"""Family routing and readouts over shared-attention MMoE decoder layers.

RoutedLayer/SharedLayer retain the fresh initialization order only; active layers
are replaced by MMoELayer before training or checkpoint loading.
"""
from copy import deepcopy
from maimai_baseline.architecture import INITIAL_DECODER_LAYERS, DECODER_LAYER_SEED_XOR, PRESERVED_DECODER_LAYERS, NEW_DECODER_LAYER_SEED_XOR
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from maimai_baseline.model.rope import heads, rotate, decoder_positions
from maimai_baseline.dual_contract import TIME, NOTE, POLICY, prefix_families, family_support
from maimai_baseline.radar_contract import is_radar


def linear_slice(attention, part):
    d = attention.embed_dim
    weight = (attention.in_proj_weight[part*d:(part+1)*d] if attention._qkv_same_embed_dim
              else (attention.q_proj_weight, attention.k_proj_weight, attention.v_proj_weight)[part])
    layer = nn.Linear(weight.shape[1], d, bias=attention.in_proj_bias is not None)
    with torch.no_grad():
        layer.weight.copy_(weight)
        if layer.bias is not None:
            layer.bias.copy_(attention.in_proj_bias[part*d:(part+1)*d])
    return layer


class MemoryKV(nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.k, self.v = linear_slice(attention, 1), linear_slice(attention, 2)
        self.heads = attention.num_heads

    def forward(self, memory, positions):
        key = heads(self.k(memory), self.heads)
        return (key if positions is None else rotate(key, positions)), heads(self.v(memory), self.heads)


class Expert(nn.Module):
    def __init__(self, layer, style_attention=None, style_norm=None):
        super().__init__()
        self.self_attention = deepcopy(layer.self_attention)
        for name in ('norm1', 'norm2', 'norm3', 'linear1', 'linear2',
                     'dropout1', 'dropout2', 'dropout3', 'dropout_ff'):
            setattr(self, name, deepcopy(getattr(layer, name)))
        self.audio_q = linear_slice(layer.cross_attention, 0)
        self.audio_o = deepcopy(layer.cross_attention.out_proj)
        if style_attention is not None:
            self.style_q = linear_slice(style_attention, 0)
            self.style_o = deepcopy(style_attention.out_proj)
            self.style_norm = deepcopy(style_norm)


def dispatch(rows, family, modules, function, *, width=None):
    """Project only selected real rows, then scatter back without severing autograd."""
    flat = rows.reshape(-1, rows.shape[-1])
    result = rows.new_zeros((flat.shape[0], width or rows.shape[-1]))
    indices = family if isinstance(family, tuple) else tuple(family.reshape(-1).eq(i).nonzero(as_tuple=False).flatten() for i in range(2))
    for index, module in enumerate(modules):
        selected = indices[index]
        if selected.numel():
            value = function(module, flat.index_select(0, selected))
            result = result.to(value.dtype).index_copy(0, selected, value)
    return result.reshape(*rows.shape[:-1], -1)


def attention(q, kv, positions, padding, *, dropout, training, past_padding=None, causal=False, rotary=True):
    if len(kv) == 4:
        from maimai_baseline.model.rope import mapped_attention
        return mapped_attention(q, kv[:2], positions, padding, kv[2], dropout=dropout,
            training=training, layout=kv[3])
    nheads = kv[0].shape[1]
    query = heads(q, nheads)
    if rotary:
        query = rotate(query, positions)
    valid = ~padding
    if causal:
        key_padding = padding if past_padding is None else torch.cat((past_padding, padding), 1)
        allowed = (~key_padding)[:, None, None, :].expand(-1, 1, q.shape[1], -1).clone()
        old = kv[0].shape[2] - q.shape[1]
        allowed &= torch.arange(kv[0].shape[2], device=q.device)[None, :] <= (old + torch.arange(q.shape[1], device=q.device))[:, None]
    else:
        allowed = valid[:, None, None, :].expand(-1, 1, q.shape[1], -1).clone()
    allowed[..., 0] |= ~allowed.any(-1)
    value = F.scaled_dot_product_attention(query, *kv, attn_mask=allowed,
        dropout_p=dropout if training else 0.).transpose(1, 2).flatten(-2)
    return value if causal else value * valid.any(-1)[:, None, None].to(value.dtype)


class RoutedLayer(nn.Module):
    def __init__(self, layer, style_attention=None, style_norm=None):
        super().__init__()
        self.experts = nn.ModuleList(Expert(layer, style_attention, style_norm) for _ in range(2))
        self.audio_kv = MemoryKV(layer.cross_attention)
        self.dropout = layer.self_attention.dropout

    def forward(self, rows, family, positions, padding, audio, audio_padding,
                *, past=None, past_padding=None, style=None, style_padding=None, gates=None, indices=None):
        if indices is None:
            indices = tuple(family.reshape(-1).eq(i).nonzero(as_tuple=False).flatten() for i in range(2))
        gate = None if gates is None else gates[family.clamp_min(0)]
        return layer_forward(self, rows, positions, padding, audio, audio_padding,
            experts=self.experts, indices=indices, past=past, past_padding=past_padding,
            style=style, style_padding=style_padding, gate=gate)


class SharedLayer(nn.Module):
    """One complete first block. Its forward API deliberately has no family input."""
    def __init__(self, fresh_routed):
        super().__init__()
        self.shared = fresh_routed.experts[0]
        self.audio_kv = fresh_routed.audio_kv
        self.dropout = fresh_routed.dropout

    def forward(self, rows, positions, padding, audio, audio_padding,
                *, past=None, past_padding=None, style=None, style_padding=None, gate=None):
        indices = ((~padding).reshape(-1).nonzero(as_tuple=False).flatten(),)
        # Match the routed gate's FP32 tensor promotion under BF16 autocast.
        gate = None if gate is None else gate.expand(padding.shape)
        return layer_forward(self, rows, positions, padding, audio, audio_padding,
            experts=(self.shared,), indices=indices, past=past, past_padding=past_padding,
            style=style, style_padding=style_padding, gate=gate)


def layer_forward(layer, rows, positions, padding, audio, audio_padding,
                  *, experts, indices, past, past_padding, style, style_padding, gate, ffn=None, style_rotary=True):
    qkv = dispatch(rows, indices, experts, lambda e, x: F.linear(e.norm1(x),
        e.self_attention.in_proj_weight, e.self_attention.in_proj_bias), width=768)
    q, k, v = qkv.chunk(3, -1)
    kv = (rotate(heads(k, 8), positions), heads(v, 8))
    if past is not None:
        kv = tuple(torch.cat((old, new), 2) for old, new in zip(past, kv))
    value = attention(q, kv, positions, padding, dropout=layer.dropout,
        training=layer.training, past_padding=past_padding, causal=True)
    rows = rows + dispatch(value, indices, experts, lambda e, x: e.dropout1(e.self_attention.out_proj(x)))
    query = dispatch(rows, indices, experts, lambda e, x: e.audio_q(e.norm2(x)))
    value = attention(query, audio, positions, audio_padding, dropout=layer.dropout, training=layer.training)
    residual = dispatch(value, indices, experts, lambda e, x: e.dropout2(e.audio_o(x)))
    present = audio[3]['present'][..., None] if len(audio) == 4 else (~audio_padding).any(-1)[:, None, None]
    residual = residual * present
    rows = rows + residual
    if style is not None:
        query = dispatch(rows, indices, experts, lambda e, x: e.style_q(e.style_norm(x)))
        if style[0].ndim == 5:
            from maimai_baseline.model.song_dense import row_style_attention
            value = row_style_attention(query, style, positions, style_padding,
                dropout=layer.dropout, training=layer.training, rotary=style_rotary)
        else:
            value = attention(query, style, positions, style_padding, dropout=layer.dropout, training=layer.training,
                              rotary=style_rotary)
        value = dispatch(value, indices, experts, lambda e, x: e.style_o(x))
        present = (~style_padding).any(-1)
        present = present[..., None] if present.ndim == 2 else present[:, None, None]
        rows = rows + value * gate[..., None] * present
    rows = rows + (ffn(rows) if ffn is not None else dispatch(rows, indices, experts,
        lambda e, x: e.dropout3(e.linear2(e.dropout_ff(F.gelu(e.linear1(e.norm3(x))))))))
    return rows.masked_fill(padding[..., None], 0.), kv, residual


class SharedStyleMemory(nn.Module):
    def __init__(self, old):
        super().__init__()
        self.project = deepcopy(old.project)
        self.memories = nn.ModuleList(MemoryKV(a) for a in old.attentions[:INITIAL_DECODER_LAYERS])
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(torch.initial_seed() ^ DECODER_LAYER_SEED_XOR)
            self.memories.extend(MemoryKV(a) for a in old.attentions[INITIAL_DECODER_LAYERS:PRESERVED_DECODER_LAYERS])
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(torch.initial_seed() ^ NEW_DECODER_LAYER_SEED_XOR)
            self.memories.extend(MemoryKV(a) for a in old.attentions[PRESERVED_DECODER_LAYERS:])
        self.gates = nn.Parameter(old.gates.detach().repeat(2, 1))

    def prepare_kv(self, context):
        memory = self.project(context.tokens)
        return tuple(m(memory, context.beats if getattr(context, 'rotary', True) else None) for m in self.memories)

    def gate_values(self):
        return self.shared_gates


class Router(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.project = nn.Linear(256, 128)
        self.norms = nn.ModuleList(nn.LayerNorm(128) for _ in range(4))
        self.self_attention = nn.MultiheadAttention(128, 4, dropout=dropout, batch_first=True)
        self.audio = nn.MultiheadAttention(128, 4, kdim=256, vdim=256, dropout=dropout, batch_first=True)
        self.style = nn.MultiheadAttention(128, 4, dropout=dropout, batch_first=True)
        self.gate = nn.Parameter(torch.tensor(1e-3))
        self.ff = nn.Sequential(nn.Linear(128, 512), nn.GELU(), nn.Dropout(dropout), nn.Linear(512, 128))
        self.final_norm = nn.LayerNorm(128)
        self.head = nn.Linear(128, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, rows, positions, padding, audio, audio_padding,
                *, past=None, past_padding=None, style=None, style_padding=None, style_rotary=True):
        from maimai_baseline.model.rope import self_attention, attend
        x = self.project(rows).masked_fill(padding[..., None], 0.)
        value, kv = self_attention(self.self_attention, self.norms[0](x), positions, padding,
                                  past=past, past_padding=past_padding)
        x = x + self.dropout(value)
        def cross(module, query, memory, mask, *, rotary=True):
            allowed = (~mask)[:, None, None, :].clone()
            allowed[..., 0] |= ~allowed.any(-1)
            value = attend(module, query, memory, allowed=allowed, positions=positions if rotary else None)
            return value * (~mask).any(-1)[:, None, None]
        x = x + self.dropout(cross(self.audio, self.norms[1](x), audio, audio_padding))
        if style is not None:
            x = x + self.gate * cross(self.style, self.norms[2](x), style, style_padding, rotary=style_rotary)
        x = x + self.dropout(self.ff(self.norms[3](x)))
        return self.head(self.final_norm(x)).masked_fill(padding[..., None], 0.), kv


def install(model, seed):
    """Convert fresh dense initialization once; never migrate trained checkpoints."""
    if hasattr(model, 'router'):
        raise ValueError('dual decoder is already installed')
    old_style = getattr(model, 'style_conditioner', None)
    # Module construction must not advance any original training/initialization RNG.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed) ^ POLICY['router']['seed_xor'])
        layers = model.decoder_layers
        def build(d):
            return RoutedLayer(layers[d], None if old_style is None else old_style.attentions[d],
                               None if old_style is None else old_style.norms[d])
        model.decoder_layers = nn.ModuleList(build(d) for d in range(INITIAL_DECODER_LAYERS))
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed) ^ DECODER_LAYER_SEED_XOR)
            model.decoder_layers.extend(build(d) for d in range(INITIAL_DECODER_LAYERS, min(len(layers), PRESERVED_DECODER_LAYERS)))
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed) ^ NEW_DECODER_LAYER_SEED_XOR)
            model.decoder_layers.extend(build(d) for d in range(PRESERVED_DECODER_LAYERS, len(layers)))
        model.decoder_norm = nn.ModuleList(deepcopy(model.decoder_norm) for _ in range(2))
        old_head = model.field_heads['event']
        model.event_heads = nn.ModuleList()
        for names in (('MOMENT', 'SEGMENT_END'), ('NOTE', 'SLIDE_BRANCH')):
            head = nn.Linear(256, 2)
            ids = [model.vocabulary.event_id(n) for n in names]
            with torch.no_grad():
                head.weight.copy_(old_head.weight[ids]); head.bias.copy_(old_head.bias[ids])
            model.event_heads.append(head)
        del model.field_heads['event']
        if old_style is not None:
            model.style_conditioner = SharedStyleMemory(old_style)
        model.router = Router(model.config.dropout)
        # Construct the original fresh v11 layout first to preserve the router RNG
        # stream and all unaffected initial tensors; discard the duplicate now.
        model.decoder_layers[0] = SharedLayer(model.decoder_layers[0])
        if old_style is not None:
            conditioner = model.style_conditioner
            gates = conditioner.gates.detach()
            conditioner.shared_gate = nn.Parameter(gates[0, 0].clone())
            conditioner.expert_gates = nn.Parameter(gates[:, 1:].clone())
            del conditioner.gates
    model.dual_decoder_identity = deepcopy(POLICY)
    from maimai_baseline.model.row_fields import install as install_fields
    install_fields(model, seed)
    from maimai_baseline.model.mmoe import install as install_mmoe
    install_mmoe(model, seed)


def audio_kv(model, context, start_tick):
    from maimai_baseline.model.rope import project_kv
    positions = context.beats[-1].float() - start_tick.float()[:, None] / 96.
    memory = context.memories[-1]
    k, v = project_kv(model.router.audio, memory)
    model.audio_kv_stats = dict(decoder_p0_kv_calls=len(model.decoder_layers), router_p0_kv_calls=1,
                                decoder_p0_kv_batch_windows=len(memory))
    if hasattr(model, 'a_encoding_stats'):
        model.a_encoding_stats.update(model.audio_kv_stats)
    return dict(experts=tuple(layer.audio_kv(memory, positions) for layer in model.decoder_layers),
                router=(rotate(k, positions), v))


@dataclass
class RouterState:
    rows: torch.Tensor
    positions: torch.Tensor
    padding: torch.Tensor
    input_events: torch.Tensor
    audio_context: object
    audio_kv: dict
    style_context: object
    style_kv: object
    past_padding: object
    logits: torch.Tensor
    support: torch.Tensor
    allowed: torch.Tensor
    kv: tuple
    style_router_kv: object


def router_forward(model, *, audio_context, input_ids, input_times, input_time_active,
           row_padding_mask, row_anchor_tick, difficulty_id, level, start_tick, end_tick,
           past_padding=None, audio_key_values=None,
           style_context=None, style_key_values=None, router_key_values=None,
           router_style_key_values=None):
    from maimai_baseline.model.rope import project_kv
    if (model.config.latent_enabled or is_radar(model.config)) and model.config.chart_style != 'none' and style_context is None:
        raise ValueError('style model requires an explicit StyleContext')
    if model.config.chart_style == 'none' and style_context is not None:
        raise ValueError('style context supplied to a baseline control model')
    rows = model._row_embedding(input_ids, input_times, input_time_active)
    rows = rows + model._condition_embedding(difficulty_id=difficulty_id, level=level,
        start_tick=start_tick, end_tick=end_tick)[:, None]
    rows = rows.masked_fill(row_padding_mask[..., None], 0.)
    positions = decoder_positions(model.config, row_anchor_tick, start_tick, row_padding_mask)
    audio_key_values = audio_kv(model, audio_context, start_tick) if audio_key_values is None else audio_key_values
    enabled_style = (model.config.latent_enabled or is_radar(model.config)) and style_context is not None and style_context.enabled
    if enabled_style:
        if style_key_values is None:
            style_key_values = model.style_conditioner.prepare_kv(style_context)
        if router_style_key_values is None:
            k, v = project_kv(model.router.style, style_context.tokens)
            router_style_key_values = (rotate(k, style_context.beats) if getattr(style_context, 'rotary', True) else k), v
    router_logits, router_kv = model.router(rows, positions, row_padding_mask,
        audio_key_values['router'], audio_context.padding_masks[-1], past=router_key_values,
        past_padding=past_padding, style=router_style_key_values if enabled_style else None,
        style_padding=~style_context.valid if enabled_style else None,
        style_rotary=getattr(style_context, 'rotary', True))
    allowed = model.grammar_mask(input_ids['event'], input_ids.get('kind'))
    allowed[..., model.vocabulary.event_id('TARGET_START')] = False
    support = family_support(allowed, model.vocabulary)
    # Control-only history is unsupervised but must still have a defined route.
    support[..., TIME] |= ~support.any(-1)
    router_logits = router_logits.masked_fill(~support, -10_000.)
    return RouterState(rows, positions, row_padding_mask, input_ids['event'], audio_context,
        audio_key_values, style_context, style_key_values, past_padding, router_logits,
        support, allowed, router_kv, router_style_key_values)


def decode(model, *, decoder_family=None, family_selector=None, apply_grammar_mask=True,
           past_key_values=None, **inputs):
    state = router_forward(model, **inputs)
    row_padding_mask = state.padding
    if decoder_family is None:
        if model.training:
            raise ValueError('training requires explicit next-output decoder_family')
        decoder_family = prefix_families(state.input_events, row_padding_mask, model.vocabulary)
        pending = decoder_family.lt(0) & ~row_padding_mask
        chosen = state.logits.argmax(-1) if family_selector is None else family_selector(state.logits, state.allowed)
        decoder_family = torch.where(pending, chosen, decoder_family)
    return experts_forward(model, state, decoder_family, past_key_values=past_key_values,
                           apply_grammar_mask=apply_grammar_mask)


def experts_forward(model, state, decoder_family, *, past_key_values=None, apply_grammar_mask=True):
    """Evaluate the requested expert rows using already prepared router/condition state."""
    rows, positions, row_padding_mask = state.rows, state.positions, state.padding
    audio_context, audio_key_values = state.audio_context, state.audio_kv
    style_context, style_key_values = state.style_context, state.style_kv
    past_padding, allowed = state.past_padding, state.allowed
    router_logits, router_kv, support = state.logits, state.kv, state.support
    router_style_key_values = state.style_router_kv
    enabled_style = (model.config.latent_enabled or is_radar(model.config)) and style_context is not None and style_context.enabled
    if decoder_family.shape != row_padding_mask.shape or bool(((decoder_family < 0) | (decoder_family > 1))[~row_padding_mask].any()):
        raise ValueError('decoder_family must specify T/N for every real row')
    decoder_family = decoder_family.masked_fill(row_padding_mask, -1)
    indices = tuple(decoder_family.reshape(-1).eq(i).nonzero(as_tuple=False).flatten() for i in range(2))
    cached, norms, ratios = [], [], []
    for d, layer in enumerate(model.decoder_layers):
        before = rows
        common = dict(
            past=None if past_key_values is None else past_key_values[d], past_padding=past_padding,
            style=style_key_values[d] if enabled_style else None,
            style_padding=~style_context.valid if enabled_style else None,
            style_rotary=getattr(style_context, 'rotary', True))
        rows, kv, residual = layer(rows, decoder_family, positions, row_padding_mask,
            audio_key_values['experts'][d], audio_context.padding_masks[-1],
            gate=model.style_conditioner.shared_gates[d] if enabled_style else None,
            collect_metrics=getattr(model, 'collect_mmoe_metrics', False), **common)
        cached.append(kv)
        from maimai_baseline.model.layers import _masked_rms
        norms.append(residual.float().norm(dim=-1).mean())
        ratios.append(_masked_rms(residual, row_padding_mask) / _masked_rms(before, row_padding_mask).clamp_min(1e-12))
    return expert_readout(model, state, decoder_family, rows, cached, norms, ratios,
                          apply_grammar_mask=apply_grammar_mask)


def expert_readout(model, state, decoder_family, rows, cached, norms, ratios, *, apply_grammar_mask=True):
    row_padding_mask, positions = state.padding, state.positions
    audio_context = state.audio_context
    router_logits, router_kv, support = state.logits, state.kv, state.support
    router_style_key_values, allowed = state.style_router_kv, state.allowed
    indices = tuple(decoder_family.reshape(-1).eq(i).nonzero(as_tuple=False).flatten() for i in range(2))
    decoded = dispatch(rows, indices, model.decoder_norm, lambda m, x: m(x))
    event = dispatch(decoded, indices, model.event_heads, lambda h, x: h(x), width=2)
    logits = {}
    masks = {}
    note_family = decoder_family.eq(NOTE)
    def selected_head(head, selected, *, fp32=False):
        shape = (*decoded.shape[:-1], head.out_features)
        flat = decoded.flatten(0, 1)
        index = selected.flatten().nonzero(as_tuple=False).flatten()
        result = decoded.new_zeros((len(flat), head.out_features), dtype=torch.float32 if fp32 else decoded.dtype)
        if index.numel():
            value = flat.index_select(0, index)
            if fp32:
                with torch.autocast(device_type=value.device.type, enabled=False):
                    value = F.linear(value.float(), head.weight.float(), head.bias.float() if head.bias is not None else None)
            else:
                value = head(value)
            result = result.to(value.dtype).index_copy(0, index, value)
        return result.reshape(shape)
    for name, head in model.field_heads.items():
        logits[name] = decoded.new_zeros((*decoded.shape[:-1], head.out_features))
        masks[name] = torch.zeros_like(note_family)
    full = decoded.new_full((*decoded.shape[:-1], len(model.vocabulary.fields['event'].values)), -10_000.)
    for f, names in enumerate((('MOMENT', 'SEGMENT_END'), ('NOTE', 'SLIDE_BRANCH'))):
        for j, name in enumerate(names):
            idx = model.vocabulary.event_id(name)
            full[..., idx] = torch.where(decoder_family.eq(f), event[..., j], full[..., idx])
    logits['event'] = full.masked_fill(~allowed, -10_000.) if apply_grammar_mask else full
    masks['event'] = ~row_padding_mask
    times, time_masks = {}, {}
    for name, head in model.time_heads.items():
        selected = decoder_family.eq(TIME) if name == 'delta_tick' else torch.zeros_like(note_family)
        value = selected_head(head, selected, fp32=True)
        times[name] = value.reshape(*value.shape[:-1], model.config.time_components, 3)
        time_masks[name] = selected
    result = dict(field_logits=logits, field_valid=masks, time_outputs=times, time_valid=time_masks,
        decoded=decoded, decoder_family=decoder_family, router_logits=router_logits,
        router_support=support, conditional_event_logits=event,
        router_key_values=router_kv, router_style_key_values=router_style_key_values,
        time_unit_scale=model.config.time_unit_scale, self_key_values=tuple(cached), decoder_positions=positions,
        audio_diagnostics={**dict(audio_context.diagnostics or {}),
            'decoder_layer_residual_norms': torch.stack(norms),
            'decoder_layer_cross_residual_ratios': torch.stack(ratios),
            'decoder_input_rows': (~row_padding_mask).sum(),
            'decoder_time_rows': decoder_family.eq(TIME).sum(), 'decoder_note_rows': note_family.sum()})
    if model.delta_candidate_enabled:
        result['delta_candidate'] = dict(query=selected_head(model.delta_candidate_query, decoder_family.eq(TIME)),
            keys=audio_context.candidate_keys, padding_mask=audio_context.candidate_padding_mask,
            beats=audio_context.candidate_beats, gate=model.delta_candidate_gate,
            lookup_left=audio_context.candidate_lookup_left, lookup_right=audio_context.candidate_lookup_right,
            lookup_weight=audio_context.candidate_lookup_weight, lookup_padding=audio_context.candidate_lookup_padding,
            lookup_start_tick=audio_context.candidate_lookup_start_tick)
    return result
