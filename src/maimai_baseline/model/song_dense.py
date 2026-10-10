"""Dense song attention, bounded local router and owning-window conditions."""
from dataclasses import replace
import torch
from torch.nn import functional as F
from torch.profiler import record_function
from maimai_baseline.model.rope import heads, rotate, project_kv
from maimai_baseline.model.dual import expert_readout
from maimai_baseline.radar_contract import is_radar, normalize
from maimai_baseline.style_contract import is_frozen, is_native, is_draft


def row_style_attention(query, kv, positions, padding, *, dropout, training, rotary=True):
    # Each row reads its owning window's condition, never other windows.
    b, r, _ = query.shape
    h, s, d = kv[0].shape[2:]
    q = heads(query, h)
    if rotary:
        q = rotate(q, positions)
    q = q.transpose(1, 2).reshape(b*r, h, 1, d)
    k, v = (x.reshape(b*r, h, s, d) for x in kv)
    valid = ~padding.reshape(b*r, s)
    mask = valid[:, None, None, :].clone()
    mask[..., 0] |= ~mask.any(-1)
    value = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
        dropout_p=dropout if training else 0.)
    return value.reshape(b, r, h*d) * valid.any(-1).reshape(b, r, 1)


def dense_layers(model, rows, family, positions, padding, memory, memory_padding, memory_beats,
                 *, style=None, style_owner=None, window_start=None, past=None, past_padding=None,
                 audio_keys=None, style_keys=None, audio_mapping=None, audio_layout=None):
    """One dense song call per layer. No query/window scheduling or recompute."""
    norms, ratios, cached = [], [], []
    from maimai_baseline.model.layers import _masked_rms
    enabled = (model.config.latent_enabled or is_radar(model.config)) and style is not None and style.enabled
    if enabled:
        absolute_style = replace(style, beats=style.beats + window_start[:, None]/96.) if getattr(style, 'rotary', True) else style
        styles = model.style_conditioner.prepare_kv(absolute_style) if style_keys is None else style_keys
        style_padding = ~style.valid[style_owner]
    if audio_mapping is not None and audio_layout is None:
        from maimai_baseline.data.attention_layout import cross_layout, to_device
        audio_layout = to_device(cross_layout(padding.cpu(), memory_padding.cpu(), audio_mapping.cpu()), rows.device)
    for d, layer in enumerate(model.decoder_layers):
        with record_function('dense.decoder.layer'):
            audio = layer.audio_kv(memory, memory_beats) if audio_keys is None else audio_keys[d]
            if audio_mapping is not None:
                audio = (*audio, audio_mapping, audio_layout)
            row_style = None if not enabled else tuple(x.transpose(1, 2)[style_owner].transpose(2, 3) for x in styles[d])
            before = rows
            rows, kv, residual = layer(rows, family, positions, padding, audio, memory_padding,
                past=None if past is None else past[d], past_padding=past_padding,
                style=row_style, style_padding=style_padding if enabled else None,
                gate=model.style_conditioner.shared_gates[d] if enabled else None,
                collect_metrics=getattr(model, 'collect_mmoe_metrics', False),
                style_rotary=getattr(style, 'rotary', True))
            cached.append(kv)
            norms.append(residual.float().norm(dim=-1).mean())
            ratios.append(_masked_rms(residual, padding)/_masked_rms(before, padding).clamp_min(1e-12))
    return rows, cached, norms, ratios


def forward(model, original):
    from maimai_baseline.model.song_chart import window_posteriors
    from maimai_baseline.data.song_audio import encode_batch
    from maimai_baseline.model.style import StylePosterior
    from maimai_baseline.model.row_fields import supervised_readout
    execution = original['song_dense_inputs']
    batch = execution['batch']
    model.a_encoding_stats = {}
    style = posterior = None
    cached_audio = execution.get('audio_context')
    if execution.get('sampled_style_context') is not None:
        if not is_frozen(model.config) or cached_audio is None or 'local_audio_context' not in execution:
            raise ValueError('sampled decoder conditions require frozen-mu A and cached audio')
        style = execution['sampled_style_context']
        model.a_encoding_stats.update(latent_enabled=True, chart_encoder_calls=0,
                                      latent_condition_windows=len(style.tokens))
    elif is_draft(model.config):
        style = model.draft_context(batch['global_style_inputs'])
        model.a_encoding_stats.update(latent_enabled=True, chart_encoder_calls=0,
            draft_generator_calls=0, latent_condition_windows=len(style.tokens))
    elif is_frozen(model.config):
        from maimai_baseline.model.style import frozen_context
        with record_function('encoding.frozen_latent_condition'):
            style = frozen_context(batch['global_style_inputs'], perturb=is_native(model.config) and model.training)
        model.a_encoding_stats.update(latent_enabled=True, chart_encoder_calls=0,
                                      latent_condition_windows=len(style.tokens))
    elif model.config.latent_enabled:
        with record_function('encoding.chart_posterior'):
            style, posterior = window_posteriors(model, batch['global_style_inputs'], sample=model.training)
    if cached_audio is not None:
        from maimai_baseline.model.common import AudioContext
        context = AudioContext(**cached_audio) if isinstance(cached_audio, dict) else cached_audio
        local_context = execution['local_audio_context']
        if isinstance(local_context, dict):
            local_context = AudioContext(**local_context)
        if is_native(model.config) and local_context.candidate_keys is None:
            from maimai_baseline.data.teacher_audio import project_candidates
            local_context = project_candidates(model, local_context, execution['teacher_interpolation'])
    else:
        if is_native(model.config):
            raise ValueError('native A requires --teacher-audio-cache; audio encoding fallback is unavailable')
        with record_function('encoding.song_audio'):
            song = encode_batch(model, batch['global_style_inputs']['song_audio'])
        with record_function('encoding.continuous_audio'):
            context = model.encode_audio(batch['audio'], batch['audio_padding_mask'], batch['frame_beats'],
                batch['frame_bpms'], start_tick=batch['audio_start_tick'], end_tick=batch['end_tick'],
                interpolation_geometry=batch['interpolation_geometry'],
                song_audio=song.for_audio_windows(batch['window_to_audio'], len(batch['audio'])))
        from maimai_baseline.data.fullaudio import window_context
        local_context = window_context(context, batch['window_to_audio'], batch['start_tick'])
    group_audio = execution.get('group_audio')
    if group_audio is None:
        group_audio = torch.stack([batch['window_to_audio'][w[0]] for w in execution['group_windows']])
    radar_prior = None
    if is_radar(model.config):
        radar = execution['radar_inputs']
        gt, radar_valid = radar['values'], radar['valid']
        window_group, window_index = radar['window_group'], radar['window_index']
        with record_function('encoding.radar_condition'):
            style = model.radar_context(gt[window_group, window_index], radar_valid[window_group, window_index])
        first_windows = torch.tensor([windows[0] for windows in execution['group_windows']], device=gt.device)
        lengths = batch['song_end_tick'][first_windows].to(gt.dtype)/96.
        with record_function('prior.radar'):
            targets = normalize(gt).masked_fill(~radar_valid[..., None], 0.)
            logits, mean, logvar = model.style_prior(model.style_prior.shifted(targets, radar_valid),
                audio=context.memories[-1], audio_valid=~context.padding_masks[-1], audio_beats=context.beats[-1],
                audio_mapping=group_audio, difficulty_id=batch['difficulty_id'][first_windows],
                span_beats=radar.get('span_beats', lengths), level=batch['level'][first_windows], valid=radar_valid)
        radar_prior = dict(logits=logits, mean=mean, logvar=logvar, target=targets, valid=radar_valid,
            window_group=window_group, window_index=window_index, target_windows=execution['target_windows'])
        model.a_encoding_stats.update(prior_audio_kv_calls=len(model.style_prior.blocks),
            prior_audio_kv_batch_variants=len(context.memories[-1]), radar_encoder_calls=1)
    fields = ('input_ids', 'input_times', 'input_time_active', 'row_padding_mask', 'row_anchor_tick',
              'difficulty_id', 'level', 'start_tick', 'end_tick')
    with record_function('dense.local_router'):
        k, v = project_kv(model.router.audio, local_context.memories[-1])
        audio_keys = dict(experts=(), router=(rotate(k,
            local_context.beats[-1] - batch['start_tick'][:, None]/96.), v))
        state = model.router_forward(audio_context=local_context, style_context=style,
            audio_key_values=audio_keys, style_key_values=(),
            **{k: batch[k] for k in fields})
    owner, row, valid = execution['owner'], execution['row'], execution['valid']
    with record_function('dense.organize'):
        rows = state.rows[owner, row].masked_fill(~valid[..., None], 0.)
        positions = (batch['row_anchor_tick'][owner, row]/96.).masked_fill(~valid, 0.)
        family = batch['decoder_family'][owner, row].masked_fill(~valid, -1)
        memory, memory_padding, memory_beats = context.memories[-1], context.padding_masks[-1], context.beats[-1]
    rows, cached, norms, ratios = dense_layers(model, rows, family, positions, ~valid,
        memory, memory_padding, memory_beats, style=style, style_owner=owner, window_start=batch['start_tick'],
        audio_mapping=group_audio, audio_layout=execution.get('audio_layout'),
        audio_keys=execution.get('decoder_audio_keys'))
    target = execution['target_windows']
    output_group, output_row = execution['output_group'], execution['output_row']
    def subset(x):
        return x.index_select(0, target)
    selected = replace(state, rows=subset(state.rows), positions=subset(state.positions),
        padding=original['row_padding_mask'], logits=subset(state.logits), support=subset(state.support),
        allowed=subset(state.allowed), audio_context=local_context.map_audio_windows(target),
        kv=tuple(subset(x) for x in state.kv))
    hidden = rows[output_group, output_row].masked_fill(original['row_padding_mask'][..., None], 0.)
    # History layouts may have different padded widths in the full-window batch.
    n = hidden.shape[1]
    selected = replace(selected, positions=selected.positions[:, :n], logits=selected.logits[:, :n],
        support=selected.support[:, :n], allowed=selected.allowed[:, :n])
    result = expert_readout(model, selected, original['decoder_family'], hidden, (), norms, ratios)
    result = model._time_outputs(model._route_outputs(result))
    result = supervised_readout(model, result, original)
    if radar_prior is not None:
        result['radar_prior'] = radar_prior
    if posterior is not None:
        result['style_posterior'] = StylePosterior(*(subset(x) for x in (posterior.mu, posterior.logvar, posterior.beats, posterior.valid)))
    result['audio_context'] = selected.audio_context
    model.a_encoding_stats.update(audio_encoder_calls=int(cached_audio is None),
        audio_encoder_batch_size=0 if cached_audio is not None else len(batch['audio']),
        decoder_p0_kv_calls=0 if execution.get('decoder_audio_keys') is not None else len(model.decoder_layers), router_p0_kv_calls=1,
        decoder_p0_kv_batch_windows=len(memory), dense_context_rows=execution['context_rows'],
        dense_padded_context_rows=valid.numel(),
        dense_padded_audio_tokens=memory_padding.numel())
    if getattr(model, 'collect_mmoe_metrics', False):
        model.a_encoding_stats['dense_audio_tokens'] = int((~memory_padding).sum())
    return result
