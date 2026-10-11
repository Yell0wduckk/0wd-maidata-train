"""Structured Decoder, reusable memory projections and split Router/content API."""
import math
import torch
from torch import nn
from .types import (Memory, KV, DecoderMemory, DecoderCache, RouterOutput, DecoderOutput, validate_memory)
from .embedding import TokenEmbedding
from .attention import ConditionAdapter
from .blocks import DecoderBlock
from .heads import ContentHeads
from .checks import require


class Decoder(nn.Module):
    def __init__(self, *, dim, audio_dim, latent_dim, layers, router_layer, heads,
                 shared_ffn_dim, expert_ffn_dim, experts, readout_ffn_dim, dropout,
                 norm_eps, rope_base, rope_position_scale, note_dims, route_dims,
                 time_base, time_scale, condition_gate_init, z_cross_attention):
        super().__init__()
        self.validate_inputs = True
        if type(z_cross_attention) is not bool:
            raise ValueError('z_cross_attention must be boolean')
        integers = (dim, audio_dim, latent_dim, layers, router_layer, heads, shared_ffn_dim,
                    expert_ffn_dim, experts, readout_ffn_dim)
        if any(type(v) is not int or v <= 0 for v in integers):
            raise ValueError('Decoder dimensions must be positive integers')
        if not 1 <= router_layer < layers or dim % heads or (dim // heads) % 2:
            raise ValueError('require 1 <= router_layer < layers and even integral head width')
        if (not 0 <= dropout < 1 or not math.isfinite(norm_eps) or norm_eps <= 0
                or not all(math.isfinite(v) for v in (rope_base, rope_position_scale, time_base, time_scale, condition_gate_init))
                or min(rope_base, time_base) <= 0):
            raise ValueError('invalid Decoder numerical configuration')
        self.dim, self.audio_dim, self.router_layer = dim, audio_dim, router_layer
        self.embedding = TokenEmbedding(dim, note_dims, route_dims, time_base, time_scale)
        self.condition = ConditionAdapter(latent_dim, dim, z_cross_attention, norm_eps)
        self.target_type = nn.Embedding(5, dim)
        self.router_norm = nn.LayerNorm(dim, eps=norm_eps)
        self.type_head = nn.Linear(dim, 5)
        self.final_norm = nn.LayerNorm(dim, eps=norm_eps)
        self.layers = nn.ModuleList(DecoderBlock(dim=dim, audio_dim=audio_dim, heads=heads,
            middle=shared_ffn_dim if i < router_layer else expert_ffn_dim,
            experts=None if i < router_layer else experts, dropout=dropout, norm_eps=norm_eps,
            rope_base=rope_base, rope_position_scale=rope_position_scale,
            condition_gate_init=condition_gate_init) for i in range(layers))
        self.heads = ContentHeads(dim, readout_ffn_dim, norm_eps)
        self.architecture = 'maidata.decoder-causal-cont-tick384.v2'

    def prepare_audio(self, audio, *, indices=None):
        """Project unique song banks once per forward/parameter version; retain gradients."""
        if self.validate_inputs:
            validate_memory(audio, self.audio_dim)
        return tuple(layer.audio_attn.project(audio, indices=indices) for layer in self.layers)

    def prepare_memory(self, audio_kv, audio_mapping, z, plan, plan_valid, *, audio_layout=None):
        """One fresh object per target window batch; conditions come from public tensors."""
        if (self.validate_inputs and (len(audio_kv) != len(self.layers) or audio_mapping.dtype != torch.long
                or audio_mapping.shape != (len(z.values),))):
            raise ValueError('invalid song-to-window mapping')
        if self.validate_inputs:
            require(((audio_mapping >= 0) & (audio_mapping < len(audio_kv[0].valid))).all(),
                    'invalid song-to-window mapping')
        if audio_layout is not None:
            if self.validate_inputs:
                require((audio_mapping == audio_layout['mapping']).all(), 'audio query layout ownership differs')
        condition = self.condition(z, plan, plan_valid)
        condition_indices = None if audio_layout is None else audio_layout.get('condition_rows', {}).get('key_indices')
        def select(kv):
            if audio_layout is not None:
                return KV(kv.key, kv.value, kv.valid, audio_layout)
            return KV(kv.key.index_select(0, audio_mapping), kv.value.index_select(0, audio_mapping),
                      kv.valid.index_select(0, audio_mapping))
        return DecoderMemory(tuple(select(kv) for kv in audio_kv),
            tuple(layer.condition_attn.project(condition, plan_index=z.values.shape[1], indices=condition_indices)
                  for layer in self.layers),
            z.values.shape[1], None if audio_layout is None else audio_layout['valid_indices'], condition.beats)

    def begin(self, tokens, memory, *, cache=None):
        """Run ONLY the shared front. No next type or content label is accepted."""
        if self.validate_inputs and (cache is not None and cache.memory is not memory):
            raise ValueError('new window/memory requires fresh self-attention prefill')
        if self.validate_inputs and (len(memory.audio) != len(self.layers) or len(memory.condition) != len(self.layers)):
            raise ValueError('memory layer count mismatch')
        x = self.embedding(tokens)
        front = []
        for i in range(self.router_layer):
            x, kv = self.layers[i](x, tokens.beats, tokens.valid, memory.audio[i], memory.condition[i],
                memory.plan_index, None, None if cache is None else cache.layers[i], memory.row_indices)
            front.append(kv)
        return RouterOutput(x, self.type_head(self.router_norm(x)), tokens.beats, tokens.valid,
                            tuple(front), memory, cache, tokens.slide_beats)

    def complete(self, routed, next_types):
        """Reusing routed with a new type recomputes ALL affected back-layer K/V."""
        if self.validate_inputs and (next_types.shape != routed.valid.shape or next_types.dtype != torch.long):
            raise ValueError('next types must be aligned valid five-class IDs')
        if self.validate_inputs:
            require(((next_types >= 0) & (next_types < 5) | ~routed.valid).all(),
                    'next types must be aligned valid five-class IDs')
        types = next_types.masked_fill(~routed.valid, 0)
        x = (routed.hidden + self.target_type(types)).masked_fill(~routed.valid[..., None], 0)
        query_beats = torch.where(types == 4, routed.slide_beats, routed.beats)
        memory, old = routed.memory, routed.previous
        layers = list(routed.front_kv)
        for i in range(self.router_layer, len(self.layers)):
            x, kv = self.layers[i](x, routed.beats, routed.valid, memory.audio[i], memory.condition[i],
                memory.plan_index, types, None if old is None else old.layers[i], memory.row_indices, cross_beats=query_beats)
            layers.append(kv)
        hidden = self.final_norm(x).masked_fill(~routed.valid[..., None], 0)
        return DecoderOutput(routed.logits, hidden, DecoderCache(tuple(layers), memory))

    def forward(self, tokens, next_types, memory):
        """Parallel causal teacher forcing, or fresh per-window history prefill."""
        return self.complete(self.begin(tokens, memory), next_types)
