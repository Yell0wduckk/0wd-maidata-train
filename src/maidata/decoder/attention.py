"""Consumer-owned projections and immutable K/V; no producer model imports."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from maidata.nn.position import apply_rope
from maidata.nn.qk_norm import normalize_qk, validate_qk_normalization
from .types import KV, Memory, validate_memory
from .checks import require
from .condition_attention import attend as attend_condition


class Attention(nn.Module):
    def __init__(self, dim, memory_dim, heads, dropout, base, scale, *, qk_normalization=False):
        super().__init__()
        self.validate_inputs = True
        self.dim, self.heads, self.dropout = dim, heads, dropout
        self.base, self.scale = base, scale
        self.qk_normalization = validate_qk_normalization(qk_normalization)
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(memory_dim, dim)
        self.value = nn.Linear(memory_dim, dim)
        self.output = nn.Linear(dim, dim)

    def split(self, x):
        return x.reshape(*x.shape[:2], self.heads, self.dim // self.heads).transpose(1, 2)

    def normalize_qk(self, value):
        return normalize_qk(value, self.qk_normalization)

    def project_query(self, values):
        """Return normalized, unrotated heads; Plan scores use these same heads."""
        projected = self.query(values)
        if projected.ndim == 2:
            projected = projected[None]
        return self.normalize_qk(self.split(projected))

    def project(self, memory, *, plan_index=None, indices=None):
        if indices is not None:
            # CPU-prepared valid positions. Project/rotate only these rows,
            # then restore the public bank shape with one differentiable copy.
            b, length = memory.valid.shape
            x = memory.values.flatten(0, 1).index_select(0, indices)[None]
            beats = memory.beats.flatten().index_select(0, indices)[None]
            raw = self.normalize_qk(self.split(self.key(x)))
            key = apply_rope(raw, beats, self.base, self.scale)
            if plan_index is not None:
                key = torch.where((indices % length >= plan_index)[None, None, :, None], raw, key)
            value = self.split(self.value(x))
            def restore(t):
                flat = t[0].transpose(0, 1)
                return flat.new_zeros((b*length, self.heads, self.dim//self.heads)).index_copy(
                    0, indices, flat).reshape(b, length, self.heads, -1).transpose(1, 2)
            return KV(restore(key), restore(value), memory.valid)
        x = memory.values.masked_fill(~memory.valid[..., None], 0)
        raw = self.normalize_qk(self.split(self.key(x)))
        k = apply_rope(raw, memory.beats.masked_fill(~memory.valid, 0), self.base, self.scale)
        if plan_index is not None:
            k = torch.cat((k[:, :, :plan_index], raw[:, :, plan_index:]), 2)
        return KV(k, self.split(self.value(x)), memory.valid)

    def forward(self, x, beats, valid, kv, *, plan_index=None, causal_offset=None):
        raw = self.project_query(x.masked_fill(~valid[..., None], 0))
        return self.attend(raw, beats, valid, kv, plan_index=plan_index, causal_offset=causal_offset)

    def attend(self, raw, beats, valid, kv, *, plan_index=None, causal_offset=None):
        q = apply_rope(raw, beats.masked_fill(~valid, 0), self.base, self.scale)
        if kv.query_layout is not None:
            if self.validate_inputs and (plan_index is not None or causal_offset is not None):
                raise ValueError('mapped queries apply only to full-song audio')
            return self.mapped(q, valid, kv)
        allowed = valid[:, None, :, None] & kv.valid[:, None, None, :]
        if causal_offset is not None:
            causal = (torch.arange(kv.key.shape[2], device=raw.device)[None, :]
                      <= causal_offset + torch.arange(valid.shape[1], device=raw.device)[:, None])
            allowed = allowed & causal[None, None]
        if plan_index is None:
            y = F.scaled_dot_product_attention(q, kv.key, kv.value, attn_mask=allowed,
                                               dropout_p=self.dropout if self.training else 0.)
        else:
            # Orthogonal halves keep rotary Z scores and unrotated plan scores
            # in ONE SDPA softmax. Explicit scale preserves the original head
            # dimension; zero value padding permits equal Q/K/V kernel widths.
            z, plan = kv.key[:, :, :plan_index], kv.key[:, :, plan_index:]
            key = torch.cat((torch.cat((z, torch.zeros_like(z)), -1),
                             torch.cat((torch.zeros_like(plan), plan), -1)), 2)
            width = q.shape[-1]
            y = attend_condition(torch.cat((q, raw), -1), key,
                F.pad(kv.value, (0, width)), kv.valid, scale=1/math.sqrt(width),
                dropout_p=self.dropout if self.training else 0.)[..., :width]
        y = y.transpose(1, 2).reshape(*valid.shape, self.dim)
        return self.output(y).masked_fill(~(valid & kv.valid.any(-1, keepdim=True))[..., None], 0)

    def mapped(self, q, valid, kv, *, packed=False):
        layout = kv.query_layout
        if self.validate_inputs and (tuple(valid.shape) != layout['shape'] or len(layout['indices']) != len(kv.valid)):
            raise ValueError('audio query layout shape mismatch')
        # Gather once before split: SplitBackward collects bucket gradients once
        # instead of each song's SliceBackward allocating the full-song batch.
        flat = q.transpose(1, 2).flatten(0, 1)
        sizes = [g[0] for g in layout['groups']]
        queries = flat.index_select(0, layout['compact_gather'] if packed else layout['gather']).split(
            [g[0]*g[1] for g in layout['groups']])
        keys = kv.key.index_select(0, layout['songs']).split(sizes)
        values = kv.value.index_select(0, layout['songs']).split(sizes)
        masks = kv.valid.index_select(0, layout['songs']).split(sizes)
        pieces = []
        for query, key, value, mask, (count, width, lo, hi) in zip(queries, keys, values, masks, layout['groups']):
            hi = kv.valid.shape[1] if hi is None else hi
            query = query.reshape(count, width, self.heads, -1).transpose(1, 2)
            result = F.scaled_dot_product_attention(query, key.narrow(2, lo, hi-lo), value.narrow(2, lo, hi-lo),
                attn_mask=mask[:, None, None, lo:hi], dropout_p=self.dropout if self.training else 0.)
            pieces.append(result.transpose(1, 2).reshape(-1, self.heads, self.dim//self.heads))
        selected = torch.cat(pieces).index_select(0, layout['take']) if pieces else flat[:0]
        output = torch.zeros_like(flat).index_copy(0, layout['compact_put'] if packed else layout['put'], selected)
        y = output.reshape(flat.shape[0], self.dim) if packed else output.reshape(*valid.shape, self.dim)
        active = valid & kv.valid.any(-1)[layout['mapping']][:, None]
        if packed:
            active = active.flatten().index_select(0, layout['valid_indices'])
        return self.output(y).masked_fill(~active[..., None], 0)

    def self_attention(self, x, beats, valid, previous):
        # Preserve separate parameter names and optimizer state while executing
        # one Q/K/V GEMM, including the incremental path.
        layers = (self.query, self.key, self.value)
        parts = F.linear(x.masked_fill(~valid[..., None], 0), torch.cat([m.weight for m in layers]),
                         torch.cat([m.bias for m in layers])).chunk(3, -1)
        raw, key, value = map(self.split, parts)
        raw, key = self.normalize_qk(raw), self.normalize_qk(key)
        current = KV(apply_rope(key, beats.masked_fill(~valid, 0), self.base, self.scale), value, valid)
        offset = 0 if previous is None else previous.valid.shape[1]
        kv = current if previous is None else KV(torch.cat((previous.key, current.key), 2),
                    torch.cat((previous.value, current.value), 2), torch.cat((previous.valid, valid), 1))
        return self.attend(raw, beats, valid, kv, causal_offset=offset), kv


class ConditionAdapter(nn.Module):
    def __init__(self, latent_dim, dim, z_cross_attention, norm_eps, z_normalization=False,
                 plan_span_beats=16, planner_layout='legacy'):
        super().__init__()
        if type(plan_span_beats) is not int or plan_span_beats not in (4, 16):
            raise ValueError('plan_span_beats must be 4 or 16')
        if planner_layout not in ('legacy', 'centered-overlap', 'window-single'):
            raise ValueError('unsupported planner_layout')
        if planner_layout in ('centered-overlap', 'window-single') and plan_span_beats != 16:
            raise ValueError(planner_layout+' requires Plan16')
        self.validate_inputs = True
        self.latent_dim = latent_dim
        self.z_cross_attention = z_cross_attention
        self.plan_span_beats = plan_span_beats
        self.planner_layout = planner_layout
        self.z = nn.Linear(latent_dim, dim)
        self.plan = nn.Sequential(nn.Linear(12, dim), nn.GELU(), nn.Linear(dim, dim))
        self.source = nn.Embedding(2, dim)
        self.plan_norm = nn.LayerNorm(dim, eps=norm_eps)
        # Bound the consumer projection + source scale without adding optimizer
        # parameters. Producer-side D normalization cannot bound this adapter.
        self.z_norm = (nn.LayerNorm(dim, eps=norm_eps, elementwise_affine=False)
                       if z_normalization else nn.Identity())

    def forward(self, z, plan, plan_valid, plan_beats=None):
        if self.planner_layout == 'centered-overlap':
            return self.positioned_plan(z, plan, plan_valid, plan_beats)
        if self.plan_span_beats == 4:
            return self.aligned_plan(z, plan, plan_valid)
        if self.validate_inputs and self.z_cross_attention:
            validate_memory(z, self.latent_dim)
        if (self.validate_inputs and (plan.shape != (len(z.values), 12) or plan_valid.shape != plan.shape[:1]
                or plan_valid.dtype != torch.bool)):
            raise ValueError('plan requires [B,12] and independent validity [B]')
        if self.validate_inputs:
            require((torch.isfinite(plan) | ~plan_valid[:, None]).all(), 'valid plan values must be finite')
        b = self.plan_norm(self.plan(plan.masked_fill(~plan_valid[:, None], 0)) + self.source.weight[1])
        if self.z_cross_attention:
            a = self.z_norm(self.z(z.values.masked_fill(~z.valid[..., None], 0)) + self.source.weight[0])
            beats, valid = z.beats, z.valid
        else:
            a = b.new_zeros((*z.valid.shape, b.shape[-1]))
            beats, valid = torch.zeros_like(z.beats), torch.zeros_like(z.valid)
        return Memory(torch.cat((a, b[:, None]), 1),
                      torch.cat((beats, beats.new_zeros((len(plan), 1))), 1),
                      torch.cat((valid, plan_valid[:, None]), 1))

    def aligned_plan(self, z, plan, plan_valid):
        """Each Plan4 slot owns the same absolute center as its corresponding Z4.

        Validity remains independent: an absent Plan never removes a valid Z,
        and a valid empty-segment Plan can survive a masked Z slot.
        """
        if self.validate_inputs and self.z_cross_attention:
            validate_memory(z, self.latent_dim)
        if (z.values.ndim != 3
                or z.beats.shape != z.values.shape[:2] or z.valid.shape != z.beats.shape
                or z.valid.dtype != torch.bool or plan.shape != (*z.values.shape[:2], 12)
                or plan_valid.shape != z.valid.shape or plan_valid.dtype != torch.bool
                or plan.device != z.values.device or plan_valid.device != z.values.device):
            raise ValueError('Plan4 requires [B,L,12] and validity [B,L] aligned with Z slots')
        if self.validate_inputs:
            require((torch.isfinite(plan) | ~plan_valid[..., None]).all()
                    & (torch.isfinite(z.beats) | ~plan_valid).all(),
                    'valid Plan4 values and aligned centers must be finite')
        b = self.plan_norm(self.plan(plan.masked_fill(~plan_valid[..., None], 0)) + self.source.weight[1])
        if self.z_cross_attention:
            a = self.z_norm(self.z(z.values.masked_fill(~z.valid[..., None], 0)) + self.source.weight[0])
            beats, valid = z.beats, z.valid
        else:
            a = b.new_zeros((*z.valid.shape, b.shape[-1]))
            beats, valid = torch.zeros_like(z.beats), torch.zeros_like(z.valid)
        return Memory(torch.cat((a, b), 1), torch.cat((beats, z.beats), 1),
                      torch.cat((valid, plan_valid), 1))

    def positioned_plan(self, z, plan, plan_valid, plan_beats):
        """Independent Plan centers share the rotary attention space with Z."""
        if self.validate_inputs and self.z_cross_attention:
            validate_memory(z, self.latent_dim)
        if (z.values.ndim != 3 or z.beats.shape != z.values.shape[:2]
                or z.valid.shape != z.beats.shape or z.valid.dtype != torch.bool
                or plan.ndim != 3 or plan.shape[0] != len(z.values) or plan.shape[-1] != 12
                or plan_valid.shape != plan.shape[:2] or plan_valid.dtype != torch.bool
                or plan_beats is None or plan_beats.shape != plan_valid.shape
                or not plan_beats.is_floating_point()
                or any(t.device != z.values.device for t in (plan, plan_valid, plan_beats))):
            raise ValueError('positioned Plan requires [B,P,12], validity and independent plan_beats [B,P]')
        if self.validate_inputs:
            require((torch.isfinite(plan) | ~plan_valid[..., None]).all()
                    & (torch.isfinite(plan_beats) | ~plan_valid).all(),
                    'valid Plan values and centers must be finite')
        b = self.plan_norm(self.plan(plan.masked_fill(~plan_valid[..., None], 0)) + self.source.weight[1])
        if self.z_cross_attention:
            a = self.z_norm(self.z(z.values.masked_fill(~z.valid[..., None], 0)) + self.source.weight[0])
            beats, valid = z.beats, z.valid
        else:
            a = b.new_zeros((*z.valid.shape, b.shape[-1]))
            beats, valid = torch.zeros_like(z.beats), torch.zeros_like(z.valid)
        return Memory(torch.cat((a, b), 1), torch.cat((beats, plan_beats), 1),
                      torch.cat((valid, plan_valid), 1))
