"""Temporary field reads; never writes main self-attention K/V or rule state."""
import torch
from torch import nn
from .attention import Attention
from .types import KV


class QueryAttention(Attention):
    """Independent Q/output, reusing the consumer's projected memory K/V."""
    def __init__(self, dim, heads, dropout, base, scale):
        nn.Module.__init__(self)
        self.dim, self.heads, self.dropout = dim, heads, dropout
        self.base, self.scale, self.validate_inputs = base, scale, True
        self.query, self.output = nn.Linear(dim, dim), nn.Linear(dim, dim)


class FieldReader(nn.Module):
    def __init__(self, dim, heads, dropout, eps, base, scale):
        super().__init__()
        self.role = nn.Embedding(5, dim)
        self.progress = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.norms = nn.ModuleList(nn.LayerNorm(dim, eps=eps) for _ in range(3))
        self.audio = QueryAttention(dim, heads, dropout, base, scale)
        self.z = QueryAttention(dim, heads, dropout, base, scale)

    def forward(self, base, memory, indices, beats, roles, origins, elapsed, *, owner_spans=None):
        """Each row is independent; GT query count/order cannot condition another row.

        ``indices`` selects committed-prefix states, while ``beats`` is temporary.
        The originating full route chain remains in the causal prefix state.
        """
        h = base[indices[:, 0], indices[:, 1]]
        features = torch.stack((elapsed.float()/384, beats.float()-origins.float()/96), -1)
        x = h + self.role(roles) + self.progress(features.to(h.dtype))
        pieces, positions = [], []
        if owner_spans is None:
            # Scalar inference and public CPU callers need no transport layout.
            # Training always supplies CPU-prepared spans, avoiding device
            # nonzero or host reads of query ownership on the hot path.
            owners = indices[:,0].tolist()
            owner_spans = []
            for start,owner in enumerate(owners):
                if not owner_spans or owner_spans[-1][0] != owner:
                    owner_spans.append([owner,start,start+1])
                else:
                    owner_spans[-1][2] = start+1
        for owner,start,stop in owner_spans:
            chosen = torch.arange(start,stop,device=indices.device)
            q, count = beats[chosen, None], len(chosen)
            valid = torch.ones((count, 1), device=x.device, dtype=torch.bool)
            def bank(kv, source, mask=None):
                key = kv.key.index_select(0, source).expand(count, -1, -1, -1)
                value = kv.value.index_select(0, source).expand(count, -1, -1, -1)
                active = kv.valid.index_select(0, source).expand(count, -1)
                return KV(key, value, active if mask is None else active & mask)
            audio = memory.audio[-1]
            source = (audio.query_layout['mapping'][owner:owner+1] if audio.query_layout is not None
                      else indices.new_tensor([owner]))
            y = x[chosen, None]
            y = y + self.audio(self.norms[0](y), q, valid, bank(audio, source))
            # Plan has already conditioned the main prefix. Field reads select
            # only local Z, with MOMENT's known-cursor forward four-beat range.
            centers = memory.condition_beats[owner:owner+1]
            slots, anchor = torch.floor(centers), torch.floor(q)
            moment = roles[chosen, None] == 0
            local = torch.where(moment, (slots >= anchor) & (slots <= torch.floor(q+4)),
                                (slots >= anchor-1) & (slots <= anchor+1))
            local[:, memory.plan_index:] = False
            condition = bank(memory.condition[-1], indices.new_tensor([owner]), local)
            y = y + self.z(self.norms[1](y), q, valid, condition)
            pieces.append(self.norms[2](y[:, 0]))
            positions.append(chosen)
        if not pieces:
            return h
        return torch.zeros_like(h).index_copy(0, torch.cat(positions), torch.cat(pieces))
