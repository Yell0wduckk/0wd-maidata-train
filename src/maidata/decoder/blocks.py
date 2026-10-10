"""Pre-LN blocks: causal self, P0, joint Z/plan, then shared FFN/dense MMoE."""
import torch
from torch import nn
from .attention import Attention


def ffn(dim, middle, dropout):
    return nn.Sequential(nn.Linear(dim, middle), nn.GELU(), nn.Dropout(dropout), nn.Linear(middle, dim))


class MMoE(nn.Module):
    def __init__(self, dim, middle, experts, dropout):
        super().__init__()
        self.experts = nn.ModuleList(ffn(dim, middle, dropout) for _ in range(experts))
        self.gates = nn.ModuleList(nn.Linear(dim, experts) for _ in range(5))

    def forward(self, x, types):
        # Dispatch gates only; all experts participate, and are independent per layer.
        gates = torch.stack([gate(x) for gate in self.gates], -2)
        index = types[..., None, None].expand(*types.shape, 1, len(self.experts))
        logits = gates.gather(-2, index).squeeze(-2)
        values = torch.stack([expert(x) for expert in self.experts], -2)
        return (logits.float().softmax(-1).to(values.dtype)[..., None] * values).sum(-2)


class DecoderBlock(nn.Module):
    def __init__(self, *, dim, audio_dim, heads, middle, experts, dropout, norm_eps,
                 rope_base, rope_position_scale, condition_gate_init):
        super().__init__()
        args = (heads, dropout, rope_base, rope_position_scale)
        self.self_attn = Attention(dim, dim, *args)
        self.audio_attn = Attention(dim, audio_dim, *args)
        self.condition_attn = Attention(dim, dim, *args)
        self.norms = nn.ModuleList(nn.LayerNorm(dim, eps=norm_eps) for _ in range(4))
        self.condition_gate = nn.Parameter(torch.tensor(float(condition_gate_init)))
        self.ffn = ffn(dim, middle, dropout) if experts is None else MMoE(dim, middle, experts, dropout)
        self.dropout = nn.Dropout(dropout)
        self.typed = experts is not None

    def forward(self, x, beats, valid, audio, condition, plan_index, types, previous, row_indices=None):
        y, kv = self.self_attn.self_attention(self.norms[0](x), beats, valid, previous)
        x = x + self.dropout(y)
        x = x + self.dropout(self.audio_attn(self.norms[1](x), beats, valid, audio))
        x = x + self.condition_gate * self.condition_attn(self.norms[2](x), beats, valid, condition,
                                                         plan_index=plan_index)
        if row_indices is None:
            normalized = self.norms[3](x)
            y = self.ffn(normalized, types) if self.typed else self.ffn(normalized)
            y = self.dropout(y)
        else:
            flat = x.flatten(0, 1)
            normalized = self.norms[3](flat.index_select(0, row_indices))
            y = (self.ffn(normalized, types.flatten().index_select(0, row_indices))
                 if self.typed else self.ffn(normalized))
            y = self.dropout(y)
            y = y.new_zeros(flat.shape).index_copy(0, row_indices, y).reshape_as(x)
        return (x + y).masked_fill(~valid[..., None], 0), kv
