"""Full-song, non-AR beat latents; independent of Encoder and Decoder classes."""
import torch
from torch import nn
from torch.nn import functional as F

from maidata.nn.position import sinusoidal
from .attention import Attention, positive_ints, numerical_config
from .types import BeatMemory, DraftInputs, ProjectedAudio


class DenseMoE(nn.Module):
    def __init__(self, width, experts, hidden, dropout):
        super().__init__()
        self.gate = nn.Linear(width, experts)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.experts = nn.ModuleList(nn.Sequential(nn.Linear(width, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, width)) for _ in range(experts))

    def forward(self, x, indices=None):
        shape = x.shape
        if indices is not None:
            x = x.flatten(0, 1).index_select(0, indices)
        weights = self.gate(x).float().softmax(-1).to(x.dtype)
        mixed = (torch.stack([expert(x) for expert in self.experts], -2)
                 * weights[..., None]).sum(-2)
        if indices is None:
            return mixed
        return mixed.new_zeros(shape[0]*shape[1], shape[2]).index_copy(0, indices, mixed).reshape(shape)


class DraftBlock(nn.Module):
    def __init__(self, audio_dim, width, heads, experts, hidden, dropout, eps, base, scale):
        super().__init__()
        self.norms = nn.ModuleList(nn.LayerNorm(width, eps=eps) for _ in range(3))
        self.self_attention = Attention(width, width, width, heads, dropout,
                                        rotary=True, base=base, scale=scale)
        self.cross_attention = Attention(width, audio_dim, width, heads, dropout,
                                         rotary=True, base=base, scale=scale)
        self.modulation = nn.Linear(width, 2 * width)
        self.experts = DenseMoE(width, experts, hidden, dropout)

    def forward(self, x, beats, valid, condition, audio_kv, audio_valid, valid_indices=None):
        scale, shift = self.modulation(condition).chunk(2, -1)
        h = self.norms[0](x) * (1 + scale[:, None]) + shift[:, None]
        kv = self.self_attention.project_memory(h, beats, valid)
        x = (x + self.self_attention(h, beats, valid, kv, valid)).masked_fill(~valid[..., None], 0)
        x = (x + self.cross_attention(self.norms[1](x), beats, valid, audio_kv,
                                      audio_valid)).masked_fill(~valid[..., None], 0)
        return (x + self.experts(self.norms[2](x), valid_indices)).masked_fill(~valid[..., None], 0)


class DraftGenerator(nn.Module):
    def __init__(self, *, audio_dim, layers, dim, heads, experts, expert_ffn_dim,
                 latent_dim, dropout, norm_eps, difficulty_vocab_size,
                 level_center, level_scale, span_scale, rope_base, rope_position_scale):
        super().__init__()
        positive_ints(audio_dim=audio_dim, layers=layers, dim=dim, heads=heads,
                      experts=experts, expert_ffn_dim=expert_ffn_dim, latent_dim=latent_dim,
                      difficulty_vocab_size=difficulty_vocab_size)
        numerical_config(dropout, norm_eps, rope_base, rope_position_scale)
        if not all(torch.isfinite(torch.tensor(v)) for v in (level_center, level_scale, span_scale)) or min(level_scale, span_scale) <= 0:
            raise ValueError('condition normalization must be finite with positive scales')
        self.audio_dim, self.dim = audio_dim, dim
        self.base, self.scale = rope_base, rope_position_scale
        self.level_center, self.level_scale, self.span_scale = level_center, level_scale, span_scale
        self.query = nn.Parameter(torch.zeros(dim))
        self.difficulty = nn.Embedding(difficulty_vocab_size, dim)
        self.level = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        self.span = nn.Linear(1, dim)
        self.blocks = nn.ModuleList(DraftBlock(audio_dim, dim, heads, experts, expert_ffn_dim,
            dropout, norm_eps, rope_base, rope_position_scale) for _ in range(layers))
        self.output_norm = nn.LayerNorm(dim, eps=norm_eps)
        self.output = nn.Linear(dim, latent_dim)
        self.latent_norm = nn.LayerNorm(latent_dim, eps=norm_eps, elementwise_affine=False)

    def project_audio(self, audio) -> ProjectedAudio:
        """One projection per layer and unique audio, reusable within this graph only."""
        return ProjectedAudio(tuple(block.cross_attention.project_memory(
            audio.values, audio.beats, audio.valid) for block in self.blocks), audio.valid)

    def forward(self, audio, inputs: DraftInputs) -> BeatMemory:
        return self.from_projected(self.project_audio(audio), inputs)

    def from_projected(self, audio: ProjectedAudio, inputs: DraftInputs) -> BeatMemory:
        beats, valid, mapping = inputs.query_beats, inputs.draft_valid, inputs.audio_mapping
        difficulty, level, span = inputs.difficulty_id, inputs.level, inputs.span_beats
        av = audio.valid.index_select(0, mapping)
        positions = beats.masked_fill(~valid, 0)
        condition = (self.difficulty(difficulty)
                     + self.level((level[:, None].to(self.query) - self.level_center) / self.level_scale)
                     + self.span(span[:, None].to(self.query) / self.span_scale))
        x = (self.query + sinusoidal(positions, self.dim, self.base, self.scale).to(condition))
        x = x.masked_fill(~valid[..., None], 0)
        valid_indices = inputs.valid_indices
        if valid_indices is None:
            valid_indices = valid.detach().cpu().flatten().nonzero().flatten().to(valid.device)
        for block, kv in zip(self.blocks, audio.key_values):
            x = block(x, positions, valid, condition,
                      tuple(value.index_select(0, mapping) for value in kv), av, valid_indices)
        # Explicit FP32 parameters also support an otherwise bf16 model.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = F.layer_norm(x.float(), self.output_norm.normalized_shape,
                             self.output_norm.weight.float(), self.output_norm.bias.float(),
                             self.output_norm.eps)
            x = F.linear(x, self.output.weight.float(), self.output.bias.float())
            x = self.latent_norm(x)
        return BeatMemory(x.masked_fill(~valid[..., None], 0), beats, valid)
