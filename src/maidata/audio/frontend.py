"""Trainable Mel frontends; all boolean masks use True for valid audio."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class _DepthwiseResidual2d(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8 if channels % 8 == 0 else 1, channels)
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels
        )
        self.expand = nn.Conv2d(channels, 2 * channels, kernel_size=1)
        self.contract = nn.Conv2d(2 * channels, channels, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, values: Tensor, valid: Tensor) -> Tensor:
        invalid = ~valid[:, None, :, None]
        residual = values.masked_fill(invalid, 0.0)
        # Normalize over valid time/frequency samples within each channel group.
        # FP32 statistics also keep empty groups finite under mixed precision.
        with torch.autocast(device_type=values.device.type, enabled=False):
            grouped = residual.float().unflatten(1, (self.norm.num_groups, -1))
            selected = valid[:, None, None, :, None]
            count = selected.sum((2, 3, 4), keepdim=True)
            count = count * grouped.shape[2] * grouped.shape[4]
            mean = (grouped * selected).sum((2, 3, 4), keepdim=True)
            mean = mean / count.clamp_min(1)
            variance = ((grouped - mean).square() * selected).sum(
                (2, 3, 4), keepdim=True
            ) / count.clamp_min(1)
            normalized = ((grouped - mean) * torch.rsqrt(variance + self.norm.eps))
            normalized = normalized.flatten(1, 2)
            normalized = (
                normalized * self.norm.weight[None, :, None, None]
                + self.norm.bias[None, :, None, None]
            )
        values = normalized.to(values.dtype).masked_fill(invalid, 0.0)
        values = F.gelu(self.depthwise(values))
        values = F.gelu(self.expand(values))
        values = self.contract(values)
        return (residual + self.dropout(values)).masked_fill(invalid, 0.0)


class _QueryFrequencyAttentionProjection(nn.Module):
    def __init__(self, frequency_dim: int, frequency_queries: int, d_model: int) -> None:
        super().__init__()
        self.frequency_dim = frequency_dim
        self.frequency_queries = frequency_queries
        self.queries = nn.Parameter(torch.zeros(frequency_queries, frequency_dim))
        centers = (torch.arange(frequency_queries).float() + 0.5) / frequency_queries
        self.raw_centers = nn.Parameter(torch.logit(centers))
        initial_width = 1 / (2 * frequency_queries)
        self.raw_widths = nn.Parameter(
            torch.full((frequency_queries,), math.log(math.expm1(initial_width)))
        )
        self.dynamic_query = nn.Sequential(
            nn.LayerNorm(frequency_dim),
            nn.Linear(frequency_dim, frequency_dim),
            nn.GELU(),
            nn.Linear(frequency_dim, frequency_queries * frequency_dim),
        )
        nn.init.zeros_(self.dynamic_query[-1].weight)
        nn.init.zeros_(self.dynamic_query[-1].bias)
        self.project = nn.Linear(frequency_queries * frequency_dim, d_model)

    def attention_weights(self, features: Tensor) -> Tensor:
        """Pool [window, time, frequency, channels] with dynamic frequency queries."""
        count = features.shape[-2]
        frequencies = (torch.arange(count, device=features.device).float() + 0.5) / count
        centers = self.raw_centers.sigmoid()
        widths = F.softplus(self.raw_widths)
        bias = -0.5 * ((frequencies[None] - centers[:, None]) / widths[:, None]).square()
        dynamic = self.dynamic_query(features.mean(-2)).unflatten(
            -1, (self.frequency_queries, self.frequency_dim)
        )
        queries = self.queries + dynamic
        scores = torch.matmul(queries, features.transpose(-1, -2))
        scores = scores / math.sqrt(self.frequency_dim)
        return (scores + bias).softmax(-1)

    def forward(self, features: Tensor) -> Tensor:
        weights = self.attention_weights(features)
        summaries = torch.matmul(weights.to(features.dtype), features)
        return self.project(summaries.flatten(-2))


class SpectralTokenizer(nn.Module):
    """Encode each Mel time position without changing the time sampling grid."""

    def __init__(
        self,
        mel_bins: int,
        stem_channels: int,
        frequency_dim: int,
        frequency_queries: int,
        d_model: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if min(mel_bins, stem_channels, frequency_dim, frequency_queries, d_model) <= 0:
            raise ValueError("Spectral tokenizer dimensions must be positive.")
        self.mel_bins = mel_bins
        self.stem = nn.Conv2d(
            1, stem_channels, kernel_size=3, stride=(1, 2), padding=1
        )
        self.residual = _DepthwiseResidual2d(stem_channels, dropout)
        self.feature_lift = nn.Linear(stem_channels, frequency_dim)
        self.frequency_pool = _QueryFrequencyAttentionProjection(
            frequency_dim, frequency_queries, d_model
        )

    def forward(self, mel: Tensor, valid: Tensor) -> Tensor:
        if mel.ndim != 3 or mel.shape[-1] != self.mel_bins:
            raise ValueError(f"mel must have shape [W,N,{self.mel_bins}].")
        if valid.shape != mel.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("valid must be boolean with shape [W,N].")
        mel = mel.masked_fill(~valid.unsqueeze(-1), 0.0)
        values = F.gelu(self.stem(mel.unsqueeze(1)))
        values = values.masked_fill(~valid[:, None, :, None], 0.0)
        values = self.residual(values, valid)
        features = F.gelu(self.feature_lift(values.permute(0, 2, 3, 1)))
        features = features.masked_fill(~valid[:, :, None, None], 0.0)
        values = self.frequency_pool(features)
        return values.masked_fill(~valid.unsqueeze(-1), 0.0)


class AudioFeatureFusion(nn.Module):
    """Fuse frequency summaries and the fixed 32-field audio representation."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        self.d_model = d_model
        self.project = nn.Linear(d_model + 32, d_model)

    def forward(self, spectral: Tensor, aux32: Tensor, valid: Tensor) -> Tensor:
        if spectral.ndim != 3 or spectral.shape[-1] != self.d_model:
            raise ValueError(f"spectral must have shape [W,N,{self.d_model}].")
        if aux32.shape != (*spectral.shape[:2], 32):
            raise ValueError("aux32 must have shape [W,N,32].")
        if valid.shape != spectral.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("valid must be boolean with shape [W,N].")
        features = torch.cat((spectral, aux32), dim=-1)
        features = features.masked_fill(~valid.unsqueeze(-1), 0.0)
        return self.project(features).masked_fill(~valid.unsqueeze(-1), 0.0)


class BeatPatchTokenizer(nn.Module):
    """Build one G0 token per beat from ordered Mel samples and center aux32."""

    def __init__(self, mel_bins: int, samples_per_beat: int, d_model: int) -> None:
        super().__init__()
        if min(mel_bins, samples_per_beat, d_model) <= 0:
            raise ValueError("Beat patch dimensions must be positive.")
        self.mel_bins = mel_bins
        self.samples_per_beat = samples_per_beat
        self.project = nn.Linear(samples_per_beat * mel_bins + samples_per_beat + 32, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, mel: Tensor, sample_valid: Tensor, aux32_centers: Tensor) -> Tensor:
        if mel.ndim != 4 or mel.shape[-2:] != (self.samples_per_beat, self.mel_bins):
            raise ValueError(
                f"mel must have shape [B,J,{self.samples_per_beat},{self.mel_bins}]."
            )
        if sample_valid.shape != mel.shape[:3] or sample_valid.dtype != torch.bool:
            raise ValueError("sample_valid must be boolean with shape [B,J,R].")
        if aux32_centers.shape != (*mel.shape[:2], 32):
            raise ValueError("aux32_centers must have shape [B,J,32].")
        valid = sample_valid.any(-1)
        mel = mel.masked_fill(~sample_valid.unsqueeze(-1), 0.0)
        # The producer clears centers outside the audio range; the tokenizer
        # additionally clears entirely invalid beats, retaining partial patches.
        aux32_centers = aux32_centers.masked_fill(~valid.unsqueeze(-1), 0.0)
        patches = torch.cat(
            (mel.flatten(-2), sample_valid.to(mel.dtype), aux32_centers), dim=-1
        )
        values = self.norm(F.gelu(self.project(patches)))
        return values.masked_fill(~valid.unsqueeze(-1), 0.0)
