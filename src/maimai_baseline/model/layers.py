"""Shared tensor operators and decoder layers."""

from __future__ import annotations
from maimai_dataset.window import AUDIO_CONTEXT_BEATS, AUDIO_BEATS_BEFORE
from maimai_baseline.model.common import BaselineModelConfig
from maimai_dataset.encoding import CHART_TICKS_PER_BEAT
from torch import nn
from torch.nn import functional
import math
import torch


class ContinuousSinusoidal(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = int(d_model)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values_fp32 = values.to(torch.float32).unsqueeze(-1)
        divisor = torch.exp(
            torch.arange(
                0, self.d_model, 2, device=values.device, dtype=torch.float32
            )
            * (-math.log(10_000.0) / self.d_model)
        )
        result = torch.zeros(
            (*values.shape, self.d_model), device=values.device, dtype=torch.float32
        )
        result[..., 0::2] = torch.sin(values_fp32 * divisor)
        result[..., 1::2] = torch.cos(values_fp32 * divisor)
        return result.to(values.dtype)


class DepthwiseResidual2d(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.norm = nn.GroupNorm(groups, channels)
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels
        )
        self.expand = nn.Conv2d(channels, channels * 2, kernel_size=1)
        self.contract = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, value: torch.Tensor, padding: torch.Tensor | None = None) -> torch.Tensor:
        residual = value
        if padding is None:
            value = self.norm(value)
        else:
            # GroupNorm over valid song samples only, independent of batch padding.
            with torch.autocast(device_type=value.device.type, enabled=False):
                x = value.float().unflatten(1,(self.norm.num_groups,-1))
                valid = (~padding)[:,None,None,:,None]
                count = valid.sum((2,3,4),keepdim=True)*x.shape[2]*x.shape[4]
                mean = (x*valid).sum((2,3,4),keepdim=True)/count.clamp_min(1)
                variance = ((x-mean).square()*valid).sum((2,3,4),keepdim=True)/count.clamp_min(1)
                x = ((x-mean)*torch.rsqrt(variance+self.norm.eps)).flatten(1,2)
                x = x*self.norm.weight[None,:,None,None]+self.norm.bias[None,:,None,None]
            value = x.to(value.dtype).masked_fill(padding[:,None,:,None],0.)
        value = self.depthwise(value)
        value = functional.gelu(value)
        value = self.expand(value)
        value = functional.gelu(value)
        value = self.contract(value)
        output = residual + self.dropout(value)
        return output if padding is None else output.masked_fill(padding[:,None,:,None],0.)


class QueryFrequencyAttentionProjection(nn.Module):
    """Four base queries plus frame-conditioned residuals and Mel-frequency priors."""

    def __init__(self, channels: int, d_model: int) -> None:
        super().__init__()
        if channels != 64 or d_model != 256:
            raise ValueError("query frequency pooling requires 64 channels and width 256")
        self.queries = nn.Parameter(torch.zeros(4, 64))
        self.raw_centers = nn.Parameter(torch.logit((torch.arange(4).float() + .5) / 4))
        self.raw_widths = nn.Parameter(torch.full((4,), math.log(math.expm1(1 / 8 - 1e-4))))
        self.register_buffer("frequencies", (torch.arange(64).float() + .5) / 64, persistent=False)
        self.project = nn.Linear(256, 256)
        # Preserve all existing parameter initialization and the caller's RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.dynamic_query = nn.Sequential(
                nn.LayerNorm(64), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 256)
            )
            nn.init.zeros_(self.dynamic_query[-1].weight)
            nn.init.zeros_(self.dynamic_query[-1].bias)

    def attention_weights(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1] != 64 or value.shape[-1] != 64:
            raise ValueError("query frequency pooling expects [batch,64,time,64]")
        features = value.permute(0, 2, 3, 1)
        centers = self.raw_centers.sigmoid()
        widths = functional.softplus(self.raw_widths) + 1e-4
        bias = -.5 * ((self.frequencies[None] - centers[:, None]) / widths[:, None]).square()
        queries = self.queries + self.dynamic_query(features.mean(-2)).unflatten(-1, (4, 64))
        scores = torch.matmul(queries, features.transpose(-1, -2)) / math.sqrt(64)
        return (scores + bias).softmax(-1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        weights = self.attention_weights(value)
        pooled = torch.matmul(weights, value.permute(0, 2, 3, 1))
        return self.project(pooled.flatten(-2))




def _masked_pool_scalar(
    values: torch.Tensor,
    padding_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = (~padding_mask).to(torch.float32).unsqueeze(1)
    numerator = functional.avg_pool1d(
        (values.to(torch.float32).unsqueeze(1) * valid),
        kernel_size=3,
        stride=2,
        padding=1,
        count_include_pad=False,
    )
    count = functional.avg_pool1d(
        valid,
        kernel_size=3,
        stride=2,
        padding=1,
        count_include_pad=False,
    )
    result = (numerator / count.clamp_min(torch.finfo(torch.float32).eps)).squeeze(1)
    padding = count.squeeze(1).eq(0)
    return result.masked_fill(padding, 0.0), padding


def _indexed_interpolate(
    values: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    weight: torch.Tensor,
    padding: torch.Tensor,
) -> torch.Tensor:
    """Apply precomputed linear interpolation geometry without coordinate search."""

    if not (
        left.shape == right.shape == weight.shape == padding.shape
        and left.ndim == 2
        and values.shape[0] == left.shape[0]
    ):
        raise ValueError("indexed interpolation tensors are not batch aligned")
    # Bounds and dtypes are validated while loading/collating the immutable
    # geometry.  Rechecking with ``.cpu()`` here would synchronize the GPU on
    # every forward and erase most of the indexed backend's benefit.
    left_index = left.to(torch.long).unsqueeze(-1).expand(-1, -1, values.shape[-1])
    right_index = right.to(torch.long).unsqueeze(-1).expand(-1, -1, values.shape[-1])
    left_value = torch.gather(values, 1, left_index)
    right_value = torch.gather(values, 1, right_index)
    interpolation_weight = weight.to(torch.float32).unsqueeze(-1)
    result = (
        left_value * (1.0 - interpolation_weight)
        + right_value * interpolation_weight
    ).to(values.dtype)
    return result.masked_fill(padding.unsqueeze(-1), 0.0)


def _fixed_beat_grid(
    start_tick: torch.Tensor,
    *,
    tokens_per_beat: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    indices = torch.arange(
        AUDIO_CONTEXT_BEATS * tokens_per_beat, device=start_tick.device, dtype=torch.float32
    )
    audio_start = start_tick.to(torch.float32) / CHART_TICKS_PER_BEAT - AUDIO_BEATS_BEFORE
    grid = audio_start[:, None] + (indices[None, :] + 0.5) / float(tokens_per_beat)
    return grid.to(dtype)




class RelativeDecoderLayer(nn.Module):
    def __init__(self, config: BaselineModelConfig) -> None:
        super().__init__()
        decoder_dim = config.resolved_decoder_d_model
        decoder_heads = config.resolved_decoder_heads
        decoder_feedforward = config.resolved_decoder_feedforward
        self.self_attention = nn.MultiheadAttention(
            decoder_dim, decoder_heads, dropout=config.dropout, batch_first=True
        )
        if decoder_dim == config.d_model:
            self.cross_attention = nn.MultiheadAttention(
                decoder_dim,
                decoder_heads,
                dropout=config.dropout,
                batch_first=True,
            )
        else:
            self.cross_attention = nn.MultiheadAttention(
                decoder_dim,
                decoder_heads,
                dropout=config.dropout,
                batch_first=True,
                kdim=config.d_model,
                vdim=config.d_model,
            )
        self.linear1 = nn.Linear(decoder_dim, decoder_feedforward)
        self.linear2 = nn.Linear(decoder_feedforward, decoder_dim)
        self.norm1 = nn.LayerNorm(decoder_dim)
        self.norm2 = nn.LayerNorm(decoder_dim)
        self.norm3 = nn.LayerNorm(decoder_dim)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)
        self.dropout_ff = nn.Dropout(config.dropout)

def _masked_rms(value: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
    active = (~padding).unsqueeze(-1).to(value.dtype)
    numerator = (value.square() * active).sum()
    denominator = (active.sum() * value.shape[-1]).clamp_min(1.0)
    return (numerator / denominator).sqrt()
