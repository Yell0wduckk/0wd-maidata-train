"""Tensor contracts, independent of Encoder, Draft and Physical implementations."""
from dataclasses import dataclass, fields
import torch
from torch import Tensor
from .checks import require


@dataclass(frozen=True)
class Memory:
    values: Tensor
    beats: Tensor
    valid: Tensor


def validate_memory(memory, width):
    x, b, v = memory.values, memory.beats, memory.valid
    if (x.ndim != 3 or x.shape[-1] != width or b.shape != x.shape[:2]
            or v.shape != b.shape or v.dtype != torch.bool
            or b.device != x.device or v.device != x.device):
        raise ValueError('memory requires values [B,L,D], beats/valid [B,L]')
    if not x.is_floating_point():
        raise ValueError('valid memory values/positions must be finite')
    require((torch.isfinite(x) | ~v[..., None]).all() & (torch.isfinite(b) | ~v).all(),
            'valid memory values/positions must be finite')


@dataclass(frozen=True)
class TokenBatch:
    types: Tensor
    beats: Tensor
    valid: Tensor
    kind: Tensor
    pad: Tensor
    flags: Tensor  # [B,L,5]; -1 = NA, never class zero
    route: Tensor
    track_break: Tensor
    time: Tensor  # delta seconds for M; beats for durations

    slide_beats: Tensor | None = None

    def __post_init__(self):
        if self.slide_beats is None:
            object.__setattr__(self, "slide_beats", torch.where(self.types >= 5,self.beats,self.beats + 1))

    def slice(self, start, stop):
        return TokenBatch(**{f.name: getattr(self, f.name)[:, start:stop] for f in fields(self)})

    def to(self, device):
        return TokenBatch(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})


@dataclass(frozen=True)
class KV:
    key: Tensor
    value: Tensor
    valid: Tensor
    query_layout: dict | None = None


@dataclass(frozen=True)
class DecoderMemory:
    audio: tuple[KV, ...]
    condition: tuple[KV, ...]
    plan_index: int
    row_indices: Tensor | None = None
    condition_beats: Tensor | None = None


@dataclass(frozen=True)
class DecoderCache:
    """Append-only self K/V. Restore by retaining a previous immutable object."""
    layers: tuple[KV, ...]
    memory: DecoderMemory  # prevents mixing self caches from different windows


@dataclass(frozen=True)
class RouterOutput:
    hidden: Tensor
    logits: Tensor
    beats: Tensor
    valid: Tensor
    front_kv: tuple[KV, ...]
    memory: DecoderMemory
    previous: DecoderCache | None
    slide_beats: Tensor


@dataclass(frozen=True)
class DecoderOutput:
    type_logits: Tensor
    hidden: Tensor
    cache: DecoderCache
