"""Public Draft tensors. True always means valid; no labels enter networks."""
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class BeatMemory:
    values: Tensor
    beats: Tensor
    valid: Tensor


def validate_memory(memory, width=None):
    x, b, v = memory.values, memory.beats, memory.valid
    if (x.ndim != 3 or x.shape[1] < 1 or v.shape != x.shape[:2]
            or b.shape != v.shape or v.dtype != torch.bool
            or (width is not None and x.shape[-1] != width)
            or b.device != x.device or v.device != x.device):
        raise ValueError('memory requires aligned values [N,L,D], beats/valid [N,L]')
    if not x.is_floating_point() or not (torch.isfinite(x) | ~v[..., None]).all() or not (torch.isfinite(b) | ~v).all():
        raise ValueError('valid memory values and beats must be finite')


@dataclass
class DraftInputs:
    query_beats: Tensor
    draft_valid: Tensor
    audio_mapping: Tensor
    difficulty_id: Tensor
    level: Tensor
    span_beats: Tensor
    valid_indices: Tensor | None = None


@dataclass
class ProjectedAudio:
    key_values: tuple[tuple[Tensor, Tensor], ...]
    valid: Tensor


@dataclass
class WindowLatents:
    memory: BeatMemory
    target_start: Tensor
    target_beats: int
    window_valid: Tensor


@dataclass
class PlannerOutput:
    continuous_logits: Tensor
    config_logits: Tensor
    plan_values: Tensor
    valid: Tensor


@dataclass
class ReconstructionOutput:
    field_logits: dict[str, Tensor]
    moment_logits: Tensor
    empty_logits: Tensor | None
    positive_count_logits: Tensor | None
    valid: Tensor
    query_valid: Tensor
    count_valid: Tensor
    count_values: Tensor | None = None

    @property
    def route_vocabulary(self):
        # The two supported readout widths have explicit, immutable meanings.
        width = self.field_logits['route_id'].shape[-1]
        if width == 37:
            return 'axis15-37'
        if width == 71:
            return 'route71'
        raise ValueError('unsupported Draft route readout width')


@dataclass
class ReconstructionTargets:
    fields: dict[str, Tensor]
    moment_tick: Tensor
    moment_counts: Tensor
    valid: Tensor
    queries: object = None


@dataclass
class ReconstructionQueries:
    """Allowed prefix conditions only; no labels, total rows, or future types."""
    roles: Tensor
    beats: Tensor
    origins: Tensor
    ordinals: Tensor
    tracks: Tensor
    note_kinds: Tensor
    route_ids: Tensor
    route_valid: Tensor
    valid: Tensor
    history_gaps: Tensor
    last_real_valid: Tensor
    note_ordinals: Tensor
    segments: Tensor
    current_route_ids: Tensor


@dataclass
class PlannerTargets:
    continuous: Tensor
    category: Tensor
    valid: Tensor
