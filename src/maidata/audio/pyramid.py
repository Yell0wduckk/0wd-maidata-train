"""Local coordinate resampling and a three-scale top-down audio pyramid."""

import math

import torch
from torch import Tensor, nn

from .geometry import interpolate, mask_values, masked_upsample
from .types import EncoderOutput, ScaleMemory


class LocalPyramid(nn.Module):
    def __init__(self, d_model, fpn_rates, context_beats, audio_before_beats):
        super().__init__()
        if len(fpn_rates) != 3:
            raise ValueError("the local pyramid requires three sampling rates")
        lengths = [context_beats * rate for rate in fpn_rates]
        if any(not math.isfinite(length) or length <= 0 or int(length) != length for length in lengths):
            raise ValueError("each FPN rate must produce a positive integer grid length")
        self.fpn_rates = tuple(fpn_rates)
        self.lengths = tuple(int(length) for length in lengths)
        self.audio_before_beats = audio_before_beats
        self.projections = nn.ModuleList(nn.Linear(d_model, d_model) for _ in range(2))

    def forward(self, stages: list[ScaleMemory], target_start_beats: Tensor) -> EncoderOutput:
        if len(stages) != 3:
            raise ValueError("the local pyramid requires three fine-to-coarse stages")
        start = target_start_beats.float().unsqueeze(1) - self.audio_before_beats
        memories = []
        for stage, rate, length in zip(stages, self.fpn_rates, self.lengths, strict=True):
            grid = start + (
                torch.arange(length, device=start.device, dtype=torch.float32) + 0.5
            ) / rate
            memories.append(interpolate(stage, grid))
        result = list(memories)
        for index in (1, 0):
            target = memories[index]
            up = masked_upsample(result[index + 1], target.beats)
            branch = mask_values(self.projections[index](up.values).to(target.values.dtype), up.valid)
            result[index] = ScaleMemory(
                mask_values(target.values + branch, target.valid), target.beats, target.valid
            )
        return EncoderOutput(*result)
