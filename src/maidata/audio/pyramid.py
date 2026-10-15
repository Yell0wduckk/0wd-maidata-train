"""Local coordinate resampling and a three-scale top-down audio pyramid."""

import math

import torch
from torch import Tensor, nn

from .geometry import interpolate, mask_values, masked_upsample
from .types import EncoderOutput, ScaleMemory


class LocalPyramid(nn.Module):
    def __init__(self, d_model, fpn_rates, context_beats, audio_before_beats, *, p0_normalization):
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
        if type(p0_normalization) is not bool:
            raise ValueError('P0 normalization must be boolean')
        # One producer boundary shared by D and M. No learned gain can undo
        # the output scale control; P1/P2 and the fusion computation stay intact.
        self.p0_norm = (nn.LayerNorm(d_model, elementwise_affine=False)
                        if p0_normalization else nn.Identity())

    def forward(self, stages: list[ScaleMemory], target_start_beats: Tensor, layout=None) -> EncoderOutput:
        start = target_start_beats.float().unsqueeze(1) - self.audio_before_beats
        memories = []
        for index, (stage, rate, length) in enumerate(zip(stages, self.fpn_rates, self.lengths)):
            grid = start + (
                torch.arange(length, device=start.device, dtype=torch.float32) + 0.5
            ) / rate
            memories.append(interpolate(stage, grid, layout=None if layout is None else layout["resample"][index]))
        result = list(memories)
        for index in (1, 0):
            target = memories[index]
            up = masked_upsample(result[index + 1], target.beats, layout=None if layout is None else layout["upsample"][index])
            branch = mask_values(self.projections[index](up.values).to(target.values.dtype), up.valid)
            result[index] = ScaleMemory(
                mask_values(target.values + branch, target.valid), target.beats, target.valid
            )
        p0 = result[0]
        result[0] = ScaleMemory(mask_values(self.p0_norm(p0.values), p0.valid), p0.beats, p0.valid)
        return EncoderOutput(*result)
