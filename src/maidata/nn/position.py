"""Continuous positions in beats, with FP32 phases and adjacent pairs."""
import torch
from torch import Tensor


def apply_rope(value: Tensor, beats: Tensor, base: float, scale: float) -> Tensor:
    with torch.autocast(device_type=value.device.type, enabled=False):
        frequencies = base ** (
            -torch.arange(0, value.shape[-1], 2, device=value.device, dtype=torch.float32)
            / value.shape[-1]
        )
        phase = beats.float()[:, None, :, None] * scale * frequencies
        cosine, sine = phase.cos(), phase.sin()
        even, odd = value.float()[..., 0::2], value.float()[..., 1::2]
        rotated = torch.stack((even * cosine - odd * sine,
                               even * sine + odd * cosine), dim=-1).flatten(-2)
    return rotated.to(value.dtype)


def sinusoidal(beats: Tensor, width: int, base: float, scale: float) -> Tensor:
    with torch.autocast(device_type=beats.device.type, enabled=False):
        frequency = base ** (-torch.arange(0, width, 2, device=beats.device,
                                         dtype=torch.float32) / width)
        phase = beats.float()[..., None] * scale * frequency
        return torch.stack((phase.sin(), phase.cos()), -1).flatten(-2)[..., :width]
