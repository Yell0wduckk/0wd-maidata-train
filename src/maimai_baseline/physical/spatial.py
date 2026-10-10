from __future__ import annotations

from typing import Final

import torch

from maimai_baseline.physical.contract import VALID_REGION_MASK


# Maimai lanes are clockwise. These are the exact four whole-chart views used
# by the Route-71 spatial cache.
LANE_PERMUTATIONS: Final = (
    (0, 1, 2, 3, 4, 5, 6, 7),
    (7, 6, 5, 4, 3, 2, 1, 0),
    (3, 2, 1, 0, 7, 6, 5, 4),
    (4, 5, 6, 7, 0, 1, 2, 3),
)
# D/E are centred on different angles from A/B/K. Match MiaCode's
# mapLaneTouchDE, which also produced the spatial chart cache.
TOUCH_DE_PERMUTATIONS: Final = (
    (0, 1, 2, 3, 4, 5, 6, 7),
    (0, 7, 6, 5, 4, 3, 2, 1),
    (4, 3, 2, 1, 0, 7, 6, 5),
    (4, 5, 6, 7, 0, 1, 2, 3),
)
_CPU_LUT: torch.Tensor | None = None
_DEVICE_LUTS: dict[str, torch.Tensor] = {}


def region_byte_lut(device: torch.device | str | None = None) -> torch.Tensor:
    """Cached uint8[view, ring family (A/B/K or D/E), byte] permutations."""
    global _CPU_LUT
    if _CPU_LUT is not None:
        cpu = _CPU_LUT
    else:
        cpu = torch.zeros((4, 2, 256), dtype=torch.uint8)
        for family, permutations in enumerate((LANE_PERMUTATIONS, TOUCH_DE_PERMUTATIONS)):
            for view, permutation in enumerate(permutations):
                for source in range(256):
                    target = 0
                    for lane in range(8):
                        if source & (1 << lane):
                            target |= 1 << permutation[lane]
                    cpu[view, family, source] = target
        _CPU_LUT = cpu
    if device is None or torch.device(device).type == "cpu":
        return cpu
    key = str(torch.device(device))
    cached = _DEVICE_LUTS.get(key)
    if cached is None:
        cached = cpu.to(device=device)
        _DEVICE_LUTS[key] = cached
    return cached


def transform_mask(mask: int, view: int) -> int:
    if int(view) == 0:
        return int(mask) & VALID_REGION_MASK
    result = int(mask) & (1 << 16)  # C is invariant.
    for offset in (0, 8, 17, 25, 33):
        permutation = (TOUCH_DE_PERMUTATIONS if offset in (17, 25) else LANE_PERMUTATIONS)[int(view)]
        byte = (int(mask) >> offset) & 0xFF
        transformed = 0
        for lane in range(8):
            if byte & (1 << lane):
                transformed |= 1 << permutation[lane]
        result |= transformed << offset
    return result & VALID_REGION_MASK


__all__ = ["LANE_PERMUTATIONS", "TOUCH_DE_PERMUTATIONS", "region_byte_lut", "transform_mask"]
