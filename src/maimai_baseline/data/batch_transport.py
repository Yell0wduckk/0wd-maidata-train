"""Copy only the current collated batch into a few dtype-contiguous IPC buffers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from maimai_dataset.timing import TimingMap


def _describe(value: Any, leaves: dict[torch.dtype, list[torch.Tensor]],
              offsets: dict[torch.dtype, int]) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.contiguous().reshape(-1)
        group = leaves.setdefault(value.dtype, [])
        offset = offsets.get(value.dtype, 0)
        group.append(tensor)
        offsets[value.dtype] = offset + value.numel()
        return ("tensor", str(value.dtype), offset, value.numel(), tuple(value.shape))
    if isinstance(value, TimingMap):
        root = getattr(value, "_offline_root", None)
        if getattr(value, "_offline_table", None) is not None and root is None:
            raise ValueError("offline TimingMap has no reopenable table reference")
        key = getattr(value, "_prepared_timing_key", None)
        if key is not None:
            return ("prepared_timing", key, value._prepared_timing_payload, root)
        return ("timing", value.to_dict(), root)
    if isinstance(value, dict):
        return ("dict", [(k, _describe(v, leaves, offsets)) for k, v in value.items()])
    if isinstance(value, list):
        return ("list", [_describe(v, leaves, offsets) for v in value])
    if isinstance(value, tuple):
        return ("tuple", [_describe(v, leaves, offsets) for v in value])
    if value is None or type(value) in (bool, int, float, str):
        return ("scalar", value)
    raise TypeError(f"unsupported batch transport scalar: {type(value)!r}")


def _restore(spec: Any, buffers: dict[str, torch.Tensor]) -> Any:
    kind = spec[0]
    if kind == "tensor":
        _, dtype, offset, length, shape = spec
        return buffers[dtype].narrow(0, offset, length).view(shape)
    if kind == "dict":
        return {k: _restore(v, buffers) for k, v in spec[1]}
    if kind == "list":
        return [_restore(v, buffers) for v in spec[1]]
    if kind == "tuple":
        return tuple(_restore(v, buffers) for v in spec[1])
    if kind == "timing":
        timing = TimingMap.from_dict(spec[1])
        if spec[2] is not None:
            from maimai_baseline.time.offline import store_for
            timing = store_for(spec[2]).bind(timing)
        return timing
    if kind == "prepared_timing":
        from maimai_baseline.time.offline import restore_prepared_timing
        return restore_prepared_timing(spec[1], spec[2], spec[3])
    if kind == "scalar":
        return spec[1]
    raise ValueError(f"invalid packed batch node: {kind!r}")


@dataclass
class PackedBatch:
    structure: Any
    buffers: dict[str, torch.Tensor]
    cpu_prepare_seconds: float = 0.0

    def pin_memory(self) -> "PackedBatch":
        return PackedBatch(self.structure, {k: v.pin_memory() for k, v in self.buffers.items()},
                           self.cpu_prepare_seconds)

    def unpack(self) -> dict[str, Any]:
        return _restore(self.structure, self.buffers)

    def to_device(self, device: torch.device) -> dict[str, Any]:
        copied = {k: v.to(device, non_blocking=True) for k, v in self.buffers.items()}
        return _restore(self.structure, copied)


def pack_batch(batch: dict[str, Any]) -> PackedBatch:
    leaves: dict[torch.dtype, list[torch.Tensor]] = {}
    structure = _describe(batch, leaves, {})
    buffers = {str(dtype): torch.cat(parts) for dtype, parts in leaves.items()}
    return PackedBatch(structure, buffers)
