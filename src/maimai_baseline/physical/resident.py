from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Mapping

import numpy as np
import torch

from maimai_baseline.physical.bitplane_cache import PhysicalBitplaneStore
from maimai_baseline.physical.contract import VALID_REGION_MASK


@dataclass
class PhysicalDeviceCache:
    tensors: Mapping[str, torch.Tensor]
    resident_bytes: int
    load_seconds: float
    rebase_seconds: float
    h2d_seconds: float
    plane_scratch: torch.Tensor | None = None

    def materialization_buffer(self, value_count: int) -> torch.Tensor:
        """Return the reusable batch-plane workspace for this process/device."""

        count = int(value_count)
        if count < 0:
            raise ValueError("Physical plane materialization size must be non-negative")
        if self.plane_scratch is None or self.plane_scratch.numel() < count:
            self.plane_scratch = torch.empty(
                (count,),
                device=self.tensors["plane_low_bytes"].device,
                dtype=torch.long,
            )
        return self.plane_scratch[:count]


_ACTIVE_STORE: PhysicalBitplaneStore | None = None
_DEVICE_CACHES: dict[tuple[int, str], PhysicalDeviceCache] = {}


def _pack_plane_masks_low41(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pack uint64 carriers into five low bytes plus one high-bit bitmap."""

    count = int(values.shape[0])
    low = np.empty((5, count), dtype=np.uint8)
    high = np.zeros(((count + 7) // 8,), dtype=np.uint8)
    chunk_size = 1 << 20  # multiple of eight keeps bitmap writes byte-aligned
    for begin in range(0, count, chunk_size):
        end = min(count, begin + chunk_size)
        chunk = np.asarray(values[begin:end], dtype=np.uint64)
        if np.bitwise_and(chunk, np.uint64(~VALID_REGION_MASK & ((1 << 64) - 1))).any():
            raise RuntimeError("Physical resident plane contains bits above region 40")
        for byte in range(5):
            low[byte, begin:end] = (chunk >> np.uint64(byte * 8)).astype(
                np.uint8, copy=False
            )
        packed = np.packbits(
            ((chunk >> np.uint64(40)) & np.uint64(1)).astype(np.uint8),
            bitorder="little",
        )
        byte_begin = begin // 8
        high[byte_begin : byte_begin + packed.shape[0]] = packed
    return low.reshape(-1), high


def bind_bitplane_store(store: PhysicalBitplaneStore | None) -> None:
    global _ACTIVE_STORE
    _ACTIVE_STORE = store
    _DEVICE_CACHES.clear()


def active_bitplane_store() -> PhysicalBitplaneStore:
    if _ACTIVE_STORE is None:
        raise RuntimeError("Physical bitplane store is not bound")
    return _ACTIVE_STORE


def _resident_host_arrays(
    store: PhysicalBitplaneStore,
) -> tuple[dict[str, np.ndarray], float]:
    started = time.perf_counter()
    offsets = store.offsets
    arrays = store.arrays
    chart_count = int(offsets["slot_counts"].shape[0])

    source_action = np.array(arrays["source_action"], dtype=np.int64, copy=True)
    source_moment = np.array(arrays["source_moment"], dtype=np.int64, copy=True)
    carry_indices = np.array(arrays["carry_indices"], dtype=np.int64, copy=True)
    carry_offsets = np.empty((int(offsets["moment_offsets"][-1]) + 1,), dtype=np.int64)
    for chart in range(chart_count):
        action_base = int(offsets["action_offsets"][chart])
        source_begin = int(offsets["source_offsets"][chart])
        source_end = int(offsets["source_offsets"][chart + 1])
        local_source = source_action[source_begin:source_end]
        valid = local_source >= 0
        local_source[valid] += action_base

        carry_begin = int(offsets["carry_index_offsets"][chart])
        carry_end = int(offsets["carry_index_offsets"][chart + 1])
        carry_indices[carry_begin:carry_end] += action_base
        local_offsets_begin = int(offsets["carry_offsets_offsets"][chart])
        local_offsets_end = int(offsets["carry_offsets_offsets"][chart + 1])
        moment_begin = int(offsets["moment_offsets"][chart])
        moment_end = int(offsets["moment_offsets"][chart + 1])
        local_offsets = arrays["carry_offsets"][local_offsets_begin:local_offsets_end]
        if int(local_offsets.shape[0]) != moment_end - moment_begin + 1:
            raise RuntimeError("Physical resident carry/moment offsets mismatch")
        carry_offsets[moment_begin : moment_end + 1] = (
            np.asarray(local_offsets, dtype=np.int64) + carry_begin
        )

    timing_counts = np.diff(offsets["timing_offsets"])
    timing_width = max(1, int(timing_counts.max(initial=0)))
    point_ticks = np.zeros((chart_count, timing_width), dtype=np.float64)
    point_seconds = np.zeros_like(point_ticks)
    point_bpms = np.ones_like(point_ticks)
    point_padding = np.ones((chart_count, timing_width), dtype=np.bool_)
    point_constant = np.zeros(chart_count, dtype=np.bool_)
    for chart in range(chart_count):
        begin = int(offsets["timing_offsets"][chart])
        end = int(offsets["timing_offsets"][chart + 1])
        count = end - begin
        point_ticks[chart, :count] = arrays["timing_ticks"][begin:end]
        point_seconds[chart, :count] = arrays["timing_seconds"][begin:end]
        point_bpms[chart, :count] = arrays["timing_bpms"][begin:end]
        point_padding[chart, :count] = False
        if count:
            expected = point_seconds[chart, 0] + (point_ticks[chart, :count] - point_ticks[chart, 0]) * 60.0 / (point_bpms[chart, 0] * 96.0)
            point_constant[chart] = bool(np.all(point_bpms[chart, :count] == point_bpms[chart, 0]) and np.array_equal(point_seconds[chart, :count], expected))

    plane_low_bytes, plane_high_bits = _pack_plane_masks_low41(arrays["planes"])
    raw_actions = arrays["action_values"]
    int32 = np.iinfo(np.int32)
    for column in range(int(raw_actions.shape[1])):
        if column == 4:  # region mask is supplied by action_masks_by_view
            continue
        values = raw_actions[:, column]
        if values.size and (
            int(values.min()) < int32.min or int(values.max()) > int32.max
        ):
            raise RuntimeError(
                f"Physical resident action column {column} exceeds int32"
            )
    compact_actions = np.asarray(raw_actions, dtype=np.int32).copy()
    canonical_action_masks = np.asarray(raw_actions[:, 4], dtype=np.int64).copy()
    compact_actions[:, 4] = 0

    host = {
        "plane_low_bytes": plane_low_bytes,
        "plane_high_bits": plane_high_bits,
        "plane_offsets": offsets["plane_offsets"],
        "slot_origins": offsets["slot_origins"],
        "slot_counts": offsets["slot_counts"],
        "action_values": compact_actions,
        "canonical_action_masks": canonical_action_masks,
        "action_floating": arrays["action_floating"],
        "action_offsets": offsets["action_offsets"],
        "source_action": source_action,
        "source_moment": source_moment,
        "source_offsets": offsets["source_offsets"],
        "moment_rows": np.asarray(arrays["moment_rows"], dtype=np.int64),
        "moment_ticks": arrays["moment_ticks"],
        "moment_slots": arrays["moment_slots"],
        "moment_offsets": offsets["moment_offsets"],
        "carry_offsets": carry_offsets,
        "carry_indices": carry_indices,
        "point_ticks": point_ticks,
        "point_seconds": point_seconds,
        "point_bpms": point_bpms,
        "point_padding_mask": point_padding,
        "point_constant": point_constant,
    }
    return host, time.perf_counter() - started


def _upload(
    array: np.ndarray,
    device: torch.device,
    *,
    preserve_uint8: bool = False,
    preserve_int32: bool = False,
) -> torch.Tensor:
    # Copy mmap/read-only arrays into owned storage exactly once.  This avoids
    # passing mmap-backed tensors to CUDA while keeping all batch workers CPU-only.
    owned = np.array(array, copy=True, order="C")
    tensor = torch.from_numpy(owned)
    if (
        (tensor.dtype == torch.int32 and not preserve_int32)
        or tensor.dtype
        in {
            torch.int16,
            torch.int8,
        }
        or (tensor.dtype == torch.uint8 and not preserve_uint8)
    ):
        tensor = tensor.to(torch.long)
    return tensor.to(device=device, non_blocking=False).contiguous()


def active_device_cache(device: torch.device | str) -> PhysicalDeviceCache:
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise RuntimeError("Physical resident cache requires CUDA")
    resolved = torch.device(
        "cuda",
        torch.cuda.current_device() if resolved.index is None else resolved.index,
    )
    key = (os.getpid(), str(resolved))
    cached = _DEVICE_CACHES.get(key)
    if cached is not None:
        return cached
    store = active_bitplane_store()
    load_started = time.perf_counter()
    host, rebase_seconds = _resident_host_arrays(store)
    h2d_started = time.perf_counter()
    tensors = {
        name: _upload(
            value,
            resolved,
            preserve_uint8=name in {"plane_low_bytes", "plane_high_bits"},
            preserve_int32=name == "action_values",
        )
        for name, value in host.items()
    }
    from maimai_baseline.physical.checker import load_cuda_extension
    from maimai_baseline.physical.spatial import region_byte_lut

    extension = load_cuda_extension()
    tensors["action_masks_by_view"] = extension.action_mask_views_v2(
        tensors.pop("canonical_action_masks"), region_byte_lut(resolved)
    )
    torch.cuda.synchronize(resolved)
    h2d_seconds = time.perf_counter() - h2d_started
    resident_bytes = sum(
        int(value.numel() * value.element_size()) for value in tensors.values()
    )
    cached = PhysicalDeviceCache(
        tensors=tensors,
        resident_bytes=resident_bytes,
        load_seconds=time.perf_counter() - load_started,
        rebase_seconds=rebase_seconds,
        h2d_seconds=h2d_seconds,
    )
    _DEVICE_CACHES[key] = cached
    return cached


def clear_device_caches() -> None:
    _DEVICE_CACHES.clear()


__all__ = [
    "PhysicalDeviceCache",
    "active_bitplane_store",
    "active_device_cache",
    "bind_bitplane_store",
    "clear_device_caches",
]
