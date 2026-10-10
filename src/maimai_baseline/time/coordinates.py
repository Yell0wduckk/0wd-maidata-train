"""Exact timing coordinates and process-local coordinate caches."""

from __future__ import annotations
from collections import OrderedDict
from fractions import Fraction
from maimai_dataset.encoding import CHART_TICKS_PER_BEAT
from maimai_dataset.timing import TimingMap
from threading import RLock
import math
import numpy as np
import os
import torch


BLOCK_SIZE = 2048  # Half ticks: centers and CDF edges use the same blocks.


MAX_BLOCKS = 4096  # 64 MiB of coordinate arrays, plus bounded index metadata.


_blocks = OrderedDict()


_pid = os.getpid()


_lock = RLock()


_hits = _misses = 0


def clear_coordinate_cache():
    global _pid, _hits, _misses
    with _lock:
        _blocks.clear()
        _pid = os.getpid()
        _hits = _misses = 0


def _after_fork():
    # Do not inherit a lock held by another thread in the parent.
    global _lock
    _lock = RLock()
    clear_coordinate_cache()


def coordinate_cache_info():
    with _lock:
        if _pid != os.getpid():
            clear_coordinate_cache()
        return dict(pid=_pid, blocks=len(_blocks), bytes=len(_blocks) * BLOCK_SIZE * 8,
                    hits=_hits, misses=_misses, max_blocks=MAX_BLOCKS, block_size=BLOCK_SIZE)


def _scalar_seconds(timing, ticks):
    return np.asarray([timing.beat_to_second(Fraction(t.item() if isinstance(t, np.generic) else t)
                                            / 96) for t in ticks], dtype=np.float64)


def seconds_at_ticks(timing, ticks):
    """Exact lookup for safely representable half ticks; scalar fallback otherwise.

    Object/Fraction inputs retain their exact rational value. In particular, a
    Fraction nearly equal to a half tick must not be rounded onto the fast path.
    Returned arrays never alias the immutable cache blocks.
    """
    global _hits, _misses
    table = getattr(timing, "_offline_table", None)
    if table is not None:
        values = np.asarray(list(ticks) if not isinstance(ticks, (np.ndarray, list, tuple)) else ticks)
        if values.ndim != 1 or values.dtype.kind not in "iuf" or not np.isfinite(values).all():
            raise ValueError("training timing coordinate is not a finite numeric vector")
        halves = values.astype(np.float64) * 2
        indices = halves.astype(np.int64) - table[1]
        if (not np.array_equal(halves, halves.astype(np.int64)) or
                np.any(indices < 0) or np.any(indices >= len(table[0]))):
            raise ValueError("training coordinate absent from accepted half-tick table")
        return np.asarray(table[0][indices], dtype=np.float64)
    if not isinstance(ticks, (np.ndarray, list, tuple)):
        ticks = list(ticks)
    values = np.asarray(ticks)
    if (values.ndim != 1 or values.dtype.kind not in "iuf" or values.dtype.itemsize > 8 or
            (values.dtype.kind == "f" and values.dtype.type not in (np.float16, np.float32, np.float64)) or
            np.any(~np.isfinite(values)) or np.any(np.abs(values.astype(np.float64)) >= 2**51)):
        return _scalar_seconds(timing, ticks)
    halves = values.astype(np.float64) * 2
    indices = halves.astype(np.int64)
    if not np.array_equal(halves, indices):
        return _scalar_seconds(timing, ticks)
    # Include every resolved control, even redundant BPM controls: accumulated
    # point.second rounding is part of the original calculation's identity.
    key = (timing.first_seconds, tuple((p.beat, p.bpm, p.second) for p in timing.points))
    block_ids = indices // BLOCK_SIZE
    result = np.empty(values.shape, dtype=np.float64)
    with _lock:
        if _pid != os.getpid():
            clear_coordinate_cache()
        for block_id in np.unique(block_ids):
            block_id = int(block_id)
            cache_key = (key, block_id)
            block = _blocks.get(cache_key)
            if block is None:
                _misses += 1
                start = block_id * BLOCK_SIZE
                block = _scalar_seconds(timing, np.arange(start, start + BLOCK_SIZE, dtype=np.int64) / 2)
                block.setflags(write=False)
                _blocks[cache_key] = block
                while len(_blocks) > MAX_BLOCKS:
                    _blocks.popitem(last=False)
            else:
                _hits += 1
                _blocks.move_to_end(cache_key)
            selected = block_ids == block_id
            result[selected] = block[indices[selected] - block_id * BLOCK_SIZE]
    return result


def half_up_nonnegative(value: float) -> int:
    """Round a nonnegative value for evaluation counters."""
    if not math.isfinite(value) or value < -1e-9:
        raise ValueError(f"rounding requires a finite non-negative value: {value}")
    return int(math.floor(max(0.0, float(value)) + 0.5))


def chart_tick_to_second(timing: TimingMap, tick: int | float) -> float:
    rounded = int(round(float(tick)))
    if not math.isclose(float(tick), float(rounded), abs_tol=1e-5):
        raise ValueError(f"chart anchor is not an integer tick: {tick!r}")
    if getattr(timing, "_offline_table", None) is not None:
        return float(seconds_at_ticks(timing, [rounded])[0])
    return float(timing.beat_to_second(Fraction(rounded, CHART_TICKS_PER_BEAT)))


def ticks_to_seconds(timing: TimingMap, ticks: torch.Tensor) -> torch.Tensor:
    values = [
        chart_tick_to_second(timing, value)
        for value in ticks.detach().cpu().reshape(-1).tolist()
    ]
    return torch.tensor(values, dtype=torch.float32).reshape(ticks.shape)


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)
