"""Inference time cells with bounded, song-local half-tick coordinate reuse."""
from collections import OrderedDict
import math

import numpy as np
import torch

from maidata.decoder.supports import TimeCandidates


class TimeCandidateCache:
    """Reuse immutable timing coordinates, never prefix-dependent rule masks.

    The old inference path caches half-tick coordinates in blocks. Keep that
    strategy local to a song here; training supports and probability semantics
    are unchanged. Rule intervals are reapplied for every causal prefix.
    """
    BLOCK_SIZE = 2048
    MAX_BLOCKS = 16

    def __init__(self):
        self.timing = None
        self.blocks = OrderedDict()

    def seconds(self, timing, half_ticks):
        if timing is not self.timing:
            self.timing = timing
            self.blocks.clear()
        half_ticks = np.asarray(half_ticks, dtype=np.int64)
        result = np.empty(half_ticks.shape, dtype=np.float64)
        block_ids = half_ticks // self.BLOCK_SIZE
        for block_id in np.unique(block_ids):
            block_id = int(block_id)
            block = self.blocks.get(block_id)
            if block is None:
                start = block_id * self.BLOCK_SIZE
                # Match TimingMap.second's existing floating half-tick path,
                # including its rounding at non-integer BPM control points.
                block = np.fromiter((timing.second(float(i) / 2)
                    for i in range(start, start + self.BLOCK_SIZE)),
                    dtype=np.float64, count=self.BLOCK_SIZE)
                self.blocks[block_id] = block
                if len(self.blocks) > self.MAX_BLOCKS:
                    self.blocks.popitem(last=False)
            else:
                self.blocks.move_to_end(block_id)
            selected = block_ids == block_id
            result[selected] = block[half_ticks[selected] - block_id * self.BLOCK_SIZE]
        return result

    def candidates(self, stage, timing, *, kind, window_end, cap_tick, device):
        cursor = stage['cursor_tick']
        basic_tick_intervals = None
        if kind == 'MOMENT':
            if not cap_tick >= window_end > cursor:
                raise ValueError('MOMENT requires cursor < window_end <= U')
            support = stage['moment']
            ticks = np.arange(cursor + 1, math.ceil(cap_tick), dtype=np.int64)
            origin = timing.second(cursor)
            cap = 1000 * (timing.second(cap_tick) - origin)
            lower = np.append(1000 * (self.seconds(timing, ticks * 2 - 1) - origin), cap)
            upper = np.append(1000 * (self.seconds(timing, ticks * 2 + 1) - origin), math.inf)
            if len(ticks):
                upper[-2] = cap
            labels = (*map(int, ticks), None)
            representatives = (*map(int, ticks), cap_tick)
            clean = np.append(ticks >= window_end, True)
        else:
            support = stage['duration']
            lo, hi = support['domain']
            minimum = {'HOLD_DURATION': 2, 'SLIDE_DURATION': 1}.get(kind)
            if minimum is None or type(lo) is not int or type(hi) is not int or lo < minimum or hi < lo:
                raise ValueError('invalid duration support')
            buckets = np.arange((lo + 2) // 4, (hi + 2) // 4 + 1, dtype=np.int64)
            ticks = np.clip(buckets * 4, lo, hi)
            scale = 1000 * timing.written_tick_seconds(cursor) if kind == 'HOLD_DURATION' else 1.
            lower, upper = buckets.astype(np.float64) * 4 - 2, buckets.astype(np.float64) * 4 + 2
            lower[0], upper[-1] = lo - .5, hi + .5
            lower, upper = lower * scale, upper * scale
            labels, representatives = tuple(map(int, buckets)), tuple(map(int, ticks))
            basic_tick_intervals = tuple((max(lo, a), min(hi, b)) for a, b, record in support['intervals']
                if not record['basic'] and max(lo, a) <= min(hi, b))
            clean = np.zeros(len(ticks), dtype=bool)

        count = len(labels)
        basic = clean.copy()
        penalty = np.zeros((count, 4), dtype=bool)
        raw = np.zeros((count, 4), dtype=bool)
        # Sorted interval endpoints let NumPy map all cells in one operation.
        intervals = support['intervals']
        if intervals and len(ticks):
            starts = np.asarray([a for a, _, _ in intervals])
            ends = np.asarray([b for _, b, _ in intervals])
            indices = np.searchsorted(ends, ticks, side='left')
            clipped = np.minimum(indices, len(intervals) - 1)
            present = (indices < len(intervals)) & (starts[clipped] <= ticks) & ~clean[:len(ticks)]
            records = [r for _, _, r in intervals]
            valid = np.asarray([not r['basic'] for r in records], dtype=bool)
            basic[:len(ticks)] |= present & valid[clipped]
            for bit in range(4):
                for field, target in (('penalty', penalty), ('physical', raw)):
                    masks = np.asarray([bool(r.get(field, 0) & (1 << bit)) for r in records])
                    target[:len(ticks), bit] = present & valid[clipped] & masks[clipped]
        def tensor(values):
            return torch.from_numpy(values).to(device=device)
        return TimeCandidates(tensor(lower), tensor(upper), tensor(basic), tensor(penalty), tensor(raw),
            labels, representatives, basic_tick_intervals)
