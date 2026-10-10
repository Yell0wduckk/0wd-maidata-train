"""BPM96 grids, candidate supports, and quantization audits."""

from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from maimai_baseline.time.coordinates import seconds_at_ticks
from maimai_dataset.timing import TimingMap
from typing import Any
import hashlib
import json
import math
import numpy as np


CHART_TICKS_PER_BEAT = 96


AUDIO_CENTERS_PER_BEAT = 24


AUDIO_GRID_TICKS = 4


TIME_CANDIDATES_PER_BEAT = 96


TIME_GRID_TICKS = 1


from maimai_dataset.window import WINDOW_POSITIONS, AUDIO_BEATS_BEFORE, WINDOW


DELTA_MS = "delta_ms"


GENERATION_CLOCK = "absolute-chart-tick1-centers-linear-ms-cdf"


SAMPLING = {
    "schema": "maimai.bpm96_sampling.v1", "centers_per_quarter": 24,
    "phase": "absolute-k/24", "sample_rounding": "half-up-float64",
    "sample_rate": 24000, "n_fft": 1024, "hann_periodic": True,
    "n_mels": 128, "f_min": 30.0, "f_max": 12000.0,
    "boundary": "reflect-waveform;outside-center-padding",
    "downsampling": [24, 12, 6, 3],
}


INTEGRATION = {
    "schema": "maimai.bpm96_linear_ms.v2", "field": DELTA_MS,
    "chart_ticks_per_quarter": CHART_TICKS_PER_BEAT,
    "candidates_per_quarter": TIME_CANDIDATES_PER_BEAT,
    "candidate_step_ticks": TIME_GRID_TICKS,
    "candidate_phase": "absolute-k/96",
    "audio_query": "linear-between-BPM96-centers-in-beat-coordinate;masked-padding",
    "unit_scale_ms": 1000.0, "minimum_scale_ms": 2.5, "components": 5,
    "cells": "beat-midpoints-mapped-through-full-timing",
    "normalization": "sum-of-legal-cell-masses", "rounding": "absolute-half-up",
    "song_boundary": "inclusive-candidate-centers;midpoint-cells-clipped-to-window",
    "durations_changed": False,
}


def identity(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def timing_identity(timing: TimingMap) -> dict[str, Any]:
    # Redundant equal-BPM controls and meter/HS metadata do not change sampling.
    points = []
    for p in timing.points:
        if not points or points[-1][2] != p.bpm:
            points.append([p.beat.numerator, p.beat.denominator, p.bpm])
    return {"first_seconds": timing.first_seconds, "bpm": points}


def half_up(value: Fraction | float) -> int:
    if not math.isfinite(float(value)):
        raise ValueError("non-finite grid coordinate")
    return math.floor(value + Fraction(1, 2))


def quantized_tick(tick: int | Fraction) -> int:
    return TIME_GRID_TICKS * half_up(Fraction(tick) / TIME_GRID_TICKS)


def first_candidate_tick(tick: int) -> int:
    return math.ceil(Fraction(tick) / TIME_GRID_TICKS) * TIME_GRID_TICKS


def grid_coordinates(timing: TimingMap, first_k: int, count: int, num_samples: int):
    k = np.arange(first_k, first_k + count, dtype=np.int64)
    seconds = seconds_at_ticks(timing, k * AUDIO_GRID_TICKS)
    samples = np.floor(seconds * 24000.0 + 0.5).astype(np.int64)
    valid = (seconds >= 0) & (seconds < num_samples / 24000.0) & (samples >= 0) & (samples < num_samples)
    return {"grid_k": k, "seconds": seconds, "samples": samples, "valid": valid}


def audio_grid(timing: TimingMap, num_samples: int):
    if num_samples < 2:
        raise ValueError("BPM features require at least two PCM samples")
    first = math.ceil(float(timing.second_to_beat(0.0)) * AUDIO_CENTERS_PER_BEAT - 1e-9)
    last = math.ceil(float(timing.second_to_beat(num_samples / 24000.0)) * AUDIO_CENTERS_PER_BEAT - 1e-9)
    return grid_coordinates(timing, first, last - first, num_samples)


def window_first_k(start_tick: int) -> int:
    if start_tick % AUDIO_GRID_TICKS:
        raise ValueError("target window start is not on the BPM96 grid")
    return start_tick // AUDIO_GRID_TICKS - AUDIO_BEATS_BEFORE * AUDIO_CENTERS_PER_BEAT


@dataclass(frozen=True)
class Candidates:
    ticks: np.ndarray
    seconds: np.ndarray
    delta_ms: np.ndarray
    lower_ms: np.ndarray
    upper_ms: np.ndarray

    def target_index(self, tick: int) -> int:
        q = quantized_tick(tick)
        where = np.flatnonzero(self.ticks == q)
        if len(where) != 1:
            raise ValueError(f"BPM96 target crosses legal window/order boundary: {tick} -> {q}")
        return int(where[0])


def candidates(timing: TimingMap, *, anchor_tick: int, start_tick: int,
               end_tick: int, allow_zero: bool, song_end_tick: int | None = None) -> Candidates:
    """Legal absolute centers in [start,end), with the original teacher anchor.

    The integration domain starts at the anchor. It is truncated at the target
    window edge; midpoint cells retain their true (possibly unequal) widths.
    """
    minimum = max(start_tick, anchor_tick + (0 if allow_zero else 1))
    ticks = np.arange(first_candidate_tick(minimum), end_tick, TIME_GRID_TICKS, dtype=np.int64)
    if song_end_tick is not None:
        ticks = ticks[ticks <= song_end_tick]
    if not len(ticks):
        raise ValueError("empty BPM96 candidate support")
    anchor_second = seconds_at_ticks(timing, [anchor_tick])[0]
    seconds = seconds_at_ticks(timing, ticks)
    lower_ticks = np.maximum(ticks - TIME_GRID_TICKS / 2, max(start_tick, anchor_tick))
    upper_ticks = np.minimum(ticks + TIME_GRID_TICKS / 2, end_tick)
    # Song end restricts emitted centers, not latent values inside their cells.
    # This also gives a terminal first-zero moment a nonzero half-cell mass.
    lower = (seconds_at_ticks(timing, lower_ticks) - anchor_second) * 1000
    upper = (seconds_at_ticks(timing, upper_ticks) - anchor_second) * 1000
    if np.any(upper <= lower):
        raise ValueError("empty BPM96 probability cell")
    return Candidates(ticks, seconds, (seconds - anchor_second) * 1000, lower, upper)


def quantization_audit(timing: TimingMap, moments, *, start_tick: int, end_tick: int):
    """Preserve every source identity; return ALL conflicts without repairing."""
    rows, conflicts = [], []
    previous = None
    for source, tick in moments:
        tick = Fraction(tick)
        q = quantized_tick(tick)
        record = {"source_row": source, "original_tick": [tick.numerator, tick.denominator],
                  "grid_tick": q, "error_ms": (timing.beat_to_second(Fraction(q, 96)) -
                    timing.beat_to_second(tick / 96)) * 1000}
        if previous is not None:
            original_interval = (tick - previous[0]) / 96
            record["interval_beats"] = [original_interval.numerator, original_interval.denominator]
            record["interval_change_ms"] = record["error_ms"] - previous[2]["error_ms"]
            if q <= previous[1]:
                conflicts.append({"kind": "same-grid" if q == previous[1] else "order", "previous": previous[2], "current": record})
        if not start_tick <= q < end_tick:
            conflicts.append({"kind": "boundary", "start_tick": start_tick, "end_tick": end_tick, "current": record})
        rows.append(record)
        previous = tick, q, record
    return {"rows": rows, "conflicts": conflicts}
