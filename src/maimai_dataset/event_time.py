"""Written-BPM durations and exact global-beat interval boundaries.

No 10 ms quantization and no Physical release margin in semantic endpoints.
"""
from __future__ import annotations
import math

IDENTITY = 'written-bpm-duration-global-wait-release22-v1'
TICKS_PER_BEAT = 96
RELEASE_TICKS = 22


def seconds(timing, tick):
    return timing.beat_to_second(float(tick) / TICKS_PER_BEAT)


def written_tick_seconds(timing, anchor):
    return 60.0 / (TICKS_PER_BEAT * timing.bpm_at_beat(float(anchor) / TICKS_PER_BEAT))


def duration_end(timing, anchor, duration, *, slide=False):
    if not math.isfinite(float(duration)) or duration < 0:
        raise ValueError('duration must be finite and nonnegative')
    return seconds(timing, anchor + (96 if slide else 0)) + duration * written_tick_seconds(timing, anchor)


def advance_global_ticks(timing, second, ticks):
    return timing.beat_to_second(timing.second_to_beat(second) + float(ticks) / 96)


def release_end(timing, anchor, duration, *, slide=False):
    return advance_global_ticks(timing, duration_end(timing, anchor, duration, slide=slide), RELEASE_TICKS)


def maximum_duration(timing, anchor, end_second, *, slide=False):
    start = seconds(timing, anchor + (96 if slide else 0))
    value = (float(end_second) - start) / written_tick_seconds(timing, anchor)
    # Only absorb arithmetic error at an exact integer boundary.
    nearest = round(value)
    if abs(value - nearest) <= 1e-9:
        value = nearest
    return math.floor(value)
