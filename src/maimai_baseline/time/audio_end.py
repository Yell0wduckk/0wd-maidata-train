"""Audio-sample boundary, independent of the source chart's End marker."""
from __future__ import annotations
import math
from dataclasses import dataclass
from maimai_baseline.time.coordinates import chart_tick_to_second

CONTRACT = {"schema": "maimai.audio_end.written_bpm.v2", "source": "decoded-samples/sample-rate",
    "onset": "strictly-before-audio-end", "duration": "endpoint-at-or-before-audio-end",
    "slide_wait_ticks": 96, "export_rest_beats": [1, 24]}

@dataclass(frozen=True)
class AudioEnd:
    seconds: float
    onset_tick: int
    tail_tick: int

def audio_end(timing, num_samples, sample_rate=24000):
    if not isinstance(num_samples, int) or num_samples <= 0 or sample_rate <= 0:
        raise ValueError("audio boundary requires positive sample count and rate")
    cached = getattr(timing, "_offline_audio_ends", None)
    if cached is not None:
        if sample_rate != 24000 or str(num_samples) not in cached:
            raise ValueError("training audio boundary absent from accepted timing table")
        return AudioEnd(**cached[str(num_samples)])
    seconds = num_samples / sample_rate
    estimate = math.floor(timing.second_to_beat(seconds) * 96)
    # Check in sample coordinates to avoid losing an exact grid boundary to FP rounding.
    def sample(tick):
        return chart_tick_to_second(timing, tick) * sample_rate
    tolerance = 1e-7
    while sample(estimate) > num_samples + tolerance:
        estimate -= 1
    while sample(estimate + 1) <= num_samples + tolerance:
        estimate += 1
    onset = estimate - int(abs(sample(estimate) - num_samples) <= tolerance)
    return AudioEnd(seconds, onset, estimate)

def duration_upper(field, anchor_tick, tail_tick=None, *, timing, end_seconds=None):
    from maimai_dataset.event_time import maximum_duration, seconds
    if end_seconds is None:
        end_seconds = seconds(timing, tail_tick)
    return maximum_duration(timing, anchor_tick, end_seconds, slide=field == "branch_duration_tick")


def export_end_beat(rows, timing, seconds):
    """Export-only E after materialized tails, audio, and timing controls."""
    from fractions import Fraction
    from maimai_dataset.event_time import duration_end
    tick = latest_tick = 0
    semantic_end = seconds
    for row in rows:
        if row.get("event") == "MOMENT":
            tick += int(row["delta_tick"])
        latest_tick = max(latest_tick, tick)
        if "hold_duration_tick" in row:
            semantic_end = max(semantic_end, duration_end(timing, tick, int(row["hold_duration_tick"])))
        if "branch_duration_tick" in row:
            semantic_end = max(semantic_end, duration_end(timing, tick, int(row["branch_duration_tick"]), slide=True))
    latest = max([Fraction(latest_tick, 96), Fraction(str(timing.second_to_beat(semantic_end))),
                  *(p.beat for p in timing.points), *(p[0] for p in timing.meter_points),
                  *(p[0] for p in timing.hs_points)])
    # E is export-only. Audio sample precision must not become simai subdivisions.
    return Fraction(math.ceil(latest * 96), 96) + Fraction(1, 24)
