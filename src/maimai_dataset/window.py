"""Shared chart-tick and window layout; independent of model/training code."""
from itertools import accumulate

CHART_TICKS_PER_BEAT = 96
TARGET_BEATS = 16
HISTORY_BEATS = 4
HISTORY_TICKS = HISTORY_BEATS * CHART_TICKS_PER_BEAT
TARGET_TICKS = TARGET_BEATS * CHART_TICKS_PER_BEAT
AUDIO_BEATS_BEFORE = 4
AUDIO_BEATS_AFTER = 2
AUDIO_CONTEXT_BEATS = AUDIO_BEATS_BEFORE + TARGET_BEATS + AUDIO_BEATS_AFTER
AUDIO_CENTERS_PER_BEAT = 24
WINDOW_POSITIONS = AUDIO_CONTEXT_BEATS * AUDIO_CENTERS_PER_BEAT
GRID_RATES = (8, 4, 2, 1)
GRID_LENGTHS = tuple(AUDIO_CONTEXT_BEATS * rate for rate in GRID_RATES)
GRID_OFFSETS = tuple(accumulate((0, *GRID_LENGTHS)))
WINDOW = {"schema": "maimai.window.v1", "ticks_per_beat": CHART_TICKS_PER_BEAT,
          "target_beats": TARGET_BEATS, "history_beats": HISTORY_BEATS,
          "audio_before_beats": AUDIO_BEATS_BEFORE, "audio_after_beats": AUDIO_BEATS_AFTER,
          "audio_positions": WINDOW_POSITIONS}


def training_window_count(last_onset_tick: int, song_end_tick: int) -> int:
    """Cover complete objects and their onsets, then append one empty window.

    An object ending exactly at a boundary needs no following coverage window;
    an onset at that boundary belongs to the following left-closed window.
    """
    occupied = last_onset_tick // TARGET_TICKS + 1
    complete = (song_end_tick + TARGET_TICKS - 1) // TARGET_TICKS
    return max(1, occupied, complete) + 1
