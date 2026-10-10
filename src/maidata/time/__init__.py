"""Chart clocks and written-BPM durations, independent of audio and models."""
from .timing import TimingMap, TICKS_PER_BEAT, duration_end, advance_ticks, split_duration
from .support import DurationCell, duration_cells, duration_target_bucket

__all__ = ['TimingMap', 'TICKS_PER_BEAT', 'duration_end', 'advance_ticks', 'split_duration',
           'DurationCell', 'duration_cells', 'duration_target_bucket']
