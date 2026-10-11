"""Chart clocks and written-BPM durations, independent of audio and models."""
from .timing import TimingMap, TICKS_PER_BEAT, duration_end, advance_ticks, split_duration
from .support import DurationCell, duration_cells, duration_target_bucket
from .discrete import (TIME_BLOCK, TIME_CLASSES, CONTINUE, TIME_REPRESENTATION,
                       TimeQuery, TimeAdvance, TimeStep, TimePath, TimeTree,
                       UnsupportedDurationError, encode_ticks, decode_ticks,
                       minimum_ticks, compile_time_target, duration_cap,
                       local_z_indices, substep_masks, compile_support_tree)

__all__ = ['TimingMap', 'TICKS_PER_BEAT', 'duration_end', 'advance_ticks', 'split_duration',
           'DurationCell', 'duration_cells', 'duration_target_bucket',
           'TIME_BLOCK', 'TIME_CLASSES', 'CONTINUE', 'TIME_REPRESENTATION',
           'TimeQuery', 'TimeAdvance', 'TimeStep', 'TimePath', 'TimeTree',
           'UnsupportedDurationError', 'encode_ticks', 'decode_ticks',
           'minimum_ticks', 'compile_time_target', 'duration_cap',
           'local_z_indices', 'substep_masks', 'compile_support_tree']
