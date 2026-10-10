"""Data-side duration buckets. No distribution heads or probability loss."""
from dataclasses import dataclass


@dataclass(frozen=True)
class DurationCell:
    bucket: int
    tick: int
    lower: float
    upper: float


def duration_cells(kind, lower, upper, *, timing, onset_tick):
    minimum = {'HOLD_DURATION': 2, 'SLIDE_DURATION': 1}.get(kind)
    if minimum is None or type(lower) is not int or type(upper) is not int or lower < minimum or upper < lower:
        raise ValueError('invalid duration support')
    first, last = (lower+2)//4, (upper+2)//4
    scale = 1000*timing.written_tick_seconds(onset_tick) if kind == 'HOLD_DURATION' else 1.
    return tuple(DurationCell(i, min(upper,max(lower,i*4)),
                              (lower-.5 if i == first else i*4-2)*scale,
                              (upper+.5 if i == last else i*4+2)*scale)
                 for i in range(first,last+1))


def duration_target_bucket(tick, lower, upper):
    if type(tick) is not int or not lower <= tick <= upper:
        raise ValueError('duration target outside legal support')
    return (tick+2)//4
