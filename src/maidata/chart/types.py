"""Structured chart values. None is NA; numeric zero is an ordinary value."""
from dataclasses import dataclass
from typing import ClassVar
import math

KINDS = ('tap', 'hold', 'touch', 'touch_hold', 'slide')
BUTTON_PADS = tuple(f'A{i}' for i in range(1, 9))
TOUCH_PADS = tuple(f'{r}{i}' for r in 'ABDE' for i in range(1, 9)) + ('C',)
FLAGS = ('ex', 'break_', 'firework', 'short_hold', 'short_touch_hold')
APPLICABLE = {
    'tap': ('ex', 'break_', 'short_hold'), 'hold': ('ex', 'break_'),
    'touch': ('firework', 'short_touch_hold'), 'touch_hold': ('firework',),
    'slide': ('ex', 'break_'),
}
GENERATED_TYPES = ('MOMENT', 'NOTE', 'ROUTE', 'HOLD_DURATION', 'SLIDE_DURATION')


class ChartError(ValueError):
    """An input cannot be represented without changing its meaning."""


def integer(value, name, minimum=None):
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ChartError(f'{name}: expected integer >= {minimum}, got {value!r}')
    return value


@dataclass(frozen=True)
class Moment:
    delta_ms: float
    type: ClassVar[str] = 'MOMENT'

    def __post_init__(self):
        if isinstance(self.delta_ms, bool) or not math.isfinite(self.delta_ms) or self.delta_ms < 0:
            raise ChartError('delta_ms must be finite and nonnegative')


@dataclass(frozen=True)
class Note:
    kind: str
    pad: str
    ex: int | None = None
    break_: int | None = None
    firework: int | None = None
    short_hold: int | None = None
    short_touch_hold: int | None = None
    type: ClassVar[str] = 'NOTE'

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ChartError(f'unknown kind: {self.kind}')
        pads = TOUCH_PADS if self.kind in ('touch', 'touch_hold') else BUTTON_PADS
        if self.pad not in pads:
            raise ChartError(f'invalid {self.kind} pad: {self.pad}')
        for name in FLAGS:
            value = getattr(self, name)
            if name in APPLICABLE[self.kind]:
                if type(value) not in (int, bool) or value not in (0, 1):
                    raise ChartError(f'{self.kind}.{name} requires a binary value')
            elif value is not None:
                raise ChartError(f'{self.kind}.{name} must be NA')

    @classmethod
    def make(cls, kind, pad, **flags):
        if kind not in APPLICABLE:
            raise ChartError(f'unknown kind: {kind}')
        return cls(kind, pad, **{**dict.fromkeys(APPLICABLE[kind], 0), **flags})


@dataclass(frozen=True)
class Route:
    route_id: int
    track_break: int | None = None
    type: ClassVar[str] = 'ROUTE'

    def __post_init__(self):
        integer(self.route_id, 'route_id', 0)
        if self.route_id > 70:
            raise ChartError('route_id must be in 0..70')
        if self.track_break is not None and (type(self.track_break) not in (int, bool) or self.track_break not in (0, 1)):
            raise ChartError('track_break must be binary or NA')


@dataclass(frozen=True)
class HoldDuration:
    ticks: int
    type: ClassVar[str] = 'HOLD_DURATION'

    def __post_init__(self):
        integer(self.ticks, 'hold ticks', 2)


@dataclass(frozen=True)
class SlideDuration:
    ticks: int
    type: ClassVar[str] = 'SLIDE_DURATION'

    def __post_init__(self):
        integer(self.ticks, 'slide ticks', 1)


Token = Moment | Note | Route | HoldDuration | SlideDuration


@dataclass(frozen=True)
class Track:
    routes: tuple[int, ...]
    track_break: int
    duration_tick: int


@dataclass(frozen=True)
class Completion:
    """A completed press or track; owner is never inferred from pad equality."""
    owner: int
    moment: int
    tick: int
    note: Note
    duration_tick: int | None = None
    track: Track | None = None
    track_index: int | None = None
    token_index: int = -1
