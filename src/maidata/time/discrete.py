"""Exact tick384/CONT field queries and data-only supervision.

Indices 0..383 represent positive remainders 1..384; index 384 is CONT.
Neither a query nor a CONT result is a chart token or a Physical action.
"""
from dataclasses import dataclass
import math

from .timing import TICKS_PER_BEAT, duration_end

TIME_BLOCK = 384
TIME_CLASSES = TIME_BLOCK + 1
CONTINUE = TIME_BLOCK
TIME_REPRESENTATION = 'relative-tick384-continue'
_MINIMUM = {'MOMENT': 1, 'HOLD_DURATION': 2, 'SLIDE_DURATION': 1}


def minimum_ticks(kind):
    try:
        return _MINIMUM[kind]
    except KeyError as exc:
        raise ValueError(f'unknown time field {kind!r}') from exc


def encode_ticks(ticks):
    """The unique canonical encoding of a positive complete integer value."""
    if type(ticks) is not int or ticks < 1:
        raise ValueError('time must be a positive integer tick value')
    n, remainder = divmod(ticks - 1, TIME_BLOCK)
    return (CONTINUE,) * n + (remainder,)


def decode_ticks(classes):
    """Decode one complete path; reject missing/extra final short values."""
    classes = tuple(classes)
    if (not classes or any(type(v) is not int or not 0 <= v < TIME_CLASSES for v in classes)
            or any(v != CONTINUE for v in classes[:-1]) or classes[-1] == CONTINUE):
        raise ValueError('expected CONT prefixes and exactly one final short value')
    return TIME_BLOCK * (len(classes) - 1) + classes[-1] + 1


@dataclass(frozen=True)
class TimeQuery:
    field_kind: str
    origin_tick: int
    elapsed_ticks: int = 0

    def __post_init__(self):
        minimum_ticks(self.field_kind)
        if (type(self.origin_tick) is not int or type(self.elapsed_ticks) is not int
                or self.elapsed_ticks < 0 or self.elapsed_ticks % TIME_BLOCK):
            raise ValueError('invalid time-field origin or CONT cursor')

    def query_beat(self, timing):
        if self.field_kind == 'MOMENT':
            return (self.origin_tick + self.elapsed_ticks) / TICKS_PER_BEAT
        # Written BPM stays fixed at the NOTE, including after crossing a BPM.
        seconds = duration_end(timing, self.origin_tick, self.elapsed_ticks,
                               slide=self.field_kind == 'SLIDE_DURATION')
        return timing.second_to_beat(seconds)

    def advance(self, choice, *, window_end_tick=None):
        if type(choice) is not int or not 0 <= choice < TIME_CLASSES:
            raise ValueError('invalid time class')
        if choice == CONTINUE:
            following = TimeQuery(self.field_kind, self.origin_tick, self.elapsed_ticks + TIME_BLOCK)
            closes = (self.field_kind == 'MOMENT' and window_end_tick is not None
                      and self.origin_tick + following.elapsed_ticks >= window_end_tick)
            lower = self.elapsed_ticks + TIME_BLOCK + 1
            if self.field_kind == 'MOMENT':
                lower += self.origin_tick
            return TimeAdvance(following, None, closes, closes, lower)
        complete = self.elapsed_ticks + choice + 1
        if complete < minimum_ticks(self.field_kind):
            raise ValueError('complete time value is below field minimum')
        closes = (self.field_kind == 'MOMENT' and window_end_tick is not None
                  and self.origin_tick + complete >= window_end_tick)
        return TimeAdvance(self, complete, closes, False, None)


@dataclass(frozen=True)
class TimeAdvance:
    query: TimeQuery
    complete_ticks: int | None
    closes_window: bool
    closed_by_continue: bool
    lower_bound_tick: int | None


@dataclass(frozen=True)
class TimeStep:
    elapsed_ticks: int
    query_beat: float
    target_classes: tuple


@dataclass(frozen=True)
class TimePath:
    steps: tuple
    exact_ticks: int | None
    censored: bool
    closes_window: bool
    closed_by_continue: bool
    censored_lower: float | None = None
    closed_lower_bound_tick: int | None = None


def compile_time_target(kind, *, ticks=None, censored_lower=None, origin_tick,
                        timing, window_end_tick=None):
    """Compile one original field, stopping at its first window-closing choice.

``ticks`` and ``censored_lower`` are complete *relative* values, never absolute
onsets. Terminal/censor metadata stays here, outside the network query input.
    """
    minimum = minimum_ticks(kind)
    if (ticks is None) == (censored_lower is None):
        raise ValueError('provide exactly one exact value or censored lower bound')
    query = TimeQuery(kind, origin_tick)
    if window_end_tick is not None and kind == 'MOMENT' and window_end_tick <= origin_tick:
        raise ValueError('MOMENT field must begin before window end')
    if ticks is not None:
        if type(ticks) is not int or ticks < minimum:
            raise ValueError('exact time is below its complete-field minimum')
        classes = encode_ticks(ticks)
    else:
        if kind != 'MOMENT' or not math.isfinite(censored_lower) or censored_lower < minimum:
            raise ValueError('only positive MOMENT lower bounds may be censored')
        lower = math.ceil(censored_lower)
        classes = None
    steps, closes, closed_by_continue, closed_lower = [], False, False, None
    while True:
        if classes is not None:
            targets = (classes[len(steps)],)
        else:
            residual = lower - query.elapsed_ticks
            targets = ((CONTINUE,) if residual > TIME_BLOCK else
                       tuple(range(max(1, residual) - 1, TIME_CLASSES)))
        steps.append(TimeStep(query.elapsed_ticks, query.query_beat(timing), targets))
        if len(targets) != 1:
            # A right-censored tail is one set NLL, not a fabricated endpoint.
            closes = (window_end_tick is not None and origin_tick + lower >= window_end_tick)
            break
        advance = query.advance(targets[0], window_end_tick=window_end_tick)
        closes, closed_by_continue = advance.closes_window, advance.closed_by_continue
        if closed_by_continue:
            closed_lower = advance.lower_bound_tick
        if closes or advance.complete_ticks is not None:
            break
        query = advance.query
    return TimePath(tuple(steps), ticks, censored_lower is not None, closes, closed_by_continue,
                    censored_lower, closed_lower)


def duration_cap(kind, *, timing, local_end_tick, audio_end_seconds,
                 onset_tick=None, origin_tick=None):
    """Largest complete written-BPM duration ending within both audio bounds."""
    if kind not in ('HOLD_DURATION', 'SLIDE_DURATION'):
        raise ValueError('duration cap requires a duration field')
    if (onset_tick is None) == (origin_tick is None):
        raise ValueError('provide exactly one NOTE origin_tick or onset_tick')
    onset_tick = origin_tick if onset_tick is None else onset_tick
    start = timing.second(onset_tick + (TICKS_PER_BEAT if kind == 'SLIDE_DURATION' else 0))
    end = min(timing.second(local_end_tick), audio_end_seconds)
    # Matches Physical's tolerant inclusive integer audio-end comparison.
    return math.floor((end - start + 1e-10) / timing.written_tick_seconds(onset_tick))


def local_z_indices(query_beat, *, moment=False):
    """Allowed absolute beat slots, selected only by the known query cursor."""
    base = math.floor(query_beat)
    return tuple(range(base, math.floor(query_beat + 4) + 1)) if moment else (base - 1, base, base + 1)


@dataclass(frozen=True)
class SubstepMasks:
    basic: tuple
    hard: tuple
    physical: tuple
    penalty: tuple
    continuation_closes: bool


def _bits(value):
    return tuple(bool(value & (1 << c)) for c in range(4))


def substep_masks(kind, intervals, *, origin_tick, elapsed_ticks=0,
                  window_end_tick=None, upper_tick=None):
    """Project full relative integer intervals into this query's 385 choices.

``basic`` permits grammar/audio candidates; ``hard`` additionally requires a
jointly legal Physical completion. CONT's penalty is intentionally empty:
Physical mass belongs to its eventual short-value leaf, or to a zero-penalty
window-close leaf. Each interval is ``(lo,hi,record)`` with source-style basic,
physical and first-attribution penalty fields.
    """
    minimum = minimum_ticks(kind)
    TimeQuery(kind, origin_tick, elapsed_ticks)
    if kind == 'MOMENT' and window_end_tick is None:
        raise ValueError('MOMENT supports require an explicit window end')
    if kind != 'MOMENT' and upper_tick is None:
        raise ValueError('duration supports require a finite complete-value cap')
    intervals = tuple(intervals)
    if upper_tick is None:
        upper_tick = max((b for _, b, _ in intervals), default=0)
    close_delta = window_end_tick - origin_tick if kind == 'MOMENT' else math.inf
    closes = elapsed_ticks + TIME_BLOCK >= close_delta

    def record_at(value):
        if value < minimum:
            return None
        if value >= close_delta:
            return dict(basic=(), physical=0, penalty=0)
        if value > upper_tick:
            return None
        return next((record for a, b, record in intervals if a <= value <= b), None)

    basic, hard, physical, penalty = [], [], [], []
    for remainder in range(1, TIME_BLOCK + 1):
        record = record_at(elapsed_ticks + remainder)
        allowed = record is not None and not record.get('basic', ())
        raw = record.get('physical', 0) if allowed else 0
        charged = record.get('penalty', raw) if allowed else 0
        basic.append(allowed)
        hard.append(allowed and raw == 0)
        physical.append(_bits(raw))
        penalty.append(_bits(charged))
    tail_lo = max(minimum, elapsed_ticks + TIME_BLOCK + 1)
    # MOMENT can always terminate in the future, even beyond effective audio.
    tail_basic = kind == 'MOMENT'
    tail_hard = kind == 'MOMENT'
    if not tail_basic:
        for a, b, record in intervals:
            if max(a, tail_lo) <= min(b, upper_tick) and not record.get('basic', ()):
                tail_basic = True
                if not record.get('physical', 0):
                    tail_hard = True
    basic.append(tail_basic)
    hard.append(tail_hard)
    physical.append((False,) * 4)
    penalty.append((False,) * 4)
    return SubstepMasks(tuple(basic), tuple(hard), tuple(physical), tuple(penalty), closes)


class UnsupportedDurationError(ValueError):
    """Exact source is retained; this local-window training case is deferred."""
    def __init__(self, exact_ticks, cap_ticks):
        self.exact_ticks, self.cap_ticks = exact_ticks, cap_ticks
        super().__init__(f'exact duration {exact_ticks} exceeds local audio cap {cap_ticks}; '
                         'source must be retained as unsupported, never clipped')


@dataclass(frozen=True)
class TimeTree:
    kind: str
    origin_tick: int
    queries: tuple
    supports: tuple
    target_path: TimePath | None

    @property
    def query_beats(self):
        return tuple(q[1] for q in self.queries)

    @property
    def elapsed_ticks(self):
        return tuple(q[0] for q in self.queries)

    @property
    def basic(self):
        return tuple(s.basic for s in self.supports)

    @property
    def hard(self):
        return tuple(s.hard for s in self.supports)

    @property
    def physical(self):
        return tuple(s.physical for s in self.supports)

    @property
    def penalty(self):
        return tuple(s.penalty for s in self.supports)

    @property
    def continuation_closes(self):
        return tuple(s.continuation_closes for s in self.supports)

    @property
    def target(self):
        steps = self.target_path.steps if self.target_path is not None else ()
        return tuple(tuple(c in steps[i].target_classes if i < len(steps) else False
                           for c in range(TIME_CLASSES)) for i in range(len(self.supports)))

    @property
    def target_steps(self):
        length = len(self.target_path.steps) if self.target_path is not None else 0
        return tuple(i < length for i in range(len(self.supports)))


def compile_support_tree(stage, timing, *, kind, window_end, cap_tick, target=None):
    """Shared exact-source compiler for migration, new production and runtime.

All reachable CONT depths are compiled, independently of the GT path length.
Existing MOMENT supports are absolute; duration supports are written ticks.
    """
    from maidata.chart.types import Moment
    from maidata.chart.windows import TimeTarget

    origin = stage['cursor_tick']
    if kind == 'MOMENT':
        support = stage['moment']
        intervals = tuple((a - origin, b - origin, r) for a, b, r in support['intervals'])
        upper = support['domain'][1] - origin
        if target is None:
            path = None
        elif isinstance(target, TimeTarget):
            if target.censored:
                path = compile_time_target(kind, censored_lower=target.absolute_tick - origin,
                    origin_tick=origin, timing=timing, window_end_tick=window_end)
            else:
                value = target.absolute_tick - origin
                if value != int(value):
                    raise ValueError('exact MOMENT target is off the integer tick grid')
                path = compile_time_target(kind, ticks=int(value), origin_tick=origin,
                                           timing=timing, window_end_tick=window_end)
        elif isinstance(target, Moment):
            path = compile_time_target(kind, ticks=timing.tick_after(origin, target.delta_ms) - origin,
                origin_tick=origin, timing=timing, window_end_tick=window_end)
        else:
            raise ValueError('MOMENT requires its exact token or terminal TimeTarget')
    else:
        support = stage['duration']
        intervals = tuple(support['intervals'])
        upper = min(support['domain'][1], duration_cap(kind, timing=timing, onset_tick=origin,
                     local_end_tick=cap_tick, audio_end_seconds=timing.second(cap_tick)))
        if target is not None and target.ticks > upper:
            raise UnsupportedDurationError(target.ticks, upper)
        path = (compile_time_target(kind, ticks=target.ticks, origin_tick=origin, timing=timing)
                if target is not None else None)
    queries, supports, elapsed = [], [], 0
    while True:
        query = TimeQuery(kind, origin, elapsed)
        masks = substep_masks(kind, intervals, origin_tick=origin, elapsed_ticks=elapsed,
                              window_end_tick=window_end, upper_tick=upper)
        if not any(masks.basic):
            raise ValueError('time field has no basic legal completion')
        queries.append((elapsed, query.query_beat(timing)))
        supports.append(masks)
        if masks.continuation_closes or not masks.basic[CONTINUE]:
            break
        elapsed += TIME_BLOCK
    tree = TimeTree(kind, origin, tuple(queries), tuple(supports), path)
    if path is not None and (len(path.steps) > len(supports) or any(not any(a and b for a, b in zip(s.basic, t))
            for s, t, active in zip(supports, tree.target, tree.target_steps) if active)):
        raise ValueError('time target has no basic legal probability mass')
    return tree
