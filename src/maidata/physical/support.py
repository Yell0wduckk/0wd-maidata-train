"""Exact integer support partitions from the interval rule change points.

No duration search limit: the basic domain ends at the actual audio boundary.
All time comparisons in rules.py are between fixed prefix times and affine
candidate motion times (release uses the inverse global 22-tick map).
Neighbouring integer ticks isolate equality and EPS cases at each crossing.
"""
import math

from maidata.chart.types import Moment, HoldDuration, SlideDuration
from maidata.time import advance_ticks
from .actions import materialize
from .rules import CATEGORIES, EPS, history_end


def decision_record(d):
    return dict(basic=list(d.basic_errors),
                physical=sum(1 << i for i, c in enumerate(CATEGORIES) if c in d.physical),
                deferred=d.deferred)


def _times(actions):
    result = set()
    for a in actions:
        result.update((a.press_start, a.press_end))
        if a.touch:
            result.add(a.press_start+.25)
        if a.track:
            result.update((a.launch, a.actual_end, a.release, a.guard_end))
            if a.handoff_release is not None:
                result.add(a.handoff_release)
            for m in a.motions:
                result.update((m.start, m.end))
                for entered, _ in m.entries:
                    result.update((entered, entered+.2))
    return result


def _cuts(roots, lower, upper):
    cuts = {lower, upper+1}
    for x in roots:
        if not math.isfinite(x) or x < lower-4 or x > upper+4:
            continue
        n = math.floor(x)
        for k in range(n-2, n+4):
            if lower <= k <= upper+1:
                cuts.add(k)
    return sorted(cuts)


def _partition(roots, lower, upper, query):
    rows = []
    cuts = _cuts(roots, lower, upper)
    for a, b in zip(cuts, cuts[1:]):
        d = decision_record(query((a+b-1)//2))
        if rows and rows[-1][2] == d and rows[-1][1]+1 == a:
            rows[-1][1] = b-1
        else:
            rows.append([a, b-1, d])
    return rows


def duration_support(state):
    s, timing = state.grammar.state, state.timing
    slide = s.phase == 'chain'
    if s.phase not in ('chain', 'hold'):
        raise ValueError('duration support requires pending duration')
    constructor = SlideDuration if slide else HoldDuration
    lower = 1 if slide else 2
    origin = timing.second(s.cursor+(96 if slide else 0))
    scale = timing.written_tick_seconds(s.cursor)
    upper = math.floor((state.audio_end_seconds-origin+EPS)/scale)
    if upper < lower:
        return dict(domain=[lower, upper], intervals=[], outside_basic='duration_outside_audio')
    _, c1 = state._trial(constructor(lower))
    _, c2 = state._trial(constructor(lower+1))
    a, b = materialize(c1,timing,state.geometry), materialize(c2,timing,state.geometry)
    fixed = _times(state.actions)
    # Include fixed candidate head/launch times for comparisons inside the action.
    fixed.update((a.press_start, a.launch if slide else a.press_start))
    if slide:
        fixed.add(a.press_end)
    moving = [(a.actual_end,b.actual_end)] if slide else []
    for m, n in zip(a.motions,b.motions):
        moving.extend(((m.start,n.start),(m.end,n.end)))
        moving.extend((x[0],y[0]) for x,y in zip(m.entries,n.entries))
    roots = []
    for x, y in moving:
        slope = y-x
        if slope:
            roots.extend(lower+(t-x)/slope for t in fixed)
    # release/hold press_end = T(T^-1(origin + duration*scale) + 22/96)
    roots.extend((advance_ticks(timing,t,-22)-origin)/scale for t in fixed)
    return dict(domain=[lower,upper], intervals=_partition(roots,lower,upper,
                lambda n: state.query(constructor(n))), outside_basic='duration_outside_audio')


def moment_support(state):
    """Absolute tick support for real onsets; window-closing queries bypass it."""
    s, timing = state.grammar.state, state.timing
    lower = s.cursor+1
    raw_end = timing.second_to_beat(state.audio_end_seconds)*96
    upper = math.ceil(raw_end)-1
    while timing.second(upper) >= state.audio_end_seconds:
        upper -= 1
    if upper < lower:
        return dict(domain=[lower,upper], intervals=[], outside_basic='onset_outside_audio')
    times = _times(state.actions) | {0.}
    roots = []
    for t in times:
        tick = timing.second_to_beat(t)*96
        roots.extend((tick,tick-2))
    # Beyond the last surviving action all real-onset candidates are equivalent.
    end = max(0., max((history_end(a) for a in state.actions), default=timing.second(lower)))
    active_upper = min(upper, math.ceil(timing.second_to_beat(end)*96)+3)
    rows = _partition(roots, lower, max(lower-1,active_upper),
                      lambda n: state.query(Moment(timing.delta_ms(s.cursor,n))))
    if active_upper < upper:
        from .state import Decision
        d = decision_record(Decision())
        if rows and rows[-1][2] == d:
            rows[-1][1] = upper
        else:
            rows.append([max(lower,active_upper+1),upper,d])
    return dict(domain=[lower,upper], intervals=rows, outside_basic='onset_outside_audio')


def interval_decision(support, tick):
    from .state import Decision
    for lo,hi,d in support['intervals']:
        if lo <= tick <= hi:
            return Decision(tuple(d['basic']),frozenset(c for i,c in enumerate(CATEGORIES)
                            if d['physical'] & (1 << i)),d['deferred'])
    return Decision((support['outside_basic'],))
