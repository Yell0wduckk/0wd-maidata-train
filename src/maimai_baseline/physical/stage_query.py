"""Causal Physical queries. No query materializes position-by-route tables."""
from dataclasses import dataclass
import numpy as np
from maimai_dataset.event_time import maximum_duration
from maimai_dataset.modifiers import parts
from maimai_baseline.time.duration import first_representative

CATEGORIES = ('multitouch', 'overlap', 'slide_too_fast', 'tap_on_slide')
KINDS = ('tap', 'touch', 'hold', 'touch_hold', 'slide', 'wifi')
STAGES = ('event', 'kind', 'position', 'route', 'duration')


def query_key(stage, kind=None, position=None, flags=(), route=None):
    """Only preceding fields enter each stage key; cosmetic flags are equivalent."""
    if stage not in STAGES:
        raise ValueError('unknown Physical stage: ' + stage)
    if stage in ('event', 'kind'):
        return (stage,)
    if stage == 'position':
        return (stage, kind)
    physical_flags = tuple(sorted(parts(flags) & {'short_hold', 'short_touch_hold'}))
    if stage == 'route':
        return (stage, kind, position, physical_flags)
    return (stage, kind, position, physical_flags, route)


def physical_query_key(catalog, *args):
    key=query_key(*args)
    if key[0]=='duration' and key[1] in (None,'slide','wifi'):
        stage,kind,position,flags,route=key
        if route is not None and route!=70:
            lane=int(position[1:]);path=catalog.path_id(lane,route)
            if path>=0:
                route=next(r for r in range(70) if catalog.path_id(lane,r)==path)
        return stage,kind,position,flags,route
    return key


@dataclass
class StageResult:
    stage: str
    values: tuple
    bad: np.ndarray
    field: str | None = None
    duration_lower: int = 0
    duration_upper: int = 0
    safe_upper: int = 0

    @property
    def allowed(self):
        return ~self.bad.any(axis=0)

    @property
    def exists(self):
        return bool(self.allowed.any())


class StageEngine:
    def __init__(self, state, tick, moment):
        self.state, self.tick, self.moment = state, tick, moment
        self.layouts={};self.completions={};self.kinds={};self.candidates={}

    def layout(self, kind):
        if kind not in self.layouts:self.layouts[kind]=self._layout(kind)
        return self.layouts[kind]

    def _layout(self, kind):
        branch = kind is None
        slide = branch or kind in ('slide', 'wifi')
        field = 'start_pad' if slide else 'pad'
        owner = self.state.owner(self.moment) if branch else None
        values = ((owner.pad,) if owner else ()) if branch else self.state._value_table[kind]
        routes = (70,) if kind == 'wifi' else tuple(range(71 if branch else 70)) if slide else ()
        duration = 'branch_duration_tick' if slide else 'hold_duration_tick' if kind in ('hold', 'touch_hold') else None
        lower = 1 if slide else 2 if duration else 0
        upper = maximum_duration(self.state.timing, self.tick, self.state.audio_end_seconds, slide=slide) if duration else 0
        return field, values, routes, duration, lower, upper

    def candidate(self, kind, position, flags=(), route=None):
        key=(kind,position,tuple(sorted(parts(flags))),route)
        if key not in self.candidates:self.candidates[key]=self._candidate(kind,position,flags,route)
        return self.candidates[key]

    def _candidate(self, kind, position, flags=(), route=None):
        field, _, routes, duration, lower, upper = self.layout(kind)
        row = dict(event='NOTE' if kind is not None else 'SLIDE_BRANCH', **{field: position}, flags='+'.join(parts(flags)))
        if kind is not None:
            row['kind'] = kind
        if duration:
            if upper < lower:
                return row, {'multitouch'}
            row[duration] = first_representative(lower, upper)
        elif parts(flags) & {'short_hold', 'short_touch_hold'}:
            if maximum_duration(self.state.timing, self.tick, self.state.audio_end_seconds) < 1:
                return row, {'multitouch'}
        if routes and (route not in routes or self.state.catalog.path_id(int(position[1:]), route) < 0):
            return row, {'slide_too_fast'}
        return row, self.state._reasons(row, self.tick, self.moment, route)

    def completion(self, kind, position):
        key=(kind,position)
        if key not in self.completions:self.completions[key]=self._completion(kind,position)
        return self.completions[key]

    def _completion(self, kind, position):
        """Existential projection: stop at the first legal future choice."""
        routes = self.layout(kind)[2]
        flags = ((), ('short_hold' if kind == 'tap' else 'short_touch_hold',)) if kind in ('tap', 'touch') else ((),)
        failures = set()
        for bits in flags:
            for route in routes or (None,):
                _, errors = self.candidate(kind, position, bits, route)
                if not errors:
                    return set()
                failures.update(errors)
        return failures or {'multitouch'}

    @staticmethod
    def mark(bad, index, errors, *, completion=False):
        # Aggregate dead ends have one earliest responsibility, avoiding duplicate
        # blame for a candidate whose different continuations fail differently.
        selected = next((c for c in CATEGORIES if c in errors), None) if completion else None
        for c in errors:
            if not completion or c == selected:
                bad[(CATEGORIES.index(c), *index)] = True

    def kind_errors(self, kind):
        if kind not in self.kinds:self.kinds[kind]=self._kind_errors(kind)
        return self.kinds[kind]

    def _kind_errors(self, kind):
        if kind is None and not self.state.branch_available(self.moment):
            return {'multitouch'}
        failures = set()
        for position in self.layout(kind)[1]:
            errors = self.completion(kind, position)
            if not errors:
                return set()
            failures.update(errors)
        return failures or {'multitouch'}

    def query(self, stage, kind=None, position=None, flags=(), route=None):
        if stage in ('kind', 'event'):
            values = KINDS if stage == 'kind' else ('NOTE', 'SLIDE_BRANCH')
            bad = np.zeros((4, len(values)), bool)
            if stage == 'kind':
                for i, k in enumerate(values):
                    self.mark(bad, (i,), self.kind_errors(k), completion=True)
            else:
                failures = set()
                for k in KINDS:
                    errors = self.kind_errors(k)
                    if not errors:
                        failures.clear()
                        break
                    failures.update(errors)
                self.mark(bad, (0,), failures, completion=True)
                self.mark(bad, (1,), self.kind_errors(None), completion=True)
            return StageResult(stage, values, bad)
        field, positions, routes, duration, lower, upper = self.layout(kind)
        if stage == 'position':
            short = kind in ('tap', 'touch')
            bad = np.zeros((4, len(positions), 2) if short else (4, len(positions)), bool)
            for p, value in enumerate(positions):
                if short:
                    flag = 'short_hold' if kind == 'tap' else 'short_touch_hold'
                    for bit in (0, 1):
                        self.mark(bad, (p, bit), self.candidate(kind, value, (flag,) if bit else ())[1])
                else:
                    self.mark(bad, (p,), self.completion(kind, value), completion=True)
            self.touch_position_mask(kind, positions, bad)
            return StageResult(stage, positions, bad, field)
        if position not in positions:
            raise ValueError('Physical query position has no valid owner/kind')
        if stage == 'route':
            if not routes:
                raise ValueError('route query for a non-track')
            bad = np.zeros((4, len(routes)), bool)
            for i, r in enumerate(routes):
                self.mark(bad, (i,), self.candidate(kind, position, flags, r)[1])
            return StageResult(stage, routes, bad, 'route')
        if stage != 'duration' or duration is None:
            raise ValueError('inapplicable Physical duration stage')
        row, errors = self.candidate(kind, position, flags, route)
        safe = lower - 1 if errors else self.state._safe_upper(row, self.tick, self.moment, route, duration, lower, upper)
        bad = np.zeros((4, 1), bool)
        self.mark(bad, (0,), errors)
        return StageResult(stage, (), bad, duration, lower, upper, int(safe))

    def touch_position_mask(self, kind, positions, bad):
        # This rule belongs only to pad: existential earlier stages and duration
        # queries deliberately use candidates without the restart prohibition.
        if kind in ('touch', 'touch_hold'):
            from .intervals import is_touch, EPS
            if not hasattr(self, '_touch_pads'):
                second = self.state.second(self.tick)
                # Candidate preparation already owns the bounded history.
                context = self.state._interval_context
                history = context[1].history if context is not None else self.state.actions
                self._touch_pads = {a.pad for a in history if is_touch(a)
                                   and -EPS <= second-a.press_start <= .25+EPS}
            for p, pad in enumerate(positions):
                if pad in self._touch_pads:
                    bad[CATEGORIES.index('overlap'), p] = True
