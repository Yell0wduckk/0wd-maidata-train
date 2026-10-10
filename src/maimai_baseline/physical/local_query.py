"""Optimistic, field-local inference masks; training keeps StageEngine unchanged."""
from collections import Counter, defaultdict
import time
import numpy as np

from .intervals import EPS, hand_segments
from .query_context import IntervalQueryContext
from .stage_query import StageEngine, StageResult, KINDS, query_key
from maimai_baseline.time.coordinates import seconds_at_ticks

POLICY = 'physical-local-fields-bounded-retry-v2'


class LocalContext:
    def __init__(self, state, tick):
        self.state, self.first_tick = state, tick
        self.identity = (state.version, len(state.actions), state.check_same_path)
        self.intervals = IntervalQueryContext(state.actions, state.second(tick), state.catalog, state.check_same_path)
        self.history = self.intervals.history
        self.segments = hand_segments(self.history)
        self.engine_key = None
        self.engine = None
        self.results = {}

    def short_support(self, ticks, moment, *, touch=True, startup=True):
        """Necessary head occupancy only, in bulk; no position/route search."""
        ticks = np.asarray(ticks, dtype=np.float64)
        ts = seconds_at_ticks(self.state.timing, ticks)
        ends = seconds_at_ticks(self.state.timing, ticks + 2)
        allowed = np.ones(len(ticks), dtype=bool)
        for lo, hi, demand in self.segments:
            if demand >= 2:
                allowed &= ~((ts < hi-EPS) & (ends > lo+EPS))
        if touch:
            # Check each precomputed panel: a candidate can straddle the
            # existing pool's release and still fit once that hand is free.
            shared=np.ones(len(ticks),dtype=bool)
            for lo,hi,_,touches,_,_,_,demand in self.intervals.panels:
                if demand+int(not touches)>2:
                    shared &= ~((ts<hi-EPS)&(ends>lo+EPS))
            allowed |= shared
        if startup:
            # Only launch instants can rescue a full two-hand interval. Check
            # matching there so already consumed credits are not reused.
            launches = {(a.pad, a.launch) for a in self.history if a.track}
            for pad, launch in launches:
                for i in np.flatnonzero(~allowed & (np.abs(ts-launch) <= EPS)):
                    for kind in ('tap','touch') if touch else ('tap',):
                        candidate = self.state._candidate(dict(event='NOTE', kind=kind, pad=pad), float(ticks[i]), moment)
                        if not self.intervals.overloaded(candidate):
                            allowed[i] = True
                            break
        return allowed

    def query(self, tick, moment, stage, kind=None, position=None, flags=(), route=None):
        if self.engine_key != (tick, moment):
            self.engine_key = (tick, moment)
            self.engine = LocalEngine(self.state, tick, moment, self)
            self.results.clear()
        key = (tick, moment, query_key(stage, kind, position, flags, route))
        if key in self.results:
            self.state.cache_hits += 1
            return self.results[key]
        result = self.engine.query(stage, kind, position, flags, route)
        self.results[key] = result
        return result


class LocalEngine(StageEngine):
    def __init__(self, state, tick, moment, context):
        super().__init__(state, tick, moment)
        self.context = context

    def _candidate(self, *args, **kwargs):
        self.state.local_counts['candidate_checks'] += 1
        key = (self.state.version, len(self.state.actions), float(self.tick), int(self.moment), self.state.check_same_path)
        self.state._interval_context = (key, self.context.intervals)
        return super()._candidate(*args, **kwargs)

    def query(self, stage, kind=None, position=None, flags=(), route=None):
        if stage == 'event':
            note = self.context.short_support([self.tick], self.moment)[0]
            branch = self.state.branch_available(self.moment) and self.layout(None)[-1] >= 1
            bad = np.zeros((4, 2), bool)
            bad[0] = [not note, not branch]
            return StageResult(stage, ('NOTE', 'SLIDE_BRANCH'), bad)
        if stage == 'kind':
            bad = np.zeros((4, len(KINDS)), bool)
            for i, k in enumerate(KINDS):
                head = self.context.short_support([self.tick], self.moment,
                    touch=k in ('touch','touch_hold'), startup=k in ('tap', 'touch', 'slide', 'wifi'))[0]
                _, _, _, duration, lower, upper = self.layout(k)
                bad[0, i] = not head or bool(duration and upper < lower)
            return StageResult(stage, KINDS, bad)
        if stage == 'position':
            field, positions, _, _, _, _ = self.layout(kind)
            short = kind in ('tap', 'touch')
            bad = np.zeros((4, len(positions), 2) if short else (4, len(positions)), bool)
            for p, value in enumerate(positions):
                if kind is None:
                    continue  # An owned branch has no new head.
                if short:
                    flag = 'short_hold' if kind == 'tap' else 'short_touch_hold'
                    for bit in (0, 1):
                        self.mark(bad, (p, bit), self.candidate(kind, value, (flag,) if bit else ())[1])
                elif kind in ('slide', 'wifi'):
                    # Only the head is known here. Do not choose a placeholder route.
                    self.mark(bad, (p,), self.candidate('tap', value)[1])
                else:
                    self.mark(bad, (p,), self.candidate(kind, value)[1])
            self.touch_position_mask(kind, positions, bad)
            return StageResult(stage, positions, bad, field)
        # Route checks just its own candidates at this start; duration computes
        # one bound only after its route has actually been selected.
        return super().query(stage, kind, position, flags, route)


def context(state, tick):
    if not hasattr(state, 'local_counts'):
        state.local_counts = Counter()
        state.local_seconds = defaultdict(float)
    current = getattr(state, '_local_context', None)
    identity = (state.version, len(state.actions), state.check_same_path)
    if current is None or current.identity != identity or tick < current.first_tick:
        start = time.perf_counter()
        current = state._local_context = LocalContext(state, tick)
        state.local_seconds['prefix'] += time.perf_counter()-start
    return current


def query(state, tick, moment, stage, *args, **kwargs):
    start = time.perf_counter()
    current = context(state, tick)
    result = current.query(tick, moment, stage, *args, **kwargs)
    elapsed = time.perf_counter()-start
    state.local_seconds[stage] += elapsed
    state.check_seconds += elapsed
    state.local_counts['query:'+stage] += 1
    return result


def moment_support(state, *, tick, target_end, first, moment, excluded=(), exists=False):
    """Family selection probes chunks and stops early; delta requests all bits."""
    start = time.perf_counter()
    current = context(state, tick)
    lower = int(tick) + (0 if first else 1)
    upper = min(int(target_end)-1, state.song_end_tick)
    excluded = set(excluded)
    blocks = []
    try:
        for begin in range(lower, upper+1, 64 if exists else max(1, upper-lower+1)):
            ticks = np.arange(begin, min(upper+1, begin+64) if exists else upper+1)
            allowed = current.short_support(ticks, moment)
            if excluded:
                allowed &= ~np.isin(ticks, list(excluded))
            state.local_counts['delta_ticks'] += len(ticks)
            if exists and allowed.any():
                return True
            if not exists:
                blocks.append((ticks.tolist(), allowed.tolist()))
        return False if exists else (blocks[0] if blocks else ([], []))
    finally:
        elapsed = time.perf_counter()-start
        state.local_seconds['delta_exists' if exists else 'delta'] += elapsed
        state.check_seconds += elapsed
