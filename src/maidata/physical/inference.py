"""Cached, field-local raw rule queries for generated prefixes.

The training attribution oracle remains separate.  These queries retain the
same complete-action rules and exact duration intervals, but existential
queries stop once a legal completion is known.  Only the current immutable
grammar/actions prefix is cached; changing or restoring it drops the cache.
"""
from maidata.chart.types import (KINDS, BUTTON_PADS, TOUCH_PADS, Completion,
                                Moment, Note, Route, HoldDuration, SlideDuration)
from .actions import materialize
from .conflicts import branch_conflicts
from .rules import CATEGORIES, prune_history
from .inference_context import InferenceIntervalContext
from .state import Decision, PhysicalState
from .support import duration_support, moment_support
from maidata.time import duration_end


def _project(decisions, *, uniform_deferred=False):
    """Raw existential projection, without training witnesses or attribution."""
    basic, supported, physical, possible = set(), False, set(), False
    for decision in decisions:
        if decision.basic_errors:
            basic.update(decision.basic_errors)
            continue
        supported = True
        if not decision.physical:
            possible = True
            if uniform_deferred or not decision.deferred:
                return decision
        physical.update(decision.physical)
    if possible:
        return Decision(deferred=True)
    if supported:
        return Decision(physical=frozenset(physical))
    return Decision(tuple(sorted(basic)) or ('no_basic_completion',))


def _record(decision):
    return dict(basic=list(decision.basic_errors), penalty=0,
                physical=sum(1 << i for i, c in enumerate(CATEGORIES) if c in decision.physical),
                deferred=decision.deferred)


def _decision(record):
    return Decision(tuple(record['basic']),
                    frozenset(c for i, c in enumerate(CATEGORIES) if record['physical'] & (1 << i)),
                    record.get('deferred', False))


def _query_token(token):
    """Cosmetic flags share support; short flags retain their audio-end test."""
    if isinstance(token, Note):
        flags = {name: getattr(token, name) for name in ('short_hold', 'short_touch_hold')
                 if getattr(token, name) is not None}
        return Note.make(token.kind, token.pad, **flags)
    if isinstance(token, Route) and token.track_break is not None:
        return Route(token.route_id, 0)
    return token


class InferencePhysicalState(PhysicalState):
    """Physical state and prefix-only sampling oracle, with no network dependency.

    Public query methods return raw ``Decision`` objects.  Prefix identity is
    checked on every query as well as normal mutations, so direct replacement
    of ``actions`` or a grammar snapshot cannot leave stale supports behind.
    Immutable snapshots and K/V ownership remain unchanged from PhysicalState.
    """
    def __init__(self, timing, geometry, *, audio_end_seconds):
        super().__init__(timing, geometry, audio_end_seconds=audio_end_seconds)
        self._prefix = None
        self._queries = {}
        self._kinds = {}
        self._positions = {}
        self._types = None
        self._times = {}
        self._context = None
        self.duration_end_seconds = None
        self._refresh()

    def set_duration_cap(self, seconds):
        if seconds != self.duration_end_seconds:
            self.duration_end_seconds = seconds
            self._prefix = None
            self._refresh()

    @property
    def physical(self):
        return self

    def _refresh(self):
        grammar, actions = self.grammar.state, self.actions
        if self._prefix is None or self._prefix[0] is not grammar or self._prefix[1] is not actions:
            self._prefix = (grammar, actions)
            self._queries.clear()
            self._kinds.clear()
            self._positions.clear()
            self._types = None
            self._times.clear()
            if self._context is not None and self._context.history is not actions:
                self._context = None

    def restore(self, snapshot):
        super().restore(snapshot)
        self._refresh()

    def inject_boundary(self, token, **kwargs):
        super().inject_boundary(token, **kwargs)
        self._refresh()

    def accept(self, token, *, window_end=None):
        result = super().accept(token, window_end=window_end)
        self._refresh()
        return result

    def accept_source(self, token):
        self._refresh()
        grammar, completion = self._trial(token)
        actions = self.actions
        decision = Decision(deferred=grammar.state.phase in ('hold', 'route', 'chain'))
        if completion is not None:
            action = materialize(completion, self.timing, self.geometry)
            if self._context is None:
                self._context = InferenceIntervalContext(actions)
            decision = Decision(self._basic(completion), self._context.reasons(action))
            actions += (action,)
        if isinstance(token, Moment):
            actions = prune_history(actions, self.timing.second(grammar.state.cursor))
        self.grammar, self.actions = grammar, actions
        self._refresh()
        return completion, decision

    def close_window(self, target):
        super().close_window(target)
        self._refresh()

    def _action(self, completion, *, touch_repeat=True, deferred=False):
        basic = self._basic(completion)
        if self.duration_end_seconds is not None:
            ticks = (completion.track.duration_tick if completion.track is not None
                     else completion.duration_tick)
            if ticks is not None and duration_end(self.timing, completion.tick, ticks,
                    slide=completion.track is not None) > self.duration_end_seconds + 1e-9:
                basic = (*basic, 'duration_outside_local_audio')
        if basic:
            return Decision(basic)
        action = materialize(completion, self.timing, self.geometry)
        if self._context is None:
            self._context = InferenceIntervalContext(self.actions)
        return Decision(physical=self._context.reasons(action, touch_repeat=touch_repeat), deferred=deferred)

    def query(self, token, *, window_end=None, touch_repeat=True):
        self._refresh()
        canonical = _query_token(token)
        # Touch restart is irrelevant for every other kind of action.
        repeat = bool(touch_repeat)
        if isinstance(token, (Moment, Route)) or isinstance(token, Note) and token.kind not in ('touch', 'touch_hold'):
            repeat = False
        key = (canonical, window_end, repeat)
        if key not in self._queries:
            self._queries[key] = self._query(canonical, window_end=window_end, touch_repeat=repeat)
        return self._queries[key]

    def _query(self, token, *, window_end=None, touch_repeat=True):
        try:
            grammar, completion = self._trial(token, window_end)
        except ValueError as exc:
            return Decision((str(exc),))
        s = grammar.state
        if isinstance(token, Moment):
            if s.phase == 'closed':
                return Decision()
            return _project(self._action(Completion(s.owner+1, s.moment, s.cursor, Note.make(kind, pad)),
                                         touch_repeat=False)
                            for kind, pads in (('tap', BUTTON_PADS), ('touch', TOUCH_PADS)) for pad in pads)
        if completion is not None:
            return self._action(completion, touch_repeat=touch_repeat)
        if isinstance(token, Note):
            held = token.kind in ('hold', 'touch_hold')
            proxy = token if held else Note.make('tap', token.pad)
            return self._action(Completion(s.owner, s.moment, s.cursor, proxy,
                                           duration_tick=2 if held else None),
                                touch_repeat=touch_repeat, deferred=True)
        if isinstance(token, Route):
            prior = self.grammar.state
            branch = branch_conflicts(self.actions, prior.owner, prior.track_count, prior.cursor,
                                      wifi=token.route_id == 70)
            if branch:
                return Decision(physical=frozenset(c.category for c in branch))
            if s.routes == (70,):
                # WiFi cannot be extended. Ordinary chains remain deferred:
                # adding routes can repair their eventual duration/coverage.
                fork = type(self)(self.timing, self.geometry, audio_end_seconds=self.audio_end_seconds)
                fork.duration_end_seconds = self.duration_end_seconds
                fork.grammar, fork.actions = grammar, self.actions
                return fork._duration_projection()
            return Decision(deferred=True)
        return Decision()

    def _kind(self, kind):
        self._refresh()
        if kind not in self._kinds:
            pads = TOUCH_PADS if kind in ('touch', 'touch_hold') else BUTTON_PADS
            self._kinds[kind] = _project((self.query(Note.make(kind, pad), touch_repeat=False) for pad in pads),
                                         uniform_deferred=True)
        return self._kinds[kind]

    def query_kinds(self):
        return {kind: self._kind(kind) for kind in KINDS}

    def query_positions(self, kind, *, touch_repeat=True):
        self._refresh()
        key = (kind, bool(touch_repeat))
        if key not in self._positions:
            pads = TOUCH_PADS if kind in ('touch', 'touch_hold') else BUTTON_PADS
            self._positions[key] = {pad: self.query(Note.make(kind, pad), touch_repeat=touch_repeat) for pad in pads}
        return self._positions[key]

    def _duration_projection(self):
        kind = 'SLIDE_DURATION' if self.grammar.state.phase == 'chain' else 'HOLD_DURATION'
        if kind not in self._times:
            minimum = self.query(SlideDuration(1) if kind == 'SLIDE_DURATION' else HoldDuration(2))
            if minimum.allowed:
                return minimum
            # An illegal shortest value does not exclude a later legal island.
            # Only a positive existence result may bypass exact partitioning.
        support = self.time_support(kind)
        if not support['intervals']:
            return Decision((support['outside_basic'],))
        return _project(_decision(record) for _, _, record in support['intervals'])

    def query_types(self):
        self._refresh()
        if self._types is None:
            result, s = {}, self.grammar.state
            for kind in self.grammar.allowed_types():
                if kind == 'MOMENT':
                    result[kind] = Decision()
                elif kind == 'NOTE':
                    result[kind] = _project(self._kind(name) for name in KINDS)
                elif kind == 'ROUTE':
                    branch = branch_conflicts(self.actions, s.owner, s.track_count, s.cursor)
                    result[kind] = Decision(physical=frozenset(c.category for c in branch), deferred=not branch)
                else:
                    result[kind] = self._duration_projection()
            self._types = result
        return self._types

    def time_support(self, kind):
        self._refresh()
        if kind not in self._times:
            producer = moment_support if kind == 'MOMENT' else duration_support
            extra = {} if kind == 'MOMENT' else {'end_seconds': self.duration_end_seconds}
            self._times[kind] = producer(self, query=self.query, record=_record, **extra)
        return self._times[kind]
