"""Token adapter over rule-only actions; speculative queries never commit."""
from dataclasses import dataclass
import math

from maidata.chart.grammar import Grammar, GrammarSnapshot
from maidata.chart.types import (ChartError, Completion, Moment, Note,
                                Route, HoldDuration, SlideDuration, BUTTON_PADS, TOUCH_PADS)
from maidata.time import duration_end
from .actions import materialize
from .rules import reasons, prune_history, EPS


@dataclass(frozen=True)
class Decision:
    basic_errors: tuple[str, ...] = ()
    physical: frozenset[str] = frozenset()
    deferred: bool = False

    @property
    def allowed(self):
        return not self.basic_errors and not self.physical


@dataclass(frozen=True)
class PhysicalSnapshot:
    grammar: GrammarSnapshot
    actions: tuple


@dataclass(frozen=True)
class DurationSupport:
    ticks: tuple[int, ...]
    decisions: tuple[Decision, ...]
    intervals: tuple[tuple[int, int], ...]  # inclusive integer tick intervals


class PhysicalError(ValueError):
    def __init__(self, decision):
        self.decision = decision
        super().__init__(f'candidate rejected: basic={decision.basic_errors}, physical={sorted(decision.physical)}')


class PhysicalState:
    def __init__(self, timing, geometry, *, audio_end_seconds):
        if not math.isfinite(audio_end_seconds) or audio_end_seconds <= 0:
            raise ValueError('audio_end_seconds must be finite and positive')
        self.timing, self.geometry = timing, geometry
        self.audio_end_seconds = float(audio_end_seconds)
        self.grammar = Grammar(timing)
        self.actions = ()

    def inject_boundary(self, token, *, start_tick, history_present=True, window_end=None):
        grammar = Grammar(self.timing)
        grammar.restore(self.grammar.snapshot())
        grammar.inject_boundary(token, start_tick=start_tick, history_present=history_present, window_end=window_end)
        self.actions = prune_history(self.actions, self.timing.second(start_tick))
        self.grammar = grammar

    def snapshot(self):
        return PhysicalSnapshot(self.grammar.snapshot(), self.actions)

    def restore(self, snapshot):
        if not isinstance(snapshot, PhysicalSnapshot):
            raise TypeError('expected PhysicalSnapshot, not a token/action count')
        self.grammar.restore(snapshot.grammar)
        self.actions = snapshot.actions

    def _basic(self, c):
        start = self.timing.second(c.tick)
        if start < -EPS or start >= self.audio_end_seconds:
            return ('onset_outside_audio',)
        duration = c.track.duration_tick if c.track else c.duration_tick
        if c.note.short_hold or c.note.short_touch_hold:
            duration = 1
        if duration is not None and duration_end(self.timing, c.tick, duration, slide=c.track is not None) > self.audio_end_seconds+EPS:
            return ('duration_outside_audio',)
        return ()

    def _trial(self, token, window_end=None):
        grammar = Grammar(self.timing)
        grammar.restore(self.grammar.snapshot())
        completion = grammar.accept(token, window_end=window_end)
        return grammar, completion

    def query(self, token, *, window_end=None):
        try:
            g, c = self._trial(token, window_end)
        except (ChartError, ValueError) as e:
            return Decision((str(e),))
        s = g.state
        if isinstance(token, Moment) and s.phase == 'closed':
            return Decision()
        if isinstance(token, Moment):
            # Window-closing queries above do not require a possible NOTE.
            failures = set()
            for kind, pads in (('tap', BUTTON_PADS), ('touch', TOUCH_PADS)):
                for pad in pads:
                    candidate = Completion(s.owner+1, s.moment, s.cursor, Note.make(kind, pad))
                    basic = self._basic(candidate)
                    if basic:
                        return Decision(basic)
                    bad = reasons(self.actions, materialize(candidate, self.timing, self.geometry), touch_repeat=False)
                    if not bad:
                        return Decision()
                    failures.update(bad)
            return Decision(physical=frozenset(failures))
        if c is not None:
            basic = self._basic(c)
            if basic:
                return Decision(basic)
            return Decision(physical=reasons(self.actions, materialize(c, self.timing, self.geometry)))
        if isinstance(token, Note):
            # Only the head is known for a slide. A placeholder route would
            # incorrectly rule out a chain that completes this prefix.
            proxy = token if token.kind in ('hold', 'touch_hold') else Note.make('tap', token.pad)
            candidate = Completion(s.owner, s.moment, s.cursor, proxy,
                                   duration_tick=2 if token.kind in ('hold', 'touch_hold') else None)
            basic = self._basic(candidate)
            if basic:
                return Decision(basic)
            return Decision(physical=reasons(self.actions, materialize(candidate, self.timing, self.geometry)), deferred=True)
        if isinstance(token, Route):
            old = self.grammar.state
            if old.phase == 'track_done' and old.track_count >= 2:
                return Decision(physical=frozenset({'multitouch'}))
            return Decision(deferred=True)
        return Decision()

    def accept(self, token, *, window_end=None):
        decision = self.query(token, window_end=window_end)
        if not decision.allowed:
            raise PhysicalError(decision)
        grammar, completion = self._trial(token, window_end)
        actions = self.actions
        if completion is not None:
            actions += (materialize(completion, self.timing, self.geometry),)
        if isinstance(token, Moment) and grammar.state.phase != 'closed':
            actions = prune_history(actions, self.timing.second(grammar.state.cursor))
        self.grammar, self.actions = grammar, actions
        return completion

    def close_window(self, target):
        grammar = Grammar(self.timing)
        grammar.restore(self.grammar.snapshot())
        grammar.close_window(target)
        self.grammar = grammar

    def accept_source(self, token):
        """Advance a real teacher prefix, retaining violations and true actions.

        Grammar remains strict. Only completed actions are judged: pending
        chains are deferred until their total duration is available. This is
        deliberately separate from candidate accept, which still rejects.
        """
        grammar, completion = self._trial(token)
        actions = self.actions
        decision = Decision(deferred=grammar.state.phase in ('hold', 'route', 'chain'))
        if completion is not None:
            action = materialize(completion, self.timing, self.geometry)
            decision = Decision(self._basic(completion), reasons(actions, action))
            actions += (action,)
        if isinstance(token, Moment):
            actions = prune_history(actions, self.timing.second(grammar.state.cursor))
        self.grammar, self.actions = grammar, actions
        return completion, decision

    def query_duration_ticks(self, ticks):
        phase = self.grammar.state.phase
        if phase not in ('hold', 'chain'):
            raise ChartError('duration support requires a pending hold or complete route prefix')
        ticks = tuple(ticks)
        if any(type(t) is not int for t in ticks) or any(a >= b for a, b in zip(ticks, ticks[1:])):
            raise ValueError('duration candidates must be increasing integer ticks')
        constructor = HoldDuration if phase == 'hold' else SlideDuration
        decisions = []
        for tick in ticks:
            try:
                decision = self.query(constructor(tick))
            except ChartError as e:
                decision = Decision((str(e),))
            decisions.append(decision)
        intervals = []
        for tick, decision in zip(ticks, decisions):
            if not decision.allowed:
                continue
            if intervals and intervals[-1][1]+1 == tick:
                intervals[-1] = intervals[-1][0], tick
            else:
                intervals.append((tick, tick))
        return DurationSupport(ticks, tuple(decisions), tuple(intervals))

    def query_note_positions(self, kind):
        """Only kind is known; flags are absent from physical support."""
        pads = TOUCH_PADS if kind in ('touch', 'touch_hold') else BUTTON_PADS
        return {pad: self.query(Note.make(kind, pad)) for pad in pads}

    def query_note_kinds(self):
        """Existential projection; stop at the first allowed completion."""
        from maidata.chart.types import KINDS
        result = {}
        for kind in KINDS:
            decisions = []
            pads = TOUCH_PADS if kind in ('touch', 'touch_hold') else BUTTON_PADS
            for pad in pads:
                if kind in ('touch', 'touch_hold'):
                    g = self.grammar.state
                    n = Note.make(kind,pad)
                    try:
                        trial, _ = self._trial(n)
                    except (ChartError, ValueError) as e:
                        d = Decision((str(e),))
                    else:
                        c = Completion(g.owner+1,trial.state.moment,g.cursor,n,
                                       duration_tick=2 if kind == 'touch_hold' else None)
                        basic = self._basic(c)
                        d = Decision(basic,frozenset() if basic else reasons(
                            self.actions,materialize(c,self.timing,self.geometry),touch_repeat=False),kind == 'touch_hold')
                else:
                    d = self.query(Note.make(kind,pad))
                decisions.append(d)
                if d.allowed:
                    result[kind] = d
                    break
            else:
                result[kind] = Decision(tuple(sorted({e for d in decisions for e in d.basic_errors})),
                                        frozenset(e for d in decisions for e in d.physical))
        return result
