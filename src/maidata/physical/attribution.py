"""Prefix-only first-responsibility queries over rule witnesses.

PhysicalState owns grammar/actions and hard rejection. This adapter owns only
the training responsibility ledger. It never receives model tensors or future
route/duration labels to decide an earlier field.
"""
from dataclasses import dataclass

from maidata.chart.types import (KINDS, BUTTON_PADS, TOUCH_PADS, Completion,
                                Moment, Note, Route, HoldDuration, SlideDuration)
from .actions import materialize
from .conflicts import Conflict, action_conflicts, branch_conflicts, capacity_conflicts
from .state import Decision, PhysicalState
from .support import CATEGORIES, duration_support, moment_support


@dataclass(frozen=True)
class Evidence:
    basic: tuple = ()
    conflicts: frozenset = frozenset()
    deferred: bool = False

    def decision(self):
        return Decision(self.basic, frozenset(c.category for c in self.conflicts), self.deferred)


@dataclass(frozen=True)
class Evaluation:
    raw: Evidence = Evidence()
    penalty: Evidence = Evidence()

    def record(self):
        def mask(e):
            return sum(1 << i for i, c in enumerate(CATEGORIES) if c in e.decision().physical)
        return dict(basic=list(self.raw.basic), physical=mask(self.raw), penalty=mask(self.penalty),
                    deferred=self.raw.deferred, penalty_deferred=self.penalty.deferred,
                    conflicts=[c.record() for c in sorted(self.raw.conflicts)],
                    new_conflicts=[c.record() for c in sorted(self.penalty.conflicts)])


def _project(values, *, attribution=False):
    """Existence of a known or still-possible completion prevents early blame."""
    supported = [v for v in values if not v.basic]
    if not supported:
        return Evidence(tuple(sorted({b for v in values for b in v.basic})) or ('no_basic_completion',))
    possible = [v for v in supported if not v.conflicts]
    if possible:
        return Evidence(deferred=all(v.deferred for v in possible))
    witnesses = frozenset(c for v in supported for c in v.conflicts)
    if attribution:
        certain = frozenset.intersection(*(v.conflicts for v in supported))
        if certain:
            # Charge the concrete conflict already unavoidable for every
            # completion, not optional same-category conflicts of long values.
            witnesses = certain
        else:
            mandatory = set.intersection(*({c.category for c in v.conflicts} for v in supported))
            choices = [v.conflicts if not mandatory else frozenset(c for c in v.conflicts if c.category in mandatory)
                       for v in supported]
            # A disjunction can exclude all completions without one universal
            # witness (e.g. every pad crowds a different hand). Keep a stable
            # cover, rather than claiming unrelated optional conflicts too.
            cover = set()
            while choices:
                candidates = set().union(*choices)
                chosen = min(candidates, key=lambda c: (-sum(c in row for row in choices), c))
                cover.add(chosen)
                choices = [row for row in choices if chosen not in row]
            witnesses = frozenset(cover)
    return Evidence(conflicts=witnesses)


def project(values):
    values = tuple(values)
    return Evaluation(_project([v.raw for v in values]), _project([v.penalty for v in values], attribution=True))


@dataclass(frozen=True)
class AttributionSnapshot:
    physical: object
    charged: frozenset


class AttributionState:
    def __init__(self, physical):
        self.physical = physical
        self.charged = frozenset()

    def snapshot(self):
        return AttributionSnapshot(self.physical.snapshot(), self.charged)

    def restore(self, snapshot):
        if not isinstance(snapshot, AttributionSnapshot):
            raise TypeError('expected AttributionSnapshot')
        self.physical.restore(snapshot.physical)
        self.charged = snapshot.charged

    def inject_boundary(self, token, **kwargs):
        self.physical.inject_boundary(token, **kwargs)
        self.charged = frozenset()

    def close_window(self, target):
        self.physical.close_window(target)

    def _leaf(self, basic=(), conflicts=frozenset(), *, inherited=frozenset(), deferred=False):
        if basic:
            return Evaluation(Evidence(tuple(basic)), Evidence(tuple(basic)))
        return Evaluation(Evidence(conflicts=conflicts, deferred=deferred),
                          Evidence(conflicts=conflicts-inherited-self.charged, deferred=deferred))

    def _action(self, completion, *, touch_repeat=True, deferred=False):
        p = self.physical
        basic = p._basic(completion)
        if basic:
            return self._leaf(basic)
        action = materialize(completion, p.timing, p.geometry)
        conflicts = action_conflicts(p.actions, action, touch_repeat=touch_repeat)
        spans = [(action.press_start, action.press_end)] + ([(action.launch, action.release)] if action.track else [])
        inherited = capacity_conflicts(p.actions, spans)
        return self._leaf(conflicts=conflicts, inherited=inherited, deferred=deferred)

    def _fork(self, grammar):
        p = self.physical
        other = PhysicalState(p.timing, p.geometry, audio_end_seconds=p.audio_end_seconds)
        other.grammar, other.actions = grammar, p.actions
        result = AttributionState(other)
        result.charged = self.charged
        return result

    def query(self, token, *, window_end=None, touch_repeat=True):
        p = self.physical
        try:
            grammar, completion = p._trial(token, window_end)
        except ValueError as exc:
            return self._leaf((str(exc),))
        s = grammar.state
        if isinstance(token, Moment):
            if s.phase == 'closed':
                return Evaluation()
            # Touch repetition belongs to pad, including when every Touch pad
            # repeats. Tap/Touch suffice for the minimum real-onset demand.
            return project(self._action(Completion(s.owner+1, s.moment, s.cursor, Note.make(kind, pad)),
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
            prior = p.grammar.state
            branch = branch_conflicts(p.actions, prior.owner, prior.track_count, prior.cursor,
                                      wifi=token.route_id == 70)
            if branch:
                return self._leaf(conflicts=branch)
            if s.routes == (70,):
                # WiFi cannot be extended; all basic durations are now known.
                return self._fork(grammar)._duration_projection()
            # A normal prefix may be repaired by more ROUTEs. No temporary
            # single-segment duration or arbitrary chain length is invented.
            return self._leaf(deferred=True)
        return Evaluation()

    def query_positions(self, kind, *, touch_repeat=True):
        pads = TOUCH_PADS if kind in ('touch', 'touch_hold') else BUTTON_PADS
        return {pad: self.query(Note.make(kind, pad), touch_repeat=touch_repeat) for pad in pads}

    def query_kinds(self):
        return {kind: project(self.query_positions(kind, touch_repeat=False).values()) for kind in KINDS}

    def _duration_values(self):
        return duration_support(self.physical, query=self.query, record=lambda value: value)

    def _duration_projection(self):
        support = self._duration_values()
        if not support['intervals']:
            return self._leaf((support['outside_basic'],))
        return project(v for _, _, v in support['intervals'])

    def query_types(self):
        p, result = self.physical, {}
        s = p.grammar.state
        for kind in p.grammar.allowed_types():
            if kind == 'MOMENT':
                # Every window has a terminal tail; it never needs a NOTE.
                result[kind] = Evaluation()
            elif kind == 'NOTE':
                result[kind] = project(self.query_kinds().values())
            elif kind == 'ROUTE':
                branch = branch_conflicts(p.actions, s.owner, s.track_count, s.cursor)
                result[kind] = self._leaf(conflicts=branch, deferred=not branch)
            else:
                result[kind] = self._duration_projection()
        return result

    def time_support(self, kind):
        producer = moment_support if kind == 'MOMENT' else duration_support
        return producer(self.physical, query=self.query, record=Evaluation.record)

    def _choose(self, evaluation):
        self.charged |= evaluation.penalty.conflicts
        return [c.record() for c in sorted(evaluation.penalty.conflicts)]

    def stage_and_accept(self, token):
        """Record each head BEFORE conditioning on its own GT selection.

        GT preceding selections condition later heads only. Raw accept_source
        retains every action, while this ledger remembers the chosen causes.
        An invalid grammar update is atomic for actions and responsibility.
        """
        before = self.snapshot()
        try:
            p, s = self.physical, self.physical.grammar.state
            p._trial(token)  # Grammar validation without future-field queries.
            types = self.query_types()
            stage = dict(cursor_tick=s.cursor, owner=s.owner, phase=s.phase,
                         grammar_types=list(types), types={k: v.record() for k, v in types.items()})
            attribution = {'type': self._choose(types[token.type])}
            if isinstance(token, Note):
                kinds = self.query_kinds()
                stage['kind'] = {k: v.record() for k, v in kinds.items()}
                attribution['kind'] = self._choose(kinds[token.kind])
                pads = self.query_positions(token.kind)
                stage.update(pad_kind=token.kind, pad={k: v.record() for k, v in pads.items()})
                attribution['pad'] = self._choose(pads[token.pad])
            elif isinstance(token, Route):
                first = s.phase in ('route', 'track_done')
                routes = [self.query(Route(r, 0 if first else None)) for r in range(71)]
                stage['route'] = [r.record() for r in routes]
                attribution['route'] = self._choose(routes[token.route_id])
            else:
                if isinstance(token, Moment):
                    self.charged = frozenset()
                name = 'moment' if isinstance(token, Moment) else 'duration'
                stage[name] = self.time_support(token.type)
                attribution[name] = self._choose(self.query(token))
            p.accept_source(token)
            stage['attribution'] = attribution
            return stage
        except Exception:
            self.restore(before)
            raise

    def terminal_stage(self):
        p, s = self.physical, self.physical.grammar.state
        return dict(cursor_tick=s.cursor, owner=s.owner, phase=s.phase,
                    grammar_types=list(p.grammar.allowed_types()),
                    types={k: v.record() for k, v in self.query_types().items()},
                    moment=self.time_support('MOMENT'), attribution={})
