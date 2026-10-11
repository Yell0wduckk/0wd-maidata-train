"""Reuse accepted full-chart supports only at equivalent teacher prefixes.

No tensors or learned state. Window controls never become real chart actions.
"""
from bisect import bisect_left
from dataclasses import replace

from maidata.chart.codec import token_from_dict
from maidata.chart.types import Moment, Route, Note
from maidata.chart.windows import TimeTarget
from maidata.physical import PhysicalState
from maidata.physical.actions import Action, Motion
from maidata.physical.attribution import AttributionState, Evaluation
from maidata.physical.conflicts import Conflict
from maidata.physical.rules import prune_history
from maidata.time import TimingMap
from maidata.time import duration_end
from maidata.physical.rules import EPS


def action_from_record(row):
    return Action(**dict(row, motions=tuple(Motion(**dict(m, entries=tuple(map(tuple, m['entries']))))
                                            for m in row['motions'])))


def equivalent(a, b):
    # token_count belongs to the serialization, not Physical action identity.
    x, y = a.physical, b.physical
    return (replace(x.grammar, token_count=0, window_end=None) ==
            replace(y.grammar, token_count=0, window_end=None)
            and x.actions == y.actions and a.charged == b.charged)


def window_record(stage, end):
    result = {k: v for k, v in stage.items() if k != 'target_token_index'}
    if 'moment' in result:
        support = result['moment']
        rows = []
        for lo, hi, value in support['intervals']:
            if lo < end:
                rows.append([lo, min(hi, end-1), value])
            if hi >= end:
                rows.append([max(lo, end), hi, Evaluation().record()])
        merged = []
        for lo, hi, value in rows:
            if merged and merged[-1][1]+1 == lo and merged[-1][2] == value:
                merged[-1][1] = hi
            else:
                merged.append([lo, hi, value])
        result['moment'] = dict(support, intervals=merged)
    return result


class ChartSupports:
    def __init__(self, record, supports, geometry):
        if (supports['chart_id'] != record['source']['chart_id']
                or supports['token_count'] != len(record['tokens'])):
            raise ValueError('fields/Physical chart or token inventory differs')
        self.tokens = tuple(token_from_dict(t) for t in record['tokens'])
        self.timing = TimingMap.from_ir(record['metadata'])
        self.ticks = [m['tick'] for m in record['token_map']]
        self.start_tick = record['start_tick']
        self.record = record
        self.stages = supports['stages']
        self.actions = {r['token_index']: action_from_record(r['action']) for r in record['actions']}
        self.state = AttributionState(PhysicalState(self.timing, geometry,
            audio_end_seconds=record['source']['audio_end_seconds']))
        self.state.inject_boundary(self.tokens[0], start_tick=self.start_tick)
        self.prefixes = [self.state.snapshot()]
        for i in range(1, len(self.tokens)):
            self._accept_cached(i)
            self.prefixes.append(self.state.snapshot())
        self.state.physical.grammar.finish()
        self.counterfactuals = {}

    def _accept_cached(self, index):
        state, token = self.state, self.tokens[index]
        p = state.physical
        completion = p.grammar.accept(token)
        if completion is not None:
            p.actions += (self.actions[index],)
        if isinstance(token, Moment):
            p.actions = prune_history(p.actions, self.timing.second(p.grammar.state.cursor))
            state.charged = frozenset()
        for witnesses in self.stages[index-1]['attribution'].values():
            state.charged |= frozenset(Conflict.from_record(r) for r in witnesses)

    def before_window(self, start_tick):
        return self.prefixes[max(0, bisect_left(self.ticks, start_tick, lo=1)-1)]

    def source_indices(self, example):
        indices = [i for i in range(bisect_left(self.ticks, example.start_tick, lo=1),
                                   bisect_left(self.ticks, example.end_tick, lo=1))
                   if not (self.ticks[i] == example.start_tick and isinstance(self.tokens[i], Moment))]
        targets = example.targets[example.boundary_index:]
        if (indices and isinstance(targets[0], Moment)
                and not isinstance(self.tokens[indices[0]], Moment)):
            indices.insert(0, None)
        if len(indices)+1 != len(targets) or not isinstance(targets[-1], TimeTarget):
            raise ValueError('window/source supervision alignment differs')
        return indices + [None]

    def _duration_source(self, state):
        """Reuse real-prefix integer runs; query only missing counterfactuals.

        Charged is deliberately absent from the key: raw witnesses do not
        depend on the attribution ledger. Consumers reattribute these runs.
        """
        p = state.physical
        grammar = replace(p.grammar.snapshot(), token_count=0, window_end=None)
        key = (grammar, p.actions)
        if key not in self.counterfactuals:
            from maidata.physical.support import duration_support
            source = state._fork(p.grammar)
            source.duration_end_seconds = None
            source.duration_support_provider = None
            self.counterfactuals[key] = duration_support(p, query=source.query, record=Evaluation.record)
        return self.counterfactuals[key]

    def window(self, example, *, duration_cap_seconds=None):
        saved = self.state.snapshot()
        previous_cap = self.state.duration_end_seconds
        previous_provider = self.state.duration_support_provider
        previous_routes = self.state.route_evaluation_provider
        reused = recomputed = 0
        try:
            self.state.restore(self.before_window(example.start_tick))
            self.state.inject_boundary(example.inputs[example.boundary_index], start_tick=example.start_tick,
                history_present=not example.history_dropped, window_end=example.end_tick)
            indices = self.source_indices(example)[:-1]
            targets = example.targets[example.boundary_index:]
            # The audio grid can start before the source's initial M(0). Its
            # first NOTE then needs a new real MOMENT absent from the source.
            rows = [None] * example.boundary_index
            for i, target in zip(indices, targets[:-1]):
                state = self.state
                state.duration_end_seconds = duration_cap_seconds
                state.duration_support_provider = self._duration_source if duration_cap_seconds is not None else None
                state.route_evaluation_provider = None
                if i is None:
                    rows.append(self.state.stage_and_accept(target))
                    recomputed += 1
                    continue
                if target != self.tokens[i] and not isinstance(target, Moment):
                    raise ValueError('window changes real chart content')
                same = target == self.tokens[i] and equivalent(state.snapshot(), self.prefixes[i-1])
                pending = (state.physical.grammar.state.phase in ('hold', 'chain')
                           or isinstance(target, Route))
                # NOTE hold-kind/pad uses the existing two-tick proxy, not an
                # existential full-duration projection. Usually four audio
                # after-beats contain it; a sharp BPM boundary can exclude
                # even that proxy, in which case this stage also changes.
                pending = pending or (isinstance(target, Note) and duration_cap_seconds is not None
                    and duration_end(self.timing, state.physical.grammar.state.cursor, 2)
                    > duration_cap_seconds+EPS)
                pending = pending and (duration_cap_seconds is not None
                    and duration_cap_seconds < state.physical.audio_end_seconds-EPS)
                if same and duration_cap_seconds is not None and pending:
                    old = self.stages[i-1]
                    if 'duration' in old:
                        grammar = replace(state.physical.grammar.snapshot(), token_count=0, window_end=None)
                        self.counterfactuals[(grammar, state.physical.actions)] = old['duration']
                    # Normal ROUTE prefixes remain repairable and their raw
                    # witnesses do not depend on duration. WiFi cannot extend,
                    # so only that candidate needs a new cap projection.
                    if 'route' in old:
                        def cached_route(current, token, records=old['route']):
                            if token.route_id == 70:
                                return None
                            record = records[token.route_id]
                            return current._leaf(record['basic'], frozenset(
                                Conflict.from_record(c) for c in record['conflicts']), deferred=record['deferred'])
                        state.route_evaluation_provider = cached_route
                    rows.append(state.stage_and_accept(target))
                    recomputed += 1
                elif same:
                    rows.append(window_record(self.stages[i-1], example.end_tick))
                    self._accept_cached(i)
                    reused += 1
                else:
                    rows.append(self.state.stage_and_accept(target))
                    recomputed += 1
            rows.append(self.state.terminal_stage())
            self.state.close_window(targets[-1])
            return tuple(rows), dict(reused=reused, recomputed=recomputed+1)
        finally:
            self.state.restore(saved)
            self.state.duration_end_seconds = previous_cap
            self.state.duration_support_provider = previous_provider
            self.state.route_evaluation_provider = previous_routes
