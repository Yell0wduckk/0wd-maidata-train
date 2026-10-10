"""Reuse accepted full-chart supports only at equivalent teacher prefixes.

No tensors or learned state. Window controls never become real chart actions.
"""
from bisect import bisect_left
from dataclasses import replace

from maidata.chart.codec import token_from_dict
from maidata.chart.types import Moment
from maidata.chart.windows import TimeTarget
from maidata.physical import PhysicalState
from maidata.physical.actions import Action, Motion
from maidata.physical.attribution import AttributionState, Evaluation
from maidata.physical.conflicts import Conflict
from maidata.physical.rules import prune_history
from maidata.time import TimingMap


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

    def window(self, example):
        saved = self.state.snapshot()
        reused = recomputed = 0
        try:
            self.state.restore(self.before_window(example.start_tick))
            self.state.inject_boundary(example.inputs[example.boundary_index], start_tick=example.start_tick,
                history_present=not example.history_dropped, window_end=example.end_tick)
            indices = [i for i in range(bisect_left(self.ticks, example.start_tick, lo=1),
                                       bisect_left(self.ticks, example.end_tick, lo=1))
                       if not (self.ticks[i] == example.start_tick and isinstance(self.tokens[i], Moment))]
            targets = example.targets[example.boundary_index:]
            # The audio grid can start before the source's initial M(0). Its
            # first NOTE then needs a new real MOMENT absent from the source.
            if (indices and isinstance(targets[0], Moment)
                    and not isinstance(self.tokens[indices[0]], Moment)):
                indices.insert(0, None)
            if len(indices)+1 != len(targets) or not isinstance(targets[-1], TimeTarget):
                raise ValueError('window/source supervision alignment differs')
            rows = [None] * example.boundary_index
            for i, target in zip(indices, targets[:-1]):
                if i is None:
                    rows.append(self.state.stage_and_accept(target))
                    recomputed += 1
                    continue
                if target != self.tokens[i] and not isinstance(target, Moment):
                    raise ValueError('window changes real chart content')
                if target == self.tokens[i] and equivalent(self.state.snapshot(), self.prefixes[i-1]):
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
