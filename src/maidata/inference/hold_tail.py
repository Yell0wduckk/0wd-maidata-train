"""Inference-only, probability-weighted repair of Hold / final Slide contacts.

No training oracle, cached Physical labels or model parameters are changed.
Candidate probabilities are those of the original filtered heads, before this
new rule is applied. All speculative states and CONT outputs are request-local.
"""
from dataclasses import dataclass, replace
from bisect import bisect_left
import math

import torch

from maidata.chart.types import (GENERATED_TYPES, Route, HoldDuration, SlideDuration)
from maidata.chart import token_to_dict
from maidata.chart.routes import chain_paths
from maidata.physical.actions import materialize
from maidata.physical.geometry import pad_bit
from maidata.physical.rules import EPS
from maidata.physical.support import _partition, decision_record
from maidata.time import advance_ticks
from maidata.time.discrete import CONTINUE, TIME_BLOCK


def contacts(state, token):
    """Only the final segment's final judge-area entry is relevant."""
    try:
        _, completion = state._trial(token)
    except ValueError:
        return ()
    if completion is None:
        return ()
    candidate = materialize(completion, state.timing, state.geometry)
    result = []
    for old in state.actions:
        hold, slide = ((old, candidate) if candidate.track else (candidate, old))
        if hold.kind not in ('hold', 'touch_hold') or not slide.track:
            continue
        if not slide.motions[-1].entries:
            continue
        entered, mask = slide.motions[-1].entries[-1]
        if not mask & pad_bit(hold.pad):
            continue
        checked = advance_ticks(state.timing, entered, -96)
        # Actions include a 22-tick release guard; recover the true hold end.
        end = advance_ticks(state.timing, hold.press_end, -22)
        if hold.press_start <= entered + EPS and checked < end - EPS:
            result.append(dict(hold_owner=hold.owner, hold_tick=hold.tick,
                hold_pad=hold.pad, hold_start=hold.press_start, hold_end=end,
                slide_owner=slide.owner, slide_tick=slide.tick,
                track_index=slide.track_index, path=slide.motions[-1].path,
                judge_area_mask=mask, entry_seconds=entered, checked_seconds=checked))
    return tuple(result)


def fork(state, snapshot=None):
    result = type(state)(state.timing, state.geometry, audio_end_seconds=state.audio_end_seconds)
    result.duration_end_seconds = state.duration_end_seconds
    result.restore(snapshot or state.snapshot())
    return result


def legal(state, token, sampler):
    return sampler.allowed(state.query(token)) and not contacts(state, token)


def strict_support(state):
    """Exact inference-only duration intervals, including new-rule crossings.

    Slide entry time is affine in written duration. Its preceding beat overlaps
    a hold when entry >= hold start and entry < hold end shifted forward a beat.
    Only the end bound shifts, including through BPM changes. No tick enumeration.
    """
    base = state.time_support('SLIDE_DURATION' if state.grammar.state.phase == 'chain' else 'HOLD_DURATION')
    lower, upper = base['domain']
    if upper < lower:
        return base
    slide = state.grammar.state.phase == 'chain'
    constructor = SlideDuration if slide else HoldDuration
    _, first = state._trial(constructor(lower))
    a = materialize(first, state.timing, state.geometry)
    roots = [v for lo, hi, _ in base['intervals'] for v in (lo, hi + 1)]
    if slide and a.motions[-1].entries:
        _, second = state._trial(constructor(lower + 1))
        b = materialize(second, state.timing, state.geometry)
        t = a.motions[-1].entries[-1][0]
        slope = b.motions[-1].entries[-1][0] - t
        mask = a.motions[-1].entries[-1][1]
        if slope:
            for hold in state.actions:
                if hold.kind in ('hold', 'touch_hold') and mask & pad_bit(hold.pad):
                    end = advance_ticks(state.timing, hold.press_end, -22)
                    roots.extend(lower + (bound - t) / slope
                                 for bound in (hold.press_start, advance_ticks(state.timing, end, 96)))
    elif not slide:
        origin = a.press_start
        scale = state.timing.written_tick_seconds(state.grammar.state.cursor)
        for old in state.actions:
            if old.track and old.motions[-1].entries:
                entered, mask = old.motions[-1].entries[-1]
                checked = advance_ticks(state.timing, entered, -96)
                if mask & pad_bit(a.pad) and origin <= entered + EPS:
                    roots.append((checked - origin) / scale)

    def query(ticks):
        token = constructor(ticks)
        decision = state.query(token)
        conflict = bool(contacts(state, token))
        if conflict:
            decision = replace(decision, physical=decision.physical | {'slide_too_fast'})
        return dict(decision_record(decision), hold_slide_final_area=conflict)

    return dict(base, intervals=_partition(roots, lower, upper, query, record=lambda value: value))


@dataclass(frozen=True)
class FieldProbabilities:
    log_probabilities: tuple
    identities: tuple


@dataclass(frozen=True)
class FieldAnchor:
    snapshot: object
    count: int
    token: object
    probabilities: FieldProbabilities
    cache: object


@dataclass(frozen=True)
class DurationLeaf:
    token: object
    log_probability: float
    cache: object


class DurationDistribution:
    """One lazy CONT chain with reusable completed outputs and full leaf mass."""
    def __init__(self, sampler, decoder, output, tree, timing):
        self.sampler, self.decoder, self.tree, self.timing = sampler, decoder, tree, timing
        self.outputs = [output]
        self.values = {}
        self.extra_forwards = 0

    def node(self, depth):
        from maidata.decoder.batching import pack_tokens
        from maidata.decoder.sequence import continue_token
        while len(self.outputs) <= depth:
            elapsed, beat = self.tree.queries[len(self.outputs)]
            previous = self.outputs[-1]
            token = continue_token(self.tree.kind, self.timing, self.tree.origin_tick, elapsed)
            batch = pack_tokens([[token]], [[beat]], device=previous.hidden.device, slide_positions=[[beat]])
            routed = self.decoder.begin(batch, previous.cache.memory, cache=previous.cache)
            following = self.decoder.complete(routed, torch.full_like(batch.types, GENERATED_TYPES.index(self.tree.kind)))
            self.outputs.append(following)
            self.extra_forwards += 1
        if depth not in self.values:
            masks = self.tree.hard
            self.values[depth] = self.sampler.filtered_scores(
                self.decoder.heads.time_logits(self.outputs[depth].hidden[0, -1], self.tree.kind), masks[depth])
        return self.outputs[depth], self.values[depth]

    def leaves(self):
        accumulated = 0.
        constructor = HoldDuration if self.tree.kind == 'HOLD_DURATION' else SlideDuration
        for depth in range(len(self.tree.queries)):
            output, values = self.node(depth)
            probabilities = self.sampler.log_probabilities(values)
            for index in torch.isfinite(values[:CONTINUE]).nonzero().flatten().tolist():
                yield DurationLeaf(constructor(depth * TIME_BLOCK + index + 1),
                                   accumulated + float(probabilities[index]), output.cache)
            if not torch.isfinite(values[CONTINUE]):
                break
            accumulated += float(probabilities[CONTINUE])

    def sampled(self):
        constructor = HoldDuration if self.tree.kind == 'HOLD_DURATION' else SlideDuration

        def visit(depth, accumulated):
            output, values = self.node(depth)
            probabilities = self.sampler.log_probabilities(values)
            for index in self.sampler.order_filtered(values):
                probability = accumulated + float(probabilities[index])
                if index == CONTINUE:
                    yield from visit(depth + 1, probability)
                else:
                    yield DurationLeaf(constructor(depth * TIME_BLOCK + index + 1), probability, output.cache)
        yield from visit(0, 0.)


@dataclass(frozen=True)
class ActionRepair:
    tokens: tuple
    caches: tuple
    record: dict


def repairs(sampler, decoder, anchor, state, duration, distribution, witnesses, *, end_tick, cap_tick):
    """Weighted no-replacement alternatives, restored at the position anchor."""
    before = fork(state, anchor.snapshot.rules)
    current = fork(state)
    pool = []
    slide = isinstance(anchor.token, Route)
    old_endpoint = chain_paths(current.grammar.state.note.pad, current.grammar.state.routes)[-1][-1] if slide else None
    for identity, probability in zip(anchor.probabilities.identities, anchor.probabilities.log_probabilities):
        if not math.isfinite(probability):
            continue
        replacement = replace(anchor.token, **({'route_id': identity} if slide else {'pad': identity}))
        if replacement == anchor.token or not sampler.allowed(before.query(replacement)):
            continue
        trial = fork(before)
        trial.accept_source(replacement)
        if slide:
            endpoint = chain_paths(trial.grammar.state.note.pad, trial.grammar.state.routes)[-1][-1]
            if endpoint == old_endpoint:
                continue
            support = strict_support(trial)
            if not any(not r['basic'] and not r.get('hold_slide_final_area')
                       and not (r['physical'] & (15 if sampler.physical_hard_mask else 1))
                       for _, _, r in support['intervals']):
                continue
        elif (not legal(trial, duration, sampler)
              or trial.tail_repair(replacement) is not None):
            continue
        pool.append((probability, replacement, trial, None))
    previously_computed = distribution.extra_forwards
    intervals = strict_support(current)['intervals']
    ends = [hi for _, hi, _ in intervals]
    for leaf in distribution.leaves():
        index = bisect_left(ends, leaf.token.ticks)
        record = intervals[index][2] if index < len(intervals) and intervals[index][0] <= leaf.token.ticks else None
        if (leaf.token != duration and record is not None and not record['basic']
                and not record.get('hold_slide_final_area')
                and not (record['physical'] & (15 if sampler.physical_hard_mask else 1))):
            pool.append((leaf.log_probability, anchor.token, current, leaf))
    if not pool:
        return
    weights = torch.tensor([entry[0] for entry in pool], dtype=torch.float64)
    while torch.isfinite(weights).any():
        index = (int(weights.argmax()) if sampler.temperature == 0 else
                 int(torch.multinomial((weights - weights.max()).softmax(0), 1, generator=sampler.generator)))
        weights[index] = -torch.inf
        _, replacement, trial, leaf = pool[index]
        extra = distribution.extra_forwards - previously_computed
        field = ('route' if slide else 'pad') if leaf is None else 'duration'
        if leaf is None:
            from maidata.decoder.batching import pack_tokens
            from maidata.decoder.sequence import expand_inputs
            beat = trial.grammar.state.cursor / 96
            sequence = expand_inputs((replacement,), (beat,), trial.timing)
            batch = pack_tokens([sequence['tokens']], [sequence['beats']], device=decoder.type_head.weight.device,
                                slide_positions=[sequence['slide_beats']])
            routed = decoder.begin(batch, anchor.cache.memory, cache=anchor.cache)
            kind = 'SLIDE_DURATION' if slide else 'HOLD_DURATION'
            output = decoder.complete(routed, torch.full_like(batch.types, GENERATED_TYPES.index(kind)))
            tree = sampler.support_tree(dict(cursor_tick=trial.grammar.state.cursor, duration=strict_support(trial)),
                trial.timing, kind=kind, window_end=end_tick, cap_tick=cap_tick)
            changed = DurationDistribution(sampler, decoder, output, tree, trial.timing)
            if slide:
                leaf = next((candidate for candidate in changed.sampled()
                             if legal(trial, candidate.token, sampler)), None)
                if leaf is None:
                    continue
            else:
                depth = (duration.ticks - 1) // TIME_BLOCK
                leaf = DurationLeaf(duration, 0., changed.node(depth)[0].cache)
            extra += 1 + changed.extra_forwards
        if not legal(trial, leaf.token, sampler):
            continue
        yield ActionRepair((replacement, leaf.token), (anchor.cache, leaf.cache), dict(
            rule='hold_slide_final_area', field=field, conflicts=list(witnesses),
            original_field=token_to_dict(anchor.token), replacement_field=token_to_dict(replacement),
            original_duration=token_to_dict(duration), replacement_duration=token_to_dict(leaf.token),
            extra_model_forwards=extra))
