"""Hierarchical field sampling against current, prefix-only rule queries."""
from itertools import product
from dataclasses import dataclass, replace

import torch

from maidata.chart.types import (GENERATED_TYPES, KINDS, FLAGS, APPLICABLE,
    BUTTON_PADS, TOUCH_PADS, Moment, Note, Route, HoldDuration, SlideDuration)
from maidata.chart.windows import TimeTarget
from maidata.time.discrete import compile_support_tree, TimeQuery, CONTINUE
from .options import resolve_seed, resolve_repetition, validate_slide_limit
from .tail_collision import TailRepair
from .hold_tail import FieldProbabilities, DurationDistribution


@dataclass(frozen=True)
class OnsetLimit:
    """Editor output boundary, never a model window terminal or chart token."""
    tick: int


class Sampler:
    def __init__(self, *, temperature=.4, seed=20261010, physical_hard_mask=True, top_k=0, top_p=1.,
                 floor_tick=None, repetition_window=32, repetition_penalty=1.,
                 pad_repetition_window=None, pad_repetition_penalty=None,
                 route_repetition_window=None, route_repetition_penalty=None, max_slide_judge_areas=40):
        # floor_tick: editor-anchored generation never adds content to an onset
        # (or empty gap) before this absolute tick. None leaves sampling unchanged.
        self.floor_tick = floor_tick
        validate_slide_limit(max_slide_judge_areas)
        self.max_slide_judge_areas = max_slide_judge_areas
        if not 0 <= temperature < float('inf'):
            raise ValueError('temperature must be finite and nonnegative')
        self.temperature = temperature
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValueError('top_k must be a nonnegative integer')
        if not 0 < top_p <= 1:
            raise ValueError('top_p must be in (0, 1]')
        self.top_k, self.top_p = top_k, top_p
        self.repetition_window, self.repetition_penalty = repetition_window, repetition_penalty
        self.repetition = resolve_repetition(repetition_window, repetition_penalty,
            pad_repetition_window=pad_repetition_window, pad_repetition_penalty=pad_repetition_penalty,
            route_repetition_window=route_repetition_window, route_repetition_penalty=route_repetition_penalty)
        self.seed = resolve_seed(seed)
        self.generator = torch.Generator(device='cpu').manual_seed(self.seed)
        self.physical_hard_mask = physical_hard_mask
        self.field_choice = self.duration_choice = None

    def filtered_scores(self, scores, allowed=None, *, identities=None, history=(), field=None):
        """Original sampling support, before the deferred Hold/final-area rule."""
        values = scores.detach().float().cpu().flatten().clone()
        if torch.isnan(values).any() or torch.isposinf(values).any():
            raise ValueError('nonfinite model sampling scores')
        if allowed is not None:
            values.masked_fill_(~torch.as_tensor(allowed, dtype=torch.bool, device='cpu'), -torch.inf)
        penalty = self.repetition.get(f'{field}_repetition_penalty', 0.)
        if penalty and identities is not None:
            recent = history[-self.repetition[f'{field}_repetition_window']:]
            seen = ({t.pad for t in recent if isinstance(t, Note)} if field == 'pad' else
                    {t.route_id for t in recent if isinstance(t, Route)} if field == 'route' else set())
            if seen:
                repeated = torch.tensor([v in seen for v in identities], dtype=torch.bool)
                values[repeated] -= penalty
        # Select the support once, before the no-replacement search. Do not
        # silently widen it when a retained candidate cannot be completed.
        # The default branch is untouched, including the old RNG sequence.
        if (self.top_k or self.top_p < 1) and torch.isfinite(values).any():
            indices = values.argsort(descending=True, stable=True)
            indices = indices[torch.isfinite(values[indices])]
            if self.top_k:
                indices = indices[:self.top_k]
            if self.top_p < 1:
                if self.temperature == 0:
                    # Zero-temperature distribution is concentrated on the
                    # deterministic argmax. Never widen this filtered support.
                    indices = indices[:1]
                else:
                    logits = values[indices]
                    probabilities = ((logits-logits.max()) / self.temperature).softmax(0)
                    # Include the candidate that crosses the cumulative threshold.
                    indices = indices[(probabilities.cumsum(0)-probabilities) < self.top_p]
            keep = torch.zeros_like(values, dtype=torch.bool)
            keep[indices] = True
            values.masked_fill_(~keep, -torch.inf)
        return values

    def log_probabilities(self, values):
        if not torch.isfinite(values).any():
            return torch.full_like(values, -torch.inf, dtype=torch.float64)
        return (values.double() / (self.temperature or 1.)).log_softmax(0)

    def order(self, scores, allowed=None, *, identities=None, history=(), field=None):
        yield from self.order_filtered(self.filtered_scores(scores, allowed,
            identities=identities, history=history, field=field))

    def order_filtered(self, values):
        """Lazy weighted sampling without replacement; unchanged normal RNG."""
        values = values.clone()
        while torch.isfinite(values).any():
            index = (int(values.argmax()) if self.temperature == 0 else
                     int(torch.multinomial(((values-values.max()) / self.temperature).softmax(0), 1, generator=self.generator)))
            values[index] = -torch.inf
            yield index

    def allowed(self, evaluation):
        d = evaluation.raw.decision() if hasattr(evaluation, 'raw') else evaluation
        return not d.basic_errors and 'multitouch' not in d.physical and (not self.physical_hard_mask or not d.physical)

    def support_tree(self, stage, timing, **kwargs):
        # Recompile CONT existence with only mandatory multitouch bits retained.
        # Checking only the final remainder would admit impossible CONT paths.
        if not self.physical_hard_mask:
            key = 'moment' if kwargs['kind'] == 'MOMENT' else 'duration'
            support = stage[key]
            stage = dict(stage, **{key: dict(support, intervals=[
                [a, b, dict(row, physical=row.get('physical', 0) & 1)]
                for a, b, row in support['intervals']])})
        return compile_support_tree(stage, timing, **kwargs)

    def contents(self, decoder, h, kind, oracle, timing, distributions, *, end_tick, cap_tick,
                 moment_remaining=None, output=None, return_cache=False, history=(), onset_end_tick=None,
                 moment_floor=None):
        heads, device = decoder.heads, h.device
        if kind == 'NOTE':
            kinds = oracle.query_kinds()
            for k in self.order(heads.kind(h), [self.allowed(kinds[n]) for n in KINDS]):
                name, kt = KINDS[k], torch.tensor(k, device=device)
                pads = TOUCH_PADS if name in ('touch', 'touch_hold') else BUTTON_PADS
                positions = oracle.query_positions(name)
                values = self.filtered_scores(heads.pad_logits(h, kt)[:len(pads)],
                    [self.allowed(positions[n]) for n in pads], identities=pads, history=history, field='pad')
                probabilities = FieldProbabilities(tuple(self.log_probabilities(values).tolist()), tuple(pads))
                for p in self.order_filtered(values):
                    logits = heads.flag_logits(h, kt, torch.tensor(p, device=device)).float().log_softmax(-1)
                    applicable = APPLICABLE[name]
                    options = [Note.make(name, pads[p], **dict(zip(applicable, bits)))
                               for bits in product((0, 1), repeat=len(applicable))]
                    # Flags are independent Bernoulli heads. Enumerating their tiny
                    # joint support also excludes a short hold beyond audio_end.
                    scores = torch.stack([sum(logits[FLAGS.index(f), getattr(n, f)] for f in applicable) for n in options])
                    for f in self.order(scores, [self.allowed(oracle.query(n)) for n in options]):
                        note = options[f]
                        repair = oracle.tail_repair(note) if hasattr(oracle, 'tail_repair') else None
                        self.field_choice = probabilities if name in ('hold', 'touch_hold') else None
                        if repair is not None:
                            # Stay in the original filtered pad support and keep
                            # kind/flags fixed. Exhaust replacements (including
                            # their downstream search) before EX/time fallback.
                            alternatives = values.clone()
                            for index, pad in enumerate(pads):
                                replacement = replace(note, pad=pad)
                                if (index == p or not torch.isfinite(alternatives[index])
                                        or not self.allowed(oracle.query(replacement))
                                        or oracle.tail_repair(replacement) is not None):
                                    alternatives[index] = -torch.inf
                            for index in self.order_filtered(alternatives):
                                self.field_choice = probabilities if name in ('hold', 'touch_hold') else None
                                candidate = TailRepair(note, repair.gaps_ms, replacement=replace(note, pad=pads[index]))
                                yield (candidate,output.cache) if return_cache else candidate
                        self.field_choice = probabilities if name in ('hold', 'touch_hold') else None
                        candidate = repair or note
                        yield (candidate,output.cache) if return_cache else candidate
                        if isinstance(repair, TailRepair) and repair.minimum_tick is not None:
                            # Pad replacements are exhausted. Retry time,
                            # never change the colliding note's kind to Touch.
                            return
        elif kind == 'ROUTE':
            first = oracle.physical.grammar.state.phase in ('route', 'track_done')
            candidates = [Route(i, 0 if first else None) for i in range(71)]
            values = self.filtered_scores(heads.route(h), [self.allowed(oracle.query(n)) for n in candidates],
                identities=range(71), history=history, field='route')
            probabilities = FieldProbabilities(tuple(self.log_probabilities(values).tolist()), tuple(range(71)))
            for r in self.order_filtered(values):
                if first:
                    for b in self.order(heads.break_logits(h, torch.tensor(r, device=device))):
                        self.field_choice = probabilities
                        yield (Route(r,b),output.cache) if return_cache else Route(r,b)
                else:
                    self.field_choice = probabilities
                    yield (candidates[r],output.cache) if return_cache else candidates[r]
        else:
            cursor = oracle.physical.grammar.state.cursor
            support = oracle.time_support(kind)
            if kind != 'MOMENT' and support['domain'][1] < support['domain'][0]:
                return
            stage = dict(cursor_tick=cursor, **{'moment' if kind == 'MOMENT' else 'duration': support})
            if output is None:
                raise ValueError('time generation requires a reusable completed main prefix')
            tree = self.support_tree(stage,timing,kind=kind,window_end=end_tick,cap_tick=cap_tick,target=None)
            if kind != 'MOMENT':
                distribution = DurationDistribution(self, decoder, output, tree, timing)
                for leaf in distribution.sampled():
                    self.duration_choice = distribution
                    yield (leaf.token, leaf.cache) if return_cache else leaf.token
                return
            from maidata.decoder.batching import pack_tokens
            from maidata.decoder.sequence import continue_token
            def emit(token, current):
                return (token,current.cache) if return_cache else token
            def visit(depth, query, current):
                qh=current.hidden[0,-1]
                valid = tree.hard[depth]
                floor = self.floor_tick if moment_floor is None else max(self.floor_tick or 0, moment_floor)
                if kind == 'MOMENT' and floor is not None:
                    # Short value c completes at cursor+elapsed+c+1; CONT stays open.
                    base = cursor + query.elapsed_ticks + 1
                    valid = tuple(ok and (c == CONTINUE or base + c >= floor)
                                  for c, ok in enumerate(valid))
                for choice in self.order(heads.time_logits(qh,kind),valid):
                    value = query.advance(choice,window_end_tick=end_tick)
                    beyond_output = (kind == 'MOMENT' and onset_end_tick is not None and
                        (value.lower_bound_tick >= onset_end_tick if choice == CONTINUE else
                         cursor + value.complete_ticks >= onset_end_tick))
                    if beyond_output:
                        yield emit(OnsetLimit(onset_end_tick), current)
                    elif value.closed_by_continue:
                        lower=value.lower_bound_tick
                        yield emit(TimeTarget(timing.delta_ms(cursor,lower),lower,False,True,True,lower,
                                              cursor+value.query.elapsed_ticks),current)
                    elif choice == CONTINUE:
                        token=continue_token(kind,timing,cursor,value.query.elapsed_ticks)
                        beat=value.query.query_beat(timing)
                        batch=pack_tokens([[token]],[[beat]],device=device,slide_positions=[[beat]])
                        types=torch.full_like(batch.types,GENERATED_TYPES.index(kind))
                        routed=decoder.begin(batch,current.cache.memory,cache=current.cache)
                        following=decoder.complete(routed,types)
                        yield from visit(depth+1,value.query,following)
                    elif kind == 'MOMENT':
                        tick=cursor+value.complete_ticks
                        delta=timing.delta_ms(cursor,tick)
                        yield emit(TimeTarget(delta,tick,False,True) if value.closes_window else Moment(delta),current)
                    else:
                        yield emit((HoldDuration if kind=='HOLD_DURATION' else SlideDuration)(value.complete_ticks),current)
            yield from visit(0,TimeQuery(kind,cursor),output)

    @staticmethod
    def slide_judge_area_count(oracle):
        from maidata.chart.routes import chain_paths
        state = oracle.physical.grammar.state
        count, previous = 0, None
        for key in chain_paths(state.note.pad, state.routes):
            entries = oracle.physical.geometry.paths[key].entries
            count += len(entries) - int(bool(entries) and previous == entries[0][1])
            if entries:
                previous = entries[-1][1]
        return count

    def choices(self, decoder, routed, oracle, timing, distributions, *, end_tick, cap_tick,
                moment_remaining=None, history=(), onset_end_tick=None, force_moment=False, moment_floor=None):
        if hasattr(oracle,'set_duration_cap'):
            oracle.set_duration_cap(timing.second(cap_tick))
        types = oracle.query_types()
        allowed = [name in types and self.allowed(types[name]) for name in GENERATED_TYPES]
        if (oracle.physical.grammar.state.phase == 'chain' and
                self.slide_judge_area_count(oracle) > self.max_slide_judge_areas):
            # Allow the chosen segment to cross the bound, then finish this
            # branch. This remains active during a Physical-only fallback.
            allowed = [ok and name == 'SLIDE_DURATION' for name, ok in zip(GENERATED_TYPES, allowed)]
        if force_moment:
            allowed = [ok and name == 'MOMENT' for name, ok in zip(GENERATED_TYPES, allowed)]
        if moment_remaining is not None and moment_remaining <= 0:
            phase = oracle.physical.grammar.state.phase
            required = {'hold': 'HOLD_DURATION', 'route': 'ROUTE',
                        'chain': 'SLIDE_DURATION'}.get(phase, 'MOMENT')
            allowed = [ok and name == required for name, ok in zip(GENERATED_TYPES, allowed)]
        if self.floor_tick is not None and oracle.physical.grammar.state.cursor < self.floor_tick:
            # The current onset lies before the anchor: only a new MOMENT may follow.
            allowed = [ok and name == 'MOMENT' for name, ok in zip(GENERATED_TYPES, allowed)]
        for index in self.order(routed.logits[0, -1], allowed):
            target_type = torch.full_like(routed.valid, index, dtype=torch.long)
            result = decoder.complete(routed, target_type)
            h = result.hidden[0,-1]
            yield from self.contents(decoder,h,GENERATED_TYPES[index],oracle,timing,distributions,
                end_tick=end_tick,cap_tick=cap_tick,moment_remaining=moment_remaining,
                output=result,return_cache=True,history=history,onset_end_tick=onset_end_tick,
                moment_floor=moment_floor)
