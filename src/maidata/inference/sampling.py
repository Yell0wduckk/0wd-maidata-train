"""Hierarchical field sampling against current, prefix-only rule queries."""
from itertools import product

import torch

from maidata.chart.types import (GENERATED_TYPES, KINDS, FLAGS, APPLICABLE,
    BUTTON_PADS, TOUCH_PADS, Moment, Note, Route, HoldDuration, SlideDuration)
from maidata.chart.windows import TimeTarget
from maidata.time.discrete import compile_support_tree, TimeQuery, CONTINUE


class Sampler:
    def __init__(self, *, temperature=1., seed=20261010, physical_hard_mask=True):
        if not 0 <= temperature < float('inf'):
            raise ValueError('temperature must be finite and nonnegative')
        self.temperature = temperature
        self.generator = torch.Generator(device='cpu').manual_seed(seed)
        self.physical_hard_mask = physical_hard_mask

    def order(self, scores, allowed=None):
        """Lazy weighted sampling without replacement; retries never repeat a choice."""
        values = scores.detach().float().cpu().flatten().clone()
        if torch.isnan(values).any() or torch.isposinf(values).any():
            raise ValueError('nonfinite model sampling scores')
        if allowed is not None:
            values.masked_fill_(~torch.as_tensor(allowed, dtype=torch.bool, device='cpu'), -torch.inf)
        while torch.isfinite(values).any():
            index = (int(values.argmax()) if self.temperature == 0 else
                     int(torch.multinomial(((values-values.max()) / self.temperature).softmax(0), 1, generator=self.generator)))
            values[index] = -torch.inf
            yield index

    def allowed(self, evaluation):
        d = evaluation.raw.decision() if hasattr(evaluation, 'raw') else evaluation
        return not d.basic_errors and (not self.physical_hard_mask or not d.physical)

    def contents(self, decoder, h, kind, oracle, timing, distributions, *, end_tick, cap_tick,
                 moment_remaining=None, output=None, return_cache=False):
        heads, device = decoder.heads, h.device
        if kind == 'NOTE':
            kinds = oracle.query_kinds()
            for k in self.order(heads.kind(h), [self.allowed(kinds[n]) for n in KINDS]):
                name, kt = KINDS[k], torch.tensor(k, device=device)
                pads = TOUCH_PADS if name in ('touch', 'touch_hold') else BUTTON_PADS
                positions = oracle.query_positions(name)
                for p in self.order(heads.pad_logits(h, kt)[:len(pads)], [self.allowed(positions[n]) for n in pads]):
                    logits = heads.flag_logits(h, kt, torch.tensor(p, device=device)).float().log_softmax(-1)
                    applicable = APPLICABLE[name]
                    options = [Note.make(name, pads[p], **dict(zip(applicable, bits)))
                               for bits in product((0, 1), repeat=len(applicable))]
                    # Flags are independent Bernoulli heads. Enumerating their tiny
                    # joint support also excludes a short hold beyond audio_end.
                    scores = torch.stack([sum(logits[FLAGS.index(f), getattr(n, f)] for f in applicable) for n in options])
                    for f in self.order(scores, [self.allowed(oracle.query(n)) for n in options]):
                        yield (options[f],output.cache) if return_cache else options[f]
        elif kind == 'ROUTE':
            first = oracle.physical.grammar.state.phase in ('route', 'track_done')
            candidates = [Route(i, 0 if first else None) for i in range(71)]
            for r in self.order(heads.route(h), [self.allowed(oracle.query(n)) for n in candidates]):
                if first:
                    for b in self.order(heads.break_logits(h, torch.tensor(r, device=device))):
                        yield (Route(r,b),output.cache) if return_cache else Route(r,b)
                else:
                    yield (candidates[r],output.cache) if return_cache else candidates[r]
        else:
            cursor = oracle.physical.grammar.state.cursor
            support = oracle.time_support(kind)
            if kind != 'MOMENT' and support['domain'][1] < support['domain'][0]:
                return
            stage = dict(cursor_tick=cursor, **{'moment' if kind == 'MOMENT' else 'duration': support})
            if output is None:
                raise ValueError('time generation requires a reusable completed main prefix')
            tree = compile_support_tree(stage,timing,kind=kind,window_end=end_tick,cap_tick=cap_tick,target=None)
            from maidata.decoder.batching import pack_tokens
            from maidata.decoder.sequence import continue_token
            def emit(token, current):
                return (token,current.cache) if return_cache else token
            def visit(depth, query, current):
                qh=current.hidden[0,-1]
                valid = tree.hard[depth] if self.physical_hard_mask else tree.basic[depth]
                for choice in self.order(heads.time_logits(qh,kind),valid):
                    value = query.advance(choice,window_end_tick=end_tick)
                    if value.closed_by_continue:
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

    def choices(self, decoder, routed, oracle, timing, distributions, *, end_tick, cap_tick,
                moment_remaining=None):
        if hasattr(oracle,'set_duration_cap'):
            oracle.set_duration_cap(timing.second(cap_tick))
        types = oracle.query_types()
        allowed = [name in types and self.allowed(types[name]) for name in GENERATED_TYPES]
        if moment_remaining is not None and moment_remaining <= 0:
            phase = oracle.physical.grammar.state.phase
            required = {'hold': 'HOLD_DURATION', 'route': 'ROUTE',
                        'chain': 'SLIDE_DURATION'}.get(phase, 'MOMENT')
            allowed = [ok and name == required for name, ok in zip(GENERATED_TYPES, allowed)]
        for index in self.order(routed.logits[0, -1], allowed):
            target_type = torch.full_like(routed.valid, index, dtype=torch.long)
            result = decoder.complete(routed, target_type)
            h = result.hidden[0,-1]
            yield from self.contents(decoder,h,GENERATED_TYPES[index],oracle,timing,distributions,
                end_tick=end_tick,cap_tick=cap_tick,moment_remaining=moment_remaining,
                output=result,return_cache=True)
