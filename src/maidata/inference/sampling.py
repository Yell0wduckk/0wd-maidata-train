"""Hierarchical field sampling against current, prefix-only rule queries."""
from itertools import product

import torch

from maidata.chart.types import (GENERATED_TYPES, KINDS, FLAGS, APPLICABLE,
    BUTTON_PADS, TOUCH_PADS, Moment, Note, Route, HoldDuration, SlideDuration)
from maidata.chart.windows import TimeTarget
from maidata.decoder.probability import interval_log_mass
from .time_support import TimeCandidateCache


class Sampler:
    def __init__(self, *, temperature=1., seed=20261010, physical_hard_mask=True):
        if not 0 <= temperature < float('inf'):
            raise ValueError('temperature must be finite and nonnegative')
        self.temperature = temperature
        self.generator = torch.Generator(device='cpu').manual_seed(seed)
        self.physical_hard_mask = physical_hard_mask
        self.time_candidates = TimeCandidateCache()

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
                 moment_remaining=None):
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
                        yield options[f]
        elif kind == 'ROUTE':
            first = oracle.physical.grammar.state.phase in ('route', 'track_done')
            candidates = [Route(i, 0 if first else None) for i in range(71)]
            for r in self.order(heads.route(h), [self.allowed(oracle.query(n)) for n in candidates]):
                if first:
                    for b in self.order(heads.break_logits(h, torch.tensor(r, device=device))):
                        yield Route(r, b)
                else:
                    yield candidates[r]
        else:
            cursor = oracle.physical.grammar.state.cursor
            support = oracle.time_support(kind)
            if kind != 'MOMENT' and support['domain'][1] < support['domain'][0]:
                return
            stage = dict(cursor_tick=cursor, **{'moment' if kind == 'MOMENT' else 'duration': support})
            candidates = self.time_candidates.candidates(stage, timing, kind=kind,
                window_end=end_tick, cap_tick=cap_tick, device=device)
            valid = candidates.valid & (~candidates.raw.any(-1) if self.physical_hard_mask else True)
            mass = interval_log_mass(heads.time_parameters(h, kind), candidates.lower, candidates.upper,
                                     distributions[kind], validate=False)
            for i in self.order(mass, valid.cpu()):
                tick = candidates.ticks[i]
                if kind == 'MOMENT':
                    delta = timing.delta_ms(cursor, tick)
                    yield (TimeTarget(delta, tick, candidates.labels[i] is None, True) if tick >= end_tick
                           else Moment(delta))
                else:
                    yield (HoldDuration if kind == 'HOLD_DURATION' else SlideDuration)(tick)

    def choices(self, decoder, routed, oracle, timing, distributions, *, end_tick, cap_tick,
                moment_remaining=None):
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
            for token in self.contents(decoder, result.hidden[0, -1], GENERATED_TYPES[index], oracle,
                                       timing, distributions, end_tick=end_tick, cap_tick=cap_tick,
                                       moment_remaining=moment_remaining):
                yield token, result.cache
