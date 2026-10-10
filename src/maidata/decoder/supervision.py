"""Small-head supervision adapter over explicit teacher prefix supports.

Networks emit states/logits. This module reads structured labels and rule records;
neither producer owns the other. It does not execute training, IO or Physical.
"""
import torch
from maidata.chart.types import (GENERATED_TYPES, KINDS, FLAGS, APPLICABLE, BUTTON_PADS,
                                TOUCH_PADS, Note, Route, Moment)
from maidata.chart.windows import TimeTarget
from maidata.time import duration_end
from .supports import categorical_support, time_candidates
from .probability import categorical_log_probabilities, interval_log_probabilities
from .losses import Reduction, DecoderLoss, nll, physical_categories


def prediction_loss(heads, h, type_logits, target, stage, *, timing, window_end, cap_tick,
                    audio_end_seconds, distributions, geometry):
    """One supervised row; the result can be added across windows/chunks before mean."""
    device = h.device
    zero = h.sum() * 0
    main, physical = Reduction(zero, 0), Reduction(zero.expand(4), 0)
    judge, trajectory = Reduction(zero, 0), Reduction(zero, 0)

    def discrete(logits, allowed, target_index):
        nonlocal main
        lp = categorical_log_probabilities(logits, allowed)
        mask = torch.arange(len(lp), device=device) == target_index
        if not allowed[target_index]:
            raise ValueError('categorical GT outside basic legal support')
        main = main + nll(lp, mask, counted=int(allowed.sum()) > 1)
        return lp

    name = 'MOMENT' if isinstance(target, TimeTarget) else target.type
    allowed, penalty, _ = categorical_support(stage['types'], GENERATED_TYPES, device=device)
    lp = discrete(type_logits, allowed, GENERATED_TYPES.index(name))
    if int(allowed.sum()) > 1:
        physical = physical + Reduction(physical_categories(lp, penalty), 1)
    if isinstance(target, Note):
        k = KINDS.index(target.kind)
        if stage['pad_kind'] != target.kind:
            raise ValueError('pad support conditioned on another kind')
        pads = TOUCH_PADS if target.kind in ('touch', 'touch_hold') else BUTTON_PADS
        p = pads.index(target.pad)
        ka, kp, _ = categorical_support(stage['kind'], KINDS, device=device)
        klp = discrete(heads.kind(h), ka, k)
        pa, pp, _ = categorical_support(stage['pad'], pads, device=device)
        kind = torch.tensor([k], dtype=torch.long, device=device)
        pad = torch.tensor([p], dtype=torch.long, device=device)
        plp = discrete(heads.pad_logits(h[None], kind)[0, :len(pads)], pa, p)
        kc, pc = int(ka.sum()) > 1, int(pa.sum()) > 1
        if kc or pc:
            terms = .5 * (physical_categories(klp, kp) if kc else zero.expand(4))
            terms = terms + .5 * (physical_categories(plp, pp) if pc else zero.expand(4))
            physical = physical + Reduction(terms, 1)
        logits = heads.flag_logits(h[None], kind, pad)[0]
        for i, flag in enumerate(FLAGS):
            if flag not in APPLICABLE[target.kind]:
                continue
            allowed = torch.ones(2, dtype=torch.bool, device=device)
            if flag in ('short_hold', 'short_touch_hold'):
                allowed[1] = duration_end(timing, stage['cursor_tick'], 1) <= audio_end_seconds + 1e-9
            discrete(logits[i], allowed, getattr(target, flag))
    elif isinstance(target, Route):
        allowed, penalty, _ = categorical_support(stage['route'], range(71), device=device)
        lp = discrete(heads.route(h), allowed, target.route_id)
        if int(allowed.sum()) > 1:
            physical = physical + Reduction(physical_categories(lp, penalty), 1)
        if geometry is None:
            raise ValueError('ROUTE supervision requires explicit fixed geometry distances')
        j, t = geometry(lp, target.route_id)
        judge, trajectory = Reduction(j, 1), Reduction(t, 1)
        first = stage['phase'] in ('route', 'track_done')
        if first != (target.track_break is not None):
            raise ValueError('track_break applicability disagrees with prefix')
        if first:
            logits = heads.break_logits(h[None], torch.tensor([target.route_id], device=device))[0]
            discrete(logits, torch.ones(2, dtype=torch.bool, device=device), target.track_break)
    else:
        candidates = time_candidates(stage, timing, kind=name, window_end=window_end, cap_tick=cap_tick, device=device)
        raw = heads.time_parameters(h, name)
        lp = interval_log_probabilities(raw, candidates.lower, candidates.upper, candidates.valid, distributions[name])
        main = main + nll(lp, candidates.target_mask(target, timing, stage['cursor_tick']))
        if candidates.choices > 1:
            physical = physical + Reduction(physical_categories(lp, candidates.penalty), 1)
    return DecoderLoss(main, physical, judge, trajectory)


def decoder_loss(heads, output, examples, stages, timings, audio_ends, *, distributions, geometry):
    """Logical-batch sum/count; history, padding and injected boundaries are excluded."""
    zero = output.hidden.sum() * 0
    result = DecoderLoss(Reduction(zero, 0), Reduction(zero.expand(4), 0),
                         Reduction(zero, 0), Reduction(zero, 0))
    if not len(examples) == len(stages) == len(timings) == len(audio_ends) == len(output.hidden):
        raise ValueError('supervision batch mismatch')
    for batch, example in enumerate(examples):
        if len(stages[batch]) != len(example.inputs):
            raise ValueError('missing known-prefix support rows')
        for row, (target, supervised) in enumerate(zip(example.targets, example.supervised)):
            if supervised:
                result = result + prediction_loss(heads, output.hidden[batch, row], output.type_logits[batch, row],
                    target, stages[batch][row], timing=timings[batch], window_end=example.end_tick,
                    cap_tick=example.cap_tick, audio_end_seconds=audio_ends[batch],
                    distributions=distributions, geometry=geometry)
    return result
