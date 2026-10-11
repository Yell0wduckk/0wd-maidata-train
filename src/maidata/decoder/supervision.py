"""Small-head supervision adapter over explicit teacher prefix supports.

Networks emit states/logits. This module reads structured labels and rule records;
neither producer owns the other. It does not execute training, IO or Physical.
"""
import torch
from maidata.chart.types import (GENERATED_TYPES, KINDS, FLAGS, APPLICABLE, BUTTON_PADS,
                                TOUCH_PADS, Note, Route, Moment)
from maidata.chart.windows import TimeTarget
from maidata.time import duration_end
from .supports import categorical_support
from .probability import categorical_log_probabilities
from .losses import Reduction, DecoderLoss, nll, physical_categories
from maidata.time.discrete import compile_support_tree
from maidata.time.distribution import time_path_nll, complete_field_log_probabilities


def prediction_loss(heads, h, type_logits, target, stage, *, timing, window_end, cap_tick,
                    audio_end_seconds, distributions, geometry, time_states=None):
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
        tree = compile_support_tree(stage,timing,kind=name,window_end=window_end,cap_tick=cap_tick,target=target)
        if time_states is None or len(time_states) != len(tree.basic):
            raise ValueError('time supervision requires a separate state at every reachable CONT query')
        allowed = torch.tensor(tree.basic,device=device)
        raw = heads.time_logits(time_states,name)
        value = time_path_nll(raw,allowed,torch.tensor(tree.target,device=device),
                              torch.tensor(tree.target_steps,device=device))
        main = main + Reduction(value,1)
        if (allowed.sum(-1)>1).any():
            lp = complete_field_log_probabilities(raw,allowed,torch.tensor(tree.continuation_closes,device=device))
            penalty = torch.tensor(tree.penalty,device=device)
            physical = physical + Reduction(physical_categories(lp.flatten(),penalty.flatten(0,1)),1)
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
                stage = stages[batch][row]
                name = 'MOMENT' if isinstance(target,TimeTarget) else target.type
                device = output.hidden.device
                index = torch.tensor([[batch,row]],device=device)
                origin = stage['cursor_tick']
                def query(beats,elapsed):
                    size = len(beats)
                    return output.field_reader(output,index.expand(size,-1),torch.tensor(beats,device=device),
                        torch.full((size,),GENERATED_TYPES.index(name),device=device,dtype=torch.long),
                        torch.full((size,),origin,device=device,dtype=torch.long),torch.tensor(elapsed,device=device))
                if name in ('NOTE','ROUTE'):
                    h,time_states = query([origin/96],[0])[0],None
                else:
                    tree = compile_support_tree(stage,timings[batch],kind=name,window_end=example.end_tick,
                                                cap_tick=example.cap_tick,target=target)
                    h,time_states = output.hidden[batch,row],query(tree.query_beats,tree.elapsed_ticks)
                result = result + prediction_loss(heads, h, output.type_logits[batch, row],
                    target, stages[batch][row], timing=timings[batch], window_end=example.end_tick,
                    cap_tick=example.cap_tick, audio_end_seconds=audio_ends[batch],
                    distributions=distributions, geometry=geometry,time_states=time_states)
    return result
