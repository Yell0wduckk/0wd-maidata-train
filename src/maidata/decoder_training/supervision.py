"""CPU-prepared supports and batched heads; same ledgers as Decoder's scalar oracle."""
from collections import defaultdict
from dataclasses import dataclass
import torch
from torch.nn.utils.rnn import pad_sequence

from maidata.chart.types import GENERATED_TYPES, KINDS, FLAGS, APPLICABLE, BUTTON_PADS, TOUCH_PADS, Note, Route
from maidata.chart.windows import TimeTarget
from maidata.time import duration_end
from maidata.decoder.supports import categorical_support, time_candidates
from maidata.decoder.probability import categorical_log_probabilities, interval_log_probabilities
from maidata.decoder.losses import Reduction, DecoderLoss, physical_categories


KEYS = ('main', 'physical', 'judge', 'trajectory')


@dataclass
class Prepared:
    groups: dict
    counts: dict
    flags: dict


def prepare(windows):
    rows = defaultdict(list)
    counts = dict.fromkeys(KEYS, 0)
    flag_rows = {}

    def add(key, index, allowed, penalty, target, *, physical=1., args=(), bounds=None, main=True):
        if not (allowed & target).any() or (target & ~allowed).any():
            raise ValueError('GT outside basic support')
        counts['main'] += int(main)
        rows[key].append((index, allowed, penalty, target, float(physical), args, bounds, main))

    def discrete(key, index, records, names, target, *, physical=1., args=()):
        allowed, penalty, _ = categorical_support(records, names, device='cpu')
        choice = int(allowed.sum()) > 1
        add(key, index, allowed, penalty, torch.arange(len(names)) == target,
            physical=physical*choice, args=args, main=choice)
        return choice

    for b, row in enumerate(windows):
        w, timing = row['example'], row['timing']
        for r, (target, active) in enumerate(zip(w.targets, w.supervised)):
            if not active:
                continue
            s, index = row['stages'][r], (b, r)
            name = 'MOMENT' if isinstance(target, TimeTarget) else target.type
            counts['physical'] += discrete('type', index, s['types'], GENERATED_TYPES, GENERATED_TYPES.index(name))
            if isinstance(target, Note):
                if s['pad_kind'] != target.kind:
                    raise ValueError('pad support conditioned on another kind')
                k = KINDS.index(target.kind)
                pads = TOUCH_PADS if target.kind in ('touch','touch_hold') else BUTTON_PADS
                p = pads.index(target.pad)
                flag_rows[index] = (k, p)
                kc = discrete('kind', index, s['kind'], KINDS, k, physical=.5)
                pc = discrete(('pad',len(pads)), index, s['pad'], pads, p, physical=.5, args=(k,))
                counts['physical'] += kc or pc
                for f, flag in enumerate(FLAGS):
                    if flag not in APPLICABLE[target.kind]:
                        continue
                    allowed = torch.ones(2, dtype=torch.bool)
                    if flag in ('short_hold','short_touch_hold'):
                        allowed[1] = duration_end(timing, s['cursor_tick'], 1) <= row['audio_end']+1e-9
                    add(('flag',f), index, allowed, torch.zeros(2,4,dtype=torch.bool),
                        torch.arange(2) == getattr(target,flag), physical=0, args=(k,p), main=int(allowed.sum())>1)
            elif isinstance(target, Route):
                counts['physical'] += discrete('route', index, s['route'], range(71), target.route_id)
                counts['judge'] += 1
                counts['trajectory'] += 1
                first = s['phase'] in ('route','track_done')
                if first != (target.track_break is not None):
                    raise ValueError('track_break applicability differs')
                if first:
                    add('break', index, torch.ones(2,dtype=torch.bool), torch.zeros(2,4,dtype=torch.bool),
                        torch.arange(2)==target.track_break, physical=0, args=(target.route_id,))
            else:
                cells = time_candidates(s,timing,kind=name,window_end=w.end_tick,cap_tick=w.cap_tick,device='cpu')
                choice = cells.choices > 1
                counts['physical'] += choice
                add(name,index,cells.valid,cells.penalty,cells.target_mask(target,timing,s['cursor_tick']),
                    physical=float(choice),bounds=(cells.lower,cells.upper))
    packed = {}
    for key, group in rows.items():
        indices, allowed, penalty, target, scale, args, bounds, main = zip(*group)
        packed[key] = dict(indices=torch.tensor(indices), allowed=pad_sequence(allowed,batch_first=True),
            penalty=pad_sequence(penalty,batch_first=True), target=pad_sequence(target,batch_first=True),
            scale=torch.tensor(scale), args=torch.tensor(args,dtype=torch.long), main=torch.tensor(main))
        packed[key]['has_physical'] = any(scale)
        if isinstance(key, tuple) and key[0] == 'flag':
            lookup = {index:i for i,index in enumerate(flag_rows)}
            packed[key]['flag_rows'] = torch.tensor([lookup[i] for i in indices])
        if bounds[0] is not None:
            packed[key].update(lower=pad_sequence([v[0] for v in bounds],batch_first=True),
                               upper=pad_sequence([v[1] for v in bounds],batch_first=True,padding_value=1))
    return Prepared(packed, counts, dict(indices=torch.tensor(list(flag_rows),dtype=torch.long).reshape(-1,2),
        args=torch.tensor(list(flag_rows.values()),dtype=torch.long).reshape(-1,2)))


def loss(heads, output, prepared, *, distributions, geometry, diagnostics=None):
    zero = output.hidden.float().sum()*0
    totals = dict(main=zero, physical=zero.expand(4), judge=zero, trajectory=zero)
    # The shared conditional adapter runs once per NOTE, not once per flag.
    f = {k:v.to(output.hidden.device, non_blocking=True) for k,v in prepared.flags.items()}
    flag_logits = None
    if len(f['indices']):
        flag_logits = heads.flag_logits(output.hidden[f['indices'][:,0],f['indices'][:,1]],f['args'][:,0],f['args'][:,1])
    for key, cpu in prepared.groups.items():
        g = {k:v.to(output.hidden.device, non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in cpu.items()}
        index = g['indices']
        h, args = output.hidden[index[:,0],index[:,1]], g['args']
        if key == 'type':
            logits = output.type_logits[index[:,0],index[:,1]]
        elif key in ('kind','route'):
            logits = getattr(heads,key)(h)
        elif key == 'break':
            logits = heads.break_logits(h,args[:,0])
        elif isinstance(key,tuple) and key[0] == 'pad':
            logits = heads.pad_logits(h,args[:,0])[:,:key[1]]
        elif isinstance(key,tuple):
            logits = flag_logits[g['flag_rows'],key[1]]
        else:
            logits = None
        lp = (categorical_log_probabilities(logits,g['allowed'],validate=False) if logits is not None else
            interval_log_probabilities(heads.time_parameters(h,key),g['lower'],g['upper'],g['allowed'],distributions[key],validate=False))
        nll = -torch.logsumexp(lp.masked_fill(~g['target'],-torch.inf),-1)
        totals['main'] = totals['main'] + (nll*g['main']).sum()
        if g['has_physical']:
            totals['physical'] = totals['physical'] + (physical_categories(lp,g['penalty'],validate=False)*g['scale'][:,None]).sum(0)
        if key == 'route':
            j,t = geometry(lp,g['target'].long().argmax(-1))
            totals['judge'], totals['trajectory'] = totals['judge']+j.sum(), totals['trajectory']+t.sum()
        if diagnostics is not None:
            name = (f'pad_{key[1]}' if key[0] == 'pad' else FLAGS[key[1]]) if isinstance(key,tuple) else key
            prediction = lp.detach().argmax(-1)
            correct = g['target'].gather(1,prediction[:,None])[:,0] & g['main']
            row = diagnostics.setdefault(name,dict(correct=0,count=0))
            row['correct'] += int(correct.sum())
            row['count'] += int(g['main'].sum())
    return DecoderLoss(*(Reduction(totals[k],prepared.counts[k]) for k in KEYS))


def objective(terms, counts, config):
    """All chunks and accumulated groups use the same logical-batch denominators."""
    normalized = DecoderLoss(*(Reduction(getattr(terms,k).total,counts[k]) for k in KEYS))
    return normalized.objective(**{k:config['decoder_train_'+k] for k in
        ('physical_weight','judge_weight','trajectory_weight')})
