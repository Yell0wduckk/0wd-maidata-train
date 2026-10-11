"""CPU-prepared supports and batched heads; same ledgers as Decoder's scalar oracle."""
from collections import defaultdict
from dataclasses import dataclass
import torch
from torch.nn.utils.rnn import pad_sequence

from maidata.chart.types import GENERATED_TYPES, KINDS, FLAGS, APPLICABLE, BUTTON_PADS, TOUCH_PADS, Note, Route
from maidata.chart.windows import TimeTarget
from maidata.time import duration_end
from maidata.decoder.supports import categorical_support
from maidata.decoder.probability import categorical_log_probabilities
from maidata.decoder.losses import Reduction, DecoderLoss, physical_categories
from maidata.time.discrete import compile_support_tree
from maidata.time.distribution import time_path_nll, complete_field_log_probabilities


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

    query = {}
    def add(key, index, allowed, penalty, target, *, physical=1., args=(), main=True, tree=None):
        if not (allowed & target).any() or (target & ~allowed).any():
            raise ValueError('GT outside basic support')
        counts['main'] += int(main)
        rows[key].append((index, allowed, penalty, target, float(physical), args, main,
                          dict(query), tree))

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
            query.clear()
            query.update(query_beats=s['cursor_tick']/96, origins=s['cursor_tick'], elapsed=0,
                         roles=GENERATED_TYPES.index(name))
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
                tree = compile_support_tree(s,timing,target=target,kind=name,
                                            window_end=w.end_tick,cap_tick=w.cap_tick)
                allowed = torch.tensor(tree.basic, dtype=torch.bool)
                penalty = torch.tensor(tree.penalty, dtype=torch.bool)
                target_mask = torch.tensor(tree.target, dtype=torch.bool)
                choice = bool((allowed.sum(-1)>1).any())
                counts['physical'] += choice
                add(name,index,allowed,penalty,target_mask,physical=float(choice),tree=tree)
    packed = {}
    for key, group in rows.items():
        indices, allowed, penalty, target, scale, args, main, queries, trees = zip(*group)
        packed[key] = dict(indices=torch.tensor(indices), allowed=pad_sequence(allowed,batch_first=True),
            penalty=pad_sequence(penalty,batch_first=True), target=pad_sequence(target,batch_first=True),
            scale=torch.tensor(scale), args=torch.tensor(args,dtype=torch.long), main=torch.tensor(main))
        packed[key]['has_physical'] = any(scale)
        spans = []
        for start,(owner,_) in enumerate(indices):
            if not spans or spans[-1][0] != owner:
                spans.append([owner,start,start+1])
            else:
                spans[-1][2] = start+1
        packed[key]['owner_spans'] = spans
        packed[key].update({name:torch.tensor([q[name] for q in queries],
            dtype=torch.long if name in ('roles','origins','elapsed') else torch.float32)
            for name in ('query_beats','origins','elapsed','roles')})
        if isinstance(key, tuple) and key[0] == 'flag':
            lookup = {index:i for i,index in enumerate(flag_rows)}
            packed[key]['flag_rows'] = torch.tensor([lookup[i] for i in indices])
        if trees[0] is not None:
            packed[key].update(query_beats=pad_sequence([torch.tensor(v.query_beats) for v in trees],batch_first=True),
                elapsed=pad_sequence([torch.tensor(v.elapsed_ticks) for v in trees],batch_first=True),
                closes=pad_sequence([torch.tensor(v.continuation_closes) for v in trees],batch_first=True),
                target_steps=pad_sequence([torch.tensor(v.target_steps) for v in trees],batch_first=True))
    content = [index for key in ('kind','route') if key in packed for index in packed[key]['indices'].tolist()]
    lookup = {tuple(index):i for i,index in enumerate(content)}
    for key,g in packed.items():
        if key != 'type' and 'target_steps' not in g:
            g['query_rows'] = torch.tensor([lookup[tuple(i)] for i in g['indices'].tolist()],dtype=torch.long)
    return Prepared(packed, counts, dict(indices=torch.tensor(list(flag_rows),dtype=torch.long).reshape(-1,2),
        args=torch.tensor(list(flag_rows.values()),dtype=torch.long).reshape(-1,2),
        query_rows=torch.tensor([lookup[index] for index in flag_rows],dtype=torch.long)))


def loss(heads, output, prepared, *, distributions, geometry, diagnostics=None, time_diagnostics=None):
    zero = output.hidden.float().sum()*0
    totals = dict(main=zero, physical=zero.expand(4), judge=zero, trajectory=zero)
    # The shared conditional adapter runs once per NOTE, not once per flag.
    f = {k:v.to(output.hidden.device, non_blocking=True) for k,v in prepared.flags.items()}
    flag_logits = None
    content_parts = []
    for key in ('kind','route'):
        if key in prepared.groups:
            cpu=prepared.groups[key]
            g = {k:v.to(output.hidden.device) if isinstance(v,torch.Tensor) else v for k,v in cpu.items()}
            content_parts.append(output.field_reader(output,g['indices'],g['query_beats'],g['roles'],g['origins'],
                g['elapsed'],owner_spans=g['owner_spans']))
    content_queries = torch.cat(content_parts) if content_parts else output.hidden.new_empty((0,output.hidden.shape[-1]))
    if len(f['indices']):
        # Reuse the NOTE query state across pad/flags instead of querying each flag.
        h = content_queries[f['query_rows']]
        flag_logits = heads.flag_logits(h,f['args'][:,0],f['args'][:,1])
    for key, cpu in prepared.groups.items():
        g = {k:v.to(output.hidden.device, non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in cpu.items()}
        index = g['indices']
        h, args = output.hidden[index[:,0],index[:,1]], g['args']
        time_group = 'target_steps' in g
        if key != 'type' and not time_group:
            h = content_queries[g['query_rows']]
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
            fields, depth = g['allowed'].shape[:2]
            query_index = index[:,None].expand(-1,depth,-1).reshape(-1,2)
            qh = output.field_reader(output,query_index,g['query_beats'].flatten(),
                g['roles'][:,None].expand(-1,depth).flatten(),
                g['origins'][:,None].expand(-1,depth).flatten(),g['elapsed'].flatten(),
                owner_spans=[(owner,start*depth,stop*depth) for owner,start,stop in g['owner_spans']])
            logits = heads.time_logits(qh,key).reshape(fields,depth,-1)
        if time_group:
            nll = time_path_nll(logits,g['allowed'],g['target'],g['target_steps']).float()
            lp = complete_field_log_probabilities(logits,g['allowed'],g['closes']).flatten(1)
            penalty = g['penalty'].flatten(1,2)
        else:
            lp = categorical_log_probabilities(logits,g['allowed'],validate=False)
            nll = -torch.logsumexp(lp.masked_fill(~g['target'],-torch.inf),-1).float()
            penalty = g['penalty']
        totals['main'] = totals['main'] + (nll*g['main']).sum()
        if g['has_physical']:
            totals['physical'] = totals['physical'] + (physical_categories(lp,penalty,validate=False).float()*g['scale'][:,None]).sum(0)
        if key == 'route':
            j,t = geometry(lp,g['target'].long().argmax(-1))
            totals['judge'], totals['trajectory'] = totals['judge']+j.sum(), totals['trajectory']+t.sum()
        if time_diagnostics is not None and time_group:
            # A merged integration region is not an emitted time candidate;
            # argmax of region mass would be a different, invalid accuracy metric.
            row = time_diagnostics.setdefault(key, dict(probability_sum=0., nll_sum=0., count=0))
            row['probability_sum'] += float(((-nll.detach()).exp()*g['main']).sum())
            row['nll_sum'] += float((nll.detach()*g['main']).sum())
            row['count'] += int(g['main'].sum())
        if diagnostics is not None and not time_group:
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
