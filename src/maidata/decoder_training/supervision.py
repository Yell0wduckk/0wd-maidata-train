"""GT-prefix supervision: one main hidden state per actual emitted time step."""
from dataclasses import dataclass
import torch
from maidata.chart.types import FLAGS
from maidata.decoder.probability import categorical_log_probabilities
from maidata.decoder.losses import Reduction, DecoderLoss, physical_categories
KEYS = ('main','physical','judge','trajectory')

@dataclass
class Prepared:
    groups: dict
    counts: dict
    flags: dict


def prepare(windows):
    from .source_labels import prepare as source_prepare
    from .sequence_cache import expand_window
    from .window_cache import collate_prepared, select_window
    rows=[]
    for row in windows:
        w=row['example']
        source=source_prepare([row])
        for g in source.groups.values():
            g['indices'][:,1]-=w.boundary_index
        source.flags['indices'][:,1]-=w.boundary_index
        value=dict(history=w.boundary_index,metadata={},labels=dict(groups=source.groups,counts=source.counts,flags=source.flags))
        rows.append(select_window(expand_window(value,w,row['timing']),False))
    return collate_prepared(rows)[2]


def loss(heads, output, prepared, *, distributions, geometry, diagnostics=None, time_diagnostics=None):
    zero = output.hidden.float().sum()*0
    totals = dict(main=zero, physical=zero.expand(4), judge=zero, trajectory=zero)
    # The shared conditional adapter runs once per NOTE, not once per flag.
    f = {k:v.to(output.hidden.device, non_blocking=True) for k,v in prepared.flags.items()}
    flag_logits = None
    if len(f['indices']):
        h = output.hidden[f['indices'][:,0],f['indices'][:,1]]
        flag_logits = heads.flag_logits(h,f['args'][:,0],f['args'][:,1])
    for key, cpu in prepared.groups.items():
        g = {k:v.to(output.hidden.device, non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in cpu.items()}
        index = g['indices']
        h, args = output.hidden[index[:,0],index[:,1]], g['args']
        time_group = key in ('MOMENT','HOLD_DURATION','SLIDE_DURATION')
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
            logits = heads.time_logits(h,key)
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
            row = time_diagnostics.setdefault(key,dict(probability_sum=0.,nll_sum=0.,count=0))
            field_count=int(g['field_id'].max())+1
            field_nll=nll.detach().new_zeros(field_count).scatter_add_(0,g['field_id'],nll.detach()*g['main'])
            row['probability_sum']+=float((-field_nll).exp().sum())
            row['nll_sum']+=float((nll.detach()*g['main']).sum())
            row['count']+=field_count
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
