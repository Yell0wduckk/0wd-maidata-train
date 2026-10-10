"""Batched moment state machine and immutable CPU-prepared supervision."""
from __future__ import annotations

import numpy as np
import torch

from maimai_baseline.contract import REMOVED_FIELDS, L3_LABEL_SMOOTHING
from maimai_baseline.loss.common import GROUP_BUDGET, GROUP_NOTE_BUDGET, L3LossResult
from maimai_baseline.loss.l3 import NOTE_ATTRIBUTE_FIELDS, note_interval_weights, structure_interval_weights
from maimai_baseline.model.rollout import IncrementalMomentDecoder
from maimai_baseline.time.coordinates import seconds_at_ticks
from maimai_baseline.time.moment import candidate_probabilities


EXECUTION = dict(backend='batched-raw-prefix-rope-v2', prefix_dropout='once-per-row-with-gradient')


def prepare(batch, vocabulary, blocks):
    """Run on CPU before moving a training batch; never rewrites offline caches."""
    if not blocks:
        return None
    targets = {k:v.detach().cpu() for k,v in batch['target_ids'].items()}
    maximum = max(len(p.notes) for p in blocks)
    rows = torch.tensor([list(p.notes)+[p.end]*(maximum+1-len(p.notes)) for p in blocks])
    windows = torch.tensor([p.batch_index for p in blocks])
    starts = torch.tensor([p.start for p in blocks])
    ends = torch.tensor([p.end for p in blocks])
    counts = torch.tensor([len(p.notes) for p in blocks])
    def encoded(row):
        value = vocabulary.encode_row(row)[0]
        from maimai_dataset.modifiers import row_bits
        value.update({'modifier_' + k: v for k, v in row_bits(row).items()})
        return {**{k:v for k,v in value.items() if k not in REMOVED_FIELDS}, 'route':0}
    plain = encoded(dict(event='NOTE',kind='tap',flags='',pad='A1'))
    head = encoded(dict(event='MOMENT',delta_tick=0.))
    # Slot order and flags are exactly those of this augmented GT batch.
    children = []
    for p in blocks:
        slots = []
        for r in p.notes:
            raw = {k:int(v[p.batch_index,r]) for k,v in targets.items() if k in vocabulary.fields and int(v[p.batch_index,r])>=0}
            value = encoded(vocabulary.decode_row(raw))
            for name in ('ex', 'break', 'short_hold'):
                value['modifier_' + name] = max(0, int(targets['modifier_' + name][p.batch_index, r]))
            slots.append(value)
        slots += [plain]*(maximum+1-len(slots))
        children.append(slots)
    inputs = {k:torch.tensor([[r[k] for r in lane] for lane in children]) for k in plain}
    grid = {}
    start_ticks, end_ticks, song_ends = [batch[k].detach().cpu().tolist() for k in ('start_tick','end_tick','song_end_tick')]
    width = max(int(e-s) for s,e in zip(start_ticks,end_ticks))
    for key in ('ticks','seconds','lower_seconds','upper_seconds','valid'):
        grid[key] = []
    for timing,s,e,last in zip(batch['sft_timings'],start_ticks,end_ticks,song_ends):
        ticks = np.arange(int(s),int(s)+width,dtype=np.int64)
        grid['ticks'].append(ticks)
        grid['seconds'].append(seconds_at_ticks(timing,ticks))
        grid['lower_seconds'].append(seconds_at_ticks(timing,np.maximum(ticks-.5,s)))
        grid['upper_seconds'].append(seconds_at_ticks(timing,np.minimum(ticks+.5,e)))
        grid['valid'].append((ticks<e)&(ticks<=last))
    grid = {k:torch.from_numpy(np.stack(v)) for k,v in grid.items()}
    event_mask = torch.zeros_like(targets['event'],dtype=torch.bool)
    note_mask = event_mask.clone()
    for p in blocks:
        event_mask[p.batch_index,p.start+1:p.end+1] = True
        note_mask[p.batch_index,list(p.notes)] = True
    note_slots=torch.arange(maximum+1)[None,:]<counts[:,None]
    attributes=[k for k in NOTE_ATTRIBUTE_FIELDS if k in targets
                and bool((targets[k][windows[:,None],rows].ne(-100)&note_slots).any())]
    return dict(attribute_fields=attributes,windows=windows,starts=starts,ends=ends,counts=counts,rows=rows,
        inputs=inputs,head=head,plain=plain,grid=grid,event_mask=event_mask,note_mask=note_mask,
        targets={k:v[windows[:,None],rows] for k,v in targets.items()},
        source_segments=[batch['segment_ids'][p.batch_index] for p in blocks])


def position_loss(logits, targets, allowed=None, *, smoothing=L3_LABEL_SMOOTHING):
    active = targets.ne(-100)
    safe = targets.clamp_min(0)
    if allowed is None:
        allowed = logits.gt(-5000.)
    else:
        allowed = allowed.expand_as(logits).clone()
    allowed = allowed.scatter(-1,safe[:,None],True)
    logits = logits.masked_fill(~allowed,-10_000.)
    logp = logits.float().log_softmax(-1)
    nll = -logp.gather(-1,safe[:,None])[:,0]
    smooth = -logp.masked_fill(~allowed,0.).sum(-1)/allowed.sum(-1).clamp_min(1)
    return ((1-smoothing)*nll+smoothing*smooth), active & allowed.sum(-1).gt(1)


def rollout_batch(model, batch, outputs, prepared, *, max_notes=33, collect_trace=False, forced=None):
    from maimai_baseline.loss.budget import reject_legacy_sft
    reject_legacy_sft()
    if max_notes<=0:
        raise ValueError('max rollout notes must be positive')
    p, vocab = prepared, model.vocabulary
    b, start, end, count = [p[k] for k in ('windows','starts','ends','counts')]
    device, n = b.device, len(b)
    moment,note = vocab.event_id('MOMENT'),vocab.event_id('NOTE')
    packed = batch['bpm96_row_index'][b,start]
    table = {k:v[packed] for k,v in batch['bpm96_candidates'].items()}
    time_key = 'delta_ms' if 'delta_ms' in outputs['time_outputs'] else 'delta_tick'
    candidate = dict(outputs['delta_candidate'],query=outputs['delta_candidate']['query'][b,start])
    logp,_ = candidate_probabilities(outputs['time_outputs'][time_key][b,start],table,candidate,b,
                                    distribution=outputs['moment_distribution'])
    prediction = logp.detach().argmax(-1)
    current = table['ticks'].gather(1,prediction[:,None])[:,0]
    if forced is not None:
        current = torch.as_tensor(forced['start_ticks'],device=device,dtype=current.dtype)
        prediction = table['ticks'].eq(current[:,None]).long().argmax(-1)
    current_seconds = table['seconds'].gather(1,prediction[:,None])[:,0]
    def gt_tick(rows):
        return batch['bpm96_candidates']['ticks'][batch['bpm96_row_index'][b,rows],batch['bpm96_target_index'][b,rows]]
    first_gt,end_gt = gt_tick(start),gt_tick(end)
    grid = p['grid']
    def seconds(ticks):
        return grid['seconds'][b,(ticks-grid['ticks'][b,0]).long()]
    delta = (current_seconds-batch['row_anchor_second'][b,start])*1000
    decoder = IncrementalMomentDecoder(model,batch,outputs,b,start,current.float())
    sw,pw = structure_interval_weights(batch,vocab),note_interval_weights(batch,vocab)
    lanes = torch.arange(n,device=device)
    inputs = {k:torch.full((n,1),v,device=device,dtype=torch.long) for k,v in p['head'].items()}
    times = {k:torch.zeros((n,1),device=device) for k in batch['input_times']}
    times['delta_tick'] = delta.float()[:,None]
    active_times = {k:torch.zeros((n,1),device=device,dtype=torch.bool) for k in times}
    active_times['delta_tick'] = torch.ones((n,1),device=device,dtype=torch.bool)
    zero = outputs['decoded'].sum()*0
    note_total,event_total,end_total = zero,zero,zero
    generated = torch.zeros(n,device=device,dtype=torch.long)
    reasons = torch.zeros(n,device=device,dtype=torch.long)
    end_valid = torch.zeros(n,device=device,dtype=torch.bool)
    end_errors = torch.zeros(n,device=device,dtype=torch.float64)
    trajectory=[]
    fields = ('kind','pad')+tuple(k for k in p['attribute_fields'] if k in outputs['field_logits'])
    masks = {}
    for k in fields:
        if k.startswith('modifier_'):
            masks[k] = torch.ones(2, device=device, dtype=torch.bool)
            continue
        mask = torch.zeros(len(vocab.fields[k].values),device=device,dtype=torch.bool)
        ids = vocab.allowed_value_ids('NOTE',k,note_kind=None if k=='kind' else 'tap')
        mask[list(ids)] = True
        masks[k]=mask
    allowed_pads = torch.tensor(list(vocab.allowed_value_ids('NOTE','pad',note_kind='tap')),device=device)
    for step in range(max_notes+1):
        out = decoder.step(lanes,inputs,times,active_times)
        # This rollout is restricted to Tap blocks. Reuse the single decoded
        # row and condition position/flags on that known kind.
        from maimai_baseline.model.row_fields import sample_readout
        out['field_logits'] = dict(out['field_logits'])
        for stage in ('kind', 'position_flags'):
            out['field_logits'].update(sample_readout(model, out, dict(event='NOTE', kind='tap'), stage))
        slot = min(step,p['rows'].shape[1]-1)
        row = p['rows'][lanes,slot]
        has_note = step<count[lanes]
        wanted = torch.where(has_note,note,moment)
        event_total = event_total + (position_loss(out['field_logits']['event'][:,0],wanted)[0]*sw[b[lanes],row]).sum()
        matched = has_note.nonzero().flatten()
        if len(matched):
            attrs=[]
            values={}
            for k in fields:
                target = p['targets'][k][lanes[matched],slot]
                logits = out['field_logits'][k][matched,0]
                mask = masks[k].expand_as(logits).clone()
                if k=='kind':
                    for name in ('touch','touch_hold'):
                        forbidden = batch['cabinet_id'][b[lanes[matched]]].eq(1)
                        mask[:,vocab.fields[k].encode(name)] &= ~forbidden
                value,informative = position_loss(logits,target,mask, smoothing=0. if k.startswith('modifier_') else L3_LABEL_SMOOTHING)
                values[k]=value
                if k in NOTE_ATTRIBUTE_FIELDS:
                    attrs.append((value,informative))
            attribute = sum(v*m for v,m in attrs)/sum(m.long() for _,m in attrs).clamp_min(1)
            note_total = note_total + (GROUP_NOTE_BUDGET['kind']*values['kind']
                + GROUP_NOTE_BUDGET['position']*values['pad']*pw[b[lanes[matched]],row[matched]]
                + GROUP_NOTE_BUDGET['attributes']*attribute).sum()
        chosen = out['field_logits']['event'][:,0,[note,moment]].detach().argmax(-1)
        chosen = torch.full_like(chosen,note) if step==0 else torch.where(chosen.eq(0),note,moment)
        pad = allowed_pads[out['field_logits']['pad'][:,0,allowed_pads].detach().argmax(-1)]
        if forced is not None:
            chosen = torch.as_tensor(forced['events'],device=device)[lanes,step]
            pad = torch.as_tensor(forced['pads'],device=device)[lanes,step]
        stop = chosen.eq(moment) | (step>=max_notes)
        if collect_trace:
            trajectory.append((lanes.detach(),chosen.detach(),pad.detach()))
        ended = stop.nonzero().flatten()
        if len(ended):
            ids = lanes[ended]
            generated[ids] = step
            reasons[ids] = torch.where(chosen[ended].eq(moment),torch.where(step<count[ids],1,0),2)
            valid = end_gt[ids]>current[ids]
            ids,ended = ids[valid],ended[valid]
            if len(ids):
                source = b[ids]
                terminal = {k:grid[k][source] for k in ('ticks','seconds','valid')}
                anchor = current_seconds[ids,None]
                terminal['valid'] = terminal['valid'] & terminal['ticks'].gt(current[ids,None])
                terminal.update(delta_ms=(terminal['seconds']-anchor)*1000,
                    lower_ms=(grid['lower_seconds'][source]-anchor)*1000,
                    upper_ms=(grid['upper_seconds'][source]-anchor)*1000,
                    first=torch.zeros(len(ids),device=device,dtype=torch.bool))
                candidate = dict(outputs['delta_candidate'],query=out['candidate_query'][ended])
                endp,_=candidate_probabilities(out['time_outputs']['delta_ms' if 'delta_ms' in out['time_outputs'] else 'delta_tick'][ended,0],terminal,candidate,source,
                                               distribution=outputs['moment_distribution'])
                target = (end_gt[ids]-terminal['ticks'][:,0]).long()
                end_total = end_total-endp.gather(1,target[:,None]).sum()
                end_valid[ids]=True
                pred_seconds=terminal['seconds'].gather(1,endp.detach().argmax(-1)[:,None])[:,0]
                end_errors[ids]=(pred_seconds-seconds(end_gt)[ids]).abs()*1000
        keep = (~stop).nonzero().flatten()
        if not len(keep):
            break
        decoder.retain(keep)
        # Only this new NOTE is embedded on the next wave. GT attributes are
        # prepared once; surplus slots already contain plain Tap defaults.
        inputs={k:v[lanes[keep],slot,None] for k,v in p['inputs'].items()}
        inputs['pad']=pad[keep,None]
        times={k:torch.zeros((len(keep),1),device=device) for k in times}
        active_times={k:torch.zeros_like(v,dtype=torch.bool) for k,v in times.items()}
        lanes=lanes[keep]
    report_values=torch.stack((generated,reasons,end_valid,end_errors,
        (current_seconds-seconds(first_gt)).abs()*1000,current,first_gt,end_gt,count),1).detach().cpu().tolist()
    traces=[[] for _ in range(n)]
    if collect_trace:
        for step,(ids,events,pads) in enumerate(trajectory):
            for i,event,pad in torch.stack((ids,events,pads),1).cpu().tolist():
                traces[i].append(dict(gt_note_cursor=min(step,int(report_values[i][8])),generated_notes=step,
                    anchor_tick=int(report_values[i][5]),target_event='NOTE' if step<report_values[i][8] else 'MOMENT',
                    chosen_event='NOTE' if event==note else 'MOMENT',pad_id=pad,
                    terminal=event==moment or step>=max_notes))
    reports=[]
    for i,(g,r,valid,error,start_error,anchor,first,last,gt_count) in enumerate(report_values):
        g,gt_count=int(g),int(gt_count)
        reports.append(dict(segment_id=p['source_segments'][i],reason=('natural','early','limit')[int(r)],
            generated_notes=g,extra_notes=max(0,g-gt_count),unvisited_notes=max(0,gt_count-g-1),
            missing_notes=max(0,gt_count-g),end_time_valid=bool(valid),start_error_ms=start_error,
            end_error_ms=error if valid else None,start_tick=int(first),end_tick=int(last),
            generated_start_tick=int(anchor),trace=traces[i]))
    return dict(notes=note_total,events=event_total,terminal=end_total,reports=reports,steps=decoder.steps)


def replace(result,positions,p,generated):
    from maimai_baseline.loss.budget import reject_legacy_sft
    reject_legacy_sft()
    weights=positions['delta_mask'].float()
    b,start,end=p['windows'],p['starts'],p['ends']
    weights[b,start]=.5
    weights[b,end]=.5
    time_sum=(positions['delta']*weights).sum()
    note_delta=positions['note'].sum()*0
    event_delta=positions['event'].sum()*0
    if generated is not None:
        note_delta=(generated['notes']-positions['note'][p['note_mask']].sum())/positions['note_mask'].sum().clamp_min(1)
        event_delta=(generated['events']-positions['event'][p['event_mask']].sum())/positions['event_mask'].sum().clamp_min(1)
        time_sum=time_sum-.5*positions['delta'][b,end].sum()+.5*generated['terminal']
    timing=time_sum/weights.sum().clamp_min(1)
    components={**result.components,'timing':timing,
        'note_semantic':result.components['note_semantic']+note_delta,
        'structure':result.components['structure']+event_delta}
    loss=result.loss+GROUP_BUDGET['timing']*(timing-result.components['timing'])+GROUP_BUDGET['note_semantic']*note_delta+GROUP_BUDGET['structure']*event_delta
    return L3LossResult(loss=loss,metrics=result.metrics,components=components)
