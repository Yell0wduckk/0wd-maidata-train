"""CPU collation of shared Physical constants; probability work stays on device."""
from __future__ import annotations
import time
import numpy as np
import torch
from .contract import PHYSICAL_CATEGORIES as CATEGORIES
from .inference_state import KINDS
from .catalog import pad_bit
from .intervals import first_note_support
from maimai_baseline.time.duration import tick_bounds, cpu_tick_seconds, first_tail_tick


def prepare(batch,store,vocabulary):
    started=time.perf_counter()
    entries={};durations={};read_seconds=merge_seconds=0.
    target={k:v.numpy() for k,v in batch['target_ids'].items()};source=batch['physical_input_source_row_ids'].numpy()
    target_source=batch['physical_target_source_row_ids'].numpy()
    input_event=batch['input_ids']['event'].numpy()
    block_ids=batch['local_block_id'].numpy()
    delta_index=batch['bpm96_row_index'].numpy()
    chart_ids=batch['physical_chart_indices'].tolist();views=batch['physical_view_indices'].tolist()
    locate_seconds=delta_seconds=pack_seconds=0.
    counters={'contexts':0,'queries':0}
    names={f:[vocabulary.fields[f].decode(i) for i in range(len(vocabulary.fields[f].values))] for f in ('kind','pad','start_pad')}
    enc={f:{str(x):i for i,x in enumerate(values)} for f,values in names.items()}
    kind_names=names['kind'];pos_names=names
    candidate_ticks=batch['bpm96_candidates']['ticks'].numpy()
    candidate_valid=batch['bpm96_candidates']['valid'].numpy()
    note_id,branch_id=vocabulary.event_id('NOTE'),vocabulary.event_id('SLIDE_BRANCH')
    def add(name,b,r,allowed,bad,weight=1.):
        entries.setdefault(name,[]).append((b,r,np.asarray(allowed,dtype=bool),np.array(bad,dtype=bool,copy=True),weight))
    delta_bad=torch.zeros_like(batch['bpm96_candidates']['ticks'],dtype=torch.bool)
    position_mask=touch_position_mask=0
    for k in ('tap','touch'):
        for i in vocabulary.allowed_value_ids('NOTE','pad',note_kind=k):
            bit=pad_bit(str(vocabulary.fields['pad'].decode(i)))
            if k=='touch':touch_position_mask |= bit
            else:position_mask |= bit
    for b in sorted(range(len(source)),key=lambda b:(chart_ids[b],views[b])):
        t=time.perf_counter()
        chart=store.chart(chart_ids[b],views[b])
        read_seconds+=time.perf_counter()-t
        moments=chart.data['source_moments']
        t=time.perf_counter()
        ids=np.unique(delta_index[b][delta_index[b]>=0])
        from maimai_baseline.time.coordinates import seconds_at_ticks
        lengths=[int(candidate_valid[i].sum()) for i in ids]
        flat=np.concatenate([candidate_ticks[i][candidate_valid[i]] for i in ids]) if len(ids) else np.empty(0)
        coordinate_pair=(seconds_at_ticks(chart.timing,flat),seconds_at_ticks(chart.timing,flat+2))
        time_bounds={};offset=0
        for i,n in zip(ids,lengths):
            time_bounds[i]=(coordinate_pair[0][offset:offset+n],coordinate_pair[1][offset:offset+n]);offset+=n
        delta_seconds+=time.perf_counter()-t
        visible_by_moment={};last_moment=-1
        blocks,counts=np.unique(block_ids[b][(target['event'][b]==note_id)|(target['event'][b]==branch_id)],return_counts=True)
        weights=dict(zip(blocks,1/np.maximum(1,counts)))
        for r in range(source.shape[1]):
            src=int(source[b,r])
            if 0<=src<len(moments) and moments[src]>=0:last_moment=moments[src]
            action=chart.action_for_source(src)
            if action is not None:visible_by_moment.setdefault(action.moment,[]).append(action)
            event=int(target['event'][b,r])
            if event<0:continue
            target_src=int(target_source[b,r])
            conditioned=event in (note_id,branch_id)
            moment=moments[target_src] if conditioned and 0<=target_src<len(moments) else moments[src] if 0<=src<len(moments) else -1
            if moment<0:
                moment=last_moment
            # TIME responsibilities use only the actions visible before the
            # decision. No future target NOTE fields enter this feasibility.
            prefix=visible_by_moment.get(moment,())
            di=int(delta_index[b,r])
            if di>=0:
                delta_started=time.perf_counter()
                history=chart.prior(moment) if moment>=0 else ()
                if moment<0:
                    anchor=float(batch['row_anchor_tick'][b,r])
                    history=tuple(a for a in chart.actions if a.tick<anchor)
                history+=tuple(prefix)
                ticks=candidate_ticks[di]
                valid=candidate_valid[di]
                bad=np.zeros(len(ticks),bool)
                bad[valid]=~first_note_support(history,ticks[valid],timing=chart.timing,position_mask=position_mask,
                    touch_position_mask=touch_position_mask,time_bounds=time_bounds[di])
                delta_bad[di]=torch.from_numpy(bad)
                delta_seconds+=time.perf_counter()-delta_started
            if moment<0:continue
            if not conditioned and int(input_event[b,r]) not in (note_id,branch_id):continue
            t=time.perf_counter()
            context=chart.context(moment,prefix)
            locate_seconds+=time.perf_counter()-t;counters['contexts']+=1
            def query(*args):
                nonlocal read_seconds,merge_seconds
                t=time.perf_counter()
                result=context.query(*args)
                counters["queries"]+=1
                elapsed=time.perf_counter()-t
                if context.large:merge_seconds+=elapsed
                else:read_seconds+=elapsed
                return result
            events=query('event')
            any_note=bool(events.allowed[0])
            owner=next((a for a in reversed(prefix) if a.row['event']=='NOTE'),None)
            branch_possible=owner is not None and owner.track
            any_branch=branch_possible and bool(events.allowed[1])
            no_family=not any_note and not any_branch
            # N/T router owns complete N-family failure; event otherwise owns
            # its uncompletable N alternative. Kind/position do not duplicate it.
            rb=np.zeros((4,2),bool)
            if no_family:
                category=next((j for j,m in enumerate(events.bad) if m.any()),0)
                rb[category,1]=True
            add('router',b,r,[True,True],rb)
            if conditioned and not no_family:
                allowed=np.zeros(len(vocabulary.fields['event'].values),bool);allowed[note_id]=True;allowed[branch_id]=branch_possible
                bad=np.zeros((4,len(allowed)),bool)
                bad[:,note_id]=events.bad[:,0];bad[:,branch_id]=events.bad[:,1]
                add('event',b,r,allowed,bad)
            if not conditioned or no_family:continue
            weight=weights[block_ids[b,r]]
            kind=str(kind_names[int(target['kind'][b,r])]) if event==note_id else None
            if event==note_id and any_note:
                ks=query('kind')
                allowed=np.zeros(len(vocabulary.fields['kind'].values),bool);bad=np.zeros((4,len(allowed)),bool)
                for j,k in enumerate(ks.values):
                    i=enc['kind'][k];allowed[i]=True;bad[:,i]=ks.bad[:,j]
                add('kind',b,r,allowed,bad,weight)
                if not ks.allowed[ks.values.index(kind)]:continue
            elif event==branch_id and not any_branch:continue
            field='start_pad' if kind is None or kind in ('slide','wifi') else 'pad'
            position=owner.pad if kind is None and owner else str(pos_names[field][int(target[field][b,r])])
            if event==note_id:
                ps=query('position',kind)
                width=len(vocabulary.fields[field].values)
                shape=(width,2) if ps.bad.ndim==3 else (width,)
                syntax=np.zeros(shape,bool);masks=np.zeros((4,*shape),bool)
                for j,value in enumerate(ps.values):
                    i=enc[field][value];syntax[i]=True;masks[:,i]=ps.bad[:,j]
                name=field+'_short_'+('hold' if kind=='tap' else 'touch_hold') if ps.bad.ndim==3 else field
                add(name,b,r,syntax,masks,weight)
                if position not in ps.values:continue
                p=ps.values.index(position)
                if not np.asarray(ps.allowed[p]).any():continue
            flags=tuple(n for n in ('short_hold','short_touch_hold') if int(target['modifier_'+n][b,r])==1)
            slide=kind is None or kind in ('slide','wifi')
            route=int(target['route'][b,r]) if slide else None
            if slide:
                rs=query('route',kind,position,flags)
                syntax=np.zeros(71,bool);syntax[list(rs.values)]=True
                masks=np.zeros((4,71),bool);masks[:,list(rs.values)]=rs.bad
                add('route',b,r,syntax,masks,weight)
                if route not in rs.values or not rs.allowed[rs.values.index(route)]:continue
            if slide or kind in ('hold','touch_hold'):
                s=query('duration',kind,position,flags,route)
                safe=s.safe_upper
                f=s.field
                lower,upper=int(batch['time_lower'][f][b,r]),int(batch['time_upper'][f][b,r])
                if upper<lower:continue
                durations.setdefault(f,[]).append((b,r,safe,lower,upper,float(chart.data['moment_ticks'][moment]),weight,chart.timing))
    packing=time.perf_counter()

    packed={}
    for rows in entries.values():rows.sort(key=lambda x:(x[0],x[1]))
    rank={'router':0,'event':1,'kind':2,'route':4}
    order=sorted(entries,key=lambda n:(entries[n][0][:2],rank.get(n,3)))
    for name in order:
        rows=entries[name]
        packed[name]=dict(rows=torch.tensor([[x[0],x[1]] for x in rows]),allowed=torch.from_numpy(np.stack([x[2] for x in rows])),
                          bad=torch.from_numpy(np.stack([x[3] for x in rows])),weight=torch.tensor([x[4] for x in rows],dtype=torch.float32))
    tails={}
    for rows in durations.values():rows.sort(key=lambda x:(x[0],x[1]))
    for f in sorted(durations,key=lambda f:durations[f][0][:2]):
        rows=durations[f]
        safe,lower,upper=[torch.tensor([x[i] for x in rows]) for i in (2,3,4)]
        anchor=torch.tensor([x[5] for x in rows])
        from maimai_dataset.event_time import written_tick_seconds
        scale=torch.tensor([written_tick_seconds(x[7],x[5]) for x in rows],dtype=torch.float64)
        threshold=first_tail_tick(safe+1,lower,upper)
        bounds=tick_bounds(f,anchor,threshold.minimum(upper),upper,lower,upper,tick_seconds=scale)
        tails[f]=dict(rows=torch.tensor([[x[0],x[1]] for x in rows]),bounds=bounds,active=threshold<=upper,
                      weight=torch.tensor([x[6] for x in rows],dtype=torch.float32))
    pack_seconds=time.perf_counter()-packing
    return dict(physical_stages=packed,physical_duration_tails=tails,physical_delta_bad=delta_bad,
                physical_interval_cost=dict(read_seconds=read_seconds,merge_seconds=merge_seconds,locate_seconds=locate_seconds,delta_seconds=delta_seconds,pack_seconds=pack_seconds,
                    **counters,prepare_seconds=time.perf_counter()-started))
