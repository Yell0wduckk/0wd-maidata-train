"""Physical-constrained row sampling."""

from __future__ import annotations
from maimai_baseline.contract import REMOVED_FIELDS
from maimai_dataset.modifiers import LEGACY_FIELDS
from maimai_baseline.inference.reporting import count, row_cached, top5, trace_enabled
from maimai_baseline.inference.sampling import choose
from maimai_baseline.physical.inference_state import KINDS, POLICY, PhysicalDecodeError
from typing import Any
import maimai_baseline.inference.sampling as g_sampling
import numpy as np
import time
import torch


def delta_options(state, *, tick, target_end, first, moment, excluded):
    from maimai_baseline.physical.local_query import moment_support
    return moment_support(state, tick=tick, target_end=target_end, first=first,
                          moment=moment, excluded=excluded)


def select_row(outputs, **kw):
    """Try remaining conditional events before escalating to the router."""
    from .reporting import failure_record
    excluded = set()
    for attempt in range(32):
        try:
            return choose_row(outputs, controls={'excluded_events': excluded}, **kw)
        except PhysicalDecodeError as error:
            if not error.context.get('recoverable') or 'non-finite' in str(error):
                raise
            failure_record(tick=kw['tick'], moment=kw['moment'], stage='event',
                           reason=str(error), detail=error.context, event_attempts=attempt+1,
                           excluded_events=sorted(excluded))
            event = error.context.get('event')
            if event not in ('NOTE', 'SLIDE_BRANCH', 'MOMENT') or event in excluded:
                raise
            excluded.add(event)
    raise PhysicalDecodeError('event budget exhausted', recoverable=True,
                              event_attempts=32, excluded_events=sorted(excluded))


def choose_row(outputs, *, model, codec, state, tick, target_end, moment, moment_count,
               audio_frame_start, drift, temperature, generator, excluded_ticks=(), controls=None):

    started = time.perf_counter()
    from maimai_baseline.model.row_fields import sample_readout
    vocabulary = model.vocabulary
    fields = {name: value[0,-1] for name,value in outputs["field_logits"].items()}
    all_logits = torch.cat([value.reshape(-1) for value in (*fields.values(),outputs["route_logits"])])
    if not bool(torch.isfinite(all_logits).all()):
        raise PhysicalDecodeError("non-finite model logits",tick=tick,moment=moment)
    event_ids = [i for i in range(fields["event"].numel())
                 if float(fields["event"][i]) > -9000 and vocabulary.fields["event"].decode(i) != "TARGET_START"]
    event_names = [vocabulary.fields["event"].decode(i) for i in event_ids]
    events = state.local_query(tick,moment,'event')
    kind_support = None
    from maimai_baseline.physical.local_query import moment_support
    has_delta = moment_support(state, tick=tick, target_end=target_end, first=moment_count==0,
        moment=moment+1, excluded=excluded_ticks, exists=True) if 'MOMENT' in event_names else False
    controls = controls or {}
    def kind_allowed(k):
        return bool(kind_support is not None and kind_support.allowed[kind_support.values.index(k)]
            and k not in controls.get('excluded_kinds',()) and (not controls.get('kind') or k==controls['kind']))
    viable = {'NOTE':bool(events.allowed[0]), 'SLIDE_BRANCH':bool(events.allowed[1]),
              'MOMENT':has_delta, 'SEGMENT_END':True}
    for name in controls.get("excluded_events", ()):
        viable[name] = False
    if controls.get("event"):
        viable = {e: v and e == controls["event"] for e, v in viable.items()}
    if not event_ids or not any(viable.get(e,False) for e in event_names):
        raise PhysicalDecodeError("no legal event completion", recoverable=True,tick=tick,moment=moment,
                                  events=event_names,stage="event")
    ei, event_detail = choose(fields["event"][event_ids], [viable.get(e,False) for e in event_names],
        temperature=temperature,generator=generator,name="event")
    event = event_names[ei]
    try:
        row: dict[str,Any] = {"event":event}
        raw_top5 = top5(fields, vocabulary)
        trace = dict(schema="maimai.baseline_prediction_trace.v1",raw_top5=raw_top5,time={},temperature=temperature)
        physical = dict(policy=POLICY,decisions=[event_detail],rules={},retry_count=0)
        physical["forced_window_closure"] = (
            event == "SEGMENT_END" and any(e != "SEGMENT_END" for e in event_names)
            and not any(viable.get(e,False) for e in event_names if e != "SEGMENT_END")
        )
        trace["physical"] = physical
        route = None
        if event == "MOMENT":
            projected, delta_allowed = delta_options(state, tick=tick, target_end=target_end,
                first=moment_count==0, moment=moment+1, excluded=excluded_ticks)
            candidate = outputs.get("delta_candidate")
            if candidate is not None:
                candidate = {"query":candidate["query"][0,-1],"keys":candidate["keys"][0],
                             "beats":candidate["beats"][0],"padding_mask":candidate["padding_mask"][0],"gate":candidate["gate"]}
            delta, detail = g_sampling._delta_time_choice(outputs["time_outputs"]["delta_ms"][0,-1],
                lower_tick=0 if moment_count==0 else 1,upper_tick=target_end-tick-1,anchor_tick=tick,
                timing=state.timing,audio_frame_start=audio_frame_start,temperature=temperature,
                generator=generator,candidate=candidate,physical_allowed=delta_allowed,
                moment_distribution=model.moment_distribution,
                song_end_tick=state.song_end_tick)
            row["delta_tick"] = delta
            detail.update(support=[0 if moment_count==0 else 1,target_end-tick-1],official_tick=None,
                          official_error_tick=None,nearest_common_tick_distance=min(abs(delta-v) for v in g_sampling.COMMON_TICKS),
                          cumulative_continuous_vs_integer_drift_tick=drift)
            trace["time"][detail["model_field"]] = detail
            physical["decisions"].append(detail["physical"])
            physical["rules"]["delta_or_no_first_note"] = len(delta_allowed)-sum(delta_allowed)
        elif event in ("NOTE","SLIDE_BRANCH"):
            kind_support = state.local_query(tick,moment,'kind') if event=='NOTE' else None
            excluded_pairs = {kind:set() for kind in (*KINDS,None)}
            exhausted=set()
            selected_kind=None
            saved_position=None
            readouts={}
            def readout(row,stage,**kw):
                key=(stage,tuple(sorted(row.items())),tuple(sorted(kw.items())))
                if key not in readouts:
                    readouts[key]=sample_readout(model,outputs,row,stage,**kw)
                return readouts[key]
            excluded_short = {k:set() for k in ('tap','touch')}
            def backtrack(stage):
                state.local_counts['backtrack:'+stage] += 1
                from .reporting import failure_record
                failure_record(tick=tick, moment=moment, stage='field', backtrack=stage,
                               event=event, kind=kind, field_attempts=attempt+1,
                               excluded_kinds=sorted(exhausted))
            def reject(index, short_bit):
                if kind in excluded_short:
                    excluded_short[kind].add((pi,int(short_bit)))
                else:
                    excluded_pairs[kind].add(index)
            for attempt in range(32):
                row = {'event':event}
                if event == 'NOTE':
                    fields.update({n:v[0,-1] for n,v in readout(row,'kind').items()})
                    ids = list(vocabulary.allowed_value_ids(event,'kind'))
                    kinds = [str(vocabulary.fields['kind'].decode(i)) for i in ids]
                    viable_kinds=[kind_allowed(k) and k not in exhausted for k in kinds]
                    if not any(viable_kinds):
                        raise PhysicalDecodeError('no legal NOTE after final-check retries',recoverable=True,tick=tick,moment=moment)
                    if selected_kind is None:
                        ki,kd=choose(fields['kind'][ids],viable_kinds,temperature=temperature,generator=generator,name='kind')
                        selected_kind=kinds[ki]
                        physical['decisions'].append(kd)
                    kind=selected_kind;row['kind']=kind
                else:
                    kind=None
                current=state.local_query(tick,moment,'position',kind)
                fields.update({n:v[0,-1] for n,v in readout(row,'position_flags',
                    owner_start=current.values[0] if kind is None else None).items()})
                allowed=current.allowed.copy()
                slide=kind is None or kind in ('slide','wifi')
                route_values=(70,) if kind=='wifi' else tuple(range(71 if kind is None else 70)) if slide else ()
                for p in range(len(current.values)):
                    if p in controls.get('excluded_positions',()):allowed[p]=False
                    if not slide and p in excluded_pairs[kind]:allowed[p]=False
                    if kind in excluded_short:
                        for bit in (0,1):
                            if (p,bit) in excluded_short[kind]:allowed[p,bit]=False
                    if slide and all(p*len(route_values)+j in excluded_pairs[kind] for j in range(len(route_values))):allowed[p]=False
                if not allowed.any():
                    if kind is None:
                        raise PhysicalDecodeError('no branch completion after retries',recoverable=True,tick=tick,moment=moment)
                    exhausted.add(kind);selected_kind=None;saved_position=None;backtrack('kind');continue
                value_ids=[vocabulary.fields[current.field].encode(v) for v in current.values]
                short_name='short_hold' if kind=='tap' else 'short_touch_hold' if kind=='touch' else None
                if saved_position is None:
                    short_bit=None
                    if kind is None:pi=0
                    elif short_name:
                        scores=torch.log_softmax(fields[current.field][value_ids].float(),-1)[:,None]+torch.log_softmax(fields['modifier_'+short_name].float(),-1)[None,:]
                        selected,detail=choose(scores,allowed,temperature=temperature,generator=generator,name=current.field+'_short_bit')
                        pi,short_bit=np.unravel_index(selected,allowed.shape)
                        physical['decisions'].append(detail)
                    else:
                        pi,detail=choose(fields[current.field][value_ids],allowed,temperature=temperature,generator=generator,name=current.field)
                        physical['decisions'].append(detail)
                    row[current.field]=current.values[pi]
                    g_sampling.sample_modifiers(row,fields,tick=tick,tail_tick=state.audio_tail_tick,timing=state.timing,audio_end_seconds=state.audio_end_seconds,
                        temperature=temperature,generator=generator,short_allowed=True,
                        fixed_short=None if short_bit is None else int(short_bit))
                    if controls.get('fixed_row'):
                        for n in ('flags','head_flags','branch_flags'):
                            if n in controls['fixed_row']:row[n]=controls['fixed_row'][n]
                    saved_position=(dict(row),pi,short_bit)
                else:
                    row,pi,short_bit=saved_position
                    row=dict(row)
                position=row[current.field]
                flags=row.get('flags',())
                route=None
                if slide:
                    rs=state.local_query(tick,moment,'route',kind,position,flags)
                    route_allowed=rs.allowed.copy()
                    for j in range(len(rs.values)):
                        if pi*len(rs.values)+j in excluded_pairs[kind]:route_allowed[j]=False
                    if not route_allowed.any():
                        excluded_pairs[kind].update(pi*len(rs.values)+j for j in range(len(rs.values)))
                        saved_position=None;backtrack('position');continue
                    route_logits=readout(row,'route',owner_start=position)['route'][0,-1]
                    ri,detail=choose(route_logits[list(rs.values)],route_allowed,temperature=temperature,generator=generator,name='route')
                    physical['decisions'].append(detail)
                    route=rs.values[ri];index=pi*len(rs.values)+ri
                    for name,value in codec.restore_factor_ids(route).items():
                        if value>=0:
                            decoded=vocabulary.fields[name].decode(value)
                            if decoded is not None:row[name]=decoded
                    trace['route']=dict(route_class=route,route_id=f'B{route:03d}',canonical_path=codec.paths[route],probability_scope='conditional-route')
                else:index=pi
                duration_support=state.local_query(tick,moment,'duration',kind,position,flags,route) if slide or kind in ('hold','touch_hold') else None
                if duration_support is not None and (not duration_support.exists or duration_support.safe_upper < duration_support.duration_lower):
                    reject(index, short_bit)
                    if not slide:saved_position=None
                    backtrack('route' if slide else 'position');continue
                for name in vocabulary.field_names:
                    if name in {"event","kind","branch_index","head_kind",current.field,*REMOVED_FIELDS,*LEGACY_FIELDS} or name not in vocabulary.applicability.get(event,()):
                        continue
                    if event=="NOTE" and name not in vocabulary.note_kind_value_constraints.get(kind,{}):
                        continue
                    candidates = vocabulary.allowed_value_ids(event,name,note_kind=kind)
                    value = g_sampling._sample_index(fields[name],candidates,temperature=temperature,generator=generator)
                    decoded = vocabulary.fields[name].decode(value)
                    if decoded is not None:
                        row[name] = decoded
                if controls.get("fixed_row"):
                    for name, value in controls["fixed_row"].items():
                        if name not in {"event", "kind", "pad", "start_pad", *REMOVED_FIELDS}:
                            row[name] = value
                if event=='SLIDE_BRANCH':
                    row['branch_index']=1+sum(a.row['event']=='SLIDE_BRANCH' and a.owner==state.owner(moment).owner for a in state.actions)
                elif kind in ('slide','wifi'):
                    row['head_kind']='star'
                if duration_support is not None:
                    duration_field=duration_support.field
                    duration_logits = readout(row,"duration",owner_start=row.get("start_pad"),route=route)
                    if "fixed_duration" in controls:
                        duration = controls["fixed_duration"]
                        detail = dict(head="fixed_retry", physical=dict(field=duration_field,
                            original_count=1, legal_count=1, removed_probability_mass=0.))
                    else:
                        try:
                            duration, detail = g_sampling._time_choice(duration_logits[duration_field][0,-1],
                                field=duration_field,lower=duration_support.duration_lower,upper=duration_support.duration_upper,
                                time_unit_scale=float(outputs.get("time_unit_scale",1.)),temperature=temperature,generator=generator,
                                duration_distribution=outputs.get("duration_distribution"), anchor_tick=tick, timing=state.timing,
                                physical_safe_upper=duration_support.safe_upper,
                                excluded_values=tuple(controls.get("excluded_durations", {}).get(index, ())))
                        except PhysicalDecodeError as exc:
                            if str(exc).startswith('empty Physical support'):
                                reject(index, short_bit)
                                if not slide:saved_position=None
                                backtrack('route' if slide else 'position');continue
                            raise
                    row[duration_field] = duration
                    detail.update(support=[duration_support.duration_lower,duration_support.duration_upper],official_tick=None,official_error_tick=None,
                                  nearest_common_tick_distance=min(abs(duration-v) for v in g_sampling.COMMON_TICKS))
                    trace["time"][duration_field] = detail
                    physical["decisions"].append(detail["physical"])
                if state.validate(row,tick,moment,route):
                    break
                # Exclude this position/path combination for the whole transaction.
                # This exceptional conservative repair is explicitly counted.
                reject(index, short_bit)
                if not slide:saved_position=None
                backtrack('final_check')
                physical["retry_count"] += 1
                count("physical_final_check_retry")
            else:
                state.local_counts["row_budget_exhausted"] += 1
                raise PhysicalDecodeError("Physical row retry budget exhausted",recoverable=True,tick=tick,moment=moment)
        trace["row"] = dict(row)
        physical["prepare_and_sampling_seconds"] = time.perf_counter()-started
        return row,trace,drift
    except PhysicalDecodeError as error:
        error.context.setdefault('event', event)
        error.context.setdefault('field_attempts', locals().get('attempt', -1)+1)
        error.context.setdefault('excluded_kinds', sorted(locals().get('exhausted', ())))
        raise



def summarize(traces):
    """Keep probability scopes separate; rule overlaps are not disjoint mass."""
    decisions, rules = {}, {}
    total_seconds = 0.
    retries = 0
    forced_closures = 0
    for trace in traces:
        physical = trace.get("physical",{})
        total_seconds += physical.get("prepare_and_sampling_seconds",0.)
        retries += physical.get("retry_count",0)
        forced_closures += int(physical.get("forced_window_closure",False))
        for item in physical.get("decisions",[]):
            result = decisions.setdefault(item["field"],dict(count=0,removed_mass_sum=0.,legal_count_sum=0))
            result["count"] += 1
            result["removed_mass_sum"] += item["removed_probability_mass"]
            result["legal_count_sum"] += item["legal_count"]
        for name,item in physical.get("rules",{}).items():
            result = rules.setdefault(name,dict(checks=0,illegal_count=0,removed_mass_sum=0.))
            result["checks"] += 1
            if isinstance(item,dict):
                result["illegal_count"] += item["illegal_count"]
                result["removed_mass_sum"] += item["removed_probability_mass"]
                result["probability_scope"] = item["probability_scope"]
            else:
                result["illegal_count"] += item
                result["probability_scope"] = "count-only"
    return dict(decisions=decisions,rules=rules,row_retries=retries,forced_window_closures=forced_closures,prepare_and_sampling_seconds=total_seconds,
                overlapping_rule_masses_are_not_additive=True)
