"""MiaCode-constrained sampling, retries, and fallback."""

from __future__ import annotations
from copy import deepcopy
from maimai_baseline.inference.physical import choose_row
from maimai_baseline.inference.reporting import count, route_of, trace_enabled, failure_record
from maimai_baseline.physical.inference_state import PhysicalDecodeError
from maimai_baseline.physical.miacode_inference import POLICY
import torch


def unconstrained(outputs, *, reason, attempts=(), **kw):
    from maimai_baseline.inference.sampling import _choose_route_row
    # Only an explicit physical dead end can arrive here. NaNs are not a
    # physical violation and must never enter the fail-open path.
    tensors = list(outputs["field_logits"].values()) + list(outputs["time_outputs"].values()) + [outputs["route_logits"]]
    if not all(bool(torch.isfinite(t).all()) for t in tensors):
        raise PhysicalDecodeError("non-finite model output in physical bypass")
    state = kw["state"]
    row, trace, drift = _choose_route_row(outputs, model=kw["model"], codec=kw["codec"],
        target_current_tick=kw["tick"], target_end_tick=kw["target_end"], song_end_tick=state.song_end_tick, audio_tail_tick=state.audio_tail_tick,audio_end_seconds=state.audio_end_seconds,
        timing=state.timing, audio_frame_start=kw["audio_frame_start"], moment_count=kw["moment_count"],
        owner_branch_count=sum(a.row['event']=='SLIDE_BRANCH' and a.owner==state.owner(kw['moment']).owner for a in state.actions) if state.owner(kw['moment']) else 0,
        owner_start=state.owner(kw["moment"]).pad if state.owner(kw["moment"]) else None,
        cumulative_continuous_drift=kw["drift"], temperature=kw["temperature"], generator=kw["generator"])
    if row["event"] == "SLIDE_BRANCH":
        owner = state.owner(kw["moment"])
        if owner is None:
            raise PhysicalDecodeError("branch has no grammar owner during bypass")
        row["start_pad"] = owner.row["start_pad"]
        trace["row"] = dict(row)
    if row["event"] == "MOMENT":
        state.bypass_first = True
    elif row["event"] in ("NOTE", "SLIDE_BRANCH"):
        state.bypass_next = True
    count("physical_bypass_rows")
    trace["miacode"] = dict(policy=POLICY, attempts=list(attempts) if trace_enabled() else [], bypass=True, reason=reason,
                            final_row=dict(row), probability_scope="grammar-only-fallback")
    trace["physical"] = dict(policy=POLICY, bypass=True, decisions=[], rules={}, retry_count=0)
    failure_record(tick=kw['tick'], moment=kw['moment'], stage='bypass', reason=reason, row=dict(row))
    return row, trace, drift


def select_row(outputs, **kw):
    state = kw["state"]
    if state.bypass_first:
        return unconstrained(outputs, reason="first-note-after-moment-bypass", **kw)
    attempts, excluded_events, excluded_kinds = [], set(), set()
    controls = {}
    for event_attempt in range(32):
        controls.update(excluded_events=excluded_events, excluded_kinds=excluded_kinds)
        try:
            row, trace, drift = choose_row(outputs, controls=controls, **kw)
        except PhysicalDecodeError as exc:
            if "non-finite" in str(exc):
                raise
            if not exc.context.get("recoverable"):
                raise
            failure_record(tick=kw['tick'], moment=kw['moment'], stage='event',
                           reason=str(exc), detail=exc.context, event_attempts=event_attempt+1,
                           excluded_events=sorted(excluded_events), excluded_kinds=sorted(excluded_kinds))
            if controls.get('kind'):
                excluded_kinds.add(controls['kind'])
                controls = {}
                continue
            failed_event = exc.context.get('event')
            if failed_event in ('NOTE', 'SLIDE_BRANCH', 'MOMENT') and failed_event not in excluded_events:
                excluded_events.add(failed_event)
                controls = {}
                continue
            exc.context.update(excluded_events=sorted(excluded_events),
                               excluded_kinds=sorted(excluded_kinds), event_attempts=event_attempt+1)
            raise
        if row["event"] not in ("NOTE", "SLIDE_BRANCH"):
            if not attempts and trace.get("physical", {}).get("forced_window_closure"):
                raise PhysicalDecodeError('physical would force window closure', recoverable=True, stage='event')
            trace["miacode"] = dict(policy=POLICY, attempts=attempts, bypass=False, final_row=dict(row))
            return row, trace, drift
        kind = row.get("kind")
        slide = row["event"] == "SLIDE_BRANCH" or kind in ("slide", "wifi")
        current = state.local_query(kw["tick"], kw["moment"], "position", kind)
        routes=(70,) if kind=='wifi' else tuple(range(71 if kind is None else 70)) if slide else ()
        duration_field='branch_duration_tick' if slide else 'hold_duration_tick' if kind in ('hold','touch_hold') else None
        fixed = dict(row)
        positions, durations = set(), {}
        maximum = 3 if slide else len(current.values)
        for attempt in range(maximum):
            route = route_of(row, kw["codec"])
            accepted, detail = state.check(row, kw["tick"], kw["moment"], route)
            count("candidate_checks")
            for rule,_,_ in {(d["rule"],d["affected"],d["cause"]) for d in detail.get("violations",())}:
                count("rejected."+rule)
            for rule,_,_ in {(d["rule"],d["affected"],d["cause"]) for d in detail.get("warnings",())}:
                count("warning."+rule)
            attempts.append(dict(row=dict(row), route=route, accepted=accepted, **(deepcopy(detail) if trace_enabled() else {})))
            if accepted:
                trace["miacode"] = dict(policy=POLICY, attempts=attempts, bypass=False, final_row=dict(row),
                    probability_scope="bounded-field-retries", event_attempts=event_attempt+1)
                return row, trace, drift
            pi = current.values.index(row[current.field])
            index = pi*len(routes)+routes.index(route) if slide else pi
            if slide:
                durations.setdefault(index, set()).add(int(row[duration_field]))
            else:
                positions.add(index)
            if attempt+1 == maximum:
                break
            count("slide_retry" if slide else "pad_retry")
            retry = dict(event=row["event"], kind=kind, fixed_row=fixed,
                         excluded_positions=positions, excluded_durations=durations)
            if not slide and duration_field:
                retry["fixed_duration"] = int(fixed[duration_field])
            try:
                row, trace, drift = choose_row(outputs, controls=retry, **kw)
            except PhysicalDecodeError as exc:
                if "non-finite" in str(exc) or not exc.context.get("recoverable"):
                    raise
                break
        if slide and kind is None:
            excluded_events.add("SLIDE_BRANCH")
            controls = {}
            attempts[-1]["fallback"] = "exclude-branch"
            count("fallback.branch->exclude-branch")
        elif kind in ("slide", "wifi", "hold", "touch_hold"):
            excluded_kinds.add(kind)
            controls = dict(event="NOTE", kind="tap")
            attempts[-1]["fallback"] = "tap"
            count("fallback."+kind+"->tap")
        else:
            excluded_kinds.add(kind)
            controls = {}
        # A finite pad support can be exhausted before the loop count. Both
        # cases follow the same explicit kind/event fallback, never recursion.
    state.local_counts["event_budget_exhausted"] += 1
    raise PhysicalDecodeError('event budget exhausted', recoverable=True, stage='event',
                              event_attempts=32, excluded_events=sorted(excluded_events),
                              excluded_kinds=sorted(excluded_kinds))


def summarize(traces):
    from collections import Counter
    rejected, fallback, warnings, kinds, durations = Counter(), Counter(), Counter(), Counter(), Counter()
    count = bypass = 0
    for trace in traces:
        detail = trace.get("miacode", {})
        bypass += int(detail.get("bypass", False))
        for attempt in detail.get("attempts", ()):
            count += 1
            # Runtime and static references may describe the same interaction.
            # Preserve both evidence records in trace, count one identity here.
            for rule, _, _ in {(d["rule"], d["affected"], d["cause"]) for d in attempt.get("violations", ())}:
                rejected[rule] += 1
            for rule, _, _ in {(d["rule"], d["affected"], d["cause"]) for d in attempt.get("warnings", ())}:
                warnings[rule] += 1
            if attempt.get("fallback"):
                fallback[f"{attempt['row'].get('kind', 'branch')}->{attempt['fallback']}"] += 1
        row = trace.get("row", {})
        if row.get("event") == "NOTE":
            kinds[row["kind"]] += 1
        for name in ("hold_duration_tick", "branch_duration_tick"):
            if name in row:
                durations[f"{name}:{row[name]}"] += 1
    return dict(policy=POLICY, candidate_checks=count, physical_bypass_rows=bypass,
        rejection_counts=dict(rejected), warning_observations=dict(warnings), fallback_counts=dict(fallback),
        note_kinds=dict(kinds), duration_counts=dict(durations), finite_retry_is_not_full_support_mass=True)
