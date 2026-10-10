"""Physical actions on an absolute time axis, without a temporal raster.

Immutable actions can be shared across windows and reordered NOTE blocks. A
branch names its NOTE owner; grouping is never inferred from pad coincidence.
"""
from __future__ import annotations
from dataclasses import dataclass, field, replace
from collections import defaultdict
from typing import Mapping, Any
from maimai_dataset.event_time import seconds, duration_end, advance_global_ticks
from maimai_dataset.modifiers import materialize_row
from .catalog import pad_bit

IDENTITY = {'schema': 'maimai.physical_intervals.v3', 'short_ticks': 2,
            'release_ticks': 22, 'head_guard_last_tick': 70, 'pass_seconds': .2,
            'duration': 'written-bpm', 'groups': 'note-owner-additive-tracks', 'raster': None,
            'max_branches': 1, 'touch_repeat_seconds': .25, 'touch_repeat_responsibility': 'pad-only',
            'same_path_wait': 'exact-path-inclusive-launch-all-shapes',
            'touch_hands': 'all-active-one-pool;short-startup-cancels-pool',
            'short_hold': 'tap-touch-equivalent-no-release',
            'touch_on_slide': 'excluded', 'path_coverage': 'directed-wifi-three-straights',
            'handoff': 'ordinary-end-to-start-one-to-one-0..22-global-ticks',
            'handoff_history': 'lock-past-matches-before-pruning'}
EPS = 1e-10


def is_touch(action):
    return action.row.get('kind') in ('touch', 'touch_hold')


def history_end(action):
    return max(action.press_end, action.release, action.guard_end,
               action.press_start + .25 if is_touch(action) else -1.,
               max((s + .2 for s, _ in action.entries), default=-1.))


def touch_repeat(history, pad, second):
    return any(is_touch(a) and a.pad == pad and -EPS <= second-a.press_start <= .25+EPS
               for a in history)


def track_weight(tracks):
    return sum(2 if a.wifi else 1 for a in tracks)


def branch_errors(history, candidate):
    if candidate.row['event'] != 'SLIDE_BRANCH':
        return set()
    tracks = [a for a in history if a.track and a.owner == candidate.owner]
    return {'multitouch'} if len(tracks) >= 2 or any(a.wifi for a in tracks) else set()


@dataclass(frozen=True)
class Action:
    row: Mapping[str, Any]
    moment: int
    owner: int
    tick: float
    pad: str
    press_start: float
    press_end: float
    launch: float
    actual_end: float
    release: float
    guard_end: float
    path: int
    wifi: bool
    entries: tuple[tuple[float, int], ...]
    source_row: Mapping[str, Any] | None = None
    route_class: int | None = None
    end_pad: str = ''
    # Causal history summaries only; never annotate actions from future GT.
    # Comparison excludes these so chart source indices retain their identity.
    handoff_release: float | None = field(default=None, compare=False)
    handoff_incoming: bool = field(default=False, compare=False)
    handoff_outgoing: bool = field(default=False, compare=False)

    @property
    def track(self):
        return self.path >= 0

    @property
    def short(self):
        return self.row.get('event') == 'NOTE' and (self.row.get('kind') in ('tap','touch','slide','wifi')
            or self.row.get('kind') in ('hold','touch_hold') and self.row.get('hold_duration_tick') == 1)


def describe(row, *, tick, moment, owner, timing, catalog, route=None, project=True, time_values=None):
    values={} if time_values is None else time_values
    def cached(key,fn):
        if key not in values:values[key]=fn()
        return values[key]
    source_row = dict(row)
    row = materialize_row(row)
    branch = row['event'] == 'SLIDE_BRANCH'
    kind = row.get('kind')
    slide = branch or kind in ('slide','wifi')
    pad = str(row.get('start_pad' if slide else 'pad', ''))
    start = cached("start",lambda:seconds(timing, tick))
    press_end = start if branch else cached("short-end",lambda:seconds(timing, tick + 2))
    if kind in ('hold','touch_hold') and row['hold_duration_tick'] != 1:
        press_end = cached(('hold',row['hold_duration_tick']),lambda:advance_global_ticks(timing, duration_end(timing,tick,row['hold_duration_tick']),22))
    launch = actual = release = guard_end = -1.
    path, entries = -1, ()
    if slide:
        if route is None:
            raise ValueError('track requires an explicit route')
        path = catalog.path_id(int(pad[1:]), int(route))
        if path < 0:
            raise ValueError('invalid start/route pair')
        launch = cached("launch",lambda:seconds(timing,tick+96))
        actual = cached(('actual',row['branch_duration_tick']),lambda:duration_end(timing,tick,row['branch_duration_tick'],slide=True))
        release = cached(("release",actual),lambda:advance_global_ticks(timing,actual,22))
        guard_end = cached("guard",lambda:seconds(timing,tick+96+70))
        begin,end = (int(x) for x in catalog.path_offsets[path:path+2])
        if project:
            entries = tuple((launch + float(p)*(actual-launch),int(mask)) for p,mask in
                            zip(catalog.path_enter_proportions[begin:end],catalog.path_masks[begin:end]))
    return Action(dict(row),int(moment),int(owner),float(tick),pad,start,press_end,
                  launch,actual,release,guard_end,path,int(route or 0)==70,entries,source_row,route,
                  'A'+catalog.path_key(path)[-1] if slide else '')


def handoff_matches(actions):
    """Deterministic per-track matching, using exact seconds and global tail ends.

    Incoming and outgoing uses are independent so a track can join a chain.
    Only accepted actions and the trial action are supplied by callers.
    """
    tracks = sorted((a for a in actions if a.track and not a.wifi),
                    key=lambda a: (a.launch, a.pad, a.end_pad, a.owner, a.path))
    waiting = defaultdict(list)
    pairs = []
    for following in tracks:
        pool=waiting[following.pad]
        pool[:]=[a for a in pool if a.release>=following.launch-EPS]
        options = [] if following.handoff_incoming else [a for a in pool
            if a.actual_end <= following.launch+EPS and a.launch < following.launch-EPS]
        if options:
            previous = min(options, key=lambda a: (a.actual_end, a.launch, a.owner, a.path))
            pool.remove(previous)
            pairs.append((previous,following))
        if not following.handoff_outgoing:
            waiting[following.end_pad].append(following)
    return tuple(pairs)


def handoff_releases(actions):
    releases = {id(a): a.release if a.handoff_release is None else a.handoff_release for a in actions if a.track}
    for previous,following in handoff_matches(actions):
        releases[id(previous)] = min(previous.release, following.launch)
    return releases


def prune_history(history, second):
    """Resolve once before pruning, retaining only facts of occurred handoffs.

    A retained successor must not acquire another predecessor after its original
    predecessor expires. Future matches remain editable by accepted same-onset
    actions; a trial's later launch cannot change these locked past matches.
    """
    history=tuple(history)
    incoming=set();outgoing={}
    for previous,following in handoff_matches(history):
        if following.launch<=second+EPS:
            incoming.add(id(following));outgoing[id(previous)]=following.launch
    return tuple(replace(a,
        handoff_incoming=a.handoff_incoming or id(a) in incoming,
        handoff_outgoing=a.handoff_outgoing or id(a) in outgoing,
        handoff_release=min(a.release,outgoing[id(a)]) if id(a) in outgoing else a.handoff_release)
        if id(a) in incoming or id(a) in outgoing else a
        for a in history if history_end(a)>=second-EPS)


def hand_segments(actions):
    """Sweep endpoints; startup credits use a bipartite matching per instant.

    Credits last exactly as long as their eligible short press. Group demand is
    recomputed at each track release, so adding a branch costs only its delta.
    """
    releases = handoff_releases(actions)
    points = sorted({p for a in actions for p in
                    (a.press_start,a.press_end,*((a.launch,a.release) if a.track else ()))})
    result = []
    for lo,hi in zip(points,points[1:]):
        if hi <= lo + EPS:
            continue
        presses = [a for a in actions if a.press_start <= lo + EPS and a.press_end > lo + EPS]
        groups = defaultdict(list)
        for a in actions:
            if a.track and a.launch <= lo + EPS and releases[id(a)] > lo + EPS:
                groups[a.owner].append(a)
        touches = [a for a in presses if is_touch(a)]
        ordinary = [a for a in presses if not is_touch(a)]
        demand = len(ordinary) + bool(touches) + sum(track_weight(tracks) for tracks in groups.values())
        edges = {i: [g for g,tracks in groups.items() if any(
            a.pad == track.pad and abs(a.press_start-track.launch) <= EPS for track in tracks)]
                 for i,a in enumerate(ordinary) if a.short}
        if touches:
            edges['touch-pool'] = sorted({g for a in touches if a.short for g, tracks in groups.items()
                if any(a.pad == track.pad and abs(a.press_start-track.launch) <= EPS for track in tracks)})
        matched = {}
        def augment(i, seen):
            for group in edges[i]:
                if group in seen:
                    continue
                seen.add(group)
                if group not in matched or augment(matched[group],seen):
                    matched[group] = i
                    return True
            return False
        demand -= sum(augment(i,set()) for i in edges)
        if result and result[-1][2] == demand and abs(result[-1][1]-lo) <= EPS:
            result[-1] = (result[-1][0],hi,demand)
        else:
            result.append((lo,hi,demand))
    return tuple(result)


def geometry_reasons(history, candidate, catalog, *, check_same_path=True):
    reasons = set()
    t = candidate.press_start
    bit = pad_bit(candidate.pad)
    launches_here = any(a.track and a.pad == candidate.pad and abs(a.launch-t)<=EPS for a in history)
    for a in history:
        if candidate.row['event'] == 'NOTE':
            if a.press_start <= t+EPS and t < a.press_end-EPS and pad_bit(a.pad)&bit:
                reasons.add('overlap')
            if a.track and not is_touch(candidate):
                protected = a.launch-EPS <= t <= a.guard_end+EPS and a.pad==candidate.pad
                if protected and not launches_here:
                    reasons.add('tap_on_slide')
                for entered,mask in a.entries:
                    # The startup exception removes only the origin's ban.
                    if entered < t-EPS and t <= entered+.2+EPS and mask&bit:
                        if not (launches_here and a.pad==candidate.pad and abs(entered-a.launch)<=EPS):
                            reasons.add('tap_on_slide')
        if candidate.track and a.track:
            if (candidate.pad == a.pad and candidate.path == a.path
                    and a.press_start-EPS <= t <= a.launch+EPS):
                reasons.add('slide_too_fast')
            if check_same_path and not int(catalog.path_flags[candidate.path])&3 and not int(catalog.path_flags[a.path])&3:
                if catalog.covers(a.path,candidate.path) and a.launch <= candidate.launch+EPS and a.actual_end >= t-EPS:
                    reasons.add('slide_too_fast')
    return reasons


def reasons(history, candidate, catalog, *, check_same_path=True, include_touch_repeat=False):
    result = geometry_reasons(history,candidate,catalog,check_same_path=check_same_path) | branch_errors(history, candidate)
    if include_touch_repeat and is_touch(candidate) and touch_repeat(history, candidate.pad, candidate.press_start):
        result.add('overlap')
    # Existing impossible history must not poison an unrelated later action.
    spans = [(candidate.press_start,candidate.press_end)]
    if candidate.track:
        spans.append((candidate.launch,candidate.release))
    def overloaded(lo,hi,n):
        return n>2 and any(lo<end-EPS and hi>start+EPS for start,end in spans)
    if any(overloaded(lo,hi,n) for lo,hi,n in hand_segments((*history,candidate))):
        result.add('multitouch')
    return result


def first_note_support(history,ticks,*,timing,position_mask,touch_position_mask=0,time_bounds=None):
    """Vectorized any-short-note feasibility at candidate global ticks.

    A legal first NOTE can always use the two-tick Tap/Touch variant. Longer
    holds or a track cannot rescue a forbidden head. This scans interval
    endpoints and path entries, never time slots or model probabilities.
    """
    import numpy as np
    ticks=np.asarray(ticks,dtype=np.float64)
    from maimai_baseline.time.coordinates import seconds_at_ticks
    if time_bounds is None:
        ts=seconds_at_ticks(timing,ticks);ends=seconds_at_ticks(timing,ticks+2)
    else:
        ts,ends=time_bounds
    if not len(ts):return np.zeros(0,dtype=bool)
    # Expired actions cannot influence any candidate. Keeping them in the hand
    # sweep would make each new-MOMENT query quadratic in the whole chart.
    first=float(ts.min())
    history=prune_history(history,first)
    occupied=np.zeros(len(ts),dtype=np.int64)
    overlap=np.zeros(len(ts),dtype=np.int64)
    slide_bans=np.zeros(len(ts),dtype=np.int64)
    launches={}
    for a in history:
        if a.track:
            launches.setdefault(a.pad,np.zeros(len(ts),dtype=bool))[:] |= np.abs(ts-a.launch)<=EPS
    for a in history:
        bit=pad_bit(a.pad)
        overlap[(ts>=a.press_start-EPS)&(ts<a.press_end-EPS)] |= bit
        if not a.track:continue
        exception=launches.get(a.pad,False)
        slide_bans[(ts>=a.launch-EPS)&(ts<=a.guard_end+EPS)&~exception] |= bit
        for entered,mask in a.entries:
            active=(ts>entered+EPS)&(ts<=entered+.2+EPS)
            if abs(entered-a.launch)<=EPS:
                active &= ~exception
            slide_bans[active] |= mask
    for lo,hi,count in hand_segments(history):
        active=(ts<hi-EPS)&(ends>lo+EPS)
        occupied[active]=np.maximum(occupied[active],count)
    tap_allowed=(~(overlap|slide_bans))&int(position_mask)
    touch_allowed=(~overlap)&int(touch_position_mask)
    allowed=tap_allowed|touch_allowed
    normal=(occupied<=1)&(allowed!=0)
    from .query_context import IntervalQueryContext
    context=IntervalQueryContext(history,first,None,False)
    # A Touch can join the pool and remain active after an old member releases.
    shared=np.ones(len(ts),dtype=bool)
    for lo,hi,_,touches,_,_,_,demand in context.panels:
        if demand+int(not touches)>2:
            shared &= ~((ts<hi-EPS)&(ends>lo+EPS))
    normal |= shared&(touch_allowed!=0)
    # A two-hand group can absorb exactly one matching short startup press.
    startup=np.zeros(len(ts),dtype=bool)
    for pad,active in launches.items():
        bit=pad_bit(pad)
        for i in np.flatnonzero(active&((allowed&bit)!=0)&~normal):
            for kind,mask in (('tap',tap_allowed),('touch',touch_allowed)):
                if not mask[i]&bit:continue
                candidate=describe(dict(event='NOTE',kind=kind,pad=pad),tick=float(ticks[i]),moment=-1,
                    owner=max((a.owner for a in history),default=-1)+1,timing=timing,catalog=None)
                if not context.overloaded(candidate):startup[i]=True;break
    return normal|startup
