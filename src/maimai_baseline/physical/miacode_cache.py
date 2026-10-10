"""Immutable prefix projections and transactional outer-judge checkpoints.

The uncached runtime remains the independent reference. Spatial/time indexes
only narrow exact predicates; no temporal look-back is imposed on the ledger.
"""
from bisect import bisect_left, bisect_right, insort
from collections import defaultdict
from functools import lru_cache
import math
import heapq

from maimai_baseline.physical.miacode_runtime import TPS
from maimai_baseline.physical.miacode_runtime import EPS
from maimai_baseline.physical.miacode_runtime import RELEASE
from maimai_baseline.physical.miacode_runtime import normalize
from maimai_baseline.physical.miacode_runtime import entry
from maimai_baseline.physical.miacode_runtime import touch_groups
from maimai_baseline.physical.miacode_runtime import Diagnostic
from maimai_baseline.physical.miacode_runtime import outer_level


class TimeIndex:
    def __init__(self, prior=None):
        self.buckets = dict(prior.buckets) if prior else {}

    def add(self, key, second, identity):
        bucket = list(self.buckets.get(key, ()))
        insort(bucket, (second, identity))
        self.buckets[key] = tuple(bucket)

    def find(self, key, second, radius):
        bucket = self.buckets.get(key, ())
        a = bisect_left(bucket, (second-radius, -math.inf))
        b = bisect_right(bucket, (second+radius, math.inf))
        return (identity for _, identity in bucket[a:b])


@lru_cache(maxsize=8192)
def normalize_one(marker, slides):
    return normalize((marker,), slides=slides)[0]


@lru_cache(maxsize=8192)
def action_events(m, group_active):
    """Per-action contributions, before cross-action pad-down union."""
    events, spans = [], []
    def put(stage, tick, pad, extra=False):
        events.append((stage, tick, pad, m.id, extra))
    def press(end):
        lo, hi = math.ceil(m.second*TPS-EPS), math.ceil(end*TPS-EPS)-1
        if hi >= lo:
            spans.append((m.pad, lo, hi, m.id))
    if m.slide:
        e = entry(m)
        if not m.after_slide:
            first = e['pad_enter_times'][0]['t']*(m.end-m.launch) if e['pad_enter_times'] else .05
            put(0, math.floor((m.launch+min(first,.05))*TPS+EPS)+1, m.pad, True)
        put(1, math.ceil(m.launch*TPS-EPS), m.pad)
        for item in e['pad_enter_times']:
            put(2, math.ceil((m.launch+item['t']*(m.end-m.launch))*TPS-EPS), item['pad'])
        if m.head:
            press(m.second+RELEASE)
    elif not (m.kind == 'tap' and m.slide_head or m.kind == 'touch' and m.on_slide and not group_active):
        press(m.end+(0 if m.tail_on_head and m.kind in ('hold','touch_hold') else RELEASE))
    return tuple(events), tuple(spans)


def merged_pad(parts, pad):
    events, spans = [], []
    for ev, sp in parts.values():
        events.extend(e for e in ev if e[2] == pad)
        spans.extend((lo,hi,cause) for p,lo,hi,cause in sp if p == pad)
    result = {}
    for stage,tick,_,cause,extra in sorted(events):
        if tick not in result or result[tick][0] <= cause:
            result[tick] = cause,extra
    changes = defaultdict(list)
    for lo,hi,cause in spans:
        changes[lo].append((cause,1));changes[hi+1].append((cause,-1))
    active = set()
    for tick, changes_here in sorted(changes.items()):
        had = bool(active)
        for cause,sign in changes_here:
            if sign > 0: active.add(cause)
            else: active.discard(cause)
        if active and not had:
            result[tick] = max(active),False
    return tuple((tick,pad,cause,extra) for tick,(cause,extra) in sorted(result.items()))


class Scene:
    def __init__(self, raw, prior=None):
        self.raw = tuple(raw)
        append = prior is not None and self.raw[:len(prior.raw)] == prior.raw
        self.index = TimeIndex(prior.index if append else None)
        slides = {m.id:m for m in raw if m.slide}
        for m in (self.raw[len(prior.raw):] if append else raw):
            if not m.slide: continue
            self.index.add(('launch',m.pad),m.launch,m.id)
            self.index.add(('end',f"A{entry(m)['end']}"),m.end,m.id)
            for e in entry(m)['pad_enter_times']:
                self.index.add(('entry',e['pad']),m.launch+e['t']*(m.end-m.launch),m.id)
        normalized = []
        for m in raw:
            relevant = set(self.index.find(('launch',m.pad),m.second,24/TPS+EPS))
            relevant.update(self.index.find(('launch',m.pad),m.end,1/TPS+EPS))
            if m.slide:
                relevant.update(self.index.find(('launch',f"A{entry(m)['end']}"),m.end,1/TPS+EPS))
                relevant.update(self.index.find(('end',m.pad),m.launch,1/TPS+EPS))
            if m.kind == 'touch':
                relevant.update(self.index.find(('entry',m.pad),m.second,24/TPS+EPS))
            normalized.append(normalize_one(m, tuple(slides[i] for i in sorted(relevant))))
        self.markers = tuple(normalized)
        self.by_id = {m.id:m for m in self.markers}
        self.groups = touch_groups(self.markers)
        active_group = {m.id for g in self.groups if not all(m.on_slide for m in g) for m in g}
        self.parts = dict(prior.parts) if append else {}
        dirty = set()
        for m in self.markers:
            part = action_events(m, m.id in active_group)
            old = self.parts.get(m.id)
            if old != part:
                for ev,sp in (old,part) if old else (part,):
                    dirty.update(e[2] for e in ev); dirty.update(s[0] for s in sp)
                self.parts[m.id] = part
        self.pad_events = dict(prior.pad_events) if append else {}
        for pad in dirty:
            self.pad_events[pad] = merged_pad(self.parts,pad)
        self.targets = tuple(sorted((m for m in self.markers if not m.slide or m.head),
                                    key=lambda m:(m.second,m.kind=='touch',m.id)))
        self.targets_by_pad = defaultdict(list)
        self.target_times = {}
        for m in self.targets: self.targets_by_pad[m.pad].append(m)
        for pad,ms in self.targets_by_pad.items(): self.target_times[pad] = tuple(m.second for m in ms)
        self.has_tap = {}
        for s in self.markers:
            if not s.slide: continue
            ts = self.target_times.get(s.pad,())
            targets = self.targets_by_pad.get(s.pad,())
            self.has_tap[s.id] = any(t.slide_head and abs(t.second-s.launch)<.5e-6
                for t in targets[bisect_left(ts,s.launch-.5e-6):bisect_right(ts,s.launch+.5e-6)])

    def static(self):
        result = []
        for s in self.markers:
            if not s.slide: continue
            ts = self.target_times.get(s.pad,())
            targets = self.targets_by_pad.get(s.pad,())
            for t in targets[bisect_left(ts,s.launch+1/TPS-1e-9):bisect_right(ts,s.launch+.2+1e-9)]:
                if t.id == s.id or t.owner == s.owner and t.owner >= 0: continue
                gap = t.second-s.launch
                if 1/TPS-1e-9 <= gap <= .2+1e-9 and t.second <= s.end+.2+1e-9:
                    result.append(Diagnostic('SlideHeadTap',t.id,s.id,
                        outer_level(t.head_protected if t.slide else t.protected,self.has_tap[s.id],abs(gap)),
                        t.second,origin='static'))
        return result

    def outer(self, resume, checkpoint_tick):
        judged, result, start = set(), [], -math.inf
        if resume:
            start, saved, diagnostics = resume
            judged, result = set(saved), list(diagnostics)
        snapshot = None
        events = heapq.merge(*(es[bisect_left(es,(start,)):] for es in self.pad_events.values()))
        event_count = 0
        expiry = 0
        membership = {m.id:i for i,g in enumerate(self.groups) for m in g}
        for tick,pad,cause,extra in events:
            event_count += 1
            if snapshot is None and tick >= checkpoint_tick:
                snapshot = (checkpoint_tick, frozenset(judged), tuple(result))
            now = tick/TPS
            affected_groups = set()
            while expiry < len(self.targets) and self.targets[expiry].second+.15+EPS < now:
                m = self.targets[expiry];expiry += 1
                judged.add(m.id)
                if m.id in membership: affected_groups.add(membership[m.id])
            targets = self.targets_by_pad.get(pad,())
            ts = self.target_times.get(pad,())
            for m in targets[bisect_left(ts,now-.15-EPS):bisect_right(ts,now+.15+EPS)]:
                if m.id in judged: continue
                if now-m.second < -.15-EPS: continue
                judged.add(m.id)
                if m.id in membership: affected_groups.add(membership[m.id])
                s = self.by_id[cause]
                if m.kind not in ('touch','touch_hold') and now < m.second-RELEASE-EPS and s.slide and (extra or pad == s.pad):
                    result.append(Diagnostic('SlideHeadTap',m.id,cause,
                        outer_level(m.head_protected if m.slide else m.protected,self.has_tap[s.id],max(0.,m.second-s.launch)),now))
                break
            for i in affected_groups:
                group = self.groups[i]
                if sum(m.id in judged for m in group) >= len(group)*.51-EPS:
                    judged.update(m.id for m in group)
        if snapshot is None:
            snapshot = (checkpoint_tick, frozenset(judged), tuple(result))
        return result,snapshot,event_count
