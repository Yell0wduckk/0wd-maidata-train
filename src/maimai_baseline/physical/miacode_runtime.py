"""Generated-prefix slide judgments, ported from the pinned MiaCode runtime.

This module deliberately has no torch, chart-cache, parser, or executable dependency.
The resource manifest contains the upstream MIT notice and resource provenance.
All times here are seconds; judgment ticks are 180 Hz, never Physical slots.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import numpy as np

EPS = 1e-6
TPS = 180
RELEASE = 3 / TPS
PADS = tuple(f"{r}{i}" for r in "ABDE" for i in range(1, 9)) + ("C",)


def pad_center(pad):
    if pad == "C":
        return np.array([270., 270.])
    ring, lane = pad[0], int(pad[1:])
    angle = math.radians((-90 if ring in "DE" else -67.5) + (lane - 1) * 45)
    distance = {"A": 205., "B": 112.75, "D": 210.125, "E": 158.875}[ring]
    return np.array([270 + math.cos(angle) * distance, 270 + math.sin(angle) * distance])


CENTERS = np.array([pad_center(p) for p in PADS])
RADII = np.array([{"A": 50., "B": 37.5, "C": 52.5, "D": 32.5, "E": 30.}[p[0]] for p in PADS])


@lru_cache(maxsize=1)
def resources():
    root = Path(__file__).resolve().parents[1] / "assets"
    data = (root / "miacode-slide-runtime.json").read_bytes()
    manifest = json.loads((root / "miacode-slide-runtime-manifest.json").read_text("utf-8"))
    if hashlib.sha256(data).hexdigest() != manifest["sha256"]:
        raise ValueError("MiaCode slide resource hash mismatch")
    return json.loads(data), {k: manifest[k] for k in ("schema", "upstream_head", "sha256")}


@dataclass(frozen=True)
class Marker:
    id: int
    kind: str
    pad: str
    second: float
    end: float
    launch: float = 0.
    path: str = ""
    owner: int = -1
    head: bool = True
    protected: bool = False
    head_protected: bool = False
    slide_head: bool = False
    tail_on_head: bool = False
    on_slide: bool = False
    before_slide: bool = False
    after_slide: bool = False
    moment: int | None = None

    @property
    def slide(self):
        return self.kind in ("slide", "wifi")


@dataclass(frozen=True)
class Diagnostic:
    rule: str
    affected: int
    cause: int
    severity: str
    second: float
    deviation: float = 0.
    origin: str = "runtime"

    @property
    def key(self):
        return self.rule, self.affected, self.cause


def entry(marker):
    data, _ = resources()
    return data["wifi" if marker.kind == "wifi" else "slides"][marker.path]


def adjacent(a, b):
    if a == b:
        return False
    ga, gb = "ABDEC".index(a[0]), "ABDEC".index(b[0])
    ia, ib = int(a[1:])-1 if a != "C" else 0, int(b[1:])-1 if b != "C" else 0
    if ga > gb:
        ga, gb, ia, ib = gb, ga, ib, ia
    if gb == 4:
        return ga == 1
    if (ga, gb) in ((0, 0), (2, 2), (3, 3), (1, 2)):
        return False
    if ga == gb == 1:
        return ib in ((ia+1)%8, (ia+7)%8)
    if (ga, gb) in ((0, 1), (2, 3)):
        return ia == ib
    return ib in (ia, (ia+1)%8)


def touch_groups(markers):
    pending = [m for m in markers if m.kind == "touch"]
    groups = []
    while pending:
        group = [pending.pop(0)]
        changed = True
        while changed:
            changed = False
            for m in pending[:]:
                if any(m.second == n.second and m.moment == n.moment and adjacent(m.pad, n.pad) for n in group):
                    group.append(m)
                    pending.remove(m)
                    changed = True
        if len(group) > 1:
            groups.append(group)
    return groups


def normalize(markers, *, slides=None):
    """Parser's causal-context flags; recomputed when a candidate changes context."""
    slides = [m for m in markers if m.slide] if slides is None else slides
    result = []
    for m in markers:
        head = any(s.pad == m.pad and abs(m.second - s.launch) < 1 / TPS for s in slides)
        tail = any(s.pad == m.pad and abs(m.end - s.launch) < 1 / TPS for s in slides)
        before = any(s.id != m.id and s.pad == f"A{entry(m)['end']}" and abs(m.end-s.launch) < 1/TPS
                     for s in slides) if m.slide else False
        after = any(s.id != m.id and f"A{entry(s)['end']}" == m.pad and abs(m.launch-s.end) < 1/TPS
                    for s in slides) if m.slide else False
        on = False
        if m.kind == "touch":
            for s in slides:
                if s.pad == m.pad and -24/TPS < m.second-s.launch < (24/TPS if s.kind == "wifi" else 1/TPS):
                    on = True
                for e in entry(s)["pad_enter_times"]:
                    if e["pad"] == m.pad and abs(m.second-(s.launch+e["t"]*(s.end-s.launch))) < 24/TPS:
                        on = True
        result.append(replace(m, slide_head=head, tail_on_head=tail, on_slide=on,
                              before_slide=before, after_slide=after))
    return tuple(result)


@lru_cache(maxsize=1024)
def path_arrays(path_key, wifi):
    data, _ = resources()
    e = data["wifi" if wifi else "slides"][path_key]
    paths = e["di_real_path_samples"] if wifi else [e["real_path_samples"]]
    return tuple(np.array([[p["x"]+270, p["y"]+270] for p in path]) for path in paths)


@lru_cache(maxsize=4096)
def motion_samples(marker):
    """Per-action vectorized 180 Hz pad coverage, cached without model data."""
    if marker.slide:
        e = entry(marker)
        paths = e["di_real_path_samples"] if marker.kind == "wifi" else [e["real_path_samples"]]
        start, duration = marker.launch, max(0., marker.end-marker.launch)
        end = marker.end + (0. if marker.kind == "wifi" or marker.before_slide else RELEASE)
        radius = 50. if marker.kind == "wifi" else 15.
    else:
        if marker.kind == "tap" and marker.slide_head or marker.kind == "touch" and marker.on_slide:
            return ()
        start, duration = marker.second, 0.
        end = marker.end + (0. if marker.tail_on_head and marker.kind in ("hold", "touch_hold") else RELEASE)
        paths = [[dict(zip(("x", "y"), pad_center(marker.pad)-270))]]
        radius = 15.
    lo = max(0, math.ceil((start-EPS)*TPS))
    hi = math.ceil((end-EPS)*TPS)
    ticks = np.arange(lo, hi, dtype=np.int64)
    if not len(ticks):
        return ()
    result = []
    arrays = path_arrays(marker.path, marker.kind == "wifi") if marker.slide else (np.array([[p["x"]+270, p["y"]+270] for p in paths[0]]),)
    for points in arrays:
        progress = np.clip((ticks/TPS-start)/duration, 0, 1) if duration > EPS else np.ones(len(ticks))
        scaled = progress * (len(points)-1)
        left = np.floor(scaled).astype(int)
        right = np.minimum(left+1, len(points)-1)
        delta = points[right]-points[left]
        centers = points[left] + delta * (scaled-left)[:, None]
        length = np.linalg.norm(delta, axis=1)
        tangents = delta / np.maximum(length[:, None], 1e-300)
        coverage = np.linalg.norm(centers[:, None, :]-CENTERS[None, :, :], axis=2) <= radius+RADII+EPS
        masks = (coverage.astype(np.uint64) * (np.uint64(1) << np.arange(33, dtype=np.uint64))).sum(axis=1)
        result.append((lo, hi, centers, tangents, length > 1e-12, masks))
    return tuple(result)


class SlideJudge:
    def __init__(self, marker):
        self.marker = marker
        e = entry(marker)
        self.wifi = marker.kind == "wifi"
        self.sequences = e["tri_judge_sequence"] if self.wifi else [e["judge_sequence"]]
        self.indices = [0]*len(self.sequences)
        self.pressing = [None]*len(self.sequences)
        self.hits = [[None]*len(s) for s in self.sequences]
        self.done = False
        self.action = None
        self.second = None
        self.is_l, self.special = e.get("is_l", False), e.get("is_special_l", False)
        tail = (1-e["critical_proportion"])*max(0., marker.end-marker.launch)
        self.critical = marker.end-tail
        self.width = min(86400., 42/TPS+max(0., tail)/4)

    def can_skip(self, lane):
        i, sequence = self.indices[lane], self.sequences[lane]
        if i >= len(sequence)-1:
            return False
        if self.wifi or self.pressing[lane] is not None:
            return True
        if self.special:
            return i not in (1, 3)
        if self.is_l:
            return i != 1
        return len(sequence) >= 4 or i != len(sequence)-2

    def update(self, tick, down, up):
        now = tick/TPS
        if self.done or now+EPS < self.marker.second-15/TPS:
            return
        for lane, sequence in enumerate(self.sequences):
            while self.indices[lane] < len(sequence):
                i = self.indices[lane]
                pressing = self.pressing[lane]
                if pressing is None:
                    pad = next((p for p in sequence[i] if p in down), None)
                    if pad is not None:
                        self.pressing[lane] = pad
                        self.hits[lane][i] = dict(tick=tick, skipped=False, cause=down[pad])
                        if i == len(sequence)-1:
                            self.indices[lane] += 1
                            self.action = down[pad]
                        continue
                elif pressing not in down:
                    self.pressing[lane] = None
                    self.indices[lane] += 1
                    continue
                if self.can_skip(lane):
                    pad = next((p for p in sequence[i+1] if p in down or p in up), None)
                    if pad is not None:
                        if self.hits[lane][i] is None and not self.wifi:
                            self.hits[lane][i] = dict(tick=tick, skipped=True, cause=None)
                        self.indices[lane] += 1
                        i += 1
                        self.pressing[lane] = pad
                        self.hits[lane][i] = dict(tick=tick, skipped=False, cause=down.get(pad, up.get(pad)))
                        if i == len(sequence)-1:
                            self.indices[lane] += 1
                            self.action = down.get(pad)
                        continue
                break
        if all(i == len(s) for i, s in zip(self.indices, self.sequences)):
            self.done, self.second = True, now

    def diagnostic(self):
        if not self.done:
            return None
        cause = self.action if self.wifi else self.hits[0][-1]["cause"]
        if cause is None or cause == self.marker.id:
            return None
        delta = self.second-self.critical
        deviation = abs(delta)-self.width
        # Inference v2 uses only the symmetric critical window. The upstream
        # shifted acceptance window is independent of the preserved launch guard.
        if deviation <= EPS:
            return None
        return Diagnostic("SlideTooFast", self.marker.id, cause, "Muri", self.second, deviation)


def outer_level(protected, has_tap, gap):
    return "Warning" if protected or (not has_tap and 1e-9 < gap <= .05+1e-9) or gap+1e-9 >= .15 else "Muri"


def outer_references(markers):
    """MiaCode static SlideHeadTap references, including head-star targets."""
    slides = [m for m in markers if m.slide]
    targets = [m for m in markers if m.kind in ("tap", "hold") or m.slide and m.head]
    diagnostics = []
    for s in slides:
        has_tap = any(t.pad == s.pad and t.slide_head and abs(t.second-s.launch) < .5e-6 for t in targets)
        for t in targets:
            if t.id == s.id or t.owner == s.owner and t.owner >= 0:
                continue
            gap = t.second-s.launch
            if t.pad == s.pad and 1/TPS-1e-9 <= gap <= .2+1e-9 and t.second <= s.end+.2+1e-9:
                diagnostics.append(Diagnostic("SlideHeadTap", t.id, s.id,
                    outer_level(t.head_protected if t.slide else t.protected, has_tap, abs(gap)),
                    t.second, origin="static"))
    return diagnostics


def outer_runtime(markers):
    """Interval pad-down timeline (distinct from the finger-motion timeline)."""
    events, intervals, targets = {}, {}, []
    by_id = {m.id: m for m in markers}
    slides = [m for m in markers if m.slide]
    groups = touch_groups(markers)
    active_group_ids = {m.id for group in groups if not all(m.on_slide for m in group) for m in group}
    def put(tick, pad, cause, extra=False, force=False):
        key = tick, pad
        if force or key not in events or events[key][0] <= cause:
            events[key] = cause, extra
    def press(m, end):
        lo, hi = math.ceil(m.second*TPS-EPS), math.ceil(end*TPS-EPS)-1
        if hi >= lo:
            intervals.setdefault(m.pad, []).append((lo, hi, m.id))
    for m in markers:
        if m.slide:
            e = entry(m)
            if not m.after_slide:
                first = e["pad_enter_times"][0]["t"]*(m.end-m.launch) if e["pad_enter_times"] else .05
                put(math.floor((m.launch+min(first, .05))*TPS+EPS)+1, m.pad, m.id, True)
    for m in markers:
        if m.slide:
            put(math.ceil(m.launch*TPS-EPS), m.pad, m.id)
    for m in markers:
        if m.slide:
            for item in entry(m)["pad_enter_times"]:
                put(math.ceil((m.launch+item["t"]*(m.end-m.launch))*TPS-EPS), item["pad"], m.id)
            if m.head:
                targets.append(m)
                press(m, m.second+RELEASE)
        else:
            targets.append(m)
            if not (m.kind == "tap" and m.slide_head or m.kind == "touch" and m.on_slide and m.id not in active_group_ids):
                press(m, m.end+(0 if m.tail_on_head and m.kind in ("hold", "touch_hold") else RELEASE))
    for pad, spans in intervals.items():
        changes = sorted(set(t for lo, hi, _ in spans for t in (lo, hi+1)))
        had = False
        for t in changes:
            active = [cause for lo, hi, cause in spans if lo <= t <= hi]
            if active and not had:
                put(t, pad, max(active), force=True)
            had = bool(active)
    targets.sort(key=lambda m: (m.second, m.kind == "touch", m.id))
    judged, result = set(), []
    for (tick, pad), (cause, extra) in sorted(events.items()):
        now = tick/TPS
        for m in targets:
            if m.id in judged or m.second+.15+EPS < now:
                judged.add(m.id)
                continue
            if m.pad != pad or now-m.second < -.15-EPS:
                continue
            judged.add(m.id)
            s = by_id[cause]
            if m.kind not in ("touch", "touch_hold") and now < m.second-RELEASE-EPS and s.slide and (extra or pad == s.pad):
                has_tap = any(t.pad == s.pad and t.slide_head and abs(t.second-s.launch) < .5e-6 for t in targets)
                result.append(Diagnostic("SlideHeadTap", m.id, cause,
                    outer_level(m.head_protected if m.slide else m.protected, has_tap, max(0., m.second-s.launch)), now))
            break
        for group in groups:
            if sum(m.id in judged for m in group) >= len(group)*.51-EPS:
                judged.update(m.id for m in group)
    return result


def _replay(markers, resume=None, checkpoint_tick=None, *, prepared=False, outer_result=None):
    """Full-prefix reference with cached vectorized action projection.

    Replaying from zero is the safe boundary when parser flags can change earlier
    actions. No history-context or arbitrary look-back truncation is used.
    """
    markers = tuple(markers) if prepared else normalize(markers)
    judges = [SlideJudge(m) for m in markers if m.slide]
    diagnostics = outer_references(markers) + outer_runtime(markers) if outer_result is None else list(outer_result)
    if not judges:
        return diagnostics, [], None
    # Synthetic stars are tap-like hand actions unless aligned to a launch;
    # attribute them to their owner so a slide cannot foreign-judge itself.
    stars = [replace(m, kind="tap", end=m.second, path="") for m in markers if m.slide and m.head]
    groups = touch_groups(markers)
    grouped = {m.id for group in groups for m in group}
    # Upstream emits individual presses, then connected Touch groups, then slides.
    group_actions = [replace(m, on_slide=False) for group in groups
                     if not all(m.on_slide for m in group) for m in group]
    actions = tuple(m for m in markers if not m.slide and m.id not in grouped) + tuple(stars) + tuple(group_actions) + tuple(m for m in markers if m.slide)
    motions = [(m.launch if m.slide else m.second, m, sample)
               for m in actions
               for sample in motion_samples(m)]
    motions.sort(key=lambda item: item[0])
    end = max((x[2][1] for x in motions), default=0)
    previous = {}
    start_tick = min((x[2][0] for x in motions), default=0)
    if resume is not None:
        start_tick, old_previous, old_saved = resume
        previous = dict(old_previous)
        saved = [j if j.done else deepcopy(j) for j in old_saved]
        saved = {j.marker.id: j for j in saved}
        judges = [saved[j.marker.id] if j.marker.id in saved and saved[j.marker.id].marker == j.marker else j for j in judges]
    snapshot = None
    active, pointer = [], 0
    pending_judges = sorted((j for j in judges if not j.done), key=lambda j:j.marker.second)
    live_judges, judge_pointer = [], 0
    for tick in range(start_tick, end):
        if tick == checkpoint_tick:
            snapshot = (tick, dict(previous), tuple(j if j.done else deepcopy(j) for j in judges))
        while pointer < len(motions) and motions[pointer][2][0] <= tick:
            active.append(motions[pointer])
            pointer += 1
        active = [a for a in active if a[2][1] > tick]
        down, accepted = {}, []
        for _, m, sample in active:
            lo, hi, centers, tangents, has_tangent, masks = sample
            if not lo <= tick < hi:
                continue
            j = tick-lo
            if m.slide:
                merged = any(other.kind == m.kind and ht and has_tangent[j]
                    and np.linalg.norm(center-centers[j]) < 10
                    and math.acos(float(np.clip(np.dot(tangent, tangents[j]), -1, 1))) < .05235389661574631
                    for other, center, tangent, ht in accepted)
                if merged:
                    continue
                accepted.append((m, centers[j], tangents[j], has_tangent[j]))
            mask = int(masks[j])
            while mask:
                bit = mask & -mask
                down[PADS[bit.bit_length()-1]] = m.id
                mask ^= bit
        up = {p: c for p, c in previous.items() if p not in down}
        while judge_pointer < len(pending_judges) and tick/TPS+EPS >= pending_judges[judge_pointer].marker.second-15/TPS:
            live_judges.append(pending_judges[judge_pointer]);judge_pointer += 1
        for judge in live_judges:
            judge.update(tick, down, up)
        live_judges = [j for j in live_judges if not j.done]
        previous = down
    diagnostics.extend(d for j in judges if (d := j.diagnostic()) is not None)
    return diagnostics, [dict(id=j.marker.id, second=j.second, critical=j.critical,
                              width=j.width, hits=j.hits) for j in judges], snapshot


def replay(markers):
    """Independent full-prefix entry point, never uses a saved judge state."""
    diagnostics, states, _ = _replay(markers)
    return diagnostics, states


def incremental_replay(markers, previous=None):
    """Resume before every changed action's earliest possible influence.

    Keep one pre-head checkpoint, not N copies of a full-song judge matrix.
    Parser-flag changes on an older action move the safe boundary backward and
    can force full replay. Already generated future tails are always replayed.
    """
    from maimai_baseline.physical.miacode_cache import Scene
    scene = Scene(tuple(markers), previous[2] if previous and len(previous)>2 else None)
    normalized = scene.markers
    old = {m.id: m for m in previous[0]} if previous else {}
    now = {m.id:m for m in normalized}
    changed = {i for i in old.keys() | now.keys() if old.get(i) != now.get(i)}
    earliest = min((m.second-.2 for i in changed for m in (old.get(i),now.get(i)) if m is not None), default=-math.inf)
    resume = previous[1] if previous and previous[1] and previous[1][0]/TPS < earliest-EPS else None
    checkpoint_tick = max(0, math.floor((max((m.second for m in normalized), default=0)-.2)*TPS))
    outer_resume = previous[3] if previous and len(previous)>3 and previous[3] and previous[3][0]/TPS < earliest-EPS else None
    outer, outer_checkpoint, event_count = scene.outer(outer_resume, checkpoint_tick)
    diagnostics, states, checkpoint = _replay(normalized, resume, checkpoint_tick,
        prepared=True, outer_result=scene.static()+outer)
    return diagnostics, states, (normalized, checkpoint, scene, outer_checkpoint), dict(
        resumed_from_tick=resume[0] if resume else None, checkpoint_tick=checkpoint[0] if checkpoint else None,
        outer_events=event_count, history_actions=len(markers),
        replay_ticks=max(0, math.ceil(max((m.end+RELEASE for m in normalized),default=0)*TPS)-(resume[0] if resume else 0)))


def new_violations(before, after):
    old = {d.key: d for d in before if d.severity == "Muri"}
    return [d for d in after if d.severity == "Muri" and
            (d.key not in old or d.rule == "SlideTooFast" and d.deviation > old[d.key].deviation+EPS)]


@lru_cache(maxsize=128)
def _component(markers):
    return replay(markers)


def analyze(markers):
    """Reuse only quiescent boundaries, including the backward judge window.

    Connected interaction intervals may span the whole song. In that case this
    deliberately uses full replay, not an unsafe duration/history cap.
    """
    groups, current, right = [], [], -math.inf
    for m in sorted(markers, key=lambda m: (m.second, m.id)):
        if current and m.second-.2 > right:
            groups.append(tuple(current))
            current = []
        current.append(m)
        right = max(right, m.end+.2)
    if current:
        groups.append(tuple(current))
    diagnostics, states = [], []
    for group in groups:
        ds, ss = _component(group)
        diagnostics.extend(ds)
        states.extend(ss)
    return diagnostics, states
