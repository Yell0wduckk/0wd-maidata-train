"""Per-prefix interval panels and category-only Physical inference checks.

This follows the old local inference preparation, using the new action and
chain identities.  Training's minimal conflict witnesses are deliberately not
constructed here; full resource matching decides the same overload category.
"""
from .geometry import Geometry, pad_bit
from .rules import EPS, handoff_releases


def _overloaded(presses, tracks):
    ordinary = [a for a in presses if not a.touch]
    touches = [a for a in presses if a.touch]
    demand = len(ordinary) + bool(touches) + sum(2 if a.wifi else 1 for a in tracks)
    if demand <= 2:
        return False
    # Branches count as separate track resources, but launch matching grants
    # at most one shared-head credit per owner, as conflicts._demand does.
    keys = {(a.owner, a.track_index, a.tick): a for a in tracks}
    edges = {('press', a.owner, a.tick, a.pad): tuple(sorted({t.owner for t in keys.values()
             if a.short and a.pad == t.pad and abs(a.press_start-t.launch) <= EPS}))
             for a in ordinary}
    if touches:
        edges[('touch_pool',)] = tuple(sorted({t.owner for t in keys.values()
            if any(a.short and a.pad == t.pad and abs(a.press_start-t.launch) <= EPS for a in touches)}))
    matched = {}

    def augment(key, seen):
        for track in sorted(edges[key]):
            if track in seen:
                continue
            seen.add(track)
            if track not in matched or augment(matched[track], seen):
                matched[track] = key
                return True
        return False

    return demand - sum(augment(key, set()) for key in edges) > 2


class InferenceIntervalContext:
    """Bounded by one immutable action prefix, shared by all local field queries."""
    def __init__(self, history):
        self.history = history
        self.geometry = {}
        self.occupancy = {}
        self.handoffs = {}
        self.panel_cache = {}
        points = sorted({p for a in history for p in
                         (a.press_start, a.press_end, *((a.launch, a.release) if a.track else ()))})
        points = [float('-inf'), *points, float('inf')]
        self.raw_panels = []
        for lo, hi in zip(points, points[1:]):
            if hi <= lo+EPS:
                continue
            presses = tuple(a for a in history if a.press_start <= lo+EPS and a.press_end > lo+EPS)
            tracks = tuple(a for a in history if a.track and a.launch <= lo+EPS and a.release > lo+EPS)
            self.raw_panels.append((lo, hi, presses, tracks))
        self.track_releases = handoff_releases(history)
        self.panels = self._panels(self.track_releases)

    def _panels(self, releases):
        key = tuple((id(a), releases[id(a)]) for a in self.history if a.track)
        if key not in self.panel_cache:
            panels = []
            for left, right, presses, raw_tracks in self.raw_panels:
                cuts = sorted({left, right, *(release for _, release in key if left < release < right)})
                for lo, hi in zip(cuts, cuts[1:]):
                    if hi > lo+EPS:
                        tracks = tuple(a for a in raw_tracks if releases[id(a)] > lo+EPS)
                        panels.append((lo, hi, presses, tracks))
            self.panel_cache[key] = tuple(panels)
        return self.panel_cache[key]

    def overloaded(self, candidate):
        if candidate in self.occupancy:
            return self.occupancy[candidate]
        c = candidate
        spans = [(c.press_start, c.press_end)] + ([(c.launch, c.release)] if c.track else [])
        points = [c.press_start, c.press_end] + ([c.launch, c.release] if c.track else [])
        releases, candidate_release = self.track_releases, c.release
        if c.track:
            future = any(a.track and a.launch > c.launch+EPS for a in self.history)
            key = (c.owner, c.pad, c.end_pad, c.path_order, c.wifi, c.launch,
                   c.actual_end if future else None, c.release if future else None,
                   c.handoff_incoming, c.handoff_outgoing)
            if key not in self.handoffs:
                full = handoff_releases((*self.history, c))
                self.handoffs[key] = ({id(a): full[id(a)] for a in self.history if a.track}, full[id(c)])
            previous, stored_release = self.handoffs[key]
            candidate_release = stored_release if future else c.release
            releases = previous
            points.append(candidate_release)
        panels = self._panels(releases) if c.track else self.panels
        for left, right, presses, tracks in panels:
            cuts = sorted({left, right, *(p for p in points if left < p < right)})
            for lo, hi in zip(cuts, cuts[1:]):
                if hi <= lo+EPS or not any(lo < end-EPS and hi > start+EPS for start, end in spans):
                    continue
                press = c.press_start <= lo+EPS and c.press_end > lo+EPS
                track = c.track and c.launch <= lo+EPS and candidate_release > lo+EPS
                if _overloaded((*presses, c) if press else presses, (*tracks, c) if track else tracks):
                    self.occupancy[candidate] = True
                    return True
        self.occupancy[candidate] = False
        return False

    def _geometry(self, c, *, touch_repeat):
        result, t, bit = set(), c.press_start, pad_bit(c.pad)
        launches_here = any(a.track and a.pad == c.pad and abs(a.launch-t) <= EPS for a in self.history)
        for a in self.history:
            if c.head:
                if a.press_start <= t+EPS and t < a.press_end-EPS and pad_bit(a.pad) & bit:
                    result.add('overlap')
                if c.touch and touch_repeat and a.touch and a.pad == c.pad and -EPS <= t-a.press_start <= .25+EPS:
                    result.add('overlap')
                if a.track and not c.touch:
                    if a.launch-EPS <= t <= a.guard_end+EPS and a.pad == c.pad and not launches_here:
                        result.add('tap_on_slide')
                    for motion in a.motions:
                        for entered, mask in motion.entries:
                            if entered < t-EPS and t <= entered+.2+EPS and mask & bit:
                                if not (launches_here and a.pad == c.pad and abs(entered-a.launch) <= EPS):
                                    result.add('tap_on_slide')
            if c.track and a.track:
                if c.pad == a.pad and c.path_key == a.path_key and a.press_start-EPS <= t <= a.launch+EPS:
                    result.add('slide_too_fast')
                for old in a.motions:
                    for new_i, new in enumerate(c.motions):
                        appearance = t if new_i == 0 else new.start
                        if (not old.exempt and not new.exempt and Geometry.covers(old.path, new.path)
                                and old.start <= new.start+EPS and old.end >= appearance-EPS):
                            result.add('slide_too_fast')
        return frozenset(result)

    def reasons(self, c, *, touch_repeat=True):
        # Later concatenated segments move when duration changes. Their actual
        # starts must be in the key, unlike a single-route launch-only cache.
        key = (c.head, c.touch, c.pad, c.press_start, c.launch, c.path_key,
               tuple((m.start, m.exempt) for m in c.motions), bool(touch_repeat and c.touch))
        if key not in self.geometry:
            self.geometry[key] = self._geometry(c, touch_repeat=touch_repeat)
        result = set(self.geometry[key])
        if c.track and c.track_index:
            own = [a for a in self.history if a.track and a.owner == c.owner]
            if c.track_index >= 2 or c.wifi or any(a.wifi for a in own):
                result.add('multitouch')
        if self.overloaded(c):
            result.add('multitouch')
        return frozenset(result)
