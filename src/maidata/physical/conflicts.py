"""Rule witnesses with stable action identities, independent of model/loss state."""
from dataclasses import dataclass
from itertools import combinations

from .geometry import Geometry, pad_bit
from .rules import EPS, handoff_releases


def press_key(a):
    return ('press', a.owner, a.tick, a.pad)


def track_key(a):
    return ('track', a.owner, a.track_index, a.tick)


@dataclass(frozen=True, order=True)
class Conflict:
    category: str
    rule: str
    participants: tuple
    detail: tuple = ()

    def record(self):
        return [self.category, self.rule, [list(p) for p in self.participants], list(self.detail)]

    @classmethod
    def from_record(cls, row):
        return cls(row[0], row[1], tuple(tuple(p) for p in row[2]), tuple(row[3]))


def branch_conflicts(actions, owner, index, tick, *, wifi=False):
    own = [a for a in actions if a.track and a.owner == owner]
    if not index:
        return frozenset()
    new = ('track', owner, index, tick)
    result = set()
    if index >= 2:
        result.add(Conflict('multitouch', 'branch_limit', (new,)))
    for old in own:
        if wifi or old.wifi:
            result.add(Conflict('multitouch', 'wifi_branch', (new, track_key(old))))
    return frozenset(result)


def first_route_coverage(actions, grammar, timing, geometry):
    """Duration-independent first-segment witnesses, identical to action_conflicts.

    Only segment zero appears at the NOTE and launches one beat later. Later
    segments depend on the eventual total duration and remain action checks.
    """
    path = geometry.path(grammar.note.pad, grammar.routes[0])
    if path.exempt:
        return frozenset()
    appearance = timing.second(grammar.cursor)
    launch = timing.second(grammar.cursor + 96)
    new = ('track', grammar.owner, grammar.track_count, grammar.cursor)
    return frozenset(
        Conflict('slide_too_fast', 'path_coverage', (new, track_key(action)), (0, old_i))
        for action in actions for old_i, old in enumerate(action.motions)
        if (not old.exempt and Geometry.covers(old.path, path.key)
            and old.start <= launch + EPS and old.end >= appearance - EPS))


def slide_launch_capacity(actions, grammar, timing, *, wifi=False):
    """Mandatory occupancy only; no guessed route or future teacher duration.

    One written tick plus the global 22-tick release is a subset of every
    possible track. Unknown endpoints cannot grant outgoing handoff credits.
    Incoming handoffs depend only on the known start pad and launch time.
    """
    from .actions import Action, Motion
    from maidata.time import advance_ticks
    launch = timing.second(grammar.cursor + 96)
    end = launch + timing.written_tick_seconds(grammar.cursor)
    release = advance_ticks(timing, end, 22)
    start = timing.second(grammar.cursor)
    head = grammar.track_count == 0
    candidate = Action(grammar.owner, grammar.moment, grammar.cursor,
        'wifi' if wifi else 'slide', grammar.note.pad, head, start,
        timing.second(grammar.cursor + 2) if head else start,
        launch, end, release, motions=(Motion(grammar.note.pad[-1]+'-'+grammar.note.pad[-1],
        launch, end, (), False),), track_index=grammar.track_count, handoff_outgoing=True)
    spans = [(launch, release)]
    return (capacity_conflicts((*actions, candidate), spans), capacity_conflicts(actions, spans))


def _demand(resources):
    presses = [r for r in resources if r[0][0] != 'track']
    tracks = [r for r in resources if r[0][0] == 'track']
    demand = len(presses) + sum(r[1] for r in tracks)
    track_keys = {r[0] for r in tracks}
    edges = {r[0]: {key[1] for key in r[2] & track_keys} for r in presses}
    matched = {}

    def augment(key, seen):
        for group in sorted(edges[key]):
            if group in seen:
                continue
            seen.add(group)
            if group not in matched or augment(matched[group], seen):
                matched[group] = key
                return True
        return False

    return demand - sum(augment(r[0], set()) for r in presses)


def capacity_conflicts(actions, spans):
    """Minimal overloaded resource sets; existing GT overloads retain their IDs.

    A minimal witness contains at most two presses and two tracks: three of
    either already overload two hands. Touches are one pool, not many presses.
    Launch matching is recomputed for each resource subset, as in hand_segments.
    """
    releases = handoff_releases(actions)
    points = sorted({p for a in actions for p in
                     (a.press_start, a.press_end, *((a.launch, a.release) if a.track else ()))})
    result = set()
    for lo, hi in zip(points, points[1:]):
        if hi <= lo+EPS or not any(lo < end-EPS and hi > start+EPS for start, end in spans):
            continue
        presses = [a for a in actions if a.press_start <= lo+EPS and a.press_end > lo+EPS]
        tracks = [a for a in actions if a.track and a.launch <= lo+EPS and releases[id(a)] > lo+EPS]
        resources = [(track_key(a), 2 if a.wifi else 1, frozenset()) for a in tracks]
        touches = [a for a in presses if a.touch]
        for a in (a for a in presses if not a.touch):
            edges = frozenset(track_key(t) for t in tracks if a.short and a.pad == t.pad and abs(a.press_start-t.launch) <= EPS)
            resources.append((press_key(a), 1, edges))
        if touches:
            edges = frozenset(track_key(t) for a in touches for t in tracks
                              if a.short and a.pad == t.pad and abs(a.press_start-t.launch) <= EPS)
            resources.append((('touch_pool',), 1, edges))
        if _demand(resources) <= 2:
            continue
        cores = []
        for width in range(1, min(4, len(resources))+1):
            for subset in combinations(resources, width):
                keys = frozenset(r[0] for r in subset)
                if any(core <= keys for core in cores) or _demand(subset) <= 2:
                    continue
                cores.append(keys)
                result.add(Conflict('multitouch', 'capacity', tuple(sorted(keys))))
    return frozenset(result)


def action_conflicts(history, c, *, touch_repeat=True, check_same_path=True):
    result = set()
    t, bit = c.press_start, pad_bit(c.pad)
    launches_here = any(a.track and a.pad == c.pad and abs(a.launch-t) <= EPS for a in history)
    for a in history:
        if c.head:
            pair = (press_key(c), press_key(a))
            if a.press_start <= t+EPS and t < a.press_end-EPS and pad_bit(a.pad) & bit:
                result.add(Conflict('overlap', 'press_overlap', pair))
            if c.touch and touch_repeat and a.touch and a.pad == c.pad and -EPS <= t-a.press_start <= .25+EPS:
                result.add(Conflict('overlap', 'touch_repeat', pair))
            if a.track and not c.touch:
                pair = (press_key(c), track_key(a))
                if a.launch-EPS <= t <= a.guard_end+EPS and a.pad == c.pad and not launches_here:
                    result.add(Conflict('tap_on_slide', 'launch_guard', pair))
                for segment, motion in enumerate(a.motions):
                    for entry, (entered, mask) in enumerate(motion.entries):
                        if entered < t-EPS and t <= entered+.2+EPS and mask & bit:
                            if not (launches_here and a.pad == c.pad and abs(entered-a.launch) <= EPS):
                                result.add(Conflict('tap_on_slide', 'path_guard', pair, (segment, entry)))
        if c.track and a.track:
            pair = (track_key(c), track_key(a))
            if c.pad == a.pad and c.path_key == a.path_key and a.press_start-EPS <= t <= a.launch+EPS:
                result.add(Conflict('slide_too_fast', 'same_path', pair))
            for old_i, old in enumerate(a.motions):
                for new_i, new in enumerate(c.motions):
                    appearance = t if new_i == 0 else new.start
                    if (check_same_path and not old.exempt and not new.exempt
                            and Geometry.covers(old.path, new.path)
                            and old.start <= new.start+EPS and old.end >= appearance-EPS):
                        result.add(Conflict('slide_too_fast', 'path_coverage', pair, (new_i, old_i)))
    branch = branch_conflicts(history, c.owner, c.track_index, c.tick, wifi=c.wifi) if c.track else frozenset()
    result.update(branch)
    spans = [(c.press_start, c.press_end)] + ([(c.launch, c.release)] if c.track else [])
    for conflict in capacity_conflicts((*history, c), spans):
        # A branch restriction and its inevitable same-owner overload are one
        # violation. External resource conflicts remain separate witnesses.
        same_owner_tracks = all(p[0] == 'track' and p[1] == c.owner for p in conflict.participants)
        if not (branch and same_owner_tracks):
            result.add(conflict)
    return frozenset(result)
