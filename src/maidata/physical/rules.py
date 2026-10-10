"""Port of the accepted interval rules; chains are one hand with many motions."""
from collections import defaultdict
from dataclasses import replace

EPS = 1e-10
CATEGORIES = ('multitouch', 'overlap', 'slide_too_fast', 'tap_on_slide')
CATEGORY_WEIGHTS = (1, 1, 4, 1)
CATEGORY_DIVISOR = 4


def history_end(a):
    return max(a.press_end, a.release, a.guard_end,
               a.press_start+.25 if a.touch else -1.,
               max((s+.2 for s, _ in a.entries), default=-1.))


def handoff_matches(actions):
    tracks = sorted((a for a in actions if a.track and not a.wifi),
                    key=lambda a: (a.launch, a.pad, a.end_pad, a.owner, a.path_order))
    waiting, pairs = defaultdict(list), []
    for following in tracks:
        pool = waiting[following.pad]
        pool[:] = [a for a in pool if a.release >= following.launch-EPS]
        options = [] if following.handoff_incoming else [a for a in pool
                   if a.actual_end <= following.launch+EPS and a.launch < following.launch-EPS]
        if options:
            previous = min(options, key=lambda a: (a.actual_end, a.launch, a.owner, a.path_order))
            pool.remove(previous)
            pairs.append((previous, following))
        if not following.handoff_outgoing:
            waiting[following.end_pad].append(following)
    return tuple(pairs)


def handoff_releases(actions):
    releases = {id(a): a.release if a.handoff_release is None else a.handoff_release for a in actions if a.track}
    for previous, following in handoff_matches(actions):
        releases[id(previous)] = min(previous.release, following.launch)
    return releases


def prune_history(history, second):
    incoming, outgoing = set(), {}
    for previous, following in handoff_matches(history):
        if following.launch <= second+EPS:
            incoming.add(id(following))
            outgoing[id(previous)] = following.launch
    return tuple(replace(a, handoff_incoming=a.handoff_incoming or id(a) in incoming,
                         handoff_outgoing=a.handoff_outgoing or id(a) in outgoing,
                         handoff_release=min(a.release, outgoing[id(a)]) if id(a) in outgoing else a.handoff_release)
                 if id(a) in incoming or id(a) in outgoing else a
                 for a in history if history_end(a) >= second-EPS)


def hand_segments(actions):
    releases = handoff_releases(actions)
    points = sorted({p for a in actions for p in
                     (a.press_start, a.press_end, *((a.launch, a.release) if a.track else ()))})
    result = []
    for lo, hi in zip(points, points[1:]):
        if hi <= lo+EPS:
            continue
        presses = [a for a in actions if a.press_start <= lo+EPS and a.press_end > lo+EPS]
        groups = defaultdict(list)
        for a in actions:
            if a.track and a.launch <= lo+EPS and releases[id(a)] > lo+EPS:
                groups[a.owner].append(a)
        touches = [a for a in presses if a.touch]
        ordinary = [a for a in presses if not a.touch]
        demand = len(ordinary)+bool(touches)+sum(2 if a.wifi else 1 for g in groups.values() for a in g)
        edges = {i: [g for g, tracks in groups.items() if any(a.pad == t.pad and abs(a.press_start-t.launch) <= EPS for t in tracks)]
                 for i, a in enumerate(ordinary) if a.short}
        if touches:
            edges['touch-pool'] = sorted({g for a in touches if a.short for g, tracks in groups.items()
                                         if any(a.pad == t.pad and abs(a.press_start-t.launch) <= EPS for t in tracks)})
        matched = {}
        def augment(i, seen):
            for group in edges[i]:
                if group in seen:
                    continue
                seen.add(group)
                if group not in matched or augment(matched[group], seen):
                    matched[group] = i
                    return True
            return False
        demand -= sum(augment(i, set()) for i in edges)
        if result and result[-1][2] == demand and abs(result[-1][1]-lo) <= EPS:
            result[-1] = result[-1][0], hi, demand
        else:
            result.append((lo, hi, demand))
    return tuple(result)


def reasons(history, c, *, touch_repeat=True, check_same_path=True):
    from .conflicts import action_conflicts
    return frozenset(w.category for w in action_conflicts(
        history, c, touch_repeat=touch_repeat, check_same_path=check_same_path))
