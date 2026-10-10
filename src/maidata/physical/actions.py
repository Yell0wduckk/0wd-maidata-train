"""Whole presses/tracks with separate per-segment motion geometry."""
from dataclasses import dataclass

from maidata.chart.routes import chain_paths
from maidata.time import duration_end, advance_ticks, split_duration


@dataclass(frozen=True)
class Motion:
    path: str
    start: float
    end: float
    entries: tuple[tuple[float, int], ...]
    exempt: bool
    path_id: int = -1


@dataclass(frozen=True)
class Action:
    owner: int
    moment: int
    tick: int
    kind: str
    pad: str
    head: bool
    press_start: float
    press_end: float
    launch: float = -1.
    actual_end: float = -1.
    release: float = -1.
    guard_end: float = -1.
    motions: tuple[Motion, ...] = ()
    track_index: int | None = None
    handoff_release: float | None = None
    handoff_incoming: bool = False
    handoff_outgoing: bool = False

    @property
    def track(self):
        return bool(self.motions)

    @property
    def wifi(self):
        return self.kind == 'wifi'

    @property
    def touch(self):
        return self.kind in ('touch', 'touch_hold')

    @property
    def short(self):
        return self.head and self.kind in ('tap', 'touch', 'slide', 'wifi')

    @property
    def end_pad(self):
        return 'A'+self.motions[-1].path[-1] if self.track else ''

    @property
    def path_key(self):
        return tuple(m.path for m in self.motions)

    @property
    def path_order(self):
        # Keep the accepted single-path catalog tie-break order for handoffs.
        return tuple((m.path_id, m.path) for m in self.motions)

    @property
    def entries(self):
        return tuple(e for m in self.motions for e in m.entries)


def materialize(completion, timing, geometry):
    c, n = completion, completion.note
    start = timing.second(c.tick)
    head = c.track_index in (None, 0)
    press_end = timing.second(c.tick+2) if head else start
    if c.duration_tick is not None:
        press_end = advance_ticks(timing, duration_end(timing, c.tick, c.duration_tick), 22)
    if c.track is None:
        return Action(c.owner, c.moment, c.tick, n.kind, n.pad, head, start, press_end)
    paths = chain_paths(n.pad, c.track.routes)
    specs = [geometry.paths[p] for p in paths]
    launch = timing.second(c.tick+96)
    actual = duration_end(timing, c.tick, c.track.duration_tick, slide=True)
    bounds = split_duration(launch, actual, [s.length for s in specs])
    motions = tuple(Motion(s.key, a, b, tuple((a+p*(b-a), mask) for p, mask in s.entries), s.exempt, s.path_id)
                    for s, (a, b) in zip(specs, bounds))
    return Action(c.owner, c.moment, c.tick, 'wifi' if c.track.routes == (70,) else 'slide', n.pad, head, start, press_end,
                  launch, actual, advance_ticks(timing, actual, 22), timing.second(c.tick+166),
                  motions, c.track_index)
