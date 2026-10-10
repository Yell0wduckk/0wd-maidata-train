"""Incremental token grammar and stable token-to-action ownership."""
from dataclasses import dataclass, replace
from .types import (ChartError, Moment, Note, Route, HoldDuration,
                    SlideDuration, Track, Completion)
from .routes import chain_paths


@dataclass(frozen=True)
class GrammarSnapshot:
    phase: str = 'start'
    cursor: int = 0
    last_real_tick: int | None = None
    moment: int = -1
    owner: int = -1
    note: Note | None = None
    routes: tuple[int, ...] = ()
    track_break: int | None = None
    track_count: int = 0
    token_count: int = 0
    window_end: int | None = None


class Grammar:
    def __init__(self, timing):
        self.timing = timing
        self.state = GrammarSnapshot()

    def snapshot(self):
        return self.state

    def restore(self, snapshot):
        if not isinstance(snapshot, GrammarSnapshot):
            raise TypeError('expected GrammarSnapshot')
        self.state = snapshot

    def inject_boundary(self, token, *, start_tick, history_present=True, window_end=None):
        """External input only; its role is runtime state, never a delta test."""
        s = self.state
        if not isinstance(token, Moment) or type(start_tick) is not int:
            raise ChartError('boundary requires MOMENT and an absolute integer anchor')
        if s.phase not in ('start', 'closed', 'done', 'track_done', 'boundary'):
            raise ChartError('cannot inject boundary during incomplete content')
        if s.last_real_tick is not None and start_tick <= s.last_real_tick:
            raise ChartError('new window must start after the last real onset')
        expected = self.timing.delta_ms(s.last_real_tick, start_tick) if history_present and s.last_real_tick is not None else 0.
        if abs(token.delta_ms-expected) > 1e-6:
            raise ChartError('boundary delta disagrees with last real onset/history visibility')
        if window_end is not None and window_end <= start_tick:
            raise ChartError('empty/reversed target range')
        self.state = replace(s, phase='boundary', cursor=start_tick, note=None,
                             routes=(), track_count=0, track_break=None, window_end=window_end,
                             token_count=s.token_count+1)

    def allowed_types(self):
        s = self.state
        return {
            'start': (), 'closed': (), 'boundary': ('NOTE', 'MOMENT'),
            'note': ('NOTE',), 'hold': ('HOLD_DURATION',), 'route': ('ROUTE',),
            'chain': ('SLIDE_DURATION',) if s.routes == (70,) else ('ROUTE', 'SLIDE_DURATION'),
            'done': ('NOTE', 'MOMENT'),
            'track_done': ('ROUTE', 'NOTE', 'MOMENT'),
        }[s.phase]

    def accept(self, token, *, window_end=None):
        s = self.state
        if token.type not in self.allowed_types():
            raise ChartError(f'{token.type} is illegal after {s.phase}')
        updates = {'token_count': s.token_count+1}
        completion = None
        if isinstance(token, Moment):
            tick = self.timing.tick_after(s.cursor, token.delta_ms)
            if token.delta_ms <= 0 or tick <= s.cursor:
                raise ChartError('generated MOMENT must advance by at least one tick')
            if s.window_end is not None and tick >= s.window_end:
                updates.update(phase='closed', note=None, routes=())
            else:
                updates.update(phase='note', cursor=tick, note=None)
        elif isinstance(token, Note):
            if s.phase in ('boundary', 'note'):
                updates.update(moment=s.moment+1, last_real_tick=s.cursor)
            updates.update(note=token, owner=s.owner+1, routes=(), track_count=0, track_break=None)
            if token.kind in ('hold', 'touch_hold'):
                updates['phase'] = 'hold'
            elif token.kind == 'slide':
                updates['phase'] = 'route'
            else:
                updates['phase'] = 'done'
                completion = Completion(s.owner+1, updates.get('moment', s.moment), s.cursor, token, token_index=s.token_count)
        elif isinstance(token, Route):
            first = s.phase in ('route', 'track_done')
            if (token.track_break is None) == first:
                raise ChartError('only the first ROUTE of a track carries track_break')
            if token.route_id == 70 and not first:
                raise ChartError('WiFi must be the sole segment of its track')
            routes = (token.route_id,) if first else s.routes+(token.route_id,)
            chain_paths(s.note.pad, routes)
            updates.update(phase='chain', routes=routes,
                           track_break=token.track_break if first else s.track_break)
        elif isinstance(token, HoldDuration):
            completion = Completion(s.owner, s.moment, s.cursor, s.note,
                                    duration_tick=token.ticks, token_index=s.token_count)
            updates['phase'] = 'done'
        elif isinstance(token, SlideDuration):
            track = Track(s.routes, s.track_break, token.ticks)
            completion = Completion(s.owner, s.moment, s.cursor, s.note, track=track,
                                    track_index=s.track_count, token_index=s.token_count)
            updates.update(phase='track_done', track_count=s.track_count+1)
        else:
            raise ChartError('unknown token class')
        self.state = replace(s, **updates)
        return completion

    def finish(self):
        if self.state.phase not in ('done', 'track_done', 'boundary', 'closed'):
            raise ChartError(f'incomplete chart: {self.state.phase}')

    def close_window(self, target):
        """Consume a terminal label without inventing a grid-aligned onset."""
        from .windows import TimeTarget
        s = self.state
        if not isinstance(target, TimeTarget) or not target.closes_window:
            raise ChartError('expected a terminal TimeTarget')
        if s.phase not in ('done', 'track_done', 'boundary'):
            raise ChartError('cannot close a window with an incomplete note')
        if (not target.censored and s.window_end is not None and target.absolute_tick < s.window_end
                or target.absolute_tick < s.cursor):
            raise ChartError('terminal target is inside the target or before the cursor')
        expected = self.timing.delta_ms(s.cursor, target.absolute_tick)
        if abs(expected-target.delta_ms) > 1e-6:
            raise ChartError('terminal time coordinates disagree')
        self.state = replace(s, phase='closed', note=None, routes=())
