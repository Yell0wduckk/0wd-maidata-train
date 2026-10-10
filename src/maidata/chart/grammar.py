"""Incremental token grammar and stable token-to-action ownership."""
from dataclasses import dataclass, replace
from .types import (ChartError, End, Moment, Note, Route, HoldDuration,
                    SlideDuration, Track, Completion)
from .routes import chain_paths


@dataclass(frozen=True)
class GrammarSnapshot:
    phase: str = 'start'
    cursor: int = 0
    first_query: bool = True
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

    def allowed_types(self):
        s = self.state
        return {
            'start': ('END',), 'closed': ('END',), 'query': ('MOMENT',),
            'note': ('NOTE',), 'hold': ('HOLD_DURATION',), 'route': ('ROUTE',),
            'chain': ('SLIDE_DURATION',) if s.note and s.note.kind == 'wifi' else ('ROUTE', 'SLIDE_DURATION'),
            'done': ('NOTE', 'MOMENT', 'END'),
            'track_done': ('NOTE', 'MOMENT', 'END') if s.note and s.note.kind == 'wifi' else ('ROUTE', 'NOTE', 'MOMENT', 'END'),
        }[s.phase]

    def accept(self, token, *, window_end=None):
        s = self.state
        if token.type not in self.allowed_types():
            raise ChartError(f'{token.type} is illegal after {s.phase}')
        updates = {'token_count': s.token_count+1}
        completion = None
        if isinstance(token, End):
            if s.last_real_tick is not None and token.start_tick <= s.last_real_tick:
                raise ChartError('new window must start after the last real onset')
            if window_end is not None and window_end <= token.start_tick:
                raise ChartError('empty/reversed target range')
            updates.update(phase='query', cursor=token.start_tick, first_query=True,
                           note=None, routes=(), track_count=0, track_break=None, window_end=window_end)
        elif isinstance(token, Moment):
            tick = self.timing.tick_after(s.cursor, token.delta_ms)
            if tick < s.cursor or (not s.first_query and tick == s.cursor):
                raise ChartError('real MOMENT must advance')
            if s.window_end is not None and tick >= s.window_end:
                updates.update(phase='closed', note=None, routes=())
            else:
                updates.update(phase='note', cursor=tick, last_real_tick=tick,
                               first_query=False, moment=s.moment+1, note=None)
        elif isinstance(token, Note):
            updates.update(note=token, owner=s.owner+1, routes=(), track_count=0, track_break=None)
            if token.kind in ('hold', 'touch_hold'):
                updates['phase'] = 'hold'
            elif token.kind in ('slide', 'wifi'):
                updates['phase'] = 'route'
            else:
                updates['phase'] = 'done'
                completion = Completion(s.owner+1, s.moment, s.cursor, token, token_index=s.token_count)
        elif isinstance(token, Route):
            first = s.phase in ('route', 'track_done')
            if (token.track_break is None) == first:
                raise ChartError('only the first ROUTE of a track carries track_break')
            if (token.route_id == 70) != (s.note.kind == 'wifi'):
                raise ChartError('WiFi and ordinary routes cannot be mixed')
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
        if self.state.phase not in ('done', 'track_done', 'query', 'closed'):
            raise ChartError(f'incomplete chart: {self.state.phase}')

    def close_window(self, target):
        """Consume a terminal label without inventing a grid-aligned onset."""
        from .windows import TimeTarget
        s = self.state
        if not isinstance(target, TimeTarget) or not target.closes_window:
            raise ChartError('expected a terminal TimeTarget')
        if s.phase not in ('done', 'track_done', 'query'):
            raise ChartError('cannot close a window with an incomplete note')
        if (not target.censored and s.window_end is not None and target.absolute_tick < s.window_end
                or target.absolute_tick < s.cursor):
            raise ChartError('terminal target is inside the target or before the cursor')
        expected = self.timing.delta_ms(s.cursor, target.absolute_tick)
        if abs(expected-target.delta_ms) > 1e-6:
            raise ChartError('terminal time coordinates disagree')
        self.state = replace(s, phase='closed', note=None, routes=())
