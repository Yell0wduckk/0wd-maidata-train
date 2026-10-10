"""Data-only window labels. Censored tails never become chart tokens."""
from dataclasses import dataclass
import math

from .types import Moment, Note
from .grammar import Grammar


@dataclass(frozen=True)
class TimeTarget:
    delta_ms: float
    absolute_tick: float
    censored: bool
    closes_window: bool

    def __post_init__(self):
        if (not math.isfinite(self.delta_ms) or self.delta_ms < 0
                or not math.isfinite(self.absolute_tick)
                or type(self.censored) is not bool or type(self.closes_window) is not bool):
            raise ValueError('invalid time target')


@dataclass(frozen=True)
class WindowSequence:
    tokens: tuple
    terminal: TimeTarget
    start_tick: int
    history_last_real_tick: int | None


def window_sequence(tokens, timing, *, start_tick, end_tick, audio_after_beats, audio_end_seconds, sequence_start_tick=0, history_present=True):
    if (end_tick <= start_tick or audio_after_beats < 0 or not math.isfinite(audio_after_beats)
            or not math.isfinite(audio_end_seconds)):
        raise ValueError('invalid window extent')
    cap_seconds = timing.second(end_tick+96*audio_after_beats)
    groups, group, g = [], None, Grammar(timing)
    g.inject_boundary(tokens[0], start_tick=sequence_start_tick)
    for token in tokens[1:]:
        old_moment = g.state.moment
        g.accept(token)
        if isinstance(token, Note) and g.state.moment != old_moment:
            group = [g.state.cursor, []]
            groups.append(group)
        if not isinstance(token, Moment):
            if group is None:
                raise ValueError('content before first NOTE')
            group[1].append(token)
    g.finish()
    previous = next((tick for tick, _ in reversed(groups) if tick < start_tick), None)
    delta = timing.delta_ms(previous, start_tick) if history_present and previous is not None else 0.
    result, cursor, next_tick = [Moment(delta)], start_tick, None
    for tick, content in groups:
        if tick < start_tick:
            continue
        if tick >= end_tick:
            next_tick = tick
            break
        if timing.second(tick) >= audio_end_seconds:
            raise ValueError('real target onset is outside effective audio')
        if tick != cursor:
            result.append(Moment(timing.delta_ms(cursor, tick)))
        result.extend(content)
        cursor = tick
    exact = next_tick is not None and timing.second(next_tick) < cap_seconds
    target_seconds = timing.second(next_tick) if exact else cap_seconds
    terminal = TimeTarget(1000*(target_seconds-timing.second(cursor)),
                          next_tick if exact else timing.second_to_beat(cap_seconds)*96,
                          not exact, True)
    return WindowSequence(tuple(result), terminal, start_tick, previous if history_present else None)
