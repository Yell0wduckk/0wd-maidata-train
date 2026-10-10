"""Data-only window labels. Censored tails never become chart tokens."""
from dataclasses import dataclass
import math

from .types import End, Moment
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


def window_sequence(tokens, timing, *, start_tick, end_tick, audio_after_beats, audio_end_seconds):
    if (end_tick <= start_tick or audio_after_beats < 0 or not math.isfinite(audio_after_beats)
            or not math.isfinite(audio_end_seconds)):
        raise ValueError('invalid window extent')
    cap_seconds = min(timing.second(end_tick+96*audio_after_beats), audio_end_seconds)
    if cap_seconds < timing.second(start_tick):
        raise ValueError('window starts after the audio end')
    groups, group, g = [], None, Grammar(timing)
    for token in tokens:
        g.accept(token)
        if isinstance(token, Moment):
            group = [g.state.cursor, []]
            groups.append(group)
        elif not isinstance(token, End):
            if group is None:
                raise ValueError('content before first MOMENT')
            group[1].append(token)
    g.finish()
    result, cursor, next_tick = [End(start_tick)], start_tick, None
    for tick, content in groups:
        if tick < start_tick:
            continue
        if tick >= end_tick:
            next_tick = tick
            break
        if timing.second(tick) >= cap_seconds:
            raise ValueError('real target onset is outside effective audio')
        result.append(Moment(timing.delta_ms(cursor, tick)))
        result.extend(content)
        cursor = tick
    exact = next_tick is not None and timing.second(next_tick) < cap_seconds
    target_seconds = timing.second(next_tick) if exact else cap_seconds
    terminal = TimeTarget(1000*(target_seconds-timing.second(cursor)),
                          next_tick if exact else timing.second_to_beat(cap_seconds)*96,
                          not exact, True)
    return WindowSequence(tuple(result), terminal)
