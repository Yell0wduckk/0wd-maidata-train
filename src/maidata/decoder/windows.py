"""GT/generated event history and window teacher forcing, without model or Physical."""
from dataclasses import dataclass
import math
from maidata.chart.grammar import Grammar
from maidata.chart.types import Moment, Note, GENERATED_TYPES
from maidata.chart.windows import window_sequence


@dataclass(frozen=True)
class Onset:
    tick: int
    delta_ms: float
    content: tuple


def real_onsets(tokens, timing, *, sequence_start_tick):
    """Read a full real chart once; window controls must not be stored in this stream."""
    grammar = Grammar(timing)
    grammar.inject_boundary(tokens[0], start_tick=sequence_start_tick)
    groups, previous, current = [], sequence_start_tick, None
    for token in tokens[1:]:
        old = grammar.state.moment
        grammar.accept(token)
        if isinstance(token, Note) and old != grammar.state.moment:
            tick = grammar.state.cursor
            current = [tick, timing.delta_ms(previous, tick), []]
            groups.append(current)
            previous = tick
        if not isinstance(token, Moment):
            current[2].append(token)
    grammar.finish()
    return tuple(Onset(t, d, tuple(c)) for t, d, c in groups)


@dataclass(frozen=True)
class WindowExample:
    inputs: tuple
    beats: tuple
    targets: tuple
    next_types: tuple
    supervised: tuple
    boundary_index: int
    start_tick: int
    end_tick: int
    cap_tick: float
    last_real_tick: int | None
    history_dropped: bool


def history_prefix(groups, timing, *, start_tick, history_beats, dropped):
    """Shared inference/training cropping; last onset is independent of token crop."""
    if not math.isfinite(history_beats) or history_beats < 0:
        raise ValueError('invalid history range')
    earlier = [g for g in groups if g.tick < start_tick]
    last = earlier[-1].tick if earlier else None
    tokens, beats = [], []
    if not dropped:
        for g in earlier:
            if g.tick >= start_tick - 96 * history_beats:
                row = (Moment(g.delta_ms), *g.content)
                tokens.extend(row)
                beats.extend([g.tick / 96] * len(row))
    boundary = Moment(0. if dropped or last is None else timing.delta_ms(last, start_tick))
    return tuple(tokens), tuple(beats), boundary, last


def build_window(tokens, timing, *, sequence_start_tick, start_tick, end_tick,
                 audio_after_beats, audio_end_seconds, history_beats, history_dropout,
                 training, rng):
    if not 0 <= history_dropout <= 1:
        raise ValueError('invalid history_dropout')
    dropped = training and rng.random() < history_dropout
    groups = real_onsets(tokens, timing, sequence_start_tick=sequence_start_tick)
    history, positions, boundary, last = history_prefix(groups, timing, start_tick=start_tick,
                                                      history_beats=history_beats, dropped=dropped)
    window = window_sequence(tokens, timing, start_tick=start_tick, end_tick=end_tick,
        audio_after_beats=audio_after_beats, audio_end_seconds=audio_end_seconds,
        sequence_start_tick=sequence_start_tick, history_present=not dropped)
    assert window.tokens[0] == boundary
    cursor = start_tick
    current_beats = [start_tick / 96]
    for token in window.tokens[1:]:
        if isinstance(token, Moment):
            cursor = timing.tick_after(cursor, token.delta_ms)
        current_beats.append(cursor / 96)
    inputs = history + window.tokens
    targets = inputs[1:] + (window.terminal,)
    next_types = tuple(GENERATED_TYPES.index(t.type) if hasattr(t, 'type') else 0 for t in targets)
    return WindowExample(inputs, positions + tuple(current_beats), targets, next_types,
        tuple(i >= len(history) for i in range(len(inputs))), len(history), start_tick, end_tick,
        end_tick + 96 * audio_after_beats, last, dropped)


def window_from_config(tokens, timing, config, *, sequence_start_tick, start_tick,
                       audio_end_seconds, training, rng):
    return build_window(tokens, timing, sequence_start_tick=sequence_start_tick, start_tick=start_tick,
        end_tick=start_tick + 96 * config['target_beats'], audio_after_beats=config['audio_after_beats'],
        audio_end_seconds=audio_end_seconds, history_beats=config['chart_history_beats'],
        history_dropout=config['decoder_history_dropout'], training=training, rng=rng)
