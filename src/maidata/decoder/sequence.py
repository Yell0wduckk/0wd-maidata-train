"""Model-only time tokens. Real chart tokens and Physical actions stay exact."""
from dataclasses import dataclass

from maidata.chart.types import Moment, HoldDuration, SlideDuration, GENERATED_TYPES
from maidata.time import duration_end
from maidata.time.discrete import encode_ticks, TimeQuery, CONTINUE

CONT_TYPES = ('MOMENT_CONT', 'HOLD_CONT', 'SLIDE_CONT')
INPUT_TYPES = GENERATED_TYPES + CONT_TYPES
TIME_TYPES = ('MOMENT', 'HOLD_DURATION', 'SLIDE_DURATION')


@dataclass(frozen=True)
class ModelTime:
    type: str
    time_value: float


def family(token):
    return TIME_TYPES[CONT_TYPES.index(token.type)] if token.type in CONT_TYPES else token.type


def continue_token(kind, timing, origin, elapsed):
    # Value encodes the emitted block, while RoPE encodes the known new cursor.
    value = (timing.delta_ms(origin + elapsed - 384, origin + elapsed) / 1000
             if kind == 'MOMENT' else 4.)
    return ModelTime(CONT_TYPES[TIME_TYPES.index(kind)], value)


def expand_inputs(tokens, beats, timing, *, boundary_index=None):
    """Return input tokens, self coordinates, Slide Q coordinates, old-row map.

    Only the designated injected boundary is exempt from time decomposition.
    Final duration tokens reset to the real onset; CONT never advances it.
    """
    result, positions, slide_positions, last_rows, source_rows, elapsed_rows = [], [], [], [], [], []
    for i, (token, beat) in enumerate(zip(tokens, beats, strict=True)):
        onset = round(beat * 96)
        launch = timing.second_to_beat(duration_end(timing, onset, 0, slide=True))
        parts = [(token, beat, launch, 0)]
        if i != boundary_index and isinstance(token, (Moment, HoldDuration, SlideDuration)):
            kind = token.type
            origin = (round(timing.second_to_beat(timing.second(onset)-token.delta_ms/1000)*96)
                      if isinstance(token, Moment) else onset)
            ticks = onset-origin if isinstance(token, Moment) else token.ticks
            if ticks > 0:
                parts = []
                elapsed = 0
                for choice in encode_ticks(ticks):
                    if choice == CONTINUE:
                        elapsed += 384
                        position = TimeQuery(kind, origin, elapsed).query_beat(timing)
                        parts.append((continue_token(kind,timing,origin,elapsed), position, position, elapsed))
                    else:
                        value = (timing.delta_ms(origin+elapsed,origin+ticks)/1000
                                 if kind == 'MOMENT' else (choice+1)/96)
                        parts.append((ModelTime(kind,value),beat,launch,elapsed))
        for item, position, slide, elapsed in parts:
            result.append(item); positions.append(position); slide_positions.append(slide)
            source_rows.append(i); elapsed_rows.append(elapsed)
        last_rows.append(len(result)-1)
    return dict(tokens=tuple(result), beats=tuple(positions), slide_beats=tuple(slide_positions),
                last_rows=tuple(last_rows), source_rows=tuple(source_rows), elapsed=tuple(elapsed_rows))


def model_window(example, timing):
    """Expand causal inputs, including nonfinal terminal-MOMENT CONT steps."""
    from maidata.time.discrete import compile_time_target
    seq=expand_inputs(example.inputs,example.beats,timing,boundary_index=example.boundary_index)
    tokens,beats,slides=map(list,(seq['tokens'],seq['beats'],seq['slide_beats']))
    target=example.targets[-1]
    origin=round(example.beats[-1]*96)
    args=({'censored_lower':target.absolute_tick-origin} if target.censored else
          {'ticks':int(target.absolute_tick-origin)})
    path=compile_time_target('MOMENT',origin_tick=origin,timing=timing,window_end_tick=example.end_tick,**args)
    for depth in range(1,len(path.steps)):
        elapsed=384*depth
        tokens.append(continue_token('MOMENT',timing,origin,elapsed))
        beat=TimeQuery('MOMENT',origin,elapsed).query_beat(timing)
        beats.append(beat);slides.append(beat)
    seq.update(tokens=tuple(tokens),beats=tuple(beats),slide_beats=tuple(slides))
    seq['next_types']=tuple(GENERATED_TYPES.index(family(t)) for t in tokens[1:])+(0,)
    return seq
