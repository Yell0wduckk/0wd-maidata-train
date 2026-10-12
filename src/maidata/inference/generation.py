"""Full-song scheduling and transactional incremental decoding of the new fields."""
from dataclasses import replace

import torch

from maidata.chart import EncodedChart, Moment, token_to_dict
from maidata.chart.windows import TimeTarget
from maidata.decoder.batching import pack_tokens
from maidata.decoder.session import DecodeSession
from maidata.decoder.windows import history_prefix, real_onsets
from maidata.draft.windows import select_windows
from maidata.model import decoder_time_distributions
from maidata.physical.inference import InferencePhysicalState
from .sampling import Sampler
from .options import GenerationOptions
from .progress import emit, stage


class GenerationError(RuntimeError):
    pass


def prefill(decoder, memory, groups, timing, start_tick, history_beats, *, device):
    """Complete onset groups and their original deltas, never former control tokens."""
    history, positions, boundary, _ = history_prefix(groups, timing, start_tick=start_tick,
                                                    history_beats=history_beats, dropped=False)
    from maidata.decoder.sequence import expand_inputs, family
    from maidata.chart.types import GENERATED_TYPES
    seq=expand_inputs((*history,boundary),(*positions,start_tick/96),timing,boundary_index=len(history))
    cache=None
    if len(seq['tokens'])>1:
        batch=pack_tokens([seq['tokens'][:-1]],[seq['beats'][:-1]],device=device,
                          slide_positions=[seq['slide_beats'][:-1]])
        next_types=torch.tensor([[GENERATED_TYPES.index(family(t)) for t in seq['tokens'][1:]]],device=device)
        cache=decoder(batch,next_types,memory).cache
    return boundary,cache,len(seq['tokens'])-1


def decode_window(decoder, memory, session, boundary, sampler, distributions, timing, *,
                  start_tick, end_tick, cap_tick, device, options, progress=None, window=1):
    # Every frame binds the complete rule state to the exact causal model prefix.
    # Keeping a bounded search frontier does NOT truncate the history fed to M.
    accepted, frames, trace, bypass = [], [], [], []
    backtracks = 0
    physical_fallback = None

    def frame(token):
        snapshot = session.snapshot()
        from maidata.decoder.sequence import expand_inputs
        seq=expand_inputs([token],[session.rules.grammar.state.cursor/96],timing,
                          boundary_index=0 if token is boundary else None)
        batch=pack_tokens([[seq['tokens'][-1]]],[[seq['beats'][-1]]],device=device,
                          slide_positions=[[seq['slide_beats'][-1]]])
        routed = decoder.begin(batch, memory, cache=session.cache)
        oracle = session.rules
        if not isinstance(oracle, InferencePhysicalState):
            # Standalone callers may still supply a plain PhysicalState. Its
            # complete-action commit remains authoritative; sampling uses the
            # same optimized raw queries as full-song generation.
            oracle = InferencePhysicalState(timing, session.rules.geometry,
                audio_end_seconds=session.rules.audio_end_seconds)
            oracle.restore(session.rules.snapshot())
        count = 0
        for previous in reversed(accepted):
            if isinstance(previous, Moment):
                break
            count += 1
        return (snapshot, len(accepted), len(bypass), sampler.choices(decoder, routed, oracle, timing,
            distributions, end_tick=end_tick, cap_tick=cap_tick,
            moment_remaining=options.max_tokens_per_moment-count),token)

    frames.append(frame(boundary))
    while frames:
        snapshot, count, bypass_count, choices, input_token = frames[-1]
        session.restore(snapshot)
        del accepted[count:]
        del bypass[bypass_count:]
        try:
            token, cache = next(choices)
        except StopIteration:
            frames.pop()
            backtracks += 1
            emit(progress, 'window_progress', window=window, tokens=count, backtracks=backtracks,
                 cursor_beat=session.rules.grammar.state.cursor / 96)
            if options.trace:
                trace.append(dict(event='backtrack', tokens=count))
            if backtracks > options.max_backtracks_per_window:
                if sampler.physical_hard_mask:
                    # Retry this exact causal prefix with only raw Physical
                    # rejection relaxed. Grammar/audio-domain validity stays.
                    sampler.physical_hard_mask = False
                    physical_fallback = dict(reason='backtracking_budget_exhausted',
                        token_index=count,tick=session.rules.grammar.state.cursor,
                        phase=session.rules.grammar.state.phase,backtracks=backtracks)
                    frames[:] = [frame(input_token)]
                    continue
                raise GenerationError(f'window {start_tick/96:g}: exhausted backtracking budget ({backtracks})')
            continue
        if isinstance(token, TimeTarget):
            session.rules.close_window(token)
            session.cache = cache
            return tuple(accepted), dict(terminal_tick=token.absolute_tick, censored=token.censored,
                closed_by_continue=token.closed_by_continue, terminal_lower_bound_tick=token.lower_bound_tick,
                terminal_query_tick=token.query_tick,
                backtracks=backtracks, physical_bypass=bypass,physical_fallback=physical_fallback,
                **({'trace': trace} if options.trace else {}))
        decision = session.rules.query(token)
        if decision.basic_errors or (sampler.physical_hard_mask and decision.physical):
            # A complete action is authoritative even after a deferred field query.
            continue
        if len(accepted) >= options.max_tokens_per_window:
            raise GenerationError(f'window {start_tick/96:g}: token limit {options.max_tokens_per_window}; no chart exported')
        if sampler.physical_hard_mask:
            session.rules.accept(token)
        else:
            session.rules.accept_source(token)
            if decision.physical:
                bypass.append(dict(token_index=len(accepted), token=token_to_dict(token), reasons=sorted(decision.physical)))
        session.cache = cache
        accepted.append(token)
        emit(progress, 'window_progress', window=window, tokens=len(accepted), backtracks=backtracks,
             cursor_beat=session.rules.grammar.state.cursor / 96)
        if options.trace:
            trace.append(dict(event='accept', token=token_to_dict(token), tick=session.rules.grammar.state.cursor))
        frames.append(frame(token))
        if len(frames) > options.rollback_tokens + 1:
            del frames[0]
    raise GenerationError(f'window {start_tick/96:g}: no legal completion in retained search frontier')


def chart_from_groups(groups, timing, metadata):
    """Recompute real inter-onset deltas across boundaries and empty windows."""
    start = min(0, groups[0].tick) if groups else 0
    cursor, tokens = start, [Moment(0)]
    for group in groups:
        if group.tick != cursor:
            tokens.append(Moment(timing.delta_ms(cursor, group.tick)))
        tokens.extend(group.content)
        cursor = group.tick
    return EncodedChart(tuple(tokens), metadata, start)


@torch.inference_mode()
def generate(decoder, conditions, timing, geometry, config, metadata, duration, *, device, options=None,
             progress=None, on_groups=None, return_chart=True):
    options = options or GenerationOptions()
    if (options.max_tokens_per_window < 1 or options.max_tokens_per_moment < 1
            or options.rollback_tokens < 1 or options.max_backtracks_per_window < 0):
        raise ValueError('invalid generation limits')
    if config.get('target_stride_beats', config['target_beats']) != config['target_beats']:
        raise ValueError('full-song generation requires contiguous non-overlapping target windows: stride == target_beats')
    decoder.eval()
    sampler = Sampler(temperature=options.temperature, seed=options.seed, physical_hard_mask=options.physical_hard_mask)
    distributions = decoder_time_distributions(config)
    session = DecodeSession(InferencePhysicalState(timing, geometry, audio_end_seconds=duration))
    with stage(progress, 'decoder_audio'):
        audio_kv = decoder.prepare_audio(conditions.audio)
    groups, windows = [], []
    for i, start in enumerate(conditions.starts):
        start_tick, end_tick = int(start*96), int((start+config['target_beats'])*96)
        emit(progress, 'window_start', window=i+1, windows=len(conditions.starts),
             start_beat=start, end_beat=end_tick/96)
        cap_tick = end_tick + config['audio_after_beats']*96
        selected = select_windows(conditions.z, torch.tensor([0], device=device),
            torch.tensor([start], device=device, dtype=torch.float32), torch.tensor([True], device=device),
            target_beats=config['target_beats'], before_beats=config['decoder_z_before_beats'],
            after_beats=config['decoder_z_after_beats'],
            z_stride_beats=config['draft_z_stride_beats']).memory if config['decoder_z_cross_attention'] else conditions.z
        audio_index = i if conditions.audio_scope == 'window' else 0
        memory = decoder.prepare_memory(audio_kv, torch.tensor([audio_index], device=device), selected,
                                        conditions.plans[i:i+1], conditions.plan_valid[i:i+1])
        session.reset_attention()
        boundary, session.cache, history_count = prefill(decoder, memory, groups, timing, start_tick,
                                                        config['chart_history_beats'], device=device)
        session.rules.inject_boundary(boundary, start_tick=start_tick, history_present=True, window_end=end_tick)
        try:
            tokens, report = decode_window(decoder, memory, session, boundary, sampler, distributions, timing,
                start_tick=start_tick, end_tick=end_tick, cap_tick=cap_tick, device=device, options=options,
                progress=progress, window=i+1)
        finally:
            # The next window starts with the requested hard-mask policy again.
            sampler.physical_hard_mask = options.physical_hard_mask
        # Parse only real content with a local anchor; the previous boundary delta
        # is input-only. Rebase the first new group against the last REAL onset.
        new = list(real_onsets((Moment(0), *tokens), timing, sequence_start_tick=start_tick))
        if new:
            previous = groups[-1].tick if groups else min(0, new[0].tick)
            new[0] = replace(new[0], delta_ms=timing.delta_ms(previous, new[0].tick))
            groups.extend(new)
            # Only committed window content is published: search rollback cannot
            # leak abandoned candidates into the playable chart.
            if on_groups is not None:
                on_groups(tuple(new))
        windows.append(dict(start_beat=start, end_beat=end_tick/96, history_tokens=history_count,
            boundary_delta_ms=boundary.delta_ms, generated_tokens=len(tokens), onsets=len(new),
            plan12=conditions.plans[i].float().cpu().tolist(),
            plan_valid=bool(conditions.plan_valid[i]),
            time_sampling='tick384-continue-substep-greedy' if options.temperature == 0 else 'tick384-continue-substep-sampling', **report))
        emit(progress, 'window_done', window=i+1, windows=len(conditions.starts), tokens=len(tokens),
             backtracks=report['backtracks'], cursor_beat=session.rules.grammar.state.cursor/96,
             onsets=len(new))
    return (chart_from_groups(groups, timing, metadata) if return_chart else None), windows
