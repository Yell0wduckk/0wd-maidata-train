"""Full-song scheduling and transactional incremental decoding of the new fields."""
from dataclasses import dataclass, replace

import torch

from maidata.chart import EncodedChart, Moment, token_to_dict
from maidata.chart.windows import TimeTarget
from maidata.decoder.batching import pack_tokens
from maidata.decoder.session import DecodeSession
from maidata.decoder.windows import history_prefix, real_onsets
from maidata.draft.windows import select_windows
from maidata.model import decoder_time_distributions
from .sampling import Sampler, OnsetLimit
from .options import GenerationOptions
from .progress import emit, stage
from .tail_collision import GenerationPhysicalState, TailRepair, generation_rules


class GenerationError(RuntimeError):
    pass


@dataclass
class SearchFrame:
    snapshot: object
    count: int
    bypass_count: int
    repair_count: int
    choices: object
    input_token: object
    routed: object
    repair_event: dict | None = None


def repetition_history(groups, limit):
    """Tail of real structural events; injected window/CONT controls are excluded."""
    tail = []
    for group in reversed(groups):
        row = ((Moment(group.delta_ms),) if group.delta_ms > 0 else ()) + group.content
        tail[:0] = row
        if len(tail) >= limit:
            break
    return tuple(tail[-limit:])


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
                  start_tick, end_tick, cap_tick, device, options, progress=None, window=1, stream=None,
                  first_token=None, stop=None, repetition_prefix=(), onset_end_tick=None):
    # Every frame binds the complete rule state to the exact causal model prefix.
    # Keeping a bounded search frontier does NOT truncate the history fed to M.
    # first_token resumes after an already-fed teacher prefix (its last real
    # token; session.cache covers every earlier expanded row). stop(accepted,
    # rules) may end the window early after a committed token.
    accepted, frames, trace, bypass, repairs = [], [], [], [], []
    backtracks = 0
    physical_fallback = None
    session.rules = generation_rules(session.rules)

    def frame(token, *, routed=None, moment_floor=None, repair_event=None):
        snapshot = session.snapshot()
        if routed is None:
            from maidata.decoder.sequence import expand_inputs
            seq=expand_inputs([token],[session.rules.grammar.state.cursor/96],timing,
                              boundary_index=0 if token is boundary else None)
            batch=pack_tokens([[seq['tokens'][-1]]],[[seq['beats'][-1]]],device=device,
                              slide_positions=[[seq['slide_beats'][-1]]])
            routed = decoder.begin(batch, memory, cache=session.cache)
        oracle = session.rules
        count = 0
        for previous in reversed(accepted):
            if isinstance(previous, Moment):
                break
            count += 1
        extra = dict(force_moment=True, moment_floor=moment_floor) if moment_floor is not None else {}
        return SearchFrame(snapshot, len(accepted), len(bypass), len(repairs), sampler.choices(decoder, routed, oracle, timing,
            distributions, end_tick=end_tick, cap_tick=cap_tick,
            moment_remaining=options.max_tokens_per_moment-count,
            history=tuple(repetition_prefix) + tuple(accepted[-options.repetition_window:]),
            onset_end_tick=onset_end_tick, **extra), token, routed, repair_event)

    emit(stream, 'window_start', window=window, start_tick=start_tick)
    frames.append(frame(boundary if first_token is None else first_token))
    while frames:
        current = frames[-1]
        count, choices = current.count, current.choices
        session.restore(current.snapshot)
        rolled_back = len(accepted) > count
        del accepted[count:]
        del bypass[current.bypass_count:]
        del repairs[current.repair_count:]
        if rolled_back:
            emit(stream, 'rollback', window=window, token_count=count)
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
                    frames[:] = [frame(current.input_token, routed=current.routed,
                        moment_floor=(current.repair_event or {}).get('minimum_tick'), repair_event=current.repair_event)]
                    continue
                raise GenerationError(f'window {start_tick/96:g}: exhausted backtracking budget ({backtracks})')
            continue
        repair_event = current.repair_event
        if isinstance(token, TailRepair):
            repair_event = token.record(session.rules.grammar.state.cursor)
            if token.replacement is not None:
                token = token.replacement
            else:
                source_index = len(frames)-1
                if session.rules.grammar.state.phase == 'note':
                    # MOMENT -> NOTE is compulsory. Replace the last unfilled
                    # onset's MOMENT, rather than appending an illegal MOMENT.
                    source_index = next((i for i in range(len(frames)-2, -1, -1)
                                         if frames[i].count == count-1), -1)
                    if source_index < 0 or not accepted or not isinstance(accepted[-1], Moment):
                        raise GenerationError('tail collision cannot restore the unfilled MOMENT prefix')
                source = frames[source_index]
                session.restore(source.snapshot)
                if len(accepted) > source.count:
                    del accepted[source.count:]
                    emit(stream, 'rollback', window=window, token_count=source.count)
                del bypass[source.bypass_count:]
                del repairs[source.repair_count:]
                del frames[source_index:]
                repair_event['minimum_tick'] = token.minimum_tick
                frames.append(frame(source.input_token, routed=source.routed,
                                    moment_floor=token.minimum_tick, repair_event=repair_event))
                continue
        if isinstance(token, OnsetLimit):
            # Do not commit the sampled out-of-range time, its CONT rows, or a
            # fabricated terminal into the rule state / reusable prefix cache.
            return tuple(accepted), dict(terminal_tick=None, stopped=True, stop_reason='onset_limit',
                backtracks=backtracks, physical_bypass=bypass, physical_fallback=physical_fallback, tail_repairs=repairs,
                **({'trace': trace} if options.trace else {}))
        if isinstance(token, TimeTarget):
            if repair_event is not None:
                repairs.append(dict(repair_event, token_index=len(accepted), replacement_type='MOMENT', advance_tick=token.absolute_tick))
            session.rules.close_window(token)
            session.cache = cache
            return tuple(accepted), dict(terminal_tick=token.absolute_tick, censored=token.censored,
                closed_by_continue=token.closed_by_continue, terminal_lower_bound_tick=token.lower_bound_tick,
                terminal_query_tick=token.query_tick,
                backtracks=backtracks, physical_bypass=bypass,physical_fallback=physical_fallback, tail_repairs=repairs,
                **({'trace': trace} if options.trace else {}))
        decision = session.rules.query(token)
        if decision.basic_errors or (sampler.physical_hard_mask and decision.physical):
            # A complete action is authoritative even after a deferred field query.
            continue
        if len(accepted) >= options.max_tokens_per_window:
            raise GenerationError(f'window {start_tick/96:g}: token limit {options.max_tokens_per_window}; generation incomplete')
        if sampler.physical_hard_mask:
            session.rules.accept(token)
        else:
            session.rules.accept_source(token)
            if decision.physical:
                bypass.append(dict(token_index=len(accepted), token=token_to_dict(token), reasons=sorted(decision.physical)))
        session.cache = cache
        if repair_event is not None:
            repairs.append(dict(repair_event, token_index=len(accepted), replacement=token_to_dict(token)))
        accepted.append(token)
        emit(stream, 'token', window=window, token=token, token_count=len(accepted),
             complete=session.rules.grammar.state.phase in ('done', 'track_done'))
        emit(progress, 'window_progress', window=window, tokens=len(accepted), backtracks=backtracks,
             cursor_beat=session.rules.grammar.state.cursor / 96)
        if options.trace:
            trace.append(dict(event='accept', token=token_to_dict(token), tick=session.rules.grammar.state.cursor))
        if stop is not None and stop(accepted, session.rules):
            return tuple(accepted), dict(terminal_tick=None, stopped=True, backtracks=backtracks,
                physical_bypass=bypass, physical_fallback=physical_fallback, tail_repairs=repairs,
                **({'trace': trace} if options.trace else {}))
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
             progress=None, on_groups=None, return_chart=True, stream=None):
    options = options or GenerationOptions()
    if (options.max_tokens_per_window < 1 or options.max_tokens_per_moment < 1
            or options.rollback_tokens < 1 or options.max_backtracks_per_window < 0):
        raise ValueError('invalid generation limits')
    if config.get('target_stride_beats', config['target_beats']) != config['target_beats']:
        raise ValueError('full-song generation requires contiguous non-overlapping target windows: stride == target_beats')
    decoder.eval()
    sampler = Sampler(temperature=options.temperature, seed=options.seed, physical_hard_mask=options.physical_hard_mask,
                      top_k=options.top_k, top_p=options.top_p, repetition_window=options.repetition_window,
                      repetition_penalty=options.repetition_penalty)
    distributions = decoder_time_distributions(config)
    session = DecodeSession(GenerationPhysicalState(timing, geometry, audio_end_seconds=duration))
    with stage(progress, 'decoder_audio'):
        audio_kv = decoder.prepare_audio(conditions.audio)
    groups, windows, repeat_prefix = [], [], ()
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
                progress=progress, window=i+1, stream=stream, repetition_prefix=repeat_prefix)
        finally:
            # The next window starts with the requested hard-mask policy again.
            sampler.physical_hard_mask = options.physical_hard_mask
        # Parse only real content with a local anchor; the previous boundary delta
        # is input-only. Rebase the first new group against the last REAL onset.
        new = list(real_onsets((Moment(0), *tokens), timing, sequence_start_tick=start_tick))
        repeat_prefix = (repeat_prefix + tokens)[-options.repetition_window:]
        if new:
            previous = groups[-1].tick if groups else min(0, new[0].tick)
            new[0] = replace(new[0], delta_ms=timing.delta_ms(previous, new[0].tick))
            groups.extend(new)
            # Only committed window content is published: search rollback cannot
            # leak abandoned candidates into the playable chart.
            if on_groups is not None:
                on_groups(tuple(new))
        emit(stream, 'window_end', window=i+1, groups=tuple(new))
        windows.append(dict(start_beat=start, end_beat=end_tick/96, history_tokens=history_count,
            boundary_delta_ms=boundary.delta_ms, generated_tokens=len(tokens), onsets=len(new),
            plan12=conditions.plans[i].float().cpu().tolist(),
            plan_valid=bool(conditions.plan_valid[i]),
            time_sampling='tick384-continue-substep-greedy' if options.temperature == 0 else 'tick384-continue-substep-sampling', **report))
        emit(progress, 'window_done', window=i+1, windows=len(conditions.starts), tokens=len(tokens),
             backtracks=report['backtracks'], cursor_beat=session.rules.grammar.state.cursor/96,
             onsets=len(new))
    return (chart_from_groups(groups, timing, metadata) if return_chart else None), windows
