"""Choose the output family before evaluating its expert; no future row access."""
import torch
from maimai_baseline.dual_contract import family_support
from maimai_baseline.inference.sampling import choose


def selector(*, vocabulary, timing, tick, target_end, song_end, tail_tick, moment_count,
             current_moment, maximum_rows, temperature, generator, trace,
             state=None, moment=0, excluded_ticks=(), audio_end_seconds=None, excluded_families=()):
    def select(logits, grammar):
        allowed = grammar[0, -1].clone()
        reasons = []
        if current_moment and len(current_moment) == 1:
            reasons.append('moment-requires-note')
        elif family_support(allowed, vocabulary).sum() == 1:
            reasons.append('grammar-single-family')
        if current_moment and len(current_moment) >= maximum_rows:
            allowed[vocabulary.event_id('NOTE')] = False
            allowed[vocabulary.event_id('SLIDE_BRANCH')] = False
            reasons.append('moment-row-limit')
        from maimai_baseline.time.audio_end import duration_upper
        from maimai_baseline.time.grid import first_candidate_tick
        lower = 0 if moment_count == 0 else 1
        if first_candidate_tick(tick+lower) >= target_end or first_candidate_tick(tick+lower) > song_end:
            allowed[vocabulary.event_id('MOMENT')] = False
            reasons.append('time-boundary')
        if duration_upper('branch_duration_tick', tick, tail_tick, timing=timing,end_seconds=audio_end_seconds) < 1:
            allowed[vocabulary.event_id('SLIDE_BRANCH')] = False
        if state is not None:
            from maimai_baseline.physical.inference_state import KINDS
            from maimai_baseline.physical.local_query import moment_support
            for name in ('NOTE', 'SLIDE_BRANCH', 'MOMENT'):
                index = vocabulary.event_id(name)
                if not bool(allowed[index]):
                    continue
                if name == 'NOTE':
                    viable = bool(state.local_query(tick, moment, 'event').allowed[0])
                elif name == 'SLIDE_BRANCH':
                    viable = bool(state.local_query(tick, moment, 'event').allowed[1])
                else:
                    viable = moment_support(state, tick=tick, target_end=target_end,
                        first=moment_count == 0, moment=moment+1, excluded=excluded_ticks, exists=True)
                if not viable:
                    allowed[index] = False
                    reasons.append('physical:' + name)
        support = family_support(allowed, vocabulary)
        for family in excluded_families:
            support[family] = False
        if not bool(support.any()):
            if excluded_families:
                from maimai_baseline.physical.inference_state import PhysicalDecodeError
                raise PhysicalDecodeError('no legal router family', recoverable=True,
                                          stage='router', excluded_families=sorted(excluded_families))
            # Initial decode must still supply logits for the ultimate bypass.
            choice = int(logits[0, -1].argmax())
            detail = dict(no_legal_completion=True)
        else:
            choice, detail = choose(logits[0, -1], support, temperature=temperature,
                                    generator=generator, name='decoder_family')
        probabilities = logits[0, -1].float().softmax(-1).tolist()
        trace.update(family=('T', 'N')[choice], probability=probabilities[choice],
                     probabilities=dict(T=probabilities[0], N=probabilities[1]),
                     forced=bool(support.sum() == 1), reasons=reasons, decision=detail)
        result = logits.argmax(-1)
        result[0, -1] = choice
        return result
    return select
