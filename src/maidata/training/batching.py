"""Complete audio-group batches; independent of networks and data loading."""
import math


def group_token_cost(group, config):
    """Charge actual E window positions/G0 beats and fixed R query capacity.

    Overlapping Encoder context is computed per window; count it each time.
    Empty/tail R windows and multiple chart conditions still consume queries.
    This is a configurable size proxy, not a claim of exact CUDA memory usage.
    """
    audio = (len(group['encoder_starts_beats']) * config['samples_per_beat'] *
             (config['audio_before_beats'] + config['target_beats'] + config['audio_after_beats'])
             + math.ceil(group['actual_audio_end_beat']) - group['first_beat'])
    rows = group['occurrences'] * config['reconstruction_query_rows']
    weights = [config['pretrain_audio_token_weight'], config['pretrain_row_token_weight']]
    if any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError('audio and row token weights must be finite and positive')
    return weights[0] * audio + weights[1] * rows


def optimizer_batches(audio_groups, config, *, token_costs=None):
    """Return optimizer steps, each holding one or more physical group batches.

    Both maximum groups and weighted token budget bound each physical batch.
    Preserve input order and each complete group. Keep a final partial batch or
    accumulation cycle; never duplicate/drop groups to fill the configured size.
    The caller merges each loss term's numerator/support across the logical
    batch before its one optimizer step, not an average of microbatch means.
    """
    size = config['pretrain_audio_groups_per_batch']
    accumulation = config['pretrain_gradient_accumulation_steps']
    if any(type(n) is not int or n < 1 for n in (size, accumulation)):
        raise ValueError('audio groups per batch and accumulation steps must be positive integers')
    groups = tuple(audio_groups)
    budget = config['pretrain_token_budget']
    if budget is not None and (isinstance(budget, bool) or not math.isfinite(budget) or budget <= 0):
        raise ValueError('token budget must be finite and positive, or null to disable')
    if budget is not None and token_costs is None:
        raise ValueError('enabled token budget requires one cost per audio group')
    costs = tuple(token_costs) if token_costs is not None else (0,) * len(groups)
    if len(costs) != len(groups) or any(not math.isfinite(c) or c < 0 for c in costs):
        raise ValueError('audio group token costs must align and be finite/nonnegative')
    batches, current, total = [], [], 0.
    for group, cost in zip(groups, costs):
        if budget is not None and cost > budget:
            raise ValueError(f'whole audio group {group!r} costs {cost:g}, above token budget {budget:g}')
        if current and (len(current) == size or (budget is not None and total + cost > budget)):
            batches.append(tuple(current))
            current, total = [], 0.
        current.append(group)
        total += cost
    if current:
        batches.append(tuple(current))
    return tuple(tuple(batches[i:i + accumulation]) for i in range(0, len(batches), accumulation))
