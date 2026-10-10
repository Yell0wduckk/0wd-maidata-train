"""Complete audio-group batches; independent of networks and data loading."""


def optimizer_batches(audio_groups, config):
    """Return optimizer steps, each holding one or more physical group batches.

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
    span = size * accumulation
    return tuple(tuple(groups[i:min(i + size, len(groups))]
                       for i in range(start, min(start + span, len(groups)), size))
                 for start in range(0, len(groups), span))
