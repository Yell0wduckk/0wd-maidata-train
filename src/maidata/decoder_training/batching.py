"""Random-window batch budgets and contiguous M computation chunks; no models."""


def joint_cost(audio_lengths, input_lengths, config):
    """Charge the allocated padded P0 bank once per distinct audio variant."""
    audio = len(audio_lengths) * max(audio_lengths, default=0)
    chart = len(input_lengths) * max(input_lengths, default=0)
    total = (audio * config['decoder_train_audio_token_weight']
             + chart * config['decoder_train_chart_token_weight'])
    return dict(audio_tokens=audio, chart_tokens=chart, weighted_tokens=total,
                oversized=total > config['decoder_train_token_budget'])


def occurrence_batches(rows, audio_lengths, input_lengths, config):
    """Greedy contiguous packing of the shuffled stream, without fill/reorder."""
    batches, current, audio, lengths = [], [], {}, []
    for row, length in zip(rows, input_lengths, strict=True):
        variant = row['audio_variant']
        candidate = dict(audio, **{variant: audio_lengths[variant]})
        if current and joint_cost(list(candidate.values()), lengths + [length], config)['oversized']:
            batches.append(current)
            current, audio, lengths = [], {}, []
        current.append(row)
        audio[variant] = audio_lengths[variant]
        lengths.append(length)
    if current:
        batches.append(current)
    # An indivisible oversized window is kept whole and reported, never cropped.
    accumulate = config['decoder_train_gradient_accumulation_steps']
    return [batches[i:i+accumulate] for i in range(0, len(batches), accumulate)]


def window_chunks(windows, token_budget):
    if type(token_budget) is not int or token_budget < 1:
        raise ValueError('Decoder token budget must be a positive integer')
    chunks, current, length = [], [], 0
    for i, row in enumerate(windows):
        size = len(row['example'].inputs)
        padded = max(length, size) * (len(current)+1)
        if current and padded > token_budget:
            chunks.append(current)
            current, length = [], 0
        current.append(i)
        length = max(length, size)
    if current:
        chunks.append(current)
    # A single large window is indivisible: keep its complete history and target.
    return chunks
