"""Random-window batch budgets and contiguous M computation chunks; no models."""


def batch_cost(audio_lengths, input_lengths, config):
    """Only padded Decoder rows constrain stage II; audio is reported separately."""
    audio = len(audio_lengths) * max(audio_lengths, default=0)
    chart = len(input_lengths) * max(input_lengths, default=0)
    return dict(audio_tokens=audio, chart_tokens=chart, row_tokens=chart,
                oversized=chart > config['decoder_train_row_budget'])


def occurrence_batches(rows, audio_lengths, input_lengths, config):
    """Greedy contiguous packing of the shuffled stream, without fill/reorder."""
    batches, current, audio, lengths = [], [], {}, []
    for row, length in zip(rows, input_lengths, strict=True):
        variant = row['audio_variant']
        candidate = dict(audio, **{variant: audio_lengths[variant]})
        if current and batch_cost(list(candidate.values()), lengths + [length], config)['oversized']:
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
    chunks, current, length = [], [], 0
    for i, row in enumerate(windows):
        size = row['length']
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
