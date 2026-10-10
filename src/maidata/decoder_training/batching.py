"""Contiguous Decoder chunks charged by actual padded input length, never R rows."""


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
