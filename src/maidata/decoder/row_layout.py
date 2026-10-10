"""CPU-only execution layouts; window order, positions and budgets are unchanged."""
import torch
from torch.nn.functional import pad


def window_layout(valid, condition_valid=None):
    """Compact holes in sequence order, then bucket independent windows by length.

    Causal order between real tokens is unchanged by removing invalid positions.
    Padded keys follow all real queries, so self SDPA can use is_causal directly.
    Condition keys have their own mask, including independently valid plan slots.
    """
    counts = valid.sum(-1).tolist()
    starts, offset = [], 0
    for count in counts:
        starts.append(offset)
        offset += count
    key_rows = None if condition_valid is None else [row.nonzero().flatten() for row in condition_valid]
    buckets = {}
    for window, count in enumerate(counts):
        if count:
            key_count = 1 if key_rows is None else max(1, len(key_rows[window]))
            key = ((count-1).bit_length(), (key_count-1).bit_length())
            buckets.setdefault(key, []).append(window)
    windows, gathers, key_gathers, key_masks, groups = [], [], [], [], []
    restore = torch.empty(offset, dtype=torch.long)
    query_offset = 0
    for _, members in sorted(buckets.items()):
        width = max(counts[w] for w in members)
        key_width = None if key_rows is None else max(1, max(len(key_rows[w]) for w in members))
        groups.append((len(members), width, key_width))
        for w in members:
            selected = torch.arange(starts[w], starts[w]+counts[w])
            gathers.append(pad(selected, (0, width-counts[w])))
            restore[selected] = torch.arange(counts[w])+query_offset
            query_offset += width
            windows.append(w)
            if key_rows is not None:
                keys = key_rows[w] + w*condition_valid.shape[1]
                key_gathers.append(pad(keys, (0, key_width-len(keys))))
                key_masks.append(torch.arange(key_width) < len(keys))
    cat = lambda rows: torch.cat(rows) if rows else torch.empty(0, dtype=torch.long)
    result = dict(shape=tuple(valid.shape), indices=valid.flatten().nonzero().flatten(),
        groups=tuple(groups), gather=cat(gathers), restore=restore,
        windows=torch.tensor(windows, dtype=torch.long))
    if key_rows is not None:
        result.update(key_gather=cat(key_gathers), key_valid=cat(key_masks).bool(),
                      key_indices=condition_valid.flatten().nonzero().flatten())
    return result
