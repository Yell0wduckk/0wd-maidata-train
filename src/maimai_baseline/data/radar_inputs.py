"""Map immutable radar labels to existing dense windows without changing the plan."""
import torch
from maimai_baseline.radar_contract import DIMENSION


def dense_inputs(cache, items, group_windows):
    lengths = [len(indices) for indices in group_windows]
    if not lengths or any(n == 0 for n in lengths):
        raise ValueError("radar input requires nonempty chart/view groups")
    values = torch.zeros((len(lengths), max(lengths), DIMENSION), dtype=torch.float32)
    valid = torch.zeros(values.shape[:2], dtype=torch.bool)
    group = torch.empty(len(items), dtype=torch.long)
    index = torch.empty_like(group)
    for g, indices in enumerate(group_windows):
        for j, w in enumerate(indices):
            item = items[w]
            cached = cache.segments[cache.lookup[(str(item['segment_id']),
                                                str(item['spatial_view_id']))]]
            if (int(item['start_tick']) != int(cached['target_start_tick'])
                    or int(item['end_tick']) != int(cached['target_end_tick'])):
                raise ValueError("MajRadar target window coordinates changed")
            values[g, j] = cache.get(str(item['segment_id']), item['spatial_view_id'])
            valid[g, j] = True
            group[w], index[w] = g, j
    return dict(values=values, valid=valid, window_group=group, window_index=index)
