"""CPU-only public tensor geometry for P0 assembly and Z window selection."""
import torch
from torch.nn.utils.rnn import pad_sequence


def prepare_p0(beats, valid, starts, mapping, coverages, *, p0_rate):
    indices = []
    for song, coverage in enumerate(coverages):
        rows = (mapping == song).nonzero().flatten()
        order = rows[torch.argsort(starts[rows])]
        chunks = []
        for row in order:
            grid, start = beats[row].float(), starts[row]
            selected = ((grid >= start) & (grid < start+coverage.target_beats)).nonzero().flatten()
            chunks.append(selected + row*beats.shape[1])
        indices.append(torch.cat(chunks))
    present = pad_sequence([torch.ones_like(i, dtype=torch.bool) for i in indices], batch_first=True)
    index = pad_sequence(indices, batch_first=True)
    return dict(indices=index, present=present, shape=tuple(beats.shape))


def prepare_selection(beats, valid, groups, starts, window_valid, *, target_beats, before_beats, after_beats):
    count = before_beats + target_beats + after_beats
    desired = starts[:, None]-before_beats + torch.arange(count).float() + .5
    sources = {}
    for group in groups.unique().tolist():
        indices = valid[group].nonzero().flatten()
        coords = beats[group, indices]
        sources[group] = (indices, coords.contiguous())
    indices, masks = [], []
    for w, group in enumerate(groups.tolist()):
        source, coords = sources[group]
        if len(source):
            index = torch.searchsorted(coords, desired[w])
            safe = index.clamp_max(len(source)-1)
            ok = (index < len(source)) & (coords[safe] == desired[w]) & window_valid[w]
            selected = source[safe] + group*beats.shape[1]
        else:
            ok = torch.zeros(count, dtype=torch.bool)
            selected = torch.zeros(count, dtype=torch.long)
        indices.append(selected); masks.append(ok)
    return dict(indices=torch.stack(indices), valid=torch.stack(masks), beats=desired, shape=tuple(beats.shape))


def to_device(layout, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k,v in layout.items()}
