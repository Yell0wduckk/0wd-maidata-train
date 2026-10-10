"""CPU-only public tensor geometry for P0 assembly and Z window selection."""
import math
import torch
from torch.nn.utils.rnn import pad_sequence


def prepare_p0(beats, valid, starts, mapping, coverages, *, p0_rate):
    if any(v.device.type != 'cpu' for v in (beats, valid, starts, mapping)):
        raise ValueError('P0 layout must be prepared on CPU')
    if (starts.shape != (len(beats),) or mapping.shape != starts.shape
            or mapping.dtype != torch.long or not coverages
            or ((mapping < 0) | (mapping >= len(coverages))).any()
            or not math.isfinite(p0_rate) or p0_rate <= 0):
        raise ValueError('invalid P0 window ownership')
    indices = []
    for song, coverage in enumerate(coverages):
        rows = (mapping == song).nonzero().flatten()
        order = rows[torch.argsort(starts[rows])]
        if not torch.equal(starts[order], starts.new_tensor(coverage.starts(include_tail=False))):
            raise ValueError('P0 coverage must include each target center exactly once')
        chunks = []
        for row in order:
            grid, start = beats[row].float(), starts[row]
            expected = grid[:1] + torch.arange(len(grid)).float()/p0_rate
            tolerance = max(1e-5, float(grid.abs().max()) * torch.finfo(torch.float32).eps * 4)
            if (not torch.isfinite(grid).all() or not (grid[1:] > grid[:-1]).all()
                    or not torch.allclose(grid, expected, atol=tolerance, rtol=0)
                    or grid[0] - .5/p0_rate > start+tolerance
                    or grid[-1] + .5/p0_rate < start+coverage.target_beats-tolerance):
                raise ValueError('P0 coordinates must cover the target on the configured rate grid')
            selected = ((grid >= start) & (grid < start+coverage.target_beats)).nonzero().flatten()
            chunks.append(selected + row*beats.shape[1])
        indices.append(torch.cat(chunks))
    present = pad_sequence([torch.ones_like(i, dtype=torch.bool) for i in indices], batch_first=True)
    index = pad_sequence(indices, batch_first=True)
    return dict(indices=index, present=present, shape=tuple(beats.shape))


def prepare_selection(beats, valid, groups, starts, window_valid, *, target_beats, before_beats, after_beats):
    if any(v.device.type != 'cpu' for v in (beats, valid, groups, starts, window_valid)):
        raise ValueError('Z layout must be prepared on CPU')
    if (any(type(v) is not int or v < 0 for v in (target_beats, before_beats, after_beats))
            or target_beats == 0 or groups.ndim != 1 or not len(groups)
            or groups.dtype != torch.long or starts.shape != groups.shape
            or window_valid.shape != starts.shape or window_valid.dtype != torch.bool
            or not torch.isfinite(starts).all() or not torch.equal(starts, starts.round())
            or ((groups < 0) | (groups >= len(beats))).any()):
        raise ValueError('invalid latent window geometry/mapping')
    count = before_beats + target_beats + after_beats
    desired = starts[:, None]-before_beats + torch.arange(count).float() + .5
    sources = {}
    for group in groups.unique().tolist():
        indices = valid[group].nonzero().flatten()
        coords = beats[group, indices]
        if len(coords) and (not torch.equal(coords-.5, (coords-.5).round())
                or not torch.equal(coords[1:]-coords[:-1], torch.ones_like(coords[1:]))):
            raise ValueError('Z must have contiguous absolute one-beat centers')
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
        if window_valid[w] and not ok[before_beats:before_beats+target_beats].all():
            raise ValueError('valid target window is outside the full Draft coverage')
        indices.append(selected); masks.append(ok)
    return dict(indices=torch.stack(indices), valid=torch.stack(masks), beats=desired, shape=tuple(beats.shape))


def to_device(layout, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k,v in layout.items()}
