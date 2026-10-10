"""Ephemeral CPU geometry, independent of learned values and model ownership.

Like the original rectangular backend, prepare indices before H2D and reuse
them across blocks. No learned representation is cached between updates.
"""
import torch
from torch.nn.utils.rnn import pad_sequence


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(to_device(v, device) for v in value)
    return value


def prepare_local(beats, valid, radius, query_width=32):
    """Group neighboring queries into rectangles; never build a full L x L mask."""
    length = beats.shape[1]
    width = min(query_width, length)
    queries, keys, query_valid, owners = [], [], [], []
    for row in range(len(beats)):
        source = valid[row].nonzero().flatten()
        coordinates = beats[row, source].float().contiguous()
        if not len(source):
            continue
        for start in range(0, length, width):
            q = torch.arange(start, min(start + width, length))
            qv = valid[row, q]
            if not qv.any():
                continue
            positions = beats[row, q[qv]].float()
            # Pooling can round differently on CPU/CUDA. This only widens the
            # candidate rectangle; the exact device beat test decides edges.
            margin = torch.finfo(torch.float32).eps * (positions.abs() + radius + 1) * 4
            lo = int(torch.searchsorted(coordinates, positions - radius - margin).min())
            hi = int(torch.searchsorted(coordinates, positions + radius + margin, right=True).max())
            queries.append(torch.nn.functional.pad(q, (0, width-len(q))))
            query_valid.append(torch.nn.functional.pad(qv, (0, width-len(qv))))
            keys.append(source[lo:hi] + row * length)
            owners.append(row)
    if not owners:
        return dict(query=torch.empty((0, width), dtype=torch.long),
                    key=torch.empty((0, 1), dtype=torch.long),
                    key_valid=torch.empty((0, 1), dtype=torch.bool),
                    valid=torch.empty((0, width), dtype=torch.bool),
                    batch=torch.empty(0, dtype=torch.long),
                    selected=torch.empty(0, dtype=torch.long), output=torch.empty(0, dtype=torch.long))
    owners = torch.tensor(owners)
    query = torch.stack(queries) + owners[:, None] * length
    qvalid = torch.stack(query_valid)
    key = pad_sequence(keys, batch_first=True)
    kv = torch.arange(key.shape[1])[None] < torch.tensor([len(k) for k in keys])[:, None]
    selected = qvalid.flatten().nonzero().flatten()
    return dict(query=query, key=key, key_valid=kv, valid=qvalid, batch=owners,
                selected=selected, output=query.flatten()[selected])


def with_local_mask(layout, beats, radius):
    q, k = layout['query'], layout['key']
    flat = beats.float().flatten()
    allowed = (flat[q][..., None] - flat[k][:, None]).abs() <= radius
    return dict(layout, allowed=allowed & layout['valid'][..., None] & layout['key_valid'][:, None])


def prepare_resample(beats, valid, targets, *, valid_only):
    lefts, rights, lw, rw, masks = [], [], [], [], []
    for b, target in enumerate(targets.float()):
        indices = valid[b].nonzero().flatten() if valid_only else torch.arange(beats.shape[1])
        if not len(indices):
            lefts.append(torch.zeros_like(target, dtype=torch.long))
            rights.append(lefts[-1])
            lw.append(torch.zeros_like(target)); rw.append(torch.zeros_like(target))
            masks.append(torch.zeros_like(target, dtype=torch.bool))
            continue
        coordinates = beats[b, indices].float().contiguous()
        if len(indices) == 1:
            left = right = torch.zeros_like(target, dtype=torch.long)
            weight = torch.zeros_like(target)
        else:
            right = torch.searchsorted(coordinates, target.contiguous()).clamp(1, len(indices)-1)
            left = right - 1
            weight = ((target-coordinates[left]) / (coordinates[right]-coordinates[left]).clamp_min(1e-8)).clamp(0, 1)
        left, right = indices[left], indices[right]
        a, bweight = (1-weight)*valid[b, left].float(), weight*valid[b, right].float()
        mask = (a+bweight).gt(0)
        if valid_only:
            mask &= target.ge(coordinates[0]) & target.le(coordinates[-1])
        lefts.append(left); rights.append(right); lw.append(a); rw.append(bweight); masks.append(mask)
    return {k: torch.stack(v) for k, v in dict(left=lefts, right=rights,
            left_weight=lw, right_weight=rw, valid=masks).items()}


def prepare_slots(beats, valid, starts, mapping, before, context):
    targets = starts[:, None].float()-before + torch.arange(context).float() + .5
    indices, masks = [], []
    for w, song in enumerate(mapping.tolist()):
        source = valid[song].nonzero().flatten()
        if not len(source):
            indices.append(torch.zeros(context, dtype=torch.long))
            masks.append(torch.zeros(context, dtype=torch.bool))
            continue
        coords = beats[song, source].float().contiguous()
        positions = torch.searchsorted(coords, targets[w].contiguous())
        chosen = positions.clamp_max(len(source)-1)
        masks.append((positions < len(source)) & (coords[chosen] == targets[w]))
        indices.append(source[chosen] + song * beats.shape[1])
    return dict(indices=torch.stack(indices), valid=torch.stack(masks), beats=targets)


def prepare_encoder_layout(song_beats, song_valid, windows, *, radii, rates, before, context):
    """Only coordinates/masks are read; all model capacities remain caller-owned."""
    from .geometry import pool_stride
    beats, valid = windows.beats.detach().cpu(), windows.valid.cpu()
    starts = windows.target_start_beats.detach().cpu()
    local, resampling, grids, masks = [], [], [], []
    for i, (radius, rate) in enumerate(zip(radii, rates, strict=True)):
        local.append(prepare_local(beats, valid, radius, (32, 32, 16)[i]))
        grid = starts[:, None]-before + (torch.arange(int(context*rate)).float()+.5)/rate
        resampling.append(prepare_resample(beats, valid, grid, valid_only=True))
        grids.append(grid); masks.append(resampling[-1]['valid'])
        if i < 2:
            pooled = pool_stride(beats[..., None], beats, valid)
            beats, valid = pooled.beats, pooled.valid
    up = tuple(prepare_resample(grids[i+1], masks[i+1], grids[i], valid_only=False) for i in (0, 1))
    return dict(local=tuple(local), resample=tuple(resampling), upsample=up, grids=tuple(grids),
        slots=prepare_slots(song_beats.detach().cpu(), song_valid.cpu(), starts,
                            windows.window_to_song.cpu(), before, context))
