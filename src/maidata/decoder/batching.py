"""Tensor collation only; never constructs audio/Draft/Physical networks."""
import torch
from maidata.chart.types import (Moment, Note, Route, KINDS, FLAGS, GENERATED_TYPES,
                                BUTTON_PADS, TOUCH_PADS)
from .types import TokenBatch


def pack_tokens(rows, positions, *, device):
    if not rows or any(not r or len(r) != len(b) for r, b in zip(rows, positions)) or len(rows) != len(positions):
        raise ValueError('tokens and known absolute positions must be aligned and nonempty')
    shape = (len(rows), max(map(len, rows)))
    def ids():
        return torch.full(shape, -1, dtype=torch.long, device=device)
    types, kind, pad, route, track_break = (ids() for _ in range(5))
    flags = torch.full((*shape, 5), -1, dtype=torch.long, device=device)
    beats, time = (torch.zeros(shape, device=device) for _ in range(2))
    valid = torch.zeros(shape, dtype=torch.bool, device=device)
    for batch, (row, coords) in enumerate(zip(rows, positions)):
        for i, (token, beat) in enumerate(zip(row, coords)):
            valid[batch, i] = True
            types[batch, i] = GENERATED_TYPES.index(token.type)
            beats[batch, i] = beat
            if isinstance(token, Note):
                kind[batch, i] = KINDS.index(token.kind)
                pads = TOUCH_PADS if token.kind in ('touch', 'touch_hold') else BUTTON_PADS
                pad[batch, i] = pads.index(token.pad)
                flags[batch, i] = torch.tensor([getattr(token, f) if getattr(token, f) is not None else -1
                                               for f in FLAGS], device=device)
            elif isinstance(token, Route):
                route[batch, i] = token.route_id
                track_break[batch, i] = -1 if token.track_break is None else token.track_break
            else:
                time[batch, i] = token.delta_ms / 1000 if isinstance(token, Moment) else token.ticks / 96
    return TokenBatch(types, beats, valid, kind, pad, flags, route, track_break, time)


def collate_windows(examples, *, device):
    batch = pack_tokens([w.inputs for w in examples], [w.beats for w in examples], device=device)
    next_types = torch.zeros_like(batch.types)
    supervised = torch.zeros_like(batch.valid)
    for i, w in enumerate(examples):
        next_types[i, :len(w.inputs)] = torch.tensor(w.next_types, device=device)
        supervised[i, :len(w.inputs)] = torch.tensor(w.supervised, device=device)
    return batch, next_types, supervised


def select_condition_windows(z, groups, starts, window_valid, config):
    """Consumer calls the public data-only selector; no learned parameters are shared."""
    from maidata.draft.windows import select_windows
    return select_windows(z, groups, starts, window_valid, target_beats=config['target_beats'],
                          before_beats=config['decoder_z_before_beats'], after_beats=config['decoder_z_after_beats'])


def audio_key_spans(valid, *, validate=True):
    if validate and (valid.device.type != 'cpu' or valid.ndim != 2 or valid.dtype != torch.bool):
        raise ValueError('audio key spans require a CPU validity matrix')
    spans = []
    for row in valid:
        index = row.nonzero().flatten()
        # An all-masked key preserves a zero gradient connection for empty audio.
        spans.append((int(index[0]),int(index[-1])+1) if len(index) else (0,min(1,len(row))))
    return tuple(spans)


def audio_query_layout(valid, mapping, audio_count, *, spans=None, condition_valid=None, validate=True):
    """CPU indices collect a song's queries without repeating its full K/V bank."""
    if validate and (valid.device.type != 'cpu' or mapping.device.type != 'cpu'):
        raise ValueError('audio query layout must be prepared on CPU')
    if validate and (valid.ndim != 2 or valid.dtype != torch.bool or mapping.shape != (len(valid),)
            or mapping.dtype != torch.long or ((mapping < 0) | (mapping >= audio_count)).any()):
        raise ValueError('invalid audio query mapping')
    if validate and spans is not None and len(spans) != audio_count:
        raise ValueError('audio key span count differs')
    indices = tuple((valid & (mapping == song)[:, None]).flatten().nonzero().flatten()
                    for song in range(audio_count))
    # Bucketing changes execution layout only, never window order or attention
    # membership. Bound query padding and group similarly sized key spans.
    buckets = {}
    for song, selected in enumerate(indices):
        if not len(selected):
            continue
        extent = 1 if spans is None else max(1, spans[song][1]-spans[song][0])
        key = ((len(selected)-1).bit_length(), (extent-1).bit_length())
        buckets.setdefault(key, []).append(song)
    songs, gather, take, put, groups = [], [], [], [], []
    offset = 0
    for _, members in sorted(buckets.items()):
        width = max(len(indices[s]) for s in members)
        lo = 0 if spans is None else min(spans[s][0] for s in members)
        hi = None if spans is None else max(spans[s][1] for s in members)
        groups.append((len(members), width, lo, hi))
        for song in members:
            selected = indices[song]
            gather.append(torch.nn.functional.pad(selected, (0, width-len(selected))))
            take.append(torch.arange(len(selected))+offset)
            put.append(selected)
            songs.append(song)
            offset += width
    cat = lambda rows: torch.cat(rows) if rows else torch.empty(0, dtype=torch.long)
    from .row_layout import window_layout
    rows = window_layout(valid)
    compact = torch.zeros(valid.numel(), dtype=torch.long)
    compact[rows['indices']] = torch.arange(len(rows['indices']))
    return dict(mapping=mapping, shape=tuple(valid.shape), spans=spans, indices=indices,
                songs=torch.tensor(songs, dtype=torch.long), groups=tuple(groups),
                gather=cat(gather), take=cat(take), put=cat(put),
                valid_indices=rows['indices'], compact_gather=compact[cat(gather)], compact_put=compact[cat(put)],
                rows=rows, condition_rows=rows if condition_valid is None else window_layout(valid, condition_valid))
