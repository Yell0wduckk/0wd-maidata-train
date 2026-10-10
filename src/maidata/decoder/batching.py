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
