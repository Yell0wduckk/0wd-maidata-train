"""Bound local Encoder activations without changing song coverage or updates."""
from dataclasses import replace

import torch
from torch.utils.checkpoint import checkpoint

from .types import EncoderOutput, ScaleMemory


def slice_windows(windows, start, stop, layout):
    names = ('mel', 'aux32', 'beats', 'valid', 'target_start_beats', 'window_to_song')
    return replace(windows, **{name: getattr(windows, name)[start:stop] for name in names}, layout=layout)


def encode_chunks(encoder, memory, windows, chunks):
    values = [[], [], []]
    beats = [[], [], []]
    masks = [[], [], []]
    for start, stop, layout in chunks:
        subset = slice_windows(windows, start, stop, layout)

        # Bind each window subset: backward recomputes after this loop advances.
        def encode(mel, aux32, subset=subset):
            output = encoder.encode_windows(memory, replace(subset, mel=mel, aux32=aux32))
            return tuple(getattr(output, name).values for name in ('P0', 'P1', 'P2'))

        if encoder.training and torch.is_grad_enabled():
            result = checkpoint(encode, subset.mel, subset.aux32, use_reentrant=False,
                                preserve_rng_state=True)
        else:
            result = encode(subset.mel, subset.aux32)
        for i, value in enumerate(result):
            values[i].append(value)
            beats[i].append(layout['grids'][i])
            masks[i].append(layout['resample'][i]['valid'])
    return EncoderOutput(*(ScaleMemory(torch.cat(v), torch.cat(b), torch.cat(m))
                           for v, b, m in zip(values, beats, masks, strict=True)))
