"""Disk cache reader and Encoder input adapters; no hidden-state caching."""
from pathlib import Path
import json

import numpy as np


def validate_grid(a):
    mel = a['mel_patches']
    if mel.ndim != 3:
        raise ValueError('mel_patches must be [J,R,M]')
    j,r,m = mel.shape
    shapes = dict(sample_valid=(j,r), beat_centers=(j,), aux32_grid=(j,r,32),
                  aux32_centers=(j,32), sample_times_sec=(j,r), sample_indices=(j,r),
                  grid_k=(j,r), first_beat=())
    if j<1 or r<1 or m<1:
        raise ValueError('empty grid')
    for key, shape in shapes.items():
        if a[key].shape != shape:
            raise ValueError(f'{key}: {a[key].shape} != {shape}')
    if a['sample_valid'].dtype != np.bool_:
        raise ValueError('sample_valid must be bool')
    for key in ('mel_patches','beat_centers','aux32_grid','aux32_centers'):
        if a[key].dtype != np.float32 or not np.isfinite(a[key]).all():
            raise ValueError(f'{key} must be finite float32')
    if not np.isfinite(a['sample_times_sec']).all():
        raise ValueError('nonfinite times')
    if not np.array_equal(a['beat_centers'], int(a['first_beat'])+np.arange(j,dtype=np.float32)+.5):
        raise ValueError('non-contiguous absolute beat centers')
    invalid = ~a['sample_valid']
    if np.any(mel[invalid]) or np.any(a['aux32_grid'][invalid]):
        raise ValueError('invalid positions must be zero')


def load_grid(path, *, mel_root=None, validate_input=True):
    path = Path(path)
    with np.load(Path(path), allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files}
    if 'mel_patches' not in result:
        meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        reference = meta['mel_reference']
        # Consumer relocation leaves the producer's metadata/identity untouched.
        from pathlib import PureWindowsPath
        feature = (Path(mel_root) / PureWindowsPath(reference['feature_path']).name
                   if mel_root is not None else Path(reference['feature_path']))
        old = np.load(feature,mmap_mode='r',allow_pickle=False)
        if validate_input and (old.dtype!=np.float32 or list(old.shape)!=reference['shape']):
            raise ValueError('referenced Mel shape/dtype mismatch')
        indices = result['mel_source_indices']
        reused = indices>=0
        if validate_input and (np.any(indices[reused]>=old.shape[1])):
            raise ValueError('referenced Mel index out of range')
        mel = np.zeros((*indices.shape,old.shape[0]),np.float32)
        mel[reused] = old[:,indices[reused]].T
        mel.reshape(-1,old.shape[0])[result['mel_new_flat_indices']] = result['mel_new_values']
        result['mel_patches'] = mel
    if validate_input:
        validate_grid(result)
    return result


def song_inputs(grids, *, validate_input=True):
    import torch
    from maidata.audio.types import SongInputs
    if validate_input and (not grids):
        raise ValueError('empty song batch')
    r,m = grids[0]['mel_patches'].shape[1:]
    j = max(len(g['beat_centers']) for g in grids)
    mel = np.zeros((len(grids),j,r,m),np.float32)
    valid = np.zeros((len(grids),j,r),bool)
    centers = np.empty((len(grids),j),np.float32)
    aux = np.zeros((len(grids),j,32),np.float32)
    for i,g in enumerate(grids):
        if validate_input:
            validate_grid(g)
        if validate_input and (g['mel_patches'].shape[1:] != (r,m)):
            raise ValueError('incompatible grid representations in song batch')
        n = len(g['beat_centers'])
        mel[i,:n],valid[i,:n],aux[i,:n] = g['mel_patches'],g['sample_valid'],g['aux32_centers']
        centers[i] = int(g['first_beat'])+np.arange(j,dtype=np.float32)+.5
    return SongInputs(*(torch.from_numpy(a) for a in (mel,valid,centers,aux)))


def window_inputs(grids, target_starts, window_to_song, *, target_beats, before_beats, after_beats, validate_input=True):
    import torch
    from maidata.audio.types import WindowInputs
    starts = np.asarray(target_starts)
    routes = np.asarray(window_to_song)
    if validate_input and (starts.ndim!=1 or routes.shape!=starts.shape or np.any(starts!=np.floor(starts))):
        raise ValueError('window starts must be integer beats and routes must align')
    if validate_input and (np.any(routes!=np.floor(routes)) or np.any(routes<0) or np.any(routes>=len(grids))):
        raise ValueError('invalid window_to_song')
    if validate_input and (any(not isinstance(v,int) or v<0 for v in (target_beats,before_beats,after_beats)) or target_beats==0):
        raise ValueError('invalid window extent')
    r,m = grids[0]['mel_patches'].shape[1:]
    c = target_beats+before_beats+after_beats
    mel = np.zeros((len(starts),c,r,m),np.float32)
    aux = np.zeros((len(starts),c,r,32),np.float32)
    valid = np.zeros((len(starts),c,r),bool)
    beats = np.empty((len(starts),c*r),np.float32)
    for w,(start,route) in enumerate(zip(starts,routes)):
        g = grids[int(route)]
        if validate_input and (g['mel_patches'].shape[1:]!=(r,m)):
            raise ValueError('incompatible grid representations')
        first = int(start)-before_beats
        indices = first+np.arange(c)-int(g['first_beat'])
        ok = (indices>=0)&(indices<len(g['beat_centers']))
        mel[w,ok],aux[w,ok],valid[w,ok] = g['mel_patches'][indices[ok]],g['aux32_grid'][indices[ok]],g['sample_valid'][indices[ok]]
        beats[w] = first+np.arange(c*r,dtype=np.float64)/r
    return WindowInputs(*(torch.from_numpy(a) for a in (
        mel.reshape(len(starts),c*r,m),aux.reshape(len(starts),c*r,32),beats,
        valid.reshape(len(starts),c*r),starts.astype(np.float32),routes.astype(np.int64))))
