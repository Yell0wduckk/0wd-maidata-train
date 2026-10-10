"""Continuous, absolute-beat audio geometry, prepared once per unique variant."""
from fractions import Fraction
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from maimai_dataset.geometry import _mapping, _pool_scalar

def arrays(feature, coords, timing, num_samples, *, geometry=None, bpms=None):
    # Integer-beat alignment fixes both stride phases even for negative audio.
    k = coords['grid_k']
    if feature.shape != (128,len(k)) or feature.dtype != np.float32 or not len(k):
        raise ValueError('continuous Mel/coordinate layout mismatch')
    first = int(k[0]) // 24 * 24
    length = max(24, ((int(k[-1]) - first + 24) // 24) * 24)
    positions = np.arange(first, first + length, dtype=np.int64)
    indices = positions - int(k[0])
    available = (indices >= 0) & (indices < len(k))
    valid = np.zeros(length, bool)
    valid[available] = coords['valid'][indices[available]]
    mel = np.zeros((length, 128), np.float32)
    mel[valid] = feature[:, indices[valid]].T
    if not np.isfinite(mel).all():
        raise ValueError('nonfinite continuous Mel input')
    beats = (positions / 24).astype(np.float32)
    if bpms is None:
        bpms = np.asarray([timing.bpm_at_beat(Fraction(int(p), 24)) for p in positions], np.float32)
    if geometry is None:
        begin = float(timing.second_to_beat(0.))
        end = float(timing.second_to_beat(num_samples / 24000.))
        geometry = geometry_for(beats, valid, begin, end)
    return dict(audio=torch.from_numpy(mel), audio_padding_mask=torch.from_numpy(~valid),
        frame_beats=torch.from_numpy(beats), frame_bpms=torch.from_numpy(bpms),
        audio_start_tick=torch.tensor(0.), interpolation_geometry=geometry)


def geometry_for(beats, valid, begin, end):
    pyramid = [(beats, ~valid)]
    for _ in range(2):
        pooled, padding = _pool_scalar(*pyramid[-1])
        padding |= pyramid[-1][1][::2]
        pooled[padding] = 0.
        pyramid.append((pooled, padding))
    geometry = {}
    for i, (source, padding) in enumerate(pyramid):
        rate = 8 // 2**i
        target = beats[0] + (np.arange(len(beats)//24*rate, dtype=np.float32)+.5)/rate
        # Preserve internal missing audio as well as the real start/end boundary.
        raw_pos = (target - beats[0]) * 24
        raw_left = np.floor(raw_pos).astype(np.int64).clip(0, len(valid)-1)
        raw_right = np.ceil(raw_pos).astype(np.int64).clip(0, len(valid)-1)
        target_padding = (target < begin) | (target >= end) | ~(valid[raw_left] | valid[raw_right])
        for key, value in _mapping(source, padding, target, target_padding).items():
            geometry[f'grid_{i}_{key}'] = torch.from_numpy(value)
        geometry[f'grid_{i}_beats'] = torch.from_numpy(target)
    # The restored timing head scores C0 together with interpolated C1. Build
    # this mapping once on the continuous lattice, alongside audio geometry.
    for key, value in _mapping(*pyramid[1], *pyramid[0]).items():
        geometry['a1_to_a0_'+key] = torch.from_numpy(value)
    from maimai_baseline.model.fusion import fixed_double_geometry
    for i,rate in enumerate((4,2)):
        for name,value in zip(('left','right','weight'),fixed_double_geometry(len(beats)//24*rate)):
            geometry[f'fusion_{i}_{name}'] = value
    return geometry


def pack(rows, mapping):
    result = {k:pad_sequence([r[k] for r in rows], batch_first=True,
        padding_value=True if k=='audio_padding_mask' else 0.)
        for k in ('audio','audio_padding_mask','frame_beats','frame_bpms')}
    result['audio_start_tick'] = torch.zeros(len(rows))
    result['window_to_audio'] = torch.tensor(mapping, dtype=torch.long)
    result['interpolation_geometry'] = {k:pad_sequence([r['interpolation_geometry'][k] for r in rows],
        batch_first=True, padding_value=True if k.endswith('padding') else 0)
        for k in rows[0]['interpolation_geometry']}
    from maimai_baseline.model.local_attention import prepare_pyramid_layouts
    result['interpolation_geometry']['local_attention_layouts'] = prepare_pyramid_layouts(
        result['frame_beats'], result['audio_padding_mask'])
    return result


class Provider:
    def __init__(self, store, *, prepared_timings=None, timing_root=None):
        self.store = store
        self.geometry = {}
        self.bpms = {}
        self.prepared_timings = prepared_timings or {}
        self.timing_root = timing_root

    def row(self, variant):
        from maimai_dataset.timing import TimingMap
        info = self.store.variants[variant]
        timing = None
        if variant not in self.geometry:
            prepared = self.prepared_timings.get(variant)
            timing = (prepared.get(self.timing_root) if prepared is not None else
                      TimingMap.from_dict(info['timing']))
        row = arrays(self.store._features[variant], self.store._coordinates[variant],
            timing, int(info['num_samples']), geometry=self.geometry.get(variant),
            bpms=self.bpms.get(variant))
        # Retain small derived geometry only, never all Mel copies in workers.
        if variant in self.geometry:
            row['interpolation_geometry'] = self.geometry[variant]
        else:
            self.geometry[variant] = row['interpolation_geometry']
            self.bpms[variant] = row['frame_bpms'].numpy()
        return row

    def batch(self, variants):
        unique = list(dict.fromkeys(variants))
        return pack([self.row(v) for v in unique], [unique.index(v) for v in variants])


def window_context(context, mapping, starts):
    """Gather the unchanged router range: pre4 + target16 + post2 beats.

    Integer starts and the absolute half-cell fusion lattices make this an
    indexed slice, with explicit out-of-song padding and no re-encoding.
    """
    from maimai_baseline.model.common import AudioContext
    from maimai_dataset.window import AUDIO_CONTEXT_BEATS, AUDIO_BEATS_BEFORE
    values, masks, coordinates = [], [], []
    rates = (8,) if len(context.memories) == 1 else (2, 4, 8)
    for memory, padding, beats, rate in zip(context.memories, context.padding_masks,
                                          context.beats, rates, strict=True):
        count = AUDIO_CONTEXT_BEATS * rate
        steps = torch.arange(count, device=starts.device)
        target = starts[:, None].float()/96. - AUDIO_BEATS_BEFORE + (steps.float()+.5)/rate
        offset = torch.round((target - beats[mapping, :1]) * rate).long()
        safe = offset.clamp(0, memory.shape[1]-1)
        invalid = (offset < 0) | (offset >= memory.shape[1]) | padding[mapping[:, None], safe]
        values.append(memory[mapping[:, None], safe].masked_fill(invalid[..., None], 0.))
        masks.append(invalid)
        coordinates.append(target)
    # MOMENT candidates cover the target sixteen beats, plus the last C0
    # interpolation neighbor. They share the continuous encoder's keys.
    source = context.candidate_beats
    steps = torch.arange(385, device=starts.device)[None]
    target = starts[:, None].float()/96. + steps.float()/24.
    origin = torch.round(source[:, 0]*24).long()
    offset = starts.long()[:, None]//4 - origin[mapping, None] + steps
    safe = offset.clamp(0, source.shape[1]-1)
    invalid = (offset < 0) | (offset >= source.shape[1]) | context.candidate_padding_mask[mapping[:, None], safe]
    keys = context.candidate_keys[mapping[:, None], safe].masked_fill(invalid[..., None], 0.)
    position = torch.arange(1537, device=starts.device)[None].expand(len(starts), -1)/4.
    left = position.floor().long()
    right = (left+1).clamp_max(384)
    return AudioContext(tuple(values), tuple(masks), tuple(coordinates),
        candidate_keys=keys, candidate_padding_mask=invalid, candidate_beats=target,
        candidate_lookup_left=left, candidate_lookup_right=right,
        candidate_lookup_weight=position-left,
        candidate_lookup_padding=invalid.gather(1, left) & invalid.gather(1, right),
        candidate_lookup_start_tick=starts, diagnostics=context.diagnostics)
