"""Frozen decoder audio with distinct native-window and continuous contracts."""
from pathlib import Path
import json

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from maimai_baseline.model.common import AudioContext
from maimai_dataset.manifest import canonical_json_sha256


SCHEMA = 'maimai.frozen_decoder_candidate_context.v1'
NATIVE_SCHEMA = 'maimai.native_window_decoder_candidate_context.v1'
CONTRACT = dict(audio='reused-continuous-full-song-P0', width=256,
                candidates='frozen-A-C0-C1-projected-keys', rate=24,
                persisted=('candidate_keys', 'candidate_padding_mask', 'candidate_beats'))
NATIVE_CONTRACT = dict(CONTRACT, audio='native-window-center-stitched-P0',
    candidates='new-A-projection-of-cached-native-C0-C1', router='owning-native-window-P0',
    layout='variant-window-candidate-width')


def _json(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def p0_identity(manifest):
    return manifest.get('manifest_identity_sha256') or canonical_json_sha256(manifest)


def validate_candidates(keys, padding, beats):
    n = len(keys)
    if (keys.ndim != 2 or n < 1 or keys.shape[1] != 256 or keys.dtype != torch.float32
            or padding.shape != (n,) or padding.dtype != torch.bool
            or beats.shape != (n,) or beats.dtype != torch.float32
            or not bool(torch.isfinite(keys).all() & torch.isfinite(beats).all())
            or (n > 1 and not bool(torch.all(beats[1:] > beats[:-1])))):
        raise ValueError('frozen decoder candidate shape/dtype/mask/coordinate mismatch')
    if n > 1 and not torch.allclose((beats[1:] - beats[:-1])*24,
                                   torch.ones(n-1), atol=2e-3, rtol=0):
        raise ValueError('frozen decoder candidates must use the continuous 24-per-beat lattice')
    if bool(keys[padding].ne(0).any()):
        raise ValueError('frozen decoder candidate padding must be zero')


class DecoderContextCache:
    def __init__(self, root, p0_root=None, *, descriptor=None):
        self.root = Path(root)
        self.manifest = _json(self.root/'manifest.json') if descriptor is None else descriptor['manifest']
        m = self.manifest
        if m.get('schema') not in (SCHEMA, NATIVE_SCHEMA) or not m.get('ready') or not m.get('audio'):
            raise ValueError('frozen decoder candidate context schema/readiness mismatch')
        self.native = m['schema'] == NATIVE_SCHEMA
        existing = p0_root if hasattr(p0_root, 'manifest') and hasattr(p0_root, 'root') else None
        self.p0_root = Path(existing.root if existing is not None else
                            p0_root or (descriptor or {}).get('p0_root') or m['p0_root'])
        self.p0_manifest = (existing.manifest if existing is not None else
            _json(self.p0_root/'manifest.json') if descriptor is None else descriptor['p0_manifest'])
        from maimai_baseline.data.frozen_context import SCHEMA as P0_SCHEMA, NATIVE_SCHEMA as NATIVE_P0_SCHEMA
        expected = NATIVE_P0_SCHEMA if self.native else P0_SCHEMA
        if self.p0_manifest.get('schema') != expected or not self.p0_manifest.get('ready'):
            raise ValueError('frozen decoder native/continuous P0 representation mismatch')
        self.teacher = None
        self.song_context = existing
        if self.native:
            from maimai_baseline.data.teacher_audio import TeacherAudioCache
            from maimai_baseline.runtime_versions import operational
            self.teacher = existing.teacher if existing is not None else TeacherAudioCache(
                (descriptor or {}).get('teacher_audio_cache') or self.p0_manifest['teacher_audio_root'])
            if operational(self.teacher.identity) != operational(self.p0_manifest.get('teacher_audio')):
                raise ValueError('native decoder teacher audio binding mismatch')
            if canonical_json_sha256(m.get('contract')) != canonical_json_sha256(NATIVE_CONTRACT):
                raise ValueError('native decoder candidate contract mismatch')
        if p0_identity(self.p0_manifest) != m['p0_manifest_identity']:
            raise ValueError('frozen decoder candidate/P0 data binding mismatch')
        if set(m['audio']) != set(self.p0_manifest['audio']):
            raise ValueError('frozen decoder candidate/P0 audio coverage mismatch')
        self.identity = dict(schema=m['schema'], manifest_identity=canonical_json_sha256(m),
                             p0_manifest_identity=m['p0_manifest_identity'])

    def descriptor(self):
        return dict(root=str(self.root), p0_root=str(self.p0_root), manifest=self.manifest,
                    p0_manifest=self.p0_manifest,
                    teacher_audio_cache=None if self.teacher is None else str(self.teacher.root))

    def batch(self, audio_keys, window_to_audio, start_tick, *, p0=None):
        """Read each unique audio once; slice all window conditions without encoding."""
        audio_keys = list(audio_keys)
        if not audio_keys or len(audio_keys) != len(set(audio_keys)):
            raise ValueError('decoder context batch requires unique audio keys')
        mapping = torch.as_tensor(window_to_audio, dtype=torch.long)
        starts = torch.as_tensor(start_tick, dtype=torch.long)
        if (mapping.ndim != 1 or starts.shape != mapping.shape or not len(mapping)
                or bool(((mapping < 0) | (mapping >= len(audio_keys))).any())
                or bool((starts < 0).any())):
            raise ValueError('invalid decoder context window mapping')
        if p0 is not None:
            a, valid, b = (p0[name] for name in ('audio', 'audio_valid', 'audio_beats'))
            if (a.ndim != 3 or a.shape[0] != len(audio_keys) or a.shape[-1] != 256
                    or valid.shape != a.shape[:2] or b.shape != valid.shape
                    or a.dtype != torch.float32 or b.dtype != torch.float32 or valid.dtype != torch.bool):
                raise ValueError('shared decoder/prior P0 layout mismatch')
        if self.native:
            return self._native_batch(audio_keys, mapping, starts, p0)
        audio, masks, coords, keys, candidate_masks, candidate_beats = [], [], [], [], [], []
        for key in audio_keys:
            if p0 is None:
                p = self.p0_manifest['audio'][key]
                path = self.p0_root/p['file']
                if path.stat().st_size != p['bytes']:
                    raise ValueError('frozen decoder P0 byte length mismatch')
                with np.load(path, allow_pickle=False) as z:
                    a, valid, b = (torch.from_numpy(z[k].copy()) for k in ('audio', 'audio_valid', 'audio_beats'))
                if (a.ndim != 2 or len(a) < 1 or a.shape[1] != 256 or a.dtype != torch.float32
                        or valid.shape != (len(a),) or valid.dtype != torch.bool or b.shape != valid.shape
                        or b.dtype != torch.float32 or not bool(torch.isfinite(a).all() & torch.isfinite(b).all())
                        or (len(b)>1 and not bool((b[1:]>b[:-1]).all()))):
                    raise ValueError('frozen decoder P0 shape/dtype/mask/coordinate mismatch')
                audio.append(a); masks.append(~valid); coords.append(b)
            c = self.manifest['audio'][key]
            path = self.root/c['file']
            if path.stat().st_size != c['bytes']:
                raise ValueError('frozen decoder candidate byte length mismatch')
            with np.load(path, allow_pickle=False) as z:
                k, mask, cb = (torch.from_numpy(z[n].copy()) for n in CONTRACT['persisted'])
            validate_candidates(k, mask, cb)
            if len(k) != c['frames'] or k.shape[1] != c['width']:
                raise ValueError('frozen decoder candidate metadata mismatch')
            keys.append(k); candidate_masks.append(mask); candidate_beats.append(cb)
        context = AudioContext((pad_sequence(audio, batch_first=True) if p0 is None else p0['audio'],),
            (pad_sequence(masks, batch_first=True, padding_value=True) if p0 is None else ~p0['audio_valid'],),
            (pad_sequence(coords, batch_first=True) if p0 is None else p0['audio_beats'],),
            candidate_keys=pad_sequence(keys, batch_first=True),
            candidate_padding_mask=pad_sequence(candidate_masks, batch_first=True, padding_value=True),
            candidate_beats=pad_sequence(candidate_beats, batch_first=True))
        from maimai_baseline.data.fullaudio import window_context
        return context, window_context(context, mapping, starts)

    def _native_batch(self, audio_keys, mapping, starts, p0):
        bundles = getattr(self.song_context, '_teacher_batch_bundles', None)
        if bundles is None or set(bundles) != set(audio_keys):
            bundles = self.teacher.bundles(audio_keys)
        audio, masks, beats = [], [], []
        rows = [None] * len(starts)
        for index, key in enumerate(audio_keys):
            if p0 is None:
                values = self.teacher.stitched(key, bundles=bundles)
                audio.append(values['audio']); masks.append(~values['audio_valid']); beats.append(values['audio_beats'])
            meta = self.manifest['audio'][key]
            path = self.root/meta['file']
            if path.stat().st_size != meta['bytes']:
                raise ValueError('native decoder candidate byte length mismatch')
            with np.load(path, allow_pickle=False) as archive:
                values = {name: torch.from_numpy(archive[name].copy()) for name in CONTRACT['persisted']}
            validate_native_candidates(values, meta['starts'])
            if (meta.get('windows') != len(meta['starts']) or meta.get('frames') != 528 or meta.get('width') != 256):
                raise ValueError('native decoder candidate metadata mismatch')
            positions = mapping.eq(index).nonzero().flatten()
            if not len(positions):
                continue
            requested = starts.index_select(0, positions)
            inventory = {int(start): i for i, start in enumerate(meta['starts'])}
            try:
                selected = torch.tensor([inventory[int(start)] for start in requested], dtype=torch.long)
            except KeyError as error:
                raise ValueError('native candidate cache lacks the requested owning window') from error
            local, geometry = self.teacher.window_context(key, requested, bundles=bundles)
            local = native_candidates(local, geometry, {name: value.index_select(0, selected) for name, value in values.items()})
            for ordinal, position in enumerate(positions.tolist()):
                rows[position] = local.map_audio_windows(torch.tensor([ordinal]))
        context = AudioContext((pad_sequence(audio, batch_first=True) if p0 is None else p0['audio'],),
            (pad_sequence(masks, batch_first=True, padding_value=True) if p0 is None else ~p0['audio_valid'],),
            (pad_sequence(beats, batch_first=True) if p0 is None else p0['audio_beats'],))
        return context, stack_native_windows(rows)


def validate_native_candidates(values, starts):
    keys, padding, beats = (values[name] for name in CONTRACT['persisted'])
    if (keys.ndim != 3 or keys.shape[:2] != (len(starts),528) or len(starts) < 1
            or padding.shape != keys.shape[:2] or beats.shape != padding.shape
            or len(set(map(int, starts))) != len(starts)):
        raise ValueError('native candidate window inventory mismatch')
    for index in range(len(starts)):
        validate_candidates(keys[index], padding[index], beats[index])


def native_candidates(local, geometry, values):
    """Keep the owning window's native halo and its exact indexed time lookup."""
    return AudioContext(local.memories[-1:], local.padding_masks[-1:], local.beats[-1:],
        **values, candidate_lookup_left=geometry['candidate_left'],
        candidate_lookup_right=geometry['candidate_right'], candidate_lookup_weight=geometry['candidate_weight'],
        candidate_lookup_padding=geometry['candidate_padding'],
        candidate_lookup_start_tick=geometry['candidate_start_tick'])


def stack_native_windows(rows):
    from dataclasses import fields
    result = {}
    for field in fields(AudioContext):
        values = [getattr(row, field.name) for row in rows]
        first = values[0]
        result[field.name] = (tuple(torch.cat([value[i] for value in values]) for i in range(len(first)))
            if isinstance(first, tuple) else torch.cat(values) if isinstance(first, torch.Tensor) else first)
    return AudioContext(**result)
