"""Frozen mu and shared P0 batches, with one read per unique chart/audio."""
from pathlib import Path
import json

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

SCHEMA = 'maimai.frozen_mu128_continuous_p0_context.v1'
NATIVE_SCHEMA = 'maimai.frozen_mu128_native_p0_context.v1'
NATIVE_CONTRACT = dict(audio='native-window-center-stitched-P0', width=256,
    coordinates='absolute-beats', router='owning-native-window-P0', target='layernorm-mu-eps1e-5')


def plan_requests(cache, refs):
    """The cache index identifies bytes; occurrence identifies supervision instances."""
    return [(cache.train_lookup[(int(r.canonical_index),int(r.view_id))],int(r.occurrence)) for r in refs]


class FrozenSongContext:
    def __init__(self, root, target, *, descriptor=None, teacher_audio_cache=None):
        self.root = Path(root)
        self.manifest = json.loads((self.root/'manifest.json').read_text('utf8')) if descriptor is None else descriptor
        m = self.manifest
        if m.get('schema') not in (SCHEMA, NATIVE_SCHEMA) or not m.get('ready'):
            raise ValueError('frozen mu/P0 context schema/readiness mismatch')
        self.native = m['schema'] == NATIVE_SCHEMA
        self.teacher = None
        if self.native:
            from maimai_baseline.data.teacher_audio import TeacherAudioCache
            from maimai_baseline.runtime_versions import operational
            path = teacher_audio_cache or m.get('teacher_audio_root')
            if path is None:
                raise ValueError('native song context requires --teacher-audio-cache')
            self.teacher = TeacherAudioCache(path)
            if operational(self.teacher.identity) != operational(m.get('teacher_audio')):
                raise ValueError('native song context teacher audio binding mismatch')
        elif teacher_audio_cache is not None:
            raise ValueError('continuous song context cannot consume native teacher audio')
        from maimai_baseline.runtime_versions import operational
        if operational(m.get('latent_data')) != operational(target.identity):
            raise ValueError('frozen context latent data differs from selected cache')
        self.target = target
        self.charts, self.mapping = m['charts'], m['record_to_chart']
        if len(self.mapping) != len(target.records):
            raise ValueError('frozen context mapping coverage mismatch')
        for i, record in enumerate(target.records):
            ci = self.mapping[i]
            if type(ci) is not int or not 0 <= ci < len(self.charts):
                raise ValueError('frozen context chart index mismatch')
            chart = self.charts[ci]
            if tuple(chart['key']) != (record['role'], str(record['record_id']), record['view_id']):
                raise ValueError('frozen context chart/window mapping mismatch')
            song_end = record.get('song_end_tick')
            if getattr(target, 'manifest', {}).get('derived_from') is not None:
                # P0/span retain the imported A corpus; derived window records
                # describe the new profile's separate object-end boundary.
                song_end = target.full_charts[tuple(chart['key'])]['song_end_tick']
            if song_end is not None and song_end != chart['song_end_tick']:
                raise ValueError('frozen context chart conditions mismatch')
            for a, b in (('difficulty_id', 'difficulty_id'), ('level_value', 'level')):
                if a in record and record[a] != chart[b]:
                    raise ValueError('frozen context chart conditions mismatch')

    def descriptor(self):
        return dict(root=str(self.root), manifest=self.manifest, target=self.target.descriptor(),
            teacher_audio_cache=None if self.teacher is None else str(self.teacher.root))

    def batch(self, indices):
        requests = [(r,0) if type(r) is int else tuple(r) for r in indices]
        if (not requests or any(len(r)!=2 or type(r[0]) is not int or not 0<=r[0]<len(self.mapping)
                or type(r[1]) is not int or r[1]<0 for r in requests)):
            raise ValueError('empty/invalid frozen context request')
        instances = list(dict.fromkeys((self.mapping[i],occ) for i,occ in requests))
        for c,occ in instances:
            actual = [i for i,o in requests if self.mapping[i]==c and o==occ]
            if len(actual)!=len(set(actual)) or set(actual)!=set(self.charts[c]['rows']):
                raise ValueError('B batch must contain every window of each selected chart occurrence exactly once')
        mus, valid, conditions, difficulties, levels, spans, audio_keys = [], [], [], [], [], [], []
        read_charts = {}
        for c,occ in instances:
            chart = self.charts[c]
            if c not in read_charts:
                mu, mask = self.target.full_chart(tuple(chart['key']))
                mu = mu.detach().clone() if isinstance(mu,torch.Tensor) else torch.from_numpy(np.array(mu,copy=True))
                mask = mask.detach().clone() if isinstance(mask,torch.Tensor) else torch.from_numpy(np.array(mask,copy=True))
                if (mu.shape != (chart['blocks'],128) or mu.dtype != torch.float32
                        or mask.shape != (chart['blocks'],) or mask.dtype != torch.bool or not bool(torch.isfinite(mu).all())):
                    raise ValueError('frozen mu shape/dtype/mask/finite mismatch')
                read_charts[c] = (mu,mask)
            mu,mask = read_charts[c]
            mus.append(mu.masked_fill(~mask[:,None],0.)); valid.append(mask)
            count = int(chart.get('condition_blocks', len(mask)))
            if count < len(mask):
                raise ValueError('frozen condition coverage truncates target blocks')
            condition = (torch.ones(count, dtype=torch.bool)
                         if 'condition_blocks' in chart else mask.clone())
            conditions.append(condition)
            difficulties.append(chart['difficulty_id']); levels.append(chart['level'])
            spans.append(chart.get('span_beats', chart['song_end_tick']/96.))
            audio_keys.append(chart['audio_key'])
        unique = list(dict.fromkeys(audio_keys))
        self._teacher_batch_bundles = self.teacher.bundles(unique) if self.native else None
        audio, masks, beats = [], [], []
        for key in unique:
            if self.native:
                values = self.teacher.stitched(key, bundles=self._teacher_batch_bundles)
                a, m, b = (values[n] for n in ('audio', 'audio_valid', 'audio_beats'))
            else:
                meta = self.manifest['audio'][key]
                path = self.root/meta['file']
                if path.stat().st_size != meta['bytes']:
                    raise ValueError('frozen P0 byte length mismatch')
                with np.load(path, allow_pickle=False) as z:
                    a, m, b = (torch.from_numpy(z[k].copy()) for k in ('audio','audio_valid','audio_beats'))
            if (a.ndim != 2 or a.shape[-1] != 256 or len(a) < 1 or m.shape != (len(a),)
                    or b.shape != m.shape or a.dtype != torch.float32 or b.dtype != torch.float32
                    or m.dtype != torch.bool or not bool(torch.isfinite(a).all() & torch.isfinite(b).all())
                    or (len(b)>1 and not bool(torch.all(b[1:]>b[:-1])))):
                raise ValueError('continuous P0 shape/dtype/mask/coordinate mismatch')
            audio.append(a); masks.append(m); beats.append(b)
        mask = pad_sequence(valid,batch_first=True)
        row = dict(mu=pad_sequence(mus,batch_first=True), valid=mask, weights=mask.float(),
            target_valid=mask.clone(), condition_valid=pad_sequence(conditions,batch_first=True),
            target_lengths=torch.tensor([len(v) for v in valid]),
            condition_lengths=torch.tensor([len(v) for v in conditions]),
            audio=pad_sequence(audio,batch_first=True), audio_valid=pad_sequence(masks,batch_first=True),
            audio_beats=pad_sequence(beats,batch_first=True),
            audio_mapping=torch.tensor([unique.index(k) for k in audio_keys]),
            difficulty_id=torch.tensor(difficulties), level=torch.tensor(levels,dtype=torch.float32),
            span_beats=torch.tensor(spans,dtype=torch.float32))
        return dict(prior_inputs=row, chart_count=len(instances), unique_chart_count=len(read_charts),
                    chart_occurrences=instances, audio_count=len(unique), reference_count=len(requests),
                    window_requests=requests)


class SongBatches(Dataset):
    def __init__(self, cache_root, context_root, *, descriptor=None):
        self.cache_root, self.context_root = str(cache_root), str(context_root)
        self.metadata, self.context = descriptor, None

    def __getstate__(self):
        return dict(self.__dict__, context=None)

    def __getitem__(self, request):
        if self.context is None:
            from maimai_baseline.data.frozen_latent import FrozenLatentCache
            target = FrozenLatentCache(self.cache_root,fields=(),
                descriptor=None if self.metadata is None else self.metadata['target'])
            self.context = FrozenSongContext(self.context_root, target,
                descriptor=None if self.metadata is None else self.metadata['manifest'],
                teacher_audio_cache=None if self.metadata is None else self.metadata.get('teacher_audio_cache'))
        indices, _, _ = request
        from maimai_baseline.data.batch_transport import pack_batch
        return pack_batch(self.context.batch(indices))


class SongLoader(DataLoader):
    def __iter__(self):
        iterator = super().__iter__()
        return (batch.unpack() for batch in iterator)
