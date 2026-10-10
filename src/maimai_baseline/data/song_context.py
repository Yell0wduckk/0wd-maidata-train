"""Immutable whole-song sidecar, retaining original window supervision."""
from pathlib import Path
import json
import numpy as np
import torch
import time
from torch.utils.data import Dataset
from maimai_dataset.manifest import canonical_json_sha256 as digest
from maimai_dataset.integrity import file_sha256
from maimai_baseline.data.style_cache import StyleLatentCache

CONTRACT = dict(schema="maimai.full_song_z128_p2_fused_audio256_prior.v1", audio_width=256, latent="four-beat-128", audio="stitched-center16-P0-8ppb",
    coordinates="absolute-four-beat-centers", conditioning="level-difficulty-P2-fused-P0", global_latent=False, sharing="step-audio-kv;record-view-posterior-t-noise",
    loss="occurrence-weighted-selected-four-beat-blocks", query_chunk=128, checkpoint_blocks=False)

SCHEMA = "maimai.full_song_chart128_p2_fused_audio256_context.v1"


def chart_key(record):
    return (record['role'], str(record['record_id']), int(record['view_id']))


def read(path):
    return json.loads(Path(path).read_text('utf8'))


def center_p0(audio, valid, beats, length):
    """Same center selection for offline production and raw-audio inference."""
    if audio.shape != (176, 256) or valid.shape != (176,) or beats.shape != (176,):
        raise ValueError('expected original window P0')
    expected = (np.arange(176, dtype=np.float32)+.5)/8-4
    if not np.allclose(beats, expected, atol=1e-5, rtol=0):
        raise ValueError('P0 window coordinate mismatch')
    return audio[32:32+length*8].copy(), valid[32:32+length*8].copy()


def layout(target, audio, bpm, note):
    # The plan's aggregate digest includes model provenance. Validate the
    # actual record/window layout below instead of reinstating a hash lock.
    charts, groups, windows = [], {}, {}
    mapping = []
    for i, r in enumerate(target.records):
        key = chart_key(r)
        variant = bpm['segments'][r['segment_id']]
        if variant != note['charts'][r['record_id']]['variant']:
            raise ValueError('audio/chart timing variant mismatch')
        audio_key = digest(dict(role=r['role'], variant=bpm['variants'][variant]['identity'],
            preprocessing=bpm['preparation_identity'], parent=target.manifest['bindings'].get('stage_a_sha256')))
        if key not in groups:
            groups[key] = len(charts)
            charts.append(dict(key=list(key), audio_key=audio_key, variant=variant,
                length=r['song_end_tick']//96+1, blocks=r['song_end_tick']//384+1, song_end_tick=r['song_end_tick'],
                difficulty_id=r['difficulty_id'], level=r['level_value'], rows=[]))
        ci = groups[key]; c = charts[ci]
        if c['audio_key'] != audio_key or c['song_end_tick'] != r['song_end_tick']:
            raise ValueError('inconsistent full-song geometry')
        if r['start_tick'] % 1536:
            raise ValueError('source windows must follow the original 16-beat grid')
        c['rows'].append(i); mapping.append(ci)
        window = (audio_key, r['start_tick'])
        ai = int(audio.mapping[i])
        if window in windows and windows[window] != ai:
            raise ValueError('same audio window has conflicting source cache entries')
        windows[window] = ai
    return charts, mapping, windows


class SongContextCache:
    def __init__(self, root, target, audio=None, *, verify=False, descriptor=None):
        self.root = Path(root)
        self.manifest = read(self.root/'manifest.json') if descriptor is None else descriptor
        m = self.manifest
        if m.get('schema') != SCHEMA or not m.get('ready'):
            raise ValueError('full-song cache format/readiness mismatch')
        self.charts, self.mapping = m['charts'], m['record_to_chart']
        if len(self.mapping) != len(target.records):
            raise ValueError('song cache mapping coverage mismatch')
        if descriptor is None:
            for i, record in enumerate(target.records):
                index = self.mapping[i]
                if type(index) is not int or not 0 <= index < len(self.charts):
                    raise ValueError('song cache mapping index mismatch')
                chart = self.charts[index]
                if tuple(chart['key']) != chart_key(record):
                    raise ValueError('song cache record mapping mismatch')
                for old, new in (('difficulty_id', 'difficulty_id'), ('level_value', 'level'), ('song_end_tick', 'song_end_tick')):
                    if old in record and chart[new] != record[old]:
                        raise ValueError('song cache chart condition mismatch')
            if verify:
                for name, meta in m['files'].items():
                    if file_sha256(self.root/name) != meta['sha256']:
                        raise ValueError('song cache payload mismatch: ' + name)
        self.target = target

    def descriptor(self):
        return dict(root=str(self.root), manifest=self.manifest, target=self.target.descriptor())

    def batch(self, indices, *, epoch, seed, deterministic, read_audio=True, whole_song=False):
        if any(type(i) is not int or not 0 <= i < len(self.mapping) for i in indices):
            raise ValueError('song cache request index out of range')
        ids = list(dict.fromkeys(self.mapping[i] for i in indices))
        if whole_song:
            if len(indices) != len(set(indices)):
                raise ValueError('whole-song batch repeats a window')
            selected = set(indices)
            expected = {i for c in ids for i in self.charts[c]['rows']}
            if selected != expected:
                raise ValueError('whole-song batch must contain every window of each chart')
        # Read only the requested unique references, never repeat full-song arrays per window.
        groups, tensors = {}, []
        for j, c in enumerate(ids):
            meta = self.charts[c]
            with np.load(self.root/f'chart-{c}.npz', allow_pickle=False) as z:
                mu = torch.from_numpy(z['mu'].copy());logvar=torch.from_numpy(z['logvar'].copy())
                valid = torch.from_numpy(z['valid'].copy())
                if mu.shape!=(meta['blocks'],128) or logvar.shape!=mu.shape or mu.dtype!=torch.float32 or logvar.dtype!=torch.float32 or not bool(torch.isfinite(mu).all()&torch.isfinite(logvar).all()):raise ValueError('invalid frozen posterior cache')
            if valid.shape!=(meta['blocks'],) or valid.dtype!=torch.bool or bool(((logvar < -6)|(logvar > 2)).any()):raise ValueError('invalid posterior mask/range')
            if whole_song:
                weights = valid.to(torch.float32)
            else:
                weights = torch.zeros(len(valid), dtype=torch.float32)
                for i in indices:
                    if self.mapping[i] == c:
                        start = self.target.records[i]['start_tick']//384
                        mask = torch.from_numpy(np.array(self.target.arrays['valid'][i], copy=True))
                        n = max(0, min(4, len(weights)-start))
                        weights[start:start+n] += mask[:n]
                if bool((weights.ne(0) & ~valid).any()):
                    raise ValueError('supervision outside song validity')
            tensors.append(dict(mu=mu,logvar=logvar, valid=valid, weights=weights,
                difficulty_id=meta['difficulty_id'], level=meta['level'], span_beats=meta['song_end_tick']/96, index=j, chart_id=c))
            groups.setdefault(meta['audio_key'], []).append(j)
        packed = []
        for key, members in groups.items():
            a = {}
            if read_audio:
                with np.load(self.root/f'audio-{key}.npz', allow_pickle=False) as z:
                    a = {k:torch.from_numpy(z[k].copy())[None] for k in ('audio','audio_valid','audio_beats')}
                tokens, mask, beats = a['audio'], a['audio_valid'], a['audio_beats']
                if (tokens.ndim != 3 or tokens.shape[0] != 1 or tokens.shape[-1] != 256
                        or mask.shape != tokens.shape[:2] or beats.shape != mask.shape
                        or mask.dtype != torch.bool or tokens.dtype != torch.float32
                        or not bool(torch.isfinite(tokens).all() & torch.isfinite(beats).all())):
                    raise ValueError('invalid full-song P0 shape/mask/values')
            a['variant'] = self.charts[ids[members[0]]]['variant']
            rows = [tensors[j] for j in members]
            a.update({k:torch.stack([r[k] for r in rows]) for k in ('mu','logvar','valid','weights')})
            a.update({k:torch.tensor([r[k] for r in rows]) for k in ('difficulty_id','level','span_beats','index','chart_id')})
            packed.append(a)
        result = dict(chart_ids=ids, groups=packed, reference_count=len(indices),
                      chart_count=len(ids), audio_count=len(groups))
        if not whole_song:
            result.update(window_groups=torch.tensor([ids.index(self.mapping[i]) for i in indices]),
                window_starts=torch.tensor([self.target.records[i]['start_tick'] for i in indices]),
                window_valid=torch.tensor(np.array(self.target.arrays['valid'][indices], copy=True)))
        return result


class SongBatches(Dataset):
    def __init__(self, target_root, audio_root, song_root, seed, *, packed=False, whole_song=False, descriptor=None):
        self.paths = tuple(map(str,(target_root,audio_root,song_root))); self.seed=seed; self.cache=None
        self.packed=packed
        self.whole_song=whole_song
        self.metadata=descriptor

    def __getstate__(self):
        return dict(self.__dict__,cache=None)

    def __getitem__(self, request):
        if self.cache is None:
            fields = () if self.whole_song else ('valid',)
            if self.metadata is None:
                target = StyleLatentCache(self.paths[0], fields=fields)
                self.cache = SongContextCache(self.paths[2], target)
            else:
                target = StyleLatentCache(self.metadata['target']['root'], fields=fields,
                    descriptor=self.metadata['target'])
                self.cache = SongContextCache(self.metadata['root'], target, descriptor=self.metadata['manifest'])
        indices,epoch,deterministic=request
        started=time.perf_counter()
        batch=self.cache.batch(indices,epoch=epoch,seed=self.seed,deterministic=deterministic,
                              whole_song=self.whole_song)
        if not self.packed:
            return batch
        from maimai_baseline.data.batch_transport import pack_batch
        result=pack_batch(batch)
        result.cpu_prepare_seconds=time.perf_counter()-started
        return result
