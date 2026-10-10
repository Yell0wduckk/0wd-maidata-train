"""Bind native teacher P0 or produce continuous full-song P0 for older A models."""
from pathlib import Path
import json
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from maimai_baseline.data.frozen_context import SCHEMA, FrozenSongContext
from maimai_baseline.data.style_cache import write_json
from maimai_baseline.runtime_versions import stamp


def _json(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def context_layout(cache, chart_manifest):
    lookup = {tuple(c['key']): c for c in chart_manifest['charts']}
    charts, mapping, indices = [], [], {}
    for i, record in enumerate(cache.records):
        key = (record['role'], str(record['record_id']), record['view_id'])
        if key not in lookup:
            raise ValueError('selected frozen latent chart missing from complete-chart metadata')
        source = lookup[key]
        if key not in indices:
            indices[key] = len(charts)
            blocks = cache.full_charts[key]['blocks']
            charts.append(dict(key=list(key), rows=[], blocks=int(blocks),
                song_end_tick=record['song_end_tick'], difficulty_id=record['difficulty_id'],
                level=record['level_value'], audio_key=str(source['variant'])))
        ci = indices[key]
        charts[ci]['rows'].append(i)
        mapping.append(ci)
    return charts, mapping


class ContinuousAudioJobs(Dataset):
    def __init__(self, bpm_manifest, song_chart_cache, song_audio_cache, variants):
        self.path = str(bpm_manifest)
        self.bpm = _json(bpm_manifest)
        self.chart_root, self.audio_root = str(song_chart_cache), str(song_audio_cache)
        self.variants = list(variants)
        self.song_audio = None

    def __getitem__(self, index):
        from maimai_baseline.data.fullaudio import arrays, pack
        from maimai_baseline.data.song_audio import SongAudioCache
        from maimai_baseline.data.song_chart_compact import SCHEMA as COMPACT_SCHEMA
        from maimai_dataset.timing import TimingMap
        variant = self.variants[index]
        info = self.bpm['variants'][variant]
        root = Path(self.path).parent
        feature = np.load(root/info['feature_path'], mmap_mode='r', allow_pickle=False)
        with np.load(root/info['coordinates_path'], allow_pickle=False) as z:
            coords = {k: z[k].copy() for k in z.files}
        timing = TimingMap.from_dict(info['timing'])
        if _json(Path(self.chart_root)/'manifest.json')['schema'] == COMPACT_SCHEMA:
            from maimai_baseline.time.offline import store_for
            timing = store_for(self.chart_root).bind(timing)
        row = arrays(feature, coords, timing, int(info['num_samples']))
        if self.song_audio is None:
            self.song_audio = SongAudioCache(self.audio_root)
        return dict(index=index, variant=variant, continuous=pack([row],[0]),
                    song_audio=self.song_audio.batch([variant], include_fusion=False))

    def __len__(self):
        return len(self.variants)


@torch.no_grad()
def encode_p0(model, batch):
    """Use the exact A graph: one song encoder and one continuous audio call."""
    from maimai_baseline.data.song_audio import encode_batch
    full = encode_batch(model, batch['song_audio'])
    row = batch['continuous']
    context = model.encode_audio(row['audio'], row['audio_padding_mask'], row['frame_beats'],
        row['frame_bpms'], start_tick=row['audio_start_tick'],
        interpolation_geometry=row['interpolation_geometry'], song_audio=full)
    return dict(audio=context.memories[-1][0].float().cpu(),
                audio_valid=(~context.padding_masks[-1][0]).cpu(), audio_beats=context.beats[-1][0].float().cpu())


def accept_context(output,cache,manifest):
    """Read the new P0 with its normal consumer, one complete audio group at a time."""
    context=FrozenSongContext(output,cache,descriptor=manifest)
    groups={}
    for chart in context.charts:
        groups.setdefault(chart['audio_key'],[]).extend(chart['rows'])
    charts=references=audio=0
    for rows in groups.values():
        batch=context.batch(rows)
        charts+=batch['chart_count']; references+=batch['reference_count']; audio+=batch['audio_count']
    if (charts!=len(context.charts) or references!=len(cache.records) or audio!=len(manifest['audio'])):
        raise ValueError('new continuous P0 consumer coverage mismatch')
    return dict(loaded_charts=charts,loaded_references=references,loaded_audio_variants=audio)


def finalize_context(output,cache,manifest):
    """Persist the new cache's actual consumer result without accepting failed bytes."""
    path=Path(output)/'manifest.json'
    pending=dict(manifest,ready=False)
    write_json(path,pending)
    try:
        coverage=accept_context(output,cache,dict(pending,ready=True))
    except Exception as error:
        write_json(path,dict(pending,loader_error=str(error)))
        raise
    write_json(path,dict(pending,ready=True,finite=True,**coverage))
    return coverage


def build_song_context(args):
    from maimai_baseline.checkpoint import load_frozen_stage_a
    from maimai_baseline.data.frozen_latent import FrozenLatentCache
    from maimai_baseline.data.style_audio_cache import worker_init
    from maimai_baseline.style_contract import is_frozen, NATIVE_STYLE_ID
    from maimai_baseline.training.steps import _device, _move
    from maimai_dataset.integrity import file_sha256

    from maimai_baseline.training.decoder_context import _paths
    args = _paths(args, names=('song_chart_cache',))
    cache = FrozenLatentCache(getattr(args, 'latent_cache', None) or args.cache)
    chart_manifest = _json(Path(args.song_chart_cache)/'manifest.json')
    charts, mapping = context_layout(cache, chart_manifest)
    variants = list(dict.fromkeys(c['audio_key'] for c in charts))
    if not variants:
        raise ValueError('no selected continuous audio variants')
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError('refusing to overwrite frozen A P0 cache')
    model, parent = load_frozen_stage_a(Path(args.stage_a_checkpoint), route_artifact=Path(args.route_artifact))
    if not is_frozen(model.config):
        raise ValueError('continuous P0 producer requires the frozen-mu stage A model')
    from maimai_baseline.runtime_versions import operational
    if operational(parent.get('baseline_manifest',{}).get('frozen_latent')) != operational(cache.identity):
        raise ValueError('continuous P0 producer latent data differs from stage A binding')
    if model.config.chart_style == NATIVE_STYLE_ID:
        return _native_context(args, cache, charts, mapping, variants, parent, output)
    if getattr(args, 'teacher_audio_cache', None) is not None:
        raise ValueError('continuous P0 producer cannot consume native teacher audio')
    args = _paths(args)
    device = _device(getattr(args, 'device', 'cuda'))
    model.to(device).eval().requires_grad_(False)
    jobs = ContinuousAudioJobs(args.bpm_audio_manifest, args.song_chart_cache, args.song_audio_cache, variants)
    loader = DataLoader(jobs, batch_size=None, num_workers=12, worker_init_fn=worker_init,
                        generator=torch.Generator().manual_seed(20260811))
    output.mkdir(parents=True)
    write_json(output/'manifest.json', dict(schema=SCHEMA, ready=False))
    audio = {}
    started = time.perf_counter()
    for raw in loader:
        batch = _move(raw, device)
        values = encode_p0(model, batch)
        n = len(values['audio'])
        if (values['audio'].shape != (n,256) or values['audio_valid'].shape != (n,)
                or values['audio_beats'].shape != (n,) or values['audio_valid'].dtype != torch.bool
                or n < 1 or not bool(torch.isfinite(values['audio']).all() & torch.isfinite(values['audio_beats']).all())):
            raise ValueError('new continuous P0 tensor layout/finite mismatch')
        variant = raw['variant']
        path = output/f"audio-{raw['index']}.npz"
        np.savez(path, **{k: v.numpy() for k,v in values.items()})
        # Hash only the new bytes; the normal consumer reads them once below.
        audio[variant] = dict(file=path.name, bytes=path.stat().st_size, sha256=file_sha256(path), frames=n)
    manifest = dict(**stamp(), schema=SCHEMA, ready=True, charts=charts, record_to_chart=mapping, audio=audio,
        latent_data=cache.identity, producer=dict(checkpoint=str(Path(args.stage_a_checkpoint).resolve()),
            epoch=parent.get('epoch'), step=parent.get('step'), runtime_version=parent.get('runtime_version')),
        contract=dict(audio='continuous-full-song-P0', width=256, coordinates='absolute-beats', target='deterministic-mu'))
    coverage=finalize_context(output,cache,manifest)
    return dict(operation='build-song-context', output=str(output), charts=len(charts),
                audio_variants=len(audio), seconds=time.perf_counter()-started, workers=12,**coverage)


def _native_context(args, cache, charts, mapping, variants, parent, output):
    """Bind accepted native bytes; no duplicate P0 production or payload scan."""
    from maimai_baseline.data.teacher_audio import TeacherAudioCache
    from maimai_baseline.data.frozen_context import NATIVE_SCHEMA, NATIVE_CONTRACT
    from maimai_baseline.runtime_versions import operational
    from maimai_baseline.time.audio_end import audio_end
    from maimai_dataset.timing import TimingMap
    from maimai_baseline.training.decoder_context import _paths
    args = _paths(args, names=('teacher_audio_cache',))
    path = getattr(args, 'teacher_audio_cache', None)
    if path is None:
        raise ValueError('native song context requires --teacher-audio-cache')
    teacher = TeacherAudioCache(path)
    if operational(parent.get('baseline_manifest', {}).get('teacher_audio')) != operational(teacher.identity):
        raise ValueError('native song context teacher audio differs from native A')
    audio = {}
    for key in variants:
        if key not in teacher.manifest['variants']:
            raise ValueError('native teacher audio lacks a selected variant')
        info = teacher.manifest['variants'][key]
        audio[key] = dict(teacher_variant=key, windows=len(info['starts']), frames=len(info['starts'])*128)
    for chart in charts:
        info = teacher.manifest['variants'][chart['audio_key']]
        chart['span_beats'] = audio_end(TimingMap.from_dict(info['timing']), int(info['num_samples'])).onset_tick / 96.
        chart['condition_blocks'] = chart['blocks']
    manifest = dict(**stamp(), schema=NATIVE_SCHEMA, ready=True, charts=charts,
        record_to_chart=mapping, audio=audio, latent_data=cache.identity,
        teacher_audio_root=str(teacher.root), teacher_audio=teacher.identity,
        producer=dict(checkpoint=str(Path(args.stage_a_checkpoint).resolve()), epoch=parent.get('epoch'),
                      step=parent.get('step'), runtime_version=parent.get('runtime_version')),
        contract=NATIVE_CONTRACT)
    # The new artifact contains only mappings. Validate their actual consumer
    # contract without re-reading accepted teacher/latent payloads.
    FrozenSongContext(output, cache, descriptor=manifest, teacher_audio_cache=teacher.root)
    output.mkdir(parents=True)
    write_json(output/'manifest.json', manifest)
    return dict(operation='build-song-context', output=str(output), charts=len(charts),
        audio_variants=len(audio), references=len(mapping), audio_encoder_calls=0,
        song_audio_encoder_calls=0, reused_teacher_audio=True)
