"""Produce frozen A timing candidates while reusing the selected P0 memory."""
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from maimai_baseline.data.decoder_context import SCHEMA, CONTRACT, NATIVE_SCHEMA, NATIVE_CONTRACT, DecoderContextCache, _json, validate_candidates, validate_native_candidates, p0_identity
from maimai_baseline.data.style_cache import write_json


@torch.no_grad()
def encode_candidates(model, batch):
    from maimai_baseline.data.song_audio import encode_batch
    full = encode_batch(model, batch['song_audio'])
    row = batch['continuous']
    context = model.encode_audio(row['audio'], row['audio_padding_mask'], row['frame_beats'],
        row['frame_bpms'], start_tick=row['audio_start_tick'],
        interpolation_geometry=row['interpolation_geometry'], song_audio=full)
    result = dict(candidate_keys=context.candidate_keys[0].float().cpu(),
                  candidate_padding_mask=context.candidate_padding_mask[0].cpu(),
                  candidate_beats=context.candidate_beats[0].float().cpu())
    validate_candidates(*result.values())
    return result


def _paths(args, names=('song_chart_cache', 'song_audio_cache', 'bpm_audio_manifest', 'route_artifact')):
    root = getattr(args, 'dataset_root', None)
    root = None if root is None else Path(root).expanduser().resolve()
    manifest = getattr(args, 'dataset_manifest', None)
    if manifest is None and root is not None:
        manifest = next((root/n for n in ('dataset-v10.json', 'dataset.json') if (root/n).is_file()), None)
    mapped = {}
    if manifest is not None:
        manifest = Path(manifest).expanduser().resolve()
        root = root or manifest.parent
        mapped = {k.replace('-', '_'):v for k,v in _json(manifest).get('inputs', {}).items()}
    for name in names:
        value = getattr(args, name, None)
        explicit = value is not None
        if value is None:
            value = mapped.get(name)
        if value is None:
            raise ValueError('missing decoder candidate production input: --'+name.replace('_', '-'))
        path = Path(value).expanduser()
        if not explicit and root is not None and not path.is_absolute(): path = root/path
        setattr(args, name, path.resolve())
    return args


def build_decoder_context_cache(args):
    from maimai_baseline.checkpoint import load_frozen_stage_a
    from maimai_baseline.data.frozen_context import SCHEMA as P0_SCHEMA, NATIVE_SCHEMA as NATIVE_P0_SCHEMA
    from maimai_baseline.data.style_audio_cache import worker_init
    from maimai_baseline.runtime_versions import stamp
    from maimai_baseline.style_contract import is_frozen
    from maimai_baseline.training.frozen_context import ContinuousAudioJobs
    from maimai_baseline.training.steps import _device, _move
    from maimai_dataset.integrity import file_sha256
    p0_root = Path(args.song_context_cache).expanduser().resolve()
    p0 = _json(p0_root/'manifest.json')
    if p0.get('schema') == NATIVE_P0_SCHEMA:
        return _build_native(args, p0_root, p0)
    args = _paths(args)
    if p0.get('schema') != P0_SCHEMA or not p0.get('ready'):
        raise ValueError('candidate producer requires ready frozen A continuous P0')
    from maimai_baseline.data.frozen_latent import FrozenLatentCache
    from maimai_baseline.runtime_versions import operational
    target = FrozenLatentCache(args.latent_cache, fields=())
    if operational(p0.get('latent_data')) != operational(target.identity):
        raise ValueError('decoder candidate cache and P0 require the selected latent data')
    variants = list(p0['audio'])
    if not variants: raise ValueError('no frozen A audio variants')
    output = Path(args.output).expanduser().resolve()
    if output.exists(): raise ValueError('refusing to overwrite decoder candidate cache')
    device = _device(getattr(args, 'device', 'cuda'))
    model, parent = load_frozen_stage_a(Path(args.stage_a_checkpoint), route_artifact=args.route_artifact)
    if not is_frozen(model.config):
        raise ValueError('candidate producer requires deterministic-mu frozen A')
    producer = p0.get('producer', {})
    if any(producer.get(k) is not None and producer[k] != parent.get(k) for k in ('epoch', 'step')):
        raise ValueError('candidate producer and reused P0 require the same frozen A training state')
    model.to(device).eval().requires_grad_(False)
    jobs = ContinuousAudioJobs(args.bpm_audio_manifest, args.song_chart_cache, args.song_audio_cache, variants)
    loader = DataLoader(jobs, batch_size=None, num_workers=12, worker_init_fn=worker_init,
                        generator=torch.Generator().manual_seed(20260811))
    output.mkdir(parents=True)
    write_json(output/'manifest.json', dict(schema=SCHEMA, ready=False))
    audio = {}; started = time.perf_counter()
    try:
        for raw in loader:
            values = encode_candidates(model, _move(raw, device))
            path = output/f"candidates-{raw['index']}.npz"
            np.savez(path, **{k:v.numpy() for k,v in values.items()})
            key = raw['variant']
            audio[key] = dict(file=path.name, bytes=path.stat().st_size, sha256=file_sha256(path),
                              frames=len(values['candidate_keys']), width=values['candidate_keys'].shape[1])
        manifest = dict(**stamp(), schema=SCHEMA, ready=True, contract=CONTRACT, audio=audio,
            p0_root=str(p0_root), p0_manifest_identity=p0_identity(p0),
            producer=dict(checkpoint=str(Path(args.stage_a_checkpoint).resolve()), epoch=parent.get('epoch'),
                          step=parent.get('step'), runtime_version=parent.get('runtime_version')))
        reader = DecoderContextCache(output, p0_root, descriptor=dict(manifest=manifest, p0_manifest=p0))
        for key in variants:
            reader.batch([key], torch.tensor([0]), torch.tensor([0]))
        write_json(output/'manifest.json', dict(manifest, finite=True, loaded_audio_variants=len(variants)))
    except Exception as error:
        write_json(output/'manifest.json', dict(schema=SCHEMA, ready=False, audio=audio, loader_error=str(error)))
        raise
    return dict(operation='build-decoder-context-cache', output=str(output), audio_variants=len(variants),
                seconds=time.perf_counter()-started, workers=12, audio_encoder_calls=len(variants),
                persisted=list(CONTRACT['persisted']))


def _build_native(args, p0_root, p0):
    from maimai_baseline.checkpoint import load_frozen_stage_a
    from maimai_baseline.data.frozen_latent import FrozenLatentCache
    from maimai_baseline.data.frozen_context import FrozenSongContext
    from maimai_baseline.data.teacher_audio import project_candidates, context_dict
    from maimai_baseline.model.common import AudioContext
    from maimai_baseline.runtime_versions import operational, stamp
    from maimai_baseline.style_contract import NATIVE_STYLE_ID
    from maimai_baseline.training.steps import _device, _move
    from maimai_dataset.integrity import file_sha256
    args = _paths(args, names=('teacher_audio_cache',))
    teacher_path = getattr(args, 'teacher_audio_cache', None)
    if teacher_path is None:
        raise ValueError('native candidate production requires --teacher-audio-cache')
    target = FrozenLatentCache(args.latent_cache, fields=())
    context = FrozenSongContext(p0_root, target, teacher_audio_cache=teacher_path)
    model, parent = load_frozen_stage_a(Path(args.stage_a_checkpoint), route_artifact=args.route_artifact)
    if model.config.chart_style != NATIVE_STYLE_ID:
        raise ValueError('native candidates require the native-audio frozen A')
    if operational(parent.get('baseline_manifest', {}).get('teacher_audio')) != operational(context.teacher.identity):
        raise ValueError('native candidates and frozen A teacher audio differ')
    if operational(parent.get('baseline_manifest', {}).get('frozen_latent')) != operational(target.identity):
        raise ValueError('native candidates and frozen A latent data differ')
    producer = p0.get('producer', {})
    if any(producer.get(k) is not None and producer[k] != parent.get(k) for k in ('epoch', 'step')):
        raise ValueError('native candidates and reused P0 require the same frozen A training state')
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise ValueError('refusing to overwrite decoder candidate cache')
    device = _device(getattr(args, 'device', 'cuda'))
    model.to(device).eval().requires_grad_(False)
    output.mkdir(parents=True)
    write_json(output/'manifest.json', dict(schema=NATIVE_SCHEMA, ready=False))
    audio = {}; started = time.perf_counter()
    try:
        for index, key in enumerate(p0['audio']):
            starts = list(context.teacher.manifest['variants'][key]['starts'])
            local, geometry = context.teacher.window_context(key, starts)
            with torch.no_grad():
                encoded = project_candidates(model, AudioContext(**_move(context_dict(local), device)), _move(geometry, device))
            values = dict(candidate_keys=encoded.candidate_keys.float().cpu(),
                candidate_padding_mask=encoded.candidate_padding_mask.cpu(),
                candidate_beats=encoded.candidate_beats.float().cpu())
            validate_native_candidates(values, starts)
            path = output/f'candidates-{index}.npz'
            np.savez(path, **{k:v.numpy() for k,v in values.items()})
            audio[key] = dict(file=path.name, bytes=path.stat().st_size, sha256=file_sha256(path),
                starts=starts, windows=len(starts), frames=values['candidate_keys'].shape[1], width=256)
        manifest = dict(**stamp(), schema=NATIVE_SCHEMA, ready=True, contract=NATIVE_CONTRACT, audio=audio,
            p0_root=str(p0_root), p0_manifest_identity=p0_identity(p0), teacher_audio=context.teacher.identity,
            producer=dict(checkpoint=str(Path(args.stage_a_checkpoint).resolve()), epoch=parent.get('epoch'),
                          step=parent.get('step'), runtime_version=parent.get('runtime_version')))
        DecoderContextCache(output, context, descriptor=dict(manifest=manifest, p0_manifest=p0))
        write_json(output/'manifest.json', dict(manifest, finite=True, loaded_audio_variants=len(audio)))
    except Exception as error:
        write_json(output/'manifest.json', dict(schema=NATIVE_SCHEMA, ready=False, audio=audio, loader_error=str(error)))
        raise
    return dict(operation='build-decoder-context-cache', output=str(output), audio_variants=len(audio),
        seconds=time.perf_counter()-started, workers=12, audio_encoder_calls=0, song_audio_encoder_calls=0,
        candidate_projection_calls=len(audio), persisted=list(NATIVE_CONTRACT['persisted']))
