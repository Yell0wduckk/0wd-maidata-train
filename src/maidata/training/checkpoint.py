"""Complete-epoch state and recipe recovery; source and inputs are provenance only."""
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from maidata.model import ENCODER_KEYS, with_legacy_defaults
from maidata.draft.route_acceptance import route_loss_recipe
from .data import read_json


SCHEMA = 'maidata.encoder-draft-training.v3'
PATH_KEYS = ('pretrain_train_plan', 'pretrain_validation_plan', 'pretrain_chart_cache',
             'pretrain_audio_cache', 'pretrain_mel_root', 'pretrain_source_root', 'pretrain_output', 'pretrain_p0_cache')


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recipe(config):
    # Unused Decoder/sampling knobs cannot change this training contract.
    value = {k: v for k, v in config.items()
        if (k in ENCODER_KEYS or k.startswith(('draft_', 'reconstruction_', 'planner_', 'pretrain_')))
        and k not in PATH_KEYS
        and k not in ('reconstruction_route_loss', 'reconstruction_route_acceptance')}
    value.update(route_loss_recipe(config))
    return json.loads(json.dumps(value, sort_keys=True))


def source_identity():
    """Optional version note only; never a checkpoint compatibility condition."""
    import subprocess
    root = Path(__file__).resolve().parents[3]
    marker = root / 'SOURCE_COMMIT'
    if marker.is_file():
        return {'commit': marker.read_text(encoding='utf-8').strip()}
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root,
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {'commit': commit}



def data_provenance(config):
    """Record configured inputs without opening, hashing or validating assets."""
    return dict(paths={k: str(Path(config[k]).resolve()) for k in PATH_KEYS
                       if k != 'pretrain_output' and config[k] is not None})


def data_binding(plans, config):
    """Legacy shared metadata binding for Decoder; stage one does not call this."""
    result = {'plans': [], 'paths': {k: str(Path(config[k]).resolve()) for k in PATH_KEYS
                                    if k != 'pretrain_output' and config[k] is not None}}
    for plan in plans:
        names = ('plan.json', *plan.manifest['files'].values())
        result['plans'].append({name: digest_file(plan.root / name) for name in names})
    for key in ('pretrain_chart_cache', 'pretrain_audio_cache'):
        root = Path(config[key])
        result[key] = {name: digest_file(root / name) for name in ('catalog.json', 'production-summary.json')}
    charts = read_json(Path(config['pretrain_chart_cache']) / 'catalog.json')
    audio = read_json(Path(config['pretrain_audio_cache']) / 'catalog.json')
    if charts['schema'] != 'maidata.full-chart-original.v2' or charts['view'] != 'original':
        raise ValueError('incorrect chart cache representation')
    chart_index = {c['chart_id']: c for c in charts['charts']}
    for plan in plans:
        if charts['producer_recipe'] != plan.manifest['sources']['chart_recipe']:
            raise ValueError('chart cache producer recipe differs from plan')
        for variant, group in plan.groups.items():
            actual = audio['variants'][variant]
            if (actual['asset'] != group['audio_asset'] or actual['timing'] != group['timing']
                    or actual['samples_per_beat'] != config['samples_per_beat']):
                raise ValueError('audio producer/plan identity differs')
        for key, chart in plan.charts.items():
            actual = chart_index[key]
            if actual['path'] != chart['chart_cache_path'] or actual['audio_variant'] != chart['audio_variant']:
                raise ValueError('chart catalog/plan identity differs')
    if set(plans[0].groups) & set(plans[1].groups):
        raise ValueError('train/validation audio groups overlap')
    if set(c['song_id'] for c in plans[0].charts.values()) & set(c['song_id'] for c in plans[1].charts.values()):
        raise ValueError('train/validation songs overlap')
    return result


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    if set(state) != {'python', 'numpy', 'torch', 'cuda'}:
        raise ValueError('checkpoint is missing complete RNG state')
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available():
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA RNG device inventory changed')
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])
    elif state['cuda']:
        raise ValueError('cannot resume CUDA training on CPU')


def atomic_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def validate_cursor(payload, steps):
    epoch, step = payload['completed_epochs'], payload['step']
    if (type(epoch) is not int or not 1 <= epoch <= len(steps) or type(step) is not int
            or step != sum(steps[:epoch]) or payload['scheduler_state']['last_epoch'] != step):
        raise ValueError('checkpoint is not at a complete epoch boundary')


def capture(model, optimizer, scheduler, *, completed_epochs, step, config, binding, source,
            steps, loaders, best, metrics):
    payload = dict(schema=SCHEMA, model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict(), completed_epochs=completed_epochs, step=step,
        recipe=recipe(config), binding=binding, source=source, steps_by_epoch=steps,
        rng=rng_state(), loader_rng=[loader.generator.get_state() for loader in loaders],
        loader_started=[loader._iterator is not None for loader in loaders],
        best_validation_loss=best, metrics=metrics, audio_producer=model.audio_producer)
    validate_cursor(payload, steps)
    return payload


def restore(payload, model, optimizer, scheduler, *, config, binding, source, steps, loaders):
    if payload.get('schema') != SCHEMA:
        raise ValueError('resume architecture, recipe or scheduler steps changed: split Drafter requires v3 checkpoint')
    current, saved = with_legacy_defaults(recipe(config)), with_legacy_defaults(payload['recipe'])
    current.setdefault('reconstruction_route_vocabulary', 'route71')
    saved.setdefault('reconstruction_route_vocabulary', 'route71')
    saved.setdefault('reconstruction_route_loss', 'ce')
    current.setdefault('planner_span_beats', 16)
    saved.setdefault('planner_span_beats', 16)
    for key, default in (('planner_layout', 'legacy'), ('planner_stride_beats', None),
                         ('draft_z_anchor_mode', 'block-center'), ('reconstruction_span_beats', None)):
        current.setdefault(key, default)
        saved.setdefault(key, default)
    saved.setdefault('pretrain_audio_mode', 'joint')
    saved.setdefault('draft_z_stride_beats', 1)  # Legacy split checkpoints are per-beat.
    if payload.get('audio_producer') != model.audio_producer:
        raise ValueError('resume frozen P0 producer changed')
    stop = 'pretrain_stop_after_epochs'
    if (payload['steps_by_epoch'] != steps
            or {k:v for k,v in current.items() if k != stop} != {k:v for k,v in saved.items() if k != stop}
            or current[stop] < saved[stop]):
        raise ValueError('resume architecture, recipe or scheduler steps changed')
    validate_cursor(payload, steps)
    if payload['completed_epochs'] > saved[stop]:
        raise ValueError('saved epoch exceeds the original absolute stop')
    expected, actual = model.state_dict(), payload['model_state']
    if (expected.keys() != actual.keys() or any(expected[k].shape != v.shape or expected[k].dtype != v.dtype
            or not torch.isfinite(v).all() for k, v in actual.items())):
        raise ValueError('model tensor inventory/shape/dtype/finite mismatch')
    if len(payload['loader_rng']) != len(loaders) or len(payload['loader_started']) != len(loaders):
        raise ValueError('DataLoader RNG inventory differs')
    model.load_state_dict(actual, strict=True)
    optimizer.load_state_dict(payload['optimizer_state'])
    scheduler.load_state_dict(payload['scheduler_state'])
    for loader, state in zip(loaders, payload['loader_rng']):
        loader.generator.set_state(state.cpu())
    restore_rng(payload['rng'])
    return payload['completed_epochs'], payload['step'], payload['best_validation_loss']
