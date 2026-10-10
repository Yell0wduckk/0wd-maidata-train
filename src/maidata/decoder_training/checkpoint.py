"""Producer provenance and independent complete-epoch Decoder checkpoints."""
import json
import random
import numpy as np
from pathlib import Path
import torch

from maidata.model import ENCODER_KEYS
from maidata.training import checkpoint as stage_one
from maidata.training.checkpoint import digest_file, rng_state, atomic_save, validate_cursor
from .plan import read_json
from .recipe import PATH_KEYS, restore_optimizer


SCHEMA = 'maidata.decoder-training.v1'
PLANNER_KEYS = ('planner_dim','planner_heads','planner_mlp_dim','planner_dropout','planner_norm_eps')


def recipe(config):
    return json.loads(json.dumps({k:v for k,v in config.items() if
        (k in ENCODER_KEYS or k in PLANNER_KEYS or k.startswith(('draft_','decoder_'))
         or k in ('target_stride_beats','chart_history_beats','complex_rhythm_min_kinds','complex_rhythm_multiplier'))
        and k not in PATH_KEYS}))


def source_identity():
    # Provenance is descriptive, never a checkpoint loading requirement.
    return stage_one.source_identity()


def data_binding(plans, config):
    # Explicit offline export only; the training runner uses data_provenance.
    alias = {k.replace('decoder_train_','pretrain_'):v for k,v in config.items() if k.startswith('decoder_train_')}
    alias['samples_per_beat'] = config['samples_per_beat']
    binding = stage_one.data_binding(plans, alias)
    binding['paths'] = {k:str(Path(config[k]).resolve()) for k in PATH_KEYS if k != 'decoder_train_output'}
    fields = read_json(Path(config['decoder_train_chart_cache'])/'production-summary.json')
    hashes = {r['chart_id']:r.get('original_storage',r)['sha256'] for r in fields['results'] if r['status']=='success'}
    root = Path(config['decoder_train_physical_cache'])
    cat, summary = read_json(root/'catalog.json'),read_json(root/'production-summary.json')
    source = read_json(Path(config['decoder_train_chart_cache'])/'catalog.json')
    if (cat['schema'] != 'maidata.training-physical-original.v2' or cat['views'] != ['original']
            or cat['source_producer_recipe'] != source['producer_recipe']
            or cat['producer_recipe'] != summary['recipe']):
        raise ValueError('Physical/source producer contract differs')
    physical = {r['chart_id']:r for r in summary['results'] if r['status']=='success'}
    catalog = {r['chart_id']:r for r in cat['charts']}
    wanted = {key for p in plans for key in p.charts}
    for p in plans:
        for key,row in p.charts.items():
            if (key not in physical or key not in hashes or key not in catalog
                    or physical[key]['source_sha256'] != hashes[key]
                    or physical[key]['path'] != row['chart_cache_path']
                    or catalog[key]['path'] != row['chart_cache_path']):
                raise ValueError(f'{key}: selected field/Physical producer binding differs')
    binding['physical'] = {name:digest_file(root/name) for name in ('catalog.json','production-summary.json')}
    geometry = Path(config['decoder_train_geometry'])
    binding['geometry'] = {name:digest_file(geometry/name) for name in
        ('paths.json','runtime-path-semantics.json','runtime-path-rules.json')}
    binding['slide_data'] = digest_file(config['decoder_train_slide_data'])
    binding['route_distances'] = digest_file(config['decoder_train_route_distances'])
    return binding, {k:hashes[k] for k in wanted}


def load_producer(model, config, binding):
    path = config['decoder_train_checkpoint']
    payload = torch.load(path,map_location='cpu',weights_only=False)
    if payload.get('schema') != stage_one.SCHEMA:
        raise ValueError('initialization requires a compatible stage-I complete-epoch checkpoint')
    validate_cursor(payload,payload['steps_by_epoch'])
    _validate_producer(payload, config, binding)
    for name in ('encoder','draft','planner'):
        state = {k[len(name)+1:]:v for k,v in payload['model_state'].items() if k.startswith(name+'.')}
        _load(getattr(model,name),state)
    return dict(sha256=digest_file(path),schema=payload['schema'],completed_epochs=payload['completed_epochs'],
                step=payload['step'],recipe=payload['recipe'],source=payload['source'],binding=payload['binding'])


def data_provenance(config):
    """Accepted inputs are selected by the operator, without startup revalidation."""
    return dict(paths={k: str(Path(config[k]).resolve()) for k in PATH_KEYS
                       if k != 'decoder_train_output' and config[k] is not None})


def cache_producer(identity, config, binding):
    return identity


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available() and state['cuda']:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def _validate_producer(payload, config, binding):
    keys = [k for k in config if k in ENCODER_KEYS or k.startswith('draft_') or k in PLANNER_KEYS]
    current = json.loads(json.dumps({k:config[k] for k in keys}))
    if current != {k:payload['recipe'].get(k) for k in keys}:
        raise ValueError('frozen producer architecture/representation differs')
    for key in ('pretrain_chart_cache','pretrain_audio_cache'):
        if payload['binding'][key] != binding[key]:
            raise ValueError('producer field/audio normalization binding differs')


def _load(module, actual):
    module.load_state_dict(actual,strict=True)


def capture(model, opt, sched, *, epoch, step, config, binding, source, producer, steps, loaders, best, metrics):
    result = dict(schema=SCHEMA,model_state=model.decoder.state_dict(),optimizer_state=opt.state_dict(),
        scheduler_state=sched.state_dict(),completed_epochs=epoch,step=step,recipe=recipe(config),binding=binding,
        source=source,producer=producer,steps_by_epoch=steps,rng=rng_state(),
        loader_rng=[l.generator.get_state() for l in loaders],loader_started=[l._iterator is not None for l in loaders],
        best_validation_loss=best,metrics=metrics)
    return result


def restore(payload, model, opt, sched, *, config, binding, source, producer, steps, loaders):
    # The user/training operator selects the checkpoint. Source, budget, recipe,
    # plan and cache provenance are recorded, not used as resume vetoes.
    _load(model.decoder,payload['model_state'])
    restore_optimizer(opt, payload['optimizer_state'])
    sched.load_state_dict(payload['scheduler_state'])
    for loader,state in zip(loaders,payload['loader_rng']):
        loader.generator.set_state(state.cpu())
    restore_rng(payload['rng'])
    return payload['completed_epochs'],payload['step'],payload['best_validation_loss']
