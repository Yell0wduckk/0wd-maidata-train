"""Producer identity and independent, strict complete-epoch Decoder checkpoints."""
import json
from pathlib import Path
import torch

from maidata.model import ENCODER_KEYS
from maidata.training import checkpoint as stage_one
from maidata.training.checkpoint import digest_file, rng_state, restore_rng, atomic_save, validate_cursor
from .plan import read_json
from .recipe import PATH_KEYS


SCHEMA = 'maidata.decoder-training.v1'
PLANNER_KEYS = ('planner_dim','planner_heads','planner_mlp_dim','planner_dropout','planner_norm_eps')


def recipe(config):
    return json.loads(json.dumps({k:v for k,v in config.items() if
        (k in ENCODER_KEYS or k in PLANNER_KEYS or k.startswith(('draft_','decoder_'))
         or k in ('target_stride_beats','chart_history_beats','complex_rhythm_min_kinds','complex_rhythm_multiplier'))
        and k not in PATH_KEYS}))


def source_identity():
    root = Path(__file__).resolve().parents[1]
    files = [root/'model.py',root/'__init__.py']
    for directory in ('audio','draft','nn','chart','chart_cache','time','physical','preparation',
                      'decoder','decoder_training','sampling','training'):
        files.extend(sorted((root/directory).glob('*.py')))
    files.extend(sorted((root/'chart/assets').glob('*.json')))
    return {p.relative_to(root).as_posix():digest_file(p) for p in files}


def data_binding(plans, config):
    # The shared metadata contract checker reads only catalog/summary/plan files.
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


def cache_producer(identity, config, binding):
    """Verify the original producer without constructing E/D/P or loading weights.

    Export already validated the complete checkpoint; its exact hash binds that
    accepted identity. Training reads only metadata and this named checkpoint.
    """
    if (identity['schema'] != stage_one.SCHEMA or identity['completed_epochs'] < 1
            or identity['step'] < 1 or identity['sha256'] != digest_file(config['decoder_train_checkpoint'])):
        raise ValueError('frozen cache producer checkpoint differs')
    _validate_producer(identity, config, binding)
    return identity


def _validate_producer(payload, config, binding):
    keys = [k for k in config if k in ENCODER_KEYS or k.startswith('draft_') or k in PLANNER_KEYS]
    current = json.loads(json.dumps({k:config[k] for k in keys}))
    if current != {k:payload['recipe'].get(k) for k in keys}:
        raise ValueError('frozen producer architecture/representation differs')
    # Retain the producer's own contract: older checkpoints carry file guards,
    # newer stage-I checkpoints carry commit provenance. Never rewrite either.
    guarded = {k:v for k,v in payload['source'].items() if
               k == 'model.py' or k.startswith(('audio/','draft/','nn/'))}
    root = Path(__file__).resolve().parents[1]
    for key,value in guarded.items():
        if not (root/key).is_file() or digest_file(root/key) != value:
            raise ValueError(f'frozen producer source differs: {key}')
    for key in ('pretrain_chart_cache','pretrain_audio_cache'):
        if payload['binding'][key] != binding[key]:
            raise ValueError('producer field/audio normalization binding differs')


def _load(module, actual):
    expected = module.state_dict()
    if (expected.keys() != actual.keys() or any(expected[k].shape != v.shape or expected[k].dtype != v.dtype
            or not torch.isfinite(v).all() for k,v in actual.items())):
        raise ValueError('model tensor inventory/shape/dtype/finite mismatch')
    module.load_state_dict(actual,strict=True)


def capture(model, opt, sched, *, epoch, step, config, binding, source, producer, steps, loaders, best, metrics):
    result = dict(schema=SCHEMA,model_state=model.decoder.state_dict(),optimizer_state=opt.state_dict(),
        scheduler_state=sched.state_dict(),completed_epochs=epoch,step=step,recipe=recipe(config),binding=binding,
        source=source,producer=producer,steps_by_epoch=steps,rng=rng_state(),
        loader_rng=[l.generator.get_state() for l in loaders],loader_started=[l._iterator is not None for l in loaders],
        best_validation_loss=best,metrics=metrics)
    validate_cursor(result,steps)
    return result


def restore(payload, model, opt, sched, *, config, binding, source, producer, steps, loaders):
    current, saved = recipe(config),payload['recipe']
    stop = 'decoder_train_stop_after_epochs'
    if (payload.get('schema') != SCHEMA or payload['source'] != source or payload['binding'] != binding
            or payload['producer'] != producer or payload['steps_by_epoch'] != steps
            or {k:v for k,v in current.items() if k != stop} != {k:v for k,v in saved.items() if k != stop}
            or current[stop] < saved[stop]):
        raise ValueError('resume source, recipe, producer or input binding changed')
    validate_cursor(payload,steps)
    if payload['completed_epochs'] > saved[stop] or len(payload['loader_rng']) != len(loaders) or len(payload['loader_started']) != len(loaders):
        raise ValueError('invalid saved epoch or loader RNG inventory')
    _load(model.decoder,payload['model_state'])
    opt.load_state_dict(payload['optimizer_state'])
    sched.load_state_dict(payload['scheduler_state'])
    for loader,state in zip(loaders,payload['loader_rng']):
        loader.generator.set_state(state.cpu())
    restore_rng(payload['rng'])
    return payload['completed_epochs'],payload['step'],payload['best_validation_loss']
