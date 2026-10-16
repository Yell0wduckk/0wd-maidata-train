"""One complete-epoch checkpoint for the shared E/D/R/P/M graph."""
import json

from maidata.model import ENCODER_KEYS, with_legacy_defaults
from maidata.draft.route_acceptance import route_loss_recipe
from maidata.training.checkpoint import (atomic_save, rng_state, restore_rng,
                                        source_identity, validate_cursor)
from maidata.decoder_training.recipe import restore_optimizer
from .recipe import PATH_KEYS

SCHEMA = 'maidata.joint-training.v1'
ARCHITECTURE = 'maidata.joint-encoder-draft-decoder.v1'


def recipe(config):
    value = {key: value for key, value in config.items()
        if (key in ENCODER_KEYS or key.startswith(('draft_', 'reconstruction_', 'planner_',
                                                 'joint_train_', 'decoder_'))
            or key in ('target_stride_beats', 'chart_history_beats', 'complex_rhythm_min_kinds',
                       'complex_rhythm_multiplier', 'pretrain_audio_groups_per_batch',
                       'pretrain_token_budget', 'pretrain_batch_reference_rows',
                       'pretrain_audio_token_weight', 'pretrain_row_token_weight',
                       'pretrain_gradient_accumulation_steps'))
        and key not in PATH_KEYS and not key.endswith(('_checkpoint', '_cache', '_output'))
        and key not in ('reconstruction_route_loss', 'reconstruction_route_acceptance')}
    value.update(route_loss_recipe(config))
    return json.loads(json.dumps(value, sort_keys=True))


def capture(model, opt, sched, *, epoch, step, config, binding, source, steps, loaders, best, metrics):
    payload = dict(schema=SCHEMA, architecture=ARCHITECTURE, model_state=model.state_dict(),
        optimizer_state=opt.state_dict(), scheduler_state=sched.state_dict(), completed_epochs=epoch,
        step=step, recipe=recipe(config), binding=binding, source=source, steps_by_epoch=steps,
        rng=rng_state(), loader_rng=[loader.generator.get_state() for loader in loaders],
        loader_started=[loader._iterator is not None for loader in loaders],
        best_validation_loss=best, metrics=metrics)
    validate_cursor(payload, steps)
    return payload


def single_plan16_upgrade(payload, config, *, binding, steps):
    """Validate one explicit objective migration without rewriting saved semantics."""
    current, saved = with_legacy_defaults(recipe(config)), with_legacy_defaults(payload['recipe'])
    if (saved.get('planner_layout') != 'centered-overlap'
            or current.get('planner_layout') != 'window-single'
            or saved.get('planner_stride_beats') != 4 or current.get('planner_stride_beats') is not None):
        raise ValueError('single Plan16 upgrade requires centered-overlap/stride4 -> window-single/no Plan stride')
    for key, expected in (('planner_span_beats', 16), ('draft_z_anchor_mode', 'integer'),
                          ('draft_z_stride_beats', 1), ('joint_train_z_stride_beats', 1),
                          ('reconstruction_span_beats', 4), ('target_beats', 16)):
        if saved.get(key) != expected or current.get(key) != expected:
            raise ValueError('single Plan16 upgrade must preserve integer Z1/R4/Plan16: '+key)
    allowed = {'planner_layout', 'planner_stride_beats'}
    if set(saved) != set(current):
        raise ValueError('single Plan16 upgrade recipe field inventory differs')
    differences = sorted(key for key in set(saved) | set(current)
                         if key not in allowed and saved.get(key) != current.get(key))
    if differences:
        raise ValueError('single Plan16 upgrade cannot change other recipe fields: '+', '.join(differences))
    if list(steps) != payload['steps_by_epoch']:
        raise ValueError('single Plan16 upgrade must preserve every epoch update count')
    previous_binding = payload['binding']
    if set(previous_binding) != set(binding) or 'paths' not in binding:
        raise ValueError('single Plan16 upgrade input binding inventory differs')
    old_paths, new_paths = previous_binding['paths'], binding['paths']
    gt_key = 'decoder_train_gt_plan_cache'
    if (set(old_paths) != set(new_paths) or gt_key not in new_paths
            or any(old_paths[key] != new_paths[key] for key in new_paths if key != gt_key)
            or any(previous_binding[key] != binding[key] for key in binding if key != 'paths')):
        raise ValueError('single Plan16 upgrade may change only the GT Plan cache input path')
    if not old_paths[gt_key] or not new_paths[gt_key] or old_paths[gt_key] == new_paths[gt_key]:
        raise ValueError('single Plan16 upgrade requires the separate existing window Plan16 cache')
    return dict(from_layout=saved['planner_layout'], to_layout=current['planner_layout'],
                from_gt_plan_cache=old_paths[gt_key], to_gt_plan_cache=new_paths[gt_key],
                source_checkpoint=payload.get('source'))


def restore(payload, model, opt, sched, *, loaders, upgrade_z_normalization=False,
            upgrade_single_plan16=False, **_):
    if payload.get('schema') != SCHEMA or payload.get('architecture') != ARCHITECTURE:
        raise ValueError('joint resume requires its complete E/D/R/P/M v1 checkpoint')
    validate_cursor(payload, payload['steps_by_epoch'])
    if len(payload['loader_rng']) != len(loaders) or len(payload['loader_started']) != len(loaders):
        raise ValueError('joint checkpoint DataLoader RNG inventory differs')
    if upgrade_single_plan16:
        if upgrade_z_normalization:
            raise ValueError('single Plan16 upgrade cannot also change Z normalization')
        if not all(key in _ for key in ('config', 'binding', 'steps')):
            raise ValueError('single Plan16 upgrade requires current recipe, input binding and epoch schedule')
        single_plan16_upgrade(payload, _['config'], binding=_['binding'], steps=_['steps'])
    if 'config' in _:
        current, saved = with_legacy_defaults(recipe(_['config'])), with_legacy_defaults(payload['recipe'])
        saved.setdefault('reconstruction_route_loss', 'ce')
        saved.setdefault('planner_span_beats', 16)
        saved.setdefault('planner_layout', 'legacy')
        saved.setdefault('planner_stride_beats', None)
        saved.setdefault('draft_z_anchor_mode', 'block-center')
        saved.setdefault('reconstruction_span_beats', None)
        previous_z_norm = saved.get('decoder_z_normalization', False)
        current_z_norm = current.get('decoder_z_normalization', False)
        if upgrade_z_normalization and not (previous_z_norm is False and current_z_norm is True):
            raise ValueError('Z normalization upgrade requires an unnormalized checkpoint and normalized model')
        keys = [key for key in current if key in ENCODER_KEYS or
                key.startswith(('draft_', 'reconstruction_', 'planner_')) or
                key.startswith('decoder_') and not key.startswith('decoder_train_')]
        def differs(key):
            if upgrade_single_plan16 and key in ('planner_layout', 'planner_stride_beats'):
                return False
            if key == 'reconstruction_route_vocabulary':
                return current[key] != saved.get(key, 'route71')
            if key == 'decoder_z_normalization':
                return current_z_norm != previous_z_norm and not upgrade_z_normalization
            return current[key] != saved.get(key)
        if any(differs(key) for key in keys):
            raise ValueError('joint checkpoint architecture/representation differs')
    model.load_state_dict(payload['model_state'], strict=True)
    restore_optimizer(opt, payload['optimizer_state'])
    sched.load_state_dict(payload['scheduler_state'])
    for loader, state in zip(loaders, payload['loader_rng']):
        loader.generator.set_state(state.cpu())
    restore_rng(payload['rng'])
    return payload['completed_epochs'], payload['step'], payload['best_validation_loss']
