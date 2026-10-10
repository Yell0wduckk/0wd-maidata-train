"""Independent complete-epoch audio-draft checkpoints and generator-only exports."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

import torch

from maimai_baseline.model.audio_draft import (CONTRACT, SUPERVISION_CONTRACT,
    AudioDraftGenerator, AudioDraftPretrainer, validate_structure)
from maimai_baseline.runtime_versions import operational, require_compatible, stamp
from maimai_baseline.training.chart_encoder_checkpoint import _atomic_save, rng_state, restore_rng


CHECKPOINT_SCHEMA = 'maimai.audio_draft_pretrain_checkpoint.v1'
DRAFT_SCHEMA = 'maimai.audio_draft_generator_export.v1'


def _check_state(state, model):
    expected = model.state_dict()
    if (not isinstance(state, Mapping) or state.keys() != expected.keys() or any(
            not isinstance(value, torch.Tensor) or value.shape != expected[name].shape
            or value.dtype != expected[name].dtype or not bool(torch.isfinite(value).all())
            for name, value in state.items())):
        raise ValueError('audio draft tensor inventory/shape/dtype/finite mismatch')


def save_checkpoint(path, *, model, optimizer, scheduler, epoch, step, recipe, data_binding,
                    steps_by_epoch, loader_generator, metrics=None, source_provenance=None,
                    validation_generator=None, best_validation_loss=None):
    if (type(epoch) is not int or epoch < 0 or epoch >= len(steps_by_epoch) or epoch >= recipe['epochs']
            or step != sum(steps_by_epoch[:epoch+1]) or scheduler.last_epoch != step):
        raise ValueError('audio draft checkpoint must be at a complete epoch boundary')
    if not isinstance(loader_generator, torch.Generator):
        raise ValueError('audio draft checkpoint requires DataLoader RNG state')
    if validation_generator is not None and not isinstance(validation_generator, torch.Generator):
        raise ValueError('audio draft validation requires DataLoader RNG state')
    payload = dict(schema=CHECKPOINT_SCHEMA, **stamp(), structure=deepcopy(model.structure),
        supervision=deepcopy(model.supervision),
        model_metadata=dict(vocabulary=model.vocabulary.to_dict(), query_rows=model.query_rows, seed=model.seed),
        model_state={name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        optimizer_state=optimizer.state_dict(), scheduler_state=scheduler.state_dict(), epoch=epoch,
        completed_epochs=epoch+1, sampler_epoch=int(recipe['offset'])+epoch, step=int(step),
        recipe=deepcopy(recipe), data_binding=deepcopy(data_binding), steps_by_epoch=list(steps_by_epoch),
        rng_state=rng_state(), loader_rng=loader_generator.get_state(), metrics=metrics or {},
        validation_loader_rng=None if validation_generator is None else validation_generator.get_state(),
        best_validation_loss=best_validation_loss,
        source_provenance=deepcopy(source_provenance))
    _atomic_save(path, payload)
    return payload


def _payload(path, operation):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(payload, Mapping) or payload.get('schema') not in (CHECKPOINT_SCHEMA, DRAFT_SCHEMA):
        raise ValueError('unsupported standalone audio draft checkpoint schema')
    require_compatible(payload, operation)
    validate_structure(payload.get('structure'))
    if operation == 'training' and payload['schema'] != CHECKPOINT_SCHEMA:
        raise ValueError('generator export cannot resume standalone draft training')
    if operation == 'training' and (payload['structure'] != CONTRACT
            or payload.get('supervision') != SUPERVISION_CONTRACT):
        raise ValueError('audio draft training representation or reconstruction supervision changed')
    return dict(payload)


def load_checkpoint(path, *, device='cpu'):
    from maimai_dataset.vocabulary import CompactVocabulary
    payload = _payload(path, 'training')
    if payload['schema'] != CHECKPOINT_SCHEMA:
        raise ValueError('generator export cannot resume standalone draft training')
    metadata = payload['model_metadata']
    model = AudioDraftPretrainer(CompactVocabulary.from_dict(metadata['vocabulary']),
        int(metadata['query_rows']), seed=int(metadata['seed']), structure=payload['structure'])
    _check_state(payload['model_state'], model)
    model.load_state_dict(payload['model_state'], strict=True)
    return model.to(device), payload


def validate_resume(payload, *, recipe, data_binding, steps_by_epoch):
    if payload.get('structure') != CONTRACT or payload.get('supervision') != SUPERVISION_CONTRACT:
        raise ValueError('audio draft resume representation or supervision changed')
    saved = payload['recipe']
    if ({k: v for k, v in saved.items() if k != 'epochs'} !=
            {k: v for k, v in recipe.items() if k != 'epochs'}):
        raise ValueError('audio draft resume recipe or scheduler horizon changed')
    if recipe['epochs'] < saved['epochs']:
        raise ValueError('audio draft resume may extend, not shorten, the stop epoch')
    if operational(payload['data_binding']) != operational(data_binding):
        raise ValueError('audio draft resume data or canonical fixed plan changed')
    if list(payload['steps_by_epoch']) != list(steps_by_epoch):
        raise ValueError('audio draft resume fixed plan update counts changed')
    epoch, step = payload.get('epoch'), payload.get('step')
    if (type(epoch) is not int or epoch < 0 or epoch >= len(steps_by_epoch) or epoch >= saved['epochs']
            or payload.get('completed_epochs') != epoch+1 or payload.get('sampler_epoch') != recipe['offset']+epoch
            or type(step) is not int or step != sum(steps_by_epoch[:epoch+1])
            or payload.get('scheduler_state', {}).get('last_epoch') != step):
        raise ValueError('audio draft resume requires a complete epoch cursor')
    return epoch+1, step


def restore_training_state(payload, *, optimizer, scheduler, loader_generator, validation_generator=None):
    validate_resume(payload, recipe=payload['recipe'], data_binding=payload['data_binding'],
                    steps_by_epoch=payload['steps_by_epoch'])
    if not isinstance(payload.get('loader_rng'), torch.Tensor):
        raise ValueError('audio draft checkpoint is missing DataLoader RNG state')
    validation_rng = payload.get('validation_loader_rng')
    if validation_generator is not None and not isinstance(validation_rng, torch.Tensor):
        raise ValueError('audio draft checkpoint is missing validation DataLoader RNG state')
    if validation_generator is None and validation_rng is not None:
        raise ValueError('audio draft checkpoint validation loader was removed')
    optimizer.load_state_dict(payload['optimizer_state'])
    scheduler.load_state_dict(payload['scheduler_state'])
    restore_rng(payload['rng_state'])
    loader_generator.set_state(payload['loader_rng'].cpu())
    if validation_generator is not None:
        validation_generator.set_state(validation_rng.cpu())
    return payload['epoch']+1, payload['step']


def export_draft(path, *, model, recipe=None, data_binding=None, source_provenance=None, completed_epochs=None):
    generator = model.generator if isinstance(model, AudioDraftPretrainer) else model
    payload = dict(schema=DRAFT_SCHEMA, **stamp(), structure=deepcopy(generator.structure), seed=generator.seed,
        completed_epochs=completed_epochs, epoch=None if completed_epochs is None else completed_epochs-1,
        generator_state={name: value.detach().cpu().clone() for name, value in generator.state_dict().items()},
        recipe=deepcopy(recipe), data_binding=deepcopy(data_binding), source_provenance=deepcopy(source_provenance))
    _atomic_save(path, payload)
    return payload


def load_draft(path, *, device='cpu'):
    """Read the generator only; an auxiliary head is never constructed here."""
    payload = _payload(path, 'inference')
    if payload['schema'] == CHECKPOINT_SCHEMA:
        seed = int(payload['model_metadata']['seed'])
        state = {name.removeprefix('generator.'): value for name, value in payload['model_state'].items()
                 if name.startswith('generator.')}
    else:
        seed, state = int(payload['seed']), payload['generator_state']
    generator = AudioDraftGenerator(seed, structure=payload['structure'])
    _check_state(state, generator)
    generator.load_state_dict(state, strict=True)
    payload['seed'] = seed
    payload['generator_state'] = state
    return generator.to(device).eval(), payload
