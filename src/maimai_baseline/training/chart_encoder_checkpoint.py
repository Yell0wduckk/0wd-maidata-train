"""Independent chart-encoder checkpoints and portable deterministic encoder exports."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
import random

import numpy as np
import torch

from maimai_baseline.runtime_versions import operational, require_compatible, stamp
from maimai_baseline.encoder_audio_contract import CONTRACT, REPRESENTATION

LEGACY_CHECKPOINT_SCHEMA = 'maimai.chart_encoder_pretrain_checkpoint.v1'
LEGACY_ENCODER_SCHEMA = 'maimai.pretrained_chart_encoder.v1'
CHECKPOINT_SCHEMA = 'maimai.chart_encoder_audio_pretrain_checkpoint.v2'
ENCODER_SCHEMA = 'maimai.pretrained_chart_encoder_audio.v2'


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    if not isinstance(state, Mapping) or not {'python', 'numpy', 'torch', 'cuda'} <= state.keys():
        raise ValueError('chart encoder checkpoint is missing complete RNG state')
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available() and state['cuda']:
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def _atomic_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def _model_metadata(model):
    return dict(vocabulary=model.vocabulary.to_dict(),
                prototypes=model.encoder.embeddings['route'].prototypes.detach().cpu().clone(),
                query_rows=int(model.query_rows), seed=int(model.seed),
                representation=REPRESENTATION, encoder_audio_contract=deepcopy(CONTRACT))


def _check_state(state, model):
    expected = model.state_dict()
    if (not isinstance(state, Mapping) or state.keys() != expected.keys() or any(
            not isinstance(value, torch.Tensor) or value.shape != expected[name].shape
            or value.dtype != expected[name].dtype or not bool(torch.isfinite(value).all())
            for name, value in state.items())):
        raise ValueError('chart encoder tensor inventory/shape/dtype/finite mismatch')


def save_checkpoint(path, *, model, optimizer, scheduler, epoch, step, recipe, data_binding,
                    steps_by_epoch, loader_generator, metrics=None, source_provenance=None):
    """The caller saves only after consuming the exact complete plan epoch."""
    if (type(epoch) is not int or epoch < 0 or epoch >= len(steps_by_epoch) or epoch >= recipe['epochs']
            or step != sum(steps_by_epoch[:epoch + 1]) or scheduler.last_epoch != step):
        raise ValueError('chart encoder checkpoint must be at a complete epoch boundary')
    if not isinstance(loader_generator, torch.Generator):
        raise ValueError('chart encoder checkpoint requires DataLoader generator state')
    payload = dict(schema=CHECKPOINT_SCHEMA, **stamp(), model_metadata=_model_metadata(model),
        model_state={name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        optimizer_state=optimizer.state_dict(), scheduler_state=scheduler.state_dict(),
        epoch=epoch, sampler_epoch=int(recipe['offset']) + epoch, step=int(step),
        recipe=deepcopy(recipe), data_binding=deepcopy(data_binding),
        steps_by_epoch=list(steps_by_epoch), rng_state=rng_state(),
        loader_rng=loader_generator.get_state(), metrics=metrics or {},
        source_provenance=deepcopy(source_provenance))
    _atomic_save(path, payload)
    return payload


def load_checkpoint(path, *, device='cpu'):
    from maimai_baseline.model.chart_encoder_pretrain import ChartEncoderPretrainer
    from maimai_dataset.vocabulary import CompactVocabulary

    payload = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(payload, Mapping) or payload.get('schema') != CHECKPOINT_SCHEMA:
        raise ValueError('unsupported independent chart encoder checkpoint schema')
    require_compatible(payload, 'training')
    metadata = payload['model_metadata']
    if metadata.get('representation') != REPRESENTATION or metadata.get('encoder_audio_contract') != CONTRACT:
        raise ValueError('native encoder pretraining representation/objective changed')
    model = ChartEncoderPretrainer(CompactVocabulary.from_dict(metadata['vocabulary']),
        metadata['prototypes'], int(metadata['query_rows']), seed=int(metadata['seed']))
    _check_state(payload['model_state'], model)
    model.load_state_dict(payload['model_state'], strict=True)
    return model.to(device), dict(payload)


def validate_resume(payload, *, recipe, data_binding, steps_by_epoch):
    saved = payload['recipe']
    # Stop is an absolute endpoint; extending it never changes the LR horizon.
    if ({k: v for k, v in saved.items() if k != 'epochs'} !=
            {k: v for k, v in recipe.items() if k != 'epochs'}):
        raise ValueError('chart encoder resume recipe or scheduler horizon changed')
    if recipe['epochs'] < saved['epochs']:
        raise ValueError('chart encoder resume may extend, not shorten, the absolute stop epoch')
    if operational(payload['data_binding']) != operational(data_binding):
        raise ValueError('chart encoder resume data or fixed plan changed')
    if list(payload['steps_by_epoch']) != list(steps_by_epoch):
        raise ValueError('chart encoder resume fixed plan update counts changed')
    epoch, step = payload.get('epoch'), payload.get('step')
    if (type(epoch) is not int or epoch < 0 or epoch >= len(steps_by_epoch) or epoch >= saved['epochs']
            or payload.get('sampler_epoch') != recipe['offset'] + epoch
            or type(step) is not int or step != sum(steps_by_epoch[:epoch + 1])
            or payload.get('scheduler_state', {}).get('last_epoch') != step):
        raise ValueError('chart encoder resume requires a complete epoch cursor')
    return epoch + 1, step


def restore_training_state(payload, *, optimizer, scheduler, loader_generator):
    validate_resume(payload, recipe=payload['recipe'], data_binding=payload['data_binding'],
                    steps_by_epoch=payload['steps_by_epoch'])
    if not isinstance(payload.get('loader_rng'), torch.Tensor):
        raise ValueError('chart encoder checkpoint is missing DataLoader RNG state')
    optimizer.load_state_dict(payload['optimizer_state'])
    scheduler.load_state_dict(payload['scheduler_state'])
    if scheduler.last_epoch != payload['step']:
        raise ValueError('chart encoder scheduler and complete epoch cursor differ')
    restore_rng(payload['rng_state'])
    loader_generator.set_state(payload['loader_rng'].cpu())
    return payload['epoch'] + 1, payload['step']


def export_encoder(path, *, model, recipe=None, data_binding=None, source_provenance=None, completed_epochs=None):
    """Export only the encoder; no auxiliary decoder, prior or latent cache."""
    payload = dict(schema=ENCODER_SCHEMA, **stamp(),
        metadata=dict(_model_metadata(model), beats_per_token=4, dimension=128,
                      completed_epochs=completed_epochs),
        encoder_state={name: value.detach().cpu().clone() for name, value in model.encoder.state_dict().items()},
        recipe=deepcopy(recipe), data_binding=deepcopy(data_binding),
        source_provenance=deepcopy(source_provenance))
    _atomic_save(path, payload)
    return payload


def load_encoder(path, *, device='cpu'):
    """Use model.chart_encoder_pretrain.encode_pretrained to obtain clean latents."""
    from maimai_baseline.model.style import ChartStyleEncoder
    from maimai_dataset.vocabulary import CompactVocabulary

    payload = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(payload, Mapping) or payload.get('schema') not in (ENCODER_SCHEMA, LEGACY_ENCODER_SCHEMA):
        raise ValueError('unsupported pretrained chart encoder export schema')
    require_compatible(payload, 'inference')
    metadata = payload['metadata']
    native = payload['schema'] == ENCODER_SCHEMA
    expected_representation = REPRESENTATION if native else 'deterministic-mu;no-posterior-sampling'
    if (metadata.get('beats_per_token') != 4 or metadata.get('dimension') != 128
            or metadata.get('representation') != expected_representation
            or (native and metadata.get('encoder_audio_contract') != CONTRACT)):
        raise ValueError('pretrained chart encoder representation changed')
    encoder = ChartStyleEncoder(CompactVocabulary.from_dict(metadata['vocabulary']),
                               metadata['prototypes'], seed=int(metadata['seed']))
    encoder.logvar.requires_grad_(False)
    if native:
        encoder.mu_representation = REPRESENTATION
    _check_state(payload['encoder_state'], encoder)
    encoder.load_state_dict(payload['encoder_state'], strict=True)
    return encoder.to(device).eval(), dict(metadata)
