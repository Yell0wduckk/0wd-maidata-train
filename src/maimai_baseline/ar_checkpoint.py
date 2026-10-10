"""Native AR checkpoints contain one complete model state and A metadata."""
from maimai_baseline.runtime_versions import stamp
from copy import deepcopy
from pathlib import Path
import hashlib
import random

import numpy as np
import torch

from maimai_baseline.ar_contract import (ARCHITECTURE, CHECKPOINT_SCHEMA, validate_identity,
    plan_counts, stage_recipe, condition_contract)
from maimai_baseline.model.song_ar import SongAR


def state_digest(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        h.update((name+'\0'+str(tensor.dtype)+'\0'+str(tuple(tensor.shape))+'\0').encode())
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def retain_parent(payload):
    excluded = {'model_state', 'optimizer_state', 'scheduler_state', 'rng_state',
                'metrics', 'profile_summary', 'parent_payload', 'ar_state', 'training'}
    return deepcopy({k:v for k,v in payload.items() if k not in excluded})


def attach_ar(model, *, seed):
    """Select B trainability in place; never reconstruct the registered prior."""
    if not isinstance(model.style_prior, SongAR):
        raise ValueError('native SongAR is required')
    model.eval().requires_grad_(False)
    model.zero_grad(set_to_none=True)
    model.style_prior.requires_grad_(True).train()
    return model


def require_frozen(model, parent):
    actual = {k:v for k,v in model.state_dict().items() if not k.startswith('style_prior.')}
    expected = {k:v for k,v in parent['model_state'].items() if not k.startswith('style_prior.')}
    if actual.keys() != expected.keys() or any(not torch.equal(v.detach().cpu(), expected[k].cpu()) for k,v in actual.items()):
        raise ValueError('AR changed a frozen A tensor')
    if any(p.requires_grad or p.grad is not None for n,p in model.named_parameters() if not n.startswith('style_prior.')):
        raise ValueError('AR must freeze all non-prior parameters')


def capture_rng(loader_generator):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [], loader=loader_generator.get_state())


def make_payload(model, parent, identity, *, epoch, step, metrics, optimizer, scheduler, loader_generator, plan_batches=None):

    if not bool(model.style_prior.ready):
        raise ValueError('AR checkpoint requires a completed epoch')
    result = dict(**stamp(), schema=CHECKPOINT_SCHEMA, architecture=ARCHITECTURE,
        source_identity=deepcopy(identity.get('source_identity')), identity=deepcopy(identity),
        parent_metadata=retain_parent(parent),
        model_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        epoch=epoch, step=step, metrics=metrics, checkpoint_kind='training',
        training=dict(optimizer_state=deepcopy(optimizer.state_dict()), scheduler_state=deepcopy(scheduler.state_dict()),
            rng_state=capture_rng(loader_generator), data_cursor=dict(next_plan_epoch=stage_recipe(identity)['offset']+epoch+1, batch=0)))
    if plan_batches is not None:
        result['training']['plan_batches'] = plan_batches
    validate_cursor(result)
    return result


def serialize_batches(batches):
    """Store actual last-epoch references, not an aggregate producer identity."""
    return [[[r.canonical_index, r.view_id, r.occurrence] for r in batch] for batch in batches]


def validate_cursor(payload):
    """Check the stored complete-epoch state, not a reconstructed fixed recipe."""
    identity = payload['identity']
    recipe = stage_recipe(identity)
    counts = plan_counts(identity)
    offset = recipe['offset']
    epoch, step = payload.get('epoch'), payload.get('step')
    saved = payload.get('training', {})
    if (type(epoch) is not int or not 0 <= epoch < recipe['epochs'] or type(step) is not int or
            step != sum(counts[offset:offset+epoch+1]) or
            saved.get('data_cursor') != dict(next_plan_epoch=offset+epoch+1, batch=0) or
            saved.get('scheduler_state', {}).get('last_epoch') != step or
            not isinstance(saved.get('optimizer_state'), dict)):
        raise ValueError('AR checkpoint is not a complete epoch/scheduler boundary')
    rng = saved.get('rng_state', {})
    if not {'python', 'numpy', 'torch', 'cuda', 'loader'} <= rng.keys():
        raise ValueError('AR complete epoch RNG missing')
    for name in ('torch', 'loader'):
        if not isinstance(rng[name], torch.Tensor) or rng[name].dtype != torch.uint8 or rng[name].ndim != 1:
            raise ValueError('invalid AR RNG state')


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, *, route_artifact, expected_identity=None, device='cpu', operation='training'):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    return load_payload(payload, route_artifact=route_artifact, expected_identity=expected_identity,
                        device=device, operation=operation)


def load_payload(payload, *, route_artifact, expected_identity=None, device='cpu', operation='training'):
    """A B checkpoint contains everything needed; no parent A file is reopened."""
    from maimai_baseline.checkpoint import _load_baseline_payload
    if not isinstance(payload, dict) or payload.get('schema') != CHECKPOINT_SCHEMA:
        raise ValueError('not a native AR checkpoint')
    if any(k in payload for k in ('parent_payload', 'ar_state')):
        raise ValueError('retired composite AR checkpoint')
    from maimai_baseline.runtime_versions import require_compatible
    require_compatible(payload, operation)
    identity = payload['identity']
    validate_identity(identity, operation=operation)
    if operation == 'training':
        validate_cursor(payload)
    if expected_identity is not None:
        # Compare only the consumed normalization and existing plan boundaries.
        if identity['normalization']['local'] != expected_identity['normalization']['local']:
            raise ValueError('AR resume normalization differs from cached targets')
    parent = payload['parent_metadata']
    view = dict(parent, model_state=payload['model_state'], ar_prior=identity)
    model, _ = _load_baseline_payload(view, route_artifact=route_artifact, device='cpu',
        strict_source=True, verify_artifact_payloads=False, operation=operation)
    model.prior_condition_contract = condition_contract(identity)
    if not bool(model.style_prior.ready):
        raise ValueError('AR checkpoint prior is not trained')
    norm = identity['normalization']['local']
    if (not torch.equal(model.style_prior.latent_mean.cpu(),torch.tensor(norm['mean'],dtype=torch.float32))
            or not torch.equal(model.style_prior.latent_std.cpu(),torch.tensor(norm['std'],dtype=torch.float32))):
        raise ValueError('AR checkpoint normalization buffers differ from cached target statistics')
    model.eval().requires_grad_(False)
    if operation == 'training':
        model.style_prior.requires_grad_(True).train()
    return model.to(device), payload


def restore_training(payload, optimizer, scheduler, loader_generator):
    validate_cursor(payload)
    saved = payload['training']
    restore_training_state(saved, optimizer, scheduler, loader_generator)
    return payload['epoch']+1, payload['step']


def restore_training_state(saved, optimizer, scheduler, loader_generator):
    """Shared restore mechanics; public checkpoint entry validates its cursor."""
    optimizer.load_state_dict(saved['optimizer_state'])
    for parameter, state in optimizer.state.items():
        for name in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
            if name in state and state[name].shape != parameter.shape:
                raise ValueError('AR optimizer slot shape mismatch: ' + name)
    scheduler.load_state_dict(saved['scheduler_state'])
    rng = saved['rng_state']
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch'].cpu())
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng['cuda'])
    loader_generator.set_state(rng['loader'].cpu())
