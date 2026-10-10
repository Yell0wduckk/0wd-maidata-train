"""Retained B representation checks and explicit stage schedule loading."""
import torch
from pathlib import Path

from maimai_baseline.model.song_ar import CONTRACT

ARCHITECTURE = "maimai-frozen-mu128-song-ar-gmm4-v1"
CHECKPOINT_SCHEMA = "maimai.frozen_mu128_song_ar_checkpoint.v1"
IDENTITY_SCHEMA = "maimai.frozen_mu128_song_ar_training.v1"
DECODER_OBJECTIVE = 'nll+decoder-l3'
TAIL_CONTRACT = 'complete-tail-empty-v1'
CONFIG = Path(__file__).resolve().parents[2]/'configs/train-b.json'
FULL_CONFIG = CONFIG.with_name('full-mmoe-decoders-ar-b50.json')
RECIPE = dict(schema='maimai.ar_consumer_config.v1', experiment_config='pilot261-mmoe-decoders-whole-song-50.json',
    stages=['B'], prior=CONTRACT['schema'], objective=CONTRACT['loss'], audio_cfg=False,
    epochs=50, scheduler_epochs=50, offset=90, warmup_steps=50, learning_rate=1e-4,
    weight_decay=.01, gradient_clip=1., precision='fp32', workers=12,
    budget=16384, budget_unit='source-target-rows', seed=20260811)
FULL_RECIPE = dict(RECIPE, experiment_config='full-mmoe-decoders-whole-song-50.json')


def validate_normalization(norm):
    """Normalize only the learned deterministic means, never unused logvar."""
    if (not isinstance(norm, dict) or norm.get('includes_posterior_variance') is not False
            or norm.get('representation') not in ('deterministic-mu', 'layernorm-mu-eps1e-5')):
        raise ValueError('AR requires deterministic mu-only normalization')
    for key in ('mean', 'std'):
        value = torch.tensor(norm.get('local', {}).get(key, []), dtype=torch.float32)
        if value.shape != (128,) or not torch.isfinite(value).all() or (key == 'std' and bool((value <= 0).any())):
            raise ValueError('invalid AR latent normalization')


def recipe_for(selection=None):
    """Historical callers may request either preset; data selection is not a gate."""
    return dict(FULL_RECIPE if selection == 'full' else RECIPE)


def config_for(selection=None):
    return FULL_CONFIG if selection == 'full' else CONFIG


def load_ar_config(path):
    from maimai_baseline.training.recipe import resolve_stage_recipe
    recipe = resolve_stage_recipe(path, stage='B')
    return dict(recipe=recipe)


def plan_counts(identity):
    """Read current or historical complete-epoch counts without producer checks."""
    plan = identity.get('plan')
    if plan is not None:
        return list(plan['optimizer_steps_by_epoch'])
    return list(identity['whole_song_training']['experiment']['optimizer_steps_by_epoch'])


def stage_recipe(identity):
    from maimai_baseline.training.recipe import resolve_stage_recipe
    value = dict(identity['recipe'])
    if 'offset' not in value and 'whole_song_training' in identity:
        value['offset'] = identity['whole_song_training']['experiment']['recipe']['stages']['B']['offset']
    return resolve_stage_recipe(value, stage='B')


def condition_contract(identity):
    """Old B inference keeps its original coverage; new B records the live mask contract."""
    if stage_recipe(identity).get('objective') == DECODER_OBJECTIVE:
        if identity.get('condition_contract') != TAIL_CONTRACT:
            raise ValueError('Decoder-supervised B tail condition contract missing')
        return TAIL_CONTRACT
    if identity.get('condition_contract') is not None:
        raise ValueError('NLL-only B cannot claim Decoder tail supervision')
    return None


def validate_identity(identity, *, operation='training'):
    """Validate consumed representation fields; provenance remains descriptive."""
    if not isinstance(identity, dict):
        raise ValueError('missing AR training metadata')
    model = identity.get('model', {})
    for field in ('dimension', 'audio_width', 'prediction', 'components', 'component_scope', 'input', 'target'):
        if model.get(field) != CONTRACT[field]:
            raise ValueError('AR representation mismatch: ' + field)
    validate_normalization(identity.get('normalization'))
    recipe = stage_recipe(identity)
    condition_contract(identity)
    counts = plan_counts(identity)
    offset = recipe['offset']
    required = offset + max(recipe['epochs'], recipe['scheduler_epochs'])
    if len(counts) < required or any(type(n) is not int or n < 1 for n in counts):
        raise ValueError('AR plan does not cover the requested schedule')
    if identity.get('total_steps') != sum(counts[offset:offset + recipe['scheduler_epochs']]):
        raise ValueError('AR scheduler horizon differs from plan update count')
    return identity.get('whole_song_training', {}).get('experiment', identity.get('plan'))
