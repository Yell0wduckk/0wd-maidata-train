"""Compatibility policy lives exclusively in runtime-versions.toml.

Hashes may describe history; they must never decide model/source compatibility.
Actual tensor, data layout and state validity checks remain the loaders' job.
"""
from collections.abc import Mapping
from pathlib import Path
import re
import tomllib

POLICY_PATH = Path(__file__).with_name('runtime-versions.toml')


def units(version):
    if not isinstance(version, str) or not re.fullmatch(r'(0|[1-9][0-9]*)\.[0-9]{2}', version):
        raise ValueError(f'Invalid runtime version: {version!r}; expected integer.two-digits')
    major, fraction = version.split('.')
    return int(major) * 100 + int(fraction)


def policy():
    with POLICY_PATH.open('rb') as stream:
        value = tomllib.load(stream)
    units(value['current'])
    for bounds in value['supported'].values():
        if units(bounds['min']) > units(bounds['max']):
            raise ValueError('Invalid runtime compatibility range')
    return value


def current_version():
    return policy()['current']


def bump(version, kind):
    step = policy()['bump'][kind]
    value = (units(version) // step + 1) * step
    return f'{value // 100}.{value % 100:02d}'


def resolve_version(payload):
    if not isinstance(payload, Mapping):
        raise ValueError('Missing runtime version metadata')
    if 'runtime_version' in payload:
        units(payload['runtime_version'])
        return payload['runtime_version']
    parent = payload.get('parent_payload', payload)
    training = parent.get('baseline_contract', parent).get('training', {})
    for rule in policy()['legacy']:
        if parent.get('architecture') != rule['architecture'] or parent.get('schema') not in (rule['schema'], rule['manifest_schema']):
            continue
        if parent is not payload and (payload.get('architecture') != rule['ar_architecture'] or payload.get('schema') != rule['ar_schema']):
            continue
        if all(parent.get(k, training.get(k)) == rule[k] for k in ('base_loss_scheme', 'loss_scheme')):
            return rule['version']
    raise ValueError('Unsupported unversioned checkpoint: only the two registered dynamic Query6 producers are supported')


def require_compatible(payload, operation):
    version = resolve_version(payload)
    bounds = policy()['supported'][operation]
    if not units(bounds['min']) <= units(version) <= units(bounds['max']):
        raise ValueError(f'{operation} cannot read runtime version {version}; supported [{bounds["min"]}, {bounds["max"]}]')
    return version


def stamp():
    return {'runtime_version': current_version()}


# Exclude only model/source provenance and objective metadata from runtime
# comparisons. Independent dataset/audio hashes are deliberately retained.
_PROVENANCE = frozenset({
    'source_identity', 'producer_source', 'consumer_source', 'source',
    'checkpoint_sha256', 'stage_a_sha256', 'parent_b_sha256',
    'readout_sha256', 'frozen_state_sha256', 'model_state_sha256',
    'identity_sha256', 'manifest_identity_sha256', 'identity',
    'runtime_version', 'execution_version', 'base_loss_scheme', 'loss_scheme',
    'loss_contract', 'physical_contract', 'training_config',
    'parent_a_sha256', 'parent_source_sha256', 'normalization_source_sha256',
    'cache_identity', 'song_context_identity', 'target_cache_identity', 'label_identity',
    'plan_sha256', 'plan_manifest_sha256', 'input_plan_sha256', 'template_plan_sha256',
    'runtime_source_identity', 'producer_script_sha256', 'parent_model_state_sha256',
    'specialized_supervision_config', 'difficulty_loss_weighting', 'physical_warmup_steps',
})


def operational(value, context=None):
    """Compare data/state contracts without recursively reinstating hash locks."""
    if isinstance(value, Mapping):
        return {k: operational(v, k) for k, v in value.items()
                if (k not in _PROVENANCE or (k == "identity" and isinstance(v, Mapping)))
                and not (k == "sha256" and context in ("plan_cache", "plan", "parent"))}
    if isinstance(value, (list, tuple)):
        return [operational(v) for v in value]
    return value


def compatible_metadata(left, right):
    return operational(left) == operational(right)
