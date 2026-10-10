"""Explicit batch experiments; optimizer updates and data progress are distinct."""
from dataclasses import dataclass, asdict, replace
import math
import json
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class TrainingRecipe:
    token_budget: int
    base_learning_rate: float = 4e-4
    reference_token_budget: int = 2048
    reference_steps_per_epoch: int = 500
    schedule: str = "target-row-progress-v1"

    @property
    def learning_rate(self):
        return self.base_learning_rate * math.sqrt(self.token_budget / self.reference_token_budget)

    def identity(self):
        return {**asdict(self), "default_learning_rate": self.learning_rate}


RECIPES = {str(budget): TrainingRecipe(budget) for budget in (2048, 3072, 4096)}


def recipe(args):
    resolved = getattr(args, 'resolved_recipe', None)
    budget = resolved['token_budget'] if resolved else int(getattr(args, 'training_recipe', None) or 4096)
    value = TrainingRecipe(budget)
    return replace(value,schedule="optimizer-step") if getattr(args,"experiment_config",None) else value


_DEFAULTS = {
    'A': dict(epochs=90, scheduler_epochs=155, offset=0, learning_rate=0.000565685424949238,
              warmup_steps=79, physical_warmup_steps=988, token_budget=4096, precision='bf16'),
    'B': dict(epochs=50, scheduler_epochs=50, offset=0, learning_rate=1e-4, objective='nll',
              warmup_steps=50, physical_warmup_steps=0, token_budget=16384, precision='fp32'),
    'sft': dict(epochs=3, scheduler_epochs=3, offset=0, learning_rate=2e-5,
                encoder_learning_rate=5e-6, warmup_steps=0, physical_warmup_steps=0,
                token_budget=4096, precision='bf16'),
}


def resolve_stage_recipe(config_or_path=None, stage='A'):
    """Resolve actual numerical settings; historical experiment labels are provenance.

    Accept a flat stage recipe or a document with ``stages.A/B/sft``. Relative
    experiment_config references in older B recipe files remain readable.
    """
    if stage not in _DEFAULTS:
        raise ValueError(f'unknown training stage: {stage}')
    path = None
    if config_or_path is None:
        document = {}
    elif isinstance(config_or_path, Mapping):
        document = dict(config_or_path)
    else:
        path = Path(config_or_path)
        document = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(document, dict):
        raise ValueError('training recipe must be a JSON object')
    document = document.get('recipe', document)
    defaults = dict(_DEFAULTS[stage], seed=20260811, workers=12, weight_decay=.01, gradient_clip=1.)
    parent = {}
    if path is not None and document.get('experiment_config'):
        parent_path = path.parent / document['experiment_config']
        parent = json.loads(parent_path.read_text(encoding='utf-8-sig'))
    stages = document.get('stages')
    local = stages.get(stage, {}) if isinstance(stages, dict) else document
    base = parent.get('stages', {}).get(stage, {})
    result = defaults | {k: v for k, v in parent.items() if k in defaults}
    result.update({k: v for k, v in document.items() if k in defaults})
    result.update({k: v for k, v in base.items() if k in defaults})
    result.update({k: v for k, v in local.items() if k in defaults})
    if 'budget' in local and 'token_budget' not in local:
        result['token_budget'] = local['budget']
    if result.get('objective') == 'joint-gmm4-nll-valid-coordinate-mean':
        result['objective'] = 'nll'
    return validate_stage_recipe(result)


def validate_stage_recipe(value):
    for key in ('epochs', 'scheduler_epochs', 'token_budget', 'workers'):
        if type(value.get(key)) is not int or value[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('offset', 'warmup_steps', 'physical_warmup_steps'):
        if type(value.get(key)) is not int or value[key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    if value['scheduler_epochs'] < value['epochs']:
        raise ValueError('scheduler horizon must cover the requested stop epoch')
    if type(value['seed']) is not int:
        raise ValueError('seed must be an integer')
    for key in ('learning_rate', 'gradient_clip', 'weight_decay'):
        number = value.get(key)
        if not isinstance(number, (int, float)) or not math.isfinite(number) or number < 0 or (key != 'weight_decay' and number == 0):
            raise ValueError(f'invalid {key}')
    if value['precision'] not in ('bf16', 'fp32'):
        raise ValueError('precision must be bf16 or fp32')
    if 'objective' in value and value['objective'] not in ('nll', 'nll+decoder-l3'):
        raise ValueError('unknown B objective')
    return value


def bind_stage_recipe(args, stage='A'):
    """Apply explicit CLI overrides once, then retain the resolved run settings."""
    if getattr(args, 'resolved_recipe', None) is not None:
        return args.resolved_recipe
    value = resolve_stage_recipe(getattr(args, 'experiment_config', None), stage)
    if getattr(args, 'training_recipe', None) is not None and getattr(args, 'token_budget', None) is None:
        value['token_budget'] = int(args.training_recipe)
    fields = dict(stop_after_epochs='epochs', scheduler_epochs='scheduler_epochs',
                  plan_offset='offset', seed='seed', num_workers='workers',
                  token_budget='token_budget', learning_rate='learning_rate',
                  warmup_steps='warmup_steps', physical_warmup_steps='physical_warmup_steps',
                  weight_decay='weight_decay', gradient_clip='gradient_clip', precision='precision')
    if stage == 'sft':
        fields['encoder_learning_rate'] = 'encoder_learning_rate'
    for attribute, key in fields.items():
        requested = getattr(args, attribute, None)
        if requested is not None:
            value[key] = requested
    validate_stage_recipe(value)
    for attribute, key in fields.items():
        setattr(args, attribute, value[key])
    args.training_recipe = str(value['token_budget'])
    args.resolved_recipe = value
    if getattr(args, 'validation_policy', None) is None:
        path = getattr(args, 'experiment_config', None)
        document = json.loads(Path(path).read_text(encoding='utf-8-sig')) if path else {}
        args.validation_policy = 'disabled' if document.get('validation', 'disabled') == 'disabled' else 'required'
    return value


def plan_steps(plans, epochs, *, offset=0):
    """Count the supplied final plan; never repack or manufacture extra epochs."""
    end = int(offset) + int(epochs)
    if offset < 0 or end > plans.available_epochs:
        raise ValueError(f'plan requires epochs [{offset}, {end}), available={plans.available_epochs}')
    return sum(plans.batch_count(e) for e in range(offset, end))


def data_progress(epoch, consumed_rows, total_rows):
    if epoch < 0 or total_rows <= 0 or not 0 <= consumed_rows <= total_rows:
        raise ValueError("invalid target-row schedule progress")
    return 500.0 * (epoch + consumed_rows / total_rows)
