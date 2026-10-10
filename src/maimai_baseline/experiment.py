"""Stage recipes and plan counts, independent of named datasets or experiments."""
import json
from copy import deepcopy
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[2] / 'configs/train-a.json'
FULL_CONFIG = CONFIG  # compatibility import; there is no separate full-data mode
SCHEMA = 'maimai.training_recipe.v1'


def validate_recipe(value):
    from maimai_baseline.training.recipe import resolve_stage_recipe
    if not isinstance(value, dict):
        raise ValueError('training recipe must be a JSON object')
    result = deepcopy(value)
    names = result.get('stages')
    names = [s for s in names if s in ('A', 'B', 'sft')] if isinstance(names, dict) else [result.get('stage', 'A')]
    if not names:
        raise ValueError('recipe needs an A, B or sft stage')
    result['stages'] = {name: resolve_stage_recipe(value, name) for name in names}
    first = result['stages'][names[0]]
    for key in ('seed', 'workers', 'token_budget'):
        result.setdefault(key, first[key])
    result.setdefault('validation', 'disabled')
    result.setdefault('excluded_record_ids', [])
    result.setdefault('occurrences_per_chart', 1)
    result.setdefault('view_weights', {'identity': 2, 'mirror-lr': 1})
    result.setdefault('plan_epochs', max(s['offset'] + max(s['epochs'], s['scheduler_epochs']) for s in result['stages'].values()))
    return result


def load_config(path, *, selection=None):
    """Read numerical settings; selection is an ignored legacy caller argument."""
    return validate_recipe(json.loads(Path(path or CONFIG).read_text(encoding='utf-8-sig')))


def recipe(experiment):
    return experiment.get('recipe', experiment)


def validation_enabled(experiment):
    return recipe(experiment).get('validation', 'disabled') != 'disabled'


def selected(row, experiment):
    """Optional explicit filters for preparation tools, never a pilot whitelist."""
    spec = recipe(experiment)
    difficulty = int(row.get('source_difficulty_id', row.get('difficulty_id', -1)))
    record = str(row.get('record_id') or f"{row['song_id']}:{difficulty}")
    return (('difficulties' not in spec or difficulty in spec['difficulties'])
            and record not in spec.get('excluded_record_ids', []))


def stage(name, experiment):
    from maimai_baseline.training.recipe import resolve_stage_recipe
    return resolve_stage_recipe(recipe(experiment), name)


def _steps(experiment, name, key):
    spec = stage(name, experiment)
    counts = experiment['optimizer_steps_by_epoch']
    start, end = spec['offset'], spec['offset'] + spec[key]
    if end > len(counts):
        raise ValueError(f'{name} requires plan epochs [{start}, {end}), available={len(counts)}')
    return sum(counts[start:end])


def step_count(experiment, name):
    return _steps(experiment, name, 'epochs')


def horizon(experiment, name):
    original = recipe(experiment)
    if (original.get('schema') == 'maimai.master_remaster_experiment.v2'
            and original.get('downstream_schedule') not in ('whole-song-50-v1', 'native-ar-b50-v1')):
        spec = stage(name, experiment)
        counts = experiment['optimizer_steps_by_epoch']
        if spec['scheduler_epochs'] > len(counts):
            raise ValueError('scheduler horizon exceeds available plan epochs')
        return sum(counts[:spec['scheduler_epochs']])
    return _steps(experiment, name, 'scheduler_epochs')


def validate_binding(value):
    if not isinstance(value, dict):
        raise ValueError('missing training plan metadata')
    counts = value.get('optimizer_steps_by_epoch', [])
    if not counts or any(type(n) is not int or n <= 0 for n in counts):
        raise ValueError('plan must provide positive per-epoch update counts')
    return value


def bound(manifest):
    return validate_binding(manifest.get('experiment'))


def check_args(args, name):
    from maimai_baseline.training.recipe import bind_stage_recipe
    bind_stage_recipe(args, name)
    config = load_config(getattr(args, 'experiment_config', None))
    config['stages'][name] = dict(args.resolved_recipe)
    config['seed'] = args.seed
    return config


def validate_plan(plans, config=None):
    """Read supplied final plan counts; provenance is not a training admission gate."""
    value = deepcopy(plans.manifest.get('experiment', {}))
    value['optimizer_steps_by_epoch'] = [plans.batch_count(e) for e in range(plans.available_epochs)]
    if config is not None:
        value['recipe'] = validate_recipe(config)
    elif 'recipe' not in value:
        value['recipe'] = load_config(CONFIG)
    value.setdefault('subsets', {})
    return validate_binding(value)


def parent_experiment(identity):
    return identity.get('whole_song_training', {}).get('experiment',
        identity.get('frozen_stage_a', {}).get('baseline_manifest', {}).get('experiment', {}))


def validate_records(records, experiment, full_charts):
    """Only check that a consumed record/view has a real full-chart mapping."""
    for row in records:
        key = (row['role'], row['record_id'], row['view_id'])
        if key not in full_charts:
            raise ValueError(f'learned cache missing chart/view mapping: {key}')
