"""Derive AR consumer metadata without scanning/rebuilding accepted payloads."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

from maimai_baseline.ar_contract import CONFIG, load_ar_config
from maimai_baseline.checkpoint import load_frozen_stage_a
from maimai_baseline.data.master_plan import metadata
from maimai_baseline.whole_song_training import import_from_payload, build_plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ar-config', type=Path, default=CONFIG)
    for name in ('stage-a-checkpoint', 'route-artifact', 'dataset-root', 'epoch-plan-cache', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    from maimai_baseline.data.curriculum import EpochPlanStore
    from maimai_baseline.query6_plan import validate as validate_a_plan
    validate_a_plan(EpochPlanStore(args.epoch_plan_cache, verify_payloads=False))
    model, payload = load_frozen_stage_a(args.stage_a_checkpoint, route_artifact=args.route_artifact)
    imported = import_from_payload(args.stage_a_checkpoint, payload)
    del model, payload
    from maimai_baseline.experiment import load_config, validate_plan
    recipe = load_ar_config(args.ar_config)['recipe']
    document = json.loads(args.ar_config.read_text(encoding='utf-8-sig'))
    if document.get('experiment_config'):
        config = load_config(args.ar_config.parent / document['experiment_config'])
    else:
        config = validate_plan(EpochPlanStore(args.epoch_plan_cache, verify_payloads=False))['recipe']
    config = dict(config, stages=dict(config['stages'], B=recipe),
                  plan_epochs=max(config['plan_epochs'], recipe['offset'] + recipe['scheduler_epochs']))
    records, _, bpm = metadata(args.dataset_root, config)
    binding = build_plan(args.epoch_plan_cache, args.output, records['train'], imported, config=config, bpm=bpm)
    print(json.dumps(dict(plan_sha256=binding['plan_sha256'], policy=binding['policy'],
        active_stage='B', absolute_epochs=[recipe['offset'] + 1, recipe['offset'] + recipe['epochs']],
        steps=binding['experiment']['optimizer_steps_by_epoch'][recipe['offset']:recipe['offset'] + recipe['epochs']])))


if __name__ == '__main__':
    main()
