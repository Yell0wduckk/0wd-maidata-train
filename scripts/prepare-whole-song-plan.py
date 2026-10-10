"""Derive only new batch boundaries from the selected A90 metadata/plan."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from maimai_baseline.checkpoint import load_frozen_stage_a
from maimai_baseline.data.master_plan import metadata
from maimai_baseline.experiment import load_config
from maimai_baseline.whole_song_training import import_from_payload, build_plan


def main():
    import torch
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('stage-a-checkpoint', 'route-artifact', 'dataset-root',
                 'experiment-config', 'epoch-plan-cache', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    payload = torch.load(args.stage_a_checkpoint, map_location='cpu', weights_only=False)
    imported = import_from_payload(args.stage_a_checkpoint, payload)
    model, _ = load_frozen_stage_a(args.stage_a_checkpoint, route_artifact=args.route_artifact,
                                  frozen_import=imported)
    del model, payload
    config = load_config(args.experiment_config)
    records, _, bpm = metadata(args.dataset_root, config)
    value = build_plan(args.epoch_plan_cache, args.output, records['train'], imported, config=config, bpm=bpm)
    print(json.dumps(dict(plan_sha256=value['plan_sha256'], policy=value['policy'],
                         steps=value['experiment']['optimizer_steps_by_epoch'])))


if __name__ == '__main__':
    main()
