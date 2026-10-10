"""Build or verify a separate 4096 plan; shared pilot payloads remain read-only."""
import argparse
import json
from pathlib import Path
import runpy
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from maimai_baseline.data.curriculum import EpochPlanStore
from maimai_baseline.data.training_dataset import inside, read
from maimai_baseline.data.training_build import CurriculumSource, plan_inputs, plan_metrics
from maimai_baseline.data.reduced_spatial import build_plans, verify_plans, validate_parent


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("build", "verify"))
    p.add_argument("--dataset-manifest", type=Path, default=PROJECT / "training-dataset/dataset.json")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    manifest = args.dataset_manifest.resolve()
    runpy.run_path(str(PROJECT / "scripts/training-dataset-args.py"))["arguments"](manifest, "4096")
    data = read(manifest)
    selection = read(inside(manifest.parent, data["selection"]))
    if (len(set(selection["train_song_ids"])) != 256 or len(set(selection["validation_song_ids"])) != 16
            or set(selection["train_song_ids"]) & set(selection["validation_song_ids"])):
        raise ValueError("reduced spatial experiment requires original 256/16 pilot")
    source = CurriculumSource(manifest.parent)
    parent = EpochPlanStore(inside(manifest.parent, data["epoch_plans"]["4096"]))
    validate_parent(parent, source, plan_inputs(manifest.parent))
    metrics = plan_metrics(manifest.parent, source)
    if args.action == "build":
        report = build_plans(parent, args.output, segments=source.segments, **metrics)
    else:
        report = verify_plans(parent, EpochPlanStore(args.output), segments=source.segments, **metrics)
    print(json.dumps({k: v for k, v in report.items() if k != "epoch_audits"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
