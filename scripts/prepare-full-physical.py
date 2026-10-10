"""Audit pilot reuse, build missing full spatial/bitplane payloads, and verify.

The audit stage reads sources and writes a report; every expensive build is an
explicit subcommand. This entrypoint never starts training or contacts a host.
"""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from maimai_baseline.data.full_dataset import FullDatasetSpec, build_epoch_plans
from full_dataset_tools.pipeline import audit, load_plan, spatial_build, bitplane_build, dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["adopt-pilot", "audit", "spatial", "bitplanes", "plans", "verify"])
    parser.add_argument("--config", type=Path, default=Path("training-dataset/full-dataset/config.json"))
    parser.add_argument("--pilot", type=Path, default=Path("training-dataset"))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    spec = FullDatasetSpec.load(args.config)
    if args.stage == "adopt-pilot":
        from full_dataset_tools.adoption import adopt
        result = adopt(spec, args.pilot, spec.root / "pilot-adoption-policy.json")
        dump(spec.reports / "pilot-adoption.json", result)
    elif args.stage == "audit":
        result = audit(spec, args.pilot)
    else:
        plan = load_plan(spec)
        if args.stage == "spatial":
            result = spatial_build(spec, plan)
        elif args.stage == "bitplanes":
            result = bitplane_build(spec, plan, args.workers)
        elif args.stage == "plans":
            result = build_epoch_plans(spec)
        else:
            from maimai_baseline.data.full_physical import verify_full_physical
            result = verify_full_physical(spec.runtime, full_hash=True)
        dump(spec.reports / ("incremental-" + args.stage + ".json"), result)
    print(json.dumps({k: result[k] for k in ("passed", "status", "counts", "identity", "chart_count", "plan_identity", "adopted_record_ids", "budgets", "epochs") if k in result}, ensure_ascii=False))
    if result.get("passed") is False:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
