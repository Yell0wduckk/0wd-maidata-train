"""Explicit post-transfer integrity verification of portable training inputs."""
from pathlib import Path
import json
import sys
import time

from maimai_baseline.data.store import contained, completed, read_json
from maimai_dataset.integrity import file_sha256, hashing_session


def input_files(args):
    """Enumerate runtime artifacts, excluding original audio and build provenance."""
    def records(root, value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                yield contained(root, value["path"]), value["sha256"]
            else:
                for child in value.values():
                    yield from records(root, child)
        elif isinstance(value, list):
            for child in value:
                yield from records(root, child)

    for name in ("chart_cache", "spatial_view_cache", "physical_interval_cache"):
        root = Path(getattr(args, name)).resolve()
        manifest = read_json(root / "manifest.json")
        yield root / "manifest.json", None
        for key in ("arrays", "offsets", "segment_index", "audio_references", "views", "charts", "segments"):
            yield from records(root, manifest.get(key, {}))

    root = Path(getattr(args, "physical_source_rows", None) or args.physical_path_catalog).resolve()
    manifest = read_json(root / "manifest.json")
    yield root / "manifest.json", None
    for view in manifest["views"].values():
        for key in ("source_row_index", "source_row_pack"):
            yield from records(root, view[key])
    # The bitplane manifest binds these resource identities to the selected cache.
    bitplane = read_json(Path(args.physical_interval_cache) / "manifest.json")
    for name, key in (("paths.json", "paths_sha256"),
                      ("runtime-path-semantics.json", "semantics_sha256"),
                      ("runtime-path-rules.json", "rules_sha256")):
        yield Path(args.physical_path_catalog).resolve() / name, bitplane["path_catalog"][key]

    root = Path(args.route_artifact).resolve()
    manifest = read_json(root / "manifest.json")
    yield root / "manifest.json", None
    from maimai_baseline.route_contract import GEOMETRY_FILE
    name = GEOMETRY_FILE
    yield root / name, manifest["outputs"][name]["sha256"]

    root = Path(args.epoch_plan_cache).resolve()
    yield root / "manifest.json", None
    yield root / "audit.jsonl", None
    for line in (root / "audit.jsonl").read_text("utf-8-sig").splitlines():
        if line.strip():
            row = json.loads(line)
            yield contained(root, row["file"]), row["file_sha256"]

    manifest_path = Path(args.bpm_audio_manifest).resolve()
    root = manifest_path.parent
    value = read_json(manifest_path)
    audit = read_json(root / "audit.json")
    material = audit["inputs"]

    def published(path, identity, expected=None):
        receipt = completed(path, identity, verify_payloads=False)
        if receipt is None:
            raise ValueError(f"missing published artifact: {path}")
        if expected is not None and receipt["sha256"] != expected:
            raise ValueError(f"artifact receipt/manifest hash mismatch: {path}")
        yield path.with_name(path.name + ".complete.json"), None
        yield path, receipt["sha256"]

    yield from published(manifest_path, material)
    yield from published(root / "audit.json", material, value["audit_sha256"])
    yield from published(contained(root, value["stats_path"]), material, value["stats_sha256"])
    for variant in value["variants"].values():
        identity = {"preparation": value["preparation_identity"], "variant": variant["identity"]}
        for field, digest in (("feature_path", "feature_sha256"), ("coordinates_path", "coordinates_sha256")):
            yield from published(contained(root, variant[field]), identity, variant[digest])
    geometry_root = Path(args.interpolation_geometry).resolve()
    yield from published(geometry_root / "manifest.json", material, value["geometry_manifest_sha256"])
    geometry = read_json(geometry_root / "manifest.json")
    profile = Path(args.dataset).resolve()
    for name in ("profile.json", "summary.json", "vocabulary.json"):
        yield profile / name, None
    stats = (profile / read_json(profile / "profile.json")["base_feature_stats"]).resolve()
    yield stats, value["stats_sha256"]
    if getattr(args, "normalized_audio_manifest", None):
        yield Path(args.normalized_audio_manifest).resolve(), value["normalization_manifest_sha256"]
    yield profile / "manifest.jsonl", value["profile_manifest_sha256"]
    yield profile / "segments.jsonl", value["segments_sha256"]
    seen = set()
    for line in (profile / "segments.jsonl").read_text("utf-8-sig").splitlines():
        if not line.strip():
            continue
        segment = json.loads(line)
        sid = str(segment["segment_id"])
        if sid not in value["segments"]:
            continue
        record = geometry["segments"][sid]
        path = contained(geometry_root, record["path"])
        variant = value["variants"][value["segments"][sid]]
        identity = {"preparation": value["preparation_identity"], "variant": variant["identity"],
                    "start_tick": int(segment["target_start_tick"])}
        signature = (path, record["sha256"], json.dumps(identity, sort_keys=True))
        if signature not in seen:
            seen.add(signature)
            yield from published(path, identity, record["sha256"])


def verify_inputs(args):
    from maimai_baseline.training.inputs import _prepare_inputs

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    report = {"schema": "maimai.training_input_verification.v1", "scope": "portable-file-integrity",
              "passed": False, "files": [], "failures": []}

    def progress(path, count):
        print(json.dumps({"stage": "verify-inputs", "path": str(path), "bytes_hashed": count}),
              file=sys.stderr, flush=True)

    with hashing_session(progress) as session:
        try:
            print("verify-inputs: checking portable files", file=sys.stderr, flush=True)
            checked = {}
            for path, expected in input_files(args):
                path = path.resolve()
                if path in checked:
                    actual = checked[path]
                    if expected is not None and actual != expected:
                        raise ValueError(f"conflicting file identity: {path}")
                    continue
                try:
                    actual = file_sha256(path)
                    if expected is not None and actual != expected:
                        raise ValueError("file SHA-256 mismatch")
                    checked[path] = actual
                    report["files"].append({"path": str(path), "sha256": actual})
                except Exception as error:
                    report["failures"].append({"path": str(path), "error": str(error)})
            if not report["failures"]:
                print("verify-inputs: checking manifest relationships", file=sys.stderr, flush=True)
                inputs = _prepare_inputs(args)
                report["baseline_manifest"] = inputs.manifest
                report["passed"] = True
        except Exception as error:
            report["failures"].append({"error": str(error)})
        report.update(files_hashed=len(session["cache"]), bytes_hashed=session["bytes"],
                      seconds=time.perf_counter() - start)
    report["exit_code"] = 0 if report["passed"] else 2
    (output / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(f"verify-inputs: passed={report['passed']} files={report['files_hashed']} seconds={report['seconds']:.2f}",
          file=sys.stderr, flush=True)
    return report
