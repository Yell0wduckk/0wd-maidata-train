"""Local dataset lifecycle. Real payloads live only in the main checkout."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import time
from maimai_dataset.window import WINDOW
from maimai_dataset.manifest import canonical_json_sha256 as identity
from maimai_dataset.integrity import file_sha256 as sha

REMOTE_ROOT = "/root/autodl-tmp/maimai-ai-charting/v4-window-conformer-pilot256/route71-delta10ms-physical-bitplane/cache/physical-bitplane-pilot256-v2"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_rows(path, values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(v, ensure_ascii=False, sort_keys=True) + "\n" for v in values), encoding="utf-8")


def inside(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f"path escapes root: {relative}")
    return path


def copy_verified(source, target, digest):
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target:
        raise ValueError("migration source equals destination")
    if sha(source) != digest:
        raise ValueError(f"source changed: {source}")
    if target.exists():
        if sha(target) != digest:
            raise ValueError(f"conflicting destination: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".partial")
    shutil.copyfile(source, temporary)
    if sha(temporary) != digest:
        raise ValueError(f"copy verification failed: {source}")
    os.replace(temporary, target)


def inventory(root):
    root = Path(root).resolve()
    result = {}
    for path in sorted(root.rglob("*")):
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"inventory path escapes root: {path}")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = {"sha256": sha(path), "bytes": path.stat().st_size}
    return result


def check_inventory(root, files):
    for name, meta in files.items():
        path = inside(root, name)
        if not path.is_file() or path.stat().st_size != meta["bytes"] or sha(path) != meta["sha256"]:
            raise ValueError(f"missing/corrupt prepared input: {path}")


def default_sources(workspace):
    return {
        "audio": str(workspace / "artifacts/bpm96-pilot-normalized-20260906-r3"),
        "profile": str(workspace / "ssh-tmp/remote-upload-codec-ablation-20260818/dataset-view/profiles/v2-test1-all-difficulties"),
        "spatial": str(workspace / "artifacts/route71-spatial-augmentation-v1/multiview-route71-cache"),
        "physical_views": str(workspace / "artifacts/route71-spatial-augmentation-v1/physical-state-multiview-v2"),
        "route": str(workspace / "artifacts/slide-route-embedding-434-v4-dual"),
        "selection": str(workspace / "artifacts/physical-loss-v1/pilot-256-rebuilt-profile-suite.json"),
        "ssh_host": "root@connect.cqa1.seetacloud.com", "ssh_port": 13223, "remote_bitplanes": REMOTE_ROOT,
    }


def audit(root, sources):
    from maimai_baseline.time.grid import SAMPLING
    audio_root = Path(sources["audio"]).resolve(); audio = read(audio_root / "bpm-audio-manifest.json")
    profile = Path(sources["profile"]).resolve(); spatial = read(Path(sources["spatial"]) / "manifest.json")
    selected = read(sources["selection"])
    train, validation = [list(map(str, selected[k])) for k in ("train_song_ids", "validation_song_ids")]
    songs = set(train + validation)
    if (len(train), len(validation), len(songs)) != (256, 16, 272) or songs != set(audio["song_ids"]):
        raise ValueError("selection must preserve the 256/16 pilot split")
    if sha(profile / "manifest.jsonl") != audio["profile_manifest_sha256"] or sha(profile / "segments.jsonl") != audio["segments_sha256"]:
        raise ValueError("canonical profile differs from audio provenance")
    if spatial["profile_manifest_sha256"] != audio["profile_manifest_sha256"] or spatial["selection_sha256"] != audio["selection_sha256"]:
        raise ValueError("spatial/audio provenance mismatch")
    if {k:v for k,v in audio["sampling"].items() if k != "window_positions"} != SAMPLING:
        raise ValueError("audio feature sampling changed; feature reuse is unsafe")
    records = [r for r in rows(profile / "manifest.jsonl") if str(r["song_id"]) in songs and r.get("status") == "eligible"]
    if len(records) != 1097:
        raise ValueError("pilot chart coverage differs from 1097")
    files, migrations = {}, []
    def source(path):
        path = Path(path).resolve()
        files[str(path)] = {"sha256": sha(path), "bytes": path.stat().st_size}
        return files[str(path)]["sha256"]
    def migrate(path, destination):
        digest = source(path); resolved = str(Path(path).resolve())
        migrations.append({"source": resolved, "destination": destination, "sha256": digest, "bytes": files[resolved]["bytes"]})
    for key in ("summary.json", "profile.json", "vocabulary.json", "manifest.jsonl", "segments.jsonl"):
        source(profile / key)
    for r in records:
        source(inside(profile, r["encoded_path"]))
    for name in ("bpm-audio-manifest.json", "audit.json"):
        source(audio_root / name)
    for variant in audio["variants"].values():
        for kind in ("feature", "coordinates"):
            relative = variant[kind + "_path"]
            path = inside(audio_root, relative)
            if source(path) != variant[kind + "_sha256"]:
                raise ValueError(f"audio source payload corrupt: {relative}")
            migrate(path, "audio/" + relative)
    migrate(audio_root / audio["stats_path"], "audio/feature-stats.json")
    state_root = Path(sources["physical_views"]); state = read(state_root / "manifest.json")
    source(state_root / "manifest.json"); source(Path(sources["spatial"]) / "manifest.json"); source(sources["selection"])
    if state["profile_fingerprint"] != spatial["profile_fingerprint"]:
        raise ValueError("Physical views/canonical provenance mismatch")
    for view, meta in state["views"].items():
        for kind in ("chart_index", "chart_pack", "source_row_index", "source_row_pack"):
            path = inside(state_root, meta[kind]["path"])
            if source(path) != meta[kind]["sha256"]:
                raise ValueError(f"Physical view input corrupt: {view}/{kind}")
            if kind.startswith("source_row"):
                migrate(path, "physical/source-rows/" + meta[kind]["path"])
    for name in ("paths.json", "runtime-path-semantics.json", "runtime-path-rules.json"):
        migrate(state_root / name, "physical/path-catalog/" + name)
    # Only these two files are consumed by the current Route reader.
    route = Path(sources["route"])
    from maimai_baseline.model.route import load_route_artifacts
    load_route_artifacts(route)
    from maimai_baseline.route_contract import GEOMETRY_FILE
    for name in ("manifest.json", GEOMETRY_FILE):
        migrate(route / name, "route-artifact/" + name)
    migrations = list({(m["source"], m["destination"]): m for m in migrations}.values())
    result = {"schema": "maimai.training_dataset_audit.v1", "window": WINDOW, "sources": sources,
              "files": files, "migrations": migrations, "selection": {"train_song_ids": train, "validation_song_ids": validation},
              "charts": len(records), "chart_ids": [r["record_id"] for r in records], "source_profile_fingerprint": spatial["profile_fingerprint"],
              "minimum_copy_bytes": sum(m["bytes"] for m in migrations), "disk_free_bytes": shutil.disk_usage(root).free}
    # Conservative upper bound includes four vector views, 22-beat geometry,
    # retained staging copies, and both 75-epoch plans; it is an estimate, not allocation.
    vector_bytes = sum(Path(sources["spatial"]).joinpath(m["path"]).stat().st_size
                       for v in spatial["views"].values() for m in v["arrays"].values())
    result["estimated_additional_peak_bytes"] = result["minimum_copy_bytes"] + 2 * 660_000_000 + 3 * vector_bytes + 4_000_000_000
    result["remote_missing_resources"] = {"host": sources["ssh_host"], "port": sources["ssh_port"],
        "root": sources["remote_bitplanes"], "action": "fetch validates manifest then exact referenced payloads"}
    result["identity"] = identity({k:v for k,v in result.items() if k != "disk_free_bytes"})
    path = root / "preparation/reports/audit.json"
    if path.exists() and read(path)["identity"] != result["identity"]:
        raise ValueError("audit inputs changed; use a new explicitly selected dataset root")
    write(path, result)
    dataset_path = root / "dataset.json"
    if not dataset_path.exists() or not read(dataset_path).get("ready"):
        write(root / "selection.json", {"schema": "maimai.small_sample_suite.v1", **result["selection"]})
        write(dataset_path, {"schema": "maimai.training_dataset.v1", "window": WINDOW, "ready": False,
            "audit_identity": result["identity"], "selection": "selection.json", "charts": len(records),
            "audit": "preparation/reports/audit.json", "preparation_status": {
                "audit": "passed", "fetch": "pending", "build": "pending", "verify": "pending", "finalize-migration": "pending"}})
    return result


def load_audit(root, *, check_sources=True):
    result = read(root / "preparation/reports/audit.json")
    if result["window"] != WINDOW or result["identity"] != identity({k:v for k,v in result.items() if k not in ("identity", "disk_free_bytes")}):
        raise ValueError("audit identity/window mismatch")
    if check_sources:
        for path, meta in result["files"].items():
            if not Path(path).is_file() or sha(Path(path)) != meta["sha256"]:
                raise ValueError(f"audited source changed or missing: {path}")
    return result


def fetch(root, audit):
    import shlex
    source = audit["sources"]
    host, port, remote = source["ssh_host"], int(source["ssh_port"]), source["remote_bitplanes"]
    if not re.fullmatch(r"[A-Za-z0-9_.@-]+", host) or host.startswith("-") or not 1 <= port <= 65535 or not re.fullmatch(r"/[A-Za-z0-9_./-]+", remote):
        raise ValueError("invalid SSH destination")
    ssh = ["ssh", "-p", str(port), "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host]
    code = "import pathlib,sys;sys.stdout.buffer.write(pathlib.Path(" + repr(remote + "/manifest.json") + ").read_bytes())"
    local = Path(source["physical_bitplanes"]).resolve() if source.get("physical_bitplanes") else None
    raw = (local / "manifest.json").read_bytes() if local else subprocess.check_output([*ssh, "python -c " + shlex.quote(code)])
    manifest = json.loads(raw)
    from maimai_baseline.physical.contract import BITPLANE_CACHE_SCHEMA
    if (manifest["profile_fingerprint"] != audit["source_profile_fingerprint"] or manifest["chart_count"] != audit["charts"]
            or manifest["schema"] != BITPLANE_CACHE_SCHEMA):
        raise ValueError("remote Physical provenance/coverage mismatch")
    target = root / "preparation/staging/physical-original"
    target.mkdir(parents=True, exist_ok=True)
    path = target / "manifest.json"
    if path.exists() and path.read_bytes() != raw:
        raise ValueError("remote Physical manifest changed")
    path.write_bytes(raw)
    files = [manifest[k] for k in ("charts", "segments")] + list(manifest["arrays"].values()) + list(manifest["offsets"].values())
    for meta in files:
        rel = meta["path"]
        if not re.fullmatch(r"[A-Za-z0-9_./-]+", rel) or PurePosixPath(rel).is_absolute() or ".." in PurePosixPath(rel).parts:
            raise ValueError("remote payload path escapes cache")
        dest = inside(target, rel)
        if dest.exists():
            if sha(dest) != meta["sha256"]: raise ValueError(f"corrupt downloaded file: {dest}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        partial = dest.with_name(dest.name + ".partial")
        if local:
            shutil.copyfile(inside(local, rel), partial)
        else:
            result = subprocess.run(["scp", "-P", str(port), "-o", "BatchMode=yes", f"{host}:{remote}/{rel}", str(partial)], capture_output=True, text=True)
            print(result.stdout); print(result.stderr, file=sys.stderr); result.check_returncode()
        if sha(partial) != meta["sha256"]: raise ValueError(f"download hash mismatch: {rel}")
        os.replace(partial, dest)
    from maimai_baseline.physical.bitplane_cache import PhysicalBitplaneStore
    store = PhysicalBitplaneStore(target); store.validate(full_hash=True, scan_contents=True)
    report = {"passed": True, "remote_manifest_sha256": sha(path), "files": inventory(target), "audit_identity": audit["identity"]}
    write(root / "preparation/reports/fetch.json", report)
    return report


def stage(root, name, audit, fn):
    receipt = root / f"preparation/reports/{name}.json"
    if receipt.exists():
        saved = read(receipt)
        if saved['audit_identity'] != audit['identity']:
            raise ValueError(f"stage configuration/source changed: {name}")
        check_inventory(root, saved["files"])
        return saved
    output_paths = fn()
    files = {}
    for path in output_paths:
        if Path(path).is_dir():
            files.update({(Path(path).relative_to(root)/k).as_posix():v for k,v in inventory(path).items()})
        else:
            files[Path(path).relative_to(root).as_posix()] = {"sha256":sha(path), "bytes":Path(path).stat().st_size}
    result = {"passed": True, "audit_identity":audit["identity"], "source_identity":code_identity(), "files":files}
    write(receipt,result)
    return result


def code_identity():
    from maimai_baseline.source_identity import baseline_source_identity
    return baseline_source_identity()


def finalize(root):
    audit = load_audit(root, check_sources=False)
    report = read(root / "preparation/reports/verify.json")
    dataset = read(root / "dataset.json")
    if dataset.get("verification_sha256") != sha(root / "preparation/reports/verify.json") or not dataset.get("ready"):
        raise ValueError("dataset does not bind a successful verification report")
    if not report.get('passed') or report['audit_identity'] != audit['identity']:
        raise ValueError("matching successful verification required before cleanup")
    check_inventory(root, report["files"])
    receipt = root / "preparation/reports/cleanup.json"
    saved = read(receipt) if receipt.exists() else {"audit_identity":audit["identity"], "deleted":[]}
    if saved["audit_identity"] != audit["identity"]: raise ValueError("cleanup identity mismatch")
    # Preflight every pending source before deleting any file.
    for item in audit["migrations"]:
        dest = inside(root,item["destination"]); source = Path(item["source"]).resolve()
        if str(source) != item["source"] or source.is_relative_to(root) or dest == source or sha(dest) != item["sha256"]:
            raise ValueError("invalid migration deletion boundary")
        if source.exists() and sha(source) != item["sha256"]:
            raise ValueError(f"migration source changed: {source}")
    saved.setdefault("results", {})
    for item in audit["migrations"]:
        source = Path(item["source"]).resolve()
        existed = source.exists()
        if existed:
            if sha(source) != item["sha256"]: raise ValueError(f"source changed during cleanup: {source}")
            source.unlink()
        if str(source) not in saved["results"]:
            saved["results"][str(source)] = {"destination": item["destination"], "sha256": item["sha256"],
                "result": "deleted" if existed else "already_absent", "verified_destination": True}
        if existed and str(source) not in saved["deleted"]: saved["deleted"].append(str(source))
        write(receipt,saved)
    dataset = read(root/"dataset.json"); dataset["migration_finalized"] = True; write(root/"dataset.json",dataset)
    return saved


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("audit","fetch","build","verify","finalize-migration"))
    p.add_argument("--main-worktree",type=Path,default=Path(__file__).resolve().parents[4]/"maidata-baseline")
    p.add_argument("--sources",type=Path,help="explicit source-location overrides JSON")
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    main_root=args.main_worktree.resolve()
    # Git registry proves this is the main checkout, not an arbitrary child worktree.
    registry=subprocess.check_output(["git","-C",str(main_root),"worktree","list","--porcelain"],text=True)
    registered=Path(registry.splitlines()[0].removeprefix("worktree ")).resolve()
    if registered != main_root: raise ValueError("dataset root must belong to the main worktree")
    root=main_root/"training-dataset"
    if root.resolve() != root: raise ValueError("training-dataset must not be a junction or symbolic link")
    root.mkdir(exist_ok=True)
    if subprocess.run(["git","-C",str(main_root),"check-ignore","-q","training-dataset/probe"]).returncode:
        raise ValueError("training-dataset must be ignored by Git before preparation")
    log=root/"preparation/logs"; log.mkdir(parents=True,exist_ok=True)
    stamp=time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns()}"; output=log/f"{stamp}-{args.action}.log"
    status=1
    try:
        from contextlib import redirect_stdout,redirect_stderr
        with output.open("w",encoding="utf-8") as stream, redirect_stdout(stream),redirect_stderr(stream):
            if args.action == "audit":
                sources=default_sources(main_root.parent.parent)
                if args.sources: sources.update(read(args.sources))
                result=audit(root,sources)
            elif args.action == "finalize-migration": result=finalize(root)
            else:
                state=load_audit(root,check_sources=args.action in ("fetch","build"))
                if args.action == "fetch": result=fetch(root,state)
                else:
                    from maimai_baseline.data.training_build import build, verify
                    dataset = read(root / "dataset.json")
                    dataset["ready"] = False
                    write(root / "dataset.json", dataset)
                    result=(build if args.action=="build" else verify)(root,state)
            if args.action == "build":
                write(root / "preparation/reports/build.json", {**result, "passed": True,
                    "audit_identity": state["identity"], "source_identity": code_identity()})
            dataset = read(root / "dataset.json")
            dataset.setdefault("preparation_status", {})[args.action] = "passed"
            write(root / "dataset.json", dataset)
            print(json.dumps({k:v for k,v in result.items() if k not in {"files", "migrations", "chart_ids", "deleted", "selection", "sources"}},ensure_ascii=False,sort_keys=True))
        status=0
    except Exception:
        import traceback
        with output.open("a", encoding="utf-8") as stream: traceback.print_exc(file=stream)
        raise
    finally:
        write(log/f"{stamp}-{args.action}-exit.json",{"exit_code":status,"log":output.name})
        print(f"{args.action}: exit={status}; log={output}")
