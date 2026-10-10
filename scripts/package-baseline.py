"""Package a clean, locally accepted checkout; no uploads or remote execution."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

from maimai_baseline.source_identity import baseline_source_identity


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_bytes((json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runbook", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    for document in (args.runbook, args.report):
        if not document.is_file():
            parser.error(f"delivery document does not exist: {document}")
    repo = Path(__file__).resolve().parents[1]
    def git(*arguments):
        return subprocess.check_output(["git", "-c", f"safe.directory={repo.as_posix()}", *arguments], cwd=repo).decode("utf8").strip()
    if git("status", "--porcelain"):
        raise ValueError("commit and review the checkout before packaging")
    accepted = json.loads(args.acceptance.read_text("utf8"))
    source = baseline_source_identity()
    if not accepted.get("passed"):
        raise ValueError("local acceptance does not match this runtime source")
    evidence_root = args.acceptance.resolve().parent
    evidence = []
    for item in accepted["evidence_files"]:
        path = (evidence_root / item["path"]).resolve()
        if (not path.is_relative_to(evidence_root) or not path.is_file()
                or digest(path) != item["sha256"]):
            raise ValueError(f"acceptance evidence is missing or changed: {item['path']}")
        evidence.append((path, path.relative_to(evidence_root)))
    if Path(__import__("maimai_baseline").__file__).resolve().parent != repo / "src/maimai_baseline":
        raise ValueError("PYTHONPATH points outside the selected worktree")
    paths = git("ls-files", "-z").split("\0")
    selected = sorted(p for p in paths if p and
        (p.split("/")[0] in {"src", "tests", "scripts", "configs", "schemas", "docs"} or
         p in {"AGENTS.md", "README.md", "pyproject.toml"}))
    entries = []
    for relative in selected:
        path = repo / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"expected regular checkout file: {path}")
        entries.append(dict(path=relative, bytes=path.stat().st_size, sha256=digest(path)))
    args.output.mkdir(parents=True, exist_ok=False)
    archive = args.output / "code.tar.gz"
    with tarfile.open(archive, "x:gz") as tar:
        for entry in entries:
            raw = (repo / entry["path"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                raise ValueError("source changed during packaging")
            info = tarfile.TarInfo("code/" + entry["path"])
            info.size, info.mode, info.mtime = len(raw), 0o644, 0
            tar.addfile(info, io.BytesIO(raw))
    # Verify archive contents, not just the inputs from which it was made.
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        if [m.name for m in members] != ["code/" + e["path"] for e in entries]:
            raise ValueError("archive member mismatch")
        for member, entry in zip(members, entries, strict=True):
            if not member.isfile() or hashlib.sha256(tar.extractfile(member).read()).hexdigest() != entry["sha256"]:
                raise ValueError(f"archive content mismatch: {member.name}")
    manifest = dict(schema="maimai.code_handoff.v1", created_at=datetime.now(timezone.utc).isoformat(),
        head=git("rev-parse", "HEAD"), branch=git("branch", "--show-current"), status="",
        source_identity=source, files=entries,
        archive=dict(path=archive.name, sha256=digest(archive), bytes=archive.stat().st_size),
        scope="Native AR six-layer baseline; acceptance scope is defined by the included evidence")
    write_json(args.output / "code-upload-manifest.json", manifest)
    shutil.copyfile(args.acceptance, args.output / "local-acceptance.json")
    shutil.copyfile(args.runbook, args.output / "training-runbook.md")
    shutil.copyfile(args.report, args.output / "implementation-report.md")
    for source_path, relative in evidence:
        target = args.output / "evidence" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target)
    # All named evidence must travel with SHA256SUMS; the runbook says to upload the complete directory.
    names = sorted(p.relative_to(args.output).as_posix() for p in args.output.rglob("*") if p.is_file())
    (args.output / "SHA256SUMS").write_bytes("".join(f"{digest(args.output/name)}  {name}\n" for name in names).encode())
    print(json.dumps(dict(output=str(args.output.resolve()), files=len(entries), head=manifest["head"],
        archive=manifest["archive"], source_sha256=source["source_sha256"]), indent=2))


if __name__ == "__main__":
    main()
