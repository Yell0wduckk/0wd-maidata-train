from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from maimai_baseline.contract import IMPLEMENTATION_ABI
from maimai_baseline.contract import SOURCE_IDENTITY_SCHEMA


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def baseline_source_identity() -> dict[str, Any]:
    """Capture source provenance once when a run is prepared.

    This snapshot is descriptive, not a training/resume admission check.
    Checkpoint writers reuse the run's snapshot; they must not call this
    scanner on each save. Compatibility uses explicit versions and actual
    model/data structures instead of equality with these file hashes.
    """

    package_root = Path(__file__).resolve().parent
    project_root = package_root.parents[1]
    dataset_root = package_root.parent / "maimai_dataset"
    training_dataset_modules = (
        "array_store.py",
        "audio.py",
        "chart_cache.py",
        "codec.py",
        "dataset.py",
        "encoding.py",
        "event_time.py",
        "geometry.py",
        "integrity.py",
        "manifest.py",
        "modifiers.py",
        "timing.py",
        "vocabulary.py",
        "window.py",
    )
    paths = sorted([
        project_root / "configs/master-remaster.json",
        project_root / "configs/full-master-remaster.json",
        *(p for p in package_root.rglob("*") if p.suffix in {".py", ".cpp", ".cu", ".h"}),
        *(package_root / "assets").glob("*.json"),
        *(dataset_root / name for name in training_dataset_modules),
    ])
    entries = [
        {
            "path": path.relative_to(project_root).as_posix(),
            "sha256": _sha256(path),
        }
        for path in paths
    ]
    canonical = "\n".join(
        f"{entry['path']}\0{entry['sha256']}" for entry in entries
    ).encode("utf-8")
    return {
        "schema": SOURCE_IDENTITY_SCHEMA,
        "implementation_abi": IMPLEMENTATION_ABI,
        "source_sha256": hashlib.sha256(canonical).hexdigest(),
        "files": entries,
    }


__all__ = ["baseline_source_identity"]
