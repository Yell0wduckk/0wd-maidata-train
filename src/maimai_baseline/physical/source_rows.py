"""Read the existing view-position -> canonical NOTE-group row maps.

This is a training-only reader. Branch ordinals retain the cache's semantics;
matching individual paths when a compound Slide is reordered is a separate task.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping
import zlib

import numpy as np
import torch

from maimai_dataset.manifest import DatasetError, load_jsonl
from maimai_baseline.contract import VIEW_NAMES, cache_view_names

SOURCE_ROW_POLICY = "view-position-to-canonical-note-group-v1"


def _sha256(path: Path) -> str:
    from maimai_dataset.integrity import file_sha256
    return file_sha256(path)


class PhysicalSourceRowStore:
    def __init__(
        self, root: Path, *, profile_fingerprint: str, vocabulary_sha256: str,
        chart_cache_manifest_sha256: str, spatial_cache_manifest_sha256: str,
        verify_payloads: bool = True,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        path = self.root / "manifest.json"
        manifest = json.loads(path.read_text("utf-8-sig"))
        if manifest.get("schema") != "maimai.physical_state.multiview.v2":
            raise DatasetError("Physical source-row manifest schema mismatch")
        if manifest.get("source_row_mapping") != "view-position-to-canonical-int32-v1":
            raise DatasetError("Physical source-row mapping policy mismatch")
        fingerprint = manifest.get("state_fingerprint")
        self.indices: dict[str, dict[str, Any]] = {}
        self.packs: dict[str, Path] = {}
        resources = {}
        try:
            available_views = cache_view_names(manifest.get("views", {}))
        except ValueError as error:
            raise DatasetError("Physical source-row view order mismatch") from error
        for view in available_views:
            metadata = manifest.get("views", {}).get(view, {})
            checked = {}
            for kind in ("source_row_index", "source_row_pack"):
                meta = metadata.get(kind)
                if not isinstance(meta, Mapping):
                    raise DatasetError(f"Physical source-row resource missing: {view}/{kind}")
                resource = (self.root / str(meta["path"])).resolve()
                if not resource.is_relative_to(self.root):
                    raise DatasetError("Physical source-row resource escaped catalog root")
                if verify_payloads and _sha256(resource) != meta["sha256"]:
                    raise DatasetError(f"Physical source-row resource hash mismatch: {view}/{kind}")
                checked[kind] = resource
            rows = load_jsonl(checked["source_row_index"])
            index = {str(row["record_id"]): row for row in rows}
            if len(index) != len(rows) or len(index) != int(manifest["chart_count"]):
                raise DatasetError(f"Physical source-row record coverage mismatch: {view}")
            size = checked["source_row_pack"].stat().st_size
            for record, entry in index.items():
                offset, length = int(entry["offset"]), int(entry["length"])
                count = int(entry["row_count"])
                if offset < 0 or length <= 0 or offset + length > size or count <= 0:
                    raise DatasetError(f"Physical source-row index bounds invalid: {view}/{record}")
                if int(entry["uncompressed_length"]) != count * 4:
                    raise DatasetError(f"Physical source-row index length invalid: {view}/{record}")
            self.indices[view] = index
            self.packs[view] = checked["source_row_pack"]
            resources[view] = {
                kind: dict(metadata[kind]) for kind in ("source_row_index", "source_row_pack")
            }
        if any(set(index) != set(self.indices[VIEW_NAMES[0]]) for index in self.indices.values()):
            raise DatasetError("Physical source-row views have different record coverage")
        self.identity = {
            "policy": SOURCE_ROW_POLICY,
            "branch_identity": "existing-group-ordinal-path-alignment-not-validated",
            "manifest_sha256": _sha256(path),
            "state_fingerprint": fingerprint,
            "resources": resources,
        }
        self._pid = os.getpid()
        self._rows: dict[tuple[str, str], torch.Tensor] = {}

    def __getstate__(self) -> dict[str, Any]:
        # Spawn workers own an identity-scoped cache and never inherit open files.
        return {**self.__dict__, "_rows": {}}

    def rows(self, view: str, record_id: str) -> torch.Tensor:
        """Return a read-only-by-convention int64 lookup, cached per process."""
        if self._pid != os.getpid():
            self._rows = {}
            self._pid = os.getpid()
        key = (str(view), str(record_id))
        if key in self._rows:
            return self._rows[key]
        try:
            entry = self.indices[key[0]][key[1]]
        except KeyError as error:
            raise DatasetError(f"Physical source-row map missing: {key}") from error
        with self.packs[key[0]].open("rb") as stream:
            stream.seek(int(entry["offset"]))
            compressed = stream.read(int(entry["length"]))
        try:
            raw = zlib.decompress(compressed)
        except zlib.error as error:
            raise DatasetError(f"Physical source-row compression invalid: {key}") from error
        count = int(entry["row_count"])
        if len(raw) != count * 4:
            raise DatasetError(f"Physical source-row record length mismatch: {key}")
        values = np.frombuffer(raw, dtype="<i4").astype(np.int64)
        if not np.array_equal(np.sort(values), np.arange(count)):
            raise DatasetError(f"Physical source-row map is not bijective: {key}")
        if view == "identity" and not np.array_equal(values, np.arange(count)):
            raise DatasetError(f"Physical identity-view source rows changed: {key}")
        tensor = torch.from_numpy(values)
        self._rows[key] = tensor
        return tensor

    def validate_records(self, row_counts: Mapping[str, int]) -> None:
        """Explicitly validate selected chart/view mappings for offline tools."""
        for record, count in row_counts.items():
            for view in self.indices:
                if self.rows(view, record).numel() != int(count):
                    raise DatasetError(f"Physical source-row/cache row count mismatch: {view}/{record}")
