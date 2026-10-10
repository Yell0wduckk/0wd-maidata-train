"""Spatial views and prebuilt epoch plans."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from maimai_baseline.contract import (
    CURRICULUM_EXCEPTIONS,
    CURRICULUM_PARTICIPANTS,
    DIFFICULTY_POOLS,
    EPOCH_PLAN_MODE,
    PLAN_AUDIT_SCHEMA,
    PLAN_CACHE_SCHEMA,
    VIEW_CACHE_SCHEMA,
    VIEW_NAMES,
    VIEW_WEIGHTS,
    cache_view_names,
)
from maimai_dataset.manifest import DatasetError, canonical_json_sha256, load_jsonl
from maimai_dataset.chart_cache import validate_vector_index
from maimai_dataset.array_store import ReadOnlyArrayStore
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import hashlib
import json
import numpy as np
import torch


def sha256_file(path: Path) -> str:
    from maimai_dataset.integrity import file_sha256
    return file_sha256(path)


class MultiViewRouteStore(ReadOnlyArrayStore):
    def __init__(
        self,
        *,
        root: Path,
        manifest: Mapping[str, Any],
        segment_indices: Mapping[str, int],
        segment_records: Mapping[str, Mapping[str, Any]],
        offsets: np.ndarray,
        arrays: Mapping[str, Mapping[str, np.ndarray]],
        route_mass_by_class: torch.Tensor,
    ) -> None:
        self.root = root
        self.manifest = dict(manifest)
        self.segment_indices = dict(segment_indices)
        self.segment_records = dict(segment_records)
        self.offsets = offsets
        self.arrays = {view: dict(values) for view, values in arrays.items()}
        self.route_mass_by_class = route_mass_by_class

    @classmethod
    def load(
        cls,
        root: Path,
        *,
        profile_fingerprint: str,
        vocabulary_sha256: str,
        verify_payloads: bool = True,
        selection_sha256: str,
    ) -> "MultiViewRouteStore":
        root = Path(root).expanduser().resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise DatasetError(f"multi-view cache manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text("utf-8-sig"))
        if manifest.get("schema") != VIEW_CACHE_SCHEMA:
            raise DatasetError("unsupported multi-view Route-71 cache schema")
        try:
            available_views = cache_view_names(manifest.get("views", {}))
        except ValueError as error:
            raise DatasetError("multi-view cache view order mismatch") from error
        offsets_meta = manifest["offsets"]
        offsets_path = root / str(offsets_meta["path"])
        if verify_payloads and sha256_file(offsets_path) != offsets_meta["sha256"]:
            raise DatasetError("multi-view cache offsets hash mismatch")
        offsets = np.load(offsets_path, mmap_mode="r", allow_pickle=False)
        segment_meta = manifest["segment_index"]
        segment_path = root / str(segment_meta["path"])
        if verify_payloads and sha256_file(segment_path) != segment_meta["sha256"]:
            raise DatasetError("multi-view cache segment index hash mismatch")
        rows = load_jsonl(segment_path)
        indices = {str(row["segment_id"]): int(row["cache_index"]) for row in rows}
        if len(indices) != int(manifest["segment_count"]):
            raise DatasetError("multi-view cache segment count mismatch")
        arrays: dict[str, dict[str, np.ndarray]] = {}
        for view in available_views:
            arrays[view] = {}
            metadata_by_name = manifest["views"][view]["arrays"]
            for name, metadata in metadata_by_name.items():
                path = root / str(metadata["path"])
                if verify_payloads and sha256_file(path) != metadata["sha256"]:
                    raise DatasetError(f"multi-view array hash mismatch: {view}/{name}")
                value = np.load(path, mmap_mode="r", allow_pickle=False)
                if list(value.shape) != list(metadata["shape"]):
                    raise DatasetError(f"multi-view array shape mismatch: {view}/{name}")
                if value.dtype.name != str(metadata["dtype"]):
                    raise DatasetError(f"multi-view array dtype mismatch: {view}/{name}")
                arrays[view][str(name)] = value
            validate_vector_index(offsets, indices, arrays[view], int(manifest["segment_count"]))
        names = [set(arrays[view]) for view in available_views]
        if any(value != names[0] for value in names[1:]):
            raise DatasetError("multi-view arrays do not share one field schema")
        removed = {"shape", "end_offset", "via_offset"}
        if any(
            name.split(".")[-1] in removed
            for name in names[0]
        ):
            raise DatasetError("multi-view Route-71 cache leaked removed route factors")
        for required in ("input_ids.route", "target_ids.route"):
            if required not in names[0]:
                raise DatasetError(f"multi-view Route-71 cache lacks {required}")
        mass_values = manifest.get("route_informative_mass_by_class")
        if not isinstance(mass_values, list) or len(mass_values) != 71:
            raise DatasetError("multi-view cache lacks the 71-class route mass table")
        route_mass_by_class = torch.tensor(mass_values, dtype=torch.float32)
        return cls(
            root=root,
            manifest=manifest,
            segment_indices=indices,
            segment_records={str(row["segment_id"]): row for row in rows},
            offsets=offsets,
            arrays=arrays,
            route_mass_by_class=route_mass_by_class,
        )

    def record(self, segment_id: str, view: str) -> dict[str, Any]:
        if view not in self.arrays:
            raise DatasetError(f"unknown spatial view {view!r}")
        try:
            index = self.segment_indices[str(segment_id)]
        except KeyError as error:
            raise DatasetError(
                f"multi-view cache does not contain segment {segment_id}"
            ) from error
        begin = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        result: dict[str, Any] = {
            "input_ids": {},
            "input_times": {},
            "input_time_active": {},
            "target_ids": {},
            "time_targets": {},
            "time_category_targets": {},
            "time_lower": {},
            "time_upper": {},
        }
        for name, values in self.arrays[view].items():
            tensor = torch.from_numpy(np.array(values[begin:end], copy=True))
            if "." not in name:
                result[name] = tensor.to(torch.float32 if name == "row_anchor_tick" else torch.long)
                continue
            group, field = name.split(".", 1)
            if group == "input_times":
                tensor = tensor.to(torch.float32)
            elif group == "input_time_active":
                tensor = tensor.to(torch.bool)
            else:
                tensor = tensor.to(torch.long)
            result[group][field] = tensor
        route_target = result["target_ids"]["route"]
        active = route_target.ne(-100)
        route_mass = torch.zeros(route_target.shape, dtype=torch.float32)
        route_mass[active] = self.route_mass_by_class.index_select(
            0, route_target[active]
        )
        result["route_informative_mass"] = route_mass
        return result


@dataclass(frozen=True, slots=True)
class ChartSlot:
    song_id: str
    difficulty_id: int
    occurrence: int = 0

    @property
    def chart_key(self) -> tuple[str, int]:
        return self.song_id, self.difficulty_id


@dataclass(frozen=True, slots=True)
class VirtualSegmentRef:
    canonical_index: int
    view_id: int
    occurrence: int

    @property
    def view(self) -> str:
        return VIEW_NAMES[self.view_id]


def _digest(*parts: Any) -> bytes:
    material = "\0".join(str(value) for value in parts).encode("utf-8")
    return hashlib.sha256(material).digest()


def _stable_order(values: Iterable[Any], *namespace: Any) -> list[Any]:
    return sorted(values, key=lambda value: (_digest(*namespace, value), str(value)))


def hamilton_quotas(
    total: int, weights: Mapping[str, int], *, tie_namespace: Sequence[Any] = ()
) -> dict[str, int]:
    total = int(total)
    if total < 0:
        raise ValueError("Hamilton total cannot be negative")
    denominator = sum(int(value) for value in weights.values())
    if denominator <= 0:
        raise ValueError("Hamilton weights must have positive mass")
    floors = {
        name: total * int(weight) // denominator for name, weight in weights.items()
    }
    remaining = total - sum(floors.values())
    order = sorted(
        weights,
        key=lambda name: (
            -(total * int(weights[name]) % denominator),
            _digest("hamilton", *tie_namespace, name),
            name,
        ),
    )
    for name in order[:remaining]:
        floors[name] += 1
    if sum(floors.values()) != total:
        raise AssertionError("Hamilton apportionment lost slots")
    return floors


def _round_half_up_fraction(count: int, numerator: int, denominator: int) -> int:
    value = (Decimal(int(count)) * Decimal(int(numerator))) / Decimal(int(denominator))
    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _allocate_exact(
    values: Sequence[str],
    counts: Mapping[int, int],
    *,
    seed: int,
    epoch: int,
    namespace: str,
) -> dict[str, int]:
    if sum(int(value) for value in counts.values()) != len(values):
        raise ValueError("exact category allocation count mismatch")
    ordered = _stable_order(values, seed, epoch, namespace)
    labels: list[int] = []
    for label in sorted(counts):
        labels.extend([int(label)] * int(counts[label]))
    labels.sort(key=lambda label: _digest(seed, epoch, namespace, "label", label))
    return dict(zip(ordered, labels, strict=True))


def curriculum_slots(
    charts: Mapping[str, Sequence[int]],
    *,
    epoch: int,
    seed: int,
    full_selection: bool = False,
) -> list[ChartSlot]:
    epoch = int(epoch)
    if not 1 <= epoch <= 75:
        raise ValueError("epoch must be within [1, 75]")
    normalized = {
        str(song): tuple(sorted({int(value) for value in values}))
        for song, values in charts.items()
    }
    if any(not d or not set(d).issubset({2, 3, 4, 5, 6}) for d in normalized.values()):
        raise DatasetError("curriculum contains empty or unsupported difficulties")
    exceptions = ({song for song, d in normalized.items() if not {2, 3, 4, 5}.issubset(d)}
                  if full_selection else set(CURRICULUM_EXCEPTIONS))
    participants = [
        song
        for song, difficulties in normalized.items()
        if song not in exceptions and {2, 3, 4, 5}.issubset(difficulties)
    ]
    unexpected = sorted(
        song
        for song, difficulties in normalized.items()
        if song not in exceptions and not {2, 3, 4, 5}.issubset(difficulties)
    )
    if unexpected:
        raise DatasetError(f"unexpected incomplete curriculum songs: {unexpected}")
    if not full_selection and len(participants) != CURRICULUM_PARTICIPANTS:
        raise DatasetError(
            "curriculum participant count drifted: "
            f"{len(participants)} != {CURRICULUM_PARTICIPANTS}"
        )
    missing_exceptions = sorted(exceptions - set(normalized))
    if missing_exceptions:
        raise DatasetError(f"curriculum exceptions missing from selection: {missing_exceptions}")

    if epoch <= 10:
        converted_count = 0
    elif epoch >= 40:
        converted_count = len(participants)
    else:
        converted_count = _round_half_up_fraction(
            len(participants), epoch - 10, 30
        )
    converted = set(
        _stable_order(participants, seed, epoch, "curriculum-converted")[
            :converted_count
        ]
    )
    drop_counts = hamilton_quotas(
        converted_count,
        {2: 60, 3: 40},
        tie_namespace=(seed, epoch, "curriculum-drop"),
    )
    add_counts = hamilton_quotas(
        converted_count,
        {4: 20, 5: 80},
        tie_namespace=(seed, epoch, "curriculum-add"),
    )
    drops = _allocate_exact(
        tuple(converted), drop_counts, seed=seed, epoch=epoch, namespace="drop"
    )
    additions = _allocate_exact(
        tuple(converted), add_counts, seed=seed, epoch=epoch, namespace="add"
    )
    result: list[ChartSlot] = []
    for song in sorted(normalized, key=lambda value: (int(value), value)):
        available = normalized[song]
        if song in exceptions or song not in converted:
            result.extend(ChartSlot(song, difficulty) for difficulty in available)
            continue
        drop = drops[song]
        add = additions[song]
        kept = [difficulty for difficulty in available if difficulty != drop]
        result.extend(ChartSlot(song, difficulty) for difficulty in kept)
        result.append(ChartSlot(song, add, 1))

    by_song: dict[str, Counter[int]] = defaultdict(Counter)
    for slot in result:
        by_song[slot.song_id][slot.difficulty_id] += 1
    for song in participants:
        counts = by_song[song]
        if counts[4] < 1 or counts[5] < 1 or counts[2] + counts[3] < 1:
            raise AssertionError(f"curriculum hard coverage failed for {song}")
        if 6 in normalized[song] and counts[6] != 1:
            raise AssertionError(f"Re:MASTER coverage failed for {song}")
    return result


def assign_spatial_views(
    slots: Sequence[ChartSlot], *, seed: int, epoch: int
) -> dict[ChartSlot, str]:
    """Allocate exact pool quotas; repeated chart occurrences may share a view."""
    if len(set(slots)) != len(slots):
        raise DatasetError("duplicate chart occurrence in spatial allocation")
    result: dict[ChartSlot, str] = {}
    by_pool: dict[str, list[ChartSlot]] = defaultdict(list)
    for slot in slots:
        by_pool[DIFFICULTY_POOLS[slot.difficulty_id]].append(slot)
    for pool, pool_slots in sorted(by_pool.items()):
        quotas = hamilton_quotas(len(pool_slots), VIEW_WEIGHTS,
                                tie_namespace=(seed, epoch, pool))
        ordered = _stable_order(pool_slots, seed, epoch, pool, "spatial-slots")
        offset = 0
        for view, count in quotas.items():
            for slot in ordered[offset:offset + count]:
                result[slot] = view
            offset += count
    return result


def validate_spatial_sampling(manifest: Mapping[str, Any]) -> None:
    from maimai_baseline.contract import spatial_sampling_identity
    if manifest.get("spatial_sampling") != spatial_sampling_identity():
        raise DatasetError("epoch plan spatial sampling policy mismatch")


class EpochPlanStore:
    def __init__(self, root: Path, *, verify_payloads: bool = True) -> None:
        self.verify_payloads = verify_payloads
        self.root = Path(root).expanduser().resolve()
        self.mode = EPOCH_PLAN_MODE
        self.manifest = json.loads((self.root / "manifest.json").read_text("utf-8"))
        if self.manifest.get("schema") not in (PLAN_CACHE_SCHEMA, "maimai.repacked_epoch_plan.v1"):
            raise DatasetError("unsupported epoch plan cache schema")
        if self.mode not in self.manifest.get("modes", {}):
            raise DatasetError(f"epoch plan cache lacks mode {self.mode}")
        audit_rows = [
            json.loads(line)
            for line in (self.root / "audit.jsonl").read_text("utf-8").splitlines()
            if line
        ]
        self._audit_by_epoch = {
            int(row["epoch"]): row
            for row in audit_rows
            if row.get("mode") == self.mode
        }
        epochs = int(self.manifest.get("epochs", 0))
        if epochs <= 0:
            raise DatasetError("epoch plan cache declares no epochs")
    @property
    def available_epochs(self) -> int:
        return int(self.manifest["epochs"])

    def load_epoch(self, epoch_index: int) -> tuple[tuple[VirtualSegmentRef, ...], ...]:
        epoch = int(epoch_index) + 1
        if not 1 <= epoch <= int(self.manifest["epochs"]):
            raise ValueError(f"plan epoch is out of range: {epoch}")
        path = self.root / self.mode / f"epoch-{epoch:03d}.npz"
        audit = self._audit_by_epoch[epoch]
        if self.verify_payloads and sha256_file(path) != audit["file_sha256"]:
            raise DatasetError(f"epoch plan file hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as payload:
            indices = payload["canonical_indices"]
            views = payload["view_ids"]
            occurrences = payload["occurrences"]
            offsets = payload["batch_offsets"]
            if any(a.ndim != 1 or a.dtype.kind not in "iu" for a in (indices, views, occurrences, offsets)):
                raise DatasetError("epoch plan array shape/dtype mismatch")
            if (views.shape != indices.shape or occurrences.shape != indices.shape
                    or len(offsets) != self.batch_count(epoch_index) + 1
                    or offsets[0] != 0 or offsets[-1] != len(indices)
                    or np.any(offsets[1:] < offsets[:-1])
                    or np.any(indices < 0) or np.any(occurrences < 0)
                    or np.any(views < 0) or np.any(views >= len(VIEW_NAMES))):
                raise DatasetError("epoch plan index bounds mismatch")
            result = []
            for begin, end in zip(offsets[:-1], offsets[1:]):
                result.append(
                    tuple(
                        VirtualSegmentRef(int(indices[i]), int(views[i]), int(occurrences[i]))
                        for i in range(int(begin), int(end))
                    )
                )
        if len(result) != self.batch_count(epoch_index):
            raise DatasetError("epoch plan batch count drifted")
        return tuple(result)

    def batch_count(self, epoch_index: int) -> int:
        row = self.audit(epoch_index)
        count = int(row.get("batch_count", self.manifest.get("fixed_batch_count", 0)))
        if count <= 0:
            raise DatasetError(f"plan epoch {epoch_index + 1} has no batches")
        return count

    def audit(self, epoch_index: int) -> Mapping[str, Any]:
        try:
            return self._audit_by_epoch[int(epoch_index) + 1]
        except KeyError as error:
            raise DatasetError(f"missing plan epoch {int(epoch_index) + 1}") from error
