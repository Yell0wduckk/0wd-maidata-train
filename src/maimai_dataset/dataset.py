from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from maimai_dataset.chart_cache import (
    ChartVectorStore,
    audio_reference,
    materialize_chart_vectors,
)
from maimai_dataset.geometry import InterpolationGeometryStore
from maimai_dataset.manifest import DatasetError, canonical_json_sha256, load_jsonl
from maimai_dataset.timing import TimingMap

from maimai_dataset.vocabulary import CompactVocabulary


AUDIO_WINDOW_GROUP_SCHEMA = "maimai.audio_window_group.v1"
AUDIO_WINDOW_REUSE_SCHEMA = "maimai.audio_window_reuse.grouped_v1"


def frame_beat_bpm(timing: TimingMap, frame_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seconds = frame_indices.astype(np.float64) * 0.01
    point_seconds = np.asarray([point.second for point in timing.points], dtype=np.float64)
    point_beats = np.asarray([float(point.beat) for point in timing.points], dtype=np.float64)
    point_bpms = np.asarray([point.bpm for point in timing.points], dtype=np.float64)
    indices = np.searchsorted(point_seconds, seconds, side="right") - 1
    indices = np.clip(indices, 0, len(point_seconds) - 1)
    beats = point_beats[indices] + (seconds - point_seconds[indices]) * point_bpms[indices] / 60.0
    bpms = point_bpms[indices]
    return beats.astype(np.float32), bpms.astype(np.float32)


class CompactSegmentDataset(Dataset):
    def __init__(
        self,
        root: Path,
        vocabulary: CompactVocabulary,
        *,
        split: str | None,
        song_ids: Iterable[str] | None = None,
        source_difficulty_ids: Iterable[int] | None = None,
        record_ids: Iterable[str] | None = None,
        geometry_root: Path | None = None,
        geometry_suite_sha256: str | None = None,
        chart_cache_root: Path | None = None,
        chart_cache_store: ChartVectorStore | None = None,
        defer_audio: bool = False,
    ) -> None:
        self.root = root.resolve()
        self.vocabulary = vocabulary
        self.defer_audio = bool(defer_audio)
        profile = json.loads((self.root / "profile.json").read_text(encoding="utf-8"))
        summary = json.loads((self.root / "summary.json").read_text(encoding="utf-8-sig"))
        self.base_root = (self.root / str(profile["base_dataset"])).resolve()
        vocabulary_path = self.root / "vocabulary.json"
        vocabulary_sha256 = hashlib.sha256(vocabulary_path.read_bytes()).hexdigest()
        if chart_cache_root is not None and chart_cache_store is not None:
            raise DatasetError("provide either chart_cache_root or chart_cache_store, not both")
        self.chart_store = chart_cache_store
        if self.chart_store is None and chart_cache_root is not None:
            self.chart_store = ChartVectorStore.load(
                chart_cache_root,
                profile_fingerprint=str(summary["profile_fingerprint"]),
                vocabulary_sha256=vocabulary_sha256,
            )
        if geometry_root is None and geometry_suite_sha256 is not None:
            raise DatasetError(
                "indexed interpolation geometry SHA cannot be provided without its root"
            )
        self.geometry_store = (
            None
            if geometry_root is None
            else InterpolationGeometryStore.load(
                geometry_root,
                profile_fingerprint=str(summary["profile_fingerprint"]),
                suite_sha256=str(geometry_suite_sha256),
            )
        )
        self._validated_geometry_segments: set[str] = set()
        selected = None if song_ids is None else {str(value) for value in song_ids}
        selected_difficulties = (
            None
            if source_difficulty_ids is None
            else {int(value) for value in source_difficulty_ids}
        )
        selected_records = None if record_ids is None else set(map(str, record_ids))
        self.segments = [
            row
            for row in load_jsonl(self.root / "segments.jsonl")
            if (split is None or row.get("split") == split)
            and (selected is None or str(row.get("song_id")) in selected)
            and (
                selected_difficulties is None
                or int(row.get("source_difficulty_id", row["difficulty_id"]))
                in selected_difficulties
            )
        ]
        if selected_records is not None:
            self.segments = [r for r in self.segments if str(r.get('record_id') or f"{r['song_id']}:{r.get('source_difficulty_id', r['difficulty_id'])}") in selected_records]
        self.target_lengths = [int(row["target_rows"]) for row in self.segments]
        self.sequence_lengths = (
            None
            if self.chart_store is None
            else [
                int(
                    self.chart_store.segment_records[str(row["segment_id"])]["length"]
                )
                for row in self.segments
            ]
        )
        self.audio_frame_lengths = [
            max(
                1,
                int(row["audio_frame_end"]) - int(row["audio_frame_start"]),
            )
            for row in self.segments
        ]
        if self.chart_store is not None:
            missing = [
                str(row["segment_id"])
                for row in self.segments
                if str(row["segment_id"]) not in self.chart_store.segment_indices
            ]
            if missing:
                raise DatasetError(
                    f"chart vector cache does not cover selected dataset segments: {missing[:8]}"
                )
        groups: dict[str, list[int]] = {}
        self.audio_group_keys: list[str] = []
        self.audio_group_material: dict[str, dict[str, Any]] = {}
        for index, segment in enumerate(self.segments):
            audio_key, material = audio_reference(segment)
            group_material = {
                "schema": AUDIO_WINDOW_GROUP_SCHEMA,
                "audio_key": audio_key,
                "target_start_tick": int(segment["target_start_tick"]),
            }
            group_key = canonical_json_sha256(group_material)
            segment["audio_key"] = audio_key
            segment["audio_group_key"] = group_key
            if group_key not in groups:
                groups[group_key] = []
                self.audio_group_keys.append(group_key)
                self.audio_group_material[group_key] = {
                    **group_material,
                    **material,
                    "representative_segment_id": str(segment["segment_id"]),
                }
            groups[group_key].append(index)
        self.audio_groups = {
            key: tuple(indices) for key, indices in groups.items()
        }
        self.audio_group_indices = tuple(
            self.audio_groups[key] for key in self.audio_group_keys
        )
        self._validated_audio_groups: set[str] = set()

    def __len__(self) -> int:
        return len(self.segments)

    def audio_group_statistics(self, *, token_budget: int | None = None) -> dict[str, Any]:
        size_counts: dict[str, int] = {}
        costs: list[int] = []
        for indices in self.audio_group_indices:
            label = str(len(indices))
            size_counts[label] = size_counts.get(label, 0) + 1
            costs.append(sum(max(1, self.target_lengths[index]) for index in indices))
        group_count = len(self.audio_group_indices)
        return {
            "schema": AUDIO_WINDOW_REUSE_SCHEMA,
            "chart_windows": len(self.segments),
            "audio_windows": group_count,
            "audio_reuse_factor": (
                len(self.segments) / group_count if group_count else 0.0
            ),
            "group_size_counts": dict(
                sorted(size_counts.items(), key=lambda item: int(item[0]))
            ),
            "maximum_group_target_rows": max(costs, default=0),
            "singleton_oversize": (
                0
                if token_budget is None
                else sum(cost > int(token_budget) for cost in costs)
            ),
        }

    def _validate_geometry_record(
        self,
        segment: Mapping[str, Any],
        geometry: Mapping[str, np.ndarray],
        *,
        audio_length: int,
    ) -> None:
        segment_id = str(segment["segment_id"])
        if segment_id in self._validated_geometry_segments:
            return
        if int(geometry["candidate_start_tick"]) != int(
            segment["target_start_tick"]
        ):
            raise DatasetError(
                f"{segment_id} indexed candidate lookup has a stale start tick"
            )
        source_lengths = [audio_length]
        for _ in range(3):
            source_lengths.append((source_lengths[-1] + 1) // 2)
        from .window import GRID_OFFSETS as offsets
        for scale, source_length in enumerate(source_lengths):
            begin, end = offsets[scale : scale + 2]
            for name in ("grid_left", "grid_right"):
                values = geometry[name][begin:end]
                if values.size and (
                    int(values.min()) < 0 or int(values.max()) >= source_length
                ):
                    raise DatasetError(
                        f"{segment_id} indexed {name} is out of A{scale} bounds"
                    )
        a1_length = source_lengths[1]
        for name in ("a1_to_a0_left", "a1_to_a0_right"):
            values = geometry[name]
            if values.size and (
                int(values.min()) < 0 or int(values.max()) >= a1_length
            ):
                raise DatasetError(
                    f"{segment_id} indexed {name} is out of A1 bounds"
                )
        for name in ("candidate_left", "candidate_right"):
            values = geometry[name]
            if values.size and (
                int(values.min()) < 0 or int(values.max()) >= audio_length
            ):
                raise DatasetError(
                    f"{segment_id} indexed {name} is out of A0 bounds"
                )
        for name in ("grid_weight", "a1_to_a0_weight", "candidate_weight"):
            values = geometry[name]
            if (
                not np.isfinite(values).all()
                or np.any(values < 0)
                or np.any(values > 1)
            ):
                raise DatasetError(
                    f"{segment_id} indexed {name} contains invalid weights"
                )
        self._validated_geometry_segments.add(segment_id)

    def feature_source(self, segment: Mapping[str, Any]) -> Path:
        return self.base_root / str(segment["feature_path"])

    def audio_item(self, group_key: str) -> dict[str, Any]:
        try:
            indices = self.audio_groups[str(group_key)]
        except KeyError as error:
            raise DatasetError(f"unknown audio group: {group_key}") from error
        segment = self.segments[indices[0]]
        frame_start = int(segment["audio_frame_start"])
        frame_end = int(segment["audio_frame_end"])
        frame_count = max(1, frame_end - frame_start)
        source = np.load(self.feature_source(segment), mmap_mode="r")
        mel = np.zeros((frame_count, 128), dtype=np.float32)
        valid = np.zeros((frame_count,), dtype=np.bool_)
        copy_start = max(0, frame_start)
        copy_end = min(int(source.shape[1]), frame_end)
        if copy_end > copy_start:
            destination_start = copy_start - frame_start
            destination_end = destination_start + copy_end - copy_start
            mel[destination_start:destination_end] = np.asarray(
                source[:, copy_start:copy_end].T, dtype=np.float32
            )
            valid[destination_start:destination_end] = True
        frame_indices = np.arange(frame_start, frame_end, dtype=np.int64)
        frame_beats, frame_bpms = frame_beat_bpm(
            TimingMap.from_dict(segment["timing"]), frame_indices
        )
        result = {
            "audio_group_key": str(group_key),
            "audio": torch.from_numpy(mel),
            "audio_valid": torch.from_numpy(valid),
            "frame_beats": torch.from_numpy(frame_beats),
            "frame_bpms": torch.from_numpy(frame_bpms),
            "audio_start_tick": float(segment["target_start_tick"]),
        }
        if self.geometry_store is not None:
            representative = self.geometry_store.record(str(segment["segment_id"]))
            self._validate_geometry_record(
                segment, representative, audio_length=frame_count
            )
            if group_key not in self._validated_audio_groups:
                for member_index in indices[1:]:
                    member = self.segments[member_index]
                    geometry = self.geometry_store.record(str(member["segment_id"]))
                    self._validate_geometry_record(
                        member, geometry, audio_length=frame_count
                    )
                    for name, expected in representative.items():
                        if name not in geometry or not np.array_equal(
                            expected, geometry[name]
                        ):
                            raise DatasetError(
                                "audio group contains non-identical interpolation geometry: "
                                f"{segment['segment_id']} != {member['segment_id']} field {name}"
                            )
                self._validated_audio_groups.add(str(group_key))
            result["interpolation_geometry"] = {
                name: torch.from_numpy(np.array(value, copy=True))
                for name, value in representative.items()
            }
        return result

    def __getitem__(self, index: int) -> dict[str, Any]:
        segment = self.segments[index]
        chart_vectors = (
            materialize_chart_vectors(
                profile_root=self.root,
                segment=segment,
                vocabulary=self.vocabulary,
            )
            if self.chart_store is None
            else self.chart_store.record(str(segment["segment_id"]))
        )

        result = {
            "segment_id": segment["segment_id"],
            "song_id": str(segment["song_id"]),
            "audio_group_key": str(segment["audio_group_key"]),
            **chart_vectors,
            "difficulty_id": int(
                segment.get("condition_difficulty_id", segment["difficulty_id"])
            ),
            "source_difficulty_id": int(
                segment.get("source_difficulty_id", segment["difficulty_id"])
            ),
            "cabinet_id": 2 if segment["cabinet"] == "DX" else 1,
            "level": float(segment.get("level_value") or 0.0),
            "start_tick": float(segment["target_start_tick"]),
            "end_tick": float(segment["target_end_tick"]),
            "song_end_tick": int(segment["song_end_tick"]),
        }
        if not self.defer_audio:
            result.update(self.audio_item(str(segment["audio_group_key"])))
        return result


def collate_compact_segments(
    items: Sequence[Mapping[str, Any]],
    *,
    audio_loader: Callable[[str], Mapping[str, Any]] | None = None,
    chart_only: bool = False,
) -> dict[str, Any]:
    if not items:
        raise ValueError("cannot collate an empty compact-v2 batch")
    batch = len(items)
    unique_keys: list[str] = []
    key_indices: dict[str, int] = {}
    window_to_audio: list[int] = []
    embedded_audio: dict[str, Mapping[str, Any]] = {}
    for item_index, item in enumerate(items if not chart_only else ()):
        key = str(item.get("audio_group_key", f"__unshared__:{item_index}"))
        if key not in key_indices:
            key_indices[key] = len(unique_keys)
            unique_keys.append(key)
            if audio_loader is None:
                if "audio" not in item:
                    raise DatasetError(
                        "deferred compact segment collation requires an audio loader"
                    )
                embedded_audio[key] = item
        window_to_audio.append(key_indices[key])
    audio_items = [
        (audio_loader(key) if audio_loader is not None else embedded_audio[key])
        for key in unique_keys
    ]
    audio_batch = len(audio_items)
    max_audio = max((item["audio"].shape[0] for item in audio_items), default=0)
    max_rows = max(next(iter(item["input_ids"].values())).shape[0] for item in items)
    audio = torch.zeros((audio_batch, max_audio, 128), dtype=torch.float32)
    audio_padding = torch.ones((audio_batch, max_audio), dtype=torch.bool)
    frame_beats = torch.zeros((audio_batch, max_audio), dtype=torch.float32)
    frame_bpms = torch.full((audio_batch, max_audio), 120.0, dtype=torch.float32)
    row_padding = torch.ones((batch, max_rows), dtype=torch.bool)
    row_anchor = torch.zeros((batch, max_rows), dtype=torch.float32)
    def long_group(source: str, fill: int) -> dict[str, torch.Tensor]:
        return {
            field: torch.full((batch, max_rows), fill, dtype=torch.long)
            for field in items[0][source]
        }

    input_ids = long_group("input_ids", 0)
    target_ids = long_group("target_ids", -100)
    time_targets = long_group("time_targets", -100)
    time_category_targets = long_group("time_category_targets", -100)
    time_lower = long_group("time_lower", 0)
    time_upper = long_group("time_upper", 0)
    input_times = {
        field: torch.zeros((batch, max_rows), dtype=torch.float32)
        for field in items[0]["input_times"]
    }
    input_time_active = {
        field: torch.zeros((batch, max_rows), dtype=torch.bool)
        for field in items[0]["input_time_active"]
    }
    for audio_index, item in enumerate(audio_items):
        audio_length = item["audio"].shape[0]
        audio[audio_index, :audio_length] = item["audio"]
        audio_padding[audio_index, :audio_length] = ~item["audio_valid"]
        frame_beats[audio_index, :audio_length] = item["frame_beats"]
        frame_bpms[audio_index, :audio_length] = item["frame_bpms"]
    for batch_index, item in enumerate(items):
        row_length = next(iter(item["input_ids"].values())).shape[0]
        row_padding[batch_index, :row_length] = False
        row_anchor[batch_index, :row_length] = item["row_anchor_tick"]
        for field in input_ids:
            input_ids[field][batch_index, :row_length] = item["input_ids"][field]
            target_ids[field][batch_index, :row_length] = item["target_ids"][field]
        for field in input_times:
            input_times[field][batch_index, :row_length] = item["input_times"][field]
            input_time_active[field][batch_index, :row_length] = item[
                "input_time_active"
            ][field]
            time_targets[field][batch_index, :row_length] = item["time_targets"][field]
            time_category_targets[field][batch_index, :row_length] = item[
                "time_category_targets"
            ][field]
            time_lower[field][batch_index, :row_length] = item["time_lower"][field]
            time_upper[field][batch_index, :row_length] = item["time_upper"][field]
    delta_active = time_targets["delta_tick"].ne(-100)
    candidate_rows = int(delta_active.sum().item())
    if candidate_rows:
        candidate_widths = (
            time_upper["delta_tick"][delta_active]
            - time_lower["delta_tick"][delta_active]
            + 1
        )
        candidate_active_cells = int(candidate_widths.sum().item())
        candidate_dense_cells = candidate_rows * int(candidate_widths.max().item())
    else:
        candidate_active_cells = 0
        candidate_dense_cells = 0
    result = {
        "segment_ids": [item["segment_id"] for item in items],
        "song_ids": [item["song_id"] for item in items],
        "audio_group_keys": unique_keys,
        "audio": audio,
        "audio_padding_mask": audio_padding,
        "frame_beats": frame_beats,
        "frame_bpms": frame_bpms,
        "audio_start_tick": torch.tensor(
            [
                item.get("audio_start_tick", item.get("start_tick", 0.0))
                for item in audio_items
            ],
            dtype=torch.float32,
        ),
        "window_to_audio": torch.tensor(window_to_audio, dtype=torch.long),
        "row_padding_mask": row_padding,
        "row_anchor_tick": row_anchor,
        "input_ids": input_ids,
        "input_times": input_times,
        "input_time_active": input_time_active,
        "target_ids": target_ids,
        "time_targets": time_targets,
        "time_category_targets": time_category_targets,
        "time_lower": time_lower,
        "time_upper": time_upper,
        "difficulty_id": torch.tensor(
            [item["difficulty_id"] for item in items], dtype=torch.long
        ),
        "source_difficulty_id": torch.tensor(
            [item["source_difficulty_id"] for item in items], dtype=torch.long
        ),
        "cabinet_id": torch.tensor(
            [item["cabinet_id"] for item in items], dtype=torch.long
        ),
        "level": torch.tensor([item["level"] for item in items], dtype=torch.float32),
        "start_tick": torch.tensor(
            [item["start_tick"] for item in items], dtype=torch.float32
        ),
        "end_tick": torch.tensor(
            [item["end_tick"] for item in items], dtype=torch.float32
        ),
        "song_end_tick": torch.tensor(
            [item["song_end_tick"] for item in items], dtype=torch.long
        ),
        "batch_stats": {
            "chart_windows": batch,
            "audio_windows": audio_batch,
            "audio_reuse_factor": batch / max(1, audio_batch),
            "audio_positions": audio_batch * max_audio,
            "audio_padding_positions": int(audio_padding.sum().item()),
            "padded_audio_frames": audio_batch * max_audio,
            "audio_padding_ratio": float(audio_padding.sum().item())
            / max(1, audio_batch * max_audio),
            "row_positions": batch * max_rows,
            "row_padding_positions": int(row_padding.sum().item()),
            "row_padding_ratio": float(row_padding.sum().item())
            / max(1, batch * max_rows),
            "candidate_rows": candidate_rows,
            "candidate_active_cells": candidate_active_cells,
            "candidate_dense_cells": candidate_dense_cells,
            "candidate_padding_ratio": float(
                candidate_dense_cells - candidate_active_cells
            )
            / max(1, candidate_dense_cells),
        },
    }
    if chart_only:
        for name in ('audio_group_keys','audio','audio_padding_mask','frame_beats','frame_bpms',
                     'audio_start_tick','window_to_audio'):
            result.pop(name)
        for name in tuple(result['batch_stats']):
            if 'audio' in name: result['batch_stats'].pop(name)
        return result
    geometry_presence = ["interpolation_geometry" in item for item in audio_items]
    if any(geometry_presence) and not all(geometry_presence):
        raise DatasetError("cannot collate mixed reference and indexed interpolation items")
    if all(geometry_presence):
        fixed_names = (
            "grid_left",
            "grid_right",
            "grid_weight",
            "grid_padding",
            "candidate_left",
            "candidate_right",
            "candidate_weight",
            "candidate_padding",
            "candidate_start_tick",
        )
        geometry_batch = {
            name: (
                torch.stack(
                    [item["interpolation_geometry"][name] for item in audio_items]
                ).to(torch.long)
                if name.endswith(("_left", "_right", "_start_tick"))
                else torch.stack(
                    [item["interpolation_geometry"][name] for item in audio_items]
                )
            )
            for name in fixed_names
        }
        variable_names = (
            "a1_to_a0_left",
            "a1_to_a0_right",
            "a1_to_a0_weight",
            "a1_to_a0_padding",
        )
        for name in variable_names:
            fill = True if name.endswith("padding") else 0
            dtype = (
                torch.long
                if name.endswith(("_left", "_right"))
                else audio_items[0]["interpolation_geometry"][name].dtype
            )
            values = torch.full((audio_batch, max_audio), fill, dtype=dtype)
            for batch_index, item in enumerate(audio_items):
                source = item["interpolation_geometry"][name]
                if source.shape[0] != item["audio"].shape[0]:
                    raise DatasetError(
                        "interpolation geometry/audio length mismatch for "
                        f"audio group {unique_keys[batch_index]}"
                    )
                values[batch_index, : source.shape[0]] = source.to(dtype)
            geometry_batch[name] = values
        result["interpolation_geometry"] = geometry_batch
    return result


__all__ = [
    "AUDIO_WINDOW_GROUP_SCHEMA",
    "AUDIO_WINDOW_REUSE_SCHEMA",
    "CompactSegmentDataset",
    "collate_compact_segments",
    "frame_beat_bpm",
]
