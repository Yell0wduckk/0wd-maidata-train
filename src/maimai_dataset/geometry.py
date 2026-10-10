from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as torch_functional

from maimai_dataset.manifest import DatasetError, load_jsonl
from maimai_dataset.timing import TimingMap


SCHEMA = "maimai.interpolation_geometry.v1"
BACKEND = "indexed-v1"
from .window import (CHART_TICKS_PER_BEAT, TARGET_TICKS, AUDIO_BEATS_BEFORE,
    AUDIO_CONTEXT_BEATS, GRID_RATES, GRID_LENGTHS, GRID_OFFSETS)
CANDIDATE_TICKS = TARGET_TICKS + 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _segment_id_sha256(segment_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for segment_id in segment_ids:
        digest.update(str(segment_id).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _pool_scalar(
    values: np.ndarray, padding: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Match the pyramid's mask-aware k=3, stride=2, padding=1 pooling."""

    values = np.asarray(values, dtype=np.float32)
    padding = np.asarray(padding, dtype=np.bool_)
    # Use the exact Torch operation used by the trainable pyramid.  A NumPy
    # mean differs by one ULP for some overlapping boundary windows, which is
    # enough to choose the other duplicate coordinate in searchsorted.
    tensor = torch.from_numpy(values).unsqueeze(0)
    mask = torch.from_numpy(padding).unsqueeze(0)
    valid = (~mask).to(torch.float32).unsqueeze(1)
    numerator = torch_functional.avg_pool1d(
        tensor.unsqueeze(1) * valid,
        kernel_size=3,
        stride=2,
        padding=1,
        count_include_pad=False,
    )
    count = torch_functional.avg_pool1d(
        valid,
        kernel_size=3,
        stride=2,
        padding=1,
        count_include_pad=False,
    )
    pooled = (
        numerator / count.clamp_min(torch.finfo(torch.float32).eps)
    ).squeeze(0).squeeze(0)
    pooled_padding = count.squeeze(0).squeeze(0).eq(0)
    pooled = pooled.masked_fill(pooled_padding, 0.0)
    return pooled.numpy().astype(np.float32, copy=False), pooled_padding.numpy()


def _mapping(
    source_beats: np.ndarray,
    source_padding: np.ndarray,
    target_beats: np.ndarray,
    target_padding: np.ndarray,
) -> dict[str, np.ndarray]:
    source_beats = np.asarray(source_beats, dtype=np.float32)
    source_padding = np.asarray(source_padding, dtype=np.bool_)
    target_beats = np.asarray(target_beats, dtype=np.float32)
    target_padding = np.asarray(target_padding, dtype=np.bool_)
    left = np.zeros(target_beats.shape, dtype=np.int32)
    right = np.zeros(target_beats.shape, dtype=np.int32)
    weight = np.zeros(target_beats.shape, dtype=np.float32)
    valid_source_indices = np.flatnonzero(~source_padding).astype(np.int32)
    valid_target_indices = np.flatnonzero(~target_padding).astype(np.int32)
    if valid_source_indices.size == 0 or valid_target_indices.size == 0:
        return {
            "left": left,
            "right": right,
            "weight": weight,
            "padding": np.ones(target_beats.shape, dtype=np.bool_),
        }
    source = source_beats[valid_source_indices]
    if source.size == 1:
        left[valid_target_indices] = valid_source_indices[0]
        right[valid_target_indices] = valid_source_indices[0]
        return {
            "left": left,
            "right": right,
            "weight": weight,
            "padding": target_padding.copy(),
        }
    targets = target_beats[valid_target_indices]
    compact_right = np.searchsorted(source, targets, side="left")
    compact_right = np.clip(compact_right, 1, source.size - 1)
    compact_left = compact_right - 1
    denominator = np.maximum(
        source[compact_right] - source[compact_left], np.float32(1e-8)
    ).astype(np.float32)
    interpolation_weight = np.clip(
        (targets - source[compact_left]) / denominator, 0.0, 1.0
    ).astype(np.float32)
    left[valid_target_indices] = valid_source_indices[compact_left]
    right[valid_target_indices] = valid_source_indices[compact_right]
    weight[valid_target_indices] = interpolation_weight
    return {
        "left": left,
        "right": right,
        "weight": weight,
        "padding": target_padding.copy(),
    }


def frame_beat_bpm(
    timing: TimingMap, frame_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    seconds = frame_indices.astype(np.float64) * 0.01
    point_seconds = np.asarray([point.second for point in timing.points], dtype=np.float64)
    point_beats = np.asarray([float(point.beat) for point in timing.points], dtype=np.float64)
    point_bpms = np.asarray([point.bpm for point in timing.points], dtype=np.float64)
    indices = np.searchsorted(point_seconds, seconds, side="right") - 1
    indices = np.clip(indices, 0, len(point_seconds) - 1)
    beats = point_beats[indices] + (seconds - point_seconds[indices]) * point_bpms[indices] / 60.0
    return beats.astype(np.float32), point_bpms[indices].astype(np.float32)


def build_segment_geometry(
    *,
    frame_beats: np.ndarray,
    frame_valid: np.ndarray,
    target_start_tick: int,
    target_end_tick: int,
) -> dict[str, np.ndarray]:
    """Build immutable interpolation geometry for one 16-beat model window."""

    frame_beats = np.asarray(frame_beats, dtype=np.float32)
    frame_valid = np.asarray(frame_valid, dtype=np.bool_)
    if frame_beats.ndim != 1 or frame_valid.shape != frame_beats.shape:
        raise DatasetError("interpolation geometry frame arrays must be aligned vectors")
    if int(target_end_tick) - int(target_start_tick) != TARGET_TICKS:
        raise DatasetError("interpolation geometry requires an exact 16-beat target window")
    pyramid_beats = [frame_beats]
    pyramid_padding = [~frame_valid]
    for _ in range(3):
        beats, padding = _pool_scalar(pyramid_beats[-1], pyramid_padding[-1])
        pyramid_beats.append(beats)
        pyramid_padding.append(padding)

    audio_start = np.float32(target_start_tick / CHART_TICKS_PER_BEAT - AUDIO_BEATS_BEFORE)
    grid_maps: list[dict[str, np.ndarray]] = []
    for beats, padding, rate in zip(
        pyramid_beats, pyramid_padding, GRID_RATES, strict=True
    ):
        target = audio_start + (
            np.arange(AUDIO_CONTEXT_BEATS * rate, dtype=np.float32) + np.float32(0.5)
        ) / np.float32(rate)
        valid_source = beats[~padding]
        target_padding = np.ones(target.shape, dtype=np.bool_)
        if valid_source.size:
            target_padding = (target < valid_source.min()) | (target > valid_source.max())
        grid_maps.append(_mapping(beats, padding, target, target_padding))

    a1_to_a0 = _mapping(
        pyramid_beats[1],
        pyramid_padding[1],
        pyramid_beats[0],
        pyramid_padding[0],
    )
    candidate_ticks = np.arange(
        int(target_start_tick), int(target_end_tick) + 1, dtype=np.int32
    )
    candidate_beats = candidate_ticks.astype(np.float32) / np.float32(
        CHART_TICKS_PER_BEAT
    )
    candidate_padding = np.zeros(candidate_beats.shape, dtype=np.bool_)
    if not frame_valid.any():
        candidate_padding.fill(True)
    candidate = _mapping(
        pyramid_beats[0], pyramid_padding[0], candidate_beats, candidate_padding
    )
    return {
        "grid_left": np.concatenate([value["left"] for value in grid_maps]),
        "grid_right": np.concatenate([value["right"] for value in grid_maps]),
        "grid_weight": np.concatenate([value["weight"] for value in grid_maps]),
        "grid_padding": np.concatenate([value["padding"] for value in grid_maps]),
        "a1_to_a0_left": a1_to_a0["left"],
        "a1_to_a0_right": a1_to_a0["right"],
        "a1_to_a0_weight": a1_to_a0["weight"],
        "a1_to_a0_padding": a1_to_a0["padding"],
        "candidate_left": candidate["left"],
        "candidate_right": candidate["right"],
        "candidate_weight": candidate["weight"],
        "candidate_padding": candidate["padding"],
        "candidate_start_tick": np.asarray(int(target_start_tick), dtype=np.int32),
    }


@dataclass(frozen=True)
class InterpolationGeometryStore:
    root: Path
    manifest: Mapping[str, Any]
    segment_indices: Mapping[str, int]
    arrays: Mapping[str, np.ndarray]

    @classmethod
    def load(
        cls,
        root: Path,
        *,
        profile_fingerprint: str,
        suite_sha256: str | None,
    ) -> "InterpolationGeometryStore":
        root = root.expanduser().resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise DatasetError(f"interpolation geometry manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if manifest.get("schema") != SCHEMA or manifest.get("backend") != BACKEND:
            raise DatasetError("interpolation geometry schema/backend is incompatible")
        if manifest.get("profile_fingerprint") != profile_fingerprint:
            raise DatasetError("interpolation geometry profile fingerprint mismatch")
        # ``None`` is the explicit full-profile selection marker.  A concrete
        # value remains mandatory for a small-sample sidecar, so a geometry
        # directory can never be silently reused for a different selection.
        if manifest.get("suite_sha256") != suite_sha256:
            raise DatasetError("interpolation geometry suite SHA mismatch")
        segment_ids = manifest.get("segment_ids")
        if not isinstance(segment_ids, list) or len(segment_ids) != int(
            manifest.get("segment_count", -1)
        ):
            raise DatasetError("interpolation geometry segment index is invalid")
        if len({str(value) for value in segment_ids}) != len(segment_ids):
            raise DatasetError("interpolation geometry contains duplicate segment IDs")
        expected_segment_id_sha = manifest.get("segment_id_sha256")
        if expected_segment_id_sha is not None and expected_segment_id_sha != _segment_id_sha256(
            [str(value) for value in segment_ids]
        ):
            raise DatasetError("interpolation geometry segment ID hash mismatch")
        selection = manifest.get("selection")
        if isinstance(selection, Mapping) and selection.get("mode") == "profile-full":
            if int(selection.get("profile_segment_count", -1)) != len(segment_ids):
                raise DatasetError(
                    "full interpolation geometry selection count is inconsistent"
                )
        arrays: dict[str, np.ndarray] = {}
        for name, metadata in manifest.get("arrays", {}).items():
            path = root / str(metadata["path"])
            if not path.is_file() or _sha256(path) != metadata["sha256"]:
                raise DatasetError(f"interpolation geometry array hash mismatch: {name}")
            arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(arrays[name].shape) != list(metadata.get("shape", [])):
                raise DatasetError(
                    f"interpolation geometry array shape mismatch: {name}"
                )
            if str(arrays[name].dtype) != str(metadata.get("dtype")):
                raise DatasetError(
                    f"interpolation geometry array dtype mismatch: {name}"
                )
        required = {
            "grid_left",
            "grid_right",
            "grid_weight",
            "grid_padding",
            "a1_to_a0_left",
            "a1_to_a0_right",
            "a1_to_a0_weight",
            "a1_to_a0_padding",
            "a1_to_a0_offsets",
            "candidate_left",
            "candidate_right",
            "candidate_weight",
            "candidate_padding",
            "candidate_start_tick",
        }
        if set(arrays) != required:
            raise DatasetError("interpolation geometry array catalog is incomplete")
        segment_count = len(segment_ids)
        for name in (
            "grid_left",
            "grid_right",
            "grid_weight",
            "grid_padding",
        ):
            if arrays[name].shape != (segment_count, GRID_OFFSETS[-1]):
                raise DatasetError(f"interpolation geometry shape is invalid: {name}")
        for name in (
            "candidate_left",
            "candidate_right",
            "candidate_weight",
            "candidate_padding",
        ):
            if arrays[name].shape != (segment_count, CANDIDATE_TICKS):
                raise DatasetError(f"interpolation geometry shape is invalid: {name}")
        offsets = arrays["a1_to_a0_offsets"]
        if offsets.shape != (segment_count + 1,) or int(offsets[0]) != 0:
            raise DatasetError("interpolation geometry flat offsets are invalid")
        if np.any(offsets[1:] < offsets[:-1]):
            raise DatasetError("interpolation geometry flat offsets are not monotonic")
        flat_length = int(offsets[-1])
        for name in (
            "a1_to_a0_left",
            "a1_to_a0_right",
            "a1_to_a0_weight",
            "a1_to_a0_padding",
        ):
            if arrays[name].shape != (flat_length,):
                raise DatasetError(f"interpolation geometry flat shape is invalid: {name}")
        return cls(
            root=root,
            manifest=manifest,
            segment_indices={str(value): index for index, value in enumerate(segment_ids)},
            arrays=arrays,
        )

    def record(self, segment_id: str) -> dict[str, np.ndarray]:
        try:
            index = self.segment_indices[str(segment_id)]
        except KeyError as error:
            raise DatasetError(
                f"interpolation geometry does not contain segment {segment_id!r}"
            ) from error
        begin = int(self.arrays["a1_to_a0_offsets"][index])
        end = int(self.arrays["a1_to_a0_offsets"][index + 1])
        result = {
            name: np.asarray(value[index])
            for name, value in self.arrays.items()
            if name not in {
                "a1_to_a0_left",
                "a1_to_a0_right",
                "a1_to_a0_weight",
                "a1_to_a0_padding",
                "a1_to_a0_offsets",
            }
        }
        for name in (
            "a1_to_a0_left",
            "a1_to_a0_right",
            "a1_to_a0_weight",
            "a1_to_a0_padding",
        ):
            result[name] = np.asarray(self.arrays[name][begin:end])
        return result


def _write_array(root: Path, name: str, value: np.ndarray) -> dict[str, Any]:
    path = root / f"{name}.npy"
    np.save(path, value, allow_pickle=False)
    return {
        "path": path.name,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def build_geometry_sidecar(
    *,
    profile_root: Path,
    chart_cache_root: Path | None = None,
    feature_root: Path | None = None,
    output: Path,
) -> dict[str, Any]:
    profile_root = profile_root.expanduser().resolve()
    chart_cache_root = (
        None if chart_cache_root is None else chart_cache_root.expanduser().resolve()
    )
    feature_root = None if feature_root is None else feature_root.expanduser().resolve()
    output = output.expanduser().resolve()
    profile_summary = json.loads(
        (profile_root / "summary.json").read_text(encoding="utf-8-sig")
    )
    profile_segment_count = int(
        profile_summary.get("segment_count")
        or sum(
            int(value)
            for value in profile_summary.get("segment_split_counts", {}).values()
        )
    )
    profile = json.loads(
        (profile_root / "profile.json").read_text(encoding="utf-8-sig")
    )
    base_root = (
        feature_root
        if feature_root is not None
        else (profile_root / str(profile["base_dataset"])).resolve()
    )
    selection_sha256: str | None = None
    if chart_cache_root is None:
        # Full-profile geometry is deliberately selected from the profile's
        # own immutable segment index.
        segments = list(load_jsonl(profile_root / "segments.jsonl"))
        selection = {
            "mode": "profile-full",
            "schema": "maimai.interpolation_geometry_selection.profile_full.v1",
            "profile_segment_count": profile_segment_count,
        }
    else:
        cache_manifest_path = chart_cache_root / "manifest.json"
        cache_manifest = json.loads(
            cache_manifest_path.read_text(encoding="utf-8-sig")
        )
        if cache_manifest.get("schema") != "maimai.baseline_chart_vector_cache.v1":
            raise DatasetError("unsupported chart vector cache for geometry selection")
        if cache_manifest.get("profile_fingerprint") != profile_summary.get(
            "profile_fingerprint"
        ):
            raise DatasetError("chart cache/profile mismatch for geometry selection")
        selection_sha256 = str(
            cache_manifest.get("selection", {}).get("selection_sha256", "")
        )
        if not selection_sha256:
            raise DatasetError("chart cache selection fingerprint is missing")
        cache_rows = load_jsonl(chart_cache_root / "segments.jsonl")
        selected_ids = {str(row["segment_id"]) for row in cache_rows}
        segments = [
            row
            for row in load_jsonl(profile_root / "segments.jsonl")
            if str(row["segment_id"]) in selected_ids
        ]
        if len(segments) != len(selected_ids):
            raise DatasetError("chart cache geometry selection is missing profile segments")
        selection = {
            "mode": "chart-cache",
            "schema": "maimai.interpolation_geometry_selection.chart_cache.v1",
            "chart_cache_manifest_sha256": _sha256(cache_manifest_path),
            "selection_sha256": selection_sha256,
            "segment_count": len(selected_ids),
        }
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise DatasetError(f"interpolation geometry output is not empty: {output}")
        output.rmdir()
    staging = output.with_name(output.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=False)
    feature_lengths: dict[str, int] = {}
    frame_counts: list[int] = []
    for segment in segments:
        feature_path = str(segment["feature_path"])
        if feature_path not in feature_lengths:
            feature_lengths[feature_path] = int(
                np.load(base_root / feature_path, mmap_mode="r").shape[1]
            )
        frame_counts.append(
            max(0, int(segment["audio_frame_end"]) - int(segment["audio_frame_start"]))
        )
    if not segments:
        raise DatasetError("interpolation geometry selection contains no segments")
    if chart_cache_root is None and profile_segment_count != len(segments):
        raise DatasetError(
            "full interpolation geometry profile segment count is inconsistent with segments.jsonl"
        )
    segment_ids = [str(segment["segment_id"]) for segment in segments]
    if len(set(segment_ids)) != len(segment_ids):
        raise DatasetError("interpolation geometry selection contains duplicate segment IDs")
    segment_count = len(segments)
    fixed_arrays: dict[str, np.ndarray] = {
        "grid_left": np.empty((segment_count, GRID_OFFSETS[-1]), dtype=np.int32),
        "grid_right": np.empty((segment_count, GRID_OFFSETS[-1]), dtype=np.int32),
        "grid_weight": np.empty((segment_count, GRID_OFFSETS[-1]), dtype=np.float32),
        "grid_padding": np.empty((segment_count, GRID_OFFSETS[-1]), dtype=np.bool_),
        "candidate_left": np.empty((segment_count, CANDIDATE_TICKS), dtype=np.int32),
        "candidate_right": np.empty((segment_count, CANDIDATE_TICKS), dtype=np.int32),
        "candidate_weight": np.empty((segment_count, CANDIDATE_TICKS), dtype=np.float32),
        "candidate_padding": np.empty((segment_count, CANDIDATE_TICKS), dtype=np.bool_),
        "candidate_start_tick": np.empty((segment_count,), dtype=np.int32),
    }
    offsets = np.zeros((segment_count + 1,), dtype=np.int64)
    offsets[1:] = np.cumsum(frame_counts, dtype=np.int64)
    flat_length = int(offsets[-1])
    flat_arrays: dict[str, np.ndarray] = {
        "a1_to_a0_left": np.empty((flat_length,), dtype=np.int32),
        "a1_to_a0_right": np.empty((flat_length,), dtype=np.int32),
        "a1_to_a0_weight": np.empty((flat_length,), dtype=np.float32),
        "a1_to_a0_padding": np.empty((flat_length,), dtype=np.bool_),
    }
    try:
        for index, segment in enumerate(segments):
            feature_path = str(segment["feature_path"])
            frame_start = int(segment["audio_frame_start"])
            frame_end = int(segment["audio_frame_end"])
            frame_indices = np.arange(frame_start, frame_end, dtype=np.int64)
            valid = (frame_indices >= 0) & (
                frame_indices < feature_lengths[feature_path]
            )
            beats, _ = frame_beat_bpm(
                TimingMap.from_dict(segment["timing"]), frame_indices
            )
            record = build_segment_geometry(
                frame_beats=beats,
                frame_valid=valid,
                target_start_tick=int(segment["target_start_tick"]),
                target_end_tick=int(segment["target_end_tick"]),
            )
            for name in fixed_arrays:
                if name == "candidate_start_tick":
                    fixed_arrays[name][index] = record[name]
                else:
                    fixed_arrays[name][index] = record[name]
            begin = int(offsets[index])
            end = int(offsets[index + 1])
            for name in flat_arrays:
                value = np.asarray(record[name])
                if value.shape != (end - begin,):
                    raise DatasetError(
                        f"interpolation geometry A1 mapping length changed at {segment_ids[index]}"
                    )
                flat_arrays[name][begin:end] = value
            if (index + 1) % 10000 == 0 or index + 1 == segment_count:
                print(
                    f"geometry {index + 1}/{segment_count}",
                    file=sys.stderr,
                    flush=True,
                )
        arrays = {**fixed_arrays, "a1_to_a0_offsets": offsets, **flat_arrays}
        array_metadata = {
            name: _write_array(staging, name, value) for name, value in arrays.items()
        }
        manifest = {
            "schema": SCHEMA,
            "backend": BACKEND,
            "profile_fingerprint": profile_summary["profile_fingerprint"],
            "suite_sha256": (
                selection_sha256 if chart_cache_root is not None else None
            ),
            "suite_file_sha256": (
                _sha256(chart_cache_root / "manifest.json")
                if chart_cache_root is not None
                else None
            ),
            "feature_root": str(base_root),
            "selection": selection,
            "segment_count": segment_count,
            "segment_ids": segment_ids,
            "segment_id_sha256": _segment_id_sha256(segment_ids),
            "grid_rates": list(GRID_RATES),
            "grid_lengths": list(GRID_LENGTHS),
            "grid_offsets": list(GRID_OFFSETS),
            "candidate_tick_count": CANDIDATE_TICKS,
            "arrays": array_metadata,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {
        "schema": SCHEMA,
        "backend": BACKEND,
        "output": str(output),
        "segment_count": len(segments),
        "profile_fingerprint": profile_summary["profile_fingerprint"],
        "suite_sha256": (
            selection_sha256 if chart_cache_root is not None else None
        ),
        "segment_id_sha256": _segment_id_sha256(segment_ids),
        "manifest_sha256": _sha256(output / "manifest.json"),
        "bytes": sum(path.stat().st_size for path in output.iterdir() if path.is_file()),
    }


__all__ = [
    "BACKEND",
    "CANDIDATE_TICKS",
    "GRID_LENGTHS",
    "GRID_OFFSETS",
    "GRID_RATES",
    "InterpolationGeometryStore",
    "SCHEMA",
    "build_geometry_sidecar",
    "build_segment_geometry",
    "frame_beat_bpm",
]
