from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np
import torch

from maimai_dataset.encoding import TIME_FIELDS
from maimai_dataset.array_store import ReadOnlyArrayStore
from maimai_dataset.manifest import (
    DatasetError,
    canonical_json_sha256,
    load_jsonl,
    write_jsonl,
)
from maimai_dataset.vocabulary import CompactVocabulary


SCHEMA = "maimai.baseline_chart_vector_cache.v1"
AUDIO_REFERENCE_SCHEMA = "maimai.baseline_audio_reference.v1"
SUPPORT_SUITE_SCHEMA = "maimai.small_sample_suite.v1"
TRAIN_ONLY_SUITE_SCHEMA = "maimai.full_train_selection.v1"


def _sha256(path: Path) -> str:
    from maimai_dataset.integrity import file_sha256
    return file_sha256(path)


@lru_cache(maxsize=128)
def _encoded(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _recenter_first_moment(
    rows: Sequence[Mapping[str, Any]], *, reference_tick: int, first_tick: int | None
) -> list[dict[str, Any]]:
    result = [dict(row) for row in rows]
    if first_tick is None:
        return result
    for row in result:
        if row.get("event") == "MOMENT":
            value = int(first_tick) - int(reference_tick)
            if value < 0:
                raise DatasetError("first moment precedes its segment reference")
            row["delta_tick"] = value
            return result
    raise DatasetError("segment metadata names a first moment, but its row slice is empty")


def _row_anchors(
    rows: Sequence[Mapping[str, Any]],
    *,
    context_tick: int,
    target_tick: int,
) -> list[int]:
    current = int(context_tick)
    anchors: list[int] = []
    for row in rows:
        event = row.get("event")
        if event == "TARGET_START":
            current = int(target_tick)
        elif event == "MOMENT":
            current += int(row["delta_tick"])
        anchors.append(current)
    return anchors


def _segment_streams(
    *,
    profile_root: Path,
    segment: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    encoded = _encoded(str(profile_root / str(segment["encoded_path"])))
    all_rows = encoded["rows"]
    context_start = int(segment["context_row_start"])
    target_start = int(segment["target_row_start"])
    target_end = int(segment["target_row_end"])
    return (
        _recenter_first_moment(
            all_rows[context_start:target_start],
            reference_tick=int(segment["context_start_tick"]),
            first_tick=segment.get("context_first_tick"),
        ),
        _recenter_first_moment(
            all_rows[target_start:target_end],
            reference_tick=int(segment["target_start_tick"]),
            first_tick=segment.get("target_first_tick"),
        ),
    )


def materialize_chart_vectors(
    *,
    profile_root: Path,
    segment: Mapping[str, Any],
    vocabulary: CompactVocabulary,
) -> dict[str, Any]:
    """Materialize only the baseline chart-side tensors for one segment.

    Audio Mel, frame timing and indexed interpolation geometry deliberately do
    not enter this function.  Keeping this as the single implementation used
    by both the dynamic dataset path and the persistent cache prevents the two
    representations from acquiring different row semantics.
    """

    context_rows, target_rows = _segment_streams(
        profile_root=profile_root,
        segment=segment,
    )
    sequence = [
        {"event": "SEGMENT_START"},
        *context_rows,
        {"event": "TARGET_START"},
        *target_rows,
        {"event": "SEGMENT_END"},
    ]
    input_rows = sequence[:-1]
    label_rows = sequence[1:]
    first_learned_label = len(context_rows) + 1

    input_ids = {field: [] for field in vocabulary.field_names}
    input_times = {field: [] for field in TIME_FIELDS}
    input_time_active = {field: [] for field in TIME_FIELDS}
    target_ids = {field: [] for field in vocabulary.field_names}
    time_targets = {field: [] for field in TIME_FIELDS}
    time_category_targets = {field: [] for field in TIME_FIELDS}
    time_lower = {field: [] for field in TIME_FIELDS}
    time_upper = {field: [] for field in TIME_FIELDS}
    for row in input_rows:
        encoded_row, _ = vocabulary.encode_row(row)
        for field in input_ids:
            input_ids[field].append(encoded_row[field])
        for field in TIME_FIELDS:
            present = field in row
            input_times[field].append(float(row.get(field, 0)))
            input_time_active[field].append(present)
    target_reference = int(segment["target_start_tick"])
    target_current = target_reference
    target_moments = 0
    for row_index, row in enumerate(label_rows):
        learn = row_index >= first_learned_label
        encoded_row, row_active = vocabulary.encode_row(row)
        for field in target_ids:
            target_ids[field].append(
                encoded_row[field] if learn and row_active[field] else -100
            )
        for field in TIME_FIELDS:
            time_targets[field].append(-100)
            time_category_targets[field].append(-100)
            time_lower[field].append(0)
            time_upper[field].append(0)
        if not learn:
            continue
        if row.get("event") == "MOMENT":
            field = "delta_tick"
            lower = 0 if target_moments == 0 else 1
            upper = max(lower, int(segment["target_end_tick"]) - target_reference)
            value = int(row[field])
            if not lower <= value <= upper:
                raise DatasetError(
                    f"{segment['segment_id']} delta {value} outside [{lower}, {upper}]"
                )
            time_targets[field][-1] = value
            time_lower[field][-1] = lower
            time_upper[field][-1] = upper
            time_category_targets[field][-1] = vocabulary.time_indices[field].get(
                value, -100
            )
            target_current = target_reference + value
            target_reference = target_current
            target_moments += 1
        if row.get("event") == "NOTE" and row.get("kind") in {
            "hold",
            "touch_hold",
        }:
            field = "hold_duration_tick"
            value = int(row[field])
            lower = 0
            upper = max(lower, int(segment["song_end_tick"]) - target_current)
            if not lower <= value <= upper:
                raise DatasetError(
                    f"{segment['segment_id']} Hold duration {value} outside [{lower}, {upper}]"
                )
            time_targets[field][-1] = value
            time_lower[field][-1] = lower
            time_upper[field][-1] = upper
            time_category_targets[field][-1] = vocabulary.time_indices[field].get(
                value, -100
            )
        if row.get("event") == "SLIDE_BRANCH" or (
            row.get("event") == "NOTE" and row.get("kind") in {"slide", "wifi"}
        ):
            field = "branch_duration_tick"
            value = int(row[field])
            lower = 1
            upper = max(lower, int(segment["song_end_tick"]) - target_current)
            if not lower <= value <= upper:
                raise DatasetError(
                    f"{segment['segment_id']} Slide duration {value} outside [{lower}, {upper}]"
                )
            time_targets[field][-1] = value
            time_lower[field][-1] = lower
            time_upper[field][-1] = upper
            time_category_targets[field][-1] = vocabulary.time_indices[field].get(
                value, -100
            )
    result = {
        "input_ids": {
            field: torch.tensor(values, dtype=torch.long)
            for field, values in input_ids.items()
        },
        "input_times": {
            field: torch.tensor(values, dtype=torch.float32)
            for field, values in input_times.items()
        },
        "input_time_active": {
            field: torch.tensor(values, dtype=torch.bool)
            for field, values in input_time_active.items()
        },
        "target_ids": {
            field: torch.tensor(values, dtype=torch.long)
            for field, values in target_ids.items()
        },
        "time_targets": {
            field: torch.tensor(values, dtype=torch.long)
            for field, values in time_targets.items()
        },
        "time_category_targets": {
            field: torch.tensor(values, dtype=torch.long)
            for field, values in time_category_targets.items()
        },
        "time_lower": {
            field: torch.tensor(values, dtype=torch.long)
            for field, values in time_lower.items()
        },
        "time_upper": {
            field: torch.tensor(values, dtype=torch.long)
            for field, values in time_upper.items()
        },
        "row_anchor_tick": torch.tensor(
            _row_anchors(
                input_rows,
                context_tick=int(segment["context_start_tick"]),
                target_tick=int(segment["target_start_tick"]),
            ),
            dtype=torch.float32,
        ),
    }
    return result


def audio_reference(segment: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    # Only the beat clock used by ``frame_beat_bpm`` belongs to the audio
    # identity.  Chart-specific end_beat/meter metadata must not prevent two
    # difficulties from sharing the same audio window.
    beat_clock = {
        "first_seconds": segment["timing"].get("first_seconds"),
        "bpm_points": segment["timing"].get("bpm_points", []),
    }
    material = {
        "schema": AUDIO_REFERENCE_SCHEMA,
        "song_id": str(segment["song_id"]),
        "feature_path": str(segment["feature_path"]),
        "audio_frame_start": int(segment["audio_frame_start"]),
        "audio_frame_end": int(segment["audio_frame_end"]),
        "beat_clock_sha256": canonical_json_sha256(beat_clock),
        "mel_bins": 128,
        "frame_hz": 100,
    }
    return canonical_json_sha256(material), material


def _array_specs(vocabulary: CompactVocabulary) -> dict[str, np.dtype[Any]]:
    specs: dict[str, np.dtype[Any]] = {}
    for field in vocabulary.field_names:
        specs[f"input_ids.{field}"] = np.dtype(np.int16)
        specs[f"target_ids.{field}"] = np.dtype(np.int16)
    for field in TIME_FIELDS:
        specs[f"input_times.{field}"] = np.dtype(np.float32)
        specs[f"input_time_active.{field}"] = np.dtype(np.bool_)
        specs[f"time_targets.{field}"] = np.dtype(np.int32)
        specs[f"time_category_targets.{field}"] = np.dtype(np.int16)
        specs[f"time_lower.{field}"] = np.dtype(np.int32)
        specs[f"time_upper.{field}"] = np.dtype(np.int32)
    specs["row_anchor_tick"] = np.dtype(np.float32)
    return specs


def _selected_segments(
    profile_root: Path, suite: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    schema = suite.get("schema")
    train = [str(value) for value in suite.get("train_song_ids", [])]
    validation = [str(value) for value in suite.get("validation_song_ids", [])]
    if schema == SUPPORT_SUITE_SCHEMA:
        if not train or not validation:
            raise DatasetError("pilot suite must contain train_song_ids and validation_song_ids")
    elif schema == TRAIN_ONLY_SUITE_SCHEMA:
        if not train:
            raise DatasetError("full train-only selection must contain train_song_ids")
        if "validation_song_ids" not in suite or validation:
            raise DatasetError("full train-only selection must declare an empty validation_song_ids list")
    else:
        raise DatasetError(f"unsupported chart-cache selection schema: {schema!r}")
    if len(train) != len(set(train)) or len(validation) != len(set(validation)):
        raise DatasetError("pilot suite song IDs must be unique within each role")
    if set(train) & set(validation):
        raise DatasetError("pilot suite train and validation song IDs overlap")
    roles = {song_id: "train" for song_id in train}
    roles.update({song_id: "validation" for song_id in validation})
    segments = [
        row
        for row in load_jsonl(profile_root / "segments.jsonl")
        if str(row["song_id"]) in roles
    ]
    present = {str(row["song_id"]) for row in segments}
    missing = sorted(set(roles) - present)
    if missing:
        raise DatasetError(f"pilot suite references songs without current segments: {missing[:8]}")
    return segments, roles


def _tensor_at(vectors: Mapping[str, Any], name: str) -> torch.Tensor:
    if "." not in name:
        value = vectors[name]
    else:
        group, field = name.split(".", 1)
        value = vectors[group][field]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"chart vector {name!r} is not a tensor")
    return value


def build_chart_vector_cache(
    *, profile_root: Path, suite_path: Path, output: Path
) -> dict[str, Any]:
    profile_root = profile_root.expanduser().resolve()
    suite_path = suite_path.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists():
        raise DatasetError(f"chart vector cache output already exists: {output}")
    summary_path = profile_root / "summary.json"
    vocabulary_path = profile_root / "vocabulary.json"
    profile_manifest_path = profile_root / "manifest.jsonl"
    for required in (summary_path, vocabulary_path, profile_manifest_path):
        if not required.is_file():
            raise DatasetError(f"profile input is missing: {required}")
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    vocabulary_payload = json.loads(vocabulary_path.read_text(encoding="utf-8-sig"))
    vocabulary = CompactVocabulary.from_dict(vocabulary_payload)
    suite = json.loads(suite_path.read_text(encoding="utf-8-sig"))
    segments, roles = _selected_segments(profile_root, suite)
    if not segments:
        raise DatasetError("pilot suite selects no segments")

    lengths_list: list[int] = []
    for row in segments:
        context_rows, target_rows = _segment_streams(
            profile_root=profile_root,
            segment=row,
        )
        lengths_list.append(len(context_rows) + len(target_rows) + 2)
    lengths = np.asarray(lengths_list, dtype=np.int64)
    offsets = np.empty((len(segments) + 1,), dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    total_rows = int(offsets[-1])
    specs = _array_specs(vocabulary)
    staging = output.with_name(f".{output.name}.building-{uuid4().hex}")
    arrays: dict[str, np.memmap] = {}
    try:
        staging.mkdir(parents=True)
        arrays_dir = staging / "arrays"
        arrays_dir.mkdir()
        for name, dtype in specs.items():
            arrays[name] = np.lib.format.open_memmap(
                arrays_dir / f"{name}.npy", mode="w+", dtype=dtype, shape=(total_rows,)
            )
        segment_index: list[dict[str, Any]] = []
        audio_references: dict[str, dict[str, Any]] = {}
        for index, segment in enumerate(segments):
            vectors = materialize_chart_vectors(
                profile_root=profile_root, segment=segment, vocabulary=vocabulary
            )
            begin = int(offsets[index])
            end = int(offsets[index + 1])
            for name, array in arrays.items():
                value = _tensor_at(vectors, name).detach().cpu().numpy()
                if value.shape != (end - begin,):
                    raise DatasetError(
                        f"{segment['segment_id']} chart vector length changed for {name}: "
                        f"{value.shape} != {(end - begin,)}"
                    )
                array[begin:end] = value.astype(array.dtype, copy=False)
            audio_key, audio_material = audio_reference(segment)
            audio_references.setdefault(audio_key, {"audio_key": audio_key, **audio_material})
            segment_index.append(
                {
                    "segment_id": str(segment["segment_id"]),
                    "cache_index": index,
                    "offset": begin,
                    "length": end - begin,
                    "song_id": str(segment["song_id"]),
                    "role": roles[str(segment["song_id"])],
                    "audio_key": audio_key,
                    "source_difficulty_id": int(
                        segment.get("source_difficulty_id", segment["difficulty_id"])
                    ),
                }
            )
            if (index + 1) % 5000 == 0 or index + 1 == len(segments):
                print(
                    f"chart vectors {index + 1}/{len(segments)}",
                    file=sys.stderr,
                    flush=True,
                )
        for array in arrays.values():
            array.flush()
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()
        arrays.clear()
        np.save(staging / "offsets.npy", offsets, allow_pickle=False)
        write_jsonl(staging / "segments.jsonl", segment_index)
        write_jsonl(
            staging / "audio-references.jsonl",
            (audio_references[key] for key in sorted(audio_references)),
        )
        array_metadata = {}
        for name, dtype in specs.items():
            path = arrays_dir / f"{name}.npy"
            array_metadata[name] = {
                "path": path.relative_to(staging).as_posix(),
                "dtype": dtype.name,
                "shape": [total_rows],
                "sha256": _sha256(path),
            }
        role_counts = {
            role: sum(row["role"] == role for row in segment_index)
            for role in ("train", "validation")
        }
        train_ids = [str(value) for value in suite["train_song_ids"]]
        validation_ids = [str(value) for value in suite["validation_song_ids"]]
        selection = {
            "schema": (
                "maimai.baseline_chart_selection.full_train.v1"
                if suite["schema"] == TRAIN_ONLY_SUITE_SCHEMA
                else "maimai.baseline_chart_selection.pilot_256.v1"
            ),
            "profile_fingerprint": summary["profile_fingerprint"],
            "train_song_ids": train_ids,
            "validation_song_ids": validation_ids,
        }
        if suite["schema"] == TRAIN_ONLY_SUITE_SCHEMA:
            selection.update(selection_mode="train-only", validation_enabled=False)
        selection["selection_sha256"] = canonical_json_sha256(selection)
        manifest = {
            "schema": SCHEMA,
            "profile_schema": summary.get("schema"),
            "profile_fingerprint": summary["profile_fingerprint"],
            "row_codec": summary.get("row_codec"),
            "profile_manifest_sha256": _sha256(profile_manifest_path),
            "vocabulary_sha256": _sha256(vocabulary_path),
            "source_suite": {
                "path": str(suite_path),
                "file_sha256": _sha256(suite_path),
                "schema": suite["schema"],
                "suite_sha256": suite.get("suite_sha256"),
                "declared_profile_fingerprint": suite.get("profile_fingerprint"),
                "used_as_song_selection_only": True,
            },
            "selection": selection,
            "segment_count": len(segments),
            "segment_role_counts": role_counts,
            "song_counts": {
                "train": len(train_ids),
                "validation": len(validation_ids),
            },
            "row_count": total_rows,
            "audio_reference_count": len(audio_references),
            "contains_audio": False,
            "contains_aio": False,
            "contains_interpolation_geometry": False,
            "audio_storage": "external-baseline-mel-and-indexed-geometry",
            "offsets": {
                "path": "offsets.npy",
                "dtype": "int64",
                "shape": [len(offsets)],
                "sha256": _sha256(staging / "offsets.npy"),
            },
            "segment_index": {
                "path": "segments.jsonl",
                "sha256": _sha256(staging / "segments.jsonl"),
            },
            "audio_references": {
                "path": "audio-references.jsonl",
                "sha256": _sha256(staging / "audio-references.jsonl"),
            },
            "arrays": array_metadata,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(output)
    except Exception:
        for array in arrays.values():
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()
        arrays.clear()
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {
        "schema": SCHEMA,
        "output": str(output),
        "profile_fingerprint": summary["profile_fingerprint"],
        "selection_sha256": selection["selection_sha256"],
        "segment_count": len(segments),
        "segment_role_counts": role_counts,
        "row_count": total_rows,
        "audio_reference_count": len(audio_references),
        "contains_audio": False,
        "contains_aio": False,
        "manifest_sha256": _sha256(output / "manifest.json"),
        "bytes": sum(path.stat().st_size for path in output.rglob("*") if path.is_file()),
    }


def validate_vector_index(offsets, indices, arrays, segment_count):
    """Check small index tables and array headers, without touching row payloads."""
    if offsets.dtype != np.int64 or offsets.shape != (segment_count + 1,):
        raise DatasetError("vector cache offsets shape/dtype mismatch")
    if (int(offsets[0]) != 0 or np.any(offsets[1:] < offsets[:-1])
            or set(indices.values()) != set(range(segment_count))):
        raise DatasetError("vector cache index bounds mismatch")
    if not arrays or any(a.ndim != 1 or a.shape[0] != int(offsets[-1]) for a in arrays.values()):
        raise DatasetError("vector cache payload length mismatch")


class ChartVectorStore(ReadOnlyArrayStore):
    def __init__(
        self,
        *,
        root: Path,
        manifest: Mapping[str, Any],
        segment_indices: Mapping[str, int],
        segment_records: Mapping[str, Mapping[str, Any]],
        offsets: np.ndarray,
        arrays: Mapping[str, np.ndarray],
    ) -> None:
        self.root = root
        self.manifest = manifest
        self.segment_indices = segment_indices
        self.segment_records = segment_records
        self.offsets = offsets
        self.arrays = arrays

    @classmethod
    def load(
        cls,
        root: Path,
        *,
        profile_fingerprint: str,
        vocabulary_sha256: str,
        verify_payloads: bool = True,
    ) -> "ChartVectorStore":
        root = root.expanduser().resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise DatasetError(f"chart vector cache manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if manifest.get("schema") != SCHEMA:
            raise DatasetError(f"unsupported chart vector cache schema: {manifest.get('schema')!r}")
        if any(
            bool(manifest.get(name))
            for name in ("contains_audio", "contains_aio", "contains_interpolation_geometry")
        ):
            raise DatasetError("baseline chart vector cache must not embed audio, AIO or geometry")
        offsets_meta = manifest["offsets"]
        offsets_path = root / str(offsets_meta["path"])
        if verify_payloads and _sha256(offsets_path) != offsets_meta["sha256"]:
            raise DatasetError("chart vector cache offsets SHA-256 mismatch")
        offsets = np.load(offsets_path, mmap_mode="r", allow_pickle=False)
        segment_meta = manifest["segment_index"]
        segment_path = root / str(segment_meta["path"])
        if verify_payloads and _sha256(segment_path) != segment_meta["sha256"]:
            raise DatasetError("chart vector cache segment index SHA-256 mismatch")
        segment_rows = load_jsonl(segment_path)
        segment_indices = {str(row["segment_id"]): int(row["cache_index"]) for row in segment_rows}
        if len(segment_indices) != int(manifest["segment_count"]):
            raise DatasetError("chart vector cache segment index is invalid")
        arrays: dict[str, np.ndarray] = {}
        for name, metadata in manifest["arrays"].items():
            path = root / str(metadata["path"])
            if verify_payloads and _sha256(path) != metadata["sha256"]:
                raise DatasetError(f"chart vector cache array SHA-256 mismatch: {name}")
            value = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(value.shape) != list(metadata["shape"]) or value.dtype.name != metadata["dtype"]:
                raise DatasetError(f"chart vector cache array metadata mismatch: {name}")
            arrays[str(name)] = value
        validate_vector_index(offsets, segment_indices, arrays, int(manifest["segment_count"]))
        return cls(
            root=root,
            manifest=manifest,
            segment_indices=segment_indices,
            segment_records={str(row["segment_id"]): row for row in segment_rows},
            offsets=offsets,
            arrays=arrays,
        )

    def song_ids(self, role: str) -> tuple[str, ...]:
        values = self.manifest.get("selection", {}).get(f"{role}_song_ids")
        if not isinstance(values, list):
            raise DatasetError(f"chart vector cache has no {role!r} song selection")
        return tuple(str(value) for value in values)

    def record(self, segment_id: str) -> dict[str, Any]:
        try:
            index = self.segment_indices[str(segment_id)]
        except KeyError as error:
            raise DatasetError(f"chart vector cache does not contain segment {segment_id}") from error
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
        for name, values in self.arrays.items():
            copied = np.array(values[begin:end], copy=True)
            tensor = torch.from_numpy(copied)
            if "." not in name:
                if name == "row_anchor_tick":
                    result[name] = tensor.to(torch.float32)
                else:
                    result[name] = tensor.to(torch.long)
                continue
            group, field = name.split(".", 1)
            if group == "input_times":
                tensor = tensor.to(torch.float32)
            elif group == "input_time_active":
                tensor = tensor.to(torch.bool)
            else:
                tensor = tensor.to(torch.long)
            result[group][field] = tensor
        return result


def verify_chart_vector_cache(*, profile_root: Path, cache_root: Path) -> dict[str, Any]:
    """Compare every cached tensor with current on-the-fly materialization."""

    profile_root = profile_root.expanduser().resolve()
    cache_root = cache_root.expanduser().resolve()
    summary = json.loads((profile_root / "summary.json").read_text(encoding="utf-8-sig"))
    vocabulary_path = profile_root / "vocabulary.json"
    vocabulary = CompactVocabulary.from_dict(
        json.loads(vocabulary_path.read_text(encoding="utf-8-sig"))
    )
    store = ChartVectorStore.load(
        cache_root,
        profile_fingerprint=str(summary["profile_fingerprint"]),
        vocabulary_sha256=_sha256(vocabulary_path),
    )
    by_id = {
        str(row["segment_id"]): row
        for row in load_jsonl(profile_root / "segments.jsonl")
        if str(row["segment_id"]) in store.segment_indices
    }
    if len(by_id) != len(store.segment_indices):
        missing = sorted(set(store.segment_indices) - set(by_id))
        raise DatasetError(f"chart vector cache references missing profile segments: {missing[:8]}")
    tensor_count = 0
    for index, segment_id in enumerate(store.segment_indices):
        expected = materialize_chart_vectors(
            profile_root=profile_root,
            segment=by_id[segment_id],
            vocabulary=vocabulary,
        )
        actual = store.record(segment_id)
        for name in store.arrays:
            expected_tensor = _tensor_at(expected, name)
            actual_tensor = _tensor_at(actual, name)
            if expected_tensor.dtype != actual_tensor.dtype or not torch.equal(
                expected_tensor, actual_tensor
            ):
                raise DatasetError(
                    f"chart vector cache mismatch at {segment_id} field {name}"
                )
            tensor_count += 1
        if (index + 1) % 10000 == 0 or index + 1 == len(store.segment_indices):
            print(
                f"verify chart vectors {index + 1}/{len(store.segment_indices)}",
                file=sys.stderr,
                flush=True,
            )
    return {
        "schema": "maimai.baseline_chart_vector_cache_audit.v1",
        "cache": str(cache_root),
        "profile_fingerprint": summary["profile_fingerprint"],
        "segment_count": len(store.segment_indices),
        "tensor_count": tensor_count,
        "all_tensors_equal": True,
        "contains_audio": False,
        "contains_aio": False,
    }


__all__ = [
    "audio_reference",
    "AUDIO_REFERENCE_SCHEMA",
    "ChartVectorStore",
    "SCHEMA",
    "build_chart_vector_cache",
    "materialize_chart_vectors",
    "TRAIN_ONLY_SUITE_SCHEMA",
    "verify_chart_vector_cache",
]
