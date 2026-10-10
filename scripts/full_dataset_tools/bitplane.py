"""Canonical v2 builder ported from the existing offline producer.
Source SHA256: 769b3dd45b086bc9913905b3bdc2d58110924eb7cd4218abbbd121f9fcf13b3d
Only current baseline imports; optional verified pilot payload reuse.
"""
from __future__ import annotations
from maimai_baseline.physical.bitplane_cache import PhysicalBitplaneStore
from maimai_dataset.window import WINDOW

from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import bisect
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

import numpy as np
import torch

from maimai_dataset.manifest import DatasetError, canonical_json_sha256, load_jsonl
from maimai_dataset.timing import TimingMap
from maimai_dataset.vocabulary import CompactVocabulary
from maimai_baseline.data.route import Route71Codec
from maimai_baseline.contract import VIEW_NAMES

from maimai_baseline.physical.catalog import CompactPathCatalog, pad_bit
from maimai_baseline.physical.contract import (
    ACTION_FLOAT_COLUMNS,
    BITPLANE_CACHE_SCHEMA,
    BITPLANE_CACHE_CONTRACT,
    CACHE_ACTION_COLUMNS,
    EVENT_NOTE,
    EVENT_SLIDE_BRANCH,
    KIND_HOLD,
    KIND_NONE,
    KIND_SLIDE,
    KIND_TAP,
    KIND_TOUCH,
    KIND_TOUCH_HOLD,
    KIND_WIFI,
    PLANE_COUNT,
    PRESS_STATE,
    SLIDE_ENTRY,
    SLIDE_LAUNCH_GUARD,
    SLIDE_OCCUPANCY,
    TOUCH_STATE,
    VALID_REGION_MASK,
)
from maimai_baseline.physical.oracle import project_path


_KIND_CODES = {
    "tap": KIND_TAP,
    "touch": KIND_TOUCH,
    "hold": KIND_HOLD,
    "touch_hold": KIND_TOUCH_HOLD,
    "slide": KIND_SLIDE,
    "wifi": KIND_WIFI,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _half_up(value: float) -> int:
    return int(math.floor(float(value) + 0.5 + 1e-8))


def _onset_slot(second: float) -> int:
    return int(math.floor(float(second) * 100.0 + 1e-8))


def _tick_second(timing: TimingMap, tick: int) -> float:
    return float(timing.beat_to_second(Fraction(int(tick), 96)))


def _route_class(codec: Route71Codec, row: Mapping[str, Any]) -> int:
    key = (
        str(row["shape"]),
        int(row["end_offset"]),
        None if "via_offset" not in row else int(row["via_offset"]),
    )
    try:
        return int(codec._factor_to_variant[key])
    except KeyError as error:
        raise DatasetError(f"route is outside Route-71: {key}") from error


def _write_plane(
    planes: np.ndarray, plane: int, origin: int, slot: int, mask: int
) -> None:
    index = int(slot) - int(origin)
    if 0 <= index < planes.shape[1]:
        planes[int(plane), index] |= np.int64(int(mask) & VALID_REGION_MASK)


def _fill_plane(
    planes: np.ndarray,
    plane: int,
    origin: int,
    start: int,
    stop: int,
    mask: int,
) -> None:
    left = max(0, int(start) - int(origin))
    right = min(planes.shape[1], int(stop) - int(origin))
    if right > left:
        planes[int(plane), left:right] |= np.int64(int(mask) & VALID_REGION_MASK)


def _array_metadata(root: Path, relative: str, value: np.ndarray) -> dict[str, Any]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ascontiguousarray(value), allow_pickle=False)
    actual = path if path.suffix == ".npy" else path.with_suffix(path.suffix + ".npy")
    return {
        "path": actual.relative_to(root).as_posix(),
        "dtype": str(value.dtype),
        "shape": [int(part) for part in value.shape],
        "bytes": int(actual.stat().st_size),
        "sha256": _sha256_file(actual),
    }


def _chart_payload(
    *,
    chart_index: int,
    record_id: str,
    encoded_path: Path,
    timing_value: Mapping[str, Any],
    song_end_tick: int,
    catalog: CompactPathCatalog,
    codec: Route71Codec,
) -> dict[str, Any]:
    encoded_sha256 = _sha256_file(encoded_path)
    encoded = json.loads(encoded_path.read_text("utf-8"))
    if str(encoded.get("record_id")) != str(record_id):
        raise DatasetError(f"encoded chart identity mismatch: {record_id}")
    from maimai_dataset.modifiers import normalize_row, materialize_row
    rows = [materialize_row(normalize_row(row)) for row in encoded.get("rows", ())]
    tick = latest = 0
    for row in rows:
        if row.get("event") == "MOMENT":
            tick += int(row["delta_tick"])
        latest = max(latest, tick + int(row.get("hold_duration_tick", 0)),
                     tick + 96 + int(row["branch_duration_tick"]) if "branch_duration_tick" in row else tick)
    song_end_tick = max(song_end_tick, latest)
    timing = TimingMap.from_dict(timing_value)
    origin = int(math.floor(float(timing.first_seconds) * 100.0)) - 16
    final_slot = int(math.ceil(_tick_second(timing, int(song_end_tick)) * 100.0)) + 16
    slot_count = max(1, final_slot - origin + 1)
    planes = np.zeros((PLANE_COUNT, slot_count), dtype=np.int64)
    source_action = np.full((len(rows),), -1, dtype=np.int32)
    source_moment = np.full((len(rows),), -1, dtype=np.int32)
    actions: list[list[int]] = []
    action_float: list[list[float]] = []
    moment_rows: list[int] = []
    moment_ticks: list[int] = []
    moment_slots: list[int] = []
    tick = 0
    moment_index = -1

    for source_row, row in enumerate(rows):
        event = str(row.get("event", ""))
        if event == "MOMENT":
            tick += int(row["delta_tick"])
            moment_index += 1
            moment_rows.append(source_row)
            moment_ticks.append(tick)
            moment_slots.append(_onset_slot(_tick_second(timing, tick)))
        source_moment[source_row] = moment_index
        if event not in {"NOTE", "SLIDE_BRANCH"}:
            continue
        if moment_index < 0:
            raise DatasetError(f"{record_id} action precedes its first MOMENT")
        event_code = EVENT_NOTE if event == "NOTE" else EVENT_SLIDE_BRANCH
        kind = _KIND_CODES.get(str(row.get("kind", "")), KIND_NONE)
        if event_code == EVENT_NOTE and kind == KIND_NONE:
            raise DatasetError(f"{record_id}:{source_row} has unsupported NOTE kind")
        path_id = -1
        region_mask = 0
        if event_code == EVENT_NOTE and kind <= KIND_TOUCH_HOLD:
            region_mask = pad_bit(str(row.get("pad", "")))
        elif event_code == EVENT_NOTE or event_code == EVENT_SLIDE_BRANCH:
            start_pad = str(row.get("start_pad", ""))
            if len(start_pad) != 2 or start_pad[0] != "A" or not start_pad[1].isdigit():
                raise DatasetError(f"{record_id}:{source_row} has invalid slide start")
            lane = int(start_pad[1])
            region_mask = pad_bit(start_pad) if event_code == EVENT_NOTE else 0
            path_id = catalog.path_id(lane, _route_class(codec, row))
            if path_id < 0:
                raise DatasetError(f"{record_id}:{source_row} has invalid physical path")
        if event_code == EVENT_NOTE and not region_mask:
            raise DatasetError(f"{record_id}:{source_row} has no region")

        head_second = _tick_second(timing, tick)
        head_slot = _onset_slot(head_second)
        duration_tick = 0
        hold_end_slot = head_slot + 1
        launch_slot = -1
        slide_end_slot = -1
        trace_units = motion_units = deadline_units = 0.0
        if kind in {KIND_HOLD, KIND_TOUCH_HOLD}:
            duration_tick = int(row.get("hold_duration_tick", 0))
            duration_second = max(
                0.0, _tick_second(timing, tick + duration_tick) - head_second
            )
            hold_end_slot = head_slot + max(
                1, int(math.ceil(duration_second * 100.0 - 1e-9))
            )
        if path_id >= 0:
            duration_tick = max(1, int(row.get("branch_duration_tick", 1)))
            trace_second = _tick_second(timing, tick + 96)
            end_second = _tick_second(timing, tick + 96 + duration_tick)
            motion_second = max(0.01, end_second - trace_second)
            trace_units = trace_second * 100.0
            motion_units = motion_second * 100.0
            launch_slot = _half_up(trace_units)
            slide_end_slot = max(launch_slot + 1, _half_up(end_second * 100.0))
            critical = float(catalog.path_critical_proportions[path_id])
            critical_second = trace_second + critical * motion_second
            deadline_units = (
                critical_second
                - (42.0 / 180.0 + (1.0 - critical) * motion_second / 4.0)
            ) * 100.0

        local_action = len(actions)
        source_action[source_row] = local_action
        actions.append(
            [
                moment_index,
                source_row,
                event_code,
                kind,
                region_mask,
                path_id,
                tick,
                duration_tick,
                head_slot,
                hold_end_slot,
                launch_slot,
                slide_end_slot,
            ]
        )
        action_float.append(
            [head_second * 100.0, trace_units, motion_units, deadline_units]
        )

        if event_code == EVENT_NOTE:
            plane = TOUCH_STATE if kind in {KIND_TOUCH, KIND_TOUCH_HOLD} else PRESS_STATE
            stop = hold_end_slot if kind in {KIND_HOLD, KIND_TOUCH_HOLD} else head_slot + 1
            _fill_plane(planes, plane, origin, head_slot, stop, region_mask)
        if path_id >= 0:
            masks_start = int(catalog.path_offsets[path_id])
            first_mask = int(catalog.path_masks[masks_start])
            _fill_plane(
                planes,
                SLIDE_LAUNCH_GUARD,
                origin,
                launch_slot - 5,
                launch_slot + 1,
                first_mask,
            )
            occupancy, entries, _ = project_path(
                catalog,
                path_id=path_id,
                trace_start_units=trace_units,
                motion_units=motion_units,
                launch_slot=launch_slot,
                end_slot=slide_end_slot,
            )
            for start, stop, mask in occupancy:
                _fill_plane(
                    planes, SLIDE_OCCUPANCY, origin, start, stop, mask
                )
            for entry_slot, mask in entries:
                _write_plane(planes, SLIDE_ENTRY, origin, entry_slot, mask)

    action_values = np.asarray(actions, dtype=np.int64).reshape(
        -1, len(CACHE_ACTION_COLUMNS)
    )
    action_floating = np.asarray(action_float, dtype=np.float64).reshape(
        -1, len(ACTION_FLOAT_COLUMNS)
    )
    carry: list[list[int]] = [[] for _ in moment_slots]
    moment_action_counts = [0] * len(moment_slots)
    for action in action_values:
        moment_action_counts[int(action[0])] += 1
    max_moment_actions = max(moment_action_counts, default=0)
    if max_moment_actions > 64:
        raise DatasetError(
            f"{record_id} has {max_moment_actions} actions in one moment; CUDA v2 limit is 64"
        )
    for action_index, action in enumerate(action_values):
        action_moment = int(action[0])
        horizon = max(
            int(action[8]) + 1,
            int(action[9]),
            int(action[10]),
            int(action[11]),
        )
        stop = bisect.bisect_right(moment_slots, horizon)
        for target_moment in range(action_moment + 1, stop):
            carry[target_moment].append(action_index)
    carry_offsets = [0]
    carry_indices: list[int] = []
    for values in carry:
        carry_indices.extend(values)
        carry_offsets.append(len(carry_indices))

    points = list(timing.points)
    return {
        "chart_index": int(chart_index),
        "record_id": str(record_id),
        "encoded_sha256": encoded_sha256,
        "slot_origin": int(origin),
        "slot_count": int(slot_count),
        "max_moment_actions": int(max_moment_actions),
        "planes": planes,
        "action_values": action_values,
        "action_floating": action_floating,
        "source_action": source_action,
        "source_moment": source_moment,
        "moment_rows": np.asarray(moment_rows, dtype=np.int32),
        "moment_ticks": np.asarray(moment_ticks, dtype=np.int64),
        "moment_slots": np.asarray(moment_slots, dtype=np.int64),
        "carry_offsets": np.asarray(carry_offsets, dtype=np.int64),
        "carry_indices": np.asarray(carry_indices, dtype=np.int32),
        "timing_ticks": np.asarray([float(p.beat) * 96.0 for p in points], dtype=np.float64),
        "timing_seconds": np.asarray([p.second for p in points], dtype=np.float64),
        "timing_bpms": np.asarray([p.bpm for p in points], dtype=np.float64),
    }


_WORKER_CATALOG: CompactPathCatalog | None = None
_WORKER_CODEC: Route71Codec | None = None


_REUSE_STORES: dict[str, PhysicalBitplaneStore] = {}


def _payload_or_reuse(*, payload, **kwargs):
    root = payload.get("reuse_root")
    if not root:
        return _chart_payload(**kwargs)
    if root not in _REUSE_STORES:
        _REUSE_STORES[root] = PhysicalBitplaneStore(Path(root))
    store = _REUSE_STORES[root]
    record = kwargs["record_id"]
    index = store.record_indices[record]
    meta = store.chart_records[index]
    result = {"chart_index": kwargs["chart_index"], "record_id": record,
              "encoded_sha256": _sha256_file(kwargs["encoded_path"]),
              "slot_origin": int(meta["slot_origin"]), "slot_count": int(meta["slot_count"]),
              "max_moment_actions": int(meta["max_moment_actions"])}
    groups = {"plane": ["planes"], "action": ["action_values", "action_floating"],
              "source": ["source_action", "source_moment"],
              "moment": ["moment_rows", "moment_ticks", "moment_slots"],
              "carry_offsets": ["carry_offsets"], "carry_index": ["carry_indices"],
              "timing": ["timing_ticks", "timing_seconds", "timing_bpms"]}
    for prefix, names in groups.items():
        begin, end = map(int, store.offsets[prefix + "_offsets"][index:index + 2])
        for name in names:
            result[name] = np.array(store.arrays[name][begin:end], copy=True)
    result["planes"] = result["planes"].reshape(PLANE_COUNT, result["slot_count"])
    timing = TimingMap.from_dict(kwargs["timing_value"])
    expected = {"timing_ticks": [float(p.beat) * 96 for p in timing.points],
                "timing_seconds": [p.second for p in timing.points],
                "timing_bpms": [p.bpm for p in timing.points]}
    for name, values in expected.items():
        if not np.array_equal(result[name], np.asarray(values, dtype=np.float64)):
            raise DatasetError(f"pilot reuse timing mismatch: {record}/{name}")
    encoded = json.loads(Path(kwargs["encoded_path"]).read_text(encoding="utf-8"))
    if len(encoded["rows"]) != len(result["source_action"]):
        raise DatasetError(f"pilot reuse source rows differ: {record}")
    return result


def _worker_init(catalog_root: str, vocabulary_payload: Mapping[str, Any]) -> None:
    global _WORKER_CATALOG, _WORKER_CODEC
    vocabulary = CompactVocabulary.from_dict(vocabulary_payload)
    _WORKER_CATALOG = CompactPathCatalog.load(Path(catalog_root), vocabulary)
    _WORKER_CODEC = Route71Codec(vocabulary)


def _worker_build(payload: Mapping[str, Any]) -> tuple[int, str, Mapping[str, Any]]:
    if _WORKER_CATALOG is None or _WORKER_CODEC is None:
        raise RuntimeError("Physical bitplane worker was not initialized")
    staging = Path(str(payload["staging"]))
    result = _payload_or_reuse(
        payload=payload,
        chart_index=int(payload["chart_index"]),
        record_id=str(payload["record_id"]),
        encoded_path=Path(str(payload["encoded_path"])),
        timing_value=payload["timing"],
        song_end_tick=int(payload["song_end_tick"]),
        catalog=_WORKER_CATALOG,
        codec=_WORKER_CODEC,
    )
    path = staging / "charts" / f"{int(payload['chart_index']):05d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
    np.savez(path, **arrays)
    metadata = {
        key: value for key, value in result.items() if not isinstance(value, np.ndarray)
    }
    metadata.update(
        {
            "action_count": int(arrays["action_values"].shape[0]),
            "source_row_count": int(arrays["source_action"].shape[0]),
            "moment_count": int(arrays["moment_rows"].shape[0]),
            "carry_count": int(arrays["carry_indices"].shape[0]),
            "timing_count": int(arrays["timing_ticks"].shape[0]),
        }
    )
    return int(payload["chart_index"]), str(path), metadata


def _prefix_offsets(counts: Iterable[int], *, scale: int = 1) -> np.ndarray:
    values = [0]
    for count in counts:
        values.append(values[-1] + int(count) * int(scale))
    return np.asarray(values, dtype=np.int64)


def _copy_npz_array(
    target: np.ndarray, offsets: np.ndarray, chart_index: int, source: np.ndarray
) -> None:
    begin = int(offsets[chart_index])
    end = int(offsets[chart_index + 1])
    if end - begin != int(source.shape[0]):
        raise DatasetError("Physical bitplane staging shape drifted")
    target[begin:end] = source


def build_physical_bitplane_cache(
    *,
    profile_root: Path,
    chart_cache_root: Path,
    spatial_cache_root: Path,
    catalog_root: Path,
    output: Path,
    workers: int = 1,
    reuse: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    profile_root = Path(profile_root).expanduser().resolve()
    chart_cache_root = Path(chart_cache_root).expanduser().resolve()
    spatial_cache_root = Path(spatial_cache_root).expanduser().resolve()
    catalog_root = Path(catalog_root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output.exists():
        old_schema = None
        manifest_path = output / "manifest.json"
        if manifest_path.is_file():
            old_schema = json.loads(manifest_path.read_text("utf-8-sig")).get("schema")
        raise DatasetError(
            f"refusing to overwrite Physical cache {output}; existing schema={old_schema!r}"
        )
    for path in (
        profile_root / "summary.json",
        profile_root / "vocabulary.json",
        profile_root / "segments.jsonl",
        chart_cache_root / "manifest.json",
        spatial_cache_root / "manifest.json",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = json.loads((profile_root / "summary.json").read_text("utf-8-sig"))
    vocabulary_path = profile_root / "vocabulary.json"
    vocabulary_payload = json.loads(vocabulary_path.read_text("utf-8-sig"))
    vocabulary = CompactVocabulary.from_dict(vocabulary_payload)
    catalog = CompactPathCatalog.load(catalog_root, vocabulary)
    chart_manifest = json.loads(
        (chart_cache_root / "manifest.json").read_text("utf-8-sig")
    )
    spatial_manifest = json.loads(
        (spatial_cache_root / "manifest.json").read_text("utf-8-sig")
    )
    if str(chart_manifest.get("profile_fingerprint")) != str(
        summary["profile_fingerprint"]
    ):
        raise DatasetError("chart cache/profile fingerprint mismatch")
    if str(spatial_manifest.get("profile_fingerprint")) != str(
        summary["profile_fingerprint"]
    ):
        raise DatasetError("spatial cache/profile fingerprint mismatch")
    if str(spatial_manifest.get("selection_sha256")) != str(
        chart_manifest["selection"]["selection_sha256"]
    ):
        raise DatasetError("spatial/chart cache selection mismatch")

    profile_segments = {
        str(row["segment_id"]): row
        for row in load_jsonl(profile_root / "segments.jsonl")
    }
    all_selected_segment_rows = load_jsonl(chart_cache_root / "segments.jsonl")
    unknown_roles = sorted(
        {
            str(row.get("role"))
            for row in all_selected_segment_rows
            if str(row.get("role")) not in {"train", "validation", "test"}
        }
    )
    if unknown_roles:
        raise DatasetError(f"chart cache contains unsupported roles: {unknown_roles}")
    selected_segment_rows = [
        row
        for row in all_selected_segment_rows
        if str(row.get("role")) in {"train", "validation"}
    ]
    selected_ids = [str(row["segment_id"]) for row in selected_segment_rows]
    missing = sorted(set(selected_ids) - set(profile_segments))
    if missing:
        raise DatasetError(f"chart cache contains unknown segments: {missing[:8]}")
    record_role: dict[str, str] = {}
    record_segment: dict[str, Mapping[str, Any]] = {}
    for cached in selected_segment_rows:
        segment = profile_segments[str(cached["segment_id"])]
        record_id = str(segment["record_id"])
        role = str(cached["role"])
        if record_id in record_role and record_role[record_id] != role:
            raise DatasetError(f"chart {record_id} appears in two roles")
        record_role[record_id] = role
        record_segment.setdefault(record_id, segment)
    record_ids = sorted(
        record_segment,
        key=lambda value: (int(value.split(":", 1)[0]), int(value.split(":", 1)[1])),
    )
    chart_index = {record_id: index for index, record_id in enumerate(record_ids)}
    staging = output.with_name(f".{output.name}.building-{uuid4().hex}")
    staging.mkdir(parents=True)
    payloads = [
        {
            "staging": str(staging),
            "reuse_root": (reuse or {}).get(record_id),
            "chart_index": chart_index[record_id],
            "record_id": record_id,
            "encoded_path": str(profile_root / str(record_segment[record_id]["encoded_path"])),
            "timing": record_segment[record_id]["timing"],
            "song_end_tick": int(record_segment[record_id]["song_end_tick"]),
        }
        for record_id in record_ids
    ]
    try:
        if int(workers) > 1:
            with ProcessPoolExecutor(
                max_workers=int(workers),
                initializer=_worker_init,
                initargs=(str(catalog_root), vocabulary_payload),
            ) as executor:
                built = list(executor.map(_worker_build, payloads, chunksize=1))
        else:
            _worker_init(str(catalog_root), vocabulary_payload)
            built = [_worker_build(payload) for payload in payloads]
        built.sort(key=lambda value: value[0])
        metadata = [dict(value[2]) for value in built]
        if [int(value["chart_index"]) for value in metadata] != list(range(len(record_ids))):
            raise DatasetError("Physical bitplane worker results are incomplete")

        slot_offsets = _prefix_offsets(value["slot_count"] for value in metadata)
        plane_offsets = _prefix_offsets(
            (value["slot_count"] for value in metadata), scale=PLANE_COUNT
        )
        action_offsets = _prefix_offsets(value["action_count"] for value in metadata)
        source_offsets = _prefix_offsets(value["source_row_count"] for value in metadata)
        moment_offsets = _prefix_offsets(value["moment_count"] for value in metadata)
        carry_offsets_offsets = _prefix_offsets(
            (value["moment_count"] + 1 for value in metadata)
        )
        carry_index_offsets = _prefix_offsets(value["carry_count"] for value in metadata)
        timing_offsets = _prefix_offsets(value["timing_count"] for value in metadata)

        arrays_root = staging / "arrays"
        arrays_root.mkdir()
        specs: dict[str, tuple[np.dtype[Any], tuple[int, ...]]] = {
            "planes": (np.dtype(np.int64), (int(plane_offsets[-1]),)),
            "action_values": (
                np.dtype(np.int64),
                (int(action_offsets[-1]), len(CACHE_ACTION_COLUMNS)),
            ),
            "action_floating": (
                np.dtype(np.float64),
                (int(action_offsets[-1]), len(ACTION_FLOAT_COLUMNS)),
            ),
            "source_action": (np.dtype(np.int32), (int(source_offsets[-1]),)),
            "source_moment": (np.dtype(np.int32), (int(source_offsets[-1]),)),
            "moment_rows": (np.dtype(np.int32), (int(moment_offsets[-1]),)),
            "moment_ticks": (np.dtype(np.int64), (int(moment_offsets[-1]),)),
            "moment_slots": (np.dtype(np.int64), (int(moment_offsets[-1]),)),
            "carry_offsets": (
                np.dtype(np.int64),
                (int(carry_offsets_offsets[-1]),),
            ),
            "carry_indices": (
                np.dtype(np.int32),
                (int(carry_index_offsets[-1]),),
            ),
            "timing_ticks": (np.dtype(np.float64), (int(timing_offsets[-1]),)),
            "timing_seconds": (np.dtype(np.float64), (int(timing_offsets[-1]),)),
            "timing_bpms": (np.dtype(np.float64), (int(timing_offsets[-1]),)),
        }
        targets = {
            name: np.lib.format.open_memmap(
                arrays_root / f"{name}.npy", mode="w+", dtype=dtype, shape=shape
            )
            for name, (dtype, shape) in specs.items()
        }
        for index, (_, path_string, _) in enumerate(built):
            with np.load(path_string, allow_pickle=False) as chart:
                plane_begin = int(plane_offsets[index])
                plane_end = int(plane_offsets[index + 1])
                targets["planes"][plane_begin:plane_end] = chart["planes"].reshape(-1)
                for name in ("action_values", "action_floating"):
                    _copy_npz_array(targets[name], action_offsets, index, chart[name])
                for name in ("source_action", "source_moment"):
                    _copy_npz_array(targets[name], source_offsets, index, chart[name])
                for name in ("moment_rows", "moment_ticks", "moment_slots"):
                    _copy_npz_array(targets[name], moment_offsets, index, chart[name])
                _copy_npz_array(
                    targets["carry_offsets"], carry_offsets_offsets, index, chart["carry_offsets"]
                )
                _copy_npz_array(
                    targets["carry_indices"], carry_index_offsets, index, chart["carry_indices"]
                )
                for name in ("timing_ticks", "timing_seconds", "timing_bpms"):
                    _copy_npz_array(targets[name], timing_offsets, index, chart[name])
        for value in targets.values():
            value.flush()
        # np.memmap keeps an open Windows file mapping until every view is
        # released; close it before atomically renaming the staging directory.
        for mapped in targets.values():
            mapped._mmap.close()
        targets.clear()
        del value
        gc.collect()

        offset_arrays = {
            "plane_offsets": plane_offsets,
            "slot_offsets": slot_offsets,
            "action_offsets": action_offsets,
            "source_offsets": source_offsets,
            "moment_offsets": moment_offsets,
            "carry_offsets_offsets": carry_offsets_offsets,
            "carry_index_offsets": carry_index_offsets,
            "timing_offsets": timing_offsets,
            "slot_origins": np.asarray(
                [value["slot_origin"] for value in metadata], dtype=np.int64
            ),
            "slot_counts": np.asarray(
                [value["slot_count"] for value in metadata], dtype=np.int64
            ),
        }
        arrays_meta: dict[str, Any] = {}
        for name in specs:
            path = arrays_root / f"{name}.npy"
            arrays_meta[name] = {
                "path": path.relative_to(staging).as_posix(),
                "dtype": str(specs[name][0]),
                "shape": list(specs[name][1]),
                "bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
        offsets_meta = {
            name: _array_metadata(staging, f"offsets/{name}.npy", value)
            for name, value in offset_arrays.items()
        }

        chart_rows = []
        for value in metadata:
            index = int(value["chart_index"])
            chart_rows.append(
                {
                    "chart_index": index,
                    "record_id": value["record_id"],
                    "encoded_sha256": value["encoded_sha256"],
                    "role": record_role[str(value["record_id"])],
                    "slot_origin": int(value["slot_origin"]),
                    "slot_count": int(value["slot_count"]),
                    "storage_bytes": int(value["slot_count"]) * PLANE_COUNT * 8,
                    "action_count": int(value["action_count"]),
                    "source_row_count": int(value["source_row_count"]),
                    "moment_count": int(value["moment_count"]),
                    "carry_count": int(value["carry_count"]),
                    "max_moment_actions": int(value["max_moment_actions"]),
                }
            )
        chart_path = staging / "charts.jsonl"
        chart_path.write_text(
            "".join(json.dumps(value, sort_keys=True) + "\n" for value in chart_rows),
            "utf-8",
        )
        segment_path = staging / "segments.jsonl"
        segment_rows = [
            {
                "segment_id": segment_id,
                "record_id": str(profile_segments[segment_id]["record_id"]),
                "chart_index": chart_index[str(profile_segments[segment_id]["record_id"])],
                "role": str(cached["role"]),
            }
            for cached, segment_id in zip(selected_segment_rows, selected_ids, strict=True)
        ]
        segment_path.write_text(
            "".join(json.dumps(value, sort_keys=True) + "\n" for value in segment_rows),
            "utf-8",
        )
        storage = np.asarray([value["storage_bytes"] for value in chart_rows], dtype=np.int64)
        raw_array_bytes = {
            name: int(np.dtype(dtype).itemsize * math.prod(shape))
            for name, (dtype, shape) in specs.items()
        }
        offset_payload_bytes = sum(int(value.nbytes) for value in offset_arrays.values())
        manifest: dict[str, Any] = {
            "schema": BITPLANE_CACHE_SCHEMA,
            "window": WINDOW,
            "reuse_record_ids": sorted(reuse or {}),
            "contract": dict(BITPLANE_CACHE_CONTRACT),
            "profile_fingerprint": str(summary["profile_fingerprint"]),
            "profile_manifest_sha256": _sha256_file(profile_root / "manifest.jsonl"),
            "segments_sha256": _sha256_file(profile_root / "segments.jsonl"),
            "vocabulary_sha256": _sha256_file(vocabulary_path),
            "chart_cache_manifest_sha256": _sha256_file(chart_cache_root / "manifest.json"),
            "spatial_cache_manifest_sha256": _sha256_file(spatial_cache_root / "manifest.json"),
            "selection_sha256": str(chart_manifest["selection"]["selection_sha256"]),
            "path_catalog": dict(catalog.identity),
            "views": list(VIEW_NAMES),
            "storage": "canonical-only; views generated after H2D",
            "chart_count": len(chart_rows),
            "segment_count": len(segment_rows),
            "slot_count": int(slot_offsets[-1]),
            "action_count": int(action_offsets[-1]),
            "moment_count": int(moment_offsets[-1]),
            "carry_index_count": int(carry_index_offsets[-1]),
            "plane_payload_bytes": int(storage.sum()),
            "index_payload_bytes": int(
                sum(raw_array_bytes.values())
                - raw_array_bytes["planes"]
                + offset_payload_bytes
            ),
            "total_numeric_payload_bytes": int(
                sum(raw_array_bytes.values()) + offset_payload_bytes
            ),
            "per_chart_storage": {
                "mean_bytes": float(storage.mean()) if storage.size else 0.0,
                "min_bytes": int(storage.min()) if storage.size else 0,
                "max_bytes": int(storage.max()) if storage.size else 0,
                "median_bytes": float(np.median(storage)) if storage.size else 0.0,
            },
            "arrays": arrays_meta,
            "offsets": offsets_meta,
            "charts": {
                "path": "charts.jsonl",
                "sha256": _sha256_file(chart_path),
            },
            "segments": {
                "path": "segments.jsonl",
                "sha256": _sha256_file(segment_path),
            },
            "forbidden_fields": ["epoch", "occurrence", "batch", "prefix_action_indices"],
        }
        manifest["manifest_identity_sha256"] = canonical_json_sha256(manifest)
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            "utf-8",
        )
        audit_store = PhysicalBitplaneStore(
            staging,
            profile_fingerprint=str(summary["profile_fingerprint"]),
            chart_cache_manifest_sha256=_sha256_file(
                chart_cache_root / "manifest.json"
            ),
            spatial_cache_manifest_sha256=_sha256_file(
                spatial_cache_root / "manifest.json"
            ),
            catalog_identity=catalog.identity,
        )
        audit_store.validate(full_hash=True)
        del audit_store
        gc.collect()
        shutil.rmtree(staging / "charts")
        os.replace(staging, output)
        return manifest
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise

