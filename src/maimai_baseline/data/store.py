"""Prebuilt feature readers and inference audio cache publication."""

from __future__ import annotations
from maimai_dataset.window import WINDOW_POSITIONS, TARGET_TICKS, GRID_OFFSETS, WINDOW
from fractions import Fraction
from importlib.metadata import version
from maimai_baseline.data.audio import (
    extract_at_centers,
    policy_contract,
    prepare_waveform,
    validate_stats,
)
from maimai_baseline.time.grid import (
    INTEGRATION,
    SAMPLING,
    audio_grid,
    grid_coordinates,
    identity,
    timing_identity,
    window_first_k,
)
from maimai_dataset.audio import sha256_file
from maimai_dataset.array_store import ReadOnlyArrayStore
from maimai_dataset.geometry import build_segment_geometry
from maimai_dataset.timing import TimingMap
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
import uuid


SCHEMA = "maimai.bpm96_audio_manifest.v1"


GEOMETRY_SCHEMA = "maimai.bpm96_geometry.v1"


def preprocessing_source_identity():
    """Fingerprint preprocessing only; changing a model distribution reuses PCM/Mel."""
    source = Path(__file__).resolve().parents[2]
    names = (
        "maimai_baseline/data/audio.py", "maimai_baseline/time/grid.py",
        "maimai_baseline/time/coordinates.py", "maimai_baseline/data/store.py", "maimai_dataset/audio.py",
        "maimai_dataset/timing.py",
    )
    return {"schema": "maimai.bpm96_preprocessing_source.v1",
            "files": {name: sha256_file(source / name) for name in names}}


def prepare_raw_bpm_audio(path, *, stats, policy, level, timing, cache_root=None):
    policy_contract(policy)
    base_policy = policy
    material = {"sampling": SAMPLING, "timing": timing_identity(timing), "source_sha256": sha256_file(Path(path)),
                "audio_contract": policy_contract(base_policy), "stats_sha256": identity(stats),
                "preprocessing_source": preprocessing_source_identity(),
                "dependencies": {n: version(n) for n in ("numpy", "torch", "torchcodec", "av", "pyloudnorm", "soundfile")}}
    cache = None if cache_root is None else Path(cache_root) / identity(material)
    if cache is not None and (cache / "metadata.json").exists():
        completed(cache / "metadata.json", material)
        meta = read_json(cache / "metadata.json")
        if meta.get("identity") != material:
            raise ValueError("BPM inference cache identity mismatch")
        for name in ("base.npy", "coordinates.npz"):
            completed(cache / name, material)
            if sha256_file(cache / name) != meta["hashes"][name]:
                raise ValueError(f"corrupt BPM inference cache: {name}")
        base = np.load(cache / "base.npy", allow_pickle=False)
        with np.load(cache / "coordinates.npz", allow_pickle=False) as archive:
            coords = {n: archive[n] for n in archive.files}
        info = meta["audio"]
        expected = audio_grid(timing, info["num_samples"])
        if set(expected) != set(coords) or any(coords[n].dtype != v.dtype or not np.array_equal(coords[n], v) for n, v in expected.items()):
            raise ValueError("corrupt BPM cache grid coordinates")
    else:
        wave, info = prepare_waveform(path, policy=base_policy)
        if info["source_sha256"] != material["source_sha256"]:
            raise ValueError("source changed after BPM cache identity was resolved")
        coords = audio_grid(timing, wave.shape[1])
        base = extract_at_centers(wave, coords, stats)
        if cache is not None:
            data = publish(cache / "base.npy", base, material)
            coordinates = publish(cache / "coordinates.npz", coords, material)
            publish(cache / "metadata.json", {"identity": material, "audio": info,
                    "hashes": {"base.npy": data["sha256"], "coordinates.npz": coordinates["sha256"]}}, material)
    if base.dtype != np.float32 or base.shape != (128, len(coords["grid_k"])) or not np.isfinite(base).all():
        raise ValueError("invalid BPM inference base feature")
    result = base
    return result, {**info, "policy": policy, "level": float(level), "level_gain_db": 0.0,
                    "base_cache_identity": material, "feature_sha256": hashlib.sha256(result.tobytes()).hexdigest()}, {
                    "coordinates": coords, "num_samples": info["num_samples"]}


def read_json(path):
    return json.loads(Path(path).read_text("utf-8-sig"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text("utf-8-sig").splitlines() if line.strip()]


def contained(root, path):
    root = Path(root).resolve()
    result = (root / path).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"artifact escapes root: {path}")
    return result


def completed(path, input_identity, *, verify_payloads=True):
    path = Path(path)
    receipt = path.with_name(path.name + ".complete.json")
    if path.with_name(path.name + ".lock").exists():
        raise ValueError(f"active/interrupted artifact lock (quarantine before retry): {path}")
    if not path.exists() and not receipt.exists():
        return None
    if not path.is_file() or not receipt.is_file():
        raise ValueError(f"incomplete published artifact (quarantine before retry): {path}")
    saved = read_json(receipt)
    if saved["identity"] != input_identity or (verify_payloads and sha256_file(path) != saved["sha256"]):
        raise ValueError(f"conflicting/corrupt artifact: {path}")
    return saved


def publish(path, value, input_identity):
    """Data first, completion receipt last. Never overwrite a completed file."""
    path = Path(path)
    receipt = path.with_name(path.name + ".complete.json")
    saved = completed(path, input_identity)
    if saved is not None:
        return saved
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    # Unique unpublished temporaries prevent concurrent writers sharing bytes.
    with temp.open("wb") as stream:
        if isinstance(value, dict) and all(isinstance(v, np.ndarray) for v in value.values()):
            np.savez(stream, **value)
        elif isinstance(value, np.ndarray):
            np.save(stream, value, allow_pickle=False)
        elif isinstance(value, bytes):
            stream.write(value)
        else:
            stream.write((json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode())
    # Validate serialization, shape and finite floating values before publishing.
    if isinstance(value, np.ndarray):
        check = np.load(temp, allow_pickle=False)
        if not np.array_equal(check, value) or (check.dtype.kind == "f" and not np.isfinite(check).all()):
            raise ValueError(f"invalid temporary array: {path}")
    elif isinstance(value, dict) and all(isinstance(v, np.ndarray) for v in value.values()):
        with np.load(temp, allow_pickle=False) as check:
            if set(check.files) != set(value) or any(not np.array_equal(check[k], v) or
                    (v.dtype.kind == "f" and not np.isfinite(v).all()) for k, v in value.items()):
                raise ValueError(f"invalid temporary coordinates: {path}")
    saved = {"identity": input_identity, "sha256": sha256_file(temp)}
    # 'x' lock protects two writers; an interrupted lock requires explicit quarantine.
    lock = path.with_name(path.name + ".lock")
    with lock.open("x"):
        if path.exists() or receipt.exists():
            raise ValueError(f"artifact appeared during publication: {path}")
        temp.replace(path)
        receipt_temp = receipt.with_name(receipt.name + ".tmp")
        receipt_temp.write_text(json.dumps(saved, sort_keys=True, allow_nan=False), encoding="utf8")
        receipt_temp.replace(receipt)
    lock.unlink()
    return saved


def window_arrays(features, coords, timing, start_tick, num_samples, *, include_geometry=True):
    wanted = grid_coordinates(timing, window_first_k(start_tick), WINDOW_POSITIONS, num_samples)
    value = np.zeros((WINDOW_POSITIONS, 128), dtype=np.float32)
    k = coords["grid_k"]
    if not len(k):
        raise ValueError("empty BPM audio grid")
    positions = wanted["grid_k"] - int(k[0])
    valid = wanted["valid"] & (positions >= 0) & (positions < len(k))
    value[valid] = features[:, positions[valid]].T
    beats = (wanted["grid_k"] / 24).astype(np.float32)
    bpms = np.asarray([timing.bpm_at_beat(Fraction(int(i), 24)) for i in wanted["grid_k"]], dtype=np.float32)
    geometry = (build_segment_geometry(frame_beats=beats, frame_valid=valid,
        target_start_tick=start_tick, target_end_tick=start_tick + TARGET_TICKS) if include_geometry else None)
    return value, valid, beats, bpms, geometry


def _validate_coordinate_structure(coords):
    types = {"grid_k": "int64", "seconds": "float64", "samples": "int64", "valid": "bool"}
    if set(coords) != set(types):
        raise ValueError("BPM coordinate fields mismatch")
    length = len(coords["grid_k"])
    if not length or any(a.shape != (length,) or a.dtype.name != types[n] for n, a in coords.items()):
        raise ValueError("BPM coordinate shape/dtype mismatch")


def _validate_geometry_structure(geometry, start_tick):
    expected = {"candidate_start_tick": ((), "int32")}
    for prefix, size in (("grid", GRID_OFFSETS[-1]), ("a1_to_a0", WINDOW_POSITIONS), ("candidate", TARGET_TICKS + 1)):
        for suffix, dtype in (("left", "int32"), ("right", "int32"), ("weight", "float32"), ("padding", "bool")):
            expected[f"{prefix}_{suffix}"] = ((size,), dtype)
    if set(geometry) != set(expected) or any((a.shape, a.dtype.name) != expected[n] for n, a in geometry.items()):
        raise ValueError("BPM geometry shape/dtype mismatch")
    if int(geometry["candidate_start_tick"]) != start_tick:
        raise ValueError("BPM geometry window start mismatch")
    for suffix in ("left", "right"):
        checks = [(geometry[f"candidate_{suffix}"], WINDOW_POSITIONS), (geometry[f"a1_to_a0_{suffix}"], WINDOW_POSITIONS // 2)]
        for i, (begin, end) in enumerate(zip(GRID_OFFSETS, GRID_OFFSETS[1:])):
            limit = WINDOW_POSITIONS // (2 ** i)
            checks.append((geometry[f"grid_{suffix}"][begin:end], limit))
        if any(np.any(a < 0) or np.any(a >= limit) for a, limit in checks):
            raise ValueError("BPM geometry index bounds mismatch")


class BpmFeatureStore(ReadOnlyArrayStore):
    def __getstate__(self):
        from maimai_dataset.array_store import mmap_pickle_state
        return mmap_pickle_state({**self.__dict__, "_geometry_cache": {}})

    def __init__(self, manifest, *, profile, stats, song_ids, selection_sha256, geometry_root,
                 expected_segment_ids=None, verify_payloads=True):
        self.verify_payloads = bool(verify_payloads)
        self.manifest = Path(manifest).resolve()
        self.root = self.manifest.parent
        self.stats = dict(stats)
        validate_stats(stats)
        value = read_json(self.manifest)
        if value.get("window") != WINDOW or value.get("schema") != SCHEMA or value.get("sampling") != SAMPLING or value.get("integration") != INTEGRATION or value.get("complete") is not True:
            raise ValueError("incomplete/incompatible BPM audio manifest")
        self.value = value
        self.policy = value["base_policy"]
        self.variants = value["variants"]
        self.segments = value["segments"]
        self.levels = {str(r["record_id"]): r["level_value"] for r in read_jsonl(Path(profile) / "manifest.jsonl") if str(r["song_id"]) in song_ids}
        self._features, self._coordinates = {}, {}
        variant_timings = {}
        for key, v in self.variants.items():
            timing = TimingMap.from_dict(v["timing"])
            variant_timings[key] = timing_identity(timing)
            fpath, cpath = (contained(self.root, v[n]) for n in ("feature_path", "coordinates_path"))
            if verify_payloads and (sha256_file(fpath) != v["feature_sha256"] or sha256_file(cpath) != v["coordinates_sha256"]):
                raise ValueError(f"corrupt BPM feature/coordinates: {key}")
            feature = np.load(fpath, mmap_mode="r", allow_pickle=False)
            with np.load(cpath, allow_pickle=False) as archive:
                coords = {n: archive[n] for n in archive.files}
            if verify_payloads:
                expected = audio_grid(timing, v["num_samples"])
                if set(coords) != set(expected) or any(coords[n].dtype != a.dtype or not np.array_equal(coords[n], a) for n, a in expected.items()):
                    raise ValueError(f"BPM coordinate mapping mismatch: {key}")
            else:
                _validate_coordinate_structure(coords)
            if (feature.dtype != np.float32 or feature.shape != (128, len(coords["grid_k"]))
                    or list(feature.shape) != v["shape"] or v["dtype"] != "float32" or (verify_payloads and not np.isfinite(feature).all())):
                raise ValueError(f"BPM feature shape/dtype/finite mismatch: {key}")
            self._features[key], self._coordinates[key] = feature, coords
        self.geometry_root = Path(geometry_root).resolve()
        geometry_path = self.geometry_root / "manifest.json"
        self.geometry = read_json(geometry_path)
        if self.geometry.get("schema") != GEOMETRY_SCHEMA:
            raise ValueError("BPM geometry schema/coverage mismatch")
        # Runtime geometry is opened and checked on first consumption only.
        self._geometry_cache = {}
        # Immutable profile metadata is checked once at startup, not every
        # time a persistent worker reads a window. No feature payload scan.
        record_timings = {}
        self.prepared_timings = {}
        checked_pairs = set()
        for segment in read_jsonl(Path(profile) / "segments.jsonl"):
            sid = str(segment["segment_id"])
            if sid not in self.segments:
                continue
            rid = str(segment["record_id"])
            payload = segment["timing"]
            if rid not in record_timings:
                from maimai_baseline.time.offline import PreparedTiming
                timing = TimingMap.from_dict(payload)
                record_timings[rid] = (payload, timing_identity(timing))
                self.prepared_timings[rid] = PreparedTiming(timing)
            saved_payload, curve = record_timings[rid]
            if payload != saved_payload:
                raise ValueError(f"BPM record contains inconsistent timing maps: {rid}")
            key = self.segments[sid]
            pair = (rid, key)
            if pair not in checked_pairs:
                if curve != variant_timings[key]:
                    raise ValueError(f"BPM feature/segment timing mismatch: {sid}")
                checked_pairs.add(pair)
            if verify_payloads:
                self.window(segment, validate_only=True)
        self.identity = {"manifest_sha256": sha256_file(self.manifest), "selection_sha256": selection_sha256,
                         "geometry_manifest_sha256": value["geometry_manifest_sha256"], "stats_file_sha256": value["stats_sha256"],
                         "preparation_identity": value["preparation_identity"], "variants": len(self.variants)}

    def window(self, segment, validate_only=False):
        verify_payloads = getattr(self, "verify_payloads", True)
        sid = str(segment["segment_id"])
        key = self.segments[sid]
        v = self.variants[key]
        record = self.geometry["segments"][sid]
        path = contained(self.geometry_root, record["path"])
        cache_key = (key, int(segment["target_start_tick"]), str(path), record["sha256"])
        if validate_only and cache_key in self._geometry_cache:
            return
        timing = self.prepared_timings[str(segment["record_id"])].get(
            getattr(self, "offline_timing_root", None)
        )
        arrays = window_arrays(self._features[key], self._coordinates[key], timing,
                               int(segment["target_start_tick"]), v["num_samples"],
                               include_geometry=verify_payloads and cache_key not in self._geometry_cache)
        if cache_key not in self._geometry_cache:
            if verify_payloads and sha256_file(path) != record["sha256"]:
                raise ValueError(f"corrupt BPM geometry: {sid}")
            with np.load(path, allow_pickle=False) as archive:
                geometry = {n: archive[n] for n in archive.files}
            _validate_geometry_structure(geometry, int(segment["target_start_tick"]))
            if verify_payloads and (set(geometry) != set(arrays[4]) or any(geometry[n].dtype != a.dtype or
                    not np.array_equal(geometry[n], a) for n, a in arrays[4].items())):
                raise ValueError(f"BPM geometry coordinate mismatch: {sid}")
            self._geometry_cache[cache_key] = geometry
        geometry = self._geometry_cache[cache_key]
        if validate_only:
            return
        mel, valid, beats, bpms, _ = arrays
        return {"audio": torch.from_numpy(mel), "audio_valid": torch.from_numpy(valid),
                "frame_beats": torch.from_numpy(beats), "frame_bpms": torch.from_numpy(bpms),
                "audio_start_tick": float(segment["target_start_tick"]),
                "interpolation_geometry": {n: torch.from_numpy(a.copy()) for n, a in geometry.items()}}
