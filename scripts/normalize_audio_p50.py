from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import importlib.metadata
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA = "maimai.audio_fixed_gain_p50.v1"
FEATURE_SCHEMA = "maimai.audio_fixed_gain_pilot_features.v1"
TARGET_DEFAULT = -8.092632368636558


def sha256_file(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line]


def safe_relative_path(text: str) -> Path:
    value = Path(text.replace("\\", "/"))
    if value.is_absolute() or ".." in value.parts:
        raise ValueError(f"unsafe relative audio path: {text!r}")
    return value


def output_audio_path(output: Path, relative: Path) -> Path:
    return output / "audio" / relative.parent.name / "track.wav"


def dependency_versions() -> dict[str, str]:
    names = ("av", "numpy", "pyloudnorm", "scipy", "soundfile", "torch", "torchcodec")
    return {name: importlib.metadata.version(name) for name in names}


def stage_identity(command: str, material: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "maimai.audio_fixed_gain_stage_identity.v1",
        "command": command,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "python": sys.version,
        "dependencies": dependency_versions(),
        **dict(material),
    }


def require_identity(path: Path, identity: Mapping[str, Any]) -> bool:
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8-sig"))
        if saved != dict(identity):
            raise ValueError(f"stage identity mismatch; refusing resume: {path}")
        return True
    else:
        atomic_json(path, identity)
        return False


def read_loudness_csv(path: Path) -> list[dict[str, Any]]:
    required = {
        "path", "sha256", "sample_rate", "channels", "window_count",
        "momentary_p10_lufs", "momentary_p50_lufs", "momentary_p90_lufs",
    }
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"loudness CSV lacks required fields: {sorted(required)}")
    paths = [str(safe_relative_path(row["path"])) for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("loudness CSV contains duplicate paths")
    return rows


def _imports_audio():
    try:
        import av
        import numpy as np
        import pyloudnorm as pyln
    except (ImportError, OSError) as error:
        raise RuntimeError("normalize/verify requires av, numpy and pyloudnorm") from error
    return av, np, pyln


def decode_planar(path: Path):
    av, np, _ = _imports_audio()
    container = av.open(str(path))
    try:
        stream = container.streams.audio[0]
        rate = int(stream.rate)
        layout = stream.layout.name
        reform = av.audio.resampler.AudioResampler(format="fltp", layout=layout, rate=rate)
        chunks = []
        for frame in container.decode(stream):
            converted = reform.resample(frame)
            if converted is None:
                continue
            if not isinstance(converted, list):
                converted = [converted]
            chunks.extend(item.to_ndarray().astype(np.float32, copy=False) for item in converted)
        flushed = reform.resample(None)
        if flushed is not None:
            if not isinstance(flushed, list):
                flushed = [flushed]
            chunks.extend(item.to_ndarray().astype(np.float32, copy=False) for item in flushed)
        if not chunks:
            raise RuntimeError(f"audio stream decoded no frames: {path}")
        value = np.concatenate(chunks, axis=1)
        if value.shape[0] != len(stream.layout.channels):
            raise RuntimeError(f"decoded channel layout mismatch: {path}")
        return value, rate, layout
    finally:
        container.close()


def write_float_wav(path: Path, planar, rate: int) -> None:
    _, np, _ = _imports_audio()
    try:
        import soundfile as sf
    except (ImportError, OSError) as error:
        raise RuntimeError("normalize requires soundfile") from error
    sf.write(
        str(path), np.ascontiguousarray(planar.T, dtype=np.float32), rate,
        format="WAV", subtype="FLOAT",
    )


def momentary_stats(planar, rate: int) -> dict[str, float | int]:
    _, np, pyln = _imports_audio()
    data = planar.T.astype(np.float64, copy=False)
    meter = pyln.Meter(rate, block_size=0.400, overlap=0.75)
    filtered = data.copy()
    for stage in meter._filters.values():
        for channel in range(filtered.shape[1]):
            filtered[:, channel] = stage.apply_filter(filtered[:, channel])
    block = int(0.400 * rate)
    hop = int(0.100 * rate)
    if len(filtered) < block:
        raise RuntimeError("audio shorter than 400 ms")
    starts = np.arange(0, len(filtered) - block + 1, hop, dtype=np.int64)
    ends = starts + block
    gains = np.ones(filtered.shape[1], dtype=np.float64)
    if filtered.shape[1] >= 4:
        gains[3 : min(5, filtered.shape[1])] = 1.41
    energy = np.sum(np.square(filtered) * gains[None, :], axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(energy, dtype=np.float64)))
    block_energy = (cumulative[ends] - cumulative[starts]) / block
    with np.errstate(divide="ignore", invalid="ignore"):
        values = -0.691 + 10.0 * np.log10(block_energy)
    finite = values[np.isfinite(values)]
    if not finite.size:
        raise RuntimeError("all momentary windows are silent")
    p10, p50, p90 = np.percentile(values, [10, 50, 90], method="linear")
    return {
        "window_count": int(values.size),
        "finite_window_count": int(finite.size),
        "silent_window_count": int(values.size - finite.size),
        "momentary_p10_lufs": float(p10),
        "momentary_p50_lufs": float(p50),
        "momentary_p90_lufs": float(p90),
        "momentary_min_lufs": float(np.min(finite)),
        "momentary_max_lufs": float(np.max(finite)),
    }


def gain_db(row: Mapping[str, Any], target: float) -> float:
    value = target - float(row["momentary_p50_lufs"])
    if not math.isfinite(value):
        raise ValueError(f"non-finite gain for {row['path']}")
    return value


def reusable_audio(record: Mapping[str, Any], source: Path, destination: Path, row: Mapping[str, Any], target: float) -> bool:
    return (
        record.get("schema") == SCHEMA
        and record.get("source_sha256") == row["sha256"]
        and float(record.get("target_p50_lufs", math.nan)) == target
        and float(record.get("source_p50_lufs", math.nan)) == float(row["momentary_p50_lufs"])
        and destination.is_file()
        and record.get("output_sha256") == sha256_file(destination)
        and source.is_file()
        and sha256_file(source) == row["sha256"]
    )


def normalize_one(
    source: Path, destination: Path, row: Mapping[str, Any], target: float, *, output_root: Path
) -> dict[str, Any]:
    actual_source_sha = sha256_file(source)
    if actual_source_sha != row["sha256"]:
        raise ValueError(f"source SHA-256 mismatch: {source}")
    waveform, rate, layout = decode_planar(source)
    if rate != int(row["sample_rate"]) or waveform.shape[0] != int(row["channels"]):
        raise ValueError(f"source audio metadata mismatch: {source}")
    adjustment = gain_db(row, target)
    multiplier = 10.0 ** (adjustment / 20.0)
    output_waveform = waveform * multiplier
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
    os.close(handle)
    temporary = Path(temp_name)
    try:
        write_float_wav(temporary, output_waveform, rate)
        decoded, output_rate, output_layout = decode_planar(temporary)
        if decoded.shape != waveform.shape or output_rate != rate:
            raise RuntimeError(
                "float WAV round-trip shape/rate mismatch: "
                f"source={waveform.shape}/{rate}/{layout}, "
                f"output={decoded.shape}/{output_rate}/{output_layout}: {destination}"
            )
        verify_waveform(waveform, decoded, multiplier)
        measured = momentary_stats(decoded, output_rate)
        for measured_name, source_name in (
            ("momentary_p10_lufs", "momentary_p10_lufs"),
            ("momentary_p50_lufs", "momentary_p50_lufs"),
            ("momentary_p90_lufs", "momentary_p90_lufs"),
        ):
            expected = float(row[source_name]) + adjustment
            if abs(float(measured[measured_name]) - expected) > 0.1:
                raise ValueError(
                    f"temporary WAV loudness mismatch for {destination}: "
                    f"{measured_name}={measured[measured_name]}, expected={expected}"
                )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "schema": SCHEMA,
        "relative_source_path": str(safe_relative_path(row["path"])).replace("\\", "/"),
        "asset_id": safe_relative_path(row["path"]).parent.name,
        "canonical_song_id": canonical_song_id(str(row["path"])),
        "source_sha256": actual_source_sha,
        "source_sample_rate": rate,
        "channels": int(waveform.shape[0]),
        "samples_per_channel": int(waveform.shape[1]),
        "source_p10_lufs": float(row["momentary_p10_lufs"]),
        "source_p50_lufs": float(row["momentary_p50_lufs"]),
        "source_p90_lufs": float(row["momentary_p90_lufs"]),
        "target_p50_lufs": target,
        "gain_db": adjustment,
        "amplitude_multiplier": multiplier,
        "source_sample_peak": float(abs(waveform).max()),
        "output_sample_peak": float(abs(decoded).max()),
        "output_momentary": measured,
        "output_relative_path": destination.relative_to(output_root).as_posix(),
        "output_sha256": sha256_file(destination),
        "output_format": "wav/pcm_f32le",
    }


def command_normalize(args: argparse.Namespace) -> None:
    rows = read_loudness_csv(args.csv.resolve())
    if args.include_path:
        requested = {safe_relative_path(value).as_posix() for value in args.include_path}
        rows = [row for row in rows if safe_relative_path(row["path"]).as_posix() in requested]
        found = {safe_relative_path(row["path"]).as_posix() for row in rows}
        if found != requested:
            raise FileNotFoundError(f"requested CSV paths missing: {sorted(requested - found)}")
    output = args.output.resolve()
    output_paths = [output_audio_path(output, safe_relative_path(row["path"])) for row in rows]
    if len(output_paths) != len(set(output_paths)):
        raise ValueError("multiple CSV rows map to the same output WAV")
    identity = stage_identity("normalize", {
        "source_root": str(args.source_root.resolve()),
        "source_csv": str(args.csv.resolve()),
        "source_csv_sha256": sha256_file(args.csv.resolve()),
        "target_p50_lufs": args.target_p50,
        "included_paths": sorted(safe_relative_path(row["path"]).as_posix() for row in rows),
    })
    identity_path = output / "normalize-identity.json"
    resumed = identity_path.exists()
    if resumed:
        require_identity(identity_path, identity)
    manifest_path = output / "audio-manifest.jsonl"
    existing = {row["relative_source_path"]: row for row in load_jsonl(manifest_path)}
    usage = shutil.disk_usage(output if output.exists() else output.parent)
    if not resumed:
        required = args.minimum_free_gib * (1024 ** 3)
    else:
        remaining = [
            row for row in rows if safe_relative_path(row["path"]).as_posix() not in existing
        ]
        required = sum(
            float(row["duration_seconds"]) * int(row["sample_rate"]) * int(row["channels"]) * 4
            for row in remaining
        ) + 5 * (1024 ** 3)
    if usage.free < required:
        raise RuntimeError(
            f"insufficient free space: {usage.free / (1024 ** 3):.2f} GiB; "
            f"required={required / (1024 ** 3):.2f} GiB"
        )
    if not resumed:
        require_identity(identity_path, identity)
    records: dict[str, dict[str, Any]] = dict(existing)
    for index, row in enumerate(rows, 1):
        relative = safe_relative_path(row["path"])
        key = relative.as_posix()
        source = args.source_root.resolve() / relative
        destination = output_audio_path(output, relative)
        old = existing.get(key)
        if old and reusable_audio(old, source, destination, row, args.target_p50):
            records[key] = old
        elif old or destination.exists():
            raise ValueError(f"existing audio identity/hash mismatch; refusing overwrite: {destination}")
        else:
            records[key] = normalize_one(
                source, destination, row, args.target_p50, output_root=output
            )
        atomic_jsonl(manifest_path, (records[name] for name in sorted(records)))
        if index == 1 or index % 25 == 0 or index == len(rows):
            print(f"NORMALIZE {index}/{len(rows)}", flush=True)
    atomic_json(output / "normalization-summary.json", {
        "schema": SCHEMA, "source_root": str(args.source_root.resolve()),
        "source_csv": str(args.csv.resolve()), "source_csv_sha256": sha256_file(args.csv.resolve()),
        "target_p50_lufs": args.target_p50, "count": len(records),
        "audio_manifest_sha256": sha256_file(manifest_path),
    })


def selection_ids(path: Path) -> tuple[list[str], str]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    selection = payload.get("selection", {})
    values = [str(v) for v in selection.get("train_song_ids", [])] + [str(v) for v in selection.get("validation_song_ids", [])]
    if len(values) != len(set(values)) or not values:
        raise ValueError("selection is empty or contains duplicate song IDs")
    return values, str(selection.get("selection_sha256", ""))


def canonical_song_id(relative_source_path: str) -> str | None:
    parts = safe_relative_path(relative_source_path).parts
    if len(parts) == 4 and parts[0:2] == ("data", "extracted") and parts[3].lower() == "track.mp3":
        return parts[2]
    return None


def resolve_pilot_audio_records(
    output: Path, song_ids: Iterable[str], profile_manifest: Path
) -> dict[str, dict[str, Any]]:
    requested = set(song_ids)
    hashes: dict[str, str] = {}
    for row in load_jsonl(profile_manifest):
        song_id = str(row.get("song_id", ""))
        if song_id not in requested:
            continue
        audio = row.get("audio")
        digest = str(audio.get("file_sha256", "")) if isinstance(audio, Mapping) else ""
        if not digest:
            raise ValueError(f"profile audio SHA-256 missing for pilot song {song_id}")
        if song_id in hashes and hashes[song_id] != digest:
            raise ValueError(f"profile has conflicting audio identities for pilot song {song_id}")
        hashes[song_id] = digest
    missing_profile = sorted(requested - set(hashes))
    if missing_profile:
        raise FileNotFoundError(f"pilot songs missing from profile manifest: {missing_profile[:10]}")
    all_audio = load_jsonl(output / "audio-manifest.jsonl")
    resolved: dict[str, dict[str, Any]] = {}
    for song_id in sorted(requested):
        candidates = [row for row in all_audio if row.get("source_sha256") == hashes[song_id]]
        if not candidates:
            raise FileNotFoundError(
                f"normalized audio with profile source hash missing for pilot song {song_id}: {hashes[song_id]}"
            )
        candidates.sort(
            key=lambda row: (
                canonical_song_id(str(row["relative_source_path"])) != song_id,
                str(row["relative_source_path"]),
            )
        )
        resolved[song_id] = candidates[0]
    return resolved


def atomic_numpy(path: Path, value) -> None:
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(handle)
    temporary = Path(name)
    try:
        with temporary.open("wb") as stream:
            np.save(stream, value, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def command_features(args: argparse.Namespace) -> None:
    import numpy as np
    from maimai_dataset.audio import N_MELS, normalized_features

    output = args.output.resolve()
    ids, selection_sha = selection_ids(args.selection_manifest.resolve())
    if args.song_id:
        requested = list(dict.fromkeys(args.song_id))
        missing_selection = sorted(set(requested) - set(ids))
        if missing_selection:
            raise ValueError(f"requested song IDs absent from selection: {missing_selection}")
        ids = requested
    profile_manifest = args.profile_manifest.resolve()
    audio_records = resolve_pilot_audio_records(output, ids, profile_manifest)
    stats_path = args.stats.resolve()
    stats = json.loads(stats_path.read_text(encoding="utf-8-sig"))
    stats_sha = sha256_file(stats_path)
    identity = stage_identity("features", {
        "selection_manifest": str(args.selection_manifest.resolve()),
        "selection_manifest_sha256": sha256_file(args.selection_manifest.resolve()),
        "selection_sha256": selection_sha,
        "song_ids": ids,
        "profile_manifest": str(profile_manifest),
        "profile_manifest_sha256": sha256_file(profile_manifest),
        "resolved_audio_sources": {
            song_id: {
                "relative_source_path": audio_records[song_id]["relative_source_path"],
                "source_sha256": audio_records[song_id]["source_sha256"],
            }
            for song_id in ids
        },
        "stats_source": str(stats_path), "stats_sha256": stats_sha,
        "old_features": str(args.old_features.resolve()),
        "audio_manifest_sha256": sha256_file(output / "audio-manifest.jsonl"),
        "audio_source_sha256": sha256_file(Path(__file__).resolve().parents[1] / "src/maimai_dataset/audio.py"),
    })
    require_identity(output / "features-identity.json", identity)
    copied_stats = output / "pilot" / "feature-stats.json"
    if copied_stats.exists() and copied_stats.read_bytes() != stats_path.read_bytes():
        raise ValueError("existing copied feature-stats.json differs from requested stats")
    if not copied_stats.exists():
        copied_stats.parent.mkdir(parents=True, exist_ok=True)
        temporary = copied_stats.with_name(copied_stats.name + ".tmp")
        shutil.copyfile(stats_path, temporary)
        os.replace(temporary, copied_stats)
    manifest_path = output / "feature-manifest.jsonl"
    existing = {row["song_id"]: row for row in load_jsonl(manifest_path)}
    records: dict[str, dict[str, Any]] = dict(existing)
    audio_source_sha = sha256_file(Path(__file__).resolve().parents[1] / "src/maimai_dataset/audio.py")
    for index, song_id in enumerate(ids, 1):
        audio_record = audio_records[song_id]
        wav = output / str(audio_record["output_relative_path"])
        if not wav.is_file() or sha256_file(wav) != audio_record["output_sha256"]:
            raise ValueError(f"normalized WAV missing or corrupt: {song_id}")
        destination = output / "pilot" / "features" / f"{song_id}.npy"
        old_feature = args.old_features.resolve() / f"{song_id}.npy"
        if not old_feature.is_file():
            raise FileNotFoundError(f"old pilot feature missing: {old_feature}")
        expected = existing.get(song_id)
        if expected and destination.is_file() and expected.get("wav_sha256") == sha256_file(wav) and expected.get("stats_sha256") == stats_sha and expected.get("audio_source_sha256") == audio_source_sha and expected.get("output_sha256") == sha256_file(destination):
            records[song_id] = expected
        elif expected or destination.exists():
            raise ValueError(f"existing feature identity/hash mismatch; refusing overwrite: {destination}")
        else:
            feature, info = normalized_features(wav, feature_stats=stats, relative_path=audio_record["output_relative_path"])
            feature = np.asarray(feature, dtype=np.float32)
            if not np.isfinite(feature).all():
                raise ValueError(f"new feature contains non-finite values: {song_id}")
            old_shape = list(np.load(old_feature, mmap_mode="r", allow_pickle=False).shape)
            if list(feature.shape) != old_shape or feature.shape[0] != N_MELS:
                raise ValueError(f"feature frame/shape drift for {song_id}: new={list(feature.shape)}, old={old_shape}")
            atomic_numpy(destination, feature)
            records[song_id] = {
                "schema": FEATURE_SCHEMA, "song_id": song_id,
                "wav_relative_path": str(audio_record["output_relative_path"]),
                "wav_sha256": sha256_file(wav), "stats_sha256": stats_sha,
                "resolved_source_path": audio_record["relative_source_path"],
                "resolved_source_sha256": audio_record["source_sha256"],
                "stats_policy": "copied-byte-for-byte-fixed-affine-transform",
                "audio_source_sha256": audio_source_sha,
                "shape": list(feature.shape), "dtype": str(feature.dtype),
                "old_shape": old_shape,
                "old_dtype": str(np.load(old_feature, mmap_mode="r", allow_pickle=False).dtype),
                "old_feature_sha256": sha256_file(old_feature),
                "output_relative_path": destination.relative_to(output).as_posix(),
                "output_sha256": sha256_file(destination), "pcm_sha256": info.pcm_sha256,
            }
        atomic_jsonl(manifest_path, (records[name] for name in sorted(records)))
        if index == 1 or index % 25 == 0 or index == len(ids):
            print(f"FEATURES {index}/{len(ids)}", flush=True)
    atomic_json(output / "feature-summary.json", {
        "schema": FEATURE_SCHEMA, "selection_manifest": str(args.selection_manifest.resolve()),
        "selection_manifest_sha256": sha256_file(args.selection_manifest.resolve()),
        "selection_sha256": selection_sha, "song_count": len(ids),
        "stats_source": str(stats_path), "stats_sha256": stats_sha,
        "audio_source_sha256": audio_source_sha,
        "feature_manifest_sha256": sha256_file(manifest_path),
    })


def verify_waveform(source, output, multiplier: float) -> tuple[float, float]:
    import numpy as np
    if source.shape != output.shape:
        raise ValueError("waveform shape changed")
    expected = source.astype(np.float64) * multiplier
    error = float(np.max(np.abs(output.astype(np.float64) - expected)))
    tolerance = max(2e-7, float(np.max(np.abs(expected))) * 2e-7)
    if error > tolerance:
        raise ValueError(f"waveform is not a fixed-gain copy: error={error}, tolerance={tolerance}")
    return error, tolerance


def command_verify(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    records = load_jsonl(output / "audio-manifest.jsonl")
    if len(records) != args.expected_audio_count:
        raise ValueError(f"audio count mismatch: {len(records)} != {args.expected_audio_count}")
    verifications = []
    for index, record in enumerate(records, 1):
        relative = safe_relative_path(record["relative_source_path"])
        source = args.source_root.resolve() / relative
        wav = output / str(record["output_relative_path"])
        if sha256_file(source) != record["source_sha256"] or sha256_file(wav) != record["output_sha256"]:
            raise ValueError(f"hash mismatch during verify: {relative}")
        source_pcm, source_rate, source_layout = decode_planar(source)
        output_pcm, output_rate, output_layout = decode_planar(wav)
        if source_rate != output_rate or source_pcm.shape[0] != output_pcm.shape[0]:
            raise ValueError(
                f"rate/channel count changed: {relative}; "
                f"source={source_rate}/{source_pcm.shape[0]}/{source_layout}, "
                f"output={output_rate}/{output_pcm.shape[0]}/{output_layout}"
            )
        error, tolerance = verify_waveform(source_pcm, output_pcm, float(record["amplitude_multiplier"]))
        measured = momentary_stats(output_pcm, output_rate)
        shift = float(record["gain_db"])
        for name, old_name in (("momentary_p10_lufs", "source_p10_lufs"), ("momentary_p50_lufs", "source_p50_lufs"), ("momentary_p90_lufs", "source_p90_lufs")):
            expected = float(record[old_name]) + shift
            if abs(float(measured[name]) - expected) > args.lufs_tolerance:
                raise ValueError(f"loudness shift mismatch for {relative} {name}: measured={measured[name]}, expected={expected}")
        if abs(float(measured["momentary_p50_lufs"]) - float(record["target_p50_lufs"])) > args.lufs_tolerance:
            raise ValueError(f"P50 target mismatch: {relative}")
        verifications.append({"relative_source_path": relative.as_posix(), **measured, "waveform_max_abs_error": error, "waveform_tolerance": tolerance})
        if index == 1 or index % 25 == 0 or index == len(records):
            print(f"VERIFY {index}/{len(records)}", flush=True)
    ids, selection_sha = selection_ids(args.selection_manifest.resolve())
    if args.song_id:
        requested = list(dict.fromkeys(args.song_id))
        if not set(requested).issubset(ids):
            raise ValueError("verify song IDs are absent from pilot selection")
        ids = requested
    resolved_audio = resolve_pilot_audio_records(
        output, ids, args.profile_manifest.resolve()
    )
    feature_records = {row["song_id"]: row for row in load_jsonl(output / "feature-manifest.jsonl")}
    if set(feature_records) != set(ids):
        raise ValueError("feature manifest does not exactly match pilot selection")
    for song_id, record in feature_records.items():
        expected_audio = resolved_audio[song_id]
        if (
            record.get("resolved_source_path") != expected_audio["relative_source_path"]
            or record.get("resolved_source_sha256") != expected_audio["source_sha256"]
        ):
            raise ValueError(f"feature/profile audio source mismatch: {song_id}")
        path = output / record["output_relative_path"]
        if not path.is_file() or sha256_file(path) != record["output_sha256"]:
            raise ValueError(f"feature missing or corrupt: {song_id}")
        import numpy as np
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        old = np.load(args.old_features.resolve() / f"{song_id}.npy", mmap_mode="r", allow_pickle=False)
        if value.dtype != np.float32 or value.shape != old.shape or value.shape[0] != 128:
            raise ValueError(f"feature dtype/shape mismatch: {song_id}")
        if not np.isfinite(value).all():
            raise ValueError(f"feature contains non-finite values: {song_id}")
    actual_feature_files = {
        path.stem for path in (output / "pilot" / "features").glob("*.npy")
    }
    if actual_feature_files != set(ids):
        raise ValueError("feature directory does not exactly match pilot selection")
    identity = stage_identity("verify", {
        "source_root": str(args.source_root.resolve()), "audio_count": len(records),
        "selection_manifest": str(args.selection_manifest.resolve()),
        "selection_manifest_sha256": sha256_file(args.selection_manifest.resolve()),
        "profile_manifest": str(args.profile_manifest.resolve()),
        "profile_manifest_sha256": sha256_file(args.profile_manifest.resolve()),
        "song_ids": ids, "old_features": str(args.old_features.resolve()),
        "audio_manifest_sha256": sha256_file(output / "audio-manifest.jsonl"),
        "feature_manifest_sha256": sha256_file(output / "feature-manifest.jsonl"),
        "lufs_tolerance": args.lufs_tolerance,
    })
    stats_sha = sha256_file(args.stats.resolve())
    copied_stats = output / "pilot" / "feature-stats.json"
    if not copied_stats.is_file() or sha256_file(copied_stats) != stats_sha:
        raise ValueError("copied feature stats are missing or differ from source stats")
    for song_id, record in feature_records.items():
        old = args.old_features.resolve() / f"{song_id}.npy"
        if sha256_file(old) != record["old_feature_sha256"]:
            raise ValueError(f"old feature identity changed: {song_id}")
    require_identity(output / "verify-identity.json", identity)
    atomic_jsonl(output / "verification.jsonl", verifications)
    atomic_json(output / "verification-summary.json", {
        "schema": "maimai.audio_fixed_gain_verification.v1", "audio_count": len(records),
        "feature_count": len(feature_records), "selection_sha256": selection_sha,
        "lufs_tolerance": args.lufs_tolerance,
        "verification_manifest_sha256": sha256_file(output / "verification.jsonl"),
    })


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Build and verify fixed-gain P50-normalized audio and pilot features")
    commands = value.add_subparsers(dest="command", required=True)
    normal = commands.add_parser("normalize")
    normal.add_argument("--source-root", type=Path, required=True)
    normal.add_argument("--csv", type=Path, required=True)
    normal.add_argument("--output", type=Path, required=True)
    normal.add_argument("--target-p50", type=float, default=TARGET_DEFAULT)
    normal.add_argument("--include-path", action="append")
    normal.add_argument("--minimum-free-gib", type=float, default=100.0)
    normal.set_defaults(function=command_normalize)
    features = commands.add_parser("features")
    features.add_argument("--output", type=Path, required=True)
    features.add_argument("--selection-manifest", type=Path, required=True)
    features.add_argument("--profile-manifest", type=Path, required=True)
    features.add_argument("--old-features", type=Path, required=True)
    features.add_argument("--stats", type=Path, required=True)
    features.add_argument("--song-id", action="append")
    features.set_defaults(function=command_features)
    verify = commands.add_parser("verify")
    verify.add_argument("--source-root", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--selection-manifest", type=Path, required=True)
    verify.add_argument("--profile-manifest", type=Path, required=True)
    verify.add_argument("--old-features", type=Path, required=True)
    verify.add_argument("--stats", type=Path, required=True)
    verify.add_argument("--expected-audio-count", type=int, default=1813)
    verify.add_argument("--lufs-tolerance", type=float, default=0.1)
    verify.add_argument("--song-id", action="append")
    verify.set_defaults(function=command_verify)
    return value


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
