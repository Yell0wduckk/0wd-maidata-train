"""Whole-song audio policy and preprocessing calculations."""

from __future__ import annotations
from maimai_dataset.audio import decode_audio, sha256_file
from pathlib import Path
from typing import Any, Mapping
import hashlib
import json
import math
import numpy as np
import tempfile
import torch


TARGET_P50 = -8.092632368636558


TRAINING_POLICIES = ("legacy", "normalized-p50")


POLICIES = TRAINING_POLICIES


SCHEMA = "maimai.audio_input.v1"


def json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def policy_contract(policy: str) -> dict[str, Any]:
    if policy not in POLICIES:
        raise ValueError(f"unknown audio policy: {policy}")
    return {
        "schema": SCHEMA, "policy": policy,
        "target_p50_lufs": None if policy == "legacy" else TARGET_P50,
        "momentary": {"window_ms": 400, "hop_ms": 100, "percentile": "linear-all-windows", "weighting": "K"},
        "level_gain": "zero",
        "feature_transform": "float64-expm1-power-gain-log1p-fixed-stats-to-float32-v1",
        "resampling": "native" if policy == "legacy" else "linear-float32-half-pixel-separate-multiply-subtract-v1",
    }


def training_audio_policy(policy: str) -> str:
    """Training and inference consume level-independent audio."""
    if policy not in TRAINING_POLICIES:
        raise ValueError("training audio policy must be legacy or normalized-p50; level gain is removed")
    return policy


def exact_level(value: Any) -> float:
    if isinstance(value, bool) or value is None or "+" in str(value):
        raise ValueError(f"audio policy requires an exact decimal level: {value!r}")
    try:
        level = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid decimal level: {value!r}") from error
    if not math.isfinite(level) or not 1 <= level <= 15:
        raise ValueError(f"audio policy level must be finite and in [1,15]: {value!r}")
    return level


def stats_arrays(stats: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    mean, std = (np.asarray(stats.get(k, []), dtype=np.float64) for k in ("mean", "std"))
    if mean.shape != (128,) or std.shape != (128,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("audio statistics require 128 finite means and positive standard deviations")
    # Existing features were standardized using float32 statistics.
    return mean.astype(np.float32).astype(np.float64), std.astype(np.float32).astype(np.float64)



def momentary_stats(planar: np.ndarray, rate: int) -> dict[str, Any]:
    """The accepted CSV meter: all 400ms windows, every 100ms, including silence."""
    import pyloudnorm as pyln
    data = np.asarray(planar, dtype=np.float64).T.copy()
    if data.ndim != 2 or data.shape[1] not in (1, 2) or not np.isfinite(data).all():
        raise ValueError("loudness preparation requires finite mono or stereo audio")
    meter = pyln.Meter(rate, block_size=0.400, overlap=0.75)
    for stage in meter._filters.values():
        for channel in range(data.shape[1]):
            data[:, channel] = stage.apply_filter(data[:, channel])
    block, hop = int(0.4 * rate), int(0.1 * rate)
    if len(data) < block or hop <= 0:
        raise ValueError("audio is shorter than 400 ms")
    starts = np.arange(0, len(data) - block + 1, hop, dtype=np.int64)
    energy = np.sum(np.square(data), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(energy, dtype=np.float64)))
    with np.errstate(divide="ignore", invalid="ignore"):
        loudness = -0.691 + 10 * np.log10((cumulative[starts + block] - cumulative[starts]) / block)
        quantiles = np.percentile(loudness, [10, 50, 90], method="linear")
    if not math.isfinite(float(quantiles[1])):
        raise ValueError("momentary median is non-finite; cannot normalize this audio")
    return {"p50_lufs": float(quantiles[1]), "window_count": int(len(loudness)),
            "silent_window_count": int(np.count_nonzero(~np.isfinite(loudness)))}


def decode_source(path: Path) -> tuple[np.ndarray, int]:
    import av
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        rate = int(stream.codec_context.sample_rate)
        if len(stream.layout.channels) not in (1, 2):
            raise ValueError("only mono/stereo source audio is supported")
        resampler = av.AudioResampler(format="fltp", layout=stream.layout.name, rate=rate)
        chunks = []
        for frame in container.decode(stream):
            chunks.extend(f.to_ndarray() for f in resampler.resample(frame))
        chunks.extend(f.to_ndarray() for f in resampler.resample(None))
    if not chunks:
        raise ValueError("source audio decoded no samples")
    return np.concatenate(chunks, axis=1).astype(np.float32), rate


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def checkpoint_audio_input(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    value = payload.get("audio_input")
    saved = payload.get("baseline_manifest", {}).get("audio_input")
    if value != saved:
        raise ValueError("checkpoint audio input differs from resume manifest")
    if value is None:
        return None
    if not isinstance(value, dict) or value.get("contract") != policy_contract(value.get("policy")):
        raise ValueError("checkpoint audio input contract mismatch")
    if value["policy"] == "legacy" or not isinstance(value.get("artifacts"), dict) or not value["artifacts"]:
        raise ValueError("new checkpoint audio policy requires normalized artifact identity")
    stats = payload.get("feature_stats")
    if not isinstance(stats, dict) or json_hash(stats) != value.get("feature_stats_sha256"):
        raise ValueError("checkpoint audio feature statistics mismatch")
    stats_arrays(stats)
    return value


def extract_at_centers(waveform: torch.Tensor, coordinates: Mapping[str, np.ndarray],
                       stats: Mapping[str, Any] | None = None) -> np.ndarray:
    """Independent centered FFTs; no interpolation of old Mel features."""
    from maimai_dataset.audio import mel_filterbank
    mono = waveform.detach().cpu().to(torch.float32).reshape(-1)
    if mono.numel() < 2 or not bool(torch.isfinite(mono).all()):
        raise ValueError("invalid PCM for BPM extraction")
    centers = np.asarray(coordinates["samples"], dtype=np.int64)
    valid = np.asarray(coordinates["valid"], dtype=bool)
    result = np.zeros((128, len(centers)), dtype=np.float32)
    bank, hann = mel_filterbank(), torch.hann_window(1024)
    offsets = torch.arange(-512, 512)
    period = 2 * (mono.numel() - 1)
    for start in range(0, len(centers), 512):
        stop = min(start + 512, len(centers))
        indices = torch.from_numpy(centers[start:stop])[:, None] + offsets
        # Repeated reflection also defines the tiny-audio case unambiguously.
        reflected = indices.remainder(period)
        reflected = torch.minimum(reflected, period - reflected)
        power = torch.fft.rfft(mono[reflected] * hann, dim=-1).abs().square().T
        result[:, start:stop] = torch.log1p(bank @ power).numpy()
    if stats is not None:
        mean, std = validate_stats(stats)
        result = ((result - mean[:, None]) / std[:, None]).astype(np.float32)
    result[:, ~valid] = 0
    if not np.isfinite(result).all():
        raise ValueError("non-finite BPM features")
    return result


def validate_stats(stats):
    from maimai_dataset.audio import TARGET_SAMPLE_RATE, N_FFT, N_MELS, F_MIN, F_MAX
    for name, value in {"sample_rate": TARGET_SAMPLE_RATE, "n_fft": N_FFT,
                        "n_mels": N_MELS, "f_min": F_MIN, "f_max": F_MAX}.items():
        if stats.get(name) != value:
            raise ValueError(f"BPM statistics definition mismatch: {name}")
    mean, std = (np.asarray(stats.get(key, []), dtype=np.float32) for key in ("mean", "std"))
    if mean.shape != (128,) or std.shape != (128,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("invalid BPM feature mean/std")
    return mean, std


def prepare_waveform(path, *, policy, already_normalized=False):
    path = Path(path)
    source_hash = sha256_file(path)
    info = {"source_sha256": source_hash, "source_p50_lufs": None, "normalization_gain_db": 0.0}
    if policy == "legacy" or already_normalized:
        wave, rate = decode_audio(path, stable_linear=policy != "legacy")
        info["source_sample_rate"] = rate
    else:
        import soundfile as sf
        planar, rate = decode_source(path)
        meter = momentary_stats(planar, rate)
        gain = TARGET_P50 - meter["p50_lufs"]
        scaled = (planar * 10 ** (gain / 20)).astype(np.float32)
        if not np.isfinite(scaled).all():
            raise ValueError("non-finite normalized PCM")
        with tempfile.TemporaryDirectory(prefix="maimai-bpm96-") as temporary:
            wav = Path(temporary) / "track.wav"
            sf.write(str(wav), scaled.T, rate, subtype="FLOAT", format="WAV")
            wave, _ = decode_audio(wav, stable_linear=True)
        info.update(source_sample_rate=rate, source_p50_lufs=meter["p50_lufs"], normalization_gain_db=gain,
                    normalized_peak=float(np.abs(scaled).max()), source_samples_per_channel=planar.shape[1])
    if sha256_file(path) != source_hash:
        raise ValueError("source changed while decoding BPM audio")
    info.update(num_samples=wave.shape[1], sample_rate=24000, duration_seconds=wave.shape[1] / 24000)
    return wave, info
