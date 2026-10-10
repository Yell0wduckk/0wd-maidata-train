from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping


TARGET_SAMPLE_RATE = 24_000
N_FFT = 1_024
HOP_LENGTH = 240
N_MELS = 128
F_MIN = 30.0
F_MAX = 12_000.0


class AudioDependencyError(RuntimeError):
    pass


_DLL_HANDLES: list[Any] = []


def _configure_windows_ffmpeg() -> None:
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return
    if _DLL_HANDLES:
        return
    candidates = []
    explicit = os.environ.get("MAIMAI_FFMPEG_DLL_DIR")
    if explicit:
        candidates.append(Path(explicit))
    candidates.append(
        Path(__file__).resolve().parents[2]
        / "tools" / "MiaCode" / "third_party" / "ffmpeg" / "windows" / "dev" / "bin"
    )
    for candidate in candidates:
        if candidate.is_dir():
            handle = os.add_dll_directory(str(candidate))
            _DLL_HANDLES.append(handle)
            os.environ["PATH"] = str(candidate) + os.pathsep + os.environ.get("PATH", "")
            return


@dataclass(frozen=True)
class AudioInfo:
    relative_path: str
    file_sha256: str
    pcm_sha256: str
    source_sample_rate: int
    sample_rate: int
    num_samples: int
    duration_seconds: float
    feature_frames: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _imports(*, require_decoder: bool = True):
    _configure_windows_ffmpeg()
    try:
        import numpy as np
        import torch
        import torch.nn.functional as functional
    except (ImportError, OSError) as error:
        raise AudioDependencyError(
            "audio preparation requires the 'train' extra and a PyTorch installation"
        ) from error
    AudioDecoder = None
    if require_decoder:
        try:
            from torchcodec.decoders import AudioDecoder as _AudioDecoder
        except (ImportError, OSError) as error:
            raise AudioDependencyError(
                "audio decoding requires the 'train' extra and a TorchCodec-compatible FFmpeg installation"
            ) from error
        AudioDecoder = _AudioDecoder
    return np, torch, functional, AudioDecoder


def sha256_file(path: Path) -> str:
    from maimai_dataset.integrity import file_sha256
    return file_sha256(path)


def resample_linear_float32(waveform, output_samples: int):
    """Match the accepted float32 half-pixel coordinates without compiler FMA.

    A fused scale * position - 0.5 changes coordinates on long recordings.
    Separate NumPy operations preserve the Windows pilot feature convention on
    Linux too. Chunking bounds temporary memory without resetting coordinates.
    """
    np, torch, _, _ = _imports(require_decoder=False)
    source = waveform.detach().cpu().numpy()
    if source.dtype != np.float32 or source.ndim != 2 or source.shape[0] != 1 or source.shape[1] == 0 or output_samples <= 0:
        raise ValueError("linear resampling requires nonempty mono float32 audio and positive output size")
    size = source.shape[1]
    if output_samples == size:
        return waveform
    scale = np.float32(size / output_samples)
    output = np.empty((1, output_samples), dtype=np.float32)
    for start in range(0, output_samples, 262144):
        stop = min(start + 262144, output_samples)
        # Convert integer positions separately: float arange can drift when
        # its step is smaller than the float32 spacing on very long inputs.
        positions = np.arange(start, stop, dtype=np.int64).astype(np.float32)
        positions += np.float32(0.5)
        positions *= scale
        positions -= np.float32(0.5)
        np.maximum(positions, np.float32(0), out=positions)
        lower = np.minimum(np.floor(positions).astype(np.int64), size - 1)
        weight = np.clip(positions - lower, 0, 1).astype(np.float32)
        output[0, start:stop] = (np.float32(1) - weight) * source[0, lower] + weight * source[0, np.minimum(lower + 1, size - 1)]
    return torch.from_numpy(output)


def decode_audio(path: Path, *, target_sample_rate: int = TARGET_SAMPLE_RATE, stable_linear: bool = False):
    _, torch, functional, AudioDecoder = _imports()
    decoder = AudioDecoder(str(path))
    samples = decoder.get_all_samples()
    waveform = samples.data.detach().to(device="cpu", dtype=torch.float32)
    source_rate = int(getattr(samples, "sample_rate", 0) or getattr(decoder.metadata, "sample_rate", 0))
    if source_rate <= 0:
        raise RuntimeError(f"TorchCodec did not report a sample rate for {path}")
    if waveform.ndim != 2 or waveform.shape[1] == 0:
        raise RuntimeError(f"decoded audio is empty or malformed: {path}")
    waveform = waveform.mean(dim=0, keepdim=True)
    if source_rate != target_sample_rate:
        output_samples = max(1, round(waveform.shape[1] * target_sample_rate / source_rate))
        if stable_linear:
            waveform = resample_linear_float32(waveform, output_samples)
        else:
            waveform = functional.interpolate(
                waveform.unsqueeze(0), size=output_samples, mode="linear", align_corners=False
            ).squeeze(0)
    return waveform.contiguous(), source_rate


def pcm_sha256(waveform) -> str:
    np, torch, _, _ = _imports(require_decoder=False)
    quantized = (
        waveform.detach().to(device="cpu", dtype=torch.float32)
        .clamp(-1.0, 1.0)
        .mul(32767.0)
        .round()
        .to(dtype=torch.int16)
        .numpy()
    )
    return hashlib.sha256(np.ascontiguousarray(quantized).tobytes()).hexdigest()


def _hz_to_mel(value):
    _, torch, _, _ = _imports(require_decoder=False)
    return 2595.0 * torch.log10(1.0 + value / 700.0)


def _mel_to_hz(value):
    _, torch, _, _ = _imports(require_decoder=False)
    return 700.0 * (torch.pow(10.0, value / 2595.0) - 1.0)


def mel_filterbank(
    *,
    sample_rate: int = TARGET_SAMPLE_RATE,
    n_fft: int = N_FFT,
    n_mels: int = N_MELS,
    f_min: float = F_MIN,
    f_max: float = F_MAX,
):
    _, torch, _, _ = _imports(require_decoder=False)
    frequencies = torch.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1, dtype=torch.float32)
    mel_min = _hz_to_mel(torch.tensor(float(f_min)))
    mel_max = _hz_to_mel(torch.tensor(float(f_max)))
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    lower = hz_points[:-2, None]
    center = hz_points[1:-1, None]
    upper = hz_points[2:, None]
    rising = (frequencies[None, :] - lower) / (center - lower).clamp_min(1e-12)
    falling = (upper - frequencies[None, :]) / (upper - center).clamp_min(1e-12)
    return torch.minimum(rising, falling).clamp_min(0.0)


def log_mel(waveform, *, sample_rate: int = TARGET_SAMPLE_RATE):
    _, torch, _, _ = _imports(require_decoder=False)
    mono = waveform.squeeze(0)
    window = torch.hann_window(N_FFT, dtype=torch.float32)
    spectrum = torch.stft(
        mono,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window=window,
        center=True,
        return_complex=True,
    )
    power = spectrum.abs().square()
    mel = mel_filterbank(sample_rate=sample_rate).matmul(power)
    return torch.log1p(mel).contiguous()


def extract_audio(path: Path, *, relative_path: str, stable_linear: bool = False) -> tuple[Any, AudioInfo]:
    waveform, source_rate = decode_audio(path, stable_linear=stable_linear)
    features = log_mel(waveform)
    info = AudioInfo(
        relative_path=relative_path,
        file_sha256=sha256_file(path),
        pcm_sha256=pcm_sha256(waveform),
        source_sample_rate=source_rate,
        sample_rate=TARGET_SAMPLE_RATE,
        num_samples=int(waveform.shape[1]),
        duration_seconds=float(waveform.shape[1] / TARGET_SAMPLE_RATE),
        feature_frames=int(features.shape[1]),
    )
    if not math.isfinite(info.duration_seconds) or info.duration_seconds <= 0:
        raise RuntimeError(f"invalid decoded duration for {path}")
    return features, info


def normalized_features(
    path: Path,
    *,
    feature_stats: Mapping[str, Any],
    relative_path: str | None = None,
    stable_linear: bool = False,
) -> tuple[Any, AudioInfo]:
    """Decode an inference asset with the exact preprocessing stored in a checkpoint."""
    np, _, _, _ = _imports(require_decoder=False)
    expected = {
        "sample_rate": TARGET_SAMPLE_RATE,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "n_mels": N_MELS,
        "f_min": F_MIN,
        "f_max": F_MAX,
    }
    mismatches = [
        f"{key}={feature_stats.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if float(feature_stats.get(key, -1)) != float(value)
    ]
    if mismatches:
        raise ValueError("checkpoint audio feature definition is incompatible: " + "; ".join(mismatches))
    raw, info = extract_audio(path, relative_path=relative_path or path.name, stable_linear=stable_linear)
    mean = np.asarray(feature_stats.get("mean", []), dtype=np.float32)
    std = np.asarray(feature_stats.get("std", []), dtype=np.float32)
    if mean.shape != (N_MELS,) or std.shape != (N_MELS,) or np.any(std <= 0):
        raise ValueError("checkpoint feature statistics must contain 128 positive standard deviations")
    value = raw.numpy().astype(np.float32, copy=False)
    return ((value - mean[:, None]) / std[:, None]).astype(np.float32), info
