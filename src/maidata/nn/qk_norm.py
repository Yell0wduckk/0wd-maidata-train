"""Parameter-free, per-head Q/K scale control before positional rotation."""
import torch


QK_NORM_EPS = 1e-6


def validate_qk_normalization(enabled):
    if type(enabled) is not bool:
        raise ValueError('qk_normalization must be boolean')
    return enabled


def normalize_qk(value, enabled):
    """Normalize the final head dimension without changing legacy math when off.

    Q/K projections call this once, before RoPE or caching. RMS accumulation and
    scaling use FP32 even under autocast; no affine gain or temperature is added.
    """
    if not enabled:
        return value
    with torch.autocast(device_type=value.device.type, enabled=False):
        floating = value.float()
        normalized = floating * torch.rsqrt(floating.square().mean(-1, keepdim=True) + QK_NORM_EPS)
    return normalized.to(value.dtype)
