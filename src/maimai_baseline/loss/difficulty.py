"""Difficulty-weighted loss presets and reductions."""

from __future__ import annotations
from collections.abc import Mapping
import torch


PRESETS = {
    "uniform": (1., 1., 1., 1.),
    "conservative": (1., 1.1, 1.25, 1.5),
    "standard": (1., 1.2, 1.5, 2.),
    "aggressive": (1., 1.5, 2., 3.),
}


def contract(preset="uniform"):
    if preset not in PRESETS:
        raise ValueError(f"unknown difficulty loss preset: {preset!r}")
    return {"schema": "maimai.difficulty_loss_weighting.v1", "preset": preset,
            "weights": dict(zip(("basic", "advanced", "expert", "master"), PRESETS[preset])),
            "source_mapping": {"2": "basic", "3": "advanced", "4": "expert", "5": "master", "6": "master"},
            "reduction": "per-existing-group-weighted-mean-v1",
            "scope": "l3-and-physical", "uniform": "frozen-v9-reduction"}


def validate_identity(value):
    if not isinstance(value, Mapping) or dict(value) != contract(value.get("preset")):
        raise ValueError("difficulty loss weighting identity mismatch")
    return value["preset"]


def validate_checkpoint(payload, *, required=False):
    value = payload.get("difficulty_loss_weighting")
    manifest = payload.get("baseline_manifest", {})
    if value is None:
        if required or manifest.get("difficulty_loss_weighting") is not None:
            raise ValueError("checkpoint lacks difficulty loss weighting")
        return "uniform"  # frozen pre-experiment checkpoints only
    preset = validate_identity(value)
    if manifest.get("difficulty_loss_weighting") != value:
        raise ValueError("checkpoint difficulty loss weighting differs from manifest")
    config = payload.get("training_config")
    if config is not None and config.get("difficulty_loss_preset", "uniform") != preset:
        raise ValueError("checkpoint difficulty loss preset differs from training config")
    return preset


def validate_records(records):
    for row in records:
        source = row.get("source_difficulty_id", row.get("difficulty_id"))
        if source not in (2, 3, 4, 5, 6):
            raise ValueError(f"unknown source difficulty {source!r}: {row.get('segment_id', row.get('song_id'))}")


def weights(batch, reference):
    preset = batch.get("difficulty_loss_preset", "uniform")
    contract(preset)
    if "difficulty_loss_preset" not in batch:
        return None, None
    source = batch.get("source_difficulty_id", batch.get("difficulty_id"))
    if source is None:
        if preset != "uniform":
            raise ValueError("difficulty loss requires source difficulty IDs")
        return None, None
    source = source.to(device=reference.device, dtype=torch.long)
    if source.ndim != 1 or source.shape[0] != reference.shape[0] or bool(((source < 2) | (source > 6)).any()):
        raise ValueError("invalid difficulty loss row mapping")
    ids = source.clamp_max(5)
    return reference.new_tensor(PRESETS[preset])[ids - 2], ids


def mean(values, active_weight):
    denominator = active_weight.sum(dim=0)
    safe = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    return (values.masked_fill(active_weight.eq(0), 0) * active_weight).sum(dim=0) / safe


def diagnostics(values, mask, base_weights, difficulty_weights, ids):
    """Detached sufficient statistics; no probability recomputation or device sync."""
    values = values.detach().masked_fill(~mask, 0)
    active = base_weights * mask
    weighted = active * difficulty_weights
    denominator = weighted.sum()
    safe = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    result = {}
    for name, select in [("all", torch.ones_like(mask)), *[(str(d), ids.eq(d)) for d in (2,3,4,5)]]:
        a = active * select
        w = weighted * select
        result[name] = {"count": (mask & select).sum(), "base_weight_sum": a.sum(),
                        "weight_sum": w.sum(), "loss_sum": (values * w).sum(),
                        "unweighted_loss_sum": (values * a).sum(),
                        "contribution": (values * w).sum() / safe}
    return result
