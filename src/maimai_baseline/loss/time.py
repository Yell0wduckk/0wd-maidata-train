"""Training adapters for MOMENT and duration distributions."""

from __future__ import annotations
from maimai_baseline.time.duration import FIELDS, distribution_name, interval_log_probability
from maimai_baseline.time.grid import DELTA_MS
from maimai_baseline.time.moment import LINEAR_MS, candidate_probabilities
import torch


def time_position_losses(outputs, batch, runtime_cache, *, collect_metrics=True):
    from maimai_baseline.time.duration import LINEAR_CONTRACT
    from maimai_baseline.time.moment import LINEAR_MS
    expected = {DELTA_MS: 1000.0, "hold_duration_tick": 96.0, "branch_duration_tick": 96.0}
    if outputs.get("time_unit_scales") != expected:
        raise ValueError("invalid v9 time scale contract")
    if outputs.get("duration_distribution") != LINEAR_CONTRACT or outputs.get("moment_distribution") != LINEAR_MS:
        raise ValueError("v9 loss requires linear MOMENT and duration distributions")
    losses, active_masks, informative, metrics = position_losses(outputs, batch, collect_metrics=collect_metrics)
    target = batch["time_targets"][DELTA_MS]
    active = target.ne(-100)
    selected = active.nonzero()
    row_map = batch["bpm96_row_index"]
    if not len(selected):
        raw = outputs["time_outputs"][DELTA_MS]
        if runtime_cache is not None:
            runtime_cache.update(delta_log_probability=raw.new_empty((0, 0)), delta_lower=raw.new_empty((0,)),
                                 delta_row_index=row_map, delta_absolute_seconds=batch["bpm96_candidates"]["seconds"],
                                 delta_absolute_ticks=batch["bpm96_candidates"]["ticks"])
        return losses, active_masks, informative, metrics
    b, r = selected.unbind(1)
    # CPU collation validates canonical active-row order; no dense candidate
    # allocation or extra GPU gather is needed here.
    table = batch["bpm96_candidates"]
    candidate = dict(outputs["delta_candidate"], query=outputs["delta_candidate"]["query"][b, r])
    distribution, detail = candidate_probabilities(outputs["time_outputs"][DELTA_MS][b, r], table, candidate, b,
                                                   distribution=outputs["moment_distribution"])
    index = batch["bpm96_target_index"][b, r]
    if bool((index < 0).any()) or bool((index >= distribution.shape[1]).any()):
        raise ValueError("invalid BPM target cell index")
    nll = -distribution.gather(1, index[:, None]).squeeze(1)
    loss = torch.zeros_like(target, dtype=torch.float32)
    loss[b, r] = nll
    losses[DELTA_MS], active_masks[DELTA_MS] = loss, active
    informative[DELTA_MS] = torch.zeros_like(active)
    informative[DELTA_MS][b, r] = table["valid"].sum(-1).gt(1)
    if collect_metrics:
        chosen = distribution.detach().argmax(-1)
        prediction = table["delta_ms"].gather(1, chosen[:, None]).squeeze(1)
        metrics[DELTA_MS] = {"loss": nll.mean().detach(), "accuracy": chosen.eq(index).float().mean(),
                             "count": float(len(b)), "median_absolute_error_ms": (prediction - target[b, r]).abs().median(),
                             "candidate_gate": detail["gate"].detach()}
    if runtime_cache is not None:
        runtime_cache.update(delta_log_probability=distribution, delta_lower=torch.zeros_like(b),
                             delta_row_index=row_map, delta_absolute_seconds=table["seconds"],
                             delta_absolute_ticks=table["ticks"])
    return losses, active_masks, informative, metrics


def position_losses(outputs, batch, *, collect_metrics=True):
    """Sparse L3 duration objectives. No duration modal search for diagnostics."""
    losses, active_masks, informative, metrics = {}, {}, {}, {}
    for field in FIELDS:
        target = batch["time_targets"][field]
        active = target.ne(-100)
        selected = active.nonzero()
        if not len(selected):
            continue
        prepared = batch.get("duration_bounds", {}).get(field)
        if prepared is None or prepared["rows"].shape != selected.shape:
            raise ValueError("duration bounds missing or active-row mapping mismatch")
        if not torch.equal(prepared["rows"], selected):
            raise ValueError("duration bounds row mapping mismatch")
        b, r = selected.unbind(1)
        raw = outputs["time_outputs"][field][b, r]
        probability = interval_log_probability(raw, prepared["bounds"], field=field,
            distribution=distribution_name(outputs["duration_distribution"]))
        loss = torch.zeros_like(target, dtype=torch.float32)
        loss[b, r] = -probability
        losses[field], active_masks[field] = loss, active
        informative[field] = active & batch["time_upper"][field].gt(batch["time_lower"][field])
        # Do not report the old artificial accuracy=1 from predicted=official.
        if collect_metrics:
            metrics[field] = {"loss": -probability.mean().detach(), "count": float(len(selected))}
    return losses, active_masks, informative, metrics
