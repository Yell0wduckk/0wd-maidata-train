"""Loss contracts, result types, reductions, and statistics."""

from __future__ import annotations
from dataclasses import dataclass, field
from maimai_baseline.contract import L3_LABEL_SMOOTHING, L3_LOSS_ID
from maimai_dataset.vocabulary import CompactVocabulary
from typing import Any, Mapping
import torch


GROUP_NOTE_BUDGET = {"kind": 0.45, "fields": 0.55}

# Explicit coefficients within F; do not normalize or redistribute missing fields.
NOTE_FIELD_BUDGET = {
    "tap": {"position": 7/15, "modifier_ex": 2/15,
            "modifier_break": 4/15, "modifier_short_hold": 2/15},
    "touch": {"position": 3/5, "modifier_firework": 1/5,
              "modifier_short_touch_hold": 1/5},
    "hold": {"position": 3/10, "modifier_ex": 1/10,
             "modifier_break": 1/5, "duration": 2/5},
    "touch_hold": {"position": 2/5, "modifier_firework": 1/5, "duration": 2/5},
    "slide": {"position": 7/30, "modifier_ex": 1/15, "modifier_break": 2/15,
              "modifier_track_break": 1/15, "route": 1/2, "duration": 1/2},
    "wifi": {"position": 7/30, "modifier_ex": 1/15, "modifier_break": 2/15,
             "modifier_track_break": 1/15, "route": 1/2, "duration": 1/2},
}
BRANCH_FIELD_BUDGET = {"route": .4, "duration": .4, "flags": .2}


GROUP_BUDGET = {
    "note_semantic": 0.65,
    "timing": 0.15,
    "structure": 0.20,
}

# Absolute L3 coefficients, sharing the full event-row denominator. Structure
# keeps its outer 0.2 coefficient; its value is therefore 2*router+conditional.
EVENT_BUDGET = {"router": 0.40, "conditional_event": 0.20}


_L3_BASE_CONTRACT = {
    "schema": "maimai.baseline_loss_contract.v2",
    "name": "L3",
    "loss_scheme": L3_LOSS_ID,
    "label_smoothing": L3_LABEL_SMOOTHING,
    "top_level_reduction": "fixed-note-delta-joint-event-budgets",
    "target_conditioning": "official-target",
    "note_budget": dict(GROUP_NOTE_BUDGET),
    "group_budget": dict(GROUP_BUDGET),
    "groups": list(GROUP_BUDGET),
    "note_field_budget": {k: dict(v) for k, v in NOTE_FIELD_BUDGET.items()},
    "branch_field_budget": dict(BRANCH_FIELD_BUDGET),
    "note_reduction": "joint-note-and-branch-row-mean",
    "flags_reduction": "explicit-per-head-coefficients-no-renormalization-or-redistribution",
    "break_budget": "double-note-modifier-break-from-position;track-break-unchanged",
    "flag_diagnostics": "additive-per-head-budgets;non-additive-budget-summary-flags",
    "route_budget": "single-ce-plus-half-judge-plus-half-trajectory-no-informative-mass",
    "event_budget": dict(EVENT_BUDGET),
    "event_reduction": "shared-all-active-event-row-mean-no-renormalization",
    "structure_value": "two-router-plus-conditional-event",
    "event_field_diagnostic": "unscaled-router-plus-conditional-event",
    "delta_owner": "all-active-delta-rows",
    "missing_group_policy": "zero-without-renormalization",
}


POSITION_INTERVAL_CONTRACT = {
    "policy": "gt-moment-delta-log2-exp2-v1",
    "fields": ["pad", "start_pad"],
    "reference_ms": 100.0,
    "minimum_weight": 0.25,
    "long_interval_half_life_ms": 100.0,
    "formula": "max(0.25, 1 + log2(100 / d) if d <= 100 else 2 ** (-(d - 100) / 100))",
    "interval_source": "teacher-forced-input-moment-delta-ms",
    "first_target_moment_weight": 1.0,
    "reduction": "original-note-count-without-weight-renormalization",
    "physical_weighting": "unchanged",
}


_L3_CONTRACT = {
    **_L3_BASE_CONTRACT,
    "position_interval_weighting": POSITION_INTERVAL_CONTRACT,
    "structure_interval_weighting": {
        "formula": "1 + log2(100 / d) if d < 100 else 1",
        "events": ["NOTE", "MOMENT"],
        "interval_source": "visible-current-gt-moment-delta-ms",
        "first_target_moment_weight": 1.0,
        "reduction": "original-count-without-weight-renormalization",
    },
    "specialized_supervision": {
        "policy": "configurable-hold-slide-v1",
        "kind_targets": ["hold", "touch_hold", "slide", "wifi"],
        "fields": ["route", "hold_duration_tick", "branch_duration_tick"],
        "event_targets": ["SLIDE_BRANCH"],
        "physical_weighting": "unchanged",
    },
    "sd_kind_mask": ["touch", "touch_hold"],
}


def loss_contract(supervision=None):
    from maimai_baseline.dual_contract import identity
    from maimai_baseline.loss.supervision import SpecializedSupervision
    config = SpecializedSupervision() if supervision is None else supervision
    return {**_L3_CONTRACT, 'dual_decoder': identity(), "specialized_supervision": {
        **_L3_CONTRACT["specialized_supervision"], **config.identity()}}


L3_LOSS_CONTRACT = loss_contract()


def masked_mean(
    values: torch.Tensor, mask: torch.Tensor, *, count: torch.Tensor | None = None,
) -> torch.Tensor:
    # Mask before reduction (rather than multiplying): inactive NaN/Inf is not
    # part of the objective. Empty groups remain differentiable zero.
    if count is None:
        count = mask.sum()
    return values.masked_fill(~mask, 0.0).sum() / count.clamp_min(1)


def materialize_metrics(
    metrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    entries = [(group, name, value) for group, row in metrics.items()
               for name, value in row.items() if isinstance(value, torch.Tensor)]
    values = (
        torch.stack([value.detach() for _, _, value in entries])
        .to(torch.float64).cpu().tolist() if entries else []
    )
    result = {group: dict(row) for group, row in metrics.items()}
    for (group, name, _), value in zip(entries, values, strict=True):
        result[group][name] = float(value)
    for group in list(result):
        row = result[group]
        omit = row.pop("_omit_empty", False)
        if omit and row["count"] == 0:
            del result[group]
    return result


@dataclass(frozen=True)
class L3LossResult:
    loss: torch.Tensor
    metrics: dict[str, dict[str, float]]
    components: dict[str, torch.Tensor] = field(default_factory=dict)


def apply_field_grammar_mask(
    logits: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    vocabulary: CompactVocabulary,
    *, sd_forbidden_kind: torch.Tensor,
) -> dict[str, torch.Tensor]:
    result = dict(logits)
    event_targets = targets.get("event")
    kind_targets = targets.get("kind")
    if event_targets is None:
        return result

    def permit(
        allowed: torch.Tensor,
        selector: torch.Tensor,
        field_name: str,
        event: str,
        note_kind: str | None,
    ) -> None:
        ids = vocabulary.allowed_value_ids(event, field_name, note_kind=note_kind)
        if not ids:
            return
        value_mask = torch.zeros(
            (allowed.shape[-1],), dtype=torch.bool, device=allowed.device
        )
        value_mask[list(ids)] = True
        allowed.logical_or_(selector.unsqueeze(-1) & value_mask)

    for field_name, values in logits.items():
        if field_name == "event" or field_name.startswith("modifier_") or field_name not in targets:
            continue
        target = targets[field_name]
        active = target.ne(-100)
        allowed = torch.zeros_like(values, dtype=torch.bool)
        for event in vocabulary.applicability:
            selector = active & event_targets.eq(vocabulary.event_id(event))
            if event == "NOTE" and field_name != "kind" and kind_targets is not None:
                for note_kind in vocabulary.note_kind_value_constraints:
                    kind_id = vocabulary.fields["kind"].encode(note_kind)
                    permit(
                        allowed,
                        selector & kind_targets.eq(kind_id),
                        field_name,
                        event,
                        note_kind,
                    )
            else:
                permit(allowed, selector, field_name, event, None)
        allowed.scatter_(-1, target.masked_fill(~active, 0).unsqueeze(-1), True)
        result[field_name] = values.masked_fill(~allowed, -10_000.0)
        if field_name == "kind":
            result[field_name] = result[field_name].masked_fill(sd_forbidden_kind, -torch.inf)
    return result


def mask_audio_support(logits, batch, vocabulary):
    """Finite-support grammar shared by supervised and Physical probabilities."""
    if "audio_tail_tick" not in batch:
        return logits  # Isolated formula fixtures have no audio metadata.
    from maimai_baseline.time.audio_end import duration_upper
    result = dict(logits)
    anchor = batch["row_anchor_tick"].long()
    tail = batch["audio_tail_tick"][:, None]
    hold_ok = batch['time_upper']['hold_duration_tick'].ge(2)
    slide_ok = batch['time_upper']['branch_duration_tick'].ge(1)
    allowed = torch.ones_like(logits["kind"], dtype=torch.bool)
    for name, valid in (("hold", hold_ok), ("touch_hold", hold_ok), ("slide", slide_ok), ("wifi", slide_ok)):
        allowed[..., vocabulary.fields["kind"].encode(name)] = valid
    result["kind"] = logits["kind"].masked_fill(~allowed, -10_000.)
    event_allowed = torch.ones_like(logits["event"], dtype=torch.bool)
    event_allowed[..., vocabulary.event_id("SLIDE_BRANCH")] = slide_ok
    result["event"] = logits["event"].masked_fill(~event_allowed, -10_000.)
    for name in ("short_hold", "short_touch_hold"):
        key = "modifier_" + name
        allowed = torch.stack((torch.ones_like(hold_ok), batch['time_upper']['hold_duration_tick'].ge(1)), -1)
        result[key] = logits[key].masked_fill(~allowed, -10_000.)
    return result


def discrete_position_losses(
    field_logits: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    *, defer_metrics: bool = False, collect_metrics: bool = True,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, dict[str, float]],
]:
    losses: dict[str, torch.Tensor] = {}
    active_masks: dict[str, torch.Tensor] = {}
    informative_masks: dict[str, torch.Tensor] = {}
    metrics: dict[str, dict[str, float]] = {}
    # One presence transfer preserves disconnected absent heads and therefore
    # AdamW's skip behavior, instead of reading a scalar for every field.
    present = torch.stack([targets[name].ne(-100).any() for name in field_logits]).cpu().tolist() if field_logits else []
    for (field_name, logits), has_target in zip(field_logits.items(), present, strict=True):
        if not has_target:
            continue
        target = targets[field_name]
        active = target.ne(-100)
        safe_target = target.masked_fill(~active, 0)
        allowed = logits.gt(-5_000.0)
        allowed.scatter_(-1, safe_target.unsqueeze(-1), True)
        log_probability = torch.log_softmax(logits, dim=-1)
        nll = -log_probability.gather(-1, safe_target.unsqueeze(-1)).squeeze(-1)
        smooth = -(
            log_probability.masked_fill(~allowed, 0.0).sum(dim=-1)
            / allowed.sum(dim=-1).clamp_min(1)
        )
        position_loss = (
            (1.0 - L3_LABEL_SMOOTHING) * nll
            + L3_LABEL_SMOOTHING * smooth
        )
        if field_name.startswith("modifier_"):
            position_loss = nll
        losses[field_name] = position_loss
        active_masks[field_name] = active
        informative_masks[field_name] = active & allowed.sum(dim=-1).gt(1)
        if collect_metrics:
            count = active.sum()
            prediction = logits.argmax(dim=-1)
            metrics[field_name] = {
                "loss": masked_mean(position_loss.detach(), active, count=count),
                "accuracy": masked_mean(prediction.eq(target).float(), active, count=count),
                "count": count,
                "_omit_empty": True,
            }
            if field_name.startswith("modifier_"):
                tp = (active & target.eq(1) & prediction.eq(1)).sum()
                fp = (active & target.eq(0) & prediction.eq(1)).sum()
                fn = (active & target.eq(1) & prediction.eq(0)).sum()
                tn = (active & target.eq(0) & prediction.eq(0)).sum()
                metrics[field_name].update(tp=tp, fp=fp, fn=fn, tn=tn,
                    positive_precision=tp.float() / (tp + fp).clamp_min(1),
                    positive_recall=tp.float() / (tp + fn).clamp_min(1),
                    predicted_positive_rate=(tp + fp).float() / count.clamp_min(1))
    if collect_metrics and not defer_metrics:
        metrics = materialize_metrics(metrics)
        losses = {key: value for key, value in losses.items() if key in metrics}
        active_masks = {key: value for key, value in active_masks.items() if key in metrics}
        informative_masks = {key: value for key, value in informative_masks.items() if key in metrics}
    return losses, active_masks, informative_masks, metrics
