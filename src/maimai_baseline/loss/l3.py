"""The L3 objective and Route/position loss composition."""

from __future__ import annotations
from maimai_baseline.contract import DELTA_FIELD, L3_LABEL_SMOOTHING, REMOVED_FIELDS
from maimai_baseline.loss.common import (
    GROUP_BUDGET,
    EVENT_BUDGET,
    L3LossResult,
    POSITION_INTERVAL_CONTRACT,
    apply_field_grammar_mask,
    discrete_position_losses,
    masked_mean,
    materialize_metrics,
)
from maimai_dataset.vocabulary import CompactVocabulary
from typing import Any, Mapping
import torch


NOTE_ATTRIBUTE_FIELDS = (
    "modifier_ex", "modifier_break", "modifier_firework",
    "modifier_short_hold", "modifier_short_touch_hold", "modifier_track_break",
    "head_kind",
    "hold_duration_tick",
    "branch_duration_tick",
)


def route_position_loss(
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    vocabulary: CompactVocabulary,
    *, defer_metrics: bool = False, diagnostic_positions=None, collect_metrics: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, dict[str, float]]]:
    """Compute the fixed Route71 CE + 0.5 Judge + 0.5 Trajectory loss."""

    logits = outputs["route_logits"].to(torch.float32)
    target = batch["target_ids"]["route"]
    active = target.ne(-100)
    safe_target = target.masked_fill(~active, 0)
    event = batch["target_ids"]["event"]
    kind = batch["target_ids"]["kind"]
    note = event.eq(vocabulary.event_id("NOTE"))
    branch = event.eq(vocabulary.event_id("SLIDE_BRANCH"))
    slide = kind.eq(vocabulary.fields["kind"].encode("slide"))
    wifi = kind.eq(vocabulary.fields["kind"].encode("wifi"))
    # Branch has no independent kind. Its route stage must choose among all
    # 71 paths; using the target route to select grammar would leak the answer.
    ordinary = note & slide
    wifi_route = note & wifi
    classified = ordinary | wifi_route | branch
    if not torch.equal(active, classified):
        mismatch = int((active ^ classified).sum().item())
        raise ValueError(f"Route71 target grammar mismatch at {mismatch} rows")
    if bool((ordinary & target.eq(70)).any()) or bool(
        (wifi_route & ~target.eq(70)).any()
    ):
        raise ValueError("Route71 target is outside its legal grammar support")

    allowed = torch.zeros_like(logits, dtype=torch.bool)
    allowed[..., :70] |= ordinary.unsqueeze(-1)
    allowed[..., 70] |= wifi_route
    allowed |= branch.unsqueeze(-1)
    allowed[..., 0] |= ~active
    masked_logits = logits.masked_fill(~allowed, -10_000.0)
    log_probability = torch.log_softmax(masked_logits, dim=-1)
    negative_log_likelihood = -log_probability.gather(
        -1, safe_target.unsqueeze(-1)
    ).squeeze(-1)
    smooth = -(
        log_probability.masked_fill(~allowed, 0.0).sum(dim=-1)
        / allowed.sum(dim=-1).clamp_min(1)
    )
    ce = (
        (1.0 - L3_LABEL_SMOOTHING) * negative_log_likelihood
        + L3_LABEL_SMOOTHING * smooth
    )
    probability = torch.softmax(masked_logits, dim=-1)
    judge_matrix = outputs["route_judge_distance"].to(torch.float32)
    trajectory_matrix = outputs["route_trajectory_distance"].to(torch.float32)
    judge = (probability * judge_matrix[safe_target]).sum(dim=-1)
    trajectory = (probability * trajectory_matrix[safe_target]).sum(dim=-1)
    total = (ce + 0.5 * judge + 0.5 * trajectory).masked_fill(~active, 0.0)

    if not collect_metrics:
        return total, active, {}
    if diagnostic_positions is not None:
        diagnostic_positions.update(route_ce=ce.detach(), route_judge_cost=judge.detach(), route_trajectory_cost=trajectory.detach())
    count = active.sum()
    prediction = masked_logits.argmax(dim=-1)
    predicted_judge = judge_matrix[safe_target, prediction]
    predicted_trajectory = trajectory_matrix[safe_target, prediction]
    def mean(value: torch.Tensor) -> torch.Tensor:
        return masked_mean(value.detach(), active, count=count)

    route_metrics = {
        "loss": mean(total),
        "ce": mean(ce),
        "judge_cost": mean(judge),
        "trajectory_cost": mean(trajectory),
        "accuracy": (
            masked_mean(prediction.eq(target).float(), active, count=count)
        ),
        "argmax_judge_distance": mean(predicted_judge),
        "argmax_trajectory_distance": mean(predicted_trajectory),
        "count": count,
        "mean_legal_support": (
            masked_mean(allowed.sum(dim=-1).to(torch.float32), active, count=count)
        ),
    }
    metrics = {
        "route": route_metrics,
        "route_ce": {"loss": route_metrics["ce"], "count": count},
        "route_judge_cost": {
            "loss": route_metrics["judge_cost"],
            "count": count,
        },
        "route_trajectory_cost": {
            "loss": route_metrics["trajectory_cost"],
            "count": count,
        },
        "route_argmax_judge_distance": {
            "loss": route_metrics["argmax_judge_distance"],
            "count": count,
        },
        "route_argmax_trajectory_distance": {
            "loss": route_metrics["argmax_trajectory_distance"],
            "count": count,
        },
        "route_top1_error": {
            "loss": 1.0 - route_metrics["accuracy"],
            "count": count,
        },
        "route_legal_support": {
            "loss": route_metrics["mean_legal_support"],
            "count": count,
        },
    }
    return total, active, metrics if defer_metrics else materialize_metrics(metrics)


def _prepared_supervision(batch):
    if "supervision" not in batch:
        raise ValueError("batch lacks CPU supervision preparation; call prepare_supervision before device transfer")
    return batch["supervision"]


def note_interval_weights(batch, vocabulary):
    milliseconds = _prepared_supervision(batch)["moment_ms"]
    note = _prepared_supervision(batch)["note_mask"]
    milliseconds = torch.where(note, milliseconds, 100.0)
    reference = POSITION_INTERVAL_CONTRACT["reference_ms"]
    weights = torch.where(
        milliseconds.le(reference),
        1.0 + torch.log2(reference / milliseconds),
        torch.exp2(-(milliseconds - reference) / POSITION_INTERVAL_CONTRACT["long_interval_half_life_ms"]),
    ).clamp_min(POSITION_INTERVAL_CONTRACT["minimum_weight"])
    return weights


def structure_interval_weights(batch, vocabulary):
    """Use the visible current MOMENT interval for NOTE/MOMENT decisions."""
    milliseconds = _prepared_supervision(batch)["moment_ms"]
    return 1.0 + torch.log2(100.0 / milliseconds.clamp_max(100.0))


def _supervision_weights(batch, vocabulary, config):
    prepared = _prepared_supervision(batch)
    kind_weight = torch.where(prepared["hold_mask"], config.hold_multiplier,
                              torch.where(prepared["slide_mask"], config.slide_multiplier, 1.0))
    event_weight = structure_interval_weights(batch, vocabulary)
    event_weight = torch.where(prepared["branch_mask"], config.slide_multiplier, event_weight)
    return {"kind": kind_weight, "event": event_weight,
            "hold_duration_tick": config.hold_multiplier, "branch_duration_tick": config.slide_multiplier}


def _note_position_loss(
    *,
    all_losses: Mapping[str, torch.Tensor],
    all_active: Mapping[str, torch.Tensor],
    note_mask: torch.Tensor,
    kind_target: torch.Tensor,
    vocabulary: CompactVocabulary,
) -> torch.Tensor:
    """Select the one official pad or start-pad position for each NOTE."""

    pad_position = torch.zeros_like(note_mask)
    start_pad_position = torch.zeros_like(note_mask)
    for kind in ("tap", "hold", "touch", "touch_hold"):
        pad_position |= kind_target.eq(vocabulary.fields["kind"].encode(kind))
    for kind in ("slide", "wifi"):
        start_pad_position |= kind_target.eq(vocabulary.fields["kind"].encode(kind))
    position_loss = torch.zeros_like(kind_target, dtype=torch.float32)
    position_valid = torch.zeros_like(note_mask)
    for field, selector in (("pad", pad_position), ("start_pad", start_pad_position)):
        if field not in all_active:
            continue
        valid = note_mask & selector & all_active[field]
        position_loss += all_losses[field] * valid
        position_valid |= valid
    if not torch.equal(position_valid, note_mask):
        raise ValueError("every NOTE row must retain one official position field")
    return position_loss


def _time_position_losses(outputs, batch, runtime_cache=None, *, collect_metrics=True):
    if outputs.get("delta_time_parameterization") != "bpm96-ms":
        raise ValueError("v9 loss requires BPM96 millisecond outputs")
    from maimai_baseline.loss.time import time_position_losses
    return time_position_losses(outputs,batch,runtime_cache,collect_metrics=collect_metrics)


def _l3_loss(*, groups, metrics, difficulty_batch=None, collect_metrics=True,
             raw_groups=None, note_terms=None, event_terms=None, time_metrics=None,
             reduction_support=None):
    """Reduce note, delta and joint event groups on their effective supports."""
    from maimai_baseline.loss.difficulty import weights, mean as weighted_mean, diagnostics
    from maimai_baseline.loss.budget import budget_statistics
    difficulty_batch = difficulty_batch or {}
    raw_groups = groups if raw_groups is None else raw_groups
    reference = groups['note_semantic'][0]
    dw, ids = weights(difficulty_batch, reference)
    weighted = difficulty_batch.get('difficulty_loss_preset', 'uniform') != 'uniform'
    w = torch.ones_like(reference) if dw is None else dw[:, None].expand_as(reference)
    components, original_means, counts = {}, {}, []
    for name, coefficient in GROUP_BUDGET.items():
        values, mask = groups[name]
        count = mask.sum()
        original = masked_mean(values, mask, count=count) if collect_metrics or not weighted else None
        if reduction_support is None:
            value = weighted_mean(values.flatten(), (w * mask).flatten()) if weighted else original
        else:
            active_weight = w * mask if weighted else mask
            denominator = torch.as_tensor(reduction_support[name], device=values.device, dtype=torch.float32)
            value = (values.masked_fill(~mask, 0.).float() * active_weight).sum() / denominator.clamp_min(1.)
        components[name] = value
        if collect_metrics:
            counts.append(count)
            original_means[name] = original.detach()
            metrics['component:' + name] = dict(loss=value.detach(), count=count, weight_sum=(w * mask).sum())
            metrics['budget:' + name] = budget_statistics(raw_groups[name][0], values, mask, w, mask, coefficient)
            if dw is not None:
                for suffix, stats in diagnostics(values, mask, torch.ones_like(values), w,
                                                  ids[:, None].expand_as(mask)).items():
                    metrics[f'difficulty:l3:{name}:{suffix}'] = stats
    total = sum(GROUP_BUDGET[name] * value for name, value in components.items())
    if not collect_metrics:
        return L3LossResult(loss=total, metrics={}, components=components)
    for name, (raw, values, mask, coefficient) in (note_terms or {}).items():
        metrics['budget:note:' + name] = budget_statistics(
            raw, values, mask, w, groups['note_semantic'][1], GROUP_BUDGET['note_semantic'] * coefficient)
    flag_terms = {}
    for name, term in (note_terms or {}).items():
        kind, field = name.split(':', 1)
        if field.startswith('modifier_'):
            flag_terms.setdefault(kind, []).append(term)
    for kind, terms in flag_terms.items():
        mass = sum(coefficient for _, _, _, coefficient in terms)
        raw = sum(raw * coefficient for raw, _, _, coefficient in terms) / mass
        values = sum(value.detach() * coefficient for _, value, _, coefficient in terms) / mass
        mask = torch.stack([mask for _, _, mask, _ in terms]).any(0)
        metrics['budget_summary:note:' + kind + ':flags'] = budget_statistics(
            raw, values, mask, w, groups['note_semantic'][1], GROUP_BUDGET['note_semantic'] * mass)
    for name, (raw, values, mask) in (event_terms or {}).items():
        metrics['budget:event:' + name] = budget_statistics(
            raw, values, mask, w, groups['structure'][1],
            EVENT_BUDGET['router' if name == 'router' else 'conditional_event'])
    time_metrics = time_metrics or {}
    metrics['total'] = dict(loss=total.detach(), group_count=float(len(GROUP_BUDGET)),
        active_fields=torch.stack(counts).gt(0).sum(),
        time_count=float(sum(v['count'] for v in time_metrics.values())),
        delta_median_absolute_error_ms=time_metrics.get(DELTA_FIELD, {}).get('median_absolute_error_ms', 0.))
    if dw is not None:
        metrics['difficulty_objective'] = dict(
            unweighted_l3=sum(GROUP_BUDGET[n] * v for n, v in original_means.items()), weighted_l3=total.detach())
    return L3LossResult(loss=total, metrics=materialize_metrics(metrics), components=components)


from maimai_baseline.loss.supervision import SpecializedSupervision


def l3_loss(
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    vocabulary: CompactVocabulary,
    *,
    runtime_cache: dict[str, Any] | None = None,
    collect_metrics: bool = True,
    supervision: SpecializedSupervision = SpecializedSupervision(),
    position_cache: dict[str, Any] | None = None,
    reduction_support: Mapping[str, Any] | None = None,
) -> L3LossResult:
    """Compute the configured Route71 BPM96 linear-time L3 objective."""
    if position_cache is not None:
        from maimai_baseline.loss.budget import reject_legacy_sft
        reject_legacy_sft()
    if any(field in outputs["field_logits"] for field in REMOVED_FIELDS):
        raise ValueError("L3 received a removed factor head")
    field_logits = apply_field_grammar_mask(
        outputs["field_logits"], batch["target_ids"], vocabulary,
        sd_forbidden_kind=_prepared_supervision(batch)["sd_forbidden_kind"],
    )
    from maimai_baseline.loss.common import mask_audio_support
    field_logits = mask_audio_support(field_logits, batch, vocabulary)
    loss_targets = dict(batch['target_ids'])
    if 'row_field_identity' in outputs:
        for name, valid in outputs['field_valid'].items():
            if name in loss_targets:
                loss_targets[name] = loss_targets[name].masked_fill(~valid, -100)
    discrete, discrete_active, _, metrics = (
        discrete_position_losses(
            field_logits,
            loss_targets,
            defer_metrics=True, collect_metrics=collect_metrics,
        )
    )
    time, time_active, _, time_metrics = _time_position_losses(
        outputs, batch, runtime_cache, collect_metrics=collect_metrics
    )
    from maimai_baseline.loss.family import router_loss
    router, family_metrics = router_loss(dict(outputs, field_logits=field_logits), batch, vocabulary,
        discrete['event'], collect_metrics=collect_metrics)
    metrics.update(family_metrics)
    if runtime_cache is not None and "delta_log_probability" not in runtime_cache:
        raw = outputs["time_outputs"][DELTA_FIELD]
        runtime_cache.update(delta_log_probability=raw.new_empty((0, 0)),
                             delta_lower=torch.empty(0, device=raw.device, dtype=torch.long),
                             delta_row_index=torch.full_like(batch["target_ids"]["event"], -1))
    metrics.update(time_metrics)
    route_positions = {} if collect_metrics and "difficulty_loss_preset" in batch else None
    route_loss, route_active, route_metrics = route_position_loss(
        outputs, batch, vocabulary, defer_metrics=True, diagnostic_positions=route_positions, collect_metrics=collect_metrics
    )
    metrics.update(route_metrics)
    all_losses = {**discrete, **time, 'conditional_event': discrete['event'],
                  'event': discrete['event'] + router, 'router': router}
    raw_losses = dict(all_losses, router=router, route=route_loss)
    supervision_weights = (_supervision_weights(batch, vocabulary, supervision) if supervision.enabled
                           else {"event": structure_interval_weights(batch, vocabulary)})
    supervision_weights.update(router=supervision_weights['event'],
                               conditional_event=supervision_weights['event'])
    if collect_metrics:
        diagnostic_weights = ({"kind": None, "hold_duration_tick": None, "branch_duration_tick": None,
                               **supervision_weights} if not supervision.enabled else supervision_weights)
        for name, weight in diagnostic_weights.items():
            if name in all_losses:
                active = discrete_active['event'] if name in ('router', 'conditional_event') else (
                    discrete_active if name in discrete_active else time_active)[name]
                metrics[f"supervision:{name}"] = {
                    "raw_loss": masked_mean(all_losses[name].detach(), active),
                    "weighted_loss": masked_mean(all_losses[name].detach() if weight is None else all_losses[name].detach() * weight, active),
                    "count": active.sum(),
                }
        metrics["supervision:route"] = {
            "raw_loss": masked_mean(route_loss.detach(), route_active),
            "weighted_loss": masked_mean((route_loss.detach() * supervision.slide_multiplier if supervision.enabled else route_loss.detach()), route_active),
            "count": route_active.sum(),
        }
    all_losses = {name: value * supervision_weights[name] if name in supervision_weights else value
                  for name, value in all_losses.items()}
    if supervision.enabled:
        route_loss = route_loss * supervision.slide_multiplier
    all_active = {**discrete_active, **time_active}
    all_active.update(router=discrete_active['event'], conditional_event=discrete_active['event'])
    event_target = batch["target_ids"]["event"]
    note_mask = event_target.eq(vocabulary.event_id("NOTE"))
    kind_target = batch["target_ids"]["kind"]
    position_loss = _note_position_loss(
        all_losses=all_losses,
        all_active=all_active,
        note_mask=note_mask,
        kind_target=kind_target,
        vocabulary=vocabulary,
    )
    position_weight = note_interval_weights(batch, vocabulary)
    weighted_position_loss = position_loss * position_weight
    if collect_metrics:
        note_count = note_mask.sum()
        metrics["position_supervision"] = {
            "raw_loss": masked_mean(position_loss.detach(), note_mask, count=note_count),
            "weighted_loss": masked_mean(weighted_position_loss.detach(), note_mask, count=note_count),
            "mean_weight": masked_mean(position_weight, note_mask, count=note_count),
            "count": note_count,
        }
    from maimai_baseline.loss.budget import note_semantics
    all_losses.update(position=weighted_position_loss, route=route_loss)
    all_active.update(position=note_mask, route=route_active, router=all_active['event'])
    raw_losses['position'] = position_loss
    note_loss, raw_note_loss, semantic_mask, note_terms = note_semantics(
        all_losses, all_active, batch['target_ids'], vocabulary, raw_losses=raw_losses,
        collect_metrics=collect_metrics)
    if collect_metrics and "difficulty_loss_preset" in batch:
        from maimai_baseline.loss.difficulty import weights as difficulty_weights
        from maimai_baseline.loss.difficulty import diagnostics
        dw, di = difficulty_weights(batch, note_loss)
        if dw is not None:
            w = dw[:, None].expand_as(note_loss)
            ids = di[:, None].expand_as(note_mask)
            fields = {name: (value, all_active[name]) for name,value in all_losses.items()}
            # Absent heads stay disconnected; log zero support without computing their loss.
            empty = (torch.zeros_like(note_loss), torch.zeros_like(note_mask))
            for name in (*field_logits, *batch["time_targets"]):
                fields.setdefault(name, empty)
            fields.update(route=(route_loss,route_active),
                **{name:(value,route_active) for name,value in route_positions.items()})
            fields.update(position_raw=(position_loss,note_mask), position_interval_weighted=(weighted_position_loss,note_mask))
            for field,(value,active) in fields.items():
                for scope,mask in (("all",active),("note",active & note_mask)):
                    for suffix,stats in diagnostics(value,mask,torch.ones_like(value),w,ids).items():
                        metrics[f"difficulty:field:{scope}:{field}:{suffix}"] = stats
    from maimai_baseline.dual_contract import event_family, TIME, NOTE
    family = event_family(event_target, vocabulary)
    event_active = all_active['event']
    empty = torch.zeros_like(note_mask)
    groups = {
        'structure': (sum(EVENT_BUDGET[name] / GROUP_BUDGET['structure'] * all_losses[name]
                          for name in EVENT_BUDGET), event_active),
        'timing': (all_losses.get(DELTA_FIELD, torch.zeros_like(note_loss)), all_active.get(DELTA_FIELD, empty)),
        'note_semantic': (note_loss, semantic_mask),
    }
    raw_groups = dict(groups, structure=(sum(
        EVENT_BUDGET[name] / GROUP_BUDGET['structure'] * raw_losses[name]
        for name in EVENT_BUDGET), event_active),
        note_semantic=(raw_note_loss, semantic_mask))
    event_terms = {
        'router': (router, all_losses['router'], event_active),
        'conditional_event': (discrete['event'], all_losses['conditional_event'], event_active),
        'time_event': (discrete['event'], all_losses['conditional_event'], event_active & family.eq(TIME)),
        'note_event': (discrete['event'], all_losses['conditional_event'], event_active & family.eq(NOTE)),
    }
    return _l3_loss(groups=groups, raw_groups=raw_groups, note_terms=note_terms,
                    event_terms=event_terms,
                    metrics=metrics, time_metrics=time_metrics,
                    difficulty_batch=batch, collect_metrics=collect_metrics,
                    reduction_support=reduction_support)


def logical_support(batch, vocabulary):
    """Exact existing L3 denominators from GT, independent of predictions."""
    from maimai_baseline.loss.difficulty import weights
    event = batch['target_ids']['event']
    masks = dict(structure=event.ne(-100),
        timing=batch['time_targets'][DELTA_FIELD].ne(-100),
        note_semantic=event.eq(vocabulary.event_id('NOTE')) | event.eq(vocabulary.event_id('SLIDE_BRANCH')))
    dw, _ = weights(batch, event.float())
    weighted = batch.get('difficulty_loss_preset', 'uniform') != 'uniform'
    return {name: (mask.float() * dw[:, None]).sum() if weighted else mask.sum()
            for name, mask in masks.items()}
