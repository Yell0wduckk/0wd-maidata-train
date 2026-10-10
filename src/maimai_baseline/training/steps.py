"""Device transfer, optimization, scheduling, and step metrics."""
from __future__ import annotations

from maimai_baseline.loss.supervision import SpecializedSupervision, from_args as supervision_from_args
from contextlib import nullcontext
from maimai_baseline.contract import LEARNING_RATE, WEIGHT_DECAY
from maimai_baseline.loss.common import GROUP_BUDGET
from maimai_baseline.training.recipe import recipe
from maimai_baseline.model.capacity import from_args as capacity_from_args
from maimai_baseline.model.baseline import BaselineTransformer, baseline_forward
from maimai_baseline.physical.loss import baseline_physical_loss
from maimai_dataset.vocabulary import CompactVocabulary
from typing import Any, Mapping
import math
import statistics
import torch


def _device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    return torch.device("cuda" if torch.cuda.is_available() and requested != "cpu" else "cpu")


def _move(value: Any, device: torch.device) -> Any:
    from maimai_baseline.data.batch_transport import PackedBatch
    if isinstance(value, PackedBatch):
        return value.to_device(device)
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: _move(child, device) for key, child in value.items()}
    if isinstance(value, list):
        return [_move(child, device) for child in value]
    if isinstance(value, tuple):
        return tuple(_move(child, device) for child in value)
    return value


def _autocast(device: torch.device, precision="bf16"):
    return (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and precision == "bf16"
        else nullcontext()
    )


def _module_gradient_norm(module: torch.nn.Module) -> float:
    squares = [
        parameter.grad.detach().to(torch.float32).square().sum()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    return float(torch.stack(squares).sum().sqrt().item()) if squares else 0.0


def _scheduler_factor(step: int, *, warmup_steps: int, total_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return min(1.0, max(1e-6, (step + 1) / warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))


def _scheduler(
    optimizer: torch.optim.Optimizer, *, warmup_steps: int, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    def factor(step: int) -> float:
        return _scheduler_factor(
            step, warmup_steps=warmup_steps, total_steps=total_steps
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _learning_rate(args: Any) -> float:
    requested = getattr(args, "learning_rate", None)
    value = recipe(args).learning_rate if requested is None else float(requested)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("learning_rate must be finite and positive")
    return value


def _optimizer(model: BaselineTransformer, *, learning_rate: float = LEARNING_RATE, weight_decay: float = WEIGHT_DECAY) -> torch.optim.AdamW:
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    return torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
        fused=bool(torch.cuda.is_available()),
    )


def _loss_step(
    model: BaselineTransformer,
    batch: Mapping[str, Any],
    vocabulary: CompactVocabulary,
    *,
    warmup_coefficient: float = 1.0,
    collect_metrics: bool = True,
    style_progress: float = 1.0,
):
    batch = dict(batch, difficulty_loss_preset=getattr(model, "difficulty_loss_preset", "uniform"))
    previous = getattr(model, "collect_mmoe_metrics", False)
    model.collect_mmoe_metrics = collect_metrics
    try:
        outputs = baseline_forward(model, batch)
    finally:
        model.collect_mmoe_metrics = previous
    result = baseline_physical_loss(outputs, batch, vocabulary, warmup_coefficient=warmup_coefficient, collect_metrics=collect_metrics,
                                    supervision=getattr(model, "specialized_supervision", SpecializedSupervision()))
    if collect_metrics:
        from dataclasses import replace
        metrics = dict(result.metrics)
        for d, layer in enumerate(model.decoder_layers):
            for task, values in layer.metrics.items():
                metrics[f"mmoe:layer_{d}:{task}"] = values
        result = replace(result, metrics=metrics)
    if 'radar_prior' in outputs:
        from dataclasses import replace
        from maimai_baseline.training.radar import prior_loss
        effective, radar_metrics = prior_loss(outputs['radar_prior'], collect_metrics=collect_metrics)
        result = replace(result, loss=result.loss + effective, metrics={**result.metrics, **radar_metrics})
    if model.config.latent_enabled and "style_posterior" in outputs:
        from dataclasses import replace
        from maimai_baseline.style_contract import style_beta
        posterior = outputs["style_posterior"]
        kl, raw = posterior.kl()
        beta = style_beta(style_progress)
        effective = beta * kl
        metrics = dict(result.metrics)
        if collect_metrics:
            metrics["style:kl"] = {"loss": float(kl.detach()), "beta": beta,
                "effective": float((beta * kl).detach()), "count": int(posterior.valid.sum())}
        if collect_metrics:
            mask = posterior.valid[..., None]
            metrics["style:posterior"] = {
                "raw_kl": float((raw.detach() * mask).sum() / mask.sum().clamp_min(1)),
                "gate_abs_mean": float(model.style_conditioner.gate_values().detach().abs().mean()),
                "mu_rms": float(((posterior.mu.detach().square() * mask).sum() / (mask.sum()*posterior.mu.shape[-1]).clamp_min(1)).sqrt())}
        result = replace(result, loss=result.loss + effective, metrics=metrics)
    if not bool(torch.isfinite(result.loss)):
        raise RuntimeError("non-finite baseline loss")
    return outputs, result


def _accumulate_loss_components(
    accumulator: dict[str, dict[str, float]],
    metrics: Mapping[str, Mapping[str, float]],
) -> None:
    """Accumulate component and field-budget metrics by their actual support."""
    for metric_name, values in metrics.items():
        if metric_name == 'radar:nll':
            target = accumulator.setdefault(metric_name, dict(loss_sum=0., count=0., windows=0.))
            target['loss_sum'] += float(values['loss']) * float(values['count'])
            target['count'] += float(values['count'])
            target['windows'] += float(values['windows'])
            continue
        if metric_name.startswith("mmoe:"):
            count = float(values["count"])
            target = accumulator.setdefault(metric_name, {"count": 0.})
            target["count"] += count
            for key, value in values.items():
                if key != "count":
                    target[key] = target.get(key, 0.) + float(value) * count
            continue
        if metric_name.startswith(('budget:', 'budget_summary:')):
            target = accumulator.setdefault(metric_name, {})
            for name, value in values.items():
                number = float(value)
                if not math.isfinite(number):
                    raise RuntimeError(f'non-finite budget metric {metric_name}:{name}')
                target[name] = target.get(name, 0.) + number
            continue
        if metric_name in {'router', 'router_time', 'router_note', 'conditional_event', 'event'}:
            count = float(values['count'])
            target = accumulator.setdefault('routing:' + metric_name,
                dict(loss_sum=0., accuracy_sum=0., count=0.))
            target['loss_sum'] += float(values['loss']) * count
            target['accuracy_sum'] += float(values['accuracy']) * count
            target['count'] += count
            continue
        if metric_name.startswith("modifier_"):
            target = accumulator.setdefault(metric_name, dict.fromkeys(("tp", "fp", "fn", "tn"), 0.))
            for key in target:
                target[key] += float(values[key])
            continue
        if metric_name.startswith("style:"):
            target = accumulator.setdefault(metric_name, {"batches": 0.})
            target["batches"] += 1
            for key, value in values.items():
                target[key] = target.get(key, 0.) + float(value)
            continue
        if metric_name.startswith("difficulty:"):
            target = accumulator.setdefault(metric_name, {})
            for key, value in values.items():
                target[key] = target.get(key, 0.0) + float(value)
            target["batches"] = target.get("batches", 0.0) + 1
            continue
        if metric_name == "difficulty_objective":
            target = accumulator.setdefault(metric_name, {"steps": 0.0})
            target["steps"] += 1
            for key, value in values.items():
                target[key] = target.get(key, 0.0) + float(value)
            continue
        if metric_name == "position_supervision" or metric_name.startswith("supervision:"):
            count = float(values["count"])
            target = accumulator.setdefault(metric_name, {
                "raw_loss_sum": 0.0, "weighted_loss_sum": 0.0,
                "mean_weight_sum": 0.0, "count": 0.0,
            })
            for name in ("raw_loss", "weighted_loss", "mean_weight") if metric_name == "position_supervision" else ("raw_loss", "weighted_loss"):
                target[name + "_sum"] += float(values[name]) * count
            target["count"] += count
        if metric_name.startswith("physical:"):
            required = ("loss", "coefficient") if metric_name == "physical:total" else ("loss", "bad_mass", "count")
            for key in required:
                if key not in values:
                    raise RuntimeError(f"missing Physical metric {metric_name}:{key}")
            for key in (*required, "weight_sum", "trigger_count"):
                if key in values and not math.isfinite(float(values[key])):
                    raise RuntimeError(f"non-finite Physical metric {metric_name}:{key}")
        if metric_name == "physical:total":
            target = accumulator.setdefault("physical:effective", {"loss_sum": 0.0, "count": 0.0})
            target["loss_sum"] += float(values["loss"]) * float(values["coefficient"])
            target["count"] += 1.0
        if metric_name.startswith("physical:") and metric_name != "physical:total":
            count = float(values.get("weight_sum", values.get("count", 0.0))) if values.get("exact_weighted_mean") else float(values.get("count", 0.0))
            target = accumulator.setdefault(metric_name, {"loss_sum": 0.0, "count": 0.0, "mass_sum": 0.0, "trigger_count": 0.0})
            # Match the loss reducer's clamp_min(1) for fractional opportunity supports.
            divisor = count if values.get("exact_weighted_mean") else max(1.0, count)
            target["loss_sum"] += float(values["loss"]) * divisor
            target["mass_sum"] += float(values["bad_mass"]) * divisor
            target["exact_weighted_mean"] = float(values.get("exact_weighted_mean", 0))
            target["count"] += count
            target["trigger_count"] += float(values.get("trigger_count", 0.0))
        if not metric_name.startswith("component:"):
            continue
        name = metric_name.removeprefix("component:")
        if name not in GROUP_BUDGET:
            continue
        count = float(values.get("weight_sum", values.get("count", 0.0)))
        loss = float(values.get("loss", 0.0))
        if count <= 0.0:
            continue
        if not math.isfinite(loss) or not math.isfinite(count):
            raise RuntimeError(
                f"non-finite baseline loss component {name!r}: loss={loss}, count={count}"
            )
        target = accumulator.setdefault(name, {"loss_sum": 0.0, "count": 0.0})
        target["loss_sum"] += loss * count
        target["count"] += count


def _finalize_loss_components(
    accumulator: Mapping[str, Mapping[str, float]],
) -> tuple[float, dict[str, float]]:
    """Reconstruct the epoch loss using the baseline's support-weighted contract."""

    def mean(name: str) -> float:
        values = accumulator.get(name)
        if values is None or float(values.get("count", 0.0)) <= 0.0:
            return 0.0
        return float(values["loss_sum"]) / float(values["count"])

    components = {name: mean(name) for name in GROUP_BUDGET}
    chart_loss = sum(GROUP_BUDGET[name] * value for name, value in components.items())
    for prefix, output in (('budget:', 'budgets'), ('budget_summary:', 'budget_summaries')):
        budgets = {}
        for name, values in accumulator.items():
            if not name.startswith(prefix):
                continue
            budgets[name.removeprefix(prefix)] = dict(values,
                raw_loss=values['raw_loss_sum'] / (values['count'] or 1.),
                loss=values['loss_sum'] / (values['weight_sum'] or 1.),
                contribution=values['contribution_sum'] / (values['denominator_weight_sum'] or 1.))
        if budgets:
            components[output] = budgets
    mmoe = {name.removeprefix("mmoe:"): dict(count=values["count"],
        **{k: v / max(1., values["count"]) for k, v in values.items() if k != "count"})
        for name, values in accumulator.items() if name.startswith("mmoe:")}
    if mmoe:
        components["mmoe"] = mmoe
    components['routing'] = {key.removeprefix('routing:'): dict(
        loss=value['loss_sum'] / max(1., value['count']),
        accuracy=value['accuracy_sum'] / max(1., value['count']), count=value['count'])
        for key, value in accumulator.items() if key.startswith('routing:')}
    flags = {}
    for key, value in accumulator.items():
        if key.startswith("modifier_"):
            tp, fp, fn, tn = (value[n] for n in ("tp", "fp", "fn", "tn"))
            flags[key.removeprefix("modifier_")] = dict(value,
                positive_precision=tp/max(1., tp+fp), positive_recall=tp/max(1., tp+fn),
                predicted_positive_rate=(tp+fp)/max(1., tp+fp+fn+tn))
    if flags:
        components["modifiers"] = flags
    if "position_supervision" in accumulator:
        position = accumulator["position_supervision"]
        components["position_supervision"] = {
            name: position[name + "_sum"] / max(1.0, position["count"])
            for name in ("raw_loss", "weighted_loss", "mean_weight")
        }
        components["position_supervision"]["count"] = position["count"]
    supervision = {
        key.removeprefix("supervision:"): {
            "raw_loss": value["raw_loss_sum"] / max(1.0, value["count"]),
            "weighted_loss": value["weighted_loss_sum"] / max(1.0, value["count"]),
            "count": value["count"],
        }
        for key, value in accumulator.items() if key.startswith("supervision:")
    }
    if supervision:
        components["supervision"] = supervision
    if "physical:effective" in accumulator:
        from maimai_baseline.physical.contract import PHYSICAL_CATEGORIES
        from maimai_baseline.physical.contract import PHYSICAL_CATEGORY_WEIGHTS
        physical_heads = {
            name: {"loss": value["loss_sum"] / ((value["count"] or 1.0) if value.get("exact_weighted_mean") else max(1.0, value["count"])),
                   "bad_mass": value["mass_sum"] / ((value["count"] or 1.0) if value.get("exact_weighted_mean") else max(1.0, value["count"])),
                   "count": value["count"], "trigger_count": value["trigger_count"]}
            for name, value in accumulator.items()
            if name.startswith("physical:") and name.count(":") >= 2 and name != "physical:effective"
        }
        categories = {}
        for category in PHYSICAL_CATEGORIES:
            heads = [v for k, v in physical_heads.items() if k.startswith(f"physical:{category}:") and v["count"] > 0]
            categories[category] = statistics.fmean(v["loss"] for v in heads) if heads else 0.0
        components.update(chart_loss=chart_loss, physical_categories=categories,
                          physical_responsibilities=physical_heads,
                          physical=sum(categories[k] * w for k, w in zip(PHYSICAL_CATEGORIES, PHYSICAL_CATEGORY_WEIGHTS, strict=True)) / 4,
                          physical_effective=mean("physical:effective"))
    if "difficulty_objective" in accumulator:
        objective = accumulator["difficulty_objective"]
        components["difficulty_objective_step_means"] = {k: v / objective["steps"] for k,v in objective.items() if k != "steps"}
    components["difficulty_diagnostics"] = {
        name: {**values, "weighted_loss": values["loss_sum"] / (values["weight_sum"] or 1.0),
               "unweighted_loss": values["unweighted_loss_sum"] / (values["base_weight_sum"] or 1.0),
               "step_mean_contribution": values["contribution"] / values["batches"]}
        for name, values in accumulator.items() if name.startswith("difficulty:")}
    if not components["difficulty_diagnostics"]:
        del components["difficulty_diagnostics"]
    else:
        components["field_losses"] = {k.removeprefix("difficulty:field:"): v for k,v in components["difficulty_diagnostics"].items() if k.startswith("difficulty:field:")}
    components["style"] = {name: {k: v / values["batches"] for k, v in values.items() if k != "batches"}
                           for name, values in accumulator.items() if name.startswith("style:")}
    style_effective = sum(values.get("effective", 0.) for values in components["style"].values())
    if not components["style"]:
        del components["style"]
    radar_effective = 0.
    if 'radar:nll' in accumulator:
        from maimai_baseline.radar_contract import LOSS_COEFFICIENT
        values = accumulator['radar:nll']
        components['radar'] = dict(nll=mean('radar:nll'), coordinates=values['count'],
                                  windows=values['windows'], coefficient=LOSS_COEFFICIENT)
        radar_effective = mean('radar:nll') * LOSS_COEFFICIENT
    return chart_loss + mean("physical:effective") + style_effective + radar_effective, components


def _validate_resume_training_config(saved: Mapping[str, Any], args: Any) -> None:
    """Restore numerical schedule/optimizer settings; ignore experiment provenance.

    Optimizer/scheduler tensors are restored by the checkpoint loader. LambdaLR's
    callable is reconstructed here from the settings that produced those tensors.
    """
    from maimai_baseline.training.recipe import bind_stage_recipe
    resolved = dict(bind_stage_recipe(args, "A"))
    previous = saved.get("resolved_recipe", {})
    for key in ("scheduler_epochs", "warmup_steps", "physical_warmup_steps", "learning_rate",
                "weight_decay", "gradient_clip", "seed", "precision", "offset"):
        if key in previous:
            resolved[key] = previous[key]
        elif key in saved:
            resolved[key] = saved[key]
        elif key == "offset" and "plan_offset" in saved:
            resolved[key] = saved["plan_offset"]
    args.resolved_recipe = resolved
    for key in ("scheduler_epochs", "warmup_steps", "physical_warmup_steps", "learning_rate",
                "weight_decay", "gradient_clip", "seed", "precision"):
        setattr(args, key, resolved[key])
    args.plan_offset = resolved["offset"]


def set_scheduler_progress(scheduler, progress):
    """Set LambdaLR by data exposure without treating it as an update counter."""
    if not math.isfinite(progress) or progress < 0:
        raise ValueError("invalid scheduler progress")
    scheduler.last_epoch = float(progress)
    values = [base * fn(progress) for base, fn in zip(scheduler.base_lrs, scheduler.lr_lambdas, strict=True)]
    for group, value in zip(scheduler.optimizer.param_groups, values, strict=True):
        group["lr"] = value
    scheduler._last_lr = values


def batch_work(batch, vocabulary):
    """Cheap CPU-side work counters, before device transfer."""
    events = batch["target_ids"]["event"]
    dense = batch.get("song_dense_inputs")
    # Cached contexts are consumed directly; there is no raw audio batch and
    # no audio-encoder evaluation to count. Do not require a placeholder tensor.
    audio_windows = (0 if dense is not None and "audio_context" in dense else
                     int((dense["batch"] if dense is not None else batch)["audio"].shape[0]))
    return {"target_rows": int(events.ne(-100).sum()),
            "note_rows": int(events.eq(vocabulary.event_id("NOTE")).sum()),
            "audio_windows": audio_windows}
