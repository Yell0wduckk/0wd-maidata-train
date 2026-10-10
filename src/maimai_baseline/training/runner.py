"""Preflight, fit, validation, benchmark, and checkpoint orchestration."""
from __future__ import annotations
from maimai_baseline.architecture import DECODER_LAYERS

from maimai_baseline.runtime_versions import stamp
from maimai_baseline.experiment import stage, bound
from maimai_baseline.model.capacity import from_args as capacity_from_args, PAIRING_SCHEME
from maimai_baseline.audio_cross import from_args as cross_from_args, identity as cross_identity
from maimai_baseline.contract import baseline_contract, frequency_pooling_identity, position_conditioning_identity, history_representation_identity, audio_attention_bias_identity, audio_fusion_identity
from maimai_baseline.loss.supervision import from_args as supervision_from_args
from maimai_baseline.checkpoint import (
    load_baseline_checkpoint,
    _load_baseline_payload,
    restore_baseline_training_state,
    save_baseline_checkpoint,
    validate_resume_manifest,
)
from maimai_baseline.contract import (
    ARCHITECTURE,
    BASE_MODEL_ARCHITECTURE,
    CHECKPOINT_EVERY_EPOCHS,
    CHECKPOINT_SCHEMA,
    FIXED_OPTIMIZER_STEPS_PER_EPOCH,
    GRADIENT_CLIP,
    GRADIENT_DIAGNOSTIC_INTERVAL,
    L3_LOSS_ID,
    SEED,
    TIME_COMPONENTS,
    TIME_HEAD,
    TRAINING_RUN_SCHEMA,
    WARMUP_STEPS,
    WEIGHT_DECAY,
)
from maimai_baseline.data.dataset import EpochPlanBatchSampler
from maimai_baseline.loss.common import loss_contract
from maimai_baseline.model.baseline import BaselineTransformer
from maimai_baseline.model.common import BaselineModelConfig, state_dict_sha256
from maimai_baseline.model.attention import from_args as attention_from_args
from maimai_baseline.physical.contract import (
    CONTRACT as PHYSICAL_CONTRACT,
    LOSS_SCHEME as PHYSICAL_LOSS_ID,
    PHYSICAL_WARMUP_STEPS,
)
from maimai_baseline.time.duration import LINEAR_CONTRACT as DURATION_DISTRIBUTION
from maimai_baseline.time.moment import LINEAR_CONTRACT as MOMENT_DISTRIBUTION
from maimai_baseline.training.initialization import _initialize_model
from maimai_baseline.training.inputs import (
    TrainingInputs,
    _ValidationTokenBudgetSampler,
    _dataset,
    _loaders,
    _prepare_inputs,
)
from maimai_baseline.training.recipe import recipe, data_progress, bind_stage_recipe, plan_steps
from maimai_baseline.training.steps import (
    set_scheduler_progress,
    batch_work,
    _scheduler_factor,
    _accumulate_loss_components,
    _autocast,
    _device,
    _finalize_loss_components,
    _learning_rate,
    _loss_step,
    _module_gradient_norm,
    _move,
    _optimizer,
    _scheduler,
    _validate_resume_training_config,
)
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Any, Mapping
import json
import random
import numpy as np
import statistics
import time
import torch


def _settings(args, inputs):
    settings = getattr(args, "resolved_recipe", None)
    if settings is not None:
        return settings
    return stage("A", inputs.manifest["experiment"])


def _physical_coefficient(args, inputs, progress):
    warmup = _settings(args, inputs)['physical_warmup_steps']
    return min(1., max(0., progress / warmup)) if warmup else 1.


def _reference_batches(batches):
    return [[[int(ref.canonical_index), int(ref.view_id), int(ref.occurrence)]
             for ref in batch] for batch in batches]


def _validate_completed_plan(payload, plans):
    """Compare only the recorded, actually consumed last complete epoch."""
    saved = payload.get('training_config', {}).get('completed_plan')
    if saved is not None and _reference_batches(plans.load_epoch(saved['epoch'])) != saved['batches']:
        raise ValueError('resumed plan changed the completed epoch references or batch boundaries')


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _validate_resume_stop(start_epoch: int, stop_after_epochs: int) -> None:
    if int(start_epoch) >= int(stop_after_epochs):
        raise ValueError(
            f"stop_after_epochs={int(stop_after_epochs)} does not exceed "
            f"the resumed checkpoint's completed epochs={int(start_epoch)}"
        )


def _validate(
    model: BaselineTransformer,
    loader: DataLoader,
    inputs: TrainingInputs,
    device: torch.device,
    *, collect_metrics: bool = True,
) -> tuple[float, dict[str, Any]]:
    # DataLoader iterator creation consumes RNG even when eval has no dropout.
    # Validation must not change the following training epoch or a resumed run.
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    generator = getattr(loader, "generator", None)
    loader_rng = generator.get_state() if generator is not None else None
    modes = [(module, module.training) for module in model.modules()]
    losses: list[float] = []
    accumulator: dict[str, dict[str, float]] = {}
    try:
        model.eval()
        with torch.no_grad():
            for batch_value in loader:
                batch = _move(batch_value, device)
                with _autocast(device, getattr(model, "training_precision", "bf16")):
                    _, result = _loss_step(model, batch, inputs.vocabulary,
                                           collect_metrics=collect_metrics)
                losses.append(float(result.loss.detach().item()))
                if collect_metrics:
                    _accumulate_loss_components(accumulator, result.metrics)
        if not losses:
            raise RuntimeError("validation loader produced no batches")
        components = _finalize_loss_components(accumulator)[1] if collect_metrics else {}
        return statistics.fmean(losses), components
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        if loader_rng is not None:
            generator.set_state(loader_rng)
        for module, training in modes:
            module.training = training


def _model_and_state(
    inputs: TrainingInputs, args: Any, device: torch.device, *, resume_payload=None
) -> tuple[BaselineTransformer, dict[str, Any] | None, dict[str, Any] | None]:
    config = BaselineModelConfig(
        chart_style=getattr(args, "chart_style", "none"),
        latent_enabled=getattr(args, "latent_enabled", False),
        route_head=getattr(args, "route_head", "prototype-distance"),
        architecture=BASE_MODEL_ARCHITECTURE,
        audio_attention_preset=attention_from_args(args).preset,
        encoder_capacity_preset=capacity_from_args(args).preset,
        audio_cross_attention=cross_from_args(args),
        decoder_rope_coordinate=getattr(args, "decoder_rope_coordinate", "window-beat"),
        time_head=TIME_HEAD,
        time_components=TIME_COMPONENTS,
        decoder_layers=DECODER_LAYERS,
        decoder_d_model=256,
        decoder_heads=8,
        decoder_feedforward=1024,
    )
    from maimai_baseline.radar_contract import is_radar
    from maimai_baseline.style_contract import is_frozen, is_draft
    if not (is_frozen(config) or is_radar(config)):
        raise ValueError('A training requires frozen encoder60 mu or MajRadar conditioning')
    if config.chart_style != 'none':
        from maimai_baseline.training.a_execution import identity
        args.a_execution = identity(inputs)
    if args.resume:
        loader = load_baseline_checkpoint if resume_payload is None else _load_baseline_payload
        source = Path(args.resume).expanduser().resolve() if resume_payload is None else resume_payload
        model, payload = loader(source, route_artifact=Path(args.route_artifact),
            device=device, strict_source=True, verify_artifact_payloads=False)
        from maimai_baseline.runtime_versions import require_compatible
        args.resumed_from_version = require_compatible(payload, "training")
        validate_resume_manifest(payload, inputs.manifest)
        if config.chart_style != 'none':
            from maimai_baseline.training.a_execution import validate_resume
            validate_resume(payload, inputs)
        if payload.get("sft") is not None:
            raise ValueError("SFT checkpoint must resume through sft, not fit")
        model.difficulty_loss_preset = getattr(args, "difficulty_loss_preset", "uniform")
        model.specialized_supervision = supervision_from_args(args)
        if model.config != config:
            raise ValueError("baseline resume model configuration mismatch")
        saved = payload.get("training_config")
        if not isinstance(saved, Mapping):
            raise ValueError("baseline resume checkpoint lacks training configuration")
        _validate_resume_training_config(saved, args)
        initialization = saved.get("model_initialization")
        return (
            model,
            payload,
            dict(initialization) if isinstance(initialization, Mapping) else None,
        )
    model, initialization = _initialize_model(
        inputs.vocabulary, config, seed=getattr(args, "seed", SEED), artifacts=inputs.artifacts
    )
    if is_draft(config):
        from maimai_baseline.training.draft_import import import_draft
        imported = import_draft(model, args.draft_checkpoint, args.draft_cache,
            teacher_audio=inputs.manifest['teacher_audio'])
        initialization = dict(initialization, draft_generator=imported)
    elif is_frozen(config):
        if not getattr(args, 'chart_encoder_checkpoint', None):
            raise ValueError('fresh frozen latent A requires --chart-encoder-checkpoint epoch-060.ckpt')
        from maimai_baseline.training.encoder_import import import_encoder60
        imported = import_encoder60(model, args.chart_encoder_checkpoint)
        initialization = dict(initialization, frozen_encoder_import=imported)
    model.difficulty_loss_preset = getattr(args, "difficulty_loss_preset", "uniform")
    model.specialized_supervision = supervision_from_args(args)
    return model.to(device), None, initialization


def _write_run(
    output: Path,
    inputs: TrainingInputs,
    args: Any,
    model: BaselineTransformer,
    initialization: Mapping[str, Any] | None,
    device: torch.device,
    train_loader: DataLoader,
    validation_loader: DataLoader | None,
    total_steps: int,
) -> None:
    args.model_initialization = None if initialization is None else dict(initialization)
    from maimai_baseline.style_contract import style_model_contract
    args.baseline_manifest = inputs.manifest
    validation_enabled = getattr(inputs, "manifest", {}).get("validation", {}).get("enabled") is not False
    _json(
        output / "run.json",
        {
            **stamp(),
            "resumed_from_version": getattr(args, "resumed_from_version", None),
            "schema": TRAINING_RUN_SCHEMA,
            "architecture": ARCHITECTURE,
            "base_model_architecture": BASE_MODEL_ARCHITECTURE,
            "parameter_count": model.parameter_count(),
            "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "initial_model_state_sha256": state_dict_sha256(model.state_dict()),
            "model_config": model.config.to_dict(),
            "route_head": model.config.route_head,
            "route_geometry": inputs.manifest["route_geometry"],
            "model_initialization": args.model_initialization,
            "baseline_contract": style_model_contract(model.config, model.vocabulary),
            "baseline_manifest": inputs.manifest,
            "moment_distribution": dict(MOMENT_DISTRIBUTION),
            "duration_distribution": dict(DURATION_DISTRIBUTION),
            "difficulty_loss_weighting": inputs.manifest["difficulty_loss_weighting"],
            "source_identity": inputs.source_identity,
            "loss_contract": loss_contract(supervision_from_args(args)),
            "physical_contract": dict(PHYSICAL_CONTRACT),
            "physical_warmup_steps": _settings(args, inputs)["physical_warmup_steps"],
            "batching": dict(args.batching),
            "a_execution": getattr(args, "a_execution", None),
            "command": args.command,
            "learning_rate": _learning_rate(args),
            "recipe": recipe(args).identity(),
            "specialized_supervision_config": supervision_from_args(args).identity(),
            "audio_attention": attention_from_args(args).identity(),
            "encoder_capacity": capacity_from_args(args).identity(),
            "frequency_pooling": frequency_pooling_identity(),
            "position_conditioning": position_conditioning_identity(getattr(args, "decoder_rope_coordinate", "window-beat")),
            "audio_attention_bias": audio_attention_bias_identity(),
            "audio_fusion": audio_fusion_identity(),
            "audio_cross_attention": cross_from_args(args),
            "audio_cross_attention_identity": cross_identity(cross_from_args(args)),
            "history_representation": history_representation_identity(),
            "decoder_rope_coordinate": getattr(args, "decoder_rope_coordinate", "window-beat"),
            "encoder_capacity_initialization": (getattr(args, "model_initialization", None) or {}).get("encoder_capacity_initialization"),
            "initialization_scheme": PAIRING_SCHEME,
            "device": str(device),
            "train_segments": len(train_loader.dataset),
            "validation_segments": (
                None if args.command == "benchmark" else
                (len(validation_loader.dataset) if validation_enabled and validation_loader is not None else 0)
            ),
            "scheduler_epochs": (
                int(args.scheduler_epochs) if args.command == "fit" else None
            ),
            "scheduler_total_steps": total_steps,
            "scheduler_step_unit": recipe(args).schedule if args.command == "fit" else "optimizer-step",
            "planned_optimizer_steps": plan_steps(inputs.plan_store, int(args.stop_after_epochs), offset=getattr(args, "plan_offset", 0)) if args.command == "fit" else total_steps,
            "stop_after_epochs": (
                int(args.stop_after_epochs) if args.command == "fit" else None
            ),
            "epoch_plan_available_epochs": int(inputs.plan_store.manifest["epochs"]),
            "training_metrics_policy": {
                "schema": "maimai.baseline_training_metrics.v2",
                "validation_enabled": validation_enabled,
                "validation_every_epochs": 5 if validation_enabled else None,
                "validation_at_stop": validation_enabled,
                "detailed_metrics": "stop-after-epochs-only",
                "train_loss_reduction": "optimizer-step-mean",
                "validation_loss_reduction": "batch-mean-full-physical" if validation_enabled else None,
            } if args.command == "fit" else None,
            "runtime_optimizations": {
                "fused_adamw": bool(torch.cuda.is_available()),
                "foreach_gradient_clip": True,
                "gradient_diagnostic_interval": GRADIENT_DIAGNOSTIC_INTERVAL,
            },
        },
    )


def _save(
    path: Path,
    *,
    inputs: TrainingInputs,
    args: Any,
    model: BaselineTransformer,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    step: int,
    epoch: int,
    metrics: Mapping[str, Any],
    kind: str,
) -> None:
    training_config = {
        "encoder_capacity_preset": model.config.encoder_capacity_preset,
        **vars(args),
        "learning_rate": _learning_rate(args),
        "recipe": recipe(args).identity(),
        "specialized_supervision_config": supervision_from_args(args).identity(),
        "audio_attention": attention_from_args(args).identity(),
        "encoder_capacity": capacity_from_args(args).identity(),
        "frequency_pooling": frequency_pooling_identity(),
        "position_conditioning": position_conditioning_identity(getattr(args, "decoder_rope_coordinate", "window-beat")),
        "audio_attention_bias": audio_attention_bias_identity(),
        "audio_fusion": audio_fusion_identity(),
        "audio_cross_attention": cross_from_args(args),
        "audio_cross_attention_identity": cross_identity(cross_from_args(args)),
        "history_representation": history_representation_identity(),
        "decoder_rope_coordinate": getattr(args, "decoder_rope_coordinate", "window-beat"),
        "encoder_capacity_initialization": (getattr(args, "model_initialization", None) or {}).get("encoder_capacity_initialization"),
        "initialization_scheme": PAIRING_SCHEME,
        "model_initialization": {**(getattr(args, "model_initialization", None) or {}),
            "encoder_capacity_initialization": model.route_initialization["encoder_capacity_initialization"]},
        "weight_decay": _settings(args, inputs)["weight_decay"],
        "warmup_steps": _settings(args, inputs)["warmup_steps"],
        "physical_warmup_steps": _settings(args, inputs)["physical_warmup_steps"],
        "gradient_clip": _settings(args, inputs)["gradient_clip"],
        "checkpoint_every_epochs": CHECKPOINT_EVERY_EPOCHS,
        "gradient_diagnostic_interval": GRADIENT_DIAGNOSTIC_INTERVAL,
        "epoch_plan_available_epochs": int(inputs.plan_store.manifest["epochs"]),
    }
    if getattr(args, "sft", None) is not None:
        import math
        training_config.update(warmup_steps=max(1, math.ceil(sum(args.batching["plan_counts"]) * .05)),
                               checkpoint_every_epochs=1, scheduler_kind="linear-warmup-constant",
                               sft_physical_coefficient=1.)
    save_baseline_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        vocabulary=inputs.vocabulary,
        step=step,
        epoch=epoch,
        profile_fingerprint=str(inputs.profile_summary["profile_fingerprint"]),
        profile_summary=inputs.profile_summary,
        feature_stats=inputs.feature_stats,
        training_config=training_config,
        metrics=metrics,
        random_seed=getattr(args, "seed", SEED),
        checkpoint_kind=kind,
        loader_generator=getattr(model, '_training_loader_generator', None),
        source_identity=getattr(inputs, "source_identity", inputs.manifest.get("source_identity")),
    )


def _fit(
    *,
    args: Any,
    inputs: TrainingInputs,
    model: BaselineTransformer,
    train_loader: DataLoader,
    sampler: EpochPlanBatchSampler,
    validation_loader: DataLoader | None,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    output: Path,
    start_epoch: int,
    global_step: int,
    resume_metrics: Mapping[str, Any] | None,
    total_steps: int | None = None,
) -> int:
    validation_enabled = getattr(inputs, "manifest", {}).get("validation", {}).get("enabled") is not False
    if validation_enabled and validation_loader is None:
        raise RuntimeError("baseline fit requires a validation loader when validation is enabled")
    saved_best = (resume_metrics or {}).get("best_validation_loss")
    best = None if not validation_enabled or saved_best is None else float(saved_best)
    stale = int((resume_metrics or {}).get("stale_epochs", 0)) if validation_enabled else 0
    stop_after_epochs = int(args.stop_after_epochs)
    if total_steps is None:
        total_steps = plan_steps(inputs.plan_store, int(args.scheduler_epochs),
                                 offset=getattr(args, "plan_offset", 0))
    from maimai_baseline.style_contract import is_frozen
    style_steps = total_steps if getattr(getattr(model, "config", None), "latent_enabled", False) and not is_frozen(getattr(model, 'config', None)) else None
    model._training_loader_generator = getattr(train_loader, 'generator', None)
    for epoch in range(start_epoch, stop_after_epochs):
        epoch_started = time.perf_counter()
        style_accumulator = {}
        collect_metrics = epoch + 1 == stop_after_epochs
        validation_performed = validation_enabled and ((epoch + 1) % 5 == 0 or collect_metrics)
        sampler.set_epoch(getattr(args, "plan_offset", 0) + epoch)
        row_counts = sampler.batch_target_rows
        if len(row_counts) != len(train_loader) or not row_counts or any(n <= 0 for n in row_counts):
            raise ValueError("invalid per-batch target row schedule")
        total_rows, consumed_rows = sum(row_counts), 0
        note_rows = audio_windows = 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        losses: list[float] = []
        accumulator: dict[str, dict[str, float]] = {}
        gradient_norms: list[float] = []
        batch_iterator = iter(train_loader)
        restored = getattr(model, '_restored_loader_rng', None)
        if epoch == start_epoch and restored is not None:
            if train_loader.persistent_workers:
                train_loader.generator.set_state(restored)
            model._restored_loader_rng = None
        for batch_index in range(len(train_loader)):
            batch_started = time.perf_counter()
            batch_value = next(batch_iterator)
            input_wait_seconds = time.perf_counter() - batch_started
            from maimai_baseline.data.batch_transport import PackedBatch
            cpu_batch = batch_value.unpack() if isinstance(batch_value, PackedBatch) else batch_value
            if isinstance(cpu_batch, Mapping):
                work = batch_work(cpu_batch, inputs.vocabulary)
                if work["target_rows"] != row_counts[batch_index]:
                    raise ValueError("collated target rows differ from epoch plan")
                note_rows += work["note_rows"]
                audio_windows += work["audio_windows"]
            progress = global_step
            set_scheduler_progress(scheduler, progress)
            step_learning_rate = optimizer.param_groups[0]["lr"]
            transfer_started = time.perf_counter()
            batch = _move(batch_value, device)
            h2d_dispatch_seconds = time.perf_counter() - transfer_started
            update_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, getattr(model, "training_precision", "bf16")):
                _, result = _loss_step(model, batch, inputs.vocabulary,
                                       warmup_coefficient=_physical_coefficient(args, inputs, progress),
                                       collect_metrics=collect_metrics,
                                       **({"style_progress": global_step / style_steps} if style_steps else {}))
            result.loss.backward()
            if collect_metrics and (global_step == 0 or global_step % GRADIENT_DIAGNOSTIC_INTERVAL == 0):
                gradient_norms.append(_module_gradient_norm(model.decoder_layers))
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), _settings(args, inputs)["gradient_clip"], foreach=True, error_if_nonfinite=True
            )
            optimizer.step()
            consumed_rows += row_counts[batch_index]
            args.schedule_progress = global_step + 1
            set_scheduler_progress(scheduler, args.schedule_progress)
            global_step += 1
            losses.append(float(result.loss.detach().item()))
            update_seconds = time.perf_counter() - update_started
            if 'global_style_inputs' in cpu_batch or 'global_style_reference_count' in cpu_batch:
                from maimai_baseline.training.song_cost import write
                write(output,'A',global_step,batch_started,cpu_batch.get('global_style_inputs'),
                      reference_count=cpu_batch.get('global_style_reference_count'),
                      encoding=getattr(model,'a_encoding_stats',None),
                      timing=dict(input_wait_seconds=input_wait_seconds,
                                  cpu_prepare_seconds=getattr(batch_value, 'cpu_prepare_seconds', None),
                                  h2d_dispatch_seconds=h2d_dispatch_seconds,
                                  update_seconds=update_seconds))
            if style_steps and collect_metrics:
                _accumulate_loss_components(style_accumulator,
                    {k: v for k, v in result.metrics.items() if k.startswith("style:")})
            if collect_metrics:
                _accumulate_loss_components(accumulator, result.metrics)
            if len(losses) == 1 or len(losses) % 50 == 0:
                print(json.dumps({
                    "event": "training_progress", "epoch": epoch, "step": global_step,
                    "epoch_step": len(losses), "epoch_steps": len(train_loader),
                    "loss": losses[-1],
                    "physical_coefficient": _physical_coefficient(args, inputs, progress),
                    "schedule_progress": progress, "consumed_target_rows": consumed_rows,
                    "learning_rate": step_learning_rate,
                    "train_elapsed_seconds": time.perf_counter() - epoch_started,
                    **({"style_kl": result.metrics["style:kl"]} if style_steps and collect_metrics else {}),
                }, sort_keys=True), flush=True)
        train_seconds = time.perf_counter() - epoch_started
        train_peak_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        if not losses or consumed_rows != total_rows or len(losses) != len(row_counts):
            raise RuntimeError("training loader did not consume the complete epoch plan")
        from maimai_baseline.radar_contract import is_radar
        if is_radar(getattr(model, 'config', None)):
            model.style_prior.ready.fill_(True)
        train_loss = statistics.fmean(losses)
        train_components = _finalize_loss_components(accumulator)[1] if collect_metrics else None
        validation_loss, validation_components = None, None
        validation_seconds = 0.0
        improved = False
        if validation_performed:
            validation_started = time.perf_counter()
            validation_loss, validation_components = _validate(
                model, validation_loader, inputs, device, collect_metrics=collect_metrics,
            )
            validation_seconds = time.perf_counter() - validation_started
            improved = best is None or validation_loss < best
            if improved:
                best, stale = validation_loss, 0
            else:
                stale += 1
        metrics = {
            "schema": "maimai.baseline_training_metrics.v2",
            "epoch": epoch,
            "step": global_step,
            "train_loss": train_loss,
            "optimizer_step_loss_mean": statistics.fmean(losses),
            "validation_loss": validation_loss,
            "best_validation_loss": best,
            "stale_epochs": stale,
            "loss": ("L3-hierarchical+Physical-dual-v6+MajRadar-GMM4-NLL" if is_radar(getattr(model, 'config', None))
                     else "L3-hierarchical+Physical-dual-v6+style-KL" if style_steps else "L3-hierarchical+Physical-dual-v6"),
            "loss_scheme": PHYSICAL_LOSS_ID,
            "base_loss_scheme": L3_LOSS_ID,
            "validation_performed": validation_performed,
            "collect_metrics": collect_metrics,
            "train_loss_reduction": "optimizer-step-mean",
            "validation_loss_reduction": "batch-mean-full-physical" if validation_enabled else None,
            "stale_epochs_unit": "validation-checks",
            "shared_decoder_gradient_norm": (
                statistics.fmean(gradient_norms) if gradient_norms else 0.0
            ),
            "train_batch_count": len(train_loader),
            "schedule_progress": args.schedule_progress,
            "schedule_unit": recipe(args).schedule,
            "target_rows": consumed_rows,
            "target_rows_per_second": consumed_rows / train_seconds,
            "note_rows": note_rows, "note_rows_per_second": note_rows / train_seconds,
            "audio_window_evaluations": audio_windows,
            "mean_audio_windows_per_batch": audio_windows / len(losses),
            "mean_step_seconds": train_seconds / len(losses),
            "peak_cuda_bytes": train_peak_bytes,
            "recipe": recipe(args).identity(),
            "specialized_supervision_config": supervision_from_args(args).identity(),
            "audio_attention": attention_from_args(args).identity(),
            "encoder_capacity": capacity_from_args(args).identity(),
            "frequency_pooling": frequency_pooling_identity(),
            "position_conditioning": position_conditioning_identity(getattr(args, "decoder_rope_coordinate", "window-beat")),
            "audio_attention_bias": audio_attention_bias_identity(),
            "audio_fusion": audio_fusion_identity(),
            "audio_cross_attention": cross_from_args(args),
            "audio_cross_attention_identity": cross_identity(cross_from_args(args)),
            "history_representation": history_representation_identity(),
            "decoder_rope_coordinate": getattr(args, "decoder_rope_coordinate", "window-beat"),
            "encoder_capacity_initialization": (getattr(args, "model_initialization", None) or {}).get("encoder_capacity_initialization"),
            "initialization_scheme": PAIRING_SCHEME,
            "validation_batch_count": len(validation_loader) if validation_performed and validation_loader is not None else 0,
            "scheduler_epochs": int(args.scheduler_epochs),
            "scheduler_total_steps": total_steps,
            "stop_after_epochs": stop_after_epochs,
            "epoch_plan_available_epochs": int(inputs.plan_store.manifest["epochs"]),
            "epoch_seconds": time.perf_counter() - epoch_started,
            "train_seconds": train_seconds,
            "validation_seconds": validation_seconds,
            "batching": dict(args.batching),
            **({"style": _finalize_loss_components(style_accumulator)[1]["style"]} if style_steps and collect_metrics else {}),
        }
        if collect_metrics:
            metrics.update(loss_components=train_components,
                           validation_loss_components=validation_components)
        save_started = time.perf_counter()
        consumed_batches = getattr(sampler, '_batches', None)
        if consumed_batches is not None:
            args.completed_plan = dict(epoch=getattr(args, 'plan_offset', 0) + epoch,
                                       batches=_reference_batches(consumed_batches))
        _save(
            output / "last.ckpt",
            inputs=inputs, args=args, model=model, optimizer=optimizer,
            scheduler=scheduler, step=global_step, epoch=epoch,
            metrics=metrics, kind="training",
        )
        if improved:
            _save(
                output / "best.ckpt",
                inputs=inputs, args=args, model=model, optimizer=optimizer,
                scheduler=scheduler, step=global_step, epoch=epoch,
                metrics=metrics, kind="training",
            )
        if (
            (epoch + 1) % CHECKPOINT_EVERY_EPOCHS == 0
            or epoch + 1 == stop_after_epochs
        ):
            _save(
                output / "checkpoints" / f"epoch-{epoch + 1:03d}.ckpt",
                inputs=inputs, args=args, model=model, optimizer=optimizer if is_frozen(model.config) else None,
                scheduler=scheduler if is_frozen(model.config) else None, step=global_step, epoch=epoch,
                metrics=metrics, kind="training" if is_frozen(model.config) else "inference",
            )
        metrics.update(save_seconds=time.perf_counter()-save_started,
                       epoch_seconds=time.perf_counter()-epoch_started,
                       optimizer_updates=len(losses))
        with (output / "metrics.jsonl").open(
            "a", encoding="utf-8", newline="\n"
        ) as stream:
            stream.write(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n")
        print(json.dumps(metrics, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


def _cycle(loader: DataLoader):
    while True:
        yield from loader


def _benchmark(
    *,
    args: Any,
    inputs: TrainingInputs,
    model: BaselineTransformer,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    output: Path,
) -> int:
    warmup = int(args.benchmark_warmup_steps)
    measured = int(args.steps)
    durations: list[float] = []
    losses: list[float] = []
    metric_accumulator: dict[str, dict[str, float]] = {}
    work_totals = dict(target_rows=0, note_rows=0, audio_windows=0)
    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    for step, batch_value in zip(range(warmup + measured), _cycle(loader), strict=False):
        batch = _move(batch_value, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, getattr(model, "training_precision", "bf16")):
            _, result = _loss_step(model, batch, inputs.vocabulary)
        result.loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), _settings(args, inputs)["gradient_clip"], foreach=True, error_if_nonfinite=True
        )
        optimizer.step()
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if step >= warmup:
            from maimai_baseline.data.batch_transport import PackedBatch
            work = batch_work(batch_value.unpack() if isinstance(batch_value, PackedBatch) else batch_value,
                              inputs.vocabulary)
            for key in work_totals:
                work_totals[key] += work[key]
            durations.append(time.perf_counter() - started)
            losses.append(float(result.loss.detach().item()))
            _accumulate_loss_components(metric_accumulator, result.metrics)
    _, components = _finalize_loss_components(metric_accumulator)
    report = {
        "schema": "maimai.baseline_benchmark.v1",
        "architecture": ARCHITECTURE,
        "parameter_count": model.parameter_count(),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "warmup_steps": warmup,
        "steps": measured,
        "median_step_seconds": statistics.median(durations),
        "p95_step_seconds": float(np.percentile(np.asarray(durations), 95)),
        "mean_loss": statistics.fmean(losses),
        "recipe": recipe(args).identity(),
        "specialized_supervision_config": supervision_from_args(args).identity(),
        "audio_attention": attention_from_args(args).identity(),
        "encoder_capacity": capacity_from_args(args).identity(),
        "frequency_pooling": frequency_pooling_identity(),
        "position_conditioning": position_conditioning_identity(getattr(args, "decoder_rope_coordinate", "window-beat")),
        "audio_attention_bias": audio_attention_bias_identity(),
        "audio_fusion": audio_fusion_identity(),
        "audio_cross_attention": cross_from_args(args),
        "audio_cross_attention_identity": cross_identity(cross_from_args(args)),
        "history_representation": history_representation_identity(),
        "decoder_rope_coordinate": getattr(args, "decoder_rope_coordinate", "window-beat"),
        "encoder_capacity_initialization": (getattr(args, "model_initialization", None) or {}).get("encoder_capacity_initialization"),
        "initialization_scheme": PAIRING_SCHEME,
        "work_totals": work_totals,
        "target_rows_per_second": work_totals["target_rows"] / sum(durations),
        "note_rows_per_second": work_totals["note_rows"] / sum(durations),
        "mean_audio_windows_per_batch": work_totals["audio_windows"] / measured,
        "loss_components": components,
        "physical_warmup_coefficient": 1.0,
        "physical_dynamic_h2d_bytes_last_step": result.metrics["physical:total"].get("cache_h2d_bytes",0),
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device),
        "first_load_and_jit_excluded": True,
        "device": str(device),
    }
    _json(output / "benchmark.json", report)
    _save(
        output / "benchmark.ckpt",
        inputs=inputs, args=args, model=model, optimizer=None, scheduler=None,
        step=warmup + measured, epoch=0, metrics=report, kind="inference",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


def preflight(args: Any) -> dict[str, Any]:
    bind_stage_recipe(args, "A")
    from maimai_baseline.training.paths import resolve_training_paths
    resolve_training_paths(args, "A")
    inputs = _prepare_inputs(args)
    from maimai_baseline.style_contract import style_identity, style_parameter_count
    from maimai_baseline.dual_contract import style_parameter_delta
    contract = baseline_contract(capacity_from_args(args), getattr(args, "decoder_rope_coordinate", "window-beat"), getattr(args, "route_head", "prototype-distance"))
    if getattr(args, "chart_style", "none") != "none":
        contract["chart_style"] = style_identity(args.chart_style)
        contract["parameter_count"] += style_parameter_count(inputs.vocabulary) + style_parameter_delta()
    return {
        "validated": True,
        "learning_rate": _learning_rate(args),
        "recipe": recipe(args).identity(),
        "specialized_supervision_config": supervision_from_args(args).identity(),
        "audio_attention": attention_from_args(args).identity(),
        "encoder_capacity": capacity_from_args(args).identity(),
        "frequency_pooling": frequency_pooling_identity(),
        "position_conditioning": position_conditioning_identity(getattr(args, "decoder_rope_coordinate", "window-beat")),
        "audio_attention_bias": audio_attention_bias_identity(),
        "audio_fusion": audio_fusion_identity(),
        "audio_cross_attention": cross_from_args(args),
        "audio_cross_attention_identity": cross_identity(cross_from_args(args)),
        "history_representation": history_representation_identity(),
        "decoder_rope_coordinate": getattr(args, "decoder_rope_coordinate", "window-beat"),
        "encoder_capacity_initialization": (getattr(args, "model_initialization", None) or {}).get("encoder_capacity_initialization"),
        "initialization_scheme": PAIRING_SCHEME,
        "cuda_qualified": False,
        "validation_scope": "runtime-metadata",
        "input_preparation_seconds": args.input_preparation_seconds,
        "architecture": ARCHITECTURE,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "training_run_schema": TRAINING_RUN_SCHEMA,
        "moment_distribution": dict(MOMENT_DISTRIBUTION),
        "duration_distribution": dict(DURATION_DISTRIBUTION),
        "baseline_contract": contract,
        "baseline_manifest": inputs.manifest,
        "scheduler_epochs": int(args.scheduler_epochs),
        "scheduler_total_steps": (
            plan_steps(inputs.plan_store, int(args.scheduler_epochs), offset=getattr(args, "plan_offset", 0))
        ),
        "stop_after_epochs": int(args.stop_after_epochs),
        "epoch_plan_available_epochs": int(inputs.plan_store.manifest["epochs"]),
    }


def run_training(args: Any) -> int:
    bind_stage_recipe(args, "A")
    from maimai_baseline.training.paths import resolve_training_paths
    resolve_training_paths(args, "A")
    device = _device(args.device)
    if device.type != "cuda":
        raise RuntimeError("baseline Physical v5 production requires CUDA; CPU is test-only")
    resume_payload = None
    if args.resume:
        # Read once on CPU: input capacity and augmentation must see the saved
        # schedule before preparation; model construction reuses this payload.
        resume_payload = torch.load(Path(args.resume).expanduser().resolve(),
                                    map_location="cpu", weights_only=False)
        saved = resume_payload.get("training_config") if isinstance(resume_payload, Mapping) else None
        if not isinstance(saved, Mapping):
            raise ValueError("baseline resume checkpoint lacks training configuration")
        _validate_resume_training_config(saved, args)
    inputs = _prepare_inputs(args)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Resume settings were restored before input preparation. Bind them into
    # the dataset metadata without moving model initialization ahead of loaders.
    experiment = dict(inputs.manifest["experiment"])
    runtime_recipe = dict(experiment["recipe"], seed=args.seed)
    runtime_recipe["stages"] = {
        **runtime_recipe.get("stages", {}), "A": dict(args.resolved_recipe)
    }
    experiment["recipe"] = runtime_recipe
    inputs.manifest["experiment"] = experiment
    train_loader, sampler, validation_loader = _loaders(inputs, args)
    # Preserve the original fresh-run construction order and final model RNG
    # reset. Loader iteration (and worker startup) still happens inside _fit.
    model, resume_payload, initialization = _model_and_state(
        inputs, args, device, resume_payload=resume_payload)
    if model.parameter_count() != model.expected_parameter_count():
        raise RuntimeError(f"baseline parameter count drifted: {model.parameter_count()}")
    total_steps = (
        plan_steps(inputs.plan_store, int(args.scheduler_epochs), offset=getattr(args, "plan_offset", 0))
        if args.command == "fit"
        else int(args.steps) + int(args.benchmark_warmup_steps)
    )
    model.training_precision = args.precision
    optimizer = _optimizer(model, learning_rate=_learning_rate(args), weight_decay=args.weight_decay)
    scheduler = _scheduler(
        optimizer, warmup_steps=_settings(args, inputs)["warmup_steps"], total_steps=total_steps
    )
    start_epoch = global_step = 0
    if resume_payload is not None:
        _validate_completed_plan(resume_payload, inputs.plan_store)
        start_epoch, global_step = restore_baseline_training_state(
            resume_payload, optimizer=optimizer, scheduler=scheduler, loader_generator=train_loader.generator
        )
        if train_loader.generator is not None:
            model._restored_loader_rng = train_loader.generator.get_state().clone()
        _validate_resume_stop(start_epoch, int(args.stop_after_epochs))
        expected_progress = plan_steps(inputs.plan_store, start_epoch, offset=getattr(args, "plan_offset", 0))
        if (resume_payload["training_config"].get("schedule_progress") != expected_progress
                or scheduler.last_epoch != expected_progress
                or global_step != expected_progress):
            raise ValueError("resume data progress or optimizer step mismatch")
    _write_run(
        output, inputs, args, model, initialization, device,
        train_loader, validation_loader, total_steps,
    )
    if args.command == "benchmark":
        return _benchmark(
            args=args, inputs=inputs, model=model, loader=train_loader,
            device=device, optimizer=optimizer, scheduler=scheduler, output=output,
        )
    return _fit(
        args=args, inputs=inputs, model=model, train_loader=train_loader,
        sampler=sampler, validation_loader=validation_loader, device=device,
        optimizer=optimizer, scheduler=scheduler, output=output,
        start_epoch=start_epoch, global_step=global_step,
        total_steps=total_steps,
        resume_metrics=(
            None if resume_payload is None else resume_payload.get("metrics")
        ),
    )
