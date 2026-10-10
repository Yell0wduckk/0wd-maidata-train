"""Master-only SFT orchestration, distinct from pretraining resume."""
from __future__ import annotations

from functools import partial
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from maimai_baseline.contract import SEED, WEIGHT_DECAY, GRADIENT_CLIP
from maimai_baseline.data.curriculum import VirtualSegmentRef, sha256_file
from maimai_baseline.physical.data import collate_physical_segments
from maimai_baseline.training.inputs import _prepare_inputs, _dataset, _ValidationTokenBudgetSampler
from maimai_baseline.training.steps import _device, _move, _autocast, _loss_step
from maimai_baseline.training.recipe import recipe
from maimai_baseline.training.sft_rollout import choose_blocks, replace_objective
from maimai_baseline.training import sft_batch


SFT_CONTRACT = {
    "schema": "maimai.single_moment_sft.v1", "scope": "source-master-pure-tap",
    "selection": "sha256-epoch-segment-nonoverlap-v1", "feedback": "greedy-detached-training-dropout",
    "note_attributes": "gt-slot-order-extra-plain-tap", "end_delta": "absolute-gt-once",
    "delta_weights": [.5, .5], "extra_structure": "sum-fixed-gt-denominator",
    "invalid_end": "zero-without-renormalization", "physical": "original-gt-full-coefficient",
    "limit": "notes-plus-terminal-query-no-summary", "missing_rows": "zero-fixed-gt-denominator",
}


def sft_config(args):
    value = {"contract": SFT_CONTRACT, "mode": args.sft_mode,
             "rollout_fraction": float(args.rollout_fraction), "max_rollout_notes": int(args.max_rollout_notes),
             "learning_rate": float(args.learning_rate), "encoder_learning_rate": float(args.encoder_learning_rate),
             "epochs": int(args.stop_after_epochs), "warmup_fraction": .05, "seed": SEED,
             "token_budget": recipe(args).token_budget, "execution": sft_batch.EXECUTION}
    validate_sft_config(value)
    return value


def validate_sft_config(value, *, historical=False):
    if not isinstance(value, dict) or (not historical and value.get("contract") != SFT_CONTRACT):
        raise ValueError("SFT loss/rollout contract mismatch")
    if "execution" in value and value["execution"] != sft_batch.EXECUTION:
        raise ValueError("SFT execution contract mismatch")
    if value.get("mode") not in ("teacher-forcing", "moment-rollout"):
        raise ValueError("invalid SFT mode")
    if not math.isfinite(value.get("rollout_fraction", float("nan"))) or not 0 <= value["rollout_fraction"] <= 1:
        raise ValueError("invalid SFT rollout fraction")
    for name in ("max_rollout_notes", "epochs", "token_budget"):
        if not isinstance(value.get(name), int) or value[name] <= 0:
            raise ValueError(f"invalid SFT {name}")
    for name in ("learning_rate", "encoder_learning_rate"):
        if not math.isfinite(value.get(name, 0)) or value.get(name, 0) <= 0:
            raise ValueError(f"invalid SFT {name}")
    if value.get("seed") != SEED or value.get("warmup_fraction") != .05:
        raise ValueError("SFT seed/schedule contract mismatch")


def validate_checkpoint_sft(payload):
    value = payload.get("sft")
    training = payload.get("training_config", {}).get("sft")
    manifest = payload.get("baseline_manifest", {}).get("sft")
    if value is None:
        if training is not None or manifest is not None:
            raise ValueError("checkpoint lacks SFT identity")
        return
    from maimai_baseline.runtime_versions import compatible_metadata
    validate_sft_config(value["config"], historical=True)
    if not compatible_metadata(value, manifest) or (payload.get("checkpoint_kind") == "training" and not compatible_metadata(value, training)):
        raise ValueError("checkpoint SFT identity mismatch")
    if not value.get("master_view", {}).get("sha256"):
        raise ValueError("checkpoint lacks SFT data provenance")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class MasterSampler(Sampler):
    """One visit per Master segment; complete audio groups and whole-chart views."""
    def __init__(self, dataset, budget=3072):
        self.dataset, self.budget = dataset, budget
        self.indices = [i for i, row in enumerate(dataset.segments)
                        if int(row.get("source_difficulty_id", row.get("difficulty_id", -1))) == 5]
        if not self.indices:
            raise ValueError("training selection has no source Master segments")
        ids = [str(dataset.segments[i]["segment_id"]) for i in self.indices]
        self.identity = {"schema": "maimai.sft_master_view.v1", "source_difficulty_id": 5,
                         "segment_ids": ids, "sha256": digest(ids)}
        self.set_epoch(0)

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.dataset.set_epoch(epoch)
        groups = {}
        for i in self.indices:
            row = self.dataset.segments[i]
            key = str(row["audio_group_key"])
            groups.setdefault(key, []).append(i)
        order = sorted(groups, key=lambda key: digest([SEED, epoch, key]))
        batches, pending, used = [], [], 0
        for key in order:
            refs = []
            for i in groups[key]:
                row = self.dataset.segments[i]
                view = int(digest([SEED, epoch, row["song_id"], row.get("source_difficulty_id", 5)])[:8], 16) % 4
                refs.append(VirtualSegmentRef(i, view, 0))
            cost = sum(max(1, int(self.dataset.target_lengths[r.canonical_index])) for r in refs)
            if pending and used + cost > self.budget:
                batches.append(pending)
                pending, used = [], 0
            pending.extend(refs)
            used += cost
        if pending:
            batches.append(pending)
        self.batches = batches
        self.plan_identity = digest([[(r.canonical_index, r.view_id) for r in b] for b in batches])

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def optimizer_for(model, config):
    audio, other = [], []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(True)
        (audio if name.startswith("audio_encoder.") else other).append(parameter)
    return torch.optim.AdamW([
        {"params": audio, "lr": config["encoder_learning_rate"], "name": "encoder"},
        {"params": other, "lr": config["learning_rate"], "name": "decoder-and-heads"}],
        weight_decay=WEIGHT_DECAY, fused=next(model.parameters()).is_cuda)


def scheduler_for(optimizer, total_steps):
    warmup = max(1, math.ceil(total_steps * .05))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1., (step+1)/warmup))


def sft_step(model, batch, vocabulary, *, epoch, config, collect_trace=False, base_step=_loss_step):
    from maimai_baseline.loss.budget import reject_legacy_sft
    reject_legacy_sft()
    positions = {}
    outputs, result = base_step(model, dict(batch, l3_position_cache=positions), vocabulary,
                                warmup_coefficient=1., collect_metrics=False)
    if 'sft_selected' in batch:
        blocks, eligible = batch['sft_selected']
        prepared = batch['sft_prepared']
    else:
        blocks, eligible = choose_blocks(batch, vocabulary, epoch=epoch,
            fraction=config['rollout_fraction'], seed=config['seed'])
        prepared = _move(sft_batch.prepare(batch,vocabulary,blocks),batch['row_anchor_tick'].device)
    generated = None
    if blocks and config['mode']=='moment-rollout':
        generated = sft_batch.rollout_batch(model,batch,outputs,prepared,
            max_notes=config['max_rollout_notes'],collect_trace=collect_trace)
    result = sft_batch.replace(result,positions,prepared,generated) if blocks else result
    reports = generated['reports'] if generated is not None else []
    metrics = {"eligible": eligible, "selected": len(blocks), "blocks": len(reports),
               "max_rollout_depth": generated["steps"] if generated is not None else 0,
               "generated_rows": sum(r["generated_notes"]+2 for r in reports),
               **{reason: sum(r["reason"] == reason for r in reports) for reason in ("natural", "early", "limit")},
               **{name: sum(r[name] for r in reports) for name in ("generated_notes", "extra_notes", "unvisited_notes", "missing_notes")},
               "invalid_end": sum(not r["end_time_valid"] for r in reports),
               "start_error_ms_sum": sum(r["start_error_ms"] for r in reports),
               "end_error_ms_sum": sum(r["end_error_ms"] or 0 for r in reports),
               "valid_end": sum(r["end_time_valid"] for r in reports)}
    return result, metrics, reports


def train_epoch(model, loader, optimizer, scheduler, *, epoch, config, device,
                step_function=sft_step, trace_limit=2):
    model.train()
    started = time.perf_counter()
    totals, losses, components, traces = {}, [], {}, []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for index, value in enumerate(loader):
        batch_started = time.perf_counter()
        blocks,eligible = choose_blocks(value,model.vocabulary,epoch=epoch,
            fraction=config['rollout_fraction'],seed=config['seed'])
        value = dict(value,sft_selected=(blocks,eligible),
                     sft_prepared=sft_batch.prepare(value,model.vocabulary,blocks))
        batch = _move(value, device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device):
            result, metrics, report = step_function(model, batch, model.vocabulary,
                epoch=epoch, config=config, collect_trace=len(traces) < trace_limit)
        if not bool(torch.isfinite(result.loss)):
            raise FloatingPointError("non-finite SFT loss")
        result.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        losses.append(float(result.loss.detach()))
        for key, val in metrics.items():
            totals[key] = max(totals.get(key,0),val) if key=="max_rollout_depth" else totals.get(key, 0) + val
        for key, val in result.components.items():
            components[key] = components.get(key, 0.) + float(val.detach())
        if len(traces) < trace_limit:
            traces.extend(report[:trace_limit-len(traces)])
        print(json.dumps(dict(event="sft_progress", epoch=epoch, batch=index+1,
                                  loss=losses[-1], elapsed_seconds=time.perf_counter()-started,
                                  batch_seconds=time.perf_counter()-batch_started, **metrics,
                                  peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)), flush=True)
    if not losses:
        raise ValueError("empty SFT epoch")
    rates = {f"{key}_rate": totals.get(key,0)/max(1,totals.get("blocks",0))
             for key in ("natural", "early", "limit", "invalid_end")}
    rates["selected_fraction"] = totals.get("selected",0)/max(1,totals.get("eligible",0))
    return dict(train_loss=statistics.fmean(losses), steps=len(losses), totals=totals, rates=rates,
                components={k: v/len(losses) for k,v in components.items()},
                seconds=time.perf_counter()-started,
                peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None), traces


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def run_sft(args):
    from maimai_baseline.loss.budget import reject_legacy_sft
    reject_legacy_sft()
    from maimai_baseline.checkpoint import load_baseline_checkpoint, restore_baseline_training_state, validate_resume_manifest
    from maimai_baseline.training.runner import _save, _validate
    from maimai_baseline.physical.checker import load_cuda_extension
    device = _device(args.device)
    if device.type != "cuda":
        raise RuntimeError("SFT Physical production requires CUDA; CPU is fixture-test-only")
    if args.difficulty_loss_preset != "uniform" or args.specialized_supervision:
        raise ValueError("SFT v1 requires uniform difficulty and no specialized supervision")
    config = sft_config(args)
    if args.scheduler_epochs != args.stop_after_epochs:
        raise ValueError("SFT v1 scheduler horizon must equal its fixed traversal count")
    output = Path(args.output).resolve()
    if not args.resume and output.exists():
        raise ValueError(f"refusing to overwrite SFT output: {output}")
    parent_path = Path(args.resume or args.init_checkpoint).resolve()
    model, payload = load_baseline_checkpoint(parent_path, route_artifact=Path(args.route_artifact),
        device=device, strict_source=bool(args.resume), verify_artifact_payloads=False)
    if model.config.chart_style != "none":
        raise ValueError("style models use stage A fit and stage B native AR; SFT is outside this experiment")
    args.decoder_rope_coordinate = model.config.decoder_rope_coordinate
    args.encoder_capacity_preset = model.config.encoder_capacity_preset
    args.audio_cross_attention = model.config.audio_cross_attention
    args.audio_attention_preset = model.config.audio_attention.preset
    inputs = _prepare_inputs(args)
    if inputs.manifest.get("validation", {}).get("enabled") is False:
        raise ValueError(
            "SFT requires an enabled validation split; the full train-only dataset is fit-only"
        )
    if model.vocabulary.to_dict() != inputs.vocabulary.to_dict() or payload["profile_fingerprint"] != inputs.profile_summary["profile_fingerprint"]:
        raise ValueError("SFT parent vocabulary/profile mismatch")
    if payload["bpm_input"] != inputs.manifest["bpm_input"]:
        raise ValueError("SFT parent BPM/audio input mismatch")
    train_dataset, validation_dataset = _dataset(inputs, "train"), _dataset(inputs, "validation")
    sampler = MasterSampler(train_dataset, recipe(args).token_budget)
    plan_counts, plan_ids = [], []
    for epoch in range(config["epochs"]):
        sampler.set_epoch(epoch)
        plan_counts.append(len(sampler))
        plan_ids.append(sampler.plan_identity)
    parent = payload.get("sft", {}).get("parent") if args.resume else {
        "path": str(parent_path), "sha256": sha256_file(parent_path), "source_identity": payload.get("source_identity"),
        "epoch": payload["epoch"], "step": payload["step"]}
    identity = {"config": config, "parent": parent, "master_view": sampler.identity,
                "epoch_plans": plan_ids}
    inputs.manifest["sft"] = identity
    if args.resume:
        from maimai_baseline.runtime_versions import compatible_metadata
        if not compatible_metadata(payload.get("sft"), identity):
            raise ValueError("SFT resume configuration/data/parent mismatch")
        validate_resume_manifest(payload, inputs.manifest)
    model.difficulty_loss_preset = "uniform"
    from maimai_baseline.loss.supervision import from_args
    model.specialized_supervision = from_args(args)
    def loader(dataset, batch_sampler):
        return DataLoader(dataset, batch_sampler=batch_sampler,
            collate_fn=partial(collate_physical_segments, vocabulary=inputs.vocabulary,
                physical_store=inputs.physical_store, audio_loader=dataset.audio_item),
            num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
            pin_memory=True, generator=torch.Generator().manual_seed(SEED))
    train_loader = loader(train_dataset, sampler)
    validation_loader = loader(validation_dataset, _ValidationTokenBudgetSampler(validation_dataset))
    optimizer = optimizer_for(model, config)
    scheduler = scheduler_for(optimizer, sum(plan_counts))
    start, step = 0, 0
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    if args.resume:
        start, step = restore_baseline_training_state(payload, optimizer=optimizer, scheduler=scheduler)
        if start >= config["epochs"] or step != sum(plan_counts[:start]) or scheduler.last_epoch != step:
            raise ValueError("SFT resume progress mismatch or already complete")
    args.sft = identity
    args.model_initialization = payload["initialization"]
    args.baseline_manifest = inputs.manifest
    args.batching = {"policy": "sft-master-audio-groups-v1", "token_budget": sampler.budget,
                     "plan_counts": plan_counts, "plan_ids": plan_ids}
    output.mkdir(parents=True, exist_ok=True)
    _json(output / "master-view.json", sampler.identity)
    _json(output / "run.json", {"schema": "maimai.sft_run.v1", "sft": identity,
        "batching": args.batching, "source_identity": inputs.source_identity,
        "baseline_manifest": inputs.manifest, "model_config": model.config.to_dict(),
        "parameter_count": model.parameter_count(), "resume_from": str(parent_path) if args.resume else None})
    for epoch in range(start, config["epochs"]):
        sampler.set_epoch(epoch)
        # Worker seeds are epoch-specific, reproducible across an epoch-boundary resume.
        train_loader.generator.manual_seed(SEED + epoch)
        metrics, traces = train_epoch(model, train_loader, optimizer, scheduler,
            epoch=epoch, config=config, device=device)
        step += metrics["steps"]
        metrics["validation_loss"], _ = _validate(model, validation_loader, inputs, device, collect_metrics=False)
        metrics.update(epoch=epoch, step=step, schema="maimai.sft_metrics.v1")
        args.schedule_progress = step
        for destination, kind in ((output/"last.ckpt", "training"),
                                  (output/"checkpoints"/f"epoch-{epoch+1:03d}.ckpt", "inference")):
            destination.parent.mkdir(parents=True, exist_ok=True)
            _save(destination, inputs=inputs, args=args, model=model,
                optimizer=optimizer if kind == "training" else None,
                scheduler=scheduler if kind == "training" else None,
                step=step, epoch=epoch, metrics=metrics, kind=kind)
        _json(output/f"trace-{epoch+1:03d}.json", traces)
        with (output/"metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics)+"\n")
    return 0
