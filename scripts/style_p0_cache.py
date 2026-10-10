"""Build P0 once from frozen A; never read chart row payloads or refit targets."""
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from global_local_cache import CacheBatches, cache_dataset, worker_init
from maimai_baseline.checkpoint import load_frozen_stage_a
from maimai_baseline.data.style_cache import StyleLatentCache
from maimai_baseline.data.style_audio_cache import AudioConditionWriter, record_key
from maimai_baseline.source_identity import baseline_source_identity
from maimai_baseline.training.inputs import _prepare_inputs, _ValidationTokenBudgetSampler
from maimai_baseline.training.steps import _device, _move
from maimai_dataset.integrity import file_sha256
from maimai_dataset.manifest import canonical_json_sha256


def cache_mapping(source, target):
    lookup = {record_key(r): i for i, r in enumerate(target.records)}
    if len(lookup) != len(target.records) or len(source.records) != len(target.records):
        raise ValueError("P0/target record inventory mismatch")
    mapping = [-1] * len(target.records)
    keys = []
    for batch in source.batches:
        for key, rows in batch:
            index = len(keys)
            keys.append(key)
            for row in rows:
                record = source.records[row]
                target_index = lookup[record_key(record)]
                saved = target.records[target_index]
                if any(record[k] != saved[k] for k in ("canonical_index", "start_tick", "end_tick", "difficulty_id")):
                    raise ValueError("P0/target window coordinates mismatch")
                if mapping[target_index] != -1:
                    raise ValueError("duplicate P0/target record")
                mapping[target_index] = index
    if -1 in mapping:
        raise ValueError("incomplete P0/target mapping")
    validation = []
    if "validation" in source.datasets:
        dataset = source.datasets["validation"]
        for batch in _ValidationTokenBudgetSampler(dataset):
            validation.append([lookup[("validation", str(dataset.segments[i]["segment_id"]), 0)] for i in batch])
    return keys, mapping, validation


@torch.no_grad()
def prepare(args, *, token_budget=4096, workers=12, cpu_threads=1):
    if workers < 0 or cpu_threads < 1:
        raise ValueError("invalid P0 cache worker/thread settings")
    torch.set_num_threads(cpu_threads)
    target = StyleLatentCache(args.cache)
    parent_hash = file_sha256(Path(args.stage_a_checkpoint))
    # Model/source digests are provenance only.
    model, parent = load_frozen_stage_a(Path(args.stage_a_checkpoint), route_artifact=Path(args.route_artifact))
    inputs = _prepare_inputs(args)
    datasets = {role:cache_dataset(inputs, role) for role in ("train", "validation")
                if role == "train" or inputs.manifest["validation"]["enabled"]}
    source = CacheBatches(datasets, inputs.vocabulary, token_budget, audio_only=True,song_audio_cache=args.song_audio_cache)
    keys, mapping, validation = cache_mapping(source, target)
    bindings = dict(stage_a_sha256=parent_hash, target_cache_identity=target.manifest["manifest_identity_sha256"],
        records_sha256=canonical_json_sha256(target.records), baseline_manifest=parent["baseline_manifest"],
        dataset_manifest_sha256=file_sha256(Path(args.dataset_manifest)),
        plan_manifest_sha256=target.manifest["bindings"]["plan_manifest_sha256"],
        validation_sampler="baseline-identity-complete-audio-group-2048")
    summary = dict(operation="style-p0-cache", groups=len(keys), records=len(mapping), written=False)
    if args.dry_run:
        return summary
    writer = AudioConditionWriter(args.output, keys=keys, record_to_audio=mapping,
        validation_batches=validation, bindings=bindings, producer=dict(source_identity=baseline_source_identity(),
        script_sha256=file_sha256(Path(__file__)), workers=workers, token_budget=token_budget))
    device = _device(args.device)
    model.to(device).eval().requires_grad_(False)
    loader = DataLoader(source, batch_size=None, num_workers=workers, worker_init_fn=worker_init,
        generator=torch.Generator().manual_seed(20260811), pin_memory=device.type == "cuda",
        **({"prefetch_factor": 2} if workers else {}))
    for i, row in enumerate(loader):
        batch = _move(row["batch"], device)
        from maimai_baseline.data.song_audio import encode_batch
        full=encode_batch(model,batch["song_audio_inputs"])
        audio = model.encode_audio(batch["audio"], batch["audio_padding_mask"], batch["frame_beats"],
            batch["frame_bpms"], start_tick=batch["audio_start_tick"],
            interpolation_geometry=batch["interpolation_geometry"],song_audio=full)
        writer.append(row["audio_keys"], dict(audio=audio.memories[-1].float(),
            audio_valid=~audio.padding_masks[-1],
            audio_beats=audio.beats[-1].float() - batch["audio_start_tick"].float()[:, None] / 96.))
        if i == 0 or (i+1) % 20 == 0 or i+1 == len(loader):
            print(f"P0 cache batch {i+1}/{len(loader)} groups {writer.offset}/{len(keys)}", flush=True)
    manifest = writer.finish()
    return dict(summary, written=True, identity=manifest["identity"])
