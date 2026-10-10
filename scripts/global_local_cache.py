"""Offline bound A target production with explicit parent and producer identities."""
from __future__ import annotations
from collections import OrderedDict
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from maimai_baseline.data.dataset import BaselineDataset
from maimai_baseline.data.global_style import pack_reference_rows
from maimai_baseline.data.style_cache import StyleCacheWriter
from maimai_baseline.style_contract import STYLE_SLOTS, STYLE_BEATS
from maimai_dataset.integrity import file_sha256
from maimai_dataset.manifest import canonical_json_sha256
from maimai_dataset.timing import TimingMap
from maimai_baseline.time.audio_end import audio_end


def producer_identity(options):
    root = Path(__file__).resolve().parent
    return {"schema": "maimai.global_local_cache_producer.v2", "implementation": "full-song-z128-exact-posterior-moments-v1",
            "files": {name: file_sha256(root / name) for name in
                      ("global_local_cache.py", "song_chart_features.py", "build-global-local-style-cache.py")},
            "execution": dict(options), "precision": "fp32-encoder-fp64-moments"}


def cache_dataset(inputs, role):
    dataset = BaselineDataset(inputs.profile, inputs.vocabulary, split=None,
        song_ids=inputs.chart_store.song_ids(role), source_difficulty_ids=inputs.manifest["experiment"]["recipe"]["difficulties"], record_ids=inputs.manifest["experiment"]["subsets"][role]["records"], geometry_root=None, geometry_suite_sha256=None,
        chart_cache_store=inputs.chart_store, spatial_store=inputs.spatial_store,
        augmentation_seed=20260811, defer_audio=True)
    dataset.set_augmentation_enabled(False)
    dataset.configure_bpm_audio(inputs.bpm_audio_store, inputs.manifest["bpm_input"]["audio_policy"])
    return dataset


def read_target_inputs(dataset, index, view, end):
    """Read only accepted input vectors. No targets, history conversion or Physical."""
    segment = dataset.segments[index]
    store = dataset.spatial_store
    row = store.segment_indices[str(segment["segment_id"])]
    begin, stop = int(store.offsets[row]), int(store.offsets[row + 1])
    arrays = store.arrays["identity" if view == 0 else "mirror-lr"]
    item = {family: {} for family in ("input_ids", "input_times", "input_time_active")}
    for name, value in arrays.items():
        family, _, field = name.partition(".")
        if family in item:
            dtype = torch.float32 if family == "input_times" else torch.bool if family == "input_time_active" else torch.long
            item[family][field] = torch.from_numpy(np.array(value[begin:stop], copy=True)).to(dtype)
    anchors = torch.from_numpy(np.array(arrays["row_anchor_tick"][begin:stop], copy=True)).float()
    marker = item["input_ids"]["event"].eq(dataset.vocabulary.event_id("TARGET_START")).nonzero().flatten()[0]
    item = {family: {name: value[int(marker):] for name, value in values.items()} for family, values in item.items()}
    item.update(row_anchor_tick=anchors[int(marker):], start_tick=int(segment["target_start_tick"]),
                end_tick=int(segment["target_end_tick"]), song_end_tick=end)
    return item


def stack_audio(items):
    # The maintained BPM96 grid has fixed length, including its masked tail.
    batch = {name: torch.stack([item[name] for item in items]) for name in
             ("audio", "frame_beats", "frame_bpms")}
    batch["audio_padding_mask"] = ~torch.stack([item["audio_valid"] for item in items])
    batch["audio_start_tick"] = torch.tensor([item["audio_start_tick"] for item in items], dtype=torch.float32)
    batch["interpolation_geometry"] = {}
    for name in items[0]["interpolation_geometry"]:
        value = torch.stack([item["interpolation_geometry"][name] for item in items])
        if name.endswith(("_left", "_right", "_start_tick")):
            value = value.long()
        batch["interpolation_geometry"][name] = value
    return batch


class CacheBatches(Dataset):
    """One dataset item is a prepared batch of complete audio groups."""
    def __init__(self, datasets, vocabulary, token_budget=4096, *, audio_only=False, song_audio_cache=None):
        if token_budget < 1:
            raise ValueError("cache token budget must be positive")
        self.datasets, self.vocabulary = datasets, vocabulary
        self.audio_only = audio_only
        self.song_audio_root=song_audio_cache
        self.song_audio_cache=None
        self.records, groups = [], OrderedDict()
        boundaries = {}
        for role, dataset in datasets.items():
            store = dataset.bpm_audio_store
            for i, segment in enumerate(dataset.segments):
                record_key = (role, str(segment["record_id"]))
                if record_key not in boundaries:
                    variant = store.variants[store.segments[str(segment["segment_id"])]]
                    boundaries[record_key] = audio_end(TimingMap.from_dict(segment["timing"]), int(variant["num_samples"])).onset_tick
                audio_key = dataset.runtime_segment_audio_keys[i]
                for view in ((0, 1) if role == "train" else (0,)):
                    row = len(self.records)
                    self.records.append(dict(role=role, canonical_index=i, view_id=view,
                        segment_id=str(segment["segment_id"]), song_id=str(segment["song_id"]),
                        record_id=str(segment["record_id"]), level_value=float(segment["level_value"]),
                        source_difficulty_id=int(segment.get("source_difficulty_id", segment["difficulty_id"])),
                        difficulty_id=int(segment.get("condition_difficulty_id", segment["difficulty_id"])),
                        start_tick=int(segment["target_start_tick"]), end_tick=int(segment["target_end_tick"]),
                        song_end_tick=boundaries[record_key]))
                    groups.setdefault((role, audio_key), []).append(row)
        self.batches, pending, cost = [], [], 0
        for key, rows in groups.items():
            dataset = datasets[key[0]]
            size = sum(max(1, int(dataset.segments[self.records[r]["canonical_index"]]["target_row_end"]) -
                              int(dataset.segments[self.records[r]["canonical_index"]]["target_row_start"])) for r in rows)
            # As in A, an indivisible audio group may exceed the target budget.
            if pending and cost + size > token_budget:
                self.batches.append(pending); pending, cost = [], 0
            pending.append((key, rows)); cost += size
        if pending:
            self.batches.append(pending)
        self.unique_audio_windows = len(groups)

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        started = time.perf_counter()
        inputs, audio_items, mapping, indices = [], [], [], []
        variants=[]
        for group_index, ((role, audio_key), rows) in enumerate(self.batches[index]):
            dataset = self.datasets[role]
            if self.audio_only:
                audio_items.append(dataset.audio_item(audio_key))
                variants.append(dataset.bpm_audio_store.segments[str(self.records[rows[0]]["segment_id"])])
            for row in rows:
                r = self.records[row]
                if not self.audio_only:
                    inputs.append(read_target_inputs(dataset, r["canonical_index"], r["view_id"], r["song_end_tick"]))
                mapping.append(group_index); indices.append(row)
        batch = stack_audio(audio_items) if self.audio_only else {}
        if self.audio_only and self.song_audio_root is not None:
            from maimai_baseline.data.song_audio import SongAudioCache
            if self.song_audio_cache is None:self.song_audio_cache=SongAudioCache(self.song_audio_root)
            batch["song_audio_inputs"]=self.song_audio_cache.batch(variants)
        if not self.audio_only:
            batch["style_inputs"] = pack_reference_rows(inputs, self.vocabulary, beats_per_token=STYLE_BEATS, slots=STYLE_SLOTS)
            batch["reference_inputs"] = pack_reference_rows(inputs, self.vocabulary)
        for name in ("start_tick", "end_tick", "song_end_tick", "difficulty_id"):
            batch[name] = torch.tensor([self.records[i][name] for i in indices])
        batch["window_to_audio"] = torch.tensor(mapping)
        return {"indices": indices, "batch": batch, "prepare_seconds": time.perf_counter() - started,
                "audio_keys": [list(key) for key, rows in self.batches[index]]}


def worker_init(_):
    torch.set_num_threads(1)


@torch.no_grad()
def prepare(args, *, token_budget=4096, workers=12, cpu_threads=1):
    from maimai_baseline.checkpoint import load_frozen_stage_a
    from maimai_baseline.training.inputs import _prepare_inputs
    from maimai_baseline.training.steps import _device
    if workers < 0 or cpu_threads < 1:
        raise ValueError("cache workers must be nonnegative and threads positive")
    build_started = started = time.perf_counter(); torch.set_num_threads(cpu_threads)
    model, parent = load_frozen_stage_a(Path(args.stage_a_checkpoint), route_artifact=Path(args.route_artifact))
    # Explicit pinned-parent initialization, not a cross-source training resume.
    from maimai_baseline.source_identity import baseline_source_identity
    inputs = _prepare_inputs(args)
    datasets = {role: cache_dataset(inputs, role) for role in ("train", "validation")
                if role == "train" or inputs.manifest["validation"]["enabled"]}
    source = CacheBatches(datasets, inputs.vocabulary, token_budget)
    ids = [str(s["segment_id"]) for s in datasets["train"].segments]
    if canonical_json_sha256(ids) != inputs.plan_store.manifest["segment_order_sha256"]:
        raise ValueError("style cache/epoch plan segment order mismatch")
    options = dict(token_budget=token_budget, workers=workers, cpu_threads=cpu_threads, prefetch=2)
    bindings = dict(stage_a_sha256=file_sha256(Path(args.stage_a_checkpoint)), baseline_manifest=parent["baseline_manifest"],
        source_identity=parent["source_identity"], train_segments=len(ids), segment_order_sha256=canonical_json_sha256(ids),
        plan_manifest_sha256=file_sha256(inputs.plan_store.root / "manifest.json"), cache_producer=producer_identity(options),
        runtime_source_identity=baseline_source_identity(),
        song_audio_input_identity=inputs.manifest["song_audio"]["identity"],
        normalization_source_sha256=file_sha256(Path(args.stage_a_checkpoint)))
    summary = dict(operation="style-cache", records=len(source.records), batches=len(source),
                   unique_audio_windows=source.unique_audio_windows, execution=options)
    if args.dry_run:
        return dict(summary, written=False)
    if Path(args.output).exists():
        raise ValueError("refusing to overwrite style cache")
    device = _device(args.device); model.to(device).eval().requires_grad_(False)
    writer = StyleCacheWriter(args.output, source.records, bindings)
    from song_chart_features import build
    return build(writer, model, args.song_chart_cache, device)
