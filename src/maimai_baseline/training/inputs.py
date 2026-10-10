"""Resolve prepared inputs once and construct datasets from the supplied plan."""

from __future__ import annotations
from maimai_baseline.model.capacity import from_args as capacity_from_args
from maimai_baseline.loss.supervision import from_args as supervision_from_args
from dataclasses import dataclass
from functools import partial
from maimai_baseline.contract import (
    EPOCH_PLAN_MODE,
    SEED,
    VALIDATION_TOKEN_BUDGET,
    baseline_manifest,
)
from maimai_baseline.data.curriculum import EpochPlanStore, MultiViewRouteStore, sha256_file
from maimai_baseline.data.dataset import BaselineDataset, EpochPlanBatchSampler
from maimai_baseline.model.route import RouteArtifacts, load_route_artifacts
from maimai_baseline.physical.interval_store import IntervalPhysicalStore
from maimai_baseline.physical.catalog import CompactPathCatalog
from maimai_baseline.physical.data import PhysicalBaselineDataset, collate_physical_segments
from maimai_baseline.physical.source_rows import PhysicalSourceRowStore
from maimai_baseline.source_identity import baseline_source_identity
from maimai_baseline.training.steps import _learning_rate
from maimai_baseline.training.recipe import recipe
from maimai_dataset.chart_cache import ChartVectorStore
from maimai_dataset.codec import PROFILE_SCHEMA, ROW_CODEC, VOCABULARY_SCHEMA
from maimai_dataset.manifest import DatasetError
from maimai_dataset.vocabulary import CompactVocabulary
from pathlib import Path
from torch.utils.data import DataLoader, Sampler
from typing import Any
import json
import torch
import time
import sys


def _collate_full_input(items, **kwargs):
    from maimai_baseline.data.batch_transport import pack_batch
    started = time.perf_counter()
    dense_provider = kwargs.pop('dense_provider', None)
    count_only = dense_provider is not None and not kwargs.get('chart_style', False)
    if dense_provider is not None:
        kwargs['chart_only'] = True
    if count_only:
        kwargs['global_reference_provider'] = None
    batch = collate_physical_segments(items, **kwargs)
    if count_only:
        batch['global_style_reference_count'] = len(items)
    if dense_provider is not None:
        batch['song_dense_inputs'] = dense_provider(items)
        batch['song_dense_inputs']['batch'].pop('sft_timings', None)
    # Only SFT consumes timing objects after collation; its collator is unchanged.
    batch.pop('sft_timings', None)
    # Both whole-song GT representations need bounded IPC storage with
    # persistent workers; individual song tensors exhaust file descriptors.
    result = pack_batch(batch)
    result.cpu_prepare_seconds = time.perf_counter() - started
    return result


@dataclass(frozen=True)
class TrainingInputs:
    profile: Path
    profile_summary: dict[str, Any]
    feature_stats: dict[str, Any]
    vocabulary: CompactVocabulary
    vocabulary_sha256: str
    chart_store: ChartVectorStore
    geometry: Path
    spatial_store: MultiViewRouteStore
    plan_store: EpochPlanStore
    artifacts: RouteArtifacts
    manifest: dict[str, Any]
    source_identity: dict[str, Any]
    physical_catalog: CompactPathCatalog
    physical_store: IntervalPhysicalStore
    physical_source_rows: PhysicalSourceRowStore
    bpm_audio_store: Any = None


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise DatasetError(f"expected JSON object: {path}")
    return value


def _manifest_identity(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    result = {
        "path": str(path.resolve()),
        "schema": str(payload["schema"]),
        "sha256": sha256_file(path),
    }
    if "query6_plan" in payload:
        result["query6_plan"] = payload["query6_plan"]
    if payload.get("manifest_identity_sha256") is not None:
        result["identity_sha256"] = str(payload["manifest_identity_sha256"])
    return result


def _positive_epoch_argument(args: Any, name: str) -> int:
    value = int(getattr(args, name))
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_epoch_plan_capacity(stop_after_epochs: int, available_epochs: int) -> None:
    # The supplied plan is authoritative; never manufacture missing epochs.
    if int(stop_after_epochs) > int(available_epochs):
        raise ValueError(
            f"stop_after_epochs={int(stop_after_epochs)} exceeds epoch-plan cache "
            f"capacity={int(available_epochs)}; extend or rebuild --epoch-plan-cache "
            f"before training beyond {int(available_epochs)} epochs"
        )


def _prepare_inputs(args: Any, *, required_physical_segments=None) -> TrainingInputs:
    from maimai_baseline.experiment import check_args
    config = check_args(args, "sft" if getattr(args, "operation", None) == "sft" else "A")
    from maimai_baseline.model.attention import from_args as attention_from_args
    attention = attention_from_args(args)
    supervision = supervision_from_args(args)
    started = previous = time.perf_counter()
    args.input_preparation_seconds = {}
    def stage(name):
        nonlocal previous
        now = time.perf_counter()
        args.input_preparation_seconds[name] = now - previous
        previous = now
        print(json.dumps({"stage": "runtime-metadata", "component": name,
                          "seconds": args.input_preparation_seconds[name]}), file=sys.stderr, flush=True)
    if not getattr(args, "bpm_audio_manifest", None):
        raise ValueError("training requires --bpm-audio-manifest and --interpolation-geometry")
    from maimai_baseline.data.audio import training_audio_policy
    _learning_rate(args)
    if args.command in ("fit", "train", "sft"):
        _positive_epoch_argument(args, "scheduler_epochs")
        stop_after_epochs = _positive_epoch_argument(args, "stop_after_epochs")
    else:
        stop_after_epochs = 1
    profile = Path(args.dataset).expanduser().resolve()
    summary = _load_json(profile / "summary.json")
    from maimai_dataset.window import WINDOW
    if summary.get("window") != WINDOW:
        raise DatasetError("profile window identity mismatch; prepare current training-dataset")
    profile_record = _load_json(profile / "profile.json")
    if summary.get("schema") != PROFILE_SCHEMA or summary.get("row_codec") != ROW_CODEC:
        raise DatasetError("baseline requires the current v2-test1 profile and row codec")
    vocabulary_path = profile / "vocabulary.json"
    vocabulary_sha = sha256_file(vocabulary_path)
    vocabulary = CompactVocabulary.from_dict(_load_json(vocabulary_path))
    if vocabulary.schema != VOCABULARY_SCHEMA:
        raise DatasetError("baseline vocabulary schema mismatch")
    feature_stats = _load_json((profile / str(profile_record["base_feature_stats"])).resolve())

    stage("profile")
    chart_cache = Path(args.chart_cache).expanduser().resolve()
    chart_store = ChartVectorStore.load(
        chart_cache,
        profile_fingerprint=str(summary["profile_fingerprint"]),
        vocabulary_sha256=vocabulary_sha,
        verify_payloads=False,
    )
    from maimai_baseline.loss.difficulty import contract as difficulty_contract
    from maimai_baseline.loss.difficulty import validate_records
    difficulty_identity = difficulty_contract(getattr(args, "difficulty_loss_preset", "uniform"))
    validate_records(chart_store.segment_records.values())
    chart_manifest = chart_store.manifest
    validation_policy = getattr(args, "validation_policy", "disabled")
    validation_enabled = validation_policy != "disabled"
    geometry = Path(args.interpolation_geometry).expanduser().resolve()
    geometry_manifest = _load_json(geometry / "manifest.json")
    selection_sha = str(chart_manifest.get("selection", {}).get("selection_sha256", ""))

    stage("chart-and-geometry-manifest")
    spatial_store = MultiViewRouteStore.load(
        Path(args.spatial_view_cache).expanduser().resolve(),
        profile_fingerprint=str(summary["profile_fingerprint"]),
        vocabulary_sha256=vocabulary_sha,
        verify_payloads=False,
        selection_sha256=selection_sha,
    )
    plan_store = EpochPlanStore(Path(args.epoch_plan_cache).expanduser().resolve(), verify_payloads=False)
    available_epochs = plan_store.available_epochs
    offset = int(args.resolved_recipe["offset"])
    _validate_epoch_plan_capacity(offset + stop_after_epochs, available_epochs)
    from maimai_baseline.experiment import validate_plan
    experiment = validate_plan(plan_store, config)
    # Old plans name their ordered record selection in subsets. The records,
    # not an experiment name or difficulty filter, define the runtime dataset.
    subsets = experiment["subsets"]
    for role in ("train", "validation"):
        if role not in subsets:
            rows = [row for row in chart_store.segment_records.values()
                    if row.get("role") == role]
            subsets[role] = {"records": list(dict.fromkeys(
                str(row.get("record_id") or str(row["segment_id"]).rsplit(":", 1)[0])
                for row in rows)), "segments": len(rows)}
    selected_records = set(subsets["train"]["records"])
    if validation_enabled:
        selected_records.update(subsets["validation"]["records"])
    selected_segments = {
        sid for sid, row in chart_store.segment_records.items()
        if str(row.get("record_id") or str(sid).rsplit(":", 1)[0]) in selected_records
    }
    stage("spatial-and-plan")
    artifacts = load_route_artifacts(Path(args.route_artifact).expanduser().resolve(), verify_payloads=False)
    physical_catalog = CompactPathCatalog.load(Path(args.physical_path_catalog), vocabulary)
    physical_store = IntervalPhysicalStore(
        Path(args.physical_interval_cache),
        profile_fingerprint=str(summary["profile_fingerprint"]),
        chart_cache_manifest_sha256=sha256_file(chart_cache / "manifest.json"),
        spatial_cache_manifest_sha256=sha256_file(spatial_store.root / "manifest.json"),
        catalog=physical_catalog, vocabulary=vocabulary, allow_partial=required_physical_segments is not None,
    )
    physical_source_rows = PhysicalSourceRowStore(
        Path(getattr(args, "physical_source_rows", None) or args.physical_path_catalog),
        profile_fingerprint=str(summary["profile_fingerprint"]),
        vocabulary_sha256=vocabulary_sha,
        verify_payloads=False,
        chart_cache_manifest_sha256=sha256_file(chart_cache / "manifest.json"),
        spatial_cache_manifest_sha256=sha256_file(spatial_store.root / "manifest.json"),
    )
    if not selected_segments.issubset(spatial_store.segment_records):
        raise ValueError("spatial cache lacks selected segments")
    if required_physical_segments is not None:
        required = set(required_physical_segments)
        if not required or not required.issubset(selected_segments):
            raise ValueError("diagnostic Physical selection is outside the experiment")
    else:
        required = selected_segments
    if not required.issubset(physical_store.segment_records):
        raise ValueError("Physical cache lacks selected segments")
    stage("physical")
    source = baseline_source_identity()
    manifest = baseline_manifest(
        encoder_capacity=capacity_from_args(args),
        decoder_rope_coordinate=getattr(args, "decoder_rope_coordinate", "window-beat"),
        profile={
            "path": str(profile),
            "profile_fingerprint": str(summary["profile_fingerprint"]),
            "vocabulary_sha256": vocabulary_sha,
        },
        chart_cache={
            **_manifest_identity(chart_cache / "manifest.json"),
            "selection_sha256": selection_sha,
            "segment_count": int(chart_manifest["segment_count"]),
        },
        interpolation_geometry={
            **_manifest_identity(geometry / "manifest.json"),
            "selection_sha256": selection_sha,
            "segment_count": int(geometry_manifest["segment_count"]),
        },
        route_artifact=artifacts.identity,
        route_head=getattr(args, "route_head", "prototype-distance"),
        view_cache=_manifest_identity(spatial_store.root / "manifest.json"),
        plan_cache=_manifest_identity(plan_store.root / "manifest.json"),
        source_identity=source,
    )
    manifest["physical_inputs"] = {
        "path_catalog": dict(physical_catalog.identity),
        "interval_cache": _manifest_identity(physical_store.root / "manifest.json"),
    }
    manifest["physical_source_rows"] = dict(physical_source_rows.identity)
    if tuple(spatial_store.arrays) != tuple(physical_source_rows.indices):
        raise ValueError("spatial cache and Physical source-row views differ")
    manifest["data_strategy"]["cache_views"] = list(spatial_store.arrays)
    manifest["validation"] = {
        "enabled": validation_enabled,
        "policy": validation_policy,
        "train_song_count": len({rid.split(':', 1)[0] for rid in subsets['train']['records']}),
        "validation_song_count": len({rid.split(':', 1)[0] for rid in subsets['validation']['records']}),
        "train_only_selection": not validation_enabled,
    }
    from maimai_baseline.data.audio import json_hash
    from maimai_baseline.data.audio import policy_contract
    from maimai_baseline.data.store import BpmFeatureStore
    from maimai_baseline.time.grid import SAMPLING
    from maimai_baseline.time.grid import INTEGRATION
    audio_manifest = getattr(args, "normalized_audio_manifest", None)
    songs = {rid.split(":", 1)[0] for rid in selected_records}
    audio_store = BpmFeatureStore(Path(args.bpm_audio_manifest), profile=profile, stats=feature_stats,
        song_ids=songs, selection_sha256=selection_sha, geometry_root=geometry,
        expected_segment_ids=selected_segments, verify_payloads=False)
    audio_policy = args.audio_policy = training_audio_policy(
        getattr(args, "audio_policy", None) or audio_store.policy)
    if audio_store.policy != ("legacy" if audio_policy == "legacy" else "normalized-p50"):
        raise ValueError("BPM audio base policy differs from training policy")
    from maimai_baseline.data.note_view import load_metadata
    note_view_path = getattr(args, 'note_view_manifest', None)
    if not note_view_path:
        raise ValueError('v10 requires --note-view-manifest; prepare-note-view.py audit/build/verify first')
    note_view = load_metadata(note_view_path, profile=profile,
                              bpm_manifest=args.bpm_audio_manifest,
                              canonical_source=True)
    if not selected_records.issubset(note_view['charts']):
        raise ValueError('note view lacks selected chart records')
    manifest['note_view'] = _manifest_identity(Path(note_view_path))
    from maimai_baseline.style_contract import is_native, is_draft
    native_audio = is_native(getattr(args, 'chart_style', None))
    if getattr(args, 'song_chart_cache', None) and not is_draft(getattr(args, 'chart_style', None)):
        from maimai_baseline.data.song_chart import SongChartCache
        song = SongChartCache(args.song_chart_cache)
        manifest['song_chart'] = {'identity':song.manifest['identity'], 'schema':song.manifest['schema']}
        if 'timing' in song.manifest:
            if not native_audio and ('song_audio' not in song.manifest or not getattr(args, 'song_audio_cache', None)):
                raise ValueError('compact GT requires its bound retained song audio cache')
            audio_store.offline_timing_root = str(Path(args.song_chart_cache).resolve())
    if not native_audio and getattr(args,'song_audio_cache',None):
        from maimai_baseline.data.song_audio import SongAudioCache
        song_audio=SongAudioCache(args.song_audio_cache)
        manifest['song_audio']={'identity':song_audio.manifest['identity'],'schema':song_audio.manifest['schema']}


    manifest["bpm_input"] = {"sampling": SAMPLING, "integration": INTEGRATION,
                              "audio_policy": audio_policy, "artifacts": audio_store.identity,
                              "feature_stats_sha256": json_hash(feature_stats)}
    if audio_policy != "legacy":
        manifest["audio_input"] = {"policy": audio_policy, "contract": policy_contract(audio_policy),
                                   "feature_stats_sha256": json_hash(feature_stats), "artifacts": audio_store.identity}
    elif audio_manifest:
        raise ValueError("--normalized-audio-manifest requires a normalized audio policy")
    stage("bpm-and-source")
    manifest["experiment"] = experiment
    manifest["difficulty_loss_weighting"] = difficulty_identity
    manifest["specialized_supervision_config"] = supervision.identity()
    from maimai_baseline.style_contract import style_identity
    manifest["chart_style"] = style_identity(getattr(args, "chart_style", "none"))
    from maimai_baseline.style_contract import is_frozen, is_draft
    frozen = is_frozen(getattr(args, 'chart_style', None))
    manifest["latent_enabled"] = frozen
    args.latent_enabled = frozen
    if is_draft(getattr(args, 'chart_style', None)):
        if not getattr(args, 'draft_cache', None):
            raise ValueError('audio draft A training requires --draft-cache')
        from maimai_baseline.data.draft_cache import AudioDraftCache
        cache = AudioDraftCache(args.draft_cache)
        cache.validate_binding(profile_fingerprint=str(summary['profile_fingerprint']),
            vocabulary_sha256=vocabulary_sha,
            segment_order_sha256=plan_store.manifest['segment_order_sha256'])
        manifest['draft_cache'] = cache.identity
    elif frozen:
        if not getattr(args, 'latent_cache', None):
            raise ValueError('frozen latent A training requires --latent-cache')
        from maimai_baseline.data.frozen_latent import FrozenLatentCache
        cache = FrozenLatentCache(args.latent_cache)
        if cache.manifest['chart_style'] != args.chart_style:
            raise ValueError('A model and mu cache representation differ')
        if cache.manifest.get('derived_from') is not None:
            coverage = cache.manifest.get('coverage', {})
            if coverage.get('valid_window_blocks', 0) < coverage.get('condition_window_blocks', 0):
                raise ValueError('A requires genuine mu for the full new profile; rebuild the complete encoder60 cache before A90')
        cache.validate_binding(profile_fingerprint=str(summary['profile_fingerprint']),
            vocabulary_sha256=vocabulary_sha,
            segment_order_sha256=plan_store.manifest['segment_order_sha256'])
        manifest['frozen_latent'] = cache.identity
    if native_audio:
        if not getattr(args, 'teacher_audio_cache', None):
            raise ValueError('native audio A requires --teacher-audio-cache')
        from maimai_baseline.data.teacher_audio import TeacherAudioCache
        manifest['teacher_audio'] = TeacherAudioCache(args.teacher_audio_cache).identity
    from maimai_baseline.radar_contract import RADAR_ID
    if getattr(args, 'chart_style', None) == RADAR_ID:
        if not getattr(args, 'majradar_cache', None):
            raise ValueError('MajRadar A training requires --majradar-cache or its dataset manifest input')
        from maimai_baseline.data.majradar import MajRadarCache
        radar = MajRadarCache(args.majradar_cache)
        manifest['majradar'] = {key: radar.manifest[key] for key in
                               ('schema', 'dimensions', 'scales', 'data_binding')}
    manifest["audio_attention"] = attention.identity()
    manifest["training_recipe"] = recipe(args).identity()
    if "song_sampling" in plan_store.manifest:
        manifest["song_sampling"] = plan_store.manifest["song_sampling"]
    args.input_preparation_seconds["total"] = time.perf_counter() - started
    return TrainingInputs(
        profile,
        summary,
        feature_stats,
        vocabulary,
        vocabulary_sha,
        chart_store,
        geometry,
        spatial_store,
        plan_store,
        artifacts,
        manifest,
        source,
        physical_catalog,
        physical_store,
        physical_source_rows,
        audio_store,
    )


class _ValidationTokenBudgetSampler(Sampler[list[int]]):
    """Frozen validation packing with complete cached-audio groups."""

    def __init__(self, dataset: BaselineDataset) -> None:
        batches: list[tuple[int, ...]] = []
        batch: list[int] = []
        used = 0
        for source_group in dataset.audio_group_indices:
            group = tuple(int(index) for index in source_group)
            cost = sum(max(1, int(dataset.target_lengths[index])) for index in group)
            if batch and used + cost > VALIDATION_TOKEN_BUDGET:
                batches.append(tuple(batch))
                batch, used = [], 0
            batch.extend(group)
            used += cost
        if batch:
            batches.append(tuple(batch))
        self.batches = tuple(batches)

    def __iter__(self):
        yield from (list(batch) for batch in self.batches)

    def __len__(self) -> int:
        return len(self.batches)


def _dataset(inputs: TrainingInputs, role: str) -> BaselineDataset:
    dataset = PhysicalBaselineDataset(
        inputs.profile,
        inputs.vocabulary,
        physical_store=inputs.physical_store,
        physical_source_rows=inputs.physical_source_rows,
        split=None,
        song_ids={record.split(':', 1)[0]
                  for record in inputs.manifest["experiment"]["subsets"][role]["records"]},
        record_ids=inputs.manifest["experiment"]["subsets"][role]["records"],
        source_difficulty_ids=None,
        geometry_root=None,
        geometry_suite_sha256=None,
        chart_cache_store=inputs.chart_store,
        spatial_store=inputs.spatial_store,
        augmentation_seed=inputs.manifest["experiment"]["recipe"]["seed"],
        defer_audio=True,
    )
    dataset.set_augmentation_enabled(role == "train")
    if inputs.bpm_audio_store is not None:
        dataset.configure_bpm_audio(inputs.bpm_audio_store, inputs.manifest["bpm_input"]["audio_policy"])
    return dataset


def _loaders(
    inputs: TrainingInputs, args: Any
) -> tuple[DataLoader, EpochPlanBatchSampler, DataLoader | None]:
    from maimai_baseline.data.song_chart import SongChartProvider
    from maimai_baseline.style_contract import is_frozen, is_native, is_draft
    frozen = is_frozen(getattr(args, 'chart_style', None))
    cached_targets = getattr(args, "cached_style_targets", False)
    dense = getattr(args, "chart_style", "none") != "none" and not cached_targets
    encode_style = dense and getattr(args, "latent_enabled", False) and not frozen
    from maimai_baseline.data.song_chart_compact import SCHEMA as COMPACT_SCHEMA
    packed = inputs.manifest.get("song_chart", {}).get("schema") == COMPACT_SCHEMA
    timing_root = str(Path(args.song_chart_cache).resolve()) if packed else None
    train_dataset = _dataset(inputs, "train")
    from maimai_dataset.manifest import canonical_json_sha256
    expected_order = inputs.plan_store.manifest.get("segment_order_sha256")
    if expected_order is not None and expected_order != canonical_json_sha256([str(s["segment_id"]) for s in train_dataset.segments]):
        raise ValueError("epoch plan segment order mismatch")
    batching = {
        "policy": "precomputed-baseline-epoch-plan-v1",
        "unit": "complete-audio-group",
        "token_budget": recipe(args).token_budget,
        "recipe": recipe(args).identity(),
        "optimizer_steps_by_epoch": [inputs.plan_store.batch_count(e) for e in range(inputs.plan_store.available_epochs)],
        "plan_cache": _manifest_identity(inputs.plan_store.root / "manifest.json"),
        "mode": EPOCH_PLAN_MODE,
    }
    args.batching = batching
    sampler = EpochPlanBatchSampler(
        inputs.plan_store, train_dataset, batching_manifest=batching
    )
    sampler.set_epoch(int(args.resolved_recipe["offset"]))
    from maimai_baseline.data.song_dense import SongDenseProvider
    dense_provider = None
    if dense:
        dense_provider = SongDenseProvider(train_dataset, SongChartProvider(args.song_chart_cache,
            train_dataset, 'train', song_audio_cache=args.song_audio_cache,
            latent_enabled=encode_style, latent_cache=getattr(args, 'latent_cache', None) if frozen and not is_draft(args.chart_style) else None,
            draft_cache=getattr(args, 'draft_cache', None) if is_draft(args.chart_style) else None,
            teacher_audio_cache=getattr(args, 'teacher_audio_cache', None) if is_native(args.chart_style) else None),
            timing_root, radar_cache=getattr(args, 'majradar_cache', None) if not frozen else None,
            teacher_audio_cache=getattr(args, 'teacher_audio_cache', None) if is_native(args.chart_style) else None)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        collate_fn=partial(
            _collate_full_input, timing_root=timing_root, dense_provider=dense_provider,
            vocabulary=inputs.vocabulary, physical_store=inputs.physical_store, audio_loader=train_dataset.audio_item,
            chart_style=encode_style, cached_style_targets=cached_targets,
            global_reference_provider=SongChartProvider(args.song_chart_cache, train_dataset, 'train', joint=cached_targets,song_audio_cache=getattr(args,"song_audio_cache",None),latent_enabled=encode_style) if getattr(args,'song_chart_cache',None) else None
        ),
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(args.num_workers) > 0,
        generator=torch.Generator().manual_seed(getattr(args, "seed", SEED)) if cached_targets or frozen else None,
    )
    if args.command == "benchmark":
        return train_loader, sampler, None
    if inputs.manifest.get("validation", {}).get("enabled") is False:
        return train_loader, sampler, None
    validation_dataset = _dataset(inputs, "validation")
    validation_dense_provider = (SongDenseProvider(validation_dataset, SongChartProvider(args.song_chart_cache,
        validation_dataset, 'validation', song_audio_cache=args.song_audio_cache,
        latent_enabled=encode_style, latent_cache=getattr(args, 'latent_cache', None) if frozen and not is_draft(args.chart_style) else None,
        draft_cache=getattr(args, 'draft_cache', None) if is_draft(args.chart_style) else None,
        teacher_audio_cache=getattr(args, 'teacher_audio_cache', None) if is_native(args.chart_style) else None),
        timing_root, radar_cache=getattr(args, 'majradar_cache', None) if not frozen else None,
        teacher_audio_cache=getattr(args, 'teacher_audio_cache', None) if is_native(args.chart_style) else None) if dense else None)
    validation_loader = DataLoader(
        validation_dataset,
        batch_sampler=_ValidationTokenBudgetSampler(validation_dataset),
        collate_fn=partial(
            _collate_full_input, timing_root=timing_root, dense_provider=validation_dense_provider,
            vocabulary=inputs.vocabulary, physical_store=inputs.physical_store, audio_loader=validation_dataset.audio_item,
            chart_style=encode_style, cached_style_targets=cached_targets,
            global_reference_provider=SongChartProvider(args.song_chart_cache, validation_dataset, 'validation', joint=cached_targets,song_audio_cache=getattr(args,"song_audio_cache",None),latent_enabled=encode_style) if getattr(args,'song_chart_cache',None) else None
        ),
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(args.num_workers) > 0,
        generator=torch.Generator().manual_seed(getattr(args, "seed", SEED)) if cached_targets or frozen else None,
    )
    return train_loader, sampler, validation_loader
