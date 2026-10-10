"""Canonical draft supervision and order-preserving standalone data preparation."""
from __future__ import annotations

from copy import deepcopy
from functools import partial
from pathlib import Path
import shutil

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from maimai_baseline.data.chart_encoder_pretrain import (ChartEncoderDataset,
    ChartEncoderLoader, ChartEncoderPretrainInputs, FixedChartPlanSampler,
    _payload_metadata, _transport_collate, _worker_init)
from maimai_baseline.data.curriculum import EpochPlanStore, MultiViewRouteStore, VirtualSegmentRef
from maimai_baseline.data.encoder_audio import (_audio_indices, _export_requests,
    _note_index, _physical_index, ensure_export_requests, resolve_paths as preparation_paths)
from maimai_baseline.data.frozen_latent import selected_segments
from maimai_baseline.data.prior_decoder import (_extended_groups, _pack_groups,
    _read, _rows, _standard_vectors, _window_cache, WindowCache)
from maimai_baseline.data.repack import flatten, reference_hash
from maimai_baseline.data.style_cache import write_json
from maimai_baseline.data.teacher_audio import TeacherAudioCache, draft_producer_identity
from maimai_baseline.data.training_build import segment_chart
from maimai_dataset.chart_cache import ChartVectorStore
from maimai_dataset.integrity import file_sha256
from maimai_dataset.manifest import DatasetError, canonical_json_sha256 as digest, load_jsonl
from maimai_dataset.vocabulary import CompactVocabulary

SCHEMA = "maimai.audio_draft_data.v1"


def condition_key(variant, level, difficulty):
    """Float32 is the actual continuous conditioning precision."""
    value = float(np.float32(level))
    if not np.isfinite(value):
        raise DatasetError("draft level is not finite")
    return str(variant), value, int(difficulty)


def segment_condition(segment, teacher):
    record = str(segment.get("record_id") or str(segment["segment_id"]).rsplit(":", 1)[0])
    variant = segment.get("audio_variant") or teacher.manifest.get("record_to_variant", {}).get(record)
    if variant is None:
        variant = teacher.resolve_variant(str(segment["song_id"]), segment.get("timing"))
    return condition_key(variant, segment.get("level_value") or 0.,
        segment.get("source_difficulty_id", segment["difficulty_id"]))


def build_draft_inputs(teacher, condition_keys, *, beats_per_token=1):
    """One visible audio bank per variant, one query sequence per condition."""
    keys = [condition_key(*key) for key in condition_keys]
    if not keys:
        raise DatasetError("empty draft condition batch")
    variants = list(dict.fromkeys(key[0] for key in keys))
    bundles = teacher.bundles(variants)
    rows = [teacher.draft_stitched(variant, bundles=bundles, beats_per_token=beats_per_token) for variant in variants]
    lengths = [rows[variants.index(key[0])]["blocks"] for key in keys]
    maximum = max(lengths)
    valid = torch.arange(maximum)[None] < torch.tensor(lengths)[:, None]
    beats = ((torch.arange(maximum).float() + .5) * beats_per_token)[None].expand(len(keys), -1).clone()
    beats.masked_fill_(~valid, 0.)
    return dict(**{name: pad_sequence([row[name] for row in rows], batch_first=True)
                   for name in ("audio", "audio_valid", "audio_beats")},
        audio_mapping=torch.tensor([variants.index(key[0]) for key in keys], dtype=torch.long),
        valid=valid, beats=beats,
        difficulty_id=torch.tensor([key[2] for key in keys], dtype=torch.long),
        level=torch.tensor([key[1] for key in keys], dtype=torch.float32),
        span_beats=torch.tensor([rows[variants.index(key[0])]["span_beats"] for key in keys]))


def _write_plans(parent, output, old, segments, variants):
    """Only the new draft and main A plans; no learned cache or B stage."""
    lengths = [int(segment["target_rows"]) for segment in segments]
    identical = [s["segment_id"] for s in old] == [s["segment_id"] for s in segments]
    audits = {"a-plan": [], "draft-plan": []}
    for name in audits:
        (output / name / parent.mode).mkdir(parents=True)
    for epoch in range(parent.available_epochs):
        original = parent.load_epoch(epoch)
        main = original if identical else _pack_groups(
            _extended_groups(original, old, segments, variants), lengths, 4096)
        # Spatial allocation selects one view of each chart occurrence. Mapping
        # that view to canonical is one-for-one, not filtering or augmentation.
        assigned = {}
        for reference in flatten(main):
            key = reference.canonical_index, reference.occurrence
            previous = assigned.setdefault(key, reference.view_id)
            if previous != reference.view_id:
                raise DatasetError("draft source repeats the same window occurrence across views")
        canonical = tuple(tuple(VirtualSegmentRef(r.canonical_index, 0, r.occurrence)
                                for r in batch) for batch in main)
        for name, batches in (("a-plan", main), ("draft-plan", canonical)):
            refs = flatten(batches)
            offsets = np.cumsum([0, *map(len, batches)], dtype=np.int64)
            relative = f"{parent.mode}/epoch-{epoch + 1:03d}.npz"
            path = output / name / relative
            np.savez(path, canonical_indices=np.array([r.canonical_index for r in refs], np.int64),
                view_ids=np.array([r.view_id for r in refs], np.int64),
                occurrences=np.array([r.occurrence for r in refs], np.int64), batch_offsets=offsets)
            counts = [sum(lengths[r.canonical_index] for r in batch) for batch in batches]
            audits[name].append(dict(epoch=epoch + 1, mode=parent.mode, file=relative,
                file_sha256=file_sha256(path), batch_count=len(batches), batch_target_rows=counts,
                target_rows=sum(counts), segment_references=len(refs),
                source_reference_sha256=reference_hash(flatten(original)),
                reference_sha256=reference_hash(refs),
                plan_sha256=digest(dict(references=reference_hash(refs), offsets=offsets.tolist()))))
    order = digest([s["segment_id"] for s in segments])
    for name, rows in audits.items():
        root = output / name
        _rows(root / "audit.jsonl", rows)
        manifest = deepcopy(parent.manifest)
        for key in ("recipe", "fixed_batch_count", "paired_first_ten_epoch_plan_sha256", "manifest_identity_sha256"):
            manifest.pop(key, None)
        experiment = manifest.setdefault("experiment", {})
        subset = experiment.setdefault("subsets", {}).setdefault("train", {})
        subset.update(segments=len(segments), records=list(dict.fromkeys(s["record_id"] for s in segments)),
                      charts=len({s["record_id"] for s in segments}), segment_order_sha256=order)
        experiment["optimizer_steps_by_epoch"] = [row["batch_count"] for row in rows]
        if "identity" in experiment:
            experiment["identity"] = digest({key: value for key, value in experiment.items() if key != "identity"})
        manifest.update(token_budget=4096, segment_count=len(segments), segment_order_sha256=order,
            profile_segments_sha256=file_sha256(output / "profile/segments.jsonl"),
            modes={parent.mode: dict(epoch_plan_aggregate_sha256=digest([row["plan_sha256"] for row in rows]))},
            audit=dict(path="audit.jsonl", rows=len(rows), sha256=file_sha256(root / "audit.jsonl")))
        if name == "draft-plan":
            manifest["spatial_sampling"] = dict(views=["identity"], policy="canonicalize-existing-references")
        manifest["manifest_identity_sha256"] = digest(manifest)
        write_json(root / "manifest.json", manifest)
        loaded = EpochPlanStore(root, verify_payloads=False)
        for epoch in range(parent.available_epochs):
            loaded.load_epoch(epoch)
    return {name: [row["batch_count"] for row in rows] for name, rows in audits.items()}


def canonicalize_plan(source, output, *, segments=None, excluded_songs=()):
    """Derive only view IDs, leaving all source references and bounds in place."""
    source = source if isinstance(source, EpochPlanStore) else EpochPlanStore(source, verify_payloads=False)
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite canonical draft plan: " + str(output))
    (output / source.mode).mkdir(parents=True)
    audits = []
    excluded_songs = set(map(str, excluded_songs))
    retained = None if not excluded_songs else [row for row in segments if str(row['song_id']) not in excluded_songs]
    remap = None if retained is None else {index: new for new, index in enumerate(
        index for index, row in enumerate(segments) if str(row['song_id']) not in excluded_songs)}
    for epoch in range(source.available_epochs):
        batches = source.load_epoch(epoch)
        refs = flatten(batches)
        assigned = {}
        for reference in refs:
            key = reference.canonical_index, reference.occurrence
            old = assigned.setdefault(key, reference.view_id)
            if old != reference.view_id:
                raise DatasetError("draft source repeats the same window occurrence across views")
        canonical_batches = [tuple(VirtualSegmentRef(ref.canonical_index if remap is None else remap[ref.canonical_index], 0, ref.occurrence)
            for ref in batch if remap is None or ref.canonical_index in remap) for batch in batches]
        canonical_batches = [batch for batch in canonical_batches if batch]
        canonical = flatten(canonical_batches)
        offsets = np.cumsum([0, *map(len, canonical_batches)], dtype=np.int64)
        relative = f"{source.mode}/epoch-{epoch + 1:03d}.npz"
        path = output / relative
        np.savez(path, canonical_indices=np.array([ref.canonical_index for ref in canonical], np.int64),
            view_ids=np.zeros(len(canonical), np.int64),
            occurrences=np.array([ref.occurrence for ref in canonical], np.int64), batch_offsets=offsets)
        audits.append(dict(source.audit(epoch), file=relative, file_sha256=file_sha256(path),
            source_reference_sha256=reference_hash(refs), reference_sha256=reference_hash(canonical),
            segment_references=len(canonical),
            plan_sha256=digest(dict(references=reference_hash(canonical), offsets=offsets.tolist()))))
        if retained is not None:
            costs = [sum(int(retained[r.canonical_index]['target_rows']) for r in batch) for batch in canonical_batches]
            audits[-1].update(batch_count=len(costs), batch_target_rows=costs, target_rows=sum(costs))
    _rows(output / "audit.jsonl", audits)
    manifest = deepcopy(source.manifest)
    manifest.pop("manifest_identity_sha256", None)
    if retained is not None:
        order = digest([row['segment_id'] for row in retained])
        manifest.pop('fixed_batch_count', None)
        manifest.update(segment_count=len(retained), segment_order_sha256=order)
        experiment = manifest.setdefault('experiment', {})
        subset = experiment.setdefault('subsets', {}).setdefault('train', {})
        subset.update(records=list(dict.fromkeys(row['record_id'] for row in retained)),
            segments=len(retained), charts=len({row['record_id'] for row in retained}), segment_order_sha256=order)
        experiment['optimizer_steps_by_epoch'] = [row['batch_count'] for row in audits]
        experiment['draft_holdout_song_ids'] = sorted(excluded_songs)
        if 'identity' in experiment:
            experiment['identity'] = digest({key: value for key, value in experiment.items() if key != 'identity'})
    manifest.update(spatial_sampling=dict(views=["identity"], policy="canonicalize-existing-references"),
        modes={source.mode: dict(epoch_plan_aggregate_sha256=digest([row["plan_sha256"] for row in audits]))},
        audit=dict(path="audit.jsonl", rows=len(audits), sha256=file_sha256(output / "audit.jsonl")))
    manifest["manifest_identity_sha256"] = digest(manifest)
    write_json(output / "manifest.json", manifest)
    return [row["batch_count"] for row in audits]


def _complete_profile(segments):
    charts = {}
    for segment in segments:
        charts.setdefault(str(segment["record_id"]), []).append(segment)
    for rows in charts.values():
        rows.sort(key=lambda row: int(row["target_start_tick"]))
        end = int(rows[-1]["target_end_tick"])
        if ([int(row["target_start_tick"]) for row in rows] != list(range(0, end, 1536))
                or any(int(row.get("condition_end_tick", -1)) != end for row in rows)
                or int(rows[-1]["target_rows"]) != 1
                or int(rows[-1]["target_start_tick"]) < max(int(row["song_end_tick"]) for row in rows)):
            return False
    return bool(charts)


def missing_teacher_requests(requests, existing):
    """Inspect only coverage metadata; an external adapter produces new slices."""
    requests = _read(requests) if isinstance(requests, (str, Path)) else requests
    teacher = existing if isinstance(existing, TeacherAudioCache) else TeacherAudioCache(existing)
    from maimai_baseline.time.grid import timing_identity
    from maimai_dataset.timing import TimingMap
    result = deepcopy(requests)
    result["variants"] = {}
    for variant, request in requests["variants"].items():
        old = teacher.manifest["variants"].get(variant)
        if old is not None:
            if (int(old["num_samples"]) != int(request["num_samples"])
                    or timing_identity(TimingMap.from_dict(old["timing"])) != timing_identity(TimingMap.from_dict(request["timing"]))):
                raise DatasetError("reused native audio variant has different sample/timing metadata: " + variant)
            covered = set(map(int, old["starts"]))
        else:
            covered = set()
        missing = [int(start) for start in request["starts"] if int(start) not in covered]
        if missing:
            result["variants"][variant] = dict(request, starts=missing)
    return result


def _prepare_training_data(args):
    """Extend only window/index assets and derive canonical draft supervision."""
    paths = preparation_paths(args)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite audio draft data: " + str(output))
    parent = EpochPlanStore(paths["epoch_plan_cache"], verify_payloads=False)
    if parent.manifest.get("token_budget", 4096) != 4096:
        raise DatasetError("audio draft preparation requires source A4096 plan")
    old = selected_segments(paths["dataset"], parent)
    bpm = _read(paths["bpm_audio_manifest"])
    if getattr(args, 'validation_from_training', False) and not _complete_profile(old):
        raise DatasetError('full draft holdout requires the accepted complete-tail profile')
    if _complete_profile(old):
        output.mkdir(parents=True)
        holdout = None
        if getattr(args, 'validation_from_training', False):
            from maimai_baseline.data.draft_validation import select_training_holdout
            holdout = select_training_holdout(load_jsonl(paths['dataset'] / 'manifest.jsonl'), old, bpm,
                count=getattr(args, 'validation_song_count', 32))
        counts = canonicalize_plan(parent, output / "draft-plan", segments=old, excluded_songs=holdout or ())
        variants = {str(segment["record_id"]): bpm["segments"][segment["segment_id"]] for segment in old}
        requests = _export_requests(paths["dataset"], paths["bpm_audio_manifest"], old, variants,
                                   _read(paths["dataset"] / "summary.json")["profile_fingerprint"])
        write_json(output / "export_requests.json", requests)
        mapped = dict(paths, draft_epoch_plan_cache=output / "draft-plan")
        for name in ("whole_song_plan", "latent_cache", "song_context_cache", "decoder_context_cache",
                     "chart_encoder_checkpoint", "stage_a_checkpoint", "song_chart_cache", "song_audio_cache"):
            mapped.pop(name, None)
        result = dict(schema=SCHEMA, ready=True, inputs={key: str(value) for key, value in mapped.items()},
            coverage=dict(charts=len(variants), old_windows=len(old), windows=len(old), new_windows=0,
                view_windows=2 * len(old), epochs=parent.available_epochs, audio_variants=len(requests["variants"])),
            batch_counts={"a-plan": [parent.batch_count(epoch) for epoch in range(parent.available_epochs)], "draft-plan": counts},
            export_requests=str(output / "export_requests.json"), learned_inputs_required=False, reused_complete_data=True)
        if holdout is not None:
            result['draft_holdout_song_ids'] = holdout
        write_json(output / "dataset.json", result)
        write_json(output / "manifest.json", result)
        return result
    variants = {}
    templates = {}
    for segment in old:
        record = str(segment["record_id"])
        variant = bpm.get("segments", {}).get(segment["segment_id"], segment.get("audio_variant"))
        if variant is None or (record in variants and variants[record] != variant):
            raise DatasetError("source chart has absent or conflicting audio variants")
        variants[record] = variant
        templates.setdefault(record, segment)
    output.mkdir(parents=True)
    write_json(output / "manifest.json", dict(schema=SCHEMA, ready=False))
    profile = output / "profile"
    profile.mkdir()
    segments = []
    for record, template in templates.items():
        windows = segment_chart(_read(paths["dataset"] / template["encoded_path"]), template)
        end = int(windows[-1]["target_end_tick"])
        for segment in windows:
            segment.update(condition_end_tick=end, audio_variant=variants[record])
        segments.extend(windows)
        destination = profile / template["encoded_path"]
        if not destination.resolve().is_relative_to(profile):
            raise DatasetError("encoded chart destination escaped new profile")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(paths["dataset"] / template["encoded_path"], destination)
    old_ids = {segment["segment_id"] for segment in old}
    if any(segment["segment_id"] not in old_ids and int(segment["target_rows"]) != 1 for segment in segments):
        raise DatasetError("new tail window unexpectedly owns a declaration")
    _rows(profile / "segments.jsonl", segments)
    records = [row for row in load_jsonl(paths["dataset"] / "manifest.jsonl")
               if str(row["record_id"]) in templates]
    _rows(profile / "manifest.jsonl", records)
    shutil.copyfile(paths["dataset"] / "vocabulary.json", profile / "vocabulary.json")
    summary = dict(_read(paths["dataset"] / "summary.json"), segment_count=len(segments),
                   profile_fingerprint=digest(dict(segments=segments, records=records)))
    for name, key in (("segment_split_counts", "split"), ("segment_source_difficulty_counts", "source_difficulty_id")):
        if name in summary:
            summary[name] = {str(value): sum(str(s.get(key, s["difficulty_id"])) == str(value) for s in segments)
                             for value in dict.fromkeys(s.get(key, s["difficulty_id"]) for s in segments)}
    if "target_row_split_counts" in summary:
        summary["target_row_split_counts"] = {"train": sum(int(s["target_rows"]) for s in segments)}
    write_json(profile / "summary.json", summary)
    profile_meta = _read(paths["dataset"] / "profile.json")
    for name in ("base_dataset", "base_feature_stats"):
        if name in profile_meta:
            profile_meta[name] = str((paths["dataset"] / profile_meta[name]).resolve())
    profile_meta.update(profile="audio-draft", summary_sha256=file_sha256(profile / "summary.json"))
    write_json(profile / "profile.json", profile_meta)
    vocabulary = CompactVocabulary.from_dict(_read(profile / "vocabulary.json"))
    canonical = ChartVectorStore.load(paths["chart_cache"], profile_fingerprint="", vocabulary_sha256="", verify_payloads=False)
    spatial = MultiViewRouteStore.load(paths["spatial_view_cache"], profile_fingerprint="", vocabulary_sha256="",
                                      selection_sha256="", verify_payloads=False)
    _standard_vectors(output, segments, old, canonical, spatial, vocabulary, summary["profile_fingerprint"])
    _window_cache(output, segments, old, spatial, vocabulary, variants, bpm)
    counts = _write_plans(parent, output, old, segments, variants)
    mapped = dict(paths, dataset=profile, chart_cache=output / "chart-cache", spatial_view_cache=output / "spatial-views",
        window_cache=output / "window-cache", epoch_plan_cache=output / "a-plan", draft_epoch_plan_cache=output / "draft-plan")
    for name in ("whole_song_plan", "latent_cache", "song_context_cache", "decoder_context_cache",
                 "chart_encoder_checkpoint", "stage_a_checkpoint", "song_chart_cache", "song_audio_cache"):
        mapped.pop(name, None)
    mapped.update(_audio_indices(paths, output, segments, variants, bpm))
    mapped.update(_physical_index(paths, output, segments))
    mapped.update(_note_index(paths, output, segments, mapped["bpm_audio_manifest"]))
    requests = _export_requests(profile, mapped["bpm_audio_manifest"], segments, variants, summary["profile_fingerprint"])
    write_json(output / "export_requests.json", requests)
    windows = WindowCache(output / "window-cache")
    for index, segment in enumerate(segments):
        for view in (0, 1):
            raw = windows.get(index, view, enabled=False)
            if int(raw["target_ids"]["event"].ne(-100).sum()) != int(segment["target_rows"]):
                raise DatasetError("new draft window loader coverage mismatch")
    result = dict(schema=SCHEMA, ready=True, inputs={key: str(value) for key, value in mapped.items()},
        coverage=dict(charts=len(templates), old_windows=len(old), windows=len(segments),
            new_windows=len(segments) - len(old), view_windows=2 * len(segments),
            epochs=parent.available_epochs, audio_variants=len(requests["variants"])),
        batch_counts=counts, export_requests=str(output / "export_requests.json"), learned_inputs_required=False)
    write_json(output / "dataset.json", result)
    write_json(output / "manifest.json", result)
    return result


def prepare_data(args):
    """Keep the accepted A plan intact and add independent canonical validation."""
    if getattr(args, 'validation_song_count', 32) < 1:
        raise DatasetError('validation song count must be positive')
    if getattr(args, 'validation_from_training', False) and getattr(args, 'validation_source_manifest', None) is None:
        raise DatasetError('full holdout requires the accepted validation source manifest')
    result = _prepare_training_data(args)
    source = getattr(args, "validation_source_manifest", None)
    if source is not None:
        from maimai_baseline.data.draft_validation import prepare_validation, merge_requests
        output = Path(args.output).expanduser().resolve()
        validation = prepare_validation(source, result, output / "validation",
            count=getattr(args, 'validation_song_count', 32), selected_songs=result.get('draft_holdout_song_ids'))
        merge_requests(output / "export_requests.json", validation["export_requests"])
        result["validation_dataset_manifest"] = str(output / "validation/dataset.json")
        result["coverage"]["validation"] = validation["coverage"]
        write_json(output / "dataset.json", result)
        write_json(output / "manifest.json", result)
    return result


def resolve_paths(args):
    """Resolve only the draft consumer's inputs; complete chart/μ are unused."""
    root = getattr(args, "dataset_root", None)
    root = Path(root).expanduser().resolve() if root is not None else None
    manifest = getattr(args, "dataset_manifest", None)
    if manifest is not None:
        manifest = Path(manifest).expanduser().resolve()
        root = root or manifest.parent
    elif root is not None:
        manifest = next((root / name for name in ("dataset-v10.json", "dataset.json") if (root / name).is_file()), None)
    values = {}
    if root is not None:
        values.update(dataset="profile", chart_cache="chart-cache", spatial_view_cache="spatial-views",
                      route_artifact="slide-geometry/route-artifact")
    if manifest is not None:
        values.update({key.replace("-", "_"): value for key, value in _read(manifest).get("inputs", {}).items()})
    explicit_plan = getattr(args, "epoch_plan_cache", None)
    for name in ("dataset", "chart_cache", "spatial_view_cache", "route_artifact", "epoch_plan_cache",
                 "draft_epoch_plan_cache", "teacher_audio_cache", "draft_checkpoint"):
        explicit = getattr(args, name, None)
        value = explicit if explicit is not None else values.get(name)
        if value is not None:
            path = Path(value).expanduser()
            if explicit is None and root is not None and not path.is_absolute():
                path = root / path
            setattr(args, name, path.resolve())
    if explicit_plan is None and getattr(args, "draft_epoch_plan_cache", None) is not None:
        args.epoch_plan_cache = args.draft_epoch_plan_cache
    return args


class DraftCollator:
    def __init__(self, teacher, query_rows):
        self.teacher, self.query_rows = teacher, int(query_rows)

    def __call__(self, items):
        keys = [segment_condition(item["segment"], self.teacher) for item in items]
        unique = list(dict.fromkeys(keys))
        draft = build_draft_inputs(self.teacher, unique)
        groups = torch.tensor([unique.index(key) for key in keys], dtype=torch.long)
        blocks = torch.tensor([int(item["segment"]["target_start_tick"]) // 96 for item in items])
        valid = torch.zeros((len(items), 48), dtype=torch.bool)
        for i, item in enumerate(items):
            start, end = int(item["segment"]["target_start_tick"]), int(item["segment"]["target_end_tick"])
            if start < 0 or start % 1536 or end != start + 1536:
                raise DatasetError("draft supervision requires complete aligned 16-beat windows")
            slots = blocks[i] - 16 + torch.arange(48)
            inside = (slots >= 0) & (slots < draft["valid"].shape[1])
            valid[i, inside] = draft["valid"][groups[i], slots[inside]]
            if not bool(valid[i, 16:32].all()):
                raise DatasetError("GT supervision exceeds audio-derived draft horizon")
        targets = {family: {} for family in ("ids", "times")}
        for family, dtype in (("ids", torch.long), ("times", torch.float32)):
            for name in items[0]["targets"][family]:
                result = torch.full((len(items), self.query_rows), -100, dtype=dtype)
                for i, item in enumerate(items):
                    value = item["targets"][family][name]
                    if len(value) > self.query_rows:
                        raise DatasetError("draft target exceeds metadata-derived row queries")
                    result[i, :len(value)] = value
                targets[family][name] = result
        targets["moment_counts"] = torch.stack([moment_counts(item["targets"]) for item in items])
        return dict(draft_inputs=draft, window_groups=groups, window_blocks=blocks, window_valid=valid,
            targets=targets, references=[dict(canonical_index=item["reference"].canonical_index,
                view_id=item["reference"].view_id, occurrence=item["reference"].occurrence) for item in items],
            segment_id=[item["segment"]["segment_id"] for item in items], spatial_view_id=["identity"] * len(items),
            record_id=[str(item["segment"]["record_id"]) for item in items])


def moment_counts(targets):
    """Count distinct MOMENT rows, never NOTE rows, in left-closed beat bins."""
    positions = targets["times"]["moment_position_tick"]
    positions = positions[positions.ne(-100)]
    if (not bool(torch.isfinite(positions).all()) or bool((positions < 0).any())
            or bool((positions >= 1536).any()) or not torch.equal(positions, positions.round())):
        raise DatasetError("MOMENT count labels must lie on the current 96-tick grid inside [0,16)")
    counts = torch.bincount((positions.long() // 96), minlength=16)
    if bool(counts.gt(96).any()):
        raise DatasetError("MOMENT count exceeds the complete 0..96 beat-grid classes")
    return counts


class CanonicalPlanSampler(FixedChartPlanSampler):
    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        if any(reference.view_id != 0 for reference in flatten(self._batches)):
            raise DatasetError("draft pretraining requires the canonical draft plan")


def prepare_inputs(args):
    args = resolve_paths(args)
    for name in ("dataset", "spatial_view_cache", "epoch_plan_cache", "teacher_audio_cache"):
        if getattr(args, name, None) is None:
            raise ValueError("missing draft input --" + name.replace("_", "-"))
    profile = Path(args.dataset)
    plans = EpochPlanStore(args.epoch_plan_cache, verify_payloads=False)
    segments = selected_segments(profile, plans)
    if not segments:
        raise DatasetError("draft plan selects no training windows")
    vocabulary_path = profile / "vocabulary.json"
    vocabulary = CompactVocabulary.from_dict(_read(vocabulary_path))
    vocabulary_sha = file_sha256(vocabulary_path)
    teacher = TeacherAudioCache(args.teacher_audio_cache)
    summary = _read(profile / "summary.json")
    chart_meta = _read(Path(args.chart_cache) / "manifest.json") if getattr(args, "chart_cache", None) else {}
    spatial_meta = _read(Path(args.spatial_view_cache) / "manifest.json")
    dataset = ChartEncoderDataset(segments, vocabulary, args.spatial_view_cache,
        profile_fingerprint=summary["profile_fingerprint"], vocabulary_sha256=vocabulary_sha,
        selection_sha256=str(chart_meta.get("selection", {}).get("selection_sha256", "")))
    validation = None
    validation_manifest = getattr(args, "validation_dataset_manifest", None)
    if validation_manifest is not None:
        from maimai_baseline.data.draft_validation import validation_inputs
        validation = validation_inputs(validation_manifest, teacher, vocabulary,
            segments, seed=int(getattr(args, "seed", 20260811)), workers=int(getattr(args, "num_workers", 12)))
    elif getattr(args, "resolved_recipe", {}).get("validation", "disabled") != "disabled":
        raise DatasetError("one-beat draft training requires --validation-dataset-manifest for 32 held-out songs")
    query_rows = max([int(segment["target_rows"]) for segment in segments]
                     + ([] if validation is None else [validation.query_rows]))
    sampler = CanonicalPlanSampler(plans, dataset)
    offset = int(getattr(args, "resolved_recipe", {}).get("offset", getattr(args, "plan_offset", 0)))
    sampler.set_epoch(offset)
    collator = DraftCollator(teacher, query_rows)
    generator = torch.Generator().manual_seed(int(getattr(args, "seed", 20260811)))
    workers = int(getattr(args, "num_workers", 12))
    loader = ChartEncoderLoader(dataset, batch_sampler=sampler, collate_fn=partial(_transport_collate, collator=collator),
        generator=generator, num_workers=workers, persistent_workers=workers > 0, worker_init_fn=_worker_init)
    audits = [{name: row[name] for name in ("epoch", "mode", "plan_sha256", "file_sha256", "batch_count", "batch_target_rows")
               if name in row} for _, row in sorted(plans._audit_by_epoch.items())]
    binding = dict(vocabulary_sha256=vocabulary_sha, row_codec=vocabulary.row_codec, query_rows=query_rows,
        profile=dict(profile_fingerprint=summary["profile_fingerprint"], window=_read(profile / "profile.json").get("window")),
        segment_order_sha256=digest([s["segment_id"] for s in segments]),
        spatial_chart=_payload_metadata(spatial_meta), chart_cache=_payload_metadata(chart_meta), native_audio=draft_producer_identity(teacher),
        plan=dict(schema=plans.manifest["schema"], mode=plans.mode, epochs=plans.available_epochs,
                  modes=plans.manifest["modes"], audits=audits))
    inputs = ChartEncoderPretrainInputs(vocabulary, None, dataset, loader, sampler, plans,
        query_rows, binding, generator, collator, offset)
    inputs.baseline_manifest = binding
    inputs.profile_manifest = _read(profile / "profile.json")
    inputs.validation = validation
    if validation is not None:
        validation.collator.query_rows = query_rows
        binding["validation"] = validation.binding
    return inputs
