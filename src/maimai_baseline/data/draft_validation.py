"""Independent canonical-song validation, without learned or Physical assets."""
from __future__ import annotations

from copy import deepcopy
from functools import partial
from pathlib import Path
import random
import shutil
from types import SimpleNamespace

import torch
from torch.utils.data import Dataset, Sampler

from maimai_baseline.data.chart_encoder_pretrain import (ChartEncoderLoader,
    canonical_targets, _transport_collate, _worker_init)
from maimai_baseline.data.curriculum import VirtualSegmentRef
from maimai_baseline.data.encoder_audio import _export_requests
from maimai_baseline.data.prior_decoder import _read, _rows
from maimai_baseline.data.route import Route71Codec
from maimai_baseline.data.style_cache import write_json
from maimai_baseline.data.training_build import segment_chart
from maimai_dataset.chart_cache import materialize_chart_vectors
from maimai_dataset.integrity import file_sha256
from maimai_dataset.manifest import DatasetError, canonical_json_sha256 as digest, load_jsonl

SCHEMA = "maimai.audio_draft_validation.v1"


def mapped_inputs(manifest):
    path = Path(manifest).expanduser().resolve()
    values = _read(path).get("inputs", {})
    result = {}
    for name, value in values.items():
        candidate = Path(value).expanduser()
        result[name.replace("-", "_")] = (candidate if candidate.is_absolute() else path.parent / candidate).resolve()
    return result


def _audio_identity(meta):
    # Existing identities only: never reread audio to prove disjointness.
    names = ("source_sha256", "source_audio_sha256", "audio_sha256", "wav_sha256",
             "normalized_audio_sha256", "feature_sha256")
    result = {("feature" if name == "feature_sha256" else "source-audio", str(meta[name]))
              for name in names if meta.get(name)}
    identity = meta.get('identity')
    if isinstance(identity, dict) and identity.get('original_sha256'):
        result.add(('source-audio', str(identity['original_sha256'])))
    if isinstance(meta.get('audio_identity'), str):
        result.add(('source-audio', meta['audio_identity']))
    return result


def select_validation(records, source_segments, source_bpm, training_segments, training_bpm, *, count=32, seed=20260811):
    train_songs = {str(row["song_id"]) for row in training_segments}
    train_variants = {training_bpm.get("segments", {}).get(row["segment_id"], row.get("audio_variant"))
                      for row in training_segments}
    forbidden_audio = set()
    for segment in training_segments:
        forbidden_audio |= _audio_identity(segment)
    for key in train_variants:
        forbidden_audio |= _audio_identity(training_bpm["variants"].get(key, {}))
    templates = {}
    for segment in source_segments:
        templates.setdefault(str(segment["record_id"]), segment)
    candidates, blocked_songs = {}, set()
    for record in records:
        rid, song = str(record["record_id"]), str(record["song_id"])
        template = templates.get(rid)
        difficulty = int(record.get("source_difficulty_id", (template or {}).get("source_difficulty_id", -1)))
        if (template is None or difficulty not in (5, 6) or song in train_songs
                or record.get("status", "eligible") != "eligible" or not record.get("encoded_path")):
            continue
        variant = source_bpm.get("segments", {}).get(template["segment_id"], template.get("audio_variant"))
        if variant not in source_bpm["variants"]:
            continue
        identities = (_audio_identity(source_bpm["variants"][variant]) | _audio_identity(record)
                      | _audio_identity(template))
        if variant in train_variants or identities & forbidden_audio:
            blocked_songs.add(song)
            continue
        candidates.setdefault(song, []).append((record, template, variant))
    for song in blocked_songs:
        candidates.pop(song, None)
    # Song IDs can alias the same recording. Keep the first stable song ID,
    # then sample only distinct existing audio identities without loading bytes.
    distinct, seen_audio = {}, set()
    for song in sorted(candidates):
        identities = set()
        for record, template, variant in candidates[song]:
            identities.add(('variant', variant))
            identities |= (_audio_identity(source_bpm['variants'][variant])
                           | _audio_identity(record) | _audio_identity(template))
        if not identities & seen_audio:
            distinct[song] = candidates[song]
            seen_audio |= identities
    candidates = distinct
    songs = sorted(candidates)
    if len(songs) < count:
        raise DatasetError(f"draft validation needs {count} independent songs; only {len(songs)} eligible full-data candidates")
    chosen = random.Random(seed).sample(songs, count)
    return chosen, [item for song in chosen for item in sorted(candidates[song], key=lambda row: str(row[0]["record_id"]))]


def select_training_holdout(records, segments, bpm, *, count=64, seed=20260811):
    """Sample songs from the accepted full pool without audio aliases leaking."""
    owners = {}
    by_record = {str(row['record_id']): row for row in records}
    for row in segments:
        song = str(row['song_id'])
        variant = bpm.get('segments', {}).get(row['segment_id'], row.get('audio_variant'))
        identities = {('variant', variant)} | _audio_identity(row)
        identities |= _audio_identity(by_record.get(str(row['record_id']), {}))
        identities |= _audio_identity(bpm['variants'].get(variant, {}))
        for identity in identities:
            owners.setdefault(identity, set()).add(song)
    aliased = set().union(*(songs for songs in owners.values() if len(songs) > 1))
    candidates = [row for row in records if str(row['song_id']) not in aliased]
    songs, _ = select_validation(candidates, segments, bpm, [], bpm, count=count, seed=seed)
    return songs


def prepare_validation(source_manifest, training, output, *, count=32, selected_songs=None):
    source = mapped_inputs(source_manifest)
    output = Path(output).resolve()
    profile = source["dataset"]
    training_profile = Path(training["inputs"]["dataset"])
    from maimai_baseline.data.curriculum import EpochPlanStore
    from maimai_baseline.data.frozen_latent import selected_segments
    train_segments = selected_segments(training_profile, EpochPlanStore(
        training["inputs"].get("draft_epoch_plan_cache", training["inputs"]["epoch_plan_cache"]), verify_payloads=False))
    source_segments = load_jsonl(profile / "segments.jsonl")
    source_bpm, train_bpm = _read(source["bpm_audio_manifest"]), _read(training["inputs"]["bpm_audio_manifest"])
    records = load_jsonl(profile / "manifest.jsonl")
    if selected_songs is not None:
        records = [row for row in records if str(row['song_id']) in set(selected_songs)]
    songs, selected = select_validation(records, source_segments,
        source_bpm, train_segments, train_bpm, count=count)
    if selected_songs is not None and set(songs) != set(selected_songs):
        raise DatasetError('selected full holdout differs from validation songs')
    if file_sha256(profile / "vocabulary.json") != file_sha256(training_profile / "vocabulary.json"):
        raise DatasetError("validation source vocabulary differs from the accepted training vocabulary")
    if output.exists():
        raise FileExistsError("refusing to overwrite draft validation data: " + str(output))
    target = output / "profile"
    target.mkdir(parents=True)
    segments, records, variants = [], [], {}
    for record, template, variant in selected:
        rid = str(record["record_id"])
        destination = target / record["encoded_path"]
        if not destination.resolve().is_relative_to(target):
            raise DatasetError("validation encoded chart path escaped profile")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(profile / record["encoded_path"], destination)
        template = dict(template, split="validation", encoded_path=record["encoded_path"])
        windows = ([dict(row) for row in source_segments if str(row['record_id']) == rid]
                   if selected_songs is not None else segment_chart(_read(destination), template))
        windows.sort(key=lambda row: int(row['target_start_tick']))
        end = int(windows[-1]["target_end_tick"])
        for segment in windows:
            segment.update(condition_end_tick=end, audio_variant=variant, split="validation")
        segments.extend(windows)
        records.append(dict(record, split="validation"))
        variants[rid] = variant
    _rows(target / "segments.jsonl", segments)
    _rows(target / "manifest.jsonl", records)
    shutil.copyfile(training_profile / "vocabulary.json", target / "vocabulary.json")
    fingerprint = digest(dict(segments=segments, records=records))
    write_json(target / "summary.json", dict(profile_fingerprint=fingerprint, segment_count=len(segments)))
    metadata = deepcopy(_read(profile / "profile.json"))
    for name in ("base_dataset", "base_feature_stats"):
        if name in metadata:
            metadata[name] = str((profile / metadata[name]).resolve())
    metadata.update(profile=f"audio-draft-held-out-{count}", summary_sha256=file_sha256(target / "summary.json"))
    write_json(target / "profile.json", metadata)
    # Persist only the new canonical labels; normal training loaders do not
    # adapt charts, consult Physical, or inspect old payloads again.
    from maimai_dataset.vocabulary import CompactVocabulary
    vocabulary = CompactVocabulary.from_dict(_read(target / "vocabulary.json"))
    codec = Route71Codec(vocabulary)
    tensor_root = output / "targets"
    tensor_root.mkdir()
    for index, segment in enumerate(segments):
        raw = codec.adapt_item(materialize_chart_vectors(profile_root=target, segment=segment, vocabulary=vocabulary))
        labels = canonical_targets(raw, vocabulary, start_tick=segment["target_start_tick"])
        if len(labels["ids"]["event"]) != int(segment["target_rows"]):
            raise DatasetError("validation target rows differ from the new window metadata")
        torch.save(labels, tensor_root / f"{index:06d}.pt")
    request = _export_requests(target, source["bpm_audio_manifest"], segments, variants, fingerprint)
    write_json(output / "export_requests.json", request)
    coverage = dict(songs=len(songs), charts=len(records), windows=len(segments), canonical_only=True)
    result = dict(schema=SCHEMA, ready=True, inputs=dict(dataset=str(target), validation_targets=str(tensor_root)),
        song_ids=songs, seed=20260811, source_manifest=str(Path(source_manifest).resolve()),
        training_song_ids=sorted({str(s["song_id"]) for s in train_segments}),
        coverage=coverage, query_rows=max(int(s["target_rows"]) for s in segments),
        export_requests=str(output / "export_requests.json"), token_budget=4096)
    write_json(output / "dataset.json", result)
    return result


def merge_requests(destination, validation_request):
    destination, validation_request = Path(destination), Path(validation_request)
    main, extra = _read(destination), _read(validation_request)
    joint = deepcopy(main)
    joint_bpm = deepcopy(_read(main["bpm_audio_manifest"]))
    joint_bpm["variants"] = {}
    joint_bpm["segments"] = {}
    stats_identity = None
    for request in (main, extra):
        path = Path(request["bpm_audio_manifest"])
        bpm = _read(path)
        identity = bpm.get("stats_sha256")
        if stats_identity is not None and identity is not None and identity != stats_identity:
            raise DatasetError("train/validation frozen native audio normalization differs")
        stats_identity = stats_identity or identity
        joint_bpm["segments"].update(bpm["segments"])
        for key, meta in bpm["variants"].items():
            if key not in request["variants"]:
                continue
            copied = deepcopy(meta)
            for field in ("feature_path", "coordinates_path", "source_audio_path", "audio_path"):
                if field in copied:
                    copied[field] = str((path.parent / copied[field]).resolve())
            if key in joint_bpm["variants"] and copied != joint_bpm["variants"][key]:
                raise DatasetError("combined native audio metadata conflict: " + key)
            joint_bpm["variants"][key] = copied
        for key, meta in request["variants"].items():
            if key in joint["variants"]:
                old = joint["variants"][key]
                if old["num_samples"] != meta["num_samples"] or old["timing"] != meta["timing"]:
                    raise DatasetError("combined native audio variant conflict: " + key)
                old["starts"] = sorted(set(old["starts"]) | set(meta["starts"]))
            else:
                joint["variants"][key] = deepcopy(meta)
        joint["record_to_variant"].update(request["record_to_variant"])
    # Limit this index to requested variants; metadata references accepted bytes.
    joint_bpm["variants"] = {key: joint_bpm["variants"][key] for key in joint["variants"]}
    joint_path = destination.parent / "combined-bpm-audio-manifest.json"
    write_json(joint_path, joint_bpm)
    joint["bpm_audio_manifest"] = str(joint_path.resolve())
    write_json(destination, joint)


class ValidationDataset(Dataset):
    def __init__(self, segments, root):
        self.segments, self.root = segments, Path(root)
        self.target_lengths = [int(row["target_rows"]) for row in segments]

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, reference):
        if not isinstance(reference, VirtualSegmentRef):
            reference = VirtualSegmentRef(int(reference), 0, 0)
        if reference.view_id != 0 or reference.occurrence != 0:
            raise DatasetError("validation is canonical occurrence0 without augmentation")
        index = reference.canonical_index
        targets = torch.load(self.root / f"{index:06d}.pt", map_location="cpu", weights_only=False)
        return dict(segment=self.segments[index], reference=reference, targets=targets)


class ValidationSampler(Sampler):
    def __init__(self, dataset, budget=4096):
        self.batches = []
        pending, used = [], 0
        for index, cost in enumerate(dataset.target_lengths):
            if pending and used + cost > budget:
                self.batches.append(pending)
                pending, used = [], 0
            pending.append(VirtualSegmentRef(index, 0, 0))
            used += cost
        if pending:
            self.batches.append(pending)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def validation_inputs(manifest, teacher, vocabulary, training_segments, *, seed, workers):
    from maimai_baseline.data.audio_draft import DraftCollator
    path = Path(manifest).resolve()
    metadata, inputs = _read(path), mapped_inputs(path)
    if metadata.get("schema") != SCHEMA or not metadata.get("ready"):
        raise DatasetError("draft validation manifest schema/readiness mismatch")
    segments = load_jsonl(inputs["dataset"] / "segments.jsonl")
    songs = {str(row["song_id"]) for row in segments}
    count = int(metadata['coverage']['songs'])
    if count < 1 or len(songs) != count or songs != set(metadata["song_ids"]):
        raise DatasetError("draft validation must contain exactly the declared songs")
    if songs & {str(row["song_id"]) for row in training_segments}:
        raise DatasetError("draft validation song leaks into the training set")
    from maimai_dataset.vocabulary import CompactVocabulary
    validation_vocabulary = CompactVocabulary.from_dict(_read(inputs["dataset"] / "vocabulary.json"))
    if validation_vocabulary.to_dict() != vocabulary.to_dict():
        raise DatasetError("draft validation vocabulary differs")
    dataset = ValidationDataset(segments, inputs["validation_targets"])
    sampler = ValidationSampler(dataset)
    query_rows = max(dataset.target_lengths)
    collator = DraftCollator(teacher, query_rows)
    generator = torch.Generator().manual_seed(seed + 1)
    loader = ChartEncoderLoader(dataset, batch_sampler=sampler,
        collate_fn=partial(_transport_collate, collator=collator), generator=generator,
        num_workers=workers, persistent_workers=workers > 0, worker_init_fn=_worker_init)
    binding = dict(schema=SCHEMA, song_ids=metadata["song_ids"], segment_order_sha256=digest(
        [row["segment_id"] for row in segments]), profile_fingerprint=_read(inputs["dataset"] / "summary.json")["profile_fingerprint"],
        query_rows=query_rows, batches=[[r.canonical_index for r in batch] for batch in sampler.batches])
    return SimpleNamespace(dataset=dataset, sampler=sampler, loader=loader,
        loader_generator=generator, collator=collator, query_rows=query_rows, binding=binding)
