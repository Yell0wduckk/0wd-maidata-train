"""Independent clean audio-draft tokens, shared by canonical and mirror views."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from maimai_baseline.data.audio_draft import (build_draft_inputs, condition_key,
    resolve_paths, segment_condition)
from maimai_baseline.data.curriculum import EpochPlanStore
from maimai_baseline.data.frozen_latent import selected_segments
from maimai_baseline.data.prior_decoder import _read
from maimai_baseline.data.style_cache import write_json
from maimai_baseline.data.teacher_audio import TeacherAudioCache, draft_producer_identity
from maimai_baseline.draft_contract import geometry
from maimai_dataset.integrity import file_sha256
from maimai_dataset.manifest import DatasetError, canonical_json_sha256 as digest

SCHEMA = "maimai.audio_draft_tokens.v1"
ARRAY_DTYPES = dict(raw="float32", valid="bool", beats="float32",
                    window_blocks="int64", window_valid="bool")


class AudioDraftCache:
    def __init__(self, root, *, descriptor=None, fields=None, verify_payloads=False):
        self.root = Path(root).expanduser().resolve()
        self.manifest = _read(self.root / "manifest.json") if descriptor is None else descriptor["manifest"]
        value = self.manifest
        if (value.get("schema") != SCHEMA or not value.get("ready")
                or value.get("dimension") != 128 or value.get("beats_per_token") not in (1, 4)):
            raise DatasetError("audio draft cache schema/shape/readiness mismatch")
        self.geometry = geometry(value)
        self.records = value["records"]
        self.producer = value["producer"]
        if geometry(self.producer["generator_structure"]) != self.geometry:
            raise DatasetError("audio draft cache and producer geometry differ")
        self.normalization = value["normalization"]
        local = self.normalization.get("local", self.normalization)
        mean, std = np.asarray(local.get("mean", ())), np.asarray(local.get("std", ()))
        if (self.normalization.get("scope") != "train-unique-chart-block"
                or mean.shape != (128,) or std.shape != (128,)
                or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any()):
            raise DatasetError("invalid audio draft perturbation statistics")
        self.conditions = {condition_key(*row["key"]): row for row in value["conditions"]}
        if len(self.conditions) != len(value["conditions"]):
            raise DatasetError("duplicate audio draft condition")
        self.train_lookup, self.segment_lookup = {}, {}
        for index, record in enumerate(self.records):
            key = str(record["role"]), str(record["segment_id"]), int(record["view_id"])
            if key in self.segment_lookup:
                raise DatasetError("duplicate audio draft segment/view")
            self.segment_lookup[key] = index
            if record["role"] == "train":
                key = int(record["canonical_index"]), int(record["view_id"])
                if key in self.train_lookup:
                    raise DatasetError("duplicate audio draft canonical/view mapping")
                self.train_lookup[key] = index
        self.arrays = {}
        for name in (ARRAY_DTYPES if fields is None else fields):
            self._array(name, verify=verify_payloads)

    @property
    def identity(self):
        return dict(schema=SCHEMA, dimension=128, beats_per_token=self.geometry.beats_per_token,
            data_binding=self.manifest["data_binding"], producer=self.producer,
            normalization=self.normalization, files=self.manifest["files"],
            conditions=self.manifest["conditions"])

    def descriptor(self):
        return dict(root=str(self.root), manifest=self.manifest)

    def __getstate__(self):
        return dict(self.__dict__, arrays={})

    def _array(self, name, *, verify=False):
        if name not in self.arrays:
            meta = self.manifest["files"][name]
            path = self.root / meta["file"]
            if verify and file_sha256(path) != meta["sha256"]:
                raise DatasetError("new audio draft payload hash mismatch: " + name)
            value = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(value.shape) != meta["shape"] or str(value.dtype) != ARRAY_DTYPES[name]:
                raise DatasetError("audio draft array header mismatch: " + name)
            self.arrays[name] = value
        return self.arrays[name]

    def validate_binding(self, *, profile_fingerprint, vocabulary_sha256, segment_order_sha256):
        for name, expected in dict(profile_fingerprint=profile_fingerprint,
            vocabulary_sha256=vocabulary_sha256, segment_order_sha256=segment_order_sha256).items():
            if self.manifest["data_binding"].get(name) != expected:
                raise DatasetError("audio draft input binding mismatch: " + name)

    def plan_indices(self, refs):
        return [self.train_lookup[(int(ref.canonical_index), int(ref.view_id))] for ref in refs]

    def condition(self, audio_variant, level, true_difficulty):
        row = self.conditions[condition_key(audio_variant, level, true_difficulty)]
        begin, count = int(row["offset"]), int(row["blocks"])
        if (begin < 0 or count < 1 or count % self.geometry.target_slots
                or begin + count > len(self._array("raw"))):
            raise DatasetError("audio draft condition offset escaped cache")
        result = {name: torch.from_numpy(np.array(self._array(name)[begin:begin + count], copy=True))
                  for name in ("raw", "valid", "beats")}
        expected = (torch.arange(count).float() + .5) * self.geometry.beats_per_token
        if (result["raw"].shape != (count, 128) or not bool(result["valid"].all())
                or not torch.isfinite(result["raw"]).all() or not torch.equal(result["beats"], expected)):
            raise DatasetError("invalid clean audio draft condition payload")
        return result

    def window(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or (indices < 0).any() or (indices >= len(self.records)).any():
            raise DatasetError("audio draft window index out of range")
        blocks = np.array(self._array("window_blocks")[indices], copy=True)
        valid = np.array(self._array("window_valid")[indices], copy=True)
        shape = self.geometry
        if (blocks.shape != (len(indices), shape.condition_slots) or valid.shape != blocks.shape
                or (shape.beats_per_token == 4 and not valid.all())):
            raise DatasetError("audio draft windows differ from selected condition geometry")
        coordinates = []
        for row, index in enumerate(indices):
            record = self.records[int(index)]
            condition = self.conditions[condition_key(*record["condition"])]
            start = int(record["start_tick"])
            begin = start // (96 * shape.beats_per_token)
            local = begin + shape.first_slot + np.arange(shape.condition_slots)
            expected_valid = (local >= 0) & (local < int(condition["blocks"]))
            expected = np.where(expected_valid, int(condition["offset"]) + local, -1)
            if (int(record["start_tick"]) % 1536 or int(record["end_tick"]) != int(record["start_tick"]) + 1536
                    or start < 0 or begin + shape.target_slots > int(condition["blocks"])
                    or not np.array_equal(valid[row], expected_valid)
                    or not np.array_equal(blocks[row], expected)):
                raise DatasetError("audio draft window mapping escaped its audio-derived condition")
            coordinates.append((local.astype(np.float32) + .5) * shape.beats_per_token)
        safe = np.maximum(blocks, 0)
        raw = torch.from_numpy(np.array(self._array("raw")[safe], copy=True))
        beats = torch.from_numpy(np.array(coordinates, dtype=np.float32))
        mask = torch.from_numpy(valid)
        if (not torch.isfinite(raw).all()
                or not np.array(self._array("valid")[safe])[valid].all()):
            raise DatasetError("invalid audio draft window tokens")
        raw = raw.masked_fill(~mask[..., None], 0.)
        return dict(raw=raw, latent_tokens=raw, valid=mask, latent_valid=mask, condition_valid=mask, beats=beats)

    def record(self, record_or_segment, view=0, role="train"):
        segment = record_or_segment["segment_id"] if isinstance(record_or_segment, dict) else record_or_segment
        index = self.segment_lookup[(role, str(segment), int(view))]
        result = self.window([index])
        return {key: value[0] for key, value in result.items()}

    def windows(self, items, role="train"):
        indices, views = [], []
        for item in items:
            view = item.get("spatial_view_id", "identity")
            if view not in ("identity", "mirror-lr"):
                raise DatasetError("unsupported audio draft spatial view")
            number = 0 if view == "identity" else 1
            index = self.segment_lookup[(role, str(item["segment_id"]), number)]
            record = self.records[index]
            for name in ("start_tick", "end_tick"):
                if name in item and int(item[name]) != int(record[name]):
                    raise DatasetError("audio draft requested window coordinate mismatch")
            indices.append(index)
            views.append(number)
        result = self.window(indices)
        records = [self.records[index] for index in indices]
        keys = [(record["record_id"], view) for record, view in zip(records, views, strict=True)]
        unique = list(dict.fromkeys(keys))
        return dict(latent_tokens=result["latent_tokens"], latent_valid=result["valid"],
            condition_view=torch.tensor(views, dtype=torch.long),
            window_variants=[record["condition"][0] for record in records],
            window_groups=torch.tensor([unique.index(key) for key in keys], dtype=torch.long),
            window_blocks=torch.tensor([int(record["start_tick"]) // (96 * self.geometry.beats_per_token)
                                        for record in records], dtype=torch.long),
            window_valid=result["valid"], coverage=1.)


def _noise_statistics(chart_conditions, generated):
    first = torch.zeros(128, dtype=torch.float64)
    second = torch.zeros_like(first)
    count = 0
    for record, key in chart_conditions.items():
        values = generated[key].double()
        first += values.sum(0)
        second += values.square().sum(0)
        count += len(values)
    if not count:
        raise DatasetError("draft noise statistics have no training chart blocks")
    mean = first / count
    std = (second / count - mean.square()).clamp_min(1e-8).sqrt()
    return dict(scope="train-unique-chart-block", algorithm="E-z-squared-minus-E-z-square;variance-clamp-1e-8",
        representation="raw-final-draft-token", local=dict(mean=mean.tolist(), std=std.tolist(), units=count))


def build_cache(args):
    """Evaluate the frozen generator once per audio/level/true-difficulty."""
    from maimai_baseline.training.audio_draft_checkpoint import load_draft
    args = resolve_paths(args)
    for name in ("dataset", "epoch_plan_cache", "teacher_audio_cache", "draft_checkpoint"):
        if getattr(args, name, None) is None:
            raise ValueError("missing draft cache input --" + name.replace("_", "-"))
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite clean draft cache: " + str(output))
    teacher = TeacherAudioCache(args.teacher_audio_cache)
    plans = EpochPlanStore(args.epoch_plan_cache, verify_payloads=False)
    profile = Path(args.dataset)
    segments = selected_segments(profile, plans)
    device = torch.device(getattr(args, "device", "cpu"))
    generator, checkpoint = load_draft(args.draft_checkpoint, device=device)
    shape = geometry(checkpoint['structure'])
    generator.eval().requires_grad_(False)
    binding = checkpoint.get("data_binding") or {}
    if "native_audio" in binding and draft_producer_identity(binding["native_audio"]) != draft_producer_identity(teacher):
        raise DatasetError("draft checkpoint and cache require the same native audio producer")
    chart_conditions = OrderedDict()
    conditions = OrderedDict()
    for segment in segments:
        key = segment_condition(segment, teacher)
        record = str(segment["record_id"])
        if record in chart_conditions and chart_conditions[record] != key:
            raise DatasetError("a chart has conflicting audio draft conditions")
        chart_conditions[record] = key
        conditions.setdefault(key, None)
    generated, rows = {}, []
    offset = 0
    audio_groups = OrderedDict()
    for key in conditions:
        audio_groups.setdefault(key[0], []).append(key)
    for keys in audio_groups.values():
        # Several levels/difficulties share the same projected audio K/V.
        inputs = {name: value.to(device) for name, value in build_draft_inputs(
            teacher, keys, beats_per_token=shape.beats_per_token).items()}
        with torch.inference_mode():
            outputs = generator.generate(inputs).detach().float().cpu()
        for index, key in enumerate(keys):
            raw = outputs[index]
            valid = inputs["valid"][index].cpu()
            if raw.shape != (len(valid), 128) or not bool(valid.all()) or not torch.isfinite(raw).all():
                raise DatasetError("generator emitted invalid clean draft tokens")
            generated[key] = raw
            rows.append(dict(key=list(key), offset=offset, blocks=len(raw)))
            conditions[key] = rows[-1]
            offset += len(raw)
    normalization = _noise_statistics(chart_conditions, generated)
    output.mkdir(parents=True)
    write_json(output / "manifest.json", dict(schema=SCHEMA, ready=False))
    raw = torch.cat(list(generated.values())).numpy()
    valid = np.ones(len(raw), dtype=np.bool_)
    beats = np.concatenate([(np.arange(row["blocks"], dtype=np.float32) + .5) * shape.beats_per_token for row in rows])
    records, windows, window_valid = [], [], []
    for index, segment in enumerate(segments):
        key = chart_conditions[str(segment["record_id"])]
        condition = conditions[key]
        start, end = int(segment["target_start_tick"]), int(segment["target_end_tick"])
        begin = start // (96 * shape.beats_per_token)
        slots = begin + shape.first_slot + np.arange(shape.condition_slots)
        slot_valid = (slots >= 0) & (slots < condition['blocks'])
        if start < 0 or start % 1536 or end != start + 1536 or begin + shape.target_slots > condition['blocks']:
            raise DatasetError("profile window exceeds audio-derived draft token coverage")
        for view in (0, 1):
            records.append(dict(role="train", canonical_index=index, segment_id=segment["segment_id"],
                record_id=segment["record_id"], view_id=view, start_tick=start, end_tick=end, condition=list(key)))
            windows.append(np.where(slot_valid, int(condition["offset"]) + slots, -1))
            window_valid.append(slot_valid)
    arrays = dict(raw=raw, valid=valid, beats=beats, window_blocks=np.array(windows, dtype=np.int64),
                  window_valid=np.array(window_valid, dtype=np.bool_))
    files = {}
    for name, value in arrays.items():
        path = output / (name + ".npy")
        np.save(path, value)
        files[name] = dict(file=path.name, shape=list(value.shape), dtype=str(value.dtype),
                           bytes=path.stat().st_size, sha256=file_sha256(path))
    producer = dict(checkpoint=str(Path(args.draft_checkpoint).expanduser().resolve()),
        completed_epochs=checkpoint.get("completed_epochs"), generator_structure=checkpoint["structure"],
        seed=checkpoint.get("seed", generator.seed), recipe=checkpoint.get("recipe"),
        data_binding=checkpoint.get("data_binding"),
        native_audio=draft_producer_identity(teacher))
    manifest = dict(schema=SCHEMA, ready=True, dimension=128, beats_per_token=shape.beats_per_token,
        representation="raw-final-draft-token", producer=producer,
        normalization=normalization, conditions=rows, records=records, files=files,
        data_binding=dict(profile_fingerprint=_read(profile / "summary.json")["profile_fingerprint"],
            vocabulary_sha256=file_sha256(profile / "vocabulary.json"),
            segment_order_sha256=digest([s["segment_id"] for s in segments])),
        coverage=dict(charts=len(chart_conditions), generated_conditions=len(conditions),
            blocks=len(raw), windows=len(segments), view_windows=len(records),
            statistic_blocks=normalization["local"]["units"]))
    write_json(output / "manifest.json", manifest)
    # Once, use the actual reader over these new bytes. Old assets are not scanned.
    cache = AudioDraftCache(output)
    for key in conditions:
        cache.condition(*key)
    for index in range(0, len(records), 256):
        cache.window(list(range(index, min(index + 256, len(records)))))
    return dict(operation="build-draft-cache", output=str(output), **manifest["coverage"],
                completed_epochs=producer["completed_epochs"])
