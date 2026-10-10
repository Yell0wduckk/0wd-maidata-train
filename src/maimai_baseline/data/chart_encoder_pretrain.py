"""Independent chart-only inputs; the accepted epoch plan owns every batch."""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from maimai_baseline.contract import EPOCH_PLAN_MODE, REMOVED_FIELDS, VIEW_CACHE_SCHEMA
from maimai_baseline.data.curriculum import EpochPlanStore, MultiViewRouteStore, VirtualSegmentRef
from maimai_baseline.data.song_chart import SCHEMA as SONG_SCHEMA
from maimai_baseline.data.song_chart_compact import SCHEMA as COMPACT_SCHEMA, _decode
from maimai_baseline.style_contract import STYLE_ID
from maimai_dataset.encoding import TIME_FIELDS as SOURCE_TIME_FIELDS
from maimai_baseline.model.chart_encoder_pretrain import TIME_FIELDS
from maimai_baseline.encoder_audio_contract import CONTRACT
from maimai_dataset.manifest import DatasetError, canonical_json_sha256, load_jsonl
from maimai_dataset.modifiers import LEGACY_FIELDS, ROLES, normalize_inputs, row_masks
from maimai_dataset.vocabulary import CompactVocabulary


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def resolve_paths(args):
    """Resolve only inputs used here, without the A runner's audio requirements."""
    names = dict(dataset="profile", chart_cache="chart-cache", spatial_view_cache="spatial-views",
                 song_chart_cache="song-chart", route_artifact="slide-geometry/route-artifact")
    root = getattr(args, "dataset_root", None)
    root = Path(root).expanduser().resolve() if root is not None else None
    manifest = getattr(args, "dataset_manifest", None)
    if manifest is not None:
        manifest = Path(manifest).expanduser().resolve()
        root = root or manifest.parent
    elif root is not None:
        manifest = next((root / n for n in ("dataset-v10.json", "dataset.json")
                         if (root / n).is_file()), None)
    mapped = dict(names) if root is not None else {}
    if manifest is not None:
        mapped.update({k.replace("-", "_"): v for k, v in _json(manifest).get("inputs", {}).items()})
        args.dataset_manifest = manifest
    for name in (*names, "epoch_plan_cache", "teacher_audio_cache"):
        explicit = getattr(args, name, None)
        value = explicit if explicit is not None else mapped.get(name)
        if value is not None:
            path = Path(value).expanduser()
            if explicit is None and root is not None and not path.is_absolute():
                path = root / path
            setattr(args, name, path.resolve())
    required = ("dataset", "spatial_view_cache", "song_chart_cache", "route_artifact", "epoch_plan_cache")
    missing = [name for name in required if not getattr(args, name, None)]
    if missing:
        raise ValueError("missing chart pretraining paths: " + ", ".join("--" + n.replace("_", "-") for n in missing))
    return args


class ChartOnlySongCache:
    """Read the existing v1/v2 chart layouts, omitting v2 timing/audio sidecars."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        path = self.root / "manifest.json"
        if not path.is_file():
            raise DatasetError(f"accepted complete-chart cache is required at --song-chart-cache: {path}")
        self.manifest = _json(path)
        if (self.manifest.get("schema") not in (SONG_SCHEMA, COMPACT_SCHEMA)
                or self.manifest.get("representation") != STYLE_ID or not self.manifest.get("ready")):
            raise DatasetError("complete-chart cache schema/representation/ready mismatch")
        if (self.manifest["schema"] == COMPACT_SCHEMA
                and self.manifest.get("layout") != "dtype-contiguous-flat-readonly-mmap-v1"):
            raise DatasetError("complete-chart compact layout mismatch")
        self.lookup = {tuple(c["key"]): c for c in self.manifest["charts"]}
        if len(self.lookup) != len(self.manifest["charts"]):
            raise DatasetError("duplicate complete-chart cache key")
        self._arrays = self._v1 = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state.update(_arrays=None, _v1=None)
        return state

    def get(self, key):
        key = tuple(key)
        try:
            chart = self.lookup[key]
        except KeyError as error:
            raise DatasetError(f"complete-chart cache lacks selected chart/view {key}") from error
        if self.manifest["schema"] == SONG_SCHEMA:
            if self._v1 is None:
                from maimai_baseline.data.song_chart import SongChartCache
                self._v1 = SongChartCache(self.root, verify=False)
            return self._v1.get(key)
        if self._arrays is None:
            arrays = {}
            for dtype, meta in self.manifest["files"].items():
                path = self.root / meta["file"]
                if path.stat().st_size != int(meta["bytes"]):
                    raise DatasetError("compact chart buffer byte length mismatch")
                arrays[dtype] = np.memmap(path, dtype=np.dtype(dtype), mode="r")
                if arrays[dtype].size != int(meta["elements"]):
                    raise DatasetError("compact chart buffer element count mismatch")
            self._arrays = arrays
        return _decode(chart["fields"], self._arrays)


def _prototype_inputs(root):
    from maimai_baseline.contract import CATALOG_IDENTITY
    from maimai_baseline.route_contract import GEOMETRY_DIM, GEOMETRY_FILE, GEOMETRY_SCHEMA, geometry_identity
    root = Path(root)
    manifest = _json(root / "manifest.json")
    if (manifest.get("schema") != GEOMETRY_SCHEMA or manifest.get("route_geometry") != geometry_identity()
            or manifest.get("catalog_identity") != CATALOG_IDENTITY):
        raise DatasetError("Route71 prototype artifact schema/geometry mismatch")
    # The encoder only needs prototypes; judge/trajectory matrices are not read.
    with np.load(root / GEOMETRY_FILE, allow_pickle=False) as archive:
        ids = tuple(archive["route71_ids"].astype(str).tolist())
        prototypes = torch.from_numpy(archive["route71_prototypes"].copy())
    if (ids != tuple(f"B{i:03d}" for i in range(71))
            or prototypes.shape != (72, GEOMETRY_DIM) or prototypes.dtype != torch.float32
            or not torch.isfinite(prototypes).all() or bool(prototypes[0].ne(0).any())):
        raise DatasetError("invalid Route71 encoder prototypes")
    expected = manifest.get("route71", {}).get("prototype_sha256")
    actual = hashlib.sha256(prototypes.numpy().tobytes()).hexdigest()
    if expected is not None and actual != expected:
        raise DatasetError("Route71 prototype data identity mismatch")
    return prototypes, dict(schema=manifest["schema"], route_geometry=manifest["route_geometry"],
                            catalog_identity=manifest["catalog_identity"], prototype_sha256=actual)


def canonical_targets(raw, vocabulary, *, start_tick=None):
    """Mask real targets, normalize modifiers, and sort complete NOTE sets."""
    original = raw["target_ids"]
    selected = original["event"].ne(-100).nonzero().flatten()
    source = {n: t.index_select(0, selected).clamp_min(0) for n, t in original.items()}
    times = {n: t.index_select(0, selected).float() for n, t in raw["time_targets"].items()}
    active = {n: t.ne(-100) for n, t in times.items()}
    ids, normalized_times, active = normalize_inputs(source,
        {n: t.masked_fill(~active[n], 0) for n, t in times.items()}, active, vocabulary)
    targets = {n: ids[n].masked_fill(original[n].index_select(0, selected).eq(-100), -100)
               for n in vocabulary.field_names if n not in (*REMOVED_FIELDS, *LEGACY_FIELDS)}
    targets["route"] = original["route"].index_select(0, selected).long().clone()
    events = targets["event"]
    for role in ROLES:
        masks = []
        for i in range(len(events)):
            event = vocabulary.fields["event"].decode(int(events[i]))
            kind = vocabulary.fields["kind"].decode(int(ids["kind"][i]))
            masks.append(row_masks(dict(event=event, kind=kind))[role])
        mask = torch.tensor(masks, dtype=torch.bool)
        targets["modifier_" + role] = ids["modifier_" + role].long().masked_fill(~mask, -100)
    times = {n: normalized_times[n].float().masked_fill(~active[n], -100) for n in SOURCE_TIME_FIELDS}
    moment, note, branch, eos = (vocabulary.event_id(n) for n in ("MOMENT", "NOTE", "SLIDE_BRANCH", "SEGMENT_END"))
    if not len(events) or int(events[-1]) != eos:
        raise DatasetError("pretraining target stream must include its EOS")
    if start_tick is not None:
        for target_index in events.eq(moment).nonzero().flatten().tolist():
            # Only the auxiliary reconstruction labels use absolute positions.
            # Its fixed row queries never receive these anchors as input.
            input_index = int(selected[target_index]) + 1
            if input_index >= raw["row_anchor_tick"].numel():
                raise DatasetError("target MOMENT lacks its input anchor")
            delta = float(raw["row_anchor_tick"][input_index]) - float(start_tick)
            if delta < 0:
                raise DatasetError("target MOMENT precedes its window")
            times["delta_tick"][target_index] = delta
    else:
        indices = events.eq(moment)
        times['delta_tick'][indices] = times['delta_tick'][indices].cumsum(0)
    times['moment_position_tick'] = times.pop('delta_tick')
    for name, value in times.items():
        active_time = value.ne(-100)
        if not torch.isfinite(value[active_time]).all() or bool(value[active_time].lt(0).any()):
            raise DatasetError(f"invalid active chart pretraining time target: {name}")
    names = sorted(n for n in targets if n != "branch_index")
    def semantic(index):
        return tuple(int(targets[n][index]) for n in names) + tuple(float(times[n][index]) for n in TIME_FIELDS)
    order, index = [], 0
    while index < len(events) - 1:
        if int(events[index]) != moment:
            raise DatasetError("pretraining target NOTE set lacks its MOMENT")
        order.append(index)
        index += 1
        blocks = []
        while index < len(events) - 1 and int(events[index]) != moment:
            if int(events[index]) != note:
                raise DatasetError("pretraining SLIDE_BRANCH lacks its owning NOTE")
            head = index
            index += 1
            branches = []
            while index < len(events) - 1 and int(events[index]) == branch:
                branches.append(index)
                index += 1
            branches.sort(key=semantic)
            # branch_index names the child ordinal, not the physical route.
            if "branch_index" in targets:
                for ordinal, child in enumerate(branches, 1):
                    if int(targets["branch_index"][child]) != -100:
                        targets["branch_index"][child] = vocabulary.fields["branch_index"].encode(ordinal)
            blocks.append((head, branches))
        blocks.sort(key=lambda block: (semantic(block[0]), tuple(semantic(c) for c in block[1])))
        for head, children in blocks:
            order.extend((head, *children))
    order.append(len(events) - 1)
    indices = torch.tensor(order, dtype=torch.long)
    return dict(ids={n: t.index_select(0, indices) for n, t in targets.items()},
                times={n: t.index_select(0, indices) for n, t in times.items()})


class ChartEncoderDataset(Dataset):
    def __init__(self, segments, vocabulary, spatial_root, *, profile_fingerprint, vocabulary_sha256,
                 selection_sha256=""):
        self.segments = tuple(dict(s) for s in segments)
        self.target_lengths = tuple(int(s["target_rows"]) for s in segments)
        self.vocabulary = vocabulary
        self.spatial_root = Path(spatial_root)
        self.spatial_binding = dict(profile_fingerprint=profile_fingerprint,
            vocabulary_sha256=vocabulary_sha256, selection_sha256=selection_sha256)
        self._spatial = None

    def __len__(self):
        return len(self.segments)

    def __getstate__(self):
        return dict(self.__dict__, _spatial=None)

    def __getitem__(self, reference):
        if not isinstance(reference, VirtualSegmentRef):
            reference = VirtualSegmentRef(int(reference), 0, 0)
        if not 0 <= reference.canonical_index < len(self.segments):
            raise IndexError(reference.canonical_index)
        if self._spatial is None:
            self._spatial = MultiViewRouteStore.load(self.spatial_root, verify_payloads=False, **self.spatial_binding)
        segment = self.segments[reference.canonical_index]
        raw = self._spatial.record(segment["segment_id"], reference.view)
        targets = canonical_targets(raw, self.vocabulary, start_tick=segment["target_start_tick"])
        if targets["ids"]["event"].numel() != int(segment["target_rows"]):
            raise DatasetError(f"target length metadata mismatch for {segment['segment_id']}")
        return dict(segment=segment, reference=reference, targets=targets)


class FixedChartPlanSampler(Sampler):
    def __init__(self, store, dataset):
        self.store, self.dataset = store, dataset
        self.epoch, self._batches = 0, None

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self._batches = self.store.load_epoch(self.epoch)
        costs = []
        for batch in self._batches:
            if not batch:
                raise DatasetError("fixed pretraining plan contains an empty batch")
            if any(not 0 <= ref.canonical_index < len(self.dataset) for ref in batch):
                raise DatasetError("fixed pretraining plan segment index is outside its bound dataset")
            costs.append(sum(self.dataset.target_lengths[r.canonical_index] for r in batch))
        expected = self.store.audit(self.epoch).get("batch_target_rows")
        if expected is not None and costs != list(expected):
            raise DatasetError("fixed pretraining plan target-row counts mismatch")
        self.batch_target_rows = tuple(costs)

    def __iter__(self):
        if self._batches is None:
            self.set_epoch(self.epoch)
        for batch in self._batches:
            yield list(batch)

    def __len__(self):
        return self.store.batch_count(self.epoch)


class ChartEncoderCollator:
    def __init__(self, cache, query_rows, audio_teacher=None, condition_ends=None):
        self.cache, self.query_rows = cache, int(query_rows)
        self.audio_teacher = audio_teacher
        self.condition_ends = condition_ends or {}

    def __call__(self, items):
        if not items:
            raise DatasetError("empty pretraining batch")
        groups, songs, window_groups, blocks, valid, variants = {}, [], [], [], [], []
        for item in items:
            segment, ref = item["segment"], item["reference"]
            record = str(segment.get("record_id") or str(segment["segment_id"]).rsplit(":", 1)[0])
            key = ("train", record, ref.view_id)
            if key not in groups:
                groups[key] = len(songs)
                packed = self.cache.get(key)
                condition_end = self.condition_ends.get(record, int(segment.get('condition_end_tick', 0)))
                if condition_end:
                    if condition_end % 384 or condition_end < 384:
                        raise DatasetError('pretraining condition end must be four-beat aligned')
                    length = condition_end // 384
                    slots = packed['moment_slot']
                    if bool(((slots < 0) | (slots >= length)).any()):
                        raise DatasetError('real chart rows exceed condition coverage; extend profile')
                    packed = dict(packed, valid=torch.ones((1, length), dtype=torch.bool))
                songs.append(packed)
            group = groups[key]
            start, end, song_end = (int(segment[n]) for n in ("target_start_tick", "target_end_tick", "song_end_tick"))
            if start < 0 or start % 384:
                raise DatasetError("pretraining window is not aligned to four-beat blocks")
            block = start // 384
            packed_valid = songs[group]["valid"].reshape(-1)
            slots = block + torch.arange(4)
            mask = start + torch.arange(4) * 384 < end
            if not self.condition_ends.get(record, int(segment.get('condition_end_tick', 0))):
                mask &= start + torch.arange(4) * 384 <= song_end
            in_bounds = slots < packed_valid.numel()
            if bool((mask & ~in_bounds).any()):
                raise DatasetError("complete-chart cache does not cover the selected window")
            mask &= in_bounds
            mask[in_bounds] &= packed_valid[slots[in_bounds]]
            window_groups.append(group)
            blocks.append(block)
            valid.append(mask)
            if self.audio_teacher is not None:
                variants.append(segment.get('audio_variant') or self.audio_teacher.manifest['record_to_variant'][record])
        targets = {}
        for family, dtype in (("ids", torch.long), ("times", torch.float32)):
            targets[family] = {}
            for name in items[0]["targets"][family]:
                out = torch.full((len(items), self.query_rows), -100, dtype=dtype)
                for i, item in enumerate(items):
                    value = item["targets"][family][name]
                    if len(value) > self.query_rows:
                        raise DatasetError("target stream exceeds the metadata-derived query limit")
                    out[i, :len(value)] = value
                targets[family][name] = out
        batch = dict(songs=songs, window_groups=torch.tensor(window_groups, dtype=torch.long),
            window_blocks=torch.tensor(blocks, dtype=torch.long), window_valid=torch.stack(valid), targets=targets,
            references=[dict(canonical_index=i["reference"].canonical_index, view_id=i["reference"].view_id,
                             occurrence=i["reference"].occurrence) for i in items],
            segment_id=[i["segment"]["segment_id"] for i in items],
            spatial_view_id=[i["reference"].view for i in items],
            record_id=[str(i["segment"].get("record_id") or str(i["segment"]["segment_id"]).rsplit(":", 1)[0]) for i in items])
        if self.audio_teacher is not None:
            batch['audio_teacher'] = self.audio_teacher.pretrain_batch(variants)
        return batch


def _transport_collate(items, collator):
    from maimai_baseline.data.batch_transport import pack_batch
    return pack_batch(collator(items))


def _worker_init(_):
    torch.set_num_threads(1)


class ChartEncoderLoader(DataLoader):
    """Few IPC buffers per batch, with a normal dict at the training boundary."""
    def __iter__(self):
        # Create workers/base_seed now, so the runner can restore the saved
        # generator state immediately after iter(loader) on a resumed epoch.
        iterator = super().__iter__()
        return (batch.unpack() for batch in iterator)


@dataclass
class ChartEncoderPretrainInputs:
    vocabulary: CompactVocabulary
    prototypes: torch.Tensor
    dataset: ChartEncoderDataset
    loader: DataLoader
    sampler: FixedChartPlanSampler
    plan_store: EpochPlanStore
    query_rows: int
    binding: dict[str, Any]
    loader_generator: torch.Generator
    collator: ChartEncoderCollator
    plan_offset: int = 0

    @property
    def train_loader(self):
        return self.loader

    @property
    def data_binding(self):
        return self.binding

    @property
    def manifest(self):
        return self.binding


def _payload_metadata(manifest):
    """Existing data hashes/layout only; producer and source labels are excluded."""
    result = {n: manifest[n] for n in ("schema", "layout", "representation", "segment_count", "views",
                                       "arrays", "offsets", "segment_index", "files") if n in manifest}
    if "charts" in manifest:
        result["charts"] = [{n: c[n] for n in ("key", "file", "sha256", "bytes", "fields") if n in c}
                            for c in manifest["charts"]]
    return result


def prepare_inputs(args):
    args = resolve_paths(args)
    profile = Path(args.dataset)
    summary, profile_meta = _json(profile / "summary.json"), _json(profile / "profile.json")
    vocabulary_path = profile / "vocabulary.json"
    vocabulary_sha = hashlib.sha256(vocabulary_path.read_bytes()).hexdigest()
    vocabulary = CompactVocabulary.from_dict(_json(vocabulary_path))
    plans = EpochPlanStore(args.epoch_plan_cache, verify_payloads=False)
    subsets = plans.manifest.get("experiment", {}).get("subsets", plans.manifest.get("subsets", {}))
    train = subsets.get("train", {})
    records = train.get("records")
    selected_records = set(map(str, records)) if records is not None else None
    # The supplied plan's canonical ordering is the profile's selected ordering.
    segments = [s for s in load_jsonl(profile / "segments.jsonl") if s.get("split") == "train"
                and (selected_records is None or str(s.get("record_id") or str(s["segment_id"]).rsplit(":", 1)[0]) in selected_records)]
    if not segments:
        raise DatasetError("fixed chart pretraining plan selects no training segments")
    order_sha = canonical_json_sha256([s["segment_id"] for s in segments])
    expected = plans.manifest.get("segment_order_sha256", train.get("segment_order_sha256"))
    if expected is not None and order_sha != expected:
        raise DatasetError("fixed chart pretraining plan segment order mismatch")
    if train.get("segments") is not None and len(segments) != int(train["segments"]):
        raise DatasetError("fixed chart pretraining plan segment count mismatch")
    query_rows = max(int(s["target_rows"]) for s in segments)  # Already includes SEGMENT_END.
    if query_rows < 1:
        raise DatasetError("target length metadata must include EOS")
    cache = ChartOnlySongCache(args.song_chart_cache)
    spatial_meta = _json(Path(args.spatial_view_cache) / "manifest.json")
    if spatial_meta.get("schema") != VIEW_CACHE_SCHEMA:
        raise DatasetError("unsupported pretraining spatial chart cache schema")
    prototypes, prototype_binding = _prototype_inputs(args.route_artifact)
    chart_meta = _json(Path(args.chart_cache) / "manifest.json") if getattr(args, "chart_cache", None) else {}
    dataset = ChartEncoderDataset(segments, vocabulary, args.spatial_view_cache,
        profile_fingerprint=str(summary["profile_fingerprint"]), vocabulary_sha256=vocabulary_sha,
        selection_sha256=str(chart_meta.get("selection", {}).get("selection_sha256", "")))
    sampler = FixedChartPlanSampler(plans, dataset)
    offset = int(getattr(args, "resolved_recipe", {}).get("offset", getattr(args, "plan_offset", 0)))
    sampler.set_epoch(offset)
    teacher_root = getattr(args, 'teacher_audio_cache', None)
    teacher = None
    if teacher_root is not None:
        from maimai_baseline.data.teacher_audio import TeacherAudioCache
        teacher = TeacherAudioCache(teacher_root)
    condition_ends = {}
    for segment in segments:
        record = str(segment.get('record_id') or str(segment['segment_id']).rsplit(':', 1)[0])
        condition_ends[record] = max(condition_ends.get(record, 0), int(segment.get('condition_end_tick', 0)))
    collator = ChartEncoderCollator(cache, query_rows, teacher, condition_ends)
    generator = torch.Generator().manual_seed(int(getattr(args, "seed", 20260811)))
    workers = int(getattr(args, "num_workers", 12))
    loader = ChartEncoderLoader(dataset, batch_sampler=sampler,
        collate_fn=partial(_transport_collate, collator=collator), generator=generator,
        num_workers=workers, persistent_workers=workers > 0, worker_init_fn=_worker_init)
    audits = [{n: row[n] for n in ("epoch", "mode", "plan_sha256", "file_sha256", "batch_count", "batch_target_rows")
               if n in row} for _, row in sorted(plans._audit_by_epoch.items())]
    binding = dict(vocabulary_sha256=vocabulary_sha, row_codec=vocabulary.row_codec, query_rows=query_rows,
        profile=dict(profile_fingerprint=summary["profile_fingerprint"], row_codec=profile_meta.get("row_codec"),
                     window=profile_meta.get("window")),
        segment_order_sha256=order_sha, song_chart=_payload_metadata(cache.manifest),
        spatial_chart=_payload_metadata(spatial_meta), chart_cache=_payload_metadata(chart_meta),
        prototypes=prototype_binding,
        plan=dict(schema=plans.manifest["schema"], mode=EPOCH_PLAN_MODE, epochs=plans.available_epochs,
                  modes=plans.manifest["modes"], segment_order_sha256=order_sha, subsets=subsets, audits=audits))
    if teacher is not None:
        binding.update(native_audio=teacher.identity, encoder_audio_contract=CONTRACT)
    return ChartEncoderPretrainInputs(vocabulary, prototypes, dataset, loader, sampler, plans,
                                     query_rows, binding, generator, collator, offset)


def fixed_diagnostic_batch(inputs, args=None, count=4):
    """Fixed first recipe-epoch references; no loader RNG or sampler mutation."""
    offset = int(getattr(args, "resolved_recipe", {}).get("offset", inputs.plan_offset))
    references = inputs.plan_store.load_epoch(offset)[0][:int(count)]
    return inputs.collator([inputs.dataset[ref] for ref in references])
