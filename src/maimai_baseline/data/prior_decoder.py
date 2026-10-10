"""Complete tail-window inputs, full encoder60 mu and reused continuous P0."""
from collections import defaultdict
from copy import deepcopy
from itertools import groupby
from pathlib import Path
from types import SimpleNamespace
import json
import shutil

import numpy as np
import torch

from maimai_baseline.contract import AUGMENTATION_SEED, VIEW_NAMES
from maimai_baseline.data.curriculum import EpochPlanStore, MultiViewRouteStore, VirtualSegmentRef
from maimai_baseline.data.frozen_latent import FrozenLatentCache, selected_segments
from maimai_baseline.data.repack import flatten, reference_hash
from maimai_baseline.data.segments import prepare_raw_prefix_item
from maimai_baseline.data.style_cache import write_json
from maimai_baseline.data.training_build import segment_chart
from maimai_baseline.runtime_versions import operational, stamp
from maimai_baseline.whole_song_training import repack_frozen
from maimai_dataset.chart_cache import ChartVectorStore, audio_reference
from maimai_dataset.integrity import file_sha256
from maimai_dataset.manifest import DatasetError, canonical_json_sha256 as digest, load_jsonl
from maimai_dataset.vocabulary import CompactVocabulary

SCHEMA = "maimai.prior_decoder_tail_data.v1"
WINDOW_SCHEMA = "maimai.prior_decoder_window_inputs.v1"


def _read(path):
    return json.loads(Path(path).read_text("utf-8-sig"))


def _rows(path, values):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(v, allow_nan=False) + "\n" for v in values), "utf8")


def resolve_paths(args):
    root = getattr(args, "dataset_root", None)
    root = Path(root).expanduser().resolve() if root is not None else None
    manifest = getattr(args, "dataset_manifest", None)
    if manifest is not None:
        manifest = Path(manifest).expanduser().resolve()
        root = root or manifest.parent
    elif root is not None:
        manifest = next((root / n for n in ("dataset-v10.json", "dataset.json") if (root / n).is_file()), None)
    mappings = dict(dataset="profile", chart_cache="chart-cache", spatial_view_cache="spatial-views",
                    song_chart_cache="song-chart", route_artifact="slide-geometry/route-artifact",
                    bpm_audio_manifest="audio/bpm-audio-manifest.json") if root is not None else {}
    if manifest is not None:
        mappings.update({k.replace("-", "_"): v for k, v in _read(manifest).get("inputs", {}).items()})
    resolved = {}
    for name in (*mappings, "dataset", "chart_cache", "spatial_view_cache", "bpm_audio_manifest",
                 "epoch_plan_cache", "latent_cache", "song_context_cache", "song_chart_cache",
                 "route_artifact", "chart_encoder_checkpoint"):
        explicit = getattr(args, name, None)
        value = explicit if explicit is not None else mappings.get(name)
        if value is None:
            continue
        path = Path(value).expanduser()
        if explicit is None and root is not None and not path.is_absolute():
            path = root / path
        resolved[name] = path.resolve()
    required = ("dataset", "chart_cache", "spatial_view_cache", "bpm_audio_manifest", "epoch_plan_cache", "song_context_cache")
    required += (("song_chart_cache", "route_artifact") if "chart_encoder_checkpoint" in resolved else ("latent_cache",))
    missing = [n for n in required if n not in resolved]
    if missing:
        raise ValueError("missing tail migration inputs: " + ", ".join("--" + n.replace("_", "-") for n in missing))
    return resolved


def validate_derived_latent(cache, parent_binding):
    """New window indices may differ; the imported A's actual mu corpus may not."""
    source = cache.manifest.get("derived_from")
    if source is None or operational(source) != operational(parent_binding):
        raise ValueError("derived latent origin differs from frozen A mu binding")
    current = cache.identity
    for name in ("schema", "chart_style", "dimension", "beats_per_token", "normalization"):
        if operational(current.get(name)) != operational(source.get(name)):
            raise ValueError("derived latent representation/normalization changed: " + name)
    for name in ("mu", "block_valid"):
        # File locations may be absolute references to the original accepted bytes.
        left = {k: v for k, v in current["files"][name].items() if k != "file"}
        right = {k: v for k, v in source["files"][name].items() if k != "file"}
        if left != right:
            raise ValueError("derived latent raw corpus changed: " + name)
    if current["data_binding"].get("vocabulary_sha256") != source["data_binding"].get("vocabulary_sha256"):
        raise ValueError("derived latent vocabulary changed")


def _extended_groups(batches, old, new, variants):
    """Insert new windows into existing song/audio groups without reordering old refs."""
    old_by_id = {s["segment_id"]: i for i, s in enumerate(old)}
    new_by_id = {s["segment_id"]: i for i, s in enumerate(new)}
    if len(new_by_id) != len(new) or not set(old_by_id).issubset(new_by_id):
        raise DatasetError("tail migration removed/duplicated original segment IDs")
    missing = defaultdict(list)
    for i, s in enumerate(new):
        if s["segment_id"] not in old_by_id:
            missing[str(s["record_id"])].append(i)
    for values in missing.values():
        values.sort(key=lambda i: int(new[i]["target_start_tick"]))
    refs = flatten(batches)
    # Original batch boundaries delimit group instances, even when the same
    # audio/start key happens to be adjacent on either side of that boundary.
    source_groups = []
    for batch in batches:
        for key, rows in groupby(batch, lambda r: (str(old[r.canonical_index]["song_id"]),
                variants[str(old[r.canonical_index]["record_id"])], int(old[r.canonical_index]["target_start_tick"]))):
            source_groups.append((key, tuple(rows)))
    result = []
    seen_songs = set()
    for song, song_groups in groupby(source_groups, lambda g: g[0][0]):
        if song in seen_songs:
            raise DatasetError("tail migration source is not song contiguous")
        seen_songs.add(song)
        groups = [(key[1:], rows) for key, rows in song_groups]
        members = tuple(r for _, rows in groups for r in rows)
        instances = dict.fromkeys((str(old[r.canonical_index]["record_id"]), r.view_id, r.occurrence) for r in members)
        additions = defaultdict(list)
        for record, view, occurrence in instances:
            for i in missing[record]:
                s = new[i]
                additions[(variants[record], int(s["target_start_tick"]))].append(VirtualSegmentRef(i, view, occurrence))
        last = {key: j for j, (key, _) in enumerate(groups)}
        for j, (key, rows) in enumerate(groups):
            remapped = [VirtualSegmentRef(new_by_id[old[r.canonical_index]["segment_id"]], r.view_id, r.occurrence) for r in rows]
            if last[key] == j:
                remapped.extend(additions.pop(key, ()))
            result.append(tuple(remapped))
        for key in sorted(additions):
            result.append(tuple(additions[key]))
    filtered = [(new[r.canonical_index]["segment_id"], r.view_id, r.occurrence) for r in flatten(result)
                if new[r.canonical_index]["segment_id"] in old_by_id]
    expected = [(old[r.canonical_index]["segment_id"], r.view_id, r.occurrence) for r in refs]
    if filtered != expected:
        raise DatasetError("tail migration changed original flattened references")
    return tuple(result)


def extend_references(batches, old, new, variants):
    return tuple(flatten(_extended_groups(batches, old, new, variants)))


def _pack_groups(groups, lengths, budget):
    result, pending, used = [], [], 0
    for group in groups:
        cost = sum(lengths[r.canonical_index] for r in group)
        if pending and used + cost > budget:
            result.append(tuple(pending)); pending, used = [], 0
        pending.extend(group); used += cost
    if pending:
        result.append(tuple(pending))
    return tuple(result)


def _plans(parent, output, old, new, variants):
    lengths = [int(s["target_rows"]) for s in new]
    all_audits = {"a-plan": [], "b-plan": []}
    for name in all_audits:
        (output / name / parent.mode).mkdir(parents=True)
    for e in range(parent.available_epochs):
        source_batches = parent.load_epoch(e)
        groups = _extended_groups(source_batches, old, new, variants)
        refs = tuple(flatten(groups))
        a = _pack_groups(groups, lengths, 4096)
        b = repack_frozen(a, new, budget=16384)
        for name, batches in (("a-plan", a), ("b-plan", b)):
            offsets = np.cumsum([0, *map(len, batches)], dtype=np.int64)
            relative = f"{parent.mode}/epoch-{e+1:03d}.npz"
            path = output / name / relative
            np.savez(path, canonical_indices=np.array([r.canonical_index for r in refs], np.int64),
                     view_ids=np.array([r.view_id for r in refs], np.int64),
                     occurrences=np.array([r.occurrence for r in refs], np.int64), batch_offsets=offsets)
            counts = [sum(lengths[r.canonical_index] for r in batch) for batch in batches]
            all_audits[name].append(dict(epoch=e+1, mode=parent.mode, file=relative, file_sha256=file_sha256(path),
                batch_count=len(batches), batch_target_rows=counts, target_rows=sum(counts),
                audio_group_instances=len(groups),
                segment_references=len(refs), reference_sha256=reference_hash(refs),
                source_reference_sha256=reference_hash(flatten(source_batches)),
                plan_sha256=digest(dict(references=reference_hash(refs), offsets=offsets.tolist()))))
    order = digest([s["segment_id"] for s in new])
    for name, audits in all_audits.items():
        root = output / name
        _rows(root / "audit.jsonl", audits)
        experiment = deepcopy(parent.manifest.get("experiment", {}))
        subset = experiment.setdefault("subsets", {}).setdefault("train", {})
        subset.update(segments=len(new), charts=len({s["record_id"] for s in new}),
                      records=list(dict.fromkeys(s["record_id"] for s in new)), segment_order_sha256=order)
        experiment["optimizer_steps_by_epoch"] = [a["batch_count"] for a in audits]
        if "identity" in experiment:
            experiment["identity"] = digest({k: v for k, v in experiment.items() if k != "identity"})
        base = {k:v for k,v in parent.manifest.items() if k not in ("recipe","fixed_batch_count","paired_first_ten_epoch_plan_sha256")}
        manifest = dict(base, **stamp(), epochs=parent.available_epochs,
            token_budget=4096 if name == "a-plan" else 16384, experiment=experiment,
            segment_count=len(new), segment_order_sha256=order,
            profile_segments_sha256=file_sha256(output / "profile/segments.jsonl"),
            modes={parent.mode: dict(epoch_plan_aggregate_sha256=digest([a["plan_sha256"] for a in audits]))},
            audit=dict(path="audit.jsonl", rows=len(audits), sha256=file_sha256(root / "audit.jsonl")))
        manifest["tail_migration"] = dict(input_plan=str(parent.root), source_recipe=parent.manifest.get("recipe"),
            source_available_epochs=parent.available_epochs, packing="original-audio-group-instances" if name == "a-plan" else "whole-song-soft-16384")
        manifest["manifest_identity_sha256"] = digest({k: v for k, v in manifest.items() if k != "manifest_identity_sha256"})
        write_json(root / "manifest.json", manifest)
        loaded = EpochPlanStore(root, verify_payloads=False)
        for e in range(parent.available_epochs):
            loaded.load_epoch(e)
    return {name: [a["batch_count"] for a in audits] for name, audits in all_audits.items()}


def _empty_raw(last, segment, vocabulary):
    """New windows are after the final declaration; retain only genuine prefix rows."""
    events, anchors = last["input_ids"]["event"], last["row_anchor_tick"]
    allowed = torch.zeros_like(events, dtype=torch.bool)
    for event in ("MOMENT", "NOTE", "SLIDE_BRANCH"):
        allowed |= events.eq(vocabulary.event_id(event))
    allowed &= anchors.ge(segment["context_start_tick"]) & anchors.lt(segment["target_start_tick"])
    selected = allowed.nonzero().flatten()
    n = len(selected) + 2
    value = {}
    for family in ("input_ids", "input_times", "input_time_active", "target_ids", "time_targets",
                   "time_category_targets", "time_lower", "time_upper"):
        value[family] = {}
        for field, old in last[family].items():
            fill = -100 if family in ("target_ids", "time_targets", "time_category_targets") else 0
            tensor = torch.full((n,), fill, dtype=old.dtype)
            if family.startswith("input"):
                tensor[1:-1] = old[selected]
            value[family][field] = tensor
    value["input_ids"]["event"][0] = vocabulary.event_id("SEGMENT_START")
    value["input_ids"]["event"][-1] = vocabulary.event_id("TARGET_START")
    value["target_ids"]["event"][-1] = vocabulary.event_id("SEGMENT_END")
    value["row_anchor_tick"] = torch.cat((torch.tensor([segment["context_start_tick"]], dtype=torch.float32),
                                          anchors[selected], torch.tensor([segment["target_start_tick"]], dtype=torch.float32)))
    moment = value["input_ids"]["event"].eq(vocabulary.event_id("MOMENT")).nonzero().flatten()
    if len(moment):
        first = int(moment[0])
        value["input_times"]["delta_tick"][first] = value["row_anchor_tick"][first] - segment["context_start_tick"]
    value["route_informative_mass"] = torch.zeros(n)
    return value


class WindowCache:
    def __init__(self, root, *, descriptor=None):
        self.root = Path(root)
        self.manifest = _read(self.root / "manifest.json") if descriptor is None else descriptor["manifest"]
        if self.manifest.get("schema") != WINDOW_SCHEMA or not self.manifest.get("ready"):
            raise DatasetError("prior Decoder window cache schema/readiness mismatch")
        self.records = self.manifest["records"]
        self.vocabulary = CompactVocabulary.from_dict(_read(self.manifest["vocabulary"]))
        self.route_mass = torch.tensor(self.manifest["route_informative_mass_by_class"], dtype=torch.float32)

    def descriptor(self):
        return dict(root=str(self.root), manifest=self.manifest)

    def get(self, canonical_index, view_id, occurrence=0, *, epoch=0, enabled=True):
        index = int(canonical_index) * 2 + int(view_id)
        if not 0 <= index < len(self.records) or int(view_id) not in (0, 1):
            raise DatasetError("prior Decoder window index outside cache")
        record = self.records[index]
        if record["canonical_index"] != int(canonical_index) or record["view_id"] != int(view_id):
            raise DatasetError("prior Decoder window canonical/view mapping mismatch")
        item = torch.load(self.root / record["file"], map_location="cpu", weights_only=True)
        item.update(_occurrence=int(occurrence), _cache_index=int(canonical_index))
        from maimai_baseline.data.augmentation import prepare_augmented_item
        item = prepare_augmented_item(item, self.vocabulary, seed=AUGMENTATION_SEED, epoch=int(epoch), enabled=bool(enabled))
        route = item["target_ids"]["route"]
        active = route.ne(-100)
        mass = torch.zeros_like(route, dtype=torch.float32)
        mass[active] = self.route_mass.index_select(0, route[active])
        item["route_informative_mass"] = mass
        return item


def _window_cache(output, segments, old, source, vocabulary, variants, bpm):
    root = output / "window-cache"
    root.mkdir()
    old_ids = {s["segment_id"] for s in old}
    latest = {}
    for s in old:
        if s["record_id"] not in latest or s["target_start_tick"] > latest[s["record_id"]]["target_start_tick"]:
            latest[s["record_id"]] = s
    records = []
    for i, s in enumerate(segments):
        variant = variants[str(s["record_id"])]
        info = bpm["variants"][variant]
        for view in (0, 1):
            name = VIEW_NAMES[view]
            raw = (source.record(s["segment_id"], name) if s["segment_id"] in old_ids else
                   _empty_raw(source.record(latest[s["record_id"]]["segment_id"], name), s, vocabulary))
            item = dict(raw, segment_id=s["segment_id"], record_id=s["record_id"], song_id=str(s["song_id"]),
                spatial_view_id=name, difficulty_id=int(s.get("condition_difficulty_id", s["difficulty_id"])),
                source_difficulty_id=int(s.get("source_difficulty_id", s["difficulty_id"])),
                cabinet_id=2 if s["cabinet"] == "DX" else 1,
                level=float(s.get("level_value") or 0.), start_tick=int(s["target_start_tick"]),
                end_tick=int(s["target_end_tick"]), song_end_tick=int(s["song_end_tick"]),
                condition_end_tick=int(s["condition_end_tick"]), _bpm96=True, _delta10ms_timing=s["timing"],
                audio_num_samples=int(info["num_samples"]), audio_group_key=(variant, int(s["target_start_tick"])),
                audio_key=(variant, int(s["target_start_tick"])), audio_reference=variant)
            sources = [-2, *range(int(s["context_row_start"]), int(s["target_row_start"])), -1,
                       *range(int(s["target_row_start"]), int(s["target_row_end"]))]
            if len(sources) != len(raw["row_anchor_tick"]):
                raise DatasetError("window input source-row ownership differs from complete chart rows")
            item["_physical_input_source_row_ids"] = torch.tensor(sources, dtype=torch.long)
            item["_physical_target_source_row_ids"] = torch.tensor([*sources[1:], -2], dtype=torch.long)
            item = prepare_raw_prefix_item(item, vocabulary)
            route = item["target_ids"]["route"]
            active = route.ne(-100)
            mass = torch.zeros_like(route, dtype=torch.float32)
            mass[active] = source.route_mass_by_class.index_select(0, route[active])
            item["route_informative_mass"] = mass
            file = f"{i:06d}-{view}.pt"
            torch.save(item, root / file)
            records.append(dict(canonical_index=i, view_id=view, segment_id=s["segment_id"], file=file,
                                bytes=(root / file).stat().st_size, sha256=file_sha256(root / file)))
    write_json(root / "manifest.json", dict(schema=WINDOW_SCHEMA, ready=True, records=records,
        profile=str(output / "profile"), vocabulary=str(output / "profile/vocabulary.json"),
        route_informative_mass_by_class=source.route_mass_by_class.tolist(),
        segment_order_sha256=digest([s["segment_id"] for s in segments])))


def _standard_vectors(output, segments, old, canonical, spatial, vocabulary, profile_fingerprint):
    """Extend the existing canonical/spatial contracts without adapting old charts."""
    old_ids = {s["segment_id"] for s in old}
    latest = {}
    for s in old:
        if s["record_id"] not in latest or s["target_start_tick"] > latest[s["record_id"]]["target_start_tick"]:
            latest[s["record_id"]] = s

    def raw(segment, view=None):
        source = canonical if view is None else spatial
        key = segment if segment["segment_id"] in old_ids else latest[segment["record_id"]]
        value = (source.record(key["segment_id"]) if view is None else source.record(key["segment_id"], view))
        return value if segment["segment_id"] in old_ids else _empty_raw(value, segment, vocabulary)

    offsets = np.cumsum([0, *(len(raw(s)["row_anchor_tick"]) for s in segments)], dtype=np.int64)
    index = []
    for i, s in enumerate(segments):
        old_row = canonical.segment_records.get(s["segment_id"], canonical.segment_records[latest[s["record_id"]]["segment_id"]])
        index.append(dict(old_row, segment_id=s["segment_id"], record_id=s["record_id"], cache_index=i,
            audio_key=audio_reference(s)[0],
            offset=int(offsets[i]), length=int(offsets[i+1]-offsets[i]), song_id=str(s["song_id"]),
            role=s["split"], source_difficulty_id=int(s.get("source_difficulty_id", s["difficulty_id"]))))

    def serialize(root, fields, view=None):
        root.mkdir(exist_ok=True, parents=True)
        arrays = {}
        for field, meta in fields.items():
            path = root / (("" if view is None else view + "-") + field + ".npy")
            array = np.lib.format.open_memmap(path, mode="w+", dtype=np.dtype(meta["dtype"]), shape=(int(offsets[-1]),))
            arrays[field] = (path, array)
        for i, segment in enumerate(segments):
            item = raw(segment, view)
            for field, (_, array) in arrays.items():
                value = item
                for part in field.split("."):
                    value = value[part]
                values = value.detach().cpu().numpy()
                if values.shape != (int(offsets[i+1]-offsets[i]),) or (values.dtype.kind == "f" and not np.isfinite(values).all()):
                    raise DatasetError("new standard window vector shape/finite mismatch: " + field)
                array[offsets[i]:offsets[i+1]] = values
        result = {}
        for field, (path, array) in arrays.items():
            array.flush()
            result[field] = dict(path=path.name, dtype=str(array.dtype), shape=list(array.shape), sha256=file_sha256(path))
            array._mmap.close()
        return result

    for name, source in (("chart-cache", canonical), ("spatial-views", spatial)):
        root = output / name
        root.mkdir()
        np.save(root / "offsets.npy", offsets, allow_pickle=False)
        _rows(root / "segments.jsonl", index)
        manifest = dict(source.manifest, segment_count=len(segments), row_count=int(offsets[-1]), profile_fingerprint=profile_fingerprint,
            profile_manifest_sha256=file_sha256(output / "profile/manifest.jsonl"),
            profile_segments_sha256=file_sha256(output / "profile/segments.jsonl"),
            segment_index=dict(path="segments.jsonl", sha256=file_sha256(root / "segments.jsonl")),
            offsets=dict(path="offsets.npy", sha256=file_sha256(root / "offsets.npy")))
        if name == "chart-cache":
            selection = dict(manifest.get("selection",{}),profile_fingerprint=profile_fingerprint)
            selection["selection_sha256"] = digest({k:v for k,v in selection.items() if k != "selection_sha256"})
            manifest["selection"] = selection
            manifest["segment_role_counts"] = {"train":len(segments),"validation":0}
            manifest["arrays"] = serialize(root, source.manifest["arrays"])
            references = {audio_reference(s)[0]: audio_reference(s)[1] for s in segments}
            _rows(root / "audio-references.jsonl", [dict(audio_key=key, **value) for key,value in sorted(references.items())])
            manifest["audio_references"] = dict(path="audio-references.jsonl", sha256=file_sha256(root / "audio-references.jsonl"))
            manifest["audio_reference_count"] = len(references)
        else:
            manifest["views"] = {view: dict(source.manifest["views"][view],
                arrays=serialize(root, source.manifest["views"][view]["arrays"], view)) for view in source.manifest["views"]}
            manifest["canonical_cache_manifest_sha256"] = file_sha256(output / "chart-cache/manifest.json")
            manifest["selection_sha256"] = _read(output / "chart-cache/manifest.json")["selection"]["selection_sha256"]
        # New consumers bind this manifest; provenance is still the original producer.
        manifest.pop("manifest_identity_sha256", None)
        manifest["manifest_identity_sha256"] = digest(manifest)
        write_json(root / "manifest.json", manifest)


def _mu_and_p0(output, source, p0_root, segments, profile_fingerprint):
    root = output / "mu"
    root.mkdir()
    origin = deepcopy(source.manifest)
    full = deepcopy(origin["full_charts"])
    by_chart = defaultdict(list)
    for i, s in enumerate(segments):
        by_chart[str(s["record_id"])].append((i, s))
    records, mappings, masks, conditions = [], [], [], []
    for chart in full:
        role, record, view = chart["key"]
        if role != "train" or str(record) not in by_chart:
            raise DatasetError("tail migration expects the bound train-only chart corpus")
        windows = by_chart[str(record)]
        end = max(int(s["target_end_tick"]) for _, s in windows)
        chart.update(condition_end_tick=end, condition_blocks=end // 384)
        mask = source._array("block_valid")[chart["offset"]:chart["offset"] + chart["blocks"]]
        for i, s in windows:
            local = int(s["target_start_tick"]) // 384 + np.arange(4)
            valid = local < chart["blocks"]
            valid[valid] &= mask[local[valid]]
            mapping = np.full(4, -1, np.int64)
            mapping[valid] = chart["offset"] + local[valid]
            records.append(dict(role=role, record_id=str(record), song_id=str(s["song_id"]), view_id=view,
                canonical_index=i, segment_id=s["segment_id"], start_tick=int(s["target_start_tick"]),
                end_tick=int(s["target_end_tick"]), song_end_tick=int(s["song_end_tick"]),
                condition_end_tick=end, difficulty_id=chart["difficulty_id"], level_value=chart["level"]))
            mappings.append(mapping); masks.append(valid); conditions.append(np.ones(4, bool))
    files = deepcopy(origin["files"])
    for name in ("mu", "block_valid"):
        files[name]["file"] = str((source.root / files[name]["file"]).resolve())
    for name, array in dict(window_blocks=np.stack(mappings), valid=np.stack(masks), condition_valid=np.stack(conditions)).items():
        path = root / (name + ".npy")
        np.save(path, array, allow_pickle=False)
        files[name] = dict(file=path.name, dtype=str(array.dtype), shape=list(array.shape), bytes=path.stat().st_size, sha256=file_sha256(path))
    write_json(root / "index.json", dict(records=records, full_charts=full))
    binding = dict(origin["data_binding"], files=files, index=dict(file="index.json", sha256=file_sha256(root / "index.json")),
                   profile_fingerprint=profile_fingerprint, segment_order_sha256=digest([s["segment_id"] for s in segments]))
    manifest = dict(origin, ready=True, records=records, full_charts=full, files=files, data_binding=binding,
                    derived_from=source.identity, condition_support="all-profile-window-slots",
                    decoder_data=dict(window_cache=str(output / "window-cache")))
    manifest["coverage"] = dict(origin.get("coverage", {}), windows=len(records), valid_window_blocks=int(np.stack(masks).sum()),
                                 condition_window_blocks=len(records)*4)
    manifest.pop("manifest_identity_sha256",None)
    manifest.pop("identity",None)
    write_json(root / "manifest.json", manifest)
    target = FrozenLatentCache(root)
    validate_derived_latent(target, source.identity)
    _reindex_p0(output, target, p0_root)
    return target


def _reindex_p0(output, target, p0_root, context=None):
    """Rebind old P0 bytes to new complete-chart/window indices without audio work."""
    context = _read(p0_root / "manifest.json") if context is None else context
    charts = deepcopy(context["charts"])
    lookup = {tuple(c["key"]): i for i, c in enumerate(charts)}
    if len(lookup) != len(charts) or set(lookup) != set(target.full_charts):
        raise DatasetError("P0 and rebuilt mu chart/view coverage differ")
    for c in charts:
        span = c.get("span_beats", c["song_end_tick"] / 96.)
        c.update(target.full_charts[tuple(c["key"])])
        c["rows"] = []
        c["span_beats"] = span
    mapping = []
    for i, r in enumerate(target.records):
        ci = lookup[(r["role"], r["record_id"], r["view_id"])]
        mapping.append(ci); charts[ci]["rows"].append(i)
    audio = deepcopy(context["audio"])
    for meta in audio.values():
        meta["file"] = str((p0_root / meta["file"]).resolve())
    (output / "p0").mkdir()
    migrated = dict(context, latent_data=target.identity, charts=charts, record_to_chart=mapping, audio=audio,
                    reindex_origin=context.get("manifest_identity_sha256") or digest(context))
    migrated.pop("manifest_identity_sha256",None)
    migrated.pop("identity",None)
    for name in ("loaded_charts","loaded_references","loaded_audio_variants"):
        migrated.pop(name,None)
    write_json(output / "p0/manifest.json", migrated)


def _rebuilt_mu_and_p0(args, paths, output, context):
    from maimai_baseline.data.frozen_latent import build_cache
    build_cache(SimpleNamespace(dataset=output / "profile", song_chart_cache=paths["song_chart_cache"],
        route_artifact=paths["route_artifact"], epoch_plan_cache=output / "a-plan",
        chart_encoder_checkpoint=paths["chart_encoder_checkpoint"], output=output / "mu",
        window_cache=output / "window-cache", device=getattr(args, "device", "cpu")))
    manifest = _read(output / "mu/manifest.json")
    manifest["rebuilt_from"] = deepcopy(context["latent_data"])
    manifest["decoder_data"] = dict(window_cache=str(output / "window-cache"))
    write_json(output / "mu/manifest.json", manifest)
    target = FrozenLatentCache(output / "mu", fields=())
    _reindex_p0(output, target, paths["song_context_cache"], context)
    return target


def prepare_data(args):
    """Produce new windows/plans and full encoder60 mu; reuse continuous P0 bytes."""
    paths = resolve_paths(args)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite prior Decoder data: " + str(output))
    parent = EpochPlanStore(paths["epoch_plan_cache"], verify_payloads=False)
    if parent.manifest.get("token_budget",4096) != 4096:
        raise DatasetError("tail migration requires the original A4096 source plan")
    old = selected_segments(paths["dataset"], parent)
    from maimai_baseline.data.frozen_context import FrozenSongContext
    rebuilt = "chart_encoder_checkpoint" in paths
    source = None
    if rebuilt:
        context = _read(paths["song_context_cache"] / "manifest.json")
        from maimai_baseline.data.frozen_context import SCHEMA as P0_SCHEMA
        if context.get("schema") != P0_SCHEMA or not context.get("ready") or "latent_data" not in context:
            raise DatasetError("frozen P0 schema/readiness/origin mismatch")
    else:
        source = FrozenLatentCache(paths["latent_cache"], fields=())
        context = FrozenSongContext(paths["song_context_cache"], source).manifest
    variants = {str(c["key"][1]): c["audio_key"] for c in context["charts"]}
    output.mkdir(parents=True)
    write_json(output / "manifest.json", dict(schema=SCHEMA, ready=False))
    profile = output / "profile"
    profile.mkdir()
    templates = {}
    for s in old:
        templates.setdefault(str(s["record_id"]), s)
    new = []
    for record, template in templates.items():
        document = _read(paths["dataset"] / template["encoded_path"])
        windows = segment_chart(document, template)
        end = windows[-1]["target_end_tick"]
        for s in windows:
            s["condition_end_tick"] = end
        new.extend(windows)
        destination = profile / template["encoded_path"]
        if not destination.resolve().is_relative_to(profile.resolve()):
            raise DatasetError("encoded destination escaped the new profile")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(paths["dataset"] / template["encoded_path"], destination)
    old_ids = {s["segment_id"] for s in old}
    for s in new:
        if s["segment_id"] not in old_ids and s["target_rows"] != 1:
            raise DatasetError("new tail window unexpectedly contains a declaration")
    _rows(profile / "segments.jsonl", new)
    manifest_rows = [r for r in load_jsonl(paths["dataset"] / "manifest.jsonl") if str(r["record_id"]) in templates]
    _rows(profile / "manifest.jsonl", manifest_rows)
    shutil.copyfile(paths["dataset"] / "vocabulary.json", profile / "vocabulary.json")
    summary = dict(_read(paths["dataset"] / "summary.json"), segment_count=len(new),
                   profile_fingerprint=digest(dict(segments=new, records=manifest_rows)))
    for name, key in (("segment_split_counts","split"),("segment_source_difficulty_counts","source_difficulty_id")):
        if name in summary:
            summary[name] = {str(value):sum(str(s.get(key,s["difficulty_id"])) == str(value) for s in new)
                             for value in dict.fromkeys(s.get(key,s["difficulty_id"]) for s in new)}
    if "target_row_split_counts" in summary:
        summary["target_row_split_counts"] = {"train":sum(int(s["target_rows"]) for s in new)}
    write_json(profile / "summary.json", summary)
    profile_meta = _read(paths["dataset"] / "profile.json")
    for name in ("base_dataset", "base_feature_stats"):
        if name in profile_meta:
            profile_meta[name] = str((paths["dataset"] / profile_meta[name]).resolve())
    profile_meta.update(profile="prior-decoder-tail", summary_sha256=file_sha256(profile / "summary.json"))
    write_json(profile / "profile.json", profile_meta)
    spatial_manifest = _read(paths["spatial_view_cache"] / "manifest.json")
    spatial = MultiViewRouteStore.load(paths["spatial_view_cache"], verify_payloads=False,
        profile_fingerprint=spatial_manifest.get("profile_fingerprint", ""),
        vocabulary_sha256=spatial_manifest.get("vocabulary_sha256", ""),
        selection_sha256=spatial_manifest.get("selection_sha256", ""))
    vocabulary = CompactVocabulary.from_dict(_read(profile / "vocabulary.json"))
    canonical_manifest = _read(paths["chart_cache"] / "manifest.json")
    canonical = ChartVectorStore.load(paths["chart_cache"], verify_payloads=False,
        profile_fingerprint=canonical_manifest.get("profile_fingerprint", ""),
        vocabulary_sha256=canonical_manifest.get("vocabulary_sha256", ""))
    _standard_vectors(output, new, old, canonical, spatial, vocabulary, summary["profile_fingerprint"])
    ChartVectorStore.load(output / "chart-cache", verify_payloads=False,
        profile_fingerprint=summary["profile_fingerprint"], vocabulary_sha256=file_sha256(profile / "vocabulary.json"))
    MultiViewRouteStore.load(output / "spatial-views", verify_payloads=False,
        profile_fingerprint=summary["profile_fingerprint"], vocabulary_sha256=file_sha256(profile / "vocabulary.json"),
        selection_sha256=_read(output / "chart-cache/manifest.json")["selection"]["selection_sha256"])
    _window_cache(output, new, old, spatial, vocabulary, variants, _read(paths["bpm_audio_manifest"]))
    batch_counts = _plans(parent, output, old, new, variants)
    target = (_rebuilt_mu_and_p0(args, paths, output, context) if rebuilt else
              _mu_and_p0(output, source, paths["song_context_cache"], new, summary["profile_fingerprint"]))
    # New mu acceptance is performed by its producer. Reindexed P0 and window
    # inputs are accepted here through normal requested loader paths.
    windows = WindowCache(output / "window-cache")
    for i, s in enumerate(new):
        for view in (0, 1):
            item = windows.get(i, view)
            if int(item["target_ids"]["event"].ne(-100).sum()) != s["target_rows"]:
                raise DatasetError("new window loader supervision coverage mismatch")
            index = target.train_lookup[(i, view)]
            if not rebuilt and not target.window([index])["condition_valid"].all():
                raise DatasetError("new window lost condition support")
    new_context = FrozenSongContext(output / "p0", target)
    audio_rows = defaultdict(list)
    for chart in new_context.charts:
        audio_rows[chart["audio_key"]].extend(chart["rows"])
    loaded_charts = loaded_references = loaded_audio = 0
    for indices in audio_rows.values():
        batch = new_context.batch(indices)
        _validate_condition_batch(batch)
        loaded_charts += batch["chart_count"]; loaded_references += batch["reference_count"]; loaded_audio += batch["audio_count"]
    if (loaded_charts,loaded_references,loaded_audio) != (len(new_context.charts),len(target.records),len(new_context.manifest["audio"])):
        raise DatasetError("new μ/P0 reindex loader coverage mismatch")
    write_json(output / "p0/manifest.json", dict(new_context.manifest,loaded_charts=loaded_charts,
        loaded_references=loaded_references,loaded_audio_variants=loaded_audio))
    mapped = {k: str(v) for k, v in paths.items()}
    mapped.update(dataset=str(profile), window_cache=str(output / "window-cache"),
                  chart_cache=str(output / "chart-cache"), spatial_view_cache=str(output / "spatial-views"),
                  epoch_plan_cache=str(output / "a-plan"), whole_song_plan=str(output / "b-plan"),
                  latent_cache=str(output / "mu"), song_context_cache=str(output / "p0"))
    coverage = dict(charts=len(templates),
                  old_windows=len(old), windows=len(new), new_windows=len(new)-len(old),
                  view_windows=len(new)*2, epochs=parent.available_epochs,
                  condition_window_blocks=len(target.records)*4,
                  missing_target_window_blocks=len(target.records)*4-int(target.manifest["coverage"]["valid_window_blocks"]))
    coverage["target_blocks" if rebuilt else "old_target_blocks"] = int(target.manifest["coverage"]["valid_blocks"])
    result = dict(schema=SCHEMA, ready=True, inputs=mapped, coverage=coverage,
                  batch_counts=batch_counts, updates=dict(
                      a90=sum(batch_counts["a-plan"][:90]) if parent.available_epochs >= 90 else None,
                      b20=sum(batch_counts["b-plan"][90:110]) if parent.available_epochs >= 110 else None,
                      b_horizon50=sum(batch_counts["b-plan"][90:140]) if parent.available_epochs >= 140 else None),
                  reused_mu=not rebuilt, rebuilt_mu=rebuilt, reused_p0=True)
    write_json(output / "dataset.json", result)
    write_json(output / "manifest.json", result)
    return result


def _validate_condition_batch(batch):
    row = batch["prior_inputs"]
    valid, lengths = row["condition_valid"], row["condition_lengths"]
    expected = torch.arange(valid.shape[1])[None] < lengths[:,None]
    if not torch.equal(valid, expected):
        raise DatasetError("new full-song condition coverage has holes or valid padding")
