"""Immutable raw-mu chart cache; audio and posterior variance never enter it."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np
import torch

from maimai_baseline.data.chart_encoder_pretrain import ChartOnlySongCache, _prototype_inputs
from maimai_baseline.data.curriculum import EpochPlanStore
from maimai_baseline.data.style_cache import write_json
from maimai_dataset.integrity import file_sha256
from maimai_dataset.manifest import DatasetError, canonical_json_sha256, load_jsonl
from maimai_dataset.vocabulary import CompactVocabulary


SCHEMA = "maimai.frozen_chart_mu_cache.v1"
ARRAY_DTYPES = dict(mu="float32", block_valid="bool", window_blocks="int64", valid="bool")
OPTIONAL_ARRAY_DTYPES = dict(condition_valid="bool")


def _resolve_cache_paths(args):
    """Map only actual cache-production inputs; no unused A path requirements."""
    names = dict(dataset="profile", song_chart_cache="song-chart", route_artifact="slide-geometry/route-artifact")
    root = getattr(args, "dataset_root", None)
    root = Path(root).expanduser().resolve() if root is not None else None
    manifest = getattr(args, "dataset_manifest", None)
    if manifest is not None:
        manifest = Path(manifest).expanduser().resolve()
        root = root or manifest.parent
    elif root is not None:
        manifest = next((root / n for n in ("dataset-v10.json", "dataset.json") if (root / n).is_file()), None)
    mapped = dict(names) if root is not None else {}
    if manifest is not None:
        document = json.loads(manifest.read_text("utf-8-sig"))
        mapped.update({k.replace("-", "_"): v for k, v in document.get("inputs", {}).items()})
    for name in (*names, "epoch_plan_cache"):
        explicit = getattr(args, name, None)
        value = explicit if explicit is not None else mapped.get(name)
        if value is None:
            raise ValueError("missing frozen mu production input: --" + name.replace("_", "-"))
        path = Path(value).expanduser()
        if explicit is None and root is not None and not path.is_absolute():
            path = root / path
        setattr(args, name, path.resolve())
    window_cache = getattr(args, "window_cache", None) or mapped.get("window_cache")
    if window_cache is not None:
        path = Path(window_cache).expanduser()
        if getattr(args, "window_cache", None) is None and root is not None and not path.is_absolute():
            path = root / path
        args.window_cache = path.resolve()
    return args


def selected_segments(profile, plans, role="train"):
    """Only plan selection and existing metadata; preserve canonical ordering."""
    subsets = plans.manifest.get("experiment", {}).get("subsets", plans.manifest.get("subsets", {}))
    subset = subsets.get(role, {})
    records = subset.get("records")
    allowed = None if records is None else set(map(str, records))
    rows = [row for row in load_jsonl(Path(profile) / "segments.jsonl")
            if (row.get("split") == role if allowed is None else
                str(row.get("record_id") or str(row["segment_id"]).rsplit(":", 1)[0]) in allowed)]
    order = canonical_json_sha256([row["segment_id"] for row in rows])
    expected = subset.get("segment_order_sha256")
    if role == "train":
        expected = plans.manifest.get("segment_order_sha256", expected)
    if expected is not None and order != expected:
        raise DatasetError("frozen latent plan segment order mismatch")
    if subset.get("segments") is not None and len(rows) != int(subset["segments"]):
        raise DatasetError("frozen latent plan segment count mismatch")
    return rows


def _statistics(charts, representation="deterministic-mu"):
    first = torch.zeros(128, dtype=torch.float64)
    second = torch.zeros_like(first)
    units = 0
    for key, mu, valid in charts:
        if key[0] != "train":
            continue
        weight = 2 if key[2] == 0 else 1
        values = mu[valid].double()
        first += values.sum(0) * weight
        second += values.square().sum(0) * weight
        units += len(values) * weight
    if not units:
        raise DatasetError("frozen latent normalization has no valid training blocks")
    mean = first / units
    std = (second / units - mean.square()).clamp_min(1e-8).sqrt()
    return dict(scope="train-only", view_weights={"identity": 2, "mirror-lr": 1},
        includes_posterior_variance=False, representation=representation,
        algorithm="weighted-E-mu-squared-minus-weighted-E-mu-square;variance-clamp-1e-8",
        local=dict(mean=mean.tolist(), std=std.tolist(), weighted_units=units))


class FrozenLatentCache:
    def __init__(self, root, *, verify_payloads=False, fields=None, descriptor=None):
        self.root = Path(root).resolve()
        self.manifest = (json.loads((self.root / "manifest.json").read_text("utf8"))
                         if descriptor is None else descriptor["manifest"])
        value = self.manifest
        from maimai_baseline.style_contract import FROZEN_STYLE_ID, NATIVE_STYLE_ID
        if (value.get("schema") != SCHEMA or not value.get("ready")
                or value.get("chart_style") not in (FROZEN_STYLE_ID, NATIVE_STYLE_ID)
                or value.get("dimension") != 128 or value.get("beats_per_token") != 4):
            raise DatasetError("frozen latent cache schema/representation/readiness mismatch")
        self.records = value["records"]
        self.full_charts = {tuple(chart["key"]): chart for chart in value["full_charts"]}
        if len(self.full_charts) != len(value["full_charts"]):
            raise DatasetError("duplicate frozen full-chart key")
        self.max_blocks = max((chart["blocks"] for chart in self.full_charts.values()), default=0)
        self.train_lookup, self.segment_lookup = {}, {}
        for i, record in enumerate(self.records):
            key = (record["role"], record["segment_id"], int(record["view_id"]))
            if key in self.segment_lookup:
                raise DatasetError("duplicate frozen latent segment/view")
            self.segment_lookup[key] = i
            if record["role"] == "train":
                key = (int(record["canonical_index"]), int(record["view_id"]))
                if key in self.train_lookup:
                    raise DatasetError("duplicate frozen latent canonical/view mapping")
                self.train_lookup[key] = i
        norm = value["normalization"]
        local = norm.get("local", {})
        mean, std = np.asarray(local.get("mean", ())), np.asarray(local.get("std", ()))
        if (norm.get("scope") != "train-only" or norm.get("includes_posterior_variance") is not False
                or norm.get("representation") != ("layernorm-mu-eps1e-5" if value['chart_style'] == NATIVE_STYLE_ID else "deterministic-mu")
                or norm.get("view_weights") != {"identity": 2, "mirror-lr": 1}
                or mean.shape != (128,) or std.shape != (128,) or not np.isfinite(mean).all()
                or not np.isfinite(std).all() or (std <= 0).any()):
            raise DatasetError("invalid deterministic-mu normalization")
        self.arrays = {}
        for name in (ARRAY_DTYPES if fields is None else fields):
            self._array(name, verify=verify_payloads)
        if verify_payloads:
            for name in (*ARRAY_DTYPES, *(n for n in OPTIONAL_ARRAY_DTYPES if n in value["files"])):
                if name not in self.arrays:
                    self._array(name, verify=True)

    def __getstate__(self):
        return dict(self.__dict__, arrays={})

    def _array(self, name, *, verify=False):
        if name not in self.arrays:
            meta = self.manifest["files"][name]
            path = self.root / meta["file"]
            if verify and file_sha256(path) != meta["sha256"]:
                raise DatasetError("frozen latent payload hash mismatch: " + name)
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            dtype = (ARRAY_DTYPES | OPTIONAL_ARRAY_DTYPES)[name]
            if list(array.shape) != meta["shape"] or str(array.dtype) != dtype:
                raise DatasetError("frozen latent array header mismatch: " + name)
            self.arrays[name] = array
        return self.arrays[name]

    @property
    def identity(self):
        return dict(schema=SCHEMA, chart_style=self.manifest["chart_style"], dimension=128,
            beats_per_token=4, data_binding=self.manifest["data_binding"],
            normalization=self.manifest["normalization"], files=self.manifest["files"])

    def validate_binding(self, *, profile_fingerprint, vocabulary_sha256, segment_order_sha256):
        binding = self.manifest["data_binding"]
        for name, expected in dict(profile_fingerprint=profile_fingerprint,
                vocabulary_sha256=vocabulary_sha256, segment_order_sha256=segment_order_sha256).items():
            if binding.get(name) != expected:
                raise DatasetError("frozen latent input binding mismatch: " + name)

    def descriptor(self):
        return dict(root=str(self.root), manifest=self.manifest, train_lookup=self.train_lookup)

    def plan_indices(self, refs):
        return [self.train_lookup[(int(r.canonical_index), int(r.view_id))] for r in refs]

    def full_chart(self, key):
        chart = self.full_charts[tuple(key)]
        start, length = int(chart["offset"]), int(chart["blocks"])
        mu = self._array("mu")
        valid = self._array("block_valid")
        if start < 0 or length < 1 or start + length > len(mu) or start + length > len(valid):
            raise DatasetError("frozen full-chart offset outside cache")
        values = torch.from_numpy(np.array(mu[start:start + length], copy=True))
        mask = torch.from_numpy(np.array(valid[start:start + length], copy=True))
        if values.shape != (length, 128) or mask.shape != (length,) or not torch.isfinite(values).all():
            raise DatasetError("invalid frozen full-chart mu/mask")
        if bool(values[~mask].ne(0).any()):
            raise DatasetError("frozen full-chart padding is not zero")
        return values, mask

    def window(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or (indices < 0).any() or (indices >= len(self.records)).any():
            raise DatasetError("frozen window record index out of range")
        blocks = np.array(self._array("window_blocks")[indices], copy=True)
        valid = np.array(self._array("valid")[indices], copy=True)
        condition_valid = (np.array(self._array("condition_valid")[indices], copy=True)
                           if "condition_valid" in self.manifest["files"] else valid.copy())
        if blocks.shape != (len(indices), 4) or valid.shape != blocks.shape or condition_valid.shape != blocks.shape:
            raise DatasetError("frozen window mapping/mask shape mismatch")
        values = np.zeros((len(indices), 4, 128), dtype=np.float32)
        for row, index in enumerate(indices):
            record = self.records[int(index)]
            chart = self.full_charts[(record["role"], record["record_id"], record["view_id"])]
            local = int(record["start_tick"]) // 384 + np.arange(4)
            expected = int(chart["offset"]) + local
            if (blocks[row, valid[row]] != expected[valid[row]]).any() or (blocks[row, ~valid[row]] != -1).any():
                raise DatasetError("frozen window mapping differs from its chart/view")
            geometry = (int(record["start_tick"]) + np.arange(4) * 384 < int(record["end_tick"]))
            # The reused full-chart corpus owns GT support. A new profile's
            # object end can be shorter without invalidating cached real mu.
            geometry &= (local >= 0) & (local < int(chart["blocks"]))
            if (valid[row] & ~geometry).any():
                raise DatasetError("frozen window mask escaped its geometry")
            support = int(record["start_tick"]) + np.arange(4) * 384 < int(record["end_tick"])
            if (condition_valid[row] & ~support).any() or (valid[row] & ~condition_valid[row]).any():
                raise DatasetError("frozen condition mask escaped window geometry or lost its target")
        if (blocks[valid] < 0).any() or (blocks[valid] >= len(self._array("mu"))).any():
            raise DatasetError("frozen window block mapping out of bounds")
        if valid.any():
            if not self._array("block_valid")[blocks[valid]].all():
                raise DatasetError("frozen window mask escaped full-chart validity")
            values[valid] = self._array("mu")[blocks[valid]]
        if not np.isfinite(values).all():
            raise DatasetError("non-finite frozen window mu")
        return dict(latent_tokens=torch.from_numpy(values), latent_valid=torch.from_numpy(valid),
                    target_valid=torch.from_numpy(valid.copy()), condition_valid=torch.from_numpy(condition_valid))

    def windows(self, items, role="train"):
        indices = []
        for item in items:
            view = item.get("spatial_view_id", "identity")
            if view not in ("identity", "mirror-lr"):
                raise DatasetError("unsupported frozen latent spatial view")
            index = self.segment_lookup[(role, str(item["segment_id"]), 0 if view == "identity" else 1)]
            record = self.records[index]
            for source, target in (("start_tick", "start_tick"), ("end_tick", "end_tick")):
                if source in item and int(item[source]) != int(record[target]):
                    raise DatasetError("frozen latent window coordinate mismatch")
            indices.append(index)
        return self.window(indices)


def build_cache(args):
    """Produce only new mu assets, from accepted full-chart inputs and encoder60."""
    from maimai_baseline.model.chart_encoder_pretrain import encode_pretrained
    from maimai_baseline.training.encoder_import import load_encoder60
    from maimai_baseline.style_contract import FROZEN_STYLE_ID, NATIVE_STYLE_ID
    args = _resolve_cache_paths(args)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise ValueError("refusing to overwrite frozen latent cache: " + str(output))
    plans = EpochPlanStore(args.epoch_plan_cache, verify_payloads=False)
    vocabulary_path = Path(args.dataset) / "vocabulary.json"
    vocabulary = CompactVocabulary.from_dict(json.loads(vocabulary_path.read_text("utf-8-sig")))
    prototypes, _ = _prototype_inputs(args.route_artifact)
    checkpoint = getattr(args, "chart_encoder_checkpoint", None) or getattr(args, "encoder_checkpoint", None)
    if checkpoint is None:
        raise ValueError("frozen mu production requires --chart-encoder-checkpoint")
    device = torch.device(getattr(args, "device", "cpu"))
    encoder, encoder_import = load_encoder60(checkpoint, vocabulary=vocabulary, prototypes=prototypes, device=device)
    representation = encoder_import['representation']
    chart_style = NATIVE_STYLE_ID if representation == 'layernorm-mu-eps1e-5' else FROZEN_STYLE_ID
    source = ChartOnlySongCache(args.song_chart_cache)
    subsets = plans.manifest.get("experiment", {}).get("subsets", plans.manifest.get("subsets", {}))
    roles = ["train"] + (["validation"] if subsets.get("validation", {}).get("records") else [])
    grouped, segments_by_role = {}, {}
    for role in roles:
        segments = selected_segments(args.dataset, plans, role)
        segments_by_role[role] = segments
        for index, segment in enumerate(segments):
            record = str(segment.get("record_id") or str(segment["segment_id"]).rsplit(":", 1)[0])
            grouped.setdefault((role, record), []).append((index, segment))
    if not grouped:
        raise DatasetError("frozen latent source plan contains no charts")
    output.mkdir(parents=True)
    write_json(output / "manifest.json", dict(schema=SCHEMA, chart_style=chart_style, ready=False))
    full_charts, values, masks, statistics, records, mappings, window_masks, condition_masks = [], [], [], [], [], [], [], []
    full_coverage = any("condition_end_tick" in segment for windows in grouped.values() for _, segment in windows)
    offset = 0
    with torch.inference_mode():
        for (role, record), windows in grouped.items():
            for view_id in (0, 1):
                key = (role, record, view_id)
                packed = source.get(key)
                packed = {name: ({field: value.to(device) for field, value in group.items()}
                                 if isinstance(group, dict) else group.to(device) if isinstance(group, torch.Tensor) else group)
                          for name, group in packed.items()}
                condition_end = max(int(segment.get("condition_end_tick", 0)) for _, segment in windows)
                if condition_end:
                    if condition_end % 384:
                        raise DatasetError("full-profile condition end is not four-beat aligned")
                    length = condition_end // 384
                    # Ownership stays complete: NOTE/BRANCH members retain their
                    # original MOMENT, and no row tensor is cropped or filtered.
                    slots = packed["moment_slot"]
                    if length < 1 or bool(((slots < 0) | (slots >= length)).any()):
                        raise DatasetError("real chart rows exceed condition coverage; extend the profile window range")
                    # Both shortening old empty padding and extending coverage
                    # re-encode all real rows plus an encoder token for each empty
                    # block. No old mu or invented zero target is appended.
                    packed["valid"] = torch.ones((1, length), dtype=torch.bool, device=device)
                mu, valid, _ = encode_pretrained(encoder, [packed])
                mu, valid = mu[0].float().cpu(), valid[0].cpu()
                if mu.shape != (len(valid), 128) or valid.dtype != torch.bool or not torch.isfinite(mu).all():
                    raise DatasetError("encoder60 produced invalid deterministic mu")
                if bool(mu[~valid].ne(0).any()):
                    raise DatasetError("encoder60 produced nonzero padded mu")
                info = source.lookup[key]
                first = windows[0][1]
                if condition_end:
                    boundaries = {int(segment["song_end_tick"]) for _, segment in windows}
                    if len(boundaries) != 1:
                        raise DatasetError("full-profile windows disagree on song end")
                    song_end = boundaries.pop()
                else:
                    song_end = int(info.get("song_end_tick", first["song_end_tick"]))
                expected_blocks = condition_end // 384 if condition_end else song_end // 384 + 1
                if len(valid) != expected_blocks:
                    raise DatasetError("full-chart mu blocks differ from accepted song geometry")
                meta = dict(key=list(key), blocks=len(valid), offset=offset, shape=list(mu.shape),
                    song_end_tick=song_end, difficulty_id=int(first.get("condition_difficulty_id", first["difficulty_id"])),
                    level=float(first.get("level_value") or 0.))
                if condition_end:
                    meta.update(condition_end_tick=condition_end, condition_blocks=expected_blocks)
                if "variant" in info:
                    meta["variant"] = info["variant"]
                full_charts.append(meta)
                values.append(mu.numpy())
                masks.append(valid.numpy())
                statistics.append((key, mu, valid))
                for canonical_index, segment in windows:
                    start, end = int(segment["target_start_tick"]), int(segment["target_end_tick"])
                    if start < 0 or start % 384:
                        raise DatasetError("frozen latent window is not four-beat aligned")
                    local = start // 384 + np.arange(4)
                    in_bounds = local < len(valid)
                    condition = (start + np.arange(4) * 384 < end) & in_bounds
                    mask = condition.copy()
                    if not condition_end:
                        mask &= start + np.arange(4) * 384 <= song_end
                    mask[in_bounds] &= valid.numpy()[local[in_bounds]]
                    mapping = np.full(4, -1, dtype=np.int64)
                    mapping[mask] = offset + local[mask]
                    records.append(dict(role=role, record_id=record, song_id=str(segment["song_id"]),
                        difficulty_id=meta["difficulty_id"], source_difficulty_id=int(segment.get("source_difficulty_id", segment["difficulty_id"])),
                        level_value=meta["level"], song_end_tick=song_end, start_tick=start, end_tick=end,
                        canonical_index=canonical_index, view_id=view_id, segment_id=str(segment["segment_id"])))
                    if condition_end:
                        records[-1]["condition_end_tick"] = condition_end
                    mappings.append(mapping)
                    window_masks.append(mask)
                    condition_masks.append(condition if condition_end else mask.copy())
                offset += len(valid)
    arrays = dict(mu=np.concatenate(values), block_valid=np.concatenate(masks),
                  window_blocks=np.stack(mappings), valid=np.stack(window_masks))
    if full_coverage:
        arrays["condition_valid"] = np.stack(condition_masks)
    files = {}
    for name, array in arrays.items():
        path = output / (name + ".npy")
        np.save(path, array, allow_pickle=False)
        files[name] = dict(file=path.name, shape=list(array.shape), dtype=str(array.dtype),
                           bytes=path.stat().st_size, sha256=file_sha256(path))
    normalization = _statistics(statistics, representation)
    record_material = dict(records=records, full_charts=full_charts)
    write_json(output / "index.json", record_material)
    index_meta = dict(file="index.json", sha256=file_sha256(output / "index.json"))
    summary = json.loads((Path(args.dataset) / "summary.json").read_text("utf-8-sig"))
    data_binding = dict(files=files, index=index_meta,
        profile_fingerprint=str(summary["profile_fingerprint"]),
        vocabulary_sha256=hashlib.sha256(vocabulary_path.read_bytes()).hexdigest(),
        segment_order_sha256=canonical_json_sha256([s["segment_id"] for s in segments_by_role["train"]]))
    coverage = dict(charts=len(full_charts), windows=len(records), blocks=offset,
        valid_blocks=int(arrays["block_valid"].sum()), valid_window_blocks=int(arrays["valid"].sum()),
        successful_charts=len(full_charts), failed_charts=0, finite=True)
    manifest = dict(schema=SCHEMA, chart_style=chart_style, representation=representation,
        ready=True, dimension=128, beats_per_token=4, records=records, full_charts=full_charts,
        files=files, data_binding=data_binding, normalization=normalization, coverage=coverage,
        producer=dict(encoder_import=encoder_import))
    if full_coverage:
        manifest["condition_support"] = "encoded-full-profile-window-slots"
        window_cache = getattr(args, "window_cache", None) or Path(args.dataset).resolve().parent / "window-cache"
        manifest["decoder_data"] = dict(window_cache=str(Path(window_cache).resolve()))
        coverage["condition_window_blocks"] = int(arrays["condition_valid"].sum())
    write_json(output / "manifest.json", manifest)
    # One acceptance for this new output: actual requested loader paths. Inputs
    # above are read normally; no old payload verification or feature rebuild.
    try:
        cache = FrozenLatentCache(output, fields=())
        for chart in full_charts:
            cache.full_chart(chart["key"])
        cache.window(range(len(records)))
    except Exception as error:
        manifest["ready"] = False
        manifest["coverage"]["loader_error"] = str(error)
        write_json(output / "manifest.json", manifest)
        raise
    return dict(output=str(output), coverage=coverage, normalization=normalization,
                data_binding=data_binding)
