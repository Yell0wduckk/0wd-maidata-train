"""Frozen, original-window teacher audio. No audio network lives in this reader."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import fields, replace
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from maimai_baseline.model.common import AudioContext
from maimai_dataset.manifest import canonical_json_sha256 as digest

SCHEMA = "maimai.native_teacher_audio.v1"
CONTRACT = dict(window_beats=22, target_beats=16, before_beats=4, after_beats=2,
    dimension=256, p0_tokens=176, c0_tokens=528, c1_tokens=264,
    precision="fp32-eval", coordinates="absolute-beats", song_p0="native-window-center-concatenation",
    saved_scales=["P0", "C0", "C1"])
GEOMETRY = ("a1_to_a0_left", "a1_to_a0_right", "a1_to_a0_weight", "a1_to_a0_padding",
            "candidate_left", "candidate_right", "candidate_weight", "candidate_padding", "candidate_start_tick")
FIELDS = tuple(name+suffix for name in ("p0","c0","c1") for suffix in ("","_valid","_beats"))


def producer_identity(identity):
    """Inference may use new songs, with the same bound teacher and output contract."""
    value = identity.identity if hasattr(identity,"identity") else identity
    producer = value["producer"]
    kept = {key:producer.get(key) for key in ("kind","run","source_commit","epoch","step","runtime_version",
        "checkpoint_recorded_sha256","original_feature_stats","scope","fusion_geometry_extension")}
    kept["source_sha256"] = producer.get("source_identity",{}).get("source_sha256")
    audio = producer.get("original_audio_input")
    kept["preprocessing"] = None if audio is None else {key:audio.get(key) for key in ("policy","contract","feature_stats_sha256")}
    return dict(schema=SCHEMA,contract=value.get("contract",CONTRACT),producer=kept)


def draft_producer_identity(identity):
    """New draft contracts bind native output/preprocessing, not source hashes."""
    raw = identity.identity if hasattr(identity, "identity") else identity
    value = (dict(raw) if "preprocessing" in raw.get("producer", {}) else producer_identity(raw))
    kept = ("kind", "run", "epoch", "step", "runtime_version", "original_feature_stats",
            "scope", "fusion_geometry_extension", "preprocessing", "output")
    producer = {key: item for key, item in value["producer"].items() if key in kept}
    if "output" in raw.get("producer", {}):
        producer["output"] = raw["producer"]["output"]
    return dict(schema=SCHEMA, contract=value.get("contract", CONTRACT), producer=producer)


def context_dict(context):
    return {f.name: getattr(context, f.name) for f in fields(context)}


def project_candidates(model, context, geometry):
    """Apply only the consuming A's candidate projection, preserving autograd."""
    if isinstance(context, dict):
        context = AudioContext(**context)
    if len(context.fine_memories) != 2:
        raise ValueError("native candidates require the original C0 and C1")
    from maimai_baseline.model.layers import _indexed_interpolate
    first, second = context.fine_memories
    second = _indexed_interpolate(second, geometry["a1_to_a0_left"], geometry["a1_to_a0_right"],
        geometry["a1_to_a0_weight"], geometry["a1_to_a0_padding"])
    keys = model.delta_candidate_key(model.delta_candidate_fusion(torch.cat((first, second), -1)))
    mask = context.fine_padding_masks[0]
    return replace(context, candidate_keys=keys.masked_fill(mask[..., None], 0.),
        candidate_padding_mask=mask, candidate_beats=context.fine_beats[0],
        candidate_lookup_left=geometry["candidate_left"], candidate_lookup_right=geometry["candidate_right"],
        candidate_lookup_weight=geometry["candidate_weight"], candidate_lookup_padding=geometry["candidate_padding"],
        candidate_lookup_start_tick=geometry["candidate_start_tick"])


def _validate_bundle(bundle, starts):
    count = len(starts)
    if set(bundle) != {*FIELDS,"interpolation_geometry"}:
        raise ValueError("native cache contains unsupported scales/fields")
    if not count or starts != list(range(0, starts[-1] + 1, 1536)):
        raise ValueError("native audio windows must cover a contiguous 16-beat lattice from zero")
    for name, size in (("p0",176),("c0",528),("c1",264)):
        for field, shape, dtype in ((name,(count,size,256),torch.float32),
                (name+"_valid",(count,size),torch.bool),(name+"_beats",(count,size),torch.float32)):
            value = bundle[field]
            if value.shape != shape or value.dtype != dtype or not torch.isfinite(value).all():
                raise ValueError("native audio shape/dtype/finite mismatch: " + field)
        expected = torch.tensor(starts)[:,None].float()/96 - 4 + (torch.arange(size).float()+(.5 if name=="p0" else 0))/(8 if name=="p0" else 24 if name=="c0" else 12)
        # Pooled C1 coordinates are mask-aware averages, unlike fixed-grid P0.
        if name == "p0" and not torch.equal(bundle[name+"_beats"], expected):
            raise ValueError("native P0 absolute fixed-grid coordinates differ")
    for key in GEOMETRY:
        value = bundle["interpolation_geometry"][key]
        shape = (count,) if key.endswith("start_tick") else (count,528 if key.startswith("a1_") else 1537)
        if value.shape != shape or not torch.isfinite(value).all():
            raise ValueError("native interpolation geometry shape/finite mismatch: " + key)
        if key.endswith(("left","right")):
            limit = 264 if key.startswith("a1_") else 528
            if value.dtype not in (torch.int32,torch.int64) or ((value<0)|(value>=limit)).any():
                raise ValueError("native interpolation index bounds differ: " + key)
        elif key.endswith("padding") and value.dtype != torch.bool:
            raise ValueError("native interpolation padding dtype differs")
        elif key.endswith("weight") and (value.dtype != torch.float32 or ((value<0)|(value>1)).any()):
            raise ValueError("native interpolation weight differs")
    if not torch.equal(bundle["interpolation_geometry"]["candidate_start_tick"].long(),torch.tensor(starts)):
        raise ValueError("native interpolation window ownership differs")


class TeacherAudioWriter:
    """Small interchange writer; the external old-source exporter supplies tensors."""
    def __init__(self, root, *, producer, record_to_variant=None, bindings=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.manifest = dict(schema=SCHEMA, ready=False, contract=CONTRACT, producer=producer,
            bindings=bindings or {}, record_to_variant=record_to_variant or {}, variants={})
        self._write()

    def _write(self):
        (self.root/"manifest.json").write_text(json.dumps(self.manifest,indent=2,allow_nan=False)+"\n",encoding="utf8")

    def append(self, variant, metadata, bundle):
        if variant in self.manifest["variants"]:
            raise ValueError("duplicate native audio variant")
        starts = [int(v) for v in metadata["starts"]]
        _validate_bundle(bundle,starts)
        arrays = {k:v.detach().cpu().numpy() for k,v in bundle.items() if k != "interpolation_geometry"}
        arrays.update({"geometry__"+k:v.detach().cpu().numpy() for k,v in bundle["interpolation_geometry"].items() if k in GEOMETRY})
        file = f"audio-{len(self.manifest['variants']):05d}.npz"
        path = self.root/file
        np.savez(path,**arrays)
        from maimai_dataset.integrity import file_sha256
        self.manifest["variants"][str(variant)] = dict(metadata,file=file,bytes=path.stat().st_size,sha256=file_sha256(path))
        self._write()

    def finish(self):
        if not self.manifest["variants"]:
            raise ValueError("native audio cache is empty")
        if set(self.manifest["record_to_variant"].values()) - set(self.manifest["variants"]):
            raise ValueError("native audio record mapping lacks variants")
        self.manifest["ready"] = True
        self.manifest["identity"] = digest({k:v for k,v in self.manifest.items() if k != "identity"})
        self._write()
        return self.manifest


class TeacherAudioCache:
    def __init__(self, root, *, descriptor=None):
        self.root = Path(root).resolve()
        self.manifest = (descriptor["manifest"] if descriptor is not None else
            json.loads((self.root/"manifest.json").read_text("utf-8-sig")))
        m = self.manifest
        if m.get("schema") != SCHEMA or m.get("contract") != CONTRACT or not m.get("ready"):
            raise ValueError("native teacher audio schema/contract/readiness mismatch")
        self.identity = dict(schema=SCHEMA,identity=m["identity"],contract=CONTRACT,producer=m["producer"],
            bindings=m.get("bindings",{}),variants=len(m["variants"]))
        self._loaded = OrderedDict()

    def descriptor(self):
        return dict(root=str(self.root),manifest=self.manifest)

    def __getstate__(self):
        return dict(self.__dict__,_loaded=OrderedDict())

    def resolve_variant(self, song_id, timing=None, *, variant=None):
        if variant is not None:
            if variant not in self.manifest["variants"]:
                raise ValueError("native audio variant absent")
            choices = [variant]
        else:
            choices = [k for k,v in self.manifest["variants"].items()
                if str(song_id) in set(map(str,v.get("song_ids",[v.get("song_id")])))]
        if timing is not None:
            from maimai_baseline.time.grid import timing_identity
            from maimai_dataset.timing import TimingMap
            wanted = timing_identity(TimingMap.from_dict(timing) if isinstance(timing,dict) else timing)
            choices = [k for k in choices if timing_identity(TimingMap.from_dict(self.manifest["variants"][k]["timing"])) == wanted]
        if len(choices) != 1:
            raise ValueError("native audio requires an unambiguous song/timing variant")
        return choices[0]

    def variant(self, variant):
        if variant not in self._loaded:
            meta = self.manifest["variants"][variant]
            path = self.root/meta["file"]
            if path.stat().st_size != meta["bytes"]:
                raise ValueError("native audio payload length mismatch")
            with np.load(path,allow_pickle=False) as archive:
                value = {k:torch.from_numpy(archive[k].copy()) for k in archive.files if not k.startswith("geometry__")}
                value["interpolation_geometry"] = {k:torch.from_numpy(archive["geometry__"+k].copy()) for k in GEOMETRY}
            _validate_bundle(value,[int(s) for s in meta["starts"]])
            self._loaded[variant] = value
            # Keep recent songs, never the complete data corpus in workers.
            while len(self._loaded)>2:
                self._loaded.popitem(last=False)
        self._loaded.move_to_end(variant)
        return self._loaded[variant]

    def _stitched(self, bundle):
        return dict(audio=bundle["p0"][:,32:160].reshape(-1,256),
            audio_valid=bundle["p0_valid"][:,32:160].reshape(-1),
            audio_beats=bundle["p0_beats"][:,32:160].reshape(-1))

    def bundles(self, variants):
        """Retain one logical batch explicitly, independently of the small LRU."""
        return {v:self.variant(v) for v in dict.fromkeys(variants)}

    def stitched(self, variant, *, bundles=None):
        return self._stitched(self.variant(variant) if bundles is None else bundles[variant])

    def draft_horizon(self, variant, *, beats_per_token=4):
        """Queries depend on decoded audio, never a reference chart's tail."""
        from maimai_baseline.time.audio_end import audio_end
        from maimai_dataset.timing import TimingMap
        from maimai_dataset.window import training_window_count
        meta = self.manifest["variants"][variant]
        boundary = audio_end(TimingMap.from_dict(meta["timing"]), int(meta["num_samples"]))
        windows = training_window_count(boundary.onset_tick, boundary.tail_tick)
        if beats_per_token not in (1, 4):
            raise ValueError("unsupported audio draft token geometry")
        return dict(windows=windows, blocks=windows * (16 // beats_per_token),
            condition_end_tick=windows * 1536, span_beats=boundary.onset_tick / 96.)

    def draft_stitched(self, variant, *, bundles=None, beats_per_token=4):
        """Crop an export's extra reference coverage before draft attention."""
        horizon = self.draft_horizon(variant, beats_per_token=beats_per_token)
        bundle = self.variant(variant) if bundles is None else bundles[variant]
        count = horizon["windows"]
        if bundle["p0"].shape[0] < count:
            raise ValueError("native audio cache does not cover the audio-derived draft horizon")
        return {**self._stitched({key: value[:count] for key, value in bundle.items()
                                 if key != "interpolation_geometry"}), **horizon}

    def pretrain_batch(self, variants):
        unique = list(dict.fromkeys(variants))
        rows = [self.stitched(v) for v in unique]
        return {**{k:pad_sequence([r[k] for r in rows],batch_first=True) for k in rows[0]},
            "window_to_audio":torch.tensor([unique.index(v) for v in variants],dtype=torch.long)}

    def _window_context(self, variants, starts, bundles):
        values = {k:[] for k in ("p0","p0_valid","p0_beats","c0","c0_valid","c0_beats","c1","c1_valid","c1_beats")}
        geometry = {k:[] for k in GEOMETRY}
        for variant,start in zip(variants,starts,strict=True):
            try:
                index = self.manifest["variants"][variant]["starts"].index(int(start))
            except ValueError as error:
                raise ValueError("native audio requested an uncached window") from error
            bundle = bundles[variant]
            for key in values:
                values[key].append(bundle[key][index])
            for key in geometry:
                geometry[key].append(bundle["interpolation_geometry"][key][index])
        v = {k:torch.stack(rows) for k,rows in values.items()}
        g = {k:torch.stack(rows) for k,rows in geometry.items()}
        context = AudioContext(memories=(v["p0"],),padding_masks=(~v["p0_valid"],),beats=(v["p0_beats"],),
            fine_memories=(v["c0"],v["c1"]),fine_padding_masks=(~v["c0_valid"],~v["c1_valid"]),
            fine_beats=(v["c0_beats"],v["c1_beats"]))
        return context,g

    def window_context(self, variant, starts, *, bundles=None):
        variants = [variant]*len(starts) if isinstance(variant,str) else list(variant)
        bundles = self.bundles(variants) if bundles is None else bundles
        return self._window_context(variants,starts,bundles)

    def dense_inputs(self, variants, starts):
        variants = list(variants)
        unique = list(dict.fromkeys(variants))
        bundles = {v:self.variant(v) for v in unique}
        rows = [self._stitched(bundles[v]) for v in unique]
        song = AudioContext(memories=(pad_sequence([r["audio"] for r in rows],batch_first=True),),
            padding_masks=(~pad_sequence([r["audio_valid"] for r in rows],batch_first=True),),
            beats=(pad_sequence([r["audio_beats"] for r in rows],batch_first=True),))
        local,geometry = self._window_context(variants,starts,bundles)
        return dict(audio_context=context_dict(song),local_audio_context=context_dict(local),
            teacher_interpolation=geometry,window_to_audio=torch.tensor([unique.index(v) for v in variants],dtype=torch.long))
