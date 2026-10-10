"""Immutable, mmap-backed posterior/audio cache for a frozen stage-A model."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from maimai_baseline.style_contract import STYLE_SLOTS
from maimai_dataset.integrity import file_sha256

CACHE_SCHEMA = "maimai.dual_decoder_feature_cache.v1"
ARRAYS = {
    "features": ("float32", (STYLE_SLOTS, 128)),
    "valid": ("bool", (STYLE_SLOTS,)),
    "reference_features": ("float32", (4, 128)),
    "reference_valid": ("bool", (4,)),
    "span_beats": ("float32", ()),
}


def representation_cache_conditions(*, start_tick, end_tick, song_end_tick):
    """Geometry only. Target production neither encodes nor stores audio."""
    from maimai_baseline.time.style import style_geometry
    _, valid, span = style_geometry(start_tick, end_tick, song_end_tick)
    _, reference_valid, _ = style_geometry(start_tick, end_tick, song_end_tick, beats_per_token=4, slots=4)
    return dict(span_beats=span, valid=valid, reference_valid=reference_valid)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class StyleCacheWriter:
    def __init__(self, root, records, bindings):
        self.root = Path(root).resolve()
        if self.root.exists():
            raise ValueError(f"refusing to overwrite style cache: {self.root}")
        self.records, self.bindings = records, bindings
        if not records:
            raise ValueError("empty style cache selection")
        self.root.mkdir(parents=True)
        self.arrays = {name: np.lib.format.open_memmap(self.root / (name + ".npy"), mode="w+",
                            dtype=dtype, shape=(len(records), *tail)) for name, (dtype, tail) in ARRAYS.items()}
        self.written = np.zeros(len(records), bool)
        write_json(self.root / "manifest.json", {"schema": CACHE_SCHEMA, "ready": False})

    def append(self, indices, features, conditions):
        indices = np.asarray(indices, np.int64)
        if indices.ndim != 1 or (indices < 0).any() or (indices >= len(self.records)).any():
            raise ValueError("invalid style cache row indices")
        if len(set(indices.tolist())) != len(indices) or self.written[indices].any():
            raise ValueError("duplicate style cache rows")
        values = dict(features=features,
                      **{name: conditions[name] for name in ("valid", "reference_features", "reference_valid", "span_beats")})
        if "song_end_tick" in conditions:
            for i, end in zip(indices, conditions["song_end_tick"].cpu().tolist()):
                self.records[int(i)]["song_end_tick"] = int(end)
        for name, value in values.items():
            dtype, tail = ARRAYS[name]
            if tuple(value.shape) != (len(indices), *tail) or str(value.dtype) != "torch." + dtype:
                raise ValueError("style cache value shape/dtype mismatch: " + name)
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"non-finite style cache data: {name}")
            self.arrays[name][indices] = value.detach().cpu().numpy()
        self.written[indices] = True

class StyleLatentCache:
    """Read the selected cache as written; workers reopen only requested arrays."""
    def __init__(self, root, *, verify_payloads=False, fields=None, descriptor=None):
        self.root = Path(root).resolve()
        self.manifest = (json.loads((self.root / 'manifest.json').read_text('utf8'))
                         if descriptor is None else descriptor['manifest'])
        value = self.manifest
        if value.get('schema') != CACHE_SCHEMA or not value.get('ready'):
            raise ValueError('style cache format/readiness mismatch')
        self.records = value['records']
        self.full_charts = {tuple(r['key']): r for r in value.get('full_charts', [])}
        self.max_blocks = max((r['shape'][0] for r in self.full_charts.values()), default=0)
        if descriptor is None:
            self.train_lookup = {}
            for i, record in enumerate(self.records):
                if record['role'] == 'train':
                    key = (record['canonical_index'], record['view_id'])
                    if key in self.train_lookup:
                        raise ValueError('duplicate style cache train index')
                    self.train_lookup[key] = i
            from maimai_baseline.ar_contract import validate_normalization
            validate_normalization(value['normalization'])
        else:
            self.train_lookup = descriptor['train_lookup']
        self.arrays = {}
        for name in (ARRAYS if fields is None else fields):
            dtype, tail = ARRAYS[name]
            info = value['files'][name]
            path = self.root / info.get('file', name + '.npy')
            if verify_payloads and file_sha256(path) != info['sha256']:
                raise ValueError('style cache payload hash mismatch: ' + name)
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            if array.shape != (len(self.records), *tail) or str(array.dtype) != dtype:
                raise ValueError('style cache array shape/dtype mismatch: ' + name)
            self.arrays[name] = array
        if verify_payloads:
            for info in self.full_charts.values():
                if file_sha256(self.root / info['file']) != info['sha256']:
                    raise ValueError('full feature payload mismatch')

    def descriptor(self):
        return dict(root=str(self.root), manifest=self.manifest, train_lookup=self.train_lookup)

    def plan_indices(self, refs):
        return [self.train_lookup[(r.canonical_index, r.view_id)] for r in refs]
