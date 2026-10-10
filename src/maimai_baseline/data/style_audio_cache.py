"""Frozen P0 sidecar. Chart features and posterior normalization stay immutable."""
from pathlib import Path
import json
import numpy as np
import torch
from maimai_baseline.data.style_cache import write_json
from maimai_dataset.manifest import canonical_json_sha256
from maimai_dataset.integrity import file_sha256

SCHEMA = "maimai.style_p0_condition_cache.v3"
CONDITION = {"scale": "P0", "shape": [176, 256], "precision": "fp32-eval",
             "coordinates": "window-relative-beats", "context_beats": [-4, 18]}
FIELDS = {"audio": ("float32", (176, 256)), "audio_valid": ("bool", (176,)),
          "audio_beats": ("float32", (176,))}


def record_key(record):
    return (record["role"], str(record["segment_id"]), int(record["view_id"]))


class AudioConditionWriter:
    def __init__(self, root, *, keys, record_to_audio, validation_batches, bindings, producer):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.metadata = dict(schema=SCHEMA, ready=False, condition=CONDITION,
            keys=[list(k) for k in keys], record_to_audio=list(record_to_audio),
            validation_batches=validation_batches, bindings=bindings, producer=producer)
        self.offset = 0
        self.arrays = {name: np.lib.format.open_memmap(self.root / (name + ".npy"), mode="w+",
            dtype=dtype, shape=(len(keys), *tail)) for name, (dtype, tail) in FIELDS.items()}
        write_json(self.root / "manifest.json", self.metadata)

    def append(self, keys, values):
        n = len(keys)
        if [list(k) for k in keys] != self.metadata["keys"][self.offset:self.offset+n]:
            raise ValueError("P0 cache group order mismatch")
        for name, (dtype, tail) in FIELDS.items():
            value = values[name].detach().cpu().numpy()
            if value.shape != (n, *tail) or str(value.dtype) != dtype or not np.isfinite(value).all():
                raise ValueError("P0 cache non-finite value or layout mismatch: " + name)
            self.arrays[name][self.offset:self.offset+n] = value
        self.offset += n

    def finish(self):
        if self.offset != len(self.metadata["keys"]):
            raise ValueError("P0 cache is incomplete")
        files = {}
        for name, array in self.arrays.items():
            array.flush()
            path = self.root / (name + ".npy")
            files[name] = dict(shape=list(array.shape), dtype=str(array.dtype),
                               bytes=path.stat().st_size, sha256=file_sha256(path))
        from maimai_baseline.runtime_versions import stamp
        value = dict(self.metadata, **stamp(), ready=True, files=files)
        value["identity"] = canonical_json_sha256(value)
        write_json(self.root / "manifest.json", value)
        return value


class AudioConditionCache:
    def __init__(self, root, target_cache, *, verify_payloads=False):
        self.root = Path(root)
        value = json.loads((self.root / "manifest.json").read_text("utf-8"))
        if value.get('schema') != SCHEMA or not value.get('ready'):
            raise ValueError('P0 condition cache format/readiness mismatch')
        self.manifest = value
        self.mapping = np.asarray(value["record_to_audio"], dtype=np.int64)
        n = len(value["keys"])
        if (self.mapping.shape != (len(target_cache.records),) or n < 1 or
                np.any(self.mapping < 0) or np.any(self.mapping >= n) or
                len({tuple(k) for k in value["keys"]}) != n):
            raise ValueError("P0 condition cache record mapping mismatch")
        for r, index in zip(target_cache.records, self.mapping):
            if value["keys"][int(index)][0] != r["role"]:
                raise ValueError("P0 condition cache role mapping mismatch")
        self.validation_batches = value["validation_batches"]
        selected = [i for batch in self.validation_batches for i in batch]
        expected = [i for i, r in enumerate(target_cache.records) if r["role"] == "validation"]
        if sorted(selected) != expected or any(not b for b in self.validation_batches):
            raise ValueError("P0 validation sampler coverage mismatch")
        self.arrays = {}
        for name, (dtype, tail) in FIELDS.items():
            info = value["files"][name]
            path = self.root / (name + ".npy")
            if verify_payloads and file_sha256(path) != info["sha256"]:
                raise ValueError("P0 condition cache payload hash mismatch")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.shape != (n, *tail) or str(array.dtype) != dtype:
                raise ValueError("P0 condition cache array header mismatch")
            self.arrays[name] = array

    def identity(self):
        return dict(identity=self.manifest["identity"], scale="P0", shape=[176, 256],
                    **{k:self.manifest["bindings"].get(k) for k in ("stage_a_sha256", "target_cache_identity")})

    def batch(self, indices, device="cpu"):
        selected = self.mapping[list(indices)]
        return {name: torch.from_numpy(np.array(array[selected], copy=True)).to(device)
                for name, array in self.arrays.items()}


def worker_init(_):
    torch.set_num_threads(1)


class BatchRequests:
    def __init__(self):
        self.requests = []

    def set_batches(self, batches, *, epoch, deterministic=False):
        self.requests = [(list(b), epoch, deterministic) for b in batches]

    def __iter__(self):
        return iter(self.requests)

    def __len__(self):
        return len(self.requests)
