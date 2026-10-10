"""Exact float64 half-tick tables for the accepted full-train input assets."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from copy import copy
from collections import OrderedDict
import json
import math
import os
import numpy as np

from maimai_baseline.time.audio_end import audio_end
from maimai_dataset.integrity import file_sha256
from maimai_dataset.manifest import canonical_json_sha256
from maimai_dataset.timing import TimingMap

SCHEMA = "maimai.full_train_timing.v1"


def identity(timing: TimingMap) -> str:
    # to_dict retains redundant controls, accumulated seconds, meter/HS and E.
    return canonical_json_sha256(timing.to_dict())


class PreparedTiming:
    """A read-only profile timing, reused without reparsing each window.

    Ordinary TimingMap construction/binding remains mutable. This wrapper is
    only for immutable runtime metadata, and keeps separate bindings per root.
    """

    def __init__(self, timing: TimingMap):
        self._payload = timing.to_dict()
        self._key = canonical_json_sha256(self._payload)
        self._base = copy(timing)
        for name in ("_offline_table", "_offline_audio_ends", "_offline_root"):
            self._base.__dict__.pop(name, None)
        self._prepare_base()

    def _prepare_base(self):
        self._base._prepared_timing_key = self._key
        self._base._prepared_timing_payload = self._payload
        self._pid = os.getpid()
        self._bindings = {}

    def get(self, root=None) -> TimingMap:
        if self._pid != os.getpid():
            self._pid = os.getpid()
            self._bindings = {}
        if root is None:
            return self._base
        path = os.path.abspath(root)
        if path not in self._bindings:
            self._bindings[path] = store_for(path)._bind(copy(self._base), self._key)
        return self._bindings[path]

    def __getstate__(self):
        # Worker startup copies only metadata, never a bound mmap/table.
        return self._base, self._key, self._payload

    def __setstate__(self, state):
        self._base, self._key, self._payload = state
        self._prepare_base()


MAX_TRANSPORT_TIMINGS = 4096
_transport_timings = OrderedDict()
_transport_pid = os.getpid()


def restore_prepared_timing(key, payload, root=None) -> TimingMap:
    """Resolve immutable IPC metadata with bounded process-local reuse.

    Returned maps are read-only runtime inputs. Ordinary mutable TimingMap
    construction and binding do not enter this cache.
    """
    global _transport_pid
    if _transport_pid != os.getpid():
        _transport_timings.clear()
        _transport_pid = os.getpid()
    prepared = _transport_timings.pop(key, None)
    if prepared is None:
        prepared = PreparedTiming.__new__(PreparedTiming)
        prepared._base = TimingMap.from_dict(payload)
        prepared._key, prepared._payload = key, payload
        prepared._prepare_base()
    _transport_timings[key] = prepared
    while len(_transport_timings) > MAX_TRANSPORT_TIMINGS:
        _transport_timings.popitem(last=False)
    return prepared.get(root)


def build(root, bpm_manifest, profile_segments):
    """Build once from accepted BPM metadata; no audio or old GT payload scan."""
    root = Path(root)
    source = Path(bpm_manifest)
    value = json.loads(source.read_text(encoding="utf8"))
    entries = {}
    curves = {}
    def include(timing, num_samples):
        key = identity(timing)
        curve_key = canonical_json_sha256({"first_seconds": timing.first_seconds,
            "points": [[p.beat.numerator, p.beat.denominator, p.bpm, p.second] for p in timing.points]})
        end = audio_end(timing, int(num_samples))
        from maimai_baseline.time.grid import AUDIO_BEATS_BEFORE, WINDOW_POSITIONS
        first_audio = math.ceil(float(timing.second_to_beat(0.0)) * 24 - 1e-9) * 4
        minimum = min(first_audio, -AUDIO_BEATS_BEFORE * 96) - 8
        maximum = end.tail_tick + WINDOW_POSITIONS * 4 + 384
        current = entries.setdefault(key, {"timing": timing.to_dict(), "minimum": minimum,
                                          "maximum": maximum, "audio_ends": {}, "curve": curve_key})
        current["minimum"] = min(current["minimum"], minimum)
        current["maximum"] = max(current["maximum"], maximum)
        current["audio_ends"][str(num_samples)] = {
            "seconds": end.seconds, "onset_tick": end.onset_tick, "tail_tick": end.tail_tick}
        curve = curves.setdefault(curve_key, {"timing": timing, "minimum": minimum, "maximum": maximum})
        curve["minimum"] = min(curve["minimum"], minimum)
        curve["maximum"] = max(curve["maximum"], maximum)
    for meta in value["variants"].values():
        include(TimingMap.from_dict(meta["timing"]), int(meta["num_samples"]))
    selected = value["segments"]
    seen = set()
    segment_path = Path(profile_segments)
    with segment_path.open(encoding="utf8") as stream:
        for line in stream:
            if not line.strip():
                continue
            segment = json.loads(line)
            sid = str(segment["segment_id"])
            if sid not in selected:
                continue
            seen.add(sid)
            variant = value["variants"][selected[sid]]
            include(TimingMap.from_dict(segment["timing"]), int(variant["num_samples"]))
    if seen != set(selected):
        raise ValueError("selected BPM segments missing from profile timing metadata")
    curve_files = {}
    for n, (key, item) in enumerate(curves.items()):
        timing = item["timing"]
        lo, hi = 2 * item["minimum"], 2 * item["maximum"] + 1
        path = root / f"timing-{n}.npy"
        array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64, shape=(hi - lo + 1,))
        for index in range(len(array)):
            array[index] = timing.beat_to_second(Fraction(lo + index, 192))
        array.flush()
        curve_files[key] = {"file": path.name, "sha256": file_sha256(path), "bytes": path.stat().st_size,
                            "first_half_tick": lo, "last_half_tick": hi}
    result = {key: {**curve_files[item["curve"]], "timing": item["timing"],
                    "audio_ends": item["audio_ends"]} for key, item in entries.items()}
    return {"schema": SCHEMA, "bpm_manifest_sha256": file_sha256(source),
            "profile_segments_sha256": file_sha256(segment_path), "tables": result}


class TimeTableStore:
    def __init__(self, root, manifest, *, verify=False):
        self.root = Path(root)
        if manifest.get("schema") != SCHEMA:
            raise ValueError("full-train timing table schema mismatch")
        self.manifest = manifest
        self.arrays = {}
        loaded = {}
        hashed = set()
        for key, meta in manifest["tables"].items():
            if key != canonical_json_sha256(meta["timing"]):
                raise ValueError("full TimingMap identity mismatch")
            path = self.root / meta["file"]
            if path.stat().st_size != meta["bytes"]:
                raise ValueError("timing table byte length mismatch")
            if verify and path not in hashed:
                if file_sha256(path) != meta["sha256"]:
                    raise ValueError("timing table hash mismatch")
                hashed.add(path)
            if path not in loaded:
                loaded[path] = np.load(path, mmap_mode="r", allow_pickle=False)
            array = loaded[path]
            if array.dtype != np.float64 or array.ndim != 1 or len(array) != meta["last_half_tick"] - meta["first_half_tick"] + 1:
                raise ValueError("timing table shape/dtype mismatch")
            self.arrays[key] = array

    def bind(self, timing: TimingMap) -> TimingMap:
        return self._bind(timing, identity(timing))

    def _bind(self, timing: TimingMap, key: str) -> TimingMap:
        if key not in self.arrays:
            raise ValueError("training TimingMap missing from accepted offline table")
        meta = self.manifest["tables"][key]
        timing._offline_table = (self.arrays[key], meta["first_half_tick"])
        timing._offline_audio_ends = meta["audio_ends"]
        timing._offline_root = str(self.root.resolve())
        return timing


_stores = {}


def store_for(root) -> TimeTableStore:
    path = str(Path(root).resolve())
    key = (os.getpid(), path)
    if key not in _stores:
        manifest = json.loads((Path(path) / "manifest.json").read_text(encoding="utf8"))
        _stores[key] = TimeTableStore(path, manifest["timing"])
    return _stores[key]
