"""Read-only dtype-contiguous storage for complete chart/view GT inputs."""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import torch

from maimai_dataset.integrity import file_sha256
from maimai_dataset.manifest import canonical_json_sha256

SCHEMA = "maimai.full_song_chart_inputs.compact.v2"


def _encode(value, streams, offsets, root):
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().contiguous().numpy()
        dtype = array.dtype.str
        start = offsets.get(dtype, 0)
        flat = array.reshape(-1)
        if dtype not in streams:
            path = root / f"dtype-{len(streams)}.bin"
            streams[dtype] = (path, path.open("wb"))
        streams[dtype][1].write(flat.tobytes(order="C"))
        offsets[dtype] = start + flat.size
        return {"kind": "tensor", "dtype": dtype, "offset": start,
                "length": int(flat.size), "shape": list(array.shape)}
    if isinstance(value, dict):
        return {"kind": "dict", "items": [[k, _encode(v, streams, offsets, root)] for k, v in value.items()]}
    if isinstance(value, list):
        return {"kind": "list", "items": [_encode(v, streams, offsets, root) for v in value]}
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [_encode(v, streams, offsets, root) for v in value]}
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"kind": "scalar", "value": value}
    raise TypeError(f"unsupported complete chart field: {type(value)!r}")


def _decode(spec, arrays):
    kind = spec["kind"]
    if kind == "tensor":
        raw = arrays[spec["dtype"]]
        start, length = int(spec["offset"]), int(spec["length"])
        if start < 0 or length < 0 or start + length > raw.size:
            raise ValueError("compact chart tensor index outside dtype buffer")
        view = raw[start:start + length].reshape(spec["shape"])
        return torch.from_numpy(view)
    if kind == "dict":
        return {k: _decode(v, arrays) for k, v in spec["items"]}
    if kind == "list":
        return [_decode(v, arrays) for v in spec["items"]]
    if kind == "tuple":
        return tuple(_decode(v, arrays) for v in spec["items"])
    if kind == "scalar":
        return spec["value"]
    raise ValueError(f"invalid compact chart node: {kind}")


def build(old_root, new_root, *, source_identity, bpm_manifest, profile_segments,
          song_audio_root=None):
    """Stream each accepted v1 chart once; keep its manifest as provenance."""
    from maimai_baseline.data.song_chart import SongChartCache, SCHEMA as OLD_SCHEMA
    old = SongChartCache(old_root)
    if old.manifest["schema"] != OLD_SCHEMA:
        raise ValueError("compact conversion requires the accepted v1 GT source")
    root = Path(new_root)
    root.mkdir(parents=True, exist_ok=False)
    from maimai_baseline.time.offline import build as build_timing
    if old.manifest["bindings"]["bpm"] != file_sha256(Path(bpm_manifest)):
        raise ValueError("compact timing source differs from accepted GT BPM binding")
    timing_manifest = build_timing(root, bpm_manifest, profile_segments)
    streams, offsets = {}, {}
    charts = []
    # First-appearance order is the producer's chart order, not a sorted key order.
    for chart in old.manifest["charts"]:
        packed = old.get(chart["key"])
        index = _encode(packed, streams, offsets, root)
        charts.append({k: v for k, v in chart.items() if k not in ("file", "sha256", "bytes")}
                      | {"fields": index})
    files = {}
    for dtype, (path, output) in streams.items():
        output.close()
        files[dtype] = {"file": path.name, "elements": offsets[dtype],
                        "bytes": path.stat().st_size, "sha256": file_sha256(path)}
    manifest = {"schema": SCHEMA, "ready": True, "representation": old.manifest["representation"],
                "layout": "dtype-contiguous-flat-readonly-mmap-v1", "charts": charts,
                "files": files, "bindings": old.manifest["bindings"],
                "source_chart_identity": old.manifest["identity"],
                "source_chart_manifest_sha256": file_sha256(Path(old_root) / "manifest.json"),
                "source_audio_root": str(Path(old_root).resolve()),
                "source_chart_files": {str(tuple(c["key"])): {k: c[k] for k in ("file", "sha256", "bytes")}
                                       for c in old.manifest["charts"]},
                "audio_files": old.manifest["audio_files"],
                "producer": source_identity}
    if song_audio_root is not None:
        from maimai_baseline.data.song_audio import SongAudioCache
        audio = SongAudioCache(song_audio_root)
        expected = {"bpm": old.manifest["bindings"]["bpm"],
                    "chart_inputs": old.manifest["identity"],
                    "plan": old.manifest["bindings"]["plan"]}
        if audio.manifest["bindings"] != expected or not {c["variant"] for c in charts}.issubset(audio.manifest["variants"]):
            raise ValueError("compact GT source/variant relation to retained song audio mismatch")
        manifest["song_audio"] = {"identity": audio.manifest["identity"],
                                  "manifest_sha256": file_sha256(audio.root / "manifest.json"),
                                  "variant_to_file": {v: audio.manifest["variants"][v]
                                                      for v in dict.fromkeys(c["variant"] for c in charts)}}
    manifest["timing"] = timing_manifest
    manifest["identity"] = canonical_json_sha256(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf8")
    return manifest


class CompactSongChartCache:
    def __init__(self, root, *, verify=False):
        self.root = Path(root)
        m = json.loads((self.root / "manifest.json").read_text(encoding="utf8"))
        if (m.get("schema") != SCHEMA or m.get("layout") != "dtype-contiguous-flat-readonly-mmap-v1"
                or not m.get("ready")):
            raise ValueError("compact chart manifest layout mismatch")
        from maimai_baseline.style_contract import STYLE_ID
        if m.get("representation") != STYLE_ID:
            raise ValueError("compact chart representation mismatch")
        self.manifest = m
        self.lookup = {tuple(c["key"]): c for c in m["charts"]}
        if len(self.lookup) != len(m["charts"]):
            raise ValueError("duplicate compact chart/view")
        self.arrays = {}
        for dtype, meta in m["files"].items():
            path = self.root / meta["file"]
            if path.stat().st_size != meta["bytes"] or meta["bytes"] != meta["elements"] * np.dtype(dtype).itemsize:
                raise ValueError("compact chart dtype buffer length mismatch")
            if verify and file_sha256(path) != meta["sha256"]:
                raise ValueError("compact chart dtype buffer hash mismatch")
            if meta["elements"]:
                self.arrays[dtype] = np.memmap(path, mode="r", dtype=np.dtype(dtype), shape=(meta["elements"],))
            else:
                self.arrays[dtype] = np.empty((0,), dtype=np.dtype(dtype))
                self.arrays[dtype].setflags(write=False)
        from maimai_baseline.time.offline import TimeTableStore
        self.time_tables = TimeTableStore(self.root, m["timing"], verify=verify)

    def get(self, key):
        return _decode(self.lookup[tuple(key)]["fields"], self.arrays)
