"""Immutable per-target-window public MajRadar scores and their explicit producer."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import os
import subprocess
import sys
from typing import Mapping

import numpy as np
import torch

from maimai_baseline.radar_contract import SCHEMA, DIMENSIONS, SCALES, BEATS_PER_TOKEN, CONTRACT


VIEWS = ("identity", "mirror-lr")


def _read(path):
    return json.loads(Path(path).read_text("utf-8-sig"))


def _rows(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", "utf-8")


def view_name(view):
    if view in (0, "identity"):
        return "identity"
    if view in (1, "mirror-lr"):
        return "mirror-lr"
    raise ValueError(f"unsupported MajRadar spatial view: {view}")


class MajRadarCache:
    """Read only metadata at construction; validate requested values on access."""
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = _read(self.root / "manifest.json")
        m = self.manifest
        if (m.get("schema") != SCHEMA or m.get("dimensions") != list(DIMENSIONS)
                or m.get("scales") != list(SCALES) or m.get("beats_per_token") != BEATS_PER_TOKEN
                or m.get("ownership") != CONTRACT["ownership"] or m.get("duration") != CONTRACT["duration"]
                or m.get("empty") != CONTRACT["empty"] or not m.get("ready")):
            raise ValueError("MajRadar cache layout or completion mismatch")
        self.segments = list(m["segments"])
        self.lookup = {(str(r["segment_id"]), str(r["view"])): i for i, r in enumerate(self.segments)}
        if len(self.lookup) != len(self.segments):
            raise ValueError("duplicate MajRadar segment/view")
        self.values = np.load(self.root / m["values"], mmap_mode="r", allow_pickle=False)
        if self.values.shape != (len(self.segments), len(DIMENSIONS)) or self.values.dtype != np.dtype("float32"):
            raise ValueError("MajRadar value array shape or dtype mismatch")

    def _tensor(self, indices):
        value = torch.from_numpy(np.array(self.values[indices], copy=True))
        if not bool(torch.isfinite(value).all()) or bool((value < 0).any()) or bool((value > value.new_tensor(SCALES)).any()):
            raise ValueError("MajRadar requested scores are non-finite or outside native ranges")
        return value

    def get(self, segment_id, view="identity"):
        return self._tensor(self.lookup[(str(segment_id), view_name(view))])

    def record_segments(self, record_id):
        return sorted((dict(r) for r in self.segments if str(r["record_id"]) == str(record_id)
                       and r["view"] == "identity"), key=lambda r: r["target_start_tick"])

    def indices(self, segments_or_dataset, record_id, view="identity"):
        segments = getattr(segments_or_dataset, "segments", segments_or_dataset)
        selected = sorted((s for s in segments if str(s["record_id"]) == str(record_id)),
                          key=lambda s: s["target_start_tick"])
        name = view_name(view)
        indices = [self.lookup[(str(s["segment_id"]), name)] for s in selected]
        for source, index in zip(selected, indices):
            cached = self.segments[index]
            if any(int(source[k]) != int(cached[k]) for k in ("target_start_tick", "target_end_tick")):
                raise ValueError("MajRadar target window coordinates changed")
        return torch.tensor(indices, dtype=torch.long)

    def group(self, segments_or_dataset, record_id, view="identity"):
        return self._tensor(self.indices(segments_or_dataset, record_id, view).numpy())


class MajRadarProvider:
    def __init__(self, root):
        self.root, self.cache = str(root), None

    def __call__(self, items):
        if self.cache is None:
            self.cache = MajRadarCache(self.root)
        return torch.stack([self.cache.get(i["segment_id"], i.get("spatial_view_id", "identity")) for i in items])


class MajRadarHost:
    """A persistent JSONL process: one parser adaptation per chart/view."""
    def __init__(self, dll):
        self.process = subprocess.Popen(["dotnet", str(Path(dll).resolve())], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=None, text=True, encoding="utf-8", bufsize=1)

    def analyze(self, request):
        self.process.stdin.write(json.dumps(request, ensure_ascii=False, allow_nan=False) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"MajRadar host stopped, exit={self.process.poll()}")
        result = json.loads(line)
        if result.get("record_id") != request["record_id"] or result.get("view") != request["view"]:
            raise ValueError("MajRadar host response ownership mismatch")
        return result

    def close(self):
        self.process.stdin.close()
        code = self.process.wait(timeout=30)
        self.process.stdout.close()
        if code:
            raise RuntimeError(f"MajRadar host exited {code}")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def build_host(host_project, build_root, majradar_project=None):
    build_root = Path(build_root).resolve()
    temporary = build_root / "system"
    temporary.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TEMP=str(temporary), TMP=str(temporary),
        DOTNET_CLI_HOME=str(build_root / "dotnet-home"), DOTNET_SKIP_FIRST_TIME_EXPERIENCE="1",
        DOTNET_GENERATE_ASPNET_CERTIFICATE="false", DOTNET_CLI_TELEMETRY_OPTOUT="1")
    command = ["dotnet", "build", str(Path(host_project).resolve()), "--artifacts-path", str(build_root / "dotnet"),
        "--output", str(build_root / "host"), "-c", "Release", "--nologo", "-v", "minimal"]
    if majradar_project is not None:
        command.append("-p:MajRadarProject=" + str(Path(majradar_project).resolve()))
    subprocess.run(command, env=env, check=True)
    return build_root / "host/MajRadarHost.dll"


def selected_segments(profile, subset_config):
    """Consume the existing selection; do not resample or repack the plan."""
    config = _read(subset_config) if not isinstance(subset_config, Mapping) else dict(subset_config)
    train_songs = set(map(str, config.get("train_song_ids", ())))
    difficulties = set(map(int, config.get("difficulties", (5, 6))))
    excluded = set(map(str, config.get("excluded_record_ids", ())))
    result = [s for s in _rows(Path(profile) / "segments.jsonl")
        if int(s.get("source_difficulty_id", s["difficulty_id"])) in difficulties
        and str(s["record_id"]) not in excluded and s.get("split") == "train"
        and (not train_songs or str(s["song_id"]) in train_songs)]
    if not result:
        raise ValueError("MajRadar selection is empty")
    if len({s["segment_id"] for s in result}) != len(result):
        raise ValueError("duplicate selected segment")
    for s in result:
        if int(s["target_end_tick"]) - int(s["target_start_tick"]) != BEATS_PER_TOKEN * 96:
            raise ValueError("MajRadar requires existing complete sixteen-beat windows")
    return result


def canonical_chart(profile, record, segments, view, note_manifest, miacode_exe):
    from fractions import Fraction
    from maimai_dataset.codec import restore_v1_rows
    from maimai_dataset.materialize import materialize_chart_ir
    from maimai_dataset.modifiers import normalize_row
    from maimai_dataset.timing import TimingMap
    from maimai_baseline.inference.materialize import render_chart
    # Reuse the maintained spatial transform, with no imports from other worktrees.
    scripts = Path(__file__).resolve().parents[3] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from full_dataset_tools.pipeline import transform_rows

    if note_manifest is not None:
        meta = note_manifest["charts"][str(record["record_id"])]
        document = _read(Path(profile).parent / "note-view" / meta["path"])
        canonical = document["canonical_rows"]
    else:
        canonical = [normalize_row(r) for r in _read(Path(profile) / record["encoded_path"])["rows"]]
    rows = transform_rows(canonical, VIEWS.index(view_name(view)))
    timing = TimingMap.from_dict(segments[0]["timing"])
    latest = max([Fraction(int(s["target_end_tick"]), 96) for s in segments]
        + [p.beat + Fraction(1, 96) for p in timing.points]
        + [b + Fraction(1, 96) for b, *_ in timing.meter_points]
        + [b + Fraction(1, 96) for b, *_ in timing.hs_points])
    ir = materialize_chart_ir(restore_v1_rows(rows), timing=timing,
        difficulty=str(record.get("difficulty", "")), render_end_beat=latest)
    inote = render_chart(Path(miacode_exe), ir)
    windows = [dict(segment_id=s["segment_id"], start_tick=int(s["target_start_tick"]),
        end_tick=int(s["target_end_tick"]),
        start_seconds=timing.beat_to_second(Fraction(int(s["target_start_tick"]), 96)) - timing.first_seconds,
        end_seconds=timing.beat_to_second(Fraction(int(s["target_end_tick"]), 96)) - timing.first_seconds)
        for s in segments]
    return dict(record_id=str(record["record_id"]), view=view_name(view), inote=inote, windows=windows)


def build_cache(profile, subset_config, output, miacode_exe, majradar_project=None,
                host_project=None, host_dll=None, build_root=None):
    import time
    from maimai_dataset.integrity import file_sha256
    from maimai_dataset.manifest import canonical_json_sha256

    started = time.perf_counter()
    profile, output = Path(profile).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError(f"MajRadar output already exists: {output}")
    segments = selected_segments(profile, subset_config)
    records = {str(r["record_id"]): r for r in _rows(profile / "manifest.jsonl")}
    grouped = defaultdict(list)
    for s in sorted(segments, key=lambda r: (str(r["record_id"]), int(r["target_start_tick"]))):
        grouped[str(s["record_id"])].append(s)
    checkout = Path(__file__).resolve().parents[3]
    if host_dll is None:
        host_project = host_project or checkout / "tools/MajRadarHost/MajRadarHost.csproj"
        build_root = build_root or checkout / ".test-temp/majradar-cache-host"
        host_dll = build_host(host_project, build_root, majradar_project)
    note_path = profile.parent / "note-view/manifest.json"
    note_manifest = _read(note_path) if note_path.is_file() else None
    index = [dict(segment_id=s["segment_id"], record_id=s["record_id"], view=view,
        target_start_tick=int(s["target_start_tick"]), target_end_tick=int(s["target_end_tick"]))
        for rid, group in grouped.items() for view in VIEWS for s in group]
    lookup = {(r["segment_id"], r["view"]): i for i, r in enumerate(index)}
    values = np.full((len(index), len(DIMENSIONS)), np.nan, dtype="float32")
    output.mkdir(parents=True)
    manifest = dict(schema=SCHEMA, dimensions=list(DIMENSIONS), scales=list(SCALES), beats_per_token=BEATS_PER_TOKEN,
        ownership=CONTRACT["ownership"], duration=CONTRACT["duration"], empty=CONTRACT["empty"],
        ready=False, values="values.npy", segments=index,
        provenance=dict(profile=str(profile), subset_config=str(subset_config),
            selection_sha256=canonical_json_sha256(index), upstream_commit="5ee318efd22c77ac2ec7ba846fe017202244813e"))
    _write(output / "manifest.json", manifest)
    failed, empty, count, tail_extensions = [], 0, 0, 0
    with MajRadarHost(host_dll) as host, (output / "analyses.jsonl").open("w", encoding="utf-8") as journal:
        for rid, group in grouped.items():
            for view in VIEWS:
                try:
                    result = host.analyze(canonical_chart(profile, records[rid], group, view, note_manifest, miacode_exe))
                except Exception as error:
                    result = dict(record_id=rid, view=view, status="error", errors=[str(error)], windows=[])
                journal.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                journal.flush()
                if result["status"] != "ok":
                    failed.extend(dict(segment_id=s["segment_id"], view=view, errors=result["errors"]) for s in group)
                else:
                    response_ids = [r["segment_id"] for r in result["windows"]]
                    if response_ids != [s["segment_id"] for s in group]:
                        raise ValueError("MajRadar response window sequence changed")
                    for r in result["windows"]:
                        if r["status"] != "ok":
                            failed.append(dict(segment_id=r["segment_id"], view=view, errors=r["errors"]))
                            continue
                        vector = np.asarray([r["scores"][d] for d in DIMENSIONS], dtype="float32")
                        if not np.isfinite(vector).all() or (vector < 0).any() or (vector > np.asarray(SCALES)).any():
                            failed.append(dict(segment_id=r["segment_id"], view=view, errors=["invalid native score vector"]))
                            continue
                        values[lookup[(r["segment_id"], view)]] = vector
                        empty += int(r["objects"] == 0)
                        tail_extensions += int(r["last_event_end_seconds"] > r["chart_end_seconds"] + 1e-9)
                count += len(group)
                print(json.dumps(dict(processed=count, total=len(index), record_id=rid, view=view,
                    failures=len(failed)), ensure_ascii=False), flush=True)
    np.save(output / "values.npy", values, allow_pickle=False)
    _write(output / "errors.json", failed)
    coverage = dict(records=len(grouped), source_segments=len(segments), vectors=len(index),
        successful=int(np.isfinite(values).all(axis=1).sum()), failed=len(failed),
        empty_vectors=empty, tail_extended_vectors=tail_extensions, seconds=time.perf_counter() - started)
    manifest.update(coverage=coverage, ready=not failed and coverage["successful"] == len(index),
        values_sha256=file_sha256(output / "values.npy"))
    manifest["data_binding"] = dict(values_sha256=manifest["values_sha256"],
        index_sha256=manifest["provenance"]["selection_sha256"])
    _write(output / "manifest.json", manifest)
    print(json.dumps(coverage, ensure_ascii=False), flush=True)
    if not manifest["ready"]:
        raise ValueError(f"MajRadar analysis failures preserved in {output / 'errors.json'}")
    return manifest


def repair_cache(profile, subset_config, output, miacode_exe, record_ids,
                 majradar_project=None, host_project=None, host_dll=None, build_root=None):
    """Repair explicitly named records in this producer's incomplete output.

    Accepted, ready caches are immutable. Keep initial attempt payloads and the
    native journal, and preserve remaining failures rather than invent labels.
    """
    import shutil
    import time
    from maimai_dataset.integrity import file_sha256
    started = time.perf_counter()
    profile, output = Path(profile).resolve(), Path(output).resolve()
    manifest = _read(output / "manifest.json")
    if manifest.get("schema") != SCHEMA or manifest.get("ready"):
        raise ValueError("repair requires an incomplete MajRadar output")
    chosen = set(map(str, record_ids))
    if not chosen:
        raise ValueError("repair requires explicit record ids")
    grouped = defaultdict(list)
    for s in selected_segments(profile, subset_config):
        if str(s["record_id"]) in chosen:
            grouped[str(s["record_id"])].append(s)
    if set(grouped) != chosen:
        raise ValueError("repair record absent from original selection")
    records = {str(r["record_id"]): r for r in _rows(profile / "manifest.jsonl") if str(r["record_id"]) in chosen}
    index = manifest["segments"]
    lookup = {(r["segment_id"], r["view"]): i for i, r in enumerate(index)}
    values = np.load(output / manifest["values"], allow_pickle=False)
    failed = _read(output / "errors.json")
    retained = [e for e in failed if str(e["segment_id"]).rsplit(":", 1)[0] not in chosen]
    snapshot = output / "initial-attempt"
    snapshot.mkdir(exist_ok=False)
    for filename in ("manifest.json", "errors.json", manifest["values"]):
        shutil.copyfile(output / filename, snapshot / filename)
    previous = {(r["record_id"], r["view"]): r for r in _rows(output / "analyses.jsonl") if str(r["record_id"]) in chosen}
    checkout = Path(__file__).resolve().parents[3]
    if host_dll is None:
        host_dll = build_host(host_project or checkout / "tools/MajRadarHost/MajRadarHost.csproj",
                             build_root or checkout / ".test-temp/majradar-cache-host", majradar_project)
    note_path = profile.parent / "note-view/manifest.json"
    note_manifest = _read(note_path) if note_path.is_file() else None
    empty_delta, tail_delta, repaired = 0, 0, 0
    with MajRadarHost(host_dll) as host, (output / "repairs.jsonl").open("w", encoding="utf-8") as journal:
        for rid in sorted(chosen):
            group = sorted(grouped[rid], key=lambda s: s["target_start_tick"])
            for view in VIEWS:
                result = host.analyze(canonical_chart(profile, records[rid], group, view, note_manifest, miacode_exe))
                journal.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                journal.flush()
                old = previous[(rid, view)]["windows"]
                empty_delta -= sum(r["status"] == "ok" and r["objects"] == 0 for r in old)
                tail_delta -= sum(r["status"] == "ok" and r["last_event_end_seconds"] > r["chart_end_seconds"] + 1e-9 for r in old)
                if result["status"] != "ok":
                    retained.extend(dict(segment_id=s["segment_id"], view=view, errors=result["errors"]) for s in group)
                    continue
                if [r["segment_id"] for r in result["windows"]] != [s["segment_id"] for s in group]:
                    raise ValueError("repair response window sequence changed")
                for r in result["windows"]:
                    key = lookup[(r["segment_id"], view)]
                    values[key] = np.nan
                    if r["status"] != "ok":
                        retained.append(dict(segment_id=r["segment_id"], view=view, errors=r["errors"]))
                        continue
                    vector = np.asarray([r["scores"][d] for d in DIMENSIONS], dtype="float32")
                    if not np.isfinite(vector).all() or (vector < 0).any() or (vector > np.asarray(SCALES)).any():
                        raise ValueError("repair returned invalid native scores")
                    values[key] = vector
                    empty_delta += int(r["objects"] == 0)
                    tail_delta += int(r["last_event_end_seconds"] > r["chart_end_seconds"] + 1e-9)
                    repaired += 1
    np.save(output / manifest["values"], values, allow_pickle=False)
    _write(output / "errors.json", retained)
    coverage = manifest["coverage"]
    coverage.update(successful=int(np.isfinite(values).all(axis=1).sum()), failed=len(retained),
                    empty_vectors=coverage["empty_vectors"] + empty_delta,
                    tail_extended_vectors=coverage["tail_extended_vectors"] + tail_delta)
    manifest.update(ready=not retained and coverage["successful"] == len(index),
        values_sha256=file_sha256(output / manifest["values"]))
    manifest["data_binding"]["values_sha256"] = manifest["values_sha256"]
    manifest["repair"] = dict(records=sorted(chosen), vectors=repaired, seconds=time.perf_counter() - started,
        reason="native-float-BPM-clock-window-boundaries", initial_attempt="initial-attempt",
        journal="repairs.jsonl")
    _write(output / "manifest.json", manifest)
    print(json.dumps(dict(coverage=coverage, repair=manifest["repair"])), flush=True)
    if not manifest["ready"]:
        raise ValueError("MajRadar repair remains incomplete; errors preserved")
    return manifest
