"""Prepare the independent native-audio experiment without any old learned cache."""
from __future__ import annotations

from copy import deepcopy
from collections import defaultdict
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np

from maimai_baseline.data.curriculum import EpochPlanStore, MultiViewRouteStore
from maimai_baseline.data.frozen_latent import selected_segments
from maimai_baseline.data.prior_decoder import (_read, _rows, _plans, _standard_vectors,
    _window_cache, WindowCache)
from maimai_baseline.data.style_cache import write_json
from maimai_baseline.data.training_build import segment_chart
from maimai_baseline.training.paths import DEFAULT_INPUTS
from maimai_dataset.chart_cache import ChartVectorStore
from maimai_dataset.integrity import file_sha256
from maimai_dataset.manifest import DatasetError, canonical_json_sha256 as digest, load_jsonl
from maimai_dataset.vocabulary import CompactVocabulary

SCHEMA = "maimai.encoder_native_audio_data.v1"
REQUEST_SCHEMA = "maimai.native_teacher_audio_requests.v1"


def resolve_paths(args):
    root = getattr(args,"dataset_root",None)
    root = Path(root).expanduser().resolve() if root is not None else None
    manifest = getattr(args,"dataset_manifest",None)
    if manifest is not None:
        manifest = Path(manifest).expanduser().resolve()
        root = root or manifest.parent
    elif root is not None:
        manifest = next((root/n for n in ("dataset-v10.json","dataset.json") if (root/n).is_file()),None)
    values = dict(DEFAULT_INPUTS) if root is not None else {}
    if manifest is not None:
        values.update({k.replace("-","_"):v for k,v in _read(manifest).get("inputs",{}).items()})
    # A new experiment never inherits learned inputs from its source dataset map.
    for name in ("latent_cache","song_context_cache","decoder_context_cache","stage_a_checkpoint","chart_encoder_checkpoint","teacher_audio_cache"):
        values.pop(name,None)
    paths = {}
    for name in (*values,"epoch_plan_cache"):
        explicit = getattr(args,name,None)
        value = explicit if explicit is not None else values.get(name)
        if value is None:
            continue
        path = Path(value).expanduser()
        paths[name] = ((root/path) if explicit is None and root is not None and not path.is_absolute() else path).resolve()
    for name in DEFAULT_INPUTS:
        if getattr(args,name,None) is not None:
            paths[name] = Path(getattr(args,name)).expanduser().resolve()
    required = ("dataset","chart_cache","spatial_view_cache","bpm_audio_manifest","epoch_plan_cache")
    missing = [name for name in required if name not in paths]
    if missing:
        raise ValueError("missing independent data inputs: "+", ".join("--"+n.replace("_","-") for n in missing))
    return paths


def _link(source,destination):
    """Reference known accepted bytes on the same filesystem without reading them."""
    source,destination = Path(source).resolve(),Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        if not os.path.samefile(source,destination):
            raise FileExistsError("input alias collision: "+str(destination))
        return
    os.link(source,destination)


def _audio_indices(paths,output,segments,variants,bpm):
    """Extend crop geometry only; immutable Mel/coordinate payloads are hard links."""
    from maimai_baseline.data.store import GEOMETRY_SCHEMA
    from maimai_baseline.time.grid import grid_coordinates,window_first_k
    from maimai_dataset.geometry import build_segment_geometry
    from maimai_dataset.timing import TimingMap
    from maimai_dataset.window import WINDOW_POSITIONS
    source_geometry_path = paths.get("interpolation_geometry")
    source_geometry = (_read(source_geometry_path/"manifest.json")
        if source_geometry_path is not None and (source_geometry_path/"manifest.json").is_file() else None)
    geometry_root,audio_root = output/"geometry",output/"audio"
    geometry_root.mkdir(); audio_root.mkdir()
    geometry_records = {}
    shared = {}
    for segment in segments:
        sid,variant,start = segment["segment_id"],variants[str(segment["record_id"])],int(segment["target_start_tick"])
        key = (variant,start)
        if key not in shared:
            relative = f"windows/{len(shared):06d}.npz"
            target = geometry_root/relative
            old = (source_geometry or {}).get("segments",{}).get(sid)
            if old is not None:
                _link(source_geometry_path/old["path"],target)
                row = dict(old,path=relative)
            else:
                timing = TimingMap.from_dict(segment["timing"])
                coords = grid_coordinates(timing,window_first_k(start),WINDOW_POSITIONS,int(bpm["variants"][variant]["num_samples"]))
                geometry = build_segment_geometry(frame_beats=(coords["grid_k"]/24).astype(np.float32),
                    frame_valid=coords["valid"],target_start_tick=start,target_end_tick=start+1536)
                target.parent.mkdir(parents=True,exist_ok=True)
                np.savez(target,**geometry)
                row = dict(path=relative,sha256=file_sha256(target))
            shared[key] = row
        geometry_records[sid] = dict(shared[key])
    geometry_manifest = dict(source_geometry or {},schema=GEOMETRY_SCHEMA,
        profile_fingerprint=_read(output/"profile/summary.json")["profile_fingerprint"],
        segment_count=len(segments),segments=geometry_records,
        reindex_origin=str(source_geometry_path) if source_geometry_path else None)
    write_json(geometry_root/"manifest.json",geometry_manifest)
    changed = deepcopy(bpm)
    changed["segments"] = {s["segment_id"]:variants[str(s["record_id"])] for s in segments}
    changed["variants"] = {key:deepcopy(bpm["variants"][key]) for key in dict.fromkeys(variants.values())}
    for i,(key,meta) in enumerate(changed["variants"].items()):
        for field in ("feature_path","coordinates_path"):
            if field in meta:
                original = paths["bpm_audio_manifest"].parent/meta[field]
                relative = f"variants/{i:05d}-"+Path(meta[field]).name
                _link(original,audio_root/relative)
                meta[field] = relative
    changed.update(geometry_root="../geometry",geometry_manifest_sha256=file_sha256(geometry_root/"manifest.json"),
        reindex_origin=str(paths["bpm_audio_manifest"]))
    write_json(audio_root/"bpm-audio-manifest.json",changed)
    return dict(bpm_audio_manifest=audio_root/"bpm-audio-manifest.json",interpolation_geometry=geometry_root)


def _physical_index(paths,output,segments):
    root = paths.get("physical_interval_cache")
    if root is None or not (root/"manifest.json").is_file():
        return {}
    manifest = _read(root/"manifest.json")
    by_record = {}
    for row in manifest["segments"]:
        record = str(row.get("record_id") or str(row["segment_id"]).rsplit(":",1)[0])
        by_record.setdefault(record,row)
    updated = []
    for segment in segments:
        record = str(segment["record_id"])
        if record not in by_record:
            raise DatasetError("Physical chart index lacks selected record: "+record)
        updated.append(dict(by_record[record],segment_id=segment["segment_id"],
            record_id=record,target_start_tick=int(segment["target_start_tick"]),target_end_tick=int(segment["target_end_tick"])))
    views = [dict(row,path=str((root/row["path"]).resolve())) for row in manifest["views"]]
    target = output/"physical-interval"
    target.mkdir()
    value = dict(manifest,segments=updated,views=views,segment_count=len(segments),
        profile_fingerprint=_read(output/"profile/summary.json")["profile_fingerprint"],
        chart_cache_manifest_sha256=file_sha256(output/"chart-cache/manifest.json"),
        spatial_cache_manifest_sha256=file_sha256(output/"spatial-views/manifest.json"),reindex_origin=str(root))
    write_json(target/"manifest.json",value)
    return dict(physical_interval_cache=target)


def _note_index(paths,output,segments,bpm_path):
    path = paths.get("note_view_manifest")
    if path is None or not path.is_file():
        return {}
    source = _read(path)
    root = output/"note-view"
    root.mkdir()
    records = set(str(s["record_id"]) for s in segments)
    charts = {key:deepcopy(meta) for key,meta in source["charts"].items() if str(key) in records}
    if set(charts) != records:
        raise DatasetError("note view lacks selected records")
    for i,meta in enumerate(charts.values()):
        if "path" in meta:
            relative = f"charts/{i:06d}.json"
            _link(path.parent/meta["path"],root/relative)
            meta["path"] = relative
    value = dict(source,charts=charts,profile=str(output/"profile"),
        profile_manifest_sha256=file_sha256(output/"profile/manifest.jsonl"),
        segments_sha256=file_sha256(output/"profile/segments.jsonl"),
        bpm_manifest=str(bpm_path),bpm_manifest_sha256=file_sha256(bpm_path),reindex_origin=str(path))
    write_json(root/"manifest.json",value)
    return dict(note_view_manifest=root/"manifest.json")


def _export_requests(profile,bpm_path,segments,variants,profile_fingerprint):
    from maimai_baseline.time.audio_end import audio_end
    from maimai_dataset.timing import TimingMap
    from maimai_dataset.window import training_window_count
    bpm = _read(bpm_path)
    requests = {}
    for variant in dict.fromkeys(variants.values()):
        info = deepcopy(bpm["variants"][variant])
        members = [s for s in segments if variants[str(s["record_id"])] == variant]
        profile_end = max(int(s["target_end_tick"]) for s in members)
        info["timing"] = info.get("timing",members[0]["timing"])
        boundary = audio_end(TimingMap.from_dict(info["timing"]),int(info["num_samples"]))
        inference_end = training_window_count(boundary.onset_tick,boundary.tail_tick)*1536
        starts = list(range(0,max(profile_end,inference_end),1536))
        info.update(starts=starts,condition_end_tick=starts[-1]+1536,profile_condition_end_tick=profile_end,
            song_ids=list(dict.fromkeys(str(s["song_id"]) for s in members)),
            audio_onset_tick=boundary.onset_tick,audio_tail_tick=boundary.tail_tick,span_beats=boundary.onset_tick/96.)
        requests[variant] = info
    stats = _read(profile/"profile.json").get("base_feature_stats")
    return dict(schema=REQUEST_SCHEMA,variants=requests,record_to_variant=variants,
        bpm_audio_manifest=str(bpm_path),profile=str(profile),
        feature_stats=str((profile/stats).resolve()) if stats is not None else None,
        profile_fingerprint=profile_fingerprint)


def ensure_export_requests(dataset_manifest, output=None):
    """Reuse an accepted complete-tail dataset; write only its missing request list."""
    dataset_manifest = Path(dataset_manifest).expanduser().resolve()
    paths = resolve_paths(SimpleNamespace(dataset_manifest=dataset_manifest))
    profile = paths["dataset"]
    fingerprint = _read(profile/"summary.json")["profile_fingerprint"]
    destination = Path(output).expanduser().resolve() if output is not None else dataset_manifest.parent/"export_requests.json"
    if destination.is_file():
        request = _read(destination)
        if (request.get("schema") != REQUEST_SCHEMA or request.get("profile_fingerprint") != fingerprint or
                Path(request["bpm_audio_manifest"]).resolve() != paths["bpm_audio_manifest"]):
            raise DatasetError("existing native export requests bind a different dataset")
        return str(destination)
    plan = EpochPlanStore(paths["epoch_plan_cache"],verify_payloads=False)
    segments = selected_segments(profile,plan)
    by_record = defaultdict(list)
    bpm = _read(paths["bpm_audio_manifest"])
    variants = {}
    for segment in segments:
        record = str(segment["record_id"])
        by_record[record].append(segment)
        variant = bpm["segments"].get(segment["segment_id"])
        if variant is None or (record in variants and variants[record] != variant):
            raise DatasetError("accepted complete-tail profile has inconsistent BPM mapping")
        variants[record] = variant
    for record,windows in by_record.items():
        windows.sort(key=lambda s:int(s["target_start_tick"]))
        end = int(windows[-1]["target_end_tick"])
        if ([int(s["target_start_tick"]) for s in windows] != list(range(0,end,1536)) or
                any(int(s.get("condition_end_tick",-1)) != end for s in windows) or
                int(windows[-1]["target_rows"]) != 1):
            raise DatasetError("PREPARED_DATA requires an accepted complete-tail profile")
    destination.parent.mkdir(parents=True,exist_ok=True)
    write_json(destination,_export_requests(profile,paths["bpm_audio_manifest"],segments,variants,fingerprint))
    return str(destination)


def prepare_data(args):
    paths = resolve_paths(args)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite encoder/native-audio data: "+str(output))
    parent = EpochPlanStore(paths["epoch_plan_cache"],verify_payloads=False)
    if parent.manifest.get("token_budget",4096) != 4096:
        raise DatasetError("independent preparation requires the source A4096 plan")
    old = selected_segments(paths["dataset"],parent)
    bpm = _read(paths["bpm_audio_manifest"])
    variants = {}
    for segment in old:
        record = str(segment["record_id"])
        key = bpm.get("segments",{}).get(segment["segment_id"],segment.get("audio_variant"))
        if key is None:
            raise DatasetError("source BPM index lacks selected window: "+segment["segment_id"])
        if record in variants and variants[record] != key:
            raise DatasetError("a chart has conflicting audio variants")
        variants[record] = key
    output.mkdir(parents=True)
    write_json(output/"manifest.json",dict(schema=SCHEMA,ready=False))
    profile = output/"profile"
    profile.mkdir()
    templates = {}
    for s in old:
        templates.setdefault(str(s["record_id"]),s)
    new = []
    for record,template in templates.items():
        windows = segment_chart(_read(paths["dataset"]/template["encoded_path"]),template)
        condition_end = windows[-1]["target_end_tick"]
        for segment in windows:
            segment.update(condition_end_tick=condition_end,audio_variant=variants[record])
        new.extend(windows)
        destination = profile/template["encoded_path"]
        if not destination.resolve().is_relative_to(profile):
            raise DatasetError("encoded chart destination escaped new profile")
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(paths["dataset"]/template["encoded_path"],destination)
    old_ids = {s["segment_id"] for s in old}
    if any(s["segment_id"] not in old_ids and int(s["target_rows"]) != 1 for s in new):
        raise DatasetError("new tail window unexpectedly owns a declaration")
    _rows(profile/"segments.jsonl",new)
    records = [r for r in load_jsonl(paths["dataset"]/"manifest.jsonl") if str(r["record_id"]) in templates]
    _rows(profile/"manifest.jsonl",records)
    shutil.copyfile(paths["dataset"]/"vocabulary.json",profile/"vocabulary.json")
    summary = dict(_read(paths["dataset"]/"summary.json"),segment_count=len(new),profile_fingerprint=digest(dict(segments=new,records=records)))
    for name,key in (("segment_split_counts","split"),("segment_source_difficulty_counts","source_difficulty_id")):
        if name in summary:
            summary[name] = {str(value):sum(str(s.get(key,s["difficulty_id"])) == str(value) for s in new)
                for value in dict.fromkeys(s.get(key,s["difficulty_id"]) for s in new)}
    if "target_row_split_counts" in summary:
        summary["target_row_split_counts"] = {"train":sum(int(s["target_rows"]) for s in new)}
    write_json(profile/"summary.json",summary)
    profile_meta = _read(paths["dataset"]/"profile.json")
    for name in ("base_dataset","base_feature_stats"):
        if name in profile_meta:
            profile_meta[name] = str((paths["dataset"]/profile_meta[name]).resolve())
    profile_meta.update(profile="encoder-native-audio",summary_sha256=file_sha256(profile/"summary.json"))
    write_json(profile/"profile.json",profile_meta)
    vocabulary = CompactVocabulary.from_dict(_read(profile/"vocabulary.json"))
    cm,sm = _read(paths["chart_cache"]/"manifest.json"),_read(paths["spatial_view_cache"]/"manifest.json")
    canonical = ChartVectorStore.load(paths["chart_cache"],profile_fingerprint=cm.get("profile_fingerprint",""),
        vocabulary_sha256=cm.get("vocabulary_sha256",""),verify_payloads=False)
    spatial = MultiViewRouteStore.load(paths["spatial_view_cache"],profile_fingerprint=sm.get("profile_fingerprint",""),
        vocabulary_sha256=sm.get("vocabulary_sha256",""),selection_sha256=sm.get("selection_sha256",""),verify_payloads=False)
    _standard_vectors(output,new,old,canonical,spatial,vocabulary,summary["profile_fingerprint"])
    _window_cache(output,new,old,spatial,vocabulary,variants,bpm)
    batch_counts = _plans(parent,output,old,new,variants)
    mapped = dict(paths,dataset=profile,chart_cache=output/"chart-cache",spatial_view_cache=output/"spatial-views",
        window_cache=output/"window-cache",epoch_plan_cache=output/"a-plan",whole_song_plan=output/"b-plan")
    mapped.update(_audio_indices(paths,output,new,variants,bpm))
    mapped.update(_physical_index(paths,output,new))
    mapped.update(_note_index(paths,output,new,mapped["bpm_audio_manifest"]))
    request = _export_requests(profile,mapped["bpm_audio_manifest"],new,variants,summary["profile_fingerprint"])
    requests = request["variants"]
    write_json(output/"export_requests.json",request)
    windows = WindowCache(output/"window-cache")
    for i,segment in enumerate(new):
        for view in (0,1):
            item = windows.get(i,view,enabled=False)
            if int(item["target_ids"]["event"].ne(-100).sum()) != int(segment["target_rows"]):
                raise DatasetError("new window loader supervision coverage mismatch")
    coverage = dict(charts=len(templates),old_windows=len(old),windows=len(new),new_windows=len(new)-len(old),
        view_windows=2*len(new),epochs=parent.available_epochs,audio_variants=len(requests),
        native_audio_windows=sum(len(r["starts"]) for r in requests.values()),condition_window_blocks=8*len(new))
    result = dict(schema=SCHEMA,ready=True,inputs={k:str(v) for k,v in mapped.items()},coverage=coverage,
        batch_counts=batch_counts,updates=dict(e60=sum(batch_counts["a-plan"][:60]) if parent.available_epochs>=60 else None,
            a90=sum(batch_counts["a-plan"][:90]) if parent.available_epochs>=90 else None,
            b20=sum(batch_counts["b-plan"][90:110]) if parent.available_epochs>=110 else None,
            b_horizon50=sum(batch_counts["b-plan"][90:140]) if parent.available_epochs>=140 else None),
        export_requests=str(output/"export_requests.json"),learned_inputs_required=False)
    write_json(output/"dataset.json",result)
    write_json(output/"manifest.json",result)
    return result
