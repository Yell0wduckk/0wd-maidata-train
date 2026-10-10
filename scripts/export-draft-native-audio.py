"""Export full-A90 native teacher outputs through its own source3d0bc68 API.

This adapter runs in an independent process and imports the external original
network. Existing native windows are reused without re-encoding them.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

SOURCE_COMMIT = "3d0bc6829bdb877be04abbc7bb05f07f516bc49a"
CHECKPOINT_RECORDED_SHA256 = "ebe72d2c47d7629badd678e98138964423bb7f782618cdc6cfa5ce8a0183cb77"
CHECKPOINT_RUN = "maidata-full-mmoe-decoders-a90-b50-3d0bc68-20260929"


def affine_teacher_features(feature, valid, source_stats, teacher_stats):
    """Change only an already accepted affine standardization; never fit stats."""
    for stats in (source_stats,teacher_stats):
        for name in ("sample_rate","n_fft","n_mels","f_min","f_max"):
            if stats.get(name) != teacher_stats.get(name):
                raise ValueError("teacher Mel transform definition differs: "+name)
    def arrays(stats):
        mean,std = (np.asarray(stats[k],np.float32) for k in ("mean","std"))
        if mean.shape != (128,) or std.shape != (128,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std<=0):
            raise ValueError("teacher normalization requires original 128-dimensional statistics")
        return mean,std
    old_mean,old_std = arrays(source_stats)
    new_mean,new_std = arrays(teacher_stats)
    if np.array_equal(old_mean,new_mean) and np.array_equal(old_std,new_std):
        return feature, "same-original-statistics"
    value = ((feature.astype(np.float64)*old_std[:,None]+old_mean[:,None]-new_mean[:,None])/new_std[:,None]).astype(np.float32)
    value[:,~valid] = 0.
    if not np.isfinite(value).all():
        raise ValueError("nonfinite affine-converted teacher Mel")
    return value,"accepted-affine-source-stats-to-original-teacher-stats-fp64-to-fp32-v1"


def extend_song_fusion_geometry(row, last_start):
    """Extend only endpoint-clamped lookup coverage, leaving original rows intact."""
    needed = max(len(row["fusion_left"]), (int(last_start)//96+22)*2)
    for name in ("fusion_left","fusion_right","fusion_weight"):
        value = row[name]
        row[name] = np.pad(value,(0,needed-len(value)),mode="edge")
    return row


def _to(value,device):
    if isinstance(value,dict):
        return {k:_to(v,device) for k,v in value.items()}
    return value.to(device) if isinstance(value,torch.Tensor) else value


def merge_native_bundles(existing, old_starts, generated, new_starts):
    """Reuse accepted prefix tensors; only newly encoded windows are supplied."""
    old_index = {int(start): index for index, start in enumerate(old_starts)}
    new_index = {int(start): index for index, start in enumerate(new_starts)}
    if old_index.keys() & new_index.keys():
        raise ValueError("native incremental production duplicated an accepted window")
    starts = sorted(old_index.keys() | new_index.keys())
    if starts != list(range(0, starts[-1] + 1, 1536)):
        raise ValueError("native incremental coverage is not a contiguous lattice")
    def combine(name, geometry=False):
        old = existing["interpolation_geometry"][name] if geometry else existing[name]
        new = generated["interpolation_geometry"][name] if geometry else generated[name]
        return torch.stack([old[old_index[start]] if start in old_index else new[new_index[start]] for start in starts])
    result = {name: combine(name) for name in generated if name != "interpolation_geometry"}
    result["interpolation_geometry"] = {name: combine(name, True) for name in generated["interpolation_geometry"]}
    return starts, result


def producer_metadata(payload, checkpoint, teacher_stats, audio_input, *, teacher_run=None, tail_compat=None):
    epoch, step = payload.get('epoch'), payload.get('step')
    if not isinstance(epoch, int) or epoch < 0 or not isinstance(step, int) or step < 1:
        raise ValueError('native teacher requires completed training checkpoint metadata')
    original = epoch == 89 and step == 27892 and teacher_run is None
    producer = dict(kind='original-full-A90-native22' if original else 'full-A-finetuned-native22',
        run=CHECKPOINT_RUN if original else str(teacher_run or Path(checkpoint).resolve().parent),
        source_commit=SOURCE_COMMIT, source_identity=payload['source_identity'], epoch=epoch, step=step,
        runtime_version=payload.get('runtime_version'), checkpoint=str(Path(checkpoint).resolve()),
        original_feature_stats=teacher_stats, original_audio_input=audio_input,
        scope='old-eval-song-audio-once-per-variant-and-independent-native22-windows',
        fusion_geometry_extension='repeat-original-endpoint-lookup-only')
    if original:
        producer['checkpoint_recorded_sha256'] = CHECKPOINT_RECORDED_SHA256
    if tail_compat is not None:
        producer['tail_compat'] = str(Path(tail_compat).resolve())
        producer['fusion_geometry_extension'] = 'original-endpoint-lookup-with-finetune-invalid-tail-zero-mask'
    return producer


def export(args):
    source = Path(args.source_snapshot).expanduser().resolve()
    if "maimai_baseline" in sys.modules:
        raise RuntimeError("run the old-native adapter in a fresh Python process")
    if not (source/"src/maimai_baseline/checkpoint.py").is_file():
        raise ValueError("missing bound old teacher source snapshot")
    sys.path.insert(0,str(source/"src"))
    from maimai_baseline.checkpoint import load_baseline_checkpoint
    from maimai_baseline.data.audio import checkpoint_audio_input,policy_contract
    from maimai_baseline.data.song_audio import raw_song,pack as pack_song
    from maimai_baseline.data.store import window_arrays
    from maimai_baseline.time.grid import SAMPLING,INTEGRATION
    from maimai_dataset.timing import TimingMap
    tail_compat = getattr(args, 'tail_compat', None)
    if tail_compat is not None:
        spec = importlib.util.spec_from_file_location('teacher_tail_compat', tail_compat)
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
        adapter.apply()
    # Linux-produced checkpoint paths are metadata when exporting on Windows.
    if sys.platform == "win32":
        import pathlib
        pathlib.PosixPath = pathlib.WindowsPath
    model,payload = load_baseline_checkpoint(Path(args.checkpoint),route_artifact=Path(args.route_artifact),
        device=args.device,strict_source=False,verify_artifact_payloads=False)
    if model.config.d_model != 256 or model.config.chart_style != "full-song-z128-no-global-v1":
        raise ValueError("full native teacher architecture differs")
    model.eval().requires_grad_(False)
    request_path = Path(args.requests).expanduser().resolve()
    request = json.loads(request_path.read_text("utf-8-sig"))
    if request.get("schema") != "maimai.native_teacher_audio_requests.v1":
        raise ValueError("native teacher request schema differs")
    bpm_path = Path(request["bpm_audio_manifest"])
    bpm = json.loads(bpm_path.read_text("utf-8-sig"))
    audio_input = checkpoint_audio_input(payload)
    policy = "legacy" if audio_input is None else audio_input["policy"]
    if bpm.get("base_policy") != policy or bpm.get("sampling") != SAMPLING or bpm.get("integration") != INTEGRATION:
        raise ValueError("accepted Mel policy/sampling differs from original teacher preprocessing")
    if audio_input is not None and audio_input.get("contract") != policy_contract(policy):
        raise ValueError("original teacher nonlinear preprocessing contract differs")
    source_stats = json.loads(Path(request["feature_stats"]).read_text("utf-8-sig"))
    teacher_stats = payload["feature_stats"]
    writer_spec = importlib.util.spec_from_file_location("native_cache_interchange",Path(args.cache_writer))
    writer_module = importlib.util.module_from_spec(writer_spec)
    writer_spec.loader.exec_module(writer_module)
    producer = producer_metadata(payload, args.checkpoint, teacher_stats, audio_input,
        teacher_run=getattr(args, 'teacher_run', None), tail_compat=tail_compat)
    reuse_root = getattr(args, "reuse_cache", None)
    existing = writer_module.TeacherAudioCache(reuse_root) if reuse_root else None
    if existing is not None and writer_module.draft_producer_identity(existing) != writer_module.draft_producer_identity(dict(producer=producer, contract=writer_module.CONTRACT)):
        raise ValueError("native reuse producer/representation differs")
    missing = {}
    for variant, metadata in request["variants"].items():
        previous = None if existing is None else existing.manifest["variants"].get(variant)
        if previous is not None and (previous["timing"] != metadata["timing"] or int(previous["num_samples"]) != int(metadata["num_samples"])):
            raise ValueError("native reuse audio/timing geometry differs")
        accepted = set() if previous is None else set(map(int, previous["starts"]))
        missing[variant] = [int(start) for start in metadata["starts"] if int(start) not in accepted]
    if existing is not None and not any(missing.values()):
        return dict(schema=existing.manifest["schema"], identity=existing.manifest["identity"],
            output=str(existing.root), reused=True, song_audio_encoder=0, audio_encoder=0, windows=0)
    writer = writer_module.TeacherAudioWriter(args.output,producer=producer,
        record_to_variant=request["record_to_variant"],bindings=dict(profile_fingerprint=request["profile_fingerprint"],
        requests=str(request_path),bpm_preparation_identity=bpm.get("preparation_identity")))
    calls = dict(song_audio_encoder=0,audio_encoder=0,variants=0,windows=0,reused_windows=0)
    changed_variants = []
    with torch.no_grad():
        for variant,metadata in request["variants"].items():
            previous = None if existing is None else existing.manifest["variants"].get(variant)
            if not missing[variant]:
                relative = f"reused-{len(writer.manifest['variants']):05d}.npz"
                os.link(existing.root / previous["file"], writer.root / relative)
                writer.manifest["variants"][variant] = dict(previous, file=relative)
                writer._write()
                calls["reused_windows"] += len(previous["starts"])
                continue
            info = bpm["variants"][variant]
            feature = np.load(bpm_path.parent/info["feature_path"],mmap_mode="r",allow_pickle=False)
            with np.load(bpm_path.parent/info["coordinates_path"],allow_pickle=False) as archive:
                coordinates = {k:archive[k].copy() for k in archive.files}
            feature,conversion = affine_teacher_features(feature,coordinates["valid"],source_stats,teacher_stats)
            timing = TimingMap.from_dict(metadata["timing"])
            starts = missing[variant]
            raw = extend_song_fusion_geometry(raw_song(feature,coordinates,timing,int(info["num_samples"])),starts[-1])
            song_row = {k:torch.from_numpy(v) for k,v in raw.items()}
            song = model.song_audio_encoder(**_to(pack_song([song_row],[0]),args.device))
            calls["song_audio_encoder"] += 1
            fields = {k:[] for k in ("p0","p0_valid","p0_beats","c0","c0_valid","c0_beats","c1","c1_valid","c1_beats")}
            geometries = {k:[] for k in writer_module.GEOMETRY}
            for offset in range(0,len(starts),args.window_batch_size):
                selected = starts[offset:offset+args.window_batch_size]
                rows = [window_arrays(feature,coordinates,timing,start,int(info["num_samples"])) for start in selected]
                mel,valid,beats,bpms = [torch.from_numpy(np.stack([r[i] for r in rows])).to(args.device) for i in range(4)]
                geometry = {k:torch.from_numpy(np.stack([r[4][k] for r in rows])).to(args.device) for k in rows[0][4]}
                context = model.audio_encoder(mel,~valid,beats,bpms,start_tick=torch.tensor(selected,device=args.device),
                    interpolation_geometry=geometry,song_audio=replace(song,window_to_song=torch.zeros(len(selected),dtype=torch.long,device=args.device)))
                calls["audio_encoder"] += 1
                for name,index in (("p0",-1),("c0",0),("c1",1)):
                    values,masks,coords = ((context.memories,context.padding_masks,context.beats) if name=="p0" else
                        (context.fine_memories,context.fine_padding_masks,context.fine_beats))
                    fields[name].append(values[index].float().cpu())
                    fields[name+"_valid"].append((~masks[index]).cpu())
                    fields[name+"_beats"].append(coords[index].float().cpu())
                for key in geometries:
                    geometries[key].append(geometry[key].cpu())
            bundle = {k:torch.cat(v) for k,v in fields.items()}
            bundle["interpolation_geometry"] = {k:torch.cat(v) for k,v in geometries.items()}
            complete_starts = starts
            if previous is not None:
                complete_starts, bundle = merge_native_bundles(existing.variant(variant), previous["starts"], bundle, starts)
                calls["reused_windows"] += len(previous["starts"])
            writer.append(variant,dict(metadata,starts=complete_starts,stats_conversion=conversion),bundle)
            changed_variants.append(variant)
            calls["variants"] += 1; calls["windows"] += len(starts)
            print(json.dumps(dict(stage="native-teacher-cache",variant=variant,**calls)),flush=True)
    writer.manifest["production_calls"] = calls
    result = writer.finish()
    # Accept only newly produced cache through the normal actual reader once.
    cache = writer_module.TeacherAudioCache(args.output)
    loaded = sum(len(cache.variant(v)["p0"]) for v in changed_variants)
    expected = sum(len(result["variants"][v]["starts"]) for v in changed_variants)
    if loaded != expected:
        raise ValueError("new native teacher loader coverage differs")
    return dict(schema=result["schema"],identity=result["identity"],output=str(Path(args.output).resolve()),**calls)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-snapshot","checkpoint","route-artifact","cache-writer","requests","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--reuse-cache",type=Path,help="accepted native cache; encode only missing windows")
    parser.add_argument('--teacher-run', help='actual fine-tuned teacher run provenance')
    parser.add_argument('--tail-compat', type=Path, help='bound fine-tuning invalid-tail adapter')
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--window-batch-size",type=int,default=16)
    args = parser.parse_args()
    if args.window_batch_size<1:
        parser.error("--window-batch-size must be positive")
    torch.set_num_threads(1)
    print(json.dumps(export(args),allow_nan=False),flush=True)


if __name__ == "__main__":
    main()
