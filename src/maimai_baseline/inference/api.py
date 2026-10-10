"""Inference input validation and artifact orchestration."""
from __future__ import annotations

from collections import Counter
from maimai_baseline.checkpoint import load_baseline_inference_checkpoint
from maimai_baseline.contract import MAX_GENERATED_MOMENT_ROWS
from maimai_baseline.inference.generation import HISTORY_TICKS, generate_baseline_rows
from maimai_baseline.inference.materialize import _json, _sha256
from maimai_baseline.time.moment import LINEAR_MS, generation_clock
from maimai_baseline.source_identity import baseline_source_identity
from maimai_dataset.codec import restore_v1_rows as restore_v2_test1_rows
from maimai_dataset.encoding import condition_difficulty_id
from maimai_dataset.materialize import materialize_chart_ir, normalize_each_semantics
from maimai_dataset.timing import TimingMap
from pathlib import Path
from typing import Any
from maimai_baseline.runtime_versions import require_compatible, current_version
from maimai_baseline.radar_contract import RADAR_ID, is_radar
from maimai_baseline.style_contract import is_frozen, is_native, is_draft
import hashlib
import json
import numpy as np
import shutil
import time
import torch


DIFFICULTIES = {2: "BAS", 3: "ADV", 4: "EXP", 5: "MAS", 6: "REM"}


def _device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    return torch.device("cuda" if torch.cuda.is_available() and requested != "cpu" else "cpu")


def _safe_output(path: Path, *, force: bool) -> None:
    marker = path / ".maimai_baseline_inference_output"
    if path.exists() and any(path.iterdir()):
        if not force:
            raise ValueError(f"baseline inference output is not empty: {path}")
        if not marker.is_file():
            raise ValueError(f"refusing to replace unmarked output directory: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    marker.write_text("maimai.baseline_inference.v1\n", encoding="utf-8")


def _inference_result(
    *,
    checkpoint: Path,
    route_artifact: Path,
    profile_root: Path,
    song_id: str,
    timing_path: Path,
    level: float,
    cabinet: str,
    difficulty_id: int,
    device: str,
    seed: int,
    temperature: float,
    max_rows_per_window: int,
    allow_truncated: bool,
    audio_path: Path | None = None,
    audio_cache: Path | None = None,
    bpm_audio_manifest: Path | None = None,
    teacher_audio_cache: Path | None = None,
    physical_hard_mask: bool = True,
    physical_path_catalog: Path | None = None,
    miacode_slide_check: bool = True,
    trace: bool = False,
    local_seed: int | None = None,
    style_mode: str | None = None,
    style_reference: Path | None = None,
    latent_temperature: float | None = None,
    mixture_temperature: float | None = None,
    stream=None,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    TimingMap,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    if miacode_slide_check and not physical_hard_mask:
        raise ValueError("--miacode-slide-check requires --physical-hard-mask")
    if physical_hard_mask and physical_path_catalog is None:
        raise ValueError("--physical-hard-mask requires --physical-path-catalog")
    target_device = _device(device)
    model, payload, artifact_identity, _profile_summary = (
        load_baseline_inference_checkpoint(
            checkpoint,
            route_artifact=route_artifact,
            profile_root=profile_root,
            device=target_device,
        )
    )
    model.eval()
    if physical_hard_mask:
        from maimai_baseline.physical.catalog import CompactPathCatalog
        catalog = CompactPathCatalog.load(physical_path_catalog, model.vocabulary)
        if dict(catalog.identity) != payload.get("physical_inputs", {}).get("path_catalog"):
            raise ValueError("inference Physical path catalog/checkpoint identity mismatch")
    from maimai_baseline.inference.style import sampling_options
    radar = is_radar(model.config)
    frozen = is_frozen(model.config)
    native = is_native(model.config)
    draft = is_draft(model.config)
    style_mode = ('draft' if draft else 'ar' if radar or frozen else 'disabled') if style_mode is None else style_mode
    sampling_options(style_mode, latent_temperature, mixture_temperature, local_seed=local_seed, reference=style_reference, radar=radar, frozen=frozen, draft=draft)
    if radar and style_mode == 'ar' and (
            not isinstance(payload.get('radar_prior'), dict) or payload['radar_prior'].get('stage') != 'A'
            or not bool(model.style_prior.ready)):
        raise ValueError('MajRadar AR generation requires a trained A radar prior')
    if frozen and style_mode == 'ar' and (not isinstance(payload.get('ar_prior'), dict)
            or not bool(model.style_prior.ready)):
        raise ValueError('frozen-mu AR generation requires a trained B latent prior')
    reference = None
    timing = TimingMap.from_dict(
        json.loads(Path(timing_path).read_text(encoding="utf-8-sig"))
    )
    teacher = None
    teacher_variant = None
    if native:
        if teacher_audio_cache is None:
            raise ValueError('native-audio inference requires --teacher-audio-cache')
        from maimai_baseline.data.teacher_audio import TeacherAudioCache, producer_identity, draft_producer_identity
        teacher = TeacherAudioCache(teacher_audio_cache)
        binding = payload.get('baseline_manifest', {}).get('teacher_audio')
        producer_contract = draft_producer_identity if draft else producer_identity
        if not isinstance(binding, dict) or producer_contract(binding) != producer_contract(teacher.identity):
            raise ValueError('native inference teacher audio producer differs from frozen A')
        teacher_variant = teacher.resolve_variant(str(song_id), timing)
        info = teacher.manifest['variants'][teacher_variant]
        feature = None
        bpm_audio = dict(num_samples=int(info['num_samples']))
        prepared_audio = dict(policy='native-teacher-window-cache', teacher_audio=teacher.identity,
            variant=teacher_variant, audio_memory='native-window-center-stitched-P0',
            router_audio='owning-native-window-P0', candidate_source='new-A-projection-of-cached-native-C0-C1')
    else:
        if teacher_audio_cache is not None:
            raise ValueError('continuous inference cannot consume native teacher audio')
        feature, prepared_audio, bpm_audio = _continuous_audio(payload, profile_root, song_id, timing,
            level, audio_path, audio_cache, bpm_audio_manifest)
    from maimai_baseline.time.audio_end import audio_end
    boundary = audio_end(timing, int(bpm_audio["num_samples"]))
    physical_state = None
    if physical_hard_mask:
        from maimai_baseline.physical.inference_state import PhysicalInferenceState
        state_type = PhysicalInferenceState
        if miacode_slide_check:
            from maimai_baseline.physical.miacode_inference import MiaCodeInferenceState
            state_type = MiaCodeInferenceState
        physical_state = state_type(timing, catalog, model.vocabulary,
            boundary.onset_tick, audio_tail_tick=boundary.tail_tick, audio_end_seconds=boundary.seconds)
    payload = dict(payload)
    payload["_prepared_audio"] = dict(prepared_audio, num_samples=int(bpm_audio["num_samples"]),
        sample_rate=24000, audio_end_seconds=boundary.seconds,
        audio_onset_tick=boundary.onset_tick, audio_tail_tick=boundary.tail_tick)
    if stream:
        stream("init", runtime_version=current_version(), checkpoint_runtime_version=require_compatible(payload, "inference"), difficulty=difficulty_id,
            level=level, cabinet=cabinet, song_id=song_id, audio_seconds=boundary.seconds)
    rows, traces, windows = generate_baseline_rows(
        model=model,
        feature=feature,
        timing=timing,
        level=level,
        cabinet=cabinet,
        difficulty_id=difficulty_id,
        seed=seed,
        local_seed=local_seed, style_mode=style_mode, style_reference=reference,
        latent_temperature=latent_temperature, mixture_temperature=mixture_temperature,
        temperature=temperature,
        device=target_device,
        max_rows_per_window=max_rows_per_window,
        allow_truncated=allow_truncated,
        physical_state=physical_state,
        bpm_audio=bpm_audio,
        trace=trace,
        stream=stream,
        teacher_audio_cache=teacher, teacher_audio_variant=teacher_variant,
    )
    if physical_state is not None and windows:
        from maimai_baseline.inference.physical import summarize
        windows[-1]["physical_summary"] = {**physical_state.report(), "probability_statistics_collected": bool(trace), **(summarize(traces) if trace else {"probability_statistics_collected": False})}
        if miacode_slide_check and trace:
            from maimai_baseline.inference.miacode import summarize as summarize_miacode
            windows[-1]["physical_summary"]["miacode"] = summarize_miacode(traces)
    public_rows = [dict(row) for row in rows]
    for row in public_rows:
        if "route" in row or "route_id" in row or "delta_10ms" in row or "delta_ms" in row:
            raise RuntimeError("internal model time field leaked into public rows")
    from maimai_dataset.modifiers import materialize_row
    from maimai_baseline.time.audio_end import export_end_beat
    materialized_rows = [materialize_row(row) for row in public_rows]
    render_end = export_end_beat(materialized_rows, timing, boundary.seconds)
    document = normalize_each_semantics(
        materialize_chart_ir(
            restore_v2_test1_rows(materialized_rows),
            timing=timing,
            difficulty=DIFFICULTIES[int(difficulty_id)],
            render_end_beat=render_end,
        )
    )
    return payload, artifact_identity, timing, public_rows, traces, windows, document


def _continuous_audio(payload, profile_root, song_id, timing, level, audio_path, audio_cache, bpm_audio_manifest):
    from maimai_baseline.data.audio import checkpoint_audio_input, _read
    from maimai_baseline.data.store import prepare_raw_bpm_audio, BpmFeatureStore
    from maimai_baseline.time.grid import timing_identity
    audio_input = checkpoint_audio_input(payload)
    policy = "legacy" if audio_input is None else audio_input["policy"]
    profile_record = _read(profile_root / "profile.json")
    stats = payload.get("feature_stats") or _read((profile_root / profile_record["base_feature_stats"]).resolve())
    if audio_path is not None:
        if bpm_audio_manifest is not None:
            raise ValueError("raw audio and BPM feature manifest are mutually exclusive")
        feature, prepared_audio, bpm_audio = prepare_raw_bpm_audio(audio_path, stats=stats, policy=policy,
            level=level, timing=timing, cache_root=audio_cache)
    else:
        if bpm_audio_manifest is None:
            raise ValueError("BPM inference requires --audio or --bpm-audio-manifest")
        m = _read(bpm_audio_manifest)
        store = BpmFeatureStore(bpm_audio_manifest, profile=profile_root, stats=stats,
            song_ids=set(m["song_ids"]), selection_sha256=m["selection_sha256"],
            geometry_root=(bpm_audio_manifest.parent / m.get("geometry_root", "geometry")).resolve())
        if store.identity != payload["bpm_input"]["artifacts"]:
            raise ValueError("BPM inference artifact differs from checkpoint")
        matches = [key for key, v in store.variants.items() if str(song_id) in v["song_ids"] and
                   v["identity"]["timing"] == timing_identity(timing)]
        if len(matches) != 1:
            raise ValueError("BPM inference song/timing variant is missing or ambiguous")
        key = matches[0]
        v = store.variants[key]
        coords = store._coordinates[key]
        feature = np.array(store._features[key], copy=True)
        bpm_audio = {"coordinates": coords, "num_samples": v["num_samples"]}
        prepared_audio = {**v["audio"], "policy": policy, "level": level, "level_gain_db": 0.0,
                          "source_sha256": v["identity"]["original_sha256"], "artifacts": store.identity,
                          "base_feature_sha256": v["feature_sha256"],
                          "feature_sha256": hashlib.sha256(feature.tobytes()).hexdigest()}
    return feature, prepared_audio, bpm_audio


def generate_baseline_inference(
    *,
    checkpoint: Path,
    route_artifact: Path,
    profile_root: Path,
    song_id: str,
    timing_path: Path,
    output: Path,
    level: float,
    cabinet: str,
    difficulty_id: int,
    force: bool,
    device: str,
    seed: int,
    temperature: float,
    max_rows_per_window: int = 512,
    allow_truncated: bool = False,
    audio_path: Path | None = None,
    audio_cache: Path | None = None,
    bpm_audio_manifest: Path | None = None,
    teacher_audio_cache: Path | None = None,
    physical_hard_mask: bool = True,
    physical_path_catalog: Path | None = None,
    miacode_slide_check: bool = True,
    trace: bool = False,
    local_seed: int | None = None,
    style_mode: str | None = None,
    style_reference: Path | None = None,
    latent_temperature: float | None = None,
    mixture_temperature: float | None = None,
    stream=None,
) -> dict[str, Any]:
    torch.set_num_threads(1)
    generation_started = time.perf_counter()
    output = Path(output).resolve()
    _safe_output(output, force=force)
    (
        payload,
        artifact_identity,
        _timing,
        rows,
        traces,
        windows,
        document,
    ) = _inference_result(
        checkpoint=Path(checkpoint),
        route_artifact=Path(route_artifact),
        profile_root=Path(profile_root),
        song_id=str(song_id),
        timing_path=Path(timing_path),
        level=level,
        cabinet=cabinet,
        difficulty_id=difficulty_id,
        device=device,
        seed=seed,
        local_seed=local_seed, style_mode=style_mode, style_reference=style_reference,
        latent_temperature=latent_temperature, mixture_temperature=mixture_temperature,
        temperature=temperature,
        max_rows_per_window=max_rows_per_window,
        allow_truncated=allow_truncated,
        audio_path=audio_path, audio_cache=audio_cache,
        bpm_audio_manifest=bpm_audio_manifest,
        teacher_audio_cache=teacher_audio_cache,
        physical_hard_mask=physical_hard_mask,
        physical_path_catalog=physical_path_catalog,
        miacode_slide_check=miacode_slide_check,
        trace=trace,
        stream=stream,
    )
    radar = payload.get('model_config', {}).get('chart_style') == RADAR_ID
    frozen = is_frozen(payload.get('model_config', {}).get('chart_style'))
    draft = is_draft(payload.get('model_config', {}).get('chart_style'))
    style_mode = ('draft' if draft else 'ar' if radar or frozen else 'disabled') if style_mode is None else style_mode
    _json(output / "model.rows.json", rows)
    _json(output / "model.ir.json", document)
    if trace:
        with (output / "prediction-trace.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as stream:
            for trace_record in traces:
                stream.write(json.dumps(trace_record, ensure_ascii=False, sort_keys=True) + "\n")

    route_counts = Counter(windows[-1].get("final_distribution", {}).get("route_counts", {}) if windows else {})
    architecture = str(payload["architecture"])
    provenance = {
        "schema": "maimai.baseline_generation.v1",
        "runtime_version": current_version(),
        "checkpoint_runtime_version": require_compatible(payload, "inference"),
        "execution_version": current_version(),
        "physical_hard_mask": {
            "enabled": physical_hard_mask,
            **(windows[-1].get("physical_summary", {}) if windows and physical_hard_mask else {}),
        },
        "trace_enabled": bool(trace),
        "operation_summary": windows[-1].get("operation_summary", {}) if windows else {},
        "physical_failures": windows[-1].get("physical_failures", []) if windows else [],
        "final_distribution": windows[-1].get("final_distribution", {}) if windows else {},
        "generation_seconds": time.perf_counter() - generation_started,
        "architecture": architecture,
        "base_model_architecture": payload.get("base_model_architecture"),
        "baseline_contract": dict(payload["baseline_contract"]),
        "bpm_input": payload.get("bpm_input"),
        **({"moment_distribution": payload["moment_distribution"]} if "moment_distribution" in payload else {}),
        **({"duration_distribution": payload["duration_distribution"]} if "duration_distribution" in payload else {}),
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": _sha256(Path(checkpoint)),
        "checkpoint_kind": payload.get("checkpoint_kind"),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "checkpoint_step": int(payload.get("step", -1)),
        "parameter_count": payload["baseline_contract"]["parameter_count"],
        "audio_cross_attention": payload["model_config"]["audio_cross_attention"],
        "audio_cross_attention_identity": payload["audio_cross_attention_identity"],
        "encoder_capacity": payload["encoder_capacity"],
        "encoder_capacity_initialization": payload["initialization"]["encoder_capacity_initialization"],
        "route_artifact": artifact_identity,
        "checkpoint_source": payload.get("source_identity"),
        "inference_source": baseline_source_identity(),
        "profile_fingerprint": payload["profile_fingerprint"],
        "song_id": str(song_id),
        "source_difficulty_id": int(difficulty_id),
        "condition_difficulty_id": condition_difficulty_id(difficulty_id),
        "level": float(level),
        "audio_input": payload.get("audio_input"),
        "audio_preparation": payload.get("_prepared_audio"),
        "timing_sha256": _sha256(Path(timing_path)),
        "cabinet": cabinet,
        "history": {
            "policy": "raw-prefix-rope-kv-v1",
            "beats": 4,
            "ticks": HISTORY_TICKS,
            "target_beats": 16,
            "audio_context_beats": 22,
            "audio_kv_cache": True,
            "raw_prefix_kv_cache": True,
            "history_delta": ("recomputed-ms-from-exact-chart-tick-anchors" if payload.get("bpm_input") is not None
                              else "recomputed-from-projected-tick-anchors"),
            "window_clock_reset": "exact-start-tick-and-second",
            "max_generated_moment_rows": MAX_GENERATED_MOMENT_ROWS,
        },
        "sampling": {
            "seed": int(seed),
            "local_seed": local_seed,
            "radar_seed": (int(seed) if local_seed is None else int(local_seed)) if radar and style_mode == 'ar' else None,
            "style_mode": style_mode if payload.get("model_config", {}).get("chart_style", "none") != "none" else "none",
            "latent_temperature": (1. if latent_temperature is None else latent_temperature) if style_mode == "ar" else None,
            "radar_temperature": (1. if latent_temperature is None else latent_temperature) if radar and style_mode == 'ar' else None,
            "mixture_temperature": (1. if mixture_temperature is None else mixture_temperature) if style_mode == "ar" else None,
            "mixture_components": 4 if style_mode == "ar" else None,
            "style_reference_sha256": _sha256(style_reference) if style_reference else None,
            "temperature": float(temperature),
            "top_p": 1.0,
            "top_k": 0,
            "deterministic": float(temperature) == 0.0 and (style_mode != "ar" or (latent_temperature == 0. and mixture_temperature == 0.)),
        },
        "resampling_backend": "indexed-v1",
        "generation_clock": generation_clock((payload.get("moment_distribution") or {}).get("schema", LINEAR_MS)),
        "official_rows_used": False,
        "chart_style": payload.get("chart_style"),
        "ar_prior": payload.get("ar_prior"),
        "radar_prior": payload.get("radar_prior"),
        "frequency_pooling": payload["baseline_contract"]["frequency_pooling"],
        "frozen_a_source": payload.get("frozen_a_source"),
        "generated_rows": len(rows),
        "generated_moments": sum(row.get("event") == "MOMENT" for row in rows),
        "route_usage": {
            "total": sum(route_counts.values()),
            "counts": dict(sorted(route_counts.items())),
        },
        "windows": windows,
        "truncated": any(window["status"] != "complete" for window in windows),
        "outputs": {
            "rows": "model.rows.json",
            "model_ir": "model.ir.json",
            "trace": "prediction-trace.jsonl" if trace else None,
        },
    }
    _json(output / "generation.json", provenance)
    return provenance
