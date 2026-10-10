"""Cross-window generation and rollback orchestration."""
from __future__ import annotations

from fractions import Fraction
from maimai_baseline.contract import (
    DELTA_FIELD,
    HISTORY_TICKS as WINDOW_HISTORY_TICKS,
    MAX_GENERATED_MOMENT_ROWS,
)
from maimai_baseline.data.route import Route71Codec
from maimai_baseline.inference.decoding import encode_prefix_rows, decode_prefix
from maimai_baseline.inference.reporting import (
    begin_row,
    count,
    distribution,
    generation_session,
    route_of,
    trace_enabled,
)
from maimai_baseline.inference.sampling import _choose_route_row
from maimai_baseline.model.baseline import BaselineTransformer
from maimai_baseline.time.coordinates import chart_tick_to_second
from maimai_dataset.encoding import CHART_TICKS_PER_BEAT, condition_difficulty_id
from maimai_dataset.geometry import build_segment_geometry
from maimai_dataset.timing import TimingMap
from maimai_dataset.window import TARGET_TICKS, AUDIO_BEATS_BEFORE, AUDIO_BEATS_AFTER
from typing import Any, Mapping, Sequence
import math
import numpy as np
import torch


# Shared full-song Route-71 loop


# This must match the raw prefix data transform exactly.
# Four history beats and all audio/target coordinates use 96 chart ticks/beat.
HISTORY_TICKS = WINDOW_HISTORY_TICKS


AUDIO_PREFIX_TICKS = AUDIO_BEATS_BEFORE * CHART_TICKS_PER_BEAT
AUDIO_SUFFIX_TICKS = AUDIO_BEATS_AFTER * CHART_TICKS_PER_BEAT


GENERATION_SCHEMA_SUFFIX = "inference.v1"


def build_interpolation_geometry(
    *,
    frame_beats: torch.Tensor,
    audio_padding: torch.Tensor,
    target_start_tick: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    geometry = build_segment_geometry(
        frame_beats=frame_beats[0].detach().to("cpu", torch.float32).numpy(),
        frame_valid=(~audio_padding[0].detach().to("cpu")).numpy(),
        target_start_tick=int(target_start_tick),
        target_end_tick=int(target_start_tick) + TARGET_TICKS,
    )
    return {
        name: torch.from_numpy(np.asarray(value)).unsqueeze(0).to(device)
        for name, value in geometry.items()
    }


def _append_window_chunks(
    rows: Sequence[Mapping[str, Any]], *, target_start_tick: int
) -> list[tuple[int, list[dict[str, Any]]]]:
    result: list[tuple[int, list[dict[str, Any]]]] = []
    current = target_start_tick
    index = 0
    while index < len(rows) and rows[index].get("event") != "SEGMENT_END":
        if rows[index].get("event") != "MOMENT":
            raise ValueError("baseline full-song window must begin each group with MOMENT")
        current += int(rows[index]["delta_tick"])
        chunk = [dict(rows[index])]
        index += 1
        while index < len(rows) and rows[index].get("event") not in {"MOMENT", "SEGMENT_END"}:
            chunk.append(dict(rows[index]))
            index += 1
        if len(chunk) == 1:
            raise ValueError("baseline full-song window contains an empty moment")
        result.append((current, chunk))
    return result


def _global_rows(
    chunks: Sequence[tuple[int, Sequence[Mapping[str, Any]]]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    previous = 0
    for tick, rows in chunks:
        copied = [dict(row) for row in rows]
        copied[0]["delta_tick"] = int(tick) - previous
        result.extend(copied)
        previous = int(tick)
    return result


def _close_truncated_window_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    last_complete = max(
        (index for index, row in enumerate(rows) if row.get("event") in {"NOTE", "SLIDE_BRANCH"}),
        default=-1,
    )
    kept = [dict(row) for row in rows[: last_complete + 1]]
    dropped = len(rows) - len(kept)
    kept.append({"event": "SEGMENT_END"})
    return kept, dropped


def _condition_tensors(
    *,
    difficulty_id: int,
    level: float,
    target_start: int,
    target_end: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "difficulty_id": torch.tensor(
            [condition_difficulty_id(difficulty_id)], dtype=torch.long, device=device
        ),
        "level": torch.tensor([float(level)], dtype=torch.float32, device=device),
        "start_tick": torch.tensor(
            [float(target_start)], dtype=torch.float32, device=device
        ),
        "end_tick": torch.tensor(
            [float(target_end)], dtype=torch.float32, device=device
        ),
    }


def _history_chunks(
    chunks: Sequence[tuple[int, Sequence[Mapping[str, Any]]]],
    *,
    target_start: int,
) -> list[tuple[int, list[dict[str, Any]]]]:
    cutoff = max(0, int(target_start) - HISTORY_TICKS)
    selected = [
        (int(tick), [dict(row) for row in rows])
        for tick, rows in chunks
        if cutoff <= int(tick) < int(target_start)
    ]
    previous = cutoff
    for tick, rows in selected:
        if not rows or rows[0].get("event") != "MOMENT":
            raise ValueError("history chunk is not a complete compact moment")
        rows[0]["delta_tick"] = int(tick) - previous
        previous = int(tick)
    return selected


def _constrain_moment_boundary(
    outputs: Mapping[str, Any],
    *,
    vocabulary: Any,
    current_moment: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Prevent empty moments and force legal closure at the configured generation row limit."""

    if not current_moment:
        return dict(outputs)
    copied = dict(outputs)
    fields = dict(outputs["field_logits"])
    events = fields["event"].clone()
    if len(current_moment) == 1:
        # A just-opened MOMENT must receive at least one semantic member.
        events[..., vocabulary.event_id("MOMENT")] = -10_000.0
        events[..., vocabulary.event_id("SEGMENT_END")] = -10_000.0
    if len(current_moment) >= MAX_GENERATED_MOMENT_ROWS:
        allowed = {
            vocabulary.event_id("MOMENT"),
            vocabulary.event_id("SEGMENT_END"),
        }
        for index in range(events.shape[-1]):
            if index not in allowed:
                events[..., index] = -10_000.0
    fields["event"] = events
    copied["field_logits"] = fields
    return copied


@torch.no_grad()
@generation_session
def generate_baseline_rows(
    *,
    model: BaselineTransformer,
    feature: np.ndarray | None,
    timing: TimingMap,
    level: float,
    cabinet: str,
    difficulty_id: int,
    seed: int,
    temperature: float,
    device: torch.device,
    max_rows_per_window: int = 512,
    allow_truncated: bool = False,
    physical_state: Any = None,
    bpm_audio: Mapping[str, Any] | None = None,
    trace: bool = False,
    local_seed: int | None = None,
    style_mode: str | None = None,
    style_reference: Mapping[str, Any] | None = None,
    latent_temperature: float | None = None,
    mixture_temperature: float | None = None,
    stream=None,
    _style_override=None,
    teacher_audio_cache=None,
    teacher_audio_variant: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    from maimai_baseline.inference.style import sampling_options
    from maimai_baseline.radar_contract import is_radar, BEATS_PER_TOKEN, DIMENSION
    from maimai_baseline.style_contract import is_frozen, is_native, is_draft, STYLE_BEATS, STYLE_SLOTS
    radar = is_radar(getattr(model, 'config', None))
    frozen = is_frozen(getattr(model, 'config', None))
    native = is_native(getattr(model, 'config', None))
    draft = is_draft(getattr(model, 'config', None))
    if native and (teacher_audio_cache is None or teacher_audio_variant is None):
        raise ValueError('native-audio inference requires --teacher-audio-cache and a matching song/timing variant')
    if not native and teacher_audio_cache is not None:
        raise ValueError('continuous inference cannot consume native teacher audio')
    complete_condition = draft or getattr(model, 'prior_condition_contract', None) == 'complete-tail-empty-v1'
    if _style_override is not None and not complete_condition:
        raise ValueError('fixed-sample latent intervention requires the complete-tail prior contract')
    style_mode = ('draft' if draft else 'ar' if radar or frozen else 'disabled') if style_mode is None else style_mode
    sampling_options(style_mode, latent_temperature, mixture_temperature, local_seed=local_seed, reference=style_reference, radar=radar, frozen=frozen, draft=draft)
    if not isinstance(model, BaselineTransformer):
        raise TypeError("baseline generation received an incompatible model")
    if bpm_audio is None:
        raise ValueError("generation feature grid/checkpoint time semantics mismatch")
    if temperature < 0:
        raise ValueError("Route-71 temperature must be non-negative")
    from maimai_baseline.time.audio_end import audio_end
    boundary = audio_end(timing, int(bpm_audio["num_samples"]))
    song_end_tick = boundary.onset_tick
    if physical_state is not None:
        physical_state.song_end_tick = boundary.onset_tick
        physical_state.audio_tail_tick = boundary.tail_tick
    generator = torch.Generator(device=device.type)
    generator.manual_seed(int(seed))
    codec = Route71Codec(model.vocabulary)
    chunks: list[tuple[int, list[dict[str, Any]]]] = []
    traces: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    physical_moment = 0
    if complete_condition:
        from maimai_dataset.window import training_window_count
        window_count = training_window_count(boundary.onset_tick, boundary.tail_tick)
    else:
        window_count = song_end_tick // TARGET_TICKS + 1

    if stream: stream("stage", stage="audio")
    full_audio = None
    if not native:
        from maimai_baseline.data.song_audio import raw_song,pack
        raw=pack([{k:torch.from_numpy(v) for k,v in raw_song(feature,bpm_audio['coordinates'],timing,bpm_audio['num_samples'],include_fusion=False).items()}],[0])
        with torch.no_grad():
            full_audio=model.song_audio_encoder(**{k:v.to(device) for k,v in raw.items()})
    song_style = None
    ar_session = None
    fused_windows = {}
    song_decoder_cache = None
    if getattr(model, 'song_dense', False):
        from maimai_baseline.inference.song_dense import prepare_audio, prepare_native_audio
        if native:
            fused_windows, song_decoder_cache = prepare_native_audio(model, teacher_audio_cache,
                teacher_audio_variant, device, window_count=window_count)
        else:
            fused_windows, song_decoder_cache = prepare_audio(model, feature, bpm_audio, timing,
                song_end_tick, device, full_audio,
                **({'window_count': window_count} if complete_condition else {}))
    if draft:
        from maimai_baseline.inference.style import draft_song
        song_style = draft_song(model, teacher_audio_cache, teacher_audio_variant,
            level=level, difficulty_id=difficulty_id, device=device)
    if (radar or frozen) and style_mode == 'ar':
        if song_decoder_cache is None:
            raise ValueError('AR sampling requires the shared song decoder P0 audio')
        # Ordinary and streaming decoding share one prior KV/RNG session. Its
        # audio is the exact P0 already prepared for the decoder.
        token_ticks = TARGET_TICKS if radar else STYLE_BEATS*96
        token_count = window_count*STYLE_SLOTS if complete_condition else song_end_tick//token_ticks+1
        starts = torch.arange(token_count, device=device) * token_ticks
        valid = torch.ones((1, token_count), dtype=torch.bool, device=device) if complete_condition else (starts <= song_end_tick)[None]
        whole = dict(audio=song_decoder_cache.memory,
            audio_valid=~song_decoder_cache.memory_padding,
            audio_beats=song_decoder_cache.memory_beats, valid=valid,
            span_beats=torch.tensor([song_end_tick / 96.], device=device))
        options = dict(temperature=1. if latent_temperature is None else latent_temperature,
                       mixture_temperature=1. if mixture_temperature is None else mixture_temperature)
        arguments = dict(**whole, difficulty_id=torch.tensor([condition_difficulty_id(difficulty_id)],device=device),
            level=torch.tensor([level],device=device),seed=seed if local_seed is None else local_seed, **options)
        ar_session = model.style_prior.begin_sample(**arguments)
    for window_index in range(window_count):
        if stream: stream("stage", stage="decoder", current=window_index, total=window_count)
        if ar_session is not None:
            song_style = ar_session.ensure((window_index + 1)*(1 if radar else STYLE_SLOTS))
        if stream: stream("window_start", index=window_index, current=window_index, total=window_count)
        target_start = window_index * TARGET_TICKS
        target_end = target_start + TARGET_TICKS
        conditions = _condition_tensors(
            difficulty_id=difficulty_id,
            level=level,
            target_start=target_start,
            target_end=target_end,
            device=device,
        )
        context_chunks = _history_chunks(chunks, target_start=target_start)
        prefix_rows = [{"event": "SEGMENT_START"}]
        prefix_anchors = [max(0, target_start - HISTORY_TICKS)]
        for tick, history_rows in context_chunks:
            prefix_rows.extend(dict(row) for row in history_rows)
            prefix_anchors.extend([tick] * len(history_rows))
        context_row_count = len(prefix_rows)
        prefix_rows.append({"event": "TARGET_START"})
        prefix_anchors.append(target_start)
        audio_frame_start = math.floor(
            timing.beat_to_second(
                Fraction(
                    target_start - AUDIO_PREFIX_TICKS,
                    CHART_TICKS_PER_BEAT,
                )
            )
            * 100.0
        )
        with torch.no_grad():
            audio_context = fused_windows.pop(target_start)
            audio_context = model.prepare_decoder_context(audio_context, **conditions)
        style_context = None
        if radar:
            if song_style is None:
                values = torch.zeros((1, DIMENSION), device=device)
                valid = torch.zeros(1, dtype=torch.bool, device=device)
            else:
                values, valid = song_style.window(target_start)
            style_context = model.radar_context(values, valid)
        elif getattr(getattr(model, "config", None), "latent_enabled", False):
            from maimai_baseline.inference.style import context_for_window
            style_context = song_style.window(target_start) if song_style is not None else context_for_window(model, audio_context, conditions,
                mode=style_mode, local_seed=local_seed,
                song_end_tick=song_end_tick, reference=style_reference)
            if _style_override is not None:
                override = _style_override.window(target_start)
                if override.tokens.shape != style_context.tokens.shape or override.valid.shape != style_context.valid.shape:
                    raise ValueError('fixed-sample latent intervention shape mismatch')
                from dataclasses import replace
                style_context = replace(style_context, tokens=torch.where(override.valid[..., None].to(device),
                    override.tokens.to(device), style_context.tokens))
            if draft:
                style_context = model.draft_context(dict(latent_tokens=style_context.tokens,
                    latent_valid=style_context.valid), perturb=False)
        style_arguments = ({} if style_context is None else {"style_context": style_context})

        from maimai_baseline.inference.cache import DecoderKVCache
        kv_cache = song_decoder_cache if song_decoder_cache is not None else DecoderKVCache(model) if hasattr(model, "decoder_layers") else None
        if song_decoder_cache is not None:
            kv_cache.begin_window(audio_context, conditions['start_tick'], style_context)
        row_budget = int(max_rows_per_window)
        local_rows = prefix_rows
        local_anchors = prefix_anchors
        current_moment: list[dict[str, Any]] = []
        current_tick = target_start
        current_anchor: int | None = None
        target_rows: list[dict[str, Any]] = []
        moment_count = 0
        drift = 0.0
        status = "truncated"
        dropped_rows = 0
        window_trace_start = len(traces)
        physical_window_cursor = len(physical_state.actions) if physical_state is not None else 0
        transaction = None
        excluded_ticks: set[int] = set()
        rollback_count = 0
        step = -1
        while step + 1 < row_budget:
            step += 1
            begin_row()
            router_trace = {}
            def decode_current(excluded_families=()):
                routing = {}
                if hasattr(model, 'router'):
                    from maimai_baseline.inference.family import selector
                    routing['family_selector'] = selector(vocabulary=model.vocabulary, timing=timing,audio_end_seconds=boundary.seconds,
                        tick=current_tick, target_end=target_end, song_end=boundary.onset_tick,
                        tail_tick=boundary.tail_tick, moment_count=moment_count, current_moment=current_moment,
                        maximum_rows=MAX_GENERATED_MOMENT_ROWS, temperature=temperature, generator=generator,
                        trace=router_trace, state=physical_state, moment=physical_moment, excluded_ticks=excluded_ticks,
                        excluded_families=excluded_families)
                with torch.no_grad():
                    begin = kv_cache.length if kv_cache is not None else 0
                    ids, times, active = encode_prefix_rows(local_rows[begin:], local_anchors[begin:],
                        timing=timing, model=model, codec=codec, device=device)
                    anchors = torch.tensor([local_anchors[begin:]], dtype=torch.float32, device=device)
                    outputs = decode_prefix(model, audio_context=audio_context,
                        input_ids=ids, input_times=times, input_time_active=active,
                        row_anchor_tick=anchors, kv_cache=kv_cache, **conditions, **style_arguments, **routing)
                return _constrain_moment_boundary(
                    outputs,
                    vocabulary=model.vocabulary,
                    current_moment=current_moment,
                )
            outputs = decode_current()

            def decode_again(excluded_families):
                nonlocal outputs
                saved = kv_cache.snapshot() if kv_cache is not None else None
                if kv_cache is not None:
                    kv_cache.restore(saved, replay_last=True)
                begin_row()
                try:
                    outputs = decode_current(excluded_families)
                except Exception:
                    if kv_cache is not None:
                        kv_cache.restore(saved)
                    raise
                return outputs
            if physical_state is not None:
                from maimai_baseline.inference.physical import select_row as choose_row
                if hasattr(physical_state, "runtime_identity"):
                    from maimai_baseline.inference.miacode import select_row as choose_row
                from maimai_baseline.physical.inference_state import PhysicalDecodeError
                before = (local_anchors, current_tick, current_anchor, current_moment, local_rows,
                          moment_count, physical_moment, drift, len(traces), len(target_rows), step,
                          kv_cache.snapshot() if kv_cache is not None else None)
                try:
                    from maimai_baseline.inference.retry import select_with_router
                    row, trace, drift = select_with_router(outputs, choose_row=choose_row,
                        decode_again=decode_again, has_router=hasattr(model, 'router'),
                        model=model, codec=codec, state=physical_state,
                        tick=current_tick, target_end=target_end, moment=physical_moment,
                        moment_count=moment_count, audio_frame_start=audio_frame_start, drift=drift,
                        temperature=temperature, generator=generator, excluded_ticks=excluded_ticks)
                except PhysicalDecodeError as error:
                    from maimai_baseline.inference.reporting import failure_record
                    if error.context.get('recoverable'):
                        failure_record(tick=current_tick, moment=physical_moment, stage='moment',
                            reason=str(error), detail=error.context, rollbacks=rollback_count,
                            excluded_ticks=sorted(excluded_ticks))
                    can_restore = error.context.get("recoverable") and transaction is not None and len(current_moment) == 1
                    if hasattr(physical_state, "runtime_identity") and (not can_restore or rollback_count >= 31):
                        if "non-finite" in str(error):
                            raise
                        if not error.context.get("recoverable"):
                            raise
                        if rollback_count >= 31:
                            physical_state.local_counts["moment_budget_exhausted"] += 1
                        from maimai_baseline.inference.miacode import unconstrained
                        row, trace, drift = unconstrained(outputs, reason="moment-budget-or-support-exhausted",
                            attempts=error.context.get("miacode_attempts", ()), model=model, codec=codec,
                            state=physical_state, tick=current_tick, target_end=target_end, moment=physical_moment,
                            moment_count=moment_count, audio_frame_start=audio_frame_start, drift=drift,
                            temperature=temperature, generator=generator)
                    else:
                        if not error.context.get("recoverable") or transaction is None or len(current_moment) != 1:
                            raise
                        excluded_ticks.add(current_tick)
                        rollback_count += 1
                        physical_state.rollbacks += 1
                        count("moment_rollback")
                        if rollback_count >= 32:
                            raise PhysicalDecodeError("Physical moment retry budget exhausted", window=window_index,
                                                      excluded_ticks=sorted(excluded_ticks)) from error
                        (local_anchors,current_tick,current_anchor,current_moment,local_rows,moment_count,
                         physical_moment,drift,trace_cursor,row_cursor,restore_step,kv_snapshot) = transaction
                        physical_state.restore(physical_window_cursor + sum(
                            r["event"] in ("NOTE", "SLIDE_BRANCH") for r in target_rows[:row_cursor]))
                        if kv_cache is not None:
                            kv_cache.restore(kv_snapshot, replay_last=True)
                        del traces[trace_cursor:]
                        del target_rows[row_cursor:]
                        if stream: stream("rollback", index=window_index, row_count=row_cursor)
                        transaction = None
                        step = restore_step - 1
                        continue
                if row["event"] == "MOMENT":
                    transaction = (*before[:-1], kv_cache.snapshot() if kv_cache is not None else None)
                elif row["event"] in ("NOTE", "SLIDE_BRANCH"):
                    physical_state.commit(row,current_tick,physical_moment,
                                          route_of(row, codec))
                    transaction = None
                    excluded_ticks.clear()
                    rollback_count = 0
            else:
                row, trace, drift = _choose_route_row(
                    outputs,
                    model=model,
                    codec=codec,
                    target_current_tick=current_tick,
                    target_end_tick=target_end,
                    song_end_tick=song_end_tick,
                    audio_tail_tick=boundary.tail_tick,audio_end_seconds=boundary.seconds,
                    timing=timing,
                    audio_frame_start=audio_frame_start,
                    moment_count=moment_count,
                    owner_branch_count=next((j for j,r in enumerate(reversed(current_moment or [])) if r.get('event')=='NOTE'),0),
                    owner_start=next((r.get("start_pad") for r in reversed(current_moment or []) if r.get("event")=="NOTE"),None),
                    cumulative_continuous_drift=drift,
                    temperature=temperature,
                    generator=generator,
                )
            if router_trace:
                trace['router'] = router_trace
            event = str(row["event"])
            if event == "MOMENT":
                physical_moment += 1
                current_tick += int(row["delta_tick"])
                current_anchor = current_tick
                current_moment = [dict(row)]
                moment_count += 1
            elif event in {"NOTE", "SLIDE_BRANCH"}:
                if not current_moment:
                    raise ValueError(f"generated {event} without an open moment")
                current_moment.append(dict(row))
            elif event == "SEGMENT_END":
                if current_moment and len(current_moment) == 1:
                    raise ValueError("generated an empty moment before SEGMENT_END")
                status = "complete"
            else:
                raise ValueError(f"unsupported Route-71 generated event: {event!r}")
            local_rows = [*local_rows, dict(row)]
            local_anchors = [*local_anchors, current_tick]
            trace.update(
                {
                    "schema": "maimai.baseline_prediction_trace.v1",
                    "global_step": len(traces),
                    "window_index": window_index,
                    "window_step": step,
                    "anchor_tick": current_tick,
                    "seed": int(seed),
                    "raw_prefix": {"context_rows": context_row_count,
                                   "input_rows": len(local_rows) - 1},
                }
            )
            if trace_enabled():
                traces.append(trace)
            target_rows.append(row)
            if stream: stream("row", index=window_index, row=dict(row), anchor_tick=current_tick)
            if status == "complete":
                break

        if status != "complete":
            if not allow_truncated:
                raise ValueError(f"Route-71 window {window_index} truncated after {row_budget} rows")
            accepted_rows, dropped_rows = _close_truncated_window_rows(target_rows)
            kept = len(accepted_rows) - 1
            physical_moment -= sum(row['event'] == 'MOMENT' for row in target_rows[kept:])
            if trace_enabled():
                del traces[window_trace_start + kept:]
            if dropped_rows and stream:
                stream('rollback', index=window_index, row_count=kept)
            target_rows = accepted_rows
            if song_decoder_cache is not None:
                # Retain only accepted inputs, including a final NOTE which
                # may not have entered K/V when the output budget was reached.
                desired = context_row_count + 1 + kept
                cursor = min(kv_cache.length, desired)
                kv_cache.trim_local(cursor)
                if cursor < desired:
                    with torch.no_grad():
                        ids, times, active = encode_prefix_rows(local_rows[cursor:desired], local_anchors[cursor:desired],
                            timing=timing, model=model, codec=codec, device=device)
                        from maimai_baseline.dual_contract import prefix_families
                        padding_rows = torch.zeros_like(ids['event'], dtype=torch.bool)
                        family = prefix_families(ids['event'], padding_rows, model.vocabulary).clamp_min(0)
                        kv_cache.decode(audio_context=audio_context, input_ids=ids, input_times=times,
                            input_time_active=active, row_padding_mask=padding_rows,
                            row_anchor_tick=torch.tensor([local_anchors[cursor:desired]], device=device).float(),
                            decoder_family=family, **conditions, **style_arguments)
            if physical_state is not None:
                kept_actions = sum(row['event'] in ('NOTE', 'SLIDE_BRANCH') for row in target_rows)
                physical_state.restore(physical_window_cursor + kept_actions)
        if stream: stream("window_end", index=window_index, current=window_index+1, total=window_count)
        new_chunks = _append_window_chunks(
            target_rows, target_start_tick=target_start
        )
        chunks.extend(new_chunks)
        windows.append(
            {
                "index": window_index,
                "start_tick": target_start,
                "end_tick": target_end,
                "audio_start_tick": target_start - AUDIO_PREFIX_TICKS,
                "audio_end_tick": target_end + AUDIO_SUFFIX_TICKS,
                "history_start_tick": max(0, target_start - HISTORY_TICKS),
                "history_end_tick": target_start,
                "history_moments": len(context_chunks),
                "decoder_cache": kv_cache.report() if kv_cache is not None else None,
                "style": None if style_context is None else ({
                    "mode": style_mode, "seed": seed if local_seed is None else local_seed,
                    "temperature": (1. if latent_temperature is None else latent_temperature) if style_mode == 'ar' else None,
                    "mixture_temperature": (1. if mixture_temperature is None else mixture_temperature) if style_mode == 'ar' else None,
                    "values": style_context.values[0].cpu().tolist(),
                    "valid": bool(style_context.valid[0].any()), "beats_per_token": BEATS_PER_TOKEN,
                } if radar else {
                    "mode": style_mode, "latent_seed": (seed if local_seed is None else local_seed) if style_mode == "ar" else None,
                    "latent_temperature": (1. if latent_temperature is None else latent_temperature) if style_mode == 'ar' else None,
                    "mixture_temperature": (1. if mixture_temperature is None else mixture_temperature) if style_mode == 'ar' else None,
                    "mixture_components": 4 if style_mode == 'ar' else None,
                    "level": float(level), "global_level_conditioned": style_mode in ("ar", "draft"),
                    "beats": style_context.beats[0].cpu().tolist(),
                    "valid": style_context.valid[0].cpu().tolist(),
                    "tokens": style_context.tokens[0].cpu().tolist(),
                }),
                "status": status,
                "rows": len(target_rows),
                "moments": len(new_chunks),
                "dropped_rows": dropped_rows,
                "trace_start": window_trace_start if trace_enabled() else None,
                "trace_end": len(traces) if trace_enabled() else None,
            }
        )
    rows = _global_rows(chunks)
    for row in rows:
        if DELTA_FIELD in row:
            raise RuntimeError("internal delta field leaked into public compact rows")
    for window in windows:
        from maimai_baseline.time.moment import generation_clock
        window["generation_clock"] = generation_clock(model.moment_distribution)
        window["start_second"] = chart_tick_to_second(timing, window["start_tick"])
    if windows:
        windows[-1]["final_distribution"] = distribution(rows, codec)
    return rows, traces, windows
