"""Row and time sampling with shared candidate constraints."""

from __future__ import annotations
from maimai_baseline.contract import REMOVED_FIELDS
from maimai_baseline.data.route import Route71Codec
from maimai_baseline.inference.reporting import distribution, row_cached, top5, trace_enabled
from maimai_baseline.model.baseline import BaselineTransformer
from maimai_baseline.physical.inference_state import PhysicalDecodeError
from maimai_baseline.time.grid import DELTA_MS, candidates
from maimai_baseline.time.moment import LINEAR_MS, candidate_probabilities
from maimai_dataset.timing import TimingMap
from typing import Any, Mapping, Sequence
import torch


COMMON_TICKS = (
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60,
    64, 72, 80, 96, 112, 120, 128, 144, 160, 168, 192, 208, 224,
    240, 256, 336, 384,
)


@torch.no_grad()
def delta_choice(output, *, lower_tick, upper_tick, anchor_tick, timing,
                 temperature, generator, candidate, physical_allowed=None, song_end_tick=None,
                 integration_end_tick=None, moment_distribution=LINEAR_MS, **unused):
    first = lower_tick == 0
    choices = candidates(timing, anchor_tick=anchor_tick, start_tick=anchor_tick,
                         end_tick=anchor_tick + upper_tick + 1 if integration_end_tick is None else integration_end_tick,
                         allow_zero=first, song_end_tick=song_end_tick)
    device = output.device
    table = {name: torch.from_numpy(getattr(choices, name)).to(device)[None]
             for name in ("ticks", "seconds", "delta_ms", "lower_ms", "upper_ms")}
    table["valid"] = torch.ones_like(table["ticks"], dtype=torch.bool)
    table["first"] = torch.tensor([first], device=device)
    if candidate is not None:
        candidate = {k: v if k == "gate" else v[None] for k, v in candidate.items()
                     if k in {"query", "keys", "padding_mask", "beats", "gate"}}
    distribution, _ = candidate_probabilities(output[None], table, candidate, torch.zeros(1, dtype=torch.long, device=device),
                                              distribution=moment_distribution)
    logp = distribution[0]
    physical = None
    if physical_allowed is not None:

        logp, physical = constrain_probabilities(logp, torch.as_tensor(physical_allowed, device=device),
                                                temperature=temperature, name=DELTA_MS)

    index = _sample_delta(logp, lower=0, temperature=temperature, generator=generator)
    tick = int(choices.ticks[index])
    delta = tick - anchor_tick
    from maimai_baseline.inference.reporting import trace_enabled
    if not trace_enabled():
        return delta, {"model_field": DELTA_MS, "continuous_mode": float(delta),
                       **({"physical": physical} if physical is not None else {})}
    parameters = {"unit_scale_ms": 1000.0, "minimum_scale_ms": 2.5}
    return delta, {"head": "logistic", "model_field": DELTA_MS, **parameters,
                   "candidate_index": index, "emitted_tick": tick,
                   "candidate_second": float(choices.seconds[index]), "selected_delta_ms": float(choices.delta_ms[index]),
                   "history_delta_ms": float(choices.delta_ms[index]), "continuous_mode": float(delta),
                   "modal_integer_tick": delta, "selected_probability": float(logp[index].exp()),
                   "support_ms": [float(choices.lower_ms[0]), float(choices.upper_ms[-1])],
                   **({"physical": physical} if physical is not None else {})}


def constrain_probabilities(log_probability, allowed, *, temperature, name):
    """Mask the ORIGINAL support; do not re-bin or move truncated tail mass."""
    allowed = allowed.to(device=log_probability.device, dtype=torch.bool)
    if allowed.shape != log_probability.shape or not bool(allowed.any()):
        raise PhysicalDecodeError("empty Physical support", field=name)
    if not bool(torch.isfinite(log_probability).all()):
        raise PhysicalDecodeError("non-finite model probabilities", field=name)
    removed = None
    if trace_enabled():
        original = torch.softmax(log_probability, dim=-1)
        removed = float(original.masked_select(~allowed).sum().item())
    masked = log_probability.masked_fill(~allowed, -torch.inf)
    masked = masked - torch.logsumexp(masked, dim=-1)
    if not trace_enabled():
        return masked, dict(field=name, probability_statistics_collected=False)
    sampling = torch.softmax(masked / (temperature if temperature > 0 else 1.), dim=-1)
    return masked, dict(field=name, original_count=allowed.numel(), legal_count=int(allowed.sum()),
                        removed_probability_mass=removed, sampling_max_probability=float(sampling.max()))


def choose(logits, allowed, *, temperature, generator, name):
    logp, detail = constrain_probabilities(logits.float().reshape(-1),
        torch.as_tensor(allowed, device=logits.device).reshape(-1), temperature=temperature, name=name)
    if temperature <= 0:
        index = int(logp.argmax())
    else:
        index = int(torch.multinomial(torch.softmax(logp / temperature, -1), 1, generator=generator))
    if not trace_enabled():
        return index, detail
    sampling = torch.softmax(logp / (temperature if temperature > 0 else 1.),dim=-1)
    detail.update(selected_index=index, selected_probability=float(sampling[index]),
                  base_conditioned_probability=float(logp[index].exp()))
    return index, detail


# Route-logit constraints


def mask_route_logits(
    logits: torch.Tensor,
    *,
    event: str,
    kind: str | None,
) -> torch.Tensor:
    if logits.shape[-1] != 71:
        raise ValueError("Route-71 logits must have 71 classes")
    allowed = torch.zeros(71, dtype=torch.bool, device=logits.device)
    if event == "NOTE" and kind == "wifi":
        allowed[70] = True
    elif event == "SLIDE_BRANCH":
        allowed[:] = True
    elif event == "NOTE" and kind == "slide":
        allowed[:70] = True
    else:
        raise ValueError(f"route is not applicable to event={event!r}, kind={kind!r}")
    return logits.masked_fill(~allowed, -10_000.0)


def _sample_index(
    logits: torch.Tensor,
    candidates: Sequence[int],
    *,
    temperature: float,
    generator: torch.Generator | None,
) -> int:
    if not candidates:
        raise ValueError("cannot choose from an empty candidate set")
    indices = torch.tensor(candidates, device=logits.device, dtype=torch.long)
    selected = logits[indices].to(torch.float32)
    if temperature <= 0:
        return int(indices[torch.argmax(selected)].item())
    probability = torch.softmax(selected / float(temperature), dim=-1)
    choice = int(torch.multinomial(probability, 1, generator=generator).item())
    return int(indices[choice].item())


def _time_choice(
    output: torch.Tensor,
    *,
    field: str,
    lower: int,
    upper: int,
    time_unit_scale: float = 1.0,
    temperature: float = 0.0,
    generator: torch.Generator | None = None,
    physical_safe_upper: int | None = None,
    excluded_values: tuple[int, ...] = (),
    duration_distribution: Mapping[str, Any] | None = None,
    anchor_tick: int | None = None,
    timing: TimingMap | None = None,
) -> tuple[int, dict[str, Any]]:
    if upper < lower:
        raise ValueError(f"empty time support [{lower}, {upper}] for {field}")
    raw = output.to(torch.float32)
    from maimai_baseline.time.duration import LINEAR_CONTRACT
    duration_distribution = LINEAR_CONTRACT if duration_distribution is None else duration_distribution
    from maimai_baseline.time.duration import candidate_ticks
    _, candidates = candidate_ticks(field=field, lower=lower, upper=upper, device=raw.device)
    def compute_mass():
        from maimai_baseline.time.duration import distribution_name
        from maimai_baseline.time.duration import candidate_probabilities
        distribution = distribution_name(duration_distribution)
        if anchor_tick is None or timing is None:
            raise ValueError("invalid duration distribution or missing timing/anchor")
        _, probability = candidate_probabilities(raw, field=field, lower=lower, upper=upper,
                                                  anchor_tick=anchor_tick, timing=timing, distribution=distribution)
        return probability, {}, output
    if timing is None or anchor_tick is None:
        raise ValueError("duration distribution requires timing and anchor")
    context = (duration_distribution["schema"], anchor_tick,
        timing.first_seconds, tuple((p.beat, p.bpm, p.second) for p in timing.points))
    # Conditioned readouts change on route/position retries. Retain the input
    # tensor in the cache value so its storage address cannot be recycled.
    log_probability, support, _ = row_cached(("duration", field, lower, upper, time_unit_scale, context,
                                             output.device, output.data_ptr()), compute_mass)
    physical_detail = None
    if excluded_values:

        allowed = ~torch.isin(candidates, torch.tensor(excluded_values, device=candidates.device))
        if physical_safe_upper is not None:
            allowed &= candidates.le(physical_safe_upper)
        log_probability, physical_detail = constrain_probabilities(
            log_probability, allowed, temperature=temperature, name=field)
        physical_safe_upper = None
    if physical_safe_upper is not None:

        log_probability, physical_detail = constrain_probabilities(
            log_probability, candidates.le(physical_safe_upper), temperature=temperature,
            name=field,
        )
    if temperature <= 0:
        tick = int(candidates[torch.argmax(log_probability)].item())
    else:
        probability = torch.softmax(log_probability / float(temperature), dim=-1)
        tick = int(candidates[torch.multinomial(probability, 1, generator=generator)].item())
    if not trace_enabled():
        return tick, ({"physical": physical_detail} if physical_detail is not None else {})
    from maimai_baseline.inference.reporting import trace_detail
    from maimai_baseline.time.duration import distribution_name
    detail = trace_detail(raw, tick, log_probability, field=field, lower=lower, anchor_tick=anchor_tick, timing=timing,
        distribution=distribution_name(duration_distribution), selected_index=int((candidates == tick).nonzero()[0]))
    if physical_detail is not None:
        detail["physical"] = physical_detail
    return tick, detail


def duration_kind_allowed(kind, tick, tail_tick, timing, end_seconds=None):
    from maimai_baseline.time.audio_end import duration_upper
    if kind in ('hold', 'touch_hold'):
        return duration_upper('hold_duration_tick', tick, tail_tick, timing=timing,end_seconds=end_seconds) >= 2
    if kind in ('slide', 'wifi'):
        return duration_upper('branch_duration_tick', tick, tail_tick, timing=timing,end_seconds=end_seconds) >= 1
    return True


def sample_modifiers(row, logits, *, tick, tail_tick, temperature, generator, short_allowed=True, fixed_short=None, timing=None, audio_end_seconds=None):
    from maimai_dataset.modifiers import LEGAL
    kind = row.get('kind')
    legal = LEGAL.get(kind, ()) if row['event'] == 'NOTE' else ('track_break',) if row['event'] == 'SLIDE_BRANCH' else ()
    groups = {'flags': [], 'head_flags': [], 'branch_flags': []}
    if timing is not None:
        from maimai_baseline.time.audio_end import duration_upper
        short_allowed = short_allowed and duration_upper('hold_duration_tick',tick,tail_tick,timing=timing,end_seconds=audio_end_seconds)>=1
    for name in legal:
        allowed = [0] if name in ('short_hold', 'short_touch_hold') and (not short_allowed or (timing is None and tick + 1 > tail_tick)) else [0, 1]
        value = fixed_short if fixed_short is not None and name in ("short_hold","short_touch_hold") else _sample_index(logits['modifier_' + name], allowed, temperature=temperature, generator=generator)
        if value:
            field = 'branch_flags' if name == 'track_break' else 'head_flags' if kind in ('slide', 'wifi') else 'flags'
            groups[field].append('break' if name == 'track_break' else name)
    if legal:
        row['flags'] = '+'.join(sorted(groups['flags']))
    if kind in ('slide', 'wifi'):
        row['head_flags'] = '+'.join(sorted(groups['head_flags']))
    if 'track_break' in legal:
        row['branch_flags'] = '+'.join(sorted(groups['branch_flags']))


def _choose_row(
    outputs: Mapping[str, Any],
    *,
    vocabulary: Any,
    model, codec, owner_start=None, audio_end_seconds=None, owner_branch_count=0,
    target_current_tick: int,
    target_end_tick: int,
    song_end_tick: int,
    audio_tail_tick: int,
    timing: TimingMap,
    audio_frame_start: int,
    moment_count: int,
    cumulative_continuous_drift: float,
    temperature: float = 0.0,
    generator: torch.Generator | None = None,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    field_logits = {
        field: values[0, -1] for field, values in outputs["field_logits"].items()
    }
    from maimai_dataset.modifiers import LEGACY_FIELDS
    from maimai_baseline.time.audio_end import duration_upper
    event_logits = field_logits["event"].clone()
    if duration_upper("branch_duration_tick", target_current_tick, audio_tail_tick, timing=timing, end_seconds=audio_end_seconds) < 1:
        event_logits[vocabulary.event_id("SLIDE_BRANCH")] = -10_000.0
    event_logits[vocabulary.event_id("TARGET_START")] = -10_000.0
    moment_lower = 0 if moment_count == 0 else 1
    moment_upper = target_end_tick - target_current_tick - 1
    moment_upper = min(moment_upper, song_end_tick - target_current_tick)
    if moment_upper < moment_lower:
        event_logits[vocabulary.event_id("MOMENT")] = -10_000.0
    from maimai_baseline.time.grid import first_candidate_tick
    first_legal = first_candidate_tick(target_current_tick + moment_lower)
    if first_legal >= target_end_tick or first_legal > song_end_tick:
        event_logits[vocabulary.event_id("MOMENT")] = -10_000.0
    event_id = _sample_index(
        event_logits,
        [index for index in range(event_logits.numel()) if event_logits[index] > -9_000],
        temperature=temperature,
        generator=generator,
    )
    event = vocabulary.fields["event"].decode(event_id)
    if not isinstance(event, str):
        raise ValueError("baseline generation selected an invalid event")
    row: dict[str, Any] = {"event": event}
    note_kind: str | None = None
    from maimai_baseline.model.row_fields import sample_readout
    if event == "NOTE":
        field_logits.update({n:v[0,-1] for n,v in sample_readout(model,outputs,row,"kind").items()})
        kind_id = _sample_index(
            field_logits["kind"],
            [i for i in vocabulary.allowed_value_ids(event, "kind")
             if duration_kind_allowed(str(vocabulary.fields["kind"].decode(i)), target_current_tick, audio_tail_tick, timing, end_seconds=audio_end_seconds)],
            temperature=temperature,
            generator=generator,
        )
        note_kind = str(vocabulary.fields["kind"].decode(kind_id))
        row["kind"] = note_kind
    if event in ("NOTE","SLIDE_BRANCH"):
        field_logits.update({n:v[0,-1] for n,v in sample_readout(model,outputs,row,"position_flags",owner_start=owner_start).items()})
    for field in vocabulary.field_names:
        if field in {"event", "kind", "branch_index", "head_kind", *LEGACY_FIELDS} or field not in vocabulary.applicability.get(event, ()):
            continue
        if event == "NOTE" and note_kind is not None and field not in vocabulary.note_kind_value_constraints.get(note_kind, {}):
            continue
        if event=="SLIDE_BRANCH" and field=="start_pad":
            row[field]=owner_start
            continue
        candidate = _sample_index(
            field_logits[field],
            vocabulary.allowed_value_ids(event, field, note_kind=note_kind),
            temperature=temperature,
            generator=generator,
        )
        value = vocabulary.fields[field].decode(candidate)
        if value is not None:
            row[field] = value

    if event=='SLIDE_BRANCH':row['branch_index']=owner_branch_count+1
    elif note_kind in ('slide','wifi'):row['head_kind']='star'

    def audio_delta_candidate() -> dict[str, torch.Tensor] | None:
        if "delta_candidate" not in outputs:
            return None
        candidate = outputs["delta_candidate"]
        return {
            "query": candidate["query"][0, -1],
            "keys": candidate["keys"][0],
            "beats": candidate["beats"][0],
            "padding_mask": candidate["padding_mask"][0],
            "gate": candidate["gate"],
            **{
                name: None if candidate.get(name) is None else candidate[name][0]
                for name in (
                    "lookup_left", "lookup_right", "lookup_weight",
                    "lookup_padding", "lookup_start_tick",
                )
            },
        }

    sample_modifiers(row, field_logits, tick=target_current_tick, tail_tick=audio_tail_tick, timing=timing,audio_end_seconds=audio_end_seconds,
                     temperature=temperature, generator=generator)
    route_trace = None
    route = None
    if event=='SLIDE_BRANCH' or note_kind in ('slide','wifi'):
        if event=='SLIDE_BRANCH':
            if owner_start is None:
                raise ValueError('branch requires known owner start')
            row['start_pad']=owner_start
        route_logits=sample_readout(model,outputs,row,'route',owner_start=owner_start)['route']
        row,route_trace=_route_choice(row,dict(outputs,route_logits=route_logits),codec=codec,temperature=temperature,generator=generator)
        from maimai_baseline.inference.reporting import route_of
        route=route_of(row,codec)
    time_trace: dict[str, Any] = {}
    if event == "MOMENT":
        field = "delta_tick"
        tick, detail = _delta_time_choice(
            outputs["time_outputs"]["delta_ms"][0, -1],
            lower_tick=moment_lower,
            upper_tick=moment_upper,
            anchor_tick=target_current_tick,
            timing=timing,
            audio_frame_start=audio_frame_start,
            temperature=temperature,
            generator=generator,
            candidate=audio_delta_candidate(),
            moment_distribution=outputs.get("moment_distribution"),
            song_end_tick=song_end_tick,
            integration_end_tick=target_end_tick,
        )
        row[field] = tick
        detail.update({
            "support": [moment_lower, moment_upper],
            "official_tick": None,
            "official_error_tick": None,
            "nearest_common_tick_distance": min(abs(tick - value) for value in COMMON_TICKS),
        })
        cumulative_continuous_drift += float(detail["continuous_mode"]) - tick
        detail["cumulative_continuous_vs_integer_drift_tick"] = cumulative_continuous_drift
        time_trace[detail["model_field"]] = detail
    duration_field = (
        "hold_duration_tick"
        if event == "NOTE" and note_kind in {"hold", "touch_hold"}
        else "branch_duration_tick"
        if event == "SLIDE_BRANCH" or (event == "NOTE" and note_kind in {"slide", "wifi"})
        else None
    )
    if duration_field is not None:
        lower = 1 if duration_field == "branch_duration_tick" else 2
        upper = duration_upper(duration_field, target_current_tick, audio_tail_tick, timing=timing,end_seconds=audio_end_seconds)
        tick, detail = _time_choice(
            sample_readout(model,outputs,row,"duration",owner_start=owner_start,route=route)[duration_field][0,-1],
            field=duration_field,
            lower=lower,
            upper=upper,
            time_unit_scale=float(outputs.get("time_unit_scale", 1.0)),
            duration_distribution=outputs.get("duration_distribution"),
            anchor_tick=target_current_tick, timing=timing,
            temperature=temperature,
            generator=generator,
        )
        row[duration_field] = tick
        detail.update({
            "support": [lower, upper],
            "official_tick": None,
            "official_error_tick": None,
            "nearest_common_tick_distance": min(abs(tick - value) for value in COMMON_TICKS),
        })
        time_trace[duration_field] = detail
    raw_top5 = top5(field_logits, vocabulary)
    return row, {
        "schema": "maimai.baseline_prediction_trace.v1",
        "row": dict(row),
        "raw_top5": raw_top5,
        **({"route":route_trace} if route_trace is not None else {}),
        "time": time_trace,
        "temperature": float(temperature),
    }, cumulative_continuous_drift


class _FactorFreeVocabulary:
    """Read-only vocabulary view that prevents legacy route factor sampling."""

    def __init__(self, vocabulary: Any) -> None:
        self._vocabulary = vocabulary
        self.field_names = tuple(
            field for field in vocabulary.field_names if field not in REMOVED_FIELDS
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._vocabulary, name)


def _route_choice(
    row: dict[str, Any],
    outputs: Mapping[str, Any],
    *,
    codec: Route71Codec,
    temperature: float,
    generator: torch.Generator,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    event = str(row["event"])
    kind = None if "kind" not in row else str(row["kind"])
    applicable = event == "SLIDE_BRANCH" or (
        event == "NOTE" and kind in {"slide", "wifi"}
    )
    if not applicable:
        return row, None
    raw = outputs["route_logits"][0, -1].to(torch.float32)
    masked = mask_route_logits(raw, event=event, kind=kind)
    candidates = [
        index for index in range(masked.numel()) if float(masked[index]) > -9_000
    ]
    route_class = _sample_index(
        masked,
        candidates,
        temperature=temperature,
        generator=generator,
    )
    selected_probability = rank = None
    if trace_enabled():
        probability = torch.softmax(
            masked[candidates] / (float(temperature) if temperature > 0 else 1.0), dim=-1)
        local_index = candidates.index(route_class)
        selected_probability = float(probability[local_index].item())
        rank = int((probability > probability[local_index]).sum().item()) + 1
    factors = codec.restore_factor_ids(route_class)
    restored = dict(row)
    for field, value_id in factors.items():
        if int(value_id) < 0:
            continue
        value = codec.vocabulary.fields[field].decode(int(value_id))
        if value is not None:
            restored[field] = value
    if not trace_enabled():
        return restored, None
    return restored, {
        "route_id": f"B{route_class:03d}",
        "route_class": route_class,
        "canonical_path": codec.paths[route_class],
        "legal_support": [f"B{index:03d}" for index in candidates],
        "legal_support_count": len(candidates),
        "selected_probability": selected_probability,
        "selected_rank": rank,
    }


def _choose_route_row(
    outputs: Mapping[str, Any],
    *,
    model: BaselineTransformer,
    codec: Route71Codec,
    target_current_tick: int,
    target_end_tick: int,
    song_end_tick: int,
    audio_tail_tick: int,
    timing: TimingMap,
    audio_frame_start: int,
    moment_count: int,
    cumulative_continuous_drift: float,
    temperature: float,
    generator: torch.Generator,
    owner_start=None, audio_end_seconds=None, owner_branch_count=0,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    factor_free = _FactorFreeVocabulary(model.vocabulary)
    row, trace, drift = _choose_row(
        outputs,
        vocabulary=factor_free, model=model, codec=codec, owner_start=owner_start,audio_end_seconds=audio_end_seconds,owner_branch_count=owner_branch_count,
        target_current_tick=target_current_tick,
        target_end_tick=target_end_tick,
        song_end_tick=song_end_tick,
        audio_tail_tick=audio_tail_tick,
        timing=timing,
        audio_frame_start=audio_frame_start,
        moment_count=moment_count,
        cumulative_continuous_drift=cumulative_continuous_drift,
        temperature=temperature,
        generator=generator,
    )
    return row, trace, drift


def _sample_delta(
    log_probability: torch.Tensor,
    *,
    lower: int,
    temperature: float,
    generator: torch.Generator | None,
) -> int:
    if temperature <= 0:
        return int(lower + int(log_probability.argmax().item()))
    probability = torch.softmax(log_probability / float(temperature), dim=-1)
    return int(
        lower
        + int(torch.multinomial(probability, 1, generator=generator).item())
    )


def _delta_time_choice(
    output: torch.Tensor,
    *,
    lower_tick: int,
    upper_tick: int,
    anchor_tick: int,
    timing: TimingMap,
    audio_frame_start: int,
    temperature: float,
    generator: torch.Generator | None,
    candidate: Mapping[str, Any] | None,
    physical_allowed: Sequence[bool] | None = None,
    song_end_tick: int | None = None,
    integration_end_tick: int | None = None,
    moment_distribution: str | None = None,
) -> tuple[int, dict[str, Any]]:

    return delta_choice(output, lower_tick=lower_tick, upper_tick=upper_tick, anchor_tick=anchor_tick,
        timing=timing, temperature=temperature, generator=generator, candidate=candidate,
        physical_allowed=physical_allowed, song_end_tick=song_end_tick, integration_end_tick=integration_end_tick,
        moment_distribution=moment_distribution or "maimai.bpm96_linear_ms.v2")
