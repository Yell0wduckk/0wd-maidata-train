"""Collation, shared audio groups, and time-candidate preparation."""

from __future__ import annotations
from collections.abc import Callable, Mapping, Sequence
from maimai_baseline.contract import REMOVED_FIELDS
from maimai_baseline.time.coordinates import seconds_at_ticks
from maimai_baseline.time.duration import FIELDS, HOLD, cpu_seconds, tick_bounds
from maimai_baseline.time.grid import DELTA_MS, candidates
from maimai_dataset.dataset import collate_compact_segments as collate_frozen_segments
from maimai_dataset.encoding import TIME_FIELDS
from maimai_dataset.timing import TimingMap
from maimai_dataset.vocabulary import CompactVocabulary
from typing import Any
import torch


def convert_batch(batch, timings):
    result = dict(batch, sft_timings=tuple(timings))
    def renamed(name):
        value = dict(batch[name])
        value[DELTA_MS] = value.pop("delta_tick")
        return value
    for name in ("input_times", "input_time_active", "time_targets", "time_lower",
                 "time_upper", "time_category_targets"):
        result[name] = renamed(name)
    def intervals(timing, start, end):
        shape = end.shape
        return torch.from_numpy((seconds_at_ticks(timing, end.reshape(-1).tolist()) -
                                 seconds_at_ticks(timing, start.reshape(-1).tolist())) * 1000).reshape(shape).float()
    for name, active_name, anchor_name in (
        ("input_times", "input_time_active", "row_anchor_tick"),
    ):
        source = result[name][DELTA_MS]
        converted = torch.zeros_like(source, dtype=torch.float32)
        for b, timing in enumerate(timings):
            active = result[active_name][DELTA_MS][b]
            anchor = batch[anchor_name][b]
            converted[b][active] = intervals(timing, anchor[active] - source[b][active], anchor[active])
        result[name][DELTA_MS] = converted
    source_target = result["time_targets"][DELTA_MS]
    target = torch.full_like(source_target, -100, dtype=torch.float32)
    target_indices = torch.full_like(source_target, -1, dtype=torch.long)
    tables = []
    width = 0
    row_seconds = torch.zeros_like(batch["row_anchor_tick"], dtype=torch.float64)
    for b, timing in enumerate(timings):
        row_seconds[b] = torch.from_numpy(seconds_at_ticks(timing, batch["row_anchor_tick"][b].tolist()))
        for r in source_target[b].ne(-100).nonzero().flatten().tolist():
            anchor = int(batch["row_anchor_tick"][b, r])
            tick = anchor + int(source_target[b, r])
            # The old cache upper bound includes the window's terminal tick.
            # v5 centers obey the original half-open target window instead.
            end = int(batch["end_tick"][b])
            first = bool(result["time_lower"][DELTA_MS][b, r] == 0)
            table = candidates(timing, anchor_tick=anchor, start_tick=anchor, end_tick=end, allow_zero=first,
                               song_end_tick=int(batch["song_end_tick"][b]))
            target_indices[b, r] = table.target_index(tick)
            target[b, r] = intervals(timing, torch.tensor(anchor), torch.tensor(tick))
            width = max(width, len(table.ticks))
            tables.append((b, r, first, table))
    row_index = torch.full_like(target_indices, -1)
    shape = (len(tables), width)
    fields = {key: torch.zeros(shape, dtype=torch.float64)
              for key in ("seconds", "delta_ms", "lower_ms", "upper_ms")}
    fields["ticks"] = torch.zeros(shape, dtype=torch.long)
    fields["valid"] = torch.zeros(shape, dtype=torch.bool)
    fields["first"] = torch.zeros(len(tables), dtype=torch.bool)
    upper = torch.zeros_like(target)
    for i, (b, r, first, table) in enumerate(tables):
        n = len(table.ticks)
        for key in ("ticks", "seconds", "delta_ms", "lower_ms", "upper_ms"):
            fields[key][i, :n] = torch.from_numpy(getattr(table, key))
        fields["valid"][i, :n] = True
        fields["first"][i] = first
        row_index[b, r] = i
        upper[b, r] = fields["upper_ms"][i].amax().float()
    result["time_targets"][DELTA_MS] = target
    result["time_category_targets"][DELTA_MS] = torch.full_like(target_indices, -100)
    # lower/upper are real ms bounds; target cell identity is separate.
    result["time_lower"][DELTA_MS] = torch.zeros_like(target)
    result["time_upper"][DELTA_MS] = upper
    result.update(bpm96_candidates=fields, bpm96_target_index=target_indices,
                  bpm96_row_index=row_index,
                  row_anchor_second=row_seconds, delta_time_parameterization="bpm96-ms")
    validate_packed_candidates(result)
    return result


def validate_packed_candidates(batch):
    """Check the internal CPU batch layout once, before Physical/H2D consumption."""
    target = batch["time_targets"][DELTA_MS]
    index = batch["bpm96_row_index"]
    table = batch["bpm96_candidates"]
    active = target.ne(-100)
    count = int(active.sum())
    if (index.dtype != torch.long or index.shape != target.shape or
            not torch.equal(index[~active], torch.full_like(index[~active], -1)) or
            not torch.equal(index[active], torch.arange(count, device=index.device))):
        raise ValueError("invalid packed BPM row mapping")
    valid = table["valid"]
    if valid.ndim != 2 or valid.shape[0] != count or valid.dtype != torch.bool:
        raise ValueError("invalid packed BPM candidate shape")
    for name in ("seconds", "delta_ms", "lower_ms", "upper_ms", "ticks"):
        dtype = torch.long if name == "ticks" else torch.float64
        if table[name].shape != valid.shape or table[name].dtype != dtype:
            raise ValueError(f"invalid packed BPM candidate field: {name}")
    if table["first"].shape != (count,) or table["first"].dtype != torch.bool:
        raise ValueError("invalid packed BPM first mapping")
    targets = batch["bpm96_target_index"]
    if targets.shape != target.shape or targets.dtype != torch.long:
        raise ValueError("invalid packed BPM target shape")
    chosen = targets[active]
    if count and (bool((chosen < 0).any()) or bool((chosen >= valid.shape[1]).any()) or
                  not bool(valid.gather(1, chosen[:, None]).all())):
        raise ValueError("invalid packed BPM target cell index")


def collate_history_context_segments(
    items: Sequence[Mapping[str, Any]],
    *,
    audio_loader: Callable[[str], Mapping[str, Any]] | None = None,
    chart_only: bool = False,
) -> dict[str, Any]:
    if not items:
        raise ValueError("cannot collate an empty history-context batch")
    if {str(item["_history_layout"]) for item in items} != {
        "raw-prefix-v1"
    }:
        raise ValueError("baseline batch has an invalid history layout")
    batch = collate_frozen_segments(items, audio_loader=audio_loader, chart_only=chart_only)
    batch["history_layout"] = "raw-prefix-v1"
    batch["context_cutoff_tick"] = torch.tensor(
        [int(item["_context_cutoff_tick"]) for item in items], dtype=torch.long
    )
    batch["original_sequence_length"] = torch.tensor(
        [int(item["_original_sequence_length"]) for item in items], dtype=torch.long
    )
    batch["history_context_sequence_length"] = torch.tensor(
        [int(item["_history_context_sequence_length"]) for item in items], dtype=torch.long
    )
    batch["batch_stats"] = {
        **dict(batch.get("batch_stats", {})),
        "original_row_positions": int(
            sum(int(item["_original_sequence_length"]) for item in items)
        ),
        "history_context_row_positions": int(
            sum(int(item["_history_context_sequence_length"]) for item in items)
        ),
        "decoder_row_positions": int(
            sum(int(item["row_anchor_tick"].shape[0]) for item in items)
        ),
    }
    blocks = torch.full_like(batch["row_padding_mask"], -2, dtype=torch.long)
    for b, item in enumerate(items):
        blocks[b, :len(item["_local_block_id"])] = item["_local_block_id"]
    batch["local_block_id"] = blocks
    batch["prefix_lengths"] = torch.tensor([item["_history_context_target_index"] for item in items])
    return batch


def collate_route71_segments(
    items: Sequence[Mapping[str, Any]], *, audio_loader: Any = None, chart_only=False
) -> dict[str, Any]:
    batch = collate_history_context_segments(items, audio_loader=audio_loader, chart_only=chart_only)
    width = int(batch["row_padding_mask"].shape[1])
    route_mass = torch.zeros((len(items), width), dtype=torch.float32)
    for index, item in enumerate(items):
        value = item["route_informative_mass"]
        route_mass[index, : value.shape[0]] = value
    batch["route_informative_mass"] = route_mass
    batch["history_layout"] = "raw-prefix-v1"
    for mapping_name in ("input_ids", "target_ids"):
        mapping = batch[mapping_name]
        if any(field in mapping for field in REMOVED_FIELDS):
            raise RuntimeError(f"{mapping_name} leaked removed route factors")
        if "route" not in mapping:
            raise RuntimeError(f"{mapping_name} is missing unified route")
    return batch


def collate_spatial_route71_segments(
    items: Sequence[Mapping[str, Any]], *, audio_loader: Any = None, chart_only=False
) -> dict[str, Any]:
    return collate_route71_segments(items, audio_loader=audio_loader, chart_only=chart_only)


def collate_baseline_segments(items: Sequence[Mapping[str, Any]], *, vocabulary: CompactVocabulary, audio_loader: Any = None,
                              timing_root=None, chart_only=False) -> dict[str, Any]:
    if not items or not all(item.get("_bpm96") for item in items):
        raise ValueError("v9 collation requires BPM96 audio for every item")
    batch=collate_spatial_route71_segments(items,audio_loader=audio_loader,chart_only=chart_only)
    from maimai_baseline.time.offline import PreparedTiming
    timings = [
        (item["_prepared_timing"] if "_prepared_timing" in item else
         PreparedTiming(TimingMap.from_dict(item["_delta10ms_timing"]))).get(timing_root)
        for item in items
    ]


    from maimai_dataset.modifiers import prepare_batch as prepare_modifiers
    batch = prepare_modifiers(batch, vocabulary)
    batch = prepare_audio_bounds(batch, items, timings, vocabulary)
    from maimai_baseline.dual_contract import prefix_families
    batch['decoder_family'] = prefix_families(batch['input_ids']['event'], batch['row_padding_mask'],
                                             vocabulary, targets=batch['target_ids']['event'])
    if bool((batch['decoder_family'].lt(0) & ~batch['row_padding_mask']).any()):
        raise ValueError('incomplete next-output family metadata')
    return prepare_supervision(prepare_batch(convert_batch(batch,timings),timings), vocabulary)


def prepare_audio_bounds(batch, items, timings, vocabulary):
    from maimai_baseline.time.audio_end import audio_end, duration_upper
    result = dict(batch)
    ends = [audio_end(t, int(item["audio_num_samples"])) for t, item in zip(timings, items, strict=True)]
    result["audio_num_samples"] = torch.tensor([int(i["audio_num_samples"]) for i in items])
    result["audio_end_seconds"] = torch.tensor([e.seconds for e in ends], dtype=torch.float64)
    result["audio_tail_tick"] = torch.tensor([e.tail_tick for e in ends])
    result["song_end_tick"] = torch.tensor([e.onset_tick for e in ends])
    event = batch["target_ids"]["event"]
    note = event.eq(vocabulary.event_id("NOTE")) | event.eq(vocabulary.event_id("SLIDE_BRANCH"))
    bad = note & (batch["row_anchor_tick"] > result["song_end_tick"][:, None])
    if bool(bad.any()):
        raise ValueError(f"GT onset outside audio boundary: rows={bad.nonzero().tolist()[:8]}")
    result["time_lower"] = dict(batch["time_lower"])
    result["time_upper"] = dict(batch["time_upper"])
    for field in FIELDS:
        lower = 2 if field == HOLD else 1
        result["time_lower"][field] = torch.full_like(batch["time_lower"][field], lower)
        result["time_upper"][field] = torch.tensor([
            [duration_upper(field, a, timing=t, end_seconds=e.seconds) for a in anchors.tolist()]
            for t, e, anchors in zip(timings, ends, batch["row_anchor_tick"], strict=True)
        ], dtype=torch.long)
        target = result["time_targets"][field]
        bad = target.ne(-100) & ((target < lower) | (target > result["time_upper"][field]))
        if bool(bad.any()):
            raise ValueError(f"GT duration outside audio boundary: {field}, rows={bad.nonzero().tolist()[:8]}")
    for name in ("short_hold", "short_touch_hold"):
        target = result["target_ids"]["modifier_" + name]
        bad = target.eq(1) & result["time_upper"][HOLD].lt(1)
        if bool(bad.any()):
            raise ValueError(f"GT {name} exceeds audio boundary: {bad.nonzero().tolist()[:8]}")
    return result


def prepare_batch(batch, timings):
    """Attach O(active duration rows) endpoints without touching existing tensors."""
    result = dict(batch)
    prepared = {}
    for field in FIELDS:
        target = batch["time_targets"].get(field)
        if target is None:
            continue
        indices = target.ne(-100).nonzero()
        bounds = torch.empty((len(indices), 4), dtype=torch.float64)
        for b, timing in enumerate(timings):
            selected = indices[:, 0].eq(b)
            rows = indices[selected, 1]
            value = target[b, rows]
            lower, upper = batch["time_lower"][field][b, rows], batch["time_upper"][field][b, rows]
            minimum = 2 if field == HOLD else 1
            if bool(((lower < minimum) | (value < lower) | (value > upper)).any()):
                raise ValueError("duration target outside legal tick support")
            from maimai_baseline.time.duration import cpu_tick_seconds
            bounds[selected] = tick_bounds(field, batch["row_anchor_tick"][b, rows], value, value, lower, upper,
                tick_seconds=cpu_tick_seconds(timing, batch["row_anchor_tick"][b, rows]))
        prepared[field] = {"rows": indices, "bounds": bounds}
    result["duration_bounds"] = prepared
    return result


def _moment_intervals(
    batch: Mapping[str, Any], vocabulary: CompactVocabulary,
) -> torch.Tensor:
    """Resolve visible current GT MOMENT intervals for NOTE/MOMENT supervision.

    Input rows and targets are shifted: the input MOMENT predicts the first
    NOTE; subsequent NOTE/branch inputs still belong to that same MOMENT.
    The first target delta measures a window offset and is therefore exempt.
    """
    note = batch["target_ids"]["event"].eq(vocabulary.event_id("NOTE"))
    inputs = batch["input_ids"]["event"]
    from maimai_baseline.time.grid import DELTA_MS
    delta_field = DELTA_MS
    delta = batch["input_times"][delta_field].detach().to(torch.float32)
    active = batch["input_time_active"][delta_field]
    blocks = batch["local_block_id"]
    if any(value.shape != note.shape for value in (inputs, delta, active, blocks)):
        raise ValueError("position interval inputs must align with NOTE targets")
    rows = torch.arange(note.shape[1], device=note.device).expand_as(inputs)
    header = torch.where(inputs.eq(vocabulary.event_id("MOMENT")), rows, -1)
    owner = header.cummax(dim=1).values
    safe_owner = owner.clamp_min(0)
    owned_delta = delta.gather(1, safe_owner)
    owned_block = blocks.gather(1, safe_owner)
    valid = owner.ge(0) & active.gather(1, safe_owner) & owned_block.eq(blocks) & blocks.ge(0)
    next_moment = batch["target_ids"]["event"].eq(vocabulary.event_id("MOMENT"))
    requires_owner = note | (next_moment & owner.ge(0) & blocks.ge(0))
    if bool((requires_owner & ~valid).any()):
        raise ValueError("supervision requires its visible GT MOMENT delta")
    selected = valid & owned_block.gt(0) & (
        note | next_moment
    )
    invalid_delta = owned_delta.le(0)
    if bool((selected & (~torch.isfinite(owned_delta) | invalid_delta)).any()):
        raise ValueError("non-first MOMENT interval must be a positive GT delta in its declared time units")
    return torch.where(selected, owned_delta, 100.0)



def prepare_supervision(batch, vocabulary):
    """Validate CPU GT and attach ephemeral, immutable-batch supervision inputs."""
    event = batch["target_ids"]["event"]
    kind = batch["target_ids"]["kind"]
    cabinet = batch["cabinet_id"]
    tensors = (event, kind, cabinet, batch["input_ids"]["event"],
               batch["input_times"][DELTA_MS], batch["input_time_active"][DELTA_MS], batch["local_block_id"])
    if any(value.device.type != "cpu" for value in tensors):
        raise ValueError("supervision preparation requires CPU tensors before device transfer")
    if kind.shape != event.shape or cabinet.shape != (event.shape[0],):
        raise ValueError("supervision metadata shapes do not match targets")
    if bool(((cabinet != 1) & (cabinet != 2)).any()):
        raise ValueError("kind supervision requires SD/DX cabinet metadata")
    sd = cabinet.eq(1)[:, None]
    forbidden = torch.zeros(len(vocabulary.fields["kind"].values), dtype=torch.bool)
    for name in ("touch", "touch_hold"):
        index = vocabulary.fields["kind"].encode(name)
        forbidden[index] = True
        if bool((sd & kind.eq(index)).any()):
            raise ValueError(f"SD target contains forbidden kind: {name}")
    note = event.eq(vocabulary.event_id("NOTE"))
    hold = kind.eq(vocabulary.fields["kind"].encode("hold")) | kind.eq(vocabulary.fields["kind"].encode("touch_hold"))
    slide = kind.eq(vocabulary.fields["kind"].encode("slide")) | kind.eq(vocabulary.fields["kind"].encode("wifi"))
    result = dict(batch)
    result["supervision"] = {
        "moment_ms": _moment_intervals(batch, vocabulary),
        "note_mask": note,
        "hold_mask": hold & note,
        "slide_mask": slide & note,
        "branch_mask": event.eq(vocabulary.event_id("SLIDE_BRANCH")),
        "sd_forbidden_kind": sd[:, :, None] & forbidden[None, None, :],
    }
    return result
