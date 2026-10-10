from __future__ import annotations

from dataclasses import dataclass
import torch

from maimai_baseline.physical.count_matrix import INF
from maimai_baseline.physical.count_matrix import query_counts
from maimai_baseline.physical.count_matrix import query_counts_reference


@dataclass
class CountMultitouchResult:
    delta_bad: torch.Tensor
    note_bad: torch.Tensor
    branch_bad: torch.Tensor
    kind_bad: torch.Tensor
    pad_bad: torch.Tensor
    duration_tick: torch.Tensor
    duration_eligible: torch.Tensor
    first_bad_slot: torch.Tensor
    head_count: torch.Tensor
    branch_min_bad: torch.Tensor


def timing_cache(batch, device):
    if "physical_bitplane_cache" in batch:
        return batch["physical_bitplane_cache"]
    from maimai_baseline.physical.resident import active_device_cache
    return {**active_device_cache(device).tensors, "batch_charts": batch["physical_chart_indices"]}


def guard_masks(batch, opportunity, heads, counts, catalog):
    positions = batch["physical_count_positions"].clone()
    if positions.shape[0]:
        path = positions[:, 4].clamp(0, catalog.path_offsets.numel()-2)
        paths = catalog.device_tensors(opportunity.device)
        views = batch["physical_view_indices"].index_select(0, positions[:, 0])
        masks = paths["path_masks_by_view"][views, paths["path_offsets"][path]]
        positions[:, 4] = torch.where(positions[:, 5].eq(1), masks, positions[:, 4])
    if opportunity.device.type == "cuda":
        from maimai_baseline.physical.checker import load_cuda_extension
        return load_cuda_extension().count_guard(positions, batch["physical_count_position_offsets"], opportunity, heads, counts)
    result = torch.zeros(opportunity.shape[0], dtype=torch.long)
    for i, op in enumerate(opportunity.tolist()):
        press = guard = 0
        for b, first, lo, hi, mask, kind in positions.tolist():
            if b != op[0] or first > op[2] or not lo <= int(heads[i]) < hi:
                continue
            if kind:
                guard |= mask
            else:
                press |= mask
        base = int(counts[i]) + 1
        if base <= 2:
            for pad in range(33):
                if base + (guard & ~press & ~(1 << pad)).bit_count() > 2:
                    result[i] |= 1 << pad
    return result


def end_slots(ticks, duration, slide, b, timing):
    """Exact exclusive occupancy endpoint including the one-slot release tail."""
    from maimai_baseline.physical.count_reference import _ticks_to_seconds
    head = _ticks_to_seconds(ticks, b, timing)
    head_slot = torch.floor(head * 100 + 1e-7).long()
    hold_end = _ticks_to_seconds(ticks + duration, b, timing)
    hold = head_slot + torch.ceil((hold_end-head).clamp_min(0)*100-1e-9).long().clamp_min(1) + 1
    launch = torch.floor(_ticks_to_seconds(ticks+96, b, timing)*100+.5+1e-8).long()
    end = torch.floor(_ticks_to_seconds(ticks+96+duration, b, timing)*100+.5+1e-8).long()
    return torch.where(slide, torch.maximum(end, launch+1)+2, hold)


def minimum_bad_duration(ticks, slide, b, timing, first_bad, lower, upper, *, steps=32):
    """Integer monotone search; no scan over the duration probability support."""
    lo, hi = lower.clone(), upper + 1
    for _ in range(steps):
        mid = torch.div(lo+hi, 2, rounding_mode="floor")
        bad = end_slots(ticks, mid, slide, b, timing).gt(first_bad)
        lo = torch.where(bad, lo, torch.minimum(mid+1,hi))
        hi = torch.where(bad, mid, hi)
    return lo


def cuda_timing(timing):
    """Resident production metadata; device-only classification for test fixtures."""
    from maimai_baseline.physical.checker import load_cuda_extension
    keys = ("point_ticks", "point_seconds", "point_bpms", "point_padding_mask")
    if "point_constant" not in timing:
        timing["point_constant"] = load_cuda_extension().count_constant(
            *(timing[k].contiguous() for k in keys))
    return [*(timing[k].contiguous() for k in keys),
            timing["point_constant"].contiguous(), timing["batch_charts"].contiguous()]


@torch.no_grad()
def analyze_counts(batch, opportunity, representatives, delta_slots, catalog, vocabulary):
    if "physical_count_groups" not in batch:
        raise ValueError("Physical v5 requires the CPU-collated count-matrix plan")
    from maimai_baseline.physical.count_reference import _ticks_to_seconds
    from maimai_baseline.physical.count_reference import _vocabulary_device_lookups
    device = opportunity.device
    b, row, tick = opportunity[:,0], opportunity[:,2], opportunity[:,7]
    timing = timing_cache(batch, device)
    if device.type == "cuda":
        from maimai_baseline.physical.checker import load_cuda_extension
        from maimai_baseline.physical.checker import count_phase
        extension = load_cuda_extension()
        timing_args = cuda_timing(timing)
        with count_phase(batch, "count_context"):
            context = extension.count_context(tick.contiguous(), b.contiguous(), timing_args)
            head_slots = context[:,1].long()
            launch = context[:,2].long()
    else:
        head_slots = torch.floor(_ticks_to_seconds(tick,b,timing)*100+1e-7).long()
        launch = torch.floor(_ticks_to_seconds(tick+96,b,timing)*100+.5+1e-8).long()
    kind_ids = batch["target_ids"]["kind"][b,row]
    codes = _vocabulary_device_lookups(vocabulary,device)["kind_code"]
    kinds = codes[kind_ids.clamp(0,codes.numel()-1)]
    events = batch["target_ids"]["event"][b,row]
    is_note = events.eq(vocabulary.event_id("NOTE"))
    is_branch = events.eq(vocabulary.event_id("SLIDE_BRANCH"))
    insertion = opportunity[:,6].eq(2)
    hold = is_note & kinds.ge(0) & kinds.lt(4)
    short_candidate = is_note & (kinds.eq(0) | kinds.eq(1))
    slide_note = is_note & (kinds.eq(4)|kinds.eq(5))
    slide = ~hold
    starts = torch.where(hold,head_slots,launch+1)
    query_batch = {**batch,"physical_count_head_slots":torch.cat((head_slots, head_slots))}
    query = query_counts if device.type == "cuda" else query_counts_reference
    combined = query(query_batch, torch.cat((opportunity, opportunity)), representatives,
                     delta_slots, torch.cat((starts, launch+1)))
    from maimai_baseline.physical.count_matrix import CountQueries
    size = len(opportunity)
    queried = CountQueries(combined.head_count[:size], combined.delta_bad, combined.first_bad[:size])
    # A separate prefix-only query: GT kind, event and duration cannot determine
    # whether the router's entire N family is impossible.
    branch_min_bad = combined.first_bad[size:].lt(end_slots(tick, torch.ones_like(tick),
        torch.ones_like(tick, dtype=torch.bool), b, timing))
    input_event = batch["input_ids"]["event"].index_select(0,b)
    input_kind = batch["input_ids"]["kind"].index_select(0,b)
    width = input_event.shape[1]
    prefix = torch.arange(width,device=device)[None,:].le(row[:,None])
    prefix &= batch["local_block_id"].index_select(0,b).eq(batch["local_block_id"][b,row,None])
    prefix &= input_event.eq(vocabulary.event_id("NOTE"))
    input_codes = codes[input_kind.clamp(0,codes.numel()-1)]
    has_touch = (prefix & (input_codes.eq(1)|input_codes.eq(3))).any(1)
    crowded = prefix.any(1) & queried.head_count.ge(2)
    note_bad = crowded & ~has_touch
    kind_bad = torch.where(crowded & has_touch, torch.full_like(tick,53), torch.zeros_like(tick))
    lower = torch.where(hold, batch["time_lower"]["hold_duration_tick"][b,row], batch["time_lower"]["branch_duration_tick"][b,row]).long().clamp_min(1)
    upper = torch.where(hold, batch["time_upper"]["hold_duration_tick"][b,row], batch["time_upper"]["branch_duration_tick"][b,row]).long().clamp_min(1)
    lower = torch.where(short_candidate, torch.ones_like(lower), lower)
    upper = torch.where(short_candidate, torch.ones_like(upper), upper)
    # Insertion has no active duration label. The legal branch minimum is one tick.
    lower = torch.where(insertion, torch.ones_like(lower), lower)
    if device.type == "cuda":
        with count_phase(batch, "count_duration"):
            threshold, unavoidable = extension.count_duration(tick.contiguous(), b.contiguous(),
                slide.contiguous(), context, queried.first_bad, lower.contiguous(), upper.contiguous(), timing_args)
    else:
        unavoidable = queried.first_bad.lt(end_slots(tick,lower,slide,b,timing))
        threshold = minimum_bad_duration(tick,slide,b,timing,queried.first_bad,lower,upper,
            steps=int(batch.get("physical_count_duration_search_steps",32)))
        threshold = torch.where(queried.first_bad.eq(INF), upper+1, threshold)
    kind_bad |= torch.where(slide_note & unavoidable, torch.ones_like(tick) << kinds.clamp_min(0), 0)
    branch_bad = unavoidable & (is_branch | (insertion & opportunity[:,9].gt(0)))
    eligible = hold | ((slide_note | is_branch) & ~unavoidable)
    if device.type == "cuda":
        with count_phase(batch, "count_guard"):
            pads = guard_masks(batch,opportunity,head_slots,queried.head_count,catalog)
    else:
        pads = guard_masks(batch,opportunity,head_slots,queried.head_count,catalog)
    return CountMultitouchResult(queried.delta_bad,note_bad,branch_bad,kind_bad,
        pads,
        threshold,eligible,queried.first_bad,queried.head_count,branch_min_bad)
