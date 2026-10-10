"""Causal action counts, prepared as interval rectangles and scanned on device.

The disk bitplanes are never used to count hands. Host preparation touches only
action descriptors and timing endpoints. The streamed logical [row, slot]
matrix has a bounded construction workspace and no autograd state.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Mapping

import numpy as np
import torch

TILE = 256
WORKSPACE_BYTES = 64 << 20
INF = 2**31 - 1


def action_intervals(action: np.ndarray | list[int]) -> list[tuple[int, int]]:
    """Half-open intervals, unioned per action, with one release slot."""
    a = action
    spans = []
    if int(a[2]) == 0:
        spans.append((int(a[8]), int(a[9]) + 1))
    if int(a[5]) >= 0:
        spans.append((int(a[10]) + 1, int(a[11]) + 2))
    spans.sort()
    result: list[tuple[int, int]] = []
    for lo, hi in spans:
        if hi <= lo:
            continue
        if result and lo <= result[-1][1]:
            result[-1] = (result[-1][0], max(hi, result[-1][1]))
        else:
            result.append((lo, hi))
    return result


def _chart_index(store: Any, chart: int) -> dict[str, Any]:
    indices = getattr(store, "_count_interval_indices", None)
    if indices is None:
        indices = store._count_interval_indices = {}
    if chart in indices:
        return indices[chart]
    arrays, offsets = store.arrays, store.offsets
    def part(name, offset):
        begin, end = map(int, offsets[offset][chart:chart + 2])
        return np.asarray(arrays[name][begin:end])
    actions = part("action_values", "action_offsets")
    intervals = [action_intervals(a) for a in actions]
    ends = np.array([max((hi for _, hi in spans), default=-1) for spans in intervals])
    result = {
        "actions": actions, "intervals": intervals,
        "end_prefix_max": np.maximum.accumulate(ends),
        "source_moment": part("source_moment", "source_offsets"),
        "source_action": part("source_action", "source_offsets"),
        "moment_ticks": part("moment_ticks", "moment_offsets"),
        "ticks": part("timing_ticks", "timing_offsets"),
        "seconds": part("timing_seconds", "timing_offsets"),
        "bpms": part("timing_bpms", "timing_offsets"),
    }
    indices[chart] = result
    return result


def _seconds(index, ticks):
    ticks = np.asarray(ticks, dtype=np.float64)
    at = np.searchsorted(index["ticks"], ticks, side="right") - 1
    at = np.maximum(at, 0)
    return index["seconds"][at] + (ticks - index["ticks"][at]) * 60 / (96 * index["bpms"][at])




def prepare_count_plan(batch: Mapping[str, Any], store: Any) -> dict[str, Any]:
    """Called after final CPU collation; no chart tensors enter the batch."""
    started = time.perf_counter()
    sources = batch["physical_input_source_row_ids"].numpy()
    targets = batch["physical_target_source_row_ids"].numpy()
    batch_size, rows = sources.shape
    rectangles: list[tuple[int, int, int, int]] = []
    position_intervals = []
    ranges = []
    for b in range(batch_size):
        index = _chart_index(store, int(batch["physical_chart_indices"][b]))
        valid_targets = targets[b][(targets[b] >= 0) & (targets[b] < len(index["source_moment"]))]
        target_moments = index["source_moment"][valid_targets]
        target_moments = target_moments[target_moments >= 0]
        if not len(target_moments):
            ranges.append((0, 0))
            continue
        first_moment = int(target_moments.min())
        head_ticks = np.zeros(rows, dtype=np.int64)
        row_valid = np.zeros(rows, dtype=np.bool_)
        for r, source in enumerate(targets[b]):
            if 0 <= source < len(index["source_moment"]):
                m = int(index["source_moment"][source])
                if m >= 0:
                    head_ticks[r] = index["moment_ticks"][m]
                    row_valid[r] = True
        head_seconds = _seconds(index, head_ticks)
        starts = np.floor(head_seconds * 100 + 1e-7).astype(np.int64)
        # Include hypothetical branch insertions at an input NOTE's moment too.
        input_ticks = head_ticks.copy()
        for r, source in enumerate(sources[b]):
            if 0 <= source < len(index["source_moment"]):
                m = int(index["source_moment"][source])
                if m >= 0:
                    input_ticks[r] = index["moment_ticks"][m]
                    if not row_valid[r]:
                        head_ticks[r] = input_ticks[r]
                        row_valid[r] = True
        if "row_padding_mask" in batch:
            row_valid &= ~np.asarray(batch["row_padding_mask"][b])
        if not row_valid.any():
            ranges.append((0, 0))
            continue
        head_seconds = _seconds(index, head_ticks)
        starts = np.floor(head_seconds * 100 + 1e-7).astype(np.int64)
        upper = batch["time_upper"]
        hold_upper = np.maximum(np.asarray(upper["hold_duration_tick"][b]), 1)
        slide_upper = np.maximum(np.asarray(upper["branch_duration_tick"][b]), 1)
        hold_seconds = _seconds(index, head_ticks + hold_upper)
        hold_ends = starts + np.maximum(1, np.ceil((hold_seconds - head_seconds) * 100 - 1e-9)).astype(np.int64) + 1
        slide_ends = np.floor(_seconds(index, head_ticks + 96 + slide_upper) * 100 + 0.5 + 1e-8).astype(np.int64) + 2
        insert_ends = np.floor(_seconds(index, input_ticks + 97) * 100 + 0.5 + 1e-8).astype(np.int64) + 3
        anchor = np.asarray(batch["row_anchor_second"][b], dtype=np.float64)
        song_end = int(batch["song_end_tick"][b])
        table = batch["bpm96_candidates"]
        seconds, valid = np.asarray(table["seconds"]), np.asarray(table["valid"])
        row_index = np.asarray(batch["bpm96_row_index"][b])
        delta_lo = np.floor(anchor * 100 + 1e-7).astype(np.int64)
        delta_hi = delta_lo.copy()
        for r in range(rows):
            packed = int(row_index[r])
            if packed >= 0 and valid[packed].any():
                slots = np.floor(seconds[packed, valid[packed]] * 100 + 1e-7).astype(np.int64)
                delta_lo[r], delta_hi[r] = slots.min(), slots.max()
        query_lo = int(min(starts[row_valid].min(), delta_lo[row_valid].min(),
                           np.floor(_seconds(index,input_ticks[row_valid])*100+1e-7).min()))
        query_hi = int(max(hold_ends[row_valid].max(), slide_ends[row_valid].max(), insert_ends[row_valid].max(), delta_hi[row_valid].max() + 1))
        visible = {}
        for r, source in enumerate(sources[b]):
            if source >= 0:
                visible.setdefault(int(source), r)
        local = []
        actions = index["actions"]
        # Prefix maximum makes ancient, already-ended actions cheap to skip.
        begin = int(np.searchsorted(index["end_prefix_max"], query_lo, side="right"))
        visible_moments = [first_moment-1, *(int(index["source_moment"][source])
            for source in sources[b] if 0 <= source < len(index["source_moment"]))]
        stop = int(np.searchsorted(actions[:,0], max(visible_moments), side="right"))
        for ai in range(begin, stop):
            action = actions[ai]
            moment, source = map(int, action[:2])
            first_row = 0 if moment < first_moment else visible.get(source)
            if first_row is None:
                continue
            from maimai_baseline.physical.spatial import transform_mask
            view = int(batch["physical_view_indices"][b])
            if int(action[2]) == 0:
                position_intervals.append((b, first_row, int(action[8]), int(action[9]) + 1,
                    transform_mask(int(action[4]), view), 0))
            if int(action[5]) >= 0:
                # Guard alignment is against the start pad, not the path occupancy.
                # Branch has no press mask; recover its owner's lane from the catalog
                # independently at binding time through a resident path start lookup.
                position_intervals.append((b, first_row, int(action[10])-5, int(action[10])+1,
                    int(action[5]), 1))
            for lo, hi in index["intervals"][ai]:
                lo, hi = max(lo, query_lo), min(hi, query_hi)
                if hi > lo:
                    local.append((b, first_row, lo, hi))
        lo = min((x[2] for x in local), default=0)
        hi = max((x[3] for x in local), default=0)
        ranges.append((lo, hi))
        rectangles.extend(local)
    result = pack_count_plan(rows, ranges, rectangles, time.perf_counter() - started)
    result["physical_count_duration_search_steps"] = max(1, int(max(batch["time_upper"]["hold_duration_tick"].max(), batch["time_upper"]["branch_duration_tick"].max(), 1)).bit_length() + 1)
    result["physical_count_positions"] = torch.tensor(position_intervals, dtype=torch.long).reshape(-1,6)
    sizes = np.bincount([x[0] for x in position_intervals], minlength=batch_size)
    result["physical_count_position_offsets"] = torch.tensor(np.r_[0,np.cumsum(sizes)], dtype=torch.long)
    result["physical_count_prepare_seconds"] = time.perf_counter() - started
    return result


def pack_count_plan(rows, ranges, rectangles, prepare_seconds=0.0):
    tiles = []
    offsets = [0]
    for b, (lo, hi) in enumerate(ranges):
        tiles.extend((b, start, min(TILE, hi - start)) for start in range(lo, hi, TILE))
        offsets.append(len(tiles))
    updates = [[] for _ in tiles]
    for b, first, lo, hi in rectangles:
        if not 0 <= first < rows:
            raise ValueError("count rectangle visibility row outside decoder")
        origin = ranges[b][0]
        for relative in range((lo - origin) // TILE, (hi - 1 - origin) // TILE + 1):
            tile = offsets[b] + relative
            start = tiles[tile][1]
            updates[tile].extend(((first * (TILE + 1) + max(lo - start, 0), 1),
                                  (first * (TILE + 1) + min(hi - start, TILE), -1)))
    flat_indices = []; values = []; update_offsets = [0]
    for tile, changes in enumerate(updates):
        for index, value in changes:
            flat_indices.append(tile * rows * (TILE + 1) + index)
            values.append(value)
        update_offsets.append(len(values))
    # Budget scans, suffix-min values/indices, count storage and query slack.
    per_tile = max(1, rows * (TILE + 1) * 64)
    if per_tile > WORKSPACE_BYTES:
        raise ValueError("decoder row count exceeds count-matrix workspace capacity")
    take = max(1, WORKSPACE_BYTES // per_tile)
    groups = [(start, min(start + take, len(tiles)), update_offsets[start],
               update_offsets[min(start + take, len(tiles))]) for start in range(0, len(tiles), take)]
    return {
        "physical_count_rectangles": torch.tensor(rectangles, dtype=torch.long).reshape(-1,4),
        "physical_count_ranges": torch.tensor(ranges, dtype=torch.long).reshape(-1, 2),
        "physical_count_tile_offsets": torch.tensor(offsets, dtype=torch.long),
        "physical_count_tiles": torch.tensor(tiles, dtype=torch.long).reshape(-1, 3),
        "physical_count_update_indices": torch.tensor(flat_indices, dtype=torch.long),
        "physical_count_update_values": torch.tensor(values, dtype=torch.int32),
        "physical_count_groups": tuple(groups),
        "physical_count_prepare_seconds": float(prepare_seconds),
        "physical_count_rows": int(rows),
        "physical_count_peak_bytes": min(take, len(tiles)) * per_tile,
    }


def count_tiles(batch):
    """Yield bounded pieces of the logical matrix; works on CPU for tests."""
    rows = int(batch["physical_count_rows"])
    indices = batch["physical_count_update_indices"]
    values = batch["physical_count_update_values"]
    stride = rows * (TILE + 1)
    for begin, end, u0, u1 in batch["physical_count_groups"]:
        difference = torch.zeros((end - begin, rows, TILE + 1), dtype=torch.int32, device=indices.device)
        difference.view(-1).index_add_(0, indices[u0:u1] - begin * stride, values[u0:u1])
        temporal = difference.cumsum(2, dtype=torch.int32)
        del difference
        causal = temporal.cumsum(1, dtype=torch.int32)
        del temporal
        counts = causal[:, :, :TILE].clamp(0, 15).to(torch.uint8)
        del causal
        yield begin, end, counts


def _gather(counts, begin, end, batch, b, row, slot):
    ranges = batch["physical_count_ranges"]
    lo, hi = ranges[b, 0], ranges[b, 1]
    tile = batch["physical_count_tile_offsets"][b] + torch.div(slot - lo, TILE, rounding_mode="floor")
    valid = (slot >= lo) & (slot < hi) & (tile >= begin) & (tile < end)
    local = (tile - begin).clamp(0, end - begin - 1)
    col = (slot - lo).remainder(TILE)
    result = counts[local, row, col]
    return torch.where(valid, result, 0)


@dataclass
class CountQueries:
    head_count: torch.Tensor
    delta_bad: torch.Tensor
    first_bad: torch.Tensor


def query_counts_legacy(batch, opportunity, representatives, delta_slots, start_slots):
    n = opportunity.shape[0]
    device = opportunity.device
    head = torch.zeros(n, dtype=torch.uint8, device=device)
    delta = torch.zeros_like(delta_slots, dtype=torch.bool)
    first = torch.full((n,), INF, device=device, dtype=torch.long)
    b, row = opportunity[:, 0], opportunity[:, 2]
    qb = representatives[:, 0, None].expand_as(delta_slots)
    # MOMENT's shifted prediction owns this row, before current-moment NOTE inputs.
    qr = representatives[:, 3, None].clamp_min(0).expand_as(delta_slots)
    for begin, end, counts in count_tiles(batch):
        head = torch.maximum(head, _gather(counts, begin, end, batch, b, row, batch["physical_count_head_slots"]))
        delta |= _gather(counts, begin, end, batch, qb, qr, delta_slots).ge(2) & delta_slots.ge(0)
        tiles = batch["physical_count_tiles"][begin:end]
        slots = (tiles[:, 1, None, None] + torch.arange(TILE, device=device)[None, None, :]).int()
        valid_columns = torch.arange(TILE, device=device)[None, None, :] < tiles[:, 2, None, None]
        bad = torch.where(counts.ge(2) & valid_columns, slots, INF)
        following = bad.flip(2).cummin(2).values.flip(2)
        # Bound the tile x opportunity gather as well as the matrix buffers.
        for q0 in range(0,n,max(1,int(batch["physical_count_rows"]))):
            q1 = min(n,q0+int(batch["physical_count_rows"]))
            starts = start_slots[q0:q1]
            columns = (starts[None,:] - tiles[:,1,None]).clamp(0,TILE-1)
            candidate = following[torch.arange(end-begin,device=device)[:,None],row[None,q0:q1],columns]
            valid = tiles[:,0,None].eq(b[None,q0:q1]) & (starts[None,:] < tiles[:,1,None]+tiles[:,2,None])
            first[q0:q1] = torch.minimum(first[q0:q1],torch.where(valid,candidate,INF).amin(0))
    return CountQueries(head, delta, first)


def query_counts(batch, opportunity, representatives, delta_slots, start_slots):
    if opportunity.device.type != "cuda":
        return query_counts_legacy(batch, opportunity, representatives, delta_slots, start_slots)
    from maimai_baseline.physical.checker import load_cuda_extension
    from maimai_baseline.physical.checker import count_phase
    extension = load_cuda_extension()
    op, reps = opportunity.contiguous(), representatives.contiguous()
    slots, starts = delta_slots.contiguous(), start_slots.contiguous()
    head = torch.zeros(op.shape[0], dtype=torch.uint8, device=op.device)
    delta = torch.zeros_like(slots, dtype=torch.bool)
    first = torch.full((op.shape[0],), INF, dtype=torch.long, device=op.device)
    if not batch["physical_count_groups"]:
        return CountQueries(head, delta, first)
    with count_phase(batch, "count_index"):
        index = extension.count_index(op, reps, slots,
            batch["physical_count_head_slots"].contiguous(), batch["physical_count_ranges"],
            batch["physical_count_tile_offsets"], batch["physical_count_tiles"].shape[0])
    pieces = iter(count_tiles(batch))
    for _ in batch["physical_count_groups"]:
        with count_phase(batch, "count_build"):
            begin, end, counts = next(pieces)
        with count_phase(batch, "count_query"):
            extension.count_query(counts.contiguous(), batch["physical_count_tiles"],
                batch["physical_count_tile_offsets"], op, starts, index,
                head, delta, first, begin, end)
    return CountQueries(head, delta, first)


def direct_counts(rows, lo, hi, rectangles):
    """Independent scalar CPU reference, deliberately no difference or scans."""
    result = torch.zeros((rows, hi - lo), dtype=torch.uint8)
    for r in range(rows):
        for s in range(lo, hi):
            result[r, s-lo] = min(15, sum(first <= r and start <= s < stop for first,start,stop in rectangles))
    return result


def query_counts_reference(batch, opportunity, representatives, delta_slots, start_slots):
    """Scalar oracle: count intervals and sweep endpoints, with no matrix scans."""
    rectangles = batch["physical_count_rectangles"].tolist()
    def visible(b, row):
        return [(lo,hi) for rb,first,lo,hi in rectangles if rb==b and first<=row]
    def count(spans, slot):
        return min(15,sum(lo<=slot<hi for lo,hi in spans))
    heads, first = [], []
    for i,op in enumerate(opportunity.tolist()):
        spans = visible(op[0],op[2])
        heads.append(count(spans,int(batch["physical_count_head_slots"][i])))
        start = int(start_slots[i])
        points = sorted({start, *(max(lo,start) for lo,hi in spans if hi>start)})
        first.append(next((s for s in points if count(spans,s)>=2),INF))
    delta = torch.zeros_like(delta_slots,dtype=torch.bool)
    for i,op in enumerate(representatives.tolist()):
        spans = visible(op[0],op[3])
        for j,slot in enumerate(delta_slots[i].tolist()):
            delta[i,j] = slot>=0 and count(spans,slot)>=2
    return CountQueries(torch.tensor(heads,dtype=torch.uint8),delta,torch.tensor(first,dtype=torch.long))
