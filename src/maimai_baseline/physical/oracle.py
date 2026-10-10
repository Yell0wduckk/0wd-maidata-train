from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch

from maimai_baseline.physical.contract import OPPORTUNITY_INSERT_SAME_ONSET

from maimai_baseline.physical.catalog import CompactPathCatalog
from maimai_baseline.physical.contract import ACTIVE_REGION_MASK
from maimai_baseline.physical.contract import EVENT_NOTE
from maimai_baseline.physical.contract import KIND_HOLD
from maimai_baseline.physical.contract import KIND_SLIDE
from maimai_baseline.physical.contract import KIND_TOUCH
from maimai_baseline.physical.contract import KIND_TOUCH_HOLD
from maimai_baseline.physical.contract import KIND_WIFI
from maimai_baseline.physical.contract import PATH_BITSET_WORDS
from maimai_baseline.physical.contract import PRESS_STATE
from maimai_baseline.physical.contract import SLIDE_ENTRY
from maimai_baseline.physical.contract import SLIDE_LAUNCH_GUARD
from maimai_baseline.physical.contract import SLIDE_OCCUPANCY
from maimai_baseline.physical.contract import TOUCH_STATE
from maimai_baseline.physical.contract import VALID_REGION_MASK
from maimai_baseline.physical.spatial import transform_mask


def _half_up(value: float) -> int:
    return int(math.floor(float(value) + 0.5 + 1e-8))


def _path_values(
    catalog: CompactPathCatalog, path_id: int
) -> tuple[list[int], list[float], int]:
    start = int(catalog.path_offsets[int(path_id)])
    stop = int(catalog.path_offsets[int(path_id) + 1])
    return (
        [int(value) for value in catalog.path_masks[start:stop].tolist()],
        [float(value) for value in catalog.path_enter_proportions[start:stop].tolist()],
        int(catalog.path_flags[int(path_id)]),
    )


def project_path(
    catalog: CompactPathCatalog,
    *,
    path_id: int,
    trace_start_units: float,
    motion_units: float,
    launch_slot: int,
    end_slot: int,
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    """Return occupancy intervals, public entry pulses, and natural FSM entries."""

    masks, proportions, _ = _path_values(catalog, path_id)
    boundaries: list[int] = []
    previous = int(launch_slot)
    for proportion in proportions:
        boundary = max(
            previous, _half_up(trace_start_units + proportion * motion_units)
        )
        boundaries.append(boundary)
        previous = boundary
    if boundaries:
        boundaries[0] = int(launch_slot)
    occupancy: list[tuple[int, int, int]] = []
    for index, mask in enumerate(masks):
        start = max(int(launch_slot) + 1, boundaries[index])
        stop = (
            boundaries[index + 1] if index + 1 < len(boundaries) else int(end_slot) + 1
        )
        if stop > start:
            occupancy.append((start, stop, mask))
    public_entries = [
        (boundaries[index], masks[index]) for index in range(1, len(masks))
    ]
    natural_entries = [(boundaries[index], masks[index]) for index in range(len(masks))]
    return occupancy, public_entries, natural_entries


class _CachedChartView:
    def __init__(
        self,
        cache: Mapping[str, Any],
        packed_chart: int,
        catalog: CompactPathCatalog,
    ) -> None:
        self.cache = cache
        self.chart = int(packed_chart)
        self.catalog = catalog
        self.view = int(cache["views"][self.chart])
        self.origin = int(cache["slot_origins"][self.chart])
        self.count = int(cache["slot_counts"][self.chart])
        self.plane_begin = int(cache["plane_offsets"][self.chart])
        self.action_begin = int(cache["action_offsets"][self.chart])
        self.source_begin = int(cache["source_offsets"][self.chart])
        self.source_end = int(cache["source_offsets"][self.chart + 1])
        self.moment_begin = int(cache["moment_offsets"][self.chart])

    def dense_read(self, plane: int, slot: int) -> int:
        local = int(slot) - self.origin
        if local < 0 or local >= self.count:
            return 0
        index = self.plane_begin + int(plane) * self.count + local
        return transform_mask(int(self.cache["planes"][index]), self.view)

    def action_for_source(self, source: int) -> int:
        if source < 0 or self.source_begin + source >= self.source_end:
            return -1
        return int(self.cache["source_action"][self.source_begin + source])

    def moment_slot(self, moment: int) -> int:
        return int(self.cache["moment_slots"][self.moment_begin + int(moment)])

    def carry(self, moment: int) -> list[int]:
        global_moment = self.moment_begin + int(moment)
        begin = int(self.cache["carry_offsets"][global_moment])
        end = int(self.cache["carry_offsets"][global_moment + 1])
        return [int(value) for value in self.cache["carry_indices"][begin:end].tolist()]

    def tick_second(self, tick: int) -> float:
        begin = int(self.cache["timing_offsets"][self.chart])
        end = int(self.cache["timing_offsets"][self.chart + 1])
        ticks = self.cache["timing_ticks"][begin:end]
        index = 0
        for cursor, value in enumerate(ticks.tolist()):
            if float(value) > float(tick):
                break
            index = cursor
        point_tick = float(ticks[index])
        second = float(self.cache["timing_seconds"][begin + index])
        bpm = float(self.cache["timing_bpms"][begin + index])
        return second + (float(tick) - point_tick) * 60.0 / (bpm * 96.0)

    def transformed_action(self, index: int) -> tuple[list[int], list[float]]:
        action = self.cache["action_values"][int(index)].tolist()
        action[4] = transform_mask(int(action[4]), self.view)
        floating = self.cache["action_floating"][int(index)].tolist()
        return action, floating

    def rebase_action(
        self, index: int, head_tick: int
    ) -> tuple[list[int], list[float]]:
        action, _ = self.transformed_action(index)
        duration_tick = int(action[7])
        head_second = self.tick_second(head_tick)
        head_slot = int(math.floor(head_second * 100.0 + 1e-8))
        action[6] = int(head_tick)
        action[8] = head_slot
        action[9] = head_slot + 1
        action[10] = -1
        action[11] = -1
        trace_units = motion_units = deadline_units = 0.0
        if int(action[3]) in {KIND_HOLD, KIND_TOUCH_HOLD}:
            duration_second = max(
                0.0, self.tick_second(head_tick + duration_tick) - head_second
            )
            action[9] = head_slot + max(
                1, int(math.ceil(duration_second * 100.0 - 1e-9))
            )
        if int(action[5]) >= 0:
            trace_second = self.tick_second(head_tick + 96)
            end_second = self.tick_second(head_tick + 96 + max(1, duration_tick))
            motion_second = max(0.01, end_second - trace_second)
            trace_units = trace_second * 100.0
            motion_units = motion_second * 100.0
            action[10] = _half_up(trace_units)
            action[11] = max(action[10] + 1, _half_up(end_second * 100.0))
            critical = float(self.catalog.path_critical_proportions[int(action[5])])
            deadline_units = (
                trace_second
                + critical * motion_second
                - (42.0 / 180.0 + (1.0 - critical) * motion_second / 4.0)
            ) * 100.0
        return action, [head_second * 100.0, trace_units, motion_units, deadline_units]

    def path_values(self, path_id: int) -> tuple[list[int], list[float], int]:
        masks, proportions, flags = _path_values(self.catalog, path_id)
        return (
            [transform_mask(mask, self.view) for mask in masks],
            proportions,
            flags,
        )


def _v2_path_projection(
    view: _CachedChartView,
    action: Sequence[int],
    floating: Sequence[float],
    *,
    transformed: bool,
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    path_id = int(action[5])
    masks, proportions, _ = (
        view.path_values(path_id)
        if transformed
        else _path_values(view.catalog, path_id)
    )
    launch = int(action[10])
    end_slot = int(action[11])
    boundaries: list[int] = []
    previous = launch
    for proportion in proportions:
        boundary = max(
            previous, _half_up(float(floating[1]) + proportion * float(floating[2]))
        )
        boundaries.append(boundary)
        previous = boundary
    if boundaries:
        boundaries[0] = launch
    occupancy: list[tuple[int, int, int]] = []
    for index, mask in enumerate(masks):
        start = max(launch + 1, boundaries[index])
        stop = boundaries[index + 1] if index + 1 < len(boundaries) else end_slot + 1
        if stop > start:
            occupancy.append((start, stop, mask))
    public = [(boundaries[i], masks[i]) for i in range(1, len(masks))]
    natural = [(boundaries[i], masks[i]) for i in range(len(masks))]
    return occupancy, public, natural


def _v2_action_state_at(
    view: _CachedChartView,
    action: Sequence[int],
    floating: Sequence[float],
    slot: int,
    plane: int,
    *,
    transformed_path: bool = True,
) -> int:
    event = int(action[2])
    kind = int(action[3])
    mask = int(action[4])
    head = int(action[8])
    if plane in {PRESS_STATE, TOUCH_STATE}:
        if event != EVENT_NOTE:
            return 0
        active = slot == head
        if kind in {KIND_HOLD, KIND_TOUCH_HOLD}:
            active = head <= slot < int(action[9])
        if not active:
            return 0
        touch = kind in {KIND_TOUCH, KIND_TOUCH_HOLD}
        return mask if (plane == TOUCH_STATE) == touch else 0
    path_id = int(action[5])
    if path_id < 0:
        return 0
    masks = (
        view.path_values(path_id)[0]
        if transformed_path
        else _path_values(view.catalog, path_id)[0]
    )
    if plane == SLIDE_LAUNCH_GUARD:
        return masks[0] if int(action[10]) - 5 <= slot <= int(action[10]) else 0
    occupancy, entries, _ = _v2_path_projection(
        view, action, floating, transformed=transformed_path
    )
    if plane == SLIDE_OCCUPANCY:
        return sum((mask for start, stop, mask in occupancy if start <= slot < stop), 0)
    if plane == SLIDE_ENTRY:
        return sum((mask for at, mask in entries if at == slot), 0)
    return 0


def _v2_prefix_indices(
    batch: Mapping[str, Any], opportunity: Sequence[int], view: _CachedChartView
) -> list[int]:
    batch_index = int(opportunity[0])
    row = int(opportunity[2])
    block = int(batch["local_block_id"][batch_index, row])
    sources = batch["physical_input_source_row_ids"][batch_index]
    blocks = batch["local_block_id"][batch_index]
    result: list[int] = []
    seen: set[int] = set()
    for decoder_row in range(row + 1):
        if int(blocks[decoder_row]) != block:
            continue
        index = view.action_for_source(int(sources[decoder_row]))
        if index >= 0 and index not in seen:
            result.append(index)
            seen.add(index)
    return result


def _v2_causal_read(
    view: _CachedChartView,
    plane: int,
    slot: int,
    boundary: int,
    carry: Sequence[int],
) -> int:
    if int(slot) < int(boundary):
        return view.dense_read(plane, slot)
    result = 0
    for index in carry:
        action, floating = view.transformed_action(index)
        result |= _v2_action_state_at(view, action, floating, slot, plane)
    return result


def _v2_prefix_read(
    view: _CachedChartView,
    rebased: Sequence[tuple[Sequence[int], Sequence[float]]],
    plane: int,
    slot: int,
) -> int:
    result = 0
    for action, floating in rebased:
        result |= _v2_action_state_at(view, action, floating, slot, plane)
    return result


def _v4_state_at(
    view: _CachedChartView,
    *,
    slot: int,
    boundary: int,
    carry: Sequence[int],
    prefix: Sequence[tuple[Sequence[int], Sequence[float]]],
) -> tuple[int, int, int, int]:
    return tuple(
        _v2_causal_read(view, plane, slot, boundary, carry)
        | _v2_prefix_read(view, prefix, plane, slot)
        for plane in (
            PRESS_STATE,
            TOUCH_STATE,
            SLIDE_LAUNCH_GUARD,
            SLIDE_OCCUPANCY,
        )
    )  # type: ignore[return-value]


def _v4_hand_count(press: int, touch: int, launch: int, occupancy: int) -> int:
    press &= ACTIVE_REGION_MASK
    touch &= ACTIVE_REGION_MASK
    return (
        (press | (launch & ~touch & VALID_REGION_MASK)).bit_count()
        + (occupancy & VALID_REGION_MASK).bit_count()
        + int(touch != 0)
    )


def _v4_candidate_count(
    state: tuple[int, int, int, int], *, mask: int, touch_kind: bool
) -> int:
    press, touch, launch, occupancy = state
    if touch_kind:
        touch |= int(mask)
    else:
        press |= int(mask)
    return _v4_hand_count(press, touch, launch, occupancy)


def analyze_reference_v4(
    batch: Mapping[str, Any],
    catalog: CompactPathCatalog,
    delta_query_slots: torch.Tensor,
    *, delta_opportunities: torch.Tensor | None = None, geometry_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return v4 illegal sets, visible-path bitsets, and bad delta support.

    This diagnostic oracle intentionally consumes the packed v2 cache.  The
    production loss calls the CUDA resident-cache analyzer with the same tensor
    layouts; neither path selects a modal model candidate.
    """

    if "physical_bitplane_cache" not in batch:
        raise ValueError("Physical v4 CPU oracle requires physical_bitplane_cache")
    opportunities = batch["physical_opportunity_plan"]["values"]
    if delta_opportunities is None:
        delta_opportunities = opportunities
    if (
        delta_query_slots.ndim != 2
        or delta_query_slots.shape[0] != delta_opportunities.shape[0]
    ):
        raise ValueError("Physical v4 delta query slots must be [opportunity,support]")
    query_count, width = map(int, delta_query_slots.shape)
    count = opportunities.shape[0]
    illegal = torch.zeros((count, 9), dtype=torch.long)
    illegal[:, 8] = -1
    active_paths = torch.zeros((count, PATH_BITSET_WORDS), dtype=torch.long)
    delta_bad = torch.zeros((query_count, width), dtype=torch.bool)

    cache = batch["physical_bitplane_cache"]
    for query_index in range(0 if geometry_only else query_count):
        query = delta_opportunities[query_index].tolist()
        view = _CachedChartView(cache, int(cache["batch_charts"][int(query[0])]), catalog)
        boundary = view.moment_slot(int(query[1]))
        carry = view.carry(int(query[1]))
        # Delta is predicted at the MOMENT boundary, before any action in that
        # moment.  It therefore sees only the causal chart/carry state and never
        # the note-prefix list collected for the later NOTE decision.
        for offset in range(width):
            slot = int(delta_query_slots[query_index, offset])
            if slot < 0:
                continue
            delta_state = _v4_state_at(
                view, slot=slot, boundary=boundary, carry=carry, prefix=()
            )
            delta_bad[query_index, offset] = _v4_hand_count(*delta_state) >= 2

    for op_index in range(count):
        opportunity = opportunities[op_index].tolist()
        batch_index = int(opportunity[0])
        packed_chart = int(cache["batch_charts"][batch_index])
        view = _CachedChartView(cache, packed_chart, catalog)
        moment = int(opportunity[1])
        boundary = view.moment_slot(moment)
        carry = view.carry(moment)
        prefix_indices = _v2_prefix_indices(batch, opportunity, view)
        head_tick = int(opportunity[7])
        prefix = [view.rebase_action(index, head_tick) for index in prefix_indices]
        head_slot = boundary

        state = _v4_state_at(
            view, slot=head_slot, boundary=boundary, carry=carry, prefix=prefix
        )
        base_count = _v4_hand_count(*state)
        prefix_note_count = sum(
            int(view.transformed_action(index)[0][2]) == EVENT_NOTE
            for index in prefix_indices
        )
        source_action = view.action_for_source(int(opportunity[4]))
        action: list[int] | None = None
        action_floating: list[float] | None = None
        if source_action >= 0:
            action, action_floating = view.rebase_action(source_action, head_tick)

        # Same-shape SlideTooFast applies to both a slide NOTE and a
        # SLIDE_BRANCH.  The action itself is not yet visible; only carry and
        # already-decoded current-moment actions can contribute active paths.
        if action is not None:
            candidate_path = int(action[5])
            if candidate_path >= 0 and not (
                int(catalog.path_flags[candidate_path]) & 3
            ):
                candidate_launch = int(action[10])
                prefix_set = set(prefix_indices)
                for index in (*carry, *prefix_indices):
                    old_action, _ = (
                        view.rebase_action(index, head_tick)
                        if index in prefix_set
                        else view.transformed_action(index)
                    )
                    old_path = int(old_action[5])
                    if old_path < 0 or (int(catalog.path_flags[old_path]) & 3):
                        continue
                    if (
                        int(old_action[10]) - 5 > candidate_launch
                        or int(old_action[11]) < head_slot
                    ):
                        continue
                    viewed_path = int(catalog.path_ids_by_view[view.view, old_path])
                    word_index = viewed_path // 64
                    word = int(active_paths[op_index, word_index]) & ((1 << 64) - 1)
                    word |= 1 << (viewed_path % 64)
                    if word >= 1 << 63:
                        word -= 1 << 64
                    active_paths[op_index, word_index] = word

        insertion = int(opportunity[6]) == OPPORTUNITY_INSERT_SAME_ONSET
        if insertion or action is None or int(action[2]) != EVENT_NOTE:
            if insertion and prefix_note_count > 0 and base_count >= 2:
                has_safe_non_touch_position = any(
                    _v4_candidate_count(state, mask=1 << position, touch_kind=False)
                    <= 2
                    for position in range(33)
                )
                if state[1] & ACTIVE_REGION_MASK and not has_safe_non_touch_position:
                    for kind in (0, KIND_HOLD, KIND_SLIDE, KIND_WIFI):
                        illegal[op_index, 1] |= 1 << kind
                elif (
                    not (state[1] & ACTIVE_REGION_MASK)
                    and not has_safe_non_touch_position
                ):
                    illegal[op_index, 0] = 1
            continue
        kind = int(action[3])
        touch_kind = kind in {KIND_TOUCH, KIND_TOUCH_HOLD}
        is_start = kind in {KIND_SLIDE, KIND_WIFI}
        if not geometry_only:
            if (
                prefix_note_count > 0
                and base_count >= 2
                and _v4_candidate_count(state, mask=int(action[4]), touch_kind=touch_kind)
                > 2
            ):
                if state[1] & ACTIVE_REGION_MASK:
                    for invalid_kind in (0, KIND_HOLD, KIND_SLIDE, KIND_WIFI):
                        illegal[op_index, 1] |= 1 << invalid_kind
                else:
                    illegal[op_index, 0] = 1

            # MultiTouch pad responsibility is specifically the launch-guard
            # alignment effect.  Other already-overfull prefix states belong to the
            # delta/event/kind branches and must not be charged to every position.
            press, touch, launch, occupancy = state
            no_guard_state = (press, touch, 0, occupancy)
            bad_position_mask = 0
            candidates = range(8) if is_start else range(33)
            for position in candidates:
                mask = 1 << position
                with_guard = _v4_candidate_count(state, mask=mask, touch_kind=touch_kind)
                without_guard = _v4_candidate_count(
                    no_guard_state, mask=mask, touch_kind=touch_kind
                )
                if with_guard > 2 and without_guard <= 2:
                    bad_position_mask |= mask
            illegal[op_index, 3 if is_start else 2] = bad_position_mask


        previous_note = _v2_causal_read(
            view, PRESS_STATE, head_slot - 1, boundary, carry
        ) | _v2_causal_read(view, TOUCH_STATE, head_slot - 1, boundary, carry)
        prefix_note = _v2_prefix_read(
            view, prefix, PRESS_STATE, head_slot
        ) | _v2_prefix_read(view, prefix, TOUCH_STATE, head_slot)
        current_history = 0
        # Source rows are canonical and monotone even when decoder blocks shuffle.
        for source in range(int(opportunity[5]) - 1, -1, -1):
            index = view.action_for_source(source)
            if index < 0:
                continue
            prior, floating = view.transformed_action(index)
            if int(prior[8]) < head_slot:
                break
            if int(prior[0]) >= moment or int(prior[8]) != head_slot:
                continue
            current_history |= _v2_action_state_at(
                view, prior, floating, head_slot, PRESS_STATE
            )
            current_history |= _v2_action_state_at(
                view, prior, floating, head_slot, TOUCH_STATE
            )
        overlap = (previous_note | prefix_note | current_history) & ACTIVE_REGION_MASK
        illegal[op_index, 5 if is_start else 4] = overlap & (
            0xFF if is_start else ACTIVE_REGION_MASK
        )

        entries = 0
        for slot in range(head_slot - 18, head_slot):
            entries |= _v2_causal_read(view, SLIDE_ENTRY, slot, boundary, carry)
            entries |= _v2_prefix_read(view, prefix, SLIDE_ENTRY, slot)
        illegal[op_index, 7 if is_start else 6] = entries & (
            0xFF if is_start else ACTIVE_REGION_MASK
        )

        if geometry_only:
            continue

        if kind in {KIND_HOLD, KIND_TOUCH_HOLD}:
            fixed_mask = int(action[4])
            # Match the CUDA bound: once all decoder-visible carry/prefix
            # actions have ended, no later slot can newly become overfull.
            dynamic_stop = head_slot + 1
            prefix_set = set(prefix_indices)
            for index in (*carry, *prefix_indices):
                visible_action, _ = (
                    view.rebase_action(index, head_tick)
                    if index in prefix_set
                    else view.transformed_action(index)
                )
                dynamic_stop = max(
                    dynamic_stop,
                    int(visible_action[9]),
                    int(visible_action[11]) + 1,
                )
            for slot in range(head_slot + 1, dynamic_stop):
                future = _v4_state_at(
                    view, slot=slot, boundary=boundary, carry=carry, prefix=prefix
                )
                before = _v4_hand_count(*future)
                after = _v4_candidate_count(
                    future, mask=fixed_mask, touch_kind=touch_kind
                )
                if before <= 2 < after:
                    illegal[op_index, 8] = slot
                    break

        if kind in {KIND_SLIDE, KIND_WIFI} and int(action[5]) >= 0:
            first_occupancy = int(action[10]) + 1
            future = _v4_state_at(
                view,
                slot=first_occupancy,
                boundary=boundary,
                carry=carry,
                prefix=prefix,
            )
            before = _v4_hand_count(*future)
            candidate_occupancy = _v2_action_state_at(
                view,
                action,
                action_floating or (0.0, 0.0, 0.0, 0.0),
                first_occupancy,
                SLIDE_OCCUPANCY,
            )
            after = _v4_hand_count(
                future[0], future[1], future[2], future[3] | candidate_occupancy
            )
            if before <= 2 < after:
                illegal[op_index, 1] |= 1 << kind

    return illegal, active_paths, delta_bad
