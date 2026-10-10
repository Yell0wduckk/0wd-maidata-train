"""Raw history prefix preparation and target supervision ownership."""

from __future__ import annotations
from collections.abc import Mapping
from maimai_baseline.contract import HISTORY_TICKS
from maimai_dataset.vocabulary import CompactVocabulary
from typing import Any
import torch


_ROW_MAPPING_KEYS = (
    "input_ids",
    "target_ids",
    "input_times",
    "input_time_active",
    "time_targets",
    "time_category_targets",
    "time_lower",
    "time_upper",
)


_ROW_TENSOR_KEYS = (
    "_physical_input_source_row_ids",
    "_physical_target_source_row_ids",
    "row_anchor_tick",
    "input_position_tick",
    "input_position_active",
    "position_target",
    "position_lower",
    "position_upper",
    "semantic_position_tick",
    "semantic_position_active",
)


def _take_rows(item: Mapping[str, Any], indices: torch.Tensor) -> dict[str, Any]:
    result = dict(item)
    for key in _ROW_MAPPING_KEYS:
        value = item.get(key)
        if isinstance(value, Mapping):
            result[key] = {
                name: tensor.index_select(0, indices).clone()
                for name, tensor in value.items()
            }
    for key in _ROW_TENSOR_KEYS:
        value = item.get(key)
        if isinstance(value, torch.Tensor):
            result[key] = value.index_select(0, indices).clone()
    return result


def _event_positions(item: Mapping[str, Any], event_id: int) -> list[int]:
    values = item["input_ids"]["event"]
    return values.eq(int(event_id)).nonzero(as_tuple=False).flatten().tolist()


def truncate_to_history_context(
    item: Mapping[str, Any], vocabulary: CompactVocabulary
) -> dict[str, Any]:
    """Drop chart history older than the configured history interval.

    The target labels remain byte-for-byte the same.  Only masked context rows
    are removed, and the first retained MOMENT delta is recentered to the new
    context boundary so its embedding still describes the true absolute tick.
    """

    length = int(item["row_anchor_tick"].shape[0])
    target_positions = _event_positions(item, vocabulary.event_id("TARGET_START"))
    if len(target_positions) != 1:
        raise ValueError(
            f"expected one TARGET_START row, found {len(target_positions)}"
        )
    target_index = target_positions[0]
    target_start = int(round(float(item["start_tick"])))
    cutoff = max(0, target_start - HISTORY_TICKS)
    event_values = item["input_ids"]["event"]
    anchors = item["row_anchor_tick"]
    moment_id = vocabulary.event_id("MOMENT")
    first_retained: int | None = None
    for index in range(1, target_index):
        if (
            int(event_values[index]) == moment_id
            and float(anchors[index]) >= float(cutoff)
        ):
            first_retained = index
            break
    context_indices = (
        [] if first_retained is None else list(range(first_retained, target_index))
    )
    selected = torch.tensor(
        [0, *context_indices, *range(target_index, length)], dtype=torch.long
    )
    result = _take_rows(item, selected)
    result["row_anchor_tick"][0] = float(cutoff)
    new_target_index = 1 + len(context_indices)
    if context_indices:
        first_new_context = 1
        result["input_times"]["delta_tick"][first_new_context] = (
            result["row_anchor_tick"][first_new_context] - float(cutoff)
        )
        result["input_time_active"]["delta_tick"][first_new_context] = True
    result["_history_context_target_index"] = new_target_index
    result["_context_cutoff_tick"] = cutoff
    result["_original_sequence_length"] = length
    result["_history_context_sequence_length"] = int(selected.numel())
    if new_target_index >= int(result["row_anchor_tick"].shape[0]):
        raise ValueError("history-context transform removed TARGET_START")
    if int(result["input_ids"]["event"][new_target_index]) != vocabulary.event_id(
        "TARGET_START"
    ):
        raise ValueError("history-context transform lost TARGET_START alignment")
    retained_context = result["row_anchor_tick"][1:new_target_index]
    if retained_context.numel() and bool(retained_context.lt(float(cutoff)).any()):
        raise ValueError("history-context transform retained chart history before its cutoff")
    return result


def _target_block_ids(
    item: Mapping[str, Any], vocabulary: CompactVocabulary
) -> torch.Tensor:
    event = item["input_ids"]["event"]
    target_id = vocabulary.event_id("TARGET_START")
    moment_id = vocabulary.event_id("MOMENT")
    block = -2
    result = torch.full(event.shape, -2, dtype=torch.long)
    for index, value in enumerate(event.tolist()):
        if int(value) == target_id:
            block = -1
        elif int(value) == moment_id and block >= -1:
            block += 1
        if block >= -1:
            result[index] = block
    return result


def prepare_raw_prefix_item(item, vocabulary):
    flat = truncate_to_history_context(item, vocabulary)
    flat["_history_layout"] = "raw-prefix-v1"
    flat["_local_block_id"] = _target_block_ids(flat, vocabulary)
    return flat
