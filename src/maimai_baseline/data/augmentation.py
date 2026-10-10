"""Deterministic permutation of notes within complete moments."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from maimai_baseline.data.segments import (
    _target_block_ids, prepare_raw_prefix_item, truncate_to_history_context
)
from maimai_dataset.vocabulary import CompactVocabulary
from typing import Any
import hashlib
import torch


# Deterministic NOTE-block augmentation


_INPUT_MAPPING_KEYS = ("input_ids", "input_times", "input_time_active")


_TARGET_MAPPING_KEYS = (
    "target_ids",
    "time_targets",
    "time_category_targets",
    "time_lower",
    "time_upper",
)


_INPUT_TENSOR_KEYS = (
    "_physical_input_source_row_ids",
    "row_anchor_tick",
    "input_position_tick",
    "input_position_active",
)


_TARGET_TENSOR_KEYS = (
    "_physical_target_source_row_ids",
    "position_target",
    "position_lower",
    "position_upper",
    "semantic_position_tick",
    "semantic_position_active",
)


def deterministic_note_permutation(
    count: int,
    *,
    seed: int,
    epoch: int,
    segment_id: str,
    moment_ordinal: int,
) -> tuple[int, ...]:
    """Return a worker-independent deterministic uniform permutation."""

    count = int(count)
    if count < 0:
        raise ValueError("note count cannot be negative")
    if count < 2:
        return tuple(range(count))
    material = (
        f"moment-order-v1\0{int(seed)}\0{int(epoch)}\0"
        f"{segment_id}\0{int(moment_ordinal)}"
    ).encode("utf-8")
    if count == 2:
        swap = hashlib.sha256(material + b"\0bernoulli").digest()[0] & 1
        return (1, 0) if swap else (0, 1)
    keys = [
        (hashlib.sha256(material + index.to_bytes(4, "little")).digest(), index)
        for index in range(count)
    ]
    return tuple(index for _, index in sorted(keys))


def _semantic_events(
    item: Mapping[str, Any], vocabulary: CompactVocabulary
) -> list[int]:
    events = [int(value) for value in item["input_ids"]["event"].tolist()]
    final_event = int(item["target_ids"]["event"][-1])
    if final_event != vocabulary.event_id("SEGMENT_END"):
        raise ValueError("moment permutation requires a final SEGMENT_END")
    return [*events, final_event]


def _note_blocks(
    events: Sequence[int],
    *,
    start: int,
    end: int,
    note_id: int,
    branch_id: int,
) -> list[list[int]]:
    blocks: list[list[int]] = []
    for position in range(start + 1, end):
        event = int(events[position])
        if event == note_id:
            blocks.append([position])
        elif event == branch_id:
            if not blocks:
                raise ValueError("SLIDE_BRANCH precedes NOTE inside a moment")
            blocks[-1].append(position)
        else:
            raise ValueError(f"unsupported row inside moment: event_id={event}")
    return blocks


def reorder_complete_moments(
    item: Mapping[str, Any],
    vocabulary: CompactVocabulary,
    *,
    seed: int,
    epoch: int,
) -> dict[str, Any]:
    """Permute NOTE blocks without adding augmentation metadata to the item."""

    events = _semantic_events(item, vocabulary)
    input_length = len(events) - 1
    order = list(range(input_length + 1))
    moment_id = vocabulary.event_id("MOMENT")
    note_id = vocabulary.event_id("NOTE")
    branch_id = vocabulary.event_id("SLIDE_BRANCH")
    target_id = vocabulary.event_id("TARGET_START")
    end_id = vocabulary.event_id("SEGMENT_END")
    boundary_ids = {moment_id, target_id, end_id}
    segment_id = str(item["segment_id"])
    moment_ordinal = -1
    position = 0
    while position < input_length:
        if int(events[position]) != moment_id:
            position += 1
            continue
        moment_ordinal += 1
        stop = position + 1
        while stop <= input_length and int(events[stop]) not in boundary_ids:
            stop += 1
        if stop > input_length:
            raise ValueError("unterminated moment in shifted sequence")
        blocks = _note_blocks(
            events,
            start=position,
            end=stop,
            note_id=note_id,
            branch_id=branch_id,
        )
        if len(blocks) >= 2:
            permutation = deterministic_note_permutation(
                len(blocks),
                seed=seed,
                epoch=epoch,
                segment_id=segment_id,
                moment_ordinal=moment_ordinal,
            )
            replacement = [position]
            for block_index in permutation:
                replacement.extend(blocks[block_index])
            if len(replacement) != stop - position:
                raise ValueError("NOTE block permutation changed moment length")
            order[position:stop] = replacement
        position = stop

    if order[0] != 0 or order[-1] != input_length:
        raise ValueError("moment permutation moved a sequence boundary")
    if sorted(order) != list(range(input_length + 1)):
        raise ValueError("moment permutation is not a row bijection")
    input_order = torch.tensor(order[:-1], dtype=torch.long)
    target_order = torch.tensor([value - 1 for value in order[1:]], dtype=torch.long)
    result = dict(item)
    for key in _INPUT_MAPPING_KEYS:
        value = item.get(key)
        if isinstance(value, Mapping):
            result[key] = {
                name: tensor.index_select(0, input_order).clone()
                for name, tensor in value.items()
            }
    for key in _TARGET_MAPPING_KEYS:
        value = item.get(key)
        if isinstance(value, Mapping):
            result[key] = {
                name: tensor.index_select(0, target_order).clone()
                for name, tensor in value.items()
            }
    for key in _INPUT_TENSOR_KEYS:
        value = item.get(key)
        if isinstance(value, torch.Tensor):
            result[key] = value.index_select(0, input_order).clone()
    for key in _TARGET_TENSOR_KEYS:
        value = item.get(key)
        if isinstance(value, torch.Tensor):
            result[key] = value.index_select(0, target_order).clone()
    return result


def prepare_augmented_item(
    item: Mapping[str, Any],
    vocabulary: CompactVocabulary,
    *,
    seed: int,
    epoch: int,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return prepare_raw_prefix_item(item, vocabulary)

    flat = truncate_to_history_context(item, vocabulary)
    shuffled = reorder_complete_moments(
        flat, vocabulary, seed=seed, epoch=epoch
    )
    shuffled["_history_layout"] = "raw-prefix-v1"
    shuffled["_local_block_id"] = _target_block_ids(shuffled, vocabulary)
    return shuffled
