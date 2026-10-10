"""Small tensor helpers for the retained count-matrix reference analyzer."""
from typing import Mapping
import torch
from maimai_dataset.vocabulary import CompactVocabulary
from maimai_baseline.physical.catalog import pad_bit
from maimai_baseline.physical.contract import KIND_NONE
_KIND_CODES = {"tap": 0, "touch": 1, "hold": 2, "touch_hold": 3, "slide": 4, "wifi": 5}
_VOCAB_DEVICE_LOOKUPS = {}
_VOCAB_LOOKUP_OWNERS = {}


def _allowed_mask(
    vocabulary: CompactVocabulary,
    field: str,
    *,
    event: str,
    note_kind: str | None = None,
    device: torch.device,
) -> torch.Tensor:
    result = torch.zeros(
        len(vocabulary.fields[field].values), dtype=torch.bool, device=device
    )
    ids = vocabulary.allowed_value_ids(event, field, note_kind=note_kind)
    if ids:
        result[list(ids)] = True
    return result


def _vocabulary_device_lookups(
    vocabulary: CompactVocabulary, device: torch.device
) -> dict[str, torch.Tensor]:
    key = (id(vocabulary), str(device))
    cached = _VOCAB_DEVICE_LOOKUPS.get(key)
    if cached is None:
        lanes: list[int] = []
        for index in range(len(vocabulary.fields["start_pad"].values)):
            decoded = vocabulary.fields["start_pad"].decode(index)
            lanes.append(
                int(decoded[1])
                if isinstance(decoded, str)
                and len(decoded) == 2
                and decoded[0] == "A"
                and decoded[1].isdigit()
                else 0
            )
        cached = {
            "start_lane": torch.tensor(lanes, device=device, dtype=torch.long),
            "pad_mask": _pad_id_masks(vocabulary, "pad", device),
        }
        cached["kind_code"] = torch.tensor([
            _KIND_CODES.get(str(vocabulary.fields["kind"].decode(i)), KIND_NONE)
            for i in range(len(vocabulary.fields["kind"].values))
        ], device=device, dtype=torch.long)
        cached["kind_allowed"] = _allowed_mask(vocabulary, "kind", event="NOTE", device=device)
        for field in ("pad", "start_pad"):
            masks = []
            for name in _KIND_CODES:
                mask = _allowed_mask(vocabulary, field, event="NOTE", note_kind=name, device=device)
                # Inapplicable kinds use a safe singleton and are zeroed below.
                if not vocabulary.allowed_value_ids("NOTE", field, note_kind=name):
                    mask[0] = True
                masks.append(mask)
            cached[field + "_allowed"] = torch.stack(masks)
        cached["valid_start_ids"] = torch.tensor([i for i, lane in enumerate(lanes) if lane > 0], device=device)
        cached["start_mask"] = torch.tensor([1 << (lane - 1) if lane > 0 else 0 for lane in lanes], device=device)
        _VOCAB_LOOKUP_OWNERS[id(vocabulary)] = vocabulary
        _VOCAB_DEVICE_LOOKUPS[key] = cached
    return cached


def _ticks_to_seconds(
    ticks: torch.Tensor,
    batch_indices: torch.Tensor,
    timing: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    chart_indices = timing["batch_charts"].index_select(0, batch_indices)
    points = timing["point_ticks"].index_select(0, chart_indices).to(torch.float64)
    padding = timing["point_padding_mask"].index_select(0, chart_indices)
    eligible = ticks.to(torch.float64)[:, None].ge(points) & ~padding
    index = eligible.sum(dim=1).sub(1).clamp_min(0)
    point_ticks = points.gather(1, index[:, None]).squeeze(1)
    point_seconds = (
        timing["point_seconds"]
        .index_select(0, chart_indices)
        .gather(1, index[:, None])
        .squeeze(1)
    )
    bpm = (
        timing["point_bpms"]
        .index_select(0, chart_indices)
        .gather(1, index[:, None])
        .squeeze(1)
    )
    return point_seconds + (ticks.to(torch.float64) - point_ticks) * 60.0 / (bpm * 96.0)


def _pad_id_masks(
    vocabulary: CompactVocabulary, field: str, device: torch.device
) -> torch.Tensor:
    values = []
    for index in range(len(vocabulary.fields[field].values)):
        decoded = vocabulary.fields[field].decode(index)
        values.append(pad_bit("" if decoded is None else str(decoded)))
    return torch.tensor(values, device=device, dtype=torch.long)
