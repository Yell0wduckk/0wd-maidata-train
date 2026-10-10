"""Raw prefix row encoding and model decoding."""
from __future__ import annotations
from typing import Any, Mapping, Sequence
import torch
from maimai_baseline.contract import REMOVED_FIELDS
from maimai_baseline.data.route import Route71Codec
from maimai_baseline.model.route import Route71Transformer
from maimai_dataset.encoding import TIME_FIELDS
from maimai_baseline.time.coordinates import chart_tick_to_second


def encode_route_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    model: Route71Transformer,
    codec: Route71Codec,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Encode public compact rows into the factor-free Route model input."""

    from maimai_dataset.modifiers import normalize_row, row_bits, ROLES
    rows = [normalize_row(row) for row in rows]
    encoded_rows = [model.vocabulary.encode_row(row)[0] for row in rows]
    factor_values = {
        field: torch.tensor(
            [int(row[field]) for row in encoded_rows], dtype=torch.long
        )
        for field in model.vocabulary.field_names
    }
    route, _ = codec.encode_mapping(factor_values, target=False)
    input_ids = {
        field: torch.tensor(
            [[int(row[field]) for row in encoded_rows]],
            dtype=torch.long,
            device=device,
        )
        for field in model.vocabulary.field_names
        if field not in REMOVED_FIELDS
    }
    input_ids["route"] = route[None, :].to(device)
    bits = [row_bits(row) for row in rows]
    for role in ROLES:
        input_ids["modifier_" + role] = torch.tensor([[b[role] for b in bits]], device=device, dtype=torch.long)
    input_times = {
        field: torch.tensor(
            [[float(row.get(field, 0)) for row in rows]],
            dtype=torch.float32,
            device=device,
        )
        for field in TIME_FIELDS
    }
    input_time_active = {
        field: torch.tensor(
            [[field in row for row in rows]], dtype=torch.bool, device=device
        )
        for field in TIME_FIELDS
    }
    return input_ids, input_times, input_time_active


def encode_prefix_rows(rows, anchors, *, timing, model, codec, device):
    if len(rows) != len(anchors):
        raise ValueError("raw prefix rows and anchors must align")
    converted = [dict(row) for row in rows]
    for row, anchor in zip(converted, anchors, strict=True):
        if row['event'] == 'MOMENT':
            reference = anchor - row['delta_tick']
            row['delta_tick'] = (chart_tick_to_second(timing, anchor) - chart_tick_to_second(timing, reference)) * 1000
    ids, times, active = encode_route_rows(converted, model=model, codec=codec, device=device)
    times['delta_ms'] = times.pop('delta_tick')
    active['delta_ms'] = active.pop('delta_tick')
    return ids, times, active


def decode_prefix(model, *, kv_cache=None, **kwargs):
    if 'row_padding_mask' not in kwargs:
        kwargs['row_padding_mask'] = torch.zeros_like(kwargs['row_anchor_tick'], dtype=torch.bool)
    if kv_cache is not None and hasattr(kv_cache, 'audio_spans'):
        kwargs['_query_padding_cpu'] = kwargs['row_padding_mask'].cpu()
    return model.decode_prefix_rows(**kwargs) if kv_cache is None else kv_cache.decode(**kwargs)
