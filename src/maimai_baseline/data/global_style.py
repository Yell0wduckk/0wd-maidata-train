"""Reference selection and GT-only packing; consume already accepted chart rows."""
from __future__ import annotations
import hashlib
import json
import random
import torch
from maimai_baseline.data.style import prepare_style_inputs
from maimai_baseline.style_contract import GLOBAL_REFERENCE_BEATS, GLOBAL_REFERENCE_SLOTS


def stable_seed(*parts):
    data = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "little") & ((1 << 63) - 1)


def local_window_seed(seed, start_tick, policy="per-window"):
    if policy not in ("per-window", "shared-song"):
        raise ValueError("unknown local noise policy")
    return stable_seed("local-noise-v1", int(seed), int(start_tick) if policy == "per-window" else 0)


def slot_groups(records, masks=None, *, beats_per_token=GLOBAL_REFERENCE_BEATS, slots_count=GLOBAL_REFERENCE_SLOTS):
    groups = {}
    for i, record in enumerate(records):
        key = (record.get("role", "train"), str(record["record_id"]), int(record.get("view_id", 0)))
        slots = groups.setdefault(key, {})
        start = int(record.get("start_tick", record.get("target_start_tick")))
        end = int(record.get("end_tick", record.get("target_end_tick")))
        for slot in range(slots_count):
            tick = start + slot * beats_per_token * 96
            valid = bool(masks[i][slot]) if masks is not None else tick < end and tick <= int(record["song_end_tick"])
            if valid:
                slots.setdefault(tick, (i, slot, tick))
    return {key: [slots[t] for t in sorted(slots)] for key, slots in groups.items()}


def pack_reference_rows(items, vocabulary, *, beats_per_token=GLOBAL_REFERENCE_BEATS, slots=GLOBAL_REFERENCE_SLOTS):
    """Items contain only one selected four-beat span, with a TARGET_START row."""
    width = max(len(i["row_anchor_tick"]) for i in items)
    batch = {}
    for family in ("input_ids", "input_times", "input_time_active"):
        batch[family] = {}
        for name, value in items[0][family].items():
            out = torch.zeros((len(items), width), dtype=value.dtype)
            for row, item in enumerate(items):
                out[row, :len(item[family][name])] = item[family][name]
            batch[family][name] = out
    batch["row_anchor_tick"] = torch.zeros(len(items), width)
    batch["row_padding_mask"] = torch.ones(len(items), width, dtype=torch.bool)
    for row, item in enumerate(items):
        n = len(item["row_anchor_tick"])
        batch["row_anchor_tick"][row, :n] = item["row_anchor_tick"]
        batch["row_padding_mask"][row, :n] = False
    for name in ("start_tick", "end_tick", "song_end_tick"):
        batch[name] = torch.tensor([i[name] for i in items])
    return prepare_style_inputs(batch, vocabulary, beats_per_token=beats_per_token, slots=slots)


def reference_span(raw, start, end, song_end, vocabulary):
    events, anchors = raw["input_ids"]["event"], raw["row_anchor_tick"]
    marker = events.eq(vocabulary.event_id("TARGET_START")).nonzero().flatten()[0]
    rows = torch.arange(len(events))
    selected = (rows > marker) & (anchors >= start) & (anchors < end)
    selected &= ~events.eq(vocabulary.event_id("SEGMENT_END"))
    indices = torch.cat((marker.reshape(1), selected.nonzero().flatten()))
    result = {family: {name: value[indices] for name, value in raw[family].items()}
              for family in ("input_ids", "input_times", "input_time_active")}
    result.update(row_anchor_tick=anchors[indices], start_tick=start, end_tick=end, song_end_tick=song_end)
    return result
