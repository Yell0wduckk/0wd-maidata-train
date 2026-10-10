"""CPU-only GT ownership preparation. Never alter decoder labels or history."""
from __future__ import annotations
import torch
from maimai_baseline.style_contract import STYLE_SLOTS, STYLE_BEATS
from maimai_baseline.time.style import style_geometry


def prepare_style_inputs(batch, vocabulary, *, beats_per_token=STYLE_BEATS, slots=STYLE_SLOTS):
    """Pack target input rows (the omitted final row is only SEGMENT_END).

    Targets are shifted and field-masked; reading target_ids as raw vocabulary
    ids would lose information. Read complete GT input rows after TARGET_START.
    Index maps are computed before H2D, with no per-row CUDA synchronization.
    GT validity is established during dataset construction; only build the
    representation here, without rechecking markers, chronology or ownership.
    """
    from maimai_dataset.modifiers import normalize_inputs
    batch = dict(batch)
    batch['input_ids'], batch['input_times'], batch['input_time_active'] = normalize_inputs(
        batch['input_ids'], batch['input_times'], batch['input_time_active'], vocabulary)
    events = batch["input_ids"]["event"]
    if events.device.type != "cpu":
        raise ValueError("style ownership must be prepared on CPU before H2D")
    bsz, width = events.shape
    ids = {name: vocabulary.event_id(name) for name in
           ("TARGET_START", "MOMENT", "NOTE", "SLIDE_BRANCH", "SEGMENT_END")}
    chosen, note_rows, branch_rows, branch_note, note_moment = [], [], [], [], []
    moment_rows, moment_slot, moment_local, moment_delta, moment_ordinal = [], [], [], [], []
    starts = batch["start_tick"].float()
    ends = batch["end_tick"].float()
    song_ends = batch["song_end_tick"].float()
    _, valid, spans = style_geometry(starts, ends, song_ends, beats_per_token=beats_per_token, slots=slots)
    for b in range(bsz):
        marker = (events[b].eq(ids["TARGET_START"]) & ~batch["row_padding_mask"][b]).nonzero().flatten()
        current_moment = current_note = None
        last_beat = [-1.] * slots
        ordinals = [0] * slots
        for r in range(int(marker[0]) + 1, width):
            if bool(batch["row_padding_mask"][b, r]):
                break
            event = int(events[b, r])
            if event == ids["SEGMENT_END"]:
                break
            tick = float(batch["row_anchor_tick"][b, r])
            beat = (tick - float(starts[b])) / 96.
            # Consume the existing data path's validated anchors; do not impose
            # a second interval convention on the same GT rows.
            slot = int(beat // beats_per_token)
            packed = len(chosen)
            chosen.append(b * width + r)
            if event == ids["MOMENT"]:
                local = beat - slot * beats_per_token
                current_moment = len(moment_rows)
                current_note = None
                moment_rows.append(packed)
                moment_slot.append(b * slots + slot)
                moment_local.append(local)
                moment_delta.append(local - max(0., last_beat[slot]))
                moment_ordinal.append(ordinals[slot])
                ordinals[slot] += 1
                last_beat[slot] = local
            elif event == ids["NOTE"]:
                current_note = len(note_rows)
                note_rows.append(packed)
                note_moment.append(current_moment)
            elif event == ids["SLIDE_BRANCH"]:
                branch_rows.append(packed)
                branch_note.append(current_note)
    def longs(value):
        return torch.tensor(value, dtype=torch.long)
    selected = longs(chosen)
    def gather(group):
        return {name: value.flatten().index_select(0, selected).clone() for name, value in group.items()}
    return {
        "ids": gather(batch["input_ids"]), "times": gather(batch["input_times"]),
        "time_active": gather(batch["input_time_active"]),
        "note_rows": longs(note_rows), "branch_rows": longs(branch_rows),
        "branch_note": longs(branch_note), "note_moment": longs(note_moment),
        "moment_rows": longs(moment_rows), "moment_slot": longs(moment_slot),
        "moment_local": torch.tensor(moment_local, dtype=torch.float32),
        "moment_delta": torch.tensor(moment_delta, dtype=torch.float32),
        "moment_ordinal": longs(moment_ordinal),
        "moment_width": max(1, max(moment_ordinal, default=-1) + 1),
        "valid": valid, "span_beats": spans,
    }
