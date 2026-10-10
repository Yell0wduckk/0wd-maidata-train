"""Supervised single-moment rollouts; generated actions are detached conditions."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import torch

from maimai_baseline.data.route import Route71Codec
from maimai_baseline.inference.decoding import encode_route_rows, decode_prefix
from maimai_baseline.loss.common import (
    GROUP_BUDGET, GROUP_NOTE_BUDGET, apply_field_grammar_mask, discrete_position_losses,
)
from maimai_baseline.loss.l3 import NOTE_ATTRIBUTE_FIELDS, note_interval_weights, structure_interval_weights
from maimai_baseline.time.grid import candidates
from maimai_baseline.time.coordinates import chart_tick_to_second
from maimai_baseline.time.moment import candidate_probabilities


@dataclass(frozen=True)
class MomentBlock:
    batch_index: int
    start: int
    end: int
    notes: tuple[int, ...]


def choose_blocks(batch, vocabulary, *, epoch, fraction=.25, seed=20260811):
    """Stable, non-overlapping target spans, independent of model predictions."""
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("rollout fraction must be in [0,1]")
    event = batch["target_ids"]["event"].detach().cpu()
    kind = batch["target_ids"]["kind"].detach().cpu()
    moment, note = vocabulary.event_id("MOMENT"), vocabulary.event_id("NOTE")
    tap = vocabulary.fields["kind"].encode("tap")
    selected, eligible = [], 0
    for b in range(len(event)):
        heads = event[b].eq(moment).nonzero().flatten().tolist()
        options = []
        for start, end in zip(heads, heads[1:]):
            rows = tuple(range(start + 1, end))
            if rows and all(event[b, r] == note and kind[b, r] == tap for r in rows):
                options.append(MomentBlock(b, start, end, rows))
        eligible += len(options)
        identity = str(batch["segment_ids"][b])
        def digest(value):
            return hashlib.sha256(f"{seed}:{epoch}:{identity}:{value}".encode()).digest()
        # Stochastic rounding is deterministic per segment/epoch, including short windows.
        exact = len(options) * fraction
        count = int(exact) + (int.from_bytes(digest("round")[:8], "big") / 2**64 < exact % 1)
        picked = []
        for block in sorted(options, key=lambda x: digest(x.start)):
            if len(picked) >= count:
                break
            if all(block.end < p.start or p.end < block.start for p in picked):
                picked.append(block)
        selected.extend(sorted(picked, key=lambda x: x.start))
    return selected, eligible


def timing_weights(positions, blocks):
    weights = positions["delta_mask"].to(torch.float32)
    for block in blocks:
        weights[block.batch_index, block.start] = .5
        weights[block.batch_index, block.end] = .5
    return weights


def target_tick(batch, b, row):
    packed = int(batch["bpm96_row_index"][b, row])
    index = int(batch["bpm96_target_index"][b, row])
    return int(batch["bpm96_candidates"]["ticks"][packed, index])


def time_distribution(outputs, table, b=0, row=-1):
    raw = outputs["time_outputs"]
    key = "delta_ms" if "delta_ms" in raw else "delta_tick"
    candidate = dict(outputs["delta_candidate"], query=outputs["delta_candidate"]["query"][b:b+1, row])
    return candidate_probabilities(raw[key][b:b+1, row], table, candidate,
                                   torch.tensor([b], device=raw[key].device),
                                   distribution=outputs["moment_distribution"])[0][0]


def packed_time_table(batch, b, row):
    index = int(batch["bpm96_row_index"][b, row])
    return {k: v[index:index+1] for k, v in batch["bpm96_candidates"].items()}


def shifted_table(batch, b, anchor):
    grid = candidates(batch["sft_timings"][b], anchor_tick=anchor, start_tick=anchor,
                      end_tick=int(batch["end_tick"][b]), allow_zero=False,
                      song_end_tick=int(batch["song_end_tick"][b]))
    device = batch["row_anchor_tick"].device
    table = {key: torch.as_tensor(getattr(grid, key), device=device)[None]
             for key in ("ticks", "seconds", "delta_ms", "lower_ms", "upper_ms")}
    table.update(valid=torch.ones_like(table["ticks"], dtype=torch.bool),
                 first=torch.zeros(1, dtype=torch.bool, device=device))
    return table


def event_loss(outputs, target, weight):
    values, *_ = discrete_position_losses(
        {"event": outputs["field_logits"]["event"][:, -1:]},
        {"event": torch.tensor([[target]], device=weight.device)}, collect_metrics=False)
    return values["event"][0, 0] * weight


def tap_loss(outputs, batch, vocabulary, b, row, position_weight):
    from maimai_baseline.loss.budget import reject_legacy_sft
    reject_legacy_sft()
    logits = {key: value[:, -1:] for key, value in outputs["field_logits"].items()}
    targets = {key: value[b:b+1, row:row+1] for key, value in batch["target_ids"].items()}
    forbidden = torch.zeros_like(logits["kind"], dtype=torch.bool)
    if int(batch["cabinet_id"][b]) == 1:
        for kind in ("touch", "touch_hold"):
            forbidden[..., vocabulary.fields["kind"].encode(kind)] = True
    masked = apply_field_grammar_mask(logits, targets, vocabulary, sd_forbidden_kind=forbidden)
    values, _, informative, _ = discrete_position_losses(masked, targets, collect_metrics=False)
    attrs = [values[k][0, 0] for k in NOTE_ATTRIBUTE_FIELDS if k in values and bool(informative[k][0, 0])]
    if not attrs:
        raise ValueError("Tap target lacks informative attributes")
    return (GROUP_NOTE_BUDGET["kind"] * values["kind"][0, 0]
            + GROUP_NOTE_BUDGET["position"] * values["pad"][0, 0] * position_weight
            + GROUP_NOTE_BUDGET["attributes"] * torch.stack(attrs).mean())


def rollout_block(model, batch, gt_outputs, block, *, max_notes=33, collect_trace=False, forced=None):
    from maimai_baseline.loss.budget import reject_legacy_sft
    reject_legacy_sft()
    if max_notes <= 0:
        raise ValueError("max rollout notes must be positive")
    b, start, end = block.batch_index, block.start, block.end
    vocab = model.vocabulary
    device = batch["row_anchor_tick"].device
    moment, note = vocab.event_id("MOMENT"), vocab.event_id("NOTE")
    first_table = packed_time_table(batch, b, start)
    initial = time_distribution(gt_outputs, first_table, b, start)
    current = int(first_table["ticks"][0, initial.detach().argmax()])
    if forced is not None:
        current = int(forced['start_tick'])
    first_gt, end_gt = target_tick(batch, b, start), target_tick(batch, b, end)
    timing = batch["sft_timings"][b]
    delta_ms = (chart_tick_to_second(timing, current)
                - float(batch["row_anchor_second"][b, start])) * 1000
    # encode_route_rows uses the internal millisecond compatibility slot.
    rows = [{"event": "MOMENT", "delta_tick": delta_ms}]
    context = gt_outputs["audio_context"].map_audio_windows(torch.tensor([b], device=device))
    conditions = {key: batch[key][b:b+1] for key in ("difficulty_id", "level", "start_tick", "end_tick")}
    codec = Route71Codec(vocab)
    sw, pw = structure_interval_weights(batch, vocab), note_interval_weights(batch, vocab)
    struct = gt_outputs["decoded"][b, start].sum() * 0
    note_total = struct
    generated, trace = 0, []
    end_loss, end_error = None, None
    reason = ""
    while True:
        ids, times, active = encode_route_rows(rows, model=model, codec=codec, device=device)
        times['delta_ms'] = times.pop('delta_tick')
        active['delta_ms'] = active.pop('delta_tick')
        ids = {k: torch.cat((batch['input_ids'][k][b:b+1, :start+1], v), 1) for k,v in ids.items()}
        times = {k: torch.cat((batch['input_times'][k][b:b+1, :start+1], v), 1) for k,v in times.items()}
        active = {k: torch.cat((batch['input_time_active'][k][b:b+1, :start+1], v), 1) for k,v in active.items()}
        anchors = torch.cat((batch['row_anchor_tick'][b:b+1, :start+1],
                             torch.full((1, len(rows)), float(current), device=device)), 1)
        output = decode_prefix(model, audio_context=context, input_ids=ids,
            input_times=times, input_time_active=active, row_anchor_tick=anchors, **conditions)
        from maimai_baseline.model.row_fields import sample_readout
        conditional=dict(output['field_logits'])
        for stage in ('kind','position_flags'):
            conditional.update(sample_readout(model,output,dict(event='NOTE',kind='tap'),stage))
        output=dict(output,field_logits=conditional)
        cursor = min(generated, len(block.notes))
        row = block.notes[cursor] if cursor < len(block.notes) else end
        wanted = note if cursor < len(block.notes) else moment
        struct = struct + event_loss(output, wanted, sw[b, row])
        if cursor < len(block.notes):
            note_total = note_total + tap_loss(output, batch, vocab, b, row, pw[b, row])
        choices = [note] if generated == 0 else [note, moment]
        chosen = choices[int(output["field_logits"]["event"][0, -1, choices].detach().argmax())]
        if forced is not None:
            chosen = int(forced['events'][generated])
        stop = chosen == moment or generated >= max_notes
        entry = dict(gt_note_cursor=cursor, generated_notes=generated, anchor_tick=current,
                     target_event="NOTE" if wanted == note else "MOMENT",
                     chosen_event="NOTE" if chosen == note else "MOMENT", terminal=stop)
        if stop:
            reason = ("early" if generated < len(block.notes) else "natural") if chosen == moment else "limit"
            if end_gt > current:
                table = shifted_table(batch, b, current)
                indices = table["ticks"][0].eq(end_gt).nonzero().flatten()
                if len(indices):
                    logp = time_distribution(output, table)
                    end_loss = -logp[indices[0]]
                    prediction = int(table["ticks"][0, logp.detach().argmax()])
                    end_error = abs(chart_tick_to_second(timing, prediction) - chart_tick_to_second(timing, end_gt)) * 1000
                    entry["target_delta_ms"] = (chart_tick_to_second(timing, end_gt) - chart_tick_to_second(timing, current)) * 1000
            if collect_trace:
                trace.append(entry)
            break
        allowed = list(vocab.allowed_value_ids("NOTE", "pad", note_kind="tap"))
        pad = allowed[int(output["field_logits"]["pad"][0, -1, allowed].detach().argmax())]
        if forced is not None:
            pad = int(forced['pads'][generated])
        if cursor < len(block.notes):
            raw = {key: int(value[b, row]) for key, value in batch["target_ids"].items()
                   if key in vocab.fields and int(value[b, row]) >= 0}
            child = vocab.decode_row(raw)
            child["flags"] = "+".join(n for n in ("ex", "break", "short_hold")
                if int(batch["target_ids"]["modifier_" + n][b, row]) == 1)
        else:
            child = {"event": "NOTE", "kind": "tap", "flags": ""}
        child.update(event="NOTE", kind="tap", pad=vocab.fields["pad"].decode(pad))
        rows.append(child)
        generated += 1
        if collect_trace:
            entry["pad"] = child["pad"]
            trace.append(entry)
    metrics = dict(segment_id=batch["segment_ids"][b], reason=reason,
                   generated_notes=generated, extra_notes=max(0, generated-len(block.notes)),
                   unvisited_notes=max(0, len(block.notes)-min(generated+1, len(block.notes))),
                   missing_notes=max(0, len(block.notes)-generated), end_time_valid=end_loss is not None,
                   start_error_ms=abs(chart_tick_to_second(timing, current)-chart_tick_to_second(timing, first_gt))*1000,
                   end_error_ms=end_error, start_tick=first_gt, end_tick=end_gt, trace=trace)
    return note_total, struct, end_loss, metrics


def replace_objective(result, positions, blocks, rollouts):
    from maimai_baseline.loss.budget import reject_legacy_sft
    reject_legacy_sft()
    """Fixed GT denominators; the unmodified Physical graph stays in result."""
    weights = timing_weights(positions, blocks)
    time_sum = (positions["delta"] * weights).sum()
    note_change = positions["note"].sum() * 0
    event_change = positions["event"].sum() * 0
    for block, (notes, events, terminal, _) in zip(blocks, rollouts, strict=bool(rollouts)):
        b = block.batch_index
        note_change = note_change + notes - positions["note"][b, list(block.notes)].sum()
        event_change = event_change + events - positions["event"][b, block.start+1:block.end+1].sum()
        time_sum = time_sum - .5 * positions["delta"][b, block.end]
        if terminal is not None:
            time_sum = time_sum + .5 * terminal
    timing = time_sum / weights.sum().clamp_min(1)
    note_delta = note_change / positions["note_mask"].sum().clamp_min(1)
    event_delta = event_change / positions["event_mask"].sum().clamp_min(1)
    loss = (result.loss + GROUP_BUDGET["timing"] * (timing-result.components["timing"])
            + GROUP_BUDGET["note_semantic"] * note_delta + GROUP_BUDGET["structure"] * event_delta)
    from maimai_baseline.loss.common import L3LossResult
    return L3LossResult(loss=loss, metrics=result.metrics, components={**result.components,
        "timing": timing, "note_semantic": result.components["note_semantic"]+note_delta,
        "structure": result.components["structure"]+event_delta})
