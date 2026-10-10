"""Canonical short-note view and independent, role-preserving modifier bits."""
from __future__ import annotations

LEGACY_FIELDS = ("flags", "head_flags", "branch_flags")
HEADS = ("ex", "break", "firework", "short_hold", "short_touch_hold", "track_break")
ROLES = ("ex", "break", "firework", "short_hold", "short_touch_hold", "head_ex", "head_break", "track_break")
LEGAL = {
    "tap": ("ex", "break", "short_hold"),
    "hold": ("ex", "break"),
    "touch": ("firework", "short_touch_hold"),
    "touch_hold": ("firework",),
    "slide": ("ex", "break", "track_break"),
    "wifi": ("ex", "break", "track_break"),
}
CONTRACT = {"schema": "maimai.modifier_bits.v1", "heads": list(HEADS), "roles": list(ROLES),
    "short_max_tick": 1, "materialized_short_tick": 1, "normal_hold_min_tick": 2,
    "loss": "independent-unsmoothed-two-class-ce;each-active-head-one-attribute",
    "head_roles": "shared-ex-break-classifiers;separate-head-history",
    "each": "derived", "legal": {k: list(v) for k, v in LEGAL.items()}}

def parts(value):
    return set(value or ()) if isinstance(value, (list, tuple, set)) else set(filter(None, str(value or "").split("+")))

def normalize_row(row):
    result = dict(row)
    kind = result.get("kind")
    if result.get("event") == "NOTE" and kind in ("hold", "touch_hold"):
        duration = int(result["hold_duration_tick"])
        if duration < 0:
            raise ValueError("negative Hold duration")
        if duration <= 1:
            result["kind"] = "tap" if kind == "hold" else "touch"
            flag = "short_hold" if kind == "hold" else "short_touch_hold"
            result["flags"] = "+".join(sorted(parts(result.get("flags")) | {flag}))
            result.pop("hold_duration_tick")
    return result

def materialize_row(row):
    result = dict(row)
    flags = parts(result.get("flags"))
    for kind, flag, restored in (("tap", "short_hold", "hold"), ("touch", "short_touch_hold", "touch_hold")):
        if flag in flags:
            if result.get("kind") != kind:
                raise ValueError(f"{flag} is illegal for {result.get('kind')}")
            result["kind"], result["hold_duration_tick"] = restored, 1
            flags.remove(flag)
    if "flags" in result:
        result["flags"] = "+".join(sorted(flags))
    return result

def row_bits(row, *, normalize=True):
    """Read flag roles; generated partial rows already use normalized kinds."""
    if normalize:
        row = normalize_row(row)
    result = dict.fromkeys(ROLES, 0)
    kind = row.get("kind")
    ordinary, head, branch = (parts(row.get(n)) for n in LEGACY_FIELDS)
    allowed = LEGAL.get(kind, ()) if row.get("event") == "NOTE" else ("track_break",) if row.get("event") == "SLIDE_BRANCH" else ()
    for name in allowed:
        if name == "track_break":
            result[name] = int("break" in branch)
        elif kind in ("slide", "wifi"):
            result["head_" + name] = int(name in head)
        else:
            result[name] = int(name in ordinary)
    return result


def row_masks(row):
    """Role applicability for an explicit prepared chart record."""
    kind, event = row.get("kind"), row.get("event")
    allowed = LEGAL.get(kind, ()) if event == "NOTE" else ("track_break",) if event == "SLIDE_BRANCH" else ()
    roles = {"head_" + n if kind in ("slide", "wifi") and n in ("ex", "break") else n for n in allowed}
    return {name: int(name in roles) for name in ROLES}

def normalize_inputs(ids, times, active, vocabulary):
    """Expand immutable source tokens; explicit bits carry generated short flags."""
    import torch
    ids, times, active = dict(ids), dict(times), dict(active)
    kind, event = ids["kind"], ids["event"]
    note = event.eq(vocabulary.event_id("NOTE"))
    iskind = {k: note & kind.eq(vocabulary.fields["kind"].encode(k)) for k in LEGAL}
    duration = times["hold_duration_tick"]
    short = {k: iskind[k] & active["hold_duration_tick"] & duration.le(1) for k in ("hold", "touch_hold")}
    if bool(((short["hold"] | short["touch_hold"]) & duration.lt(0)).any()):
        raise ValueError("negative Hold duration")
    bits = {}
    for role in ROLES:
        source = "head_flags" if role.startswith("head_") else "branch_flags" if role == "track_break" else "flags"
        atom = role[5:] if role.startswith("head_") else "break" if role == "track_break" else role
        if "modifier_" + role in ids:
            bits[role] = ids["modifier_" + role]
        else:
            lookup = torch.tensor([int(atom in parts(vocabulary.fields[source].decode(i))) for i in range(len(vocabulary.fields[source].values))],
                                  device=kind.device, dtype=torch.long)
            bits[role] = lookup[ids[source].clamp_min(0)]
    bits["short_hold"] = bits["short_hold"].bool() | short["hold"]
    bits["short_touch_hold"] = bits["short_touch_hold"].bool() | short["touch_hold"]
    changed = short["hold"] | short["touch_hold"]
    ids["kind"] = torch.where(short["hold"], vocabulary.fields["kind"].encode("tap"),
                             torch.where(short["touch_hold"], vocabulary.fields["kind"].encode("touch"), kind))
    times["hold_duration_tick"] = duration.masked_fill(changed, 0)
    active["hold_duration_tick"] = active["hold_duration_tick"] & ~changed
    for role in ROLES:
        mask = torch.zeros_like(note)
        for k, legal in LEGAL.items():
            names = tuple("head_" + n if k in ("slide", "wifi") and n in ("ex", "break") else n for n in legal)
            if role in names:
                mask |= note & ids["kind"].eq(vocabulary.fields["kind"].encode(k))
        if role == "track_break":
            mask |= event.eq(vocabulary.event_id("SLIDE_BRANCH"))
        ids["modifier_" + role] = bits[role].long().masked_fill(~mask, 0)
    return ids, times, active

def prepare_batch(batch, vocabulary):
    import torch
    result = dict(batch)
    result["input_ids"], result["input_times"], result["input_time_active"] = normalize_inputs(
        batch["input_ids"], batch["input_times"], batch["input_time_active"], vocabulary)
    original = batch["target_ids"]
    # Inactive target IDs are not source vocabulary IDs.
    source = {k: v.clamp_min(0) for k, v in original.items()}
    times = {k: v.masked_fill(v.eq(-100), 0) for k, v in batch["time_targets"].items()}
    active = {k: v.ne(-100) for k, v in batch["time_targets"].items()}
    ids, _, normalized_active = normalize_inputs(source, times, active, vocabulary)
    targets = dict(original)
    targets["kind"] = ids["kind"].masked_fill(original["kind"].eq(-100), -100)
    event, kind = original["event"], targets["kind"]
    for name in HEADS:
        mask = torch.zeros_like(event, dtype=torch.bool)
        values = ids["modifier_" + name]
        for k, legal in LEGAL.items():
            selected = event.eq(vocabulary.event_id("NOTE")) & kind.eq(vocabulary.fields["kind"].encode(k))
            if name in legal:
                mask |= selected
                if name in ("ex", "break") and k in ("slide", "wifi"):
                    values = torch.where(selected, ids["modifier_head_" + name], values)
        if name == "track_break":
            mask |= event.eq(vocabulary.event_id("SLIDE_BRANCH"))
        targets["modifier_" + name] = values.masked_fill(~mask, -100)
    result["target_ids"] = targets
    result["time_targets"] = dict(batch["time_targets"])
    result["time_targets"]["hold_duration_tick"] = batch["time_targets"]["hold_duration_tick"].masked_fill(
        ~normalized_active["hold_duration_tick"], -100)
    return result
