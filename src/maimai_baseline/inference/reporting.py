"""Optional traces and generation summaries."""

from __future__ import annotations
from collections import Counter
from contextvars import ContextVar
from functools import wraps
from maimai_baseline.time.coordinates import seconds_at_ticks
from maimai_baseline.time.duration import (
    HOLD,
    LINEAR_CONTRACT,
    LINEAR_DURATION,
    MINIMUM_SCALES,
    REFERENCES,
    distribution_contract,
)
from torch.nn import functional as F


_session = ContextVar("inference_reporting", default=None)


def trace_detail(raw, tick, log_probability, *, field, lower, anchor_tick, timing, selected_index, distribution=LINEAR_DURATION):
    distribution_contract(distribution)
    raw, log_probability = raw.detach(), log_probability.detach()
    detail = {"head": LINEAR_DURATION, "coordinate": LINEAR_CONTRACT["fields"][field]["coordinate"],
        "modal_integer_tick": tick, "bucket_ticks": 4, "bucket_id": (tick+2)//4,
        "selected_probability": float(log_probability[selected_index].exp()),
        "components": [{"pi": float(p), "center": float(c), "scale": float(s)}
            for p,c,s in zip(raw[:,0].softmax(-1), raw[:,1]*REFERENCES[field],
                F.softplus(raw[:,2])*REFERENCES[field]+MINIMUM_SCALES[field])]}
    if field == HOLD:
        ts=seconds_at_ticks(timing,[anchor_tick,anchor_tick+tick])
        detail["duration_ms"]=float((ts[1]-ts[0])*1000)
    return detail


class Reporting:
    def __init__(self, trace):
        self.trace = bool(trace)
        self.counts = Counter()
        self.row_cache = {}
        self.failures = []


def failure_record(**record):
    """Small audit trail survives trace=False; no logits or tensors are retained."""
    session = _session.get()
    if session is not None:
        session.failures.append(record)


def trace_enabled():
    session = _session.get()
    return True if session is None else session.trace


def count(name, amount=1):
    session = _session.get()
    if session is not None:
        session.counts[name] += amount


def row_cached(key, compute):
    session = _session.get()
    if session is None:
        return compute()
    if key not in session.row_cache:
        session.row_cache[key] = compute()
    return session.row_cache[key]


def begin_row():
    session = _session.get()
    if session is not None:
        session.row_cache.clear()


def top5(fields, vocabulary):
    if not trace_enabled():
        return {}
    def compute():
        import torch
        blocks, metadata = [], []
        for name, logits in fields.items():
            values, indices = torch.softmax(logits.float(),-1).topk(min(5,logits.numel()))
            blocks.append(torch.stack((indices.to(values.dtype),values),dim=-1))
            metadata.append((name,len(values)))
        packed=torch.cat(blocks).detach().cpu().tolist()
        result={};offset=0
        for name,length in metadata:
            result[name]=[dict(id=int(i),token=bool(i) if name.startswith("modifier_") else vocabulary.fields[name].decode(int(i)),probability=p)
                          for i,p in packed[offset:offset+length]]
            offset+=length
        return result
    return row_cached(('raw_top5',),compute)


def generation_session(function):
    @wraps(function)
    def run(*args, **kwargs):
        session = Reporting(kwargs.get("trace", False))
        token = _session.set(session)
        try:
            rows, traces, windows = function(*args, **kwargs)
            if windows:
                windows[-1]["operation_summary"] = dict(session.counts)
                windows[-1]["physical_failures"] = session.failures
            return rows, traces, windows
        finally:
            _session.reset(token)
    return run


def route_of(row, codec):
    if row.get("event") != "SLIDE_BRANCH" and row.get("kind") not in ("slide", "wifi"):
        return None
    return codec._factor_to_variant[(row["shape"], int(row["end_offset"]),
                                    None if "via_offset" not in row else int(row["via_offset"]))]


def distribution(rows, codec):
    kinds, durations, routes = Counter(), Counter(), Counter()
    for row in rows:
        if row.get("event") == "NOTE":
            kinds[row["kind"]] += 1
        for field in ("hold_duration_tick", "branch_duration_tick"):
            if field in row:
                durations[f"{field}:{row[field]}"] += 1
        route = route_of(row, codec)
        if route is not None:
            routes[f"B{route:03d}"] += 1
    return dict(note_kinds=dict(kinds), duration_counts=dict(durations), route_counts=dict(routes))
