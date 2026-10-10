"""Offline local attack evidence. No training imports or dataset writes at import time.

An operational sharp-attack detector, not perceptual onset ground truth. All
thresholds are relative to local amplitude, and ambiguous attacks are retained.
"""
from __future__ import annotations
from fractions import Fraction
from importlib.metadata import version
from maimai_baseline.data.store import completed, contained, publish, read_json, read_jsonl
from maimai_baseline.time.grid import identity, timing_identity
from maimai_dataset.audio import _imports, sha256_file
from maimai_dataset.timing import TimingMap
from pathlib import Path
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import find_peaks
import argparse
import csv
import hashlib
import html
import io
import json
import math
import numpy as np
import platform
import sys


RULES = {
    "schema": "maimai.local_attack_alignment.v1", "beats": [4, 8, 12],
    "search_ms": 30, "context_ms": 100, "spectral_windows_ms": [8, 16, 32],
    "spectral_hop_ms": 1, "envelope_ms": .5,
    "max_rise_ms": 6, "max_stability_width_ms": 2,
    "agreement_ms": 2, "alignment_tolerance_ms": 1,
    "candidate_relative_prominence": .12, "candidate_relative_height": .18,
    "spectral_relative_support": .12, "max_background_ratio": .08,
    "similar_candidate_ratio": .35, "minimum_clear_positions": 2,
    "onset": "baseline-intercept-of-early-power-rise;box-window-support-compensated",
    "rise_fit_power_fractions": [[.01, .03], [.03, .06], [.06, .1]],
    "timing_uncertainty_margin_ms": .25,
    "interval": "detector-sensitivity-envelope-not-statistical-confidence",
    "applied": False,
}


def spectral_curves(wave, rate):
    """Centered FFTs on a true 1ms float64 time lattice; no Mel interpolation."""
    count = wave.shape[1]
    times = np.arange(0., count / rate, .001)
    centers = np.floor(times * rate + .5).astype(int)
    result = {}
    for ms in RULES["spectral_windows_ms"]:
        size = max(4, round(rate * ms / 1000))
        offsets = np.arange(size) - size // 2
        valid = (centers + offsets[0] >= 0) & (centers + offsets[-1] < count)
        values = np.zeros(len(centers))
        indices = centers[valid, None] + offsets
        # Channel powers avoid cancellation. mono cancellation is checked separately.
        power = np.abs(np.fft.rfft(wave[:, indices] * np.hanning(size), axis=-1)) ** 2
        mag = np.sqrt(power.mean(0)) / size
        logs = np.log1p(100 * mag)
        flux = np.maximum(np.diff(logs, axis=0), 0).mean(axis=1)
        where = np.flatnonzero(valid)
        values[where[1:]] = flux
        # A difference represents the midpoint of its two frame centers.
        result[str(ms)] = {"seconds": times - .0005, "values": values, "valid": valid}
    return result


def channel_candidates(wave, rate, spectral_seeds=None):
    """Find energy-rise events independently of the expected beat position."""
    smooth = max(1, round(rate * RULES["envelope_ms"] / 1000))
    if smooth % 2 == 0:
        smooth += 1
    env = np.sqrt(np.maximum(uniform_filter1d(wave ** 2, smooth, mode="nearest"), 0))
    derivative = np.maximum(gaussian_filter1d(np.gradient(env), max(1., rate * .0002)), 0)
    maximum = float(derivative.max(initial=0))
    if maximum == 0:
        return [], env
    peaks, props = find_peaks(derivative, height=maximum * .18,
                             prominence=maximum * .12, distance=max(1, round(rate * .004)))
    strengths = props["prominences"]
    if spectral_seeds is not None:
        peaks = np.asarray([round(t * rate) for t, _ in spectral_seeds])
        strengths = np.asarray([strength for _, strength in spectral_seeds])
    candidates = []
    for seed, prominence in zip(peaks, strengths):
        # Side context is mandatory; never reflect missing audio into onset evidence.
        a, b = seed - round(rate * .018), seed + round(rate * (.006 if spectral_seeds is not None else .015))
        if a < 0 or b >= len(env):
            continue
        pre = env[a:seed - max(1, round(rate * (.006 if spectral_seeds is not None else .003)))]
        baseline = float(np.median(pre))
        noise = float(np.median(np.abs(pre - baseline)))
        peak_start = seed - round(rate * .003) if spectral_seeds is not None else seed
        peak_index = peak_start + int(np.argmax(env[peak_start:b + 1]))
        amplitude = float(env[peak_index] - baseline)
        if amplitude <= max(8 * noise, 1e-8):
            continue
        rising = env[a:peak_index + 1]
        crossings = []
        for q in (.1, .2, .3, .6, .8, .9):
            hits = np.flatnonzero(rising >= baseline + q * amplitude)
            crossings.append(a + int(hits[0]) if len(hits) else peak_index)
        # RMS has a concave square-root attack and extrapolates systematically
        # early. Fit early POWER instead, then compensate the known centered
        # box support. Fraction variations expose noisy/curved initial growth.
        power = env ** 2
        floor = float(np.median(pre ** 2))
        power_height = float(power[peak_index] - floor)
        # A preceding, completed attack inside the context is not the current
        # event's first crossing. Separate events only across a >=1ms quiet run.
        quiet = power[a:peak_index + 1] <= floor + .005 * power_height
        quiet_length, fit_start = 0, a
        for index, is_quiet in enumerate(quiet):
            quiet_length = quiet_length + 1 if is_quiet else 0
            if quiet_length >= max(1, round(rate * .001)):
                fit_start = a + index
        power_rise = power[fit_start:peak_index + 1]
        estimates = []
        for qlo, qhi in RULES["rise_fit_power_fractions"]:
            lo = fit_start + int(np.flatnonzero(power_rise >= floor + qlo * power_height)[0])
            hi = fit_start + int(np.flatnonzero(power_rise >= floor + qhi * power_height)[0])
            support = (smooth - 1) / (2 * rate)
            if hi <= lo:
                estimates.append(lo / rate + support)
                continue
            xx = np.arange(lo, hi + 1, dtype=np.float64) / rate
            slope, intercept = np.polyfit(xx - xx[0], power[lo:hi + 1] - floor, 1)
            if slope <= 0:
                continue
            estimates.append(float(xx[0] - intercept / slope + support))
        if not estimates:
            continue
        onset = float(np.median(estimates))
        # Include threshold crossing and smoothing support: a sensitivity interval,
        # not a confidence interval or a guarantee of physical onset truth.
        interval = [min(estimates) - .00025, max(estimates) + .00025]
        reasons = []
        rise_ms = (crossings[5] - crossings[0]) / rate * 1000
        if rise_ms > RULES["max_rise_ms"]:
            reasons.append("slow-or-composite-rise")
        following = env[b:min(len(env), seed + round(rate * .025))]
        if spectral_seeds is not None and float(following.max(initial=0)) > float(env[peak_index]) * 1.8:
            reasons.append("continuing-slow-or-composite-attack")
        if baseline / max(float(env[peak_index]), 1e-12) > RULES["max_background_ratio"]:
            reasons.append("background-or-preringing")
        if not a / rate <= onset <= peak_index / rate + (smooth - 1) / (2 * rate) + 1e-12:
            reasons.append("unstable-extrapolation")
        if (interval[1] - interval[0]) * 1000 > 2:
            reasons.append("unstable-rise-fit")
        candidates.append({"onset": onset, "interval": interval, "energy_peak": peak_index / rate,
                           "rise_ms": rise_ms, "strength": float(prominence), "reasons": reasons})
    return candidates, env


def detect(wave, rate, start_seconds, beat_seconds):
    """Analyze one explicitly bounded PCM excerpt, returning report and plot data."""
    wave = np.asarray(wave, dtype=np.float64)
    if wave.ndim != 2 or wave.shape[0] not in (1, 2) or not np.isfinite(wave).all() or rate <= 0:
        raise ValueError("expected finite mono/stereo PCM and positive sample rate")
    nominal_sample = math.floor(beat_seconds * 24000 + .5)
    report = {"beat_seconds": beat_seconds, "sample_rate": int(rate),
              "nominal_center_seconds": beat_seconds, "model_sample_index": nominal_sample,
              "model_center_seconds": nominal_sample / 24000,
              "sample_rounding_error_ms": (nominal_sample / 24000 - beat_seconds) * 1000,
              "read_seconds": [start_seconds, start_seconds + wave.shape[1] / rate],
              "candidates": [], "status": "missing-evidence", "reasons": [], "applied": False}
    peak = float(np.abs(wave).max(initial=0))
    if peak == 0 or wave.shape[1] < rate * .05:
        report["reasons"] = ["silence" if peak == 0 else "insufficient-context"]
        return report, {"wave": wave, "rate": rate, "start": start_seconds, "curves": {}, "envelopes": []}
    # Analysis-only scalar normalization; never changes audio or training features.
    normalized = wave / peak
    curves = spectral_curves(normalized, rate)
    channels = [*normalized, normalized.mean(0)]
    # Locate attacks with spectral change first. An energy maximum much later
    # in a kick's low-frequency body must not define the initial attack window.
    fast = curves["8"]
    top = float(fast["values"].max(initial=0))
    seed_indices, seed_props = find_peaks(fast["values"], height=top * .18,
                                         prominence=top * .12, distance=4)
    seeds = [(float(fast["seconds"][i]), float(s)) for i, s in zip(seed_indices, seed_props["prominences"])]
    found = [channel_candidates(c, rate, seeds) for c in channels]
    all_events = sorted([(c["onset"], ch, c) for ch, (events, _) in enumerate(found)
                         for c in events], key=lambda x: x[0])
    groups = []
    for event in all_events:
        if not groups or event[0] - groups[-1][0][0] > .002:
            groups.append([event])
        else:
            groups[-1].append(event)
    for group in groups:
        onset = float(np.median([e[0] for e in group])) + start_seconds
        delta = (onset - beat_seconds) * 1000
        if abs(delta) > 30:
            continue
        reasons = sorted({reason for _, _, c in group for reason in c["reasons"]})
        interval = [min(c["interval"][0] for _, _, c in group) + start_seconds,
                    max(c["interval"][1] for _, _, c in group) + start_seconds]
        evidence = {}
        for ms, curve in curves.items():
            dt = np.abs(curve["seconds"] + start_seconds - onset)
            near = dt <= int(ms) / 2000 + .002
            score = float(curve["values"][near].max(initial=0))
            maximum = float(curve["values"].max(initial=0))
            evidence[ms] = score / maximum if maximum > 0 else 0.
        if any(x < .12 for x in evidence.values()):
            reasons.append("weak-multiscale-spectral-evidence")
        present = {e[1] for e in group}
        # Missing silent stereo channel is allowed, but missing mono evidence isn't.
        active = {i for i, c in enumerate(channels) if np.max(np.abs(c), initial=0) > .1}
        if not active.issubset(present) or len(channels) - 1 not in present:
            reasons.append("channel-disagreement-or-cancellation")
        if (interval[1] - interval[0]) * 1000 > 2:
            reasons.append("channel-or-fit-instability")
        if abs(delta) >= 29 or (interval[0] - beat_seconds) * 1000 <= -30 or (interval[1] - beat_seconds) * 1000 >= 30:
            reasons.append("search-edge")
        report["candidates"].append({"onset_seconds": onset, "offset_ms": delta,
            "stability_interval_seconds": interval,
            "offset_interval_ms": [(x - beat_seconds) * 1000 for x in interval],
            "energy_peak_seconds": float(np.median([c["energy_peak"] for _, _, c in group])) + start_seconds,
            "strength": max(c["strength"] for _, _, c in group),
            "rise_ms": max(c["rise_ms"] for _, _, c in group),
            "spectral_support": evidence, "channels": sorted(present),
            "channel_evidence": [{"channel": ch, **c} for _, ch, c in group],
            "reasons": sorted(set(reasons))})
    candidates = report["candidates"]
    if not candidates:
        report["reasons"] = ["no-local-sharp-attack"]
    else:
        strongest = max(candidates, key=lambda c: c["strength"])
        significant = [c for c in candidates if c["strength"] >= strongest["strength"] * .35]
        reasons = list(strongest["reasons"])
        if len(significant) != 1:
            reasons.append("multiple-comparable-attacks")
        report.update(status="uncertain" if reasons else "clear", reasons=sorted(set(reasons)))
        if not reasons:
            report["selected"] = strongest
            low, high = strongest["offset_interval_ms"]
            report["alignment"] = "within-1ms-tolerance" if low >= -1 and high <= 1 else "offset-evidence"
    return report, {"wave": wave, "rate": rate, "start": start_seconds, "curves": curves,
                    "envelopes": [v[1] * peak for v in found]}


def aggregate(positions):
    clear = [p["selected"] for p in positions if p["status"] == "clear"]
    if len(clear) < 2:
        return {"status": "insufficient-evidence", "suggested_offset_ms": None}
    offsets = [c["offset_ms"] for c in clear]
    if max(offsets) - min(offsets) > 2:
        return {"status": "inconsistent", "suggested_offset_ms": None}
    lower = min(c["offset_interval_ms"][0] for c in clear)
    upper = max(c["offset_interval_ms"][1] for c in clear)
    return {"status": "within-1ms-tolerance" if lower >= -1 and upper <= 1 else "consistent-offset",
            "suggested_offset_ms": float(np.median(offsets)),
            "evidence_envelope_ms": [lower, upper], "clear_positions": len(clear)}


def read_excerpt(decoder, second):
    start = max(0., second - .1)
    stop = second + .1
    duration = decoder.metadata.duration_seconds
    if duration is not None:
        stop = min(float(duration), stop)
    if stop <= start:
        return None
    value = decoder.get_samples_played_in_range(start_seconds=start, stop_seconds=stop)
    return value.data.numpy(), int(value.sample_rate), float(value.pts_seconds)


def svg_plot(report, plot):
    """Standalone scientific SVG, with literal time axes and no JS dependencies."""
    width, height = 1100, 660
    beat = report["beat_seconds"]
    def x(second):
        return 75 + (second - beat + .1) * 5000
    chunks = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
              '<rect width="100%" height="100%" fill="white"/>',
              '<g font-family="sans-serif" font-size="13">',
              f'<text x="30" y="22">Beat {report["beat"]}: {html.escape(report["status"])}. Each row independently scaled; details in JSON.</text>',
              f'<text x="30" y="40">{html.escape(", ".join(report["reasons"])[:135])}</text>']
    times = plot["start"] + np.arange(plot["wave"].shape[1]) / plot["rate"]
    series = [(f"wave channel {i}", c, times, "#576f9f") for i, c in enumerate(plot["wave"])]
    if plot["envelopes"]:
        series.append(("mono RMS envelope", plot["envelopes"][-1], times, "#ce6c25"))
    for ms, curve in plot["curves"].items():
        series.append((f"spectral change {ms} ms (relative)", curve["values"],
                       curve["seconds"] + plot["start"], "#25745c"))
    for i, (label, values, ts, color) in enumerate(series):
        top = 65 + i * 85
        maximum = float(np.abs(values).max(initial=0)) or 1
        stride = max(1, len(values) // 2000)
        points = " ".join(f"{x(t):.2f},{top + 35 - v / maximum * 28:.2f}"
                          for t, v in zip(ts[::stride], values[::stride]))
        chunks += [f'<text x="5" y="{top - 5}">{label}</text>',
                   f'<polyline fill="none" stroke="{color}" stroke-width="1" points="{points}"/>']
    for delta in range(-100, 101, 10):
        xx = x(beat + delta / 1000)
        chunks += [f'<line x1="{xx}" x2="{xx}" y1="45" y2="590" stroke="#dddddd"/>',
                   f'<text x="{xx - 10}" y="610">{delta}</text>']
    for delta, color in ((0, "#b22222"), (-30, "#888"), (30, "#888")):
        xx = x(beat + delta / 1000)
        chunks.append(f'<line x1="{xx}" x2="{xx}" y1="45" y2="590" stroke="{color}" stroke-dasharray="4 3"/>')
    for c in report["candidates"]:
        left, right = [min(1075, max(75, x(t))) for t in c["stability_interval_seconds"]]
        xx = x(c["onset_seconds"])
        peak_x = x(c["energy_peak_seconds"])
        chunks += [f'<rect x="{left}" y="45" width="{max(.2, right-left)}" height="545" fill="#389660" opacity=".15"/>',
                   f'<line x1="{xx}" x2="{xx}" y1="45" y2="590" stroke="#148f45"/>',
                   f'<line x1="{peak_x}" x2="{peak_x}" y1="45" y2="310" stroke="#6757a5" stroke-dasharray="2 5"/>']
    chunks += ['<text x="75" y="640">Relative time (ms). Red: beat; green: attack; purple: local envelope peak; bands: sensitivity, not truth.</text>', '</g></svg>']
    return "\n".join(chunks).encode()


def tool_identity():
    root = Path(__file__).resolve().parent
    from maimai_baseline.source_identity import baseline_source_identity
    return {"files": {p.name: sha256_file(p) for p in (root / "attack_alignment.py",)},
            "runtime": baseline_source_identity(), "python": sys.version, "interpreter": sys.executable,
            "platform": platform.platform(),
            "dependencies": {n: version(n) for n in ("numpy", "scipy", "torch", "torchcodec", "av", "soundfile")}}


def source_item(path, expected, timing, ids, origin):
    # Actual bytes are checked in the per-source worker so batch failures are listed.
    return {"source_path": str(Path(path).resolve()), "source_sha256": expected,
            "timing": timing.to_dict(), "song_ids": sorted(ids), "origin": origin}


def pilot_items(chart_path, bpm_path, mapping_path=None):
    chart_path, bpm_path = Path(chart_path), Path(bpm_path)
    if chart_path.is_dir():
        chart_path /= "manifest.json"
    chart, bpm = read_json(chart_path), read_json(bpm_path)
    selection = chart["selection"]
    songs = selection["train_song_ids"] + selection["validation_song_ids"]
    if len(songs) != len(set(songs)) or set(songs) != set(bpm["song_ids"]):
        raise ValueError("pilot song selection mismatch")
    if bpm.get("complete") is not True or bpm["selection_sha256"] != selection["selection_sha256"]:
        raise ValueError("incomplete BPM manifest or selection identity mismatch")
    if bpm["profile_manifest_sha256"] != chart["profile_manifest_sha256"]:
        raise ValueError("pilot profile identity mismatch")
    if len(bpm["segments"]) != chart["segment_count"]:
        raise ValueError("pilot segment coverage mismatch")
    index_path = contained(chart_path.parent, chart["segment_index"]["path"])
    if sha256_file(index_path) != chart["segment_index"]["sha256"]:
        raise ValueError("pilot segment index SHA mismatch")
    rows = read_jsonl(index_path)
    if len(rows) != chart["segment_count"] or {r["segment_id"] for r in rows} != set(bpm["segments"]):
        raise ValueError("pilot exact segment selection mismatch")
    if set(bpm["segments"].values()) != set(bpm["variants"]):
        raise ValueError("BPM variant coverage mismatch")
    mapping = read_json(mapping_path) if mapping_path else {}
    if set(mapping) - set(bpm["variants"]):
        raise ValueError("unknown variant in source map")
    items, seen, covered = [], {}, set()
    for key, v in sorted(bpm["variants"].items()):
        timing = TimingMap.from_dict(v["timing"])
        if timing_identity(timing) != v["identity"]["timing"] or v["source_sha256"] != v["identity"]["source_sha256"]:
            raise ValueError(f"BPM timing/source identity mismatch: {key}")
        if identity(v["identity"]) != key:
            raise ValueError(f"BPM variant key mismatch: {key}")
        if any(str(r["song_id"]) not in v["song_ids"] for r in rows if bpm["segments"][r["segment_id"]] == key):
            raise ValueError(f"BPM segment/song mapping mismatch: {key}")
        covered.update(v["song_ids"])
        dedup = identity({"source": v["source_sha256"], "timing": timing_identity(timing)})
        # Only explicit variant -> path relocation; never choose by song number/hash alias.
        path = mapping.get(key, v["source_path"])
        if dedup in seen:
            seen[dedup]["song_ids"] = sorted(set(seen[dedup]["song_ids"]) | set(v["song_ids"]))
            seen[dedup]["origin"]["variant_ids"].append(key)
            continue
        item = source_item(path, v["source_sha256"], timing, v["song_ids"], {"variant_ids": [key],
            "original_path": v["original_path"], "original_sha256": v["identity"]["original_sha256"]})
        seen[dedup] = item
        items.append(item)
    if covered != set(songs):
        raise ValueError("variant songs do not exactly cover pilot")
    return items, {"chart_manifest_sha256": sha256_file(chart_path), "bpm_manifest_sha256": sha256_file(bpm_path),
                   "source_map_sha256": sha256_file(Path(mapping_path)) if mapping_path else None,
                   "selection_sha256": selection["selection_sha256"], "song_count": len(songs),
                   "chart_count": len({(r["song_id"], r["source_difficulty_id"]) for r in rows}),
                   "segment_index_sha256": sha256_file(index_path),
                   "segment_count": len(bpm["segments"]), "variant_count": len(bpm["variants"])}


def run_item(item, output, common):
    key = identity({"source": item["source_sha256"], "timing": timing_identity(TimingMap.from_dict(item["timing"]))})
    material = {"common": common, "item": item}
    folder = output / "sources" / key
    result_path = folder / "result.json"
    if sha256_file(Path(item["source_path"])) != item["source_sha256"]:
        raise ValueError("source SHA mismatch")
    if completed(result_path, material):
        result = read_json(result_path)
        for name, expected in result["files"].items():
            if completed(folder / name, material) is None or sha256_file(folder / name) != expected:
                raise ValueError("corrupt/missing result evidence")
        return result
    decoder = _imports()[3](item["source_path"])
    timing = TimingMap.from_dict(item["timing"])
    positions, files = [], {}
    for beat in (4, 8, 12):
        second = timing.beat_to_second(Fraction(beat))
        excerpt = read_excerpt(decoder, second)
        if excerpt is None:
            report = {"beat": beat, "beat_seconds": second, "status": "missing-evidence",
                      "reasons": ["outside-audio"], "candidates": [], "applied": False}
        else:
            wave, rate, start = excerpt
            report, plot = detect(wave, rate, start, second)
            report["beat"] = beat
            name = f"beat-{beat}.svg"
            files[name] = publish(folder / name, svg_plot(report, plot), material)["sha256"]
        positions.append(report)
    if sha256_file(Path(item["source_path"])) != item["source_sha256"]:
        raise ValueError("source changed during detection")
    result = {"key": key, "identity": material, "positions": positions, "files": files,
              "summary": aggregate(positions), "applied": False, "complete": True}
    publish(result_path, result, material)
    return result


def execute(items, inputs, output):
    output = Path(output)
    common = {"rules": RULES, "tool": tool_identity(), "inputs": inputs}
    request = {"common": common, "items": items}
    # Enforce a single immutable request per output root before any processing.
    publish(output / "request.json", request, request)
    results, failures = [], []
    for i, item in enumerate(items):
        try:
            result = run_item(item, output, common)
            results.append(result)
            print(f"{i+1}/{len(items)} {','.join(item['song_ids'])}: {result['summary']['status']}", flush=True)
        except Exception as error:
            failures.append({"source": item, "type": type(error).__name__, "error": str(error)})
            print(f"{i+1}/{len(items)} FAILED: {error}", flush=True)
    report = {"schema": RULES["schema"], "complete": not failures and len(results) == len(items),
              "applied": False, "expected_sources": len(items), "completed_sources": len(results),
              "inputs": inputs, "failures": failures, "results": [{"key": r["key"], **r["summary"]} for r in results],
              "position_status_counts": {status: sum(p["status"] == status for r in results for p in r["positions"])
                                         for status in ("clear", "uncertain", "missing-evidence")}}
    # Failed attempts are separate immutable snapshots; a successful rerun can
    # finish without rewriting earlier failure evidence or completed sources.
    final = output if report["complete"] else output / "failed-attempts" / identity(report)
    publish(final / "summary.json", report, request)
    stream = io.StringIO(newline="")
    fields = ["source_key", "song_ids", "beat", "status", "offset_ms", "lower_ms", "upper_ms", "reasons"]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for r in results:
        for p in r["positions"]:
            selected = p.get("selected", {})
            bounds = selected.get("offset_interval_ms", [None, None])
            writer.writerow(dict(source_key=r["key"], song_ids=",".join(r["identity"]["item"]["song_ids"]),
                                 beat=p["beat"], status=p["status"], offset_ms=selected.get("offset_ms"),
                                 lower_ms=bounds[0], upper_ms=bounds[1], reasons=";".join(p["reasons"])))
    publish(final / "positions.csv", stream.getvalue().encode("utf-8-sig"), request)
    chinese = (f"# 局部拍位攻击检测\n\n执行完整：{report['complete']}；完成 {len(results)}/{len(items)} 个音源/timing。\n\n"
               f"仅检查拍位 4、8、12；搜索 ±30ms，各读取 ±100ms 上下文。\n\n"
               f"位置统计：{report['position_status_counts']}；工具失败 {len(failures)}。\n\n"
               "稳定性区间是检测器敏感性范围，不是统计置信区间或真实起音标注。完整执行不代表全部对齐。\n\n"
               "正偏移表示攻击晚于原拍位。未修改 timing、音频、特征或训练数据。详见 summary.json、positions.csv 与 sources 下逐位置 SVG。\n")
    publish(final / "report.md", chinese.encode(), request)
    print(f"Invocation complete={report['complete']}; current report: {final / 'summary.json'}", flush=True)
    return 0 if report["complete"] else 1


def qualify(output, manifest=None, songs=()):
    import av
    import soundfile as sf

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    calibration = []
    for rate in (44100, 48000):
        for kind in ("noise", "phase-varied-kick"):
            for seed in range(30):
                x = np.zeros(round(.2 * rate))
                onset = round(.1 * rate)
                t = np.arange(len(x) - onset) / rate
                attack = (np.random.default_rng(seed).normal(size=len(t)) if kind == "noise"
                          else np.sin(t * 2 * np.pi * 160 + seed / 30 * 2 * np.pi))
                x[onset:] = attack * np.exp(-t / .012)
                report, _ = detect(np.stack([x, x]), rate, 0, .1)
                selected = report.get("selected")
                calibration.append({"rate": rate, "kind": kind, "seed": seed,
                    "status": report["status"], "reasons": report["reasons"],
                    "error_ms": (selected["onset_seconds"] - onset / rate) * 1000 if selected else None,
                    "interval_ms": selected["offset_interval_ms"] if selected else None})
    mp3 = []
    for rate in (44100, 48000):
        timing = TimingMap(first_seconds=.123456, bpm_points=[(Fraction(0), 300)])
        x = np.zeros(round(2.8 * rate), dtype=np.float32)
        for beat in (4, 8, 12):
            at = round((timing.beat_to_second(Fraction(beat)) + .008) * rate)
            t = np.arange(min(round(.07 * rate), len(x) - at)) / rate
            x[at:at + len(t)] = np.cos(t * 2 * np.pi * 160) * np.exp(-t / .012)
        path = output / f"codec-{rate}.mp3"
        wav = output / f"codec-{rate}.wav"
        sf.write(wav, np.stack([x, x]).T, rate, subtype="FLOAT")
        with av.open(str(path), "w") as container:
            stream = container.add_stream("libmp3lame", rate=rate)
            stream.bit_rate = 192000
            frame = av.AudioFrame.from_ndarray(np.stack([x, x]), format="fltp", layout="stereo")
            frame.sample_rate = rate
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode(None):
                container.mux(packet)
        decoder = _imports()[3](str(path))
        full = decoder.get_all_samples()
        for beat in (4, 8, 12):
            second = timing.beat_to_second(Fraction(beat))
            local, sr, start = read_excerpt(decoder, second)
            offset = round((start - float(full.pts_seconds)) * sr)
            reference = full.data.numpy()[:, offset:offset + local.shape[1]]
            error = float(np.max(np.abs(local - reference)))
            result, _ = detect(local, sr, start, second)
            with sf.SoundFile(wav) as f:
                f.seek(round(start * rate))
                original = f.read(local.shape[1], dtype="float32", always_2d=True).T
            pcm_result, _ = detect(original, rate, start, second)
            selected = result.get("selected")
            # Deliberately keep the encoded file's original timing test above.
            # A second, fixture-only probe near its known strongest attack
            # separates codec/container priming from detector localization.
            full_wave = full.data.numpy()
            lo, hi = round((second - .04) * sr), round((second + .06) * sr)
            peak_index = lo + int(np.argmax(np.abs(full_wave[0, lo:hi])))
            codec_peak_second = peak_index / sr + float(full.pts_seconds)
            shifted_wave, shifted_rate, shifted_start = read_excerpt(decoder, codec_peak_second)
            shifted, _ = detect(shifted_wave, shifted_rate, shifted_start, codec_peak_second)
            codec_onset = shifted.get("selected", {}).get("onset_seconds")
            fixture_onset = round((second + .008) * rate) / rate
            mp3.append({"rate": rate, "beat": beat, "full_local_pcm_max_error": error,
                "status": result["status"], "reasons": result["reasons"],
                "pcm_offset_ms": pcm_result.get("selected", {}).get("offset_ms"),
                "mp3_offset_ms": selected["offset_ms"] if selected else None,
                "fixture_only_shifted_probe_status": shifted["status"],
                "codec_plus_detector_shift_ms": (codec_onset - fixture_onset) * 1000 if codec_onset is not None else None,
                "codec_peak_seconds": codec_peak_second,
                "mp3_sha256": sha256_file(path), "wav_sha256": sha256_file(wav)})
        item = source_item(path, sha256_file(path), timing, [f"synthetic-mp3-{rate}"], {})
        execute([item], {"qualification": "MP3 codec fixture"}, output / f"mp3-evidence-{rate}")
    real_reports = []
    if manifest:
        if not songs or len(songs) > 3:
            raise ValueError("qualification requires 1-3 explicitly selected real songs")
        value = read_json(manifest)
        for song in songs:
            matches = [(key, v) for key, v in value["variants"].items() if song in v["song_ids"]]
            if not matches:
                raise ValueError(f"missing explicitly selected song: {song}")
            for key, v in matches:
                item = source_item(v["source_path"], v["source_sha256"], TimingMap.from_dict(v["timing"]),
                                   v["song_ids"], {"variant_ids": [key], "original_path": v["original_path"]})
                target = output / "real" / song / key
                code = execute([item], {"bpm_manifest_sha256": sha256_file(Path(manifest)), "sample_only": True}, target)
                real_reports.append({"song": song, "variant": key, "exit": code, "output": str(target)})
    clear = [r for r in calibration if r["error_ms"] is not None]
    result = {"tool": tool_identity(), "rules": RULES, "calibration": calibration, "codec": mp3,
              "real_samples": real_reports, "calibration_count": len(calibration), "clear_count": len(clear),
              "max_absolute_error_ms": max(abs(r["error_ms"]) for r in clear),
              "sensitivity_intervals_containing_truth": int(sum(r["interval_ms"][0] <= 0 <= r["interval_ms"][1] for r in clear)),
              "full_pilot_executed": False, "timing_modified": False,
              "codec_note": "Synthetic PyAV encoder/container may retain priming. Fixture-only shifted probes do not change production +/-30ms search or timing.",
              "passed": bool(clear) and all(abs(r["error_ms"]) <= 1 for r in clear)
                        and all(r["full_local_pcm_max_error"] <= 1e-7 for r in mp3)
                        and all(r["exit"] == 0 for r in real_reports)}
    (output / "qualification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("tool", "rules", "calibration", "codec")}, ensure_ascii=False))
    return 0 if result["passed"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    single = sub.add_parser("single")
    single.add_argument("--audio", type=Path, required=True)
    single.add_argument("--timing", type=Path, required=True)
    single.add_argument("--expected-audio-sha256", required=True)
    single.add_argument("--output", type=Path, required=True)
    pilot = sub.add_parser("pilot")
    pilot.add_argument("--chart-cache", type=Path, required=True)
    pilot.add_argument("--bpm-audio-manifest", type=Path, required=True)
    pilot.add_argument("--source-map", type=Path)
    pilot.add_argument("--output", type=Path, required=True)
    calibration = sub.add_parser("qualify", help="bounded synthetic and selected real calibration")
    calibration.add_argument("--output", type=Path, required=True)
    calibration.add_argument("--bpm-audio-manifest", type=Path)
    calibration.add_argument("--songs", nargs="*", default=[])
    args = parser.parse_args(argv)
    if args.mode == "qualify":
        return qualify(args.output, args.bpm_audio_manifest, args.songs)
    if args.mode == "single":
        timing = TimingMap.from_dict(read_json(args.timing))
        items = [source_item(args.audio, args.expected_audio_sha256, timing, ["single"], {})]
        inputs = {"timing_file_sha256": sha256_file(args.timing), "mode": "single"}
    else:
        items, inputs = pilot_items(args.chart_cache, args.bpm_audio_manifest, args.source_map)
    return execute(items, inputs, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
