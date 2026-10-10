from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
from itertools import zip_longest
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


TARGET_VERSIONS = (
    "maimai DX FESTiVAL",
    "maimai DX FESTiVAL PLUS",
    "maimai DX BUDDiES",
    "maimai DX BUDDiES PLUS",
    "maimai DX PRiSM",
    "maimai DX PRiSM PLUS",
    "maimai DX CiRCLE",
    "maimai DX CiRCLE PLUS",
)

PATH_RE = re.compile(r"^([1-8])(pp|qq|[-^v<>szw]|p|q)([1-8])$")
V_PATH_RE = re.compile(r"^([1-8])V([1-8])([1-8])$")
PAD_RE = re.compile(r"^([A-Z]+)([1-8])$")


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    workspace = script.parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Audit complete Slide routes directly from five-digit raw maidata files."
        )
    )
    parser.add_argument(
        "--corpus", type=Path, default=workspace / "data" / "extracted"
    )
    parser.add_argument(
        "--miacode",
        type=Path,
        default=workspace / "tools" / "MiaCode" / "build" / "Release" / "miacode_chart_ir.exe",
    )
    parser.add_argument(
        "--slide-data",
        type=Path,
        default=workspace / "tools" / "MiaCode" / "assets" / "reference" / "slide_data.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script.parents[1] / "artifacts" / "raw-slide-routes-festival-plus-v1",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar_metadata(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
        for line in stream:
            if line.startswith("&inote_"):
                break
            if not line.startswith("&") or "=" not in line:
                continue
            key, value = line[1:].rstrip("\r\n").split("=", 1)
            result[key] = value
    return result


def selected_songs(corpus: Path) -> list[dict[str, Any]]:
    songs: list[dict[str, Any]] = []
    for directory in sorted(corpus.iterdir(), key=lambda value: value.name):
        # Exactly five decimal digits: this excludes the six-digit entertainment
        # charts as well as legacy three/four-digit song directories.
        if not directory.is_dir() or re.fullmatch(r"\d{5}", directory.name) is None:
            continue
        maidata = directory / "maidata.txt"
        if not maidata.is_file():
            continue
        metadata = scalar_metadata(maidata)
        version = metadata.get("version", "")
        if version not in TARGET_VERSIONS:
            continue
        songs.append(
            {
                "song_id": directory.name,
                "shortid": metadata.get("shortid", ""),
                "title": metadata.get("title", ""),
                "version": version,
                "cabinet": metadata.get("cabinet", ""),
                "maidata": maidata,
            }
        )
    return songs


def dump_song(
    song: Mapping[str, Any], executable: Path, timeout_seconds: int
) -> dict[str, Any]:
    command = [
        str(executable),
        "dump",
        "--maidata",
        str(song["maidata"]),
        "--difficulty",
        "ALL",
        "--output",
        "-",
    ]
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        return {"song": dict(song), "fatal_error": f"timeout: {error}"}
    stderr = completed.stderr.decode("utf-8", errors="replace").strip()
    if completed.returncode != 0:
        return {
            "song": dict(song),
            "fatal_error": f"exit={completed.returncode}: {stderr}",
        }
    try:
        bundle = json.loads(completed.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        return {"song": dict(song), "fatal_error": f"invalid JSON: {error}; {stderr}"}
    return {"song": dict(song), "bundle": bundle, "stderr": stderr}


def rotate_digit(value: int, shift: int) -> int:
    return ((value - 1 + shift) % 8) + 1


def rotate_path(path: str, shift: int) -> str:
    return re.sub(
        r"[1-8]", lambda match: str(rotate_digit(int(match.group(0)), shift)), path
    )


def rotate_pad(pad: str, shift: int) -> str:
    match = PAD_RE.fullmatch(pad)
    if match is None:
        return pad
    return f"{match.group(1)}{rotate_digit(int(match.group(2)), shift)}"


def build_caret_aliases(slides: Mapping[str, Any]) -> dict[str, str]:
    by_signature: dict[str, list[str]] = defaultdict(list)
    for key, resource in slides.items():
        signature = json.dumps(
            resource.get("judge_sequence", []),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        by_signature[signature].append(str(key))
    aliases: dict[str, str] = {}
    for key, resource in slides.items():
        if "^" not in key:
            continue
        signature = json.dumps(
            resource.get("judge_sequence", []),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        start, end = key[0], key[-1]
        candidates = [
            candidate
            for candidate in by_signature[signature]
            if "^" not in candidate
            and candidate.startswith(start)
            and candidate.endswith(end)
        ]
        if len(candidates) != 1:
            raise ValueError(f"caret alias is ambiguous for {key}: {candidates}")
        aliases[str(key)] = candidates[0]
    return aliases


def factor_path(path: str) -> str:
    match_v = V_PATH_RE.fullmatch(path)
    if match_v is not None:
        start, via, end = (int(match_v.group(index)) for index in (1, 2, 3))
        return f"V:{(via - start) % 8}:{(end - start) % 8}"
    match = PATH_RE.fullmatch(path)
    if match is None:
        raise ValueError(f"unsupported Slide path: {path!r}")
    start, shape, end = int(match.group(1)), match.group(2), int(match.group(3))
    return f"{shape}:{(end - start) % 8}"


def merged_judge_sequence(
    segment_paths: Sequence[str],
    slides: Mapping[str, Any],
    wifi: Mapping[str, Any],
) -> tuple[list[list[str]], list[str]]:
    merged: list[list[str]] = []
    missing: list[str] = []
    for path in segment_paths:
        resource = slides.get(path) or wifi.get(path)
        if resource is None:
            missing.append(path)
            continue
        sequence = [
            [str(pad) for pad in area]
            for area in resource.get("judge_sequence", [])
        ]
        if not sequence and resource.get("tri_judge_sequence"):
            # Wi-Fi is three simultaneous paths rather than a chained route.
            # Preserve one time-ordered signature by unioning the three lane
            # areas at each judge ordinal; chain statistics remain unaffected.
            lanes = resource["tri_judge_sequence"]
            sequence = [
                [str(pad) for area in ordinal for pad in area]
                for ordinal in zip_longest(*lanes, fillvalue=[])
            ]
        if not sequence:
            missing.append(path)
            continue
        if merged and merged[-1] == sequence[0]:
            merged.extend(sequence[1:])
        else:
            merged.extend(sequence)
    return merged, missing


def trajectory_signature(sequence: Sequence[Sequence[str]], shift: int) -> str:
    normalized = [
        sorted(rotate_pad(str(pad), shift) for pad in area) for area in sequence
    ]
    return ">".join("/".join(area) for area in normalized)


def iter_notes(chart: Mapping[str, Any]) -> Iterable[tuple[Mapping[str, Any], int]]:
    for moment in chart.get("moments", []):
        tick = int(moment.get("tick", 0))
        for group in moment.get("groups", []):
            for note in group.get("notes", []):
                yield note, tick


def extract_occurrences(
    result: Mapping[str, Any],
    slides: Mapping[str, Any],
    wifi: Mapping[str, Any],
    caret_aliases: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    song = result["song"]
    bundle = result["bundle"]
    occurrences: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    counts = Counter(charts=0, slide_notes=0, slide_branches=0, wifi_branches=0)
    for failure in bundle.get("failures", []):
        failures.append(
            {
                "song_id": song["song_id"],
                "title": song["title"],
                "version": song["version"],
                "difficulty": failure.get("difficulty", ""),
                "error": failure.get("error", ""),
                "scope": "difficulty",
            }
        )
    for chart in bundle.get("charts", []):
        counts["charts"] += 1
        difficulty = str(chart.get("difficulty", ""))
        chart_key = f"{song['song_id']}:{difficulty}"
        for note, tick in iter_notes(chart):
            kind = str(note.get("kind", ""))
            if kind not in {"slide", "wifi"}:
                continue
            counts["slide_notes"] += 1
            branches = note.get("branches", [])
            for branch_index, branch in enumerate(branches):
                segments = [
                    str(segment.get("path", ""))
                    for segment in branch.get("segments", [])
                    if str(segment.get("path", ""))
                ]
                if not segments:
                    failures.append(
                        {
                            "song_id": song["song_id"],
                            "title": song["title"],
                            "version": song["version"],
                            "difficulty": difficulty,
                            "error": "Slide branch has no path segments",
                            "scope": "branch",
                        }
                    )
                    continue
                start = int(segments[0][0])
                shift = 1 - start
                aliased = [caret_aliases.get(path, path) for path in segments]
                rotated = [rotate_path(path, shift) for path in aliased]
                try:
                    factors = tuple(factor_path(path) for path in rotated)
                except ValueError as error:
                    failures.append(
                        {
                            "song_id": song["song_id"],
                            "title": song["title"],
                            "version": song["version"],
                            "difficulty": difficulty,
                            "error": str(error),
                            "scope": "branch",
                        }
                    )
                    continue
                judge_sequence, missing = merged_judge_sequence(segments, slides, wifi)
                readable_route = "|".join(rotated)
                raw = str(note.get("source", {}).get("raw", ""))
                occurrences.append(
                    {
                        "song_id": song["song_id"],
                        "title": song["title"],
                        "version": song["version"],
                        "difficulty": difficulty,
                        "chart_key": chart_key,
                        "tick": tick,
                        "branch_index": branch_index,
                        "kind": kind,
                        "raw": raw,
                        "segment_count": len(segments),
                        "raw_segments": "|".join(segments),
                        "canonical_segments": readable_route,
                        "factor_route": "|".join(factors),
                        "judge_area_count": len(judge_sequence),
                        "trajectory_signature": trajectory_signature(judge_sequence, shift),
                        "resource_missing": "|".join(missing),
                        "timing_mode": str(branch.get("timing_mode", "")),
                    }
                )
                counts["slide_branches"] += 1
                if kind == "wifi":
                    counts["wifi_branches"] += 1
    return occurrences, failures, dict(counts)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def histogram(values: Iterable[int]) -> dict[str, int]:
    return {
        str(key): value for key, value in sorted(Counter(int(x) for x in values).items())
    }


def nearest_rank(values: Sequence[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(int(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def route_rows(occurrences: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in occurrences:
        grouped[(str(row["kind"]), str(row["factor_route"]))].append(row)
    rows: list[dict[str, Any]] = []
    for (kind, factor_route), members in grouped.items():
        example = members[0]
        areas = [int(member["judge_area_count"]) for member in members]
        songs = {str(member["song_id"]) for member in members}
        charts = {str(member["chart_key"]) for member in members}
        trajectories = {str(member["trajectory_signature"]) for member in members}
        rows.append(
            {
                "kind": kind,
                "factor_route": factor_route,
                "canonical_segments": example["canonical_segments"],
                "segment_count": int(example["segment_count"]),
                "occurrences": len(members),
                "song_count": len(songs),
                "chart_count": len(charts),
                "trajectory_type_count": len(trajectories),
                "judge_areas_min": min(areas),
                "judge_areas_mean": round(sum(areas) / len(areas), 4),
                "judge_areas_max": max(areas),
                "example_song_id": example["song_id"],
                "example_title": example["title"],
                "example_difficulty": example["difficulty"],
                "example_raw": example["raw"],
                "example_trajectory": example["trajectory_signature"],
            }
        )
    rows.sort(
        key=lambda row: (
            int(row["segment_count"]),
            -int(row["occurrences"]),
            str(row["factor_route"]),
        )
    )
    return rows


def policy_table(routes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    chains = [row for row in routes if int(row["segment_count"]) >= 2]
    total = sum(int(row["occurrences"]) for row in chains)
    policies: list[dict[str, Any]] = []
    observed_max = max((int(row["segment_count"]) for row in chains), default=2)
    for max_segments in sorted({2, 3, 4, 5, 6, 8, observed_max}):
        for min_occurrences in (1, 2, 3, 5, 10):
            selected = [
                row
                for row in chains
                if int(row["segment_count"]) <= max_segments
                and int(row["occurrences"]) >= min_occurrences
            ]
            covered = sum(int(row["occurrences"]) for row in selected)
            policies.append(
                {
                    "max_segments": max_segments,
                    "min_occurrences": min_occurrences,
                    "route_types": len(selected),
                    "covered_occurrences": covered,
                    "chain_coverage": round(covered / total, 6) if total else 0.0,
                }
            )
    return policies


def markdown_report(summary: Mapping[str, Any], routes: Sequence[Mapping[str, Any]]) -> str:
    counts = summary["counts"]
    quantiles = summary["chain_quantiles"]
    buckets = summary["recommended_length_buckets"]
    frequency = summary["chain_frequency"]
    overlap = summary["trajectory_overlap"]
    version_rows = "\n".join(
        f"| {version} | {row['songs']} | {row['charts']} | {row['branches']} | {row['chains']} |"
        for version, row in summary["by_version"].items()
    )
    segment_rows = "\n".join(
        f"| {segments} | {amount} |"
        for segments, amount in summary["segment_count_occurrence_histogram"].items()
    )
    top = sorted(
        (row for row in routes if int(row["segment_count"]) >= 2),
        key=lambda row: (-int(row["occurrences"]), str(row["factor_route"])),
    )[:30]
    top_rows = "\n".join(
        f"| {row['factor_route']} | {row['segment_count']} | {row['occurrences']} | "
        f"{row['song_count']} | {row['example_song_id']} {row['example_difficulty']} |"
        for row in top
    )
    return f"""# FESTiVAL 及以后原始 maidata Slide 路线统计

本统计直接读取 `data/extracted/<五位数ID>/maidata.txt`，由 MiaCode 原生 Simai
解析器即时解析完整 Slide branch。未读取现有训练/验证 Chart IR；六位数娱乐型谱面和
FESTiVAL 以前版本均未进入样本。

## 范围

- 歌曲：{counts['songs']} 首
- 成功解析谱面：{counts['charts']} 张
- 谱面级解析失败：{counts['difficulty_failures']} 张
- Slide/Wi-Fi note：{counts['slide_notes']} 个
- 完整 branch：{counts['branches']} 条
- 单段基础 branch：{counts['base_branches']} 条
- 拼接 branch：{counts['chain_branches']} 条
- 资源定义的基础语义类型：{counts['expected_base_route_types']} 种；本范围实际出现
  {counts['observed_base_route_types']} 种（缺少 `{', '.join(counts['missing_base_routes'])}`）
- 拼接路线类型（旋转归一化，`^` 等价化）：{counts['chain_route_types']} 种
- 拼接判定区轨迹类型（旋转归一化）：{counts['chain_trajectory_types']} 种
- 所有拼接均使用整体时长：{counts['chain_total_timing_branches']}/{counts['chain_branches']}

## 版本分布

| 版本 | 歌曲 | 谱面 | branch | 拼接 branch |
|---|---:|---:|---:|---:|
{version_rows}

## 拼接长度

| 基础段数 | branch 数 |
|---:|---:|
{segment_rows}

拼接 branch 的段数分位数：P50={quantiles['segments']['p50']}、
P90={quantiles['segments']['p90']}、P95={quantiles['segments']['p95']}、
P99={quantiles['segments']['p99']}、max={quantiles['segments']['max']}。
合并后的判定区长度分位数：P50={quantiles['judge_areas']['p50']}、
P90={quantiles['judge_areas']['p90']}、P95={quantiles['judge_areas']['p95']}、
P99={quantiles['judge_areas']['p99']}、max={quantiles['judge_areas']['max']}。

按段数 P99 的下一档作为“极长”边界，本数据建议：

- 基础：1 段，{counts['base_branches']} 条；
- 短/常规拼接：2–{buckets['regular_max_segments']} 段，
  {buckets['regular_occurrences']} 条，占拼接 {buckets['regular_chain_coverage']:.2%}；
- 极长拼接：≥{buckets['extreme_min_segments']} 段，
  {buckets['extreme_occurrences']} 条、{buckets['extreme_route_types']} 种，
  仅占拼接 {buckets['extreme_chain_coverage']:.2%}。

但“短”不等于“高频”：{frequency['singleton_route_types']} 个拼接路线类型只出现一次，
对应 {frequency['singleton_occurrences']} 条 branch（占拼接
{frequency['singleton_chain_coverage']:.2%}）。因此若完整路线直接作为离散类别，长尾仍然很重；
短拼接更适合由基础段/判定区序列组合编码。

判定区层面确有语义碰撞：{overlap['chain_route_types_equivalent_to_base']} 种拼接路线、
{overlap['chain_occurrences_equivalent_to_base']} 条实际 branch 与某个基础形状拥有相同的
旋转归一化判定区轨迹。统计仍保留 `segment_count`/路线身份，没有把它们合并成基础形状。

## 最常见拼接路线

`factor_route` 的每段写作 `shape:end_offset`；V 形写作
`V:via_offset:end_offset`。每段都相对自身起点，因此整体旋转不改变键。

| factor_route | 段数 | 次数 | 歌曲数 | 示例 |
|---|---:|---:|---:|---|
{top_rows}

## 文件

- `summary.json`：完整计数、分位数、候选截断策略及输入指纹。
- `selected_songs.csv`：实际纳入的五位数歌曲及 maidata SHA-256。
- `occurrences.csv`：每条 branch 的来源、原始 Simai、完整段列表和判定区轨迹。
- `chain_branches.csv`：只含全部拼接 branch 的轻量缓存，后续复核无需重扫 maidata。
- `route_vocab.csv`：旋转归一化后的完整路线词表及频次。
- `extreme_examples.csv`：最长路线的可核对样例。
- `parse_failures.csv`：解析器或 branch 资源异常；为空也保留表头。

这里没有把“常见”硬编码成一个结论。`summary.json` 的 `candidate_policies`
枚举了最大段数和最小出现次数组合，可用覆盖率选择 embedding 词表边界，避免先验阈值
掩盖真实长尾。
"""


def main() -> int:
    args = parse_args()
    corpus = args.corpus.resolve()
    executable = args.miacode.resolve()
    slide_data_path = args.slide_data.resolve()
    output_dir = args.output_dir.resolve()
    if not corpus.is_dir():
        raise SystemExit(f"corpus does not exist: {corpus}")
    if not executable.is_file():
        raise SystemExit(f"MiaCode executable does not exist: {executable}")
    if not slide_data_path.is_file():
        raise SystemExit(f"slide data does not exist: {slide_data_path}")

    songs = selected_songs(corpus)
    if not songs:
        raise SystemExit("no target five-digit FESTiVAL-or-newer maidata files found")
    output_dir.mkdir(parents=True, exist_ok=True)

    slide_data = json.loads(slide_data_path.read_text(encoding="utf-8"))
    slides = slide_data.get("slides", {})
    wifi = slide_data.get("wifi", {})
    aliases = build_caret_aliases(slides)

    dumped: list[dict[str, Any]] = []
    workers = max(1, int(args.workers))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(dump_song, song, executable, args.timeout_seconds): song
            for song in songs
        }
        for index, future in enumerate(as_completed(futures), 1):
            dumped.append(future.result())
            if index % 25 == 0 or index == len(futures):
                print(f"parsed {index}/{len(futures)} songs", file=sys.stderr)
    dumped.sort(key=lambda row: str(row["song"]["song_id"]))

    occurrences: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    totals = Counter()
    version_counts: dict[str, Counter[str]] = {
        version: Counter() for version in TARGET_VERSIONS
    }
    for result in dumped:
        song = result["song"]
        version_counts[song["version"]]["songs"] += 1
        if "fatal_error" in result:
            failures.append(
                {
                    "song_id": song["song_id"],
                    "title": song["title"],
                    "version": song["version"],
                    "difficulty": "",
                    "error": result["fatal_error"],
                    "scope": "song",
                }
            )
            continue
        rows, item_failures, counts = extract_occurrences(
            result, slides, wifi, aliases
        )
        occurrences.extend(rows)
        failures.extend(item_failures)
        totals.update(counts)
        version_counts[song["version"]]["charts"] += counts.get("charts", 0)
        version_counts[song["version"]]["branches"] += len(rows)
        version_counts[song["version"]]["chains"] += sum(
            int(row["segment_count"]) >= 2 for row in rows
        )

    occurrences.sort(
        key=lambda row: (
            str(row["song_id"]),
            str(row["difficulty"]),
            int(row["tick"]),
            int(row["branch_index"]),
        )
    )
    routes = route_rows(occurrences)
    chains = [row for row in occurrences if int(row["segment_count"]) >= 2]
    chain_routes = [row for row in routes if int(row["segment_count"]) >= 2]
    chain_segments = [int(row["segment_count"]) for row in chains]
    chain_areas = [int(row["judge_area_count"]) for row in chains]
    missing_resources = [row for row in occurrences if row["resource_missing"]]
    difficulty_failures = sum(row["scope"] == "difficulty" for row in failures)

    base_rows = [row for row in occurrences if int(row["segment_count"]) == 1]
    observed_base_routes = {str(row["factor_route"]) for row in base_rows}
    expected_base_routes = {
        factor_path(aliases.get(str(key), str(key)))
        for key in slides
        if str(key).startswith("1")
    } | {
        factor_path(str(key)) for key in wifi if str(key).startswith("1")
    }
    base_trajectories = {str(row["trajectory_signature"]) for row in base_rows}
    chain_trajectories = {str(row["trajectory_signature"]) for row in chains}
    overlap_trajectories = base_trajectories & chain_trajectories
    overlapping_chains = [
        row for row in chains if str(row["trajectory_signature"]) in base_trajectories
    ]
    chain_route_frequency = Counter(str(row["factor_route"]) for row in chains)
    singleton_occurrences = sum(
        count for count in chain_route_frequency.values() if count == 1
    )

    def quantiles(values: Sequence[int]) -> dict[str, int]:
        return {
            "p50": nearest_rank(values, 0.50),
            "p90": nearest_rank(values, 0.90),
            "p95": nearest_rank(values, 0.95),
            "p99": nearest_rank(values, 0.99),
            "max": max(values, default=0),
        }

    segment_quantiles = quantiles(chain_segments)
    extreme_min_segments = segment_quantiles["p99"] + 1
    regular_chains = [
        row for row in chains if int(row["segment_count"]) < extreme_min_segments
    ]
    extreme_chains = [
        row for row in chains if int(row["segment_count"]) >= extreme_min_segments
    ]

    summary = {
        "schema": "maimai.raw_slide_route_inventory.festival_plus.v1",
        "selection": {
            "id_regex": "^\\d{5}$",
            "excluded_six_digit_entertainment": True,
            "versions": list(TARGET_VERSIONS),
            "source": str(corpus),
            "uses_existing_chart_ir": False,
            "parser_mode": "MiaCode raw maidata dump, difficulty ALL",
        },
        "counts": {
            "songs": len(songs),
            "charts": totals["charts"],
            "difficulty_failures": difficulty_failures,
            "all_failures": len(failures),
            "slide_notes": totals["slide_notes"],
            "branches": len(occurrences),
            "ordinary_slide_branches": len(
                [row for row in occurrences if row["kind"] == "slide"]
            ),
            "wifi_branches": len(
                [row for row in occurrences if row["kind"] == "wifi"]
            ),
            "base_branches": len(occurrences) - len(chains),
            "chain_branches": len(chains),
            "expected_base_route_types": len(expected_base_routes),
            "observed_base_route_types": len(observed_base_routes),
            "missing_base_routes": sorted(expected_base_routes - observed_base_routes),
            "route_types": len(routes),
            "chain_route_types": len(chain_routes),
            "trajectory_types": len(
                {str(row["trajectory_signature"]) for row in occurrences}
            ),
            "chain_trajectory_types": len(
                {str(row["trajectory_signature"]) for row in chains}
            ),
            "resource_missing_branches": len(missing_resources),
            "caret_aliases": len(aliases),
            "chain_total_timing_branches": sum(
                str(row["timing_mode"]) == "total" for row in chains
            ),
        },
        "by_version": {
            version: {
                "songs": version_counts[version]["songs"],
                "charts": version_counts[version]["charts"],
                "branches": version_counts[version]["branches"],
                "chains": version_counts[version]["chains"],
            }
            for version in TARGET_VERSIONS
        },
        "segment_count_occurrence_histogram": histogram(
            int(row["segment_count"]) for row in occurrences
        ),
        "segment_count_route_type_histogram": histogram(
            int(row["segment_count"]) for row in routes
        ),
        "chain_route_frequency_histogram": histogram(
            int(row["occurrences"]) for row in chain_routes
        ),
        "chain_quantiles": {
            "segments": segment_quantiles,
            "judge_areas": quantiles(chain_areas),
        },
        "recommended_length_buckets": {
            "derivation": "extreme begins one segment above the occurrence-level P99",
            "regular_max_segments": extreme_min_segments - 1,
            "extreme_min_segments": extreme_min_segments,
            "regular_occurrences": len(regular_chains),
            "regular_route_types": len(
                {str(row["factor_route"]) for row in regular_chains}
            ),
            "regular_chain_coverage": round(
                len(regular_chains) / len(chains), 6
            ) if chains else 0.0,
            "extreme_occurrences": len(extreme_chains),
            "extreme_route_types": len(
                {str(row["factor_route"]) for row in extreme_chains}
            ),
            "extreme_chain_coverage": round(
                len(extreme_chains) / len(chains), 6
            ) if chains else 0.0,
        },
        "chain_frequency": {
            "singleton_route_types": sum(
                count == 1 for count in chain_route_frequency.values()
            ),
            "singleton_occurrences": singleton_occurrences,
            "singleton_chain_coverage": round(
                singleton_occurrences / len(chains), 6
            ) if chains else 0.0,
            "repeated_route_types": sum(
                count >= 2 for count in chain_route_frequency.values()
            ),
            "repeated_occurrences": sum(
                count for count in chain_route_frequency.values() if count >= 2
            ),
        },
        "trajectory_overlap": {
            "base_trajectory_types": len(base_trajectories),
            "chain_trajectory_types": len(chain_trajectories),
            "shared_trajectory_types": len(overlap_trajectories),
            "chain_route_types_equivalent_to_base": len(
                {str(row["factor_route"]) for row in overlapping_chains}
            ),
            "chain_occurrences_equivalent_to_base": len(overlapping_chains),
        },
        "candidate_policies": policy_table(routes),
        "provenance": {
            "miacode_executable": str(executable),
            "miacode_sha256": sha256_file(executable),
            "slide_data": str(slide_data_path),
            "slide_data_sha256": sha256_file(slide_data_path),
            "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        },
    }

    song_rows = [
        {
            "song_id": song["song_id"],
            "shortid": song["shortid"],
            "title": song["title"],
            "version": song["version"],
            "cabinet": song["cabinet"],
            "maidata": str(Path(song["maidata"]).resolve()),
            "maidata_sha256": sha256_file(Path(song["maidata"])),
        }
        for song in songs
    ]
    write_csv(
        output_dir / "selected_songs.csv",
        song_rows,
        ("song_id", "shortid", "title", "version", "cabinet", "maidata", "maidata_sha256"),
    )
    occurrence_fields = (
        "song_id",
        "title",
        "version",
        "difficulty",
        "chart_key",
        "tick",
        "branch_index",
        "kind",
        "raw",
        "segment_count",
        "raw_segments",
        "canonical_segments",
        "factor_route",
        "judge_area_count",
        "trajectory_signature",
        "resource_missing",
        "timing_mode",
    )
    write_csv(output_dir / "occurrences.csv", occurrences, occurrence_fields)
    write_csv(output_dir / "chain_branches.csv", chains, occurrence_fields)
    route_fields = (
        "kind",
        "factor_route",
        "canonical_segments",
        "segment_count",
        "occurrences",
        "song_count",
        "chart_count",
        "trajectory_type_count",
        "judge_areas_min",
        "judge_areas_mean",
        "judge_areas_max",
        "example_song_id",
        "example_title",
        "example_difficulty",
        "example_raw",
        "example_trajectory",
    )
    write_csv(output_dir / "route_vocab.csv", routes, route_fields)
    extremes = sorted(
        chains,
        key=lambda row: (
            -int(row["segment_count"]),
            -int(row["judge_area_count"]),
            str(row["song_id"]),
        ),
    )[:200]
    write_csv(output_dir / "extreme_examples.csv", extremes, occurrence_fields)
    write_csv(
        output_dir / "parse_failures.csv",
        failures,
        ("song_id", "title", "version", "difficulty", "scope", "error"),
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(
        markdown_report(summary, routes), encoding="utf-8"
    )
    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2))
    print(f"wrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
