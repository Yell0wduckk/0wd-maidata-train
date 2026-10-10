from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence


MIN_ROUTE_ID_OCCURRENCES = 5
PATH_RE = re.compile(r"^([1-8])(pp|qq|[-^v<>sz]|p|q)([1-8])$")
V_PATH_RE = re.compile(r"^([1-8])V([1-8])([1-8])$")


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    workspace = script.parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Merge cached chains by judge sequence, mirror one representative with "
            "MiaCode, normalize it to start lane 1, then merge judge sequences again."
        )
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=script.parents[1] / "artifacts" / "raw-slide-routes-festival-plus-v1",
    )
    parser.add_argument(
        "--transform-exe",
        type=Path,
        default=(
            workspace
            / "tools"
            / "MiaCode"
            / "build"
            / "Debug"
            / "chart_batch_transform_spec.exe"
        ),
    )
    parser.add_argument(
        "--chart-ir-exe",
        type=Path,
        default=(
            workspace
            / "tools"
            / "MiaCode"
            / "build"
            / "Release"
            / "miacode_chart_ir.exe"
        ),
    )
    parser.add_argument(
        "--slide-data",
        type=Path,
        default=(
            workspace
            / "tools"
            / "MiaCode"
            / "assets"
            / "reference"
            / "slide_data.json"
        ),
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_signature(signature: str) -> str:
    return ">".join(
        "/".join(sorted(pad for pad in area.split("/") if pad))
        for area in signature.split(">")
    )


def complete_route(canonical_segments: str) -> str:
    segments = canonical_segments.split("|")
    if not segments or any(not segment for segment in segments):
        raise ValueError(f"empty canonical segment in {canonical_segments!r}")
    for previous, current in zip(segments, segments[1:]):
        if previous[-1] != current[0]:
            raise ValueError(f"discontinuous route: {previous} -> {current}")
    return segments[0] + "".join(segment[1:] for segment in segments[1:])


def maidata_text(chart: str, *, title: str) -> str:
    return "\n".join(
        (
            f"&title={title}",
            "&wholebpm=180",
            "&artist=maimai-ai-charting",
            "&des_5=Native Mirror Probe",
            "&shortid=0",
            "&cabinet=DX",
            "&first=0",
            "&lv_5=15",
            "&inote_5=",
            chart.strip(),
            "",
        )
    )


def run_native_transform(executable: Path, op: str, maidata: Path) -> str:
    completed = subprocess.run(
        [str(executable), "--dump", op, str(maidata)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=120,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"MiaCode transform {op} failed: {stderr}")
    return completed.stdout.decode("utf-8", errors="strict")


def dump_chart_ir(executable: Path, maidata: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            str(executable),
            "dump",
            "--maidata",
            str(maidata),
            "--difficulty",
            "MAS",
            "--output",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=120,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"MiaCode chart IR dump failed: {stderr}")
    return json.loads(completed.stdout)


def iter_notes(chart: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    for moment in chart.get("moments", []):
        for group in moment.get("groups", []):
            yield from group.get("notes", [])


def merged_judge_sequence(
    segment_paths: Sequence[str], slides: Mapping[str, Any]
) -> list[list[str]]:
    merged: list[list[str]] = []
    for path in segment_paths:
        resource = slides.get(path)
        if resource is None:
            raise KeyError(f"MiaCode slide_data is missing transformed path {path!r}")
        sequence = [
            sorted(str(pad) for pad in area)
            for area in resource.get("judge_sequence", [])
        ]
        if not sequence:
            raise ValueError(f"transformed path has no judge sequence: {path}")
        if merged and merged[-1] == sequence[0]:
            merged.extend(sequence[1:])
        else:
            merged.extend(sequence)
    return merged


def judge_signature(sequence: Sequence[Sequence[str]]) -> str:
    return ">".join("/".join(sorted(area)) for area in sequence)


def factor_path(path: str) -> str:
    match_v = V_PATH_RE.fullmatch(path)
    if match_v is not None:
        start, via, end = (int(match_v.group(index)) for index in (1, 2, 3))
        return f"V:{(via - start) % 8}:{(end - start) % 8}"
    match = PATH_RE.fullmatch(path)
    if match is None:
        raise ValueError(f"unsupported transformed path: {path!r}")
    start, shape, end = int(match.group(1)), match.group(2), int(match.group(3))
    return f"{shape}:{(end - start) % 8}"


def examples(rows: Iterable[Mapping[str, str]], limit: int = 5) -> str:
    result: list[str] = []
    seen: set[str] = set()
    for row in rows:
        value = f"{row['song_id']}:{row['difficulty']}:{row['raw']}"
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
        if len(result) >= limit:
            break
    return " || ".join(result)


def choose_representative(members: Sequence[Mapping[str, str]]) -> Mapping[str, str]:
    route_counts = Counter(row["factor_route"] for row in members)
    preferred_route = min(route_counts, key=lambda key: (-route_counts[key], key))
    candidates = [row for row in members if row["factor_route"] == preferred_route]
    return min(
        candidates,
        key=lambda row: (
            row["song_id"],
            row["difficulty"],
            int(row["tick"]),
            int(row["branch_index"]),
        ),
    )


def main() -> int:
    args = parse_args()
    root = args.inventory.resolve()
    transform_exe = args.transform_exe.resolve()
    chart_ir_exe = args.chart_ir_exe.resolve()
    slide_data_path = args.slide_data.resolve()
    for required in (transform_exe, chart_ir_exe, slide_data_path):
        if not required.is_file():
            raise SystemExit(f"required MiaCode file is missing: {required}")

    branches = read_csv(root / "chain_branches.csv")
    route_rows = read_csv(root / "chain_routes.csv")
    if len(branches) != 4968 or any(int(row["segment_count"]) < 2 for row in branches):
        raise SystemExit("cached chain inventory does not match its frozen contract")
    slide_data = json.loads(slide_data_path.read_text(encoding="utf-8"))
    slides = slide_data["slides"]

    exact_members: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in branches:
        exact_members[normalized_signature(row["trajectory_signature"])].append(row)
    exact_signatures = sorted(exact_members)
    representatives = [choose_representative(exact_members[key]) for key in exact_signatures]

    # `canonical_segments` in the frozen inventory was produced by rotating only
    # lane digits.  That representation is useful for counting, but it is not
    # safe to serialize because MiaCode may also rewrite < and > when a route is
    # rotated across screen quadrants.  Start from the real parsed `raw_segments`
    # instead and normalize each start-lane bucket exclusively with MiaCode's
    # native counter-clockwise rotation.
    representatives_by_start: dict[int, list[tuple[str, Mapping[str, str]]]] = (
        defaultdict(list)
    )
    for signature, representative in zip(exact_signatures, representatives):
        raw_segments = representative["raw_segments"]
        start = int(raw_segments[0])
        representatives_by_start[start].append((signature, representative))

    normalized_raw: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="maimai-native-normalize-") as temp_name:
        temp_root = Path(temp_name)
        for start, items in sorted(representatives_by_start.items()):
            source_lines = ["(180){1}"]
            for signature, representative in items:
                source_lines.append(
                    f"{complete_route(representative['raw_segments'])}[2:1],"
                )
            source_lines.append("E")
            current_maidata = temp_root / f"start-{start}-step-0.txt"
            current_maidata.write_text(
                maidata_text(
                    "\n".join(source_lines) + "\n",
                    title=f"Chain Judge Source Start {start}",
                ),
                encoding="utf-8",
            )
            for step in range(1, start):
                rotated_chart = run_native_transform(
                    transform_exe, "ccw45", current_maidata
                )
                current_maidata = temp_root / f"start-{start}-step-{step}.txt"
                current_maidata.write_text(
                    maidata_text(
                        rotated_chart,
                        title=f"Chain Judge Source Start {start} Step {step}",
                    ),
                    encoding="utf-8",
                )

            normalized_ir = dump_chart_ir(chart_ir_exe, current_maidata)
            normalized_notes = [
                note
                for note in iter_notes(normalized_ir)
                if note.get("kind") == "slide"
            ]
            if len(normalized_notes) != len(items):
                raise RuntimeError(
                    f"native start normalization lost slides for start {start}: "
                    f"expected={len(items)} actual={len(normalized_notes)}"
                )
            for (signature, _representative), note in zip(items, normalized_notes):
                note_branches = note.get("branches", [])
                if note.get("start_pad") != "A1" or len(note_branches) != 1:
                    raise RuntimeError(
                        f"native start normalization failed for {signature}"
                    )
                paths = [
                    str(segment["path"])
                    for segment in note_branches[0].get("segments", [])
                ]
                reparsed_signature = judge_signature(
                    merged_judge_sequence(paths, slides)
                )
                if reparsed_signature != signature:
                    raise RuntimeError(
                        "native-normalized representative changed judge sequence: "
                        f"cached={signature} reparsed={reparsed_signature}"
                    )
                normalized_raw[signature] = str(note["source"]["raw"])

    chart_lines = ["(180){1}"]
    for index, signature in enumerate(exact_signatures, 1):
        chart_lines.append(
            f"|| exact_group={index:04d} occurrences={len(exact_members[signature])}"
        )
        chart_lines.append(f"{normalized_raw[signature]},")
    chart_lines.append("E")
    representative_chart = "\n".join(chart_lines) + "\n"

    representative_maidata = root / "chain_judge_representatives_maidata.txt"
    representative_maidata.write_text(
        maidata_text(representative_chart, title="Chain Judge Representatives"),
        encoding="utf-8",
    )

    mirrored_chart = run_native_transform(transform_exe, "lr", representative_maidata)
    mirrored_simai = root / "chain_judge_representatives_mirrored_lr.simai"
    mirrored_simai.write_text(mirrored_chart, encoding="utf-8")
    mirrored_maidata = root / "chain_judge_representatives_mirrored_lr_maidata.txt"
    mirrored_maidata.write_text(
        maidata_text(mirrored_chart, title="Chain Judge Representatives LR"),
        encoding="utf-8",
    )

    # Native LR maps canonical start A1 to A8. One native clockwise rotation
    # returns every representative to A1 without reimplementing lane rules.
    normalized_chart = run_native_transform(transform_exe, "cw45", mirrored_maidata)
    normalized_simai = root / "chain_judge_representatives_mirrored_lr_normalized.simai"
    normalized_simai.write_text(normalized_chart, encoding="utf-8")
    normalized_maidata = (
        root / "chain_judge_representatives_mirrored_lr_normalized_maidata.txt"
    )
    normalized_maidata.write_text(
        maidata_text(normalized_chart, title="Chain Judge Representatives LR Normalized"),
        encoding="utf-8",
    )

    # A direct second native LR + normalization must return every representative
    # to its original judge sequence.  This tests the actual serialized samples,
    # independently of which representative another exact-sequence group chose.
    mirrored_twice_lr_chart = run_native_transform(
        transform_exe, "lr", normalized_maidata
    )
    mirrored_twice_lr_simai = (
        root / "chain_judge_representatives_mirrored_twice_lr.simai"
    )
    mirrored_twice_lr_simai.write_text(mirrored_twice_lr_chart, encoding="utf-8")
    mirrored_twice_lr_maidata = (
        root / "chain_judge_representatives_mirrored_twice_lr_maidata.txt"
    )
    mirrored_twice_lr_maidata.write_text(
        maidata_text(
            mirrored_twice_lr_chart,
            title="Chain Judge Representatives LR Twice Pre-Normalize",
        ),
        encoding="utf-8",
    )
    mirrored_twice_chart = run_native_transform(
        transform_exe, "cw45", mirrored_twice_lr_maidata
    )
    mirrored_twice_simai = (
        root / "chain_judge_representatives_mirrored_twice_normalized.simai"
    )
    mirrored_twice_simai.write_text(mirrored_twice_chart, encoding="utf-8")
    mirrored_twice_maidata = (
        root / "chain_judge_representatives_mirrored_twice_normalized_maidata.txt"
    )
    mirrored_twice_maidata.write_text(
        maidata_text(
            mirrored_twice_chart,
            title="Chain Judge Representatives LR Twice Normalized",
        ),
        encoding="utf-8",
    )
    mirrored_twice_ir = dump_chart_ir(chart_ir_exe, mirrored_twice_maidata)
    mirrored_twice_notes = [
        note for note in iter_notes(mirrored_twice_ir) if note.get("kind") == "slide"
    ]
    if len(mirrored_twice_notes) != len(exact_signatures):
        raise RuntimeError(
            "double native mirror lost representatives: "
            f"expected={len(exact_signatures)} actual={len(mirrored_twice_notes)}"
        )
    for index, (signature, note) in enumerate(
        zip(exact_signatures, mirrored_twice_notes), 1
    ):
        note_branches = note.get("branches", [])
        if note.get("start_pad") != "A1" or len(note_branches) != 1:
            raise RuntimeError(f"double native mirror failed at group {index}")
        paths = [
            str(segment["path"])
            for segment in note_branches[0].get("segments", [])
        ]
        roundtrip_signature = judge_signature(merged_judge_sequence(paths, slides))
        if roundtrip_signature != signature:
            raise RuntimeError(
                "double native mirror did not return the original judge group: "
                f"group={index} original={signature} roundtrip={roundtrip_signature}"
            )

    transformed_chart_ir = dump_chart_ir(chart_ir_exe, normalized_maidata)
    transformed_notes = [
        note for note in iter_notes(transformed_chart_ir) if note.get("kind") == "slide"
    ]
    if len(transformed_notes) != len(exact_signatures):
        raise RuntimeError(
            "native mirror probe lost representatives: "
            f"expected={len(exact_signatures)} actual={len(transformed_notes)}"
        )

    native_mirror: dict[str, str] = {}
    exact_rows: list[dict[str, Any]] = []
    transformed_records: list[dict[str, Any]] = []
    for index, (signature, representative, note) in enumerate(
        zip(exact_signatures, representatives, transformed_notes), 1
    ):
        if note.get("start_pad") != "A1":
            raise RuntimeError(
                f"native mirror normalization did not return group {index} to A1"
            )
        note_branches = note.get("branches", [])
        if len(note_branches) != 1:
            raise RuntimeError(f"representative {index} did not remain one branch")
        segment_paths = [
            str(segment["path"]) for segment in note_branches[0].get("segments", [])
        ]
        mirrored_signature = judge_signature(merged_judge_sequence(segment_paths, slides))
        native_mirror[signature] = mirrored_signature
        members = exact_members[signature]
        factor_routes = sorted({row["factor_route"] for row in members})
        canonical_routes = sorted({row["canonical_segments"] for row in members})
        segments = [int(row["segment_count"]) for row in members]
        transformed_source = str(note.get("source", {}).get("raw", ""))
        transformed_factor = "|".join(factor_path(path) for path in segment_paths)
        exact_rows.append(
            {
                "exact_group_index": index,
                "trajectory_signature": signature,
                "native_mirror_trajectory_signature": mirrored_signature,
                "native_mirror_group_key": min(signature, mirrored_signature),
                "occurrences": len(members),
                "song_count": len({member["song_id"] for member in members}),
                "chart_count": len({member["chart_key"] for member in members}),
                "version_count": len({member["version"] for member in members}),
                "judge_area_count": int(members[0]["judge_area_count"]),
                "segment_count_min": min(segments),
                "segment_count_max": max(segments),
                "factor_route_alias_count": len(factor_routes),
                "factor_route_aliases": " || ".join(factor_routes),
                "canonical_route_aliases": " || ".join(canonical_routes),
                "representative_factor_route": representative["factor_route"],
                "representative_canonical_segments": representative[
                    "canonical_segments"
                ],
                "representative_raw": representative["raw"],
                "native_mirror_normalized_raw": transformed_source,
                "native_mirror_normalized_segments": "|".join(segment_paths),
                "native_mirror_normalized_factor_route": transformed_factor,
                "candidate_min5": len(members) >= MIN_ROUTE_ID_OCCURRENCES,
                "examples": examples(members),
            }
        )
        transformed_records.append(
            {
                "exact_group_index": index,
                "representative_canonical_segments": representative[
                    "canonical_segments"
                ],
                "native_mirror_normalized_raw": transformed_source,
                "native_mirror_normalized_segments": "|".join(segment_paths),
                "original_trajectory_signature": signature,
                "native_mirror_trajectory_signature": mirrored_signature,
            }
        )

    # The direct double-transform above proves the native operation is an
    # involution on every chosen representative.  If the mirrored judge sequence
    # is also observed, its independently chosen representative must map back.
    for signature, mirrored in native_mirror.items():
        if mirrored in native_mirror and native_mirror[mirrored] != signature:
            raise RuntimeError(
                "native mirror mapping is not involutive for observed pair: "
                f"{signature} -> {mirrored} -> {native_mirror[mirrored]}"
            )

    for row in transformed_records:
        row["native_mirror_group_key"] = min(
            row["original_trajectory_signature"],
            row["native_mirror_trajectory_signature"],
        )

    mirror_members: dict[str, list[dict[str, str]]] = defaultdict(list)
    for branch in branches:
        signature = normalized_signature(branch["trajectory_signature"])
        mirror_members[min(signature, native_mirror[signature])].append(branch)

    mirror_rows: list[dict[str, Any]] = []
    for group_key, members in mirror_members.items():
        observed_signatures = sorted(
            {normalized_signature(row["trajectory_signature"]) for row in members}
        )
        factor_routes = sorted({row["factor_route"] for row in members})
        canonical_routes = sorted({row["canonical_segments"] for row in members})
        segments = [int(row["segment_count"]) for row in members]
        mirror_rows.append(
            {
                "native_mirror_group_key": group_key,
                "observed_exact_trajectory_count": len(observed_signatures),
                "observed_exact_trajectories": " || ".join(observed_signatures),
                "occurrences": len(members),
                "song_count": len({row["song_id"] for row in members}),
                "chart_count": len({row["chart_key"] for row in members}),
                "version_count": len({row["version"] for row in members}),
                "judge_area_count": int(members[0]["judge_area_count"]),
                "segment_count_min": min(segments),
                "segment_count_max": max(segments),
                "factor_route_alias_count": len(factor_routes),
                "factor_route_aliases": " || ".join(factor_routes),
                "canonical_route_aliases": " || ".join(canonical_routes),
                "candidate_min5": len(members) >= MIN_ROUTE_ID_OCCURRENCES,
                "examples": examples(members),
            }
        )
    mirror_rows.sort(
        key=lambda row: (-int(row["occurrences"]), row["native_mirror_group_key"])
    )
    exact_rows.sort(key=lambda row: (-int(row["occurrences"]), row["trajectory_signature"]))

    syntax_min5 = [row for row in route_rows if int(row["occurrences"]) >= 5]
    exact_min5 = [row for row in exact_rows if bool(row["candidate_min5"])]
    mirror_min5 = [row for row in mirror_rows if bool(row["candidate_min5"])]
    mirror_lt5 = [row for row in mirror_rows if not bool(row["candidate_min5"])]

    review_root = (
        Path(__file__).resolve().parents[2]
        / "artifacts"
        / "slide-route-id-review-native-mirror-min5-maidata-v1"
    )
    review_root.mkdir(parents=True, exist_ok=True)
    review_lines = ["(180){1}"]
    for rank, row in enumerate(mirror_min5, 1):
        group_key = str(row["native_mirror_group_key"])
        signature_counts = Counter(
            normalized_signature(member["trajectory_signature"])
            for member in mirror_members[group_key]
        )
        representative_signature = min(
            signature_counts,
            key=lambda signature: (-signature_counts[signature], signature),
        )
        review_lines.append(
            "|| "
            f"rank={rank:03d} occurrences={int(row['occurrences'])} "
            f"songs={int(row['song_count'])} "
            f"exact_sequences={int(row['observed_exact_trajectory_count'])}"
        )
        review_lines.append(f"{normalized_raw[representative_signature]},")
    review_lines.append("E")
    review_maidata = review_root / "maidata.txt"
    review_maidata.write_text(
        maidata_text(
            "\n".join(review_lines) + "\n",
            title="189 Native-Mirror Chain Slide Candidates",
        ),
        encoding="utf-8",
    )
    review_ir = dump_chart_ir(chart_ir_exe, review_maidata)
    review_slides = [
        note for note in iter_notes(review_ir) if note.get("kind") == "slide"
    ]
    if len(review_slides) != len(mirror_min5) or len(review_slides) != 189:
        raise RuntimeError(
            "candidate review maidata does not contain exactly 189 slides: "
            f"parsed={len(review_slides)} candidates={len(mirror_min5)}"
        )
    if any(note.get("start_pad") != "A1" for note in review_slides):
        raise RuntimeError("candidate review maidata contains a non-A1 slide")

    def covered(rows: Sequence[Mapping[str, Any]]) -> int:
        return sum(int(row["occurrences"]) for row in rows)

    syntax_covered = covered(syntax_min5)
    exact_covered = covered(exact_min5)
    mirror_covered = covered(mirror_min5)
    summary = {
        "schema": "maimai.chain_judge_native_mirror_merge.v1",
        "input": {
            "source": str(root / "chain_branches.csv"),
            "branches": len(branches),
            "does_not_rescan_maidata": True,
        },
        "method": {
            "first_merge": "complete ordered judge-area sequence",
            "representatives": len(representatives),
            "representative_source": "real parsed raw_segments",
            "representative_start_normalization": (
                "MiaCode native Rotate45CounterClockwise to A1"
            ),
            "mirror": "MiaCode native MirrorLeftRight",
            "renormalize": "MiaCode native Rotate45Clockwise (A8 back to A1)",
            "second_merge": (
                "unordered pairs of original/native-mirrored complete ordered "
                "judge-area sequences"
            ),
            "double_native_mirror_roundtrip_verified": True,
            "no_handwritten_mirror_lane_mapping": True,
        },
        "counts": {
            "syntax_route_types": len(route_rows),
            "exact_judge_trajectory_groups": len(exact_rows),
            "native_mirror_merged_groups": len(mirror_rows),
            "syntax_to_exact_reduction": len(route_rows) - len(exact_rows),
            "exact_to_native_mirror_reduction": len(exact_rows) - len(mirror_rows),
            "syntax_to_native_mirror_reduction": len(route_rows) - len(mirror_rows),
            "observed_native_mirror_pair_groups": sum(
                int(row["observed_exact_trajectory_count"]) == 2 for row in mirror_rows
            ),
            "single_observed_orientation_groups": sum(
                int(row["observed_exact_trajectory_count"]) == 1 for row in mirror_rows
            ),
            "max_observed_exact_trajectories_per_group": max(
                int(row["observed_exact_trajectory_count"]) for row in mirror_rows
            ),
        },
        "min5": {
            "syntax_route_ids": len(syntax_min5),
            "syntax_covered_branches": syntax_covered,
            "syntax_coverage": round(syntax_covered / len(branches), 6),
            "exact_trajectory_ids": len(exact_min5),
            "exact_covered_branches": exact_covered,
            "exact_coverage": round(exact_covered / len(branches), 6),
            "native_mirror_trajectory_ids": len(mirror_min5),
            "native_mirror_covered_branches": mirror_covered,
            "native_mirror_coverage": round(mirror_covered / len(branches), 6),
            "candidate_max_source_segments": max(
                (int(row["segment_count_max"]) for row in mirror_min5), default=0
            ),
            "candidates_containing_7plus_segment_source": sum(
                int(row["segment_count_max"]) >= 7 for row in mirror_min5
            ),
            "candidates_with_at_most_2_songs": sum(
                int(row["song_count"]) <= 2 for row in mirror_min5
            ),
        },
        "long_tail_lt5": {
            "groups": len(mirror_lt5),
            "branches": covered(mirror_lt5),
            "coverage": round(covered(mirror_lt5) / len(branches), 6),
            "single_occurrence_groups": sum(
                int(row["occurrences"]) == 1 for row in mirror_lt5
            ),
            "single_song_groups": sum(
                int(row["song_count"]) == 1 for row in mirror_lt5
            ),
            "groups_with_at_most_2_songs": sum(
                int(row["song_count"]) <= 2 for row in mirror_lt5
            ),
            "max_source_segments": max(
                (int(row["segment_count_max"]) for row in mirror_lt5), default=0
            ),
            "max_judge_areas": max(
                (int(row["judge_area_count"]) for row in mirror_lt5), default=0
            ),
        },
        "provenance": {
            "transform_executable": str(transform_exe),
            "transform_executable_sha256": sha256_file(transform_exe),
            "chart_ir_executable": str(chart_ir_exe),
            "chart_ir_executable_sha256": sha256_file(chart_ir_exe),
            "slide_data": str(slide_data_path),
            "slide_data_sha256": sha256_file(slide_data_path),
            "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "review_maidata": {
            "path": str(review_maidata),
            "slides": len(review_slides),
            "bpm": 180,
            "spacing_beats": 4,
            "duration": "[2:1]",
            "track_required": False,
            "sha256": sha256_file(review_maidata),
        },
    }

    exact_fields = tuple(exact_rows[0])
    mirror_fields = tuple(mirror_rows[0])
    transform_fields = tuple(transformed_records[0])
    write_csv(root / "chain_judge_exact_groups.csv", exact_rows, exact_fields)
    write_csv(root / "chain_judge_native_mirror_map.csv", transformed_records, transform_fields)
    write_csv(root / "chain_judge_native_mirror_groups.csv", mirror_rows, mirror_fields)
    write_csv(root / "chain_judge_native_mirror_min5.csv", mirror_min5, mirror_fields)
    write_csv(root / "chain_judge_native_mirror_lt5.csv", mirror_lt5, mirror_fields)
    (root / "chain_judge_native_mirror_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    counts = summary["counts"]
    min5 = summary["min5"]
    tail = summary["long_tail_lt5"]
    (root / "chain_judge_native_mirror_report.md").write_text(
        f"""# 拼接 Slide：判定区序列 + MiaCode 原生镜像归并

输入为缓存的 {len(branches)} 条拼接 branch，没有重新扫描 maidata，也没有在统计脚本中
实现镜像编号规则。

## 流程

1. 先按完整有序判定区序列合并为 {len(exact_rows)} 组；同一步的可选判定区按集合处理。
2. 每组选择一条真实完整 Slide 作为代表，共 {len(representatives)} 条；从真实
   `raw_segments` 出发，并调用 MiaCode 原生旋转将起点归一化为 A1。
3. 调用 MiaCode 原生 `MirrorLeftRight`。
4. 调用 MiaCode 原生 `Rotate45Clockwise`，把镜像后的 A8 起点归一化回 A1。
5. 用 MiaCode 重新解析归一化结果，以无序的“原序列—原生镜像序列”对完成合并。
6. 对全部代表再做一次原生左右镜像和归一化，严格验证判定区序列逐条回到原组。

中间的代表谱面、原生镜像结果、归一化结果和逐代表映射均已保留，可逐条审计。

## 结果

- 原始 Simai 拼接路线：{counts['syntax_route_types']} 种。
- 首次判定区合并：{counts['exact_judge_trajectory_groups']} 组，减少
  {counts['syntax_to_exact_reduction']}。
- 原生镜像后再次合并：{counts['native_mirror_merged_groups']} 组，再减少
  {counts['exact_to_native_mirror_reduction']}；相对原始共减少
  {counts['syntax_to_native_mirror_reduction']}。
- 两个镜像方向都在语料中出现：{counts['observed_native_mirror_pair_groups']} 组；
  只出现一侧：{counts['single_observed_orientation_groups']} 组。
- 单组最多含 {counts['max_observed_exact_trajectories_per_group']} 个实际出现的判定区序列；
  两次原生镜像回原组校验已通过。

## 至少出现 5 次

| 口径 | ID 数 | 覆盖 branch | 覆盖率 |
|---|---:|---:|---:|
| 原始完整路线 | {min5['syntax_route_ids']} | {min5['syntax_covered_branches']} | {min5['syntax_coverage']:.2%} |
| 首次判定区合并 | {min5['exact_trajectory_ids']} | {min5['exact_covered_branches']} | {min5['exact_coverage']:.2%} |
| MiaCode 原生镜像合并 | {min5['native_mirror_trajectory_ids']} | {min5['native_mirror_covered_branches']} | {min5['native_mirror_coverage']:.2%} |

原生镜像口径的 min-5 候选不限制段数；实际来源最长
{min5['candidate_max_source_segments']} 段，其中
{min5['candidates_containing_7plus_segment_source']} 组包含至少一条 7 段以上来源。
另有 {min5['candidates_with_at_most_2_songs']} 组只来自不超过两首歌，需要后续人工复核。

## 低于 5 次的长尾

- 共 {tail['groups']} 组、{tail['branches']} 条 branch，占全部拼接 branch 的
  {tail['coverage']:.2%}。
- {tail['single_occurrence_groups']} 组仅出现一次；{tail['single_song_groups']} 组只来自一首歌；
  {tail['groups_with_at_most_2_songs']} 组来自不超过两首歌。
- 来源最长 {tail['max_source_segments']} 段，完整判定区序列最长
  {tail['max_judge_areas']} 个判定阶段。

这些组单独保存在 `chain_judge_native_mirror_lt5.csv`，用于后续有损量化与人工审计。
""",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
