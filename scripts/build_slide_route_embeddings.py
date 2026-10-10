from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import pearsonr, spearmanr


SCHEMA = "maimai.slide_route_embedding_catalog.v2"
MANIFEST_SCHEMA = "maimai.slide_route_embedding_audit.v3"
EMBEDDING_DIM = 256
EXPECTED_BASIC_ROUTES = 71
EXPECTED_CHAIN_GROUPS = 189
EXPECTED_CANONICAL_CHAIN_GROUPS = 182
EXPECTED_SOURCE_CHAIN_ROUTES = 377
EXPECTED_CHAIN_ALIASES_TO_BASIC = 14
EXPECTED_CHAIN_ROUTES = 363
EXPECTED_TOTAL_ROUTES = 434
EXPECTED_OBSERVED_PAIRS = 184
EXPECTED_SYNTHESIZED_PAIRS = 4
EXPECTED_CANONICAL_SYNTHESIZED_ROUTES = 3
EXPECTED_SELF_MIRROR = 1
PAD_RE = re.compile(r"^(?:[ABDE][1-8]|C)$")


Stage = tuple[str, ...]
Lane = tuple[Stage, ...]
Lanes = tuple[Lane, ...]


@dataclass(frozen=True)
class SourcePaths:
    workspace: Path
    basic_catalog: Path
    chain_inventory: Path
    slide_data: Path
    muri_config: Path
    output: Path
    document: Path


@dataclass(frozen=True)
class RouteRecord:
    embedding_index: int
    route_id: str
    family: str
    orientation: str
    lanes: Lanes
    trajectory_signature: str
    representative_simai: str
    representative_segments: str
    mirror_partner_id: str | None
    observed: bool
    synthesized_by_native_mirror: bool
    exact_occurrences: int | None
    group_occurrences: int | None
    song_count: int | None
    chart_count: int | None
    base_variant_id: int | None
    base_motion: str | None
    chain_group_rank: int | None
    chain_group_key: str | None
    examples: str

    @property
    def lane_count(self) -> int:
        return len(self.lanes)

    @property
    def judge_area_count(self) -> int:
        return sum(len(lane) for lane in self.lanes)

    @property
    def sequence_key(self) -> str:
        return json.dumps(self.lanes, ensure_ascii=False, separators=(",", ":"))

    def to_json(self) -> dict[str, Any]:
        return {
            "embedding_index": self.embedding_index,
            "route_id": self.route_id,
            "family": self.family,
            "orientation": self.orientation,
            "lanes": [[list(stage) for stage in lane] for lane in self.lanes],
            "trajectory_signature": self.trajectory_signature,
            "representative_simai": self.representative_simai,
            "representative_segments": self.representative_segments,
            "mirror_partner_id": self.mirror_partner_id,
            "observed": self.observed,
            "synthesized_by_native_mirror": self.synthesized_by_native_mirror,
            "exact_occurrences": self.exact_occurrences,
            "group_occurrences": self.group_occurrences,
            "song_count": self.song_count,
            "chart_count": self.chart_count,
            "base_variant_id": self.base_variant_id,
            "base_motion": self.base_motion,
            "chain_group_rank": self.chain_group_rank,
            "chain_group_key": self.chain_group_key,
            "lane_count": self.lane_count,
            "judge_area_count": self.judge_area_count,
            "sequence_key": self.sequence_key,
            "examples": self.examples,
        }


@dataclass(frozen=True)
class RouteAlias:
    alias_route_id: str
    canonical_route_id: str
    alias_orientation: str
    alias_representative_simai: str
    alias_representative_segments: str
    trajectory_signature: str
    mirror_partner_alias_route_id: str
    observed: bool
    synthesized_by_native_mirror: bool
    exact_occurrences: int
    group_occurrences: int
    song_count: int
    chart_count: int
    chain_group_rank: int
    chain_group_key: str
    canonical_representative_simai: str
    canonical_base_variant_id: int
    canonical_base_motion: str
    reason: str = "exact_judge_sequence_matches_basic"

    def to_json(self) -> dict[str, Any]:
        return {
            "alias_route_id": self.alias_route_id,
            "canonical_route_id": self.canonical_route_id,
            "alias_orientation": self.alias_orientation,
            "alias_representative_simai": self.alias_representative_simai,
            "alias_representative_segments": self.alias_representative_segments,
            "trajectory_signature": self.trajectory_signature,
            "mirror_partner_alias_route_id": self.mirror_partner_alias_route_id,
            "observed": self.observed,
            "synthesized_by_native_mirror": self.synthesized_by_native_mirror,
            "exact_occurrences": self.exact_occurrences,
            "group_occurrences": self.group_occurrences,
            "song_count": self.song_count,
            "chart_count": self.chart_count,
            "chain_group_rank": self.chain_group_rank,
            "chain_group_key": self.chain_group_key,
            "canonical_representative_simai": self.canonical_representative_simai,
            "canonical_base_variant_id": self.canonical_base_variant_id,
            "canonical_base_motion": self.canonical_base_motion,
            "reason": self.reason,
        }


def default_paths() -> SourcePaths:
    script = Path(__file__).resolve()
    project = script.parents[1]
    workspace = script.parents[2]
    inventory = workspace / "artifacts" / "raw-slide-routes-festival-plus-v1"
    return SourcePaths(
        workspace=workspace,
        basic_catalog=(
            workspace
            / "0wd-maidata-v2"
            / "src"
            / "maimai_chart_ir"
            / "assets"
            / "slide_variant_catalog.json"
        ),
        chain_inventory=inventory,
        slide_data=(
            workspace
            / "tools"
            / "MiaCode"
            / "assets"
            / "reference"
            / "slide_data.json"
        ),
        muri_config=workspace / "tools" / "MiaCode" / "src" / "common" / "MuriConfig.h",
        output=workspace / "artifacts" / "slide-route-embedding-434-v3",
        document=project / "docs" / "slide-route-embedding-434-v3.md",
    )


def parse_args() -> argparse.Namespace:
    defaults = default_paths()
    parser = argparse.ArgumentParser(
        description="Build the canonical 434-route judge-trajectory embedding audit."
    )
    parser.add_argument("--basic-catalog", type=Path, default=defaults.basic_catalog)
    parser.add_argument("--chain-inventory", type=Path, default=defaults.chain_inventory)
    parser.add_argument("--slide-data", type=Path, default=defaults.slide_data)
    parser.add_argument("--muri-config", type=Path, default=defaults.muri_config)
    parser.add_argument("--output", type=Path, default=defaults.output)
    parser.add_argument("--document", type=Path, default=defaults.document)
    return parser.parse_args()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_stage(values: Iterable[str]) -> Stage:
    stage = tuple(sorted({str(value) for value in values}))
    if not stage:
        raise ValueError("judge stage cannot be empty")
    invalid = [pad for pad in stage if PAD_RE.fullmatch(pad) is None]
    if invalid:
        raise ValueError(f"invalid judge pad(s): {invalid!r}")
    return stage


def parse_signature(value: str) -> Lanes:
    lane = tuple(normalize_stage(stage.split("/")) for stage in str(value).split(">"))
    if not lane:
        raise ValueError(f"empty trajectory signature: {value!r}")
    return (lane,)


def lane_signature(lane: Lane) -> str:
    return ">".join("/".join(stage) for stage in lane)


def lanes_signature(lanes: Lanes) -> str:
    return " || ".join(lane_signature(lane) for lane in lanes)


def normalize_signature(value: str) -> str:
    lanes = parse_signature(value)
    return lane_signature(lanes[0])


def complete_route(segments_value: str) -> str:
    segments = [part for part in str(segments_value).split("|") if part]
    if not segments or segments[0][0] != "1":
        raise ValueError(f"route is not normalized to A1: {segments_value!r}")
    for previous, current in zip(segments, segments[1:]):
        if previous[-1] != current[0]:
            raise ValueError(f"discontinuous chained route: {previous!r} -> {current!r}")
    return segments[0] + "".join(segment[1:] for segment in segments[1:])


def _basic_lanes(path: str, slide_data: Mapping[str, Any]) -> Lanes:
    if path in slide_data["slides"]:
        sequence = slide_data["slides"][path].get("judge_sequence")
        if not sequence:
            raise ValueError(f"basic Slide lacks judge_sequence: {path}")
        return (tuple(normalize_stage(stage) for stage in sequence),)
    if path in slide_data["wifi"]:
        lanes = slide_data["wifi"][path].get("tri_judge_sequence")
        if not lanes or len(lanes) != 3:
            raise ValueError(f"Wifi lacks three tri_judge_sequence lanes: {path}")
        return tuple(
            tuple(normalize_stage(stage) for stage in lane) for lane in lanes
        )
    raise KeyError(f"MiaCode slide_data does not contain basic path {path!r}")


def build_route_catalog_with_aliases(
    paths: SourcePaths,
) -> tuple[list[RouteRecord], dict[str, Any], list[RouteAlias]]:
    basic = _json(paths.basic_catalog)
    slide_data = _json(paths.slide_data)
    inventory = paths.chain_inventory
    groups = _csv(inventory / "chain_judge_native_mirror_min5.csv")
    exact_rows = _csv(inventory / "chain_judge_exact_groups.csv")
    mirror_rows = _csv(inventory / "chain_judge_native_mirror_map.csv")

    variants = sorted(basic["variants"], key=lambda row: int(row["variant_id"]))
    if len(variants) != EXPECTED_BASIC_ROUTES:
        raise ValueError(f"expected 71 basic variants, got {len(variants)}")
    if [int(row["variant_id"]) for row in variants] != list(range(71)):
        raise ValueError("basic variant IDs are not contiguous 0..70")

    records: list[RouteRecord] = []
    for variant in variants:
        variant_id = int(variant["variant_id"])
        path = str(variant["canonical_path"])
        lanes = _basic_lanes(path, slide_data)
        if any("A1" not in lane[0] for lane in lanes):
            raise ValueError(f"basic route is not normalized to A1: {path}")
        records.append(
            RouteRecord(
                embedding_index=len(records),
                route_id=f"B{variant_id:03d}",
                family="basic",
                orientation="catalog",
                lanes=lanes,
                trajectory_signature=lanes_signature(lanes),
                representative_simai=path,
                representative_segments=path,
                mirror_partner_id=None,
                observed=True,
                synthesized_by_native_mirror=False,
                exact_occurrences=None,
                group_occurrences=None,
                song_count=None,
                chart_count=None,
                base_variant_id=variant_id,
                base_motion=str(variant["motion"]),
                chain_group_rank=None,
                chain_group_key=None,
                examples="",
            )
        )

    exact_by_signature: dict[str, dict[str, str]] = {}
    for row in exact_rows:
        signature = normalize_signature(row["trajectory_signature"])
        if signature in exact_by_signature:
            raise ValueError(f"duplicate exact trajectory signature: {signature}")
        exact_by_signature[signature] = row

    mirror_by_signature: dict[str, str] = {}
    segments_by_signature: dict[str, str] = {}
    for row in mirror_rows:
        original = normalize_signature(row["original_trajectory_signature"])
        mirrored = normalize_signature(row["native_mirror_trajectory_signature"])
        for source, target in ((original, mirrored), (mirrored, original)):
            prior = mirror_by_signature.setdefault(source, target)
            if prior != target:
                raise ValueError(f"native mirror mapping is not functional for {source}")
        segments_by_signature.setdefault(
            original, str(row["representative_canonical_segments"])
        )
        segments_by_signature.setdefault(
            mirrored, str(row["native_mirror_normalized_segments"])
        )
    for source, target in mirror_by_signature.items():
        if mirror_by_signature.get(target) != source:
            raise ValueError(f"native mirror mapping is not involutive: {source} -> {target}")

    if len(groups) != EXPECTED_CHAIN_GROUPS:
        raise ValueError(f"expected 189 chain mirror groups, got {len(groups)}")
    groups.sort(key=lambda row: (-int(row["occurrences"]), row["native_mirror_group_key"]))
    observed_pairs = 0
    synthesized_pairs = 0
    self_mirrors = 0
    chain_count = 0

    def record_for_signature(
        *,
        signature: str,
        route_id: str,
        partner_id: str,
        orientation: str,
        rank: int,
        group: Mapping[str, str],
    ) -> RouteRecord:
        exact = exact_by_signature.get(signature)
        observed = exact is not None
        segments = (
            str(exact["representative_canonical_segments"])
            if exact is not None
            else segments_by_signature.get(signature, "")
        )
        if not segments:
            raise ValueError(f"no representative segments for trajectory {signature}")
        lanes = parse_signature(signature)
        if "A1" not in lanes[0][0]:
            raise ValueError(f"chain route is not normalized to A1: {signature}")
        return RouteRecord(
            embedding_index=-1,
            route_id=route_id,
            family="chain",
            orientation=orientation,
            lanes=lanes,
            trajectory_signature=signature,
            representative_simai=complete_route(segments),
            representative_segments=segments,
            mirror_partner_id=partner_id,
            observed=observed,
            synthesized_by_native_mirror=not observed,
            exact_occurrences=(int(exact["occurrences"]) if exact is not None else 0),
            group_occurrences=int(group["occurrences"]),
            song_count=(int(exact["song_count"]) if exact is not None else 0),
            chart_count=(int(exact["chart_count"]) if exact is not None else 0),
            base_variant_id=None,
            base_motion=None,
            chain_group_rank=rank,
            chain_group_key=normalize_signature(group["native_mirror_group_key"]),
            examples=(str(exact["examples"]) if exact is not None else ""),
        )

    for rank, group in enumerate(groups, 1):
        group_key = normalize_signature(group["native_mirror_group_key"])
        mirrored = mirror_by_signature.get(group_key)
        if mirrored is None:
            raise ValueError(f"group key has no native mirror mapping: {group_key}")
        observed_count = int(group["observed_exact_trajectory_count"])
        if group_key == mirrored:
            self_mirrors += 1
            route_id = f"C{rank:03d}-S"
            record = record_for_signature(
                signature=group_key,
                route_id=route_id,
                partner_id=route_id,
                orientation="self",
                rank=rank,
                group=group,
            )
            records.append(replace(record, embedding_index=len(records)))
            chain_count += 1
            if observed_count != 1:
                raise ValueError("self-mirror group must contain one exact trajectory")
            continue

        route_o = f"C{rank:03d}-O"
        route_m = f"C{rank:03d}-M"
        original = record_for_signature(
            signature=group_key,
            route_id=route_o,
            partner_id=route_m,
            orientation="original",
            rank=rank,
            group=group,
        )
        mirror = record_for_signature(
            signature=mirrored,
            route_id=route_m,
            partner_id=route_o,
            orientation="native-mirror",
            rank=rank,
            group=group,
        )
        actual_observed = int(original.observed) + int(mirror.observed)
        if actual_observed != observed_count:
            raise ValueError(
                f"observed-orientation count mismatch for group {rank}: "
                f"summary={observed_count}, exact={actual_observed}"
            )
        if actual_observed == 2:
            observed_pairs += 1
        elif actual_observed == 1:
            synthesized_pairs += 1
        else:
            raise ValueError(f"chain group {rank} has no observed orientation")
        records.append(replace(original, embedding_index=len(records)))
        records.append(replace(mirror, embedding_index=len(records)))
        chain_count += 2

    if chain_count != EXPECTED_SOURCE_CHAIN_ROUTES:
        raise ValueError(
            f"source chain route count mismatch: {chain_count} != "
            f"{EXPECTED_SOURCE_CHAIN_ROUTES}"
        )

    basic_by_sequence = {
        record.sequence_key: record for record in records if record.family == "basic"
    }
    aliases: list[RouteAlias] = []
    canonical_records: list[RouteRecord] = []
    for record in records:
        canonical = basic_by_sequence.get(record.sequence_key)
        if record.family == "chain" and canonical is not None:
            if (
                record.exact_occurrences is None
                or record.group_occurrences is None
                or record.song_count is None
                or record.chart_count is None
                or record.chain_group_rank is None
                or record.chain_group_key is None
                or canonical.base_variant_id is None
                or canonical.base_motion is None
            ):
                raise ValueError(f"incomplete chain-to-basic alias metadata: {record.route_id}")
            aliases.append(
                RouteAlias(
                    alias_route_id=record.route_id,
                    canonical_route_id=canonical.route_id,
                    alias_orientation=record.orientation,
                    alias_representative_simai=record.representative_simai,
                    alias_representative_segments=record.representative_segments,
                    trajectory_signature=record.trajectory_signature,
                    mirror_partner_alias_route_id=record.mirror_partner_id or "",
                    observed=record.observed,
                    synthesized_by_native_mirror=record.synthesized_by_native_mirror,
                    exact_occurrences=record.exact_occurrences,
                    group_occurrences=record.group_occurrences,
                    song_count=record.song_count,
                    chart_count=record.chart_count,
                    chain_group_rank=record.chain_group_rank,
                    chain_group_key=record.chain_group_key,
                    canonical_representative_simai=canonical.representative_simai,
                    canonical_base_variant_id=canonical.base_variant_id,
                    canonical_base_motion=canonical.base_motion,
                )
            )
            continue
        canonical_records.append(replace(record, embedding_index=len(canonical_records)))
    records = canonical_records

    alias_ids = {alias.alias_route_id for alias in aliases}
    if len(aliases) != EXPECTED_CHAIN_ALIASES_TO_BASIC:
        raise ValueError(
            f"chain-to-basic alias count mismatch: {len(aliases)} != "
            f"{EXPECTED_CHAIN_ALIASES_TO_BASIC}"
        )
    if len(alias_ids) != len(aliases):
        raise ValueError("chain-to-basic alias IDs are not unique")
    if any(alias.mirror_partner_alias_route_id not in alias_ids for alias in aliases):
        raise ValueError("a removed chain alias has a surviving or missing mirror partner")

    canonical_route_ids = {record.route_id for record in records}
    if any(
        record.family == "chain" and record.mirror_partner_id not in canonical_route_ids
        for record in records
    ):
        raise ValueError("canonical chain route points to a removed mirror partner")

    counts = {
        "basic_routes": sum(record.family == "basic" for record in records),
        "source_chain_groups": len(groups),
        "canonical_chain_groups": len(
            {
                record.chain_group_rank
                for record in records
                if record.family == "chain"
            }
        ),
        "source_chain_routes": chain_count,
        "chain_aliases_to_basic": len(aliases),
        "chain_routes": sum(record.family == "chain" for record in records),
        "total_routes": len(records),
        "source_observed_direction_pairs": observed_pairs,
        "source_one_observed_plus_synthesized_pairs": synthesized_pairs,
        "source_self_mirror_groups": self_mirrors,
        "source_synthesized_chain_routes": EXPECTED_SYNTHESIZED_PAIRS,
        "synthesized_chain_routes": sum(
            record.synthesized_by_native_mirror for record in records
        ),
        "wifi_routes": sum(record.lane_count == 3 for record in records),
    }
    expected = {
        "basic_routes": EXPECTED_BASIC_ROUTES,
        "source_chain_groups": EXPECTED_CHAIN_GROUPS,
        "canonical_chain_groups": EXPECTED_CANONICAL_CHAIN_GROUPS,
        "source_chain_routes": EXPECTED_SOURCE_CHAIN_ROUTES,
        "chain_aliases_to_basic": EXPECTED_CHAIN_ALIASES_TO_BASIC,
        "chain_routes": EXPECTED_CHAIN_ROUTES,
        "total_routes": EXPECTED_TOTAL_ROUTES,
        "source_observed_direction_pairs": EXPECTED_OBSERVED_PAIRS,
        "source_one_observed_plus_synthesized_pairs": EXPECTED_SYNTHESIZED_PAIRS,
        "source_self_mirror_groups": EXPECTED_SELF_MIRROR,
        "source_synthesized_chain_routes": EXPECTED_SYNTHESIZED_PAIRS,
        "synthesized_chain_routes": EXPECTED_CANONICAL_SYNTHESIZED_ROUTES,
        "wifi_routes": 1,
    }
    if counts != expected:
        raise ValueError(f"route catalog count mismatch: {counts!r} != {expected!r}")
    if [record.embedding_index for record in records] != list(range(len(records))):
        raise ValueError("embedding indices are not contiguous")
    if len({record.route_id for record in records}) != len(records):
        raise ValueError("route IDs are not unique")
    return records, counts, aliases


MURI_CONTRACT_SNIPPETS = (
    "constexpr double kLogicalCanvasSize = 540.0;",
    "constexpr double kPadDistanceA = kLogicalCanvasSize * 410.0 / 1080.0;",
    "constexpr double kPadDistanceB = kPadDistanceA * 2.2 / 4.0;",
    "constexpr double kPadDistanceD = kPadDistanceA * 4.1 / 4.0;",
    "constexpr double kPadDistanceE = kPadDistanceA * 3.1 / 4.0;",
    "constexpr double kPadAngleBaseDegrees = -67.5;",
    "constexpr double kPadAngleStepDegrees = 45.0;",
)


def verify_muri_geometry_contract(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    missing = [value for value in MURI_CONTRACT_SNIPPETS if value not in source]
    if missing:
        raise ValueError(f"MiaCode MuriConfig geometry contract changed: {missing!r}")


def pad_center(pad: str) -> tuple[float, float]:
    if pad == "C":
        return 0.0, 0.0
    if PAD_RE.fullmatch(pad) is None:
        raise ValueError(f"invalid pad token: {pad!r}")
    ring = pad[0]
    lane = int(pad[1])
    distance_a = 540.0 * 410.0 / 1080.0
    distances = {
        "A": distance_a,
        "B": distance_a * 2.2 / 4.0,
        "D": distance_a * 4.1 / 4.0,
        "E": distance_a * 3.1 / 4.0,
    }
    angle = (-90.0 if ring in {"D", "E"} else -67.5) + (lane - 1) * 45.0
    radians = math.radians(angle)
    return math.cos(radians) * distances[ring], math.sin(radians) * distances[ring]


class JudgeTrajectoryDistance:
    def __init__(self, records: Sequence[RouteRecord]) -> None:
        tokens = sorted(
            {
                pad
                for record in records
                for lane in record.lanes
                for stage in lane
                for pad in stage
            }
        )
        centers = {pad: np.asarray(pad_center(pad), dtype=np.float64) for pad in tokens}
        maximum = max(
            float(np.linalg.norm(centers[left] - centers[right]))
            for left in tokens
            for right in tokens
        )
        if not math.isfinite(maximum) or maximum <= 0:
            raise ValueError("invalid judge-pad geometry scale")
        self.tokens = tuple(tokens)
        self.centers = centers
        self.pad_distance_scale = maximum
        self._stage_cache: dict[tuple[Stage, Stage], float] = {}
        self._lane_cache: dict[tuple[Lane, Lane], float] = {}

    @staticmethod
    def _ordered_pair(left: Any, right: Any) -> tuple[Any, Any]:
        return (left, right) if left <= right else (right, left)

    def stage_distance(self, left: Stage, right: Stage) -> float:
        key = self._ordered_pair(left, right)
        cached = self._stage_cache.get(key)
        if cached is not None:
            return cached
        if left == right:
            value = 0.0
        else:
            def directed_nearest(source: Stage, target: Stage) -> float:
                return float(
                    np.mean(
                        [
                            min(
                                float(np.linalg.norm(self.centers[a] - self.centers[b]))
                                for b in target
                            )
                            for a in source
                        ]
                    )
                )

            value = (
                directed_nearest(left, right) + directed_nearest(right, left)
            ) / (2.0 * self.pad_distance_scale)
        result = float(np.clip(value, 0.0, 1.0))
        self._stage_cache[key] = result
        return result

    def lane_distance(self, left: Lane, right: Lane) -> float:
        key = self._ordered_pair(left, right)
        cached = self._lane_cache.get(key)
        if cached is not None:
            return cached
        left_size = len(left)
        right_size = len(right)
        if not left_size or not right_size:
            raise ValueError("judge lane cannot be empty")
        previous = np.arange(right_size + 1, dtype=np.float64)
        for left_index in range(1, left_size + 1):
            current = np.empty(right_size + 1, dtype=np.float64)
            current[0] = float(left_index)
            for right_index in range(1, right_size + 1):
                substitution = previous[right_index - 1] + self.stage_distance(
                    left[left_index - 1], right[right_index - 1]
                )
                current[right_index] = min(
                    previous[right_index] + 1.0,
                    current[right_index - 1] + 1.0,
                    substitution,
                )
            previous = current
        result = float(np.clip(previous[-1] / max(left_size, right_size), 0.0, 1.0))
        self._lane_cache[key] = result
        return result

    def route_distance(self, left: Lanes, right: Lanes) -> float:
        lane_count = max(len(left), len(right))
        costs = np.ones((lane_count, lane_count), dtype=np.float64)
        for left_index, left_lane in enumerate(left):
            for right_index, right_lane in enumerate(right):
                costs[left_index, right_index] = self.lane_distance(
                    left_lane, right_lane
                )
        rows, columns = linear_sum_assignment(costs)
        return float(np.clip(costs[rows, columns].sum() / lane_count, 0.0, 1.0))

    def matrix(self, records: Sequence[RouteRecord]) -> np.ndarray:
        count = len(records)
        result = np.zeros((count, count), dtype=np.float64)
        for left in range(count):
            for right in range(left + 1, count):
                value = self.route_distance(records[left].lanes, records[right].lanes)
                result[left, right] = value
                result[right, left] = value
        validate_distance_matrix(result)
        return result


def validate_distance_matrix(distance: np.ndarray) -> None:
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("distance matrix must be square")
    if not np.isfinite(distance).all():
        raise ValueError("distance matrix contains NaN/Inf")
    if not np.array_equal(distance, distance.T):
        raise ValueError("distance matrix is not exactly symmetric")
    if not np.array_equal(np.diag(distance), np.zeros(distance.shape[0])):
        raise ValueError("distance matrix diagonal is not exactly zero")
    if float(distance.min()) < 0.0 or float(distance.max()) > 1.0:
        raise ValueError("distance matrix is outside [0,1]")


def exact_equivalence_groups(records: Sequence[RouteRecord]) -> list[list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for record in records:
        groups[record.sequence_key].append(record.embedding_index)
    return sorted(
        (indices for indices in groups.values() if len(indices) > 1),
        key=lambda indices: (indices[0], len(indices)),
    )


def classical_metric_mds(
    distance: np.ndarray,
    *,
    dimensions: int = EMBEDDING_DIM,
    equivalent_groups: Sequence[Sequence[int]] = (),
) -> tuple[np.ndarray, dict[str, Any]]:
    validate_distance_matrix(distance)
    count = distance.shape[0]
    squared = distance.astype(np.float64, copy=False) ** 2
    row_mean = squared.mean(axis=1, keepdims=True)
    centered = -0.5 * (squared - row_mean - row_mean.T + squared.mean())
    centered = (centered + centered.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(centered)
    order = np.argsort(eigenvalues, kind="stable")[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    maximum = max(float(eigenvalues[0]), 0.0)
    threshold = maximum * 1e-12
    positive = np.flatnonzero(eigenvalues > threshold)
    retained = min(dimensions, int(positive.size))
    coordinates = np.zeros((count, dimensions), dtype=np.float64)
    if retained:
        values = eigenvalues[:retained]
        vectors = eigenvectors[:, :retained].copy()
        for dimension in range(retained):
            pivot = int(np.argmax(np.abs(vectors[:, dimension])))
            if vectors[pivot, dimension] < 0:
                vectors[:, dimension] *= -1.0
        coordinates[:, :retained] = vectors * np.sqrt(values)[None, :]

    for group in equivalent_groups:
        indices = np.asarray(group, dtype=np.int64)
        coordinates[indices] = coordinates[indices].mean(axis=0, keepdims=True)
    coordinates -= coordinates.mean(axis=0, keepdims=True)

    source_values = squareform(distance, checks=False)
    embedded_values = pdist(coordinates)
    denominator = float(np.dot(embedded_values, embedded_values))
    scale = (
        float(np.dot(embedded_values, source_values) / denominator)
        if denominator > 0
        else 1.0
    )
    coordinates *= scale
    final = coordinates.astype(np.float32)
    embedded_distance = cdist(final.astype(np.float64), final.astype(np.float64))
    np.fill_diagonal(embedded_distance, 0.0)
    metrics = embedding_metrics(distance, embedded_distance)
    metrics.update(
        {
            "requested_dimensions": dimensions,
            "positive_eigenvalue_count": int(positive.size),
            "retained_dimensions": retained,
            "coordinate_scale": scale,
            "largest_eigenvalue": float(eigenvalues[0]),
            "most_negative_eigenvalue": float(eigenvalues[-1]),
            "positive_eigenvalue_sum": float(eigenvalues[eigenvalues > 0].sum()),
            "retained_positive_eigenvalue_fraction": (
                float(eigenvalues[:retained].sum() / eigenvalues[eigenvalues > 0].sum())
                if retained and bool((eigenvalues > 0).any())
                else 0.0
            ),
        }
    )
    return final, metrics


def _nearest_indices(distance: np.ndarray, index: int, k: int) -> list[int]:
    candidates = [candidate for candidate in range(distance.shape[0]) if candidate != index]
    candidates.sort(key=lambda candidate: (float(distance[index, candidate]), candidate))
    return candidates[:k]


def embedding_metrics(source: np.ndarray, embedded: np.ndarray) -> dict[str, Any]:
    source_values = squareform(source, checks=False)
    embedded_values = squareform(embedded, checks=False)
    residual = embedded_values - source_values
    stress = math.sqrt(
        float(np.dot(residual, residual)) / max(float(np.dot(source_values, source_values)), 1e-30)
    )
    nearest: dict[str, float] = {}
    for k in (1, 5, 10):
        preserved = []
        for index in range(source.shape[0]):
            source_set = set(_nearest_indices(source, index, k))
            embedded_set = set(_nearest_indices(embedded, index, k))
            preserved.append(len(source_set & embedded_set) / k)
        nearest[f"top{k}"] = float(np.mean(preserved))
    row_mae = np.mean(np.abs(embedded - source), axis=1)
    return {
        "kruskal_stress_1": stress,
        "distance_pearson": float(pearsonr(source_values, embedded_values).statistic),
        "distance_spearman": float(spearmanr(source_values, embedded_values).statistic),
        "distance_mean_absolute_error": float(np.mean(np.abs(residual))),
        "distance_max_absolute_error": float(np.max(np.abs(residual))),
        "nearest_neighbor_preservation": nearest,
        "worst_route_index": int(np.argmax(row_mae)),
        "worst_route_mean_absolute_error": float(np.max(row_mae)),
    }


def _write_matrix(path: Path, route_ids: Sequence[str], values: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["route_id", *route_ids])
        for route_id, row in zip(route_ids, values, strict=True):
            writer.writerow([route_id, *(format(float(value), ".9g") for value in row)])


def build_top_neighbors(
    records: Sequence[RouteRecord],
    source_distance: np.ndarray,
    embedded_distance: np.ndarray,
    *,
    k: int = 10,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        index = record.embedding_index
        for rank, neighbor_index in enumerate(
            _nearest_indices(embedded_distance, index, k), 1
        ):
            neighbor = records[neighbor_index]
            source_value = float(source_distance[index, neighbor_index])
            embedded_value = float(embedded_distance[index, neighbor_index])
            rows.append(
                {
                    "source_route_id": record.route_id,
                    "rank": rank,
                    "neighbor_route_id": neighbor.route_id,
                    "source_family": record.family,
                    "neighbor_family": neighbor.family,
                    "source_simai": record.representative_simai,
                    "neighbor_simai": neighbor.representative_simai,
                    "source_occurrences": record.exact_occurrences,
                    "neighbor_occurrences": neighbor.exact_occurrences,
                    "source_similarity": 1.0 - source_value,
                    "embedding_similarity": 1.0 - float(np.clip(embedded_value, 0.0, 1.0)),
                    "source_distance": source_value,
                    "embedding_distance": embedded_value,
                    "absolute_distance_error": abs(embedded_value - source_value),
                    "exact_judge_equivalent": source_value == 0.0,
                }
            )
    return rows


def _cluster_order(distance: np.ndarray) -> np.ndarray:
    condensed = squareform(distance, checks=False)
    tree = linkage(condensed, method="average", optimal_ordering=True)
    return leaves_list(tree).astype(np.int64)


def _plot_heatmap(
    path: Path,
    matrix: np.ndarray,
    order: np.ndarray,
    *,
    title: str,
    label: str,
    cmap: str,
    vmin: float,
    vmax: float,
) -> None:
    figure, axis = plt.subplots(figsize=(14, 12), constrained_layout=True)
    image = axis.imshow(
        matrix[np.ix_(order, order)],
        interpolation="nearest",
        aspect="equal",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    axis.set_title(title)
    axis.set_xlabel("Routes (source-distance cluster order)")
    axis.set_ylabel("Routes (source-distance cluster order)")
    axis.set_xticks([])
    axis.set_yticks([])
    figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02, label=label)
    figure.savefig(
        path,
        dpi=180,
        metadata={"Software": "maimai-ai-charting"},
    )
    plt.close(figure)


def _plot_projection(path: Path, records: Sequence[RouteRecord], embeddings: np.ndarray) -> None:
    figure, axis = plt.subplots(figsize=(14, 11), constrained_layout=True)
    categories = (
        ("basic", "#1f77b4", "o", "71 basic routes"),
        ("chain-observed", "#ff7f0e", "o", "observed chain orientations"),
        ("chain-synthesized", "#d62728", "^", "native-mirror synthesized"),
    )
    for category, color, marker, label in categories:
        if category == "basic":
            indices = [record.embedding_index for record in records if record.family == "basic"]
        elif category == "chain-observed":
            indices = [
                record.embedding_index
                for record in records
                if record.family == "chain" and record.observed
            ]
        else:
            indices = [
                record.embedding_index
                for record in records
                if record.synthesized_by_native_mirror
            ]
        axis.scatter(
            embeddings[indices, 0],
            embeddings[indices, 1],
            s=28 if category == "basic" else 18,
            c=color,
            marker=marker,
            alpha=0.78,
            linewidths=0.25,
            edgecolors="white",
            label=label,
        )
    highlighted = [
        record
        for record in records
        if record.family == "chain" and record.chain_group_rank is not None and record.chain_group_rank <= 8
    ]
    highlighted.extend(record for record in records if record.lane_count == 3)
    for record in highlighted:
        axis.annotate(
            record.route_id,
            (embeddings[record.embedding_index, 0], embeddings[record.embedding_index, 1]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=6,
            alpha=0.85,
        )
    axis.axhline(0.0, color="#999999", linewidth=0.5)
    axis.axvline(0.0, color="#999999", linewidth=0.5)
    axis.set_title(
        f"{len(records)} canonical directed Slide routes: first two metric-MDS coordinates"
    )
    axis.set_xlabel("MDS dimension 1")
    axis.set_ylabel("MDS dimension 2")
    axis.legend(loc="best", frameon=True)
    axis.grid(alpha=0.15)
    figure.savefig(
        path,
        dpi=180,
        metadata={"Software": "maimai-ai-charting"},
    )
    plt.close(figure)


def _markdown_table(rows: Sequence[Sequence[Any]], headers: Sequence[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def _relative(path: Path, base: Path) -> str:
    return Path(path).resolve().relative_to(base.resolve()).as_posix()


def _report_text(
    *,
    records: Sequence[RouteRecord],
    counts: Mapping[str, Any],
    metrics: Mapping[str, Any],
    equivalences: Sequence[Sequence[int]],
    source_distance: np.ndarray,
    embedded_distance: np.ndarray,
    source_summary: Mapping[str, Any],
    core_hashes: Mapping[str, str],
    aliases: Sequence[RouteAlias],
) -> str:
    cross_pairs: list[tuple[float, int, int]] = []
    distortion_pairs: list[tuple[float, int, int]] = []
    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            distance = float(source_distance[left, right])
            if records[left].family != records[right].family:
                cross_pairs.append((distance, left, right))
            distortion_pairs.append(
                (abs(float(embedded_distance[left, right]) - distance), left, right)
            )
    cross_pairs.sort(key=lambda value: (value[0], records[value[1]].route_id, records[value[2]].route_id))
    distortion_pairs.sort(key=lambda value: (-value[0], records[value[1]].route_id, records[value[2]].route_id))
    nearest_rows = [
        (
            records[left].route_id,
            records[left].representative_simai,
            records[right].route_id,
            records[right].representative_simai,
            f"{1.0 - distance:.6f}",
        )
        for distance, left, right in cross_pairs[:12]
    ]
    distortion_rows = [
        (
            records[left].route_id,
            records[right].route_id,
            f"{source_distance[left, right]:.6f}",
            f"{embedded_distance[left, right]:.6f}",
            f"{error:.6f}",
        )
        for error, left, right in distortion_pairs[:12]
    ]
    equivalence_rows = [
        (
            ", ".join(records[index].route_id for index in group),
            " / ".join(records[index].representative_simai for index in group),
            lane_signature(records[group[0]].lanes[0]),
        )
        for group in equivalences[:30]
    ]
    alias_rows = [
        (
            alias.alias_route_id,
            alias.alias_representative_simai,
            alias.canonical_route_id,
            alias.canonical_representative_simai,
            alias.exact_occurrences,
        )
        for alias in aliases
    ]
    long_tail = source_summary["long_tail_lt5"]
    min5 = source_summary["min5"]
    return f"""# 434 类 Slide 路线嵌入与相似度审计

## 结论

- 本资产只构建离线路线目录、语义距离和256维嵌入；没有修改模型、损失或训练脚本。
- 路线目录包含 `{counts['basic_routes']}` 个基础路线和 `{counts['chain_routes']}` 个有向拼接路线，共 `{counts['total_routes']}` 类。
- 原始189个拼接镜像组共有377个有向路线；其中14条与基础路线判定序列完全相同，已作为输入别名映射到对应基础路线，不再占用嵌入类别。
- 14条别名恰好组成7个完整镜像组，因此规范目录保留182个拼接镜像组和363个有向拼接路线。
- 语义距离完全来自有序判定区序列。拼接段数、拼接点、出现次数和Simai语法均不进入距离。
- 规范目录不再保留与basic完全一致的chain ID；原谱中的对应拼接语法应按 `chain_to_basic_aliases.csv` 直接映射到basic ID。

## 数据来源与保留统计

- Festival及之后版本的缓存中共有4,968条拼接Slide branch；本流程不重新扫描maidata。
- 原始Simai链语法1,568种，完整判定轨迹1,285组，原生镜像合并后872组。
- 至少出现5次的189组覆盖 `{min5['native_mirror_covered_branches']}` / 4,968 = `{min5['native_mirror_coverage']:.4%}`。
- 未进入当前目录的长尾有 `{long_tail['groups']}` 组、`{long_tail['branches']}` 条branch，占 `{long_tail['coverage']:.4%}`；既有长尾审计继续保留。

## 距离与嵌入

- 每个判定阶段是一个无序可接受判定区集合；集合使用对称最近邻几何距离，只有集合完全相同才为0。
- 单轨使用单位插入/删除代价的归一化编辑距离；Wifi保留三轨，并用Hungarian匹配和虚拟轨道比较。
- {counts['total_routes']}×{counts['total_routes']}距离矩阵使用deterministic classical metric MDS拟合为256维FP32向量，最终解码语义应使用欧氏距离。
- Kruskal stress-1：`{metrics['kruskal_stress_1']:.8f}`；距离Pearson：`{metrics['distance_pearson']:.8f}`；Spearman：`{metrics['distance_spearman']:.8f}`。
- Top-1/5/10近邻保持率：`{metrics['nearest_neighbor_preservation']['top1']:.4%}` / `{metrics['nearest_neighbor_preservation']['top5']:.4%}` / `{metrics['nearest_neighbor_preservation']['top10']:.4%}`。
- 距离平均绝对误差：`{metrics['distance_mean_absolute_error']:.8f}`；最大绝对误差：`{metrics['distance_max_absolute_error']:.8f}`。

## 基础路线与拼接路线的最近邻

{_markdown_table(nearest_rows, ('基础/拼接ID', 'Simai', '另一侧ID', 'Simai', '源相似度'))}

## 最大嵌入失真路线对

{_markdown_table(distortion_rows, ('路线A', '路线B', '源距离', '嵌入距离', '绝对误差'))}

## 精确判定语义等价组

{_markdown_table(equivalence_rows, ('路线ID', '代表Simai', '判定序列')) if equivalence_rows else '没有发现不同ID的精确判定语义等价。'}

完整等价组见 `zero_distance_equivalences.csv`，完整近邻见 `top10_neighbors.csv`。

## 映射到基础路线的拼接别名

{_markdown_table(alias_rows, ('原拼接ID', '原Simai', '规范basic ID', '规范Simai', '出现次数'))}

完整映射及频次、歌曲数、谱面数和镜像伙伴见 `chain_to_basic_aliases.csv` 与 `chain_to_basic_aliases.json`。

## 核心资产哈希

{_markdown_table([(name, value) for name, value in sorted(core_hashes.items())], ('资产', 'SHA-256'))}

## 未来训练接线（本轮未实现）

后续可让decoder从当前hidden回归256维路线向量，并按到合法路线原型的欧氏距离得到候选分布或最近邻结果。pilot-256仍只有71个基础Slide正样本，因此本资产只能提供统一的合法路线空间，不能证明模型已经学会何时主动使用拼接Slide。
"""


def _document_text(
    counts: Mapping[str, Any], metrics: Mapping[str, Any], catalog_identity: str
) -> str:
    return f"""# Slide 路线目录与256维距离嵌入（434类）

当前规范目录保留左右方向，但移除与基础路线判定序列完全一致的拼接类别。原始189个拼接镜像组共有377个有向路线，其中14条恰好组成7个完整镜像组并映射到14条基础路线。因此目录为71个基础路线加363个有向拼接路线，共434类。

本轮没有接入任何训练代码。完整产物位于 [`artifacts/slide-route-embedding-434-v3`](../../artifacts/slide-route-embedding-434-v3/)，详细结果见该目录的 [`report.md`](../../artifacts/slide-route-embedding-434-v3/report.md)。路线目录identity为 `{catalog_identity}`。

## 冻结口径

- 起点统一旋转到A1，但保留左右方向。
- 判定区集合内部无序，`A2/B2` 表示经过任意一个即可。
- 判定集合使用对称最近邻几何距离：只有集合完全相同才为0；部分重叠仍由未重叠判定区产生正距离。
- 普通及拼接Slide使用一条有序判定序列；Wifi保留三条并行判定序列。
- 距离不读取拼接段数、拼接点、出现次数或语法类别。
- 判定序列完全相同的拼接语法不再分配嵌入ID，而是通过 `chain_to_basic_aliases` 映射到对应基础路线。

## 嵌入方式

完整语义距离矩阵通过classical metric MDS拟合为 `[{counts['total_routes']},256]` FP32向量。语义相似度为 `1-source_distance`；嵌入相似度为 `1-clamp(euclidean_distance,0,1)`。当前拟合stress-1为 `{metrics['kruskal_stress_1']:.8f}`，Top-10近邻保持率为 `{metrics['nearest_neighbor_preservation']['top10']:.4%}`。

## 已保留的历史审计

- 4,968条Festival及之后版本的拼接branch缓存和1,285个完整判定轨迹组不变。
- 189组min-5清单、审阅maidata及683组长尾审计不变。
- 本目录只从上述缓存和MiaCode参考数据派生，不重新扫描maidata。

## 未来使用

后续训练可让decoder预测路线向量，并从434个规范路线原型中按欧氏距离解码；输入预处理应先将14条chain别名改写为对应basic ID。本轮只提供和审计原型空间，没有实现训练、损失或生成接线。
"""


def build_artifacts(paths: SourcePaths) -> dict[str, Any]:
    paths = replace(
        paths,
        workspace=paths.workspace.resolve(),
        basic_catalog=paths.basic_catalog.resolve(),
        chain_inventory=paths.chain_inventory.resolve(),
        slide_data=paths.slide_data.resolve(),
        muri_config=paths.muri_config.resolve(),
        output=paths.output.resolve(),
        document=paths.document.resolve(),
    )
    verify_muri_geometry_contract(paths.muri_config)
    records, counts, aliases = build_route_catalog_with_aliases(paths)
    engine = JudgeTrajectoryDistance(records)
    source_distance = engine.matrix(records)
    equivalences = exact_equivalence_groups(records)
    if equivalences:
        raise ValueError("canonical route catalog still contains duplicate judge sequences")
    embeddings, metrics = classical_metric_mds(
        source_distance,
        dimensions=EMBEDDING_DIM,
        equivalent_groups=equivalences,
    )
    if embeddings.shape != (EXPECTED_TOTAL_ROUTES, EMBEDDING_DIM):
        raise ValueError(f"unexpected embedding shape: {embeddings.shape}")
    if not np.isfinite(embeddings).all():
        raise ValueError("embedding contains NaN/Inf")
    embedded_distance = cdist(embeddings.astype(np.float64), embeddings.astype(np.float64))
    np.fill_diagonal(embedded_distance, 0.0)
    source_similarity = 1.0 - source_distance
    embedded_similarity = 1.0 - np.clip(embedded_distance, 0.0, 1.0)
    route_ids = [record.route_id for record in records]
    output = paths.output
    output.mkdir(parents=True, exist_ok=True)

    catalog_payload = {
        "schema": SCHEMA,
        "policy": (
            "directed-a1-normalized-71-basic-plus-363-canonical-chain-"
            "with-14-chain-to-basic-aliases-v2"
        ),
        "embedding_dimension": EMBEDDING_DIM,
        "counts": counts,
        "routes": [record.to_json() for record in records],
    }
    catalog_identity = canonical_json_sha256(catalog_payload)
    catalog_payload["identity"] = catalog_identity
    catalog_json = output / "route_catalog.json"
    _write_json(catalog_json, catalog_payload)
    catalog_csv = output / "route_catalog.csv"
    catalog_fields = (
        "embedding_index",
        "route_id",
        "family",
        "orientation",
        "representative_simai",
        "representative_segments",
        "trajectory_signature",
        "mirror_partner_id",
        "observed",
        "synthesized_by_native_mirror",
        "exact_occurrences",
        "group_occurrences",
        "song_count",
        "chart_count",
        "base_variant_id",
        "base_motion",
        "chain_group_rank",
        "chain_group_key",
        "lane_count",
        "judge_area_count",
        "lanes_json",
        "examples",
    )
    _write_csv(
        catalog_csv,
        (
            {
                **record.to_json(),
                "lanes_json": json.dumps(
                    record.to_json()["lanes"], ensure_ascii=False, separators=(",", ":")
                ),
            }
            for record in records
        ),
        catalog_fields,
    )

    alias_schema = "maimai.slide_route_chain_to_basic_aliases.v1"
    alias_json = output / "chain_to_basic_aliases.json"
    _write_json(
        alias_json,
        {
            "schema": alias_schema,
            "canonical_catalog_identity": catalog_identity,
            "count": len(aliases),
            "aliases": [alias.to_json() for alias in aliases],
        },
    )
    alias_csv = output / "chain_to_basic_aliases.csv"
    alias_fields = (
        "alias_route_id",
        "canonical_route_id",
        "alias_orientation",
        "alias_representative_simai",
        "alias_representative_segments",
        "trajectory_signature",
        "mirror_partner_alias_route_id",
        "observed",
        "synthesized_by_native_mirror",
        "exact_occurrences",
        "group_occurrences",
        "song_count",
        "chart_count",
        "chain_group_rank",
        "chain_group_key",
        "canonical_representative_simai",
        "canonical_base_variant_id",
        "canonical_base_motion",
        "reason",
    )
    _write_csv(alias_csv, (alias.to_json() for alias in aliases), alias_fields)

    matrix_paths = {
        "source_distance.csv": source_distance,
        "source_similarity.csv": source_similarity,
        "embedding_distance.csv": embedded_distance,
        "embedding_similarity.csv": embedded_similarity,
    }
    for name, matrix in matrix_paths.items():
        _write_matrix(output / name, route_ids, matrix)

    npz_path = output / "route_embeddings_256d.npz"
    np.savez_compressed(
        npz_path,
        schema=np.asarray(SCHEMA),
        catalog_identity=np.asarray(catalog_identity),
        route_ids=np.asarray(route_ids, dtype="U16"),
        embeddings=embeddings,
        source_distance=source_distance.astype(np.float32),
        source_similarity=source_similarity.astype(np.float32),
        embedding_distance=embedded_distance.astype(np.float32),
        embedding_similarity=embedded_similarity.astype(np.float32),
    )

    neighbor_rows = build_top_neighbors(records, source_distance, embedded_distance)
    neighbors_path = output / "top10_neighbors.csv"
    neighbor_fields = tuple(neighbor_rows[0].keys())
    _write_csv(neighbors_path, neighbor_rows, neighbor_fields)

    equivalence_path = output / "zero_distance_equivalences.csv"
    equivalence_rows: list[dict[str, Any]] = []
    for group_index, group in enumerate(equivalences, 1):
        for member_index, route_index in enumerate(group, 1):
            record = records[route_index]
            equivalence_rows.append(
                {
                    "equivalence_group": group_index,
                    "member_index": member_index,
                    "route_id": record.route_id,
                    "family": record.family,
                    "representative_simai": record.representative_simai,
                    "trajectory_signature": record.trajectory_signature,
                }
            )
    _write_csv(
        equivalence_path,
        equivalence_rows,
        (
            "equivalence_group",
            "member_index",
            "route_id",
            "family",
            "representative_simai",
            "trajectory_signature",
        ),
    )

    order = _cluster_order(source_distance)
    cluster_path = output / "cluster_order.csv"
    _write_csv(
        cluster_path,
        (
            {
                "cluster_position": position,
                "embedding_index": int(index),
                "route_id": records[int(index)].route_id,
                "family": records[int(index)].family,
            }
            for position, index in enumerate(order)
        ),
        ("cluster_position", "embedding_index", "route_id", "family"),
    )
    _plot_heatmap(
        output / "source_similarity_heatmap.png",
        source_similarity,
        order,
        title=f"Source judge-trajectory similarity ({len(records)} routes)",
        label="1 - normalized trajectory distance",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    error_matrix = np.abs(embedded_distance - source_distance)
    _plot_heatmap(
        output / "embedding_error_heatmap.png",
        error_matrix,
        order,
        title="256D metric-MDS absolute distance error",
        label="absolute distance error",
        cmap="magma",
        vmin=0.0,
        vmax=float(np.quantile(error_matrix, 0.995)),
    )
    _plot_projection(output / "embedding_projection_2d.png", records, embeddings)

    source_summary = _json(paths.chain_inventory / "chain_judge_native_mirror_summary.json")
    input_paths = {
        "basic_catalog": paths.basic_catalog,
        "chain_exact_groups": paths.chain_inventory / "chain_judge_exact_groups.csv",
        "chain_mirror_groups_min5": paths.chain_inventory
        / "chain_judge_native_mirror_min5.csv",
        "chain_mirror_map": paths.chain_inventory / "chain_judge_native_mirror_map.csv",
        "chain_summary": paths.chain_inventory / "chain_judge_native_mirror_summary.json",
        "slide_data": paths.slide_data,
        "muri_config": paths.muri_config,
        "builder": Path(__file__).resolve(),
    }
    input_hashes = {
        name: sha256_file(path) for name, path in sorted(input_paths.items())
    }
    core_paths = {
        "route_catalog.json": catalog_json,
        "chain_to_basic_aliases.json": alias_json,
        "route_embeddings_256d.npz": npz_path,
        "source_distance.csv": output / "source_distance.csv",
        "top10_neighbors.csv": neighbors_path,
    }
    core_hashes = {name: sha256_file(path) for name, path in core_paths.items()}
    report_path = output / "report.md"
    report_path.write_text(
        _report_text(
            records=records,
            counts=counts,
            metrics=metrics,
            equivalences=equivalences,
            source_distance=source_distance,
            embedded_distance=embedded_distance,
            source_summary=source_summary,
            core_hashes=core_hashes,
            aliases=aliases,
        ),
        encoding="utf-8",
    )
    paths.document.parent.mkdir(parents=True, exist_ok=True)
    paths.document.write_text(
        _document_text(counts, metrics, catalog_identity), encoding="utf-8"
    )

    output_files = sorted(
        path for path in output.iterdir() if path.is_file() and path.name != "manifest.json"
    )
    output_hashes = {path.name: sha256_file(path) for path in output_files}
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "catalog_schema": SCHEMA,
        "alias_schema": alias_schema,
        "catalog_identity": catalog_identity,
        "counts": counts,
        "distance": {
            "stage_set": "symmetric-chamfer-normalized-miacode-pad-center-distance-v2",
            "lane_sequence": "unit-gap-normalized-edit-distance-v1",
            "multi_lane": "hungarian-with-unit-dummy-lanes-v1",
            "segment_count_used": False,
            "syntax_used": False,
            "occurrence_used": False,
            "range": [0.0, 1.0],
            "pad_tokens": list(engine.tokens),
            "pad_distance_scale": engine.pad_distance_scale,
        },
        "embedding": {
            "method": "deterministic-classical-metric-mds-v1",
            "dimensions": EMBEDDING_DIM,
            "dtype": "float32",
            "distance": "euclidean",
            "metrics": metrics,
        },
        "equivalence_group_count": len(equivalences),
        "canonicalization": {
            "policy": "exact-chain-judge-sequence-to-basic-v1",
            "chain_alias_count": len(aliases),
            "alias_csv": alias_csv.name,
            "alias_json": alias_json.name,
        },
        "inputs": {
            name: {
                "path": _relative(path, paths.workspace),
                "sha256": input_hashes[name],
            }
            for name, path in sorted(input_paths.items())
        },
        "outputs": {
            name: {"sha256": value, "bytes": (output / name).stat().st_size}
            for name, value in output_hashes.items()
        },
        "document": {
            "path": _relative(paths.document, paths.workspace),
            "sha256": sha256_file(paths.document),
        },
    }
    _write_json(output / "manifest.json", manifest)
    validate_generated_artifacts(output, paths.document)
    return manifest


def validate_generated_artifacts(output: Path, document: Path) -> None:
    manifest = _json(output / "manifest.json")
    if manifest["counts"]["total_routes"] != EXPECTED_TOTAL_ROUTES:
        raise ValueError(
            f"generated manifest route count is not {EXPECTED_TOTAL_ROUTES}"
        )
    catalog = _json(output / "route_catalog.json")
    if len(catalog["routes"]) != EXPECTED_TOTAL_ROUTES:
        raise ValueError(
            f"generated catalog route count is not {EXPECTED_TOTAL_ROUTES}"
        )
    route_ids = {row["route_id"] for row in catalog["routes"]}
    aliases = _json(output / "chain_to_basic_aliases.json")
    if aliases["count"] != EXPECTED_CHAIN_ALIASES_TO_BASIC:
        raise ValueError("generated chain-to-basic alias count is invalid")
    if len(aliases["aliases"]) != EXPECTED_CHAIN_ALIASES_TO_BASIC:
        raise ValueError("generated chain-to-basic alias payload is incomplete")
    for alias in aliases["aliases"]:
        if alias["alias_route_id"] in route_ids:
            raise ValueError("chain alias remains in canonical route catalog")
        if alias["canonical_route_id"] not in route_ids:
            raise ValueError("chain alias points outside canonical route catalog")
        if not alias["canonical_route_id"].startswith("B"):
            raise ValueError("chain alias does not map to a basic route")
    with np.load(output / "route_embeddings_256d.npz") as archive:
        if archive["embeddings"].shape != (EXPECTED_TOTAL_ROUTES, EMBEDDING_DIM):
            raise ValueError("generated NPZ embedding shape is invalid")
        if not np.isfinite(archive["embeddings"]).all():
            raise ValueError("generated NPZ contains non-finite embeddings")
    neighbor_rows = _csv(output / "top10_neighbors.csv")
    if len(neighbor_rows) != EXPECTED_TOTAL_ROUTES * 10:
        raise ValueError("Top-10 neighbor file has the wrong row count")
    if any(row["source_route_id"] == row["neighbor_route_id"] for row in neighbor_rows):
        raise ValueError("Top-10 neighbor file contains self-neighbors")
    for name, detail in manifest["outputs"].items():
        path = output / name
        if sha256_file(path) != detail["sha256"]:
            raise ValueError(f"output hash mismatch: {name}")
    if sha256_file(document) != manifest["document"]["sha256"]:
        raise ValueError("document hash mismatch")
    for name in (
        "source_similarity_heatmap.png",
        "embedding_error_heatmap.png",
        "embedding_projection_2d.png",
    ):
        if (output / name).stat().st_size < 10_000:
            raise ValueError(f"generated figure appears empty: {name}")


def main() -> int:
    args = parse_args()
    defaults = default_paths()
    paths = replace(
        defaults,
        basic_catalog=args.basic_catalog,
        chain_inventory=args.chain_inventory,
        slide_data=args.slide_data,
        muri_config=args.muri_config,
        output=args.output,
        document=args.document,
    )
    manifest = build_artifacts(paths)
    print(
        json.dumps(
            {
                "output": str(paths.output.resolve()),
                "document": str(paths.document.resolve()),
                "catalog_identity": manifest["catalog_identity"],
                "counts": manifest["counts"],
                "embedding_metrics": manifest["embedding"]["metrics"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
