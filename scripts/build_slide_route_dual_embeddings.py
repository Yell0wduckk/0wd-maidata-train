from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import zipfile

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from build_slide_route_embeddings import (
    classical_metric_mds,
    embedding_metrics,
    sha256_file,
    validate_distance_matrix,
)


SCHEMA = "maimai.slide_route_dual_embedding.v4"
MANIFEST_SCHEMA = "maimai.slide_route_dual_embedding_manifest.v4"
CATALOG_IDENTITY = "acae86c44abc032dfb2a4a77921f4ca6f741407d24a27a56a5be503f0b528cee"
ROUTE_COUNT = 434
BASIC_ROUTE_COUNT = 71
MDS_DIMENSION = 256
PROTOTYPE_DIMENSION = 256
TRAJECTORY_POINTS = 256
TRAJECTORY_DISTANCE_SCALE = 410.0


def default_paths() -> tuple[Path, Path, Path]:
    script = Path(__file__).resolve()
    workspace = script.parents[2]
    return (
        workspace / "artifacts" / "slide-route-embedding-434-v3",
        workspace / "tools" / "MiaCode" / "assets" / "reference" / "slide_data.json",
        workspace / "artifacts" / "slide-route-embedding-434-v4-dual",
    )


def parse_args() -> argparse.Namespace:
    v3, slide_data, output = default_paths()
    parser = argparse.ArgumentParser(
        description="Build the frozen 434-route judge/continuous-trajectory dual embedding."
    )
    parser.add_argument("--judge-v3", type=Path, default=v3)
    parser.add_argument("--slide-data", type=Path, default=slide_data)
    parser.add_argument("--output", type=Path, default=output)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def write_csv(
    path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_matrix(path: Path, route_ids: Sequence[str], matrix: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("route_id", *route_ids))
        for route_id, row in zip(route_ids, matrix, strict=True):
            writer.writerow((route_id, *(format(float(value), ".9g") for value in row)))


def write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write an np.load-compatible archive without wall-clock ZIP metadata."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.asanyarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def points_from_samples(samples: Sequence[Mapping[str, Any]]) -> np.ndarray:
    points = np.asarray([[sample["x"], sample["y"]] for sample in samples], dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (2,) or points.shape[0] < 2:
        raise ValueError(f"invalid MiaCode trajectory samples: {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("MiaCode trajectory contains NaN/Inf")
    return points


def concatenate_segments(segments: Sequence[np.ndarray]) -> np.ndarray:
    if not segments:
        raise ValueError("route has no trajectory segments")
    result = segments[0]
    for segment in segments[1:]:
        if np.linalg.norm(result[-1] - segment[0]) > 1e-3:
            raise ValueError(
                "MiaCode chain trajectory is discontinuous: "
                f"{result[-1].tolist()} -> {segment[0].tolist()}"
            )
        result = np.concatenate((result, segment[1:]), axis=0)
    return result


def resample_by_arclength(points: np.ndarray, count: int = TRAJECTORY_POINTS) -> np.ndarray:
    deltas = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate(([True], deltas > 1e-12))
    points = points[keep]
    if points.shape[0] < 2:
        raise ValueError("trajectory has zero arc length")
    arclength = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    if arclength[-1] <= 0.0:
        raise ValueError("trajectory has zero arc length")
    target = np.linspace(0.0, arclength[-1], count, dtype=np.float64)
    return np.column_stack(
        (
            np.interp(target, arclength, points[:, 0]),
            np.interp(target, arclength, points[:, 1]),
        )
    )


def route_lanes(record: Mapping[str, Any], slide_data: Mapping[str, Any]) -> tuple[np.ndarray, ...]:
    segments = str(record["representative_segments"]).split("|")
    if len(segments) == 1 and segments[0] in slide_data["wifi"]:
        lane_samples = slide_data["wifi"][segments[0]]["lane_samples"]
        return tuple(resample_by_arclength(points_from_samples(lane)) for lane in lane_samples)
    lane_segments: list[np.ndarray] = []
    for segment in segments:
        if segment not in slide_data["slides"]:
            raise KeyError(f"MiaCode slide_data has no ordinary segment {segment!r}")
        lane_segments.append(points_from_samples(slide_data["slides"][segment]["samples"]))
    return (resample_by_arclength(concatenate_segments(lane_segments)),)


def lane_distance(left: np.ndarray, right: np.ndarray) -> float:
    rms = np.sqrt(np.mean(np.sum((left - right) ** 2, axis=1)))
    return float(np.clip(rms / TRAJECTORY_DISTANCE_SCALE, 0.0, 1.0))


def route_distance(left: tuple[np.ndarray, ...], right: tuple[np.ndarray, ...]) -> float:
    lane_count = max(len(left), len(right))
    costs = np.ones((lane_count, lane_count), dtype=np.float64)
    for left_index, left_lane in enumerate(left):
        for right_index, right_lane in enumerate(right):
            costs[left_index, right_index] = lane_distance(left_lane, right_lane)
    rows, columns = linear_sum_assignment(costs)
    return float(np.clip(costs[rows, columns].sum() / lane_count, 0.0, 1.0))


def trajectory_matrix(lanes: Sequence[tuple[np.ndarray, ...]]) -> np.ndarray:
    result = np.zeros((len(lanes), len(lanes)), dtype=np.float64)
    for left in range(len(lanes)):
        for right in range(left + 1, len(lanes)):
            value = route_distance(lanes[left], lanes[right])
            result[left, right] = value
            result[right, left] = value
    validate_distance_matrix(result)
    return result


def embedded_distance(embeddings: np.ndarray) -> np.ndarray:
    result = cdist(embeddings.astype(np.float64), embeddings.astype(np.float64))
    np.fill_diagonal(result, 0.0)
    return result


def normalized_block(block: np.ndarray) -> tuple[np.ndarray, float]:
    centered = block.astype(np.float64) - block.astype(np.float64).mean(axis=0, keepdims=True)
    rms = float(np.sqrt(np.mean(centered**2)))
    if not np.isfinite(rms) or rms <= 0.0:
        raise ValueError("prototype block has invalid global RMS")
    return (centered / rms).astype(np.float32), rms


def build(judge_v3: Path, slide_data_path: Path, output: Path) -> dict[str, Any]:
    judge_v3 = judge_v3.resolve()
    slide_data_path = slide_data_path.resolve()
    output = output.resolve()
    catalog = read_json(judge_v3 / "route_catalog.json")
    if catalog["identity"] != CATALOG_IDENTITY:
        raise ValueError(f"unexpected frozen 434-v3 catalog identity: {catalog['identity']}")
    records = catalog["routes"]
    if len(records) != ROUTE_COUNT:
        raise ValueError(f"expected {ROUTE_COUNT} routes, found {len(records)}")
    route_ids = [str(record["route_id"]) for record in records]
    expected_basic_ids = [f"B{index:03d}" for index in range(BASIC_ROUTE_COUNT)]
    if route_ids[:BASIC_ROUTE_COUNT] != expected_basic_ids:
        raise ValueError("B000-B070 are not the first 71 frozen catalog routes")

    with np.load(judge_v3 / "route_embeddings_256d.npz") as judge_archive:
        judge_ids = judge_archive["route_ids"].astype(str).tolist()
        judge_embeddings = judge_archive["embeddings"].astype(np.float32)
        judge_source_distance = judge_archive["source_distance"].astype(np.float64)
    if judge_ids != route_ids or judge_embeddings.shape != (ROUTE_COUNT, MDS_DIMENSION):
        raise ValueError("frozen judge embedding does not match the route catalog")
    judge_distance = np.clip(embedded_distance(judge_embeddings), 0.0, 1.0)

    slide_data = read_json(slide_data_path)
    lanes = [route_lanes(record, slide_data) for record in records]
    trajectory_source_distance = trajectory_matrix(lanes)
    trajectory_embeddings, trajectory_metrics = classical_metric_mds(
        trajectory_source_distance, dimensions=MDS_DIMENSION
    )
    trajectory_embedding_distance = embedded_distance(trajectory_embeddings)
    trajectory_distance = np.clip(trajectory_embedding_distance, 0.0, 1.0)

    judge_block, judge_block_rms = normalized_block(judge_embeddings[:, :128])
    trajectory_block, trajectory_block_rms = normalized_block(trajectory_embeddings[:, :128])
    global_prototypes = np.concatenate((judge_block, trajectory_block), axis=1).astype(np.float32)
    route71_prototypes = np.zeros((BASIC_ROUTE_COUNT + 1, PROTOTYPE_DIMENSION), dtype=np.float32)
    route71_prototypes[1:] = global_prototypes[:BASIC_ROUTE_COUNT]
    judge_route71 = judge_distance[:BASIC_ROUTE_COUNT, :BASIC_ROUTE_COUNT].astype(np.float32)
    trajectory_route71 = trajectory_distance[:BASIC_ROUTE_COUNT, :BASIC_ROUTE_COUNT].astype(np.float32)

    straight = route_ids.index("B000")
    arc = next(
        index
        for index, record in enumerate(records)
        if record["representative_segments"] == "1>3" and record["family"] == "basic"
    )
    one_three_distance = float(trajectory_source_distance[straight, arc])
    if not 0.1240 <= one_three_distance <= 0.1255:
        raise ValueError(f"1-3/1>3 trajectory distance drifted: {one_three_distance}")

    output.mkdir(parents=True, exist_ok=True)
    write_matrix(output / "trajectory_source_distance.csv", route_ids, trajectory_source_distance)
    write_matrix(output / "trajectory_embedding_distance.csv", route_ids, trajectory_embedding_distance)
    write_matrix(output / "judge_route71_distance.csv", expected_basic_ids, judge_route71)
    write_matrix(output / "trajectory_route71_distance.csv", expected_basic_ids, trajectory_route71)
    write_csv(
        output / "route71_prototypes.csv",
        (
            {
                "route_input_id": index + 1,
                "route_id": route_id,
                "representative_simai": records[index]["representative_simai"],
                "base_variant_id": records[index]["base_variant_id"],
            }
            for index, route_id in enumerate(expected_basic_ids)
        ),
        ("route_input_id", "route_id", "representative_simai", "base_variant_id"),
    )
    npz_path = output / "dual_route_embeddings_256d.npz"
    write_deterministic_npz(
        npz_path,
        {
            "schema": np.asarray(SCHEMA),
            "catalog_identity": np.asarray(CATALOG_IDENTITY),
            "route_ids": np.asarray(route_ids, dtype="U16"),
            "judge_embeddings": judge_embeddings,
            "judge_source_distance": judge_source_distance.astype(np.float32),
            "judge_embedding_distance": embedded_distance(judge_embeddings).astype(np.float32),
            "trajectory_embeddings": trajectory_embeddings,
            "trajectory_source_distance": trajectory_source_distance.astype(np.float32),
            "trajectory_embedding_distance": trajectory_embedding_distance.astype(np.float32),
            "route71_ids": np.asarray(expected_basic_ids, dtype="U4"),
            "route71_prototypes": route71_prototypes,
            "judge_route71_distance": judge_route71,
            "trajectory_route71_distance": trajectory_route71,
        },
    )

    judge_metrics = embedding_metrics(
        judge_source_distance, embedded_distance(judge_embeddings)
    )
    report = f"""# 434 类 Slide Judge / Trajectory 双距离资产（dual-v4）

本目录在冻结的434-v3路线目录上增加连续轨迹距离，没有重新扫描maidata，也没有改变434类身份。

- Judge block：原434-v3判定序列MDS，完整复用。
- Trajectory block：MiaCode原生XY轨迹逐段首尾拼接，按整条路线弧长重采样{TRAJECTORY_POINTS}点。
- 单轨距离：点位RMS / {TRAJECTORY_DISTANCE_SCALE:g}，裁剪至[0,1]。
- Wifi：保留三轨，以Hungarian匹配；缺失轨道代价为1。
- 训练距离：分别取两套434类MDS的B000-B070欧氏距离，再裁剪至[0,1]。
- Route输入原型：Judge前128维与Trajectory前128维分别中心化、缩放到单位全局RMS后拼接；第0行保留为不适用。

校准对 `1-3` / `1>3` 的连续轨迹源距离为 `{one_three_distance:.9f}`。

Judge MDS stress-1：`{judge_metrics['kruskal_stress_1']:.9f}`。  
Trajectory MDS stress-1：`{trajectory_metrics['kruskal_stress_1']:.9f}`。
"""
    (output / "report.md").write_text(report, encoding="utf-8")

    core_files = (
        "dual_route_embeddings_256d.npz",
        "judge_route71_distance.csv",
        "trajectory_route71_distance.csv",
        "route71_prototypes.csv",
        "trajectory_source_distance.csv",
        "trajectory_embedding_distance.csv",
        "report.md",
    )
    hashes = {name: sha256_file(output / name) for name in core_files}
    prototype_hash = hashlib.sha256(route71_prototypes.tobytes(order="C")).hexdigest()
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "catalog_identity": CATALOG_IDENTITY,
        "counts": {"all_routes": ROUTE_COUNT, "basic_training_routes": BASIC_ROUTE_COUNT},
        "judge": {
            "source": "frozen-slide-route-embedding-434-v3",
            "embedding_dimensions": MDS_DIMENSION,
            "metrics": judge_metrics,
        },
        "trajectory": {
            "source": "MiaCode-native-xy-v1",
            "points": TRAJECTORY_POINTS,
            "distance_scale": TRAJECTORY_DISTANCE_SCALE,
            "single_lane": "normalized-pointwise-rms-v1",
            "multi_lane": "hungarian-with-unit-dummy-lanes-v1",
            "embedding_dimensions": MDS_DIMENSION,
            "metrics": trajectory_metrics,
            "one_three_straight_vs_arc_source_distance": one_three_distance,
        },
        "route71": {
            "ids": expected_basic_ids,
            "judge_distance": "clipped-euclidean-distance-in-434-judge-mds",
            "trajectory_distance": "clipped-euclidean-distance-in-434-trajectory-mds",
            "prototype": "centered-unit-global-rms-judge128-plus-trajectory128-v1",
            "judge_block_pre_normalization_rms": judge_block_rms,
            "trajectory_block_pre_normalization_rms": trajectory_block_rms,
            "prototype_sha256": prototype_hash,
        },
        "inputs": {
            "judge_catalog": {
                "path": str((judge_v3 / "route_catalog.json").as_posix()),
                "sha256": sha256_file(judge_v3 / "route_catalog.json"),
            },
            "judge_embedding": {
                "path": str((judge_v3 / "route_embeddings_256d.npz").as_posix()),
                "sha256": sha256_file(judge_v3 / "route_embeddings_256d.npz"),
            },
            "chain_aliases": {
                "path": str((judge_v3 / "chain_to_basic_aliases.json").as_posix()),
                "sha256": sha256_file(judge_v3 / "chain_to_basic_aliases.json"),
            },
            "slide_data": {
                "path": str(slide_data_path.as_posix()),
                "sha256": sha256_file(slide_data_path),
            },
            "builder": {
                "path": str(Path(__file__).resolve().as_posix()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "outputs": {
            name: {"sha256": digest, "bytes": (output / name).stat().st_size}
            for name, digest in hashes.items()
        },
    }
    write_json(output / "manifest.json", manifest)
    validate(output)
    return manifest


def validate(output: Path) -> None:
    manifest = read_json(output / "manifest.json")
    if manifest["catalog_identity"] != CATALOG_IDENTITY:
        raise ValueError("dual manifest catalog identity mismatch")
    with np.load(output / "dual_route_embeddings_256d.npz") as archive:
        expected_shapes = {
            "judge_embeddings": (ROUTE_COUNT, MDS_DIMENSION),
            "trajectory_embeddings": (ROUTE_COUNT, MDS_DIMENSION),
            "route71_prototypes": (BASIC_ROUTE_COUNT + 1, PROTOTYPE_DIMENSION),
            "judge_route71_distance": (BASIC_ROUTE_COUNT, BASIC_ROUTE_COUNT),
            "trajectory_route71_distance": (BASIC_ROUTE_COUNT, BASIC_ROUTE_COUNT),
        }
        for name, shape in expected_shapes.items():
            if archive[name].shape != shape or not np.isfinite(archive[name]).all():
                raise ValueError(f"invalid {name}: {archive[name].shape}")
        for name in ("judge_route71_distance", "trajectory_route71_distance"):
            validate_distance_matrix(archive[name].astype(np.float64))
        if not np.array_equal(archive["route71_prototypes"][0], np.zeros(PROTOTYPE_DIMENSION)):
            raise ValueError("route_input_id=0 prototype is not zero")
    for name, detail in manifest["outputs"].items():
        if sha256_file(output / name) != detail["sha256"]:
            raise ValueError(f"output hash mismatch: {name}")


def main() -> int:
    args = parse_args()
    manifest = build(args.judge_v3, args.slide_data, args.output)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "catalog_identity": manifest["catalog_identity"],
        "prototype_sha256": manifest["route71"]["prototype_sha256"],
        "trajectory_1-3_vs_1>3": manifest["trajectory"]["one_three_straight_vs_arc_source_distance"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
