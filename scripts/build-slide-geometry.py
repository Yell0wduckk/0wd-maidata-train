"""Audit, build and verify the small, independent rendered Route-71 asset."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from maimai_baseline.contract import CATALOG_IDENTITY
from maimai_baseline.route_contract import GEOMETRY_FILE, GEOMETRY_SCHEMA, geometry_identity


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_deterministic_npz(path, arrays):
    """Keep asset bytes stable without importing historical plotting tools."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.asanyarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def prepare(slide_data, judge_artifact):
    from maimai_baseline.data.route_geometry import describe_lane, route_features, route_distances, pca_prototypes
    reference = json.loads(slide_data.read_text("utf-8"))
    catalog_path = PROJECT / "src/maimai_baseline/assets/slide_variant_catalog.json"
    catalog = json.loads(catalog_path.read_text("utf-8"))["variants"]
    judge_manifest = json.loads((judge_artifact / "manifest.json").read_text("utf-8"))
    source = judge_artifact / "dual_route_embeddings_256d.npz"
    if (judge_manifest["catalog_identity"] != CATALOG_IDENTITY or
            sha(source) != judge_manifest["outputs"][source.name]["sha256"]):
        raise ValueError("source Judge artifact identity/hash mismatch")
    with np.load(source, allow_pickle=False) as archive:
        if archive["route_ids"][:71].tolist() != [f"B{i:03d}" for i in range(71)]:
            raise ValueError("source Judge class ordering mismatch")
        judge = archive["judge_source_distance"][:71, :71].copy()
    routes = []
    names = []
    for i, variant in enumerate(catalog):
        if variant["variant_id"] != i:
            raise ValueError("basic class catalog ordering mismatch")
        name = variant["canonical_path"]
        lanes = (reference["wifi"][name]["lane_samples"] if name in reference["wifi"] else
                 [reference["slides"][name]["samples"]])
        routes.append(tuple(describe_lane([[p["x"], p["y"]] for p in lane]) for lane in lanes))
        names.append(name)
    maximum = max(lane.length for lanes in routes for lane in lanes)
    features = route_features(routes, maximum)
    prototypes, pca, mean, basis = pca_prototypes(features)
    shape, components = route_distances(routes, maximum)
    points = np.zeros((71, 3, 128, 2), dtype=np.float32)
    lengths = np.zeros((71, 3), dtype=np.float64)
    mask = np.zeros((71, 3), dtype=np.bool_)
    for i, lanes in enumerate(routes):
        for slot, lane in zip(([1] if len(lanes) == 1 else [0, 1, 2]), lanes):
            points[i, slot], lengths[i, slot], mask[i, slot] = lane.points, lane.length, True
    arrays = dict(route71_ids=np.asarray([f"B{i:03d}" for i in range(71)]),
                  route71_prototypes=prototypes, judge_route71_distance=judge.astype(np.float32),
                  trajectory_route71_distance=shape.astype(np.float32),
                  features=features, pca_mean=mean, pca_basis=basis,
                  points=points, lengths=lengths, lane_mask=mask, route_names=np.asarray(names),
                  **{f"distance_{key}": value.astype(np.float32) for key, value in components.items()})
    provenance = {"slide_data": {"path": str(slide_data), "sha256": sha(slide_data)},
                  "judge_source": {"path": str(source), "sha256": sha(source)},
                  "catalog": {"path": str(catalog_path), "sha256": sha(catalog_path)},
                  "builder": {"sha256": sha(Path(__file__))},
                  "geometry_implementation": {"sha256": sha(PROJECT / "src/maimai_baseline/data/route_geometry.py")}}
    return arrays, dict(inputs=provenance, maximum_lane_length=maximum, pca=pca)


def plots(output, arrays):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names, points, masks = arrays["route_names"], arrays["points"], arrays["lane_mask"]
    def draw(ax, i):
        for lane in points[i, masks[i]]:
            ax.plot(lane[:, 0], lane[:, 1], lw=1.5)
            a, b = lane[60], lane[67]
            ax.annotate("", b, a, arrowprops=dict(arrowstyle="->", lw=1))
        ax.set(xlim=(-280, 280), ylim=(280, -280), aspect="equal")
        ax.set_axis_off()
    fig, axes = plt.subplots(8, 9, figsize=(15, 14))
    for i, ax in enumerate(axes.flat):
        ax.set_axis_off()
        if i < 71:
            draw(ax, i)
            ax.set_title(f"B{i:03d} {names[i]}  L={arrays['lengths'][i].max()/410:.2f}D", fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "route-atlas.png", dpi=140)
    plt.close(fig)
    chosen = [0, 2, 5, 12, 20, 30, 40, 50, 60, 65, 69, 70]
    fig, axes = plt.subplots(len(chosen), 6, figsize=(13, 22))
    for row, i in enumerate(chosen):
        nearest = np.argsort(arrays["trajectory_route71_distance"][i], kind="stable")
        nearest = [i] + [int(j) for j in nearest if j != i][:5]
        for ax, j in zip(axes[row], nearest):
            draw(ax, j)
            ax.set_title(f"{names[j]}  d={arrays['trajectory_route71_distance'][i,j]:.3f}", fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "shape-neighbors.png", dpi=130)
    plt.close(fig)


def verify(output):
    from maimai_baseline.model.route import load_route_artifacts
    manifest = json.loads((output / "manifest.json").read_text("utf-8"))
    artifact = load_route_artifacts(output)
    for name, record in manifest["outputs"].items():
        if sha(output / name) != record["sha256"]:
            raise ValueError(f"geometry output hash mismatch: {name}")
    with np.load(output / GEOMETRY_FILE, allow_pickle=False) as a:
        centered = a["features"] - a["pca_mean"]
        projected = centered @ a["pca_basis"].T
        if not np.allclose(projected @ a["pca_basis"], centered, atol=1e-9, rtol=1e-9):
            raise ValueError("geometry PCA reconstruction mismatch")
        if not np.allclose(projected / manifest["pca"]["scale"], artifact.prototypes[1:].numpy(), atol=1e-6):
            raise ValueError("geometry prototype reconstruction mismatch")
        if a["lane_mask"].sum(axis=1).tolist() != [1] * 70 + [3]:
            raise ValueError("rendered lane counts mismatch")
    return {"passed": True, "route_count": 71, "identity": artifact.identity}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("action", choices=("audit", "build", "verify"))
    p.add_argument("--slide-data", type=Path)
    p.add_argument("--judge-artifact", type=Path)
    p.add_argument("--output", type=Path, default=PROJECT / "training-dataset/slide-geometry/route-artifact")
    args = p.parse_args(argv)
    output = args.output.resolve()
    if args.action == "verify":
        print(json.dumps(verify(output)))
        return 0
    if not args.slide_data or not args.judge_artifact:
        p.error("audit/build require --slide-data and --judge-artifact")
    arrays, metadata = prepare(args.slide_data.resolve(), args.judge_artifact.resolve())
    if args.action == "audit":
        print(json.dumps(dict(written=False, routes=71, **metadata), ensure_ascii=False))
        return 0
    if output.exists():
        raise ValueError("refusing to overwrite geometry output")
    output.mkdir(parents=True)
    write_deterministic_npz(output / GEOMETRY_FILE, arrays)
    plots(output, arrays)
    with (output / "neighbors.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "neighbor", "shape", "xy", "direction", "turn", "length", "judge"])
        for i, name in enumerate(arrays["route_names"]):
            neighbors = [j for j in np.argsort(arrays["trajectory_route71_distance"][i], kind="stable") if j != i][:10]
            for j in neighbors:
                writer.writerow([name, arrays["route_names"][j], arrays["trajectory_route71_distance"][i,j],
                                 *[arrays[f"distance_{k}"][i,j] for k in ("xy", "direction", "turn", "length")],
                                 arrays["judge_route71_distance"][i,j]])
    manifest = dict(schema=GEOMETRY_SCHEMA, catalog_identity=CATALOG_IDENTITY,
                    route_geometry=geometry_identity(), **metadata,
                    route71={"prototype_sha256": hashlib.sha256(arrays["route71_prototypes"].tobytes()).hexdigest()},
                    outputs={name: {"sha256": sha(output / name)} for name in
                             (GEOMETRY_FILE, "route-atlas.png", "shape-neighbors.png", "neighbors.csv")})
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = verify(output)
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(output=str(output), **report), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
