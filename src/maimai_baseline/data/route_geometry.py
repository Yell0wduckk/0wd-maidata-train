"""Offline rendered-curve descriptors; no model or training dependencies."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment
from maimai_baseline.route_contract import (
    GEOMETRY_DIM, GEOMETRY_POINTS, GEOMETRY_SCALE, GEOMETRY_WEIGHTS,
)


@dataclass(frozen=True)
class LaneGeometry:
    points: np.ndarray
    direction: np.ndarray
    turn: np.ndarray
    length: float


def describe_lane(points, count=GEOMETRY_POINTS):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("invalid rendered XY curve")
    points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-12]]
    if len(points) < 2 or count < 3:
        raise ValueError("zero-length curve or insufficient sampling points")
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    targets = np.linspace(0., arc[-1], count)
    sampled = np.column_stack([np.interp(targets, arc, points[:, i]) for i in range(2)])
    delta = np.diff(sampled, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    if np.any(lengths <= 1e-12):
        raise ValueError("resampled curve contains a zero-length edge")
    direction = delta / lengths[:, None]
    a, b = direction[:-1], direction[1:]
    turn = np.column_stack((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], (a * b).sum(1)))
    return LaneGeometry(sampled, direction, turn, float(arc[-1]))


def lane_blocks(lane, maximum_length):
    if not np.isfinite(maximum_length) or maximum_length <= 0:
        raise ValueError("invalid catalog length scale")
    return dict(xy=lane.points / GEOMETRY_SCALE / np.sqrt(len(lane.points)),
                direction=lane.direction / 2 / np.sqrt(len(lane.direction)),
                turn=lane.turn / 2 / np.sqrt(len(lane.turn)),
                length=np.asarray([lane.length / maximum_length]))


def lane_distance_components(left, right, maximum_length):
    a, b = lane_blocks(left, maximum_length), lane_blocks(right, maximum_length)
    return {key: float(np.linalg.norm(a[key] - b[key])) for key in GEOMETRY_WEIGHTS}


def lane_distance(left, right, maximum_length):
    parts = lane_distance_components(left, right, maximum_length)
    return float(np.sqrt(sum(GEOMETRY_WEIGHTS[k] * v * v for k, v in parts.items())))


def route_distances(routes, maximum_length):
    count = len(routes)
    result = np.zeros((count, count), dtype=np.float64)
    parts = {key: np.zeros_like(result) for key in GEOMETRY_WEIGHTS}
    for i in range(count):
        for j in range(i + 1, count):
            n = max(len(routes[i]), len(routes[j]))
            costs = np.ones((n, n))
            components = {key: np.ones((n, n)) for key in parts}
            for a, left in enumerate(routes[i]):
                for b, right in enumerate(routes[j]):
                    values = lane_distance_components(left, right, maximum_length)
                    costs[a, b] = np.sqrt(sum(GEOMETRY_WEIGHTS[k] * v*v for k, v in values.items()))
                    for key, value in values.items():
                        components[key][a, b] = value
            a, b = linear_sum_assignment(costs)
            result[i, j] = result[j, i] = np.clip(costs[a, b].mean(), 0, 1)
            for key in parts:
                parts[key][i, j] = parts[key][j, i] = components[key][a, b].mean()
    return result, parts


def route_features(routes, maximum_length):
    result = []
    for lanes in routes:
        if len(lanes) not in (1, 3):
            raise ValueError("rendered route requires one or three lanes")
        flat = [np.concatenate([np.sqrt(GEOMETRY_WEIGHTS[k]) * v.ravel()
                                for k, v in lane_blocks(lane, maximum_length).items()]) for lane in lanes]
        mask = [1., 1., 1.]
        if len(flat) == 1:
            flat = [np.zeros_like(flat[0]), flat[0], np.zeros_like(flat[0])]
            mask = [0., 1., 0.]
        result.append(np.r_[np.concatenate(flat), mask])
    return np.asarray(result, dtype=np.float64)


def pca_prototypes(features):
    features = np.asarray(features, dtype=np.float64)
    if features.shape[0] != 71 or not np.isfinite(features).all():
        raise ValueError("PCA requires 71 finite route descriptors")
    mean = features.mean(0)
    centered = features - mean
    u, s, vh = np.linalg.svd(centered, full_matrices=False)
    retained = min(GEOMETRY_DIM, int((s > s[0] * 1e-12).sum()))
    if not retained:
        raise ValueError("route descriptors have no geometric variation")
    basis = np.zeros((GEOMETRY_DIM, features.shape[1]))
    basis[:retained] = vh[:retained]
    for vector in basis[:retained]:
        if vector[np.argmax(np.abs(vector))] < 0:
            vector *= -1
    projected = centered @ basis.T
    if not np.allclose(projected @ basis, centered, atol=1e-9, rtol=1e-9):
        raise ValueError("PCA discarded nonzero geometry components")
    scale = float(np.sqrt(np.mean(np.sum(projected**2, axis=1))))
    prototypes = np.zeros((72, GEOMETRY_DIM), dtype=np.float32)
    prototypes[1:] = projected / scale
    pair = np.linalg.norm(prototypes[1:, None] - prototypes[None, 1:], axis=-1)
    if np.min(pair[np.triu_indices(71, 1)]) <= 1e-6:
        raise ValueError("geometry cannot distinguish all route classes")
    return prototypes, dict(rank=retained, scale=scale), mean, basis
