"""Explicit read-only Route-71 asset adapter; no legacy Python dependency."""
from .losses import RouteGeometryLoss


def load_route_distances(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as archive:
        ids = tuple(archive['route71_ids'].astype(str).tolist())
        if ids != tuple(f'B{i:03d}' for i in range(71)):
            raise ValueError('Route-71 geometry order mismatch')
        return RouteGeometryLoss(archive['judge_route71_distance'].copy(),
                                 archive['trajectory_route71_distance'].copy())
