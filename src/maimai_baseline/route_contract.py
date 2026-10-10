"""Fixed slide geometry and the prototype query route head."""
from copy import deepcopy

GEOMETRY_ID = "rendered-xy-direction-turn-length-pca70-v1"
GEOMETRY_SCHEMA = "maimai.route_geometry.v1"
GEOMETRY_FILE = "route_geometry.npz"
GEOMETRY_DIM = 70
GEOMETRY_POINTS = 128
GEOMETRY_SCALE = 410.0
GEOMETRY_WEIGHTS = dict(xy=0.5, direction=0.2, turn=0.2, length=0.1)
ROUTE_HEADS = ("prototype-distance",)
ROUTE_TEMPERATURE = 1.0
GEOMETRY_SEED_XOR = 0x47454F4D
GEOMETRY_CONTRACT = {
    "name": GEOMETRY_ID, "schema": GEOMETRY_SCHEMA, "dimension": GEOMETRY_DIM,
    "points": GEOMETRY_POINTS, "coordinate_scale": GEOMETRY_SCALE,
    "features": ["xy", "unit-edge-tangent", "adjacent-tangent-sin-cos", "native-arclength"],
    "distance_weights": GEOMETRY_WEIGHTS,
    "length_scale": "maximum-native-lane-length-in-catalog",
    "multi_lane_distance": "hungarian;unit-dummy;mean;clip-0-1",
    "prototype": "weighted-flattened-three-lanes-and-mask;centered-PCA-all-components;pad70;mean-squared-norm1",
    "ordinary_lane_slot": 1, "wifi_lane_order": "native-reference",
    "input": "frozen-codebook;independent-biasless-70-to-256-and-70-to-128-projections;normal-0-1",
    "judge_distance": "original-source-distance;no-MDS",
}


def geometry_identity():
    return deepcopy(GEOMETRY_CONTRACT)


def route_head_identity(name="prototype-distance"):
    if name not in ROUTE_HEADS:
        raise ValueError(f"unsupported route head: {name!r}")
    return {"name": name, "query_dimension": GEOMETRY_DIM,
            "temperature": ROUTE_TEMPERATURE,
            "score": "negative-squared-euclidean-fp32",
            "objective": "CE+0.5*judge-expectation+0.5*shape-expectation"}
