"""Fixed public MajRadar supervision; release ranges live in runtime-versions.toml."""
from copy import deepcopy

RADAR_ID = "majradar16-scores-gmm4-v1"
SCHEMA = "maimai.majradar16_scores.v1"
DIMENSIONS = ("note", "peak", "sweep", "slide_tricky", "slide_sequence", "jack", "fitted_constant")
SCALES = (250., 250., 250., 250., 250., 250., 18.)
BEATS_PER_TOKEN = 16
DIMENSION = 7
LOSS_COEFFICIENT = 1.
CONTRACT = {
    "schema": SCHEMA, "name": RADAR_ID, "dimensions": list(DIMENSIONS),
    "scales": list(SCALES), "beats_per_token": BEATS_PER_TOKEN,
    "ownership": "half-open-start-or-slide-declaration;complete-owned-tails",
    "duration": "max-target-window-end-and-owned-event-end;window-relative",
    "empty": "native-positive-duration-empty-analysis;not-seven-zero-padding",
    "encoder": "chart-encoder-frozen-and-skipped",
    "decoder": "gt-in-A;prior-in-inference;seven-semantic-axis-tokens-no-time-rope",
    "prior": {"type": "native-radar-ar-gmm4-moe", "dimension": DIMENSION,
              "layers": 8, "components": 4, "input": "shifted-normalized-GT",
              "positions": "absolute-sixteen-beat-centers", "logvar_bounds": [-6., 2.]},
    "loss": "joint-gmm4-nll-valid-selected-coordinate-mean",
    "coefficient": LOSS_COEFFICIENT,
    "sampling": "categorical-then-Gaussian;clip-normalized-feedback-and-condition-to-unit-range",
}


def is_radar(config):
    return getattr(config, "chart_style", None) == RADAR_ID


def identity():
    return deepcopy(CONTRACT)


def normalize(values):
    return values / values.new_tensor(SCALES)


def denormalize(values):
    return values * values.new_tensor(SCALES)
