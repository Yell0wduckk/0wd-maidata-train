from __future__ import annotations

from maimai_dataset.modifiers import CONTRACT as MODIFIER_CONTRACT
from maimai_baseline.time.audio_end import CONTRACT as AUDIO_END_CONTRACT
from maimai_baseline.song_audio_contract import CONTRACT as SONG_AUDIO_CONTRACT
from typing import Any, Final, Mapping
from dataclasses import dataclass
import copy
from maimai_baseline.audio_cross import DEFAULT as DEFAULT_AUDIO_CROSS, scales as audio_cross_scales, identity as audio_cross_identity, EDGE_PARAMETERS
import math
from maimai_baseline.architecture import DECODER_LAYERS, INITIAL_DECODER_LAYERS, DENSE_DECODER_LAYER_PARAMETERS

from maimai_baseline.physical.contract import CONTRACT as PHYSICAL_CONTRACT
from maimai_baseline.physical.contract import LOSS_SCHEME as PHYSICAL_LOSS_ID

ARCHITECTURE: Final = "maimai-row-fields-interval-physical-dynamic-query4x64-lift64-audio256-native-ar-decoder8-v1"
BASE_MODEL_ARCHITECTURE: Final = "v4-window-conformer-decoder-wide-v1"
CHECKPOINT_SCHEMA: Final = "maimai.baseline_checkpoint.dynamic_query4x64_lift64_audio256_native_ar_decoder8.v1"
TRAINING_RUN_SCHEMA: Final = "maimai.baseline_training_run.dynamic_query4x64_lift64_audio256_native_ar_decoder8.v1"
SOURCE_IDENTITY_SCHEMA: Final = "maimai.baseline_source_identity.v1"
BASELINE_CONTRACT_SCHEMA: Final = "maimai.baseline_contract.dynamic_query4x64_lift64_audio256_native_ar_decoder8.v1"
IMPLEMENTATION_ABI: Final = 15

POSITION_CONDITIONING_ID: Final = "flat-history-cross-rope-fp32-v2"
DECODER_ROPE_COORDINATES: Final = ("window-beat",)
POSITION_CONDITIONING: Final = {
    "schema": "maimai.position_conditioning.v1", "implementation": POSITION_CONDITIONING_ID,
    "origin": "target-window-start-tick", "ticks_per_beat": 96,
    "encoder_rope": "absolute-actual-center-beats;all-encoder-stages",
    "query_time_source": "input-prefix-only;no-next-target-time",
    "window_condition": "[0,0,(end_tick-start_tick)/(96*256)]",
    "additive_position_embeddings": "none", "cross_attention_rope": {
        "query": "(known-row-anchor-tick-start_tick)/96",
        "key": "AudioContext.beats-start_tick/96;matching-resampled-memory",
        "memory_rates_per_beat": [8, 4, 2], "fused_song_rate": 1, "song_alignment": "absolute-beat-linear-to-M2",
        "edges": "selected-independent-projections",
        "cache": "rotated-K-and-raw-V;bound-to-audio-window-and-origin",
        "distance_penalty": "none",
    },
    "alignment_coordinates": "unchanged-absolute", "coordinate_dtype": "float32",
    "rotary_compute_dtype": "float32-autocast-disabled", "rotary_output_dtype": "activation-dtype",
    "theta": 10000, "rotary_dim": 32, "pairing": "adjacent-even-odd",
    "frequency": "10000**(-2*i/32);i=0..15", "scale": 1,
    "rotated": "projected-Q-and-K;not-V", "visibility": "full-causal-raw-prefix",
}


def position_conditioning_identity(decoder_rope_coordinate="window-beat") -> dict[str, Any]:
    if decoder_rope_coordinate not in DECODER_ROPE_COORDINATES:
        raise ValueError("unsupported decoder RoPE coordinate")
    return {**copy.deepcopy(POSITION_CONDITIONING), "decoder_rope_coordinate": decoder_rope_coordinate,
            "decoder_rope": "(known-row-anchor-tick-start_tick)/96"}


AUDIO_ATTENTION_BIAS_ID: Final = "none-v1"
HISTORY_REPRESENTATION_ID: Final = "raw-prefix-v1"
AUDIO_ATTENTION_BIAS: Final = {
    "schema": "maimai.audio_attention_bias.v1",
    "implementation": AUDIO_ATTENTION_BIAS_ID,
    "distance_penalty": "none",
    "mask": "boolean-SDPA;zero-on-valid;negative-infinity-on-padding;safe-empty-row",
    "visibility": "decoder:all-valid-continuous-song-P0;router:pre4-target16-post2",
    "edges": f"{DECODER_LAYERS}-fused-P0-projections",
}
HISTORY_REPRESENTATION: Final = {
    "schema": "maimai.history_representation.v1", "implementation": HISTORY_REPRESENTATION_ID,
    "memory": "raw-input-rows", "history_cross_attention": "absent",
    "visibility": "decoder:complete-causal-song-prefix;router:original-window-raw-prefix",
    "prefix_supervision": "masked", "cross_window": "decoder:persistent-song-KV;router:rebuild-original-raw-prefix-and-prefill",
}


def audio_attention_bias_identity() -> dict[str, Any]:
    return copy.deepcopy(AUDIO_ATTENTION_BIAS)


def history_representation_identity() -> dict[str, Any]:
    return copy.deepcopy(HISTORY_REPRESENTATION)


AUDIO_FUSION_ID: Final = "song-p2-topdown-linear-audio256-v1"
AUDIO_FUSION_ADDED_PARAMETERS: Final = 3 * (256 * 256 + 256)
AUDIO_FUSION: Final = {
    "schema": "maimai.audio_fusion.v1", "implementation": AUDIO_FUSION_ID,
    "location": "after-conformer-fixed-grid-resampling;before-cross-KV-projection",
    "input_order": "M0,M1,M2", "rates_per_beat": [8, 4, 2],
    "formula": "P2=M2+Linear_song(Align_song_absolute_beats);Pi=Mi+Linear_i(Upsample(Pi+1));i=1,0",
    "projection": {"count": 3, "in_features": 256, "out_features": 256, "bias": True},
    "interpolation": "fixed-beat-centers;linear;clamped-endpoints;valid-weight-renormalization",
    "interpolation_compute": "float32-autocast-disabled;output-source-dtype",
    "fusion_output_dtype": "target-memory-dtype",
    "padding": "preserve-target-mask;zero-missing-source-after-projection;zero-target-padding",
    "geometry": "independent-song-to-M2-sidecar;nonpersistent-fixed-double-grid-buffers",
    "output_order": "P2,P1,P0;unchanged-lengths-and-beat-coordinates",
    "fine_memories_and_candidate_keys": "unfused-original-fine-features",
    "cross_attention": "selected-independent-cross-rope-edges;unchanged-target-grid-positions",
    "context": "E0-E2-local;song-context-through-P2",
    "extra_processing": "none;no-convolution-ffn-norm-activation-gate-dropout",
    "initialization": "torch-linear-default;fork-and-restore-construction-RNG;capacity-shared",
    "added_parameters": AUDIO_FUSION_ADDED_PARAMETERS,
}


def audio_fusion_identity() -> dict[str, Any]:
    return copy.deepcopy(AUDIO_FUSION)


FREQUENCY_POOLING_ID: Final = "dynamic-query4x64-lift64-audio256-v1"
FREQUENCY_POOLING: Final = {
    "schema": "maimai.frequency_pooling.v3", "implementation": FREQUENCY_POOLING_ID,
    "input_mel_bins": 128, "frequency_positions": 64, "queries": 4, "channels": 64, "convolution_channels": 32, "feature_lift": "shared-per-frequency-Linear32-64-bias-GELU;fork-restore-construction-RNG",
    "score": "q-dot-x/sqrt64-minus-half-squared-normalized-frequency-distance",
    "query_initialization": "zero", "centers": "sigmoid-raw;initial-(i+0.5)/4",
    "widths": "softplus-raw+1e-4;initial-1/8", "frequency": "(f+0.5)/64",
    "normalization": "softmax-all-frequencies", "concatenation": "fixed-query-index",
    "dynamic_queries": True, "kv_projection": False,
    "dynamic_query": {
        "summary": "per-frame-frequency-mean-after-nonlinear-feature-lift",
        "network": "LayerNorm64-affine-eps1e-5;Linear64-64;GELU;Linear64-256;both-bias",
        "composition": "base-query-plus-frame-residual;reshape-fixed-4x64",
        "initialization": "first-linear-default;last-linear-zero;fork-restore-construction-RNG",
        "added_parameters": 20928, "macs_per_frame": 20480,
        "temporal_aggregation": False, "dynamic_centers_widths": False,
    },
    "projection": {"in_features": 256, "out_features": 256, "bias": True},
    "parameters": 86984,
}
FREQUENCY_POOLING_ADDED_PARAMETERS: Final = 86984 + 2112 - (33 + 32 * 256 + 256)


def frequency_pooling_identity() -> dict[str, Any]:
    return copy.deepcopy(FREQUENCY_POOLING)

DEFAULT_ENCODER_CAPACITY_PRESET: Final = "ffn1024-depth112-song2-p2-audio256"
ENCODER_CAPACITY_PRESETS: Final = (DEFAULT_ENCODER_CAPACITY_PRESET,)


@dataclass(frozen=True)
class EncoderCapacityConfig:
    """The single maintained encoder; this does not identify offline data."""
    preset: str = DEFAULT_ENCODER_CAPACITY_PRESET
    audio_cross_attention: str = DEFAULT_AUDIO_CROSS

    def __post_init__(self):
        audio_cross_scales(self.audio_cross_attention)
        if self.preset not in ENCODER_CAPACITY_PRESETS:
            raise ValueError(f"unsupported encoder capacity preset: {self.preset!r}")

    @property
    def ffn_dim(self):
        return 1024

    @property
    def stage_depths(self):
        return (1, 1, 2)

    def parameter_count(self, component="baseline"):
        return {"encoder": 6565672, "wide": 11179718,
                "route": 11201264, "baseline": 13640432}[component] + (
                    0 if component == "encoder" else
                    (DECODER_LAYERS - INITIAL_DECODER_LAYERS) * DENSE_DECODER_LAYER_PARAMETERS)

    def identity(self):
        return {"schema": "maimai.encoder_capacity.v1", "preset": self.preset,
                "audio_cross_attention": self.audio_cross_attention,
                "model_dim": 256, "heads": 8, "ffn_dim": self.ffn_dim,
                "stage_depths": list(self.stage_depths), "parameter_count": self.parameter_count(),
                "encoder_parameter_count": self.parameter_count("encoder")}


ROUTE_PARAMETER_COUNT: Final = EncoderCapacityConfig().parameter_count("route")
from maimai_baseline.dual_contract import identity as dual_identity, parameter_delta as dual_parameter_delta
PARAMETER_COUNT: Final = EncoderCapacityConfig().parameter_count() + dual_parameter_delta()
CANDIDATE_PROTECTED_INTERVAL_MS: Final = 25.0
CANDIDATE_FULL_INTERVAL_MS: Final = 40.0
SEED: Final = 20260811
TIME_HEAD: Final = "logistic"
TIME_COMPONENTS: Final = 5
L3_LOSS_ID: Final = "dual-decoder-field-budget-v6"
L3_LABEL_SMOOTHING: Final = 0.05
RESAMPLING_BACKEND: Final = "indexed-v1"
EPOCHS: Final = 75
TOKEN_BUDGET: Final = 4096
VALIDATION_TOKEN_BUDGET: Final = 2048
FIXED_OPTIMIZER_STEPS_PER_EPOCH: Final = 500
EPOCH_PLAN_MODE: Final = "spatial-aug-curriculum"
DEFAULT_SCHEDULER_EPOCHS: Final = EPOCHS
DEFAULT_STOP_AFTER_EPOCHS: Final = 45
LEARNING_RATE: Final = 4e-4 * math.sqrt(4096 / 2048)
WEIGHT_DECAY: Final = 0.01
WARMUP_STEPS: Final = 200
GRADIENT_CLIP: Final = 1.0
CHECKPOINT_EVERY_EPOCHS: Final = 5
GRADIENT_DIAGNOSTIC_INTERVAL: Final = 32

# Cached chart units stay fixed; runtime BPM96 units are defined in time.grid.
DELTA_FIELD: Final = "delta_ms"
SOURCE_DELTA_FIELD: Final = "delta_tick"
TIME_FIELDS: Final = (DELTA_FIELD, "hold_duration_tick", "branch_duration_tick")
TIME_UNIT_SCALES: Final = {
    DELTA_FIELD: 1000.0,
    "hold_duration_tick": 96.0,
    "branch_duration_tick": 96.0,
}
QUANTUM_SECONDS: Final = 0.01

from maimai_baseline.route_contract import geometry_identity, route_head_identity

from maimai_dataset.window import (CHART_TICKS_PER_BEAT, HISTORY_BEATS, HISTORY_TICKS,
    TARGET_BEATS, AUDIO_BEATS_BEFORE, AUDIO_BEATS_AFTER, AUDIO_CONTEXT_BEATS, WINDOW)
DECODER_DIM: Final = 256
DECODER_HEADS: Final = 8
DECODER_FFN_DIM: Final = 1024
MAX_GENERATED_MOMENT_ROWS: Final = 32
DECODER_WIDE_PARAMETER_COUNT: Final = EncoderCapacityConfig().parameter_count("wide")
ROUTE_CLASSES: Final = 71
ROUTE_INPUT_VOCABULARY: Final = 72
ROUTE_DIMENSION: Final = 256
ROUTE_HEAD_SEED_XOR: Final = 0x52543731
REMOVED_FIELDS: Final = ("shape", "end_offset", "via_offset")
CATALOG_IDENTITY: Final = (
    "acae86c44abc032dfb2a4a77921f4ca6f741407d24a27a56a5be503f0b528cee"
)

AUGMENTATION_ID: Final = "random-note-block-permutation-v1"
AUGMENTATION_POLICY: Final = "sha256-independent-random-keys-v1"
AUGMENTATION_SEED: Final = SEED
VIEW_CACHE_SCHEMA: Final = "maimai.route71_multiview_chart_cache.v1"
PLAN_CACHE_SCHEMA: Final = "maimai.route71_spatial_epoch_plan_cache.v1"
PLAN_AUDIT_SCHEMA: Final = "maimai.route71_spatial_epoch_plan_audit.v1"
SPATIAL_AUGMENTATION_ID: Final = "route71-whole-chart-lr-2to1-v1"
VIEW_NAMES: Final = ("identity", "mirror-lr", "mirror-ud", "rotate-180")
VIEW_OPERATIONS: Final = {
    "identity": "identity",
    "mirror-lr": "lr",
    "mirror-ud": "ud",
    "rotate-180": "180",
}
VIEW_WEIGHTS: Final = {
    "identity": 2,
    "mirror-lr": 1,
}
TRAINING_VIEW_NAMES: Final = tuple(VIEW_WEIGHTS)


def cache_view_names(views) -> tuple[str, ...]:
    """Two active views or an ordered four-view preparation source."""
    names = tuple(views)
    if names not in (TRAINING_VIEW_NAMES, VIEW_NAMES):
        raise ValueError("cache views must be ordered identity/lr or identity/lr/ud/180")
    return names


def spatial_sampling_identity() -> dict[str, Any]:
    return {
        "id": SPATIAL_AUGMENTATION_ID,
        "views": list(TRAINING_VIEW_NAMES),
        "view_weights": dict(VIEW_WEIGHTS),
        "unit": "whole-chart-occurrence",
        "allocation": "hamilton-per-difficulty-pool",
        "assignment": "sha256-seed-epoch-pool-slot-v1",
        "repeated_views": "allowed",
        "seed": SEED,
    }
DIFFICULTY_POOLS: Final = {
    2: "basic",
    3: "advanced",
    4: "expert",
    5: "master-remaster",
    6: "master-remaster",
}
CURRICULUM_EXCEPTIONS: Final = ("649", "11195", "11209", "11311", "11362")
CURRICULUM_PARTICIPANTS: Final = 251


def augmentation_manifest(*, seed: int = SEED) -> dict[str, Any]:
    return {
        "id": AUGMENTATION_ID,
        "policy": AUGMENTATION_POLICY,
        "seed": int(seed),
        "training_only": True,
        "unit": "NOTE-plus-following-SLIDE_BRANCH-block",
        "validation_order": "canonical",
    }


from maimai_baseline.time.grid import SAMPLING
from maimai_baseline.time.grid import INTEGRATION
from maimai_baseline.time.moment import LINEAR_CONTRACT as MOMENT_DISTRIBUTION
from maimai_baseline.time.duration import LINEAR_CONTRACT as DURATION_DISTRIBUTION

BASELINE_CONTRACT: Final = {
    "schema": BASELINE_CONTRACT_SCHEMA,
    "implementation_abi": IMPLEMENTATION_ABI,
    "architecture": ARCHITECTURE,
    "base_model_architecture": BASE_MODEL_ARCHITECTURE,
    "checkpoint_schema": CHECKPOINT_SCHEMA,
    "training_run_schema": TRAINING_RUN_SCHEMA,
    "parameter_count": PARAMETER_COUNT,
    "dual_decoder": dual_identity(),
    "modifiers": MODIFIER_CONTRACT,
    "audio_end": AUDIO_END_CONTRACT,
    "song_audio_encoder": SONG_AUDIO_CONTRACT,
    "encoder_capacity": EncoderCapacityConfig().identity(),
    "frequency_pooling": frequency_pooling_identity(),
    "audio_fusion": audio_fusion_identity(),
    "position_conditioning": position_conditioning_identity(),
    "decoder_rope_coordinate": "window-beat",
    "audio_attention_bias": audio_attention_bias_identity(),
    "history_representation": history_representation_identity(),
    "cabinet_conditioning": "none; SD/DX metadata only",
    "seed": SEED,
    "history": {"beats": HISTORY_BEATS, "ticks": HISTORY_TICKS},
    "audio_context_beats": AUDIO_CONTEXT_BEATS,
    "window": dict(WINDOW),
    "route_geometry": geometry_identity(),
    "route_head": route_head_identity(),
    "route": {
        "classes": ROUTE_CLASSES,
        "removed_fields": list(REMOVED_FIELDS),
        "judge_distance_weight": 0.5,
        "trajectory_distance_weight": 0.5,
    },
    "timing": INTEGRATION,
    "moment_distribution": MOMENT_DISTRIBUTION,
    "duration_distribution": DURATION_DISTRIBUTION,
    "audio_sampling": SAMPLING,
    "training": {
        "epochs": EPOCHS,
        "token_budget": TOKEN_BUDGET,
        "loss": "L3-hierarchical+Physical-dual-v6",
        "loss_scheme": PHYSICAL_LOSS_ID,
        "base_loss_scheme": L3_LOSS_ID,
        "label_smoothing": L3_LABEL_SMOOTHING,
        "resampling_backend": RESAMPLING_BACKEND,
        "time_head": TIME_HEAD,
        "time_components": TIME_COMPONENTS,
        "curriculum": True,
        "spatial_augmentation": True,
        "random_note_block": True,
    },
    "cross_attention": audio_cross_identity(),
    "audio_cross_attention_identity": audio_cross_identity(),
    "level_conditioning": {
        "name": "zero-input-zero-column-v1",
        "weight": "condition.0.weight",
        "column": 0,
        "difficulty_embedding_enabled": True,
    },
    "delta_candidate": {
        "policy": "short-ramp-25-40ms-v1",
        "protected_interval_ms": CANDIDATE_PROTECTED_INTERVAL_MS,
        "full_interval_ms": CANDIDATE_FULL_INTERVAL_MS,
        "window_start_exempt": True,
        "preserve_protected_k5_mass": True,
    },
    "unreasonable_detection": True,
    "physical": PHYSICAL_CONTRACT,
}


def baseline_contract(capacity: EncoderCapacityConfig, decoder_rope_coordinate="window-beat", route_head="prototype-distance") -> dict[str, Any]:
    value = copy.deepcopy(BASELINE_CONTRACT)
    value["position_conditioning"] = position_conditioning_identity(decoder_rope_coordinate)
    value["decoder_rope_coordinate"] = decoder_rope_coordinate
    from maimai_baseline.dual_contract import identity, parameter_delta
    value['dual_decoder'] = identity()
    value["parameter_count"] = capacity.parameter_count() + parameter_delta()
    value["route_head"] = route_head_identity(route_head)
    value["encoder_capacity"] = capacity.identity()
    value["cross_attention"] = audio_cross_identity(capacity.audio_cross_attention)
    value["audio_cross_attention_identity"] = audio_cross_identity(capacity.audio_cross_attention)
    return value


def baseline_manifest(
    *,
    profile: Mapping[str, Any],
    chart_cache: Mapping[str, Any],
    interpolation_geometry: Mapping[str, Any],
    route_artifact: Mapping[str, Any],
    view_cache: Mapping[str, Any],
    plan_cache: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    encoder_capacity: EncoderCapacityConfig = EncoderCapacityConfig(),
    decoder_rope_coordinate: str = "window-beat",
    route_head: str = "prototype-distance",
) -> dict[str, Any]:
    from maimai_baseline.runtime_versions import stamp
    return {
        **stamp(),
        **baseline_contract(encoder_capacity, decoder_rope_coordinate, route_head),
        "profile": dict(profile),
        "chart_cache": dict(chart_cache),
        "interpolation_geometry": dict(interpolation_geometry),
        "route_artifact": dict(route_artifact),
        "data_strategy": {
            "mode": EPOCH_PLAN_MODE,
            "views": list(TRAINING_VIEW_NAMES),
            "cache_views": list(VIEW_NAMES),
            "view_weights": dict(VIEW_WEIGHTS),
            "spatial_sampling": spatial_sampling_identity(),
            "augmentation": augmentation_manifest(),
            "view_cache": dict(view_cache),
            "plan_cache": dict(plan_cache),
        },
        "source_identity": dict(source_identity),
    }


__all__ = [
    "EncoderCapacityConfig", "DEFAULT_ENCODER_CAPACITY_PRESET", "ENCODER_CAPACITY_PRESETS",
    "VALIDATION_TOKEN_BUDGET", "baseline_contract",
    "ARCHITECTURE",
    "AUDIO_CONTEXT_BEATS",
    "AUGMENTATION_SEED",
    "BASELINE_CONTRACT",
    "CATALOG_IDENTITY",
    "CHECKPOINT_EVERY_EPOCHS",
    "CHECKPOINT_SCHEMA",
    "CURRICULUM_EXCEPTIONS",
    "CURRICULUM_PARTICIPANTS",
    "DECODER_DIM",
    "DECODER_FFN_DIM",
    "DECODER_HEADS",
    "DECODER_LAYERS",
    "DEFAULT_SCHEDULER_EPOCHS",
    "DEFAULT_STOP_AFTER_EPOCHS",
    "DELTA_FIELD",
    "DIFFICULTY_POOLS",
    "EPOCHS",
    "EPOCH_PLAN_MODE",
    "FIXED_OPTIMIZER_STEPS_PER_EPOCH",
    "GRADIENT_CLIP",
    "GRADIENT_DIAGNOSTIC_INTERVAL",
    "HISTORY_BEATS",
    "HISTORY_TICKS",
    "IMPLEMENTATION_ABI",
    "L3_LABEL_SMOOTHING",
    "L3_LOSS_ID",
    "LEARNING_RATE",
    "PARAMETER_COUNT",
    "PLAN_AUDIT_SCHEMA",
    "PLAN_CACHE_SCHEMA",
    "QUANTUM_SECONDS",
    "REMOVED_FIELDS",
    "RESAMPLING_BACKEND",
    "ROUTE_CLASSES",
    "ROUTE_DIMENSION",
    "ROUTE_HEAD_SEED_XOR",
    "ROUTE_INPUT_VOCABULARY",
    "SEED",
    "SOURCE_DELTA_FIELD",
    "SOURCE_IDENTITY_SCHEMA",
    "SPATIAL_AUGMENTATION_ID",
    "TIME_COMPONENTS",
    "TIME_FIELDS",
    "TIME_HEAD",
    "TIME_UNIT_SCALES",
    "TOKEN_BUDGET",
    "TRAINING_RUN_SCHEMA",
    "VIEW_CACHE_SCHEMA",
    "VIEW_NAMES",
    "VIEW_OPERATIONS",
    "VIEW_WEIGHTS",
    "WARMUP_STEPS",
    "WEIGHT_DECAY",
    "augmentation_manifest",
    "baseline_manifest",
]
