"""Fixed encoder identity and the provenance of its deterministic initialization."""
from collections.abc import Mapping
from maimai_baseline.audio_cross import from_args as cross_from_args, validate_checkpoint as validate_cross
from maimai_baseline.contract import (
    EncoderCapacityConfig, DEFAULT_ENCODER_CAPACITY_PRESET,
)

PAIRING_SCHEME = "independent-global-audio-initialization-v1"
INDEPENDENT_SEED_XOR = 0x454E4344


def from_args(args):
    values = args if isinstance(args, Mapping) else vars(args)
    return EncoderCapacityConfig(values.get("encoder_capacity_preset", DEFAULT_ENCODER_CAPACITY_PRESET), cross_from_args(values))


def is_shared_parameter(name):
    from maimai_baseline.style_contract import STYLE_PREFIXES
    return not name.startswith(STYLE_PREFIXES)


def validate_checkpoint_capacity(payload):
    config = payload.get("model_config")
    if not isinstance(config, Mapping) or "encoder_capacity_preset" not in config:
        raise ValueError("checkpoint lacks explicit encoder capacity model configuration")
    EncoderCapacityConfig(config["encoder_capacity_preset"])
    if "audio_cross_attention" not in config:
        raise ValueError("checkpoint lacks explicit audio cross attention topology")
    capacity = EncoderCapacityConfig(config["encoder_capacity_preset"], config["audio_cross_attention"])
    identity = capacity.identity()
    for value in (payload, payload.get("baseline_manifest"), payload.get("baseline_contract")):
        if not isinstance(value, Mapping) or value.get("encoder_capacity") != identity:
            raise ValueError("checkpoint encoder capacity identity mismatch")
    if payload.get("checkpoint_kind") == "training":
        saved = payload.get("training_config", {})
        if (not isinstance(saved, Mapping)
                or saved.get("encoder_capacity_preset") != capacity.preset
                or saved.get("encoder_capacity") != identity):
            raise ValueError("checkpoint encoder capacity training configuration mismatch")
    metadata = payload.get("initialization")
    initialization = metadata.get("encoder_capacity_initialization") if isinstance(metadata, Mapping) else None
    seed = payload.get("random_seed")
    if (not isinstance(seed, int) or not isinstance(initialization, Mapping)
            or initialization.get("scheme") != PAIRING_SCHEME
            or initialization.get("preset") != capacity.preset
            or initialization.get("seed") != seed
):
        raise ValueError("checkpoint encoder capacity initialization mismatch")
    for key in ("shared_state_sha256", "initial_model_state_sha256"):
        value = initialization.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("checkpoint encoder capacity initialization hash missing/invalid")
    if not isinstance(initialization.get("shared_parameters"), dict) or not initialization["shared_parameters"]:
        raise ValueError("checkpoint encoder capacity shared parameter inventory missing")
    if payload.get("checkpoint_kind") == "training":
        saved = payload["training_config"].get("model_initialization", {})
        if not isinstance(saved, Mapping) or saved.get("encoder_capacity_initialization") != initialization:
            raise ValueError("checkpoint encoder capacity initialization differs from training configuration")
    validate_cross(payload)
    return capacity
