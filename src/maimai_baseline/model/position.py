"""Identity guards for FP32 window positions and distance-free attention."""
from typing import Any, Mapping

from maimai_baseline.contract import (
    POSITION_CONDITIONING_ID, position_conditioning_identity,
    HISTORY_REPRESENTATION_ID, history_representation_identity,
    AUDIO_ATTENTION_BIAS_ID, audio_attention_bias_identity,
)


def validate_checkpoint_position(payload: Mapping[str, Any]) -> None:
    config = payload.get("model_config")
    coordinate = config.get("decoder_rope_coordinate") if isinstance(config, Mapping) else None
    if coordinate not in ("window-beat",):
        raise ValueError("checkpoint lacks explicit decoder RoPE coordinate")
    records = [payload, payload.get("baseline_manifest"), payload.get("baseline_contract"), payload.get("initialization")]
    if payload.get("checkpoint_kind") == "training":
        saved = payload.get("training_config")
        records.extend((saved,
            saved.get("model_initialization") if isinstance(saved, Mapping) else None,
            saved.get("baseline_manifest") if isinstance(saved, Mapping) else None))
    # Fresh initialization metadata also carries a copy of the model contract.
    records.extend(record["baseline_contract"] for record in tuple(records)
                   if isinstance(record, Mapping) and "baseline_contract" in record and record is not payload)
    if any(not isinstance(record, Mapping) or record.get("decoder_rope_coordinate") != coordinate for record in records):
        raise ValueError("checkpoint decoder RoPE coordinate identity mismatch")
    for field, implementation, identity in (
        ("audio_attention_bias", AUDIO_ATTENTION_BIAS_ID, audio_attention_bias_identity()),
        ("position_conditioning", POSITION_CONDITIONING_ID, position_conditioning_identity(coordinate)),
        ("history_representation", HISTORY_REPRESENTATION_ID, history_representation_identity()),
    ):
        label = field.replace("_", " ")
        if not isinstance(config, Mapping) or config.get(field) != implementation:
            raise ValueError(f"checkpoint lacks supported explicit {label} model configuration")
        if any(not isinstance(record, Mapping) or record.get(field) != identity for record in records):
            raise ValueError(f"checkpoint {label} identity mismatch")
