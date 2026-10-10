"""Dynamic six-query frequency pooling checkpoint identity."""
from typing import Any, Mapping

from maimai_baseline.contract import FREQUENCY_POOLING_ID, frequency_pooling_identity


def validate_checkpoint_pooling(payload: Mapping[str, Any]) -> None:
    config = payload.get("model_config")
    if not isinstance(config, Mapping) or config.get("frequency_pooling") != FREQUENCY_POOLING_ID:
        raise ValueError("checkpoint lacks supported explicit frequency pooling model configuration")
    identity = frequency_pooling_identity()
    for value in (payload, payload.get("baseline_manifest"), payload.get("baseline_contract")):
        if not isinstance(value, Mapping) or value.get("frequency_pooling") != identity:
            raise ValueError("checkpoint frequency pooling identity mismatch")
    if payload.get("checkpoint_kind") == "training":
        saved = payload.get("training_config")
        if not isinstance(saved, Mapping) or saved.get("frequency_pooling") != identity:
            raise ValueError("checkpoint frequency pooling training configuration mismatch")
