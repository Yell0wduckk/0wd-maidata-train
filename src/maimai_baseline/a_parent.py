"""Legacy parent hooks; actual weights are checked by the checkpoint loader."""
from maimai_baseline.runtime_versions import require_compatible


def validate_parent(path, payload, *, frozen_import=None):
    """Accept compatible A metadata without pinning an epoch, recipe or source."""
    from maimai_baseline.contract import ARCHITECTURE, CHECKPOINT_SCHEMA
    require_compatible(payload, "training")
    if payload.get("architecture") != ARCHITECTURE or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported frozen A architecture/schema")
    return payload.get("checkpoint_sha256")


def validate_inputs(parent, inputs):
    from maimai_baseline.checkpoint import validate_resume_manifest
    validate_resume_manifest(parent, inputs.manifest)
