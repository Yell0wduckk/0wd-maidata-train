"""Read historical frozen-A provenance without producer recipe restrictions."""
from pathlib import Path

SCHEMA = "maimai.dynamic_query4x64_lift64_audio256_native_ar_decoder6_same_source_a90.v1"
A_CONFIG = Path(__file__).resolve().parents[2] / "configs/pilot261-mmoe-decoders-a90.json"
FULL_A_CONFIG = A_CONFIG.with_name("full-mmoe-decoders-a90.json")


def a_config_for(recipe):
    """Legacy launcher helper; new stages use their explicit recipe path."""
    return FULL_A_CONFIG if recipe.get('selection') == 'full' else A_CONFIG


def import_from_payload(path, payload):
    """Capture existing producer information without rewriting its identity."""
    return dict(schema=SCHEMA, runtime_version=payload.get('runtime_version'),
        checkpoint_sha256=payload.get('checkpoint_sha256'),
        input_plan_sha256=payload.get('baseline_manifest', {}).get('data_strategy', {}).get('plan_cache', {}).get('sha256'),
        producer_source=payload.get('source_identity'), producer_manifest=payload.get('baseline_manifest', {}),
        frozen_state_sha256=payload.get('frozen_state_sha256'),
        cursor={k: payload.get(k) for k in ('epoch', 'sampler_epoch', 'step')})
