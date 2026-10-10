from maimai_baseline.architecture import DECODER_LAYERS
"""Selected fused audio scales, shared by runtime and checkpoint contracts."""
from collections.abc import Mapping

PRESETS = ("p0-p2-fused",)
DEFAULT = "p0-p2-fused"
EDGE_PARAMETERS = 263168


def scales(preset=DEFAULT):
    if preset not in PRESETS:
        raise ValueError(f"unsupported audio cross attention: {preset!r}")
    return (0,)


def identity(preset=DEFAULT):
    selected = scales(preset)
    return {
        "schema": "maimai.audio_cross_attention.p0_p2_fused.v1", "topology": preset,
        "decoder_encoders": [["P0"] for _ in range(DECODER_LAYERS)],
        "edges": [f"d{d}_{p}" for d in range(DECODER_LAYERS) for p in ("p0",)],
        "attention_modules": DECODER_LAYERS * len(selected),
        "parameter_count": EDGE_PARAMETERS * DECODER_LAYERS * len(selected),
        "attention_policy": "independent-per-edge",
        "fusion": "song-aligned-residual-at-P2",
        "initialization": "fresh-p0-only-with-p2-song-projection-v1",
        "memory_order": "P2/P1/P0", "song_memory": "encoder-internal-only", "coordinate": {"P0":"decoder:absolute-beat;router:window-beat"},
    }


def from_args(args):
    values = args if isinstance(args, Mapping) else vars(args)
    name = values.get("audio_cross_attention", DEFAULT)
    scales(name)
    return name


def validate_checkpoint(payload):
    config = payload.get("model_config", {})
    if not isinstance(config, Mapping) or "audio_cross_attention" not in config:
        raise ValueError("checkpoint lacks explicit audio cross attention topology")
    expected = identity(config["audio_cross_attention"])
    records = [payload, payload.get("baseline_manifest"), payload.get("baseline_contract"),
               payload.get("initialization")]
    if payload.get("checkpoint_kind") == "training":
        saved = payload.get("training_config", {})
        if not isinstance(saved, Mapping) or saved.get("audio_cross_attention") != config["audio_cross_attention"]:
            raise ValueError("checkpoint audio cross attention training topology mismatch")
        records.extend((saved, saved.get("model_initialization")))
    if any(not isinstance(r, Mapping) or r.get("audio_cross_attention_identity") != expected
           for r in records):
        raise ValueError("checkpoint audio cross attention identity mismatch")
    if payload.get("baseline_contract", {}).get("cross_attention") != expected:
        raise ValueError("checkpoint audio cross attention contract mismatch")
