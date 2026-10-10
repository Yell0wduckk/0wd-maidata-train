"""Audio self-attention presets, independent of data and batch recipes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
from maimai_baseline.model.local_attention import QUERY_BLOCKS


# Selected baseline: optimized local computation with the original beat radii.
DEFAULT_PRESET = "local-e012"
PRESETS = (DEFAULT_PRESET,)


@dataclass(frozen=True)
class AudioAttentionConfig:
    preset: str = DEFAULT_PRESET

    def __post_init__(self) -> None:
        if self.preset not in PRESETS:
            raise ValueError(f"unsupported audio attention preset: {self.preset!r}")

    @property
    def radii(self) -> tuple[float | None, ...]:
        return (1.,2.,4.)

    def identity(self) -> dict[str, Any]:
        return {"preset": self.preset, "stage_attention_beat_radii": list(self.radii),
                "implementation": "chunked-local-rope-sdpa-v2", "query_blocks": list(QUERY_BLOCKS),
                "input": "continuous-whole-song;absolute-beat-phase",
                "song_attention": "independent-global-rope-sdpa"}


def from_args(args: Any) -> AudioAttentionConfig:
    value = args if isinstance(args, Mapping) else vars(args)
    return AudioAttentionConfig(value.get("audio_attention_preset", DEFAULT_PRESET))


def validate_checkpoint_attention(payload: Mapping[str, Any]) -> None:
    config = payload.get("model_config", {})
    if not isinstance(config, Mapping) or "audio_attention_preset" not in config:
        raise ValueError("checkpoint lacks explicit audio attention model configuration")
    attention = AudioAttentionConfig(config["audio_attention_preset"])
    identity = attention.identity()
    manifest = payload.get("baseline_manifest")
    if (not isinstance(manifest, Mapping) or payload.get("audio_attention") != identity
            or manifest.get("audio_attention") != identity):
        raise ValueError("checkpoint audio attention identity mismatch")
    if payload.get("checkpoint_kind") == "training":
        saved = payload.get("training_config", {})
        if (not isinstance(saved, Mapping) or saved.get("audio_attention_preset") != attention.preset
                or saved.get("audio_attention") != identity):
            raise ValueError("checkpoint audio attention training configuration mismatch")
