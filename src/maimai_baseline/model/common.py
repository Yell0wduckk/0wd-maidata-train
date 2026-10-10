"""Model configuration, shared audio context, and weight summaries."""

from __future__ import annotations
from dataclasses import asdict, dataclass, replace
from maimai_baseline.model.attention import AudioAttentionConfig, DEFAULT_PRESET
from maimai_baseline.contract import EncoderCapacityConfig, DEFAULT_ENCODER_CAPACITY_PRESET
from maimai_baseline.contract import FREQUENCY_POOLING_ID, POSITION_CONDITIONING_ID, HISTORY_REPRESENTATION_ID, AUDIO_ATTENTION_BIAS_ID, AUDIO_FUSION_ID
from maimai_baseline.contract import BASE_MODEL_ARCHITECTURE, DECODER_DIM, DECODER_FFN_DIM, DECODER_HEADS, DECODER_LAYERS
from maimai_dataset.encoding import CHART_TICKS_PER_BEAT
from maimai_baseline.style_contract import STYLE_ID, is_frozen
from maimai_baseline.radar_contract import RADAR_ID
from typing import Any, Iterator, Mapping
import hashlib
import json
import torch


from maimai_dataset.window import GRID_OFFSETS as _GRID_OFFSETS, AUDIO_CONTEXT_BEATS, AUDIO_BEATS_BEFORE


@dataclass(frozen=True)
class BaselineModelConfig:
    architecture: str = BASE_MODEL_ARCHITECTURE
    mel_bins: int = 128
    d_model: int = 256
    heads: int = 8
    encoder_layers: int = 3
    decoder_layers: int = DECODER_LAYERS
    feedforward: int = 1024
    dropout: float = 0.1
    chart_style: str = RADAR_ID
    latent_enabled: bool = False
    route_head: str = "prototype-distance"
    audio_attention_preset: str = DEFAULT_PRESET
    encoder_capacity_preset: str = DEFAULT_ENCODER_CAPACITY_PRESET
    frequency_pooling: str = FREQUENCY_POOLING_ID
    position_conditioning: str = POSITION_CONDITIONING_ID
    audio_attention_bias: str = AUDIO_ATTENTION_BIAS_ID
    audio_fusion: str = AUDIO_FUSION_ID
    audio_cross_attention: str = "p0-p2-fused"
    history_representation: str = HISTORY_REPRESENTATION_ID
    decoder_rope_coordinate: str = "window-beat"
    time_head: str = "logistic"
    time_components: int = 5
    time_parameterization: str = "quarter_beat"
    decoder_d_model: int | None = DECODER_DIM
    decoder_heads: int | None = DECODER_HEADS
    decoder_feedforward: int | None = DECODER_FFN_DIM
    _initialization_reference: bool = False

    def __post_init__(self) -> None:
        if type(self.latent_enabled) is not bool:
            raise ValueError("latent_enabled must be boolean")
        if is_frozen(self):
            object.__setattr__(self, 'latent_enabled', True)
        if self.chart_style == RADAR_ID and self.latent_enabled:
            raise ValueError('public radar conditioning keeps the chart posterior disabled')
        from maimai_baseline.style_contract import style_identity
        style_identity(self.chart_style)
        from maimai_baseline.route_contract import route_head_identity
        route_head_identity(self.route_head)
        if self.audio_fusion != AUDIO_FUSION_ID:
            raise ValueError("unsupported audio fusion")
        if self.decoder_rope_coordinate not in ("window-beat",):
            raise ValueError("unsupported decoder RoPE coordinate")
        if self.position_conditioning != POSITION_CONDITIONING_ID:
            raise ValueError("unsupported position conditioning")
        if self.history_representation != HISTORY_REPRESENTATION_ID:
            raise ValueError("unsupported history representation")
        if self.audio_attention_bias != AUDIO_ATTENTION_BIAS_ID:
            raise ValueError("unsupported audio attention bias")
        AudioAttentionConfig(self.audio_attention_preset)
        # The private initializer overrides this property for its transient layout.
        _ = self.encoder_capacity
        if self.frequency_pooling != FREQUENCY_POOLING_ID or self.mel_bins != 128 or self.d_model != 256:
            raise ValueError("query frequency pooling fixes dynamic-query4x64-lift64-audio256-v1, 128 Mel bins and width 256")
        if self.heads != 8:
            raise ValueError("encoder RoPE fixes eight heads with 32 dimensions each")
        if self.encoder_layers != 3:
            raise ValueError("encoder exposes exactly three resolution stages")
        if self.architecture != BASE_MODEL_ARCHITECTURE:
            raise ValueError("baseline model configuration has an unsupported architecture")
        if self.time_head != "logistic" or self.time_components != 5:
            raise ValueError("baseline model fixes the Logistic K5 time head")
        if self.time_parameterization != "quarter_beat":
            raise ValueError("baseline model fixes quarter-beat time parameterization")
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")
        decoder_overrides = (
            self.decoder_d_model,
            self.decoder_heads,
            self.decoder_feedforward,
        )
        if not self._initialization_reference:
            expected = (DECODER_DIM, DECODER_HEADS, DECODER_FFN_DIM)
            if decoder_overrides != expected:
                raise ValueError(
                    "wide Window Conformer fixes decoder dimensions to "
                    f"{expected}, got {decoder_overrides}"
                )
            if self.decoder_layers != DECODER_LAYERS:
                raise ValueError("wide Window Conformer requires exactly eight decoder layers")
        elif any(value is not None for value in decoder_overrides):
            raise ValueError(
                "the private encoder initialization reference uses base decoder dimensions"
            )
        if self.resolved_decoder_d_model % self.resolved_decoder_heads:
            raise ValueError("decoder d_model must be divisible by decoder heads")
        if self.decoder_layers != DECODER_LAYERS:
            raise ValueError("baseline model fixes eight decoder layers")

    @property
    def audio_attention(self) -> AudioAttentionConfig:
        return AudioAttentionConfig(self.audio_attention_preset)

    @property
    def encoder_capacity(self) -> EncoderCapacityConfig:
        return EncoderCapacityConfig(self.encoder_capacity_preset, self.audio_cross_attention)

    def parameter_count(self, component="baseline"):
        return self.encoder_capacity.parameter_count(component)

    @property
    def time_unit_scale(self) -> float:
        return float(CHART_TICKS_PER_BEAT if self.time_parameterization == "quarter_beat" else 1)

    @property
    def resolved_decoder_d_model(self) -> int:
        return int(self.d_model if self.decoder_d_model is None else self.decoder_d_model)

    @property
    def resolved_decoder_heads(self) -> int:
        return int(self.heads if self.decoder_heads is None else self.decoder_heads)

    @property
    def resolved_decoder_feedforward(self) -> int:
        return int(
            self.feedforward
            if self.decoder_feedforward is None
            else self.decoder_feedforward
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in asdict(self).items()
            if value is not None and not key.startswith("_")
        }


@dataclass(frozen=True)
class AudioContext:
    """Cached multi-resolution audio state consumed by the baseline decoder."""

    memories: tuple[torch.Tensor, ...]
    padding_masks: tuple[torch.Tensor, ...]
    beats: tuple[torch.Tensor, ...]
    fine_memories: tuple[torch.Tensor, ...] = ()
    fine_padding_masks: tuple[torch.Tensor, ...] = ()
    fine_beats: tuple[torch.Tensor, ...] = ()
    candidate_keys: torch.Tensor | None = None
    candidate_padding_mask: torch.Tensor | None = None
    candidate_beats: torch.Tensor | None = None
    candidate_lookup_left: torch.Tensor | None = None
    candidate_lookup_right: torch.Tensor | None = None
    candidate_lookup_weight: torch.Tensor | None = None
    candidate_lookup_padding: torch.Tensor | None = None
    candidate_lookup_start_tick: torch.Tensor | None = None
    diagnostics: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.memories or not (
            len(self.memories) == len(self.padding_masks) == len(self.beats)
        ):
            raise ValueError("audio context memory/mask/beat tuples must be non-empty and aligned")

    def __iter__(self) -> Iterator[torch.Tensor]:
        yield self.memories[0]
        yield self.padding_masks[0]
        yield self.beats[0]

    def map_audio_windows(self, window_to_audio: torch.Tensor) -> "AudioContext":
        index = window_to_audio.to(
            device=self.memories[0].device, dtype=torch.long
        )
        if index.ndim != 1:
            raise ValueError("window_to_audio must be a one-dimensional tensor")

        def selected(value: torch.Tensor | None) -> torch.Tensor | None:
            return None if value is None else value.index_select(0, index)

        return replace(
            self,
            memories=tuple(value.index_select(0, index) for value in self.memories),
            padding_masks=tuple(
                value.index_select(0, index) for value in self.padding_masks
            ),
            beats=tuple(value.index_select(0, index) for value in self.beats),
            fine_memories=tuple(
                value.index_select(0, index) for value in self.fine_memories
            ),
            fine_padding_masks=tuple(
                value.index_select(0, index) for value in self.fine_padding_masks
            ),
            fine_beats=tuple(
                value.index_select(0, index) for value in self.fine_beats
            ),
            candidate_keys=selected(self.candidate_keys),
            candidate_padding_mask=selected(self.candidate_padding_mask),
            candidate_beats=selected(self.candidate_beats),
            candidate_lookup_left=selected(self.candidate_lookup_left),
            candidate_lookup_right=selected(self.candidate_lookup_right),
            candidate_lookup_weight=selected(self.candidate_lookup_weight),
            candidate_lookup_padding=selected(self.candidate_lookup_padding),
            candidate_lookup_start_tick=selected(
                self.candidate_lookup_start_tick
            ),
        )

    def for_decoder_layers(self, count: int) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]:
        if len(self.memories) == 1:
            return tuple(
                (self.memories[0], self.padding_masks[0], self.beats[0])
                for _ in range(count)
            )
        if len(self.memories) != count:
            raise ValueError(
                f"audio context provides {len(self.memories)} memories for {count} decoder layers"
            )
        return tuple(zip(self.memories, self.padding_masks, self.beats, strict=True))


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(
            json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii")
        )
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()
