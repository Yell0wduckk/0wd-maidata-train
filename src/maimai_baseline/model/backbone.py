"""Core encoder-decoder backbone."""

from __future__ import annotations
from maimai_baseline.model.common import AudioContext, BaselineModelConfig
from maimai_baseline.model.layers import (
    ContinuousSinusoidal,
    RelativeDecoderLayer,
    _indexed_interpolate,
)
from maimai_dataset.encoding import CHART_TICKS_PER_BEAT, TIME_FIELDS
from maimai_dataset.vocabulary import CompactVocabulary
from maimai_dataset.modifiers import LEGACY_FIELDS, HEADS, normalize_inputs
from maimai_baseline.model.modifiers import ModifierProjection
from torch import nn
from torch.profiler import record_function
from typing import Any, Mapping
import torch
import torch.nn.functional as functional


class _BaselineCoreTransformer(nn.Module):
    def __init__(
        self,
        vocabulary: CompactVocabulary,
        config: BaselineModelConfig | None = None,
    ) -> None:
        super().__init__()
        self.vocabulary = vocabulary
        self.config = config or BaselineModelConfig()
        # This private layout is constructed and replaced solely to preserve
        # the baseline's recorded fixed-seed RNG/weight initialization.
        decoder_dim = self.config.resolved_decoder_d_model
        decoder_heads = self.config.resolved_decoder_heads
        self.embeddings = nn.ModuleDict(
            {
                field: nn.Embedding(
                    len(field_vocabulary.values),
                    decoder_dim,
                    padding_idx=0,
                )
                for field, field_vocabulary in vocabulary.fields.items() if field not in LEGACY_FIELDS
            }
        )
        self.time_embeddings = nn.ModuleDict(
            {field: ContinuousSinusoidal(decoder_dim) for field in TIME_FIELDS}
        )
        self.difficulty_embedding = nn.Embedding(8, decoder_dim)
        self.condition = nn.Sequential(
            nn.Linear(3, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, decoder_dim),
        )
        from maimai_baseline.architecture import INITIAL_DECODER_LAYERS, DECODER_LAYER_SEED_XOR, PRESERVED_DECODER_LAYERS, NEW_DECODER_LAYER_SEED_XOR
        self.decoder_layers = nn.ModuleList(
            RelativeDecoderLayer(self.config) for _ in range(INITIAL_DECODER_LAYERS)
        )
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(torch.initial_seed() ^ DECODER_LAYER_SEED_XOR)
            self.decoder_layers.extend(RelativeDecoderLayer(self.config)
                for _ in range(min(self.config.decoder_layers, PRESERVED_DECODER_LAYERS) - INITIAL_DECODER_LAYERS))
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(torch.initial_seed() ^ NEW_DECODER_LAYER_SEED_XOR)
            self.decoder_layers.extend(RelativeDecoderLayer(self.config)
                for _ in range(self.config.decoder_layers - PRESERVED_DECODER_LAYERS))
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.delta_candidate_fusion = nn.Sequential(
            nn.Linear(self.config.d_model * 2, decoder_dim),
            nn.LayerNorm(decoder_dim),
        )
        self.delta_candidate_key = nn.Linear(decoder_dim, decoder_dim, bias=False)
        self.delta_candidate_query = nn.Linear(decoder_dim, decoder_dim, bias=False)
        self.delta_candidate_gate = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.delta_candidate_enabled = True
        self.field_heads = nn.ModuleDict(
            {
                field: nn.Linear(decoder_dim, len(field_vocabulary.values))
                for field, field_vocabulary in vocabulary.fields.items() if field not in LEGACY_FIELDS
            }
        )
        self.modifier_projection = ModifierProjection(decoder_dim)
        self.field_heads.update({f"modifier_{name}": nn.Linear(decoder_dim, 2) for name in HEADS})
        self.time_heads = nn.ModuleDict(
            {
                field: nn.Linear(decoder_dim, self.config.time_components * 3)
                for field in TIME_FIELDS
            }
        )
        initial_centers = [0.0, 24.0, 48.0, 96.0, 192.0]
        initial_scales = [12.0, 12.0, 16.0, 24.0, 48.0]
        unit_scale = self.config.time_unit_scale
        with torch.no_grad():
            for head in self.time_heads.values():
                head.weight.zero_()
                bias = head.bias.reshape(self.config.time_components, 3)
                bias[:, 0].zero_()
                bias[:, 1].copy_(torch.tensor(initial_centers) / unit_scale)
                bias[:, 2].copy_(
                    torch.log(
                        torch.expm1(
                            (torch.tensor(initial_scales) - 0.25) / unit_scale
                        )
                    )
                )

        from maimai_baseline.model.audio import WindowConformerEncoder
        from maimai_baseline.model.audio import conformer_manifest

        self.audio_encoder = WindowConformerEncoder(self.config)
        self.window_conformer_metadata = conformer_manifest(self.config)
        if not self.config._initialization_reference and self.parameter_count() != self.config.encoder_capacity.parameter_count("wide"):
            raise RuntimeError("wide backbone encoder capacity parameter count drifted")

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _row_embedding(
        self,
        input_ids: Mapping[str, torch.Tensor],
        input_times: Mapping[str, torch.Tensor],
        input_time_active: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        input_ids, input_times, input_time_active = normalize_inputs(input_ids, input_times, input_time_active, self.vocabulary)
        total = self.modifier_projection(input_ids)
        event = input_ids["event"]
        active = (event.eq(self.vocabulary.event_id("NOTE")) | event.eq(self.vocabulary.event_id("SLIDE_BRANCH"))).unsqueeze(-1).to(total.dtype)
        for field, embedding in self.embeddings.items():
            field_ids = input_ids[field]
            value = embedding(field_ids)
            present = field_ids.ne(0).unsqueeze(-1).to(value.dtype)
            total = value if total is None else total + value
            active = present if active is None else active + present
        assert total is not None and active is not None
        for field, embedding in self.time_embeddings.items():
            present = input_time_active[field].unsqueeze(-1)
            value = embedding(
                (input_times[field] / CHART_TICKS_PER_BEAT).to(total.dtype)
            ).masked_fill(~present, 0.0)
            total = total + value
            active = active + present.to(active.dtype)
        return total / active.clamp_min(1).sqrt()

    def grammar_mask(
        self,
        previous_event_ids: torch.Tensor,
        previous_kind_ids: torch.Tensor | None = None,
        previous_delta_tick: torch.Tensor | None = None,
        previous_delta_active: torch.Tensor | None = None,
    ) -> torch.Tensor:
        size = len(self.vocabulary.fields["event"].values)
        result = torch.zeros(
            (*previous_event_ids.shape, size),
            dtype=torch.bool,
            device=previous_event_ids.device,
        )
        for previous in torch.unique(previous_event_ids).tolist():
            for event_id in self.vocabulary.allowed_next_event_ids(int(previous)):
                result[..., event_id] |= previous_event_ids.eq(int(previous))
        if previous_kind_ids is not None:
            if previous_kind_ids.shape != previous_event_ids.shape:
                raise ValueError("previous kind IDs must match previous event IDs")
            note = previous_event_ids.eq(self.vocabulary.event_id("NOTE"))
            slide_kind = self.vocabulary.fields["kind"].encode("slide")
            wifi_kind = self.vocabulary.fields["kind"].encode("wifi")
            may_branch = previous_kind_ids.eq(slide_kind) | previous_kind_ids.eq(
                wifi_kind
            )
            result[..., self.vocabulary.event_id("SLIDE_BRANCH")] &= ~note | may_branch
        return result

    def _condition_embedding(
        self,
        *,
        difficulty_id: torch.Tensor,
        level: torch.Tensor,
        start_tick: torch.Tensor,
        end_tick: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.difficulty_embedding(difficulty_id)
            + self.condition(
                torch.stack(
                    (
                        level / 15.0,
                        torch.zeros_like(start_tick),
                        (end_tick - start_tick) / (CHART_TICKS_PER_BEAT * 256.0),
                    ),
                    dim=-1,
                )
            )
        )

    def prepare_decoder_context(
        self,
        context: AudioContext,
        *,
        difficulty_id: torch.Tensor,
        level: torch.Tensor,
        start_tick: torch.Tensor,
        end_tick: torch.Tensor,
    ) -> AudioContext:
        return context

    def encode_audio(
        self,
        audio: torch.Tensor,
        audio_padding_mask: torch.Tensor,
        frame_beats: torch.Tensor,
        frame_bpms: torch.Tensor,
        *,
        start_tick: torch.Tensor | None = None,
        end_tick: torch.Tensor | None = None,
        interpolation_geometry: Mapping[str, torch.Tensor] | None = None,
        song_audio=None,
    ) -> AudioContext:
        del end_tick
        if start_tick is None:
            raise ValueError("baseline audio encoding requires target start_tick")
        encoded = self.audio_encoder(
            audio,
            audio_padding_mask,
            frame_beats,
            frame_bpms,
            start_tick=start_tick,
            interpolation_geometry=interpolation_geometry, song_audio=song_audio,
        )
        if not self.delta_candidate_enabled:
            return encoded
        if len(encoded.fine_memories) < 2:
            raise ValueError("audio delta candidate requires C0 and C1 memories")
        if interpolation_geometry is None:
            raise ValueError("baseline audio delta candidate requires indexed geometry")
        first = encoded.fine_memories[0]
        first_mask = encoded.fine_padding_masks[0]
        first_beats = encoded.fine_beats[0]
        with record_function("resampling.candidate_c1_to_c0"):
            second = _indexed_interpolate(
                encoded.fine_memories[1],
                interpolation_geometry["a1_to_a0_left"],
                interpolation_geometry["a1_to_a0_right"],
                interpolation_geometry["a1_to_a0_weight"],
                interpolation_geometry["a1_to_a0_padding"],
            )
        keys = self.delta_candidate_key(
            self.delta_candidate_fusion(torch.cat((first, second), dim=-1))
        ).masked_fill(first_mask.unsqueeze(-1), 0.0)
        return AudioContext(
            memories=encoded.memories,
            padding_masks=encoded.padding_masks,
            beats=encoded.beats,
            fine_memories=encoded.fine_memories,
            fine_padding_masks=encoded.fine_padding_masks,
            fine_beats=encoded.fine_beats,
            candidate_keys=keys,
            candidate_padding_mask=first_mask,
            candidate_beats=first_beats,
            candidate_lookup_left=interpolation_geometry.get("candidate_left"),
            candidate_lookup_right=interpolation_geometry.get("candidate_right"),
            candidate_lookup_weight=interpolation_geometry.get("candidate_weight"),
            candidate_lookup_padding=interpolation_geometry.get("candidate_padding"),
            candidate_lookup_start_tick=interpolation_geometry.get("candidate_start_tick"),
            diagnostics=encoded.diagnostics,
        )
