"""Trainable song preparation and independently encoded local windows."""

import math

import torch
from torch import nn

from .blocks import SharedConformerBlock, StridedTransition
from .frontend import AudioFeatureFusion, BeatPatchTokenizer, SpectralTokenizer
from .geometry import gather_slots
from .pyramid import LocalPyramid
from .types import EncoderOutput, ScaleMemory, SongInputs, SongMemory, WindowInputs


class AudioEncoder(nn.Module):
    def __init__(
        self, *, mel_bins, samples_per_beat, d_model, heads, ffn_dim,
        stage_depths, stage_attention_radii, temporal_kernel_size, stem_channels,
        frequency_dim, frequency_queries, fpn_rates, target_beats,
        audio_before_beats, audio_after_beats, dropout, rope_base, rope_position_scale,
    ):
        super().__init__()
        dimensions = (mel_bins, samples_per_beat, d_model, heads, ffn_dim,
                      temporal_kernel_size, stem_channels, frequency_dim, frequency_queries)
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError("encoder dimensions must be positive integers")
        if d_model % heads or (d_model // heads) % 2:
            raise ValueError("d_model / heads must be an even integer for full-head RoPE")
        if temporal_kernel_size % 2 == 0:
            raise ValueError("temporal_kernel_size must be odd for symmetric padding")
        if not (len(stage_depths) == len(stage_attention_radii) == len(fpn_rates) == 3):
            raise ValueError("the encoder has three stages, radii and output rates")
        if any(type(depth) is not int or depth <= 0 for depth in stage_depths):
            raise ValueError("stage depths must be positive integers")
        if any(not math.isfinite(radius) or radius < 0 for radius in stage_attention_radii):
            raise ValueError("attention radii must be finite and nonnegative")
        if (type(target_beats) is not int or target_beats <= 0
                or any(type(n) is not int or n < 0 for n in (audio_before_beats, audio_after_beats))):
            raise ValueError("window beats must be integers with T > 0 and A/Z >= 0")
        context = audio_before_beats + target_beats + audio_after_beats
        if any(not math.isfinite(rate) or rate <= 0 or not float(context * rate).is_integer()
               for rate in fpn_rates):
            raise ValueError("each positive FPN rate must produce an integer grid length")
        if not 0 <= dropout <= 1:
            raise ValueError("dropout must be between zero and one")
        if not math.isfinite(rope_base) or rope_base <= 0 or not math.isfinite(rope_position_scale):
            raise ValueError("RoPE requires a positive finite base and finite position scale")

        self.stage_attention_radii = tuple(stage_attention_radii)
        self.mel_bins = mel_bins
        self.samples_per_beat = samples_per_beat
        self.context_beats = context
        self.audio_before_beats = audio_before_beats
        self.spectral = SpectralTokenizer(
            mel_bins, stem_channels, frequency_dim, frequency_queries, d_model, dropout,
        )
        self.feature_fusion = AudioFeatureFusion(d_model)
        self.song_tokenizer = BeatPatchTokenizer(mel_bins, samples_per_beat, d_model)
        self.stages = nn.ModuleList([
            nn.ModuleList([
                SharedConformerBlock(
                    d_model, heads, ffn_dim, temporal_kernel_size, radius,
                    dropout, rope_base, rope_position_scale,
                ) for _ in range(depth)
            ]) for depth, radius in zip(stage_depths, stage_attention_radii)
        ])
        self.transitions = nn.ModuleList([StridedTransition(d_model, dropout) for _ in range(2)])
        self.pyramid = LocalPyramid(d_model, fpn_rates, context, audio_before_beats)

    def prepare_song(self, inputs: SongInputs) -> SongMemory:
        """Create one G0 graph and one K/V pair per non-final block, per song."""
        mel = inputs.mel_patches
        valid = inputs.sample_valid.any(-1)
        g0 = ScaleMemory(
            self.song_tokenizer(mel, inputs.sample_valid, inputs.aux32_centers),
            inputs.beat_centers.float(), valid,
        )
        blocks = [block for stage in self.stages for block in stage]
        kv = tuple(block.project_song(g0) for block in blocks[:-1]) + (None,)
        return SongMemory(g0, kv)

    def encode_windows(self, memory: SongMemory, inputs: WindowInputs) -> EncoderOutput:
        """Read a prepared song graph; slots and all local states remain window-specific."""
        mel = inputs.mel
        from .layout import prepare_encoder_layout, to_device, with_local_mask
        layout = inputs.layout
        if layout is None:
            layout = to_device(prepare_encoder_layout(memory.g0.beats, memory.g0.valid, inputs,
                radii=self.stage_attention_radii, rates=self.pyramid.fpn_rates,
                before=self.audio_before_beats, context=self.context_beats), mel.device)
        slots = gather_slots(
            memory.g0, inputs.target_start_beats, inputs.window_to_song,
            self.audio_before_beats, self.context_beats, layout=layout["slots"],
        )
        local = ScaleMemory(
            self.feature_fusion(self.spectral(mel, inputs.valid), inputs.aux32, inputs.valid),
            inputs.beats.float(), inputs.valid,
        )
        song_valid = memory.g0.valid.index_select(0, inputs.window_to_song)
        stages = []
        block_index = 0
        for stage_index, stage in enumerate(self.stages):
            attention_layout = with_local_mask(layout["local"][stage_index], local.beats, self.stage_attention_radii[stage_index])
            for block in stage:
                song_kv = memory.block_kv[block_index]
                if song_kv is not None:
                    song_kv = tuple(value.index_select(0, inputs.window_to_song) for value in song_kv)
                local, slots = block(
                    local, slots, song_kv, song_valid,
                    update_slots=block_index < len(memory.block_kv) - 1, attention_layout=attention_layout,
                )
                block_index += 1
            stages.append(local)
            if stage_index < 2:
                local = self.transitions[stage_index](local)
        return self.pyramid(stages, inputs.target_start_beats, layout=layout)

    def forward(self, song_inputs: SongInputs, window_inputs: WindowInputs) -> EncoderOutput:
        memory = self.prepare_song(song_inputs)
        return self.encode_windows(memory, window_inputs)
