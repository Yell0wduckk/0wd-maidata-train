"""Local E0/E1/E2 audio encoder."""

from __future__ import annotations
from maimai_dataset.window import GRID_OFFSETS as _GRID_OFFSETS
from maimai_baseline.model.common import AudioContext, BaselineModelConfig
from maimai_baseline.model.layers import (
    DepthwiseResidual2d,
    QueryFrequencyAttentionProjection,
    _fixed_beat_grid,
    _indexed_interpolate,
    _masked_pool_scalar,
)
from maimai_baseline.model.local_attention import local_attention, prepare_pyramid_layouts, with_local_mask
from maimai_dataset.window import CHART_TICKS_PER_BEAT
from maimai_baseline.model.rope import self_attention
from maimai_baseline.model.fusion import TopDownAudioFusion
from maimai_baseline.contract import audio_fusion_identity
from torch import nn
from torch.profiler import record_function
import torch
import torch.nn.functional as functional


ARCHITECTURE = "v4-window-conformer-v1"


BASELINE_ARCHITECTURE = "v3-test2-1-opt"


MODEL_DIM = 256


HEADS = 8




CONV_KERNEL = 15


DOWNSAMPLE_KERNEL = 3




STAGE_FRAME_STRIDES = (1, 2, 4)


STAGE_GRID_RATES = (8, 4, 2)


CHECKPOINT_SCHEMA = "maimai.chart_transformer_checkpoint.v4_window_conformer.v1"


TRAINING_RUN_SCHEMA = "maimai.compact_training_run.v4_window_conformer.v1"


MANIFEST_SCHEMA = "maimai.window_conformer_manifest.v1"


PARAMETER_COUNT = 6_565_672




def conformer_manifest(config: BaselineModelConfig) -> dict[str, object]:
    return {
        "schema": MANIFEST_SCHEMA,
        "architecture": ARCHITECTURE,
        "base_architecture": BASELINE_ARCHITECTURE,
        "model_dim": MODEL_DIM,
        "heads": HEADS,
        "ffn_dim": config.encoder_capacity.ffn_dim,
        "conv_kernel": CONV_KERNEL,
        "downsample_kernel": DOWNSAMPLE_KERNEL,
        "stage_depths": list(config.encoder_capacity.stage_depths),
        "encoder_capacity": config.encoder_capacity.identity(),
        "stage_frame_strides": list(STAGE_FRAME_STRIDES),
        "audio_attention": config.audio_attention.identity(),
        "audio_attention_preset": config.audio_attention_preset,
        "stage_attention_beat_radii": list(config.audio_attention.radii),
        "stage_grid_rates": list(STAGE_GRID_RATES),
        "input_scope": "continuous-whole-song;absolute-beat-phase",
        "full_song_attention": False,
        "decoder_memory_order": ["P2", "P1", "P0"],
        "audio_fusion": audio_fusion_identity(),
        "candidate_memory_order": ["C0", "C1"],
    }


def _masked_pool_features(
    value: torch.Tensor, padding: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask-aware k=3/s=2/p=1 pooling matching dataset geometry."""

    valid = (~padding).to(value.dtype).unsqueeze(1)
    numerator = functional.avg_pool1d(
        value.transpose(1, 2) * valid,
        kernel_size=DOWNSAMPLE_KERNEL,
        stride=2,
        padding=DOWNSAMPLE_KERNEL // 2,
        count_include_pad=False,
    )
    count = functional.avg_pool1d(
        valid,
        kernel_size=DOWNSAMPLE_KERNEL,
        stride=2,
        padding=DOWNSAMPLE_KERNEL // 2,
        count_include_pad=False,
    )
    pooled = (numerator / count.clamp_min(torch.finfo(value.dtype).eps)).transpose(1, 2)
    pooled_padding = count.squeeze(1).eq(0)
    return pooled.masked_fill(pooled_padding.unsqueeze(-1), 0.0), pooled_padding


class SpectralTokenizer(nn.Module):
    """The baseline A0 spectral stem without temporal subsampling."""

    def __init__(self, config: BaselineModelConfig) -> None:
        super().__init__()
        self.mel_bins = int(config.mel_bins)
        self.stem = nn.Conv2d(1, 32, kernel_size=3, stride=(1, 2), padding=1)
        self.residual = DepthwiseResidual2d(32, config.dropout)
        self.frequency_pool = QueryFrequencyAttentionProjection(64, config.d_model)
        # Shared per-frequency nonlinear lift; no extra time/frequency tokens.
        with torch.random.fork_rng(devices=[]):
            self.feature_lift = nn.Linear(32, 64)

    def forward(self, mel: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
        if mel.ndim != 3 or mel.shape[-1] != self.mel_bins:
            raise ValueError(
                f"window Conformer expects [batch,time,{self.mel_bins}] Mel input"
            )
        mel = mel.masked_fill(padding.unsqueeze(-1), 0.0)
        value = functional.gelu(self.stem(mel.unsqueeze(1)))
        value = value.masked_fill(padding[:,None,:,None],0.)
        value = self.residual(value, padding)
        value = functional.gelu(self.feature_lift(value.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
        value = self.frequency_pool(value)
        return value.masked_fill(padding.unsqueeze(-1), 0.0)


class MacaronFeedForward(nn.Module):
    def __init__(self, config: BaselineModelConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.linear1 = nn.Linear(config.d_model, config.encoder_capacity.ffn_dim)
        self.linear2 = nn.Linear(config.encoder_capacity.ffn_dim, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.norm(value)
        value = self.linear1(value)
        value = functional.silu(value)
        value = self.dropout(value)
        value = self.linear2(value)
        return self.dropout(value)


class TemporalConvolutionModule(nn.Module):
    def __init__(self, config: BaselineModelConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.expand = nn.Linear(config.d_model, config.d_model * 2)
        self.depthwise = nn.Conv1d(
            config.d_model,
            config.d_model,
            kernel_size=CONV_KERNEL,
            padding=CONV_KERNEL // 2,
            groups=config.d_model,
        )
        self.project = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, value: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
        value = self.norm(value)
        value = functional.glu(self.expand(value), dim=-1)
        value = value.masked_fill(padding.unsqueeze(-1), 0.0)
        value = self.depthwise(value.transpose(1, 2)).transpose(1, 2)
        value = functional.silu(value)
        value = self.project(value)
        return self.dropout(value).masked_fill(padding.unsqueeze(-1), 0.0)


class WindowConformerBlock(nn.Module):
    """Non-causal Macaron Conformer with an optional beat-local attention mask."""

    def __init__(
        self,
        config: BaselineModelConfig,
        *,
        attention_beat_radius: float | None,
        stage_index: int = 0,
    ) -> None:
        super().__init__()
        self.attention_beat_radius = attention_beat_radius
        self.stage_index = stage_index
        self.ffn1 = MacaronFeedForward(config)
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = nn.MultiheadAttention(
            config.d_model,
            config.heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(config.dropout)
        self.convolution = TemporalConvolutionModule(config)
        self.ffn2 = MacaronFeedForward(config)
        self.final_norm = nn.LayerNorm(config.d_model)
        self.heads = int(config.heads)

    def attend(self, normalized: torch.Tensor, padding: torch.Tensor, beats: torch.Tensor, positions: torch.Tensor, attention_layout=None) -> torch.Tensor:
        if self.attention_beat_radius is not None:
            return local_attention(self.attention, normalized, padding, beats,
                                   radius=self.attention_beat_radius, stage=self.stage_index, positions=positions, layout=attention_layout)
        return self_attention(self.attention, normalized, positions, padding, causal=False)[0]

    def forward(
        self, value: torch.Tensor, padding: torch.Tensor, beats: torch.Tensor, positions: torch.Tensor, attention_layout=None
    ) -> torch.Tensor:
        value = value + 0.5 * self.ffn1(value)
        value = value.masked_fill(padding.unsqueeze(-1), 0.0)
        normalized = self.attention_norm(value)
        attended = self.attend(normalized, padding, beats, positions, attention_layout)
        value = value + self.attention_dropout(attended)
        value = value.masked_fill(padding.unsqueeze(-1), 0.0)
        value = value + self.convolution(value, padding)
        value = value.masked_fill(padding.unsqueeze(-1), 0.0)
        value = value + 0.5 * self.ffn2(value)
        return self.final_norm(value).masked_fill(padding.unsqueeze(-1), 0.0)


class WindowConformerStage(nn.Module):
    """One resolution, conditioned once by the encoder before its first block."""

    def __init__(self, config: BaselineModelConfig, *, stage_index: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            WindowConformerBlock(config,
                attention_beat_radius=config.audio_attention.radii[stage_index],
                stage_index=stage_index)
            for _ in range(config.encoder_capacity.stage_depths[stage_index])
        )

    def forward(self, value, padding, beats, positions, attention_layout=None):
        if attention_layout is not None:
            attention_layout = with_local_mask(attention_layout, beats, padding, self.blocks[0].attention_beat_radius)
        for block in self.blocks:
            value = block(value, padding, beats, positions, attention_layout)
        return value


class StridedConformerTransition(nn.Module):
    """Learned stride-2 main path plus a mask-aware residual path."""

    def __init__(self, config: BaselineModelConfig) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            config.d_model,
            config.d_model,
            kernel_size=DOWNSAMPLE_KERNEL,
            stride=2,
            padding=DOWNSAMPLE_KERNEL // 2,
            groups=config.d_model,
        )
        self.project = nn.Conv1d(config.d_model, config.d_model, kernel_size=1)
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        value: torch.Tensor,
        padding: torch.Tensor,
        beats: torch.Tensor,
        bpms: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        masked = value.masked_fill(padding.unsqueeze(-1), 0.0)
        main = self.depthwise(masked.transpose(1, 2))
        main = functional.silu(main)
        main = self.project(main).transpose(1, 2)
        residual, feature_padding = _masked_pool_features(masked, padding)
        pooled_beats, beat_padding = _masked_pool_scalar(beats, padding)
        pooled_bpms, bpm_padding = _masked_pool_scalar(bpms, padding)
        pooled_padding = feature_padding | beat_padding | bpm_padding | padding[:, ::2]
        if main.shape != residual.shape or main.shape[:2] != pooled_padding.shape:
            raise ValueError("strided Conformer transition geometry is misaligned")
        output = self.norm(residual + self.dropout(main))
        output = output.masked_fill(pooled_padding.unsqueeze(-1), 0.0)
        return output, pooled_padding, pooled_beats, pooled_bpms


class WindowConformerEncoder(nn.Module):
    """E0--E2 Conformer at 24/12/6 audio positions per beat."""

    def __init__(self, config: BaselineModelConfig) -> None:
        super().__init__()
        if config.d_model != MODEL_DIM or config.heads != HEADS:
            raise ValueError(
                f"window Conformer fixes d_model={MODEL_DIM}, heads={HEADS}"
            )
        self.config = config
        self.tokenizer = SpectralTokenizer(config)
        self.bpm_embedding = nn.Sequential(
            nn.Linear(1, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.stages = nn.ModuleList(
            WindowConformerStage(config, stage_index=index) for index in range(3)
        )
        self.transitions = nn.ModuleList(
            StridedConformerTransition(config) for _ in range(2)
        )
        self.fusion = TopDownAudioFusion()
        if sum(p.numel() for p in self.parameters()) != config.encoder_capacity.parameter_count("encoder"):
            raise RuntimeError("encoder capacity parameter count drifted")

    def _condition(
        self,
        value: torch.Tensor,
        padding: torch.Tensor,
        beats: torch.Tensor,
        bpms: torch.Tensor,
        start_tick: torch.Tensor,
    ) -> torch.Tensor:
        conditioned = value + self.bpm_embedding(
            torch.log(bpms.to(torch.float32).clamp_min(1e-3) / 120.0)
            .unsqueeze(-1)
            .to(value.dtype)
        )
        return conditioned.masked_fill(padding.unsqueeze(-1), 0.0)

    def encode_stages(self, mel, padding_mask, frame_beats, frame_bpms, *, start_tick, local_layouts=None):
        """Local E0--E2 outputs."""
        if not (
            padding_mask.shape == frame_beats.shape == frame_bpms.shape == mel.shape[:2]
        ):
            raise ValueError("window Conformer audio tensors are not time aligned")
        if local_layouts is None:
            from maimai_baseline.data.attention_layout import to_device
            local_layouts = to_device(prepare_pyramid_layouts(frame_beats.detach().cpu(),
                padding_mask.cpu(), self.config.audio_attention.radii), mel.device)
        value = self.tokenizer(mel, padding_mask)
        padding = padding_mask
        beats = frame_beats.to(torch.float32)
        bpms = frame_bpms.to(torch.float32)
        prefix = {}
        for index, stage in enumerate(self.stages[:3]):
            with torch.autocast(device_type=value.device.type, enabled=False):
                positions = beats.float() - start_tick.float()[:, None] / CHART_TICKS_PER_BEAT
            value = stage(self._condition(value, padding, beats, bpms, start_tick), padding, beats, positions, local_layouts[index])
            for name, tensor in (("value", value), ("padding", padding), ("beats", beats), ("bpms", bpms)):
                prefix[f"{name}_{index}"] = tensor
            if index < 2:
                value, padding, beats, bpms = self.transitions[index](
                    value, padding, beats, bpms
                )
        return prefix

    def forward(self, mel, padding_mask, frame_beats, frame_bpms, *, start_tick,
                interpolation_geometry=None, song_audio=None) -> AudioContext:
        prefix = self.encode_stages(mel, padding_mask, frame_beats, frame_bpms, start_tick=start_tick,
            local_layouts=None if interpolation_geometry is None else interpolation_geometry.get("local_attention_layouts"))
        raw_values = [prefix[f"value_{i}"] for i in range(3)]
        raw_masks = [prefix[f"padding_{i}"] for i in range(3)]
        raw_beats = [prefix[f"beats_{i}"] for i in range(3)]
        raw_bpms = [prefix[f"bpms_{i}"] for i in range(3)]
        raw_lengths = [int(v.shape[1]) for v in raw_values]

        memories: list[torch.Tensor] = []
        memory_masks: list[torch.Tensor] = []
        memory_beats: list[torch.Tensor] = []
        for index, (source, source_mask, source_beats, rate) in enumerate(
            zip(raw_values, raw_masks, raw_beats, STAGE_GRID_RATES, strict=True)
        ):
            if interpolation_geometry is None:
                raise ValueError("indexed Window Conformer resampling requires geometry")
            continuous = f"grid_{index}_beats" in interpolation_geometry
            begin, end = _GRID_OFFSETS[index : index + 2]
            target_mask = (interpolation_geometry[f"grid_{index}_padding"] if continuous
                           else interpolation_geometry["grid_padding"][:, begin:end])
            target_beats = (interpolation_geometry[f"grid_{index}_beats"] if continuous else
                _fixed_beat_grid(start_tick, tokens_per_beat=rate, dtype=torch.float32))
            def geometry(name):
                return (interpolation_geometry[f"grid_{index}_{name}"] if continuous else
                        interpolation_geometry[f"grid_{name}"][:, begin:end])
            with record_function("resampling.continuous_conformer_to_absolute_grid"):
                memory = _indexed_interpolate(
                    source,
                    geometry("left"),
                    geometry("right"),
                    geometry("weight"),
                    target_mask,
                )
            memories.append(memory)
            memory_masks.append(target_mask)
            memory_beats.append(target_beats)

        return AudioContext(
            memories=tuple(reversed(self.fusion(memories, memory_masks, song_audio=song_audio,
                target_beats=memory_beats[-1], geometry=interpolation_geometry))),
            padding_masks=tuple(reversed(memory_masks)),
            beats=tuple(reversed(memory_beats)),
            fine_memories=(raw_values[0], raw_values[1]),
            fine_padding_masks=(raw_masks[0], raw_masks[1]),
            fine_beats=(raw_beats[0], raw_beats[1]),
            diagnostics={
                "audio_fusion": audio_fusion_identity(),
                "unfused_memories": tuple(memories),
                "window_conformer_raw_lengths": torch.tensor(
                    raw_lengths, device=mel.device, dtype=torch.long
                ),
                "window_conformer_memory_lengths": torch.tensor(
                    [value.shape[1] for value in reversed(memories)],
                    device=mel.device,
                    dtype=torch.long,
                ),
                "window_conformer_attention_radii": self.config.audio_attention.radii,
                "audio_attention": self.config.audio_attention.identity(),
                "encoder_capacity": self.config.encoder_capacity.identity(),
                "audio_attention_preset": self.config.audio_attention_preset,
                "window_conformer_raw_padding_masks": tuple(raw_masks),
                "window_conformer_raw_beats": tuple(raw_beats),
                "window_conformer_raw_bpms": tuple(raw_bpms),
            },
        )
