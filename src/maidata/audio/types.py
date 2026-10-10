"""Tensor containers; boolean masks use True for valid audio."""

from dataclasses import dataclass

from torch import Tensor


@dataclass
class SongInputs:
    mel_patches: Tensor
    sample_valid: Tensor
    beat_centers: Tensor
    aux32_centers: Tensor


@dataclass
class WindowInputs:
    mel: Tensor
    aux32: Tensor
    beats: Tensor
    valid: Tensor
    target_start_beats: Tensor
    window_to_song: Tensor


@dataclass
class ScaleMemory:
    values: Tensor
    beats: Tensor
    valid: Tensor


@dataclass
class SongMemory:
    g0: ScaleMemory
    block_kv: tuple[tuple[Tensor, Tensor] | None, ...]


@dataclass
class EncoderOutput:
    P0: ScaleMemory
    P1: ScaleMemory
    P2: ScaleMemory
