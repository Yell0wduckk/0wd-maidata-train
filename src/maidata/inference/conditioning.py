"""Audio preparation and one-time E/D/P execution. No chart targets or Decoder calls."""
from dataclasses import dataclass, fields, is_dataclass

import torch

from maidata.preparation.cache import song_inputs, window_inputs
from maidata.draft.conditions import ChartCondition, make_draft_inputs
from maidata.draft.types import BeatMemory
from maidata.draft.windows import song_coverage, audio_windows, assemble_full_p0, select_windows


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if is_dataclass(value):
        return type(value)(**{f.name: to_device(getattr(value, f.name), device) for f in fields(value)})
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(to_device(v, device) for v in value)
    return value


def audio_grid(audio_path, timing, stats, *, samples_per_beat):
    """Use exactly the training feature algorithms and supplied fixed Mel statistics."""
    from maidata.preparation.features import (read_wav, linear_resample, acoustic18,
        grid_coordinates, mel_at_centers, derive_grid)
    from maidata.preparation.timing import TempoMap
    y, sr = read_wav(audio_path)
    duration = len(y) / sr
    pcm, tempo = linear_resample(y, sr, 24000), TempoMap(timing)
    coordinates = grid_coordinates(tempo, duration, len(pcm), samples_per_beat)
    shape = coordinates['grid_k'].shape
    mel = mel_at_centers(pcm, coordinates['sample_indices'].ravel(),
                         coordinates['sample_valid'].ravel(), stats).reshape(*shape, 128)
    times, acoustic = acoustic18(y, sr)
    return derive_grid(mel, coordinates, times, acoustic, tempo, duration), duration


@dataclass(frozen=True)
class Conditions:
    audio: BeatMemory
    z: BeatMemory
    plans: torch.Tensor
    plan_valid: torch.Tensor
    starts: tuple
    coverage: object


@torch.inference_mode()
def prepare_conditions(encoder, draft, planner, config, grid, timing, duration, *,
                       difficulty, level, device, encoder_window_batch=8):
    """G0 once, every audio window once, full-song Z once, target-only plan12 once."""
    c = config
    coverage = song_coverage(timing, duration, first_beat=int(grid['first_beat']), target_beats=c['target_beats'])
    starts, mapping = audio_windows([coverage], device='cpu')
    song = encoder.prepare_song(to_device(song_inputs([grid], validate_input=False), device))
    pieces = []
    for at in range(0, len(starts), encoder_window_batch):
        windows = window_inputs([grid], starts[at:at+encoder_window_batch].numpy(),
            mapping[at:at+encoder_window_batch].numpy(), target_beats=c['target_beats'],
            before_beats=c['audio_before_beats'], after_beats=c['audio_after_beats'], validate_input=False)
        pieces.append(encoder.encode_windows(song, to_device(windows, device)).P0)
    local = BeatMemory(*(torch.cat([getattr(p, name) for p in pieces]) for name in ('values', 'beats', 'valid')))
    audio = assemble_full_p0(local, starts.to(device), mapping.to(device), [coverage], p0_rate=c['fpn_rates'][0])
    inputs, _ = make_draft_inputs([ChartCondition('inference', difficulty, level, str(level))],
        ['inference'], [coverage], device=device, validate_input=False)
    z = draft(audio, inputs)
    all_starts = coverage.starts(include_tail=True)
    target_starts = torch.tensor(all_starts, device=device, dtype=torch.float32)
    groups, valid = torch.zeros(len(all_starts), device=device, dtype=torch.long), torch.ones(len(all_starts), device=device, dtype=torch.bool)
    windows = select_windows(z, groups, target_starts, valid, target_beats=c['target_beats'], before_beats=0, after_beats=0)
    plans = planner(windows)
    return Conditions(audio, z, plans.plan_values, plans.valid, all_starts, coverage)
