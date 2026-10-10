"""Audio preparation, fixed E_D/D/P and optional joint-trained E_M execution."""
from dataclasses import dataclass, fields, is_dataclass
from contextlib import nullcontext

import torch

from maidata.preparation.cache import song_inputs, window_inputs
from maidata.audio.layout import prepare_encoder_layout
from maidata.draft.conditions import ChartCondition, make_draft_inputs
from maidata.draft.types import BeatMemory
from maidata.draft.windows import song_coverage, audio_windows, assemble_full_p0, select_windows
from .progress import emit, stage


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
                       difficulty, level, device, encoder_window_batch=8, progress=None,
                       decoder_encoder=None):
    """Fixed E_D -> D/P; optional independent E_M -> Decoder audio, each encoded once."""
    c = config
    coverage = song_coverage(timing, duration, first_beat=int(grid['first_beat']), target_beats=c['target_beats'])
    starts, mapping = audio_windows([coverage], device='cpu')
    with stage(progress, 'encoder_global'):
        host_song = song_inputs([grid], validate_input=False)
        inputs_song = to_device(host_song, device)
        song = encoder.prepare_song(inputs_song)
    decoder_song = None
    if decoder_encoder is not None:
        with stage(progress, 'decoder_encoder_global'):
            decoder_song = decoder_encoder.prepare_song(inputs_song)
    pieces, decoder_pieces = [], []
    decoder_stage = stage(progress, 'decoder_encoder_windows') if decoder_encoder is not None else nullcontext()
    with stage(progress, 'encoder_windows'), decoder_stage:
        for at in range(0, len(starts), encoder_window_batch):
            windows = window_inputs([grid], starts[at:at+encoder_window_batch].numpy(),
                mapping[at:at+encoder_window_batch].numpy(), target_beats=c['target_beats'],
                before_beats=c['audio_before_beats'], after_beats=c['audio_after_beats'], validate_input=False)
            # Layout depends on the shared input coordinates/configuration, not
            # either encoder's parameters. Build/transfer it once for both E's.
            windows.layout = prepare_encoder_layout(host_song.beat_centers, host_song.sample_valid.any(-1), windows,
                radii=c['stage_attention_radii'], rates=c['fpn_rates'], before=c['audio_before_beats'],
                context=c['audio_before_beats']+c['target_beats']+c['audio_after_beats'])
            windows = to_device(windows, device)
            pieces.append(encoder.encode_windows(song, windows).P0)
            complete = min(at+encoder_window_batch, len(starts))
            emit(progress, 'encoder_progress', encoder='draft', completed=complete, total=len(starts))
            if decoder_encoder is not None:
                decoder_pieces.append(decoder_encoder.encode_windows(decoder_song, windows).P0)
                emit(progress, 'encoder_progress', encoder='decoder', completed=complete, total=len(starts))
        device_starts, device_mapping = starts.to(device), mapping.to(device)
        def assemble(parts):
            local = BeatMemory(*(torch.cat([getattr(p, name) for p in parts]) for name in ('values', 'beats', 'valid')))
            return assemble_full_p0(local, device_starts, device_mapping, [coverage], p0_rate=c['fpn_rates'][0])
        audio = assemble(pieces)
        decoder_audio = assemble(decoder_pieces) if decoder_encoder is not None else audio
    inputs, _ = make_draft_inputs([ChartCondition('inference', difficulty, level, str(level))],
        ['inference'], [coverage], device=device, validate_input=False)
    with stage(progress, 'draft'):
        z = draft(audio, inputs)
    all_starts = coverage.starts(include_tail=True)
    target_starts = torch.tensor(all_starts, device=device, dtype=torch.float32)
    groups, valid = torch.zeros(len(all_starts), device=device, dtype=torch.long), torch.ones(len(all_starts), device=device, dtype=torch.bool)
    windows = select_windows(z, groups, target_starts, valid, target_beats=c['target_beats'], before_beats=0, after_beats=0)
    with stage(progress, 'planner'):
        plans = planner(windows)
    return Conditions(decoder_audio, z, plans.plan_values, plans.valid, all_starts, coverage)
