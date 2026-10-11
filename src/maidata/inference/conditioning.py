"""Audio preparation for fixed teachers, independent E_M, or one shared joint E."""
from dataclasses import dataclass, fields, is_dataclass, replace
from contextlib import nullcontext

import torch

from maidata.preparation.cache import song_inputs, window_inputs
from maidata.model import ENCODER_KEYS
from maidata.audio.layout import prepare_encoder_layout
from maidata.draft.conditions import ChartCondition, make_draft_inputs
from maidata.draft.types import BeatMemory
from maidata.draft.windows import song_coverage, audio_windows, assemble_full_p0, select_windows
from .input_contract import check_shared_grid_config
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
    audio_scope: str = 'full_song'


def with_plan_source(conditions, source, document=None):
    """GT rows are explicit per-window inputs; none masks the key completely."""
    if source == 'predicted':
        return conditions
    plans = torch.zeros_like(conditions.plans)
    valid = torch.zeros_like(conditions.plan_valid)
    if source == 'gt':
        if document is None:
            raise ValueError('GT plan requires a window plan JSON')
        rows = document['windows']
        by_start = {float(row['start_beat']): row for row in rows}
        if len(by_start) != len(rows):
            raise ValueError('duplicate GT plan window start')
        for i, start in enumerate(conditions.starts):
            row = by_start[float(start)]
            value = torch.as_tensor(row['plan12'], dtype=plans.dtype, device=plans.device)
            if value.shape != (12,) or not torch.isfinite(value).all() or row.get('valid', True) is not True:
                raise ValueError('GT plan requires twelve finite values and a valid row')
            plans[i] = value
            valid[i] = True
    elif source != 'none':
        raise ValueError('unknown plan source')
    return replace(conditions, plans=plans, plan_valid=valid)


@torch.inference_mode()
def prepare_decoder_conditions(encoder, config, grid, timing, duration, *, device,
                               encoder_window_batch=8, progress=None):
    """Z-off GT/none inference runs only the independently trained E_M."""
    c = config
    coverage = song_coverage(timing, duration, first_beat=int(grid['first_beat']), target_beats=c['target_beats'])
    starts = coverage.starts(include_tail=True)
    host = song_inputs([grid], validate_input=False)
    with stage(progress, 'decoder_encoder_global'):
        song = encoder.prepare_song(to_device(host, device))
    pieces = []
    with stage(progress, 'decoder_encoder_windows'):
        for at in range(0, len(starts), encoder_window_batch):
            chosen = starts[at:at+encoder_window_batch]
            windows = window_inputs([grid], chosen, [0]*len(chosen), target_beats=c['target_beats'],
                before_beats=c['audio_before_beats'], after_beats=c['audio_after_beats'], validate_input=False)
            windows.layout = prepare_encoder_layout(host.beat_centers, host.sample_valid.any(-1), windows,
                radii=c['stage_attention_radii'], rates=c['fpn_rates'], before=c['audio_before_beats'],
                context=c['audio_before_beats']+c['target_beats']+c['audio_after_beats'])
            pieces.append(encoder.encode_windows(song, to_device(windows, device)).P0)
            emit(progress, 'encoder_progress', encoder='decoder', completed=at+len(chosen), total=len(starts))
    audio = BeatMemory(*(torch.cat([getattr(p, name) for p in pieces]) for name in ('values', 'beats', 'valid')))
    if c.get('decoder_train_audio_scope', 'full_song') != 'window':
        # The extra complete tail is a target, not part of the full-song bank.
        local = BeatMemory(*(getattr(audio, name)[:-1] for name in ('values', 'beats', 'valid')))
        audio = assemble_full_p0(local, torch.tensor(starts[:-1], device=device),
            torch.zeros(len(starts)-1, dtype=torch.long, device=device), [coverage], p0_rate=c['fpn_rates'][0])
    z = BeatMemory(audio.values.new_zeros((1, 0, c['draft_latent_dim'])),
                   audio.beats.new_zeros((1, 0)), torch.zeros((1, 0), dtype=torch.bool, device=device))
    return Conditions(audio, z, torch.zeros((len(starts), 12), device=device),
                      torch.zeros(len(starts), dtype=torch.bool, device=device), starts, coverage,
                      c.get('decoder_train_audio_scope', 'full_song'))


@torch.inference_mode()
def prepare_joint_conditions(encoder, draft, planner, config, grid, timing, duration, *,
                             difficulty, level, device, encoder_window_batch=8, progress=None):
    """Encode each local window once; share its center P0 with the full-song D."""
    c = dict(config, decoder_train_audio_scope='window')
    local = prepare_decoder_conditions(encoder, c, grid, timing, duration, device=device,
        encoder_window_batch=encoder_window_batch, progress=progress)
    coverage, starts = local.coverage, local.starts
    # The final complete target tail needs local audio for M, but contributes no
    # new center coverage to D. Both banks reuse values from the same E forward.
    centers = BeatMemory(*(getattr(local.audio, name)[:-1] for name in ('values', 'beats', 'valid')))
    audio = assemble_full_p0(centers, torch.tensor(starts[:-1], device=device, dtype=torch.float32),
        torch.zeros(len(starts)-1, device=device, dtype=torch.long), [coverage], p0_rate=c['fpn_rates'][0])
    inputs, _ = make_draft_inputs([ChartCondition('inference', difficulty, level, str(level))],
        ['inference'], [coverage], device=device, validate_input=False,
        z_stride_beats=c['draft_z_stride_beats'])
    with stage(progress, 'draft'):
        z = draft(audio, inputs)
    target_starts = torch.tensor(starts, device=device, dtype=torch.float32)
    groups = torch.zeros(len(starts), device=device, dtype=torch.long)
    valid = torch.ones(len(starts), device=device, dtype=torch.bool)
    windows = select_windows(z, groups, target_starts, valid, target_beats=c['target_beats'],
        before_beats=0, after_beats=0, z_stride_beats=c['draft_z_stride_beats'])
    with stage(progress, 'planner'):
        plans = planner(windows)
    return Conditions(local.audio, z, plans.plan_values, plans.valid, starts, coverage, 'window')


@torch.inference_mode()
def prepare_conditions(encoder, draft, planner, config, grid, timing, duration, *,
                       difficulty, level, device, encoder_window_batch=8, progress=None,
                       decoder_encoder=None, teacher_config=None):
    """Fixed E_D -> D/P; optional independent E_M -> Decoder audio, each encoded once."""
    if decoder_encoder is encoder:
        return prepare_joint_conditions(encoder, draft, planner, config, grid, timing, duration,
            difficulty=difficulty, level=level, device=device,
            encoder_window_batch=encoder_window_batch, progress=progress)
    c = config if teacher_config is None else teacher_config
    if decoder_encoder is not None:
        check_shared_grid_config(c, config)
    local_decoder = decoder_encoder is not None and config.get('decoder_train_audio_scope', 'full_song') == 'window'
    separate_decoder = decoder_encoder is not None and any(c[k] != config[k] for k in ENCODER_KEYS)
    shared_decoder = None if separate_decoder else decoder_encoder
    coverage = song_coverage(timing, duration, first_beat=int(grid['first_beat']), target_beats=c['target_beats'])
    starts, mapping = audio_windows([coverage], device='cpu')
    with stage(progress, 'encoder_global'):
        host_song = song_inputs([grid], validate_input=False)
        inputs_song = to_device(host_song, device)
        song = encoder.prepare_song(inputs_song)
    decoder_song = None
    if shared_decoder is not None:
        with stage(progress, 'decoder_encoder_global'):
            decoder_song = shared_decoder.prepare_song(inputs_song)
    pieces, decoder_pieces = [], []
    decoder_stage = stage(progress, 'decoder_encoder_windows') if shared_decoder is not None else nullcontext()
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
            if shared_decoder is not None:
                decoder_pieces.append(shared_decoder.encode_windows(decoder_song, windows).P0)
                emit(progress, 'encoder_progress', encoder='decoder', completed=complete, total=len(starts))
        device_starts, device_mapping = starts.to(device), mapping.to(device)
        def assemble(parts):
            local = BeatMemory(*(torch.cat([getattr(p, name) for p in parts]) for name in ('values', 'beats', 'valid')))
            return assemble_full_p0(local, device_starts, device_mapping, [coverage], p0_rate=c['fpn_rates'][0])
        audio = assemble(pieces)
        if local_decoder and shared_decoder is not None:
            # The complete supervised tail also has its own local audio context.
            tail = window_inputs([grid], [coverage.starts(include_tail=True)[-1]], [0],
                target_beats=c['target_beats'], before_beats=c['audio_before_beats'],
                after_beats=c['audio_after_beats'], validate_input=False)
            tail.layout = prepare_encoder_layout(host_song.beat_centers, host_song.sample_valid.any(-1), tail,
                radii=c['stage_attention_radii'], rates=c['fpn_rates'], before=c['audio_before_beats'],
                context=c['audio_before_beats']+c['target_beats']+c['audio_after_beats'])
            decoder_pieces.append(shared_decoder.encode_windows(decoder_song, to_device(tail, device)).P0)
            decoder_audio = BeatMemory(*(torch.cat([getattr(p, name) for p in decoder_pieces])
                                        for name in ('values', 'beats', 'valid')))
        else:
            decoder_audio = assemble(decoder_pieces) if shared_decoder is not None else audio
    if separate_decoder:
        independent = prepare_decoder_conditions(decoder_encoder, config, grid, timing, duration,
            device=device, encoder_window_batch=encoder_window_batch, progress=progress)
        decoder_audio = independent.audio
    inputs, _ = make_draft_inputs([ChartCondition('inference', difficulty, level, str(level))],
        ['inference'], [coverage], device=device, validate_input=False,
        z_stride_beats=c['draft_z_stride_beats'])
    with stage(progress, 'draft'):
        z = draft(audio, inputs)
    all_starts = coverage.starts(include_tail=True)
    target_starts = torch.tensor(all_starts, device=device, dtype=torch.float32)
    groups, valid = torch.zeros(len(all_starts), device=device, dtype=torch.long), torch.ones(len(all_starts), device=device, dtype=torch.bool)
    windows = select_windows(z, groups, target_starts, valid, target_beats=c['target_beats'], before_beats=0, after_beats=0,
        z_stride_beats=c['draft_z_stride_beats'])
    with stage(progress, 'planner'):
        plans = planner(windows)
    return Conditions(decoder_audio, z, plans.plan_values, plans.valid, all_starts, coverage,
                      'window' if local_decoder else 'full_song')
