"""First anchors chart/audio input at beat zero, independently of WAV time zero."""
from unittest.mock import patch

import numpy as np
import pytest
import torch

from test_inference import config, producer_config
from test_chart_physical import geometry
from maidata.chart import Moment, decode_chart
from maidata.decoder.windows import real_onsets
from maidata.inference.__main__ import ir_metadata
from maidata.inference.conditioning import prepare_conditions, prepare_decoder_conditions
from maidata.inference.generation import generate, GenerationOptions
from maidata.inference.inputs import beat_zero_grid
from maidata.model import build_model, build_draft, build_planner, build_decoder
from maidata.preparation.cache import validate_grid, window_inputs
from maidata.preparation.features import grid_coordinates
from maidata.preparation.timing import TempoMap
from maidata.time import TimingMap


def cached_grid(config, first, duration=2.7):
    clock = dict(first_seconds=first, bpm=[[0, 1, 120], [1, 1, 180]])
    tempo = TempoMap(clock)
    coordinates = grid_coordinates(tempo, duration, round(duration*24000), config['samples_per_beat'])
    j, r = coordinates['grid_k'].shape
    valid = coordinates['sample_valid']
    mel = np.ones((j, r, config['mel_bins']), np.float32)
    aux = np.ones((j, r, 32), np.float32)
    mel[~valid], aux[~valid] = 0, 0
    center_times = tempo.to_seconds(coordinates['beat_centers'])
    centers = np.ones((j, 32), np.float32)
    centers[(center_times < 0) | (center_times >= duration)] = 0
    return dict(**coordinates, mel_patches=mel, aux32_grid=aux, aux32_centers=centers), TimingMap.from_audio_timing(clock)


@pytest.mark.parametrize('first', [.02, .125, 0., -.125, -1.25, -3.])
def test_cache_view_drops_pre_first_audio_and_pads_before_wav(config, first):
    original, timing = cached_grid(config, first)
    saved = {k: v.copy() for k, v in original.items()}
    grid = beat_zero_grid(original, timing)
    validate_grid(grid)
    assert int(grid['first_beat']) == 0
    assert grid['beat_centers'][0] == .5
    assert grid['sample_times_sec'][0, 0] == pytest.approx(first)
    for key in saved:
        np.testing.assert_array_equal(original[key], saved[key])
    for i, center in enumerate(original['beat_centers']):
        beat = int(np.floor(center))
        if beat >= 0:
            for key in ('mel_patches', 'sample_valid', 'aux32_grid', 'aux32_centers', 'sample_times_sec'):
                np.testing.assert_array_equal(grid[key][beat], original[key][i])
    before_wav = grid['sample_times_sec'] < 0
    assert not grid['sample_valid'][before_wav].any()
    assert not grid['mel_patches'][before_wav].any()
    assert not grid['aux32_grid'][before_wav].any()
    windows = window_inputs([grid], [0, 3], [0, 0], target_beats=3, before_beats=4, after_beats=1)
    r = config['samples_per_beat']
    assert not windows.valid[0, :4*r].any()
    assert not windows.mel[0, :4*r].any()
    assert not windows.aux32[0, :4*r].any()
    # Later windows can see real context after beat zero, never the removed prefix.
    assert not windows.valid[1, :r].any()
    torch.testing.assert_close(windows.mel[1, r:4*r], torch.from_numpy(grid['mel_patches'][:3].reshape(3*r, -1)))


@pytest.mark.parametrize('first', [.02, 0., -1.25, -3.])
@pytest.mark.parametrize('mode', ['joint', 'two_stage', 'separate_recipe', 'z_off'])
def test_all_inference_encoders_and_generation_start_at_beat_zero(config, geometry, first, mode):
    c = dict(config, audio_before_beats=4, decoder_train_audio_scope='window')
    if mode == 'z_off':
        c['decoder_z_cross_attention'] = False
    grid, timing = cached_grid(c, first)
    teacher_config = c
    if mode == 'separate_recipe':
        c = dict(c, audio_after_beats=2)
    encoder, draft, planner = (factory(teacher_config).eval() for factory in (build_model, build_draft, build_planner))
    decoder_encoder = encoder if mode == 'joint' else build_model(c).eval()
    options = dict(device='cpu', encoder_window_batch=1)
    with patch.object(encoder, 'prepare_song', wraps=encoder.prepare_song) as teacher_song, \
         patch.object(decoder_encoder, 'encode_windows', wraps=decoder_encoder.encode_windows) as decoder_windows:
        if mode == 'z_off':
            prepared = prepare_decoder_conditions(decoder_encoder, c, grid, timing, 2.7, **options)
        else:
            prepared = prepare_conditions(encoder, draft, planner, c, grid, timing, 2.7,
                decoder_encoder=decoder_encoder, teacher_config=teacher_config,
                difficulty=5, level=14.5, **options)
        assert prepared.starts[0] == 0
        assert prepared.starts == tuple(range(0, (prepared.coverage.audio_windows+1)*c['target_beats'], c['target_beats']))
        if mode != 'z_off':
            song = teacher_song.call_args.args[0]
            assert song.beat_centers[0, 0] == .5
            if first < 0:
                assert not song.sample_valid[0, :2].any()
                assert not song.mel_patches[0, :2].any()
                assert not song.aux32_centers[0, :2].any()
        first_window = decoder_windows.call_args_list[0].args[1]
        r = c['samples_per_beat']
        assert first_window.target_start_beats[0] == 0
        assert not first_window.valid[0, :4*r].any()
        assert not first_window.mel[0, :4*r].any()
        assert not first_window.aux32[0, :4*r].any()
    # Exercise real Decoder/CONT and IR export on the resulting conditions.
    decoder = build_decoder(c).eval()
    with torch.no_grad():
        decoder.type_head.weight.zero_()
        decoder.type_head.bias.fill_(-100)
        decoder.type_head.bias[0] = 100
        decoder.heads.times['MOMENT'].weight.zero_()
        decoder.heads.times['MOMENT'].bias.fill_(-100)
        decoder.heads.times['MOMENT'].bias[384] = 100
    encoded, windows = generate(decoder, prepared, timing, geometry, c, ir_metadata(timing, 2.7),
        2.7, device='cpu', options=GenerationOptions(temperature=0))
    assert encoded.start_tick == 0 and encoded.tokens == (Moment(0),)
    assert windows[0]['start_beat'] == 0
    ir = decode_chart(encoded, geometry)
    assert ir['first_seconds'] == first


@pytest.mark.parametrize('first', [.02, 0., -1.25, -3.])
def test_nonempty_generation_preserves_first_and_never_generates_negative_beats(config, geometry, first):
    c = dict(config, audio_before_beats=4, decoder_train_audio_scope='window', decoder_z_cross_attention=False)
    grid, timing = cached_grid(c, first)
    prepared = prepare_decoder_conditions(build_model(c).eval(), c, grid, timing, 2.7, device='cpu')
    decoder = build_decoder(c).eval()
    with torch.no_grad():
        decoder.type_head.weight.zero_()
        decoder.type_head.bias.fill_(-100)
        decoder.type_head.bias[0], decoder.type_head.bias[1] = 80, 100
        decoder.heads.kind.weight.zero_()
        decoder.heads.kind.bias.fill_(-100)
        decoder.heads.kind.bias[0] = 100
        decoder.heads.times['MOMENT'].weight.zero_()
        decoder.heads.times['MOMENT'].bias.fill_(-100)
        decoder.heads.times['MOMENT'].bias[384] = 100
    encoded, _ = generate(decoder, prepared, timing, geometry, c, ir_metadata(timing, 2.7),
        2.7, device='cpu', options=GenerationOptions(temperature=0, max_tokens_per_moment=1))
    groups = real_onsets(encoded.tokens, timing, sequence_start_tick=encoded.start_tick)
    assert groups and encoded.start_tick == 0
    assert all(group.tick >= 0 and timing.second(group.tick) >= -1e-8 for group in groups)
    ir = decode_chart(encoded, geometry)
    assert ir['first_seconds'] == first
    if first >= 0:
        assert groups[0].tick == 0
        assert ir['moments'][0]['second'] == pytest.approx(first)
