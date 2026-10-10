"""Public E -> full P0 -> D -> R/P and external consumer gradient contracts."""
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch

from maidata.audio.types import SongInputs, WindowInputs
from maidata.entry import CONFIG
from maidata.model import build_model, build_draft, build_planner, build_reconstruction
from maidata.time import TimingMap
from maidata.draft.types import BeatMemory, DraftInputs
from maidata.draft.windows import (song_coverage, audio_windows, assemble_full_p0,
                                   draft_queries, select_windows)
from maidata.draft.chart_data import ChartData, OnsetGroup, PlanNote
from maidata.chart.types import Note
from maidata.draft.plan_targets import MusicPlanTargets
from maidata.draft.targets import reconstruction_target, collate_reconstruction, collate_planner
from maidata.draft.supervision import predict_auxiliary, auxiliary_losses


def test_ownership_sorted_centers_padding_and_nonzero_origins():
    timing = TimingMap([(0, 120)], first_seconds=.25)
    c = song_coverage(timing, 2., first_beat=-1, target_beats=3)
    assert c.starts(include_tail=False) == (-1, 2)
    assert c.starts(include_tail=True) == (-1, 2, 5)
    starts = torch.tensor([2., -1.])  # Deliberately out of order.
    positions = starts[:, None] - 2 + (torch.arange(12).float()[None] + .5) / 2
    values = torch.arange(48).float().reshape(2, 12, 2).requires_grad_()
    memory = BeatMemory(values, positions, (positions >= -.5) & (positions < 3.5))
    bank = assemble_full_p0(memory, starts, torch.tensor([0, 0]), [c], p0_rate=2)
    torch.testing.assert_close(bank.beats[0], -1 + (torch.arange(12).float() + .5) / 2)
    torch.testing.assert_close(bank.values[0, :6], memory.values[1, 4:10].masked_fill(~memory.valid[1, 4:10, None], 0))
    bank.values.sum().backward()
    assert values.grad[:, :4].count_nonzero() == 0 and values.grad[:, 10:].count_nonzero() == 0
    with pytest.raises(ValueError, match='exactly once'):
        assemble_full_p0(memory, torch.tensor([-1., -1.]), torch.tensor([0, 0]), [c], p0_rate=2)
    q, valid = draft_queries([c], torch.tensor([0]))
    z = BeatMemory(torch.randn(1, 9, 5, requires_grad=True), q, valid)
    tail = select_windows(z, torch.tensor([0]), torch.tensor([5.]), torch.tensor([True]),
                          target_beats=3, before_beats=2, after_beats=1)
    assert tail.memory.valid.tolist() == [[True, True, True, True, True, False]]
    assert q[0, -1] == 7.5 and valid.all()
    tail.memory.values.sum().backward()
    assert z.values.grad[0, :4].count_nonzero() == 0
    assert z.values.grad[0, 4:].abs().sum() > 0
    with pytest.raises(ValueError, match='outside'):
        select_windows(z, torch.tensor([0]), torch.tensor([8.]), torch.tensor([True]),
                       target_beats=3, before_beats=0, after_beats=0)


def small_chain():
    torch.manual_seed(5)
    config = dict(CONFIG, mel_bins=7, samples_per_beat=5, d_model=18, heads=3, ffn_dim=28,
        stage_depths=(1, 1, 1), temporal_kernel_size=3, stem_channels=4, frequency_dim=6,
        frequency_queries=2, fpn_rates=(3, 2, 1), target_beats=3,
        audio_before_beats=2, audio_after_beats=1, dropout=0.,
        draft_layers=2, draft_dim=24, draft_heads=3, draft_experts=3, draft_expert_ffn_dim=29,
        draft_latent_dim=10, planner_dim=16, planner_heads=2, planner_mlp_dim=23, planner_dropout=0.,
        reconstruction_dim=12, reconstruction_heads=3, reconstruction_ffn_dim=25,
        reconstruction_dropout=0., reconstruction_query_rows=12,
        reconstruction_before_beats=2, reconstruction_after_beats=1)
    timing = TimingMap([(0, 120)])
    coverage = song_coverage(timing, 1.2, first_beat=0, target_beats=3)
    grid = torch.arange(15).reshape(1, 3, 5).float() / 5
    valid = grid < 2.4
    song = SongInputs(torch.randn(1, 3, 5, 7), valid, torch.tensor([[.5, 1.5, 2.5]]),
                       torch.randn(1, 3, 32))
    positions = torch.arange(30).float()[None] / 5 - 2
    indices = (torch.arange(30) - 10).clamp(0, 14)
    wvalid = (positions >= 0) & (positions < 2.4)
    starts, mapping = audio_windows([coverage], device='cpu')
    window = WindowInputs(song.mel_patches.flatten(1, 2)[:, indices].masked_fill(~wvalid[..., None], 0),
                          torch.randn(1, 30, 32).masked_fill(~wvalid[..., None], 0),
                          positions, wvalid, starts, mapping)
    e, d, p, r = build_model(config), build_draft(config), build_planner(config), build_reconstruction(config)
    beats, draft_valid = draft_queries([coverage], torch.tensor([0]))
    inputs = DraftInputs(beats, draft_valid, torch.tensor([0]), torch.tensor([5]),
                         torch.tensor([14.6]), torch.tensor([coverage.span_beats]))
    return config, timing, coverage, song, window, e, d, p, r, inputs


@pytest.mark.parametrize('target,after,rate', [(2, 1, 1.5), (3, 0, .75)])
def test_fractional_encoder_grid_is_preserved_by_full_p0(target, after, rate):
    config = dict(CONFIG, mel_bins=7, samples_per_beat=4, d_model=12, heads=3, ffn_dim=20,
                  stage_depths=(1, 1, 1), temporal_kernel_size=3, stem_channels=4,
                  frequency_dim=4, frequency_queries=2, target_beats=target,
                  audio_before_beats=1, audio_after_beats=after,
                  fpn_rates=(rate, 1., .5), dropout=0.)
    encoder = build_model(config).eval()
    coverage = song_coverage(TimingMap([(0, 120)]), float(target), first_beat=0, target_beats=target)
    starts, mapping = audio_windows([coverage], device='cpu')
    n = 2 * target
    song = SongInputs(torch.randn(1, n, 4, 7), torch.ones(1, n, 4, dtype=torch.bool),
                      torch.arange(n).float()[None] + .5, torch.randn(1, n, 32))
    beats = starts[:, None] - 1 + torch.arange(16).float()[None] / 4
    valid = (beats >= 0) & (beats < n)
    windows = WindowInputs(torch.randn(2, 16, 7).masked_fill(~valid[..., None], 0),
                          torch.randn(2, 16, 32).masked_fill(~valid[..., None], 0),
                          beats, valid, starts, mapping)
    p0 = encoder(song, windows).P0
    p0.values.retain_grad()
    bank = assemble_full_p0(p0, starts, mapping, [coverage], p0_rate=rate)
    selected = (p0.beats >= starts[:, None]) & (p0.beats < starts[:, None] + target)
    assert torch.equal(bank.beats[0], p0.beats[selected])
    assert torch.equal(bank.valid[0], p0.valid[selected])
    assert torch.equal(bank.values[0], p0.values[selected].masked_fill(~p0.valid[selected][:, None], 0))
    assert (bank.beats[0, 1:] > bank.beats[0, :-1]).all()
    bank.values.sum().backward()
    assert torch.equal(p0.values.grad, (selected & p0.valid)[..., None].expand_as(p0.values).float())
    assert encoder.spectral.stem.weight.grad.abs().sum() > 0
    damaged = p0.beats.clone()
    damaged[0, 1] += .1
    with pytest.raises(ValueError, match='configured rate grid'):
        assemble_full_p0(replace(p0, beats=damaged), starts, mapping, [coverage], p0_rate=rate)


def test_trainable_chain_empty_tail_and_auxiliary_gradients():
    config, timing, coverage, song, window, e, d, p, r, inputs = small_chain()
    with patch.object(e, 'prepare_song', wraps=e.prepare_song) as prepare:
        local = e(song, window)
        bank = assemble_full_p0(local.P0, window.target_start_beats, window.window_to_song,
                                [coverage], p0_rate=config['fpn_rates'][0])
        z = d(bank, inputs)
        assert prepare.call_count == 1
    assert bank.values.shape == (1, 9, 18)
    assert z.values.shape == (1, 6, 10) and z.valid.all()
    assert z.values[0, 3:].abs().sum() > 0  # No local audio, but a valid supervised tail.
    groups, starts, valid = torch.tensor([0, 0, 0]), torch.tensor([0., 3., 6.]), torch.tensor([True, True, False])
    ro, po = predict_auxiliary(z, groups, starts, valid, r, p, config)
    chart = ChartData((OnsetGroup(0, (Note.make('tap', 'A1'),)),
                       OnsetGroup(120, (Note.make('touch', 'C'),))),
                      (PlanNote(0, 'tap', 1), PlanNote(120, 'touch', None)), timing, 3)
    rt = collate_reconstruction([reconstruction_target(chart, start_tick=s * 96, target_beats=3)
                                 for s in (0, 3)] + [None], query_rows=12, target_beats=3, device='cpu')
    labels = MusicPlanTargets(chart)
    pt = collate_planner([labels.window(0, 288), labels.window(288, 576), None], device='cpu')
    terms = auxiliary_losses(ro, po, rt, pt, config)
    # Unit weights here are a synthetic gradient probe, not a training recipe.
    sum(term.mean for term in terms.values()).backward()
    for module in (e.spectral.stem, e.feature_fusion.project, e.song_tokenizer.project,
                   d.blocks[0].cross_attention.key, d.output, p.continuous,
                   r.field_heads['field_type'], r.count_head):
        assert module.weight.grad is not None
        assert torch.isfinite(module.weight.grad).all() and module.weight.grad.abs().sum() > 0
    assert terms['count'].count == 6 and terms['planner_category'].count == 2


@pytest.mark.parametrize('condition_path', ['Z', 'plan'])
def test_external_consumer_gradient_and_actual_freezing(condition_path):
    config, _, coverage, song, window, e, d, p, _, inputs = small_chain()

    def forward():
        bank = assemble_full_p0(e(song, window).P0, window.target_start_beats, window.window_to_song,
                                [coverage], p0_rate=config['fpn_rates'][0])
        z = d(bank, inputs)
        selected = select_windows(z, torch.tensor([0]), torch.tensor([3.]), torch.tensor([True]),
                                  target_beats=3, before_beats=0, after_beats=0)
        return selected.memory.values if condition_path == 'Z' else p(selected).plan_values

    # A generic external readout probes the interface; this is not a Decoder implementation.
    condition = forward()
    consumer = torch.nn.Linear(condition.shape[-1], 1)
    consumer(condition).sum().backward()
    assert e.spectral.stem.weight.grad.abs().sum() > 0
    assert d.output.weight.grad.abs().sum() > 0
    if condition_path == 'plan':
        assert p.continuous.weight.grad.abs().sum() > 0
    for module in (e, d, p):
        module.zero_grad(set_to_none=True)
        module.requires_grad_(False).eval()
    consumer.zero_grad(set_to_none=True)
    a, b = forward(), forward()
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    consumer(a).sum().backward()
    assert consumer.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for module in (e, d, p) for parameter in module.parameters())


def test_default_network_shapes_and_decoupled_imports():
    d, p = build_draft(CONFIG).eval(), build_planner(CONFIG).eval()
    r = build_reconstruction(dict(CONFIG, reconstruction_query_rows=9)).eval()
    audio = BeatMemory(torch.randn(1, 5, 256), torch.arange(5).float()[None], torch.ones(1, 5, dtype=torch.bool))
    inputs = DraftInputs(torch.arange(32).float()[None] + .5, torch.ones(1, 32, dtype=torch.bool),
                         torch.tensor([0]), torch.tensor([5]), torch.tensor([14.6]), torch.tensor([4.5]))
    with torch.no_grad():
        z = d(audio, inputs)
        rw = select_windows(z, torch.tensor([0]), torch.tensor([16.]), torch.tensor([True]),
                            target_beats=16, before_beats=16, after_beats=16)
        pw = select_windows(z, torch.tensor([0]), torch.tensor([16.]), torch.tensor([True]),
                            target_beats=16, before_beats=0, after_beats=0)
        assert z.values.shape == (1, 32, 128) and rw.memory.values.shape == (1, 48, 128)
        assert p(pw).plan_values.shape == (1, 12)
        assert r(rw).count_logits.shape == (1, 16, 97)
    source = str(Path(__file__).resolve().parents[1] / 'src')
    env = dict(os.environ, PYTHONPATH=source)
    code = '''
import sys
from maidata.entry import CONFIG
from maidata.model import build_draft, build_reconstruction, build_planner
build_draft(CONFIG)
build_reconstruction(dict(CONFIG, reconstruction_query_rows=4))
build_planner(CONFIG)
from maidata.draft.chart_data import from_record
from maidata.draft.supervision import predict_auxiliary
assert not any(n.startswith(('maidata.audio', 'maidata.physical', 'maidata.chart_cache',
                            'maidata.preparation', 'maimai_baseline', 'maimai_dataset', 'maichart')) for n in sys.modules)
'''
    result = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
