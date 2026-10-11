"""Original deltas, empty histories, BPM cells, censoring and nonlinear supports."""
import random
import math
import pytest
import torch
from maidata.chart.types import Moment, Note, Route, SlideDuration, HoldDuration
from maidata.chart.windows import TimeTarget
from maidata.time import TimingMap
from maidata.decoder.windows import build_window
from maidata.decoder.batching import collate_windows
from maidata.decoder.supports import time_candidates
from maidata.decoder.probability import interval_log_probabilities, interval_log_mass, TimeDistribution


def sequence(timing):
    return (Moment(0), Moment(timing.delta_ms(0, 10*96)), Note.make('tap', 'A1'),
            Moment(timing.delta_ms(10*96, 13*96)), Note.make('slide', 'A2'), Route(0, 0), Route(1), SlideDuration(960),
            Moment(timing.delta_ms(13*96, 15*96)), Note.make('touch', 'C'),
            Moment(timing.delta_ms(15*96, 20*96)), Note.make('hold', 'A3'), HoldDuration(2880))


def window(tokens, timing, start, *, dropout=0., training=True):
    return build_window(tokens, timing, sequence_start_tick=0, start_tick=start*96, end_tick=(start+16)*96,
        audio_after_beats=4, audio_end_seconds=100, history_beats=4, history_dropout=dropout,
        training=training, rng=random.Random(1))


def test_history_original_delta_empty_crop_dropout_and_shift():
    timing = TimingMap([(0, 120), (14, 240)])
    tokens = sequence(timing)
    w = window(tokens, timing, 16)
    assert w.inputs[0] == Moment(1500.) and w.beats[0] == 13
    assert w.inputs[4] == SlideDuration(960)  # complete chain retained across boundary
    assert w.inputs[5] == Moment(750.) and w.beats[5] == 15
    assert w.inputs[w.boundary_index] == Moment(250.)
    assert w.next_types[w.boundary_index-1] == 0 and not w.supervised[w.boundary_index-1]
    assert w.supervised[w.boundary_index]
    empty = window(tokens, timing, 32)
    assert empty.boundary_index == 0 and empty.inputs[0] == Moment(3000.)
    assert empty.last_real_tick == 20*96
    assert empty.targets[0].censored and empty.targets[0].absolute_tick == 52*96
    dropped = window(tokens, timing, 32, dropout=1.)
    assert dropped.inputs[0] == Moment(0) and dropped.last_real_tick == empty.last_real_tick
    assert dropped.targets == empty.targets  # only context changes
    validation = window(tokens, timing, 32, dropout=1., training=False)
    assert validation.inputs[0] == empty.inputs[0]
    batch, kinds, loss = collate_windows([w, empty], device='cpu')
    from maidata.decoder.sequence import model_window
    expanded=model_window(w,timing)
    assert batch.valid[1].sum()==4  # Empty terminal emits three CONT inputs before closure.
    assert loss.sum()==len(expanded['tokens'])-expanded['last_rows'][w.boundary_index]+4
    assert kinds[0,expanded['last_rows'][w.boundary_index]-1]==0


@pytest.mark.parametrize('next_tick,censored', [(32*96, False), (34*96, False), (36*96, True), (40*96, True)])
def test_terminal_observation_and_exact_window_end(next_tick, censored):
    timing = TimingMap([(0, 120)])
    tokens = (Moment(0), Note.make('tap','A1'), Moment(timing.delta_ms(0, next_tick)), Note.make('tap', 'A2'))
    w = window(tokens, timing, 16)
    target = w.targets[-1]
    assert target.censored == censored and target.closes_window
    assert target.absolute_tick == (36*96 if censored else next_tick)
    assert len(w.inputs) == 1 and w.last_real_tick == 0


def record(penalty=0, basic=()):
    return dict(basic=list(basic), penalty=penalty, physical=penalty)


def test_moment_cap_tail_and_basic_audio_are_separate():
    timing = TimingMap([(0, 120), (1, 60)])
    stage = dict(cursor_tick=94, moment=dict(domain=[95, 98], intervals=[[95, 98, record(4)]], outside_basic='outside'))
    table = time_candidates(stage, timing, kind='MOMENT', window_end=98, cap_tick=101.2, device='cpu')
    assert table.labels == (95,96,97,98,99,100,101,None)
    assert table.valid.all() and not table.penalty[3:].any()  # closing cells bypass real-onset checks
    assert table.lower[1] == pytest.approx(timing.delta_ms(94, 95.5))
    assert table.upper[-2] == table.lower[-1] and math.isinf(table.upper[-1])
    spec = TimeDistribution(1000., 2.5, (), ())
    raw = torch.tensor([[0., .1, -.5], [0., 1., -2.]], requires_grad=True)
    lp = interval_log_probabilities(raw, table.lower, table.upper, table.valid, spec)
    torch.testing.assert_close(lp.exp().sum(), torch.tensor(1.), atol=1e-6, rtol=0)
    mask = table.target_mask(TimeTarget(timing.delta_ms(94,101.2),101.2,True,True),timing,94)
    (-lp[mask].sum()).backward()
    assert torch.isfinite(raw.grad).all() and raw.grad.abs().sum() > 0


@pytest.mark.parametrize('kind', ['HOLD_DURATION', 'SLIDE_DURATION'])
def test_duration_candidates_match_field_representatives_and_disjoint_cdf(kind):
    from maidata.chart_cache.training import duration_candidate_cells
    from maidata.physical.rules import CATEGORIES
    timing = TimingMap([(0, 120), (1, 60)])
    stage = dict(cursor_tick=0, phase='hold' if kind == 'HOLD_DURATION' else 'chain', duration=dict(domain=[2, 18],
        intervals=[[2, 3, record()], [4, 5, record(4)], [6, 11, record()],
                   [12, 13, record(4)], [14, 18, record()]], outside_basic='outside'))
    for _, _, r in stage['duration']['intervals']:
        r.update(deferred=False, penalty_deferred=False)
    table = time_candidates(stage,timing,kind=kind,window_end=96,cap_tick=192,device='cpu')
    reference = duration_candidate_cells(stage, timing)
    assert table.labels == tuple(cell.bucket for cell, _ in reference)
    assert table.ticks == tuple(cell.tick for cell, _ in reference)
    assert len(table.labels) == table.choices == 5
    assert all((tick+2)//4 == label for tick, label in zip(table.ticks, table.labels))
    assert table.penalty.tolist() == [[c in decision.physical for c in CATEGORIES] for _, decision in reference]
    scale = 1000*timing.written_tick_seconds(0) if kind == 'HOLD_DURATION' else 1.
    assert table.lower[0] == pytest.approx(1.5*scale) and table.upper[-1] == pytest.approx(18.5*scale)
    raw = torch.tensor([[0., .08, -2.]], dtype=torch.float64, requires_grad=True)
    spec = TimeDistribution(1000., 2.5, (), ())
    lp = interval_log_probabilities(raw, table.lower, table.upper, table.valid, spec)
    q = lp.exp()[table.penalty[:,2]].sum()
    s = torch.nn.functional.softplus(raw[0,2])*1000+2.5
    cdf = lambda t: torch.sigmoid((t*scale-80)/s)
    expected = ((cdf(6)-cdf(1.5))+(cdf(14)-cdf(10)))/(cdf(18.5)-cdf(1.5))
    torch.testing.assert_close(q.double(),expected,atol=1e-7,rtol=1e-6)
    token = HoldDuration if kind == 'HOLD_DURATION' else SlideDuration
    assert table.target_mask(token(2),timing,0).any()
    assert table.target_mask(token(18),timing,0).any()


def test_duration_target_cannot_alias_outside_basic_support():
    timing = TimingMap([(0, 120)])
    stage = dict(cursor_tick=0, duration=dict(domain=[2, 3],
        intervals=[[2, 3, record(4)]], outside_basic='outside'))
    table = time_candidates(stage,timing,kind='HOLD_DURATION',window_end=96,cap_tick=192,device='cpu')
    # Real GT may be Physical-invalid. Basic range errors must not be rounded away.
    assert table.target_mask(HoldDuration(2),timing,0).any()
    assert table.target_mask(HoldDuration(3),timing,0).any()
    with pytest.raises(ValueError, match='basic'):
        table.target_mask(HoldDuration(5),timing,0)
    stage['duration'] = dict(domain=[2, 10], intervals=[
        [2, 2, record(basic=['audio_range'])], [3, 10, record()]], outside_basic='outside')
    table = time_candidates(stage,timing,kind='HOLD_DURATION',window_end=96,cap_tick=192,device='cpu')
    assert table.target_mask(HoldDuration(3),timing,0).any()
    with pytest.raises(ValueError, match='basic'):
        table.target_mask(HoldDuration(2),timing,0)


def test_logistic_extreme_tails_and_legacy_reference():
    from maimai_baseline.time.moment import interval_log_probabilities as old
    raw = torch.randn(2,5,3,requires_grad=True)
    lower = torch.tensor([[1., 2., 1000.], [1e-4, 1e5, 2e5]],dtype=torch.float64)
    upper = lower+torch.tensor([1e-5,1.,1.],dtype=torch.float64)
    valid = torch.ones_like(lower,dtype=torch.bool)
    spec = TimeDistribution(1000.,2.5,(),())
    new = interval_log_probabilities(raw,lower,upper,valid,spec)
    torch.testing.assert_close(new,old(raw,lower,upper,valid))
    (-new.sum()).backward()
    assert torch.isfinite(raw.grad).all()
    tail = interval_log_mass(raw.detach(),lower,torch.full_like(upper,float('inf')),spec)
    assert torch.isfinite(tail).all()
