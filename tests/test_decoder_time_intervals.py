"""Compact time integration agrees with the dense-cell oracle, including gradients."""
from types import SimpleNamespace
from unittest.mock import patch
import random
import torch
import pytest

from maidata.decoder.probability import TimeDistribution, interval_log_probabilities
from maidata.decoder.losses import physical_categories
from maidata.decoder_training.intervals import compact_time_group, compact_labels, time_intervals
from maidata.decoder.supports import time_candidates
from maidata.chart import Moment, HoldDuration, SlideDuration
from maidata.chart.windows import TimeTarget
from maidata.time import TimingMap
from maidata.decoder_training.compact_windows import convert_window
from maidata.decoder_training.window_cache import tensor_pack, tensor_unpack, select_window
from maidata.decoder_training.supervision import loss, Prepared
from maidata.model import build_decoder, decoder_time_distributions
from test_decoder_network import config


def decision(*, basic=False, penalty=0):
    return dict(basic=['outside'] if basic else [], penalty=penalty, physical=penalty)


def compare_direct(stage, timing, target, kind, *, end, cap):
    dense = time_candidates(stage, timing, kind=kind, window_end=end, cap_tick=cap, device='cpu')
    expected = compact_time_group(dict(lower=dense.lower[None], upper=dense.upper[None],
        allowed=dense.valid[None], penalty=dense.penalty[None],
        target=dense.target_mask(target, timing, stage['cursor_tick'])[None]))
    direct = time_intervals(stage, timing, target, kind=kind, window_end=end, cap_tick=cap)
    assert direct.choices == dense.choices
    for key in ('lower', 'upper', 'penalty', 'target', 'allowed'):
        actual = getattr(direct, 'valid' if key == 'allowed' else key)
        torch.testing.assert_close(actual, expected[key][0], rtol=0, atol=0)


@pytest.mark.parametrize('cursor,end,cap', [(-30, 40, 200), (0, 80, 140.25),
    (0, 1, 1), (0, 1, 1.25), (0, 1, 1.75), (0, .25, .5), (180, 190, 200)])
def test_direct_moment_partitions_match_dense_bpm_holes_and_terminal(cursor, end, cap):
    timing = TimingMap([(0, 131), (1, 177), (2, 83)])
    support = dict(domain=[-29, 149], outside_basic='onset_outside_audio', intervals=[
        [-29, -2, decision(penalty=1)], [0, 29, decision()], [30, 34, decision(basic=True)],
        [35, 42, decision(penalty=3)], [43, 65, decision(penalty=8)],
        [70, 90, decision(penalty=3)], [91, 149, decision()]])
    stage = dict(cursor_tick=cursor, moment=support)
    cells = time_candidates(stage, timing, kind='MOMENT', window_end=end, cap_tick=cap, device='cpu')
    # All valid GT cells plus the censored tail, including exact out-of-window labels.
    for label, valid in zip(cells.labels, cells.valid.tolist()):
        if not valid:
            continue
        tick = cap if label is None else label
        target = TimeTarget(timing.delta_ms(cursor, tick), tick, label is None, tick >= end)
        compare_direct(stage, timing, target, 'MOMENT', end=end, cap=cap)
    if end > cursor+1 and cells.valid[0]:
        compare_direct(stage, timing, Moment(timing.delta_ms(cursor, cursor+1)), 'MOMENT', end=end, cap=cap)


@pytest.mark.parametrize('kind,constructor,minimum', [('HOLD_DURATION', HoldDuration, 2),
                                                     ('SLIDE_DURATION', SlideDuration, 1)])
def test_direct_duration_partitions_match_dense_clamped_representatives(kind, constructor, minimum):
    timing, rng = TimingMap([(0, 131), (1, 177), (2, 83)]), random.Random(817)
    for n in range(40):
        lower, upper = minimum+n%7, minimum+n%7+rng.randrange(50)
        intervals, at = [], lower
        while at <= upper:
            stop = min(upper, at+rng.randrange(6))
            intervals.append([at, stop, decision(basic=rng.random()<.2, penalty=rng.randrange(16))])
            at = stop+1
        stage = dict(cursor_tick=n*7-10, duration=dict(domain=[lower, upper],
                     intervals=intervals, outside_basic='duration_outside_audio'))
        cells = time_candidates(stage, timing, kind=kind, window_end=100, cap_tick=200, device='cpu')
        for tick in range(lower-1, upper+2):
            if tick < minimum:
                continue
            target = constructor(tick)
            try:
                cells.target_mask(target, timing, stage['cursor_tick'])
            except ValueError:
                with pytest.raises(ValueError):
                    time_intervals(stage, timing, target, kind=kind, window_end=100, cap_tick=200)
            else:
                compare_direct(stage, timing, target, kind, end=100, cap=200)


def test_direct_duration_retains_exact_gt_legality_and_representative_legality():
    support = dict(domain=[2, 18], outside_basic='duration_outside_audio', intervals=[
        [2, 2, decision(basic=True)], [3, 7, decision()], [8, 8, decision(basic=True)], [9, 18, decision()]])
    stage, timing = dict(cursor_tick=0, duration=support), TimingMap([(0, 120)])
    # GT 2 is illegal even though its representative 4 is legal; GT 6 is legal
    # but its bucket representative 8 is illegal. Neither may be silently kept.
    for tick in (2, 6):
        with pytest.raises(ValueError):
            time_intervals(stage, timing, HoldDuration(tick), kind='HOLD_DURATION', window_end=96, cap_tick=192)
    compare_direct(stage, timing, HoldDuration(18), 'HOLD_DURATION', end=96, cap=192)


@pytest.mark.parametrize('kind', ['MOMENT', 'HOLD_DURATION', 'SLIDE_DURATION'])
def test_direct_producer_uses_boundaries_instead_of_expanding_huge_domains(kind):
    upper, timing = 10**9, TimingMap([(0, 120), (3, 137)])
    support = dict(domain=[2, upper], intervals=[[2, upper, decision()]], outside_basic='outside')
    stage = dict(cursor_tick=0, moment=support, duration=support)
    target = {'MOMENT': Moment(timing.delta_ms(0, 96)),
              'HOLD_DURATION': HoldDuration(96), 'SLIDE_DURATION': SlideDuration(96)}[kind]
    with patch('maidata.decoder.supports.time_candidates', side_effect=AssertionError('dense candidates')), \
         patch('maidata.time.support.duration_cells', side_effect=AssertionError('dense duration')), \
         patch.object(timing, 'delta_ms', wraps=timing.delta_ms) as endpoints:
        cells = time_intervals(stage, timing, target, kind=kind, window_end=upper-96, cap_tick=upper)
    assert len(cells.lower) == 3
    assert cells.choices == (upper-1 if kind == 'MOMENT' else upper//4)
    assert endpoints.call_count <= 10


def dense_group():
    # Variable cell widths include a BPM transition; holes stay disconnected.
    width = torch.cat((torch.full((800,), 2.7), torch.full((1120,), 5.3))).double()
    edges = torch.cat((torch.tensor([.4]).double(), .4+width.cumsum(0)))
    lower = edges.expand(4, -1).clone()
    upper = torch.cat((edges[1:], torch.tensor([torch.inf]))).expand(4, -1).clone()
    allowed = torch.ones_like(lower, dtype=torch.bool)
    allowed[0, 30:35] = False
    allowed[1, 20:55] = False
    allowed[2, 9:] = False  # padding/short support, one category forbids everything
    penalty = torch.zeros((*allowed.shape, 4), dtype=torch.bool)
    penalty[0, :17, 0] = True
    penalty[0, 70:88, 0] = True
    penalty[0, 13:72, 1] = True
    penalty[1, 55:900, 2] = True
    penalty[1, 700:, 3] = True
    penalty[2, :, 0] = True
    penalty[2, :4, 3] = True
    target = torch.zeros_like(allowed)
    target[0, 60] = True
    target[1, -1] = True  # censored tail
    target[2, 0] = True
    target[3, 100:102] = True  # union of touching GT cells
    return dict(lower=lower, upper=upper, allowed=allowed, penalty=penalty, target=target,
        indices=torch.tensor([[0, i] for i in range(4)]), args=torch.empty(4, 0, dtype=torch.long),
        scale=torch.tensor([1., .5, 1., 1.]), main=torch.ones(4, dtype=torch.bool), has_physical=True)


@pytest.mark.parametrize('extreme', [False, True])
def test_time_mass_and_gradients_match_dense_with_holes_penalties_bpm_and_tail(extreme):
    torch.set_num_threads(1)
    dense = dense_group()
    compact = compact_time_group(dense)
    assert compact['allowed'].shape[1] < 16
    assert torch.equal(compact_time_group(compact)['lower'], compact['lower'])
    for key in ('indices', 'args', 'scale', 'main'):
        assert torch.equal(compact[key], dense[key])
    torch.manual_seed(140)
    raw = torch.randn(4, 5, 3, requires_grad=True)
    if extreme:
        with torch.no_grad():
            raw[..., 1] *= 300
            raw[..., 2] -= 15
    spec = TimeDistribution(1000., 2.5, (), ())
    results, gradients = [], []
    for group in (dense, compact):
        lp = interval_log_probabilities(raw, group['lower'], group['upper'], group['allowed'], spec,
                                       output_dtype=torch.float64)
        main = -torch.logsumexp(lp.masked_fill(~group['target'], -torch.inf), -1)
        physical = physical_categories(lp, group['penalty'])
        result = torch.cat((main[:, None], physical), -1)
        results.append(result)
        gradients.append(torch.autograd.grad(result.sum(), raw)[0])
    torch.testing.assert_close(results[0], results[1], rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(gradients[0], gradients[1], rtol=2e-4, atol=2e-5)
    assert torch.isfinite(gradients[1]).all()


def test_compaction_preserves_decision_counts_and_does_not_bridge_real_gaps():
    group = dense_group()
    # The equal label/penalty cells here have a real hole in probability space.
    group['lower'][3, 80:] += 1
    group['upper'][3, 80:] += 1
    labels = dict(groups={'MOMENT': group}, counts={'main': 4, 'physical': 4, 'judge': 0, 'trajectory': 0}, flags={})
    compact = compact_labels(labels)
    assert compact['counts'] == labels['counts']
    merged = compact['groups']['MOMENT']
    edge = group['upper'][3, 79]
    assert (merged['upper'][3] == edge).any()
    assert (merged['lower'][3] == group['lower'][3, 80]).any()


def test_convert_existing_window_without_opening_chart_or_physical(tmp_path):
    source, output = tmp_path/'source', tmp_path/'output'
    source.mkdir()
    labels = dict(groups={'MOMENT': dense_group()}, counts={'main': 4, 'physical': 4, 'judge': 0, 'trajectory': 0},
                  flags=dict(indices=torch.empty(0, 2, dtype=torch.long), args=torch.empty(0, 2, dtype=torch.long)))
    value = dict(history=2, tokens={'time': torch.arange(7.), 'types': torch.zeros(7, dtype=torch.long)},
                 next_types=torch.zeros(7, dtype=torch.long), labels=labels)
    torch.save(tensor_pack(value), source/'a.pt')
    row = dict(path='a.pt', lengths=[5, 2], chart_id='1:5')
    with patch('maidata.chart_cache.cache.load_chart', side_effect=AssertionError('chart access')), \
         patch('maidata.chart_cache.training.load_supports', side_effect=AssertionError('Physical access')):
        key, metadata, before, after = convert_window((str(source), str(output), 'a', row))
    restored = tensor_unpack(torch.load(output/'a.pt', weights_only=True))
    assert key == 'a' and metadata['lengths'] == [5, 2]
    assert before > after and metadata['bytes'] < (source/'a.pt').stat().st_size
    assert restored['labels']['counts'] == labels['counts']
    for dropped in (False, True):
        actual, expected = select_window(restored, dropped), select_window(value, dropped)
        assert actual['length'] == expected['length'] and actual['offset'] == expected['offset']
        assert torch.equal(actual['tokens']['time'], expected['tokens']['time'])


def test_final_time_diagnostic_reports_gt_mass_not_merged_region_argmax():
    c = config()
    model = build_decoder(c)
    group = compact_time_group(dense_group())
    labels = Prepared({'MOMENT': group}, {'main': 4, 'physical': 4, 'judge': 0, 'trajectory': 0},
                      dict(indices=torch.empty(0, 2, dtype=torch.long), args=torch.empty(0, 2, dtype=torch.long)))
    output = SimpleNamespace(hidden=torch.randn(1, 4, c['decoder_dim']), type_logits=torch.randn(1, 4, 5))
    discrete, time = {}, {}
    terms = loss(model.heads, output, labels, distributions=decoder_time_distributions(c), geometry=None,
                 diagnostics=discrete, time_diagnostics=time)
    assert not discrete and time['MOMENT']['count'] == 4
    assert time['MOMENT']['nll_sum'] == float(terms.main.total.detach())
    assert 0 < time['MOMENT']['probability_sum'] <= 4
