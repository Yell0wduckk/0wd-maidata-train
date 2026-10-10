"""Compact time integration agrees with the dense-cell oracle, including gradients."""
from types import SimpleNamespace
from unittest.mock import patch
import torch
import pytest

from maidata.decoder.probability import TimeDistribution, interval_log_probabilities
from maidata.decoder.losses import physical_categories
from maidata.decoder_training.intervals import compact_time_group, compact_labels
from maidata.decoder_training.compact_windows import convert_window
from maidata.decoder_training.window_cache import tensor_pack, tensor_unpack, select_window
from maidata.decoder_training.supervision import loss, Prepared
from maidata.model import build_decoder, decoder_time_distributions
from test_decoder_network import config


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
