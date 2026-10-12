import pytest
import torch
from dataclasses import asdict
from unittest.mock import patch

from maidata.inference.sampling import Sampler
from maidata.inference.__main__ import parser
from maidata.inference.options import GenerationOptions
from maidata.chart import Moment, Note, Route, HoldDuration


def test_default_preserves_legacy_rng_and_order():
    scores = torch.tensor([0., 1., -torch.inf, 2.])
    generator = torch.Generator().manual_seed(123)
    remaining = scores.clone()
    expected = []
    while torch.isfinite(remaining).any():
        index = int(torch.multinomial(((remaining-remaining.max()) / .8).softmax(0), 1, generator=generator))
        expected.append(index)
        remaining[index] = -torch.inf
    assert list(Sampler(temperature=.8, seed=123).order(scores)) == expected


def test_random_seed_resolves_once_per_run_and_can_be_replayed():
    state = torch.get_rng_state().clone()
    with patch('maidata.inference.options.secrets.randbelow', side_effect=[17, 23]) as entropy:
        first = GenerationOptions(seed=-1)
        sampler = Sampler(temperature=.8, seed=first.seed)
        second = GenerationOptions(seed=-1)
    assert entropy.call_count == 2
    assert asdict(first)['seed'] == sampler.seed == sampler.generator.initial_seed() == 17
    assert second.seed == 23
    assert torch.equal(state, torch.get_rng_state())
    scores = torch.arange(20).float()
    assert list(sampler.order(scores)) == list(Sampler(temperature=.8, seed=17).order(scores))


def test_direct_sampler_accepts_random_seed_sentinel():
    with patch('maidata.inference.options.secrets.randbelow', return_value=41) as entropy:
        sampler = Sampler(seed=-1)
    entropy.assert_called_once_with(2**31)
    assert sampler.seed == sampler.generator.initial_seed() == 41


@pytest.mark.parametrize('seed', [0, 123, 20261010, 2147483647])
def test_fixed_seed_never_draws_entropy(seed):
    with patch('maidata.inference.options.secrets.randbelow', side_effect=AssertionError('fixed seed')):
        options = GenerationOptions(seed=seed)
        assert options.seed == Sampler(seed=options.seed).generator.initial_seed() == seed


def test_mask_precedes_filters_and_candidates_never_expand():
    sampler = Sampler(temperature=1, top_k=2, top_p=1)
    result = list(sampler.order(torch.tensor([100., 3., 2., 1.]), [False, True, True, True]))
    assert set(result) == {1, 2}
    assert len(result) == 2


def test_nucleus_includes_threshold_crossing_and_no_duplicates():
    scores = torch.tensor([.6, .3, .1]).log()
    assert set(Sampler(temperature=1, top_p=.8).order(scores)) == {0, 1}
    assert list(Sampler(temperature=1, top_p=.1).order(scores)) == [0]
    assert list(Sampler(top_k=20, temperature=0).order(scores)) == [0, 1, 2]
    assert list(Sampler(top_k=1, temperature=0, top_p=.2).order(scores)) == [0]
    assert list(Sampler(top_k=1).order(scores, [False]*3)) == []
    assert list(Sampler(temperature=0, top_p=.8).order(scores)) == [0]


@pytest.mark.parametrize('kwargs', [{'top_k': -1}, {'top_k': 1.5}, {'top_p': 0}, {'top_p': 1.1}, {'top_p': float('nan')}])
def test_invalid_filters(kwargs):
    with pytest.raises(ValueError):
        Sampler(**kwargs)


def test_cli_exposes_filters():
    args = parser().parse_args(['--joint-checkpoint', 'x', '--audio', 'x', '--bpm', '120',
        '--mel-stats', 'x', '--geometry', 'x', '--slide-data', 'x', '--difficulty', '5',
        '--level', '14.5', '--output', 'x', '--top-k', '8', '--top-p', '.9', '--seed', '-1',
        '--repetition-window', '9', '--repetition-penalty', '2.5'])
    assert (args.top_k, args.top_p) == (8, .9)
    assert args.seed == -1
    assert (args.repetition_window, args.repetition_penalty) == (9, 2.5)


@pytest.mark.parametrize('field,identities,history', [
    ('pad', ('A1', 'A2', 'B1'), (Note.make('tap', 'A1'), Note.make('touch', 'A1'))),
    ('route', (0, 1, 2), (Route(0, 1), Route(0, 0))),
])
def test_repetition_subtracts_once_before_top_filters(field, identities, history):
    sampler = Sampler(temperature=0, repetition_window=32, repetition_penalty=1., top_k=1)
    scores = torch.tensor([.8, .4, -.5])
    assert list(sampler.order(scores, identities=identities, history=history, field=field)) == [1]
    assert list(sampler.order(scores, [True, False, False], identities=identities, history=history, field=field)) == [0]
    # Same history is ignored by type, flags, time and track-break heads.
    assert list(sampler.order(scores, history=history)) == [0]
    assert scores.tolist() == pytest.approx([.8, .4, -.5])


def test_repetition_window_counts_all_structure_tokens_and_expires():
    sampler = Sampler(temperature=0, repetition_window=2)
    scores = torch.tensor([.8, .4])
    history = (Note.make('tap', 'A1'), Moment(10.), HoldDuration(12))
    assert list(sampler.order(scores, identities=('A1','A2'), history=history, field='pad'))[0] == 0
    assert list(sampler.order(scores, identities=('A1','A2'), history=history[-2:] + (Note.make('tap','A1'),), field='pad'))[0] == 1


def test_disabled_repetition_preserves_rng_with_prefix_and_filters():
    scores = torch.tensor([-.1, -.2, -.3, -.4])
    history = (Note.make('tap','A1'), Route(0,0))
    sampler = Sampler(temperature=.7, seed=123, top_k=3, top_p=.9, repetition_penalty=0)
    baseline = Sampler(temperature=.7, seed=123, top_k=3, top_p=.9)
    assert list(sampler.order(scores, identities=('A1','A2','A3','A4'), history=history, field='pad')) == list(baseline.order(scores))
    assert torch.equal(sampler.generator.get_state(), baseline.generator.get_state())


@pytest.mark.parametrize('kwargs', [dict(repetition_window=0), dict(repetition_window=1.5),
    dict(repetition_window=True), dict(repetition_penalty=-1), dict(repetition_penalty=float('nan')),
    dict(repetition_penalty=float('inf'))])
def test_invalid_repetition_parameters(kwargs):
    with pytest.raises(ValueError):
        Sampler(**kwargs)
