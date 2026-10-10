"""Inference timing reuse preserves the existing dense candidate definition."""
from fractions import Fraction
from unittest.mock import patch

import pytest
import torch

from maidata.decoder.supports import time_candidates
from maidata.inference.time_support import TimeCandidateCache
from maidata.time import TimingMap


def decision(*, basic=False, physical=0, penalty=0):
    return dict(basic=['outside'] if basic else [], physical=physical, penalty=penalty)


def compare(cache, stage, timing, kind, end, cap):
    expected = time_candidates(stage, timing, kind=kind, window_end=end, cap_tick=cap, device='cpu')
    actual = cache.candidates(stage, timing, kind=kind, window_end=end, cap_tick=cap, device='cpu')
    for field in ('lower', 'upper', 'valid', 'raw', 'penalty'):
        torch.testing.assert_close(getattr(actual, field), getattr(expected, field), rtol=0, atol=0)
    assert (actual.labels, actual.ticks, actual.basic_tick_intervals) == (
        expected.labels, expected.ticks, expected.basic_tick_intervals)
    return actual


@pytest.mark.parametrize('cursor,end,cap', [(-30,40,200), (0,80,140.25),
    (0,1,1), (0,1,1.25), (0,1,1.75), (0,.25,.5), (180,190,200)])
def test_moment_cells_exact_for_bpm_changes_holes_and_tail(cursor, end, cap):
    timing = TimingMap([(0,131), (Fraction(1,3),177), (2,83)], first_seconds=-.17)
    support = dict(domain=[-29,149], outside_basic='outside', intervals=[
        [-29,-2,decision(physical=1)], [0,29,decision()], [30,34,decision(basic=True)],
        [35,42,decision(physical=3,penalty=1)], [43,65,decision(physical=8)],
        [70,90,decision(physical=3)], [91,149,decision()]])
    compare(TimeCandidateCache(), dict(cursor_tick=cursor,moment=support), timing, 'MOMENT', end, cap)


@pytest.mark.parametrize('kind,lower', [('HOLD_DURATION',2), ('SLIDE_DURATION',1)])
def test_duration_cells_keep_nonmonotonic_support_and_representatives(kind,lower):
    timing = TimingMap([(0,131),(1,177),(2,83)])
    cache = TimeCandidateCache()
    for upper in (lower, lower+1, 9, 96, 1001):
        support = dict(domain=[lower,upper],outside_basic='outside', intervals=[
            [lower,min(upper,4),decision(physical=2)], [5,min(upper,7),decision(basic=True)],
            [8,min(upper,11),decision()], [12,upper,decision(physical=1,penalty=1)]])
        support['intervals'] = [row for row in support['intervals'] if row[1]>=row[0]]
        compare(cache,dict(cursor_tick=103,duration=support),timing,kind,1536,1920)


def test_coordinate_reuse_does_not_reuse_rule_masks_or_another_song():
    timing, cache = TimingMap([(0,120)]), TimeCandidateCache()
    support = dict(domain=[1,1919],outside_basic='outside',intervals=[[1,1919,decision()]])
    stage = dict(cursor_tick=0,moment=support)
    first = compare(cache,stage,timing,'MOMENT',1536,1920)
    support['intervals'][0][2] = decision(physical=4)
    with patch.object(timing,'second',wraps=timing.second) as coordinates:
        second = cache.candidates(stage,timing,kind='MOMENT',window_end=1536,cap_tick=1920,device='cpu')
        assert coordinates.call_count == 2  # cursor and nominal U only
    assert not first.raw.any() and second.raw[:-385,2].all()
    assert not second.raw[-385:].any()  # window-closing choices stay unmasked
    other = TimingMap([(0,180)])
    third = compare(cache,stage,other,'MOMENT',1536,1920)
    assert not torch.equal(second.lower,third.lower)
    assert len(cache.blocks)<=cache.MAX_BLOCKS
