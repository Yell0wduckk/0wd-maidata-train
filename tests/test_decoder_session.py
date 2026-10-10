"""Atomic runtime rollback and dependency/resource boundaries."""
from pathlib import Path
import os
import subprocess
import sys
import numpy as np
import pytest
import torch
from maidata.decoder.session import DecodeSession
from maidata.decoder.resources import load_route_distances
from maidata.physical import PhysicalState
from maidata.physical.attribution import AttributionState
from maidata.chart.types import Moment, Note, Route, SlideDuration, HoldDuration
from maidata.time import TimingMap
from test_decoder_supervision import geometry


def test_snapshot_pending_chain_branch_actions_and_last_onset():
    state = AttributionState(PhysicalState(TimingMap([(0,120)]),geometry(),audio_end_seconds=3))
    state.inject_boundary(Moment(0),start_tick=0)
    state.stage_and_accept(Note.make('slide','A1'))
    state.stage_and_accept(Route(0,0))
    session = DecodeSession(state)
    cache, routed, completed = object(), object(), object()
    session.cache, session.routed, session.completed = cache,routed,completed
    session.pending = {'routes':[0], 'selected_type':2}
    checkpoint = session.snapshot()
    state.stage_and_accept(Route(1))
    state.stage_and_accept(SlideDuration(32))
    state.stage_and_accept(Route(4,0))
    session.pending['routes'].append(1)
    session.cache = object()
    session.restore(checkpoint)
    assert session.cache is cache and session.routed is routed and session.completed is completed
    assert session.pending == {'routes':[0], 'selected_type':2}
    assert state.snapshot() == checkpoint.rules and state.physical.grammar.state.routes == (0,)
    with pytest.raises(ValueError,match='incomplete'):
        session.reset_attention()
    session.pending.clear()
    state.stage_and_accept(SlideDuration(32))
    before = state.snapshot()
    session.reset_attention()
    assert session.cache is None and state.snapshot() == before


def test_explicit_geometry_read_only_and_route_order(tmp_path):
    path = tmp_path/'distances.npz'
    matrix = (np.arange(71)[:,None] != np.arange(71)[None,:]).astype(np.float32)
    np.savez(path,route71_ids=np.array([f'B{i:03d}' for i in range(71)]),
             judge_route71_distance=matrix,trajectory_route71_distance=matrix)
    loss = load_route_distances(path)
    assert loss.judge.shape == (71,71) and not list(loss.parameters())
    lp = torch.randn(71,requires_grad=True).log_softmax(-1)
    j,t = loss(lp,70)
    (j+t).backward()
    assert loss.judge.grad is None


def test_real_kv_rollback_discards_abandoned_content_and_matches_fresh_prefix():
    from maidata.decoder.batching import pack_tokens
    from maidata.model import build_decoder
    from test_decoder_network import config, inputs
    torch.manual_seed(19)
    model = build_decoder(config()).eval()
    rules = AttributionState(PhysicalState(TimingMap([(0,120)]),geometry(),audio_end_seconds=3))
    rules.inject_boundary(Moment(0),start_tick=0)
    session = DecodeSession(rules)
    _,audio,z,plan = inputs(model)
    z = type(z)(z.values[:1],z.beats[:1],z.valid[:1])
    memory = model.prepare_memory(model.prepare_audio(audio),torch.tensor([0]),z,plan[:1],torch.tensor([True]))
    boundary = pack_tokens([(Moment(0),)],[(0,)],device='cpu')
    session.routed = model.begin(boundary,memory)
    session.completed = model.complete(session.routed,torch.tensor([[1]]))
    session.cache = session.completed.cache
    session.pending = {'kind':'slide'}
    saved = session.snapshot()
    rules.stage_and_accept(Note.make('slide','A1'))
    slide = pack_tokens([(Note.make('slide','A1'),)],[(0,)],device='cpu')
    session.routed = model.begin(slide,memory,cache=session.cache)
    session.completed = model.complete(session.routed,torch.tensor([[2]]))
    session.cache = session.completed.cache
    rules.stage_and_accept(Route(0,0))
    session.restore(saved)
    assert session.cache is saved.cache and rules.snapshot() == saved.rules
    rules.stage_and_accept(Note.make('tap','A2'))
    tap = pack_tokens([(Note.make('tap','A2'),)],[(0,)],device='cpu')
    actual = model.complete(model.begin(tap,memory,cache=session.cache),torch.tensor([[0]]))
    fresh = model(pack_tokens([(Moment(0),Note.make('tap','A2'))],[(0,0)],device='cpu'),
                  torch.tensor([[1,0]]),memory)
    torch.testing.assert_close(actual.hidden,fresh.hidden[:,1:],atol=2e-6,rtol=2e-5)
    for recovered, reference in zip(actual.cache.layers,fresh.cache.layers):
        torch.testing.assert_close(recovered.key,reference.key,atol=2e-6,rtol=2e-5)
        torch.testing.assert_close(recovered.value,reference.value,atol=2e-6,rtol=2e-5)


@pytest.mark.parametrize('wrapped', [False, True])
@pytest.mark.parametrize('prefix', [
    (Note.make('hold','A1'),),
    (Note.make('slide','A1'),),
    (Note.make('slide','A1'), Route(0,0)),
])
def test_window_cache_reset_checks_rule_pending_content(wrapped, prefix):
    physical = PhysicalState(TimingMap([(0,120)]),geometry(),audio_end_seconds=3)
    rules = AttributionState(physical) if wrapped else physical
    session = DecodeSession(rules)
    session.reset_attention()  # fresh start is valid
    rules.inject_boundary(Moment(0),start_tick=0)
    accept = rules.stage_and_accept if wrapped else rules.accept
    for token in prefix:
        accept(token)
    session.cache, session.routed, session.completed = object(), object(), object()
    before = session.snapshot()
    assert not session.pending  # rule state alone still prohibits switching windows
    with pytest.raises(ValueError,match='incomplete'):
        session.reset_attention()
    assert session.snapshot() == before
    if physical.grammar.state.phase == 'route':
        accept(Route(0,0))
    accept(HoldDuration(32) if prefix[0].kind == 'hold' else SlideDuration(32))
    completed = rules.snapshot()
    session.reset_attention()
    assert session.cache is session.routed is session.completed is None
    assert rules.snapshot() == completed


def test_build_has_no_legacy_or_producer_network_dependency():
    source = str(Path(__file__).resolve().parents[1]/'src')
    script = '''
import sys
from maidata.entry import CONFIG
from maidata.model import build_decoder
model = build_decoder(CONFIG)
assert not any(n.startswith(('maimai_baseline','maimai_dataset','maidata.audio','maidata.draft','maidata.physical','maidata.chart_cache')) for n in sys.modules)
'''
    result = subprocess.run([sys.executable,'-c',script],env=dict(os.environ,PYTHONPATH=source),
                             capture_output=True,text=True)
    assert result.returncode == 0,result.stderr
