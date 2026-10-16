"""Integer Z2 coverage, real gather slots, frozen conditions and inference caches."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from maidata.decoder.batching import select_condition_windows
from maidata.decoder.types import Memory
from maidata.decoder_training.conditioning import collate_conditions
from maidata.draft.layout import prepare_selection
from maidata.draft.planner import select_planner_windows
from maidata.draft.spacing import integer_offsets, integer_z_count
from maidata.draft.types import BeatMemory
from maidata.draft.windows import SongCoverage, draft_queries, select_windows
from maidata.inference.cache import (DRAFT_SCHEMA, EncodedAudio, InvalidCacheError,
    JointCache, atomic_save, draft_identity, encoder_recipe, identity_key, memory_payload)
from maidata.inference.conditioning import Conditions, decoder_z_window, empty_plan_geometry
from maidata.entry import CONFIG


def config(stride=2, **updates):
    return dict(target_beats=16, draft_latent_dim=4, draft_z_stride_beats=stride,
        draft_z_anchor_mode='integer', reconstruction_span_beats=4,
        planner_layout='window-single', planner_span_beats=16, planner_stride_beats=None,
        decoder_z_cross_attention=True, decoder_z_before_beats=4, decoder_z_after_beats=4,
        decoder_train_plan_source='gt', rope_base=10000., rope_position_scale=1., **updates)


def memory(coverage, stride=2, *, grad=False):
    beats, valid = draft_queries([coverage], torch.tensor([0]),
        z_stride_beats=stride, anchor_mode='integer')
    values = torch.arange(beats.numel()*4).reshape(1, -1, 4).float().requires_grad_(grad)
    return BeatMemory(values, beats, valid)


def conditions(coverage, z):
    starts = coverage.starts(include_tail=True)
    return Conditions(z, z, torch.zeros(len(starts), 12), torch.ones(len(starts), dtype=torch.bool),
                      starts, coverage, 'window')


def test_integer_offsets_and_coverage_endpoint_keep_phase_and_padding():
    assert integer_offsets(-3, 19, 2) == tuple(range(-2, 19, 2))
    assert integer_offsets(0, 16, 2) == tuple(range(0, 17, 2))
    assert integer_offsets(-8, 24, 2) == tuple(range(-8, 25, 2))
    assert integer_offsets(-3, 19, 1) == tuple(range(-3, 20))
    assert integer_z_count(48, 2) == 25
    with pytest.raises(ValueError, match='stride one or two'):
        integer_offsets(0, 16, 4)
    with pytest.raises(ValueError, match='divisible'):
        integer_z_count(33, 2)
    coverages = [SongCoverage(0, 2, 16, 31.25), SongCoverage(5, 1, 16, 15.25)]
    beats, valid = draft_queries(coverages, torch.tensor([0, 1]), z_stride_beats=2, anchor_mode='integer')
    assert beats[0].tolist() == list(range(0, 49, 2))
    assert beats[1, :17].tolist() == list(range(5, 38, 2))
    assert valid.sum(-1).tolist() == [25, 17] and not valid[1, 17:].any()
    # Empty-tail centers and the terminal endpoint remain real Draft queries.
    assert valid[0, beats[0] >= 32].all() and beats[0, -1] == 48
    old, old_valid = draft_queries(coverages, torch.tensor([0, 1]), z_stride_beats=4)
    assert old[0].tolist() == list(range(2, 48, 4))
    assert old[1, :8].tolist() == list(range(7, 37, 4))
    assert old_valid.sum(-1).tolist() == [12, 8]


@pytest.mark.parametrize('first', [0, 5])
def test_r_and_p_nine_slots_m_eleven_slots_exact_edges_and_gradient(first):
    coverage = SongCoverage(first, 1, 16, 15.25)
    z = memory(coverage, grad=True)
    starts = torch.tensor([float(first), float(first+16)])
    groups, valid = torch.tensor([0, 0]), torch.ones(2, dtype=torch.bool)
    c = config()
    r_layout = prepare_selection(z.beats, z.valid, groups, starts, valid,
        target_beats=16, before_beats=0, after_beats=0, z_stride_beats=2, anchor_mode='integer')
    assert r_layout['beats'].tolist() == [[float(first+s+t) for t in range(0, 17, 2)] for s in (0, 16)]
    assert r_layout['valid'].all()
    p = select_planner_windows(z, groups, starts, valid, c)
    torch.testing.assert_close(p.memory.beats, r_layout['beats'])
    assert p.memory.valid.all() and p.memory.values.shape == (2, 9, 4)
    selected = select_condition_windows(z, groups, starts, valid, c).memory
    expected = [[float(first+s+t) for t in range(-2, 19, 2)] for s in (0, 16)]
    assert selected.beats.tolist() == expected and selected.values.shape == (2, 11, 4)
    assert selected.valid.tolist() == [[False]+[True]*10, [True]*10+[False]]
    assert selected.values[~selected.valid].count_nonzero() == 0
    # Selection must retain the actual Z graph, including shared boundary nodes.
    selected.values.sum().backward()
    assert z.values.grad[0, 7:10].eq(2).all()
    assert z.values.grad[0, :7].eq(1).all() and z.values.grad[0, 10:].eq(1).all()
    source = conditions(coverage, z)
    for index in range(2):
        online = decoder_z_window(source, index, c, 'cpu')
        torch.testing.assert_close(online.beats[0], selected.beats[index])
        torch.testing.assert_close(online.valid[0], selected.valid[index])
        torch.testing.assert_close(online.values[0], selected.values[index])


@pytest.mark.parametrize('stride,count', [(1, 23), (2, 11)])
def test_disabled_z_condition_geometries_match_online_and_frozen(stride, count):
    coverage = SongCoverage(0, 1, 16, 15.25)
    c = config(stride)
    c['decoder_z_cross_attention'] = False
    z, plans, valid = empty_plan_geometry((0, 16), coverage, c, 'cpu')
    assert z.values.shape == (2, count, 4) and not z.valid.any()
    assert plans.shape == (2, 12) and valid.shape == (2,)
    source = conditions(coverage, memory(coverage, stride))
    torch.testing.assert_close(decoder_z_window(source, 1, c, 'cpu').beats, z.beats[1:])
    windows = [dict(identity=(f'chart:{start}', 0), chart_id='chart', start=start) for start in (0, 16)]
    bank = dict(planner_layout='legacy', plan_span_beats=16,
        window_index={f'chart:{start}': i for i, start in enumerate((0, 16))},
        plan_values=torch.zeros(2, 12), valid=torch.ones(2, dtype=torch.bool))
    item = dict(variant='audio', windows=windows, dp={}, gt_plan=bank)
    audio = Memory(torch.zeros(1, 2, 4), torch.tensor([[0., 1.]]), torch.ones(1, 2, dtype=torch.bool))
    frozen = collate_conditions([item], windows, c, audio=audio)
    torch.testing.assert_close(frozen.decoder_z.beats, z.beats)
    assert frozen.decoder_z.values.shape == z.values.shape and not frozen.decoder_z.valid.any()
    assert frozen.plan_beats is None and frozen.plan_valid.all()


def test_z2_inference_cache_length_and_identity_reject_z1_payload(tmp_path):
    coverage = SongCoverage(0, 1, 16, 15.25)
    z = memory(coverage)
    encoded = EncodedAudio(conditions(coverage, z), z, 7.625, 'audio-key', {})
    checkpoint = SimpleNamespace(config=config(), identity=dict(path='checkpoint', size=7, mtime_ns=1))
    identity = draft_identity(checkpoint, encoded.key, difficulty=5, level=14.5,
                              device='cpu', precision='fp32')
    prior = draft_identity(SimpleNamespace(config=config(1), identity=checkpoint.identity), encoded.key,
                           difficulty=5, level=14.5, device='cpu', precision='fp32')
    assert identity_key(identity) != identity_key(prior)
    path = tmp_path/(identity_key(identity)+'.pt')
    payload = dict(schema=DRAFT_SCHEMA, complete=True, identity=identity, key=path.stem,
        z=memory_payload(z), starts=list(encoded.local.starts),
        plans=encoded.local.plans, plan_valid=encoded.local.plan_valid)
    atomic_save(path, payload)
    cache = JointCache(tmp_path, checkpoint, device='cpu', precision='fp32')
    restored = cache._read_draft(path, identity, encoded)
    assert restored.z.values.shape == (1, 17, 4) and restored.z.beats[0, -1] == 32
    assert restored.plan_beats is None
    torch.testing.assert_close(restored.z.values, z.values)
    payload['z'] = memory_payload(memory(coverage, stride=1))
    atomic_save(path, payload)
    with pytest.raises(InvalidCacheError, match='invalid cached memory shape'):
        cache._read_draft(path, identity, encoded)


def test_disabled_qk_options_keep_historical_cache_identities():
    old = {key: value for key, value in CONFIG.items() if key not in
           ('qk_normalization', 'draft_qk_normalization', 'planner_qk_normalization')}
    disabled = dict(old, qk_normalization=False, draft_qk_normalization=False,
                    planner_qk_normalization=False)
    assert encoder_recipe(old) == encoder_recipe(disabled)
    assert 'qk_normalization' not in encoder_recipe(old)
    enabled = dict(disabled, qk_normalization=True, draft_qk_normalization=True,
                   planner_qk_normalization=True)
    assert encoder_recipe(enabled)['qk_normalization'] is True
    def identity(c):
        return draft_identity(SimpleNamespace(config=c, identity=dict(path='same', size=7, mtime_ns=1)),
                              'audio-key', difficulty=5, level=14.5, device='cpu', precision='fp32')
    assert identity(old) == identity(disabled)
    assert identity_key(identity(enabled)) != identity_key(identity(disabled))
