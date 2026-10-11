"""Coarse Z blocks retain per-beat labels, direct owners and fixed data plans."""
from copy import deepcopy
from dataclasses import replace
import json
from unittest.mock import patch

import pytest
import torch

from test_pretrain_runtime import config, item, make_plan
from test_split_drafter_consumers import paired
from maidata.chart.types import Note
from maidata.draft.chart_data import ChartData, OnsetGroup
from maidata.draft.plan_targets import MusicPlanTargets
from maidata.draft.targets import reconstruction_target, collate_reconstruction
from maidata.draft.types import BeatMemory
from maidata.draft.windows import song_coverage, draft_queries, select_windows, assemble_full_p0
from maidata.draft.schema import QUERY_ROLES
from maidata.model import build_reconstruction, build_planner, ENCODER_KEYS
from maidata.training.data import AudioPlan, collate_audio
from maidata.training.network import Pretrainer
from maidata.training.pretrain import make_optimizer, make_scheduler
from maidata.training.runner import update
from maidata.training import checkpoint
from maidata.training.frozen_p0 import FrozenP0, ENCODER_SCHEMA
from maidata.decoder_training.p0_cache import save_bank
from maidata.decoder_training.dp_cache import CachedP0Dataset, save_outputs
from maidata.decoder_training.cache_schema import P0_SCHEMA, DP_SCHEMA, Z_REPRESENTATION
from maidata.decoder_training.frozen_cache import FrozenCache, condition_config
from maidata.decoder_training.conditioning import collate_conditions
from maidata.inference.checkpoint import load_models
from maidata.inference.conditioning import prepare_conditions
from maidata.time import TimingMap


def source(config):
    row = item(0, config)
    coverage = song_coverage(TimingMap([(0, 120)]), 1.2, first_beat=0, target_beats=16)
    chart = ChartData(tuple(OnsetGroup(t, (Note.make('tap', 'A1'),))
                           for t in (0, 95, 96, 384, 1535)), (), TimingMap([(0, 120)]), 2)
    planner = MusicPlanTargets(chart)
    row['coverage'] = coverage
    row['windows'] = [(f'w{start}', '0:5', start,
        reconstruction_target(chart, start_tick=start*96, target_beats=16),
        planner.window(start*96, (start+16)*96)) for start in (0, 16)]
    return row


@pytest.mark.parametrize('stride', [1, 2, 4, 8, 16])
@pytest.mark.parametrize('precision', ['fp32', 'bf16'])
def test_real_pretrainer_update_keeps_labels_and_tail(config, stride, precision):
    c = dict(config, target_beats=16, draft_z_stride_beats=stride, pretrain_precision=precision)
    data = source(c)
    batch = collate_audio([data], config=c, packed=True).to('cpu')
    reference = collate_audio([data], config=dict(c, draft_z_stride_beats=1))
    assert batch.window_ids == reference.window_ids
    torch.testing.assert_close(batch.reconstruction.moment_tick, reference.reconstruction.moment_tick)
    torch.testing.assert_close(batch.reconstruction.moment_counts, reference.reconstruction.moment_counts)
    assert batch.draft_inputs.query_beats.shape == (1, 32//stride)
    assert batch.layouts['planner']['valid'].sum(-1).tolist() == [16//stride]*2
    model = Pretrainer(c)
    opt = make_optimizer(model.parameters(), c)
    sched = make_scheduler(opt, [1]*8, c)
    metrics, audit = update(model, [batch], opt, sched, c, audit=True)
    assert all(value > 0 for value in audit['gradients'].values())
    assert metrics.sums['empty'][1] == 32
    assert metrics.sums['moment96'][1] == 5
    assert metrics.sums['positive_count'][1] == 4


@pytest.mark.parametrize('stride', [1, 2, 4, 8, 16])
def test_negative_origin_block_owners_positions_and_planner(config, stride):
    torch.manual_seed(213)
    c = dict(config, target_beats=16, draft_z_stride_beats=stride)
    coverage = song_coverage(TimingMap([(0, 120)]), 10, first_beat=-3, target_beats=16)
    beats, valid = draft_queries([coverage, replace(coverage, audio_windows=1)],
                                torch.tensor([0, 1]), z_stride_beats=stride)
    assert valid.sum(-1).tolist() == [48//stride, 32//stride]
    assert beats[0, 0] == -3+stride/2
    values = torch.randn(2, beats.shape[1], 24, requires_grad=True)
    z = BeatMemory(values, beats, valid)
    windows = select_windows(z, torch.tensor([0]), torch.tensor([13.]), torch.tensor([True]),
        target_beats=16, before_beats=0, after_beats=0, z_stride_beats=stride)
    chart = ChartData((OnsetGroup(14*96+95, (Note.make('tap', 'A1'),)),), (), TimingMap([(0, 120)]), 20)
    targets = collate_reconstruction([reconstruction_target(chart, start_tick=13*96, target_beats=16)],
                                    target_beats=16, device='cpu')
    r = build_reconstruction(c).eval()
    out = r(windows, targets.queries)
    assert out.query_valid.equal(targets.queries.valid) and out.count_valid.all()
    moment = (targets.queries.roles[0] == QUERY_ROLES.index('MOMENT')).nonzero()[0, 0]
    assert targets.moment_tick[0, moment] == 95
    grad = torch.autograd.grad(out.moment_logits[0, moment, 0], values, retain_graph=True)[0]
    owner = 16//stride + 1//stride
    assert grad.abs().sum(-1).nonzero().tolist() == [[0, owner]]
    if stride > 1:
        # The same block is decoded differently at different known beat offsets.
        assert not torch.equal(out.empty_logits[0, 0], out.empty_logits[0, 1])
        assert not torch.equal(out.positive_count_logits[0, 0], out.positive_count_logits[0, 1])
    planner = build_planner(c)(windows)
    grad = torch.autograd.grad(planner.continuous_logits.sum(), values)[0].abs().sum(-1)
    assert grad.nonzero().tolist() == [[0, i] for i in range(16//stride, 32//stride)]
    expanded = select_windows(z, torch.tensor([0]), torch.tensor([13.]), torch.tensor([True]),
        target_beats=16, before_beats=4, after_beats=4, z_stride_beats=stride).memory
    assert len(expanded.beats[0].unique()) == expanded.beats.shape[1]
    centers = beats[0]
    overlaps = (centers+stride/2 > 9) & (centers-stride/2 < 33)
    torch.testing.assert_close(expanded.beats[expanded.valid], centers[overlaps])


def test_cached_training_export_and_decoder_condition_roundtrip(config, tmp_path):
    c = dict(config, target_beats=16, draft_z_stride_beats=8)
    data = source(c)
    online = collate_audio([data], config=c)
    original = Pretrainer(c).eval()
    with torch.no_grad():
        local = original.encoder(online.songs, online.audio_windows).P0
        bank = assemble_full_p0(local, online.audio_windows.target_start_beats,
            online.audio_windows.window_to_song, online.coverages, p0_rate=3, layout=online.layouts['p0'])
        _, (_, expected_plan, expected_z) = original(online)
    row = save_bank(tmp_path/'banks/0.pt', bank, 0, data['coverage'], variant='0', producer_sha256='p', p0_rate=3)
    row['path'] = 'banks/0.pt'
    encoder_config = json.loads(json.dumps({k:c[k] for k in ENCODER_KEYS}))
    torch.save(dict(schema=ENCODER_SCHEMA, producer_sha256='p', encoder_config=encoder_config,
                    model_state=original.encoder.state_dict()), tmp_path/'encoder.pt')
    manifest = dict(schema=P0_SCHEMA, complete=True, producer={'sha256':'p'},
        encoder=dict(path='encoder.pt', sha256='e'), encoder_config=encoder_config,
        precision='fp32', p0_rate=3, groups={'0': row})
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    frozen = FrozenP0(tmp_path)
    cached_c = dict(c, pretrain_audio_mode='cached', pretrain_p0_cache=str(tmp_path))
    cached = Pretrainer(cached_c).eval()
    cached.load_state_dict(original.state_dict())
    p0, _ = frozen.read('0')
    cached_batch = collate_audio([dict(data, p0=p0, grid=None)], config=cached_c)
    with patch.object(cached.encoder, 'forward', side_effect=AssertionError('frozen E execution')):
        _, (_, actual_plan, actual_z) = cached(cached_batch)
        torch.testing.assert_close(actual_z.values, expected_z.values, rtol=0, atol=0)
        torch.testing.assert_close(actual_plan.plan_values, expected_plan.plan_values, rtol=0, atol=0)
        opt = make_optimizer(cached.parameters(), cached_c)
        update(cached, [cached_batch], opt, make_scheduler(opt, [1]*8, cached_c), cached_c)
    assert all(p.grad is None for p in cached.encoder.parameters())
    # The actual offline dataset also takes spacing from the teacher configuration.
    records = [dict(audio_variant='0', chart_id='0:5', window_id=w[0], start_beat=w[2]) for w in data['windows']]
    (tmp_path/'windows.jsonl').write_text('\n'.join(json.dumps(w) for w in records))
    plan = dict(audio_groups=[dict(audio_variant='0', chart_ids=['0:5'], encoder_starts_beats=[0], draft_end_beat=32)],
        charts=[dict(chart_id='0:5', chart_cache_path='chart')], policy=dict(split='train'), files=dict(windows='windows.jsonl'))
    export_c = dict(c, decoder_train_train_plan=str(tmp_path), decoder_train_chart_cache=str(tmp_path),
                    decoder_train_source_root=str(tmp_path))
    chart = dict(source=dict(chart_id='0:5', audio_variant='0', song_id='song', difficulty='5', level_text='14.5'))
    with patch('maidata.decoder_training.dp_cache.load_chart', return_value=chart):
        batch = CachedP0Dataset(export_c, tmp_path, manifest, [plan])[0].to('cpu')
    torch.testing.assert_close(batch['inputs'].query_beats, online.draft_inputs.query_beats)
    with torch.no_grad():
        z = original.draft(batch['bank'], batch['inputs'])
        p = original.planner(select_windows(z, batch['groups'], batch['starts'], batch['valid'],
            target_beats=16, before_beats=0, after_beats=0, layout=batch['layout'], z_stride_beats=8))
    torch.testing.assert_close(z.values, expected_z.values, rtol=0, atol=0)
    root = tmp_path/'dp'
    row = save_outputs(root/'banks/0.pt', batch, z, p, 'teacher', z_stride_beats=8)
    dp_manifest = dict(schema=DP_SCHEMA, z_representation=Z_REPRESENTATION, z_width=24, z_stride_beats=8,
        producer={'recipe': {'draft_z_stride_beats':8}}, groups={'0':row})
    (root/'manifest.json').write_text(json.dumps(dp_manifest))
    m_config = dict(c, draft_z_stride_beats=1, decoder_train_dp_cache=str(root), decoder_train_plan_source='predicted')
    cache = FrozenCache([], m_config, {})
    m_config = condition_config(m_config, cache)
    assert m_config['draft_z_stride_beats'] == 8
    dp = cache.read('0')
    windows = [dict(chart_id='0:5', start=w[2], identity=(w[0], 0)) for w in data['windows']]
    cond = collate_conditions([dict(variant='0', dp=dp)], windows+windows[:1], m_config, audio=bank)
    assert cond.occurrence_mapping.tolist() == [0, 1, 0]
    assert cond.decoder_z.valid.sum(-1).tolist() == [3, 3]
    assert cond.decoder_z.values.shape[1] == 4  # target two plus two context blocks
    assert torch.equal(cond.decoder_z.values[0, 1:3], z.values[0, :2])
    from test_decoder_network import config as decoder_config, inputs as decoder_inputs
    from maidata.model import build_decoder
    decoder = build_decoder(dict(decoder_config(), draft_latent_dim=24))
    tokens, *_ = decoder_inputs(decoder)
    memory = decoder.prepare_memory(decoder.prepare_audio(bank), cond.audio_mapping,
                                    cond.decoder_z, cond.plan, cond.plan_valid)
    out = decoder(tokens, torch.ones_like(tokens.types), memory)
    out.type_logits.square().mean().backward()
    assert torch.isfinite(out.hidden).all()
    assert decoder.condition.z.weight.grad.abs().sum() > 0
    assert not cond.decoder_z.values.requires_grad
    dp['z_stride_beats'] = 4
    torch.save(dp, root/'banks/0.pt')
    with pytest.raises(ValueError, match='representation/width differs'):
        cache.read('0')


def test_checkpoint_spacing_restore_and_inference(config, tmp_path):
    c = dict(config, target_beats=16, draft_z_stride_beats=4)
    batch = collate_audio([source(c)], config=c)
    model = Pretrainer(c)
    opt = make_optimizer(model.parameters(), c)
    sched = make_scheduler(opt, [1]*8, c)
    update(model, [batch], opt, sched, c)
    state = deepcopy(checkpoint.capture(model, opt, sched, completed_epochs=1, step=1, config=c,
        binding={}, source={}, steps=[1]*8, loaders=[], best=0., metrics={}))
    assert state['recipe']['draft_z_stride_beats'] == 4
    restored = Pretrainer(c)
    other_opt = make_optimizer(restored.parameters(), c)
    other_sched = make_scheduler(other_opt, [1]*8, c)
    checkpoint.restore(state, restored, other_opt, other_sched, config=c, binding={}, source={}, steps=[1]*8, loaders=[])
    update(model, [batch], opt, sched, c)
    update(restored, [batch], other_opt, other_sched, c)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match='recipe or scheduler steps changed'):
        checkpoint.restore(state, restored, other_opt, other_sched, config=dict(c, draft_z_stride_beats=8),
            binding={}, source={}, steps=[1]*8, loaders=[])
    paths, teacher, consumer, _, _ = paired(c, tmp_path)
    models = load_models(*paths, device='cpu')
    conditions = prepare_conditions(models.encoder, models.draft, models.planner, models.config,
        source(c)['grid'], TimingMap([(0,120)]), 1.2, difficulty=5, level=14.5, device='cpu',
        decoder_encoder=models.decoder_encoder, teacher_config=models.teacher_config)
    assert conditions.z.values.shape == (1, 8, 30) and conditions.plan_valid.all()
    saved = torch.load(paths[1], weights_only=False)
    saved['recipe']['draft_z_stride_beats'] = 2
    torch.save(saved, paths[1])
    with pytest.raises(ValueError, match='Z spacing differs'):
        load_models(*paths, device='cpu')


def test_cli_values_and_unchanged_batch_boundaries(config, tmp_path, capsys):
    from maidata.training.__main__ import main
    from maidata.draft.spacing import z_stride
    main(['--z-stride-beats', '8', '--show-config'])
    assert json.loads(capsys.readouterr().out)['draft_z_stride_beats'] == 8
    for bad in (0, 3, 6, 32, True, 1.5):
        with pytest.raises(ValueError, match='one of'):
            z_stride(bad)
    with pytest.raises(SystemExit):
        main(['--z-stride-beats', '3'])
    make_plan(tmp_path, config)
    plan = AudioPlan(tmp_path, config)
    for stride in (1, 2, 4, 8, 16):
        for epoch in (1, 20, 50):
            assert plan.batches(epoch, dict(config, draft_z_stride_beats=stride)) == plan.batches(epoch, config)


def test_coarse_queries_do_not_read_target_tick_future_or_counts(config):
    from test_draft_local_queries import chart, labels, indices
    c = dict(config, target_beats=16, draft_z_stride_beats=16)
    z = BeatMemory(torch.randn(1, 2, 24), torch.tensor([[8., 24.]]), torch.ones(1, 2, dtype=torch.bool))
    windows = select_windows(z, torch.tensor([0]), torch.tensor([0.]), torch.tensor([True]),
        target_beats=16, before_beats=0, after_beats=0, z_stride_beats=16)
    a = labels(chart([(97, [Note.make('tap', 'A1')])]))
    b = labels(chart([(191, [Note.make('tap', 'A8')]), (300, [Note.make('touch', 'C')]*3)]))
    head = build_reconstruction(c).eval()
    x, y = head(windows, a.queries), head(windows, b.queries)
    # Different query counts select different CPU GEMM shapes (observed ~1e-7
    # rounding); prefix inputs and the dependency graph remain identical.
    torch.testing.assert_close(x.moment_logits[0, indices(a, 'MOMENT')[0]],
                               y.moment_logits[0, indices(b, 'MOMENT')[0]])
    torch.testing.assert_close(x.empty_logits, y.empty_logits, rtol=0, atol=0)
    torch.testing.assert_close(x.positive_count_logits, y.positive_count_logits, rtol=0, atol=0)


def test_legacy_per_beat_teacher_still_exports(config, tmp_path):
    from maidata.decoder_training.producer import load_export_producer
    paths, teacher, consumer, payload, binding = paired(config, tmp_path)
    payload['recipe'].pop('draft_z_stride_beats')
    torch.save(payload, paths[0])
    _, _, exported = load_export_producer(dict(consumer, decoder_train_checkpoint=str(paths[0])), binding)
    assert exported['draft_z_stride_beats'] == 1
    assert load_models(*paths, device='cpu').teacher_config['draft_z_stride_beats'] == 1
