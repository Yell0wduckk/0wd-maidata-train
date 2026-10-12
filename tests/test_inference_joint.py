"""One joint checkpoint and one real Encoder pass for full-song E/D/P/M."""
from types import SimpleNamespace
from unittest.mock import patch
import json

import pytest
import torch

from test_inference import config, producer_config
from test_chart_physical import geometry
from test_pretrain_runtime import item
from test_inference_materialize import write_audio
from maidata.inference.__main__ import main
from maidata.inference.checkpoint import load_joint_models
from maidata.inference.conditioning import prepare_conditions
from maidata.model import build_model, build_draft, build_reconstruction, build_planner, build_decoder
from maidata.draft.types import BeatMemory
from maidata.draft.windows import assemble_full_p0
from maidata.time import TimingMap


@pytest.fixture
def joint_config(config):
    return dict(config, target_beats=16, target_stride_beats=16,
        audio_before_beats=4, audio_after_beats=4, draft_z_stride_beats=4,
        joint_train_plan_source='gt', joint_train_stop_after_epochs=90,
        joint_train_scheduler_epochs=150, joint_train_learning_rate=4e-4)


def joint_checkpoint(config, tmp_path):
    modules = dict(encoder=build_model(config), draft=build_draft(config),
        reconstruction=build_reconstruction(config), planner=build_planner(config),
        decoder=build_decoder(config))
    # Real Decoder execution closes the empty window with its categorical CONT.
    with torch.no_grad():
        decoder = modules['decoder']
        decoder.type_head.weight.zero_()
        decoder.type_head.bias.fill_(-100)
        decoder.type_head.bias[0] = 100
        decoder.heads.times['MOMENT'].weight.zero_()
        decoder.heads.times['MOMENT'].bias.fill_(-100)
        decoder.heads.times['MOMENT'].bias[384] = 100
    payload = dict(schema='maidata.joint-training.v1',
        architecture='maidata.joint-encoder-draft-decoder.v1', recipe=config,
        model_state=torch.nn.ModuleDict(modules).state_dict(),
        completed_epochs=90, step=123, source=dict(commit='joint-source'))
    path = tmp_path/'joint.ckpt'
    torch.save(payload, path)
    return path, modules, payload


def test_joint_loader_builds_one_encoder_and_strictly_loads_four_modules(joint_config, tmp_path):
    path, original, _ = joint_checkpoint(joint_config, tmp_path)
    with patch('maidata.inference.checkpoint.build_model', wraps=build_model) as build:
        models = load_joint_models(path, device='cpu')
    assert build.call_count == 1
    assert models.encoder is models.decoder_encoder
    for name in ('encoder', 'draft', 'planner', 'decoder'):
        model = getattr(models, name)
        assert all(torch.equal(value, model.state_dict()[key])
                   for key, value in original[name].state_dict().items())
        assert not model.training and not any(p.requires_grad for p in model.parameters())
    assert models.config['draft_z_stride_beats'] == 4
    assert models.config['decoder_train_audio_scope'] == 'window'
    assert models.provenance['joint']['epoch'] == 90
    assert models.provenance['training_plan_source'] == 'gt'
    assert models.provenance['inference_plan_source'] == 'predicted'
    assert models.provenance['decoder_audio_encoder'] == 'shared_joint'


def test_actual_joint_checkpoint_capture_is_a_standalone_inference_input(joint_config, tmp_path):
    from maidata.joint_training import checkpoint, recipe
    from maidata.joint_training.network import JointTrainer
    c = recipe.resolve(joint_config)
    trainer = JointTrainer(c)
    optimizer = recipe.optimizer(trainer.parameters(), c)
    steps = [1]*c['joint_train_scheduler_epochs']
    scheduler = recipe.scheduler(optimizer, steps, c)
    optimizer.step()
    scheduler.step()
    loaders = [SimpleNamespace(generator=torch.Generator().manual_seed(seed), _iterator=None)
               for seed in (10, 11)]
    payload = checkpoint.capture(trainer, optimizer, scheduler, epoch=1, step=1, config=c,
        binding=dict(paths={'accepted_input': 'not-required-for-inference'}), source=dict(commit='capture'),
        steps=steps, loaders=loaders, best=None, metrics={})
    path = tmp_path/'captured.ckpt'
    torch.save(payload, path)
    loaded = load_joint_models(path, device='cpu')
    assert loaded.encoder is loaded.decoder_encoder
    assert loaded.config['draft_z_stride_beats'] == 4
    for name in ('encoder', 'draft', 'planner', 'decoder'):
        for key, value in getattr(trainer, name).state_dict().items():
            torch.testing.assert_close(value, getattr(loaded, name).state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize('prefix', ['encoder.', 'draft.', 'planner.', 'decoder.'])
def test_joint_loader_never_substitutes_missing_weights(joint_config, tmp_path, prefix):
    path, _, payload = joint_checkpoint(joint_config, tmp_path)
    payload['model_state'] = {key: value for key, value in payload['model_state'].items()
                              if not key.startswith(prefix)}
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match='Missing key'):
        load_joint_models(path, device='cpu')


@pytest.mark.parametrize('key,value', [('schema', 'maidata.decoder-training.v4'),
                                     ('architecture', 'other-joint-architecture')])
def test_joint_loader_rejects_other_checkpoint_families(joint_config, tmp_path, key, value):
    path, _, payload = joint_checkpoint(joint_config, tmp_path)
    payload[key] = value
    torch.save(payload, path)
    with patch('maidata.inference.checkpoint.build_model', side_effect=AssertionError('reject before building')):
        with pytest.raises(ValueError, match='v1 joint'):
            load_joint_models(path, device='cpu')


def test_joint_loader_only_omits_reconstruction_and_keeps_strict_state(joint_config, tmp_path):
    path, _, payload = joint_checkpoint(joint_config, tmp_path)
    payload['model_state']['unknown.weight'] = torch.ones(1)
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match='Unexpected key'):
        load_joint_models(path, device='cpu')
    payload['model_state'].pop('unknown.weight')
    key = next(key for key in payload['model_state'] if key.startswith('encoder.'))
    payload['model_state'][key] = payload['model_state'][key].flatten()[:1]
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match='size mismatch'):
        load_joint_models(path, device='cpu')


@pytest.mark.parametrize('batch_size', [1, 8])
def test_shared_encoder_reuses_exact_local_p0_including_tail(joint_config, tmp_path, batch_size):
    path, _, _ = joint_checkpoint(joint_config, tmp_path)
    models = load_joint_models(path, device='cpu')
    grid, timing = item(0, joint_config)['grid'], TimingMap([(0, 120)])
    with patch.object(models.encoder, 'prepare_song', wraps=models.encoder.prepare_song) as song, \
         patch.object(models.encoder, 'encode_windows', wraps=models.encoder.encode_windows) as local, \
         patch.object(models.draft, 'forward', wraps=models.draft.forward) as draft, \
         patch.object(models.planner, 'forward', wraps=models.planner.forward) as planner:
        conditions = prepare_conditions(models.encoder, models.draft, models.planner,
            models.config, grid, timing, 1.2, difficulty=5, level=14.5, device='cpu',
            encoder_window_batch=batch_size, decoder_encoder=models.decoder_encoder,
            teacher_config=models.teacher_config)
    assert song.call_count == draft.call_count == planner.call_count == 1
    assert local.call_count == (len(conditions.starts)+batch_size-1)//batch_size
    encoded_starts = torch.cat([call.args[1].target_start_beats for call in local.call_args_list])
    assert encoded_starts.tolist() == list(conditions.starts) == [0, 16]
    assert conditions.audio_scope == 'window'
    assert conditions.audio.values.shape[:2] == (2, 24*joint_config['fpn_rates'][0])
    assert conditions.audio.beats[0, 0] == -4+.5/joint_config['fpn_rates'][0]
    assert conditions.audio.beats[-1, 0] == 12+.5/joint_config['fpn_rates'][0]
    centers = BeatMemory(*(getattr(conditions.audio, name)[:-1] for name in ('values', 'beats', 'valid')))
    expected = assemble_full_p0(centers, torch.tensor([0.]), torch.tensor([0]),
        [conditions.coverage], p0_rate=joint_config['fpn_rates'][0])
    actual = draft.call_args.args[0]
    for name in ('values', 'beats', 'valid'):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)
    assert actual.values.shape[1] == 16*joint_config['fpn_rates'][0]
    assert conditions.z.beats.tolist() == [[2., 6., 10., 14., 18., 22., 26., 30.]]
    assert conditions.z.valid.all() and conditions.plan_valid.all()
    assert conditions.plans.shape == (2, 12)


def arguments(path, tmp_path, precision='fp32'):
    return ['--joint-checkpoint', str(path), '--audio', str(tmp_path/'audio.wav'), '--grid', 'unused.npz',
        '--bpm', '120', '--geometry', 'unused', '--slide-data', 'unused', '--difficulty', '5',
        '--level', '14.5', '--device', 'cpu', '--precision', precision, '--temperature', '0',
        '--output', str(tmp_path/'result'), '--no-progress']


@pytest.mark.parametrize('precision', ['fp32', 'bf16'])
def test_single_checkpoint_cli_generates_with_predicted_plan(joint_config, tmp_path, geometry, precision):
    path, _, _ = joint_checkpoint(joint_config, tmp_path)
    write_audio(tmp_path/'audio.wav')
    with patch('maidata.preparation.cache.load_grid', return_value=item(0, joint_config)['grid']), \
         patch('soundfile.info', return_value=SimpleNamespace(frames=28800, samplerate=24000)), \
         patch('maidata.physical.Geometry.from_resources', return_value=geometry):
        main(arguments(path, tmp_path, precision))
    report = json.loads((tmp_path/'result/generation.json').read_text(encoding='utf-8'))
    assert report['complete'] and report['physical_bypass_count'] == 0
    assert report['checkpoints']['decoder_audio_encoder'] == 'shared_joint'
    assert report['checkpoints']['training_plan_source'] == 'gt'
    assert report['checkpoints']['inference_plan_source'] == 'predicted'
    assert len(report['windows']) == 2 and all(w['plan_valid'] for w in report['windows'])
    assert report['config']['draft_z_stride_beats'] == 4
    assert all('trace' not in window for window in report['windows'])


@pytest.mark.parametrize('extra', [['--decoder-checkpoint', 'unused.ckpt'],
                                  ['--encoder-draft-checkpoint', 'unused.ckpt']])
def test_joint_cli_rejects_mixed_checkpoint_inputs(tmp_path, extra):
    with pytest.raises(SystemExit):
        main([*arguments(tmp_path/'joint.ckpt', tmp_path), *extra])


def test_joint_z_ablation_loads_original_projection_before_disabling(joint_config, tmp_path):
    path, _, _ = joint_checkpoint(joint_config, tmp_path)
    model = load_joint_models(path, device='cpu', z_cross_attention=False)
    assert model.encoder is model.decoder_encoder
    assert model.decoder.condition.z_cross_attention is False
    path, _, _ = joint_checkpoint(dict(joint_config, decoder_z_cross_attention=False), tmp_path)
    with pytest.raises(ValueError, match='cannot enable Z'):
        load_joint_models(path, device='cpu', z_cross_attention=True)
