"""Stage-II audio weights stay independent of the fixed stage-I Draft path."""
from types import SimpleNamespace
from unittest.mock import patch
import json

import pytest
import torch

from test_inference import config, producer_config
from test_chart_physical import geometry
from test_pretrain_runtime import item
from maidata.inference.__main__ import main
from maidata.inference import conditioning
from maidata.inference.checkpoint import load_models
from maidata.inference.conditioning import prepare_conditions
from maidata.model import build_model, build_draft, build_planner, build_decoder
from maidata.time import TimingMap


def checkpoints(config, tmp_path, *, schema='maidata.decoder-training.v4'):
    teacher = dict(encoder=build_model(config), draft=build_draft(config), planner=build_planner(config))
    encoder, decoder = build_model(config), build_decoder(config)
    # Emit empty windows, exercising real E/D/P/M without unbounded sampling.
    with torch.no_grad():
        decoder.type_head.weight.zero_()
        decoder.type_head.bias.fill_(-100)
        decoder.type_head.bias[0] = 100
        decoder.heads.times['MOMENT'].weight.zero_()
        decoder.heads.times['MOMENT'].bias.fill_(-100)
        decoder.heads.times['MOMENT'].bias[384] = 100
    producer = dict(schema='maidata.encoder-draft-training.v3', recipe=config,
                    model_state={name+'.'+key: value for name, module in teacher.items()
                                 for key, value in module.state_dict().items()})
    state = torch.nn.ModuleDict(dict(encoder=encoder, decoder=decoder)).state_dict()
    consumer = dict(schema=schema,architecture='maidata.decoder-causal-cont-tick384.v2',recipe=config,model_state=state)
    paths = tmp_path/'ed.ckpt', tmp_path/'m.ckpt'
    torch.save(producer, paths[0])
    torch.save(consumer, paths[1])
    return paths, teacher, encoder, decoder, consumer


@pytest.mark.parametrize('schema', ['maidata.decoder-training.v4'])
def test_inference_loads_correct_stage_specific_audio_weights(config, tmp_path, schema):
    paths, teacher, encoder, decoder, _ = checkpoints(config, tmp_path, schema=schema)
    models = load_models(*paths, device='cpu')
    for name in ('encoder', 'draft', 'planner'):
        assert all(torch.equal(value, getattr(models, name).state_dict()[key])
                   for key, value in teacher[name].state_dict().items())
    assert all(torch.equal(value, models.decoder.state_dict()[key]) for key, value in decoder.state_dict().items())
    if schema.endswith('.v4'):
        assert models.decoder_encoder is not models.encoder
        assert all(torch.equal(value, models.decoder_encoder.state_dict()[key])
                   for key, value in encoder.state_dict().items())
        assert any(not torch.equal(value, models.decoder_encoder.state_dict()[key])
                   for key, value in models.encoder.state_dict().items())
        assert models.provenance['decoder_audio_encoder'] == 'stage_ii'
    else:
        assert models.decoder_encoder is None and models.provenance['decoder_audio_encoder'] == 'stage_i'
    for model in (models.encoder, models.draft, models.planner, models.decoder, models.decoder_encoder):
        if model is not None:
            assert not model.training and not any(p.requires_grad for p in model.parameters())
            assert all(not m.validate_inputs for m in model.modules() if hasattr(m, 'validate_inputs'))


@pytest.mark.parametrize('missing', ['encoder.', 'decoder.'])
def test_joint_checkpoint_cannot_silently_fall_back_to_stage_one_weights(config, tmp_path, missing):
    paths, _, _, _, consumer = checkpoints(config, tmp_path)
    consumer['model_state'] = {key: value for key, value in consumer['model_state'].items()
                               if not key.startswith(missing)}
    torch.save(consumer, paths[1])
    with pytest.raises(RuntimeError, match='Missing key'):
        load_models(*paths, device='cpu')


def test_two_audio_encoders_share_only_inputs_and_layouts_not_draft_features(config):
    encoder, draft, planner, decoder_encoder = [factory(config).eval()
        for factory in (build_model, build_draft, build_planner, build_model)]
    timing, grid = TimingMap([(0, 120)]), item(0, config)['grid']
    options = dict(difficulty=5, level=14.5, device='cpu', encoder_window_batch=1)
    teacher = prepare_conditions(encoder, draft, planner, config, grid, timing, 2.7, **options)
    events = []
    with patch.object(encoder, 'prepare_song', wraps=encoder.prepare_song) as draft_song, \
         patch.object(decoder_encoder, 'prepare_song', wraps=decoder_encoder.prepare_song) as decoder_song, \
         patch.object(encoder, 'encode_windows', wraps=encoder.encode_windows) as draft_windows, \
         patch.object(decoder_encoder, 'encode_windows', wraps=decoder_encoder.encode_windows) as decoder_windows, \
         patch.object(draft, 'forward', wraps=draft.forward) as draft_call, \
         patch.object(planner, 'forward', wraps=planner.forward) as plan_call, \
         patch('maidata.inference.conditioning.window_inputs', wraps=conditioning.window_inputs) as windows_input, \
         patch('maidata.inference.conditioning.prepare_encoder_layout', wraps=conditioning.prepare_encoder_layout) as layouts:
        joint = prepare_conditions(encoder, draft, planner, config, grid, timing, 2.7,
                                   decoder_encoder=decoder_encoder, progress=events.append, **options)
    assert draft_song.call_count == decoder_song.call_count == draft_call.call_count == plan_call.call_count == 1
    assert draft_song.call_args.args[0] is decoder_song.call_args.args[0]
    expected = joint.coverage.audio_windows
    assert draft_windows.call_count == expected
    assert decoder_windows.call_count == windows_input.call_count == layouts.call_count == expected + 1
    for first, second in zip(draft_windows.call_args_list, decoder_windows.call_args_list):
        assert first.args[0] is not second.args[0]  # independent G0 and projected song features
        assert first.args[1] is second.args[1] and first.args[1].layout is not None
    assert torch.equal(draft_call.call_args.args[0].values, teacher.audio.values)
    assert torch.equal(joint.z.values, teacher.z.values) and torch.equal(joint.plans, teacher.plans)
    assert joint.audio_scope == 'window'
    assert joint.audio.values.shape[:2] == (len(joint.starts), 18)
    assert joint.audio.beats[0, 0] == -2 + .5/3
    assert joint.audio.beats[-1, 0] == joint.starts[-1] - 2 + .5/3
    assert joint.starts == teacher.starts and joint.coverage == teacher.coverage
    assert {e['encoder'] for e in events if e['event'] == 'encoder_progress'} == {'draft', 'decoder'}


def test_cli_v3_runs_both_encoders_and_reports_the_audio_source(config, tmp_path, geometry, capsys):
    paths, _, _, _, _ = checkpoints(config, tmp_path)
    arguments = ['--encoder-draft-checkpoint', str(paths[0]), '--decoder-checkpoint', str(paths[1]),
                 '--audio', str(tmp_path/'audio.wav'), '--grid', str(tmp_path/'grid.npz'), '--bpm', '120',
                 '--geometry', 'unused', '--slide-data', 'unused', '--difficulty', '5', '--level', '14.5',
                 '--device', 'cpu', '--precision', 'fp32', '--temperature', '0', '--output', str(tmp_path/'result')]
    with patch('maidata.preparation.cache.load_grid', return_value=item(0, config)['grid']), \
         patch('soundfile.info', return_value=SimpleNamespace(frames=28800, samplerate=24000)), \
         patch('maidata.physical.Geometry.from_resources', return_value=geometry):
        main(arguments)
    captured = capsys.readouterr()
    assert json.loads(captured.out)['windows'] == 2
    assert 'Draft encoder global audio: started' in captured.err
    assert 'Decoder encoder global audio: started' in captured.err
    report = json.loads((tmp_path/'result/generation.json').read_text(encoding='utf-8'))
    assert report['complete'] and report['checkpoints']['decoder_audio_encoder'] == 'stage_ii'
    assert report['checkpoints']['decoder']['schema'] == 'maidata.decoder-training.v4'


@pytest.mark.parametrize('schema',['maidata.decoder-training.v1','maidata.decoder-training.v2'])
def test_old_time_architecture_is_not_silently_loaded(config,tmp_path,schema):
    paths,*_=checkpoints(config,tmp_path,schema=schema)
    with pytest.raises(ValueError,match='v4 causal CONT'):
        load_models(*paths,device='cpu')


def test_unnormalized_cont_architecture_is_not_silently_loaded(config,tmp_path):
    paths,*_=checkpoints(config,tmp_path)
    payload = torch.load(paths[1],map_location='cpu',weights_only=False)
    payload['architecture'] = 'maidata.decoder-causal-cont-tick384.v1'
    torch.save(payload,paths[1])
    with pytest.raises(ValueError,match='v4 causal CONT'):
        load_models(*paths,device='cpu')


@pytest.mark.parametrize('plan_source', ['gt', 'none'])
def test_z_off_cli_uses_only_stage_two_and_masks_absent_plan(config, tmp_path, geometry, plan_source):
    c = dict(config, decoder_z_cross_attention=False, decoder_train_audio_scope='window')
    paths, _, _, _, _ = checkpoints(c, tmp_path)
    plan = tmp_path/'gt-plan.json'
    # Audio duration 1.2 seconds produces one real window and one complete tail.
    coverage = conditioning.song_coverage(TimingMap([(0, 120)]), 1.2,
        first_beat=int(item(0, c)['grid']['first_beat']), target_beats=c['target_beats'])
    values = [0.]*8+[1., 0., 0., 0.]
    plan.write_text(json.dumps(dict(windows=[dict(start_beat=s, plan12=values)
                    for s in coverage.starts(include_tail=True)])))
    args = ['--decoder-checkpoint', str(paths[1]), '--no-z-cross-attention', '--plan-source', plan_source,
            '--audio', 'unused.wav', '--grid', 'unused.npz', '--bpm', '120', '--geometry', 'unused',
            '--slide-data', 'unused', '--difficulty', '5', '--level', '14.5', '--device', 'cpu',
            '--precision', 'fp32', '--encoder-window-batch', '1', '--output', str(tmp_path/'result'), '--no-progress']
    if plan_source == 'gt':
        args += ['--gt-plan', str(plan)]
    with patch('maidata.inference.checkpoint.build_draft', side_effect=AssertionError('must not run D')), \
         patch('maidata.inference.checkpoint.build_planner', side_effect=AssertionError('must not run P')), \
         patch('maidata.preparation.cache.load_grid', return_value=item(0, c)['grid']), \
         patch('soundfile.info', return_value=SimpleNamespace(frames=28800, samplerate=24000)), \
         patch('maidata.physical.Geometry.from_resources', return_value=geometry):
        main(args)
    report = json.loads((tmp_path/'result/generation.json').read_text())
    assert report['complete'] and report['options']['temperature'] == 0
    assert report['checkpoints']['encoder_draft'] is None
    assert report['checkpoints']['inference_plan_source'] == plan_source
    assert report['config']['decoder_z_cross_attention'] is False
    assert all(w['plan_valid'] == (plan_source == 'gt') for w in report['windows'])
    assert all(w['plan12'] == (values if plan_source == 'gt' else [0.]*12) for w in report['windows'])


def test_z_ablation_loads_original_weights_strictly(config, tmp_path):
    paths, _, _, _, _ = checkpoints(config, tmp_path)
    model = load_models(paths[0], paths[1], device='cpu', z_cross_attention=False, plan_source='gt')
    assert model.encoder is model.draft is model.planner is None
    assert model.decoder.condition.z_cross_attention is False
    c = dict(config, decoder_z_cross_attention=False)
    paths, _, _, _, _ = checkpoints(c, tmp_path)
    with pytest.raises(ValueError, match='cannot enable Z'):
        load_models(None, paths[1], device='cpu', z_cross_attention=True, plan_source='gt')
