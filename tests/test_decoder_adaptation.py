"""Frozen E/P0/Z -> warm M, including export, gradients and complete-epoch restore."""
from copy import deepcopy
from dataclasses import replace
from functools import partial
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader

from maidata.model import ENCODER_KEYS, build_model, build_decoder
from maidata.training.frozen_p0 import FrozenP0, ENCODER_SCHEMA
from maidata.decoder_training import adaptation, checkpoint, recipe, runner
from maidata.decoder_training.network import DecoderTrainer
from maidata.decoder_training.data import collate_audio, DecoderDataset
from maidata.decoder_training.frozen_cache import FrozenCache, condition_config
from test_decoder_training import config, encoder_config, distances
from test_decoder_frozen_inputs import assets
from test_decoder_joint_gradients import assert_trees_equal, direct_update


@pytest.fixture
def adapted(assets):
    c, plans, binding, items, roots = assets
    with torch.random.fork_rng():
        torch.manual_seed(614)  # Same public-P0 producer as the synthetic cache.
        encoder = build_model(c)
    old = dict(c, decoder_z_cross_attention=False, draft_latent_dim=7)
    decoder = build_decoder(old)
    model = torch.nn.ModuleDict(dict(encoder=encoder, decoder=decoder))
    payload = dict(schema=checkpoint.SCHEMA, architecture=checkpoint.ARCHITECTURE,
                   completed_epochs=45, step=900, source={'commit':'original-M'},
                   recipe=checkpoint.recipe(old), model_state=deepcopy(model.state_dict()))
    path = roots['p0'].parent/'epoch-045.ckpt'
    torch.save(payload, path)
    manifest_path = roots['p0']/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    encoder_config = json.loads(json.dumps({k:c[k] for k in ENCODER_KEYS}))
    torch.save(dict(schema=ENCODER_SCHEMA, producer_sha256='fixed-producer', encoder_config=encoder_config,
                    model_state=encoder.state_dict()), roots['p0']/'encoder.pt')
    manifest.update(encoder=dict(path='encoder.pt',sha256='encoder-hash'),encoder_config=encoder_config)
    manifest_path.write_text(json.dumps(manifest))
    p0 = FrozenP0(roots['p0'])
    dp_path = roots['dp']/'manifest.json'
    dp = json.loads(dp_path.read_text())
    dp['producer']['audio_producer'] = p0.identity
    dp_path.write_text(json.dumps(dp))
    c.update(decoder_train_audio_mode='cached',decoder_train_audio_scope='full_song',
             decoder_z_cross_attention=True, decoder_train_init_checkpoint=str(path),
             decoder_train_plan_source='gt')
    cache = FrozenCache(plans,c,binding)
    c, saved = adaptation.prepare(c)
    c = condition_config(c,cache)
    selected = []
    for i,item in enumerate(items):
        selected.append(dict(item, grid=None, p0=p0.read(item['variant'])[0],
            dp=cache.read(item['variant']),
            gt_plan=dict(item['dp']['planner'],window_index=item['dp']['window_index']),
            windows=[dict(w,sequence_index=2*j+i) for j,w in enumerate(item['windows'])]))
    return c, saved, selected, cache, p0


@pytest.mark.parametrize('bf16', [False, True])
def test_only_m_updates_and_z_receives_gradient_from_cached_conditions(adapted, bf16):
    c, saved, items, cache, p0 = adapted
    adaptation.bind_conditions(cache,p0)
    batch = collate_audio(items,config=c,packed=True)
    assert batch.online_audio is None
    raw = batch.to('cpu')
    assert raw.conditions.audio.values.shape[-1] == c['d_model']
    expected_order = sorted([w for item in items for w in item['windows']],key=lambda w:w['sequence_index'])
    assert [w['identity'] for w in raw.windows] == [w['identity'] for w in expected_order]
    if bf16:
        cond = raw.conditions
        raw.conditions = replace(cond,audio=replace(cond.audio,values=cond.audio.values.bfloat16()),
            decoder_z=replace(cond.decoder_z,values=cond.decoder_z.values.bfloat16()),plan=cond.plan.bfloat16())
    model = DecoderTrainer(c)
    adaptation.initialize(model,saved,c['decoder_train_init_checkpoint'])
    assert model.decoder.condition.z_cross_attention
    reset = {'decoder.'+key for key in model.initialization['reset_decoder_parameters']}
    assert reset == {'decoder.condition.z.weight','decoder.condition.z.bias'}
    for key,value in model.state_dict().items():
        if key not in reset:
            assert torch.equal(value,saved['model_state'][key]), key
    model.train()
    assert not model.encoder.training and all(not p.requires_grad for p in model.encoder.parameters())
    opt = recipe.optimizer(model.parameters(),c)
    assert {id(p) for g in opt.param_groups for p in g['params']} == {id(p) for p in model.decoder.parameters()}
    reference = deepcopy(model)
    reference_opt = recipe.optimizer(reference.parameters(),c)
    with patch.object(model.encoder,'forward',side_effect=AssertionError('no E forward')), \
         patch('maidata.model.build_draft',side_effect=AssertionError('no D construction')):
        metrics,audit = runner.update(model,[raw],opt,recipe.scheduler(opt,[1]*6,c),c,distances(),audit=True)
    direct = direct_update(reference,[raw],reference_opt,recipe.scheduler(reference_opt,[1]*6,c),c)
    assert audit['gradients']['encoder'] == audit['changed_parameter_tensors']['encoder'] == 0
    assert audit['gradients']['decoder'] > 0 and audit['changed_parameter_tensors']['decoder'] > 0
    assert model.decoder.condition.z.weight.grad.abs().sum() > 0
    assert metrics.counts == direct.counts
    assert metrics.report(c)['loss'] == pytest.approx(direct.report(c)['loss'],rel=2e-6)
    for (name,a),(_,b) in zip(model.named_parameters(),reference.named_parameters()):
        torch.testing.assert_close(a,b,atol=2e-5,rtol=4e-4,msg=name)
    assert all(not t.requires_grad for t in (raw.conditions.audio.values,raw.conditions.decoder_z.values,raw.conditions.plan))


def test_cached_dataset_never_opens_mel_and_reuses_window_labels(adapted):
    c, saved, items, cache, p0 = adapted
    item = items[0]
    rows = [dict(window_id=w['identity'][0],occurrence=w['identity'][1],chart_id=w['chart_id'],
                 start_beat=w['start'],sequence_index=i) for i,w in enumerate(item['windows'])]
    plan = SimpleNamespace(groups={},training=False,rows=lambda *args:rows)
    labels = {w['identity'][0]:dict(history=w['offset'],tokens=w['tokens'],next_types=w['next_types'],
        labels=w['labels'],metadata=dict(logical_lengths=[w['length']-w['offset'],w['offset']])) for w in item['windows']}
    # GT cache was validated independently; this fixture supplies its selected read.
    with patch('maidata.decoder_training.gt_plan_cache.GTPlanCache',return_value=SimpleNamespace(read=lambda v:item['gt_plan'])), \
         patch('maidata.decoder_training.data.load_grid',side_effect=AssertionError('no Mel IO')):
        dataset = DecoderDataset(plan,c,{},cache,SimpleNamespace(read=lambda key:labels[key]))
        got = dataset[('0',1)]
    assert got['grid'] is None
    assert torch.equal(got['p0'].values,item['p0'].values)
    assert [w['identity'] for w in got['windows']] == [w['identity'] for w in item['windows']]


def test_adaptation_runner_resume_is_exact_and_does_not_reopen_initial_m(adapted,tmp_path,monkeypatch):
    c, saved, items, cache, p0 = adapted
    c.update({k:str(tmp_path/k) for k in recipe.PATH_KEYS if c[k] is None})
    c.update(decoder_train_stop_after_epochs=3,decoder_train_scheduler_epochs=4)
    class Plan:
        def __init__(self,*args,**kwargs): pass
        def batches(self,*args): return [None]
    def loader(*args):
        return DataLoader(items,batch_size=2,collate_fn=partial(collate_audio,config=c,packed=True),
                          generator=torch.Generator().manual_seed(72))
    monkeypatch.setattr(runner,'DecoderPlan',Plan)
    monkeypatch.setattr(runner,'FrozenCache',lambda *args:cache)
    monkeypatch.setattr(runner,'WindowCache',lambda *args:SimpleNamespace(binding='windows'))
    monkeypatch.setattr(runner,'make_loader',loader)
    monkeypatch.setattr(runner,'load_route_distances',lambda *args:distances())
    full, interrupted = dict(c,decoder_train_output=str(tmp_path/'full')),dict(c,decoder_train_output=str(tmp_path/'partial'))
    runner.run(full)
    update, calls = runner.update, 0
    def fail(*args,**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('interrupted')
        return update(*args,**kwargs)
    monkeypatch.setattr(runner,'update',fail)
    with pytest.raises(RuntimeError,match='interrupted'):
        runner.run(interrupted)
    path = tmp_path/'partial/checkpoints/last.ckpt'
    assert torch.load(path,weights_only=False)['completed_epochs'] == 1
    monkeypatch.setattr(runner,'update',update)
    interrupted['decoder_train_init_checkpoint'] = 'missing-original.ckpt'
    runner.run(interrupted,resume=path)
    actual = torch.load(path,weights_only=False)
    expected = torch.load(tmp_path/'full/checkpoints/last.ckpt',weights_only=False)
    for key in ('model_state','optimizer_state','scheduler_state','rng','loader_rng','step','completed_epochs'):
        assert_trees_equal(actual[key],expected[key])
    assert actual['recipe']['decoder_z_cross_attention']
    assert actual['recipe']['decoder_train_audio_scope'] == 'full_song'
    assert actual['initialization']['completed_epochs'] == 45
    # Completed adaptation still carries a real frozen E plus M for normal inference.
    from maidata.decoder_training.producer import FrozenProducer
    from maidata.inference.checkpoint import load_models
    from maidata.training.checkpoint import SCHEMA
    teacher = tmp_path/'teacher.ckpt'
    torch.save(dict(schema=SCHEMA,recipe=c,model_state=FrozenProducer(c).state_dict()),teacher)
    loaded = load_models(teacher,path,device='cpu',plan_source='gt')
    assert loaded.decoder.condition.z_cross_attention
    assert loaded.config['decoder_train_audio_scope'] == 'full_song'
    assert_trees_equal(loaded.decoder_encoder.state_dict(),{k.removeprefix('encoder.'):v
                      for k,v in actual['model_state'].items() if k.startswith('encoder.')})


def test_adaptation_rejects_wrong_producer_or_implicit_optimizer_resume(adapted):
    c,saved,_,cache,p0 = adapted
    with pytest.raises(ValueError,match='--init-checkpoint'):
        adaptation.prepare(c,resume=c['decoder_train_init_checkpoint'])
    other = deepcopy(cache)
    other.producer['audio_producer']['producer_sha256'] = 'different-E'
    with pytest.raises(ValueError,match='frozen P0 producer'):
        adaptation.bind_conditions(other,p0)
    bad = deepcopy(saved)
    key = next(k for k in bad['model_state'] if k.startswith('encoder.'))
    bad['model_state'][key] += 1
    with pytest.raises(ValueError,match='weights differ'):
        adaptation.initialize(DecoderTrainer(c),bad,'wrong.ckpt')
    bad = deepcopy(saved)
    bad['model_state']['decoder.heads.kind.weight'] = torch.empty(1)
    with pytest.raises((RuntimeError,ValueError)):
        adaptation.initialize(DecoderTrainer(c),bad,'wrong-M.ckpt')


def test_adapt_cli_is_explicit_and_show_config_does_not_load_inputs(capsys):
    from maidata.decoder_training.__main__ import main
    with patch('torch.load',side_effect=AssertionError('help/config cannot load models')):
        main(['--init-checkpoint','epoch-045.ckpt','--p0-cache','p0','--dp-cache','dp','--show-config'],adaptation=True)
    c = json.loads(capsys.readouterr().out)
    assert c['decoder_train_audio_mode'] == 'cached' and c['decoder_z_cross_attention']
    assert c['decoder_train_plan_source'] == 'gt' and c['decoder_train_workers'] == 12
    assert c['decoder_train_init_checkpoint'] == 'epoch-045.ckpt'


def test_dp_export_reuses_stage_two_p0_after_frozen_drafter_training(config,tmp_path):
    from pathlib import Path
    from test_pretrain_frozen_p0 import inputs
    from maidata.training.p0_cache import export as export_p0
    from maidata.training import runner as train_draft
    from maidata.decoder_training.dp_cache import export as export_dp
    from maidata.decoder_training.producer import load_export_producer
    from maidata.decoder_training.dp_cache import CachedP0Dataset
    c, source = inputs(tmp_path,config)
    p0_root, dp_root = tmp_path/'p0',tmp_path/'dp'
    export_p0(c,source,p0_root)
    c.update(pretrain_audio_mode='cached',pretrain_p0_cache=str(p0_root),
             pretrain_output=str(tmp_path/'draft'),pretrain_stop_after_epochs=1,pretrain_scheduler_epochs=2)
    train_draft.run(c)
    for side in ('train','validation'):
        path = Path(c[f'pretrain_{side}_plan'])/'plan.json'
        plan = json.loads(path.read_text())
        plan['policy'] = {'split':side}
        plan['audio_groups'].reverse()  # M order need not match stage-I P0 export order.
        path.write_text(json.dumps(plan))
    export_config = dict(c,decoder_train_checkpoint=str(tmp_path/'draft/checkpoints/last.ckpt'),
        decoder_train_p0_cache=str(p0_root),decoder_train_train_plan=c['pretrain_train_plan'],
        decoder_train_validation_plan=c['pretrain_validation_plan'],decoder_train_chart_cache=c['pretrain_chart_cache'],
        decoder_train_source_root=c['pretrain_source_root'])
    with patch('maidata.audio.encoder.AudioEncoder.forward',side_effect=AssertionError('export must reuse P0')), \
         patch('maidata.decoder_training.dp_cache.digest_file',wraps=checkpoint.digest_file) as digest:
        report = export_dp(export_config,p0_root,dp_root)
    # Export hashes only its newly created outputs, never scans old input/cache payloads.
    assert all(Path(call.args[0]).is_relative_to(dp_root) for call in digest.call_args_list)
    assert report['complete'] and report['first_bank_roundtrip_exact']
    assert report['encoder_forward_calls'] == report['decoder_forward_calls'] == 0
    cache = FrozenCache([],dict(export_config,decoder_train_dp_cache=str(dp_root)),{})
    p0 = FrozenP0(p0_root)
    adaptation.bind_conditions(cache,p0)
    assert cache.latent_dim == c['draft_dim']
    teacher,_,teacher_config = load_export_producer(export_config,{})
    plans = [json.loads((Path(c[f'pretrain_{side}_plan'])/'plan.json').read_text()) for side in ('train','validation')]
    dataset = CachedP0Dataset(teacher_config,p0_root,p0.manifest,plans)
    with torch.no_grad():
        batch = dataset[0].to('cpu')
        expected = teacher.draft(batch['bank'],batch['inputs'])
    actual = cache.read(batch['variant'])
    torch.testing.assert_close(actual['z']['values'],expected.values,rtol=0,atol=0)
    # A new teacher trained on another Encoder cannot silently reuse these banks.
    path = Path(export_config['decoder_train_checkpoint'])
    payload = torch.load(path,weights_only=False)
    payload['audio_producer']['producer_sha256'] = 'wrong'
    torch.save(payload,tmp_path/'wrong-teacher.ckpt')
    with pytest.raises(ValueError,match='not trained on'):
        export_dp(dict(export_config,decoder_train_checkpoint=str(tmp_path/'wrong-teacher.ckpt')),
                  p0_root,tmp_path/'wrong-export')
