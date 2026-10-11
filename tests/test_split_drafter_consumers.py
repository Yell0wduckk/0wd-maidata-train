"""Split teacher Z export, stage-specific construction and unchanged M paths."""
from unittest.mock import patch
import json

import pytest
import torch

from test_pretrain_runtime import config, item
from test_decoder_network import config as decoder_config
from maidata.model import build_model, build_draft, build_planner, build_decoder
from maidata.training import checkpoint as stage_one
from maidata.decoder.types import Memory
from maidata.decoder_training.producer import load_export_producer
from maidata.decoder_training.frozen_cache import FrozenCache, condition_config
from maidata.decoder_training.cache_schema import DP_SCHEMA, Z_REPRESENTATION
from maidata.decoder_training.dp_cache import save_outputs
from maidata.draft.types import BeatMemory
from maidata.draft.windows import select_windows
from maidata.inference.checkpoint import load_models
from maidata.inference.conditioning import prepare_conditions
from maidata.time import TimingMap


def paired(config, tmp_path, *, use_z=True, consumer_width=30):
    # E_D has a different hidden size and local window from E_M. M's unused
    # draft settings intentionally differ, including its old no-Z width.
    teacher = dict(config, d_model=24, draft_dim=30, draft_latent_dim=30,
                   audio_before_beats=1, audio_after_beats=2)
    consumer = dict(config, draft_latent_dim=consumer_width)
    consumer.update({k: v for k, v in decoder_config().items() if k.startswith('decoder_')})
    consumer['decoder_z_cross_attention'] = use_z
    modules = dict(encoder=build_model(teacher), draft=build_draft(teacher), planner=build_planner(teacher))
    binding = dict(pretrain_chart_cache={'catalog.json': 'fields'}, pretrain_audio_cache={'catalog.json': 'audio'})
    payload = dict(schema=stage_one.SCHEMA, recipe=stage_one.recipe(teacher), binding=binding, source={},
                   completed_epochs=1, step=1, steps_by_epoch=[1], scheduler_state={'last_epoch': 1},
                   model_state={name+'.'+k: v for name, module in modules.items() for k, v in module.state_dict().items()})
    second = torch.nn.ModuleDict(dict(encoder=build_model(consumer), decoder=build_decoder(consumer)))
    state = dict(schema='maidata.decoder-training.v4', architecture='maidata.decoder-causal-cont-tick384.v2',
                 recipe=consumer, model_state=second.state_dict())
    paths = tmp_path/'teacher.ckpt', tmp_path/'consumer.ckpt'
    torch.save(payload, paths[0])
    torch.save(state, paths[1])
    return paths, teacher, consumer, payload, binding


@pytest.mark.parametrize('precision', ['fp32', 'bf16'])
def test_distinct_teacher_and_m_configs_execute_with_direct_z_and_planner(config, tmp_path, precision):
    paths, teacher, consumer, _, _ = paired(config, tmp_path)
    models = load_models(*paths, device='cpu')
    assert models.config['d_model'] == 18 and models.teacher_config['d_model'] == 24
    assert models.draft.dim == 30
    assert models.config['draft_dim'] == 24 and models.teacher_config['draft_dim'] == 30
    assert models.decoder.condition.z.in_features == 30
    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=precision == 'bf16'):
        conditions = prepare_conditions(models.encoder, models.draft, models.planner, models.config,
            item(0, consumer)['grid'], TimingMap([(0, 120)]), 1.2, difficulty=5, level=14.5,
            device='cpu', decoder_encoder=models.decoder_encoder, teacher_config=models.teacher_config)
        assert conditions.z.values.shape[-1] == 30
        assert conditions.audio.values.shape[-1] == 18
        assert conditions.audio_scope == 'window' and conditions.z.valid.all()
        local = Memory(conditions.audio.values, conditions.audio.beats, conditions.audio.valid)
        selected = select_windows(conditions.z, torch.zeros(2, dtype=torch.long),
            torch.tensor(conditions.starts).float(), torch.ones(2, dtype=torch.bool),
            target_beats=3, before_beats=0, after_beats=0).memory
        prepared = models.decoder.prepare_memory(models.decoder.prepare_audio(local), torch.arange(2),
            Memory(selected.values, selected.beats, selected.valid), conditions.plans, conditions.plan_valid)
        assert torch.isfinite(prepared.condition[0].value).all()


def test_export_builds_teacher_recipe_instead_of_consumer_recipe_and_roundtrips(config, tmp_path):
    paths, teacher, consumer, payload, binding = paired(config, tmp_path)
    # Match the actual v3 runner/capture provenance. These accepted input paths
    # deliberately do not exist: loading/export must not scan or re-hash them.
    teacher.update(pretrain_chart_cache=str(tmp_path/'accepted-fields'),
                   pretrain_audio_cache=str(tmp_path/'accepted-audio'))
    payload['binding'] = stage_one.data_provenance(teacher)
    torch.save(payload, paths[0])
    model, producer, exported = load_export_producer(dict(consumer, decoder_train_checkpoint=str(paths[0])), binding)
    assert exported['draft_latent_dim'] == exported['draft_dim'] == 30
    assert exported['d_model'] == 24 and exported['audio_before_beats'] == 1
    assert producer['schema'] == stage_one.SCHEMA
    assert producer['binding'] == payload['binding'] and set(producer['binding']) == {'paths'}
    values = torch.randn(1, 32, 30, requires_grad=True)
    z = BeatMemory(values, torch.arange(32)[None].float()+.5, torch.ones(1, 32, dtype=torch.bool))
    windows = select_windows(z, torch.zeros(2, dtype=torch.long), torch.tensor([0., 16.]),
                            torch.ones(2, dtype=torch.bool), target_beats=16, before_beats=0, after_beats=0)
    plan = model.planner(windows)
    plan.plan_values[:, :8].sum().backward()
    assert (values.grad.abs().sum(-1) > 0).all()  # P reads every beat in both 16-beat windows.
    batch = dict(variant='song', chart_to_group={'chart': 0}, conditions=[], window_ids=['a', 'tail'],
                 chart_ids=['chart', 'chart'], groups=torch.zeros(2, dtype=torch.long), starts=torch.tensor([0., 16.]))
    root = tmp_path/'fresh-dp'
    row = save_outputs(root/'banks/song.pt', batch, z, plan, producer['sha256'])
    manifest = dict(schema=DP_SCHEMA, z_representation=Z_REPRESENTATION, z_width=30,
                    producer=producer, groups={'song': row})
    (root/'manifest.json').write_text(json.dumps(manifest))
    m_config = dict(consumer, draft_latent_dim=128, decoder_train_dp_cache=str(root),
                    decoder_train_plan_source='predicted')
    with patch('torch.load', side_effect=AssertionError('startup cannot scan existing payloads')):
        cache = FrozenCache([], m_config, {})
    adapted = condition_config(m_config, cache)
    assert adapted['draft_latent_dim'] == 30 and adapted['draft_dim'] == consumer['draft_dim']
    assert build_decoder(adapted).condition.z.in_features == 30
    saved = cache.read('song')
    torch.testing.assert_close(saved['z']['values'], values.detach(), rtol=0, atol=0)
    torch.testing.assert_close(saved['planner']['plan_values'], plan.plan_values.detach(), rtol=0, atol=0)
    saved['z']['values'] = saved['z']['values'][..., :10]
    torch.save(saved, root/'banks/song.pt')
    with pytest.raises(ValueError, match='width differs'):
        cache.read('song')


def test_predicted_plan_with_z_off_keeps_old_m_width_and_gt_none_never_load_teacher(config, tmp_path):
    paths, _, consumer, _, _ = paired(config, tmp_path, use_z=False, consumer_width=128)
    models = load_models(*paths, device='cpu', plan_source='predicted')
    assert models.teacher_config['draft_latent_dim'] == 30
    assert models.config['draft_latent_dim'] == models.decoder.condition.z.in_features == 128
    for source in ('gt', 'none'):
        actual_load = torch.load
        def only_consumer(path, *args, **kwargs):
            assert path == paths[1]
            return actual_load(path, *args, **kwargs)
        with patch('torch.load', side_effect=only_consumer):
            loaded = load_models(paths[0], paths[1], device='cpu', plan_source=source)
        assert loaded.encoder is loaded.draft is loaded.planner is loaded.teacher_config is None
        assert loaded.decoder.condition.z_cross_attention is False


def test_old_teacher_and_z_width_mismatch_are_explicit(config, tmp_path):
    paths, _, _, payload, binding = paired(config, tmp_path, consumer_width=128)
    with pytest.raises(ValueError, match='Z width differs'):
        load_models(*paths, device='cpu')
    payload['schema'] = 'maidata.encoder-draft-training.v2'
    torch.save(payload, paths[0])
    with pytest.raises(ValueError, match='stage-I v3'):
        load_models(*paths, device='cpu')
    with pytest.raises(ValueError, match='stage-I v3'):
        load_export_producer(dict(config, decoder_train_checkpoint=str(paths[0])), binding)
    root = tmp_path/'old-dp'
    root.mkdir()
    (root/'manifest.json').write_text(json.dumps(dict(schema='maidata.frozen-draft-planner.v1', groups={})))
    with pytest.raises(ValueError, match='old Z cannot be expanded'):
        FrozenCache([], dict(config, decoder_z_cross_attention=True, decoder_train_dp_cache=str(root),
                             decoder_train_plan_source='gt'), {})
