"""Main-capacity recipe with historical block Z4 and one unpositioned Plan16."""
import json
from pathlib import Path
from unittest.mock import patch

import torch

from test_inference import config, producer_config
from test_inference_joint import joint_config, joint_checkpoint
from test_pretrain_runtime import item
from maidata.chart import Moment, Note
from maidata.decoder.batching import pack_tokens
from maidata.draft.types import BeatMemory
from maidata.draft.windows import select_windows
from maidata.entry import CONFIG
from maidata.inference.checkpoint import JointCheckpoint
from maidata.inference.conditioning import prepare_joint_conditions
from maidata.joint_training.__main__ import main as training_cli
from maidata.joint_training.recipe import resolve
from maidata.time import TimingMap


RECIPE = Path(__file__).resolve().parents[1]/'configs/joint-main320-z4-plan16-gt.json'


def test_main320_legacy_recipe_is_accepted_with_fixed_training_contract(capsys):
    assert training_cli(['--config', str(RECIPE), '--show-config']) == 0
    c = json.loads(capsys.readouterr().out)
    assert (c['d_model'], c['draft_dim'], c['draft_latent_dim'], c['decoder_dim']) == (320,)*4
    assert (c['heads'], c['draft_heads'], c['decoder_heads'], c['decoder_layers']) == (8,)*4
    assert c['ffn_dim'] == c['decoder_shared_ffn_dim'] == 1280
    assert c['draft_expert_ffn_dim'] == c['decoder_expert_ffn_dim'] == 640
    assert c['draft_z_stride_beats'] == c['joint_train_z_stride_beats'] == 4
    assert c['planner_span_beats'] == 16 and c['joint_train_plan_source'] == 'gt'
    assert c.get('planner_layout', 'legacy') == 'legacy'
    assert c.get('draft_z_anchor_mode', 'block-center') == 'block-center'
    assert c['pretrain_audio_groups_per_batch'] == 3 and c['pretrain_token_budget'] == 80000
    assert c['decoder_train_chunk_token_budget'] == 32768
    assert c['joint_train_learning_rate'] == 6e-4
    assert c['joint_train_scheduler_epochs'] == c['joint_train_stop_after_epochs'] == 150
    assert c['joint_train_workers'] == 12 and c['joint_train_precision'] == 'bf16'


def test_full_320_m8_checkpoint_strictly_loads_original_plan16_on_main(tmp_path):
    torch.set_num_threads(1)
    c = resolve(dict(CONFIG, **json.loads(RECIPE.read_text('utf-8'))))
    path, original, payload = joint_checkpoint(c, tmp_path)
    expected = dict(encoder=11115880, draft=21488672, reconstruction=815703,
                    planner=151692, decoder=22606325)
    assert {name: sum(p.numel() for p in model.parameters())
            for name, model in original.items()} == expected
    assert sum(expected.values()) == 56178272
    loaded = JointCheckpoint(path).models(device='cpu')
    assert loaded.encoder is loaded.decoder_encoder
    assert loaded.decoder.dim == loaded.draft.dim == 320
    assert len(loaded.decoder.layers) == 8 and loaded.decoder.plan_span_beats == 16
    assert loaded.planner.span_beats == 16
    assert loaded.config['draft_z_stride_beats'] == 4
    for name in ('encoder', 'draft', 'planner', 'decoder'):
        actual = getattr(loaded, name)
        assert not actual.training and not any(p.requires_grad for p in actual.parameters())
        for key, value in actual.state_dict().items():
            torch.testing.assert_close(value, payload['model_state'][name+'.'+key], rtol=0, atol=0)


def test_small_m8_real_inference_has_block_centers_and_one_unpositioned_plan(joint_config, tmp_path):
    c = dict(joint_config, decoder_layers=8, planner_span_beats=16)
    path, _, _ = joint_checkpoint(c, tmp_path)
    models = JointCheckpoint(path).models(device='cpu')
    with patch.object(models.encoder, 'encode_windows', wraps=models.encoder.encode_windows) as encode, \
         patch.object(models.draft, 'forward', wraps=models.draft.forward) as draft, \
         patch.object(models.planner, 'forward', wraps=models.planner.forward) as planner:
        conditions = prepare_joint_conditions(models.encoder, models.draft, models.planner, models.config,
            item(0, c)['grid'], TimingMap([(0, 120)]), 1.2,
            difficulty=5, level=14.5, device='cpu')
    assert encode.call_count == draft.call_count == planner.call_count == 1
    assert conditions.starts == (0, 16)
    assert conditions.z.beats.tolist() == [[2., 6., 10., 14., 18., 22., 26., 30.]]
    assert conditions.plans.shape == (2, 12) and conditions.plan_valid.tolist() == [True, True]
    with torch.inference_mode():
        for index, start in enumerate(conditions.starts):
            z = select_windows(conditions.z, torch.tensor([0]), torch.tensor([float(start)]),
                torch.tensor([True]), target_beats=c['target_beats'],
                before_beats=c['decoder_z_before_beats'], after_beats=c['decoder_z_after_beats'],
                z_stride_beats=c['draft_z_stride_beats']).memory
            assert z.beats.tolist() == [[start-2., start+2., start+6., start+10., start+14., start+18.]]
            assert z.valid.tolist() == ([[False, True, True, True, True, True]] if index == 0
                                        else [[True, True, True, True, True, False]])
            audio = BeatMemory(*(getattr(conditions.audio, key)[index:index+1]
                                for key in ('values', 'beats', 'valid')))
            plan, valid = conditions.plans[index:index+1], conditions.plan_valid[index:index+1]
            condition = models.decoder.condition(z, plan, valid)
            memory = models.decoder.prepare_memory(models.decoder.prepare_audio(audio),
                torch.zeros(1, dtype=torch.long), z, plan, valid)
            assert memory.plan_index == 6 and condition.values.shape[1] == 7
            assert condition.beats[:, -1].count_nonzero() == 0
            for layer, projected in zip(models.decoder.layers, memory.condition):
                attention = layer.condition_attn
                raw = attention.split(attention.key(condition.values.masked_fill(~condition.valid[..., None], 0)))
                # The single Plan16 key retains its raw projection without RoPE.
                torch.testing.assert_close(projected.key[:, :, 6:], raw[:, :, 6:], rtol=0, atol=0)
            tokens = pack_tokens([[Moment(0), Note.make('tap', 'A1')]],
                                 [[float(start), start+.5]], device='cpu')
            output = models.decoder(tokens, torch.tensor([[1, 0]]), memory)
            assert torch.isfinite(output.hidden).all() and torch.isfinite(output.type_logits).all()
