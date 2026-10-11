from dataclasses import replace
from unittest.mock import patch
import pytest
import torch

from maidata.model import build_decoder
from maidata.decoder.batching import audio_query_layout, audio_key_spans
from maidata.decoder.types import Memory
from maidata.decoder_training.conditioning import collate_conditions
from maidata.decoder_training.frozen_cache import FrozenCache
from test_decoder_network import config, inputs


@pytest.mark.parametrize('packed', [False, True])
def test_disabled_z_is_inert_for_router_content_and_field_queries_but_plan_audio_have_gradients(packed):
    torch.set_num_threads(1)
    model = build_decoder(dict(config(), decoder_z_cross_attention=False)).eval()
    tokens, audio, z, plan = inputs(model, gradients=True)
    mapping = torch.tensor([0, 0])
    validity = torch.ones(2, dtype=torch.bool)
    layout = (audio_query_layout(tokens.valid, mapping, 1, spans=audio_key_spans(audio.valid),
              condition_valid=torch.cat((z.valid, validity[:, None]), -1)) if packed else None)
    def evaluate(z, plan):
        memory = model.prepare_memory(model.prepare_audio(audio), mapping, z, plan, validity, audio_layout=layout)
        assert not memory.condition[0].valid[:, :-1].any()
        assert memory.condition[0].valid[:, -1].all()
        out = model(tokens, torch.ones_like(tokens.types), memory)
        fields = model.query_fields(out, torch.tensor([[0, 1], [1, 1]]), torch.tensor([0., 2.]),
                                    torch.tensor([1, 0]), torch.tensor([0., 192.]), torch.tensor([0., 0.]))
        return torch.cat((out.hidden.flatten(), out.type_logits.flatten(), fields.flatten()))
    with patch.object(model.condition.z, 'forward', side_effect=AssertionError('Z projection used')), \
         patch.object(model.field_reader.z, 'forward', side_effect=AssertionError('Z field attention used')):
        result = evaluate(z, plan)
        changed = replace(z, values=torch.full_like(z.values, float('nan')),
                          beats=z.beats+1000, valid=~z.valid)
        torch.testing.assert_close(result, evaluate(changed, plan), rtol=0, atol=0)
        assert not torch.equal(result, evaluate(z, plan+5))
        result.square().mean().backward()
    assert z.values.grad is None
    assert model.condition.z.weight.grad is None
    assert model.field_reader.z.query.weight.grad is None
    for value in (audio.values, plan):
        assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0


def test_gt_only_conditions_do_not_open_dp_cache_and_preserve_plan_mapping():
    c = dict(config(), decoder_z_cross_attention=False, decoder_train_plan_source='gt',
             decoder_train_dp_cache=None)
    cache = FrozenCache([], c, {})
    assert cache.read('audio') == {} and cache.binding == {} and cache.producer == {}
    windows = [dict(identity=('second', 0), chart_id='chart', start=16),
               dict(identity=('first', 0), chart_id='chart', start=0),
               dict(identity=('second', 1), chart_id='chart', start=16)]
    plans = torch.arange(24).reshape(2, 12).float()
    items = [dict(variant='audio', dp={}, windows=windows,
                  gt_plan=dict(window_index={'first': 0, 'second': 1}, plan_values=plans,
                               valid=torch.ones(2, dtype=torch.bool)))]
    audio = Memory(torch.zeros(1, 2, 18), torch.zeros(1, 2), torch.ones(1, 2, dtype=torch.bool))
    got = collate_conditions(items, windows, c, audio=audio)
    assert got.decoder_z.values.shape == (2, 0, c['draft_latent_dim'])
    assert got.occurrence_mapping.tolist() == [0, 1, 0]
    torch.testing.assert_close(got.plan, plans[[1, 0]])
