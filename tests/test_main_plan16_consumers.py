"""Main accepts original single-window plans and rejects segmented producers."""
import json

import pytest
import torch

from maidata.decoder_training.gt_plan_cache import GTPlanCache, SCHEMA, export
from maidata.decoder_training.cache_schema import DP_SCHEMA, Z_REPRESENTATION
from maidata.decoder_training.frozen_cache import FrozenCache


@pytest.mark.parametrize('schema', [None, SCHEMA])
def test_legacy_gt_bank_remains_single_plan_per_window(tmp_path, schema):
    values = torch.rand(2, 12)
    bank = dict(plan_values=values, valid=torch.ones(2, dtype=torch.bool), window_ids=['a:0', 'a:16'])
    torch.save(bank, tmp_path/'bank.pt')
    manifest = dict(groups={'audio': dict(path='bank.pt')})
    if schema is not None:
        manifest['schema'] = schema
    (tmp_path/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    result = GTPlanCache(tmp_path, dict(planner_span_beats=16)).read('audio')
    assert result['window_index'] == {'a:0': 0, 'a:16': 1}
    torch.testing.assert_close(result['plan_values'], values)
    assert result['valid'].shape == (2,)


def test_gt_segmented_schema_config_and_export_are_rejected(tmp_path):
    manifest = dict(schema='maidata.gt-plan4x12.v1', groups={})
    (tmp_path/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='schema'):
        GTPlanCache(tmp_path)
    manifest['schema'] = SCHEMA
    (tmp_path/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='single Plan16'):
        GTPlanCache(tmp_path, dict(planner_span_beats=4))
    with pytest.raises(ValueError, match='single Plan16'):
        export(dict(planner_span_beats=4), tmp_path/'not-created')
    assert not (tmp_path/'not-created').exists()


def test_frozen_predicted_bank_keeps_legacy_and_rejects_other_spans(tmp_path):
    manifest = dict(schema=DP_SCHEMA, z_representation=Z_REPRESENTATION, z_width=12, z_stride_beats=4,
        producer=dict(recipe=dict(draft_z_stride_beats=4)), groups={'audio': dict(path='bank.pt')})
    bank = dict(schema=DP_SCHEMA, z_representation=Z_REPRESENTATION, z_width=12, z_stride_beats=4,
        z=dict(values=torch.randn(1, 8, 12), beats=torch.arange(2, 32, 4).float()[None],
               valid=torch.ones(1, 8, dtype=torch.bool)),
        planner=dict(plan_values=torch.rand(2, 12), valid=torch.ones(2, dtype=torch.bool),
                     window_ids=['a:0', 'a:16']))
    torch.save(bank, tmp_path/'bank.pt')
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps(manifest), encoding='utf-8')
    config = dict(decoder_z_cross_attention=True, decoder_train_plan_source='predicted',
                  decoder_train_dp_cache=str(tmp_path), target_beats=16)
    cache = FrozenCache([], config, {})
    assert cache.read('audio')['planner']['plan_values'].shape == (2, 12)
    manifest['plan_span_beats'] = 4
    path.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='single Plan16'):
        FrozenCache([], config, {})
    manifest.pop('plan_span_beats')
    path.write_text(json.dumps(manifest), encoding='utf-8')
    bank['plan_span_beats'] = 4
    torch.save(bank, tmp_path/'bank.pt')
    with pytest.raises(ValueError, match='representation'):
        FrozenCache([], config, {}).read('audio')
