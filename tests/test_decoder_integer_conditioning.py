"""Frozen conditions preserve chart ownership, integer anchors and separate masks."""
from types import SimpleNamespace

import pytest
import torch

from maidata.decoder.types import Memory
from maidata.decoder_training import conditioning, preparation, checkpoint
from maidata.decoder_training.gt_plan_cache import CENTERED_SCHEMA


def config(**changes):
    return dict(dict(planner_layout='centered-overlap',planner_span_beats=16,planner_stride_beats=4,
        draft_z_anchor_mode='integer',draft_z_stride_beats=1,reconstruction_span_beats=4,
        target_beats=16,draft_latent_dim=4,decoder_z_before_beats=4,decoder_z_after_beats=4,
        decoder_z_cross_attention=True,decoder_train_plan_source='gt',decoder_train_row_budget=8192,
        decoder_train_chunk_token_budget=9,decoder_train_plan_input=True),**changes)


def items():
    beats = torch.arange(49.).expand(2,-1)
    z_valid = torch.ones_like(beats,dtype=torch.bool)
    z_valid[0,16] = False
    z_values = (beats+torch.tensor([0.,100.])[:,None])[...,None].expand(-1,-1,4).clone()
    centers = torch.arange(0.,49.,4.).repeat(2)
    values = torch.zeros(26,12)
    values[:,0] = centers+torch.tensor([0.]*13+[100.]*13)
    values[:,8] = 1.
    values[12,:8] = 0.  # Empty extra-tail endpoint at center 48 remains valid.
    plan_valid = torch.ones(26,dtype=torch.bool)
    plan_valid[5] = False  # Missing center20 does not remove Z20.
    bank = dict(schema=CENTERED_SCHEMA,planner_layout='centered-overlap',plan_span_beats=16,
        plan_stride_beats=4,chart_ids=['A']*13+['B']*13,center_beats=centers,plan_values=values,valid=plan_valid)
    dp = dict(chart_to_group={'A':0,'B':1},window_index={},planner=bank,
        z=dict(values=z_values,beats=beats,valid=z_valid))
    windows = [dict(identity=(f'{chart}:{start}',occurrence),chart_id=chart,start=start,
        length=3,history_dropped=False) for chart,start,occurrence in
        [('A',0,0),('A',16,0),('A',16,1),('A',32,0),('B',16,0)]]
    return [dict(variant='song',dp=dp,gt_plan=bank,windows=windows)],windows


@pytest.mark.parametrize('use_z',[False,True])
@pytest.mark.parametrize('source',['gt','predicted'])
def test_collation_keeps_23_and_five_geometry_without_copying_edges(use_z,source):
    source_items,windows = items()
    c = config(decoder_z_cross_attention=use_z,decoder_train_plan_source=source)
    audio = Memory(torch.empty(1,2,0),torch.zeros(1,2),torch.ones(1,2,dtype=torch.bool))
    conditions = conditioning.collate_conditions(source_items,windows,c,audio=audio)
    assert conditions.occurrence_mapping.tolist() == [0,1,1,2,3]
    assert conditions.decoder_z.values.shape == (4,23,4)
    assert conditions.plan.shape == (4,5,12)
    assert conditions.decoder_z.beats.tolist() == [
        list(range(-3,20)),list(range(13,36)),list(range(29,52)),list(range(13,36))]
    assert conditions.plan_beats.tolist() == [
        [0,4,8,12,16],[16,20,24,28,32],[32,36,40,44,48],[16,20,24,28,32]]
    assert not conditions.plan_valid[1,1]
    assert conditions.plan_valid[2,-1] and conditions.plan[2,-1,8] == 1
    assert conditions.plan[2,-1,:8].count_nonzero() == 0
    assert conditions.plan[3,0,0] == 116  # Chart B owns its own center16.
    if use_z:
        assert not conditions.decoder_z.valid[0,:3].any()
        assert not conditions.decoder_z.valid[2,-3:].any()
        assert not conditions.decoder_z.valid[1,3] and conditions.plan_valid[1,0]
        assert conditions.decoder_z.valid[1,7] and not conditions.plan_valid[1,1]
        assert conditions.decoder_z.values[3,3,0] == 116
    else:
        assert not conditions.decoder_z.valid.any()


@pytest.mark.parametrize('plan_input',[False,True])
def test_packed_indices_and_plan_gate_use_independent_28_slot_mask(monkeypatch,plan_input):
    source_items,windows = items()
    c = config(decoder_train_plan_input=plan_input)
    audio = Memory(torch.empty(1,2,0),torch.zeros(1,2),torch.ones(1,2,dtype=torch.bool))
    conditions = conditioning.collate_conditions(source_items,windows,c,audio=audio)
    batch = SimpleNamespace(conditions=conditions,audio_lengths=[2],windows=windows)
    monkeypatch.setattr(preparation,'collate_prepared',lambda rows:
        (SimpleNamespace(valid=torch.ones(len(rows),3,dtype=torch.bool)),None,None))
    preparation.prepare_batch(batch,c)
    assert len(batch.chunks) == 2
    for chunk in batch.chunks:
        selected = batch.conditions.occurrence_mapping[chunk.indices]
        expected = torch.cat((batch.conditions.decoder_z.valid[selected],batch.conditions.plan_valid[selected]),-1)
        actual = chunk.audio_layout['condition_rows']['key_indices']
        assert torch.equal(actual,expected.flatten().nonzero().flatten())
        if not plan_input:
            assert not (actual%28 >= 23).any()
    if not plan_input:
        assert not batch.conditions.plan_valid.any() and not batch.conditions.plan.any()
    assert torch.equal(batch.conditions.plan_beats,conditions.plan_beats)


def test_checkpoint_representation_defaults_and_new_semantic_identity():
    current = config()
    recorded = checkpoint.recipe(current)
    for key in ('planner_layout','planner_stride_beats','planner_span_beats','draft_z_anchor_mode',
                'reconstruction_span_beats'):
        assert recorded[key] == current[key]
    legacy = dict(planner_span_beats=16,draft_z_stride_beats=4)
    normalized = dict(legacy,planner_layout='legacy',planner_stride_beats=None,
        draft_z_anchor_mode='block-center',reconstruction_span_beats=None)
    checkpoint._validate_producer(dict(recipe=legacy),normalized,{})
    with pytest.raises(ValueError,match='representation differs'):
        checkpoint._validate_producer(dict(recipe=legacy),current,{})
    checkpoint._validate_producer(dict(recipe=current),current,{})


@pytest.mark.parametrize('resume',[False,True])
def test_adaptation_inherits_centered_geometry_before_constructing_m(tmp_path,resume):
    import json
    from maidata.decoder_training import adaptation
    from maidata.decoder_training.cache_schema import P0_SCHEMA
    from maidata.model import encoder_config,build_decoder
    from test_decoder_network import config as small_config
    saved = dict(small_config(),planner_layout='centered-overlap',planner_stride_beats=4,
        planner_span_beats=16,draft_z_anchor_mode='integer',draft_z_stride_beats=1,
        reconstruction_span_beats=4,decoder_train_audio_scope='window',
        decoder_train_audio_mode='cached' if resume else 'joint')
    source = build_decoder(saved)
    path = tmp_path/'centered.ckpt'
    torch.save(dict(schema=checkpoint.SCHEMA,architecture=checkpoint.ARCHITECTURE,
        recipe=checkpoint.recipe(saved),model_state={f'decoder.{key}':value for key,value in source.state_dict().items()}),path)
    cache = tmp_path/'p0'
    cache.mkdir()
    metadata = dict(schema=P0_SCHEMA,complete=True,encoder_config=encoder_config(saved),
        encoder=dict(path='encoder.pt',sha256='encoder-producer'),producer=dict(sha256='accepted-producer'),
        precision='fp32',p0_rate=saved['fpn_rates'][0],groups={})
    (cache/'manifest.json').write_text(json.dumps(metadata),encoding='utf-8')
    current = dict(small_config(),decoder_train_audio_mode='cached',decoder_train_p0_cache=str(cache),
        decoder_train_init_checkpoint=str(path),draft_z_stride_beats=4)
    resolved,payload = adaptation.prepare(current,resume=str(path) if resume else None)
    assert resolved['draft_z_anchor_mode'] == 'integer'
    assert resolved['draft_z_stride_beats'] == 1 and resolved['reconstruction_span_beats'] == 4
    assert resolved['draft_latent_dim'] == current['draft_latent_dim']
    decoder = build_decoder(resolved)
    decoder.load_state_dict({key.removeprefix('decoder.'):value for key,value in payload['model_state'].items()},strict=True)
    assert decoder.planner_layout == 'centered-overlap'
