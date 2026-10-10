"""Structural throughput regressions with value/gradient and mask oracles."""
from copy import deepcopy
from dataclasses import replace
import pickle
from unittest.mock import patch
import pytest
import torch

from maidata.chart import Moment, Note
from maidata.decoder.batching import pack_tokens, audio_query_layout, audio_key_spans
from maidata.decoder.types import Memory
from maidata.model import build_decoder, decoder_time_distributions
from maidata.decoder_training.data import collate_audio
from maidata.decoder_training.network import DecoderTrainer
from maidata.decoder_training import runner, recipe
from maidata.decoder_training.supervision import loss
from test_decoder_network import config as decoder_config
from test_decoder_training import config, encoder_config, training_item, distances
from test_decoder_training import fields, ChartSupports


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


@pytest.mark.parametrize('empty_audio', [False, True])
def test_mapped_full_song_kv_matches_copied_windows_and_gradients(empty_audio):
    torch.manual_seed(4)
    model = build_decoder(decoder_config()).eval()
    tokens = pack_tokens([(Moment(0),Note.make('tap','A1')), (Moment(500),),
                          (Moment(1000),Note.make('touch','C'),Moment(200))],
                         [(0,0),(1,),(2,2,2.4)],device='cpu')
    mapping = torch.tensor([1,0,1])
    valid = torch.tensor([[True]*9+[False]*4,[not empty_audio]*13])
    audio = Memory(torch.randn(2,13,model.audio_dim,requires_grad=True),
                   torch.arange(13).float()[None].expand(2,-1),valid)
    z = Memory(torch.randn(3,6,model.condition.latent_dim,requires_grad=True),
               torch.arange(6).float()[None].expand(3,-1),torch.ones(3,6,dtype=torch.bool))
    plan = torch.randn(3,12,requires_grad=True)
    keys = model.prepare_audio(audio)
    arguments = keys,mapping,z,plan,torch.tensor([True,False,True])
    baseline = model.prepare_memory(*arguments)
    mapped = model.prepare_memory(*arguments,audio_layout=audio_query_layout(tokens.valid,mapping,2,spans=audio_key_spans(valid)))
    for before,after,original in zip(baseline.audio,mapped.audio,keys):
        assert before.key.shape[0]==3 and after.key.shape[0]==2
        assert after.key.data_ptr()==original.key.data_ptr()
        assert after.value.data_ptr()==original.value.data_ptr()
    types = torch.ones_like(tokens.types)
    a,b = model(tokens,types,baseline),model(tokens,types,mapped)
    torch.testing.assert_close(a.hidden,b.hidden,atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(a.type_logits,b.type_logits,atol=2e-6,rtol=2e-5)
    assert b.hidden[~tokens.valid].count_nonzero()==0
    params = (*model.parameters(),audio.values,z.values,plan)
    ga = torch.autograd.grad(a.hidden.square().sum()+a.type_logits.square().sum(),params,retain_graph=True,allow_unused=True)
    gb = torch.autograd.grad(b.hidden.square().sum()+b.type_logits.square().sum(),params,allow_unused=True)
    for x,y in zip(ga,gb):
        if x is None or y is None:
            assert x is y
        else:
            torch.testing.assert_close(x,y,atol=2e-5,rtol=3e-4)


def test_worker_packet_has_bounded_buffers_and_no_runtime_cpu_geometry(config,monkeypatch):
    items = [training_item(0,config,monkeypatch,remaster=True),training_item(1,config,monkeypatch)]
    plain = collate_audio(items,config=config)
    packet = pickle.loads(pickle.dumps(collate_audio(items,config=config,packed=True)))
    assert len(packet.context.buffers)<=5
    assert all(len(c.buffers)<=5 for c in packet.chunks)
    assert all(set(w)=={'identity','history_dropped'} for w in packet.windows)
    assert len(packet.chunks)>1 and packet.context.chunks is None
    with patch.object(torch.Tensor,'pin_memory',lambda value:value.clone()):
        pinned = packet.pin_memory()
    for original,copy in zip(packet.chunks,pinned.chunks):
        for dtype,buffer in original.buffers.items():
            torch.testing.assert_close(copy.buffers[dtype],buffer,rtol=0,atol=0)
            assert copy.buffers[dtype].data_ptr()!=buffer.data_ptr()
    assert [w['identity'] for w in packet.windows]==[w['identity'] for w in plain.windows]
    assert sum([c.indices.tolist() for c in packet.chunks],[])==list(range(len(packet.windows)))
    model = DecoderTrainer(config)
    other = deepcopy(model)
    opts = [recipe.optimizer(m.decoder.parameters(),config) for m in (model,other)]
    scheds = [recipe.scheduler(o,[1]*6,config) for o in opts]
    # Everything below executes after DataLoader collation. Neither a full
    # chart/token replay nor GPU-to-CPU index preparation is permitted here.
    forbidden = ['maidata.audio.layout.prepare_encoder_layout','maidata.draft.layout.prepare_p0',
        'maidata.draft.layout.prepare_selection','maidata.decoder_training.supervision.prepare',
        'maidata.decoder.batching.collate_windows']
    from contextlib import ExitStack
    with ExitStack() as stack:
        for name in forbidden:
            stack.enter_context(patch(name,side_effect=AssertionError('work belongs in DataLoader workers')))
        a,_ = runner.update(model,[packet],opts[0],scheds[0],config,distances(),audit=True)
        b,_ = runner.update(other,[plain],opts[1],scheds[1],config,distances(),audit=True)
    assert a.counts==b.counts
    for x,y in zip(model.decoder.parameters(),other.decoder.parameters()):
        torch.testing.assert_close(x,y,rtol=0,atol=0)
    assert a.report(config)==b.report(config)


def test_decoder_hot_path_has_no_dynamic_boolean_gathers_and_flags_run_once(config,monkeypatch):
    batch = collate_audio([training_item(0,config,monkeypatch)],config=config)
    model = DecoderTrainer(config)
    conditions = model.produce(batch)
    keys = model.decoder.prepare_audio(Memory(conditions.audio.values,conditions.audio.beats,conditions.audio.valid))
    chunk = batch.chunks[0]
    mem = model.memory(conditions,keys,chunk.indices,chunk.audio_layout)
    with patch.object(model.decoder.heads.flags_adapter,'forward',wraps=model.decoder.heads.flags_adapter.forward) as flags:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            out = model.decoder(chunk.tokens,chunk.next_types,mem)
            terms = loss(model.decoder.heads,out,chunk.labels,distributions=decoder_time_distributions(config),geometry=distances())
        assert flags.call_count==1
    assert 'aten::nonzero' not in {e.key for e in profile.key_averages()}
    terms.objective(physical_weight=.1,judge_weight=.2,trajectory_weight=.3).backward()
    assert model.decoder.layers[0].audio_attn.key.weight.grad.abs().sum()>0
    assert model.decoder.condition.plan[0].weight.grad.abs().sum()>0
    assert all(p.grad is None for m in (model.encoder,model.draft,model.planner) for p in m.parameters())


def test_parsed_onsets_preserve_window_labels_without_full_chart_replay(monkeypatch):
    import random
    from maidata.chart.windows import window_sequence
    from maidata.decoder.windows import real_onsets,build_window
    tokens = [Moment(0),Note.make('tap','A1'),Moment(6000),Note.make('touch','C')]
    record,supports,geo = fields(monkeypatch,tokens,audio_end=12.,bpm=60)
    chart = ChartSupports(record,supports,geo)
    onsets = real_onsets(chart.tokens,chart.timing,sequence_start_tick=0)
    for start in (-96,0,96*4,96*6,96*14):
        for drop in (False,True):
            expected = window_sequence(chart.tokens,chart.timing,start_tick=start,end_tick=start+192,
                audio_after_beats=1,audio_end_seconds=12.,history_present=not drop)
            with patch('maidata.decoder.windows.real_onsets',side_effect=AssertionError('replayed full chart')):
                actual = build_window(chart.tokens,chart.timing,sequence_start_tick=0,start_tick=start,end_tick=start+192,
                    audio_after_beats=1,audio_end_seconds=12.,history_beats=4,history_dropout=float(drop),
                    training=True,rng=random.Random(0),onsets=onsets)
            assert actual.inputs[actual.boundary_index:]==expected.tokens
            assert actual.targets[-1]==expected.terminal


def test_dense_embedding_and_gates_match_partitioned_reference_gradients():
    from maidata.nn.position import sinusoidal
    from maidata.decoder.blocks import MMoE
    from test_decoder_network import inputs
    model = build_decoder(decoder_config()).eval()
    tokens,*_ = inputs(model)
    embedding = model.embedding
    actual = embedding(tokens)
    expected = torch.zeros_like(actual)
    # Per-token reference follows the field quotas directly, with NA contributing
    # no embedding and without the dense implementation's masks or gathers.
    for b,row in enumerate(tokens.types):
        for i,kind in enumerate(row):
            if not tokens.valid[b,i]:
                continue
            if kind==1:
                k,p = tokens.kind[b,i],tokens.pad[b,i]
                pad = embedding.touch(p) if k in (2,3) else embedding.button(p)
                parts = [embedding.kind(k),pad]
                for f,table in enumerate(embedding.flags):
                    value = tokens.flags[b,i,f]
                    parts.append(table(value) if value>=0 else table.weight.new_zeros(table.embedding_dim))
                content = torch.cat(parts)
            elif kind==2:
                value = tokens.track_break[b,i]
                flag = embedding.track_break(value) if value>=0 else embedding.track_break.weight.new_zeros(embedding.track_break.embedding_dim)
                content = torch.cat((embedding.route(tokens.route[b,i]),flag))
            else:
                content = sinusoidal(tokens.time[b,i],embedding.dim,embedding.base,embedding.scale)
            expected[b,i] = content+embedding.input_type(kind)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    def same_gradients(a,b,params):
        ga = torch.autograd.grad(a.square().sum(),params,retain_graph=True,allow_unused=True)
        gb = torch.autograd.grad(b.square().sum(),params,allow_unused=True)
        for x,y in zip(ga,gb):
            if x is None or y is None:
                assert x is y or (x if x is not None else y).count_nonzero()==0
            else:
                torch.testing.assert_close(x,y,atol=2e-5,rtol=2e-5)
    same_gradients(actual,expected,tuple(embedding.parameters()))
    gate = MMoE(24,17,3,0.)
    x = torch.randn(2,9,24,requires_grad=True)
    types = torch.arange(18).reshape(2,9)%5
    scores = x.new_zeros(2,9,3)
    for kind,head in enumerate(gate.gates):
        chosen = types==kind
        scores[chosen] = head(x[chosen])
    values = torch.stack([e(x) for e in gate.experts],-2)
    expected = (scores.softmax(-1)[...,None]*values).sum(-2)
    actual = gate(x,types)
    torch.testing.assert_close(actual,expected)
    same_gradients(actual,expected,(*gate.parameters(),x))
