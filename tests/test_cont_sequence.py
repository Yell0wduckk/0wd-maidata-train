from fractions import Fraction
import torch
import pytest
from maidata.chart import Moment, Note, HoldDuration, SlideDuration
from maidata.decoder.sequence import expand_inputs, CONT_TYPES, ModelTime
from maidata.time import TimingMap, duration_end
from maidata.decoder_training.sequence_cache import step_penalty


@pytest.mark.parametrize('ticks',[1,384,385,768,769])
@pytest.mark.parametrize('kind',['MOMENT','HOLD_DURATION','SLIDE_DURATION'])
def test_exact_sequence_roundtrip(ticks,kind):
    timing=TimingMap([(0,60),(Fraction(4),120)])
    if kind=='HOLD_DURATION' and ticks==1:
        return
    token=(Moment(timing.delta_ms(0,ticks)) if kind=='MOMENT' else
           (HoldDuration if kind=='HOLD_DURATION' else SlideDuration)(ticks))
    beat=ticks/96 if kind=='MOMENT' else 0
    sequence=expand_inputs([token],[beat],timing)
    parts=sequence['tokens']
    assert len(parts)==(ticks-1)//384+1
    assert all(t.type in CONT_TYPES for t in parts[:-1])
    if kind=='MOMENT':
        assert sum(t.time_value for t in parts)==pytest.approx(timing.delta_ms(0,ticks)/1000)
    else:
        assert sum(t.time_value for t in parts)*96==pytest.approx(ticks)
    assert sequence['beats'][-1]==beat
    for i,b in enumerate(sequence['beats'][:-1],1):
        expected=(i*4 if kind=='MOMENT' else timing.second_to_beat(duration_end(timing,0,i*384,slide=kind=='SLIDE_DURATION')))
        assert b==pytest.approx(expected)


def test_boundary_not_split_and_no_note_expansion():
    timing=TimingMap([(0,60)])
    seq=expand_inputs([Moment(8000),Note.make('tap','A1')],[8.,8.],timing,boundary_index=0)
    assert len(seq['tokens'])==2
    assert not isinstance(seq['tokens'][0],ModelTime)


def test_cont_default_slide_coordinate_has_no_second_launch_offset():
    from maidata.decoder.batching import pack_tokens
    batch=pack_tokens([[ModelTime('SLIDE_CONT',4.)]],[[12.]],device='cpu')
    assert batch.slide_beats.item()==12.


def test_cont_penalty_is_existential_per_category_and_close_is_zero():
    group=dict(allowed=torch.zeros(1,2,385,dtype=torch.bool),
        penalty=torch.zeros(1,2,385,4,dtype=torch.bool),closes=torch.zeros(1,2,dtype=torch.bool))
    group['allowed'][0,1,:2]=True
    group['penalty'][0,1,:2,0]=True
    group['penalty'][0,1,0,1]=True
    assert step_penalty(group,0,0)[384].tolist()==[True,False,False,False]
    group['closes'][0,0]=True
    assert not step_penalty(group,0,0)[384].any()


def test_gt_prefix_loss_sums_ce_and_means_physical_once_per_field():
    from types import SimpleNamespace
    from maidata.decoder_training.supervision import Prepared,loss
    from maidata.decoder.losses import physical_categories
    logits=torch.randn(2,385,requires_grad=True)
    allowed=torch.ones(2,385,dtype=torch.bool)
    target=torch.zeros_like(allowed);target[0,384]=True;target[1,0]=True
    penalty=torch.zeros(2,385,4,dtype=torch.bool)
    penalty[0,:100,0]=True;penalty[1,50:200,2]=True
    group=dict(indices=torch.tensor([[0,0],[0,1]]),allowed=allowed,target=target,penalty=penalty,
        main=torch.ones(2,dtype=torch.bool),scale=torch.full((2,),.5),args=torch.empty(2,0,dtype=torch.long),
        field_id=torch.zeros(2,dtype=torch.long),has_physical=True)
    labels=Prepared({'HOLD_DURATION':group},dict(main=1,physical=1,judge=0,trajectory=0),
        dict(indices=torch.empty(0,2,dtype=torch.long),args=torch.empty(0,2,dtype=torch.long)))
    output=SimpleNamespace(hidden=torch.ones(1,2,4,requires_grad=True))
    heads=SimpleNamespace(time_logits=lambda hidden,kind:logits)
    terms=loss(heads,output,labels,distributions=None,geometry=None)
    lp=logits.log_softmax(-1)
    torch.testing.assert_close(terms.main.total,-lp[0,384]-lp[1,0])
    torch.testing.assert_close(terms.physical.total,physical_categories(lp,penalty).mean(0))
    assert terms.main.count==terms.physical.count==1
    (terms.main.total+terms.physical.total.sum()).backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum(1).min()>0
