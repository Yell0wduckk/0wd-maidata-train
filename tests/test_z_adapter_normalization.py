"""Z scale bounds, live gradients, legacy behavior and explicit full-state upgrade."""
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from maidata.decoder.attention import ConditionAdapter
from maidata.decoder.types import Memory
from maidata.joint_training import checkpoint, recipe
from maidata.joint_training.network import JointTrainer
from test_decoder_training import config, encoder_config
from test_joint_runtime import joint_config, assert_state_equal


def test_normalizes_after_projection_and_source_without_changing_plan_or_state():
    torch.manual_seed(31)
    legacy = ConditionAdapter(10, 24, True, 1e-5)
    normalized = ConditionAdapter(10, 24, True, 1e-5, True)
    normalized.load_state_dict(legacy.state_dict(), strict=True)
    with torch.no_grad():
        normalized.z.weight.mul_(10000)
        normalized.source.weight[0].mul_(1000)
    z_values = torch.randn(2, 4, 10, requires_grad=True)
    valid = torch.tensor([[1,1,0,0],[1,1,1,0]], dtype=torch.bool)
    z = Memory(z_values, torch.arange(4).float()[None].expand(2,-1), valid)
    plan = torch.randn(2,12,requires_grad=True)
    plan_valid = torch.tensor([True,False])
    result = normalized(z, plan, plan_valid)
    expected = torch.nn.functional.layer_norm(
        normalized.z(z_values.masked_fill(~valid[...,None],0)) + normalized.source.weight[0], (24,), eps=1e-5)
    torch.testing.assert_close(result.values[:,:4], expected)
    torch.testing.assert_close(result.values[:,-1], legacy(z,plan,plan_valid).values[:,-1])
    torch.testing.assert_close(result.values[:,:4][valid].square().mean(-1), torch.ones(5))
    assert normalized.state_dict().keys() == legacy.state_dict().keys()
    assert isinstance(normalized.z_norm, nn.LayerNorm) and not list(normalized.z_norm.parameters())
    torch.testing.assert_close(result.valid[:,:4],valid)
    torch.testing.assert_close(result.beats[:,:4],z.beats)
    (result.values[:,:4]*torch.arange(24)[None,None]*valid[...,None]).sum().backward()
    for value in (z_values,normalized.z.weight,normalized.z.bias,normalized.source.weight):
        assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum()>0
    assert not torch.count_nonzero(z_values.grad[~valid])


def test_disabled_z_does_not_read_or_backpropagate_z_even_with_normalization():
    adapter = ConditionAdapter(10,24,False,1e-5,True)
    values = torch.randn(1,4,10,requires_grad=True)
    result = adapter(Memory(values,torch.zeros(1,4),torch.ones(1,4,dtype=torch.bool)),
                     torch.ones(1,12),torch.ones(1,dtype=torch.bool))
    result.values.square().sum().backward()
    assert values.grad is None and adapter.z.weight.grad is None
    assert not result.valid[:,:4].any()


def test_explicit_normalization_upgrade_preserves_all_optimizer_scheduler_rng_and_weights(config,tmp_path):
    old = dict(joint_config(config,tmp_path/'old'), decoder_z_normalization=False)
    model = JointTrainer(old)
    opt = recipe.optimizer(model.parameters(),old)
    sched = recipe.scheduler(opt,[1]*6,old)
    for p in model.parameters():
        p.grad = torch.ones_like(p)*.01
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    loaders = [SimpleNamespace(generator=torch.Generator().manual_seed(49),_iterator=None)]
    saved = checkpoint.capture(model,opt,sched,epoch=1,step=1,config=old,
        binding={},source={'commit':'legacy'},steps=[1]*6,loaders=loaders,best=None,metrics={})
    new = dict(old,decoder_z_normalization=True)
    upgraded = JointTrainer(new)
    newopt = recipe.optimizer(upgraded.parameters(),new)
    newsched = recipe.scheduler(newopt,[1]*6,new)
    with pytest.raises(ValueError,match='architecture/representation'):
        checkpoint.restore(saved,upgraded,newopt,newsched,config=new,loaders=loaders)
    result = checkpoint.restore(saved,upgraded,newopt,newsched,config=new,loaders=loaders,
                                upgrade_z_normalization=True)
    assert result == (1,1,None)
    assert_state_equal(saved['model_state'],upgraded.state_dict())
    assert_state_equal(saved['optimizer_state'],newopt.state_dict())
    assert_state_equal(saved['scheduler_state'],newsched.state_dict())
    assert_state_equal(saved['rng'],checkpoint.rng_state())
    assert_state_equal(saved['loader_rng'][0],loaders[0].generator.get_state())
    changed = dict(new,decoder_dropout=.25)
    with pytest.raises(ValueError,match='architecture/representation'):
        checkpoint.restore(saved,upgraded,newopt,newsched,config=changed,loaders=loaders,
                           upgrade_z_normalization=True)
    with pytest.raises(ValueError,match='requires an unnormalized checkpoint'):
        checkpoint.restore(saved,model,opt,sched,config=old,loaders=loaders,
                           upgrade_z_normalization=True)
    del saved['recipe']['decoder_z_normalization']
    checkpoint.restore(saved,model,opt,sched,config=old,loaders=loaders)
    checkpoint.restore(saved,upgraded,newopt,newsched,config=new,loaders=loaders,
                       upgrade_z_normalization=True)


@pytest.mark.parametrize('normalized',[False,True])
def test_inference_reads_saved_normalization_and_legacy_missing_is_false(config,tmp_path,normalized):
    from maidata.inference.checkpoint import load_joint_models
    c = dict(joint_config(config,tmp_path/'unused'),decoder_z_normalization=normalized)
    model = JointTrainer(c)
    payload = dict(schema=checkpoint.SCHEMA,architecture=checkpoint.ARCHITECTURE,
                   recipe=checkpoint.recipe(c),model_state=model.state_dict(),completed_epochs=55,step=47788)
    if not normalized:
        del payload['recipe']['decoder_z_normalization']
    path = tmp_path/'joint.ckpt';torch.save(payload,path)
    loaded = load_joint_models(path,device='cpu')
    assert loaded.config['decoder_z_normalization'] is normalized
    assert isinstance(loaded.decoder.condition.z_norm,nn.LayerNorm if normalized else nn.Identity)
