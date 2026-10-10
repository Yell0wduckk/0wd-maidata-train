from copy import deepcopy

import torch

import maidata.audio.blocks as blocks
from maidata.audio.blocks import SharedConformerBlock, StridedTransition
from maidata.audio.types import ScaleMemory


def make_block():
    return SharedConformerBlock(
        d_model=12, heads=3, ffn_dim=19, temporal_kernel_size=3,
        attention_radius=0.75, dropout=0.0, rope_base=1000.0,
        rope_position_scale=0.6,
    )


def memory(length, *, batch=1):
    return ScaleMemory(
        torch.randn(batch, length, 12),
        (torch.arange(length).float() * 0.25 - 1.5)[None].expand(batch, -1),
        torch.ones(batch, length, dtype=torch.bool),
    )


def count_calls(module, names):
    calls = {name: 0 for name in names}
    handles = []
    for name in names:
        def count(_module, _inputs, _output, key=name):
            calls[key] += 1
        handles.append(getattr(module, name).register_forward_hook(count))
    return calls, handles


def test_synchronous_reading_reuses_projected_inputs_and_skips_last_slot_update(monkeypatch):
    torch.manual_seed(4)
    block = make_block().eval()
    local, slots, song = memory(7), memory(3), memory(5)
    song_kv = block.project_song(song)
    captured = {}

    def local_attention(q, k_local, v_local, k_slots, v_slots, *_args):
        captured['local'] = k_local, v_local, k_slots, v_slots
        return v_slots.mean(-2, keepdim=True).expand_as(q)

    def global_attention(q, k_song, v_song, k_local, v_local, *_args):
        captured['global'] = k_local, v_local
        return v_local.mean(-2, keepdim=True).expand_as(q)

    monkeypatch.setattr(blocks, 'local_joint_attention', local_attention)
    monkeypatch.setattr(blocks, 'global_joint_attention', global_attention)
    names = ('ffn1', 'ffn2', 'attention_norm', 'q', 'k', 'v', 'out', 'final_norm', 'convolution')
    calls, handles = count_calls(block, names)
    full_local, full_slots = block(local, slots, song_kv, song.valid, update_slots=True)
    assert captured['global'][0] is captured['local'][0]
    assert captured['global'][1] is captured['local'][1]
    assert calls == dict(ffn1=2, ffn2=2, attention_norm=2, q=2, k=2, v=2,
                         out=2, final_norm=2, convolution=1)
    calls.update({name: 0 for name in calls})
    last_local, last_slots = block(local, slots, None, None, update_slots=False)
    assert calls == dict(ffn1=2, ffn2=1, attention_norm=2, q=1, k=2, v=2,
                         out=1, final_norm=1, convolution=1)
    assert last_slots is slots
    torch.testing.assert_close(full_local.values, last_local.values, rtol=0, atol=0)
    assert not torch.allclose(full_slots.values, slots.values)
    for handle in handles:
        handle.remove()
    other = make_block()
    for name in ('q', 'k', 'v', 'out', 'ffn1', 'ffn2', 'attention_norm', 'final_norm'):
        assert getattr(block, name) is not getattr(other, name)


def test_song_projection_uses_shared_ln_kv_without_ffn_and_keeps_gradients():
    torch.manual_seed(5)
    block = make_block().eval()
    song = memory(5)
    song.valid[:, -1] = False
    song.values[:, -1] = float('nan')
    song.values.requires_grad_()
    calls, handles = count_calls(block, ('ffn1', 'attention_norm', 'q', 'k', 'v', 'out'))
    key, value = block.project_song(song)
    assert calls == dict(ffn1=0, attention_norm=1, q=0, k=1, v=1, out=0)
    assert key.shape == value.shape == (1, 3, 5, 4)
    assert torch.isfinite(key).all() and torch.isfinite(value).all()
    assert not key[:, :, -1].count_nonzero() and not value[:, :, -1].count_nonzero()
    (key.square().sum() + value.square().sum()).backward()
    assert song.values.grad[:, :-1].abs().sum() > 0
    assert not song.values.grad[:, -1].count_nonzero()
    for module in (block.attention_norm, block.k, block.v):
        assert module.weight.grad is not None and module.weight.grad.abs().sum() > 0
    assert block.ffn1.linear1.weight.grad is None and block.q.weight.grad is None
    for handle in handles:
        handle.remove()


def test_both_branches_backpropagate_through_the_same_registered_parameters():
    torch.manual_seed(6)
    block = make_block().train()
    local, slots, song = memory(7), memory(3), memory(5)
    for item in (local, slots, song):
        item.values.requires_grad_()
    local_out, slots_out = block(
        local, slots, block.project_song(song), song.valid, update_slots=True
    )
    loss = (local_out.values * torch.randn_like(local_out.values)).sum()
    loss = loss + (slots_out.values * torch.randn_like(slots_out.values)).sum()
    loss.backward()
    for parameter in block.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    for module in (block.q, block.k, block.v, block.out, block.ffn1.linear1,
                   block.ffn2.linear1, block.convolution.depthwise):
        assert module.weight.grad.abs().sum() > 0
    for item in (local, slots, song):
        assert item.values.grad is not None and item.values.grad.abs().sum() > 0


def test_padding_cannot_change_valid_outputs_and_all_empty_audio_is_zero():
    torch.manual_seed(7)
    block = make_block().eval()
    local, slots, song = memory(7, batch=2), memory(3, batch=2), memory(5, batch=2)
    for item in (local, slots, song):
        item.valid[0, -1] = False
        item.valid[1] = False
    expected = block(local, slots, block.project_song(song), song.valid, update_slots=True)
    dirty = deepcopy((local, slots, song))
    for item in dirty:
        item.values[~item.valid] = float('nan')
        item.beats = item.beats.clone()
        item.beats[~item.valid] = float('nan')
    actual = block(dirty[0], dirty[1], block.project_song(dirty[2]), dirty[2].valid, update_slots=True)
    for clean, changed in zip(expected, actual):
        assert torch.isfinite(changed.values).all()
        assert not changed.values[~changed.valid].count_nonzero()
        torch.testing.assert_close(clean.values, changed.values, rtol=0, atol=0)


def test_slot_attention_out_bias_is_masked_when_no_allowed_key_exists(monkeypatch):
    block = make_block().eval()
    local, slots, song = memory(5), memory(3), memory(4)
    local.valid[:] = False
    song.valid[:] = False
    with torch.no_grad():
        block.out.bias.fill_(9.0)
    seen = []
    original = block._attention_residual

    def inspect(attended, usable):
        value = original(attended, usable)
        seen.append(value)
        return value

    monkeypatch.setattr(block, '_attention_residual', inspect)
    block(local, slots, block.project_song(song), song.valid, update_slots=True)
    assert not seen[0].count_nonzero() and not seen[1].count_nonzero()


def test_stride_transition_keeps_center_validity_masked_coordinates_and_learned_branch():
    torch.manual_seed(8)
    transition = StridedTransition(4, 0.0).eval()
    local = ScaleMemory(
        torch.randn(1, 5, 4),
        torch.tensor([[-1.0, -0.5, 0.0, 0.5, 1.0]]),
        torch.tensor([[True, True, False, True, True]]),
    )
    local.values.requires_grad_()
    output = transition(local)
    assert output.values.shape == (1, 3, 4)
    assert output.valid.tolist() == [[True, False, True]]
    torch.testing.assert_close(output.beats, torch.tensor([[-0.75, 0.0, 0.75]]))
    assert not output.values[:, 1].count_nonzero()
    pooled_only = deepcopy(transition)
    with torch.no_grad():
        pooled_only.project.weight.zero_()
        pooled_only.project.bias.zero_()
    assert not torch.allclose(output.values, pooled_only(local).values)
    (output.values * torch.randn_like(output.values)).sum().backward()
    assert transition.depthwise.weight.grad.abs().sum() > 0
    assert transition.project.weight.grad.abs().sum() > 0
    assert not local.values.grad[~local.valid].count_nonzero()

    dirty_values = local.values.detach().clone()
    dirty_values[~local.valid] = float('nan')
    changed = transition(ScaleMemory(dirty_values, local.beats, local.valid))
    torch.testing.assert_close(changed.values, output.values.detach(), rtol=0, atol=0)
    empty = transition(ScaleMemory(dirty_values, local.beats, torch.zeros_like(local.valid)))
    assert torch.isfinite(empty.values).all() and not empty.values.count_nonzero()
