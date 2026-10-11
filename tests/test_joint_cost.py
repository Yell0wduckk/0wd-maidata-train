"""Cost reports use the same complete-group CPU batches as joint updates."""
import json

import pytest
import torch

from maidata.joint_training.cost import cpu_copy, workload
from maidata.joint_training.data import collate_audio
from test_joint_data import config, encoder_config, joint_item


@pytest.mark.parametrize('packed', [False, True])
def test_original_batch_workload_report_covers_shared_and_repeated_work(config, monkeypatch, packed):
    items = [joint_item(0, config, monkeypatch, remaster=True),
             joint_item(1, config, monkeypatch)]
    batch = collate_audio(items, config=config, packed=packed)
    if packed:
        batch = batch.to('cpu')
    result = workload([batch])
    assert result['audio_groups'] == [batch.audio_variants]
    assert result['auxiliary_base_windows'] == 6
    assert result['decoder_occurrences'] == 9
    assert result['unique_decoder_windows'] == 6
    assert result['local_encoder_windows'] == 4
    assert result['coverage_encoder_windows'] == 2
    # Window validity counts local input samples, including reused context and
    # the real tail. SongInputs.sample_valid is a different representation.
    assert result['valid_audio_input_samples'] == batch.audio_windows.valid.sum().item() > 0
    target = batch.reconstruction
    assert result['actual_reconstruction_queries'] == (
        target.queries.valid.sum().item() + target.valid.sum().item()*target.moment_counts.shape[1])
    assert result['valid_m_rows'] == sum(chunk.tokens.valid.sum().item() for chunk in batch.chunks) > 0
    assert result['padded_m_rows'] >= result['valid_m_rows']
    assert result['m_chunks'] == len(batch.chunks) > 0
    assert result['m_batch_costs'] == [batch.budget_cost]
    # The diagnostic is written with strict JSON serialization by the runner.
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    accumulated = workload([batch, batch])
    for name, value in result.items():
        assert accumulated[name] == (value + value if isinstance(value, list) else 2*value)


def test_cost_cpu_copy_is_detached_and_independent_for_nested_optimizer_state():
    parameter = torch.nn.Parameter(torch.tensor([1.]))
    optimizer = torch.optim.AdamW([parameter], lr=.01, fused=False)
    parameter.square().sum().backward()
    optimizer.step()
    original = dict(model={'weight': parameter}, optimizer=optimizer.state_dict(),
                    extras=[(parameter.grad,)])
    snapshot = cpu_copy(original)
    assert isinstance(snapshot['extras'], list)
    assert isinstance(snapshot['extras'][0], tuple)
    parameter.data.add_(2.)
    parameter.grad.add_(3.)
    for value in optimizer.state[parameter].values():
        value.add_(4.)
    torch.testing.assert_close(snapshot['model']['weight'], torch.tensor([.9899]))
    torch.testing.assert_close(snapshot['extras'][0][0], torch.tensor([2.]))
    state = snapshot['optimizer']['state'][0]
    torch.testing.assert_close(state['step'], torch.tensor(1.))
    torch.testing.assert_close(state['exp_avg'], torch.tensor([.2]))
    torch.testing.assert_close(state['exp_avg_sq'], torch.tensor([.004]))
    for tensor in (snapshot['model']['weight'], snapshot['extras'][0][0], *state.values()):
        assert tensor.device.type == 'cpu' and not tensor.requires_grad
