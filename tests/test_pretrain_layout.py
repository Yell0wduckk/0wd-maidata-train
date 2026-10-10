"""Prepared CPU layouts preserve the public tensor path and packed IPC."""
from copy import deepcopy
from dataclasses import fields, is_dataclass, replace
from unittest.mock import patch
import pickle

import torch

from test_pretrain_runtime import config, item
from maidata.training.data import collate_audio
from maidata.training.network import Pretrainer
from maidata.training.transport import pack_batch
from maidata.training.pretrain import weighted_loss


def tensors(x):
    if isinstance(x, torch.Tensor):
        yield x
    elif is_dataclass(x):
        for f in fields(x):
            yield from tensors(getattr(x, f.name))
    elif isinstance(x, dict):
        for v in x.values():
            yield from tensors(v)
    elif isinstance(x, (tuple, list)):
        for v in x:
            yield from tensors(v)


def test_layout_pack_and_prepared_chain_match_standalone_gradients(config):
    batch = collate_audio([item(0, config, True), item(1, config)], config=config)
    packed = pickle.loads(pickle.dumps(pack_batch(batch)))
    assert len(packed.buffers) == 3
    copied = packed.to('cpu')
    for a, b in zip(tensors(batch), tensors(copied), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    standalone = replace(batch, audio_windows=replace(batch.audio_windows, layout=None),
                         draft_inputs=replace(batch.draft_inputs, valid_indices=None), layouts=None)
    model = Pretrainer(config).eval()
    reference = deepcopy(model)
    # Prepared production must not call a CPU fallback during model execution.
    with (patch('maidata.audio.layout.prepare_encoder_layout', side_effect=AssertionError('CPU fallback')),
         patch('maidata.draft.layout.prepare_p0', side_effect=AssertionError('CPU fallback')),
         patch('maidata.draft.layout.prepare_selection', side_effect=AssertionError('CPU fallback'))):
        a, outputs = model(copied)
    b, expected = reference(standalone)
    for x, y in zip(tensors(outputs), tensors(expected), strict=True):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    weighted_loss(a, config).backward()
    weighted_loss(b, config).backward()
    for (name, x), (other, y) in zip(model.named_parameters(), reference.named_parameters(), strict=True):
        assert name == other
        if x.grad is not None:
            torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)


def test_local_joint_attention_uses_one_sdpa_across_windows():
    from maidata.audio.attention import local_joint_attention
    from torch.nn import functional as F
    n = 133
    values = [torch.randn(3, 2, n, 4) for _ in range(3)] + [torch.randn(3, 2, 5, 4) for _ in range(2)]
    beats = torch.arange(n).float()[None].expand(3, -1)/7-2
    valid = torch.ones(3, n, dtype=torch.bool)
    valid[1, 5:8] = False
    with patch.object(F, 'scaled_dot_product_attention', wraps=F.scaled_dot_product_attention) as sdpa:
        local_joint_attention(*values, beats, valid, torch.ones(3, 5, dtype=torch.bool), .7, 0.)
    assert sdpa.call_count == 1
