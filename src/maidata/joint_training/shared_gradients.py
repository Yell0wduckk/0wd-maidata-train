"""Release M chunk graphs, then traverse the shared E/D graph once."""
from dataclasses import replace

import torch


class SharedJointGradients:
    """Bridge audio K/V and live Z; GT plan never enters the bridge."""
    def __init__(self, audio_kv, z):
        self.real_audio = audio_kv
        self.real_z = z
        self.memory = tuple(replace(kv,
            key=kv.key.detach().requires_grad_(kv.key.requires_grad),
            value=kv.value.detach().requires_grad_(kv.value.requires_grad)) for kv in audio_kv)
        self.z = replace(z, values=z.values.detach().requires_grad_(z.values.requires_grad))
        self.gradients = {}

    def _leaves(self):
        for index, kv in enumerate(self.memory):
            for name in ('key', 'value'):
                yield ('audio', index, name), getattr(kv, name), getattr(self.real_audio[index], name)
        yield ('z', 0, 'values'), self.z.values, self.real_z.values

    def collect(self):
        """Accumulate chunk contributions in FP32 without holding M graphs."""
        for key, leaf, _ in self._leaves():
            if leaf.grad is not None:
                gradient = leaf.grad.detach().float()
                if key in self.gradients:
                    self.gradients[key].add_(gradient)
                else:
                    self.gradients[key] = gradient.clone()
                leaf.grad = None

    def backward(self, auxiliary):
        """Merge R/P and M branches in a single E/D autograd traversal."""
        tensors, gradients = [auxiliary], [None]
        for key, _, real in self._leaves():
            if real.requires_grad and key in self.gradients:
                tensors.append(real)
                gradients.append(self.gradients[key].to(real.dtype))
        torch.autograd.backward(tensors, gradients)
