"""Accumulate M-chunk audio gradients, then backpropagate through E once."""
from dataclasses import replace
import torch


class SharedAudioGradients:
    def __init__(self, audio_kv):
        self.real = audio_kv
        self.memory = tuple(replace(kv,
            key=kv.key.detach().requires_grad_(kv.key.requires_grad),
            value=kv.value.detach().requires_grad_(kv.value.requires_grad)) for kv in audio_kv)
        self.gradients = {}

    def collect(self):
        # Sum BF16 chunk gradients in FP32 without retaining the M graphs.
        for index, kv in enumerate(self.memory):
            for name in ('key', 'value'):
                leaf = getattr(kv, name)
                if leaf.grad is not None:
                    key = (index, name)
                    gradient = leaf.grad.detach().float()
                    if key in self.gradients:
                        self.gradients[key].add_(gradient)
                    else:
                        self.gradients[key] = gradient.clone()
                    leaf.grad = None

    def backward(self):
        tensors, gradients = [], []
        for (index, name), gradient in self.gradients.items():
            tensor = getattr(self.real[index], name)
            if tensor.requires_grad:
                tensors.append(tensor)
                gradients.append(gradient.to(tensor.dtype))
        if tensors:
            torch.autograd.backward(tensors, gradients)
