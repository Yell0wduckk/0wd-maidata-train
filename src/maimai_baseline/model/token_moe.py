"""Token-wise dense MoE with observational, support-weighted diagnostics."""
from contextlib import contextmanager
import torch
from torch import nn


class DenseTokenMoE(nn.Module):
    def __init__(self, width, dense_hidden, *, activation='gelu', dropout=0., salt=0):
        super().__init__()
        if dense_hidden % 2:
            raise ValueError('MoE requires an even dense hidden width')
        # Call after the original FFN construction. Preserve its RNG consumption
        # and isolate every newly introduced parameter (including CUDA RNG).
        with torch.random.fork_rng(devices=[]):
            seed = int(torch.randint(0, 2**62, ())) ^ int(salt) ^ 0x4D4F45
            torch.random.default_generator.manual_seed(seed)
            act = {'gelu': nn.GELU, 'silu': nn.SiLU}[activation]
            self.experts = nn.ModuleList(nn.Sequential(
                nn.Linear(width, dense_hidden // 2), act(), nn.Dropout(dropout),
                nn.Linear(dense_hidden // 2, width)) for _ in range(4))
            self.gate = nn.Linear(width, 4)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
        self.collect_metrics = False
        self.metric_sums = {}

    def forward(self, x, valid=None):
        if valid is None:
            valid = torch.ones(x.shape[:-1], dtype=torch.bool, device=x.device)
        if valid.shape != x.shape[:-1] or valid.dtype != torch.bool:
            raise ValueError('MoE validity must match token dimensions')
        h = x[valid]
        # Even zero-token batches retain a differentiable graph and never read pad.
        values = torch.stack([expert(h) for expert in self.experts], dim=1)
        weights = self.gate(h).float().softmax(-1)
        mixed = (values * weights.to(values.dtype).unsqueeze(-1)).sum(1)
        output = x.new_zeros(x.shape, dtype=mixed.dtype)
        output[valid] = mixed
        if self.collect_metrics and len(h):
            w = weights.detach()
            current = dict(count=len(h), entropy=float(-(w * w.clamp_min(1e-12).log()).sum()),
                           **{f'expert_{i}': float(w[:, i].sum()) for i in range(4)})
            for key, value in current.items():
                self.metric_sums[key] = self.metric_sums.get(key, 0.) + value
        return output


@contextmanager
def capture_moe_metrics(model, enabled):
    """Collect across all calls in one step; restore flags even on exceptions."""
    modules = [(name, module, module.collect_metrics, module.metric_sums)
               for name, module in model.named_modules() if isinstance(module, DenseTokenMoE)]
    result = {}
    for _, module, _, _ in modules:
        module.collect_metrics, module.metric_sums = enabled, {}
    try:
        yield result
    finally:
        for name, module, flag, previous in modules:
            sums = module.metric_sums
            if enabled and sums.get('count', 0):
                result[name] = {k: v if k == 'count' else v / sums['count'] for k, v in sums.items()}
            module.collect_metrics, module.metric_sums = flag, previous
