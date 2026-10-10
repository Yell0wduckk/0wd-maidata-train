"""Main, geometric and first-responsibility losses have separate sum/count ledgers."""
from dataclasses import dataclass
import torch
from torch import nn


@dataclass(frozen=True)
class Reduction:
    total: torch.Tensor
    count: int

    def mean(self):
        return self.total / max(1, self.count)

    def __add__(self, other):
        return Reduction(self.total + other.total, self.count + other.count)


def nll(logp, target_mask, *, counted=True):
    if not target_mask.any():
        raise ValueError('target outside basic legal support')
    if not counted:
        return Reduction(logp[torch.isfinite(logp)].sum() * 0, 0)
    return Reduction(-torch.logsumexp(logp[target_mask], -1), 1)


def physical_categories(logp, forbidden):
    """One candidate's mass counted once per category, before any hard mask."""
    if forbidden.shape != (*logp.shape, 4) or forbidden.dtype != torch.bool:
        raise ValueError('expected four independent Physical supports')
    # Using legal complement avoids catastrophic q -> 1 cancellation. A finite
    # floor handles a category that forbids all basic choices without NaN grads.
    legal = ~forbidden & torch.isfinite(logp)[..., None]
    values = logp[..., None].expand_as(forbidden).masked_fill(~legal, -torch.inf)
    has = legal.any(-2)
    values = torch.where(has.unsqueeze(-2), values, torch.zeros_like(values))
    complement = torch.logsumexp(values, -2)
    floor = torch.log(logp.new_tensor(torch.finfo(torch.float32).eps))
    return -torch.where(has, complement, floor).clamp(min=floor, max=0)


def physical_mean(reduction):
    weights = reduction.total.new_tensor((1., 1., 4., 1.))
    return (reduction.mean() * weights).sum() / 4


class RouteGeometryLoss(nn.Module):
    def __init__(self, judge, trajectory):
        super().__init__()
        for name, value in (('judge', judge), ('trajectory', trajectory)):
            tensor = torch.as_tensor(value, dtype=torch.float32).detach().clone()
            if (tensor.shape != (71, 71) or not torch.isfinite(tensor).all() or (tensor < 0).any()
                    or tensor.diag().ne(0).any()
                    or not torch.allclose(tensor, tensor.T)):
                raise ValueError('Route geometry requires fixed nonnegative 71x71 distances')
            self.register_buffer(name, tensor)

    def forward(self, logp, target):
        p = logp.exp()
        return ((p * self.judge[target]).sum(-1), (p * self.trajectory[target]).sum(-1))


@dataclass(frozen=True)
class DecoderLoss:
    main: Reduction
    physical: Reduction  # four sums, ONE common valid-unit count
    judge: Reduction
    trajectory: Reduction

    def __add__(self, other):
        return DecoderLoss(*(getattr(self, key) + getattr(other, key)
                             for key in ('main', 'physical', 'judge', 'trajectory')))

    def objective(self, *, physical_weight, judge_weight, trajectory_weight):
        # Recipe-owned coefficients are required, never guessed by the network.
        return (self.main.mean() + physical_weight * physical_mean(self.physical)
                + judge_weight * self.judge.mean() + trajectory_weight * self.trajectory.mean())
