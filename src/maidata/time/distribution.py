"""Differentiable complete-field probabilities for the single CONT chain.

This optional Torch adapter owns no learned parameters and does not query rules.
Data-only maidata.time imports remain usable without Torch.
"""
import torch

from .discrete import CONTINUE, TIME_CLASSES


def masked_log_probs(logits, basic):
    """FP32 normalization over basic choices, including safe padded depths."""
    if logits.shape != basic.shape or logits.shape[-1] != TIME_CLASSES:
        raise ValueError('expected matching [...,385] logits and basic masks')
    basic = basic.bool()
    has = basic.any(-1, keepdim=True)
    with torch.autocast(device_type=logits.device.type, enabled=False):
        values = logits.float().masked_fill(~basic, -torch.inf)
        values = torch.where(has, values, torch.zeros_like(values))
        return torch.log_softmax(values, -1).masked_fill(~basic, -torch.inf)


def complete_field_log_probabilities(logits, basic, continuation_closes):
    """Return all short leaves and first-cross-window CONT leaves, same shape.

Input shape is ``[...,depth,385]``. Padded depths have all-false basic masks.
No probability is discarded after the GT depth: callers must provide every
reachable depth, ending with no CONT or a window-closing CONT.
    """
    if logits.ndim < 2 or continuation_closes.shape != logits.shape[:-1]:
        raise ValueError('expected [...,depth,385] and [...,depth] closure flags')
    logp = masked_log_probs(logits, basic)
    has = basic.any(-1)
    next_has = torch.cat((has[..., 1:], torch.zeros_like(has[..., :1])), -1)
    if (has & basic[..., CONTINUE] & ~continuation_closes.bool() & ~next_has).any():
        raise ValueError('incomplete CONT chain would discard uncomputed tail probability')
    cont = logp[..., CONTINUE].masked_fill(continuation_closes.bool(), -torch.inf)
    prefix = torch.cat((torch.zeros_like(cont[..., :1]), torch.cumsum(cont[..., :-1], -1)), -1)
    leaves = logp + prefix[..., None]
    keep = torch.ones_like(basic, dtype=torch.bool)
    keep[..., CONTINUE] = continuation_closes.bool()
    return leaves.masked_fill(~keep, -torch.inf)


def time_path_nll(logits, basic, target, target_steps):
    """Exact-path CE sum or final censored set NLL, once per original field."""
    if target.shape != logits.shape or target_steps.shape != logits.shape[:-1]:
        raise ValueError('time target shapes do not match the query tree')
    active = target_steps.bool()
    selected = target.bool() & basic.bool()
    if (active & ~selected.any(-1)).any():
        raise ValueError('active time target outside basic support')
    logp = masked_log_probs(logits, basic)
    values = logp.masked_fill(~selected, -torch.inf)
    # Avoid differentiating logsumexp(-inf,...,-inf) on unlabelled depths.
    values = torch.where(active[..., None], values, torch.zeros_like(values))
    return -torch.logsumexp(values, -1).masked_fill(~active, 0).sum(-1)


def physical_categories(leaf_logp, penalty):
    """Stable four-category legal complement over the entire field tree.

The returned ``[...,4]`` is one original-field contribution. Window-closing
leaves must have zero penalty; CONT itself never receives a blanket penalty.
    """
    if penalty.shape != (*leaf_logp.shape, 4):
        raise ValueError('expected four penalty bits for every complete leaf')
    values = leaf_logp.float().flatten(-2)
    forbidden = penalty.bool().flatten(-3, -2)
    legal = ~forbidden & torch.isfinite(values)[..., None]
    has = legal.any(-2)
    logvalues = values[..., None].expand_as(legal).masked_fill(~legal, -torch.inf)
    logvalues = torch.where(has[..., None, :], logvalues, torch.zeros_like(logvalues))
    complement = torch.logsumexp(logvalues, -2)
    floor = torch.log(values.new_tensor(torch.finfo(torch.float32).eps))
    return -torch.where(has, complement, floor).clamp(min=floor, max=0)


def physical_mean(categories):
    """Existing weights and fixed-four reduction; field denominator is external."""
    return (categories * categories.new_tensor((1., 1., 4., 1.))).sum(-1) / 4
