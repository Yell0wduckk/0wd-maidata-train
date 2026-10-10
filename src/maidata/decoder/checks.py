"""Preserve tensor value contracts without a CUDA-to-host scalar per check."""
import torch


def require(condition, message):
    if condition.device.type == 'cuda':
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise ValueError(message)
