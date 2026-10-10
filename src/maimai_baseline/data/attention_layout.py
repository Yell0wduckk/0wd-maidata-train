"""Ephemeral CPU query indices and unique-song key spans, reused across layers."""
import torch


def key_spans(padding):
    if padding.device.type != 'cpu':
        raise ValueError('key spans must be prepared on CPU')
    spans = []
    for row in padding:
        valid = (~row).nonzero().flatten()
        spans.append((int(valid[0]), int(valid[-1]) + 1) if len(valid) else (0, 0))
    return tuple(spans)


def cross_layout(query_padding, key_padding, mapping, *, spans=None):
    if query_padding.device.type != 'cpu' or mapping.device.type != 'cpu':
        raise ValueError('cross query layout must be prepared on CPU')
    spans = key_spans(key_padding) if spans is None else spans
    if mapping.shape != (query_padding.shape[0],) or ((mapping < 0) | (mapping >= len(spans))).any():
        raise ValueError('invalid unique-song query mapping')
    valid = ~query_padding
    indices = tuple((valid & (mapping == audio)[:, None]).flatten().nonzero().flatten()
                    for audio in range(len(spans)))
    present = torch.tensor([hi > lo for lo, hi in spans], dtype=torch.bool)[mapping, None] & valid
    return dict(indices=indices, spans=spans, present=present)


def to_device(layout, device):
    if isinstance(layout, torch.Tensor):
        return layout.to(device)
    if isinstance(layout, dict):
        return {k: to_device(v, device) for k, v in layout.items()}
    if isinstance(layout, tuple):
        return tuple(to_device(v, device) for v in layout)
    return layout
