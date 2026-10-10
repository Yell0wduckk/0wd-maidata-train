"""Pack one collated batch into a few dtype buffers for IPC and H2D."""
from dataclasses import dataclass, fields, is_dataclass
import torch


def _describe(value, leaves, offsets):
    if isinstance(value, torch.Tensor):
        dtype, count = value.dtype, value.numel()
        offset = offsets.get(dtype, 0)
        offsets[dtype] = offset + count
        leaves.setdefault(dtype, []).append(value.contiguous().flatten())
        return ('tensor', dtype, offset, count, tuple(value.shape))
    if is_dataclass(value):
        return ('dataclass', type(value), {f.name: _describe(getattr(value, f.name), leaves, offsets) for f in fields(value)})
    if isinstance(value, dict):
        return ('dict', {k: _describe(v, leaves, offsets) for k,v in value.items()})
    if isinstance(value, (tuple, list)):
        return ('sequence', type(value), [_describe(v, leaves, offsets) for v in value])
    return ('scalar', value)


def _restore(spec, buffers):
    kind = spec[0]
    if kind == 'tensor':
        _, dtype, offset, count, shape = spec
        return buffers[dtype].narrow(0, offset, count).view(shape)
    if kind == 'dataclass':
        return spec[1](**{k: _restore(v, buffers) for k,v in spec[2].items()})
    if kind == 'dict':
        return {k: _restore(v, buffers) for k,v in spec[1].items()}
    if kind == 'sequence':
        return spec[1](_restore(v, buffers) for v in spec[2])
    return spec[1]


@dataclass
class PackedBatch:
    structure: tuple
    buffers: dict
    _view: object = None

    def unpack(self):
        if self._view is None:
            self._view = _restore(self.structure, self.buffers)
        return self._view

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return getattr(self.unpack(), name)

    def pin_memory(self):
        return PackedBatch(self.structure, {k: v.pin_memory() for k,v in self.buffers.items()})

    def to(self, device):
        return _restore(self.structure, {k: v.to(device, non_blocking=True) for k,v in self.buffers.items()})


def pack_batch(batch):
    leaves, offsets = {}, {}
    structure = _describe(batch, leaves, offsets)
    return PackedBatch(structure, {k: torch.cat(v) for k,v in leaves.items()})
