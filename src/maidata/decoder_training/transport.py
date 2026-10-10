"""Bounded dtype-packed IPC and H2D: context once, one Decoder chunk at a time."""
from dataclasses import dataclass, replace
from maidata.training.transport import pack_batch


@dataclass
class Packet:
    context: object
    chunks: list

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return getattr(self.context, name)

    def pin_memory(self):
        return Packet(self.context.pin_memory(), [c.pin_memory() for c in self.chunks])

    def to(self, device):
        return replace(self.context.to(device), chunks=self.chunks)


def pack(batch):
    return Packet(pack_batch(replace(batch, chunks=None)), [pack_batch(c) for c in batch.chunks])
