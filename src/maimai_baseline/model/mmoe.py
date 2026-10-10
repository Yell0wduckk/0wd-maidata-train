"""Dense task-gated FFNs; attention and condition projections are shared."""
from maimai_baseline.architecture import PRESERVED_DECODER_LAYERS, NEW_DECODER_LAYER_SEED_XOR
import torch
from torch import nn
from torch.nn import functional as F


class FFNExpert(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.linear1 = nn.Linear(256, 512)
        self.linear2 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class MMoELayer(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.shared = original.shared if hasattr(original, 'shared') else original.experts[0]
        self.audio_kv = original.audio_kv
        self.dropout = original.dropout
        del self.shared.linear1, self.shared.linear2
        self.experts = nn.ModuleList(FFNExpert(self.dropout) for _ in range(4))
        self.gates = nn.ModuleList(nn.Linear(256, 4) for _ in range(2))
        for gate in self.gates:
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)
        self.metrics = {}

    def mix(self, rows, family, indices, collect_metrics):
        self.metrics = {}
        flat = rows.reshape(-1, 256)
        selected = indices[0]
        result = torch.zeros_like(flat)
        if not selected.numel():
            return result.reshape_as(rows)
        x = self.shared.norm3(flat.index_select(0, selected))
        tasks = family.reshape(-1).index_select(0, selected)
        # Every real row evaluates all four experts exactly once.
        values = torch.stack([expert(x) for expert in self.experts], dim=1)
        logits = x.new_zeros((len(x), 4))
        for task, gate in enumerate(self.gates):
            index = tasks.eq(task).nonzero(as_tuple=False).flatten()
            if index.numel():
                value = gate(x.index_select(0, index))
                logits = logits.to(value.dtype).index_copy(0, index, value)
        weights = logits.float().softmax(-1)
        mixed = (values * weights.to(values.dtype).unsqueeze(-1)).sum(1)
        if collect_metrics:
            for task, name in enumerate(('time', 'note')):
                w = weights.detach()[tasks.eq(task)]
                if len(w):
                    self.metrics[name] = dict(count=len(w),
                        entropy=float(-(w * w.clamp_min(1e-12).log()).sum(-1).mean()),
                        **{f'expert_{i}': float(w[:, i].mean()) for i in range(4)})
        mixed = self.shared.dropout3(mixed)
        return result.to(mixed.dtype).index_copy(0, selected, mixed).reshape_as(rows)

    def forward(self, rows, family, positions, padding, audio, audio_padding,
                *, past=None, past_padding=None, style=None, style_padding=None,
                gate=None, collect_metrics=False, style_rotary=True):
        from maimai_baseline.model.dual import layer_forward
        indices = ((~padding).reshape(-1).nonzero(as_tuple=False).flatten(),)
        gate = None if gate is None else gate.expand(padding.shape)
        return layer_forward(self, rows, positions, padding, audio, audio_padding,
            experts=(self.shared,), indices=indices, past=past, past_padding=past_padding,
            style=style, style_padding=style_padding, gate=gate,
            ffn=lambda x: self.mix(x, family, indices, collect_metrics), style_rotary=style_rotary)


def install(model, seed):
    # Retain the original fresh construction stream for all unaffected modules.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed) ^ 0x4D4D4F45)
        layers = model.decoder_layers
        model.decoder_layers = nn.ModuleList(MMoELayer(layer) for layer in layers[:PRESERVED_DECODER_LAYERS])
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed) ^ NEW_DECODER_LAYER_SEED_XOR)
            model.decoder_layers.extend(MMoELayer(layer) for layer in layers[PRESERVED_DECODER_LAYERS:])
    if hasattr(model, 'style_conditioner'):
        c = model.style_conditioner
        c.shared_gates = nn.Parameter(torch.cat((c.shared_gate.detach().reshape(1),
                                               c.expert_gates.detach()[0])))
        del c.shared_gate, c.expert_gates
