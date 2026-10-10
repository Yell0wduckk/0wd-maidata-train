"""Worker-side tensor preparation; no learned modules or GPU-side geometry."""
from dataclasses import dataclass
import torch

from maidata.decoder.batching import collate_windows, audio_query_layout, audio_key_spans
from .batching import window_chunks, joint_cost
from .supervision import prepare


@dataclass
class Chunk:
    indices: torch.Tensor
    tokens: object
    next_types: torch.Tensor
    labels: object
    audio_layout: dict
    cost: int

    def to(self, device):
        from maidata.training.data import map_tensors
        return map_tensors(self, lambda t: t.to(device, non_blocking=True))


def prepare_batch(batch, config):
    batch.chunks = []
    batch.budget_cost = joint_cost(batch.audio_lengths,
        [len(w['example'].inputs) for w in batch.windows], config)
    if batch.budget_cost['oversized'] and len(batch.windows) != 1:
        raise ValueError('multi-window batch exceeds the joint token budget')
    conditions = batch.conditions
    spans = audio_key_spans(conditions.audio.valid)
    for indices in window_chunks(batch.windows, config['decoder_train_chunk_token_budget']):
        rows = [batch.windows[i] for i in indices]
        tokens, types, _ = collate_windows([w['example'] for w in rows], device='cpu')
        index = torch.tensor(indices, dtype=torch.long)
        audio = conditions.audio_mapping[conditions.occurrence_mapping[index]]
        batch.chunks.append(Chunk(index, tokens, types, prepare(rows),
            audio_query_layout(tokens.valid, audio, len(batch.audio_variants), spans=spans), tokens.valid.numel()))
    batch.token_budget = config['decoder_train_chunk_token_budget']
    # Transport only tensors + small occurrence metadata. Rule objects, complete
    # support dictionaries and GT token trees stay inside the worker.
    batch.windows = [dict(identity=w['identity'], history_dropped=w['example'].history_dropped)
                     for w in batch.windows]
