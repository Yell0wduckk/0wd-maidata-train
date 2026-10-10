"""Worker-side tensor preparation; no learned modules or GPU-side geometry."""
from dataclasses import dataclass
import torch

from maidata.decoder.batching import audio_query_layout, audio_key_spans
from .batching import window_chunks, batch_cost
from .window_cache import collate_prepared


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
    batch.budget_cost = batch_cost(batch.audio_lengths, [w['length'] for w in batch.windows], config)
    conditions = batch.conditions
    spans = audio_key_spans(conditions.audio.valid, validate=False)
    for indices in window_chunks(batch.windows, config['decoder_train_chunk_token_budget']):
        rows = [batch.windows[i] for i in indices]
        tokens, types, labels = collate_prepared(rows)
        index = torch.tensor(indices, dtype=torch.long)
        audio = conditions.audio_mapping[conditions.occurrence_mapping[index]]
        unique = conditions.occurrence_mapping[index]
        condition_valid = torch.cat((conditions.decoder_z.valid[unique], conditions.plan_valid[unique, None]), -1)
        batch.chunks.append(Chunk(index, tokens, types, labels,
            audio_query_layout(tokens.valid, audio, len(batch.audio_variants), spans=spans,
                               condition_valid=condition_valid, validate=False), tokens.valid.numel()))
    batch.token_budget = config['decoder_train_chunk_token_budget']
    # Transport only tensors + small occurrence metadata. Rule objects, complete
    # support dictionaries and GT token trees stay inside the worker.
    batch.windows = [dict(identity=w['identity'], history_dropped=w['history_dropped'])
                     for w in batch.windows]
