"""Worker-side tensor preparation; no learned modules or GPU-side geometry."""
from dataclasses import dataclass
import torch

from maidata.audio.layout import prepare_encoder_layout
from maidata.draft.layout import prepare_p0, prepare_selection
from maidata.decoder.batching import collate_windows, audio_query_layout, audio_key_spans
from .batching import window_chunks
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
    layout = prepare_encoder_layout(batch.songs.beat_centers, batch.songs.sample_valid.any(-1),
        batch.audio_windows, radii=config['stage_attention_radii'], rates=config['fpn_rates'],
        before=config['audio_before_beats'], context=config['audio_before_beats']+config['target_beats']+config['audio_after_beats'])
    batch.audio_windows.layout = layout
    draft = batch.draft_inputs
    draft.valid_indices = draft.draft_valid.flatten().nonzero().flatten()
    batch.layouts = dict(p0=prepare_p0(layout['grids'][0], layout['resample'][0]['valid'],
        batch.audio_windows.target_start_beats, batch.audio_windows.window_to_song,
        batch.coverages, p0_rate=config['fpn_rates'][0]))
    keys, unique, occurrences = {}, [], []
    for i, key in enumerate(zip(batch.groups.tolist(), batch.starts.tolist())):
        if key not in keys:
            keys[key] = len(unique)
            unique.append(i)
        occurrences.append(keys[key])
    index = torch.tensor(unique, dtype=torch.long)
    batch.layouts.update(unique=index, occurrences=torch.tensor(occurrences, dtype=torch.long))
    for name, before, after in (('planner', 0, 0),
            ('decoder', config['decoder_z_before_beats'], config['decoder_z_after_beats'])):
        batch.layouts[name] = prepare_selection(draft.query_beats, draft.draft_valid,
            batch.groups[index], batch.starts[index], batch.valid[index], target_beats=config['target_beats'],
            before_beats=before, after_beats=after)
    batch.chunks = []
    p0 = batch.layouts['p0']
    spans = audio_key_spans(layout['resample'][0]['valid'].flatten()[p0['indices']] & p0['present'])
    for indices in window_chunks(batch.windows, config['decoder_train_token_budget']):
        rows = [batch.windows[i] for i in indices]
        tokens, types, _ = collate_windows([w['example'] for w in rows], device='cpu')
        index = torch.tensor(indices, dtype=torch.long)
        audio = draft.audio_mapping[batch.groups[index]]
        batch.chunks.append(Chunk(index, tokens, types, prepare(rows),
            audio_query_layout(tokens.valid, audio, len(batch.coverages), spans=spans), tokens.valid.numel()))
    batch.token_budget = config['decoder_train_token_budget']
    # Transport only tensors + small occurrence metadata. Rule objects, complete
    # support dictionaries and GT token trees stay inside the worker.
    batch.windows = [dict(identity=w['identity'], history_dropped=w['example'].history_dropped)
                     for w in batch.windows]
