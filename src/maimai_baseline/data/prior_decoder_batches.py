"""Occurrence-preserving B16384 batches and whole-chart Decoder4096 microbatches."""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import torch
from torch.utils.data import Dataset

from maimai_baseline.data.batch import collate_baseline_segments
from maimai_baseline.data.frozen_context import FrozenSongContext
from maimai_baseline.data.frozen_latent import FrozenLatentCache


def _audio_dict(context):
    return {field.name:getattr(context, field.name) for field in fields(context)}


def dense_execution(batch, items, group_windows):
    """One BOS, each owning window marker and each real prefix row exactly once."""
    owners, rows, maps = [], [], []
    for windows in group_windows:
        ws, rs, sources, markers = [], [], {}, {}
        for ordinal, window in enumerate(windows):
            item = items[window]
            marker = int(item['_history_context_target_index'])
            chosen = ([0] if ordinal == 0 else [])+list(range(marker, len(item['row_anchor_tick'])))
            markers[window] = len(ws)+(1 if ordinal == 0 else 0)
            for row in chosen:
                source = int(item['_physical_input_source_row_ids'][row])
                if source >= 0:
                    if source in sources:
                        raise ValueError('duplicate owned chart-occurrence input source row')
                    sources[source] = len(ws)
                ws.append(window); rs.append(row)
        owners.append(torch.tensor(ws)); rows.append(torch.tensor(rs)); maps.append((sources,markers))
    owner = torch.nn.utils.rnn.pad_sequence(owners, batch_first=True)
    row = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True)
    valid = torch.arange(owner.shape[1])[None] < torch.tensor([len(x) for x in owners])[:,None]
    output_group = torch.zeros_like(batch['row_padding_mask'], dtype=torch.long)
    output_row = torch.zeros_like(output_group)
    for group, windows in enumerate(group_windows):
        sources, markers = maps[group]
        for window in windows:
            item = items[window]
            output_group[window] = group
            for local, source in enumerate(item['_physical_input_source_row_ids'].tolist()):
                output_row[window,local] = sources[source] if source >= 0 else markers[window] if source == -1 else 0
    return dict(batch=batch, owner=owner, row=row, valid=valid, context_rows=int(valid.sum()),
        group_windows=group_windows, target_windows=torch.arange(len(items)),
        output_group=output_group, output_row=output_row)


class PriorDecoderBatches(Dataset):
    def __init__(self, cache_root, context_root, decoder_context_root, *, window_cache_root=None,
                 disable_note_augmentation=False, descriptor=None):
        self.cache_root, self.context_root, self.decoder_context_root = map(str, (cache_root, context_root, decoder_context_root))
        self.window_cache_root = None if window_cache_root is None else str(window_cache_root)
        self.metadata = descriptor
        self.disable_note_augmentation = bool(disable_note_augmentation or (descriptor or {}).get('disable_note_augmentation', False))
        self.context = self.windows = self.decoder = None

    def __getstate__(self):
        return dict(self.__dict__, context=None, windows=None, decoder=None)

    def _open(self):
        if self.context is not None:
            return
        from maimai_baseline.data.prior_decoder import WindowCache
        from maimai_baseline.data.decoder_context import DecoderContextCache
        meta = self.metadata or {}
        target = FrozenLatentCache(self.cache_root, fields=(), descriptor=meta.get('target'))
        self.context = FrozenSongContext(self.context_root, target, descriptor=meta.get('context', meta.get('manifest')),
            teacher_audio_cache=meta.get('teacher_audio_cache'))
        path = self.window_cache_root or meta.get('window_cache_root') or target.manifest.get('decoder_data',{}).get('window_cache')
        if path is None:
            raise ValueError('prior decoder batch requires the migrated GT window cache')
        path = Path(path)
        if not path.is_absolute():
            path = Path(self.cache_root)/path
        self.window_cache_root = str(path)
        self.windows = WindowCache(path)
        self.decoder = DecoderContextCache(self.decoder_context_root, self.context, descriptor=meta.get('decoder'))

    def descriptor(self):
        self._open()
        return dict(target=self.context.target.descriptor(), context=self.context.manifest,
                    teacher_audio_cache=None if self.context.teacher is None else str(self.context.teacher.root),
                    decoder=self.decoder.descriptor(), window_cache_root=self.window_cache_root,
                    disable_note_augmentation=self.disable_note_augmentation)

    def batch(self, requests, *, epoch=0):
        self._open()
        requests = [(int(index),int(occ)) for index,occ in requests]
        result = self.context.batch(requests)
        instances = result['chart_occurrences']
        items, groups, audio_keys = [], [[] for _ in instances], []
        for position, (index,occurrence) in enumerate(requests):
            record = self.context.target.records[index]
            item = self.windows.get(int(record['canonical_index']), int(record['view_id']), occurrence,
                                    epoch=epoch, enabled=not self.disable_note_augmentation)
            group = instances.index((self.context.mapping[index],occurrence))
            groups[group].append(position)
            items.append(item)
            audio_keys.append(self.context.charts[instances[group][0]]['audio_key'])
        unique_audio = list(dict.fromkeys(audio_keys))
        mapping = torch.tensor([unique_audio.index(key) for key in audio_keys])
        starts = torch.tensor([int(item['start_tick']) for item in items])
        continuous, local = self.decoder.batch(unique_audio, mapping, starts, p0=result['prior_inputs'])
        result['decoder_audio_context'], result['decoder_local_audio_context'] = _audio_dict(continuous), _audio_dict(local)
        packed, pending, used = [], [], 0
        for group, positions in enumerate(groups):
            positions.sort(key=lambda i:int(items[i]['start_tick']))
            cost = sum(int(items[i]['target_ids']['event'].ne(-100).sum()) for i in positions)
            if pending and used+cost > 4096:
                packed.append(pending); pending, used = [], 0
            pending.append(group); used += cost
        if pending:
            packed.append(pending)
        micros = []
        vocabulary = self.windows.vocabulary
        for selected in packed:
            positions = [position for group in selected for position in groups[group]]
            selected_items = [items[position] for position in positions]
            batch = collate_baseline_segments(selected_items, vocabulary=vocabulary, chart_only=True)
            offsets, offset = [], 0
            for group in selected:
                offsets.append(list(range(offset,offset+len(groups[group])))); offset += len(groups[group])
            execution = dense_execution(batch, selected_items, offsets)
            audio_indices = list(dict.fromkeys(mapping[positions].tolist()))
            group_audio = torch.tensor([audio_indices.index(int(mapping[groups[group][0]])) for group in selected])
            from maimai_baseline.data.attention_layout import cross_layout
            execution.update(group_audio=group_audio,
                audio_layout=cross_layout(~execution['valid'], continuous.padding_masks[-1][audio_indices], group_audio),
                audio_indices=torch.tensor(audio_indices), audio_window_index=torch.tensor(positions),
                latent_group=torch.tensor([group for group in selected for _ in groups[group]]),
                latent_block=batch['start_tick'].div(384,rounding_mode='floor').long())
            micros.append(dict(batch, song_dense_inputs=execution))
        result['decoder_batches'] = micros
        return result

    def __getitem__(self, request):
        from maimai_baseline.data.batch_transport import pack_batch
        indices, epoch, _ = request
        return pack_batch(self.batch(indices, epoch=epoch))
