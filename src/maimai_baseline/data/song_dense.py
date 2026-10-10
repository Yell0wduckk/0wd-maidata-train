"""Step-local whole-song decoder inputs; original supervision remains untouched."""
from collections import defaultdict
import torch
from maimai_baseline.data.curriculum import VirtualSegmentRef
from maimai_baseline.data.batch import collate_baseline_segments


class SongDenseProvider:
    def __init__(self, dataset, chart_provider, timing_root=None, radar_cache=None, teacher_audio_cache=None):
        self.dataset, self.chart_provider, self.timing_root = dataset, chart_provider, timing_root
        self.radar_cache_root, self.radar_cache = radar_cache, None
        self.teacher_audio_root, self.teacher_audio = teacher_audio_cache, None
        self.records = defaultdict(list)
        self.lookup = {}
        for i, s in enumerate(dataset.segments):
            self.records[str(s['record_id'])].append(i)
            self.lookup[str(s['segment_id'])] = str(s['record_id'])
        for indices in self.records.values():
            indices.sort(key=lambda i: dataset.segments[i]['target_start_tick'])

    def __call__(self, items):
        # Reuse original target items verbatim. Missing context windows use the
        # same dataset epoch and NOTE augmentation as their owning target window.
        groups = list(dict.fromkeys((self.lookup[str(i['segment_id'])], i['spatial_view_id'], int(i.get('occurrence', 0))) for i in items))
        originals = {(str(i['segment_id']), i['spatial_view_id'], int(i.get('occurrence', 0))): i for i in items}
        windows, group_windows = [], []
        for record, view, occurrence in groups:
            indices = []
            for index in self.records[record]:
                segment = self.dataset.segments[index]
                key = str(segment['segment_id']), view, occurrence
                item = originals.get(key)
                if item is None:
                    item = self.dataset[VirtualSegmentRef(index, 0 if view == 'identity' else 1, occurrence)]
                indices.append(len(windows))
                windows.append(item)
            group_windows.append(indices)
        batch = collate_baseline_segments(windows, vocabulary=self.dataset.vocabulary,
            audio_loader=self.dataset.audio_item, timing_root=self.timing_root,
            chart_only=True)
        batch['global_style_inputs'] = self.chart_provider(windows)
        radar = None
        if self.radar_cache_root is not None:
            from maimai_baseline.data.majradar import MajRadarCache
            from maimai_baseline.data.radar_inputs import dense_inputs
            if self.radar_cache is None:
                self.radar_cache = MajRadarCache(self.radar_cache_root)
            radar = dense_inputs(self.radar_cache, windows, group_windows)
        variants = [self.dataset.bpm_audio_store.segments[str(i['segment_id'])] for i in windows]
        native = None
        if self.teacher_audio_root is not None:
            from maimai_baseline.data.teacher_audio import TeacherAudioCache
            if self.teacher_audio is None:
                self.teacher_audio = TeacherAudioCache(self.teacher_audio_root)
            native = self.teacher_audio.dense_inputs(variants, batch['start_tick'])
            batch['window_to_audio'] = native.pop('window_to_audio')
        elif not hasattr(self, 'fullaudio'):
            from maimai_baseline.data.fullaudio import Provider
            prepared = {variant:item['_prepared_timing'] for variant,item in zip(variants, windows)
                        if '_prepared_timing' in item}
            self.fullaudio = Provider(self.dataset.bpm_audio_store,
                prepared_timings=prepared, timing_root=self.timing_root)
        elif self.teacher_audio_root is None:
            self.fullaudio.prepared_timings.update({variant:item['_prepared_timing']
                for variant,item in zip(variants, windows) if '_prepared_timing' in item})
        if self.teacher_audio_root is None:
            batch.update(self.fullaudio.batch(variants))
        keys = {(str(i['segment_id']), i['spatial_view_id'], int(i.get('occurrence', 0))): w for w, i in enumerate(windows)}
        target_windows = torch.tensor([keys[str(i['segment_id']), i['spatial_view_id'], int(i.get('occurrence', 0))] for i in items])
        # Dense rows contain one BOS, every window marker, and each real input
        # once. SEGMENT_END is a target, never an inserted historical input.
        owners, local_rows, source_maps, offsets = [], [], [], []
        for indices in group_windows:
            ws, rs, sources, marker_map = [], [], {}, {}
            for ordinal, w in enumerate(indices):
                item = windows[w]
                marker = int(item['_history_context_target_index'])
                chosen = ([0] if ordinal == 0 else []) + list(range(marker, len(item['row_anchor_tick'])))
                marker_map[w] = len(ws) + (1 if ordinal == 0 else 0)
                for r in chosen:
                    source = int(item['_physical_input_source_row_ids'][r])
                    if source >= 0:
                        if source in sources:
                            raise ValueError('duplicate owned song input source row')
                        sources[source] = len(ws)
                    ws.append(w); rs.append(r)
            owners.append(torch.tensor(ws)); local_rows.append(torch.tensor(rs))
            source_maps.append((sources, marker_map)); offsets.append(indices)
        owner = torch.nn.utils.rnn.pad_sequence(owners, batch_first=True)
        row = torch.nn.utils.rnn.pad_sequence(local_rows, batch_first=True)
        valid = torch.arange(owner.shape[1])[None] < torch.tensor([len(x) for x in owners])[:, None]
        width = max(len(i['row_anchor_tick']) for i in items)
        output_group = torch.zeros((len(items), width), dtype=torch.long)
        output_row = torch.zeros_like(output_group)
        for b, (item, w) in enumerate(zip(items, target_windows.tolist(), strict=True)):
            group = groups.index((self.lookup[str(item['segment_id'])], item['spatial_view_id'], int(item.get('occurrence', 0))))
            sources, markers = source_maps[group]
            output_group[b] = group
            for r, source in enumerate(item['_physical_input_source_row_ids'].tolist()):
                output_row[b, r] = sources[source] if source >= 0 else markers[w] if source == -1 else 0
            marker = int(item['_history_context_target_index'])
            mapped = output_row[b, marker:len(item['row_anchor_tick'])]
            ws, rs = owners[group][mapped], local_rows[group][mapped]
            for name in ('input_ids', 'input_times', 'input_time_active'):
                for field, value in batch[name].items():
                    # Every original supervised query remains at the owning
                    # window's exact augmented input, including branch order.
                    if not torch.equal(value[ws, rs], value[w, marker:len(item['row_anchor_tick'])]):
                        raise ValueError('dense supervised input mapping changed '+name+'/'+field)
        from maimai_baseline.data.attention_layout import cross_layout
        group_audio = torch.stack([batch['window_to_audio'][w[0]] for w in offsets])
        audio_padding = (native['audio_context']['padding_masks'][-1] if native is not None else batch['interpolation_geometry']['grid_0_padding'])
        audio_layout = cross_layout(~valid, audio_padding, group_audio)
        result = dict(radar_inputs=radar, group_audio=group_audio, audio_layout=audio_layout, batch=batch, owner=owner, row=row, valid=valid, context_rows=int(valid.sum()),
            group_windows=offsets, target_windows=target_windows,
            output_group=output_group, output_row=output_row)
        if native is not None:
            result.update(native)
        return result
