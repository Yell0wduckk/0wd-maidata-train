"""Differentiable batched raw-prefix decoding for independent SFT lanes."""
import torch


class IncrementalMomentDecoder:
    def __init__(self, model, batch, outputs, windows, heads, anchors):
        self.model = model
        self.context = outputs['audio_context'].map_audio_windows(windows)
        self.conditions = {k: batch[k].index_select(0, windows)
                           for k in ('difficulty_id', 'level', 'start_tick', 'end_tick')}
        self.anchors = anchors
        self.lengths = heads + 1
        width = int(self.lengths.max())
        self.padding = torch.arange(width, device=heads.device)[None] >= self.lengths[:, None]
        self.padding |= batch['row_padding_mask'].index_select(0, windows)[:, :width]
        self.local = tuple(tuple(t.index_select(0, windows)[:, :, :width].masked_fill(
            self.padding[:, None, :, None], 0.) for t in kv) for kv in outputs['self_key_values'])
        # Audio projection is shared by source window, then indexed per lane.
        unique, inverse = torch.unique(windows, sorted=True, return_inverse=True)
        shared = model.audio_kv(outputs['audio_context'].map_audio_windows(unique),
            start_tick=batch['start_tick'].index_select(0, unique))
        self.audio = shared
        self.audio_windows = inverse
        self.steps = 0

    def retain(self, indices):
        self.local = tuple(tuple(t.index_select(0, indices) for t in kv) for kv in self.local)
        self.padding = self.padding.index_select(0, indices)

    def step(self, lanes, input_ids, input_times, input_active):
        times, active = dict(input_times), dict(input_active)
        times['delta_ms'] = times.pop('delta_tick')
        active['delta_ms'] = active.pop('delta_tick')
        audio = tuple(tuple(tuple(t.index_select(0, self.audio_windows[lanes]) for t in kv) for kv in edges) for edges in self.audio)
        padding = torch.zeros((len(lanes), 1), device=lanes.device, dtype=torch.bool)
        output = self.model.decode_prefix_rows(audio_context=self.context.map_audio_windows(lanes),
            input_ids=input_ids, input_times=times, input_time_active=active,
            row_anchor_tick=self.anchors[lanes, None], row_padding_mask=padding,
            **{k:v.index_select(0, lanes) for k,v in self.conditions.items()},
            past_key_values=self.local, past_padding=self.padding, audio_key_values=audio)
        self.local = output['self_key_values']
        self.padding = torch.cat((self.padding, padding), 1)
        self.steps += 1
        output['candidate_query'] = output['delta_candidate']['query'][:, 0]
        return output
