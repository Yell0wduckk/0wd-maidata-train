"""Persistent dense decoder history, independently reset local router history."""
from dataclasses import replace
import torch
from maimai_baseline.model.rope import project_kv, rotate
from maimai_baseline.model.song_dense import dense_layers
from maimai_baseline.model.dual import expert_readout
from maimai_baseline.dual_contract import prefix_families


class SongDenseKVCache:
    def __init__(self, model, memory, padding, beats, *, audio_spans=None):
        if model.training:
            raise ValueError('song cache requires eval')
        self.model, self.memory, self.memory_padding, self.memory_beats = model, memory, padding, beats
        self.audio = tuple(layer.audio_kv(memory, beats) for layer in model.decoder_layers)
        from maimai_baseline.data.attention_layout import key_spans
        self.audio_spans = key_spans(padding.cpu()) if audio_spans is None else audio_spans
        self.kv = self.padding = self.positions = self.anchors = self.families = None
        self.router_kv = self.router_padding = self.router_style_kv = None
        self.local_length = 0
        self.hits = 0
        self.context = self.start = self.style = None
        self.style_keys = None
        self.window_base = 0
        self.local_history_length = None

    @property
    def length(self):
        # Generation's slicing cursor counts the original local router prefix.
        return self.local_length

    @property
    def song_length(self):
        return 0 if self.padding is None else self.padding.shape[1]

    def begin_window(self, context, start, style):
        self.window_base = self.song_length
        self.local_history_length = None
        self.context, self.start, self.style = context, start, style
        from maimai_baseline.radar_contract import is_radar
        enabled = is_radar(self.model.config) or self.model.config.latent_enabled
        condition = style if style is None or not getattr(style, 'rotary', True) else replace(style, beats=style.beats+start[:, None]/96.)
        self.style_keys = None if not enabled or style is None or not style.enabled else self.model.style_conditioner.prepare_kv(condition)
        k, v = project_kv(self.model.router.audio, context.memories[-1])
        self.router_audio = (rotate(k, context.beats[-1]-start[:, None]/96.), v)
        self.router_kv = self.router_padding = self.router_style_kv = None
        self.local_length = 0

    def snapshot(self):
        return (self.kv, self.padding, self.positions, self.anchors, self.families,
            self.router_kv, self.router_padding, self.router_style_kv, self.local_length,
            self.context, self.start, self.style, self.router_audio, self.style_keys,
            self.window_base, self.local_history_length)

    def restore(self, state, *, replay_last=False):
        (self.kv, self.padding, self.positions, self.anchors, self.families,
         self.router_kv, self.router_padding, self.router_style_kv, self.local_length,
         self.context, self.start, self.style, self.router_audio, self.style_keys,
         self.window_base, self.local_history_length) = state
        if replay_last and self.song_length:
            self.kv = tuple(tuple(x[:, :, :-1] for x in pair) for pair in self.kv)
            self.padding, self.positions, self.anchors, self.families = (x[:, :-1] for x in
                (self.padding, self.positions, self.anchors, self.families))
            self.router_kv = tuple(x[:, :, :-1] for x in self.router_kv)
            self.router_padding = self.router_padding[:, :-1]
            self.local_length -= 1

    def trim_local(self, length):
        """Discard a rejected window suffix using the current prefix storage.

        No per-row snapshots are required: causal K/V for retained rows are
        immutable, and the local router cursor maps to the dense window suffix.
        """
        if self.local_history_length is None or not self.local_history_length <= length <= self.local_length:
            raise ValueError('trim must retain local history and stay within the current window')
        song_length = self.window_base + length - self.local_history_length
        self.kv = tuple(tuple(x[:, :, :song_length] for x in pair) for pair in self.kv)
        self.padding, self.positions, self.anchors, self.families = (x[:, :song_length] for x in
            (self.padding, self.positions, self.anchors, self.families))
        self.router_kv = tuple(x[:, :, :length] for x in self.router_kv)
        self.router_padding = self.router_padding[:, :length]
        self.local_length = length

    def decode(self, *, audio_context, **inputs):
        if audio_context is not self.context or not torch.equal(inputs['start_tick'], self.start):
            raise ValueError('call begin_window before changing window')
        selector = inputs.pop('family_selector', None)
        family = inputs.pop('decoder_family', None)
        apply_mask = inputs.pop('apply_grammar_mask', True)
        query_padding_cpu = inputs.pop('_query_padding_cpu', None)
        if query_padding_cpu is None:
            query_padding_cpu = inputs['row_padding_mask'].cpu()
        state = self.model.router_forward(audio_context=audio_context, **inputs,
            audio_key_values=dict(experts=(), router=self.router_audio),
            style_key_values=(),
            router_key_values=self.router_kv, past_padding=self.router_padding,
            router_style_key_values=self.router_style_kv)
        if family is None:
            family = prefix_families(state.input_events, state.padding, self.model.vocabulary)
            chosen = state.logits.argmax(-1) if selector is None else selector(state.logits, state.allowed)
            family = torch.where(family.lt(0) & ~state.padding, chosen, family)
        # At later windows the local history is only for the router. Decoder
        # history already contains the original hidden states and conditions.
        begin = 0
        if self.local_length == 0 and self.song_length:
            marker = state.input_events[0].eq(self.model.vocabulary.event_id('TARGET_START')).nonzero().flatten()
            if len(marker) != 1:
                raise ValueError('window prefill requires one marker')
            begin = int(marker[0])
        if self.local_length == 0:
            self.local_history_length = begin
        selected = replace(state, rows=state.rows[:, begin:], positions=state.positions[:, begin:],
            padding=state.padding[:, begin:], logits=state.logits[:, begin:],
            support=state.support[:, begin:], allowed=state.allowed[:, begin:])
        positions = inputs['row_anchor_tick'][:, begin:]/96.
        family = family[:, begin:]
        owner = torch.zeros_like(family)
        from maimai_baseline.data.attention_layout import cross_layout, to_device
        mapping = (torch.zeros(query_padding_cpu.shape[0], dtype=torch.long) if len(self.audio_spans) == 1
                   else torch.arange(query_padding_cpu.shape[0]))
        layout, mapping = to_device((cross_layout(query_padding_cpu[:, begin:], None,
            mapping, spans=self.audio_spans), mapping), self.memory.device)
        rows, kv, norms, ratios = dense_layers(self.model, selected.rows, family, positions,
            selected.padding, self.memory, self.memory_padding, self.memory_beats,
            style=self.style, style_owner=owner, window_start=self.start,
            past=self.kv, past_padding=self.padding, audio_keys=self.audio, style_keys=self.style_keys,
            audio_mapping=mapping, audio_layout=layout)
        selected = replace(selected, positions=positions)
        result = expert_readout(self.model, selected, family, rows, kv, norms, ratios,
            apply_grammar_mask=apply_mask)
        result = self.model._time_outputs(self.model._route_outputs(result))
        self.hits += self.song_length > 0
        self.kv, self.router_kv, self.router_style_kv = kv, state.kv, state.style_router_kv
        def append(old, new):
            return new if old is None else torch.cat((old, new), 1)
        self.padding = append(self.padding, selected.padding)
        self.positions = append(self.positions, positions)
        self.anchors = append(self.anchors, inputs['row_anchor_tick'][:, begin:])
        self.families = append(self.families, family)
        self.router_padding = append(self.router_padding, state.padding)
        self.local_length += state.padding.shape[1]
        return result

    def report(self):
        return dict(backend='song-dense-sdpa', prefix_reuse_steps=self.hits,
                    prefix_rows=self.song_length, local_router_rows=self.length)


@torch.no_grad()
def prepare_audio(model, feature, bpm_audio, timing, song_end_tick, device, full_audio, *, window_count=None):
    from maimai_baseline.data.fullaudio import arrays, pack, window_context
    from maimai_baseline.training.steps import _move
    from maimai_baseline.data.attention_layout import key_spans
    batch = pack([arrays(feature, bpm_audio['coordinates'], timing, bpm_audio['num_samples'])], [0])
    spans = key_spans(batch['interpolation_geometry']['grid_0_padding'])
    batch = _move(batch, device)
    context = model.encode_audio(batch['audio'], batch['audio_padding_mask'], batch['frame_beats'],
        batch['frame_bpms'], start_tick=batch['audio_start_tick'],
        interpolation_geometry=batch['interpolation_geometry'], song_audio=full_audio)
    starts = (range(0, song_end_tick+1, 1536) if window_count is None else
              range(0, int(window_count)*1536, 1536))
    windows = {s: window_context(context, torch.tensor([0], device=device), torch.tensor([s], device=device))
               for s in starts}
    return windows, SongDenseKVCache(model, context.memories[-1], context.padding_masks[-1],
                                    context.beats[-1], audio_spans=spans)


@torch.no_grad()
def prepare_native_audio(model, teacher, variant, device, *, window_count):
    from maimai_baseline.data.teacher_audio import project_candidates
    from maimai_baseline.model.common import AudioContext
    from maimai_baseline.training.steps import _move
    from maimai_baseline.data.attention_layout import key_spans
    starts = [index * 1536 for index in range(int(window_count))]
    inputs = teacher.dense_inputs([variant] * len(starts), starts)
    full = AudioContext(**_move(inputs['audio_context'], device))
    local = AudioContext(**_move(inputs['local_audio_context'], device))
    local = project_candidates(model, local, _move(inputs['teacher_interpolation'], device))
    windows = {start: local.map_audio_windows(torch.tensor([index], device=device))
               for index, start in enumerate(starts)}
    spans = key_spans(inputs['audio_context']['padding_masks'][-1])
    return windows, SongDenseKVCache(model, full.memories[-1], full.padding_masks[-1],
                                    full.beats[-1], audio_spans=spans)
