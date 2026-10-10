"""One-window raw prefix KV state. Snapshots share immutable tensors."""
import torch


class DecoderKVCache:
    def __init__(self, model):
        if model.training:
            raise ValueError("decoder KV cache requires eval mode")
        self.model = model
        self.audio_context = self.audio = self.start_tick = None
        self.kv = self.padding = self.positions = self.anchors = None
        self.hits = 0
        self.style_signature = None
        self.style_kv = None
        self.router_kv = self.router_style_kv = self.families = None

    @property
    def length(self):
        return 0 if self.padding is None else self.padding.shape[1]

    def snapshot(self):
        return (self.kv, self.padding, self.positions, self.anchors, self.style_signature,
                self.router_kv, self.families)

    def restore(self, state, *, replay_last=False):
        if state[4] != self.style_signature:
            raise ValueError("decoder snapshot belongs to a different style context")
        self.kv, self.padding, self.positions, self.anchors = state[:4]
        self.router_kv, self.families = state[5:]
        if replay_last and self.length:
            self.kv = tuple(tuple(t[:, :, :-1] for t in pair) for pair in self.kv)
            self.padding, self.positions, self.anchors = (v[:, :-1] for v in (self.padding, self.positions, self.anchors))
            self.router_kv = tuple(t[:, :, :-1] for t in self.router_kv)
            self.families = self.families[:, :-1]

    def decode(self, *, audio_context, **rows):
        if self.model.training:
            raise ValueError("decoder KV cache requires eval mode")
        from maimai_baseline.radar_contract import is_radar
        style = rows.get("style_context") if self.model.config.latent_enabled or is_radar(self.model.config) else None
        signature = None if style is None else style.signature()
        if self.length and signature != self.style_signature:
            raise ValueError("decoder KV style context changed; rebuild cache and prefill the complete prefix")
        if signature != self.style_signature:
            self.style_signature = signature
            self.style_kv = None if style is None else self.model.style_conditioner.prepare_kv(style)
            self.router_style_kv = None
        if self.start_tick is not None and not torch.equal(self.start_tick, rows["start_tick"]):
            raise ValueError("decoder KV cache cannot change window origin")
        if self.audio_context is not audio_context:
            if self.audio_context is not None:
                raise ValueError("decoder KV cache cannot cross audio windows")
            self.audio_context = audio_context
            self.start_tick = rows['start_tick'].detach().clone()
            self.audio = self.model.audio_kv(audio_context, start_tick=rows['start_tick'])
        self.hits += self.length > 0
        output = self.model.decode_prefix_rows(audio_context=audio_context, **rows,
            past_key_values=self.kv, past_padding=self.padding, audio_key_values=self.audio,
            style_key_values=self.style_kv, router_key_values=self.router_kv,
            router_style_key_values=self.router_style_kv)
        self.kv = output['self_key_values']
        self.router_kv = output['router_key_values']
        self.router_style_kv = output['router_style_key_values']
        def append(old, new):
            return new if old is None else torch.cat((old, new), 1)
        self.padding = append(self.padding, rows['row_padding_mask'])
        self.positions = append(self.positions, output['decoder_positions'])
        self.anchors = append(self.anchors, rows['row_anchor_tick'])
        self.families = append(self.families, output['decoder_family'])
        return output

    def report(self):
        seen = set()
        def size(value):
            if isinstance(value, torch.Tensor):
                storage = value.untyped_storage()
                key = value.device, storage.data_ptr()
                if key in seen:
                    return 0
                seen.add(key)
                return storage.nbytes()
            if isinstance(value, dict):
                return sum(size(v) for v in value.values())
            return sum(size(v) for v in value) if isinstance(value, (tuple, list)) else 0
        return dict(backend='mmoe-shared-attention-selected-task-kv-v1', prefix_reuse_steps=self.hits,
                    prefix_rows=self.length, tensor_storage_bytes=size((self.kv, self.audio, self.style_kv,
                        self.router_kv, self.router_style_kv, self.families, self.padding, self.positions, self.anchors)))
