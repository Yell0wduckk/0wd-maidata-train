"""Causal decoding over the complete raw chart prefix."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from maimai_baseline.model.backbone import _BaselineCoreTransformer
from maimai_baseline.model.layers import _masked_rms
from maimai_baseline.model.rope import decoder_positions, project_kv, self_attention, attend, rotate
from maimai_baseline.radar_contract import is_radar


class RawPrefixTransformer(_BaselineCoreTransformer):
    def audio_kv(self, context, *, start_tick):
        """Project each edge, then rotate its K at the matching memory's beat."""
        if hasattr(self, 'router'):
            from maimai_baseline.model.dual import audio_kv
            return audio_kv(self, context, start_tick)
        with torch.autocast(device_type=start_tick.device.type, enabled=False):
            positions = tuple(beats.float() - start_tick.float()[:, None] / 96.
                              for beats in context.beats)
        def edge(d, e):
            k, v = project_kv(self.attention_for_edge(d, e), context.memories[-1-e])
            return rotate(k, positions[-1-e]), v
        result=tuple(tuple(edge(d, e) for e in self.audio_scales) for d in range(self.config.decoder_layers))
        self.audio_kv_stats=dict(decoder_p0_kv_calls=sum(len(row) for row in result),
            decoder_p0_kv_batch_windows=len(context.memories[-1]))
        if hasattr(self,'a_encoding_stats'):self.a_encoding_stats.update(self.audio_kv_stats)
        return result

    def _decode_layers(self, rows, *, audio_context, positions, start_tick, row_padding_mask,
                       past_key_values=None, past_padding=None, audio_key_values=None,
                       style_context=None, style_key_values=None):
        audio_key_values = self.audio_kv(audio_context, start_tick=start_tick) if audio_key_values is None else audio_key_values
        cached, norms, ratios = [], [], []
        for d, layer in enumerate(self.decoder_layers):
            value, kv = self_attention(layer.self_attention, layer.norm1(rows), positions,
                row_padding_mask, past=None if past_key_values is None else past_key_values[d],
                past_padding=past_padding)
            cached.append(kv)
            rows = rows + layer.dropout1(value)
            before_audio, query = rows, layer.norm2(rows)
            residuals = []
            for slot, e in enumerate(self.audio_scales):
                valid = ~audio_context.padding_masks[-1-e]
                allowed = valid[:, None, None, :].clone()
                allowed[..., 0] |= ~allowed.any(-1)
                value = attend(self.attention_for_edge(d, e), query, audio_key_values[d][slot], allowed=allowed, positions=positions)
                value = value * valid.any(-1)[:, None, None].to(value.dtype)
                residuals.append(layer.dropout2(value))
            residual = torch.stack(residuals).sum(0)
            rows = before_audio + residual
            if style_context is not None:
                rows = self.style_conditioner(d, rows, positions, style_context, style_key_values)
            norms.append(residual.float().norm(dim=-1).mean())
            ratios.append(_masked_rms(residual, row_padding_mask) / _masked_rms(before_audio, row_padding_mask).clamp_min(1e-12))
            rows = rows + layer.dropout3(layer.linear2(layer.dropout_ff(F.gelu(layer.linear1(layer.norm3(rows))))))
            rows = rows.masked_fill(row_padding_mask.unsqueeze(-1), 0.)
        return rows, norms, ratios, tuple(cached)

    def decode_prefix_rows(self, *, audio_context, input_ids, input_times, input_time_active,
                           row_padding_mask, row_anchor_tick, difficulty_id, level, start_tick, end_tick,
                           apply_grammar_mask=True, past_key_values=None, past_padding=None,
                           audio_key_values=None, style_context=None, style_key_values=None,
                           decoder_family=None, family_selector=None, router_key_values=None,
                           router_style_key_values=None):
        if hasattr(self, 'router'):
            from maimai_baseline.model.dual import decode
            return decode(self, audio_context=audio_context, input_ids=input_ids,
                input_times=input_times, input_time_active=input_time_active,
                row_padding_mask=row_padding_mask, row_anchor_tick=row_anchor_tick,
                difficulty_id=difficulty_id, level=level, start_tick=start_tick, end_tick=end_tick,
                apply_grammar_mask=apply_grammar_mask, past_key_values=past_key_values,
                past_padding=past_padding, audio_key_values=audio_key_values,
                style_context=style_context, style_key_values=style_key_values,
                decoder_family=decoder_family, family_selector=family_selector,
                router_key_values=router_key_values, router_style_key_values=router_style_key_values)
        if (self.config.latent_enabled or is_radar(self.config)) and self.config.chart_style != "none":
            if style_context is None:
                raise ValueError("style model requires an explicit StyleContext")
            if style_key_values is None:
                style_key_values = self.style_conditioner.prepare_kv(style_context)
        elif style_context is not None:
            raise ValueError("style context supplied to a baseline control model")
        rows = self._row_embedding(input_ids, input_times, input_time_active)
        positions = decoder_positions(self.config, row_anchor_tick, start_tick, row_padding_mask)
        condition = self._condition_embedding(difficulty_id=difficulty_id, level=level,
                                              start_tick=start_tick, end_tick=end_tick)
        rows = (rows + condition[:, None]).masked_fill(row_padding_mask.unsqueeze(-1), 0.)
        rows, norms, ratios, kv = self._decode_layers(rows, audio_context=audio_context,
            positions=positions, start_tick=start_tick, row_padding_mask=row_padding_mask, past_key_values=past_key_values,
            past_padding=past_padding, audio_key_values=audio_key_values,
            style_context=style_context, style_key_values=style_key_values)
        decoded = self.decoder_norm(rows).masked_fill(row_padding_mask.unsqueeze(-1), 0.)
        logits = {field: head(decoded) for field, head in self.field_heads.items()}
        if apply_grammar_mask:
            allowed = self.grammar_mask(input_ids['event'], input_ids.get('kind'),
                input_times.get('delta_tick'), input_time_active.get('delta_tick'))
            logits['event'] = logits['event'].masked_fill(~allowed, -10_000.)
        times = {}
        for field, head in self.time_heads.items():
            with torch.autocast(device_type=decoded.device.type, enabled=False):
                value = F.linear(decoded.float(), head.weight.float(), None if head.bias is None else head.bias.float())
            times[field] = value.reshape(*value.shape[:-1], self.config.time_components, 3)
        result = dict(field_logits=logits, time_outputs=times, decoded=decoded,
            time_unit_scale=self.config.time_unit_scale, self_key_values=kv,
            decoder_positions=positions,
            audio_diagnostics={**dict(audio_context.diagnostics or {}),
                'decoder_layer_residual_norms': torch.stack(norms),
                'decoder_layer_cross_residual_ratios': torch.stack(ratios),
                'decoder_input_rows': (~row_padding_mask).sum()})
        if self.delta_candidate_enabled:
            result['delta_candidate'] = dict(query=self.delta_candidate_query(decoded),
                keys=audio_context.candidate_keys, padding_mask=audio_context.candidate_padding_mask,
                beats=audio_context.candidate_beats, gate=self.delta_candidate_gate,
                lookup_left=audio_context.candidate_lookup_left, lookup_right=audio_context.candidate_lookup_right,
                lookup_weight=audio_context.candidate_lookup_weight, lookup_padding=audio_context.candidate_lookup_padding,
                lookup_start_tick=audio_context.candidate_lookup_start_tick)
        return result

    def forward_prefix(self, *, audio, audio_padding_mask, frame_beats, frame_bpms,
                       audio_start_tick, window_to_audio, interpolation_geometry, song_audio, **rows):
        song_audio = song_audio.for_audio_windows(window_to_audio, len(audio))
        context = self.encode_audio(audio, audio_padding_mask, frame_beats, frame_bpms,
            start_tick=rows['start_tick'] if audio_start_tick is None else audio_start_tick,
            end_tick=rows['end_tick'], interpolation_geometry=interpolation_geometry, song_audio=song_audio)
        if window_to_audio is not None:
            context = context.map_audio_windows(window_to_audio)
        context = self.prepare_decoder_context(context, **{k: rows[k] for k in ('difficulty_id', 'level', 'start_tick', 'end_tick')})
        result = self.decode_prefix_rows(audio_context=context, **rows)
        result['audio_context'] = context
        return result
