"""Independent deterministic chart bottleneck and latent-only reconstruction."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from maimai_baseline.contract import REMOVED_FIELDS
from maimai_baseline.encoder_audio_contract import (CONTRACT, REPRESENTATION, normalize_mu, perturb_mu,
    NLL_COEFFICIENT, VARIANCE_COEFFICIENT, VARIANCE_EPS, VARIANCE_FLOOR)
from maimai_baseline.model.layers import ContinuousSinusoidal
from maimai_baseline.model.song_chart import encode_songs
from maimai_baseline.model.style import ChartStyleEncoder
from maimai_dataset.modifiers import LEGACY_FIELDS, ROLES


TIME_FIELDS = ('moment_position_tick', 'hold_duration_tick', 'branch_duration_tick')


def encode_pretrained(encoder, songs):
    """Public deterministic export adapter; never run posterior/logvar/sample."""
    features, stats = encode_songs(encoder, songs)
    with torch.autocast(device_type=features[0].device.type, enabled=False):
        pooled = nn.utils.rnn.pad_sequence([value[0] for value in features], batch_first=True)
        latents = encoder.mu(encoder.local_hidden(pooled.float()))
    valid = nn.utils.rnn.pad_sequence([song['valid'][0] for song in songs], batch_first=True)
    if getattr(encoder, 'mu_representation', None) == REPRESENTATION:
        latents = normalize_mu(latents, valid)
    return latents.masked_fill(~valid[..., None], 0.), valid, stats


class AudioPredictionLayer(nn.Module):
    """Audio-only block queries, with K/V projected once per unique audio."""
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(128, 128)
        self.key_value = nn.Linear(128, 256)
        self.output = nn.Linear(128, 128)
        self.norm = nn.LayerNorm(128)
        self.ffn_norm = nn.LayerNorm(128)
        self.ffn = nn.Sequential(nn.Linear(128, 512), nn.GELU(), nn.Dropout(.1), nn.Linear(512, 128))
        self.dropout = nn.Dropout(.1)

    def forward(self, query, memory, padding):
        keys, values = self.key_value(memory).chunk(2, -1)
        def heads(value):
            return value.reshape(*value.shape[:2], 4, 32).transpose(1, 2)
        q = heads(self.query(self.norm(query)))
        k, v = heads(keys), heads(values)
        visible = ~padding[:, None, None, :]
        value = F.scaled_dot_product_attention(q, k, v, attn_mask=visible,
            dropout_p=.1 if self.training else 0.).transpose(1, 2).reshape_as(query)
        query = query + self.dropout(self.output(value))
        return query + self.dropout(self.ffn(self.ffn_norm(query)))


class AudioOnlyPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(256, 128)
        self.position = ContinuousSinusoidal(128)
        self.layers = nn.ModuleList(AudioPredictionLayer() for _ in range(2))
        self.norm = nn.LayerNorm(128)
        self.logits = nn.Linear(128, 4)
        self.mean = nn.Linear(128, 4 * 128)
        self.logvar = nn.Linear(128, 4 * 128)

    def forward(self, teacher, blocks, valid):
        audio, mask, beats, mapping = (teacher[n] for n in
            ('audio', 'audio_valid', 'audio_beats', 'window_to_audio'))
        if (audio.ndim != 3 or audio.shape[-1] != 256 or mask.shape != audio.shape[:2]
                or mask.dtype != torch.bool or beats.shape != mask.shape or mapping.shape != blocks.shape
                or mapping.dtype != torch.long or bool((~mask.any(-1)).any())):
            raise ValueError('audio predictor requires unique native P0, validity, beats and window mapping')
        # No GT event positions, view labels or previous latent enters this head.
        memory = self.projection(audio.detach()) + self.position(beats).to(audio)
        memory = memory.masked_fill(~mask[..., None], 0.)
        block_indices = blocks[:, None] + torch.arange(4, device=blocks.device)
        pairs = torch.stack((mapping[:, None].expand_as(block_indices), block_indices), -1).reshape(-1, 2)
        unique, inverse = torch.unique(pairs, dim=0, return_inverse=True)
        # Repeated views/occurrences share exactly one stochastic prediction,
        # including when overlapping windows contain the same four-beat block.
        counts = torch.bincount(unique[:, 0], minlength=len(audio))
        offsets = counts.cumsum(0) - counts
        local = torch.arange(len(unique), device=blocks.device) - offsets[unique[:, 0]]
        query = memory.new_zeros((len(audio), int(counts.max()), 128))
        query[unique[:, 0], local] = self.position(unique[:, 1].float() * 4 + 2).to(memory)
        for layer in self.layers:
            query = layer(query, memory, ~mask)
        hidden = self.norm(query)[unique[:, 0], local][inverse].reshape(*valid.shape, 128)
        return dict(logits=self.logits(hidden), mean=self.mean(hidden).reshape(*valid.shape, 4, 128),
                    logvar=self.logvar(hidden).reshape(*valid.shape, 4, 128).clamp(-6., 2.))


class ReconstructionLayer(nn.Module):
    """Fixed row queries read four bottleneck tokens; no row self-attention."""
    def __init__(self):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(128, 4, dropout=.1, batch_first=True)
        self.norms = nn.ModuleList((nn.LayerNorm(128), nn.LayerNorm(128)))
        self.ffn = nn.Sequential(nn.Linear(128, 512), nn.GELU(), nn.Dropout(.1), nn.Linear(512, 128))
        self.dropout = nn.Dropout(.1)

    def forward(self, query, memory, valid):
        # An all-padding window gets one zero fallback key. Its final outputs
        # are masked below, while empty but valid musical spans stay visible.
        padding = ~valid.clone()
        padding[:, 0] &= valid.any(-1)
        value = self.cross_attention(self.norms[0](query), memory, memory,
            key_padding_mask=padding, need_weights=False)[0]
        query = query + self.dropout(value)
        return query + self.dropout(self.ffn(self.norms[1](query)))


class ChartEncoderPretrainer(nn.Module):
    def __init__(self, vocabulary, prototypes, query_rows, seed=20260811):
        super().__init__()
        if type(query_rows) is not int or query_rows < 1:
            raise ValueError('query_rows must be a positive fixed integer')
        self.vocabulary, self.query_rows, self.seed = vocabulary, query_rows, int(seed)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed)
            self.encoder = ChartStyleEncoder(vocabulary, prototypes, seed=self.seed)
            self.encoder.mu_representation = REPRESENTATION
            self.encoder.logvar.requires_grad_(False)
            self.audio_predictor = AudioOnlyPredictor()
            self.layers = nn.ModuleList(ReconstructionLayer() for _ in range(2))
            self.output_norm = nn.LayerNorm(128)
            sizes = {name: len(field.values) for name, field in vocabulary.fields.items()
                     if name not in (*REMOVED_FIELDS, *LEGACY_FIELDS)}
            sizes['route'] = 71
            sizes.update({'modifier_' + name: 2 for name in ROLES})
            self.field_heads = nn.ModuleDict({name: nn.Linear(128, size) for name, size in sizes.items()})
            self.time_heads = nn.ModuleDict({name: nn.Linear(128, 1) for name in TIME_FIELDS})
        position = ContinuousSinusoidal(128)
        self.register_buffer('row_queries', position(torch.arange(query_rows).float()))
        self.register_buffer('slot_positions', position(torch.tensor([2., 6., 10., 14.])))

    def decode(self, latents, valid):
        """Consume only z, four-slot validity and the same fixed Q row queries."""
        if (latents.ndim != 3 or latents.shape[1:] != (4, 128)
                or valid.shape != latents.shape[:2] or valid.dtype != torch.bool):
            raise ValueError('reconstruction requires [batch,4,128] latents and boolean validity')
        latents = latents.masked_fill(~valid[..., None], 0.)
        memory = (latents + self.slot_positions.to(latents)).masked_fill(~valid[..., None], 0.)
        query = self.row_queries.to(latents)[None].expand(len(latents), -1, -1)
        for layer in self.layers:
            query = layer(query, memory, valid)
        hidden = self.output_norm(query)
        absent = ~valid.any(-1)[:, None, None]
        return dict(field_logits={name: head(hidden).masked_fill(absent, 0.)
                                  for name, head in self.field_heads.items()},
                    time_predictions={name: head(hidden).squeeze(-1).masked_fill(absent[..., 0], 0.)
                                      for name, head in self.time_heads.items()},
                    latents=latents, valid=valid)

    def forward(self, batch):
        # Targets and all original row tensors deliberately stay outside decode.
        songs = batch['songs']
        latents, packed_valid, stats = encode_pretrained(self.encoder, songs)
        groups, blocks, window_valid = batch['window_groups'], batch['window_blocks'], batch['window_valid']
        if (groups.ndim != 1 or blocks.shape != groups.shape or groups.dtype != torch.long
                or blocks.dtype != torch.long or window_valid.shape != (len(groups), 4)
                or window_valid.dtype != torch.bool):
            raise ValueError('reconstruction window group/block/mask mismatch')
        indices = blocks[:, None] + torch.arange(4, device=blocks.device)
        lengths = torch.tensor([song['valid'].shape[1] for song in songs], device=groups.device)
        covered = (indices >= 0) & (indices < lengths[groups, None])
        safe = indices.clamp(0, latents.shape[1]-1)
        valid = window_valid & covered & packed_valid[groups[:, None], safe]
        selected = latents[groups[:, None], safe].masked_fill(~valid[..., None], 0.)
        condition, noisy = perturb_mu(selected, valid, training=self.training, return_mask=True)
        outputs = self.decode(condition, valid)
        outputs.update(latents=selected, condition_latents=condition, noisy_windows=noisy)
        # Occurrences and overlapping windows contribute once to dispersion.
        selected_pairs = torch.stack((groups[:, None].expand_as(safe)[valid], safe[valid]), -1)
        unique = torch.unique(selected_pairs, dim=0)
        outputs['variance_latents'] = latents[unique[:, 0], unique[:, 1]]
        if 'audio_teacher' in batch:
            outputs['audio_prediction'] = self.audio_predictor(batch['audio_teacher'], blocks, valid)
        outputs['encoding_stats'] = dict(stats, posterior_calls=0, reconstruction_calls=1)
        outputs['encoding_stats'].update(audio_encoder_calls=0,
            audio_predictor_kv_calls=2 if 'audio_teacher' in batch else 0,
            unique_variance_blocks=len(unique), noisy_windows=int(noisy.sum()))
        return outputs


def pretraining_loss(outputs, targets, collect_metrics=True):
    """Reconstruction + attached-target audio GMM NLL + deduplicated dispersion."""
    from maimai_baseline.training.song_ar import mixture_terms
    reconstruction, metrics = reconstruction_loss(outputs, targets, collect_metrics)
    if 'audio_prediction' not in outputs:
        raise ValueError('native encoder pretraining requires cached audio_teacher')
    prediction = outputs['audio_prediction']
    numerator, support, _ = mixture_terms(prediction['logits'], prediction['mean'], prediction['logvar'],
        outputs['latents'], outputs['valid'], collect_metrics=False)
    nll = numerator / support.clamp_min(1)
    with torch.autocast(device_type=reconstruction.device.type, enabled=False):
        unique = outputs['variance_latents'].float()
        dispersion = ((VARIANCE_FLOOR - (unique.var(0, unbiased=False) + VARIANCE_EPS).sqrt()).clamp_min(0).square().mean()
                      if len(unique) >= 2 else unique.reshape(-1)[:0].sum())
        loss = reconstruction + NLL_COEFFICIENT * nll + VARIANCE_COEFFICIENT * dispersion
    if collect_metrics:
        metrics['audio:nll'] = dict(loss=float(nll.detach()), count=int(support))
        metrics['latent:variance_floor'] = dict(loss=float(dispersion.detach()), count=1,
                                                unique_blocks=len(unique))
        metrics['latent:noise'] = dict(loss=float(outputs['noisy_windows'].float().mean()),
                                      count=len(outputs['noisy_windows']))
    return loss, metrics


def reconstruction_loss(outputs, targets, collect_metrics=True, return_terms=False):
    """FP32 mean over active categorical coordinates plus active time coordinates."""
    reference = next(iter(outputs['field_logits'].values()))
    metrics = {}
    with torch.autocast(device_type=reference.device.type, enabled=False):
        categorical_sum = reference.float().reshape(-1)[:0].sum()
        categorical_count = torch.zeros((), device=reference.device, dtype=torch.long)
        categorical_correct = torch.zeros_like(categorical_count)
        for name, target in targets['ids'].items():
            active = target.ne(-100)
            logits = outputs['field_logits'][name].float().masked_fill(~active[..., None], 0.)
            total = F.cross_entropy(logits.transpose(1, 2), target,
                                    ignore_index=-100, label_smoothing=.05, reduction='sum')
            count = active.sum()
            categorical_sum = categorical_sum + total
            categorical_count = categorical_count + count
            if collect_metrics:
                correct = (logits.argmax(-1).eq(target) & active).sum()
                categorical_correct = categorical_correct + correct
                metrics['reconstruction:field:' + name] = dict(loss=float(total.detach()/count.clamp_min(1)),
                    count=int(count), accuracy=float(correct/count.clamp_min(1)))
        time_sum = reference.float().reshape(-1)[:0].sum()
        time_count = torch.zeros_like(categorical_count)
        absolute_sum = reference.new_zeros((), dtype=torch.float32)
        for name, target in targets['times'].items():
            active = target.ne(-100)
            beats = target.float().masked_fill(~active, 0.) / 96.
            prediction = outputs['time_predictions'][name].float().masked_fill(~active, 0.)
            target_value = beats if name == 'moment_position_tick' else torch.log1p(beats)
            total = F.smooth_l1_loss(prediction, target_value, beta=1., reduction='none').masked_fill(~active, 0.).sum()
            count = active.sum()
            time_sum = time_sum + total
            time_count = time_count + count
            if collect_metrics:
                predicted_beats = prediction if name == 'moment_position_tick' else prediction.expm1().clamp_min(0.)
                absolute = (predicted_beats-beats).abs().masked_fill(~active, 0.).sum()
                absolute_sum = absolute_sum + absolute
                metrics['reconstruction:time:' + name] = dict(loss=float(total.detach()/count.clamp_min(1)),
                    count=int(count), mae_beats=float(absolute.detach()/count.clamp_min(1)))
        categorical = categorical_sum/categorical_count.clamp_min(1)
        time = time_sum/time_count.clamp_min(1)
        loss = categorical + time
    if collect_metrics:
        metrics['reconstruction:categorical'] = dict(loss=float(categorical.detach()), count=int(categorical_count),
            accuracy=float(categorical_correct/categorical_count.clamp_min(1)))
        metrics['reconstruction:time'] = dict(loss=float(time.detach()), count=int(time_count),
            mae_beats=float(absolute_sum.detach()/time_count.clamp_min(1)))
    if return_terms:
        return loss, metrics, dict(ce=dict(sum=categorical_sum, count=categorical_count),
                                  time=dict(sum=time_sum, count=time_count))
    return loss, metrics
