"""Audio-only beat drafts and three-window latent-only reconstruction."""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from maimai_baseline.contract import REMOVED_FIELDS
from maimai_baseline.draft_contract import geometry
from maimai_baseline.model.chart_encoder_pretrain import ReconstructionLayer, TIME_FIELDS, reconstruction_loss
from maimai_baseline.model.layers import ContinuousSinusoidal
from maimai_baseline.model.rope import project_kv, rotate, self_attention
from maimai_baseline.model.song_ar import ARBlock, chunk_attention
from maimai_baseline.moe_contract import identity as moe_identity
from maimai_dataset.modifiers import LEGACY_FIELDS, ROLES


LEGACY_CONTRACT = dict(schema='maimai.audio_draft_generator.v1', audio_width=256, width=256,
    heads=8, ffn=1024, main_blocks=8, dimension=128, beats_per_token=4,
    queries='absolute-four-beat-centers;audio-derived-coverage',
    attention='bidirectional-self;full-native-P0-cross',
    condition='continuous-level;true-source-difficulty;audio-span',
    representation='deterministic-ln-no-affine-128', layer_norm_eps=1e-5,
    prediction='continuous-token;no-GMM;no-GT-input',
    ffn_moe=moe_identity(256, 1024, 'gelu', 0.))

CONTRACT = dict(deepcopy(LEGACY_CONTRACT), schema='maimai.audio_draft_generator.beat.v1',
    beats_per_token=1, queries='absolute-one-beat-centers;audio-derived-coverage')

SUPERVISION_CONTRACT = dict(schema='maimai.audio_draft_reconstruction.beat.v1',
    condition_slots=48, relative_centers='-15.5..31.5', row_queries='fixed-metadata-capacity',
    count_queries=16, count_classes=97, count_intervals='left-closed-right-open-beats',
    loss='field-CE+absolute-MOMENT/log1p-duration-SmoothL1+count-CE', coefficients=[1., 1., 1.],
    label_smoothing=.05)


def validate_structure(structure):
    if structure not in (CONTRACT, LEGACY_CONTRACT):
        raise ValueError('audio draft architecture or conditioning contract changed')
    return deepcopy(structure)


@dataclass(frozen=True)
class ProjectedDraftAudio:
    main: tuple
    audio_valid: torch.Tensor


class AudioDraftGenerator(nn.Module):
    """No chart, previous latent or target row enters this network."""
    def __init__(self, seed=20260811, structure=None):
        super().__init__()
        self.seed = int(seed)
        self.structure = validate_structure(CONTRACT if structure is None else structure)
        self.geometry = geometry(self.structure)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed)
            self.query = nn.Parameter(torch.zeros(256))
            self.difficulty = nn.Embedding(8, 256)
            self.level = nn.Sequential(nn.Linear(1, 256), nn.GELU(), nn.Linear(256, 256))
            self.length = nn.Linear(1, 256)
            self.blocks = nn.ModuleList(ARBlock(index) for index in range(8))
            self.output_norm = nn.LayerNorm(256)
            self.output = nn.Linear(256, 128)
            self.token_norm = nn.LayerNorm(128, eps=1e-5, elementwise_affine=False)
        self.position = ContinuousSinusoidal(256)

    def project_audio(self, audio, audio_valid, audio_beats):
        if (audio.ndim != 3 or audio.shape[-1] != 256 or audio.shape[1] < 1
                or audio_valid.shape != audio.shape[:2] or audio_valid.dtype != torch.bool
                or audio_beats.shape != audio_valid.shape or bool((~audio_valid.any(-1)).any())
                or not bool(torch.isfinite(audio[audio_valid]).all())
                or not bool(torch.isfinite(audio_beats[audio_valid]).all())):
            raise ValueError('draft requires finite unique native P0, absolute beats and valid audio')
        # Frozen teacher features never receive gradients, including through reusable K/V.
        memory = audio.detach().masked_fill(~audio_valid[..., None], 0.)
        coordinates = audio_beats.masked_fill(~audio_valid, 0.)
        return ProjectedDraftAudio(tuple((rotate(k, coordinates), v) for block in self.blocks
            for k, v in (project_kv(block.cross_attention, memory),)), audio_valid)

    def forward(self, *, audio, audio_valid, audio_beats, valid, difficulty_id, level,
                span_beats, beats=None, audio_mapping=None, projected=None):
        if valid.ndim != 2 or valid.dtype != torch.bool or valid.shape[1] < 1:
            raise ValueError('draft requires nonempty [conditions, blocks] boolean validity')
        batch, count = valid.shape
        step = self.geometry.beats_per_token
        positions = ((torch.arange(count, device=valid.device).float()+.5)*step)[None].expand(batch, -1)
        if beats is not None and (beats.shape != valid.shape or not torch.equal(beats[valid].float(), positions[valid])):
            raise ValueError('draft queries must match fixed representation beat centers')
        if (difficulty_id.shape != (batch,) or difficulty_id.dtype != torch.long
                or bool(((difficulty_id < 0) | (difficulty_id >= 8)).any())
                or level.numel() != batch or span_beats.numel() != batch
                or not bool(torch.isfinite(level).all()) or not bool(torch.isfinite(span_beats).all())):
            raise ValueError('draft requires true source difficulty, finite continuous level and audio span')
        if audio_mapping is None:
            if len(audio) not in (1, batch):
                raise ValueError('draft requires condition-to-audio mapping')
            audio_mapping = (torch.zeros(batch, dtype=torch.long, device=valid.device) if len(audio) == 1
                             else torch.arange(batch, device=valid.device))
        if (audio_mapping.shape != (batch,) or audio_mapping.dtype != torch.long
                or bool(((audio_mapping < 0) | (audio_mapping >= len(audio))).any())):
            raise ValueError('draft condition-to-audio mapping mismatch')
        if projected is None:
            projected = self.project_audio(audio, audio_valid, audio_beats)
        if len(projected.main) != len(self.blocks) or projected.audio_valid.shape != audio_valid.shape:
            raise ValueError('draft projected audio inventory mismatch')
        condition = (self.difficulty(difficulty_id) + self.level((level.reshape(-1, 1).float()-10.)/5.)
                     + self.length(span_beats.reshape(-1, 1).float()/16.))
        value = (self.query[None, None] + self.position(positions).to(condition)).masked_fill(~valid[..., None], 0.)
        av = projected.audio_valid.index_select(0, audio_mapping)
        for block, audio_kv in zip(self.blocks, projected.main):
            scale, shift = block.modulation(condition).chunk(2, -1)
            query = block.norms[0](value)*(1+scale[:, None]) + shift[:, None]
            residual, _ = self_attention(block.self_attention, query, positions, ~valid, causal=False)
            value = value + residual
            kv = tuple(item.index_select(0, audio_mapping) for item in audio_kv)
            value = value + chunk_attention(block.cross_attention, block.norms[1](value), kv, positions, av)
            value = value + block.ffn(block.norms[2](value), valid)
            value = value.masked_fill(~valid[..., None], 0.)
        # FP32 normalization defines the cache and Decoder conditioning scale.
        with torch.autocast(device_type=value.device.type, enabled=False):
            tokens = self.token_norm(self.output(self.output_norm(value.float())))
        return tokens.masked_fill(~valid[..., None], 0.)

    def generate(self, draft_inputs):
        return self(**draft_inputs)


class DraftReconstructionHead(nn.Module):
    """Standalone fixed row queries; no chart encoder or GT prefix is instantiated."""
    def __init__(self, vocabulary, query_rows, seed=20260811, structure=None):
        super().__init__()
        if type(query_rows) is not int or query_rows < 1:
            raise ValueError('query_rows must be a positive fixed integer')
        self.vocabulary, self.query_rows = vocabulary, query_rows
        self.geometry = geometry(CONTRACT if structure is None else validate_structure(structure))
        self.count_enabled = self.geometry.beats_per_token == 1
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed))
            self.layers = nn.ModuleList(ReconstructionLayer() for _ in range(2))
            self.output_norm = nn.LayerNorm(128)
            sizes = {name: len(field.values) for name, field in vocabulary.fields.items()
                     if name not in (*REMOVED_FIELDS, *LEGACY_FIELDS)}
            sizes['route'] = 71
            sizes.update({'modifier_' + name: 2 for name in ROLES})
            self.field_heads = nn.ModuleDict({name: nn.Linear(128, size) for name, size in sizes.items()})
            self.time_heads = nn.ModuleDict({name: nn.Linear(128, 1) for name in TIME_FIELDS})
            if self.count_enabled:
                self.count_head = nn.Linear(128, 97)
        position = ContinuousSinusoidal(128)
        self.register_buffer('row_queries', position(torch.arange(query_rows).float()))
        self.register_buffer('slot_positions', position(torch.tensor(self.geometry.centers)))
        if self.count_enabled:
            self.register_buffer('count_queries', position(torch.arange(16).float()+.5))

    def forward(self, latents, valid):
        if (latents.ndim != 3 or latents.shape[1:] != (self.geometry.condition_slots, 128)
                or valid.shape != latents.shape[:2] or valid.dtype != torch.bool):
            raise ValueError('draft reconstruction condition slot geometry or validity mismatch')
        latents = latents.masked_fill(~valid[..., None], 0.)
        memory = (latents+self.slot_positions.to(latents)).masked_fill(~valid[..., None], 0.)
        query = self.row_queries.to(latents)[None].expand(len(latents), -1, -1)
        if self.count_enabled:
            query = torch.cat((query, self.count_queries.to(latents)[None].expand(len(latents), -1, -1)), 1)
        for layer in self.layers:
            query = layer(query, memory, valid)
        hidden = self.output_norm(query[:, :self.query_rows])
        absent = ~valid.any(-1)[:, None, None]
        outputs = dict(field_logits={name: head(hidden).masked_fill(absent, 0.) for name, head in self.field_heads.items()},
            time_predictions={name: head(hidden).squeeze(-1).masked_fill(absent[..., 0], 0.)
                              for name, head in self.time_heads.items()}, latents=latents, valid=valid)
        if self.count_enabled:
            count_valid = valid[:, 16:32]
            outputs.update(count_logits=self.count_head(self.output_norm(query[:, self.query_rows:])).masked_fill(
                ~count_valid[..., None], 0.), count_valid=count_valid)
        return outputs


class AudioDraftPretrainer(nn.Module):
    def __init__(self, vocabulary, query_rows, seed=20260811, structure=None):
        super().__init__()
        self.vocabulary, self.query_rows, self.seed = vocabulary, int(query_rows), int(seed)
        self.structure = validate_structure(CONTRACT if structure is None else structure)
        self.geometry = geometry(self.structure)
        self.supervision = deepcopy(SUPERVISION_CONTRACT) if self.structure == CONTRACT else None
        self.generator = AudioDraftGenerator(seed, structure=self.structure)
        self.reconstruction = DraftReconstructionHead(vocabulary, query_rows, seed, structure=self.structure)

    def decode(self, latents, valid):
        return self.reconstruction(latents, valid)

    def forward(self, batch):
        inputs = batch['draft_inputs']
        latents = self.generator.generate(inputs)
        groups, blocks, window_valid = (batch[name] for name in ('window_groups', 'window_blocks', 'window_valid'))
        if (groups.ndim != 1 or groups.dtype != torch.long or blocks.shape != groups.shape
                or blocks.dtype != torch.long or window_valid.shape != (len(groups), self.geometry.condition_slots)
                or window_valid.dtype != torch.bool or bool(((groups < 0) | (groups >= len(latents))).any())):
            raise ValueError('draft reconstruction window mapping mismatch')
        indices = blocks[:, None] + self.geometry.first_slot + torch.arange(self.geometry.condition_slots, device=blocks.device)
        covered = (indices >= 0) & (indices < latents.shape[1])
        safe = indices.clamp(0, latents.shape[1]-1)
        valid = window_valid & covered & inputs['valid'][groups[:, None], safe]
        if bool((window_valid & ~valid).any()):
            raise ValueError('draft audio-derived query coverage does not include target window')
        selected = latents[groups[:, None], safe].masked_fill(~valid[..., None], 0.)
        outputs = self.decode(selected, valid)
        outputs['encoding_stats'] = dict(audio_encoder_calls=0, chart_encoder_calls=0,
            generator_calls=1, unique_conditions=len(latents), unique_audio=len(inputs['audio']),
            audio_kv_calls=len(self.generator.blocks), reconstruction_calls=1)
        return outputs


def pretraining_loss(outputs, targets, collect_metrics=True):
    """FP32 field CE + time SmoothL1 + per-beat count CE, each independently averaged."""
    loss, metrics, terms = reconstruction_loss(outputs, targets, collect_metrics=collect_metrics, return_terms=True)
    outputs['loss_terms'] = terms
    if 'count_logits' not in outputs:
        return loss, metrics  # Explicit legacy four-beat reconstruction.
    logits, valid, counts = outputs['count_logits'], outputs['count_valid'], targets.get('moment_counts')
    if (logits.shape[1:] != (16, 97) or valid.shape != logits.shape[:2] or valid.dtype != torch.bool
            or not isinstance(counts, torch.Tensor) or counts.shape != valid.shape or counts.dtype != torch.long
            or bool((((counts < 0) | (counts > 96)) & valid).any())):
        raise ValueError('draft requires 16 per-beat MOMENT count labels in 0..96')
    with torch.autocast(device_type=logits.device.type, enabled=False):
        target = counts.masked_fill(~valid, -100)
        numerator = F.cross_entropy(logits.float().transpose(1, 2), target,
            ignore_index=-100, label_smoothing=.05, reduction='sum')
        support = valid.sum()
        count_loss = numerator/support.clamp_min(1)
        loss = loss + count_loss
        terms['count'] = dict(sum=numerator, count=support)
    if collect_metrics:
        predicted = logits.detach().argmax(-1)
        metrics['reconstruction:moment_count'] = dict(loss=float(count_loss.detach()), count=int(support),
            accuracy=float(((predicted == counts) & valid).sum()/support.clamp_min(1)),
            mae=float((predicted-counts).abs().masked_fill(~valid, 0).sum()/support.clamp_min(1)))
    return loss, metrics
