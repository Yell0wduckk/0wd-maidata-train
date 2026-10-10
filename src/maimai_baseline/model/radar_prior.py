"""Sixteen-beat, seven-coordinate causal joint GMM4 radar prior."""
from dataclasses import dataclass
import math

import torch
from torch import nn

from maimai_baseline.model.song_ar import SongAR, ARCache, causal_block, sample_mixture
from maimai_baseline.radar_contract import BEATS_PER_TOKEN, DIMENSION, SCALES


@dataclass(frozen=True)
class SongRadarContext:
    values: torch.Tensor
    valid: torch.Tensor

    def __post_init__(self):
        if (self.values.ndim != 3 or self.values.shape != (*self.valid.shape, DIMENSION)
                or self.valid.dtype != torch.bool):
            raise ValueError('song radar requires [batch,windows,7] and boolean validity')

    def window(self, start_tick):
        start = int(start_tick)
        ticks = BEATS_PER_TOKEN * 96
        if start < 0 or start % ticks or start//ticks >= self.values.shape[1]:
            raise ValueError('radar window must start on a covered nonnegative sixteen-beat boundary')
        return self.values[:, start//ticks], self.valid[:, start//ticks]


class RadarPrior(SongAR):
    def __init__(self):
        # Keep the existing eight-block construction and seed stream; replace
        # only the representation's input/output layers and normalization.
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(torch.initial_seed() ^ 0x52414441)
            self.input = nn.Linear(DIMENSION, 256)
            self.bos = nn.Parameter(torch.zeros(DIMENSION))
            self.mean = nn.Linear(256, 4*DIMENSION)
            self.logvar = nn.Linear(256, 4*DIMENSION)
            nn.init.zeros_(self.logvar.weight)
            nn.init.zeros_(self.logvar.bias)
        del self.latent_mean, self.latent_std
        self.register_buffer('radar_scales', torch.tensor(SCALES))

    def normalize(self, values):
        return values/self.radar_scales.to(values)

    def denormalize(self, values):
        return values*self.radar_scales.to(values)

    def set_normalization(self, *args):
        raise ValueError('radar uses fixed public-score scales, not latent statistics')

    def project_audio(self, audio, audio_valid, audio_beats, blocks=None):
        return super().project_audio(audio, audio_valid, audio_beats, blocks)

    def forward(self, inputs, *, audio, audio_valid, audio_beats, difficulty_id,
                span_beats, valid, level, projected=None, audio_mapping=None,
                cache=None, return_cache=False):
        if inputs.shape != (*valid.shape, DIMENSION) or valid.dtype != torch.bool or inputs.shape[1] < 1:
            raise ValueError('radar AR requires nonempty [group,windows,7] input')
        if audio.shape[:2] != audio_valid.shape or audio_beats.shape != audio_valid.shape:
            raise ValueError('radar AR audio shape mismatch')
        if audio_mapping is None:
            if len(audio) not in (1, len(inputs)):
                raise ValueError('radar AR requires an explicit chart-to-audio mapping')
            audio_mapping = torch.zeros(len(inputs), device=inputs.device, dtype=torch.long) if len(audio)==1 else torch.arange(len(inputs), device=inputs.device)
        if audio_mapping.shape != (len(inputs),) or audio_mapping.dtype != torch.long:
            raise ValueError('radar AR chart-to-audio mapping mismatch')
        offset = 0 if cache is None else cache.valid.shape[1]
        if cache is not None and (len(cache.main) != len(self.blocks) or len(cache.valid) != len(inputs)):
            raise ValueError('radar AR cache inventory mismatch')
        if projected is None:
            if cache is not None:
                raise ValueError('incremental radar AR requires shared audio projections')
            projected = self.project_audio(audio, audio_valid, audio_beats)
        positions = ((torch.arange(inputs.shape[1], device=inputs.device).float()+offset)*BEATS_PER_TOKEN+BEATS_PER_TOKEN/2)[None].expand_as(valid)
        condition = (self.difficulty(difficulty_id) + self.length(span_beats.float()[:, None]/16.)
                     + self.level((level.float().reshape(-1, 1)-10.)/5.))
        value = self.input(inputs).masked_fill(~valid[..., None], 0.)
        av = projected.audio_valid.index_select(0, audio_mapping)
        past_valid = None if cache is None else cache.valid
        caches = []
        for index, block in enumerate(self.blocks):
            kv = tuple(x.index_select(0, audio_mapping) for x in projected.main[index])
            value, kv_self = causal_block(block, value, condition, positions, valid, kv, av,
                None if cache is None else cache.main[index], past_valid)
            caches.append(kv_self)
        output = self.output_norm(value)
        logits = self.mixture(output).masked_fill(~valid[..., None], 0.)
        mean = self.mean(output).reshape(*valid.shape, 4, DIMENSION).masked_fill(~valid[..., None, None], 0.)
        logvar = self.logvar(output).reshape(*valid.shape, 4, DIMENSION).clamp(-6., 2.).masked_fill(~valid[..., None, None], 0.)
        if not return_cache:
            return logits, mean, logvar
        seen = valid if cache is None else torch.cat((cache.valid, valid), 1)
        return logits, mean, logvar, ARCache(caches, seen)

    @torch.no_grad()
    def sample(self, **kwargs):
        session = self.begin_sample(**kwargs)
        return session.ensure(kwargs['valid'].shape[1])

    def begin_sample(self, **kwargs):
        return RadarSamplingSession(self, **kwargs)


class RadarSamplingSession:
    def __init__(self, model, *, audio, audio_valid, audio_beats, difficulty_id,
                 span_beats, valid, level, seed, temperature=1., mixture_temperature=1., audio_mapping=None):
        if model.training or not bool(model.ready):
            raise ValueError('radar sampling requires a trained eval prior')
        if any(not math.isfinite(t) or t < 0 for t in (temperature, mixture_temperature)):
            raise ValueError('radar temperatures must be finite and nonnegative')
        self.model, self.valid = model, valid
        self.arguments = dict(audio=audio, audio_valid=audio_valid, audio_beats=audio_beats,
            difficulty_id=difficulty_id, span_beats=span_beats, level=level, audio_mapping=audio_mapping,
            projected=model.project_audio(audio, audio_valid, audio_beats))
        self.generator = torch.Generator(device=audio.device).manual_seed(int(seed))
        self.temperature, self.mixture_temperature = temperature, mixture_temperature
        self.inputs = model.bos[None, None].expand(len(valid), 1, -1)
        self.cache, self.result = None, []

    @torch.no_grad()
    def ensure(self, count):
        count = min(int(count), self.valid.shape[1])
        if count < 1:
            return SongRadarContext(self.inputs.new_empty((len(self.valid), 0, DIMENSION)), self.valid[:, :0])
        while len(self.result) < count:
            index = len(self.result)
            mask = self.valid[:, index:index+1]
            logits, mean, logvar, self.cache = self.model(self.inputs, **self.arguments,
                valid=mask, cache=self.cache, return_cache=True)
            value = sample_mixture(logits, mean, logvar, generator=self.generator,
                temperature=self.temperature, mixture_temperature=self.mixture_temperature)
            if not torch.isfinite(value).all():
                raise RuntimeError('non-finite radar sample')
            self.inputs = value.clamp(0., 1.).masked_fill(~mask[..., None], 0.)
            self.result.append(self.inputs)
        values = self.model.denormalize(torch.cat(self.result[:count], 1))
        return SongRadarContext(values, self.valid[:, :count])
