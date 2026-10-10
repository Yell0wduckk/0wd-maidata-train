"""Causal four-beat four-component joint Gaussian mixture prior; the frozen A representation is unchanged."""
from dataclasses import dataclass
import math

import torch
from torch import nn

from maimai_baseline.model.rope import self_attention
from maimai_baseline.model.rope import project_kv, rotate, attend
from maimai_baseline.model.style import StyleContext


CONTRACT = dict(schema="maimai.song_latent_ar_gmm4_moe.v1", audio_width=256, width=256, heads=8, ffn=1024,
    main_blocks=8, control_blocks=0, dimension=128, beats_per_token=4,
    input="learned-bos;shifted-deterministic-mu", target="deterministic-mu", attention="causal-main",
    audio="full-P0-cross-attention-only;always-conditioned",
    positions="absolute-target-four-beat-centers", prediction="joint-diagonal-gaussian-mixture", components=4,
    component_scope="one-per-128d-vector",
    reduction="sum-dimension-logprob;logsumexp-components;sum-valid-block-nll/valid-coordinates",
    logvar_bounds=[-6., 2.], loss="joint-gmm4-nll-valid-coordinate-mean",
    sampling="categorical-component-then-vector-gaussian;separate-temperatures", cfg=False)


from maimai_baseline.moe_contract import identity as moe_identity
CONTRACT['ffn_moe'] = moe_identity(256, 1024, 'gelu', 0.)

@dataclass(frozen=True)
class SongStyleContext:
    tokens: torch.Tensor
    valid: torch.Tensor

    def __post_init__(self):
        if self.tokens.ndim!=3 or self.tokens.shape!=(*self.valid.shape,128) or self.valid.dtype!=torch.bool:
            raise ValueError('song context requires [batch,K,128] and boolean validity')

    def window(self, start_tick):
        start_tick = int(start_tick)
        if start_tick < 0 or start_tick % 384:
            raise ValueError("song latent window must start at a nonnegative four-beat boundary")
        start = start_tick // 384
        n = max(0, min(4, self.tokens.shape[1] - start))
        tokens = self.tokens.new_zeros((len(self.tokens), 4, 128))
        valid = torch.zeros(tokens.shape[:2], device=tokens.device, dtype=torch.bool)
        tokens[:, :n] = self.tokens[:, start:start+n]
        valid[:, :n] = self.valid[:, start:start+n]
        beats = (torch.arange(4, device=tokens.device).float()*4+2)[None].expand_as(valid)
        return StyleContext(tokens, beats, valid)


def chunk_attention(module, query, kv, positions, key_valid, chunk=128):
    allowed = key_valid[:, None, None, :].clone()
    allowed[..., 0] |= ~allowed.any(-1)
    pieces = [attend(module, query[:, i:i+chunk], kv, allowed=allowed,
                    positions=positions[:, i:i+chunk]) for i in range(0, query.shape[1], chunk)]
    return torch.cat(pieces, 1) * key_valid.any(-1)[:, None, None]


class ARBlock(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(256, 8, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(256, 8, kdim=256, vdim=256, batch_first=True)
        self.norms = nn.ModuleList(nn.LayerNorm(256) for _ in range(3))
        self.modulation = nn.Linear(256, 512)
        from maimai_baseline.model.token_moe import DenseTokenMoE
        self.ffn = DenseTokenMoE(256, 1024, salt=index)


@dataclass
class ARCache:
    main: list
    valid: torch.Tensor

    def select_prefix(self, indices, length):
        """Trim a teacher-forced batch for equal-length, differentiable AR tails."""
        if not 0 <= int(length) <= self.valid.shape[1]:
            raise ValueError('invalid AR prefix cache length')
        return ARCache([tuple(x.index_select(0, indices)[:, :, :length] for x in kv)
                        for kv in self.main], self.valid.index_select(0, indices)[:, :length])


@dataclass(frozen=True)
class ProjectedARAudio:
    main: tuple
    audio_valid: torch.Tensor


def causal_block(block, value, condition, positions, valid, audio_kv, audio_valid,
                 past=None, past_valid=None):
    scale, shift = block.modulation(condition).chunk(2, -1)
    query = block.norms[0](value) * (1 + scale[:, None]) + shift[:, None]
    residual, cache = self_attention(block.self_attention, query, positions, ~valid,
        past=past, past_padding=None if past_valid is None else ~past_valid, causal=True)
    value = value + residual
    value = value + chunk_attention(block.cross_attention, block.norms[1](value),
                                   audio_kv, positions, audio_valid)
    value = value + block.ffn(block.norms[2](value), valid)
    return value.masked_fill(~valid[..., None], 0.), cache


class SongAR(nn.Module):
    """Training consumes shifted targets; incremental inference uses the same graph."""
    # Share immutable audio geometry/projection implementations, not diffusion state.
    def audio_kv(self, audio, audio_beats):
        return tuple((rotate(k, audio_beats), v) for k, v in
            (project_kv(b.cross_attention, audio) for b in self.blocks))

    def project_audio(self, audio, audio_valid, audio_beats, blocks):
        # `blocks` retains the training/generation interface; full P0 is independent
        # of receiver length. Target alignment comes from absolute query RoPE.
        if (audio.ndim != 3 or audio.shape[-1] != 256 or audio.shape[1] < 1 or
                audio.shape[:2] != audio_valid.shape or audio_beats.shape != audio_valid.shape or
                audio_valid.dtype != torch.bool):
            raise ValueError("AR requires P0 [group, frames, 256] with coordinates and boolean mask")
        return ProjectedARAudio(self.audio_kv(audio, audio_beats), audio_valid)

    def __init__(self):
        super().__init__()
        self.input = nn.Linear(128, 256)
        self.bos = nn.Parameter(torch.zeros(128))
        self.difficulty = nn.Embedding(8, 256)
        self.length = nn.Linear(1, 256)
        self.level = nn.Sequential(nn.Linear(1, 256), nn.GELU(), nn.Linear(256, 256))
        self.blocks = nn.ModuleList(ARBlock(index) for index in range(CONTRACT['main_blocks']))
        self.output_norm = nn.LayerNorm(256)
        self.mean = nn.Linear(256, 4*128)
        self.logvar = nn.Linear(256, 4*128)
        self.mixture = nn.Linear(256, 4)
        nn.init.zeros_(self.mixture.weight)
        nn.init.zeros_(self.mixture.bias)
        nn.init.zeros_(self.logvar.weight)
        nn.init.zeros_(self.logvar.bias)
        self.register_buffer("latent_mean", torch.zeros(128))
        self.register_buffer("latent_std", torch.ones(128))
        self.register_buffer("ready", torch.tensor(False))

    def set_normalization(self, mean, std):
        if (mean.shape != (128,) or std.shape != (128,) or
                not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any()):
            raise ValueError("invalid AR latent normalization")
        self.latent_mean.copy_(mean.to(self.latent_mean))
        self.latent_std.copy_(std.to(self.latent_std))

    def shifted(self, target, valid):
        clean = target.masked_fill(~valid[..., None], 0.)
        return torch.cat((self.bos[None, None].expand(len(target), 1, -1), clean[:, :-1]), 1)

    def forward(self, inputs, *, audio, audio_valid, audio_beats, difficulty_id,
                span_beats, valid, level, projected=None, audio_mapping=None, cache=None, return_cache=False):
        if inputs.shape != (*valid.shape, 128) or valid.dtype != torch.bool or inputs.shape[1] < 1:
            raise ValueError("AR requires nonempty [batch, K, 128] input and boolean validity")
        if audio.shape[:2] != audio_valid.shape or audio_beats.shape != audio_valid.shape:
            raise ValueError("AR audio group shape mismatch")
        if audio_mapping is None:
            if len(audio) not in (1, len(inputs)):
                raise ValueError('AR requires a chart-to-audio mapping')
            audio_mapping = (torch.zeros(len(inputs), device=inputs.device, dtype=torch.long) if len(audio)==1
                             else torch.arange(len(inputs), device=inputs.device))
        if (audio_mapping.shape != (len(inputs),) or audio_mapping.dtype != torch.long
                or bool(((audio_mapping < 0) | (audio_mapping >= len(audio))).any())):
            raise ValueError('AR chart-to-audio mapping mismatch')
        offset = 0 if cache is None else cache.valid.shape[1]
        if cache is not None and (len(cache.main) != len(self.blocks) or len(cache.valid) != len(inputs)):
            raise ValueError("AR cache inventory mismatch")
        if projected is None:
            if cache is not None:
                raise ValueError("incremental AR requires shared full-song audio projections")
            projected = self.project_audio(audio, audio_valid, audio_beats, inputs.shape[1])
        positions = ((torch.arange(inputs.shape[1], device=inputs.device).float()+offset)*4+2)[None].expand_as(valid)
        condition = (self.difficulty(difficulty_id) + self.length(span_beats.float()[:, None]/16.)
                     + self.level((level.float().reshape(-1, 1)-10.)/5.))
        value = self.input(inputs).masked_fill(~valid[..., None], 0.)
        av = projected.audio_valid.index_select(0, audio_mapping)
        past_valid = None if cache is None else cache.valid
        main_cache = []
        for i, block in enumerate(self.blocks):
            kv = tuple(x.index_select(0, audio_mapping) for x in projected.main[i])
            value, kv_self = causal_block(block, value, condition, positions, valid, kv, av,
                None if cache is None else cache.main[i], past_valid)
            main_cache.append(kv_self)
        output = self.output_norm(value)
        logits = self.mixture(output).masked_fill(~valid[..., None], 0.)
        mean = self.mean(output).reshape(*valid.shape, 4, 128).masked_fill(~valid[..., None, None], 0.)
        logvar = self.logvar(output).reshape(*valid.shape, 4, 128).clamp(-6., 2.).masked_fill(~valid[..., None, None], 0.)
        if not return_cache:
            return logits, mean, logvar
        seen = valid if cache is None else torch.cat((cache.valid, valid), 1)
        return logits, mean, logvar, ARCache(main_cache, seen)

    @torch.no_grad()
    def sample(self, *, audio, audio_valid, audio_beats, difficulty_id, span_beats,
               valid, level, seed, temperature=1., mixture_temperature=1.):
        session = self.begin_sample(audio=audio, audio_valid=audio_valid, audio_beats=audio_beats,
            difficulty_id=difficulty_id, span_beats=span_beats, valid=valid, level=level,
            seed=seed, temperature=temperature, mixture_temperature=mixture_temperature)
        return session.ensure(valid.shape[1])

    def begin_sample(self, **kwargs):
        return ARSamplingSession(self, **kwargs)


class ARSamplingSession:
    """One song's KV and RNG. Pausing between decoder windows does not reseed."""
    def __init__(self, model, *, audio, audio_valid, audio_beats, difficulty_id,
                 span_beats, valid, level, seed, temperature=1., mixture_temperature=1.):
        if model.training or not bool(model.ready):
            raise ValueError('AR sampling requires a trained eval prior')
        if any(not math.isfinite(t) or t < 0 for t in (temperature, mixture_temperature)):
            raise ValueError('AR temperature must be finite and nonnegative')
        self.model, self.valid = model, valid
        self.arguments = dict(audio=audio, audio_valid=audio_valid, audio_beats=audio_beats,
            difficulty_id=difficulty_id, span_beats=span_beats, level=level,
            projected=model.project_audio(audio, audio_valid, audio_beats, valid.shape[1]))
        self.generator = torch.Generator(device=audio.device).manual_seed(int(seed))
        self.temperature, self.mixture_temperature = temperature, mixture_temperature
        self.inputs = model.bos[None, None].expand(len(valid), 1, -1)
        self.cache, self.result = None, []

    @torch.no_grad()
    def ensure(self, count):
        count = min(int(count), self.valid.shape[1])
        while len(self.result) < count:
            i = len(self.result)
            mask = self.valid[:, i:i+1]
            logits, mean, logvar, self.cache = self.model(self.inputs, **self.arguments,
                valid=mask, cache=self.cache, return_cache=True)
            value = sample_mixture(logits, mean, logvar, generator=self.generator,
                temperature=self.temperature, mixture_temperature=self.mixture_temperature)
            self.inputs = value.masked_fill(~mask[..., None], 0.)
            self.result.append(self.inputs)
        value = torch.cat(self.result, 1)*self.model.latent_std + self.model.latent_mean
        valid = self.valid[:, :len(self.result)]
        value = value.masked_fill(~valid[..., None], 0.)
        if not torch.isfinite(value).all():
            raise RuntimeError('non-finite AR latent')
        return SongStyleContext(value, valid)


def sample_mixture(logits, mean, logvar, *, generator, temperature, mixture_temperature):
    """Choose one component for the whole vector; categorical draw precedes noise."""
    if any(not math.isfinite(t) or t < 0 for t in (temperature, mixture_temperature)):
        raise ValueError("AR temperatures must be finite and nonnegative")
    if mixture_temperature == 0:
        selected = logits.argmax(-1)
    else:
        # Center before dividing so tiny positive temperatures cannot overflow.
        centered = logits - logits.amax(-1, keepdim=True)
        probs = (centered.double() / mixture_temperature).softmax(-1)
        selected = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1,
                                     generator=generator).reshape(logits.shape[:-1])
    index = selected[..., None, None].expand(*selected.shape, 1, mean.shape[-1])
    mu = mean.gather(-2, index).squeeze(-2)
    lv = logvar.gather(-2, index).squeeze(-2)
    if temperature == 0:
        return mu
    return mu + temperature * (.5*lv).exp() * torch.randn(
        mu.shape, device=mu.device, dtype=mu.dtype, generator=generator)
