"""GT posterior and the extra, fully visible style attention memory."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from maimai_baseline.model.layers import ContinuousSinusoidal
from maimai_baseline.model.rope import attend, project_kv, rotate
from maimai_baseline.contract import REMOVED_FIELDS
from maimai_dataset.encoding import TIME_FIELDS
from maimai_dataset.modifiers import LEGACY_FIELDS
from maimai_baseline.model.modifiers import ModifierProjection
from maimai_baseline.architecture import DECODER_LAYERS, INITIAL_DECODER_LAYERS, DECODER_LAYER_SEED_XOR, PRESERVED_DECODER_LAYERS, NEW_DECODER_LAYER_SEED_XOR
from maimai_baseline.style_contract import STYLE_DIM, STYLE_SLOTS, STYLE_BEATS


@dataclass(frozen=True)
class StyleContext:
    tokens: torch.Tensor
    beats: torch.Tensor
    valid: torch.Tensor
    enabled: bool = True

    def __post_init__(self):
        if (self.valid.ndim != 2 or self.valid.shape[1] < 1
                or self.tokens.shape != (*self.valid.shape, STYLE_DIM)):
            raise ValueError("style context must be [batch,slots,128]")
        if self.beats.shape != self.valid.shape or self.valid.dtype != torch.bool:
            raise ValueError("style coordinates/mask mismatch")

    def signature(self):
        # Tensor version counters detect in-place edits without copying GPU data.
        return (self.enabled, *((id(v), v._version, v.device, v.dtype, tuple(v.shape))
                                for v in (self.tokens, self.beats, self.valid) if v is not None))


def _cached_context(inputs, *, detach):
    tokens, valid = inputs['latent_tokens'], inputs['latent_valid']
    if (not isinstance(tokens, torch.Tensor) or tokens.ndim != 3
            or tokens.shape[1:] != (STYLE_SLOTS, STYLE_DIM) or not tokens.is_floating_point()
            or not isinstance(valid, torch.Tensor) or valid.shape != tokens.shape[:2]
            or valid.dtype != torch.bool or valid.device != tokens.device
            or not bool(torch.isfinite(tokens).all())):
        raise ValueError('frozen-mu condition requires finite [windows,4,128] and boolean validity')
    beats = (torch.arange(STYLE_SLOTS, device=tokens.device, dtype=torch.float32)*STYLE_BEATS+2)[None].expand_as(valid)
    value = tokens.detach() if detach else tokens
    return StyleContext(value.masked_fill(~valid[..., None], 0.), beats, valid)


def frozen_context(inputs, *, perturb=False):
    """Consume final deterministic mu; never execute an encoder or posterior."""
    if perturb:
        from maimai_baseline.encoder_audio_contract import perturb_mu
        inputs = dict(inputs, latent_tokens=perturb_mu(inputs['latent_tokens'].detach(), inputs['latent_valid']))
    return _cached_context(inputs, detach=True)


def draft_context(inputs, contract):
    """Use the selected draft geometry, retaining trainable view gradients."""
    from maimai_baseline.draft_contract import geometry
    shape = geometry(contract)
    tokens, valid = inputs['latent_tokens'], inputs['latent_valid']
    if (not isinstance(tokens, torch.Tensor) or tokens.ndim != 3
            or tokens.shape[1:] != (shape.condition_slots, STYLE_DIM)
            or not tokens.is_floating_point() or not isinstance(valid, torch.Tensor)
            or valid.shape != tokens.shape[:2] or valid.dtype != torch.bool
            or valid.device != tokens.device or not bool(torch.isfinite(tokens).all())):
        raise ValueError('draft condition requires selected finite token geometry and boolean validity')
    beats = ((torch.arange(shape.condition_slots, device=tokens.device, dtype=torch.float32)
              + shape.first_slot + .5) * shape.beats_per_token)[None].expand_as(valid)
    return StyleContext(tokens.masked_fill(~valid[..., None], 0.), beats, valid)


def sampled_context(tokens, valid):
    """A frozen decoder still differentiates with respect to prior samples."""
    return _cached_context(dict(latent_tokens=tokens, latent_valid=valid), detach=False)


@dataclass(frozen=True)
class StylePosterior:
    mu: torch.Tensor
    logvar: torch.Tensor
    beats: torch.Tensor
    valid: torch.Tensor

    def context(self, *, sample=True, generator=None):
        value = self.mu
        if sample:
            noise = torch.randn(value.shape, device=value.device, dtype=value.dtype, generator=generator)
            value = value + torch.exp(0.5 * self.logvar) * noise
        return StyleContext(value.masked_fill(~self.valid[..., None], 0.), self.beats, self.valid)

    def kl(self, free_bits=0.25):
        with torch.autocast(device_type=self.mu.device.type, enabled=False):
            raw = 0.5 * (self.mu.float().square() + self.logvar.float().exp() - 1 - self.logvar.float())
            per_slot = raw.clamp_min(free_bits).mean(-1) * 128
            loss = (per_slot * self.valid).sum() / self.valid.sum().clamp_min(1)
        return loss, raw


class ChartStyleEncoder(nn.Module):
    def __init__(self, vocabulary, prototypes, *, seed=20260811):
        super().__init__()
        width = 128
        sizes = {n: len(v.values) for n, v in vocabulary.fields.items() if n not in (*REMOVED_FIELDS, *LEGACY_FIELDS)}
        self.embeddings = nn.ModuleDict({n: nn.Embedding(size, width, padding_idx=0) for n, size in sizes.items()})
        from maimai_baseline.model.route import GeometryProjection
        from maimai_baseline.route_contract import GEOMETRY_SEED_XOR
        self.embeddings["route"] = GeometryProjection(prototypes, width, seed=seed ^ GEOMETRY_SEED_XOR ^ 128)
        # MOMENT intervals are encoded in local beats, independent of prior history.
        self.time_embeddings = nn.ModuleDict({n: ContinuousSinusoidal(width) for n in TIME_FIELDS if n != "delta_tick"})
        self.modifier_projection = ModifierProjection(width)
        self.note_event_id = vocabulary.event_id("NOTE")
        self.branch_event_id = vocabulary.event_id("SLIDE_BRANCH")
        self.position = ContinuousSinusoidal(width)
        self.note_combine = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, width))
        self.moment_combine = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, width))
        self.empty = nn.Parameter(torch.zeros(width))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(width, 4, 512, 0.1, activation="gelu", batch_first=True, norm_first=True),
            2, norm=nn.LayerNorm(width), enable_nested_tensor=False)
        self.pool = nn.Linear(width, 1)
        self.mu = nn.Linear(width, STYLE_DIM)
        self.logvar = nn.Linear(width, STYLE_DIM)
        self.local_hidden = nn.Sequential(nn.Linear(128, 128), nn.GELU())

    def encode_features(self, packed):
        ids = packed["ids"]
        rows = next(iter(ids.values())).shape[0]
        device = self.empty.device
        value = self.modifier_projection(ids)
        mass = (ids["event"].eq(self.note_event_id) | ids["event"].eq(self.branch_event_id))[:, None].to(value.dtype)
        for name, embedding in self.embeddings.items():
            active = ids[name].ne(0)[:, None]
            value = value + embedding(ids[name]) * active
            mass = mass + active
        for name, embedding in self.time_embeddings.items():
            active = packed["time_active"][name][:, None]
            value = value + embedding(packed["times"][name].float() / 96.) * active
            mass = mass + active
        value = value / mass.clamp_min(1).sqrt()
        notes = value.index_select(0, packed["note_rows"])
        branches = value.new_zeros(notes.shape).index_add(0, packed["branch_note"],
                                                        value.index_select(0, packed["branch_rows"]))
        notes = self.note_combine(torch.cat((notes, branches), -1))
        moments = value.index_select(0, packed["moment_rows"])
        # MLP output may be BF16 under autocast; accumulate the set in FP32.
        members = value.new_zeros(moments.shape).index_add(0, packed["note_moment"], notes.to(value.dtype))
        moments = self.moment_combine(torch.cat((moments, members), -1))
        moments = moments + self.position(packed["moment_local"]) + self.position(packed["moment_delta"])
        count, width = packed["valid"].numel(), packed["moment_width"]
        flat = moments.new_zeros((count * width, 128))
        indices = packed["moment_slot"] * width + packed["moment_ordinal"]
        flat = flat.index_copy(0, indices, moments)
        mask = torch.ones(count * width, dtype=torch.bool, device=device)
        mask[indices] = False
        mask = mask.reshape(count, width)
        no_moments = mask.all(1)
        sequence = flat.reshape(count, width, 128)
        # Empty spans remain valid data; only absent tail spans are padding.
        sequence = sequence + (no_moments[:, None, None] &
                               torch.arange(width, device=device)[None, :, None].eq(0)) * self.empty
        mask[:, 0] &= ~no_moments
        encoded = self.transformer(sequence, src_key_padding_mask=mask)
        weights = torch.softmax(self.pool(encoded).squeeze(-1).float().masked_fill(mask, -torch.inf), -1)
        pooled = (encoded.float() * weights[..., None]).sum(1)
        return pooled.float().reshape(*packed["valid"].shape, 128).masked_fill(~packed["valid"][..., None], 0.)

    def posterior(self, features, valid):
        with torch.autocast(device_type=features.device.type, enabled=False):
            value = self.local_hidden(features.float())
            mu, logvar = self.mu(value), self.logvar(value).clamp(-6., 2.)
        beats = (torch.arange(features.shape[1], device=features.device).float() * STYLE_BEATS + STYLE_BEATS / 2)[None].expand_as(valid)
        return StylePosterior(mu.masked_fill(~valid[..., None], 0.), logvar.masked_fill(~valid[..., None], 0.), beats, valid)

    def forward(self, packed):
        return self.posterior(self.encode_features(packed), packed["valid"])


class StyleConditioner(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(STYLE_DIM, 256)
        self.attentions = nn.ModuleList(nn.MultiheadAttention(256, 8, dropout=0.1, batch_first=True) for _ in range(INITIAL_DECODER_LAYERS))
        self.norms = nn.ModuleList(nn.LayerNorm(256) for _ in range(INITIAL_DECODER_LAYERS))
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(torch.initial_seed() ^ DECODER_LAYER_SEED_XOR)
            self.attentions.extend(nn.MultiheadAttention(256, 8, dropout=0.1, batch_first=True)
                for _ in range(PRESERVED_DECODER_LAYERS - INITIAL_DECODER_LAYERS))
            self.norms.extend(nn.LayerNorm(256) for _ in range(PRESERVED_DECODER_LAYERS - INITIAL_DECODER_LAYERS))
        # Preserve the historical attention-then-norm RNG order.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(torch.initial_seed() ^ NEW_DECODER_LAYER_SEED_XOR)
            self.attentions.extend(nn.MultiheadAttention(256, 8, dropout=0.1, batch_first=True)
                for _ in range(DECODER_LAYERS - PRESERVED_DECODER_LAYERS))
            self.norms.extend(nn.LayerNorm(256) for _ in range(DECODER_LAYERS - PRESERVED_DECODER_LAYERS))
        self.gates = nn.Parameter(torch.full((DECODER_LAYERS,), 1e-3))

    def prepare_kv(self, context):
        memory = self.project(context.tokens)
        return tuple((rotate(k, context.beats), v) for k, v in
                     (project_kv(a, memory) for a in self.attentions))

    def forward(self, layer, rows, positions, context, kv):
        if not context.enabled:
            return rows
        allowed = context.valid[:, None, None, :].clone()
        allowed[..., 0] |= ~allowed.any(-1)
        value = attend(self.attentions[layer], self.norms[layer](rows), kv[layer],
                       allowed=allowed, positions=positions)
        value = value * context.valid.any(-1)[:, None, None]
        return rows + self.gates[layer] * value
