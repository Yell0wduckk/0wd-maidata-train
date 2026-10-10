"""Current v9 model and forward adapter."""

from __future__ import annotations
from maimai_baseline.contract import BASELINE_CONTRACT, SOURCE_DELTA_FIELD, TIME_UNIT_SCALES
from maimai_baseline.model.common import AudioContext, BaselineModelConfig, state_dict_sha256
from maimai_baseline.model.layers import _masked_rms
from maimai_baseline.model.route import Route71Transformer, RouteArtifacts, route71_forward
from maimai_dataset.vocabulary import CompactVocabulary
from torch import nn
from torch.nn import functional
from typing import Any, Mapping
import copy
import torch
from maimai_baseline.audio_cross import scales as audio_cross_scales, identity as audio_cross_identity
from maimai_baseline.radar_contract import is_radar
from maimai_baseline.style_contract import is_frozen, is_native, is_draft


def _internal_times(values: Mapping[str, torch.Tensor], field: str = "delta_ms") -> dict[str, torch.Tensor]:
    result = dict(values)
    if field not in result:
        raise ValueError(f"model input is missing {field}")
    if SOURCE_DELTA_FIELD in result:
        raise ValueError("model input leaked delta_tick")
    result[SOURCE_DELTA_FIELD] = result.pop(field)
    return result


class BaselineTransformer(Route71Transformer):
    """Selected fused-scale Route-71, no level, and seconds-scaled delta inputs."""

    def __init__(
        self,
        vocabulary: CompactVocabulary,
        config: BaselineModelConfig,
        *,
        artifacts: RouteArtifacts,
        seed: int = 20260811,
        _dense_initialization: bool = False,
    ) -> None:
        from maimai_baseline.time.moment import LINEAR_MS
        from maimai_baseline.time.duration import LINEAR_DURATION
        self.time_semantics = "bpm96-ms"
        self.moment_distribution = LINEAR_MS
        self.duration_distribution = LINEAR_DURATION
        super().__init__(vocabulary, config, artifacts=artifacts, seed=seed)
        from maimai_baseline.model.song_audio import SongAudioEncoder
        self.song_audio_encoder=SongAudioEncoder()
        self.initialize_without_level()
        if self.parameter_count() != config.parameter_count("baseline"):
            raise RuntimeError("baseline selected audio parameter count drifted")
        self.route_initialization.update({
            "cross_attention": audio_cross_identity(config.audio_cross_attention),
            "audio_cross_attention_identity": audio_cross_identity(config.audio_cross_attention),
            "level_conditioning": dict(BASELINE_CONTRACT["level_conditioning"]),
            "model_state_sha256": state_dict_sha256(self.state_dict()),
        })
        if config.chart_style != "none":
            self.initialize_style(seed)
        if not _dense_initialization:
            from maimai_baseline.model.dual import install
            install(self, seed)
        self.song_dense = True
        if not _dense_initialization:
            self.freeze_disabled_latent()

    def freeze_disabled_latent(self):
        """Freeze after construction; keep every parent initialization tensor."""
        if is_draft(self.config):
            self.draft_generator.requires_grad_(False).eval()
            self.audio_encoder.requires_grad_(False).eval()
            self.song_audio_encoder.requires_grad_(False).eval()
            return
        if is_frozen(self.config):
            self.chart_style_encoder.requires_grad_(False).eval()
            self.style_prior.requires_grad_(False).eval()
            if is_native(self.config):
                self.audio_encoder.requires_grad_(False).eval()
                self.song_audio_encoder.requires_grad_(False).eval()
            return
        if is_radar(self.config):
            self.chart_style_encoder.requires_grad_(False)
            return
        if self.config.latent_enabled:
            return
        for name, parameter in self.named_parameters():
            if (name.startswith(('chart_style_encoder.', 'style_conditioner.', 'style_prior.',
                                 'router.style.', 'router.norms.2.'))
                    or name == 'router.gate'
                    or (name.startswith('decoder_layers.') and '.shared.style_' in name)):
                parameter.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if is_draft(self.config):
            self.draft_generator.eval()
        elif is_frozen(self.config):
            self.chart_style_encoder.eval()
            if not any(parameter.requires_grad for parameter in self.style_prior.parameters()):
                self.style_prior.eval()
        if is_native(self.config):
            self.audio_encoder.eval()
            self.song_audio_encoder.eval()
        return self

    def initialize_style(self, seed):
        from maimai_baseline.model.style import ChartStyleEncoder, StyleConditioner
        from maimai_baseline.model.song_ar import SongAR
        from maimai_baseline.architecture import PRIOR_SEED_XOR
        from maimai_baseline.style_contract import STYLE_SEED_XOR
        if hasattr(self, "style_conditioner"):
            raise ValueError("style modules are already initialized")
        if is_draft(self.config):
            from maimai_baseline.model.audio_draft import AudioDraftGenerator, CONTRACT, LEGACY_CONTRACT
            from maimai_baseline.draft_contract import geometry
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(int(seed) ^ STYLE_SEED_XOR)
                self.style_conditioner = StyleConditioner()
                self.condition_view = nn.Embedding(2, 128)
            structure = CONTRACT if geometry(self.config).beats_per_token == 1 else LEGACY_CONTRACT
            self.draft_generator = AudioDraftGenerator(seed=seed, structure=structure).requires_grad_(False).eval()
            self.register_buffer('draft_condition_std', torch.ones(128))
            self.draft_generator_metadata = None
            if self.parameter_count() != self.expected_parameter_count():
                raise RuntimeError('draft model parameter count drifted')
            return
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed) ^ STYLE_SEED_XOR)
            self.chart_style_encoder = ChartStyleEncoder(self.vocabulary, self.route_embedding.prototypes, seed=seed)
            self.style_conditioner = StyleConditioner()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed) ^ PRIOR_SEED_XOR)
            if is_radar(self.config):
                from maimai_baseline.model.radar_prior import RadarPrior
                self.style_prior = RadarPrior()
            else:
                self.style_prior = SongAR()
                self.style_prior.requires_grad_(False)
        if is_radar(self.config):
            from maimai_baseline.model.radar import RadarEncoder
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(int(seed) ^ 0x52414445)
                self.radar_encoder = RadarEncoder()
        if self.parameter_count() != self.expected_parameter_count():
            raise RuntimeError("style model parameter count drifted")

    def expected_parameter_count(self):
        from maimai_baseline.style_contract import style_parameter_count
        from maimai_baseline.dual_contract import parameter_delta
        return self.config.parameter_count() + (parameter_delta(self.vocabulary, self.config.chart_style != 'none') if hasattr(self, 'router') else 0) + (
            style_parameter_count(self.vocabulary, self.config.chart_style) if self.config.chart_style != "none" else 0)

    def radar_context(self, values, valid=None):
        if not is_radar(self.config):
            raise ValueError('radar context requires the public-score model')
        return self.radar_encoder(values, valid)

    def encode_chart_style(self, inputs):
        return self.chart_style_encoder(inputs)

    def draft_context(self, inputs, *, perturb=None):
        """One draw shared by router and every Decoder condition residual."""
        from maimai_baseline.model.style import draft_context
        tokens, valid = inputs['latent_tokens'].detach(), inputs['latent_valid']
        if perturb is None:
            perturb = self.training
        if perturb:
            chosen = torch.rand((len(tokens), 1, 1), device=tokens.device) < 0.5
            tokens = tokens + chosen * torch.randn_like(tokens) * self.draft_condition_std * 0.05
        view = inputs.get('condition_view')
        if view is None:
            view = torch.zeros(len(tokens), dtype=torch.long, device=tokens.device)
        if view.shape != (len(tokens),) or view.dtype != torch.long or bool(((view < 0) | (view > 1)).any()):
            raise ValueError('draft condition view must be identity/mirror ids')
        tokens = tokens + self.condition_view(view)[:, None]
        return draft_context(dict(latent_tokens=tokens, latent_valid=valid), self.config)

    def encode_audio(self, *args, **kwargs):
        if is_native(self.config):
            raise ValueError('native audio requires --teacher-audio-cache prepared with the external full producer')
        return super().encode_audio(*args, **kwargs)

    def sample_style(self, *args, **kwargs):
        raise ValueError("sample the whole song with style_prior.sample, then slice decoder windows")

    @property
    def audio_scales(self):
        return audio_cross_scales(self.config.audio_cross_attention)

    def initialize_without_level(self) -> None:
        with torch.no_grad():
            self.condition[0].weight[:, 0].zero_()

    def require_zero_level_column(self) -> None:
        if not bool(torch.all(self.condition[0].weight[:, 0].detach() == 0)):
            raise ValueError("baseline checkpoint requires an exactly zero level column")

    def _condition_embedding(
        self,
        *,
        difficulty_id: torch.Tensor,
        level: torch.Tensor,
        start_tick: torch.Tensor,
        end_tick: torch.Tensor,
    ) -> torch.Tensor:
        return super()._condition_embedding(
            difficulty_id=difficulty_id,
            level=torch.zeros_like(level),
            start_tick=start_tick,
            end_tick=end_tick,
        )

    def attention_for_edge(self, decoder: int, encoder: int) -> nn.MultiheadAttention:
        if not (0 <= decoder < self.config.decoder_layers and encoder in self.audio_scales):
            raise ValueError("local P0 edge requires a valid decoder index")
        if encoder == 0:
            if hasattr(self, 'router'):
                return self.decoder_layers[decoder].audio_kv
            return self.decoder_layers[decoder].cross_attention
        raise ValueError("only local P0 is a decoder scale")

    def _row_embedding(
        self,
        input_ids: Mapping[str, torch.Tensor],
        input_times: Mapping[str, torch.Tensor],
        input_time_active: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        adjusted = dict(input_times)
        if SOURCE_DELTA_FIELD not in adjusted:
            raise ValueError("internal millisecond row input is missing its compatibility slot")
        # The frozen Route-71 implementation divides every timing input by 96.
        # Rescale only delta so that its resulting coordinate is delta_ms/1000
        # (seconds); duration coordinates remain byte-for-byte unchanged.
        adjusted[SOURCE_DELTA_FIELD] = (
            adjusted[SOURCE_DELTA_FIELD] * (96.0 / 1000.0)
        )
        return super()._row_embedding(
            input_ids,
            adjusted,
            input_time_active,
        )

    def decode_prefix_rows(self, **kwargs: Any) -> dict[str, Any]:
        values = dict(kwargs)
        field = "delta_ms"
        values["input_times"] = _internal_times(values["input_times"], field)
        values["input_time_active"] = _internal_times(
            values["input_time_active"], field
        )
        return self._time_outputs(super().decode_prefix_rows(**values))

    def router_forward(self, **inputs):
        """Prefix-only router pass; accepts no GT family or expert hidden state."""
        from maimai_baseline.model.dual import router_forward
        values = dict(inputs)
        for name in ('input_times', 'input_time_active'):
            values[name] = _internal_times(values[name], 'delta_ms')
        return router_forward(self, **values)

    def experts_forward(self, router_state, decoder_family, **kwargs):
        """Explicit family pass; returns selected conditional heads and validity masks."""
        from maimai_baseline.model.dual import experts_forward
        return self._time_outputs(self._route_outputs(experts_forward(
            self, router_state, decoder_family, **kwargs)))

    def _time_outputs(self, result):
        field = 'delta_ms'
        time_outputs = dict(result["time_outputs"])
        if SOURCE_DELTA_FIELD in time_outputs:
            time_outputs[field] = time_outputs.pop(SOURCE_DELTA_FIELD)
        result["time_outputs"] = time_outputs
        if 'time_valid' in result and SOURCE_DELTA_FIELD in result['time_valid']:
            result['time_valid'][field] = result['time_valid'].pop(SOURCE_DELTA_FIELD)
        result["time_unit_scales"] = dict(TIME_UNIT_SCALES)
        result["delta_time_parameterization"] = self.time_semantics
        result["moment_distribution"] = self.moment_distribution
        from maimai_baseline.time.duration import distribution_contract
        result["duration_distribution"] = distribution_contract(self.duration_distribution)
        return result


def baseline_forward(
    model: BaselineTransformer, batch: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(model, BaselineTransformer):
        raise TypeError("baseline forward received an incompatible model")
    if 'song_dense_inputs' in batch:
        from maimai_baseline.model.song_dense import forward
        return forward(model, batch)
    if is_radar(model.config) and 'style_context' not in batch:
        if 'radar_values' not in batch:
            raise ValueError('radar model requires the current GT window radar')
        batch = dict(batch, style_context=model.radar_context(batch['radar_values'], batch.get('radar_valid')))
    posterior = None
    if is_draft(model.config) and 'style_context' not in batch:
        batch = dict(batch, style_context=model.draft_context(batch['global_style_inputs']))
    elif is_frozen(model.config) and "style_context" not in batch:
        from maimai_baseline.model.style import frozen_context
        batch = dict(batch, style_context=frozen_context(batch['global_style_inputs'],
            perturb=is_native(model.config) and model.training))
    elif model.config.latent_enabled and model.config.chart_style != "none" and "style_context" not in batch:
        if "global_style_inputs" not in batch:
            raise ValueError("style training requires CPU-prepared GT ownership")
        from maimai_baseline.model.song_chart import window_posteriors
        context, posterior = window_posteriors(model, batch['global_style_inputs'], sample=model.training)
        batch = dict(batch, style_context=context)
    if "song_audio" not in batch:
        from maimai_baseline.data.song_audio import encode_batch
        batch=dict(batch,song_audio=encode_batch(model,batch["song_audio_inputs"] if "song_audio_inputs" in batch else batch["global_style_inputs"]["song_audio"]))
    if posterior is None:
        return route71_forward(model, batch)
    # Observe actual encoder calls; the normal prefix path already batches audio.
    audio_calls = []
    handle = model.audio_encoder.register_forward_pre_hook(lambda _m, xs: audio_calls.append(len(xs[0])))
    try:
        result = route71_forward(model, batch)
    finally:
        handle.remove()
    model.a_encoding_stats.update(audio_encoder_calls=len(audio_calls),
                                  audio_encoder_batch_size=sum(audio_calls))
    if posterior is not None:
        result["style_posterior"] = posterior
    return result
