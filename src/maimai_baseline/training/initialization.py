"""Deterministic initialization of the independent global-audio model."""

from __future__ import annotations
from dataclasses import asdict, replace
from maimai_baseline.contract import ARCHITECTURE, BASELINE_CONTRACT, BASE_MODEL_ARCHITECTURE, TIME_COMPONENTS, baseline_contract, EncoderCapacityConfig
from maimai_baseline.model.backbone import _BaselineCoreTransformer
from maimai_baseline.model.baseline import BaselineTransformer
from maimai_baseline.model.common import BaselineModelConfig, state_dict_sha256
from maimai_baseline.model.route import Route71Transformer, RouteArtifacts
from maimai_baseline.time.moment import LINEAR_CONTRACT as MOMENT_DISTRIBUTION, LINEAR_MS
from maimai_dataset.vocabulary import CompactVocabulary
from typing import Any
import numpy as np
import random
import torch
from maimai_baseline.audio_cross import identity as audio_cross_identity


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _module_state_sha256(module: torch.nn.Module) -> str:
    return state_dict_sha256(module.state_dict())


def _initialize_wide_model(
    vocabulary: CompactVocabulary, config: BaselineModelConfig, *, seed: int
) -> tuple[_BaselineCoreTransformer, dict[str, Any]]:
    if config.architecture != BASE_MODEL_ARCHITECTURE:
        raise ValueError("baseline initialization received a non-baseline model config")
    reference_config = replace(
        config,
        decoder_d_model=None,
        decoder_heads=None,
        decoder_feedforward=None,
        _initialization_reference=True,
    )
    _set_seed(seed)
    reference = _BaselineCoreTransformer(vocabulary, reference_config)
    encoder_state = {
        name: value.detach().cpu().clone()
        for name, value in reference.audio_encoder.state_dict().items()
    }
    reference_hash = _module_state_sha256(reference.audio_encoder)
    del reference
    _set_seed(seed)
    model = _BaselineCoreTransformer(vocabulary, config)
    model.audio_encoder.load_state_dict(encoder_state, strict=True)
    actual_hash = _module_state_sha256(model.audio_encoder)
    if actual_hash != reference_hash:
        raise RuntimeError("baseline paired encoder initialization hash mismatch")
    return model, {
        "mode": "paired-baseline-encoder-v1",
        "seed": int(seed),
        "encoder_state_sha256": actual_hash,
    }


def _initialize_delta_head(model: BaselineTransformer) -> dict[str, Any]:
    head = model.time_heads["delta_tick"]
    if head.bias is None or int(model.config.time_components) != TIME_COMPONENTS or model.moment_distribution != LINEAR_MS:
        raise ValueError("fresh v9 training requires linear MOMENT K5")
    # Exact v5 arithmetic, including float32 rounding at the 150 BPM reference.
    grids_per_tick = 60.0 / 150.0 * 100.0 / 96.0
    centers = torch.tensor((0., 24., 48., 96., 192.), device=head.bias.device, dtype=head.bias.dtype) * grids_per_tick
    scales = torch.tensor((12., 12., 16., 24., 48.), device=head.bias.device, dtype=head.bias.dtype) * grids_per_tick
    with torch.no_grad():
        bias = head.bias.reshape(5, 3)
        bias[:, 1].copy_(centers / 100.0)
        bias[:, 2].copy_(torch.log(torch.expm1((scales - .25) / 100.0)))
    return {"scheme": "baseline-tick-prior-at-reference-bpm", "reference_bpm": 150.0,
        "centers_ms": (centers * 10).detach().cpu().tolist(),
        "scales_ms": (scales * 10).detach().cpu().tolist(),
        "minimum_scale_ms": 2.5, "unit_scale_ms": 1000.0,
        "moment_distribution": dict(MOMENT_DISTRIBUTION)}


def _initialize_unpaired_model(
    vocabulary: CompactVocabulary,
    config: BaselineModelConfig,
    *,
    seed: int,
    artifacts: RouteArtifacts,
) -> tuple[BaselineTransformer, dict[str, Any]]:
    wide, wide_metadata = _initialize_wide_model(vocabulary, config, seed=seed)
    route, route_metadata = Route71Transformer.from_wide(
        wide, artifacts=artifacts, seed=seed
    )
    model = BaselineTransformer(vocabulary, config, artifacts=artifacts, seed=seed, _dense_initialization=True)
    transfer = model.load_state_dict(route.state_dict(), strict=False)
    expected_missing = {
        name for name in model.state_dict() if name.startswith(("song_audio_encoder.",))
    }
    if set(transfer.missing_keys) != expected_missing or transfer.unexpected_keys:
        raise RuntimeError("baseline expanded initialization transfer mismatch")
    delta_initialization = _initialize_delta_head(model)
    from maimai_baseline.time.duration import initialize_heads
    duration_initialization = initialize_heads(model)
    model.route_initialization = {
        **dict(route.route_initialization),
        "delta_time_head": dict(delta_initialization),
        "duration_time_heads": duration_initialization,
    }
    mismatches = [
        name
        for name, value in route.state_dict().items()
        if not torch.equal(value, model.state_dict()[name])
    ]
    if set(mismatches) != {"time_heads.delta_tick.bias", "time_heads.hold_duration_tick.bias"}:
        raise RuntimeError(f"baseline paired initialization drifted: {mismatches[:8]}")
    model.initialize_without_level()
    model.route_initialization.update({
        "cross_attention": audio_cross_identity(config.audio_cross_attention),
        "audio_cross_attention_identity": audio_cross_identity(config.audio_cross_attention),
        "level_conditioning": dict(BASELINE_CONTRACT["level_conditioning"]),
        "model_state_sha256": state_dict_sha256(model.state_dict()),
    })
    return model, {
        **wide_metadata,
        **dict(route_metadata),
        **model.route_initialization,
        "architecture": ARCHITECTURE,
        "baseline_contract": baseline_contract(config.encoder_capacity, config.decoder_rope_coordinate, config.route_head),
        "paired_route71_state_equal_except": [*mismatches, "condition.0.weight"],
        "delta_time_head_initialization": dict(delta_initialization),
        "duration_time_head_initialization": duration_initialization,
    }


def _initialize_dense_model(vocabulary, config, *, seed, artifacts):
    """Preserve the current model's seed streams without exposing a control model."""
    if config.chart_style != "none":
        model, metadata = _initialize_dense_model(vocabulary, replace(config, chart_style="none"),
                                            seed=seed, artifacts=artifacts)
        model.config = config
        model.initialize_style(seed)
        from maimai_baseline.style_contract import style_identity, style_model_contract
        digest = state_dict_sha256(model.state_dict())
        model.route_initialization["model_state_sha256"] = digest
        model.route_initialization["encoder_capacity_initialization"]["initial_model_state_sha256"] = digest
        model.route_initialization["chart_style"] = style_identity(config.chart_style)
        metadata.update(model.route_initialization)
        metadata["baseline_contract"] = style_model_contract(config, vocabulary)
        _record_audio_initialization(model, metadata)
        _set_seed(seed)
        return model, metadata
    from maimai_baseline.model.capacity import PAIRING_SCHEME
    model, metadata = _initialize_unpaired_model(vocabulary, config, seed=seed, artifacts=artifacts)
    state_hash=state_dict_sha256(model.state_dict())
    parameters=dict(model.named_parameters())
    record=dict(scheme=PAIRING_SCHEME, preset=config.encoder_capacity_preset, seed=int(seed),
        shared_parameters={n:list(p.shape) for n,p in parameters.items()},shared_parameter_count=sum(p.numel() for p in parameters.values()),
        shared_state_sha256=state_hash, initial_model_state_sha256=state_hash)
    model.route_initialization.update(encoder_capacity_initialization=record,model_state_sha256=state_hash)
    metadata.update(model.route_initialization)
    metadata.update(encoder_capacity=config.encoder_capacity.identity())
    _record_audio_initialization(model,metadata)
    _set_seed(seed)
    return model,metadata


def _initialize_model(vocabulary, config, *, seed, artifacts):
    model, metadata = _initialize_dense_model(vocabulary, config, seed=seed, artifacts=artifacts)
    from maimai_baseline.model.dual import install
    from maimai_baseline.dual_contract import identity
    from maimai_baseline.style_contract import style_model_contract
    install(model, seed)
    model.freeze_disabled_latent()
    digest = state_dict_sha256(model.state_dict())
    model.route_initialization.update(dual_decoder=identity(), model_state_sha256=digest)
    model.route_initialization['encoder_capacity_initialization']['initial_model_state_sha256'] = digest
    from maimai_baseline.model.capacity import is_shared_parameter
    shared = {n: p for n, p in sorted(model.named_parameters()) if is_shared_parameter(n)}
    model.route_initialization['encoder_capacity_initialization'].update(
        shared_parameters={n: list(p.shape) for n, p in shared.items()},
        shared_parameter_count=sum(p.numel() for p in shared.values()),
        shared_state_sha256=state_dict_sha256(shared))
    metadata.update(model.route_initialization)
    metadata['baseline_contract'] = style_model_contract(config, vocabulary)
    _record_audio_initialization(model, metadata)
    _set_seed(seed)
    return model, metadata


def _record_audio_initialization(model, metadata):
    # The P0 edge is registered under cross_attention in every decoder layer.
    # Record local and independent song paths under the new initialization identity.
    from maimai_baseline.contract import frequency_pooling_identity
    shared = {n: p for n, p in model.named_parameters()
              if not n.startswith(("song_audio_encoder.",))}
    record = {
        "scheme": "dynamic-query4x64-lift64-audio256-song-p2-audio-initialization-v1",
        "frequency_pooling": frequency_pooling_identity(),
        "shared_state_sha256": state_dict_sha256(shared),
        "shared_parameter_count": sum(p.numel() for p in shared.values()),
        "song_encoder_sha256": state_dict_sha256(model.song_audio_encoder.state_dict()),
        "edges": {**{f"d{d}_p{e}": state_dict_sha256(model.attention_for_edge(d,e).state_dict()) for d in range(model.config.decoder_layers) for e in model.audio_scales}},
    }
    model.route_initialization["audio_cross_initialization"] = record
    metadata["audio_cross_initialization"] = record
