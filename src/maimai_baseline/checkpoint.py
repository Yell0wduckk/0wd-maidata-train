from __future__ import annotations

from copy import deepcopy
from maimai_baseline.runtime_versions import require_compatible, stamp
import json
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from maimai_dataset.vocabulary import CompactVocabulary

from maimai_baseline.contract import ARCHITECTURE
from maimai_baseline.contract import BASE_MODEL_ARCHITECTURE
from maimai_baseline.contract import CHECKPOINT_SCHEMA
from maimai_baseline.contract import L3_LOSS_ID
from maimai_baseline.loss.common import loss_contract
from maimai_baseline.loss.supervision import SpecializedSupervision, from_args as supervision_from_args
from maimai_baseline.physical.contract import CONTRACT as PHYSICAL_CONTRACT
from maimai_baseline.physical.contract import LOSS_SCHEME as PHYSICAL_LOSS_ID
from maimai_baseline.model.common import BaselineModelConfig
from maimai_baseline.model.baseline import BaselineTransformer
from maimai_baseline.contract import frequency_pooling_identity, position_conditioning_identity, history_representation_identity, audio_attention_bias_identity, audio_fusion_identity
from maimai_baseline.model.route import load_route_artifacts, require_fixed_route_state
from maimai_baseline.route_contract import geometry_identity, route_head_identity
from maimai_baseline.time.moment import LINEAR_CONTRACT as MOMENT_DISTRIBUTION
from maimai_baseline.time.duration import LINEAR_CONTRACT as DURATION_DISTRIBUTION
from maimai_baseline.style_contract import style_identity, style_model_contract, is_frozen, is_draft


def _draft_cache_binding(identity):
    """Keep data and semantic export bindings; paths/source hashes are provenance."""
    if identity is None:
        return None
    from maimai_baseline.data.teacher_audio import draft_producer_identity
    from maimai_baseline.runtime_versions import operational
    value = {key: item for key, item in identity.items() if key != 'producer'}
    producer = identity.get('producer', {})
    binding = producer.get('data_binding') or {}
    value['producer'] = dict(
        generator_structure=producer.get('generator_structure'),
        completed_epochs=producer.get('completed_epochs'), seed=producer.get('seed'),
        recipe=operational(producer.get('recipe')),
        data_binding=operational({key: item for key, item in binding.items()
            if key not in ('native_audio', 'teacher_audio')}),
        native_audio=draft_producer_identity(producer['native_audio'])
            if producer.get('native_audio') is not None else None)
    return value


def validate_resume_manifest(payload, expected_manifest):
    """Check behavior flags, not descriptions of the producer or its files.

    Actual parameters are checked by the model loader. The runner owns the
    complete-epoch cursor and the supplied plan; this hook does not reload them.
    """
    saved = payload.get('model_config', {})
    if saved.get('latent_enabled', False) != expected_manifest.get('latent_enabled', False):
        raise ValueError('resume latent behavior differs from the requested model')
    requested_style = expected_manifest.get('chart_style')
    requested_style = requested_style.get('name') if isinstance(requested_style, Mapping) else requested_style
    if requested_style is not None and saved.get('chart_style') != requested_style:
        raise ValueError('resume chart condition representation differs from the requested model')
    if is_frozen(saved.get('chart_style')) and payload.get('baseline_manifest', {}).get('frozen_latent') != expected_manifest.get('frozen_latent'):
        raise ValueError('resume deterministic latent data or layout differs from the saved training inputs')
    if is_draft(saved.get('chart_style')) and _draft_cache_binding(payload.get('baseline_manifest', {}).get('draft_cache')) != _draft_cache_binding(expected_manifest.get('draft_cache')):
        raise ValueError('resume draft data or layout differs from the saved training inputs')
    from maimai_baseline.radar_contract import RADAR_ID
    if saved.get('chart_style') == RADAR_ID:
        previous = payload.get('baseline_manifest', {}).get('majradar')
        if previous != expected_manifest.get('majradar'):
            raise ValueError('resume MajRadar label data or layout differs from the saved training inputs')

def restore_baseline_rng(payload: Mapping[str, Any]) -> None:
    state = payload.get("rng_state")
    if not isinstance(state, Mapping):
        raise ValueError("baseline checkpoint is missing RNG state")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch_state = state["torch"]
    if isinstance(torch_state, torch.Tensor):
        torch_state = torch_state.detach().to("cpu")
    torch.set_rng_state(torch_state)
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all(
            [
                value.detach().to("cpu")
                if isinstance(value, torch.Tensor)
                else value
                for value in state["cuda"]
            ]
        )


def restore_baseline_training_state(payload, *, optimizer, scheduler, loader_generator=None):
    """Restore the saved execution state without re-auditing its provenance."""
    if payload.get('checkpoint_kind') != 'training':
        raise ValueError('baseline inference checkpoint cannot be resumed')
    epoch, step = payload.get('epoch'), payload.get('step')
    if (type(epoch) is not int or epoch < 0 or type(step) is not int or step < 0
            or payload.get('sampler_epoch') != epoch):
        raise ValueError('checkpoint is not at a complete epoch boundary')
    optimizer.load_state_dict(payload['optimizer_state'])
    scheduler.load_state_dict(payload['scheduler_state'])
    if scheduler.last_epoch != step:
        raise ValueError('checkpoint scheduler and update cursor differ')
    restore_baseline_rng(payload)
    loader_rng = payload.get('loader_rng')
    if is_frozen(payload.get('model_config', {}).get('chart_style')) and not _valid_loader_rng(loader_rng):
        raise ValueError('frozen-mu resume requires the DataLoader RNG state')
    if loader_generator is not None and loader_rng is not None:
        if not _valid_loader_rng(loader_rng):
            raise ValueError('invalid DataLoader RNG state')
        loader_generator.set_state(loader_rng.cpu())
    return epoch + 1, step


def _valid_loader_rng(value):
    return isinstance(value, torch.Tensor) and value.ndim == 1 and value.dtype == torch.uint8


def save_baseline_checkpoint(
    path: Path,
    *,
    model: BaselineTransformer,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    vocabulary: CompactVocabulary,
    step: int,
    epoch: int,
    profile_fingerprint: str,
    profile_summary: Mapping[str, Any],
    feature_stats: Mapping[str, Any],
    training_config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    random_seed: int,
    checkpoint_kind: str,
    source_identity: Mapping[str, Any] | None = None,
    loader_generator: torch.Generator | None = None,
) -> None:
    if checkpoint_kind not in {"training", "inference"}:
        raise ValueError(f"unsupported baseline checkpoint kind: {checkpoint_kind!r}")
    if checkpoint_kind == "training" and (optimizer is None or scheduler is None):
        raise ValueError("training checkpoint requires optimizer and scheduler")
    if not isinstance(model, BaselineTransformer):
        raise ValueError("baseline checkpoint received an incompatible model")
    from maimai_baseline.loss.difficulty import contract as difficulty_contract
    from maimai_baseline.audio_cross import identity as cross_identity

    manifest = dict(training_config.get('baseline_manifest', {}))
    # Provenance is optional; reuse the run snapshot, never rescan on save.
    source = source_identity if source_identity is not None else manifest.get('source_identity')
    training_config = dict(training_config)
    difficulty_identity = difficulty_contract(training_config.get('difficulty_loss_preset', 'uniform'))
    supervision = supervision_from_args(training_config)
    identity = supervision.identity()
    chart_style = style_identity(model.config.chart_style)
    head, geometry = route_head_identity(model.config.route_head), geometry_identity()
    attention_identity = model.config.audio_attention.identity()
    position_identity = position_conditioning_identity(model.config.decoder_rope_coordinate)
    audio_bias_identity = audio_attention_bias_identity()
    history_identity = history_representation_identity()
    cross = cross_identity(model.config.audio_cross_attention)
    fusion_identity = audio_fusion_identity()
    pooling_identity = frequency_pooling_identity()
    capacity_identity = model.config.encoder_capacity.identity()
    manifest.update(difficulty_loss_weighting=difficulty_identity,
        specialized_supervision_config=identity, chart_style=chart_style,
        latent_enabled=model.config.latent_enabled, route_head=head,
        route_geometry=geometry, audio_attention=attention_identity)
    training_config.update(baseline_manifest=manifest, specialized_supervision_config=identity,
        chart_style=model.config.chart_style, latent_enabled=model.config.latent_enabled,
        route_head=model.config.route_head, route_geometry=geometry,
        audio_attention_preset=model.config.audio_attention.preset, audio_attention=attention_identity,
        position_conditioning=position_identity, decoder_rope_coordinate=model.config.decoder_rope_coordinate,
        audio_attention_bias=audio_bias_identity, history_representation=history_identity,
        audio_cross_attention=model.config.audio_cross_attention, audio_cross_attention_identity=cross,
        audio_fusion=fusion_identity, frequency_pooling=pooling_identity,
        encoder_capacity_preset=model.config.encoder_capacity_preset, encoder_capacity=capacity_identity,
        model_initialization=dict(model.route_initialization))
    payload: dict[str, Any] = {
        "chart_style": chart_style,
        "route_head": head, "route_geometry": geometry,
        "frequency_pooling": pooling_identity,
        "position_conditioning": position_identity,
        "decoder_rope_coordinate": model.config.decoder_rope_coordinate,
        "audio_attention_bias": audio_bias_identity,
        "audio_fusion": fusion_identity,
        "audio_cross_attention_identity": cross,
        "history_representation": history_identity,
        "encoder_capacity": capacity_identity,
        "audio_attention": attention_identity,
        "schema": CHECKPOINT_SCHEMA,
        "architecture": ARCHITECTURE,
        "base_model_architecture": BASE_MODEL_ARCHITECTURE,
        "row_codec": vocabulary.row_codec,
        "checkpoint_kind": checkpoint_kind,
        "model_config": model.config.to_dict(),
        "vocabulary": vocabulary.to_dict(),
        "model_state": model.state_dict(),
        "step": int(step),
        "epoch": int(epoch),
        "sampler_epoch": int(epoch),
        "profile_fingerprint": str(profile_fingerprint),
        "metrics": dict(metrics),
        "random_seed": int(random_seed),
        "loss_scheme": PHYSICAL_LOSS_ID,
        "base_loss_scheme": L3_LOSS_ID,
        "loss_contract": loss_contract(supervision),
        "specialized_supervision_config": identity,
        "baseline_contract": style_model_contract(model.config, vocabulary),
        "moment_distribution": dict(MOMENT_DISTRIBUTION),
        "duration_distribution": dict(DURATION_DISTRIBUTION),
        "baseline_manifest": dict(manifest),
        "difficulty_loss_weighting": difficulty_identity,
        "route_artifact": dict(model.route_artifact_identity),
        "initialization": dict(model.route_initialization),
        **stamp(),
        "source_identity": deepcopy(source),
        "physical_contract": dict(PHYSICAL_CONTRACT),
        "physical_inputs": manifest.get("physical_inputs"),
        "physical_warmup_steps": training_config.get("physical_warmup_steps", 988),
    }
    # Raw MP3 inference also needs statistics in compact inference checkpoints.
    payload["feature_stats"] = dict(feature_stats)
    payload["bpm_input"] = manifest.get("bpm_input")
    if "audio_input" in manifest:
        payload["audio_input"] = dict(manifest["audio_input"])
    from maimai_baseline.radar_contract import is_radar, identity as radar_identity
    if is_radar(model.config):
        payload['radar_prior'] = ({'stage': 'A', 'contract': radar_identity()}
                                  if bool(model.style_prior.ready) else None)
    if is_frozen(model.config):
        payload['ar_prior'] = None
        if is_draft(model.config):
            payload['draft_generator'] = deepcopy(model.draft_generator_metadata)
        if getattr(model, 'frozen_encoder_import', None) is not None:
            payload['frozen_encoder_import'] = deepcopy(model.frozen_encoder_import)
    if checkpoint_kind == "training":
        if is_frozen(model.config) and loader_generator is None:
            raise ValueError('frozen-mu training checkpoint requires its DataLoader generator')
        payload.update(
            {
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "profile_summary": dict(profile_summary),
                "feature_stats": dict(feature_stats),
                "training_config": dict(training_config),
                "rng_state": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": (
                        torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available()
                        else []
                    ),
                },
                "loader_rng": None if loader_generator is None else loader_generator.get_state().cpu(),
            }
        )
    if training_config.get("sft") is not None:
        payload["sft"] = training_config["sft"]
    if is_frozen(model.config):
        from maimai_baseline.style_contract import validate_checkpoint_style
        validate_checkpoint_style(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_baseline_checkpoint(
    path: Path,
    *,
    route_artifact: Path,
    device: torch.device | str = "cpu",
    strict_source: bool = True,
    verify_artifact_payloads: bool = True,
) -> tuple[BaselineTransformer, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    return _load_baseline_payload(payload, route_artifact=route_artifact, device=device,
        strict_source=strict_source, verify_artifact_payloads=verify_artifact_payloads)


def load_frozen_stage_a(path: Path, *, route_artifact: Path, frozen_import=None):
    """Load a selected A once; historical import metadata is provenance only."""
    model, payload = load_baseline_checkpoint(Path(path), route_artifact=route_artifact,
        strict_source=True, verify_artifact_payloads=False)
    model.eval().requires_grad_(False)
    return model, payload


def _load_baseline_payload(payload, *, route_artifact, device, strict_source,
                           verify_artifact_payloads, frozen_import=None, operation=None):
    """Reconstruct the declared model and load its actual tensor inventory.

    Producer manifests and software versions do not describe tensor layout.
    Model configuration and strict state loading are the compatibility boundary.
    """
    if not isinstance(payload, Mapping):
        raise ValueError('checkpoint payload is not a mapping')
    require_compatible(payload, operation or ('training' if strict_source else 'inference'))
    if payload.get('schema') != CHECKPOINT_SCHEMA or payload.get('architecture') != ARCHITECTURE:
        raise ValueError('unsupported baseline checkpoint architecture/schema')
    config = BaselineModelConfig(**payload['model_config'])
    if config.chart_style != 'none':
        from maimai_baseline.style_contract import validate_checkpoint_style
        validate_checkpoint_style(payload)
    if is_frozen(config) and payload.get('checkpoint_kind') == 'training' and not _valid_loader_rng(payload.get('loader_rng')):
        raise ValueError('frozen-mu training checkpoint requires its DataLoader RNG state')
    vocabulary = CompactVocabulary.from_dict(payload['vocabulary'])
    artifacts = load_route_artifacts(Path(route_artifact), verify_payloads=verify_artifact_payloads)
    model = BaselineTransformer(vocabulary, config, artifacts=artifacts,
        seed=int(payload.get('random_seed', 20260811)))
    saved_state = payload['model_state']
    expected_state = model.state_dict()
    if saved_state.keys() != expected_state.keys() or any(
            not isinstance(value, torch.Tensor) or value.shape != expected_state[name].shape
            or value.dtype != expected_state[name].dtype or not torch.isfinite(value).all()
            for name, value in saved_state.items()):
        raise ValueError('baseline tensor inventory/shape/dtype/finite mismatch')
    model.load_state_dict(saved_state, strict=True)
    model.require_zero_level_column()
    require_fixed_route_state(model)
    model.route_initialization = dict(payload.get('initialization', model.route_initialization))
    model.specialized_supervision = SpecializedSupervision.from_identity(
        payload.get('specialized_supervision_config', SpecializedSupervision().identity()))
    if is_frozen(config):
        if is_draft(config):
            model.draft_generator_metadata = deepcopy(payload['draft_generator'])
        if payload.get('frozen_encoder_import') is not None:
            model.frozen_encoder_import = deepcopy(payload['frozen_encoder_import'])
        if _valid_loader_rng(payload.get('loader_rng')):
            model._restored_loader_rng = payload['loader_rng'].detach().cpu().clone()
    return model.to(device), dict(payload)


def load_baseline_inference_checkpoint(
    checkpoint: Path,
    *,
    route_artifact: Path,
    profile_root: Path,
    device: torch.device | str,
) -> tuple[BaselineTransformer, dict[str, Any], dict[str, Any], dict[str, Any]]:
    checkpoint = Path(checkpoint).resolve()
    profile_root = Path(profile_root).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"baseline checkpoint is missing: {checkpoint}")
    summary_path = profile_root / "summary.json"
    profile_path = profile_root / "profile.json"
    if not summary_path.is_file() or not profile_path.is_file():
        raise FileNotFoundError(f"compact profile is incomplete: {profile_root}")
    from maimai_baseline.ar_contract import CHECKPOINT_SCHEMA as AR_SCHEMA
    probe = torch.load(checkpoint, map_location="cpu", weights_only=False)
    require_compatible(probe, "inference")
    if probe.get("schema") == AR_SCHEMA:
        from maimai_baseline.ar_checkpoint import load_payload
        model, outer = load_payload(probe, route_artifact=route_artifact,
            device=device, operation='inference')
        parent = outer['parent_metadata']
        payload = dict(parent, runtime_version=require_compatible(outer, 'inference'),
            architecture=outer['architecture'], schema=outer['schema'],
            source_identity=outer.get('source_identity'), ar_prior=outer['identity'],
            frozen_a_source=parent.get('source_identity'), checkpoint_kind=outer['checkpoint_kind'],
            epoch=outer['epoch'], step=outer['step'])
    else:
        model, payload = _load_baseline_payload(probe,
            route_artifact=route_artifact, device=device,
            strict_source=False, verify_artifact_payloads=False)
    profile_summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    from maimai_dataset.window import WINDOW
    if profile_summary.get("window") != WINDOW:
        raise ValueError("baseline inference profile window mismatch")
    return model, payload, dict(model.route_artifact_identity), profile_summary


__all__ = [
    "load_baseline_checkpoint",
    "load_baseline_inference_checkpoint",
    "restore_baseline_rng",
    "restore_baseline_training_state",
    "save_baseline_checkpoint",
    "validate_resume_manifest",
]
