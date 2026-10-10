"""Import only the frozen deterministic encoder from the completed 60th epoch."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import torch

from maimai_baseline.model.style import ChartStyleEncoder
from maimai_baseline.runtime_versions import require_compatible
from maimai_baseline.training.chart_encoder_checkpoint import (CHECKPOINT_SCHEMA, LEGACY_CHECKPOINT_SCHEMA,
    ENCODER_SCHEMA, LEGACY_ENCODER_SCHEMA)
from maimai_baseline.encoder_audio_contract import CONTRACT, REPRESENTATION
from maimai_dataset.vocabulary import CompactVocabulary


def load_encoder60(path, *, vocabulary=None, prototypes=None, device="cpu"):
    """Do not instantiate the auxiliary reconstruction head or export loader."""
    path = Path(path).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema") not in (
            CHECKPOINT_SCHEMA, LEGACY_CHECKPOINT_SCHEMA, ENCODER_SCHEMA, LEGACY_ENCODER_SCHEMA):
        raise ValueError("encoder60 import requires an independent pretraining checkpoint")
    require_compatible(payload, "training")
    exported = payload['schema'] in (ENCODER_SCHEMA, LEGACY_ENCODER_SCHEMA)
    native = payload['schema'] in (CHECKPOINT_SCHEMA, ENCODER_SCHEMA)
    if not exported and (type(payload.get("epoch")) is not int or payload["epoch"] != 59):
        raise ValueError("encoder60 import requires epoch-060.ckpt with epoch59")
    metadata = payload.get("metadata" if exported else "model_metadata")
    state = payload.get("encoder_state" if exported else "model_state")
    if not isinstance(metadata, Mapping) or not isinstance(state, Mapping):
        raise ValueError("encoder60 checkpoint is missing encoder metadata/state")
    if exported and metadata.get('completed_epochs') != 60:
        raise ValueError('encoder60 export requires completed_epochs=60')
    if native and (metadata.get('representation') != REPRESENTATION
                   or metadata.get('encoder_audio_contract') != CONTRACT):
        raise ValueError('native encoder60 normalization/objective contract mismatch')
    stored_vocabulary = CompactVocabulary.from_dict(metadata["vocabulary"])
    if vocabulary is not None and vocabulary.to_dict() != stored_vocabulary.to_dict():
        raise ValueError("encoder60 vocabulary differs from the current chart vocabulary")
    vocabulary = stored_vocabulary if vocabulary is None else vocabulary
    stored_prototypes = metadata.get("prototypes")
    from maimai_baseline.route_contract import GEOMETRY_DIM
    if (not isinstance(stored_prototypes, torch.Tensor) or stored_prototypes.shape != (72, GEOMETRY_DIM)
            or stored_prototypes.dtype != torch.float32 or not torch.isfinite(stored_prototypes).all()
            or bool(stored_prototypes[0].ne(0).any())):
        raise ValueError("encoder60 checkpoint has invalid Route71 prototypes")
    if prototypes is not None and not torch.equal(prototypes.detach().cpu(), stored_prototypes):
        raise ValueError("encoder60 Route71 prototypes differ from the current artifact")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(metadata["seed"]))
        encoder = ChartStyleEncoder(vocabulary, stored_prototypes, seed=int(metadata["seed"]))
    encoder_state = dict(state) if exported else {name.removeprefix("encoder."): value for name, value in state.items()
                                                 if name.startswith("encoder.")}
    expected = encoder.state_dict()
    if (encoder_state.keys() != expected.keys() or any(
            not isinstance(value, torch.Tensor) or value.shape != expected[name].shape
            or value.dtype != expected[name].dtype or not bool(torch.isfinite(value).all())
            for name, value in encoder_state.items())):
        raise ValueError("encoder60 tensor inventory/shape/dtype/finite mismatch")
    if not torch.equal(encoder_state["embeddings.route.prototypes"], stored_prototypes):
        raise ValueError("encoder60 embedded prototypes differ from its data metadata")
    encoder.load_state_dict(encoder_state, strict=True)
    if native:
        encoder.mu_representation = REPRESENTATION
    encoder.requires_grad_(False).eval().to(device)
    provenance = dict(path=str(path), schema=payload["schema"], epoch=59, completed_epochs=60,
        runtime_version=payload["runtime_version"], representation=REPRESENTATION if native else "deterministic-mu",
        beats_per_token=4, dimension=128)
    return encoder, provenance


def import_encoder60(model, path):
    """Replace only the current A model's encoder tensors, then freeze them all."""
    target = model.chart_style_encoder
    encoder, metadata = load_encoder60(path, vocabulary=model.vocabulary,
        prototypes=target.embeddings["route"].prototypes, device=target.empty.device)
    from maimai_baseline.style_contract import NATIVE_STYLE_ID
    if (getattr(getattr(model, 'config', None), 'chart_style', None) == NATIVE_STYLE_ID) != (metadata['representation'] == REPRESENTATION):
        raise ValueError('frozen encoder representation differs from the A style contract')
    target.load_state_dict(encoder.state_dict(), strict=True)
    if metadata['representation'] == REPRESENTATION:
        target.mu_representation = REPRESENTATION
    target.requires_grad_(False).eval()
    model.frozen_encoder_import = dict(metadata)
    return metadata
