"""Route artifacts and the Route-71 model layer."""

from __future__ import annotations
from maimai_baseline.contract import position_conditioning_identity, history_representation_identity, audio_attention_bias_identity, audio_fusion_identity
from dataclasses import dataclass
from maimai_baseline.contract import (
    CATALOG_IDENTITY,
    REMOVED_FIELDS,
    ROUTE_CLASSES,
    ROUTE_DIMENSION,
    ROUTE_HEAD_SEED_XOR,
)
from maimai_baseline.model.common import BaselineModelConfig, state_dict_sha256
from maimai_baseline.model.decoder import RawPrefixTransformer
from maimai_dataset.encoding import CHART_TICKS_PER_BEAT
from maimai_dataset.vocabulary import CompactVocabulary
from pathlib import Path
from torch import nn
from typing import Any, Mapping
import hashlib
import json
import numpy as np
import torch
from maimai_baseline.route_contract import (GEOMETRY_DIM, GEOMETRY_FILE, GEOMETRY_SCHEMA,
    GEOMETRY_SEED_XOR, ROUTE_TEMPERATURE, geometry_identity, route_head_identity)


def _sha256(path: Path) -> str:
    from maimai_dataset.integrity import file_sha256
    return file_sha256(path)


@dataclass(frozen=True)
class RouteArtifacts:
    root: Path
    route_ids: tuple[str, ...]
    prototypes: torch.Tensor
    judge_distance: torch.Tensor
    trajectory_distance: torch.Tensor
    identity: dict[str, Any]


def load_route_artifacts(root: Path, *, verify_payloads: bool = True) -> RouteArtifacts:
    """Load the required Route-71 artifact from an explicit path."""
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    npz_path = root / GEOMETRY_FILE
    manifest = json.loads(manifest_path.read_text("utf-8"))
    if manifest.get("schema") != GEOMETRY_SCHEMA or manifest.get("route_geometry") != geometry_identity():
        raise RuntimeError("Route geometry schema/contract mismatch")
    if manifest.get("catalog_identity") != CATALOG_IDENTITY:
        raise RuntimeError("Route-71 dual artifact catalog identity mismatch")
    expected_npz_hash = manifest["outputs"][npz_path.name]["sha256"]
    actual_npz_hash = _sha256(npz_path) if verify_payloads else expected_npz_hash
    if actual_npz_hash != expected_npz_hash:
        raise RuntimeError("Route-71 dual artifact NPZ hash mismatch")
    with np.load(npz_path) as archive:
        route_ids = tuple(archive["route71_ids"].astype(str).tolist())
        prototypes = torch.from_numpy(archive["route71_prototypes"].copy())
        judge = torch.from_numpy(archive["judge_route71_distance"].copy())
        trajectory = torch.from_numpy(
            archive["trajectory_route71_distance"].copy()
        )
    if route_ids != tuple(f"B{index:03d}" for index in range(ROUTE_CLASSES)):
        raise RuntimeError("Route-71 dual artifact route order mismatch")
    if prototypes.shape != (ROUTE_CLASSES + 1, GEOMETRY_DIM):
        raise RuntimeError("Route-71 prototype shape mismatch")
    if judge.shape != (ROUTE_CLASSES, ROUTE_CLASSES):
        raise RuntimeError("Route-71 judge distance shape mismatch")
    if trajectory.shape != (ROUTE_CLASSES, ROUTE_CLASSES):
        raise RuntimeError("Route-71 trajectory distance shape mismatch")
    if not all(torch.isfinite(value).all() for value in (prototypes, judge, trajectory)):
        raise RuntimeError("Route-71 artifact contains NaN/Inf")
    prototype_hash = hashlib.sha256(prototypes.numpy().tobytes(order="C")).hexdigest()
    if (prototypes.dtype != torch.float32 or bool(prototypes[0].ne(0).any()) or
            prototype_hash != manifest["route71"]["prototype_sha256"]):
        raise RuntimeError("Route geometry prototype identity/NA mismatch")
    for matrix in (judge, trajectory):
        if (not torch.equal(matrix, matrix.T) or bool(matrix.diag().ne(0).any()) or
                bool(((matrix < 0) | (matrix > 1)).any())):
            raise RuntimeError("Route geometry distance matrix invalid")
    identity = {
        "route_geometry": geometry_identity(),
        "schema": str(manifest["schema"]),
        "catalog_identity": CATALOG_IDENTITY,
        "manifest_sha256": _sha256(manifest_path),
        "npz_sha256": actual_npz_hash,
        "prototype_sha256": str(manifest["route71"]["prototype_sha256"]),
        "judge_route71_sha256": hashlib.sha256(
            judge.numpy().tobytes(order="C")
        ).hexdigest(),
        "trajectory_route71_sha256": hashlib.sha256(
            trajectory.numpy().tobytes(order="C")
        ).hexdigest(),
    }
    return RouteArtifacts(root, route_ids, prototypes, judge, trajectory, identity)


class GeometryProjection(nn.Module):
    """Frozen route descriptors followed by a learned, biasless input projection."""
    def __init__(self, prototypes, width, *, seed):
        super().__init__()
        if prototypes.shape != (72, GEOMETRY_DIM) or bool(prototypes[0].ne(0).any()):
            raise ValueError("invalid geometry projection codebook")
        self.register_buffer("prototypes", prototypes.detach().float().clone())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.projection = nn.Linear(GEOMETRY_DIM, width, bias=False)
            nn.init.normal_(self.projection.weight)

    def forward(self, ids):
        return self.projection(self.prototypes[ids])


class PrototypeRouteHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(ROUTE_DIMENSION, GEOMETRY_DIM)

    def forward(self, rows, prototypes):
        q = self.query(rows)
        with torch.autocast(device_type=q.device.type, enabled=False):
            q, e = q.float(), prototypes.float()
            return (2 * (q @ e.T) - q.square().sum(-1, keepdim=True)
                    - e.square().sum(-1)) / ROUTE_TEMPERATURE


def require_fixed_route_state(model):
    """Checkpoint buffers must still equal the immutable input artifact."""
    values = [(model.route_embedding.prototypes, "prototype_sha256"),
              (model.route_judge_distance, "judge_route71_sha256"),
              (model.route_trajectory_distance, "trajectory_route71_sha256")]
    if hasattr(model, "chart_style_encoder"):
        values.append((model.chart_style_encoder.embeddings["route"].prototypes, "prototype_sha256"))
    for value, key in values:
        digest = hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
        if digest != model.route_artifact_identity[key]:
            raise ValueError(f"fixed route geometry state differs from artifact: {key}")


def validate_checkpoint_route(payload):
    config = payload.get("model_config", {})
    if "route_head" not in config:
        raise ValueError("checkpoint lacks explicit route head")
    head, geometry = route_head_identity(config["route_head"]), geometry_identity()
    for record in (payload, payload.get("baseline_manifest", {}), payload.get("baseline_contract", {})):
        if record.get("route_head") != head or record.get("route_geometry") != geometry:
            raise ValueError("checkpoint route head/geometry identity mismatch")
    if payload.get("checkpoint_kind") == "training":
        record = payload.get("training_config", {})
        if record.get("route_head") != config["route_head"] or record.get("route_geometry") != geometry:
            raise ValueError("checkpoint route training identity mismatch")


# Route-71 model


class Route71Transformer(RawPrefixTransformer):
    def __init__(
        self,
        vocabulary: CompactVocabulary,
        config: BaselineModelConfig,
        *,
        artifacts: RouteArtifacts,
        frozen_wide_state: Mapping[str, torch.Tensor] | None = None,
        seed: int = 20260811,
    ) -> None:
        super().__init__(vocabulary, config)
        if frozen_wide_state is not None:
            self.load_state_dict(frozen_wide_state, strict=True)
        for field in REMOVED_FIELDS:
            if field not in self.embeddings or field not in self.field_heads:
                raise RuntimeError(f"wide baseline is missing {field!r}")
            del self.embeddings[field]
            del self.field_heads[field]
        self.route_embedding = GeometryProjection(artifacts.prototypes, ROUTE_DIMENSION,
                                                  seed=int(seed) ^ GEOMETRY_SEED_XOR)
        head_seed = int(seed) ^ ROUTE_HEAD_SEED_XOR
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(head_seed)
            self.route_head = PrototypeRouteHead()
        self.register_buffer(
            "route_judge_distance", artifacts.judge_distance.to(torch.float32)
        )
        self.register_buffer(
            "route_trajectory_distance",
            artifacts.trajectory_distance.to(torch.float32),
        )
        self.route_artifact_identity = dict(artifacts.identity)
        self.route_initialization = {
            "route_geometry": geometry_identity(),
            "route_head": route_head_identity(config.route_head),
            "position_conditioning": position_conditioning_identity(config.decoder_rope_coordinate),
            "audio_attention_bias": audio_attention_bias_identity(),
            "audio_fusion": audio_fusion_identity(),
            "history_representation": history_representation_identity(),
            "decoder_rope_coordinate": config.decoder_rope_coordinate,
            "prototype_sha256": artifacts.identity["prototype_sha256"],
            "head_seed": head_seed,
            "head_seed_derivation": "seed-xor-0x52543731",
            "head_tied_to_embedding": False,
        }
        if self.parameter_count() != config.parameter_count("route"):
            raise RuntimeError(
                "Route-71 parameter count drifted: "
                f"expected={config.parameter_count('route')}, actual={self.parameter_count()}"
            )
        self.route_initialization["model_state_sha256"] = state_dict_sha256(
            self.state_dict()
        )

    @classmethod
    def from_wide(
        cls,
        wide: RawPrefixTransformer,
        *,
        artifacts: RouteArtifacts,
        seed: int,
    ) -> tuple["Route71Transformer", dict[str, Any]]:
        source_state = {
            name: value.detach().cpu().clone()
            for name, value in wide.state_dict().items()
        }
        model = cls(
            wide.vocabulary,
            wide.config,
            artifacts=artifacts,
            frozen_wide_state=source_state,
            seed=seed,
        )
        route_state = model.state_dict()
        removed_prefixes = tuple(
            f"embeddings.{field}." for field in REMOVED_FIELDS
        ) + tuple(f"field_heads.{field}." for field in REMOVED_FIELDS)
        shared_source = {
            name: value
            for name, value in source_state.items()
            if not name.startswith(removed_prefixes)
        }
        mismatch = [
            name
            for name, value in shared_source.items()
            if name not in route_state
            or not torch.equal(value, route_state[name].detach().cpu())
        ]
        if mismatch:
            raise RuntimeError(f"Route-71 shared initialization mismatch: {mismatch[:8]}")
        shared_hash = state_dict_sha256(shared_source)
        return model, {
            "mode": "paired-canonical-wide-state-plus-route71-v1",
            "seed": int(seed),
            "shared_state_sha256": shared_hash,
            "shared_tensor_count": len(shared_source),
            **model.route_initialization,
        }

    def _row_embedding(
        self,
        input_ids: Mapping[str, torch.Tensor],
        input_times: Mapping[str, torch.Tensor],
        input_time_active: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        if any(field in input_ids for field in REMOVED_FIELDS):
            raise ValueError("Route-71 input leaked shape/end_offset/via_offset")
        if "route" not in input_ids:
            raise ValueError("Route-71 input is missing route_input_id")
        from maimai_dataset.modifiers import normalize_inputs
        input_ids, input_times, input_time_active = normalize_inputs(input_ids, input_times, input_time_active, self.vocabulary)
        total = self.modifier_projection(input_ids)
        event = input_ids["event"]
        active = (event.eq(self.vocabulary.event_id("NOTE")) | event.eq(self.vocabulary.event_id("SLIDE_BRANCH"))).unsqueeze(-1).to(total.dtype)
        for field, embedding in self.embeddings.items():
            field_ids = input_ids[field]
            value = embedding(field_ids)
            present = field_ids.ne(0).unsqueeze(-1).to(value.dtype)
            total = value if total is None else total + value
            active = present if active is None else active + present
        assert total is not None and active is not None
        route_ids = input_ids["route"]
        if bool((route_ids < 0).any()) or bool((route_ids > 71).any()):
            raise ValueError("route_input_id is outside 0..71")
        route_value = self.route_embedding(route_ids)
        route_present = route_ids.ne(0).unsqueeze(-1)
        total = total + route_value
        active = active + route_present.to(active.dtype)
        for field, embedding in self.time_embeddings.items():
            present = input_time_active[field].unsqueeze(-1)
            value = embedding(
                (input_times[field] / CHART_TICKS_PER_BEAT).to(total.dtype)
            ).masked_fill(~present, 0.0)
            total = total + value
            active = active + present.to(active.dtype)
        return total / active.clamp_min(1).sqrt()

    def decode_prefix_rows(self, **kwargs: Any) -> dict[str, Any]:
        return self._route_outputs(super().decode_prefix_rows(**kwargs))

    def _route_outputs(self, result):
        if any(field in result["field_logits"] for field in REMOVED_FIELDS):
            raise RuntimeError("Route-71 output leaked removed factor heads")
        if hasattr(self, 'row_fields'):
            selected = torch.zeros_like(result['decoder_family'], dtype=torch.bool)
            result['route_logits'] = result['decoded'].new_zeros((*selected.shape, 71))
            result['route_valid'] = selected
        elif 'decoder_family' in result:
            selected = result['decoder_family'].eq(1)
            flat = result['decoded'].flatten(0, 1)
            index = selected.flatten().nonzero(as_tuple=False).flatten()
            logits = flat.new_zeros((len(flat), 71))
            if index.numel():
                value = self.route_head(flat.index_select(0, index), self.route_embedding.prototypes[1:])
                logits = logits.to(value.dtype).index_copy(0, index, value)
            result['route_logits'] = logits.reshape(*selected.shape, 71)
            result['route_valid'] = selected
        else:
            result["route_logits"] = self.route_head(result["decoded"], self.route_embedding.prototypes[1:])
        result["route_judge_distance"] = self.route_judge_distance
        result["route_trajectory_distance"] = self.route_trajectory_distance
        return result


def route71_forward(
    model: Route71Transformer, batch: Mapping[str, Any]
) -> dict[str, Any]:
    outputs = model.forward_prefix(
        audio=batch["audio"],
        audio_padding_mask=batch["audio_padding_mask"],
        frame_beats=batch["frame_beats"],
        frame_bpms=batch["frame_bpms"],
        audio_start_tick=batch.get("audio_start_tick"),
        window_to_audio=batch.get("window_to_audio"),
        input_ids=batch["input_ids"],
        decoder_family=batch['decoder_family'],
        input_times=batch["input_times"],
        input_time_active=batch["input_time_active"],
        row_padding_mask=batch["row_padding_mask"],
        row_anchor_tick=batch["row_anchor_tick"],
        difficulty_id=batch["difficulty_id"],
        level=batch["level"],
        start_tick=batch["start_tick"],
        end_tick=batch["end_tick"],
        interpolation_geometry=batch.get("interpolation_geometry"),
        style_context=batch.get("style_context"),
        song_audio=batch.get("song_audio"),
    )
    from maimai_baseline.model.row_fields import supervised_readout
    return supervised_readout(model, outputs, batch)
