"""Lightweight top-down fusion on the existing continuous absolute audio beat grids."""
from __future__ import annotations

from collections.abc import Mapping
import torch
from torch import nn

from maimai_baseline.contract import AUDIO_FUSION_ID, audio_fusion_identity


def validate_checkpoint_fusion(payload: Mapping) -> None:
    config = payload.get("model_config")
    if not isinstance(config, Mapping) or config.get("audio_fusion") != AUDIO_FUSION_ID:
        raise ValueError("checkpoint lacks supported explicit audio fusion configuration")
    records = [payload, payload.get("baseline_manifest"), payload.get("baseline_contract"),
               payload.get("initialization")]
    if payload.get("checkpoint_kind") == "training":
        saved = payload.get("training_config")
        records.extend((saved, saved.get("model_initialization") if isinstance(saved, Mapping) else None,
                        saved.get("baseline_manifest") if isinstance(saved, Mapping) else None))
    records.extend(record["baseline_contract"] for record in tuple(records)
                   if isinstance(record, Mapping) and "baseline_contract" in record and record is not payload)
    identity = audio_fusion_identity()
    if any(not isinstance(record, Mapping) or record.get("audio_fusion") != identity for record in records):
        raise ValueError("checkpoint audio fusion identity mismatch")


def fixed_double_geometry(source_length: int) -> tuple[torch.Tensor, ...]:
    source = torch.arange(source_length, dtype=torch.float32) + 0.5
    target = (torch.arange(source_length * 2, dtype=torch.float32) + 0.5) / 2.0
    if source_length == 1:
        left = right = torch.zeros((2,), dtype=torch.long)
        weight = torch.zeros((2,), dtype=torch.float32)
    else:
        right = torch.searchsorted(source, target).clamp(1, source_length - 1)
        left = right - 1
        denominator = (source[right] - source[left]).clamp_min(1e-8)
        weight = ((target - source[left]) / denominator).clamp(0.0, 1.0)
    return left, right, weight


def masked_upsample(source, padding, left, right, weight):
    """Renormalize linear interpolation over valid endpoints, entirely in FP32."""
    with torch.autocast(device_type=source.device.type, enabled=False):
        clean = source.float().masked_fill(padding.unsqueeze(-1), 0.)
        def selected(value,index):
            if index.ndim==1:return value.index_select(1,index)
            if value.ndim==3:index=index[...,None].expand(-1,-1,value.shape[-1])
            return value.gather(1,index.long())
        left_weight = (~selected(padding,left)).float() * (1. - weight.float())
        right_weight = (~selected(padding,right)).float() * weight.float()
        mass = left_weight + right_weight
        value = (selected(clean,left) * left_weight.unsqueeze(-1)
                 + selected(clean,right) * right_weight.unsqueeze(-1))
        value = value / mass.clamp_min(1e-12).unsqueeze(-1)
        valid = mass > 0
        value = value.masked_fill(~valid.unsqueeze(-1), 0.)
    return value.to(source.dtype), valid


class TopDownAudioFusion(nn.Module):
    """M2 -> P1 -> P0, with independent projections and unchanged masks."""
    def __init__(self) -> None:
        super().__init__()
        # Keep construction isolated; the new experiment has its own initialization identity.
        with torch.random.fork_rng(devices=[]):
            self.song_projection = nn.Linear(256,256,bias=True)
            self.projections = nn.ModuleList(nn.Linear(256, 256, bias=True) for _ in range(2))
    def forward(self, memories, padding_masks, *, song_audio, target_beats, geometry=None):
        if len(memories) != 3 or len(padding_masks) != 3:
            raise ValueError("top-down fusion requires three fine-to-coarse memories")
        result = list(memories)
        if song_audio is None:raise ValueError("P2 fusion requires explicit whole-song audio")
        up,valid = song_audio.aligned(target_beats)
        branch = self.song_projection(up).to(memories[-1].dtype).masked_fill(~valid[...,None],0.)
        result[-1] = (memories[-1]+branch).masked_fill(padding_masks[-1][...,None],0.)
        for index in (1, 0):
            source, target = result[index + 1], memories[index]
            if (source.shape[-1] != 256 or target.shape[-1] != 256
                    or source.shape[1] * 2 != target.shape[1]
                    or source.shape[:2] != padding_masks[index + 1].shape
                    or target.shape[:2] != padding_masks[index].shape
                    ):
                raise ValueError("top-down fusion requires aligned absolute beat grids")
            indices = (tuple(geometry[f'fusion_{index}_{name}'] for name in ('left','right','weight'))
                       if geometry is not None and f'fusion_{index}_left' in geometry else
                       tuple(x.to(source.device) for x in fixed_double_geometry(source.shape[1])))
            up, valid = masked_upsample(source, padding_masks[index + 1], *indices)
            # Mask after the projection too: its learned bias is not valid audio.
            branch = self.projections[index](up).to(target.dtype)
            branch = branch.masked_fill(~valid.unsqueeze(-1), 0.)
            result[index] = (target + branch).masked_fill(padding_masks[index].unsqueeze(-1), 0.)
        return tuple(result)
