"""Explicit inference diagnostics and reproducible per-window style sampling."""
from __future__ import annotations
import json
from pathlib import Path
import torch
import math
from maimai_baseline.model.style import StyleContext
from maimai_baseline.time.style import style_geometry
from maimai_baseline.style_contract import STYLE_ID, STYLE_DIM, STYLE_SLOTS, style_identity, representation_identity


def sampling_options(mode, temperature=None, mixture_temperature=None, *, local_seed=None, reference=None, radar=False, frozen=False, draft=False):
    if draft:
        if mode != 'draft' or any(value is not None for value in (temperature, mixture_temperature, local_seed, reference)):
            raise ValueError('audio draft uses deterministic draft mode without latent sampling options or references')
        return
    if (radar or frozen) and mode == 'ar':
        if reference is not None:
            raise ValueError('AR inference predicts conditions from audio; chart references are unavailable')
        if any(value is not None and (not math.isfinite(value) or value < 0)
               for value in (temperature, mixture_temperature)):
            raise ValueError('AR sampling temperatures must be finite and nonnegative')
        return
    if mode != 'disabled' or temperature is not None or mixture_temperature is not None or local_seed is not None or reference is not None:
        if not radar and not frozen:
            raise ValueError('this no-latent model requires disabled; AR/posterior and latent sampling options are unavailable')
        raise ValueError('this model requires disabled, or AR sampling without posterior/reference inputs')


def read_reference(path, *, stage_a_sha256):
    value = json.loads(Path(path).read_text("utf-8"))
    if (value.get('schema') != 'maimai.full_song_z128_reference.v1' or representation_identity(value.get('chart_style')) != representation_identity(style_identity(STYLE_ID))):
        raise ValueError("style reference belongs to a different GT encoder/checkpoint")
    starts, bank = set(), []
    for window in value["windows"]:
        if window["start_tick"] in starts:
            raise ValueError("duplicate reference window")
        starts.add(window["start_tick"])
        tokens, valid = torch.tensor(window["tokens"]), torch.tensor(window["valid"])
        if tokens.shape != (STYLE_SLOTS,STYLE_DIM) or valid.shape != (STYLE_SLOTS,) or valid.dtype != torch.bool or not torch.isfinite(tokens).all():
            raise ValueError("invalid style reference vectors")
        bank.extend(tokens[valid].tolist())
    if not bank:
        raise ValueError("reference contains no valid style tokens")
    return value


def context_for_window(model, audio_context, conditions, *, mode, local_seed,
                       song_end_tick, reference=None):
    start, end = conditions["start_tick"], conditions["end_tick"]
    beats, valid, _ = style_geometry(start, end, song_end_tick)
    if mode == "disabled":
        return StyleContext(torch.zeros((*valid.shape,STYLE_DIM), device=start.device), beats, valid, enabled=False)
    if mode != "posterior" or reference is None:
        raise ValueError("posterior diagnostic requires a reference")
    windows = {w["start_tick"]: w for w in reference["windows"]}
    if not bool(valid.any()):
        tokens = [[0.]*STYLE_DIM for _ in range(STYLE_SLOTS)]
    elif int(start[0]) not in windows or windows[int(start[0])]["valid"] != valid[0].cpu().tolist():
        raise ValueError("posterior reference does not cover the target window")
    else:
        tokens = windows[int(start[0])]["tokens"]
    value = torch.tensor([tokens], device=start.device, dtype=torch.float32)
    return StyleContext(value.masked_fill(~valid[..., None], 0.), beats, valid)


class DraftSongStyle:
    """Clean whole-song tokens are reused across decoder windows and rollback."""
    def __init__(self, tokens, valid, *, structure=None):
        from maimai_baseline.draft_contract import LEGACY_STYLE_ID, geometry
        self.geometry = geometry(LEGACY_STYLE_ID if structure is None else structure)
        if (valid.ndim != 2 or valid.dtype != torch.bool or tokens.shape != (*valid.shape, 128)
                or tokens.shape[0] != 1 or tokens.shape[1] % self.geometry.target_slots
                or not bool(torch.isfinite(tokens).all())):
            raise ValueError('draft song style requires complete windows of selected token geometry')
        self.tokens, self.valid = tokens, valid

    def window(self, start_tick):
        shape = self.geometry
        start_tick = int(start_tick)
        if start_tick < 0 or start_tick % 1536:
            raise ValueError('draft inference requires a nonnegative aligned decoder window')
        begin = start_tick // (96 * shape.beats_per_token)
        if begin + shape.target_slots > self.tokens.shape[1]:
            raise ValueError('draft inference horizon does not cover decoder window')
        indices = (torch.arange(shape.condition_slots, device=self.tokens.device)
                   + begin + shape.first_slot)
        in_range = (indices >= 0) & (indices < self.tokens.shape[1])
        safe = indices.clamp(0, self.tokens.shape[1] - 1)
        valid = self.valid[:, safe] & in_range[None]
        tokens = self.tokens[:, safe].masked_fill(~valid[..., None], 0.)
        beats = ((torch.arange(shape.condition_slots, device=tokens.device).float()
                  + shape.first_slot + .5) * shape.beats_per_token)[None]
        return StyleContext(tokens, beats, valid)


@torch.no_grad()
def draft_song(model, teacher, variant, *, level, difficulty_id, device):
    """Generate once in FP32 from the audio-derived horizon, never chart length."""
    from maimai_baseline.draft_contract import geometry
    structure = model.draft_generator.structure
    row = teacher.draft_stitched(variant, beats_per_token=geometry(structure).beats_per_token)
    count = int(row['blocks'])
    valid = torch.ones((1, count), dtype=torch.bool, device=device)
    arguments = {key: row[key].to(device)[None] for key in ('audio', 'audio_valid', 'audio_beats')}
    arguments.update(valid=valid, difficulty_id=torch.tensor([difficulty_id], dtype=torch.long, device=device),
        level=torch.tensor([level], dtype=torch.float32, device=device),
        span_beats=torch.tensor([row['span_beats']], dtype=torch.float32, device=device))
    with torch.autocast(device_type=torch.device(device).type, enabled=False):
        tokens = model.draft_generator.generate(arguments)
    return DraftSongStyle(tokens, valid, structure=structure)
