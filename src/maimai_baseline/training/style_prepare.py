"""Offline preparation and export; never starts either training stage."""
from __future__ import annotations
from pathlib import Path
import json
import torch
from maimai_baseline.checkpoint import load_baseline_checkpoint, validate_resume_manifest
from maimai_baseline.data.style import prepare_style_inputs
from maimai_baseline.data.style_cache import StyleCacheWriter, StyleLatentCache, write_json
from maimai_baseline.data.curriculum import VirtualSegmentRef
from maimai_baseline.data.batch import collate_baseline_segments
from maimai_baseline.data.style_cache import representation_cache_conditions
from maimai_baseline.training.inputs import _prepare_inputs, _dataset
from maimai_baseline.training.steps import _move
from maimai_baseline.style_contract import STYLE_ID, style_identity
from maimai_dataset.integrity import file_sha256
from maimai_dataset.manifest import canonical_json_sha256


def prepare_cache(args):
    # The CLI and the explicit worker entry share the same full-song producer.
    import sys
    scripts=Path(__file__).resolve().parents[3]/'scripts'
    if str(scripts) not in sys.path:sys.path.insert(0,str(scripts))
    from global_local_cache import prepare
    return prepare(args,workers=12)


def export_reference(args):
    cache = StyleLatentCache(args.cache)
    parent = Path(args.stage_a_checkpoint)
    # Model/source provenance does not gate runtime compatibility.
    from maimai_baseline.checkpoint import load_frozen_stage_a
    model, _ = load_frozen_stage_a(parent, route_artifact=Path(args.route_artifact))
    model.eval().requires_grad_(False)
    selected = [(i, r) for i, r in enumerate(cache.records) if r["song_id"] == args.song_id and
                r["source_difficulty_id"] == args.difficulty and r["view_id"] == 0]
    selected.sort(key=lambda pair: pair[1]["start_tick"])
    if not selected:
        raise ValueError("reference song/difficulty is absent from cache")
    if len({r["record_id"] for _, r in selected}) != 1:
        raise ValueError("reference selection must identify one source chart")
    output = Path(args.output)
    if output.exists():
        raise ValueError("refusing to overwrite style reference")
    windows = []
    with torch.no_grad():
        for i,r in selected:
            h=torch.from_numpy(__import__('numpy').array(cache.arrays['features'][i:i+1],copy=True))
            valid=torch.from_numpy(__import__('numpy').array(cache.arrays['valid'][i:i+1],copy=True))
            local=model.chart_style_encoder.posterior(h,valid)
            windows.append(dict(start_tick=r['start_tick'],tokens=local.mu[0].tolist(),valid=local.valid[0].tolist()))
    from maimai_baseline.runtime_versions import stamp
    value=dict(**stamp(),schema='maimai.full_song_z128_reference.v1',chart_style=style_identity(STYLE_ID),
        stage_a_sha256=file_sha256(parent),cache_identity=cache.manifest['manifest_identity_sha256'],
        song_id=args.song_id,difficulty=args.difficulty,record_id=selected[0][1]['record_id'],windows=windows)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, value)
    return {"operation": "style-export", "output": str(output), "windows": len(windows)}
