"""Song-first A plans derived only from accepted reference/length metadata."""

from maimai_baseline.runtime_versions import stamp
from copy import deepcopy
from pathlib import Path
import json
import numpy as np

from maimai_baseline.data.curriculum import EpochPlanStore
from maimai_baseline.experiment import load_config, validate_plan
from maimai_baseline.experiment import CONFIG as A_CONFIG
FULL_A_CONFIG = A_CONFIG
from maimai_baseline.source_identity import baseline_source_identity
from maimai_baseline.data.master_plan import metadata
from maimai_baseline.data.song_sampling import song_first, statistics, SONG_FIRST_POLICY
from maimai_baseline.data.repack import flatten, reference_hash
from maimai_dataset.integrity import file_sha256
from maimai_dataset.manifest import canonical_json_sha256 as digest


def validate(plans, config=None):
    """Compatibility name: consume final plans without experimental admission."""
    return validate_plan(plans, config)


def build_full(dataset, output):
    """Create the new full plan from accepted metadata, never rebuild payloads."""
    from maimai_baseline.data.master_plan import build
    build(dataset, output, FULL_A_CONFIG)
    # Repack the freshly generated metadata plan through the same pilot path.
    return _repack(EpochPlanStore(output, verify_payloads=False), Path(output), dataset,
                   config=load_config(FULL_A_CONFIG), in_place=True)


def _repack(original, output, dataset, *, config, in_place=False):
    rows, subsets, bpm = metadata(dataset, config)
    records = rows['train']
    packing = dict(lengths=[int(r['target_rows']) for r in records],
        song_ids=[str(r['song_id']) for r in records],
        audio_keys=[(bpm['segments'][r['segment_id']], int(r['target_start_tick'])) for r in records])
    template_hash = file_sha256(original.root/'manifest.json')
    if not in_place:
        output.mkdir(parents=True, exist_ok=False)
        (output/original.mode).mkdir()
    audits = []
    reports = []
    for e in range(original.available_epochs):
        old = original.load_epoch(e)
        batches = song_first(old, **packing)
        refs = flatten(batches)
        offsets = np.cumsum([0, *map(len, batches)], dtype=np.int64)
        source_audit = original.audit(e)
        name = source_audit['file']
        # Batch-dependent diagnostics from the old plan cannot describe these
        # offsets. Keep only the neutral audit fields, and report comparisons
        # separately so B never inherits A packing statistics as its own.
        audit = dict(schema=source_audit['schema'], mode=original.mode,
            epoch=e+1, file=name, segment_references=len(refs))
        target = (output/name).resolve()
        target.relative_to(output.resolve())
        np.savez(target, canonical_indices=np.array([r.canonical_index for r in refs], dtype=np.int64),
            view_ids=np.array([r.view_id for r in refs], dtype=np.int64),
            occurrences=np.array([r.occurrence for r in refs], dtype=np.int64), batch_offsets=offsets)
        stats = statistics(batches, record_ids=[r['record_id'] for r in records], **packing)
        before = statistics(old, record_ids=[r['record_id'] for r in records], **packing)
        audit.update(file_sha256=file_sha256(target), batch_count=len(batches),
            batch_target_rows=stats['batch_target_rows'], target_rows=stats['target_rows'],
            plan_sha256=digest(dict(references=reference_hash(refs), offsets=offsets.tolist())))
        audits.append(audit)
        reports.append(dict(epoch=e+1, old_batch_count=len(old), new_batch_count=len(batches),
            old_fill_ratio=before['target_rows']/(4096*len(old)) if old else 0,
            new_fill_ratio=stats['target_rows']/(4096*len(batches)) if batches else 0,
            old_song_update_counts=before['song_update_counts'],
            song_update_counts=stats['song_update_counts'],
            oversized_audio_groups=[dict(batch_index=i, target_rows=stats['batch_target_rows'][i],
                audio_key=packing['audio_keys'][batches[i][0].canonical_index],
                canonical_indices=[r.canonical_index for r in batches[i]]) for i in stats['oversized_batches']]))
    manifest = dict(deepcopy(original.manifest), **stamp())
    manifest['modes'] = {original.mode: dict(epoch_plan_aggregate_sha256=digest([a['plan_sha256'] for a in audits]))}
    semantics = manifest['modes'][original.mode]['epoch_plan_aggregate_sha256']
    exp = manifest['experiment']
    exp.update(optimizer_steps_by_epoch=[a['batch_count'] for a in audits],
        a_batch_plan=dict(packing=SONG_FIRST_POLICY, references_and_offsets_sha256=semantics))
    exp['identity'] = digest({k:v for k,v in exp.items() if k != 'identity'})
    (output/'audit.jsonl').write_text(''.join(json.dumps(a)+'\n' for a in audits), 'utf8')
    manifest['audit'] = dict(path='audit.jsonl', rows=len(audits), sha256=file_sha256(output/'audit.jsonl'))
    manifest['query6_plan'] = dict(**stamp(), schema='maimai.dynamic_query6_a_plan.v1',
        template_plan_sha256=template_hash, source=baseline_source_identity(), packing=SONG_FIRST_POLICY)
    manifest['manifest_identity_sha256'] = digest({k:v for k,v in manifest.items() if k != 'manifest_identity_sha256'})
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', 'utf8')
    derived = EpochPlanStore(output)
    validate(derived, config)
    for e, audit in enumerate(audits):
        batches = derived.load_epoch(e)
        semantic_hash = digest(dict(references=reference_hash(flatten(batches)),
            offsets=np.cumsum([0, *map(len,batches)], dtype=np.int64).tolist()))
        if semantic_hash != audit['plan_sha256']:
            raise ValueError('song-first plan readback mismatch')
    from maimai_baseline.experiment import step_count, horizon
    result = dict(plan=str(output), sha256=file_sha256(output/'manifest.json'),
        plan_manifest_sha256=file_sha256(output/'manifest.json'), subsets=subsets,
        experiment_identity=exp['identity'], epochs=len(audits),
        packing=SONG_FIRST_POLICY, epochs_packing=reports,
        stages={'A':dict(steps=step_count(exp,'A'),scheduler_steps=horizon(exp,'A'),**config['stages']['A'])})
    (output/'report.json').write_text(json.dumps(result, indent=2)+'\n', 'utf8')
    return result


def derive(original_root, output, dataset_root):
    original = EpochPlanStore(original_root, verify_payloads=False)
    config = original.manifest.get('experiment', {}).get('recipe') or load_config(A_CONFIG)
    validate_plan(original, config)
    return _repack(original, Path(output), dataset_root, config=config)
