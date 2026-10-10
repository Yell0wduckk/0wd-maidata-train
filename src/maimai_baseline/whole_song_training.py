"""Explicit plan preparation; training consumes its final supplied boundaries."""
from maimai_baseline.runtime_versions import stamp
from collections import Counter
from itertools import groupby
from pathlib import Path
import json

import numpy as np

from maimai_baseline.data.curriculum import EpochPlanStore
from maimai_baseline.data.repack import reference_hash
from maimai_baseline.experiment import validate_plan
from maimai_baseline.source_identity import baseline_source_identity
from maimai_dataset.integrity import file_sha256
from maimai_dataset.manifest import canonical_json_sha256 as digest


POLICY = dict(schema="maimai.whole_song_training.v1", token_budget=16384,
    budget_unit="source-target-rows", packing="song-contiguous-greedy-soft-limit",
    oversize="one-whole-song-group", sampling="one-posterior-per-chart-view-per-epoch",
    supervision="unit-weight-all-valid-latent-blocks", occurrences=1)

def import_from_payload(path, payload):
    from maimai_baseline.query6_parent import import_from_payload as import_dynamic
    return import_dynamic(path, payload)


def repack(batches, records):
    """Preserve the flattened order; only close batches between whole songs."""
    refs = tuple(r for batch in batches for r in batch)
    if (Counter(r.canonical_index for r in refs) != Counter(range(len(records))) or
            any(r.occurrence != 0 or r.view_id not in (0, 1) for r in refs)):
        raise ValueError("whole-song plan requires every source window exactly once")
    views = {}
    for r in refs:
        key = records[r.canonical_index]['record_id']
        if views.setdefault(key, r.view_id) != r.view_id:
            raise ValueError("chart has multiple selected views")
    result, batch, used, seen = [], [], 0, set()
    for song, members in groupby(refs, key=lambda r: str(records[r.canonical_index]['song_id'])):
        if song in seen:
            raise ValueError("source plan is not song contiguous")
        seen.add(song)
        members = tuple(members)
        cost = sum(max(1, int(records[r.canonical_index]['target_rows'])) for r in members)
        if batch and used + cost > POLICY['token_budget']:
            result.append(tuple(batch)); batch, used = [], 0
        batch.extend(members); used += cost
    if batch:
        result.append(tuple(batch))
    return tuple(result)


def build_plan(original_root, output, records, frozen_import, *, config=None, bpm=None):
    original = EpochPlanStore(original_root, verify_payloads=False)
    producer = validate_plan(original)
    if digest([r['segment_id'] for r in records]) != original.manifest['segment_order_sha256']:
        raise ValueError("whole-song source record order mismatch")
    config = producer['recipe'] if config is None else config
    from maimai_baseline.data.master_plan import epoch_batches
    def source_batches(e):
        if e < original.available_epochs:
            return original.load_epoch(e)
        if bpm is None:
            raise ValueError('extended plan requires BPM metadata')
        return epoch_batches(records, bpm, e + 1, config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/original.mode).mkdir()
    audits = []
    for e in range(config['plan_epochs']):
        batches = repack(source_batches(e), records)
        refs = tuple(r for b in batches for r in b)
        offsets = np.cumsum([0, *map(len, batches)], dtype=np.int64)
        name = f'{original.mode}/epoch-{e+1:03d}.npz'
        np.savez(output/name, canonical_indices=np.array([r.canonical_index for r in refs], dtype=np.int64),
            view_ids=np.array([r.view_id for r in refs], dtype=np.int64),
            occurrences=np.zeros(len(refs), dtype=np.int64), batch_offsets=offsets)
        counts = [sum(max(1, int(records[r.canonical_index]['target_rows'])) for r in b) for b in batches]
        template = original.audit(e) if e < original.available_epochs else dict(
            schema=original.audit(0)['schema'], mode=original.mode, epoch=e+1, segment_references=len(refs))
        audit = dict(template, file=name, file_sha256=file_sha256(output/name),
            batch_count=len(batches), batch_target_rows=counts, target_rows=sum(counts),
            plan_sha256=digest(dict(references=reference_hash(refs), offsets=offsets.tolist())))
        audits.append(audit)
    experiment = dict(producer, recipe=config, optimizer_steps_by_epoch=[a['batch_count'] for a in audits],
        whole_song=dict(policy=POLICY, input_plan_sha256=frozen_import["input_plan_sha256"],
                        producer_experiment_identity=producer['identity']))
    experiment['identity'] = digest({k:v for k,v in experiment.items() if k != 'identity'})
    (output/'audit.jsonl').write_text(''.join(json.dumps(a)+'\n' for a in audits), 'utf8')
    manifest = dict(original.manifest, **stamp(), epochs=config['plan_epochs'], token_budget=POLICY['token_budget'], experiment=experiment,
        whole_song=dict(policy=POLICY, frozen_a_import=frozen_import, consumer_source=baseline_source_identity()),
        modes={original.mode:dict(epoch_plan_aggregate_sha256=digest([a['plan_sha256'] for a in audits]))},
        audit=dict(path='audit.jsonl', rows=len(audits), sha256=file_sha256(output/'audit.jsonl')))
    manifest['manifest_identity_sha256'] = digest({k:v for k,v in manifest.items() if k != 'manifest_identity_sha256'})
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', 'utf8')
    plans = EpochPlanStore(output)
    for e in range(plans.available_epochs):
        if plans.load_epoch(e) != repack(source_batches(e), records):
            raise ValueError("new whole-song plan readback mismatch")
    return training_binding(plans)


def training_binding(plans):
    return dict(plans.manifest.get('whole_song', {}),
        experiment=plans.manifest.get('experiment', {}),
        plan_sha256=plans.manifest.get('manifest_identity_sha256'))


def consumer_plans(original, path, config=None):
    """Open the selected final plan; no parent plan or recipe comparison."""
    plans = original if path is None else EpochPlanStore(path, verify_payloads=False)
    if plans is None:
        raise ValueError('B requires a final epoch plan')
    return plans, training_binding(plans)


def repack_frozen(batches, records, budget=16384):
    """Keep every supplied reference in order; close only between complete songs."""
    refs = tuple(r for batch in batches for r in batch)
    if not refs:
        raise ValueError('cannot repack an empty fixed epoch')
    expected = {}
    for i, record in enumerate(records):
        expected.setdefault(str(record['record_id']), set()).add(i)
    selected = {}
    for ref in refs:
        if not 0 <= ref.canonical_index < len(records):
            raise ValueError('B source plan canonical index out of bounds')
        record = str(records[ref.canonical_index]['record_id'])
        selected.setdefault((record, ref.view_id, ref.occurrence), []).append(ref.canonical_index)
    for (record, _, _), indices in selected.items():
        if len(indices) != len(set(indices)) or set(indices) != expected[record]:
            raise ValueError('B source plan contains an incomplete/repeated chart occurrence')
    result, pending, cost, seen = [], [], 0, set()
    for song, members in groupby(refs, key=lambda r: str(records[r.canonical_index]['song_id'])):
        if song in seen:
            raise ValueError('B cannot preserve order when a song is noncontiguous in the source plan')
        seen.add(song)
        members = tuple(members)
        size = sum(max(1,int(records[r.canonical_index]['target_rows'])) for r in members)
        if pending and cost+size > budget:
            result.append(tuple(pending)); pending, cost = [], 0
        pending.extend(members); cost += size
    if pending:
        result.append(tuple(pending))
    return tuple(result)


def _profile_path(args):
    """Use the maintained manifest mapping semantics without unrelated requirements."""
    root = getattr(args,'dataset_root',None)
    root = Path(root).expanduser().resolve() if root is not None else None
    path = getattr(args,'dataset_manifest',None)
    if path is not None:
        path = Path(path).expanduser().resolve()
        root = root or path.parent
    elif root is not None:
        path = next((root/n for n in ('dataset-v10.json','dataset.json') if (root/n).is_file()),None)
    mapped = ({k.replace('-','_'):v for k,v in json.loads(path.read_text('utf-8-sig')).get('inputs',{}).items()}
              if path is not None else {})
    explicit = getattr(args,'dataset',None)
    value = explicit if explicit is not None else mapped.get('dataset','profile' if root is not None else None)
    if value is None:
        raise ValueError('B plan needs --dataset or a dataset root/manifest')
    result = Path(value).expanduser()
    if explicit is None and root is not None and not result.is_absolute():
        result = root/result
    return result.resolve()


def build_frozen_plan(args):
    """Read only profile/plan metadata and existing references; never extend epochs."""
    from maimai_baseline.data.frozen_latent import selected_segments
    profile = _profile_path(args)
    original = EpochPlanStore(args.epoch_plan_cache, verify_payloads=False)
    records = selected_segments(Path(profile), original)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/original.mode).mkdir()
    audits = []
    for e in range(original.available_epochs):
        batches = repack_frozen(original.load_epoch(e), records)
        refs = tuple(r for b in batches for r in b)
        offsets = np.cumsum([0,*map(len,batches)], dtype=np.int64)
        name = f'{original.mode}/epoch-{e+1:03d}.npz'
        path = output/name
        np.savez(path, canonical_indices=np.array([r.canonical_index for r in refs],dtype=np.int64),
            view_ids=np.array([r.view_id for r in refs],dtype=np.int64),
            occurrences=np.array([r.occurrence for r in refs],dtype=np.int64), batch_offsets=offsets)
        rows = [sum(max(1,int(records[r.canonical_index]['target_rows'])) for r in b) for b in batches]
        audits.append(dict(original.audit(e), file=name, file_sha256=file_sha256(path),
            batch_count=len(batches), batch_target_rows=rows, target_rows=sum(rows),
            plan_sha256=digest(dict(references=reference_hash(refs),offsets=offsets.tolist()))))
        with np.load(path,allow_pickle=False) as z:
            actual = list(zip(z['canonical_indices'].tolist(),z['view_ids'].tolist(),z['occurrences'].tolist()))
            if actual != [(r.canonical_index,r.view_id,r.occurrence) for r in refs]:
                raise ValueError('new B plan reference readback mismatch')
    policy = dict(POLICY, sampling='original-fixed-references', supervision='deterministic-mu-valid-blocks')
    policy.pop('occurrences',None)
    experiment = dict(original.manifest.get('experiment', {}),
                      optimizer_steps_by_epoch=[r['batch_count'] for r in audits])
    (output/'audit.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in audits),'utf8')
    manifest = dict(original.manifest, **stamp(), token_budget=16384, experiment=experiment,
        whole_song=dict(policy=policy, input_plan=str(original.root)),
        modes={original.mode:dict(epoch_plan_aggregate_sha256=digest([r['plan_sha256'] for r in audits]))},
        audit=dict(path='audit.jsonl',rows=len(audits),sha256=file_sha256(output/'audit.jsonl')))
    write = output/'manifest.json'
    write.write_text(json.dumps(manifest,indent=2)+'\n','utf8')
    return dict(operation='build-b-plan',output=str(output),epochs=len(audits),budget=16384,
                optimizer_steps_by_epoch=[r['batch_count'] for r in audits])
