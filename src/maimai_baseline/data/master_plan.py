"""Metadata-only subset planning; accepted payloads remain in place."""
from collections import defaultdict, Counter
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
from maimai_baseline.experiment import selected, load_config, validate_plan, step_count, horizon, validation_enabled
from maimai_baseline.contract import EPOCH_PLAN_MODE, VIEW_NAMES, PLAN_AUDIT_SCHEMA, spatial_sampling_identity
from maimai_baseline.data.curriculum import ChartSlot, VirtualSegmentRef, assign_spatial_views, EpochPlanStore
from maimai_baseline.data.song_sampling import arrange
from maimai_baseline.data.repack import reference_hash
from maimai_baseline.training.recipe import RECIPES
from maimai_dataset.manifest import canonical_json_sha256 as digest
from maimai_dataset.integrity import file_sha256


def metadata(dataset, config):
    root = Path(dataset)
    all_rows = [json.loads(x) for x in (root/'profile/segments.jsonl').read_text('utf8').splitlines() if x]
    selection = json.loads((root/'chart-cache/manifest.json').read_text('utf8'))['selection']
    if bool(selection.get('validation_song_ids')) != validation_enabled(config):
        raise ValueError('dataset selection and experiment validation policy differ')
    allowed = set(config.get('train_song_ids', selection['train_song_ids']))
    if not allowed.issubset(set(map(str, selection['train_song_ids']))):
        raise ValueError('pilot subset contains songs outside accepted full selection')
    rows = {role: [r for r in all_rows if str(r['song_id']) in (allowed if role == 'train' else set(map(str,selection['validation_song_ids']))) and selected(r, config)]
            for role in ('train','validation')}
    if config.get('train_song_ids') is not None:
        covered = {str(r['song_id']) for r in rows['train']}
        if covered != allowed:
            raise ValueError(f'pilot subset lacks difficulty 5/6 records: {sorted(allowed-covered)}')
    subsets = {role: dict(segments=len(rs), charts=len({r['record_id'] for r in rs}),
        segment_order_sha256=digest([r['segment_id'] for r in rs]),
        records=sorted({r['record_id'] for r in rs}),
        difficulty_charts=dict(Counter(str(d) for rid,d in {(r['record_id'],r['source_difficulty_id']) for r in rs})))
        for role,rs in rows.items()}
    bpm = json.loads((root/'audio/bpm-audio-manifest.json').read_text('utf8'))
    return rows, subsets, bpm


def epoch_batches(records, bpm, epoch, config):
    charts = defaultdict(list)
    for i,r in enumerate(records): charts[str(r['song_id']),int(r['source_difficulty_id'])].append(i)
    slots = [ChartSlot(s,d) for s,d in sorted(charts)]
    views = assign_spatial_views(slots,seed=config['seed'],epoch=epoch)
    keys = [(bpm['segments'][r['segment_id']],int(r['target_start_tick'])) for r in records]
    groups = defaultdict(list)
    for slot in slots:
        for i in charts[slot.chart_key]: groups[keys[i]].append(VirtualSegmentRef(i,VIEW_NAMES.index(views[slot]),0))
    refs = tuple(r for _,members in sorted(groups.items()) for r in members)
    return arrange(refs,epoch=epoch,lengths=[int(r['target_rows']) for r in records],
                   audio_keys=keys,song_ids=[str(r['song_id']) for r in records])


def build(dataset, output, experiment_config):
    config = load_config(experiment_config)
    root, output = Path(dataset), Path(output)
    rows, subsets, bpm = metadata(root, config)
    records = rows['train']
    if not records or (validation_enabled(config) and not rows['validation']): raise ValueError('empty experiment subset')
    output.mkdir(parents=True,exist_ok=False)
    (output/EPOCH_PLAN_MODE).mkdir()
    audits=[]
    for e in range(1,config['plan_epochs']+1):
        batches = epoch_batches(records,bpm,e,config)
        refs = tuple(r for b in batches for r in b)
        offsets = np.cumsum([0,*map(len,batches)],dtype=np.int64)
        name=f'{EPOCH_PLAN_MODE}/epoch-{e:03d}.npz'
        np.savez(output/name,canonical_indices=np.array([r.canonical_index for r in refs],dtype=np.int64),
            view_ids=np.array([r.view_id for r in refs],dtype=np.int64),
            occurrences=np.array([r.occurrence for r in refs],dtype=np.int64),batch_offsets=offsets)
        lengths=[sum(max(1,int(records[r.canonical_index]['target_rows'])) for r in b) for b in batches]
        audits.append(dict(schema=PLAN_AUDIT_SCHEMA,mode=EPOCH_PLAN_MODE,epoch=e,file=name,
            file_sha256=file_sha256(output/name),batch_count=len(batches),batch_target_rows=lengths,
            target_rows=sum(lengths),segment_references=len(refs),
            plan_sha256=digest(dict(references=reference_hash(refs),offsets=offsets.tolist()))))
    bindings = {k:file_sha256(root/p) for k,p in dict(profile_segments='profile/segments.jsonl',
        chart_manifest='chart-cache/manifest.json', spatial_manifest='spatial-views/manifest.json',
        bpm_manifest='audio/bpm-audio-manifest.json').items()}
    exp = dict(recipe=config,subsets=subsets,inputs=bindings,optimizer_steps_by_epoch=[a['batch_count'] for a in audits])
    exp['identity']=digest(exp)
    (output/'audit.jsonl').write_text(''.join(json.dumps(a)+'\n' for a in audits),'utf8')
    from maimai_dataset.window import WINDOW
    m = dict(schema='maimai.repacked_epoch_plan.v1',epochs=config['plan_epochs'],seed=config['seed'],token_budget=4096,
        recipe=replace(RECIPES['4096'],schedule='optimizer-step').identity(),window=WINDOW,experiment=exp,repack_inputs=bindings,
        spatial_sampling=spatial_sampling_identity(),segment_count=len(records),
        segment_order_sha256=subsets['train']['segment_order_sha256'],profile_segments_sha256=bindings['profile_segments'],
        modes={EPOCH_PLAN_MODE:dict(epoch_plan_aggregate_sha256=digest([a['plan_sha256'] for a in audits]))},
        audit=dict(path='audit.jsonl',rows=config['plan_epochs'],sha256=file_sha256(output/'audit.jsonl')))
    m['manifest_identity_sha256']=digest(m)
    (output/'manifest.json').write_text(json.dumps(m,indent=2),'utf8')
    plans=EpochPlanStore(output)
    validate_plan(plans)
    for e in range(config['plan_epochs']):
        if plans.load_epoch(e)!=epoch_batches(records,bpm,e+1,config):raise ValueError('plan readback mismatch')
    result=dict(subsets=subsets,plan_manifest_sha256=file_sha256(output/'manifest.json'),experiment_identity=exp['identity'],
        stages={s:dict(steps=step_count(exp,s),scheduler_steps=horizon(exp,s),**config['stages'][s]) for s in config['stages']})
    (output/'report.json').write_text(json.dumps(result,indent=2),'utf8')
    return result
