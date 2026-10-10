"""Derive explicitly selected interval tables from accepted action metadata.

No old bitplane payload is read, rebuilt or hashed. The new manifest records
its producers and the exact accepted input manifest bindings separately.
"""
from pathlib import Path
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from maimai_baseline.physical.catalog import CompactPathCatalog,pad_bit
from maimai_baseline.physical.intervals import describe
from maimai_baseline.physical.interval_store import build_chart
from maimai_baseline.physical.shared_cache import CACHE_IDENTITY
from maimai_baseline.physical.spatial import transform_mask
from maimai_dataset.vocabulary import CompactVocabulary
from maimai_dataset.timing import TimingMap
from maimai_dataset.manifest import canonical_json_sha256,load_jsonl
from maimai_dataset.integrity import file_sha256


def build_job(job):
    import torch
    torch.set_num_threads(1)
    started=time.perf_counter()
    root=job.pop('root');index=job.pop('index');view=job.pop('view_index')
    result=build_chart(root,**job)
    result.update(chart_index=index,view_index=view,build_seconds=time.perf_counter()-started)
    result['files']=[dict(path=root.name+'/'+n,sha256=file_sha256(root/n)) for n in ('chart.json','supports.npy')]
    result['bytes']=sum((root/n).stat().st_size for n in ('chart.json','supports.npy'))
    return result


def build(args):
    root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    source=Path(args.action_cache)
    manifest=json.loads((source/'manifest.json').read_text('utf-8'))
    profile=Path(args.profile)
    vocabulary=CompactVocabulary.from_dict(json.loads((profile/'vocabulary.json').read_text('utf-8')))
    catalog=CompactPathCatalog.load(Path(args.path_catalog),vocabulary)
    charts=list(load_jsonl(source/'charts.jsonl'));segments=list(load_jsonl(source/'segments.jsonl'))
    profile_segments={r['segment_id']:r for r in load_jsonl(profile/'segments.jsonl')}
    bpm=json.loads(Path(args.bpm_manifest).read_text('utf-8'))
    names=('action_values','source_action','source_moment','moment_ticks')
    arrays={n:np.load(source/manifest['arrays'][n]['path'],mmap_mode='r',allow_pickle=False) for n in names}
    offsets={n:np.load(source/manifest['offsets'][n]['path'],mmap_mode='r',allow_pickle=False) for n in ('action_offsets','source_offsets','moment_offsets')}
    def section(n,group,index):return arrays[n][int(offsets[group][index]):int(offsets[group][index+1])]
    selected=set(args.chart_indices)
    pad_names=['C']+[ring+str(i) for ring in 'ABDEK' for i in range(1,9)]
    pads={pad_bit(p):p for p in pad_names}
    inverse={catalog.path_id(p,r):(p,r) for p in range(1,9) for r in range(71) if catalog.path_id(p,r)>=0}
    from maimai_baseline.contract import VIEW_NAMES as view_names
    from maimai_baseline.data.route import Route71Codec
    codec=Route71Codec(vocabulary)
    jobs=[]
    for chart in charts:
        index=int(chart['chart_index'])
        if index not in selected:continue
        segment=next(s for s in segments if int(s['chart_index'])==index)
        meta=profile_segments[segment['segment_id']]
        timing=TimingMap.from_dict(meta['timing'])
        variant=bpm['variants'][bpm['segments'][segment['segment_id']]]
        end=float(variant['num_samples'])/24000
        raw=section('action_values','action_offsets',index)
        encoded=json.loads((profile/meta['encoded_path']).read_text('utf-8'))['rows']
        for view in args.views:
            actions=[];owner=-1
            for i,a in enumerate(raw):
                branch=bool(a[2]);kind=('tap','touch','hold','touch_hold','slide','wifi')[int(a[3])] if not branch else None
                if not branch:owner=i
                row={'event':'SLIDE_BRANCH' if branch else 'NOTE'}
                if not branch:row['kind']=kind
                source_row=encoded[int(a[1])]
                route=None
                if int(a[5])>=0:
                    path=int(catalog.path_ids_by_view[view,int(a[5])])
                    lane,_=inverse[path]
                    source_route=codec._factor_to_variant[(str(source_row['shape']),int(source_row['end_offset']),None if 'via_offset' not in source_row else int(source_row['via_offset']))]
                    route=70 if source_route==70 else next(r for r in range(70) if catalog.path_id(lane,r)==path)
                    row.update(start_pad='A'+str(lane),branch_duration_tick=int(a[7]))
                else:
                    row['pad']=pads[transform_mask(int(a[4]),view)]
                    if kind in ('hold','touch_hold'):row['hold_duration_tick']=int(a[7])
                for f in ('flags','head_flags','branch_flags'):
                    if f in source_row:row[f]=source_row[f]
                actions.append(describe(row,tick=int(a[6]),moment=int(a[0]),owner=owner,timing=timing,catalog=catalog,route=route))
            folder=f'chart-{index}-view-{view}'
            jobs.append(dict(root=root/folder,index=index,view_index=view,chart=chart['record_id'],view=view_names[view],timing=timing,
                source_moments=section('source_moment','source_offsets',index),source_actions=section('source_action','source_offsets',index),
                moment_ticks=section('moment_ticks','moment_offsets',index),actions=actions,catalog=catalog,vocabulary=vocabulary,audio_end_seconds=end))
    views=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(build_job,jobs):
            views.append(result);print(json.dumps(result),flush=True)
    from maimai_baseline.source_identity import baseline_source_identity
    result=dict(schema=CACHE_IDENTITY['schema'],contract=CACHE_IDENTITY,path_catalog=dict(catalog.identity),
        producer=baseline_source_identity(),action_metadata_manifest_sha256=file_sha256(source/'manifest.json'),
        charts=[c for c in charts if int(c['chart_index']) in selected],views=views,
        segments=[s for s in segments if int(s['chart_index']) in selected],
        **{k:manifest[k] for k in ('profile_fingerprint','chart_cache_manifest_sha256','spatial_cache_manifest_sha256')})
    result['manifest_identity_sha256']=canonical_json_sha256(result)
    (root/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('action-cache','profile','path-catalog','bpm-manifest','output'):p.add_argument('--'+n,required=True,type=Path)
    p.add_argument('--chart-indices',type=int,nargs='+',required=True)
    p.add_argument('--workers',type=int,default=12)
    p.add_argument('--views',type=int,nargs='+',choices=range(4),default=[0,1])
    build(p.parse_args())
