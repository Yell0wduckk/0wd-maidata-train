"""audit/build/verify complete chart inputs, independently of training outputs."""
from pathlib import Path
import sys, json, time, runpy
import torch
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from maimai_baseline.data.song_chart import SCHEMA, SongChartCache, pack_rows
from maimai_baseline.data.style_cache import write_json
from maimai_baseline.style_contract import STYLE_ID
from maimai_baseline.source_identity import baseline_source_identity
from maimai_dataset.integrity import file_sha256
from maimai_dataset.manifest import canonical_json_sha256 as digest
from full_dataset_tools.pipeline import transform_rows
from global_local_cache import CacheBatches, cache_dataset, worker_init


class Charts(Dataset):
    def __init__(self, records, note, vocabulary):
        self.note = Path(note); self.manifest = json.loads(self.note.read_text('utf8'))
        self.vocabulary=vocabulary
        self.records=list({(r['role'],str(r['record_id']),r['view_id']):r for r in records}.values())
    def __len__(self): return len(self.records)
    def __getitem__(self, index):
        r=self.records[index]; meta=self.manifest['charts'][str(r['record_id'])]
        path=self.note.parent/meta['path']
        if not path.is_file() or file_sha256(path)!=meta['sha256']: raise ValueError('missing/changed complete GT')
        document=json.loads(path.read_text('utf8')); rows=document['canonical_rows']
        if document['errors'] or document['audio_onset_tick']!=r['song_end_tick']: raise ValueError('GT audio boundary mismatch')
        packed=pack_rows(transform_rows(rows,r['view_id']),r['song_end_tick'],self.vocabulary)
        return dict(key=[r['role'],str(r['record_id']),r['view_id']], packed=packed, song_end_tick=r['song_end_tick'],
            variant=meta['variant'], source_rows=len(rows), moments=len(packed['moment_rows']), blocks=packed['valid'].numel())


class AudioWindows(Dataset):
    def __init__(self, manifest, variants):
        self.path=Path(manifest);self.meta=json.loads(self.path.read_text('utf8'));self.variants=list(variants)
    def __len__(self):return len(self.variants)
    def __getitem__(self,index):
        import numpy as np
        from maimai_dataset.timing import TimingMap
        from maimai_baseline.data.store import window_arrays
        from maimai_baseline.time.audio_end import audio_end
        key=self.variants[index];v=self.meta['variants'][key];root=self.path.parent
        feature=np.load(root/v['feature_path'],mmap_mode='r',allow_pickle=False)
        with np.load(root/v['coordinates_path'],allow_pickle=False) as f:coords={k:f[k] for k in f.files}
        timing=TimingMap.from_dict(v['timing']);end=audio_end(timing,int(v['num_samples'])).onset_tick
        return key,[dict(start=start,raw=window_arrays(feature,coords,timing,start*96,v['num_samples'])) for start in range(0,end//96+1,16)]


def main(action, tail):
    from maimai_baseline.cli import parser
    from maimai_baseline.training.inputs import _prepare_inputs
    args=parser().parse_args(['fit', *tail])
    root=Path(args.output)
    if action=='verify':
        cache=SongChartCache(root,verify=True)
        return dict(passed=True,identity=cache.manifest['identity'],charts=len(cache.lookup))
    if action not in ('audit','build'): raise ValueError('choose audit/build/verify')
    started=time.perf_counter(); torch.set_num_threads(1)
    inputs=_prepare_inputs(args)
    plan_sha = file_sha256(inputs.plan_store.root/'manifest.json')
    datasets={r:cache_dataset(inputs,r) for r in ('train','validation') if r == 'train' or inputs.manifest['validation']['enabled']}
    windows=CacheBatches(datasets,inputs.vocabulary)
    data=Charts(windows.records,args.note_view_manifest,inputs.vocabulary)
    if action=='build': root.mkdir(parents=True,exist_ok=False)
    charts=[]
    for index,row in enumerate(DataLoader(data,batch_size=None,num_workers=12,worker_init_fn=worker_init)):
        packed=row.pop('packed')
        if action=='build':
            path=root/f'chart-{index}.pt';torch.save(packed,path)
            row.update(file=path.name,sha256=file_sha256(path),bytes=path.stat().st_size)
        charts.append(row)
    audio_files={}
    variants=list(dict.fromkeys(c['variant'] for c in charts))
    if action=='build':
        for index,(key,parts) in enumerate(DataLoader(AudioWindows(args.bpm_audio_manifest,variants),batch_size=None,num_workers=12,worker_init_fn=worker_init)):
            path=root/f'audio-{index}.pt';torch.save(parts,path)
            audio_files[key]=dict(file=path.name,sha256=file_sha256(path),bytes=path.stat().st_size,windows=len(parts))
    audit=dict(charts=len(charts),max_moments=max(c['moments'] for c in charts),max_blocks=max(c['blocks'] for c in charts),seconds=time.perf_counter()-started)
    if action=='build':
        manifest=dict(schema=SCHEMA,ready=True,representation=STYLE_ID,charts=charts,audio_files=audio_files,audit=audit,
            producer=baseline_source_identity(),bindings=dict(plan=plan_sha,note_view=file_sha256(Path(args.note_view_manifest)),
                vocabulary=inputs.vocabulary_sha256,bpm=inputs.manifest['bpm_input']['artifacts']['manifest_sha256']))
        manifest['identity']=digest(manifest);write_json(root/'manifest.json',manifest)
    return audit

if __name__=='__main__': print(json.dumps(main(sys.argv[1],sys.argv[2:]),allow_nan=False))
