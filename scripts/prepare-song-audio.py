"""Audit/build/verify independent whole-song mel inputs, without model execution."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from torch.utils.data import Dataset,DataLoader
from maimai_baseline.data.song_audio import *
from maimai_baseline.data.style_audio_cache import worker_init
from maimai_dataset.timing import TimingMap
from maimai_baseline.source_identity import baseline_source_identity


class Work(Dataset):
    def __init__(self,bpm,jobs):self.bpm=Path(bpm);self.jobs=jobs
    def __len__(self):return len(self.jobs)
    def __getitem__(self,index):
        name,v=self.jobs[index];root=self.bpm.parent
        feature=np.load(root/v['feature_path'],mmap_mode='r',allow_pickle=False)
        with np.load(root/v['coordinates_path'],allow_pickle=False) as z:coords={k:z[k] for k in z.files}
        return name,raw_song(feature,coords,TimingMap.from_dict(v['timing']),v['num_samples'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['audit','build','verify'])
    for name in ('bpm-audio-manifest','song-chart-cache','epoch-plan-cache','output'):p.add_argument('--'+name,required=True)
    args=p.parse_args();started=time.perf_counter()
    root=Path(args.output);bpm=json.loads(Path(args.bpm_audio_manifest).read_text('utf8'))
    charts=json.loads((Path(args.song_chart_cache)/'manifest.json').read_text('utf8'))
    bindings=dict(bpm=file_sha256(args.bpm_audio_manifest),chart_inputs=charts['identity'],plan=file_sha256(Path(args.epoch_plan_cache)/'manifest.json'))
    if charts['bindings']['bpm']!=bindings['bpm'] or charts['bindings']['plan']!=bindings['plan']:raise ValueError('song audio input bindings disagree')
    variants={};jobs={}
    for c in charts['charts']:
        key=c['variant'];v=bpm['variants'][key]
        name=digest(dict(audio=v['identity'],preprocessing=bpm['preparation_identity']))+'.npz'
        variants[key]=name;jobs.setdefault(name,v)
    if args.action=='verify':
        cache=SongAudioCache(root,verify=True)
        if cache.manifest['bindings']!=bindings or cache.manifest['variants']!=variants:raise ValueError('song audio cache base mismatch')
    if args.action=='build':root.mkdir(parents=True,exist_ok=False)
    files={};tokens=0
    for name,row in DataLoader(Work(args.bpm_audio_manifest,list(jobs.items())),batch_size=None,num_workers=12,worker_init_fn=worker_init):
        row={k:np.asarray(v) for k,v in row.items()};tokens+=len(row['beats'])
        if args.action=='build':
            np.savez(root/name,**row);files[name]=dict(bytes=(root/name).stat().st_size,sha256=file_sha256(root/name))
        elif args.action=='verify':
            with np.load(root/name,allow_pickle=False) as z:
                for k,v in row.items():np.testing.assert_array_equal(z[k],v)
    result=dict(songs=len(jobs),tokens=tokens,raw_bytes=tokens*(24*128*4+24+8),seconds=time.perf_counter()-started,workers=12)
    if args.action=='build':
        m=dict(schema=SCHEMA,ready=True,contract=CONTRACT,bindings=bindings,variants=variants,files=files,audit=result,producer=producer_identity())
        m['identity']=digest(m);(root/'manifest.json').write_text(json.dumps(m,indent=2),encoding='utf8')
    print(json.dumps(dict(passed=True,**result)))


if __name__=='__main__':main()
