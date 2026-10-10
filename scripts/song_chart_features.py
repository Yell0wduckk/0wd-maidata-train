"""Bound A full-song feature production and conditional FP64 moments."""
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from maimai_baseline.data.song_chart import SongChartCache
from maimai_baseline.data.batch_transport import pack_batch, PackedBatch
from maimai_baseline.data.song_chart_compact import SCHEMA as COMPACT_SCHEMA
from maimai_baseline.data.global_style import stable_seed
from maimai_baseline.data.style_cache import CACHE_SCHEMA, representation_cache_conditions, write_json
from maimai_baseline.model.song_chart import encode_song
from maimai_baseline.training.steps import _move
from maimai_baseline.style_contract import STYLE_ID, style_identity
from maimai_dataset.manifest import canonical_json_sha256 as digest
from maimai_dataset.integrity import file_sha256


class Work(Dataset):
    def __init__(self, root, keys):
        self.root=str(root);self.cache=SongChartCache(root);self.keys=list(dict.fromkeys(keys))
        if not set(self.keys).issubset(self.cache.lookup): raise ValueError("selected chart/view missing from GT")
    def __len__(self):return len(self.keys)
    def __getitem__(self,i):
        packed=self.cache.get(self.keys[i])
        return i,pack_batch(packed)


@torch.no_grad()
def build(writer, model, chart_root, device):
    started=time.perf_counter(); work=Work(chart_root,((r["role"],str(r["record_id"]),r["view_id"]) for r in writer.records)); full={}; infos=[]
    for i,packed in DataLoader(work,batch_size=None,num_workers=12):
        if isinstance(packed,PackedBatch):packed=packed.unpack()
        h=encode_song(model.chart_style_encoder,_move(packed,device))[0].cpu().numpy()
        key=work.keys[i]; full[key]=h
        path=writer.root/f'full-chart-{i}.npy';np.save(path,h,allow_pickle=False)
        infos.append(dict(key=list(key),file=path.name,shape=list(h.shape),bytes=path.stat().st_size,sha256=file_sha256(path)))
    for i,r in enumerate(writer.records):
        key=(r['role'],str(r['record_id']),r['view_id']);start=r['start_tick']//384
        h=full[key];n=min(4,len(h)-start)
        f=np.zeros((1,4,128),np.float32);f[0,:n]=h[start:start+n]
        c=representation_cache_conditions(**{k:torch.tensor([r[k]]) for k in ('start_tick','end_tick','song_end_tick')})
        c['reference_features']=torch.from_numpy(f)
        writer.append([i],torch.from_numpy(f),c)
    totals={k:[torch.zeros(128,dtype=torch.float64,device=device),torch.zeros(128,dtype=torch.float64,device=device),0.] for k in ('local',)}
    def add(name,mu,lv,weight):
        mu,lv=mu.double().reshape(-1,128),lv.double().reshape(-1,128)
        totals[name][0].add_(mu.sum(0),alpha=weight);totals[name][1].add_((mu.square()+lv.exp()).sum(0),alpha=weight);totals[name][2]+=weight*len(mu)
    for key,h in full.items():
        if key[0]!='train':continue
        h=torch.from_numpy(h)[None].to(device);valid=torch.ones(h.shape[:2],dtype=torch.bool,device=device)
        weight=2. if key[2]==0 else 1.
        lp=model.chart_style_encoder.posterior(h,valid)
        add('local',lp.mu,lp.logvar,weight)
    norm=dict(scope='train-only',view_weights={'identity':2,'mirror-lr':1},includes_posterior_variance=True,
        reference_draws=0,seed=20260811,algorithm='full-song-all-blocks-exact-posterior-moments-v1')
    for kind,(first,second,weight) in totals.items():
        mean=first/weight;std=(second/weight-mean.square()).clamp_min(1e-8).sqrt()
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():raise ValueError('invalid normalization')
        norm[kind]=dict(mean=mean.cpu().tolist(),std=std.cpu().tolist(),weighted_units=weight)
    if not writer.written.all():raise ValueError("incomplete full-song feature cache")
    files={}
    for name,array in writer.arrays.items():
        array.flush();path=writer.root/(name+'.npy')
        files[name]=dict(file=path.name,dtype=str(array.dtype),shape=list(array.shape),bytes=path.stat().st_size,sha256=file_sha256(path))
    from maimai_baseline.runtime_versions import stamp
    m=dict(**stamp(), schema=CACHE_SCHEMA,ready=True,chart_style=style_identity(STYLE_ID),bindings=writer.bindings,
        records=writer.records,files=files,normalization=norm,full_charts=infos,song_chart_identity=work.cache.manifest['identity'])
    m['manifest_identity_sha256']=digest(m);write_json(writer.root/'manifest.json',m)
    return dict(written=True,identity=m['manifest_identity_sha256'],seconds=time.perf_counter()-started,charts=len(infos))
