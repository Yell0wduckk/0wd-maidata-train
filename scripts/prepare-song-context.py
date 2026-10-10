"""Audit/build/verify the independent full-song sidecar. No training."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_convert
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from maimai_baseline.data.song_context import CONTRACT, SCHEMA, SongContextCache, center_p0, layout, read
from maimai_baseline.data.style_cache import StyleLatentCache, write_json
from maimai_baseline.data.style_audio_cache import AudioConditionCache
from maimai_baseline.data.song_chart_compact import SCHEMA as COMPACT_CHART_SCHEMA
from maimai_dataset.integrity import file_sha256
from maimai_dataset.manifest import canonical_json_sha256 as digest
from maimai_baseline.data.song_chart import SongChartCache
from maimai_baseline.data.style_audio_cache import worker_init
from maimai_baseline.data.store import window_arrays
from maimai_dataset.timing import TimingMap
from maimai_baseline.checkpoint import load_frozen_stage_a
from maimai_baseline.source_identity import baseline_source_identity
from maimai_baseline.data.batch_transport import pack_batch


def packed_convert(row):
    """Preserve DataLoader conversion while sending one IPC buffer per dtype."""
    return pack_batch(default_convert(row))


def require_array(value, shape, dtype):
    if value.shape!=shape or value.dtype!=dtype or not np.isfinite(value).all():
        raise ValueError('invalid song cache tensor shape, dtype or finite values')


def window_p0(audio, part, encoded):
    """Reuse accepted bytes; only missing windows consume newly encoded P0."""
    previous=int(part['previous'])
    if previous>=0:
        result=center_p0(*(audio.arrays[k][previous] for k in ('audio','audio_valid','audio_beats')),part['length'])
    else:
        result=center_p0(encoded.memories[-1][0].cpu().numpy(),(~encoded.padding_masks[-1][0]).cpu().numpy(),
            encoded.beats[-1][0].cpu().numpy()-part['start'],part['length'])
    require_array(result[0],(part['length']*8,256),np.float32)
    require_array(result[1],(part['length']*8,),np.bool_)
    return result


def verify_diagnostic(path, length):
    """Validate stored diagnostic geometry and payloads without running fusion."""
    starts=np.arange(0,length,16,dtype=np.int64)*96
    fields={'global_audio','global_valid','global_beats','starts','fusion_left','fusion_right','fusion_weight'}
    fields.update(f'{name}{j}' for j in range(3) for name in ('m','padding','beats'))
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!=fields:raise ValueError('invalid diagnostic cache fields')
        require_array(z['global_audio'],(length,256),np.float32)
        require_array(z['global_valid'],(length,),np.bool_)
        require_array(z['global_beats'],(length,),np.float32)
        require_array(z['starts'],starts.shape,np.int64)
        np.testing.assert_array_equal(z['starts'],starts)
        np.testing.assert_array_equal(z['global_beats'],np.arange(length,dtype=np.float32)+.5)
        for name,dtype in (('left',np.int64),('right',np.int64),('weight',np.float32)):
            require_array(z['fusion_'+name],((length+22)*2,),dtype)
        left,right,weight=(z['fusion_'+name] for name in ('left','right','weight'))
        if ((left<0)|(right>=length)|(left>right)|(weight<0)|(weight>1)).any():
            raise ValueError('invalid diagnostic fusion geometry')
        for j,n in enumerate((176,88,44)):
            require_array(z[f'm{j}'],(len(starts),n,256),np.float32)
            require_array(z[f'padding{j}'],(len(starts),n),np.bool_)
            require_array(z[f'beats{j}'],(len(starts),n),np.float32)
            expected=starts[:,None]/96+(np.arange(n,dtype=np.float32)+.5)/(8//2**j)-4
            np.testing.assert_array_equal(z[f'beats{j}'],expected.astype(np.float32))


class Work(Dataset):
    def __init__(self,args,charts,windows):
        self.args=args; self.charts=charts; self.windows=windows; self.target=self.audio=None
        self.bpm=read(args.bpm_audio_manifest); self.note=read(args.note_view_manifest)
        self.audio_meta={c['audio_key']:c for c in charts}
        self.jobs=[('chart',i) for i in range(len(charts))]+[('audio',k) for k in self.audio_meta]

    def __len__(self):return len(self.jobs)

    def __getitem__(self,i):
        if self.target is None:
            self.target=StyleLatentCache(self.args.cache)
            self.audio=AudioConditionCache(self.args.audio_condition_cache,self.target)
        kind,key=self.jobs[i]
        if kind=='chart':
            c=self.charts[key]
            info=self.target.full_charts[tuple(c['key'])]
            features=np.load(self.target.root/info['file'],allow_pickle=False)
            return dict(kind=kind,key=key,features=features,missing=np.zeros(c['blocks'],bool),valid=np.ones(c['blocks'],bool))
        c=self.audio_meta[key]; v=self.bpm['variants'][c['variant']]
        result=dict(kind=kind,key=key,length=c['length'],parts=[])
        base=coords=None
        for start in range(0,c['length'],16):
            n=min(16,c['length']-start); old=self.windows.get((key,start*96))
            if base is None:
                root=Path(self.args.bpm_audio_manifest).parent
                base=np.load(root/v['feature_path'],mmap_mode='r',allow_pickle=False)
                with np.load(root/v['coordinates_path'],allow_pickle=False) as z:coords={k:z[k] for k in z.files}
            timing=TimingMap.from_dict(v['timing'])
            if self.target.manifest['bindings']['baseline_manifest']['song_chart']['schema'] == COMPACT_CHART_SCHEMA:
                from maimai_baseline.time.offline import store_for
                if self.args.song_chart_cache is None:
                    raise ValueError('full input context needs --song-chart-cache for timing tables')
                timing=store_for(self.args.song_chart_cache).bind(timing)
            values=window_arrays(base,coords,timing,start*96,v['num_samples'])
            result['parts'].append(dict(start=start,raw=values,length=n,previous=-1 if old is None else old))
        return result


def prepare(args):
    from maimai_baseline.experiment import load_config
    config = load_config(args.experiment_config)
    started=time.perf_counter(); torch.set_num_threads(1)
    target=StyleLatentCache(args.cache); audio=AudioConditionCache(args.audio_condition_cache,target)
    base=target.manifest['bindings']['baseline_manifest']
    if config != base['experiment']['recipe']: raise ValueError('context experiment differs from parent')
    if base['song_chart']['schema'] == COMPACT_CHART_SCHEMA:
        if args.song_chart_cache is None or SongChartCache(args.song_chart_cache).manifest['identity'] != base['song_chart']['identity']:
            raise ValueError('full context song-chart timing/input identity mismatch')
    if (file_sha256(args.bpm_audio_manifest)!=base['bpm_input']['artifacts']['manifest_sha256']
            or file_sha256(args.note_view_manifest)!=base['note_view']['sha256']):
        raise ValueError('supplemental context inputs do not match frozen A')
    from maimai_baseline.data.song_audio import SongAudioCache,pack
    global_inputs=SongAudioCache(args.song_audio_cache)
    if global_inputs.manifest['identity']!=target.manifest['bindings'].get('song_audio_input_identity',base['song_audio']['identity']):raise ValueError('song audio producer identity mismatch')
    if args.action=='verify':
        verified_audio=set()
        cache=SongContextCache(args.output,target,audio,verify=True)
        # Inspect payloads and direct copies only. Numerical model equivalence is tested separately.
        for ci,c in enumerate(cache.charts):
            with np.load(Path(args.output)/f'chart-{ci}.npz',allow_pickle=False) as z:
                if set(z.files)!={'features','mu','logvar','valid'}:raise ValueError('invalid song chart fields')
                for field in ('features','mu','logvar'):require_array(z[field],(c['blocks'],128),np.float32)
                require_array(z['valid'],(c['blocks'],),np.bool_)
                if not z['valid'].all() or ((z['logvar'] < -6)|(z['logvar'] > 2)).any():
                    raise ValueError('invalid song chart payload')
                for i in c['rows']:
                    r=target.records[i]; start=r['start_tick']//384; n=min(4,c['blocks']-start)
                    mask=target.arrays['valid'][i,:n]
                    np.testing.assert_array_equal(z['features'][start:start+n][mask],target.arrays['features'][i,:n][mask])
            with np.load(Path(args.output)/f"audio-{c['audio_key']}.npz",allow_pickle=False) as z:
                expected=(np.arange(c['length']*8,dtype=np.float32)+.5)/8
                if set(z.files)!={'audio','audio_valid','audio_beats'}:raise ValueError('invalid song audio fields')
                require_array(z['audio'],(len(expected),256),np.float32)
                require_array(z['audio_valid'],(len(expected),),np.bool_)
                require_array(z['audio_beats'],expected.shape,np.float32)
                np.testing.assert_array_equal(z['audio_beats'],expected)
                for i in c['rows']:
                    start=target.records[i]['start_tick']//96; n=min(16,c['length']-start); ai=int(audio.mapping[i])
                    a,m=center_p0(*(audio.arrays[k][ai] for k in ('audio','audio_valid','audio_beats')),n)
                    np.testing.assert_array_equal(z['audio'][start*8:(start+n)*8],a)
                    np.testing.assert_array_equal(z['audio_valid'][start*8:(start+n)*8],m)
        for c in cache.charts:
            if c['audio_key'] in verified_audio:continue
            verify_diagnostic(cache.root/f"diagnostic-{c['audio_key']}.npz",c['length'])
            verified_audio.add(c['audio_key'])
        return dict(passed=True,identity=cache.manifest['identity'],charts=len(cache.charts))
    charts,mapping,windows=layout(target,audio,read(args.bpm_audio_manifest),read(args.note_view_manifest))
    work=Work(args,charts,windows)
    summary=dict(charts=len(charts),audio_groups=len(work.audio_meta),max_beats=max(c['length'] for c in charts),
        missing_beats=0,supplemented_audio_windows=0,workers=12)
    build=args.action=='build'; root=Path(args.output)
    if build:
        root.mkdir(parents=True,exist_ok=False)
        parent_sha=target.manifest['bindings'].get('stage_a_sha256')
        model,parent=load_frozen_stage_a(Path(args.stage_a_checkpoint),route_artifact=Path(args.route_artifact))
        model.to(args.device).eval().requires_grad_(False)
    files={}
    loader=DataLoader(work,batch_size=None,num_workers=12,worker_init_fn=worker_init,
        collate_fn=packed_convert, generator=torch.Generator().manual_seed(20260811))
    for number,packed in enumerate(loader):
        row=packed.unpack()
        kind=row['kind']; key=row['key']
        if kind=='chart':
            summary['missing_beats']+=int(row['missing'].sum())
            if build:
                f=row['features'].numpy()
                with torch.no_grad():
                    h=model.chart_style_encoder.local_hidden(row['features'].to(args.device).float())
                    mu=model.chart_style_encoder.mu(h).cpu().numpy();lv=model.chart_style_encoder.logvar(h).clamp(-6,2).cpu().numpy()
                for value in (f,mu,lv):require_array(value,(charts[key]['blocks'],128),np.float32)
                name=f'chart-{key}.npz'; np.savez(root/name,features=f,mu=mu,logvar=lv,valid=row['valid'].numpy())
        else:
            summary['supplemented_audio_windows']+=sum(int(part['previous'])<0 for part in row['parts'])
            if build:
                raw=pack([global_inputs.get(work.audio_meta[key]['variant'])],[0])
                with torch.no_grad():full=model.song_audio_encoder(**{k:v.to(args.device) for k,v in raw.items()})
                values=[];masks=[];unfused=[[],[],[]];paddings=[[],[],[]];coordinates=[[],[],[]];starts=[]
                for part in row['parts']:
                    mel,valid,beats,bpms,geometry=part['raw']
                    with torch.no_grad():
                        out=model.encode_audio(mel[None].to(args.device),~valid[None].to(args.device),
                            beats[None].to(args.device),bpms[None].to(args.device),song_audio=full,
                            start_tick=torch.tensor([part['start']*96],device=args.device),
                            interpolation_geometry={k:v[None].to(args.device) for k,v in geometry.items()})
                    # Encoding above supplies missing diagnostic intermediates, not a P0 comparison.
                    a,m=window_p0(audio,part,out)
                    values.append(a);masks.append(m);starts.append(int(part['start'])*96)
                    for j in range(3):
                        unfused[j].append(out.diagnostics['unfused_memories'][j][0].cpu().numpy())
                        paddings[j].append(out.padding_masks[2-j][0].cpu().numpy())
                        coordinates[j].append(out.beats[2-j][0].cpu().numpy())
                diagnostic=dict(global_audio=full.memory[0].cpu().numpy(),global_valid=full.valid[0].cpu().numpy(),
                    global_beats=full.beats[0].cpu().numpy(),starts=np.asarray(starts,np.int64))
                diagnostic.update({f'fusion_{k}':v[0].cpu().numpy() for k,v in full.geometry.items()})
                for j in range(3):
                    diagnostic.update({f'm{j}':np.stack(unfused[j]),f'padding{j}':np.stack(paddings[j]),f'beats{j}':np.stack(coordinates[j])})
                diagnostic_name=f'diagnostic-{key}.npz';np.savez(root/diagnostic_name,**diagnostic)
                verify_diagnostic(root/diagnostic_name,row['length'])
                files[diagnostic_name]=dict(bytes=(root/diagnostic_name).stat().st_size,sha256=file_sha256(root/diagnostic_name))
                name=f'audio-{key}.npz';np.savez(root/name,audio=np.concatenate(values),audio_valid=np.concatenate(masks),
                    audio_beats=(np.arange(row['length']*8,dtype=np.float32)+.5)/8)
        if build:files[name]=dict(bytes=(root/name).stat().st_size,sha256=file_sha256(root/name))
        if number%100==0:print(json.dumps(dict(prepared=number,total=len(work))),flush=True)
    summary['seconds']=time.perf_counter()-started
    if build:
        plan_sha=target.manifest['bindings']['plan_manifest_sha256']
        from maimai_baseline.runtime_versions import stamp
        m=dict(**stamp(), schema=SCHEMA,ready=True,contract=CONTRACT,bindings=dict(parent=parent_sha,plan=plan_sha,
            target=target.manifest['manifest_identity_sha256'],audio=audio.manifest['identity'],song_audio=global_inputs.manifest['identity']),
            charts=charts,record_to_chart=mapping,files=files,audit=summary,producer=baseline_source_identity(),
            producer_script_sha256=file_sha256(Path(__file__)),
            inputs=dict(bpm_manifest=file_sha256(args.bpm_audio_manifest),note_view=file_sha256(args.note_view_manifest)))
        m['identity']=digest(m);write_json(root/'manifest.json',m)
    return summary


def parser():
    p=argparse.ArgumentParser();p.add_argument('--experiment-config',type=Path,required=True);p.add_argument('action',choices=['audit','build','verify'])
    for name in ('cache','audio-condition-cache','bpm-audio-manifest','note-view-manifest','stage-a-checkpoint','route-artifact','song-audio-cache','output'):
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--song-chart-cache',type=Path)
    p.add_argument('--device',default='cuda');p.add_argument('--cache-workers',type=int,default=12,choices=[12])
    return p

if __name__=='__main__':print(json.dumps(prepare(parser().parse_args()),allow_nan=False))
