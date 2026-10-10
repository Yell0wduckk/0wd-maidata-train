"""Controlled tail probes and suffix rollouts; leave the bound model source intact.

Probes restore the identical native cache before each current-query intervention.
Suffix interventions start at the original chart's final onset. Latents and the
pre-intervention prefix stay fixed, including information already in old K/V.
"""
import argparse
from collections import Counter
from copy import copy, deepcopy
import inspect
import json
from pathlib import Path
import sys
import time
import torch
from torch.nn import functional as F

MODES = ('local_audio_only', 'no_direct_audio', 'no_history_notes')
CUT = 41808  # original MAS last onset: 435.5 beats, 142.010869565 s


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')


def clone_state(state):
    # Immutable catalog/vocabulary and committed runtime snapshot are shared.
    # Caches/engines and trial dictionaries are private; no CF row is committed.
    other = copy(state)
    for name, value in vars(state).items():
        if isinstance(value, dict):
            setattr(other, name, value.copy())
        elif isinstance(value, list):
            setattr(other, name, value.copy())
    other._stage_engine = other._engine_key = other._interval_context = None
    other._cache = {}; other._time_values = {}; other._time_tick = None
    other._trial_cache = {}; other._trial_states = {}
    return other


class Experiment:
    def __init__(self, root, mode):
        self.root, self.run_mode = root, mode
        self.mode = 'baseline'
        self.active = None
        self.last = None
        self.probing = False
        self.records = []
        self.rows = []
        self.stats = Counter()
        self.native_sdpa = F.scaled_dot_product_attention
        self.patches = []
        self.handles = []
        reference=json.loads(Path('/root/autodl-tmp/dense-b50-11739-attention-20261004/journal.json').read_text())['accepted_predictions']
        self.cut_count=next(i for i,x in enumerate(reference) if x['anchor_tick']>CUT)

    def patch(self, obj, name, value):
        self.patches.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def install(self):
        from maimai_baseline.inference import api, retry, family, generation, reporting
        from maimai_baseline.inference.song_dense import SongDenseKVCache as Cache
        from maimai_baseline.model import rope
        self.Cache, self.family, self.generation, self.reporting = Cache, family, generation, reporting
        loaded = api.load_baseline_inference_checkpoint
        def load(*a, **kw):
            result = loaded(*a, **kw)
            self.model = result[0]
            self.note_ids = {self.model.vocabulary.event_id(x) for x in ('NOTE', 'SLIDE_BRANCH')}
            for layer in self.model.decoder_layers:
                # Remove the entire residual, including output projection bias.
                for name, module in layer.named_modules():
                    if name.endswith('audio_o'):
                        self.handles.append(module.register_forward_hook(lambda m,a,o: torch.zeros_like(o)
                            if self.active is not None and self.mode == 'no_direct_audio' else o))
            return result
        self.patch(api, 'load_baseline_inference_checkpoint', load)
        native_decode, native_snapshot, native_restore = Cache.decode, Cache.snapshot, Cache.restore
        native_begin, native_trim = Cache.begin_window, Cache.trim_local
        def snapshot(cache):
            return (native_snapshot(cache), list(getattr(cache,'_tail_events',[])), list(getattr(cache,'_tail_router_events',[])))
        def restore(cache, state, *, replay_last=False):
            native_restore(cache, state[0], replay_last=replay_last)
            cache._tail_events = state[1][:cache.song_length]
            cache._tail_router_events = state[2][:cache.length]
        def begin(cache, *a, **kw):
            native_begin(cache,*a,**kw); cache._tail_router_events=[]
        def trim(cache, length):
            native_trim(cache,length)
            cache._tail_events=cache._tail_events[:cache.song_length]
            cache._tail_router_events=cache._tail_router_events[:cache.length]
        def decode(cache, **kw):
            before = snapshot(cache)
            events=kw['input_ids']['event'][0].tolist()
            first=0
            if not cache.length and cache.song_length:
                first=events.index(cache.model.vocabulary.event_id('TARGET_START'))
            dense_events=before[1]+events[first:]
            router_events=before[2]+events
            if not self.probing:
                self.mode=self.run_mode if self.run_mode in MODES and len(self.rows)>=self.cut_count else 'baseline'
            self.active=dict(cache=cache,dense_events=dense_events,router_events=router_events)
            try:
                result=native_decode(cache,**kw)
                if self.mode=='no_direct_audio' and 'delta_candidate' in result:
                    result['delta_candidate']=dict(result['delta_candidate'],gate=torch.zeros_like(result['delta_candidate']['gate']))
                    self.stats['delta_candidate_disabled']+=1
                cache._tail_events=dense_events;cache._tail_router_events=router_events
                assert len(dense_events)==cache.song_length and len(router_events)==cache.length
                if not self.probing:
                    self.last=(cache,before,dict(kw))
                return result
            finally:
                self.active=None
        self.decode=decode
        self.patch(Cache,'snapshot',snapshot);self.patch(Cache,'restore',restore)
        self.patch(Cache,'begin_window',begin);self.patch(Cache,'trim_local',trim)
        self.patch(Cache,'decode',decode)
        attend=rope.attend
        def attended(attention, query, kv, **kw):
            if self.active is not None and self.mode=='no_direct_audio' and attention is self.model.router.audio:
                self.stats['router_audio_disabled']+=1
                return query.new_zeros((*query.shape[:-1],attention.embed_dim))
            return attend(attention,query,kv,**kw)
        self.patch(rope,'attend',attended)
        self.patch(F,'scaled_dot_product_attention',self.sdpa)
        selected=retry.select_with_router
        def select(outputs, **kw):
            # Probe slightly before the cutoff too, to include the crossing MOMENT.
            if self.run_mode!='probe' or kw['tick'] < CUT-96:
                return selected(outputs,**kw)
            cache,before,inputs=self.last
            after=cache.snapshot()
            record=dict(tick=kw['tick'],query_seconds=kw['tick']/96*60/184,
                        window_start=int(cache.start[0]),prefix_rows=cache.song_length,conditions={})
            original_generator_state=kw['generator'].get_state()
            selector_values=inspect.getclosurevars(inputs['family_selector']).nonlocals
            for mode in ('baseline_repeat',)+MODES:
                self.probing=True;self.mode='baseline' if mode=='baseline_repeat' else mode
                cache.restore(before)
                state=clone_state(kw['state'])
                generator=torch.Generator(device=kw['generator'].device)
                generator.set_state(original_generator_state)
                report=self.reporting.Reporting(True)
                token=self.reporting._session.set(report)
                trace={}
                selector_args={name:selector_values[name] for name in inspect.signature(self.family.selector).parameters
                               if name in selector_values}
                selector_args.update(state=state,generator=generator,trace=trace)
                def infer(excluded=(), replay=False):
                    values=dict(inputs)
                    if replay:
                        cache.restore(cache.snapshot(),replay_last=True)
                        for field in ('input_ids','input_times','input_time_active'):
                            values[field]={k:v[:,-1:] for k,v in values[field].items()}
                        for field in ('row_anchor_tick','row_padding_mask'):
                            if field in values:values[field]=values[field][:,-1:]
                    values['family_selector']=self.family.selector(**dict(selector_args,excluded_families=excluded))
                    self.reporting.begin_row()
                    result=self.decode(cache,**values)
                    return self.generation._constrain_moment_boundary(result,vocabulary=self.model.vocabulary,
                        current_moment=selector_args['current_moment'])
                try:
                    candidate=infer()
                    new_kw=dict(kw,state=state,generator=generator,decode_again=lambda excluded:infer(excluded,True))
                    prediction,detail,drift=selected(candidate,**new_kw)
                    record['conditions'][mode]=dict(row=prediction,trace=detail,drift=drift,router=trace)
                except Exception as exc:
                    record['conditions'][mode]=dict(error=repr(exc),router=trace)
                finally:
                    self.reporting._session.reset(token)
            cache.restore(after);self.probing=False;self.mode='baseline'
            assert torch.equal(kw['generator'].get_state(),original_generator_state)
            prediction,detail,drift=selected(outputs,**kw)
            record['baseline']=dict(row=prediction,trace=detail,drift=drift)
            repeated=record['conditions']['baseline_repeat']
            assert 'error' not in repeated and repeated['row']==prediction, (record['tick'],repeated,prediction)
            self.records.append(record)
            save(self.root/'fixed-prefix-probes.json',self.records)
            print('PROBE',record['tick'],prediction,{m:x.get('row',x.get('error')) for m,x in record['conditions'].items()},flush=True)
            return prediction,detail,drift
        self.patch(retry,'select_with_router',select)

    def sdpa(self,q,k,v,attn_mask=None,dropout_p=0.,is_causal=False,**kw):
        if self.active is not None and self.mode in ('local_audio_only','no_history_notes'):
            frame=sys._getframe(1); frames=[]
            try:
                for _ in range(10):
                    if frame is None:break
                    frames.append((frame.f_code.co_name,frame.f_globals.get('__name__',''),frame.f_locals))
                    frame=frame.f_back
                dual=next((loc for fn,mod,loc in frames if fn=='attention' and mod.endswith('model.dual')),None)
                router=next((loc for fn,mod,loc in frames if fn=='self_attention' and mod.endswith('model.rope')),None)
                allowed=attn_mask
                if self.mode=='local_audio_only' and dual is not None and not dual['causal']:
                    cache=self.active['cache'];start=cache.start[:,None]/96
                    local=(cache.memory_beats>=start-4)&(cache.memory_beats<start+18)
                    assert k.shape[2]==local.shape[1]
                    allowed=allowed & local[:,None,None,:]
                    self.stats['local_audio_mask_calls']+=1
                    self.stats['remote_keys_masked']+=int((~local & ~cache.memory_padding).sum())
                elif self.mode=='no_history_notes' and ((dual is not None and dual['causal']) or router is not None):
                    ev=self.active['dense_events' if dual is not None else 'router_events']
                    assert len(ev)==k.shape[2]
                    note=torch.tensor([e in self.note_ids for e in ev],device=q.device)
                    key=torch.arange(k.shape[2],device=q.device)
                    qi=k.shape[2]-q.shape[2]+torch.arange(q.shape[2],device=q.device)
                    drop=note[None,:] & (key[None,:]<qi[:,None])
                    allowed=allowed & ~drop[None,None]
                    self.stats['history_mask_calls']+=1
                    self.stats['history_note_edges_masked']+=int(drop.sum())
                attn_mask=allowed
            finally:
                del frame,frames
        return self.native_sdpa(q,k,v,attn_mask=attn_mask,dropout_p=dropout_p,is_causal=is_causal,**kw)

    def stream(self,event,**data):
        if event=='row':self.rows.append(dict(data))
        elif event=='rollback':
            current=[x for x in self.rows if x['index']==data['index']]
            self.rows=[x for x in self.rows if x['index']!=data['index']]+current[:data['row_count']]
        elif event=='window_end':print('WINDOW',data['index'],flush=True)

    def close(self):
        for handle in self.handles:handle.remove()
        for obj,name,value in reversed(self.patches):setattr(obj,name,value)
        save(self.root/'accepted-rows.json',self.rows)
        save(self.root/'intervention-counts.json',dict(self.stats))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=('probe',)+MODES,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(args.source/'src'))
    from maimai_baseline.inference.api import generate_baseline_inference
    exp=Experiment(args.output,args.mode);exp.install()
    base=Path('/root/autodl-tmp');data=base/'maimai-full-dataset-lr-fixed-20260911'
    inputs=base/'moment-simai-c50-five-song-20261003/inputs/11739'
    start=time.time()
    save(args.output/'contract.json',dict(mode=args.mode,source=str(args.source),cutoff_tick=CUT,
        far_audio='outside original window [start-4,start+18) beats',
        all_audio='disable decoder audio residuals, router audio residual, delta candidate audio gate',
        history='mask strictly earlier NOTE and SLIDE_BRANCH keys in decoder and router; retain current query, MOMENT and controls',
        retained='fixed AR latent, pre-intervention K/V and audio encoder representations; not removal of all indirect information'))
    try:
        result=generate_baseline_inference(checkpoint=base/'maidata-decoder8-song-dense-20261003/training/branch-090/B/ar.ckpt',
            route_artifact=data/'slide-geometry/route-artifact',profile_root=data/'profile',song_id='11739',
            timing_path=inputs/'timing.json',output=args.output/'inference',level=14.,cabinet='DX',difficulty_id=5,
            force=False,device='cuda',seed=20260811,temperature=0.,audio_path=inputs/'audio.mp3',
            physical_hard_mask=True,physical_path_catalog=data/'physical/path-catalog',miacode_slide_check=True,
            trace=True,local_seed=20260812,style_mode='ar',latent_temperature=1.,mixture_temperature=1.,stream=exp.stream)
        reference=json.loads((base/'dense-b50-11739-attention-20261004/journal.json').read_text())['accepted_predictions']
        projected=lambda xs:[{k:x[k] for k in ('index','row','anchor_tick')} for x in xs]
        pre=lambda xs:[x for x in xs if x['anchor_tick']<=CUT]
        same_prefix=projected(exp.rows[:exp.cut_count])==projected(reference[:exp.cut_count])
        assert same_prefix
        same_all=projected(exp.rows)==projected(reference)
        if args.mode=='probe':assert same_all
        save(args.output/'completion.json',dict(exit_code=0,seconds=time.time()-start,same_prefix=same_prefix,
            same_all=same_all,rows=result['generated_rows'],truncated=result['truncated'],probes=len(exp.records),
            preserved_prefix_predictions=exp.cut_count))
    except Exception as exc:
        save(args.output/'failure.json',dict(error=repr(exc),seconds=time.time()-start));raise
    finally:exp.close()


if __name__=='__main__':main()
