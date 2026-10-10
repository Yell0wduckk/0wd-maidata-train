"""Pilot original-batch CUDA validation for conditional row fields and intervals.

Consumes explicit exported original-plan batches. Reconstructs only their selected
whole-song GT packs from accepted pilot note-view rows, in 12 workers. No fitting
run, checkpoint, source-input overwrite, or batch resizing is performed.
"""
import argparse,json,time,statistics,sys
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.utils.data import Dataset,DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

class SelectedCharts(Dataset):
    def __init__(self,keys,lookup,note,vocabulary):
        self.keys,self.lookup,self.note,self.vocabulary=keys,lookup,Path(note),vocabulary
        self.meta=json.loads(self.note.read_text())['charts']
    def __len__(self):return len(self.keys)
    def __getitem__(self,i):
        from full_dataset_tools.pipeline import transform_rows
        from maimai_baseline.data.song_chart import pack_rows
        key=self.keys[i];meta=self.meta[key[1]]
        document=json.loads((self.note.parent/meta['path']).read_text())
        end=self.lookup[key]['song_end_tick']
        if document['errors'] or document['audio_onset_tick']!=end:raise ValueError('selected chart boundary mismatch')
        return pack_rows(transform_rows(document['canonical_rows'],key[2]),end,self.vocabulary)

def worker_init(_):torch.set_num_threads(1)
def single(items):return items[0]

def run(args):
    from maimai_baseline.physical.interval_store import IntervalPhysicalStore
    from maimai_baseline.physical.catalog import CompactPathCatalog
    from maimai_baseline.physical.stage_plan import prepare
    from maimai_baseline.data.batch import prepare_audio_bounds,prepare_batch
    from maimai_baseline.data.song_chart import SongChartCache
    from maimai_baseline.data.song_audio import SongAudioCache
    from maimai_baseline.time.style import style_geometry
    from maimai_baseline.model.common import BaselineModelConfig
    from maimai_baseline.contract import BASE_MODEL_ARCHITECTURE, TIME_HEAD
    from maimai_baseline.training.initialization import _initialize_model
    from maimai_baseline.training.steps import _move,_autocast,_loss_step,_optimizer
    from maimai_baseline.source_identity import baseline_source_identity
    import maimai_baseline.physical.stage_loss as stage_loss
    torch.set_num_threads(1)
    seed=20260811;root=args.batch_dir
    saved=torch.load(root/'model-inputs.pt',weights_only=False,map_location='cpu');options=saved['args'];v=saved['vocabulary']
    catalog=CompactPathCatalog.load(Path(options['physical_path_catalog']),v)
    store=IntervalPhysicalStore(args.interval_cache,catalog=catalog,vocabulary=v)
    song=SongChartCache(options['song_chart_cache']);audio=SongAudioCache(options['song_audio_cache'])
    config=BaselineModelConfig(architecture=BASE_MODEL_ARCHITECTURE,chart_style=options['chart_style'],
        route_head=options['route_head'],audio_cross_attention=options['audio_cross_attention'],
        audio_attention_preset=options['audio_attention_preset'],encoder_capacity_preset=options['encoder_capacity_preset'],
        decoder_rope_coordinate=options['decoder_rope_coordinate'],time_head=TIME_HEAD,time_components=5,
        decoder_layers=4,decoder_d_model=256,decoder_heads=8,decoder_feedforward=1024)
    # Use the maintained config's exact time-head spelling.
    from maimai_baseline.contract import TIME_HEAD
    from dataclasses import replace
    config=replace(config,time_head=TIME_HEAD)
    report=dict(schema='maimai.row_stages_cuda_cost.v1',source=baseline_source_identity(),
        scope='pilot original batches; full forward+L3+Physical+KL+backward+clip+AdamW',workers=12,
        warmup=args.warmup,measured=args.steps,device=torch.cuda.get_device_name(),torch=torch.__version__,checks=[],
        selected_song_inputs='reconstructed from existing pilot canonical note-view rows; no old payload changed',
        interval_manifest=store.manifest['manifest_identity_sha256'])
    native=stage_loss.compute;events=[]
    def timed(*a,**kw):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record();out=native(*a,**kw);end.record();events.append((start,end));return out
    stage_loss.compute=timed
    selection=json.loads((root/'selection.json').read_text())
    report['plan_sha256']=selection['plan_sha256']
    try:
        for case in selection['cases']:
            package=torch.load(root/(case['name']+'.pt'),weights_only=False,map_location='cpu')
            raw=package['batch'];items=package['items']
            raw={k:x for k,x in raw.items() if not k.startswith('physical_') or k in ('physical_input_source_row_ids','physical_target_source_row_ids','physical_chart_indices','physical_view_indices')}
            assert list(raw['input_ids']['event'].shape)==case['shape']
            keys=[('train',str(s).rsplit(':',1)[0],int(view)) for s,view in zip(raw['segment_ids'],raw['physical_view_indices'])]
            unique=list(dict.fromkeys(keys))
            loader=DataLoader(SelectedCharts(unique,song.lookup,options['note_view_manifest'],v),batch_size=1,
                collate_fn=single,num_workers=12,worker_init_fn=worker_init)
            songs=list(loader)
            starts,ends=raw['start_tick'],raw['end_tick']
            boundaries=torch.tensor([song.lookup[k]['song_end_tick'] for k in keys])
            raw['global_style_inputs']=dict(song_audio=audio.batch([song.lookup[k]['variant'] for k in keys]),songs=songs,
                window_groups=torch.tensor([unique.index(k) for k in keys]),window_blocks=(starts/384).long(),
                window_valid=style_geometry(starts,ends,boundaries)[1],coverage=1.)
            raw=prepare_audio_bounds(raw,items,raw['sft_timings'],v)
            raw=prepare_batch(raw,raw['sft_timings'])
            costs=[]
            for _ in range(2):
                constants=prepare(raw,store,v);costs.append(constants['physical_interval_cost'])
            raw.update(constants)
            batch=_move(raw,torch.device('cuda'))
            torch.manual_seed(seed);model,_=_initialize_model(v,config,seed=seed,artifacts=saved['artifacts']);model=model.cuda().train()
            optimizer=_optimizer(model,learning_rate=.000565685424949238)
            times=[];prob=[];changed=None
            weight=model.row_fields.adapters['route'].mlp[0].weight
            before=weight.detach().clone()
            for step in range(args.warmup+args.steps):
                optimizer.zero_grad(set_to_none=True);events.clear()
                if step==args.warmup:torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize();started=time.perf_counter()
                with _autocast(torch.device('cuda')):
                    output,loss=_loss_step(model,batch,v,collect_metrics=False,warmup_coefficient=1.,style_progress=1.)
                loss.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();torch.cuda.synchronize()
                if step>=args.warmup:
                    times.append(time.perf_counter()-started);prob.append(sum(a.elapsed_time(b) for a,b in events)/1000)
                del output,loss
            changed=not torch.equal(before,weight.detach())
            row=dict(name=case['name'],batch=case['batch'],shape=case['shape'],input_prepare=costs,
                seconds=times,mean_seconds=statistics.mean(times),physical_probability_seconds=statistics.mean(prob),
                peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
                parameters=model.parameter_count(),route_adapter_updated=changed)
            report['checks'].append(row);print(json.dumps(row),flush=True)
            del model,optimizer,batch,raw,songs,constants,weight,before
            import gc
            gc.collect();torch.cuda.empty_cache()
        report['completed']=True
    except Exception as e:
        report.update(completed=False,error=repr(e));raise
    finally:
        args.output.write_text(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('batch-dir','interval-cache','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--warmup',type=int,default=10);p.add_argument('--steps',type=int,default=30)
    options=p.parse_args()
    if options.output.exists():raise ValueError('choose a new output report')
    run(options)
