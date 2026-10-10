"""Isolated fixed-trajectory SFT comparison; never changes an existing run.

Invoke with the JSON argv captured from the existing SFT process and a NEW
report path. Select the GPU externally using CUDA_VISIBLE_DEVICES.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time
from functools import partial
import numpy as np
import torch
from torch.utils.data import DataLoader

from maimai_baseline.cli import parser
from maimai_baseline.checkpoint import load_baseline_checkpoint
from maimai_baseline.training import sft, sft_batch, sft_rollout
from maimai_baseline.training.steps import _move, _autocast, _loss_step
from maimai_baseline.training.inputs import _prepare_inputs, _dataset
from maimai_baseline.physical.data import collate_physical_segments
from maimai_baseline.physical.resident import bind_bitplane_store, active_device_cache
from maimai_baseline.physical.checker import load_cuda_extension
from maimai_baseline.loss.supervision import from_args
from maimai_baseline.source_identity import baseline_source_identity


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument('--training-args-json',required=True)
    cli.add_argument('--report',required=True)
    cli.add_argument('--smoke-only',action='store_true')
    opts=cli.parse_args()
    report_path=Path(opts.report)
    if report_path.exists(): raise ValueError('report already exists')
    args=parser().parse_args(json.loads(Path(opts.training_args_json).read_text()))
    device=torch.device('cuda')
    model,payload=load_baseline_checkpoint(Path(args.init_checkpoint),route_artifact=Path(args.route_artifact),
        device=device,strict_source=False,verify_artifact_payloads=False)
    args.encoder_capacity_preset=model.config.encoder_capacity_preset
    args.audio_attention_preset=model.config.audio_attention.preset
    args.decoder_rope_coordinate=model.config.decoder_rope_coordinate
    inputs=_prepare_inputs(args)
    if payload['profile_fingerprint']!=inputs.profile_summary['profile_fingerprint'] or payload['bpm_input']!=inputs.manifest['bpm_input']:
        raise ValueError('parent/input mismatch')
    dataset=_dataset(inputs,'train')
    sampler=sft.MasterSampler(dataset,3072)
    loader=DataLoader(dataset,batch_sampler=sampler, num_workers=0,
        collate_fn=partial(collate_physical_segments,vocabulary=inputs.vocabulary,
                          physical_store=inputs.physical_store,audio_loader=dataset.audio_item))
    chosen=[]
    for index,value in enumerate(loader):
        blocks,eligible=sft_rollout.choose_blocks(value,model.vocabulary,epoch=0,fraction=.25)
        print(json.dumps(dict(event='scan',batch=index,selected=len(blocks))),flush=True)
        if index==0: chosen.append((index,value,blocks,eligible))
        if index>0 and (len(chosen)==1 or len(blocks)>len(chosen[1][2])):
            if len(chosen)==1: chosen.append((index,value,blocks,eligible))
            else: chosen[1]=(index,value,blocks,eligible)
        if index==15: break
    if len(chosen)>1 and len(chosen[0][2])>=len(chosen[1][2]): chosen=chosen[:1]
    bind_bitplane_store(inputs.physical_store)
    load_cuda_extension()
    active_device_cache(device)
    model.specialized_supervision=from_args(args)
    model.difficulty_loss_preset='uniform'
    model.train()
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    cfg=sft.sft_config(args)
    result=dict(source=baseline_source_identity()['source_sha256'],
        parent_sha256=hashlib.sha256(Path(args.init_checkpoint).read_bytes()).hexdigest(),
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,config=cfg,batches=[])
    if opts.smoke_only:
        optimizer=sft.optimizer_for(model,cfg)
        scheduler=sft.scheduler_for(optimizer,len(chosen))
        metrics,traces=sft.train_epoch(model,[entry[1] for entry in chosen],optimizer,scheduler,
            epoch=0,config=cfg,device=device)
        result.update(natural_training_loop=metrics,traces=traces,passed=metrics['steps']==len(chosen))
        report_path.write_text(json.dumps(result,indent=2)+'\n')
        return
    for index,cpu,blocks,eligible in chosen:
        model.load_state_dict(initial)
        batch=_move(cpu,device)
        prepared=_move(sft_batch.prepare(cpu,model.vocabulary,blocks),device)
        torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
        # Capture natural greedy choices once. Timed variants replay exactly
        # these events/pads/anchors while retaining their original loss graphs.
        with _autocast(device):
            outputs,loss=_loss_step(model,batch,model.vocabulary,collect_metrics=False)
            natural=sft_batch.rollout_batch(model,batch,outputs,prepared,max_notes=33,collect_trace=True)
        moment=model.vocabulary.event_id('MOMENT')
        note=model.vocabulary.event_id('NOTE')
        events=np.full((len(blocks),34),moment,dtype=np.int64)
        pads=np.zeros_like(events)
        for i,report in enumerate(natural['reports']):
            for j,row in enumerate(report['trace']):
                events[i,j]=note if row['chosen_event']=='NOTE' else moment
                pads[i,j]=row['pad_id']
        forced=dict(start_ticks=[r['generated_start_tick'] for r in natural['reports']],events=events.tolist(),pads=pads.tolist())
        record=dict(batch=index,selected=len(blocks),eligible=eligible,
            generated_notes=sum(r['generated_notes'] for r in natural['reports']),
            depth=natural['steps'],natural_counts={k:sum(r['reason']==k for r in natural['reports']) for k in ('natural','early','limit')},
            trajectory_sha256=hashlib.sha256(json.dumps(forced,sort_keys=True).encode()).hexdigest(),variants={})
        # Natural-path backward is also a CUDA gradient qualification.
        (natural['notes']+natural['events']+natural['terminal']).backward()
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise FloatingPointError('natural CUDA gradients are nonfinite')
        del outputs,loss,natural
        model.zero_grad(set_to_none=True)
        gc.collect();torch.cuda.empty_cache()
        for variant in ('serial','batched'):
            elapsed=[];peaks=[];losses=[]
            for repetition in range(4):
                model.load_state_dict(initial)
                optimizer=sft.optimizer_for(model,cfg)
                optimizer.zero_grad(set_to_none=True)
                torch.manual_seed(cfg['seed'])
                gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize();started=time.perf_counter()
                positions={}
                with _autocast(device):
                    out,base=_loss_step(model,dict(batch,l3_position_cache=positions),model.vocabulary,collect_metrics=False)
                    if variant=='serial':
                        trajectories=[sft_rollout.rollout_block(model,batch,out,p,max_notes=33,
                            forced=dict(start_tick=forced['start_ticks'][i],events=events[i],pads=pads[i])) for i,p in enumerate(blocks)]
                        objective=sft_rollout.replace_objective(base,positions,blocks,trajectories)
                    else:
                        trajectories=sft_batch.rollout_batch(model,batch,out,prepared,max_notes=33,forced=forced)
                        objective=sft_batch.replace(base,positions,prepared,trajectories)
                objective.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                torch.cuda.synchronize();seconds=time.perf_counter()-started
                peak=torch.cuda.max_memory_allocated()
                value=float(objective.loss.detach())
                if repetition:
                    elapsed.append(seconds);peaks.append(peak);losses.append(value)
                print(json.dumps(dict(event='measurement',batch=index,variant=variant,repetition=repetition,
                    seconds=seconds,peak_bytes=peak,loss=value)),flush=True)
                del out,base,trajectories,objective,positions,optimizer
                model.zero_grad(set_to_none=True)
            record['variants'][variant]=dict(seconds=elapsed,median_seconds=statistics.median(elapsed),peak_bytes=max(peaks),losses=losses)
        record['speedup']=record['variants']['serial']['median_seconds']/record['variants']['batched']['median_seconds']
        result['batches'].append(record)
        report_path.write_text(json.dumps(result,indent=2)+'\n')
        del batch,prepared
    result['passed']=all(r['speedup']>=5 and r['variants']['batched']['peak_bytes']<24*1024**3 for r in result['batches'])
    report_path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(event='completed',passed=result['passed'])),flush=True)


if __name__=='__main__': main()
