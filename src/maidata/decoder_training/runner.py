"""Stage-II updates, validation and complete-epoch orchestration."""
from itertools import islice
import json
from pathlib import Path
import time
import torch

from maidata.decoder.resources import load_route_distances
from maidata.decoder.types import Memory
from maidata.training.data import epoch_iterator
from . import checkpoint, recipe
from .data import make_loader
from .network import DecoderTrainer
from .plan import DecoderPlan
from .frozen_cache import FrozenCache
from .window_cache import WindowCache
from .supervision import KEYS, loss, objective


class Metrics:
    def __init__(self):
        self.totals = None
        self.counts = dict.fromkeys(KEYS,0)

    def add(self, terms):
        values = torch.cat([getattr(terms,k).total.detach().double().reshape(-1) for k in KEYS])
        self.totals = values if self.totals is None else self.totals+values
        for k in KEYS:
            self.counts[k] += getattr(terms,k).count

    def merge(self, other):
        if other.totals is not None:
            self.totals = other.totals if self.totals is None else self.totals+other.totals
        for k in KEYS:
            self.counts[k] += other.counts[k]

    def report(self, config):
        values = [0.]*7 if self.totals is None else self.totals.cpu().tolist()
        sums = dict(main=values[0],physical=values[1:5],judge=values[5],trajectory=values[6])
        means = {k: sums[k]/max(1,self.counts[k]) for k in KEYS if k != 'physical'}
        categories = [v/max(1,self.counts['physical']) for v in sums['physical']]
        means['physical'] = sum(v*w for v,w in zip(categories,(1,1,4,1)))/4
        total = means['main']+sum(means[k]*config['decoder_train_'+k+'_weight'] for k in KEYS[1:])
        return dict(loss=total,means=means,physical_categories=categories,counts=self.counts,sums=sums)


def prepared_chunks(batch, config):
    return batch.chunks


def chunk_loss(model, conditions, audio_kv, chunk, config, geometry, *, diagnostics=None, time_diagnostics=None):
    memory = model.memory(conditions,audio_kv,chunk.indices,chunk.audio_layout)
    output = model.decoder(chunk.tokens,chunk.next_types,memory)
    terms = loss(model.decoder.heads,output,chunk.labels,distributions=model.distributions,
                 geometry=geometry,diagnostics=diagnostics,time_diagnostics=time_diagnostics)
    return terms,output


def update(model, batches, opt, sched, config, geometry, *, audit=False):
    model.train()
    prepared = [prepared_chunks(b,config) for b in batches]
    counts = {k:sum(p.labels.counts[k] for chunks in prepared for p in chunks) for k in KEYS}
    before = {name:[p.detach().clone() for p in m.parameters()] for name,m in model.named_children()} if audit else {}
    opt.zero_grad(set_to_none=True)
    metrics, costs = Metrics(), []
    device = next(model.decoder.parameters()).device
    for raw in batches:
        batch = raw.to(device)
        chunks = batch.chunks
        with recipe.autocast(device,config):
            conditions = model.conditions(batch)
            bank = conditions.audio
            audio_kv = model.decoder.prepare_audio(Memory(bank.values,bank.beats,bank.valid))
        for j,cpu_chunk in enumerate(chunks):
            chunk = cpu_chunk.to(device)
            with recipe.autocast(device,config):
                terms,output = chunk_loss(model,conditions,audio_kv,chunk,config,geometry)
                value = objective(terms,counts,config)
            value.backward(retain_graph=j+1<len(chunks))
            metrics.add(terms)
            costs.append(chunk.cost)
            del terms,output,value,chunk
        del batch,conditions,audio_kv
    norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(),config['decoder_train_gradient_clip'])
    gradients = {name:sum(float(p.grad.float().square().sum()) for p in m.parameters() if p.grad is not None)**.5
                 for name,m in model.named_children()} if audit else {}
    opt.step()
    sched.step()
    changed = {name:sum(not torch.equal(a,b) for a,b in zip(before[name],m.parameters()))
               for name,m in model.named_children()} if audit else {}
    if audit and (set(changed) != {'decoder'} or not gradients['decoder'] or not changed['decoder']):
        raise ValueError('offline stage II must construct and update M only')
    return metrics,dict(gradient_norm=float(norm),gradients=gradients,changed_parameter_tensors=changed,
                        chunks=len(costs),padded_token_costs=costs,
                        batch_costs=[b.budget_cost for b in batches])


@torch.no_grad()
def validate(model, loader, config, geometry, *, final, restored_rng=None):
    mode,rng = model.training,checkpoint.rng_state()
    model.eval()
    device = next(model.decoder.parameters()).device
    metrics,diagnostics = Metrics(),dict(field_accuracy={},time_target_probability={},windows=0,history_dropped=0)
    try:
        for raw in epoch_iterator(loader,restored_rng):
            prepared_chunks(raw,config)
            batch = raw.to(device)
            with recipe.autocast(device,config):
                conditions = model.conditions(batch)
                b = conditions.audio
                audio_kv = model.decoder.prepare_audio(Memory(b.values,b.beats,b.valid))
                for cpu_chunk in batch.chunks:
                    chunk = cpu_chunk.to(device)
                    terms,output = chunk_loss(model,conditions,audio_kv,chunk,config,geometry,
                        diagnostics=diagnostics['field_accuracy'] if final else None,
                        time_diagnostics=diagnostics['time_target_probability'] if final else None)
                    metrics.add(terms)
                    del terms,output,chunk
            if final:
                diagnostics['windows'] += len(raw.windows)
                diagnostics['history_dropped'] += sum(w['history_dropped'] for w in raw.windows)
        result = metrics.report(config)
        if final:
            result['diagnostics'] = diagnostics
        return result
    finally:
        model.train(mode)
        checkpoint.restore_rng(rng)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def run(config, *, resume=None, cuda_check=False, check_step=0):
    recipe.validate(config)
    device = torch.device(config['decoder_train_device'])
    if device.type == 'cuda' and (not torch.cuda.is_available() or
            config['decoder_train_precision']=='bf16' and not torch.cuda.is_bf16_supported()):
        raise ValueError('requested CUDA/bf16 runtime is unavailable')
    if cuda_check and (device.type != 'cuda' or resume):
        raise ValueError('CUDA check requires a separate fresh stage-II CUDA model')
    output = Path(config['decoder_train_output'])
    if not resume and not cuda_check and output.exists() and any(output.iterdir()):
        raise FileExistsError('fresh Decoder output must be empty')
    plans = [DecoderPlan(config['decoder_train_'+key],config,training=i==0)
             for i,key in enumerate(('train_plan','validation_plan'))]
    binding,hashes = checkpoint.data_provenance(config),{}
    cache = FrozenCache(plans,config,binding)
    windows = WindowCache(plans,config,binding)
    producer = checkpoint.cache_producer(cache.producer,config,binding)
    binding['frozen_conditions'] = cache.binding
    binding['window_tensors'] = windows.binding
    source = checkpoint.source_identity()
    loaders = [make_loader(p,config,hashes,cache,windows) for p in plans]
    steps = [len(plans[0].batches(e,config)) for e in range(1,config['decoder_train_scheduler_epochs']+1)]
    if not resume:
        recipe.seed(config)
    model = DecoderTrainer(config)
    model.to(device)
    geometry = load_route_distances(config['decoder_train_route_distances']).to(device)
    opt = recipe.optimizer(model.decoder.parameters(),config)
    sched = recipe.scheduler(opt,steps,config)
    completed,step,best,restored = 0,0,None,[None,None]
    if resume:
        payload = torch.load(resume,map_location='cpu',weights_only=False)
        completed,step,best = checkpoint.restore(payload,model,opt,sched,config=config,binding=binding,
            source=source,producer=producer,steps=steps,loaders=loaders)
        restored = [r if started else None for r,started in zip(payload['loader_rng'],payload['loader_started'])]
        del payload
    if cuda_check:
        if not 0 <= check_step < steps[0]:
            raise ValueError('CUDA check step outside first epoch')
        loaders[0].batch_sampler.epoch = 1
        loaders[0].batch_sampler.selected_step = check_step
        data_start = time.perf_counter()
        batches = list(islice(epoch_iterator(loaders[0]),config['decoder_train_gradient_accumulation_steps']))
        data_seconds = time.perf_counter()-data_start
        calls = dict.fromkeys(('encoder','draft','planner','decoder'),0)
        handles = []
        for key in dict(model.named_children()):
            def count(module,args,name=key):
                calls[name] += 1
            handles.append(getattr(model,key).register_forward_pre_hook(count))
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        metrics,audit = update(model,batches,opt,sched,config,geometry,audit=True)
        torch.cuda.synchronize(device)
        report = dict(metrics=metrics.report(config),**audit,component_forward_calls=calls,
            conditioning='offline_cache',constructed_components=list(dict(model.named_children())),
            data_wait_seconds=data_seconds,
            seconds=time.perf_counter()-start,peak_allocated=torch.cuda.max_memory_allocated(device),
            peak_reserved=torch.cuda.max_memory_reserved(device),optimizer_step_in_epoch=check_step+1,
            audio_groups=[b.audio_variants for b in batches],source=source,binding=binding,producer=producer,config=config)
        for h in handles:
            h.remove()
        write_json(output.with_name(output.name+'-cuda-check.json'),report)
        return report
    if completed >= config['decoder_train_stop_after_epochs']:
        return dict(completed_epochs=completed,step=step)
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/('resume.json' if resume else 'run.json'),dict(config=config,source=source,binding=binding,
        producer=producer,steps_by_epoch=steps,resume=str(resume) if resume else None,starting_completed_epochs=completed))
    for epoch in range(completed+1,config['decoder_train_stop_after_epochs']+1):
        start = time.perf_counter()
        loaders[0].batch_sampler.epoch = epoch
        iterator = epoch_iterator(loaders[0],restored[0])
        restored[0] = None
        metrics,updates,group_instances = Metrics(),0,0
        groups, identities = set(), []
        while True:
            data_start = time.perf_counter()
            batches = list(islice(iterator,config['decoder_train_gradient_accumulation_steps']))
            data_seconds = time.perf_counter()-data_start
            if not batches:
                break
            update_start = time.perf_counter()
            current,_ = update(model,batches,opt,sched,config,geometry)
            update_seconds = time.perf_counter()-update_start
            metrics.merge(current)
            updates += 1
            step += 1
            group_instances += sum(len(b.audio_variants) for b in batches)
            groups.update(v for b in batches for v in b.audio_variants)
            identities.extend(w['identity'] for b in batches for w in b.windows)
            print(json.dumps(dict(epoch=epoch,update=updates,updates=steps[epoch-1],step=step,
                                  loss=current.report(config)['loss'],lr=opt.param_groups[0]['lr'],
                                  data_wait_seconds=data_seconds,update_seconds=update_seconds,
                                  batch_costs=[b.budget_cost for b in batches])),flush=True)
        final = epoch == config['decoder_train_stop_after_epochs']
        result = dict(epoch=epoch,step=step,train=metrics.report(config),seconds=time.perf_counter()-start,
                      audio_groups=len(groups),audio_group_instances=group_instances,windows=len(identities))
        improved = False
        if epoch % 5 == 0 or final:
            result['validation'] = validate(model,loaders[1],config,geometry,final=final,restored_rng=restored[1])
            restored[1] = None
            value = result['validation']['loss']
            improved = best is None or value < best
            best = value if improved else best
        payload = checkpoint.capture(model,opt,sched,epoch=epoch,step=step,config=config,binding=binding,source=source,
            producer=producer,steps=steps,loaders=loaders,best=best,metrics=result)
        checkpoint.atomic_save(output/'checkpoints/last.ckpt',payload)
        if improved:
            checkpoint.atomic_save(output/'checkpoints/best.ckpt',payload)
        if epoch%5==0 or final:
            checkpoint.atomic_save(output/f'checkpoints/epoch-{epoch:03d}.ckpt',payload)
        with (output/'metrics.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n')
        print(json.dumps(dict(completed_epoch=epoch,step=step,validation_loss=result.get('validation',{}).get('loss'))),flush=True)
    return dict(completed_epochs=epoch,step=step)
