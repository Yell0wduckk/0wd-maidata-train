"""Stage-II updates, validation and complete-epoch orchestration."""
from itertools import islice
import json
from pathlib import Path
import time
import torch

from maidata.decoder.batching import collate_windows
from maidata.decoder.resources import load_route_distances
from maidata.decoder.types import Memory
from maidata.model import decoder_time_distributions
from maidata.training.data import epoch_iterator
from . import checkpoint, recipe
from .batching import window_chunks
from .data import make_loader
from .network import DecoderTrainer
from .plan import DecoderPlan
from .supervision import KEYS, prepare, loss, objective


class Metrics:
    def __init__(self):
        self.sums = dict(main=0.,physical=[0.]*4,judge=0.,trajectory=0.)
        self.counts = dict.fromkeys(KEYS,0)

    def add(self, terms):
        for k in KEYS:
            value = getattr(terms,k)
            total = value.total.detach().double().cpu().tolist()
            self.sums[k] = ([a+b for a,b in zip(self.sums[k],total)] if k == 'physical' else self.sums[k]+total)
            self.counts[k] += value.count

    def merge(self, other):
        for k in KEYS:
            self.sums[k] = ([a+b for a,b in zip(self.sums[k],other.sums[k])] if k == 'physical' else self.sums[k]+other.sums[k])
            self.counts[k] += other.counts[k]

    def report(self, config):
        means = {k: self.sums[k]/max(1,self.counts[k]) for k in KEYS if k != 'physical'}
        categories = [v/max(1,self.counts['physical']) for v in self.sums['physical']]
        means['physical'] = sum(v*w for v,w in zip(categories,(1,1,4,1)))/4
        total = means['main']+sum(means[k]*config['decoder_train_'+k+'_weight'] for k in KEYS[1:])
        return dict(loss=total,means=means,physical_categories=categories,counts=self.counts,sums=self.sums)


def prepared_chunks(batch, config):
    return [(indices,prepare([batch.windows[i] for i in indices]))
            for indices in window_chunks(batch.windows,config['decoder_train_token_budget'])]


def chunk_loss(model, batch, conditions, audio_kv, indices, prepared, config, geometry, *, diagnostics=None):
    device = next(model.decoder.parameters()).device
    index = torch.tensor(indices,device=device)
    examples = [batch.windows[i]['example'] for i in indices]
    # Build labels in one CPU collation; transfer aligned tensors together.
    from maidata.training.data import map_tensors
    tokens,types,_ = map_tensors(collate_windows(examples,device='cpu'),lambda t:t.to(device,non_blocking=True))
    memory = model.memory(conditions,audio_kv,index)
    output = model.decoder(tokens,types,memory)
    terms = loss(model.decoder.heads,output,prepared,distributions=decoder_time_distributions(config),
                 geometry=geometry,diagnostics=diagnostics)
    return terms,output


def update(model, batches, opt, sched, config, geometry, *, audit=False):
    model.train()
    prepared = [prepared_chunks(b,config) for b in batches]
    counts = {k:sum(p.counts[k] for chunks in prepared for _,p in chunks) for k in KEYS}
    before = {name:[p.detach().clone() for p in m.parameters()] for name,m in model.named_children()} if audit else {}
    opt.zero_grad(set_to_none=True)
    metrics, costs = Metrics(), []
    device = next(model.decoder.parameters()).device
    for raw,chunks in zip(batches,prepared):
        batch = raw.to(device)
        with recipe.autocast(device,config):
            conditions = model.produce(batch)
            bank = conditions.audio
            audio_kv = model.decoder.prepare_audio(Memory(bank.values,bank.beats,bank.valid))
        for j,(indices,labels) in enumerate(chunks):
            with recipe.autocast(device,config):
                terms,output = chunk_loss(model,batch,conditions,audio_kv,indices,labels,config,geometry)
                value = objective(terms,counts,config)
            if not torch.isfinite(value):
                raise ValueError('nonfinite Decoder objective')
            value.backward(retain_graph=j+1<len(chunks))
            metrics.add(terms)
            costs.append(len(indices)*max(len(raw.windows[i]['example'].inputs) for i in indices))
            del terms,output,value
        del batch,conditions,audio_kv
    norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(),config['decoder_train_gradient_clip'],error_if_nonfinite=True)
    gradients = {name:sum(float(p.grad.float().square().sum()) for p in m.parameters() if p.grad is not None)**.5
                 for name,m in model.named_children()} if audit else {}
    opt.step()
    sched.step()
    changed = {name:sum(not torch.equal(a,b) for a,b in zip(before[name],m.parameters()))
               for name,m in model.named_children()} if audit else {}
    if audit and (any(gradients[n] or changed[n] for n in ('encoder','draft','planner'))
                  or not gradients['decoder'] or not changed['decoder']):
        raise ValueError('stage II must update M only, with frozen E/D/P')
    return metrics,dict(gradient_norm=float(norm),gradients=gradients,changed_parameter_tensors=changed,
                        chunks=len(costs),padded_token_costs=costs)


@torch.no_grad()
def validate(model, loader, config, geometry, *, final, restored_rng=None):
    mode,rng = model.training,checkpoint.rng_state()
    model.eval()
    device = next(model.decoder.parameters()).device
    metrics,diagnostics = Metrics(),dict(field_accuracy={},windows=0,history_dropped=0)
    try:
        for raw in epoch_iterator(loader,restored_rng):
            chunks = prepared_chunks(raw,config)
            batch = raw.to(device)
            with recipe.autocast(device,config):
                conditions = model.produce(batch)
                b = conditions.audio
                audio_kv = model.decoder.prepare_audio(Memory(b.values,b.beats,b.valid))
                for indices,labels in chunks:
                    terms,output = chunk_loss(model,batch,conditions,audio_kv,indices,labels,config,geometry,
                        diagnostics=diagnostics['field_accuracy'] if final else None)
                    metrics.add(terms)
            if final:
                diagnostics['windows'] += len(raw.windows)
                diagnostics['history_dropped'] += sum(w['example'].history_dropped for w in raw.windows)
        result = metrics.report(config)
        if not torch.isfinite(torch.tensor(result['loss'])):
            raise ValueError('nonfinite validation loss')
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
    binding,hashes = checkpoint.data_binding(plans,config)
    source = checkpoint.source_identity()
    loaders = [make_loader(p,config,hashes) for p in plans]
    steps = [len(plans[0].batches(e,config)) for e in range(1,config['decoder_train_scheduler_epochs']+1)]
    if not resume:
        recipe.seed(config)
    model = DecoderTrainer(config)
    producer = checkpoint.load_producer(model,config,binding)
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
        batches = list(islice(epoch_iterator(loaders[0]),config['decoder_train_gradient_accumulation_steps']))
        calls = dict.fromkeys(('encoder','draft','planner'),0)
        handles = []
        for key in calls:
            def count(module,args,name=key):
                calls[name] += 1
            handles.append(getattr(model,key).register_forward_pre_hook(count))
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        metrics,audit = update(model,batches,opt,sched,config,geometry,audit=True)
        torch.cuda.synchronize(device)
        report = dict(metrics=metrics.report(config),**audit,component_forward_calls=calls,
            seconds=time.perf_counter()-start,peak_allocated=torch.cuda.max_memory_allocated(device),
            peak_reserved=torch.cuda.max_memory_reserved(device),optimizer_step_in_epoch=check_step+1,
            audio_groups=[b.audio_variants for b in batches],source=source,binding=binding,producer=producer,config=config)
        for h in handles:
            h.remove()
        write_json(output.with_name(output.name+'-cuda-check.json'),report)
        return report
    if completed == config['decoder_train_stop_after_epochs']:
        return dict(completed_epochs=completed,step=step)
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/('resume.json' if resume else 'run.json'),dict(config=config,source=source,binding=binding,
        producer=producer,steps_by_epoch=steps,resume=str(resume) if resume else None,starting_completed_epochs=completed))
    for epoch in range(completed+1,config['decoder_train_stop_after_epochs']+1):
        start = time.perf_counter()
        loaders[0].batch_sampler.epoch = epoch
        iterator = epoch_iterator(loaders[0],restored[0])
        restored[0] = None
        metrics,updates,groups,windows = Metrics(),0,0,0
        while batches := list(islice(iterator,config['decoder_train_gradient_accumulation_steps'])):
            current,_ = update(model,batches,opt,sched,config,geometry)
            metrics.merge(current)
            updates += 1
            step += 1
            groups += sum(len(b.audio_variants) for b in batches)
            windows += sum(len(b.windows) for b in batches)
            print(json.dumps(dict(epoch=epoch,update=updates,updates=steps[epoch-1],step=step,
                                  loss=current.report(config)['loss'],lr=opt.param_groups[0]['lr'])),flush=True)
        expected = len(plans[0].occurrences(epoch))
        if updates != steps[epoch-1] or groups != len(plans[0].groups) or windows != expected:
            raise ValueError('incomplete occurrence epoch; checkpoint not saved')
        final = epoch == config['decoder_train_stop_after_epochs']
        result = dict(epoch=epoch,step=step,train=metrics.report(config),seconds=time.perf_counter()-start,
                      audio_groups=groups,windows=windows)
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
