"""Export frozen D/P outputs from existing P0, without Encoder or training."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Dataset

from maidata.entry import CONFIG
from maidata.chart_cache.cache import load_chart
from maidata.draft.conditions import read_condition, make_draft_inputs
from maidata.draft.layout import prepare_selection
from maidata.draft.types import BeatMemory
from maidata.draft.windows import SongCoverage, select_windows
from maidata.training.data import worker_init
from maidata.training.transport import pack_batch
from maidata.training.checkpoint import digest_file
from .checkpoint import load_producer
from .producer import FrozenProducer
from .recipe import autocast
from .runner import write_json
from .cache_schema import P0_SCHEMA, DP_SCHEMA as SCHEMA


class CachedP0Dataset(Dataset):
    def __init__(self, config, p0_root, p0_manifest, plans):
        self.config, self.root, self.p0 = config, Path(p0_root), p0_manifest
        self.groups, self.charts, self.windows = {}, {}, defaultdict(list)
        for plan in plans:
            self.groups.update({g['audio_variant']: g for g in plan['audio_groups']})
            self.charts.update({c['chart_id']: c for c in plan['charts']})
            path = Path(config['decoder_train_'+('train_plan' if plan['policy']['split']=='train' else 'validation_plan')])
            for line in (path/plan['files']['windows']).read_text('utf-8').splitlines():
                row = json.loads(line)
                self.windows[row['audio_variant']].append(row)
        self.variants = list(self.groups)

    def __len__(self):
        return len(self.variants)

    def __getitem__(self, index):
        variant = self.variants[index]
        meta, row = self.groups[variant], self.p0['groups'][variant]
        payload = torch.load(self.root/row['path'], map_location='cpu', weights_only=True)
        if (payload['schema'] != P0_SCHEMA or payload['audio_variant'] != variant
                or payload['producer_sha256'] != self.p0['producer']['sha256']
                or payload['coverage'] != row['coverage']):
            raise ValueError('cached P0 identity differs')
        coverage = SongCoverage(**payload['coverage'])
        if (list(coverage.starts(include_tail=False)) != meta['encoder_starts_beats']
                or coverage.first_beat+coverage.draft_beats != meta['draft_end_beat']):
            raise ValueError('cached P0 and plan coverage differ')
        bank = BeatMemory(*(payload[k].unsqueeze(0) for k in ('values','beats','valid')))
        keys, conditions = meta['chart_ids'], []
        for key in keys:
            record = load_chart(Path(self.config['decoder_train_chart_cache'])/self.charts[key]['chart_cache_path'])
            source = dict(record['source'])
            if source['chart_id'] != key or source['audio_variant'] != variant:
                raise ValueError('chart condition ownership differs')
            source['source_path'] = str(Path(self.config['decoder_train_source_root'])/source['song_id']/'maidata.txt')
            conditions.append(read_condition(source))
        inputs, mapping = make_draft_inputs(conditions, [variant], [coverage], device='cpu')
        inputs.valid_indices = inputs.draft_valid.flatten().nonzero().flatten()
        ownership = dict(zip(keys, mapping.tolist()))
        windows = self.windows[variant]
        expected = {(key,start) for key in keys for start in coverage.starts(include_tail=True)}
        if {(w['chart_id'],w['start_beat']) for w in windows} != expected or len(windows) != len(expected):
            raise ValueError('D/P export requires every unique base and tail window')
        groups = torch.tensor([ownership[w['chart_id']] for w in windows],dtype=torch.long)
        starts = torch.tensor([w['start_beat'] for w in windows],dtype=torch.float32)
        valid = torch.ones(len(windows),dtype=torch.bool)
        layout = prepare_selection(inputs.query_beats, inputs.draft_valid, groups, starts, valid,
            target_beats=self.config['target_beats'],before_beats=0,after_beats=0)
        return pack_batch(dict(variant=variant,bank=bank,inputs=inputs,groups=groups,starts=starts,
            valid=valid,layout=layout,chart_to_group=ownership,conditions=[asdict(c) for c in conditions],
            window_ids=[w['window_id'] for w in windows],chart_ids=[w['chart_id'] for w in windows]))


def save_outputs(path, batch, z, plan, producer_sha):
    payload = dict(schema=SCHEMA,audio_variant=batch['variant'],producer_sha256=producer_sha,
        chart_to_group=batch['chart_to_group'],conditions=batch['conditions'],
        z={k:getattr(z,k).detach().cpu() for k in ('values','beats','valid')},
        planner=dict(window_ids=batch['window_ids'],chart_ids=batch['chart_ids'],
            groups=batch['groups'].cpu(),starts=batch['starts'].cpu(),
            plan_values=plan.plan_values.detach().cpu(),valid=plan.valid.cpu()))
    if (not torch.isfinite(payload['z']['values']).all()
            or not torch.isfinite(payload['planner']['plan_values']).all()
            or payload['planner']['plan_values'].shape != (len(batch['window_ids']),12)
            or not payload['planner']['valid'].all()):
        raise ValueError('invalid frozen D/P output')
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(payload,temporary)
    temporary.replace(path)
    return dict(path='banks/'+path.name,sha256=digest_file(path),bytes=path.stat().st_size,
        z_shape=list(z.values.shape),z_dtype=str(z.values.dtype),
        plan_shape=list(plan.plan_values.shape),plan_dtype=str(plan.plan_values.dtype),
        chart_to_group=batch['chart_to_group'],windows=len(batch['window_ids']))


@torch.no_grad()
def export(config, p0_root, output):
    output, p0_root = Path(output), Path(p0_root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('D/P output must be new or empty')
    device = torch.device(config['decoder_train_device'])
    if device.type != 'cuda' or config['decoder_train_precision'] != 'bf16' or not torch.cuda.is_bf16_supported():
        raise ValueError('export requires CUDA bf16')
    p0 = json.loads((p0_root/'manifest.json').read_text())
    if p0['schema'] != P0_SCHEMA or not p0['complete'] or p0['precision'] != config['decoder_train_precision']:
        raise ValueError('P0 cache is incomplete or incompatible')
    plans = [json.loads((Path(config['decoder_train_'+side+'_plan'])/'plan.json').read_text())
             for side in ('train','validation')]
    splits = {side:[g['audio_variant'] for g in plan['audio_groups']]
              for side,plan in zip(('train','validation'),plans)}
    if splits != p0['splits']:
        raise ValueError('P0 split binding differs')
    binding = {key:{name:digest_file(Path(config[key.replace('pretrain_','decoder_train_')])/name)
                   for name in ('catalog.json','production-summary.json')}
               for key in ('pretrain_chart_cache','pretrain_audio_cache')}
    model = FrozenProducer(config)
    producer = load_producer(model,config,binding)
    if producer != p0['producer'] or binding != p0['normalization'] or p0['p0_rate'] != config['fpn_rates'][0]:
        raise ValueError('P0 producer or normalization differs')
    draft, planner = model.draft.to(device).eval(), model.planner.to(device).eval()
    # Encoder and Decoder stay on CPU and are never forwarded.
    dataset = CachedP0Dataset(config,p0_root,p0,plans)
    workers = config['decoder_train_workers']
    loader = DataLoader(dataset,batch_size=None,num_workers=workers,persistent_workers=workers>0,
        worker_init_fn=worker_init,pin_memory=True,
        generator=torch.Generator().manual_seed(config['decoder_train_seed']))
    output.mkdir(parents=True,exist_ok=True)
    report = dict(schema=SCHEMA,complete=False,producer=producer,normalization=binding,
        p0_manifest_sha256=digest_file(p0_root/'manifest.json'),p0_root=str(p0_root),
        precision=config['decoder_train_precision'],workers=workers,splits=splits,groups={},
        plans={side:{name:digest_file(Path(config['decoder_train_'+side+'_plan'])/name)
                     for name in ('plan.json',plan['files']['windows'])}
               for side,plan in zip(('train','validation'),plans)},
        source=config.get('_export_source_commit'),encoder_forward_calls=0,decoder_forward_calls=0)
    write_json(output/'manifest.json',report)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for raw in loader:
        batch = raw.to(device)
        with autocast(device,config):
            z = draft(batch['bank'],batch['inputs'])
            target = select_windows(z,batch['groups'],batch['starts'],batch['valid'],
                target_beats=config['target_beats'],before_beats=0,after_beats=0,layout=batch['layout'])
            plan = planner(target)
        path = output/'banks'/f'{batch["variant"]}.pt'
        row = save_outputs(path,batch,z,plan,producer['sha256'])
        if not report['groups']:
            saved = torch.load(path,map_location='cpu',weights_only=True)
            assert all(torch.equal(saved['z'][k],getattr(z,k).cpu()) for k in ('values','beats','valid'))
            assert torch.equal(saved['planner']['plan_values'],plan.plan_values.cpu())
            report['first_bank_roundtrip_exact'] = True
        report['groups'][batch['variant']] = row
        write_json(output/'manifest.json',report)
        print(json.dumps(dict(completed=len(report['groups']),total=len(dataset),
            variant=batch['variant'],seconds=time.perf_counter()-started)),flush=True)
        del batch,z,target,plan
    torch.cuda.synchronize(device)
    report.update(complete=True,seconds=time.perf_counter()-started,
        peak_allocated=torch.cuda.max_memory_allocated(device),peak_reserved=torch.cuda.max_memory_reserved(device),
        bytes=sum(r['bytes'] for r in report['groups'].values()),
        charts=sum(len(r['chart_to_group']) for r in report['groups'].values()),
        windows=sum(r['windows'] for r in report['groups'].values()))
    write_json(output/'manifest.json',report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--p0-cache',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    config = dict(CONFIG)
    values = json.loads(args.config.read_text(encoding='utf-8-sig'))
    if set(values)-set(config):
        parser.error('unknown config keys')
    config.update(values)
    marker = Path(__file__).resolve().parents[3]/'SOURCE_COMMIT'
    config['_export_source_commit'] = marker.read_text().strip() if marker.exists() else None
    export(config,args.p0_cache,args.output)


if __name__ == '__main__':
    main()
