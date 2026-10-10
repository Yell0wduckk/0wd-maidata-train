"""Export/compare actual original-batch stage constants in isolated interpreters."""
import argparse,json,sys,time
from pathlib import Path
import torch


def run(a):
    sys.path.insert(0,str(a.source_root/'src'))
    from maimai_baseline.physical.interval_store import IntervalPhysicalStore
    from maimai_baseline.physical.catalog import CompactPathCatalog
    from maimai_baseline.physical.stage_plan import prepare
    from maimai_baseline.data.batch import prepare_audio_bounds,prepare_batch
    torch.set_num_threads(1)
    saved=torch.load(a.batch_dir/'model-inputs.pt',weights_only=False,map_location='cpu')
    v=saved['vocabulary'];catalog=CompactPathCatalog.load(Path(saved['args']['physical_path_catalog']),v)
    store=IntervalPhysicalStore(a.interval_cache,catalog=catalog,vocabulary=v)
    a.output.mkdir(exist_ok=False,parents=True)
    checks=[]
    for case in json.loads((a.batch_dir/'selection.json').read_text())['cases']:
        p=torch.load(a.batch_dir/(case['name']+'.pt'),weights_only=False,map_location='cpu')
        raw=p['batch'];raw=prepare_batch(prepare_audio_bounds(raw,p['items'],raw['sft_timings'],v),raw['sft_timings'])
        costs=[]
        for _ in range(2):
            result=prepare(raw,store,v);costs.append(result.pop('physical_interval_cost'))
        if a.reference:
            old=torch.load(a.reference/(case['name']+'.pt'),weights_only=False)
            def same(x,y,path=''):
                if isinstance(x,dict):
                    assert list(x)==list(y),path
                    for k in x:same(x[k],y[k],path+'/'+k)
                elif torch.is_tensor(x):
                    assert x.dtype==y.dtype and x.shape==y.shape,(path,x.shape,y.shape)
                    torch.testing.assert_close(x,y,rtol=0,atol=0,msg=path)
                else:assert x==y,path
            same(result,old)
        torch.save(result,a.output/(case['name']+'.pt'))
        checks.append(dict(name=case['name'],costs=costs,equal=True if a.reference else None))
        print(json.dumps(checks[-1]),flush=True)
    (a.output/'report.json').write_text(json.dumps(checks,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','batch-dir','interval-cache','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--reference',type=Path);run(p.parse_args())
