"""Explicit v2-to-v3 stage payload repack; no Physical/geometry computation."""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from maimai_baseline.physical.shared_cache import CACHE_IDENTITY
from maimai_baseline.source_identity import baseline_source_identity
from maimai_dataset.manifest import canonical_json_sha256
from maimai_dataset.integrity import file_sha256


def repack(source,output):
    source,output=Path(source),Path(output)
    manifest=json.loads((source/'manifest.json').read_text('utf-8'))
    expected=dict(CACHE_IDENTITY,schema='maimai.shared_physical_moments.v2',representation='causal-stage-results-v2')
    payload=dict(manifest);identity=payload.pop('manifest_identity_sha256')
    if manifest['contract']!=expected or canonical_json_sha256(payload)!=identity:
        raise ValueError('repack requires a valid causal-stage v2 manifest')
    output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();views=[];count=0
    for view in manifest['views']:
        folder=output/view['path'];folder.mkdir()
        data=json.loads((source/view['path']/'chart.json').read_text('utf-8'))
        if data['contract']!=expected:raise ValueError('chart semantics mismatch')
        arrays=[];offset=0;entries=[]
        with np.load(source/view['path']/'supports.npz',allow_pickle=False) as old:
            for table in data['tables']:
                if len(table['blocks'])>=4 and table['entries']:raise ValueError('unexpected large-chord table')
                for queries in table['entries'].values():
                    for key,e in queries.items():
                        a=old[e.pop('array')]
                        if a.dtype!=np.bool_ or a.shape[0]!=4:raise ValueError('stage dtype/shape mismatch')
                        stage=json.loads(key)[0]
                        if stage=='route' and (a.ndim!=2 or a.shape[1] not in (1,70,71)):raise ValueError('expanded route payload')
                        if stage=='duration' and a.shape!=(4,1):raise ValueError('expanded duration payload')
                        e.update(offset=offset,shape=list(a.shape));offset+=a.size
                        arrays.append(a.reshape(-1));entries.append(e);count+=1
        merged=np.concatenate(arrays) if arrays else np.empty(0,dtype=bool)
        np.save(folder/'supports.npy',merged,allow_pickle=False)
        # Check only the newly written payload against the actual source arrays.
        check=np.load(folder/'supports.npy',mmap_mode='r',allow_pickle=False)
        if not np.array_equal(check,merged):raise ValueError('repacked payload mismatch')
        check._mmap.close()
        data['contract']=CACHE_IDENTITY
        (folder/'chart.json').write_text(json.dumps(data,separators=(',',':')),encoding='utf-8')
        record=dict(view)
        record['files']=[dict(path=folder.name+'/'+n,sha256=file_sha256(folder/n)) for n in ('chart.json','supports.npy')]
        record['bytes']=sum((folder/n).stat().st_size for n in ('chart.json','supports.npy'))
        views.append(record)
    result=dict(payload,schema=CACHE_IDENTITY['schema'],contract=CACHE_IDENTITY,views=views,
        repack=dict(source_manifest=identity,producer=baseline_source_identity(),seconds=time.perf_counter()-started,
                    entries=count,verified=True,source_bytes=sum(v['bytes'] for v in manifest['views']),
                    bytes=sum(v['bytes'] for v in views)))
    result['manifest_identity_sha256']=canonical_json_sha256(result)
    (output/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result['repack'].items() if k!='producer'}))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();repack(a.source,a.output)
