"""List every valid recorded query/head/key probability strictly over 0.30."""
import argparse
import csv
import importlib.util
import json
from collections import Counter
from pathlib import Path
import zipfile
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    root = parser.parse_args().root
    index = json.loads((root/'index.json').read_text(encoding='utf8'))
    journal = json.loads((root/'journal.json').read_text(encoding='utf8'))
    report = json.loads((root/'aggregation.json').read_text(encoding='utf8'))
    spec = importlib.util.spec_from_file_location('aggregate', Path(__file__).with_name('aggregate-dense-attention.py'))
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    seconds = lambda x: module.seconds(np.asarray(x), report['timing'])
    accepted = {x['query_node_id']:x for x in journal['accepted_predictions']}
    final = set(journal['final_history_node_ids'])
    nodes = {x['id']:x for x in journal['nodes']}
    layers = [x['label'] for x in report['streams']]
    data = []
    fields = ['layer','kind','head','probability','query_seconds','key_seconds','query_beat','key_beat',
              'scope','query_node_id','key_node_id','input_event','predicted_event','key_event',
              'record_path','batch_index','query_index','key_index']
    counts,default_counts = Counter(),Counter()
    scopes = ['audio_global','audio_target','audio_halo','final_prediction','retained_control','discarded_attempt']
    defaults = {'audio_global','audio_target','final_prediction'}
    with (root/'attention-over-30-all.csv').open('w',encoding='utf-8-sig',newline='') as all_file, \
         (root/'attention-over-30-final.csv').open('w',encoding='utf-8-sig',newline='') as final_file, \
         zipfile.ZipFile(root/'raw-attention.zip') as archive:
        all_writer,final_writer = csv.DictWriter(all_file,fieldnames=fields),csv.DictWriter(final_file,fieldnames=fields)
        all_writer.writeheader();final_writer.writeheader()

        def emit(p, label, kind, qb, kb, valid, record, arrays, decoder=False):
            # Promote before comparing: do not round 0.30 to float16.
            hits = np.argwhere((p.astype(np.float32) > .30) & valid[:,None,:,None])
            qtime,ktime = seconds(qb),seconds(kb)
            for b,h,q,k in hits:
                qnode = int(arrays['query_node_ids'][q]) if decoder else -1
                knode = int(arrays['key_node_ids'][k]) if kind=='decoder_self' else -1
                if decoder:
                    scope = 'final_prediction' if qnode in accepted else 'retained_control' if qnode in final else 'discarded_attempt'
                else:
                    scope = 'audio_global' if kind=='audio_global' else 'audio_target' if arrays['canonical_query'][b,q] else 'audio_halo'
                def event(node):
                    if node < 0:return ''
                    value=nodes[node]['event']
                    try:return json.loads(value)
                    except (TypeError,ValueError):return value
                probability = float(p[b,h,q,k])
                row=dict(layer=label,kind=kind,head=int(h)+1,probability=probability,
                    query_seconds=float(qtime[b,q]),key_seconds=float(ktime[b,k]),
                    query_beat=float(qb[b,q]),key_beat=float(kb[b,k]),scope=scope,
                    query_node_id=qnode,key_node_id=knode,input_event=event(qnode),
                    predicted_event=accepted[qnode]['row']['event'] if qnode in accepted else '',key_event=event(knode),
                    record_path=record['path'],batch_index=int(b),query_index=int(q),key_index=int(k))
                all_writer.writerow(row);counts[kind]+=1
                if scope in defaults:final_writer.writerow(row);default_counts[kind]+=1
                data.append([layers.index(label),int(h)+1,probability,round(row['query_seconds'],6),
                    round(row['key_seconds'],6),scopes.index(scope),qnode,knode,row['input_event'],
                    row['predicted_event'],row['key_event'],record['path'],int(b),int(q),int(k)])

        for i,record in enumerate(index):
            with archive.open(record['path']) as stream,np.load(stream,allow_pickle=False) as a:
                if record['kind']=='decoder':
                    for layer in range(8):
                        for kind in ['self','cross']:
                            emit(a[f'{kind}_{layer:02d}'],f'decoder_{kind}_L{layer+1}','decoder_'+kind,
                                 a['query_beats'],a[f'{kind}_key_beats'],a['query_valid'],record,a,True)
                else:
                    emit(a['probability'],record['layer'],record['kind'],a['query_beats'],a['key_beats'],a['query_valid'],record,a)
            if i%300==0:print('Scanned',i,'/',len(index),'matches',len(data),flush=True)
    data.sort(key=lambda x:(-x[2],x[3],x[0],x[1]))
    meta=dict(threshold=.30,comparison='strictly greater, after FP16 probabilities are promoted to FP32',
        layers=layers,scopes=scopes,all_counts=dict(counts),default_counts=dict(default_counts),
        total=len(data),default_total=sum(default_counts.values()),
        columns=['layer_index','head','probability','query_seconds','key_seconds','scope_index','query_node_id',
                 'key_node_id','input_event','predicted_event','key_event','record_path','batch_index','query_index','key_index'])
    (root/'threshold-summary.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8')
    (root/'threshold-data.js').write_text('window.THRESHOLD_META='+json.dumps(meta,ensure_ascii=False)+';\nwindow.THRESHOLD_ROWS='+json.dumps(data,ensure_ascii=False,separators=(',',':'))+';\n',encoding='utf8')
    print(json.dumps(meta,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
