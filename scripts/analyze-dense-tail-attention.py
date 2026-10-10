"""Read only the captured queries predicting rows after the reference final onset."""
import argparse
from collections import Counter
import io
import json
from pathlib import Path
import zipfile
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('--gt',type=Path,required=True)
    a=p.parse_args();root=a.root
    gt=json.loads(a.gt.read_text(encoding='utf8'))
    last=gt['moments'][-1]
    last_beat=last['position_beats']['num']/last['position_beats']['den']
    # This song has one 184 BPM point and first_seconds=0, verified in both inputs.
    assert gt['first_seconds']==0 and gt['controls'][0]['bpm']==184
    end=last_beat*60/184
    tail_end=max(m['second']+(n.get('wait',{}).get('seconds',0)+max(b['duration']['seconds'] for b in n['branches'])
        if 'branches' in n else n.get('duration',{}).get('seconds',0))
        for m in gt['moments'] for g in m['groups'] for n in g['notes'])
    j=json.loads((root/'journal.json').read_text(encoding='utf8'))
    nodes={x['id']:x for x in j['nodes']}
    idx=json.loads((root/'index.json').read_text())
    by_call={r['call_id']:r for r in idx if r['kind']=='decoder'}
    predictions=[dict(x) for x in j['accepted_predictions'] if x['anchor_tick']>last_beat*96]
    results=[]
    with zipfile.ZipFile(root/'raw-attention.zip') as z:
        for prediction in predictions:
            node=nodes[prediction['query_node_id']];record=by_call[node['call_id']]
            result=dict(prediction,output_seconds=prediction['anchor_tick']/96*60/184,
                query_seconds=node['beat']*60/184,query_event=json.loads(node['event']),
                raw_path=record['path'],layers=[])
            with np.load(io.BytesIO(z.read(record['path'])),allow_pickle=False) as d:
                qi=list(d['query_node_ids']).index(node['id'])
                result['distant_semantic_examples']={}
                for kind in ('cross','self'):
                    seconds=d[f'{kind}_key_beats'][0].astype(float)*60/184
                    distant=seconds<result['query_seconds']-16*60/184
                    if kind=='self':
                        distant &= np.array([json.loads(nodes[int(n)]['event']) in ('NOTE','SLIDE_BRANCH') for n in d['key_node_ids']])
                    for layer in range(8):
                        probs=d[f'{kind}_{layer:02d}'][0,:,qi,:].astype(float)
                        dh,dk=np.unravel_index((probs*distant[None,:]).argmax(),probs.shape)
                        candidate=dict(layer=layer+1,head=int(dh)+1,key=int(dk),seconds=float(seconds[dk]),
                                       probability=float(probs[dh,dk]*distant[dk]))
                        if kind=='self':candidate['event']=json.loads(nodes[int(d['key_node_ids'][dk])]['event'])
                        old=result['distant_semantic_examples'].get(kind)
                        if old is None or candidate['probability']>old['probability']:
                            result['distant_semantic_examples'][kind]=candidate
                        mean=probs.mean(0);order=np.argsort(-mean)[:5]
                        top=[]
                        for k in order:
                            item=dict(key=int(k),seconds=float(seconds[k]),mean_probability=float(mean[k]))
                            if kind=='self':
                                kn=nodes[int(d['key_node_ids'][k])]
                                item.update(node_id=kn['id'],event=json.loads(kn['event']))
                            top.append(item)
                        h,k=np.unravel_index(probs.argmax(),probs.shape)
                        start=node['window_start_tick']/96*60/184
                        local=(seconds>=start-4*60/184)&(seconds<start+18*60/184)
                        result['layers'].append(dict(kind=kind,layer=layer+1,head_mean_top=top,
                            max_head=dict(head=int(h)+1,key=int(k),seconds=float(seconds[k]),probability=float(probs[h,k])),
                            mean_mass_before_original_end=float(mean[seconds<=end].sum()),
                            mean_mass_outside_local_audio_window=float(mean[~local].sum()) if kind=='cross' else None,
                            mean_mass_more_than_16_beats_back=float(mean[seconds<result['query_seconds']-16*60/184].sum())))
            results.append(result)
    summary=dict(original_last_onset_seconds=end,original_last_body_end_seconds=tail_end,
        tail_event_counts=dict(Counter(x['row']['event'] for x in results)),
        note_count_after_original_body_end=sum(x['row']['event']=='NOTE' and x['output_seconds']>tail_end+1e-6 for x in results),
        queries=results)
    try:
        import soundfile as sf
        audio,sr=sf.read(root/'audio.mp3',always_2d=True)
        summary['audio_duration_seconds']=len(audio)/sr
        summary['tail_audio_rms_dbfs']=[dict(start=s,end=min(s+1,len(audio)/sr),
            rms_dbfs=float(20*np.log10(max(1e-12,np.sqrt(np.mean(audio[int(s*sr):int((s+1)*sr)]**2))))))
            for s in range(140,151) if int(s*sr)<len(audio)]
    except ImportError:
        summary['tail_audio_rms_dbfs']='soundfile unavailable'
    (root/'tail-attention.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in summary.items() if k!='queries'},ensure_ascii=False))
    for q in results:
        if q['row']['event']!='NOTE':continue
        cross=next(x for x in q['layers'] if x['kind']=='cross' and x['layer']==8)
        own=next(x for x in q['layers'] if x['kind']=='self' and x['layer']==8)
        print(round(q['output_seconds'],3),q['row'],'crossL8',cross['head_mean_top'][:2],'selfL8',own['head_mean_top'][:2])


if __name__=='__main__':main()
