"""Small per-step accounting; no extra forward or diagnostic sampling."""
import json,time
import torch


def write(output,stage,step,started,inputs=None,targets=None,*,reference_count=None,encoding=None,timing=None):
    if reference_count is not None:
        refs = reference_count
        charts = None
        detail = dict(latent_enabled=False)
    elif not inputs:
        return
    elif targets is None and 'songs' not in inputs:
        refs = len(inputs['song_audio']['window_to_song'])
        charts = None
        detail = dict(latent_enabled=False)
    elif targets is None:
        songs=inputs['songs'];refs=len(inputs['window_groups']);charts=len(songs)
        detail=dict(moments=[len(s['moment_rows']) for s in songs],blocks=[s['valid'].numel() for s in songs],
            supervised_blocks=int(inputs['window_valid'].sum()))
    else:
        refs=targets['reference_count'];charts=targets['chart_count']
        detail=dict(audio_groups=targets['audio_count'],audio_windows=sum(map(len,inputs['audio'].values())),
            blocks=[g['valid'].shape[1] for g in targets['groups']],supervised_blocks=sum(int(g['weights'].sum()) for g in targets['groups']))
        detail.update(targets.get('audio_encoding',{}))
    if encoding is not None:detail.update(encoding)
    row=dict(stage=stage,step=step,seconds=time.perf_counter()-started,references=refs,charts=charts,reused_windows=None if charts is None else refs-charts,
        memory_allocated=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,**detail)
    if timing is not None:row.update(timing)
    with (output/'batch-costs.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(row)+'\n')
