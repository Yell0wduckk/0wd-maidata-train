"""Recompute interventions from immutable pre-fusion features, never Encoders."""
from dataclasses import replace
import numpy as np
import torch
from maimai_baseline.model.song_audio import SongAudioContext


@torch.no_grad()
def rebuild(model, cache, key, *, mode, seed, device):
    if mode not in ('native','global_off','global_shuffle'):
        raise ValueError('unknown P2 intervention')
    with np.load(cache.root/f'diagnostic-{key}.npz',allow_pickle=False) as z:
        data={k:torch.from_numpy(z[k].copy()).to(device) for k in z.files}
    memories=tuple(data[f'm{i}'] for i in range(3))
    masks=tuple(data[f'padding{i}'] for i in range(3))
    full=SongAudioContext(data['global_audio'][None],data['global_valid'][None],data['global_beats'][None],
        torch.zeros(len(memories[0]),device=device,dtype=torch.long),
        {k:data['fusion_'+k][None] for k in ('left','right','weight')})
    if mode=='global_off':full=replace(full,valid=torch.zeros_like(full.valid))
    if mode=='global_shuffle':
        g=torch.Generator(device=device).manual_seed(seed);ids=full.valid[0].nonzero().flatten()
        memory=full.memory.clone();memory[0,ids]=memory[0,ids[torch.randperm(len(ids),generator=g,device=device)]]
        full=replace(full,memory=memory)
    with torch.autocast(device_type=device.type,enabled=False):
        p0=model.audio_encoder.fusion(memories,masks,song_audio=full,target_beats=data['beats2'])[0]
    length=len(data['global_valid'])
    return dict(audio=p0[:,32:160].reshape(1,-1,256)[:,:length*8],
        audio_valid=(~masks[0][:,32:160]).reshape(1,-1)[:,:length*8],
        audio_beats=data['beats0'][:,32:160].reshape(1,-1)[:,:length*8],
        windows=p0,starts=data['starts'])
