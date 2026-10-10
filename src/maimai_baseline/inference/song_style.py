"""Generate one full-song latent; decoder windows only slice it."""
import numpy as np
import torch
from maimai_baseline.data.store import window_arrays
from maimai_baseline.data.song_context import center_p0


@torch.no_grad()
def song_audio(model, feature, bpm_audio, timing, song_end_tick, device, full_audio=None, window_contexts=None):
    length=max(0,song_end_tick//96+1)
    if length==0:raise ValueError('audio has no nonnegative chart beats')
    if full_audio is None:
        from maimai_baseline.data.song_audio import raw_song,pack
        inputs=pack([{k:torch.from_numpy(v) for k,v in raw_song(feature,bpm_audio['coordinates'],timing,bpm_audio['num_samples']).items()}],[0])
        full_audio=model.song_audio_encoder(**{k:v.to(device) for k,v in inputs.items()})
    chunks=[];masks=[]
    for start in range(0,length,16):
        mel,valid,beats,bpms,geo=window_arrays(feature,bpm_audio['coordinates'],timing,start*96,bpm_audio['num_samples'])
        tensor=lambda x:torch.from_numpy(np.asarray(x))[None].to(device)
        a = None if window_contexts is None else window_contexts.get(start*96)
        if a is None:
            a=model.encode_audio(tensor(mel),~tensor(valid),tensor(beats),tensor(bpms),
                start_tick=torch.tensor([start*96],device=device),
                interpolation_geometry={k:tensor(v) for k,v in geo.items()},song_audio=full_audio)
        if window_contexts is not None:window_contexts[start*96]=a
        values,mask=center_p0(a.memories[-1][0].float().cpu().numpy(),(~a.padding_masks[-1][0]).cpu().numpy(),
            a.beats[-1][0].cpu().numpy()-start,min(16,length-start))
        chunks.append(values);masks.append(mask)
    return dict(audio=torch.from_numpy(np.concatenate(chunks))[None].to(device),
        audio_valid=torch.from_numpy(np.concatenate(masks))[None].to(device),
        audio_beats=((torch.arange(length*8,device=device).float()+.5)/8)[None],
        valid=torch.ones((1,song_end_tick//384+1),dtype=torch.bool,device=device),
        span_beats=torch.tensor([song_end_tick/96],device=device))
