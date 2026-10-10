"""Immutable full-song beat patches; no learned features in this input pack."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from maimai_dataset.integrity import file_sha256
from maimai_dataset.manifest import canonical_json_sha256 as digest
from maimai_baseline.song_audio_contract import INPUT_CONTRACT as CONTRACT

SCHEMA='maimai.song_audio_p2_input.v1'
REUSED_SONG_INPUT_ID='7fa139c573545dc3e37e92a473f857810f4a1fb69e91d70b4e9f61ffc85a35b1'


def producer_identity():
    from maimai_baseline.source_identity import baseline_source_identity
    return dict(source=baseline_source_identity(),script_sha256=file_sha256(Path(__file__).resolve().parents[3]/'scripts/prepare-song-audio.py'))


def fusion_geometry(length):
    if not isinstance(length,int) or length<1:raise ValueError('song geometry needs at least one beat')
    # Covers every original or supplemental window start through the audio end.
    target=(np.arange((length+22)*2,dtype=np.float32)+.5)/2-4
    if length==1:left=right=np.zeros(len(target),np.int64);weight=np.zeros(len(target),np.float32)
    else:
        left=np.floor(target-.5).astype(np.int64).clip(0,length-2);right=left+1
        weight=(target-(left+.5)).clip(0,1).astype(np.float32)
    return dict(fusion_left=left,fusion_right=right,fusion_weight=weight)


def raw_song(feature,coords,timing,num_samples, *, include_fusion=True):
    from fractions import Fraction
    from maimai_baseline.time.audio_end import audio_end
    end=audio_end(timing,int(num_samples)).onset_tick
    n=max(1,end//96+1)
    k=np.arange(n*24,dtype=np.int64)
    pos=k-int(coords['grid_k'][0])
    available=(pos>=0)&(pos<len(coords['grid_k']))
    valid=np.zeros(len(k),bool)
    valid[available]=coords['valid'][pos[available]]
    valid &= k*4<=end
    mel=np.zeros((n*24,128),np.float32)
    mel[valid]=feature[:,pos[valid]].T
    return dict(**(fusion_geometry(n) if include_fusion else {}),mel=mel.reshape(n,24,128),frame_valid=valid.reshape(n,24),
        beats=np.arange(n,dtype=np.float32)+.5,
        bpms=np.asarray([timing.bpm_at_beat(Fraction(2*b+1,2)) for b in range(n)],np.float32))


def pack(rows,mapping):
    fields = ('mel','frame_valid','beats','bpms')
    if 'fusion_left' in rows[0]: fields += ('fusion_left','fusion_right','fusion_weight')
    return {**{k:pad_sequence([r[k] for r in rows],batch_first=True) for k in fields},
            'window_to_song':torch.as_tensor(mapping,dtype=torch.long)}


def encode_batch(model,inputs):
    context=model.song_audio_encoder(**inputs)
    model.song_audio_stats=dict(song_audio_encoder_calls=1,unique_audio_songs=len(context.memory),
        song_audio_tokens=int(context.valid.sum()),padded_song_audio_tokens=context.valid.numel())
    if hasattr(model,'a_encoding_stats'):model.a_encoding_stats.update(model.song_audio_stats)
    return context


class SongAudioCache:
    def __init__(self,root,*,verify=False):
        self.root=Path(root);m=json.loads((self.root/'manifest.json').read_text('utf8'));self.manifest=m
        if m.get('schema') != SCHEMA or not m.get('ready') or m.get('contract') != CONTRACT:
            raise ValueError('whole-song audio input layout mismatch')
        if set(m['variants'].values())!=set(m['files']):raise ValueError('whole-song audio coverage mismatch')
        if verify:
            for name,meta in m['files'].items():
                if (self.root/name).stat().st_size!=meta['bytes'] or file_sha256(self.root/name)!=meta['sha256']:raise ValueError('song audio payload mismatch')

    def get(self,variant, *, include_fusion=True):
        name=self.manifest['variants'][variant];path=self.root/name
        if path.stat().st_size!=self.manifest['files'][name]['bytes']:raise ValueError('song audio payload length mismatch')
        fields = ('mel','frame_valid','beats','bpms')
        if include_fusion: fields += ('fusion_left','fusion_right','fusion_weight')
        with np.load(path,allow_pickle=False) as z:
            row={k:torch.from_numpy(z[k].copy()) for k in fields}
        n=len(row['beats'])
        if include_fusion:
            for k,dtype in (('fusion_left',torch.long),('fusion_right',torch.long),('fusion_weight',torch.float32)):
                if row[k].shape!=((n+22)*2,) or row[k].dtype!=dtype:raise ValueError('invalid song-to-M2 geometry')
            if bool(((row['fusion_left']<0)|(row['fusion_right']>=n)|(row['fusion_weight']<0)|(row['fusion_weight']>1)).any()):raise ValueError('invalid song-to-M2 geometry bounds')
        if any(row[k].dtype!=torch.float32 for k in ('mel','beats','bpms')) or row['bpms'].shape!=(n,):raise ValueError('song audio dtype/shape mismatch')
        if row['mel'].shape!=(n,24,128) or row['frame_valid'].shape!=(n,24) or row['frame_valid'].dtype!=torch.bool or not torch.equal(row['beats'],torch.arange(n).float()+.5):
            raise ValueError('song audio geometry mismatch')
        if any(not torch.isfinite(x).all() for x in row.values()):raise ValueError('nonfinite song audio')
        return row

    def batch(self,variants, *, include_fusion=True):
        names=[self.manifest['variants'][v] for v in variants]
        unique=list(dict.fromkeys(names))
        return pack([self.get(variants[names.index(n)], include_fusion=include_fusion) for n in unique],[unique.index(n) for n in names])
