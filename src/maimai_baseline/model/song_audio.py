"""Independent, beat-patch whole-song audio and step-local shared projections."""
from dataclasses import dataclass
import torch
from torch import nn
from maimai_baseline.model.rope import project_kv, rotate, attend

from maimai_baseline.song_audio_contract import CONTRACT


def attention(module, query, kv, positions, valid):
    allowed=valid[:,None,None,:].clone()
    allowed[...,0] |= ~allowed.any(-1)
    output=torch.cat([attend(module,query[:,i:i+128],kv,allowed=allowed,
        positions=positions[:,i:i+128]) for i in range(0,query.shape[1],128)],1)
    return output * valid.any(-1)[:,None,None]


@dataclass(frozen=True)
class SongAudioContext:
    memory: torch.Tensor
    valid: torch.Tensor
    beats: torch.Tensor
    window_to_song: torch.Tensor
    geometry: dict | None = None

    def __post_init__(self):
        if (self.memory.ndim!=3 or self.memory.shape[-1]!=256 or
            self.memory.shape[:2]!=self.valid.shape or self.beats.shape!=self.valid.shape or
            self.valid.dtype!=torch.bool or self.memory.shape[1]<1 or
            self.window_to_song.ndim!=1 or self.window_to_song.dtype!=torch.long):
            raise ValueError('invalid independent song audio context')
        if bool(((self.window_to_song<0)|(self.window_to_song>=len(self.memory))).any()):
            raise ValueError('song audio mapping out of range')

    def for_audio_windows(self, mapping, count):
        from dataclasses import replace
        if mapping is None:
            if len(self.window_to_song)!=count:raise ValueError('song/local window mapping mismatch')
            return self
        first=[]
        for i in range(count):
            ids=(mapping==i).nonzero().flatten()
            if not len(ids):raise ValueError('unreferenced local audio window')
            songs=self.window_to_song.index_select(0,ids)
            if not bool((songs==songs[0]).all()):raise ValueError('same window mapped to different song identities')
            first.append(songs[0])
        return replace(self,window_to_song=torch.stack(first))

    def aligned(self, target_beats):
        # Derived at normal read/forward time on the absolute half-beat lattice.
        song=self.window_to_song[:,None]
        n=self.memory.shape[1]
        left=(target_beats.float()-.5).floor().long().clamp(0,max(0,n-2))
        right=(left+1).clamp_max(n-1)
        weight=(target_beats.float()-(left.float()+.5)).clamp(0,1) if n>1 else torch.zeros_like(target_beats)
        with torch.autocast(device_type=self.memory.device.type,enabled=False):
            lw=(1-weight)*self.valid[song,left];rw=weight*self.valid[song,right];mass=lw+rw
            x=self.memory.float().masked_fill(~self.valid[...,None],0.)
            value=(x[song,left]*lw[...,None]+x[song,right]*rw[...,None])/mass.clamp_min(1e-12)[...,None]
            value=value.masked_fill((mass==0)[...,None],0.)
        return value.to(self.memory.dtype),mass>0



class SongAudioBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1=nn.LayerNorm(256);self.norm2=nn.LayerNorm(256)
        self.attention=nn.MultiheadAttention(256,8,dropout=.1,batch_first=True)
        self.ffn=nn.Sequential(nn.Linear(256,1024),nn.GELU(),nn.Dropout(.1),nn.Linear(1024,256),nn.Dropout(.1))
        self.dropout=nn.Dropout(.1)

    def forward(self,x,valid,beats):
        q=self.norm1(x);k,v=project_kv(self.attention,q)
        x=x+self.dropout(attention(self.attention,q,(rotate(k,beats),v),beats,valid))
        return (x+self.ffn(self.norm2(x))).masked_fill(~valid[...,None],0.)


class SongAudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer=nn.Sequential(nn.Linear(3096,256),nn.GELU(),nn.LayerNorm(256))
        self.bpm=nn.Sequential(nn.Linear(1,256),nn.GELU(),nn.Linear(256,256))
        self.blocks=nn.ModuleList(SongAudioBlock() for _ in range(2))

    def forward(self,mel,frame_valid,bpms,beats,window_to_song,fusion_left=None,fusion_right=None,fusion_weight=None):
        if mel.ndim!=4 or mel.shape[-2:]!=(24,128) or frame_valid.shape!=mel.shape[:-1] or bpms.shape!=beats.shape or beats.shape!=mel.shape[:2]:
            raise ValueError('song audio expects ordered [songs,beats,24,128] input')
        valid=frame_valid.any(-1)
        patch=torch.cat((mel.masked_fill(~frame_valid[...,None],0.).flatten(-2),frame_valid.to(mel.dtype)),-1)
        x=self.tokenizer(patch)+self.bpm(torch.log(bpms.float().clamp_min(1e-3)/120.)[...,None].to(mel.dtype))
        x=x.masked_fill(~valid[...,None],0.)
        for block in self.blocks:x=block(x,valid,beats)
        geometry = None if fusion_left is None else dict(left=fusion_left,right=fusion_right,weight=fusion_weight)
        return SongAudioContext(x,valid,beats,window_to_song,geometry)
