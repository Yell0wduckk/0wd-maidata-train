"""Fixed Mel128 and event-audio-v4-rhythm acoustic18/aux32 algorithms.

Acoustic/structure formulas adapted from simai_transformer event_audio.py
(c9aa8e2e0b54b1b476c97aa2723de9d3c1147b3e); no runtime dependency on it.
"""
import numpy as np

ACOUSTIC_KEYS = tuple([f'chroma_{i:02d}' for i in range(12)] + [
    'onset_strength', 'rms', 'spectral_flux', 'relative_rms',
    'intensity_slope', 'harmonic_novelty'])
RHYTHM_KEYS = ('beat_onset_local', 'beat_onset_half', 'beat_flux_local', 'beat_rms_local')
CONTEXT_KEYS = ('song_progress', 'song_duration', 'local_tempo_bpm', 'beat_phase_sin',
                'beat_phase_cos', 'beat_progress', 'bar_phase_sin', 'bar_phase_cos',
                'phrase_phase_sin', 'phrase_phase_cos')
AUX_KEYS = ACOUSTIC_KEYS + RHYTHM_KEYS + CONTEXT_KEYS
MEL_RECIPE = dict(sample_rate=24000, n_fft=1024, n_mels=128,
                  f_min=30., f_max=12000., transform='log1p(power)', center=True,
                  pad_mode='reflect', hann_periodic=True, mel_scale='htk', mel_norm=None,
                  mono='float32-channel-mean', resample='float32-half-pixel-linear',
                  phase='absolute-k/R', sample_rounding='half-up-float64',
                  offsets='[-512,511]', extraction='independent-centered-FFT')


def read_wav(path):
    import soundfile as sf
    y, sr = sf.read(path, dtype='float32', always_2d=True)
    if not len(y) or not np.isfinite(y).all():
        raise ValueError(f'empty/nonfinite WAV: {path}')
    # Float WAV samples above one are intentional; never clamp or apply gain.
    return y.mean(axis=1, dtype=np.float32), sr


def linear_resample(y, sr, target):
    if sr == target:
        return y
    size = max(1, round(len(y) * target / sr))
    out = np.empty(size, np.float32)
    scale = np.float32(len(y)/size)
    for start in range(0, size, 262144):
        stop = min(size, start+262144)
        p = np.arange(start, stop, dtype=np.int64).astype(np.float32)
        p += np.float32(.5)
        p *= scale
        p -= np.float32(.5)
        np.maximum(p, 0, out=p)
        lo = np.minimum(np.floor(p).astype(np.int64), len(y)-1)
        w = np.clip(p-lo, 0, 1).astype(np.float32)
        out[start:stop] = (np.float32(1)-w)*y[lo] + w*y[np.minimum(lo+1, len(y)-1)]
    return out


def validate_stats(stats):
    for key in ('sample_rate', 'n_fft', 'n_mels', 'f_min', 'f_max'):
        if stats[key] != MEL_RECIPE[key]:
            raise ValueError(f'incompatible Mel stats: {key}')
    mean, std = np.asarray(stats['mean'], np.float32), np.asarray(stats['std'], np.float32)
    if mean.shape != (128,) or std.shape != (128,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError('invalid fixed Mel stats')
    return mean, std


def mel_at_centers(pcm, centers, valid, stats):
    """Independent 1024-sample FFT at each integer PCM center, matching old full."""
    import torch
    mono = torch.from_numpy(pcm)
    if len(pcm)<2 or not np.isfinite(pcm).all():
        raise ValueError('invalid PCM')
    f = torch.linspace(0., 12000., 513)
    bounds = torch.linspace(2595*torch.log10(torch.tensor(1+30/700)),
                            2595*torch.log10(torch.tensor(1+12000/700)), 130)
    hz = 700 * (torch.pow(10., bounds/2595)-1)
    bank = torch.minimum((f[None]-hz[:-2,None])/(hz[1:-1,None]-hz[:-2,None]).clamp_min(1e-12),
                         (hz[2:,None]-f[None])/(hz[2:,None]-hz[1:-1,None]).clamp_min(1e-12)).clamp_min(0)
    mean, std = validate_stats(stats)
    result = np.zeros((len(centers),128),np.float32)
    offsets, hann = torch.arange(-512,512),torch.hann_window(1024)
    period = 2*(len(pcm)-1)
    for start in range(0,len(centers),512):
        stop = min(start+512,len(centers))
        indices = torch.from_numpy(np.asarray(centers[start:stop],np.int64))[:,None]+offsets
        reflected = indices.remainder(period)
        reflected = torch.minimum(reflected,period-reflected)
        power = torch.fft.rfft(mono[reflected]*hann,dim=-1).abs().square().T
        raw = torch.log1p(bank@power).numpy().T
        result[start:stop] = (raw-mean)/std
    result[~np.asarray(valid,bool)] = 0
    return result


def robust_normalize(values):
    median = np.nanmedian(values, axis=0, keepdims=True)
    scale = np.maximum(np.nanpercentile(values, 75, axis=0, keepdims=True) -
                       np.nanpercentile(values, 25, axis=0, keepdims=True), 1e-4)
    return np.clip((np.nan_to_num(values)-median)/scale, -8, 8).astype(np.float32)


def structure_channels(rms, chroma, frame_rate=50.):
    frames = len(rms)
    if not frames:
        return np.zeros((0, 3), np.float32)
    relative = np.clip(rms/max(float(np.median(rms)), 1e-6), 0, 4)
    lag = max(1, round(frame_rate))
    slope = np.zeros_like(relative)
    if frames > lag:
        slope[lag:] = np.clip(relative[lag:]-relative[:-lag], -2, 2)
    novelty = np.zeros(frames, np.float32)
    width = max(1, round(2*frame_rate))
    if frames > 2*width:
        positive = np.clip(chroma, 0, None).astype(np.float64)
        cumulative = np.vstack([np.zeros((1, 12)), np.cumsum(positive, axis=0)])
        i = np.arange(frames)
        lo, hi = np.maximum(0, i-width), np.minimum(frames, i+width)
        before = (cumulative[i]-cumulative[lo])/np.maximum(i-lo, 1)[:, None]
        after = (cumulative[hi]-cumulative[i])/np.maximum(hi-i, 1)[:, None]
        bn, an = np.linalg.norm(before, axis=1), np.linalg.norm(after, axis=1)
        similarity = np.einsum('ij,ij->i', before, after)/np.maximum(bn*an, 1e-12)
        similarity = np.where(np.maximum(bn, an)<1e-6, 1., similarity)
        novelty = np.clip(1-similarity, 0, 1).astype(np.float32)
    return np.column_stack([relative, slope, novelty]).astype(np.float32)


def acoustic18(y, sr):
    import librosa
    duration = len(y)/sr
    y = librosa.resample(y, orig_sr=sr, target_sr=22050, res_type='soxr_hq') if sr != 22050 else y
    y = np.pad(y, (0, max(0, 2048-len(y))))
    chroma = librosa.feature.chroma_stft(y=y, sr=22050, n_fft=2048, hop_length=441).T
    onset = librosa.onset.onset_strength(y=y, sr=22050, hop_length=441)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=441, center=True)[0]
    spectrum = np.abs(librosa.stft(y=y, n_fft=2048, hop_length=441, center=True))
    normalized = spectrum / np.maximum(spectrum.sum(axis=0, keepdims=True), 1e-8)
    delta = np.diff(normalized, axis=1, prepend=normalized[:, :1])
    flux = np.sqrt(np.maximum(delta, 0).sum(axis=0)).astype(np.float32)
    count = max(len(chroma), len(onset), len(rms), len(flux), int(np.ceil(duration*50)))
    def pad(x):
        return np.concatenate([x, np.repeat(x[-1:], count-len(x), axis=0)]) if len(x)<count else x
    chroma, onset, rms, flux = map(pad, (chroma, onset, rms, flux))
    base = robust_normalize(np.column_stack([chroma, onset, rms, flux]))
    values = np.column_stack([base, structure_channels(rms, chroma)])
    times = np.arange(max(1, int(np.ceil(duration*50))), dtype=np.float64)/50
    return times, sample_rows(values, np.arange(count)/50, times, duration)


def sample_rows(values, source, targets, duration):
    shape = np.shape(targets)
    t = np.asarray(targets, np.float64).reshape(-1)
    # Clamp to nearest feature endpoint inside actual WAV, zero only outside WAV.
    p = np.clip(t, source[0], source[-1])
    if len(source) == 1:
        rows = np.repeat(values[:1], len(t), axis=0).copy()
    else:
        i = np.clip(np.searchsorted(source, p, side='left'), 1, len(source)-1)
        w = (p-source[i-1])/(source[i]-source[i-1])
        lo = values[i-1].astype(np.float64)
        rows = (lo+(values[i].astype(np.float64)-lo)*w[:, None]).astype(np.float32)
    rows[(t<0)|(t>=duration)] = 0
    return rows.reshape(*shape, values.shape[-1])


def rhythm4(acoustic, times, tempo):
    beats = tempo.to_beats(times)
    beat_sec = 60/tempo.tempo(beats)
    if len(times)>1:
        derivative = np.gradient(beats, times, edge_order=1)
        derived = 1/np.maximum(derivative, 1e-6)
        beat_sec = np.where(np.isfinite(derived)&(derivative>1e-6), derived, beat_sec)
    local = np.maximum(beat_sec, .02)
    result = []
    # Search each endpoint independently: unlike a one-way sweep this remains
    # correct if a BPM jump makes the neighbourhood's left boundary move back.
    for column, radius in ((12,.25), (12,.5), (14,.25), (13,.25)):
        lo = np.searchsorted(times, times-radius*local, side='left')
        hi = np.searchsorted(times, times+radius*local, side='right')
        cumulative = np.r_[0., np.cumsum(acoustic[:,column], dtype=np.float64)]
        result.append((cumulative[hi]-cumulative[lo])/np.maximum(hi-lo,1))
    return np.column_stack(result).astype(np.float32)


def aux32(acoustic_rhythm, times, beats, tempo, duration):
    beats = np.asarray(beats, np.float64)
    t = tempo.to_seconds(beats)
    context = np.stack([np.clip(t/max(duration,1e-9),0,1), np.full_like(t,np.clip(duration/300,0,1)),
        np.clip(tempo.tempo(beats)/240,0,2), np.sin(2*np.pi*(beats-np.floor(beats))),
        np.cos(2*np.pi*(beats-np.floor(beats))), np.clip(beats/max(float(tempo.to_beats(duration)),1),0,1),
        np.sin(2*np.pi*beats/4), np.cos(2*np.pi*beats/4),
        np.sin(2*np.pi*beats/16), np.cos(2*np.pi*beats/16)], axis=-1).astype(np.float32)
    values = np.concatenate([sample_rows(acoustic_rhythm,times,t,duration),context],axis=-1)
    values[(t<0)|(t>=duration)] = 0
    return values


def grid_coordinates(tempo, duration, num_samples, samples_per_beat, *, first_beat=None):
    r = samples_per_beat
    if not isinstance(r,int) or r<=0:
        raise ValueError('samples_per_beat must be a positive integer')
    first = int(np.floor(tempo.to_beats(0.))) if first_beat is None else first_beat
    last = int(np.ceil(tempo.to_beats(duration)))
    integer_beats = np.arange(first, max(last, first+1))
    k = integer_beats[:,None]*r+np.arange(r,dtype=np.int64)
    grid = k/r
    t = tempo.grid_seconds(k,r)
    samples = np.floor(t*24000+.5).astype(np.int64)
    valid = (t>=0)&(t<duration)&(samples>=0)&(samples<num_samples)
    return dict(grid_k=k,sample_times_sec=t,sample_indices=samples,sample_valid=valid,
                beat_centers=(integer_beats+.5).astype(np.float32),
                first_beat=np.array(first,dtype=np.int64))


def derive_grid(mel_patches, coordinates, times, acoustic, tempo, duration):
    grid = coordinates['grid_k']/coordinates['grid_k'].shape[1]
    rhythm = rhythm4(acoustic, times, tempo)
    base = np.column_stack([acoustic,rhythm])
    fine_aux = aux32(base,times,grid,tempo,duration)
    fine_aux[~coordinates['sample_valid']] = 0
    return dict(**coordinates,mel_patches=mel_patches,
                aux32_grid=fine_aux,
                aux32_centers=aux32(base,times,coordinates['beat_centers'],tempo,duration),
                rhythm4=rhythm)
