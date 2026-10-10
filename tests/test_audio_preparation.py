import numpy as np
import pytest

from maidata.preparation.features import (AUX_KEYS, acoustic18, aux32, derive_grid,
    grid_coordinates, mel_at_centers, read_wav, rhythm4, robust_normalize, structure_channels)
from maidata.preparation.timing import TempoMap, chart_timing, fields
from maidata.preparation.cache import load_grid, song_inputs, validate_grid, window_inputs
from maidata.preparation.produce import grid_mel, save_arrays, write_json


def stats():
    return dict(sample_rate=24000,n_fft=1024,hop_length=240,n_mels=128,
                f_min=30.,f_max=12000.,mean=[0.]*128,std=[1.]*128)


def synthetic_grid(tempo,duration,r=24,m=128):
    coordinates = grid_coordinates(tempo,duration,round(duration*24000),r)
    mel = np.ones((*coordinates['grid_k'].shape,m),np.float32)
    mel[~coordinates['sample_valid']] = 0
    count = int(np.ceil(duration*50))
    return derive_grid(mel,coordinates,np.arange(count)/50,np.zeros((count,18),np.float32),tempo,duration)


def test_float_gain_read_and_mel(tmp_path):
    import soundfile as sf
    y = (1.7*np.sin(np.arange(24000)*2*np.pi*220/24000)).astype(np.float32)
    path = tmp_path/'gain.wav'
    sf.write(path,np.column_stack([y,y]),24000,subtype='FLOAT')
    actual,sr = read_wav(path)
    np.testing.assert_array_equal(actual,y)
    assert actual.max()>1
    centers = np.arange(100)*240
    raw = mel_at_centers(actual,centers,np.ones(100,bool),stats())
    doubled = mel_at_centers(actual*2,centers,np.ones(100,bool),stats())
    np.testing.assert_allclose(np.expm1(doubled),4*np.expm1(raw),rtol=2e-5,atol=.05)
    assert raw.shape==(100,128)


def test_timing_controls_ignore_notes_and_keep_variants():
    f = fields('&first=-.25\n&inote_2=(120){4}1[8:3],,\n(240){8},E\n&inote_3=(120){4},')
    a = chart_timing(f['inote_2'],100,f['first'])
    assert a['bpm']==[[0,1,120.],[2,1,240.]]
    t = TempoMap(a)
    b = np.array([-2.,0.,1.9,2.,2.1,20.])
    np.testing.assert_allclose(t.to_beats(t.to_seconds(b)),b,atol=1e-12)
    assert t.tempo(2.)==240
    assert t.to_seconds(0)==-.25
    assert chart_timing('(120){#0.25},(180),',120,0)['bpm'][1][:2]==[1,2]
    with pytest.raises(ValueError):
        chart_timing('(0),',120,0)


def test_structure_and_robust_scales():
    x = np.arange(101,dtype=np.float32)[:,None]
    got = robust_normalize(x)
    np.testing.assert_allclose(got[:,0],(x[:,0]-50)/50)
    rms = np.r_[np.ones(100),np.full(150,2)].astype(np.float32)
    c = structure_channels(rms,np.ones((250,12),np.float32))
    assert c[:,0].min()==.5 and c[:,0].max()==1
    assert np.all(c[:50,1]==0)
    assert c[0,2]==1  # Reference: empty preceding context vs nonempty following.
    np.testing.assert_allclose(c[1:,2],0,atol=1e-7)
    assert not structure_channels(np.zeros(250),np.zeros((250,12))).any()


def test_acoustic_order_and_silence():
    assert len(AUX_KEYS)==32 and AUX_KEYS[12:18]==('onset_strength','rms','spectral_flux',
        'relative_rms','intensity_slope','harmonic_novelty')
    times,features = acoustic18(np.zeros(24000,np.float32),24000)
    assert features.shape==(50,18) and np.isfinite(features).all()
    assert not features.any()
    np.testing.assert_allclose(times,np.arange(50)/50)


def test_rhythm_against_full_song_brute_force_at_jumps():
    tempo = TempoMap(dict(first_seconds=.11,bpm=[[0,1,240.],[4,1,45.],[6,1,200.]]))
    times = np.arange(400)/50
    acoustic = np.random.default_rng(3).normal(size=(400,18)).astype(np.float32)
    actual = rhythm4(acoustic,times,tempo)
    beat_sec = 1/np.gradient(tempo.to_beats(times),times)
    for i,t in enumerate(times):
        for j,(column,radius) in enumerate(((12,.25),(12,.5),(14,.25),(13,.25))):
            width = max(beat_sec[i],.02)*radius
            mask = (times>=t-width)&(times<=t+width)
            np.testing.assert_allclose(actual[i,j],acoustic[mask,column].mean(dtype=np.float64),atol=1e-6)


def test_grid_edges_phase_roundtrip_and_cross_window(tmp_path):
    tempo = TempoMap(dict(first_seconds=.17,bpm=[[0,1,120.],[3,1,240.]]))
    duration = 3.11
    times = np.arange(156)/50
    acoustic = np.random.default_rng(5).normal(size=(156,18)).astype(np.float32)
    mel = np.ones((312,128),np.float32)
    coordinates = grid_coordinates(tempo,duration,round(duration*24000),24)
    mel = np.ones((*coordinates['grid_k'].shape,128),np.float32)
    mel[~coordinates['sample_valid']] = 0
    grid = derive_grid(mel,coordinates,times,acoustic,tempo,duration)
    assert grid['first_beat']==-1
    validate_grid(grid)
    save_arrays(tmp_path/'grid.npz',grid,grid=True)
    grid = load_grid(tmp_path/'grid.npz')
    w = window_inputs([grid],[0,1],[0,0],target_beats=4,before_beats=2,after_beats=2)
    np.testing.assert_array_equal(w.mel[0,24:],w.mel[1,:-24])
    np.testing.assert_array_equal(w.aux32[0,24:],w.aux32[1,:-24])
    np.testing.assert_array_equal(w.valid[0,24:],w.valid[1,:-24])
    assert not w.valid[0,:24].any()
    # Evaluate tempo at the exact jump, and phases directly (not interpolated).
    sampled = aux32(np.zeros((156,22),np.float32),times,np.array([2.999,3.,3.001]),tempo,duration)
    np.testing.assert_allclose(sampled[:,24],[.5,1.,1.])
    np.testing.assert_allclose(sampled[:,25],np.sin(2*np.pi*np.array([2.999,3.,3.001])),atol=1e-7)
    outside = aux32(np.ones((156,22),np.float32),times,np.array([-10.,100.]),tempo,duration)
    assert not outside.any()


def test_song_window_encoder_connection():
    import torch
    from maidata.entry import CONFIG
    from maidata.model import build_model
    torch.set_num_threads(1)
    tempo = TempoMap(dict(first_seconds=.13,bpm=[[0,1,120.],[2,1,180.]]))
    grid = synthetic_grid(tempo,2.)
    song = song_inputs([grid,grid])
    windows = window_inputs([grid,grid],[0,1],[0,1],target_beats=4,before_beats=1,after_beats=1)
    config = dict(CONFIG,d_model=16,heads=2,ffn_dim=32,stem_channels=4,frequency_dim=8,
                  frequency_queries=2,target_beats=4,audio_before_beats=1,audio_after_beats=1,dropout=0.)
    encoder = build_model(config).eval()
    with torch.no_grad():
        result = encoder(song,windows)
    for name,rate in zip(('P0','P1','P2'),(8,4,2)):
        output = getattr(result,name)
        assert output.values.shape==(2,6*rate,16)
        assert torch.isfinite(output.values).all()


@pytest.mark.parametrize('r',[1,7,24,32])
def test_representation_driven_rate(r):
    grid = synthetic_grid(TempoMap(dict(first_seconds=0,bpm=[[0,1,120.]])),1.,r,16)
    assert grid['mel_patches'].shape==(2,r,16)
    validate_grid(grid)


def test_direct_fft_matches_accepted_algorithm():
    import torch
    from fractions import Fraction
    # Read-only numerical oracle from the copied source, never a production import.
    from maimai_baseline.data.audio import extract_at_centers
    from maimai_baseline.time.grid import grid_coordinates as old_coordinates
    from maimai_dataset.timing import TimingMap
    torch.set_num_threads(1)
    pcm = np.random.default_rng(17).normal(size=17213).astype(np.float32)*1.3
    timing = dict(first_seconds=.0527,bpm=[[0,1,173.],[1,1,247.],[3,2,61.]])
    tempo = TempoMap(timing)
    new = grid_coordinates(tempo,len(pcm)/24000,len(pcm),24)
    k = new['grid_k'].reshape(-1)
    old = old_coordinates(TimingMap(first_seconds=timing['first_seconds'],
        bpm_points=[(Fraction(n,d),b) for n,d,b in timing['bpm']]),int(k[0]),len(k),len(pcm))
    np.testing.assert_array_equal(new['sample_indices'].reshape(-1),old['samples'])
    np.testing.assert_array_equal(new['sample_valid'].reshape(-1),old['valid'])
    np.testing.assert_allclose(new['sample_times_sec'].reshape(-1),old['seconds'],rtol=0,atol=1e-15)
    actual = mel_at_centers(pcm,old['samples'],old['valid'],stats())
    expected = extract_at_centers(torch.from_numpy(pcm),old,stats()).T
    np.testing.assert_array_equal(actual,expected)
    # The first L sample is integer beat j; G0 is j+.5, a different coordinate.
    assert new['grid_k'][0,0]==int(new['first_beat'])*24


def test_quantized_pcm_end_validity():
    tempo = TempoMap(dict(first_seconds=.99999,bpm=[[0,1,120.]]))
    c = grid_coordinates(tempo,1.,24000,24)
    zero = np.argwhere(c['grid_k']==0)[0]
    assert c['sample_times_sec'][tuple(zero)]<1.
    assert c['sample_indices'][tuple(zero)]==24000
    assert not c['sample_valid'][tuple(zero)]


def test_existing_mel_reference_and_missing_position_only(tmp_path):
    tempo = TempoMap(dict(first_seconds=.05,bpm=[[0,1,120.]]))
    pcm = np.random.default_rng(11).normal(size=24000).astype(np.float32)
    coordinates = grid_coordinates(tempo,1.,len(pcm),24)
    valid = coordinates['sample_valid'].reshape(-1)
    keep = np.flatnonzero(valid)[2:-2]
    old_coordinates = {new:coordinates[key].reshape(-1)[keep] for new,key in
        (('grid_k','grid_k'),('seconds','sample_times_sec'),('samples','sample_indices'),('valid','sample_valid'))}
    old_mel = mel_at_centers(pcm,old_coordinates['samples'],old_coordinates['valid'],stats()).T
    np.save(tmp_path/'old.npy',old_mel)
    np.savez(tmp_path/'old-coordinates.npz',**old_coordinates)
    candidate = dict(feature_path=str(tmp_path/'old.npy'),coordinates_path=str(tmp_path/'old-coordinates.npz'),
                     shape=list(old_mel.shape),num_samples=len(pcm))
    mel,reused,new,unmatched,indices = grid_mel(pcm,coordinates,candidate,stats())
    assert reused==len(keep) and new==4 and unmatched==0
    expected = mel_at_centers(pcm,coordinates['sample_indices'].reshape(-1),valid,stats())
    # Different FFT/GEMM batch lengths can round float32 by about 1e-6.
    np.testing.assert_allclose(mel.reshape(-1,128),expected,rtol=2e-7,atol=2e-6)
    grid = derive_grid(mel,coordinates,np.arange(50)/50,np.zeros((50,18),np.float32),tempo,1.)
    grid.pop('mel_patches')
    grid['mel_source_indices'] = indices
    missing = valid&(indices.reshape(-1)<0)
    grid['mel_new_flat_indices'] = np.flatnonzero(missing)
    grid['mel_new_values'] = mel.reshape(-1,128)[missing]
    path = tmp_path/'reference.npz'
    save_arrays(path,grid,grid=True,mel_for_validation=mel)
    write_json(path.with_suffix('.json'),dict(mel_reference=candidate))
    loaded = load_grid(path)
    np.testing.assert_array_equal(loaded['mel_patches'],mel)
    with np.load(path) as a:
        assert 'mel_patches' not in a.files and len(a['mel_new_values'])==4


@pytest.mark.integration
def test_real_cache_samples():
    import json
    import os
    from pathlib import Path
    import torch
    from maidata.entry import CONFIG
    from maidata.model import build_model
    from maidata.preparation.features import linear_resample
    from maimai_baseline.data.audio import extract_at_centers
    root = Path(os.environ['MAIDATA_SAMPLE_CACHE'])
    catalog = json.loads((root/'catalog.json').read_text(encoding='utf-8'))
    selected = {'10021','505','11899','429','100548'}
    torch.set_num_threads(1)
    encoder = build_model(dict(CONFIG,dropout=0.)).eval()
    report = []
    for song in catalog['songs']:
        if song['song_id'] not in selected:
            continue
        variants = set(song['charts'].values())|set(song.get('legacy_full_variants',[]))
        if song.get('audio_only_variant'):
            variants.add(song['audio_only_variant'])
        for variant_id in variants:
            v = catalog['variants'][variant_id]
            a = catalog['assets'][v['asset']]
            g = load_grid(root/'grids'/f'{variant_id}.npz')
            y,sr = read_wav(a['wav_path'])
            pcm = linear_resample(y,sr,24000)
            flat = g['sample_valid'].reshape(-1)
            chosen = np.unique(np.linspace(0,len(flat)-1,31).astype(int))
            coordinates = dict(samples=g['sample_indices'].reshape(-1)[chosen],valid=flat[chosen])
            reference = extract_at_centers(torch.from_numpy(pcm),coordinates,catalog['stats']).T
            actual = g['mel_patches'].reshape(-1,128)[chosen]
            error = float(np.abs(reference-actual).max())
            np.testing.assert_allclose(actual,reference,rtol=2e-5,atol=2e-5)
            first = int(g['first_beat'])
            w = window_inputs([g],[first,first+1],[0,0],target_beats=16,before_beats=4,after_beats=4)
            np.testing.assert_array_equal(w.mel[0,24:],w.mel[1,:-24])
            np.testing.assert_array_equal(w.aux32[0,24:],w.aux32[1,:-24])
            np.testing.assert_array_equal(w.beats[0,24:],w.beats[1,:-24])
            with torch.no_grad():
                out = encoder(song_inputs([g]),w)
            for name,n in (('P0',192),('P1',96),('P2',48)):
                output = getattr(out,name)
                assert output.values.shape==(2,n,256) and torch.isfinite(output.values).all()
            report.append(dict(song_id=song['song_id'],variant=variant_id,
                legacy=variant_id in song.get('legacy_full_variants',[]),
                fft_max_abs_error=error,grid_shape=list(g['mel_patches'].shape),
                variable_bpm=len(v['timing']['bpm'])>1,default_encoder_shapes=[192,96,48]))
    assert {r['song_id'] for r in report}==selected
    write_json(root/'sample-validation.json',dict(passed=True,records=report))
