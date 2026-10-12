"""Exercise both user entrypoints with small real networks and Python simai output."""
import json
from pathlib import Path
import wave
import shutil
from unittest.mock import patch

import numpy as np
import pytest
import torch

from test_inference import config, producer_config
from test_pretrain_runtime import item
from maidata.model import build_model, build_draft, build_planner, build_decoder
from maidata.inference.__main__ import main
from maidata.inference.conditioning import audio_grid
from maidata.preparation.features import MEL_RECIPE
from maidata.chart import Note
from maidata.inference.streaming import StreamingPreview


@pytest.mark.parametrize('precision',['fp32','bf16'])
def test_cli_end_to_end_cpu(config,tmp_path,precision,capsys):
    root=Path(__file__).resolve().parents[3]
    geometry=root/'repos/maidata-baseline/training-dataset/full-dataset/runtime-lr-repaired/physical/path-catalog'
    slides=root/'tools/MiaCode/assets/reference/slide_data.json'
    if not (geometry/'paths.json').is_file() or not slides.is_file():
        pytest.skip('local reference geometry is unavailable')
    modules=[build_model(config),build_draft(config),build_planner(config),build_decoder(config)]
    with torch.no_grad():
        modules[-1].type_head.weight.zero_()
        modules[-1].type_head.bias.fill_(-100)
        modules[-1].type_head.bias[0]=80
        modules[-1].type_head.bias[1]=100
        modules[-1].heads.kind.weight.zero_()
        modules[-1].heads.kind.bias.fill_(-100)
        modules[-1].heads.kind.bias[0]=100
        head=modules[-1].heads.times['MOMENT']
        head.weight.zero_()
        head.bias.fill_(-100)
        head.bias[384]=100
    torch.save(dict(schema='maidata.encoder-draft-training.v3',recipe=config,
        model_state={n+'.'+k:v for n,m in zip(('encoder','draft','planner'),modules) for k,v in m.state_dict().items()}),tmp_path/'ed.ckpt')
    decoder_encoder=build_model(config)
    torch.save(dict(schema='maidata.decoder-training.v4',architecture='maidata.decoder-causal-cont-tick384.v2',
        recipe=config,model_state={n+'.'+k:v for n,m in (('encoder',decoder_encoder),('decoder',modules[-1]))
            for k,v in m.state_dict().items()}),tmp_path/'m.ckpt')
    np.savez(tmp_path/'grid.npz',**item(0,config)['grid'])
    with wave.open(str(tmp_path/'audio.wav'),'wb') as f:
        f.setparams((1,2,24000,0,'NONE','not compressed'))
        f.writeframes(bytes(28800*2))
    arguments=['--encoder-draft-checkpoint',str(tmp_path/'ed.ckpt'),'--decoder-checkpoint',str(tmp_path/'m.ckpt'),
        '--audio',str(tmp_path/'audio.wav'),'--grid',str(tmp_path/'grid.npz'),'--bpm','120',
        '--geometry',str(geometry),'--slide-data',str(slides),'--difficulty','5','--level','14.5',
        '--device','cpu','--precision',precision,'--temperature','0','--output',str(tmp_path/'result')]
    snapshots = []
    original_stream = StreamingPreview.__call__
    def inspect_stream(live,event):
        original_stream(live,event)
        if event['event'] == 'token' and event['complete']:
            assert not (tmp_path/'result/generation.json').exists()
            snapshots.append((event['window'], (tmp_path/'result/maidata.txt').read_bytes()))
    # Default inference spells live fields without constructing IR/token bundles.
    with patch('maidata.chart.decode_chart',side_effect=AssertionError('IR construction')), \
         patch('maidata.inference.generation.chart_from_groups',side_effect=AssertionError('token bundle')), \
         patch.object(StreamingPreview,'__call__',inspect_stream), \
         patch('maidata.inference.materialize.shutil.copy2',wraps=shutil.copy2) as copies:
        main(arguments)
    assert copies.call_count == 1
    assert snapshots and snapshots[0][0] == 1
    captured=capsys.readouterr()
    assert json.loads(captured.out)['windows']==2
    assert 'Model loading: started' in captured.err and 'Writing outputs: done' in captured.err
    assert 'Window 1/2' in captured.err and 'Window 2/2' in captured.err
    report=json.loads((tmp_path/'result/generation.json').read_text(encoding='utf-8'))
    assert report['complete'] and report['physical_bypass_count']==0 and len(report['windows'])==2
    assert report['moments']>0
    assert (tmp_path/'result/maidata.txt').is_file()
    assert (tmp_path/'result/track.wav').read_bytes() == (tmp_path/'audio.wav').read_bytes()
    assert not (tmp_path/'result/model.ir.json').exists()
    assert not (tmp_path/'result/model.tokens.json').exists()
    main([*arguments,'--no-stream-output','--no-progress','--output',str(tmp_path/'plain')])
    plain = json.loads((tmp_path/'plain/generation.json').read_text(encoding='utf-8'))
    assert report['stream_output'] and not plain['stream_output']
    assert report['windows'] == plain['windows'] and report['options'] == plain['options']
    assert (tmp_path/'plain/maidata.txt').read_bytes() == (tmp_path/'result/maidata.txt').read_bytes()
    with pytest.raises(FileExistsError):
        main(arguments)
    capsys.readouterr()
    with patch('maidata.inference.generation.generate',side_effect=RuntimeError('generation failed')), \
         pytest.raises(RuntimeError,match='generation failed'):
        main([*arguments,'--force','--no-progress'])
    captured=capsys.readouterr()
    assert captured.out==captured.err==''
    assert (tmp_path/'result/generation-error.json').is_file()
    assert not (tmp_path/'result/generation.json').exists()
    assert not (tmp_path/'result/model.ir.json').exists()
    assert not (tmp_path/'result/maidata.txt').exists()
    def fail_with_preview(*args,stream,**kwargs):
        stream(dict(event='window_start',window=1,start_tick=0))
        stream(dict(event='token',window=1,token=Note.make('tap','A1'),token_count=1,complete=True))
        raise RuntimeError('failed after live note')
    with patch('maidata.inference.generation.generate',side_effect=fail_with_preview), \
         pytest.raises(RuntimeError,match='failed after live note'):
        main([*arguments,'--force','--no-progress'])
    assert '&inote_5={384}(120)1,E' in (tmp_path/'result/maidata.txt').read_text(encoding='utf-8')
    assert not (tmp_path/'result/generation.json').exists()
    error = json.loads((tmp_path/'result/generation-error.json').read_text(encoding='utf-8'))
    assert error['complete'] is False and error['stream_output'] is True
    assert capsys.readouterr().out == ''


@pytest.mark.parametrize('first', [.125, 0., -.125, -1.25])
def test_raw_audio_uses_training_features_from_chart_beat_zero(tmp_path, first):
    import soundfile as sf
    signal=(.1*np.sin(np.arange(24000)*2*np.pi*440/24000)).astype(np.float32)
    sf.write(tmp_path/'audio.wav',signal,24000,subtype='FLOAT')
    timing=dict(first_seconds=first,bpm=[[0,1,120],[1,1,180]])
    stats=dict(MEL_RECIPE,mean=[0.]*128,std=[1.]*128)
    grid,duration=audio_grid(tmp_path/'audio.wav',timing,stats,samples_per_beat=24)
    assert duration==1 and int(grid['first_beat'])==0
    assert grid['sample_times_sec'][0,0]==pytest.approx(first)
    assert grid['beat_centers'][0]==.5
    if first < 0:
        assert not grid['sample_valid'][grid['sample_times_sec'] < 0].any()
    assert grid['mel_patches'].shape[1:]==(24,128)
    assert grid['aux32_centers'].shape[-1]==32
    assert np.isfinite(grid['aux32_grid']).all()
    assert not np.any(grid['mel_patches'][~grid['sample_valid']])
    assert not np.any(grid['aux32_grid'][~grid['sample_valid']])
