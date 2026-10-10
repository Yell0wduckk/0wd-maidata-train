"""Native playable-file round trips, independently of the inference networks."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import wave

import pytest

from test_chart_fields import geometry, complex_tokens
from maidata.chart import EncodedChart, Moment, Note, Route, SlideDuration, decode_chart
from maidata.time import TimingMap
from maidata.inference.__main__ import ir_metadata
from maidata.inference.materialize import materialize, write_json, check_roundtrip, render_origin


@pytest.fixture
def native_miacode():
    configured = os.environ.get('MIACODE_CHART_IR')
    path = Path(configured) if configured else Path(__file__).resolve().parents[3]/'tools/MiaCode/build-handseq-20260820/Release/miacode_chart_ir.exe'
    if not path.is_file():
        pytest.skip('set MIACODE_CHART_IR to run native materialization integration')
    return path


@pytest.mark.parametrize('case',['complex','empty','negative','wait_across_bpm'])
def test_native_playable_materialization(case,geometry,tmp_path,native_miacode):
    timing=TimingMap([(0,120),(1.5,240)],.125)
    tokens,start = complex_tokens(timing),0
    if case=='empty':
        tokens=(Moment(0),)
    elif case=='negative':
        tokens,start=(Moment(0),Note.make('tap','A1'),Moment(timing.delta_ms(-12,0)),Note.make('touch','C')),-12
    elif case=='wait_across_bpm':
        tokens=(Moment(0),Moment(timing.delta_ms(0,96)),Note.make('slide','A1'),Route(0,1),SlideDuration(97))
    ir=decode_chart(EncodedChart(tokens,ir_metadata(timing,5.),start),geometry)
    write_json(tmp_path/'model.ir.json',ir)
    with wave.open(str(tmp_path/'audio.wav'),'wb') as f:
        f.setparams((1,2,24000,0,'NONE','not compressed'))
        f.writeframes(bytes(24000*2*5))
    report=materialize(tmp_path/'model.ir.json',tmp_path/'audio.wav',tmp_path/'playable',native_miacode,
        difficulty=5,level=14.5,title='中文曲名',artist='test',designer='model')
    assert report['roundtrip']
    result=(tmp_path/'playable/maidata.txt').read_text(encoding='utf-8')
    assert '&title=中文曲名' in result and '&lv_5=14.5' in result and '&inote_5=' in result
    assert (tmp_path/'playable/track.wav').read_bytes()==(tmp_path/'audio.wav').read_bytes()
    recovered=json.loads((tmp_path/'playable/reparsed.ir.json').read_text(encoding='utf-8'))
    check_roundtrip(render_origin(ir),recovered)
    with pytest.raises(FileExistsError):
        materialize(tmp_path/'model.ir.json',tmp_path/'audio.wav',tmp_path/'playable',native_miacode,difficulty=5,level=14.5)


def test_semantic_roundtrip_rejects_changed_fields(geometry):
    timing=TimingMap([(0,120)])
    ir=decode_chart(EncodedChart((Moment(0),Note.make('tap','A1')),ir_metadata(timing,1.)),geometry)
    changed=deepcopy(ir)
    changed['moments'][0]['groups'][0]['notes'][0]['pad']='A2'
    with pytest.raises(ValueError,match='changed generated'):
        check_roundtrip(ir,changed)


def test_materializer_import_does_not_load_models_torch_or_legacy_packages():
    result=subprocess.run([sys.executable,'-c',
        "import sys; import maidata.inference.materialize; assert 'torch' not in sys.modules; "
        "assert not any(k.startswith(('maimai_baseline','maimai_dataset','maimai_chart_ir')) for k in sys.modules)"],
        capture_output=True,text=True)
    assert result.returncode==0,result.stderr
