"""Native playable-file round trips, independently of the inference networks."""
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
from maidata.inference.materialize import materialize, write_json, render_origin


def write_audio(path):
    with wave.open(str(path), 'wb') as audio:
        audio.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
        audio.writeframes(bytes(28800*2))


@pytest.fixture
def native_miacode():
    configured = os.environ.get('MIACODE_CHART_IR')
    path = Path(configured) if configured else Path(__file__).resolve().parents[3]/'tools/MiaCode/build-handseq-20260820/Release/miacode_chart_ir.exe'
    if not path.is_file():
        pytest.skip('set MIACODE_CHART_IR to run native materialization integration')
    return path


@pytest.mark.parametrize('case',['complex','empty','negative','wait_across_bpm'])
def test_playable_materialization(case,geometry,tmp_path):
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
    report=materialize(tmp_path/'model.ir.json',tmp_path/'audio.wav',tmp_path/'playable',
        difficulty=5,level=14.5,title='中文曲名',artist='test',designer='model')
    assert report['renderer'] == 'python-simai'
    assert 'roundtrip' not in report and 'miacode' not in report
    result=(tmp_path/'playable/maidata.txt').read_text(encoding='utf-8')
    assert '&title=中文曲名' in result and '&lv_5=14.5' in result and '&inote_5=' in result
    assert (tmp_path/'playable/track.wav').read_bytes()==(tmp_path/'audio.wav').read_bytes()
    assert not (tmp_path/'playable/reparsed.ir.json').exists()
    with pytest.raises(FileExistsError):
        materialize(tmp_path/'model.ir.json',tmp_path/'audio.wav',tmp_path/'playable',difficulty=5,level=14.5)


def check_roundtrip(expected, actual):
    # Test-only independent parser comparison; production never reads back.
    from maidata.chart import encode_chart, token_to_dict
    from maidata.chart.types import Note
    from maidata.decoder.windows import real_onsets
    def events(document):
        encoded = encode_chart(document)
        result = []
        for group in real_onsets(encoded.tokens, encoded.timing, sequence_start_tick=encoded.start_tick):
            notes = []
            for token in group.content:
                if isinstance(token, Note):
                    notes.append([])
                notes[-1].append(token_to_dict(token))
            result.append((group.tick, sorted(json.dumps(n, sort_keys=True) for n in notes)))
        return encoded.timing, result
    first, original = events(expected)
    second, recovered = events(actual)
    assert original == recovered
    assert first.first_seconds == pytest.approx(second.first_seconds, abs=1e-8)
    for beat in {b for b, _ in first.points} | {b for b, _ in second.points}:
        assert first.bpm_at_beat(beat) == pytest.approx(second.bpm_at_beat(beat), abs=1e-8)
        assert first.beat_to_second(beat) == pytest.approx(second.beat_to_second(beat), abs=1e-8)


def test_materializer_import_does_not_load_models_torch_or_legacy_packages():
    result=subprocess.run([sys.executable,'-c',
        "import sys; import maidata.inference.materialize; assert 'torch' not in sys.modules; "
        "assert not any(k.startswith(('maimai_baseline','maimai_dataset','maimai_chart_ir')) for k in sys.modules)"],
        capture_output=True,text=True)
    assert result.returncode==0,result.stderr
