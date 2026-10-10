"""Run the serial launcher against a shell mock, never a training process."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


LAUNCHER = Path(__file__).resolve().parents[1] / 'scripts/run-decoder-plan-comparison.sh'


def bash_executable():
    if os.name == 'nt':
        git = shutil.which('git')
        if git:
            candidate = Path(git).parent.parent / 'bin/bash.exe'
            if candidate.is_file():
                return str(candidate)
        pytest.skip('native Git Bash is unavailable; WSL cannot consume this Windows fixture directly')
    bash = shutil.which('bash')
    if bash is None:
        pytest.skip('bash is unavailable')
    return bash


@pytest.mark.parametrize('fail_source,expected', [('', ['predicted', 'gt']), ('predicted', ['predicted'])])
def test_launcher_runs_independent_sources_in_order_and_stops_after_failure(tmp_path, fail_source, expected):
    mock = tmp_path / 'mock python'
    mock.write_text('''#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "$CALLS"
source=''
while [[ $# -gt 0 ]]; do
  if [[ "$1" == '--plan-source' ]]; then source=$2; break; fi
  shift
done
printf 'mock %s\\n' "$source"
if [[ "$source" == "$FAIL_SOURCE" ]]; then exit 7; fi
''', encoding='ascii', newline='\n')
    mock.chmod(0o755)
    config, predicted, gt, logs = (tmp_path / p for p in ('base config.json', 'predicted out', 'gt out', 'logs'))
    config.write_text('{}', encoding='ascii')
    calls = tmp_path / 'calls.txt'
    env = dict(os.environ, PYTHON=mock.as_posix(), CALLS=calls.as_posix(), FAIL_SOURCE=fail_source)
    result = subprocess.run([bash_executable(), LAUNCHER.as_posix(), config.as_posix(),
                             predicted.as_posix(), gt.as_posix(), logs.as_posix()],
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == (7 if fail_source else 0), result.stderr
    commands = calls.read_text().splitlines()
    assert len(commands) == len(expected)
    for command, source, output in zip(commands, expected, (predicted, gt)):
        assert command == (f'-u -m maidata.decoder_training --config {config.as_posix()} '
                           f'--plan-source {source} --output {output.as_posix()}')
    assert 'mock predicted' in (logs / 'predicted.log').read_text()
    assert (logs / 'gt.log').exists() == (not fail_source)
    assert not predicted.exists() and not gt.exists()  # only the mock ran
