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


@pytest.mark.parametrize('predicted_status,gt_status', [(0, 0), (7, 0), (0, 9), (7, 9)])
def test_launcher_runs_both_sources_in_order_even_after_failure(tmp_path, predicted_status, gt_status):
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
case "$source" in
  predicted) exit "$PREDICTED_STATUS" ;;
  gt) exit "$GT_STATUS" ;;
esac
''', encoding='ascii', newline='\n')
    mock.chmod(0o755)
    config, predicted, gt, logs = (tmp_path / p for p in ('base config.json', 'predicted out', 'gt out', 'logs'))
    config.write_text('{}', encoding='ascii')
    calls = tmp_path / 'calls.txt'
    env = dict(os.environ, PYTHON=mock.as_posix(), CALLS=calls.as_posix(),
               PREDICTED_STATUS=str(predicted_status), GT_STATUS=str(gt_status))
    result = subprocess.run([bash_executable(), LAUNCHER.as_posix(), config.as_posix(),
                             predicted.as_posix(), gt.as_posix(), logs.as_posix()],
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == (predicted_status or gt_status), result.stderr
    commands = calls.read_text().splitlines()
    assert len(commands) == 2
    for command, source, output in zip(commands, ('predicted', 'gt'), (predicted, gt)):
        assert command == (f'-u -m maidata.decoder_training --config {config.as_posix()} '
                           f'--plan-source {source} --output {output.as_posix()}')
    assert 'mock predicted' in (logs / 'predicted.log').read_text()
    assert 'mock gt' in (logs / 'gt.log').read_text()
    assert f'predicted={predicted_status} gt={gt_status}' in result.stdout
    assert not predicted.exists() and not gt.exists()  # only the mock ran
