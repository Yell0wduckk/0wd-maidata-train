"""The configuration UI uses stdlib; building imports only the new encoder."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("flag", ["--help", "--show-config"])
def test_stdlib_entry(flag):
    result = subprocess.run([sys.executable, "-S", "-m", "maidata", flag],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "Maidata" in result.stdout if flag == "--help" else "'mel_bins': 128" in result.stdout


def test_build_command_and_no_legacy_imports():
    script = """
import sys
from maidata.entry import main
assert main([]) == 0
assert not any(name.startswith(('maimai_baseline', 'maimai_dataset', 'maichart')) for name in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "AudioEncoder built" in result.stdout
    assert "training are not implemented" in result.stdout


@pytest.fixture(autouse=True)
def package_path(monkeypatch):
    source = str(Path(__file__).resolve().parents[1] / "src")
    monkeypatch.setenv("PYTHONPATH", source + os.pathsep + os.environ.get("PYTHONPATH", ""))
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
