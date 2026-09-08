"""The Windows entrypoint must work from another working directory."""
import os
from pathlib import Path
import subprocess

import pytest


@pytest.mark.skipif(os.name != 'nt', reason='Windows BAT entrypoint')
def test_bat_starts_its_own_local_environment(tmp_path):
    root = Path(__file__).resolve().parents[1]
    python = root / '.venv/Scripts/python.exe'
    if not python.is_file():
        pytest.skip('Local GUI environment has not been installed')
    environment = os.environ.copy()
    environment.pop('PYTHONPATH', None)
    environment['QT_QPA_PLATFORM'] = 'offscreen'
    result = subprocess.run(
        f'cmd.exe /d /c call "{root / "launch_gui.bat"}" --smoke-test',
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert str(python) in result.stdout
