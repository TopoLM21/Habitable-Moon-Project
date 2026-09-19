"""Diagnostic collection must not crash or change the numerical worker."""
from pathlib import Path
import json
import subprocess
import sys

from moon_gui import diagnostics


def test_watchdog_and_requested_stacks_survive_active_numpy_frames(tmp_path):
    # Exercise the diagnostic thread while the main thread repeatedly enters
    # NumPy/Python frames, the trigger seen in the real 880->900 Myr worker.
    code = '''
from pathlib import Path
import faulthandler, json, sys, time
import numpy as np
from moon_gui import diagnostics
def forbidden_native_dump(*args, **kwargs):
    raise AssertionError("native frame walker must not run")
faulthandler.dump_traceback_later = forbidden_native_dump
faulthandler.cancel_dump_traceback_later = forbidden_native_dump
faulthandler.dump_traceback = forbidden_native_dump
diagnostics.WATCHDOG_SECONDS = .05
def numpy_workload():
    private_local = 'must-not-appear-in-stack-output'
    values = np.linspace(-2., 2., 200)
    stop = time.monotonic() + 1.2
    while time.monotonic() < stop:
        np.clip(values, -1., 1.)
diag = diagnostics.WorkerDiagnostics(Path(sys.argv[1]))
with diag.stage('active_numpy'):
    numpy_workload()
    response = diag.dump('requested', 'test-request')
diag.close()
print(json.dumps(response))
'''
    project = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, '-X', 'utf8', '-c', code, str(tmp_path)],
                            cwd=project, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    watchdog = (tmp_path/'stacks.txt').read_text(encoding='utf-8')
    assert 'Cooperative Python thread stacks' in watchdog
    assert 'numpy_workload' in watchdog
    assert 'must-not-appear-in-stack-output' not in watchdog
    response = json.loads(result.stdout.splitlines()[-1])
    assert Path(response['stack']).is_file()
    assert 'Cooperative Python thread stacks' in Path(response['stack']).read_text(encoding='utf-8')
    report = json.loads(Path(response['report']).read_text(encoding='utf-8'))
    assert report['stage']['name'] == 'active_numpy'
    assert report['failure'] is None


def test_watchdog_is_one_shot_and_real_progress_rearms_it(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, 'WATCHDOG_SECONDS', 0.)
    diag = diagnostics.WorkerDiagnostics(tmp_path)
    # Poll explicitly to avoid timer scheduling determining assertion order.
    monkeypatch.setattr(diag, 'start', lambda: None)
    with diag.stage('work'):
        diag._poll_watchdog()
        first = (tmp_path/'stacks.txt').read_text(encoding='utf-8')
        diag._poll_watchdog()
        assert (tmp_path/'stacks.txt').read_text(encoding='utf-8') == first
        diag.update(done=1)
        diag._poll_watchdog()
        second = (tmp_path/'stacks.txt').read_text(encoding='utf-8')
        assert second.count('Cooperative Python thread stacks') == 2
    diag.close()
    diag._poll_watchdog()
    assert diag._watchdog_deadline is None


def test_failure_report_keeps_original_exception_and_stage(tmp_path):
    diag = diagnostics.WorkerDiagnostics(tmp_path)
    try:
        with diag.stage('numerical_step', time_myr=900.):
            raise ValueError('original numerical failure')
    except ValueError:
        response = diag.dump('failure', 'failure-case')
    finally:
        diag.close()
    report = json.loads(Path(response['report']).read_text(encoding='utf-8'))
    assert report['failure']['stage']['name'] == 'numerical_step'
    assert report['failure']['exception_type'] == 'ValueError'
    assert 'original numerical failure' in report['exception']


def test_stack_io_failure_does_not_fail_a_calculation(tmp_path, monkeypatch):
    diag = diagnostics.WorkerDiagnostics(tmp_path)
    def cannot_write(*args, **kwargs):
        raise OSError('simulated full disk')
    monkeypatch.setattr(diagnostics, '_python_thread_stacks', cannot_write)
    with diag.stage('work'):
        diag._watchdog_deadline = 0.
        diag._poll_watchdog()
        response = diag.dump('requested', 'io-failure')
    diag.close()
    assert Path(response['report']).is_file()
