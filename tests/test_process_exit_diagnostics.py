"""A dead numerical process must still leave its exit code in the GUI report."""
import json
from pathlib import Path
import subprocess
import sys


def test_native_exit_and_python_failure_are_preserved_in_saved_reports(tmp_path):
    code = '''
from pathlib import Path
import sys
from PySide6.QtCore import QCoreApplication, QProcess
from moon_gui.app import SimulationController
app = QCoreApplication([])
for name, prior in [('native', None), ('python', {'exception_type':'ValueError', 'exception_message':'original'})]:
    folder = Path(sys.argv[1])/name
    folder.mkdir()
    controller = SimulationController()
    controller.diagnostics.segment_dir = folder
    controller.diagnostics.stage = {'name':'late rifting', 'details':{'time_myr':900.}}
    controller.diagnostics.stage_path = ['step', 'late rifting']
    controller.diagnostics.failure = prior
    controller.current_time = 880.
    controller._process_finished(-1073741819, QProcess.ExitStatus.CrashExit)
    assert controller.state == 'Error'
    assert controller.current_time == 880.
    controller.diagnostics_timer.stop()
'''
    result = subprocess.run([sys.executable, '-X', 'utf8', '-c', code, str(tmp_path)],
                            cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    for name, exception in [('native', 'ProcessExit'), ('python', 'ValueError')]:
        report = json.loads(next((tmp_path/name).glob('gui-report-*.json')).read_text(encoding='utf-8'))
        failure = report['failure']
        assert failure['exit_code'] == -1073741819
        assert failure['exit_code_hex'] == '0xC0000005'
        assert failure['exit_status'] == 'CrashExit'
        assert failure['exception_type'] == exception
        assert failure['stage']['details']['time_myr'] == 900.
        assert any('exited with code' in line for line in report['log_tail'])
