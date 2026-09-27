"""Readiness checks run as cancellable children and never masquerade as handoff."""
import json
import os
from pathlib import Path
from time import monotonic

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QEventLoop, QProcess, QTimer
from PySide6.QtWidgets import QApplication
import pytest

from moon_gui.genesis_contact_dialog import GenesisContactDialog


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def _checkpoint(path, *, contact=True):
    np.savez(path, metadata=np.array(json.dumps({
        "format": "genesis-contact-0.1" if contact else "genesis-faults-0.1",
        "state": {"elapsed_years": 1000. if contact else 0.}})))
    return path


def _pump(predicate, timeout=10):
    deadline = monotonic()+timeout
    while not predicate() and monotonic() < deadline:
        loop = QEventLoop()
        QTimer.singleShot(10, loop.quit)
        loop.exec()
    assert predicate(), "Qt readiness child did not settle"


def _worker(root, *, marker=True, exit_code=0, delay=.15):
    script = """import argparse,json,time,sys
from pathlib import Path
from PIL import Image
p=argparse.ArgumentParser()
p.add_argument('--checkpoint',required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--probe-years',type=float,required=True)
p.add_argument('--intervals',type=int,required=True)
a=p.parse_args()
print('readiness worker started',flush=True)
time.sleep(DELAY)
a.output.mkdir(parents=True)
(a.output/'received.json').write_text(json.dumps({'checkpoint':a.checkpoint,'probe_years':a.probe_years,'intervals':a.intervals}))
(a.output/'handoff_report.json').write_text(json.dumps({'handoff_ready':False,'blockers':[{'code':'nonrigid','message':'Regions are not rigid.'}]}))
(a.output/'handoff_report.md').write_text('Readiness report only.')
Image.new('RGB',(100,70),'navy').save(a.output/'handoff_assessment.png')
if MARKER: print('HANDOFF_SCREEN_COMPLETE ready=false',flush=True)
sys.exit(EXIT_CODE)
"""
    script = script.replace("DELAY", repr(delay)).replace("MARKER", repr(marker)).replace("EXIT_CODE", repr(exit_code))
    (root/"run_genesis_handoff.py").write_text(script, encoding="utf-8")


def test_readiness_requires_actual_contact_format_and_preserves_contact_route(app, tmp_path):
    path = _checkpoint(tmp_path/"contact_checkpoint.npz", contact=False)
    dialog = GenesisContactDialog(project_root=tmp_path, checkpoint=path)
    try:
        assert dialog.start_button.isEnabled()
        assert not dialog.handoff_button.isEnabled()
        with pytest.raises(ValueError, match="contact_checkpoint"):
            dialog.handoff_arguments(tmp_path/"out")
        dialog.start_handoff()
        assert not dialog.is_running() and dialog.output_path is None
        assert "--checkpoint" in dialog.arguments(tmp_path/"out")
        assert Path(dialog.arguments(tmp_path/"out")[1]).name == "run_genesis_contact.py"
        source = _checkpoint(tmp_path/"renamed.npz")
        assert dialog.set_checkpoint(source)
        args = dialog.handoff_arguments(tmp_path/"report")
        assert dialog.handoff_button.isEnabled()
        assert Path(args[1]).name == "run_genesis_handoff.py"
        assert args[args.index("--checkpoint")+1] == str(source.resolve())
        assert args[args.index("--probe-years")+1] == "10"
        assert args[args.index("--intervals")+1] == "3"
        assert "--resume" not in args
        assert "--resume" in dialog.arguments(tmp_path/"continue")
        dialog._set_busy(True)
        assert not dialog.handoff_button.isEnabled()
    finally:
        dialog.close()


def test_completed_contact_checkpoint_precedes_selected_source_and_clears_on_selection(app, tmp_path):
    source = _checkpoint(tmp_path/"selected.npz")
    dialog = GenesisContactDialog(project_root=tmp_path, checkpoint=source)
    try:
        result = tmp_path/"contact_result"
        result.mkdir()
        produced = _checkpoint(result/"contact_checkpoint.npz")
        (result/"summary.json").write_text(json.dumps({"status":"completed","final":{
            "elapsed_years":1010.,"max_opening_m":2.,"max_abs_jump_m":1.}}))
        dialog.output_path = result
        dialog._finished(0, QProcess.ExitStatus.NormalExit)
        args = dialog.handoff_arguments(tmp_path/"out")
        assert args[args.index("--checkpoint")+1] == str(produced)
        assert dialog.handoff_button.isEnabled()
        # A newly selected fault snapshot must not accidentally test old contact results.
        dialog.set_checkpoint(_checkpoint(tmp_path/"other_fault.npz", contact=False))
        assert dialog.handoff_checkpoint() is None
        assert not dialog.handoff_button.isEnabled()
    finally:
        dialog.close()


def test_readiness_child_is_responsive_shows_failure_of_physical_gate_and_uses_new_folder(app, tmp_path):
    source = _checkpoint(tmp_path/"selected.npz")
    original = source.read_bytes()
    _worker(tmp_path)
    dialog = GenesisContactDialog(project_root=tmp_path, checkpoint=source)
    ticks = []
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(1))
    try:
        timer.start(10)
        dialog.start_handoff()
        assert dialog.is_running() and not dialog.handoff_button.isEnabled()
        _pump(lambda: not dialog.is_running() and dialog.handoff_button.isEnabled())
        assert ticks and not dialog._pixmap.isNull()
        assert "пока не готово" in dialog.status.text()
        assert "Regions are not rigid." in dialog.status.text()
        assert "Зрелая тектоника не запускалась" in dialog.status.text()
        assert "Рассчитано" not in dialog.status.text()
        assert dialog.result_view.count() == 1
        first = dialog.output_path
        report = (first/"handoff_report.json").read_bytes()
        actual = json.loads((first/"received.json").read_text())
        assert actual == {"checkpoint":str(source.resolve()),"probe_years":10.,"intervals":3}
        dialog.start_handoff()
        _pump(lambda: not dialog.is_running() and dialog.handoff_button.isEnabled())
        assert dialog.output_path != first and dialog.output_path.name.startswith("handoff_")
        assert (first/"handoff_report.json").read_bytes() == report
        assert source.read_bytes() == original
    finally:
        timer.stop()
        dialog.close()


@pytest.mark.parametrize("marker,exit_code", [(False,0),(True,3)])
def test_missing_completion_or_failed_child_never_reports_physical_success(app, tmp_path, marker, exit_code):
    source = _checkpoint(tmp_path/"source.npz")
    _worker(tmp_path, marker=marker, exit_code=exit_code)
    dialog = GenesisContactDialog(project_root=tmp_path, checkpoint=source)
    try:
        dialog.start_handoff()
        _pump(lambda: not dialog.is_running() and dialog.handoff_button.isEnabled())
        assert "Проверка завершена" not in dialog.status.text()
        assert "ошибкой" in dialog.status.text() or "Не удалось прочитать" in dialog.status.text()
        assert dialog.start_button.isEnabled() and not dialog.stop_button.isEnabled()
    finally:
        dialog.close()


def test_readiness_failed_to_start_restores_controls(app, tmp_path, monkeypatch):
    import moon_gui.genesis_contact_dialog as module
    source = _checkpoint(tmp_path/"source.npz")
    dialog = GenesisContactDialog(project_root=tmp_path, checkpoint=source)
    try:
        monkeypatch.setattr(module.sys, "executable", str(tmp_path/"missing-python.exe"))
        dialog.start_handoff()
        _pump(lambda: not dialog.is_running() and dialog.handoff_button.isEnabled())
        assert "Не удалось запустить Python" in dialog.status.text()
        assert not dialog.stop_button.isEnabled()
    finally:
        dialog.close()


def test_readiness_close_cancels_child_and_stale_kill_cannot_target_new_run(app, tmp_path):
    source = _checkpoint(tmp_path/"source.npz")
    original = source.read_bytes()
    _worker(tmp_path, delay=30.)
    dialog = GenesisContactDialog(project_root=tmp_path, checkpoint=source)
    dialog.show()
    try:
        dialog.start_handoff()
        _pump(lambda: "readiness worker started" in dialog.log.toPlainText())
        dialog._kill_if_current(dialog._generation-1)
        assert dialog.is_running()
        dialog.close()
        _pump(lambda: not dialog.is_running() and not dialog.isVisible())
        assert dialog._stopping
        assert "Проверка перехода остановлена" in dialog.status.text()
        assert source.read_bytes() == original
        assert dialog.handoff_checkpoint() == source.resolve()
    finally:
        dialog.close()


def test_existing_output_directory_is_never_reused(app, tmp_path, monkeypatch):
    import moon_gui.genesis_contact_dialog as module

    class FixedTime:
        @classmethod
        def now(cls):
            return cls()

        def strftime(self, pattern):
            return "handoff_fixed"

    dialog = GenesisContactDialog(project_root=tmp_path)
    try:
        monkeypatch.setattr(module, "datetime", FixedTime)
        existing = tmp_path/"results"/"genesis_runs"/"handoff_fixed"
        existing.mkdir(parents=True)
        marker = existing/"keep.txt"
        marker.write_text("existing result")
        assert dialog._new_output("handoff") != existing
        assert marker.read_text() == "existing result"
    finally:
        dialog.close()
