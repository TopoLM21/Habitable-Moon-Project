"""Coupled physical-time launcher and cooling/contact visualization checks."""
import json
import os
from pathlib import Path
from time import monotonic

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PIL import Image
from PySide6.QtCore import QEventLoop, QProcess, QTimer
from PySide6.QtWidgets import QApplication
import pytest

from moon_gui.genesis_contact_dialog import GenesisContactDialog
from moon_gui.genesis_coupled_dialog import GenesisCoupledDialog
from tectonics.mesh import build_icosphere
from visualization.genesis_coupled import save_coupled_snapshot


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def _source(path, kind="genesis-faults-0.1", time=1.4):
    np.savez(path, metadata=np.array(json.dumps({"format": kind, "source_time_myr": 1.4,
                                               "state": {"time_myr": time}})))
    return path


def _pump(app, predicate, timeout=10):
    until = monotonic() + timeout
    while not predicate() and monotonic() < until:
        loop = QEventLoop()
        QTimer.singleShot(10, loop.quit)
        loop.exec()
    assert predicate(), "Coupled Qt worker did not settle"


def test_coupled_launcher_routes_faults_in_physical_years(app, tmp_path):
    dialog = GenesisCoupledDialog(project_root=tmp_path)
    try:
        assert not dialog.start_button.isEnabled()
        assert "одно физическое время" in dialog.introduction.text()
        path = _source(tmp_path / "fault.npz")
        assert dialog.set_checkpoint(path)
        args = dialog.arguments(tmp_path / "out")
        assert Path(args[1]).name == "run_genesis_coupled.py"
        assert "--checkpoint" in args and "--resume" not in args
        assert float(args[args.index("--step-years") + 1]) == 100.
        assert dialog.handoff_button.isHidden() and dialog.coupled_button.isHidden()
        dialog._set_busy(True)
        assert not any(widget.isEnabled() for widget in (dialog.start_button, dialog.select_button, dialog.duration, dialog.step))
    finally:
        dialog.close()


def test_coupled_resume_uses_original_fault_clock(app, tmp_path):
    source = _source(tmp_path / "coupled.npz", "genesis-coupled-0.2", 1.4012)
    dialog = GenesisCoupledDialog(project_root=tmp_path, checkpoint=source)
    try:
        assert dialog._source_elapsed == pytest.approx(1200.)
        assert dialog.duration.value() == pytest.approx(2200.)
        assert "--resume" in dialog.arguments(tmp_path / "out")
        dialog.duration.setValue(1000.)
        dialog.start()
        assert not dialog.is_running() and "позже сохранённого физического" in dialog.status.text()
    finally:
        dialog.close()


@pytest.mark.parametrize("kind", ["genesis-contact-0.1", "genesis-mobile-0.1"])
def test_coupled_rejects_incompatible_source_without_losing_selection(app, tmp_path, kind):
    good = _source(tmp_path / "fault.npz")
    dialog = GenesisCoupledDialog(project_root=tmp_path, checkpoint=good)
    try:
        assert not dialog.set_checkpoint(_source(tmp_path / "bad.npz", kind))
        assert dialog.checkpoint_path == good.resolve()
        assert "замороженный контакт несовместим" in dialog.status.text()
    finally:
        dialog.close()


def test_contact_entry_passes_only_original_fault_source(app, tmp_path, monkeypatch):
    import moon_gui.genesis_coupled_dialog as module
    (tmp_path / "run_genesis_coupled.py").touch()
    records = []

    class Stub:
        def __init__(self, parent, **kwargs):
            records.append(kwargs)
        def choose_checkpoint(self):
            records.append("picker")
        def exec(self):
            records.append("exec")

    monkeypatch.setattr(module, "GenesisCoupledDialog", Stub)
    source = _source(tmp_path / "fault.npz")
    dialog = GenesisContactDialog(project_root=tmp_path, checkpoint=source)
    try:
        assert dialog.coupled_button.isEnabled()
        dialog._open_coupled()
        assert records == [{"project_root": tmp_path, "checkpoint": source.resolve()}, "exec"]
        records.clear()
        dialog.set_checkpoint(_source(tmp_path / "contact.npz", "genesis-contact-0.1"))
        dialog._open_coupled()
        assert records == [{"project_root": tmp_path, "checkpoint": None}, "picker", "exec"]
        dialog._set_busy(True)
        assert not dialog.coupled_button.isEnabled()
    finally:
        dialog.close()


def test_coupled_qprocess_completion_is_responsive_and_loads_images(app, tmp_path):
    source = _source(tmp_path / "fault.npz")
    (tmp_path / "run_genesis_coupled.py").write_text('''import argparse,json,time
from pathlib import Path
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--output',type=Path);a,_=p.parse_known_args()
print('physical worker ready',flush=True);time.sleep(.15)
a.output.mkdir(parents=True)
(a.output/'summary.json').write_text(json.dumps({'model':'genesis-coupled','status':'completed','final':{'time_myr':1.4001,'elapsed_years':100.,'surface_temperature_k':285.,'mean_lid_thickness_km':8.}}))
Image.new('RGB',(100,70),'navy').save(a.output/'genesis_coupled.png')
Image.new('RGB',(100,70),'navy').save(a.output/'genesis_coupled.gif')
print('GENESIS_COUPLED_COMPLETE',flush=True)
''', encoding="utf-8")
    dialog = GenesisCoupledDialog(project_root=tmp_path, checkpoint=source)
    ticks = []
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(1))
    try:
        timer.start(10)
        dialog.start()
        _pump(app, lambda: not dialog.is_running() and dialog.start_button.isEnabled())
        assert ticks and dialog.result_view.count() == 2 and not dialog._pixmap.isNull()
        assert "285 K" in dialog.status.text() and "8 км" in dialog.status.text()
        assert dialog.output_path.name.startswith("coupled_")
        dialog.result_view.setCurrentIndex(1)
        assert dialog._movie is not None and dialog._movie.isValid()
    finally:
        timer.stop()
        dialog.close()


def test_coupled_close_stops_process_and_preserves_resume_guidance(app, tmp_path):
    source = _source(tmp_path / "fault.npz")
    (tmp_path / "run_genesis_coupled.py").write_text("import time\nprint('ready',flush=True)\ntime.sleep(20)\n")
    dialog = GenesisCoupledDialog(project_root=tmp_path, checkpoint=source)
    try:
        dialog.show()
        dialog.start()
        _pump(app, lambda: "ready" in dialog.log.toPlainText())
        dialog._kill_if_current(dialog._generation - 1)
        assert dialog.is_running()
        dialog.close()
        _pump(app, lambda: not dialog.is_running() and not dialog.isVisible())
        assert "coupled_checkpoint.npz" in dialog.status.text()
    finally:
        dialog.close()


@pytest.mark.parametrize("count", [0, 5])
def test_coupled_map_accepts_no_seams_and_shows_evolving_temperatures(tmp_path, monkeypatch, count):
    import matplotlib.figure
    import visualization.genesis_coupled as plots
    mesh = build_icosphere(1)
    fields = {"damage": np.linspace(0, 1, mesh.cell_count), "seam_centers_xyz": mesh.centroids[:count],
              "seam_gap_m": np.tile([-.1, 2.], (count, 1)), "seam_slip_m": np.tile([-1., .5], (count, 1))}
    history = [{"time_myr": 1.4 + i / 1e4, "elapsed_years": 100. * i, "surface_temperature_k": 600. - i * 10.,
                "mantle_temperature_k": 1600. - i, "ocean_fraction": i / 2., "mean_lid_thickness_km": 8. + i / 100.}
               for i in range(3)]
    observed = []
    original = matplotlib.figure.Figure.savefig

    def save(figure, *args, **kwargs):
        observed.extend(text.get_text() for text in figure.texts)
        return original(figure, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", save)
    path = save_coupled_snapshot(mesh, fields, tmp_path / "coupled.png", history, {"status": "completed"})
    assert "развиваются совместно" in "\n".join(observed)
    assert "зафиксированы" not in "\n".join(observed)
    with Image.open(path) as picture:
        assert picture.width >= 1500 and picture.height >= 1000
