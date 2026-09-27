"""Contact launcher routing, cancellable child processes and gap-safe maps."""
from dataclasses import replace
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
from moon_gui.genesis_dialog import GenesisDialog, ROOT
from tectonics.mesh import build_icosphere
from visualization.genesis_contact import material_polygons, save_contact_snapshot


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def _checkpoint(path, *, contact=False, elapsed=0.):
    np.savez(path, metadata=np.array(json.dumps({
        "format": "genesis-contact-0.1" if contact else "genesis-faults-0.1",
        "state": {"elapsed_years": elapsed}})))
    return path


def _pump(app, predicate, timeout=10):
    deadline = monotonic()+timeout
    while not predicate() and monotonic() < deadline:
        loop = QEventLoop()
        QTimer.singleShot(10, loop.quit)
        loop.exec()
    assert predicate(), "Qt contact child did not settle"


def test_contact_requires_source_and_routes_fault_checkpoint_in_years(app, tmp_path):
    dialog = GenesisContactDialog(project_root=tmp_path)
    try:
        assert not dialog.start_button.isEnabled()
        assert "зафиксированы" in dialog.introduction.text()
        assert "Остывание здесь не продолжается" in dialog.introduction.text()
        with pytest.raises(ValueError):
            dialog.arguments(tmp_path/"out")
        path = _checkpoint(tmp_path/"fault_checkpoint.npz")
        assert dialog.set_checkpoint(path)
        args = dialog.arguments(tmp_path/"out")
        assert Path(args[1]).name == "run_genesis_contact.py"
        assert args[args.index("--checkpoint")+1] == str(path.resolve())
        assert "--resume" not in args and "--duration-myr" not in args
        assert float(args[args.index("--duration-years")+1]) == 1000
        assert float(args[args.index("--step-years")+1]) == 10
        assert dialog.start_button.isEnabled()
        dialog._set_busy(True)
        assert not any(w.isEnabled() for w in (dialog.select_button, dialog.duration, dialog.step, dialog.start_button))
    finally:
        dialog.close()


def test_contact_resume_detected_and_end_time_advances_past_saved_time(app, tmp_path):
    path = _checkpoint(tmp_path/"renamed.npz", contact=True, elapsed=1200.)
    dialog = GenesisContactDialog(project_root=tmp_path, checkpoint=path)
    try:
        args = dialog.arguments(tmp_path/"out")
        assert "--resume" in args and "--checkpoint" not in args
        assert dialog.duration.value() == 2200
        assert "1200" in dialog.source_kind.text()
        dialog.duration.setValue(1100)
        dialog.start()
        assert not dialog.is_running()
        assert "позже сохранённого" in dialog.status.text()
    finally:
        dialog.close()


@pytest.mark.parametrize("bad", ["missing", "garbage", "old_mode", "nan_elapsed"])
def test_invalid_source_is_rejected_without_losing_valid_selection(app, tmp_path, bad):
    source = _checkpoint(tmp_path/"good.npz")
    dialog = GenesisContactDialog(project_root=tmp_path, checkpoint=source)
    target = tmp_path/"invalid.npz"
    try:
        if bad == "garbage":
            target.write_bytes(b"not npz")
        elif bad == "old_mode":
            np.savez(target, metadata=np.array('{"format":"genesis-mobile-0.1"}'))
        elif bad == "nan_elapsed":
            _checkpoint(target, contact=True, elapsed=float("nan"))
        assert not dialog.set_checkpoint(target)
        assert dialog.checkpoint_path == source.resolve()
        assert "Не удалось выбрать" in dialog.status.text()
    finally:
        dialog.close()


def test_picker_uses_returned_path_and_cancel_keeps_source(app, tmp_path, monkeypatch):
    import moon_gui.genesis_contact_dialog as module
    path = _checkpoint(tmp_path/"fault.npz")
    dialog = GenesisContactDialog(project_root=tmp_path)
    try:
        monkeypatch.setattr(module.QFileDialog, "getOpenFileName", lambda *a, **k: (str(path), ""))
        dialog.choose_checkpoint()
        assert dialog.checkpoint_path == path.resolve()
        monkeypatch.setattr(module.QFileDialog, "getOpenFileName", lambda *a, **k: ("", ""))
        dialog.choose_checkpoint()
        assert dialog.checkpoint_path == path.resolve()
    finally:
        dialog.close()


def _result(path, status="completed"):
    path.mkdir(exist_ok=True)
    (path/"summary.json").write_text(json.dumps({"status": status, "final": {
        "elapsed_years": 30., "max_opening_m": 1.25, "max_abs_jump_m": 2.5}}), encoding="utf-8")
    for name in ("genesis_contact.png", "genesis_contact.gif"):
        Image.new("RGB", (100, 70), "navy").save(path/name)


def test_contact_result_loads_png_gif_and_reports_limits(app, tmp_path):
    dialog = GenesisContactDialog(project_root=tmp_path)
    try:
        _result(tmp_path/"out", "small_sliding_limit")
        dialog.output_path = tmp_path/"out"
        dialog._finished(0, QProcess.ExitStatus.NormalExit)
        assert "30 лет" in dialog.status.text()
        assert "1.25 м" in dialog.status.text() and "2.5 м" in dialog.status.text()
        assert "small_sliding_limit" in dialog.status.text()
        assert not dialog._pixmap.isNull()
        assert dialog.folder_button.isEnabled()
        assert dialog.result_view.count() == 2
        dialog.result_view.setCurrentIndex(1)
        assert dialog._movie is not None and dialog._movie.isValid()
        dialog._stopping = True
        dialog._finished(-1, QProcess.ExitStatus.CrashExit)
        assert "contact_checkpoint.npz" in dialog.status.text()
    finally:
        dialog.close()


def test_real_child_completion_keeps_qt_responsive_and_shows_preview(app, tmp_path):
    path = _checkpoint(tmp_path/"fault.npz")
    worker = '''import argparse,json,time
from pathlib import Path
from PIL import Image
p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
a,_=p.parse_known_args()
print('contact worker ready',flush=True)
time.sleep(.15)
a.output.mkdir(parents=True)
(a.output/'summary.json').write_text(json.dumps({'status':'completed','final':{'elapsed_years':10.,'max_opening_m':.5,'max_abs_jump_m':1.}}))
Image.new('RGB',(100,70),'navy').save(a.output/'genesis_contact.png')
'''
    (tmp_path/"run_genesis_contact.py").write_text(worker, encoding="utf-8")
    dialog = GenesisContactDialog(project_root=tmp_path, checkpoint=path)
    ticks = []
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(1))
    try:
        timer.start(10)
        dialog.start()
        assert not dialog.start_button.isEnabled()
        _pump(app, lambda: not dialog.is_running() and dialog.start_button.isEnabled())
        assert ticks and not dialog._pixmap.isNull()
        assert "10 лет" in dialog.status.text()
        assert dialog.output_path.name.startswith("contact_")
    finally:
        timer.stop()
        dialog.close()


def test_close_terminates_child_and_stale_timer_cannot_kill_new_generation(app, tmp_path):
    path = _checkpoint(tmp_path/"fault.npz")
    (tmp_path/"run_genesis_contact.py").write_text("import time\nprint('ready',flush=True)\ntime.sleep(20)\n")
    dialog = GenesisContactDialog(project_root=tmp_path, checkpoint=path)
    dialog.show()
    dialog.start()
    _pump(app, lambda: "ready" in dialog.log.toPlainText())
    dialog._kill_if_current(dialog._generation-1)
    assert dialog.is_running()
    dialog.close()
    _pump(app, lambda: not dialog.is_running() and not dialog.isVisible())
    assert dialog._stopping


def test_main_dialog_contact_button_preserves_genesis_mode_and_passes_checkpoint(app, tmp_path, monkeypatch):
    import moon_gui.genesis_contact_dialog as module
    (tmp_path/"configs").mkdir()
    (tmp_path/"configs/genesis_moon.yaml").write_bytes((ROOT/"configs/genesis_moon.yaml").read_bytes())
    for name in ("run_genesis_faults.py", "run_genesis_contact.py"):
        (tmp_path/name).touch()
    recorded = []

    class Stub:
        def __init__(self, parent, **kwargs):
            recorded.append(kwargs)
        def choose_checkpoint(self):
            recorded.append("picker")
        def exec(self):
            recorded.append("exec")

    monkeypatch.setattr(module, "GenesisContactDialog", Stub)
    dialog = GenesisDialog(project_root=tmp_path)
    try:
        assert dialog.mode.currentData() == "faults" and dialog.mode.count() == 5
        assert dialog.contact_button.isEnabled()
        dialog.output_path = tmp_path
        source = _checkpoint(tmp_path/"fault_checkpoint.npz")
        dialog._open_contact()
        assert recorded == [{"project_root": tmp_path, "checkpoint": source}, "exec"]
        recorded.clear()
        dialog.output_path = None
        dialog._open_contact()
        assert recorded == [{"project_root": tmp_path, "checkpoint": None}, "picker", "exec"]
        dialog._set_busy(True)
        assert not dialog.contact_button.isEnabled()
    finally:
        dialog.close()


def _fields(mesh, count=5):
    return {"damage": np.linspace(0, 1, mesh.cell_count),
            "seam_centers_xyz": mesh.centroids[:count],
            "seam_gap_m": np.tile([-.2, 2.], (count, 1)),
            "seam_slip_m": np.tile([-1., .5], (count, 1))}


@pytest.mark.parametrize("count", [0, 5])
def test_contact_map_draws_current_triangles_without_filling_gaps(tmp_path, monkeypatch, count):
    import matplotlib.figure
    import visualization.genesis_contact as plots
    mesh = build_icosphere(1)
    current = replace(mesh, vertices=mesh.vertices.copy())
    observed, labels = [], []
    original_polygons, savefig = plots.material_polygons, matplotlib.figure.Figure.savefig

    def polygons(actual):
        observed.append(actual)
        return original_polygons(actual)

    def output(figure, *args, **kwargs):
        labels.extend(text.get_text() for text in figure.texts)
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(plots, "material_polygons", polygons)
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", output)
    path = save_contact_snapshot(current, _fields(current, count), tmp_path/"contact.png", 1.4, 20.,
        [{"elapsed_years":0., "contact_dissipation_j":0., "equilibrium_residual":0.},
         {"elapsed_years":20., "contact_dissipation_j":1e10, "equilibrium_residual":1e-10}])
    assert observed == [current]
    assert "зафиксированы" in "\n".join(labels)
    with Image.open(path) as image:
        assert image.width >= 1500 and image.height >= 1000


@pytest.mark.parametrize("key", ["damage", "seam_centers_xyz", "seam_gap_m", "seam_slip_m"])
def test_contact_map_rejects_nonfinite_fields(tmp_path, key):
    mesh = build_icosphere(1)
    fields = _fields(mesh)
    fields[key].flat[0] = np.nan
    with pytest.raises(ValueError):
        save_contact_snapshot(mesh, fields, tmp_path/"bad.png", 1.4, 20.)
    assert not (tmp_path/"bad.png").exists()


def test_material_polygons_clip_antimeridian_and_keep_face_ownership():
    mesh = build_icosphere(1)
    polygons, owners = material_polygons(mesh)
    assert set(owners) == set(range(mesh.cell_count))
    assert all(np.isfinite(p).all() and np.max(np.abs(p[:, 0])) <= np.pi+1e-12 for p in polygons)
