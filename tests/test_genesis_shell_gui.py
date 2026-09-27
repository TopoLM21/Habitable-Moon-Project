"""Spatial experiment selection, result compatibility, and real mesh-edge maps."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PIL import Image
from PySide6.QtCore import QProcess
from PySide6.QtWidgets import QApplication
import pytest

from moon_gui.genesis_dialog import GenesisDialog, ROOT
from tectonics.mesh import build_icosphere
from visualization.genesis_shell import crack_segments, save_shell_animation, save_shell_snapshot


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def dialog(app, tmp_path):
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "genesis_moon.yaml").write_bytes((ROOT / "configs/genesis_moon.yaml").read_bytes())
    (tmp_path / "run_genesis_shell.py").touch()
    widget = GenesisDialog(project_root=tmp_path)
    yield widget
    widget.close()


def test_shell_default_and_thermal_switch_preserve_numeric_precision(dialog, tmp_path):
    assert dialog.mode.currentData() == "shell"
    args = dialog.arguments(tmp_path / "result")
    assert Path(args[1]).name == "run_genesis_shell.py"
    assert args[args.index("--subdivisions") + 1] == "3"
    assert float(args[args.index("--shell-step-myr") + 1]) == 0.002
    assert float(args[args.index("--duration-myr") + 1]) == 3
    assert "--stellar-flux-w-m2" not in args
    assert "--water-volume-km3" not in args
    assert "--initial-temperature-k" not in args
    assert "--convective-traction-mpa" not in args
    assert dialog.convective_traction.value() == 0.02
    dialog.convective_traction.setValue(0.05)
    args = dialog.arguments(tmp_path / "result")
    assert float(args[args.index("--convective-traction-mpa") + 1]) == 0.05
    dialog.intact_control.setChecked(True)
    dialog.subdivisions.setCurrentIndex(2)
    dialog.shell_step.setValue(0.001)
    args = dialog.arguments(tmp_path / "result")
    assert args[args.index("--control") + 1] == "intact"
    assert "--convective-traction-mpa" not in args
    assert args[args.index("--subdivisions") + 1] == "4"
    assert float(args[args.index("--shell-step-myr") + 1]) == 0.001
    dialog.mode.setCurrentIndex(dialog.mode.findData("thermal"))
    args = dialog.arguments(tmp_path / "thermal")
    assert Path(args[1]).name == "run_genesis.py"
    assert "--subdivisions" not in args
    assert "--shell-step-myr" not in args
    assert "--control" not in args
    assert "--convective-traction-mpa" not in args
    assert not dialog.shell_step.isEnabled()
    assert not dialog.intact_control.isEnabled()
    dialog.mode.setCurrentIndex(dialog.mode.findData("shell"))
    assert dialog.shell_step.isEnabled()
    dialog._set_busy(True)
    assert not dialog.mode.isEnabled()
    assert not dialog.shell_step.isEnabled()
    dialog._set_busy(False)
    assert dialog.mode.isEnabled()
    assert dialog.shell_step.isEnabled()


def test_missing_shell_runner_defaults_to_thermal(app, tmp_path):
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "genesis_moon.yaml").write_bytes((ROOT / "configs/genesis_moon.yaml").read_bytes())
    dialog = GenesisDialog(project_root=tmp_path)
    assert dialog.mode.currentData() == "thermal"
    assert Path(dialog.arguments(tmp_path / "result")[1]).name == "run_genesis.py"
    dialog.close()


def _write_preview(path, color="red"):
    Image.new("RGB", (80, 40), color).save(path)


@pytest.mark.parametrize("new_summary", [False, True])
def test_summary_and_available_image_fallback(dialog, tmp_path, new_summary):
    output = tmp_path / "results"
    output.mkdir()
    summary = {"events_myr": {"ocean_start": 1.05}, "status": "completed", "final": {}}
    if new_summary:
        summary["shell"] = {"first_fracture_time_myr": 1.2, "failed_edge_fraction": 0.08,
                            "cracked_length_km": 100, "intact_region_count": 2,
                            "damaged_area_fraction": 0.08}
    (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    _write_preview(output / "genesis_history.png")
    dialog.output_path = output
    dialog._finished(0, QProcess.ExitStatus.NormalExit)
    assert "1.050" in dialog.status.text()
    assert dialog.result_view.count() == 1
    assert not dialog._pixmap.isNull()
    if new_summary:
        assert "1.200" in dialog.status.text()
        assert "8.0%" in dialog.status.text()
    _write_preview(output / "genesis_shell.png", "blue")
    Image.new("RGB", (80, 40), "green").save(output / "genesis_shell.gif")
    dialog._load_previews(output)
    assert dialog.result_view.count() == 3
    assert dialog.result_view.currentData().endswith("genesis_shell.png")
    dialog.result_view.setCurrentIndex(1)
    assert dialog._movie is not None and dialog._movie.isValid()
    dialog.result_view.setCurrentIndex(2)
    assert dialog._movie is None
    assert not dialog._pixmap.isNull()


def test_missing_summary_does_not_hide_available_maps(dialog, tmp_path):
    _write_preview(tmp_path / "genesis_shell.png")
    dialog.output_path = tmp_path
    dialog._finished(0, QProcess.ExitStatus.NormalExit)
    assert "Не удалось прочитать" in dialog.status.text()
    assert dialog.result_view.count() == 1
    assert not dialog._pixmap.isNull()
    dialog._load_previews(tmp_path / "missing")
    assert not dialog.result_view.isEnabled()
    assert dialog._pixmap.isNull()


@pytest.mark.parametrize("status,expected,absent", [
    ("shell_small_strain_limit", "предел малых деформаций", "замерзание"),
    ("surface_reached_freezing_limit_ice_not_modelled", "замерзание воды", "малых деформаций"),
    ("future_stop_reason", "future_stop_reason", "замерзание"),
])
def test_stop_reason_distinguishes_geometry_from_thermal_limit(dialog, tmp_path, status, expected, absent):
    summary = {"status": status, "events_myr": {}, "shell": {"first_fracture_time_myr": None}}
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    dialog.output_path = tmp_path
    dialog._finished(0, QProcess.ExitStatus.NormalExit)
    assert expected in dialog.status.text()
    assert absent not in dialog.status.text()


def test_cracks_use_actual_edge_vertices_and_split_antimeridian():
    lon = np.deg2rad([179.0, -179.0])
    vertices = np.column_stack((np.cos(lon), np.sin(lon), np.zeros(2)))
    mesh = SimpleNamespace(vertices=vertices, shared_edges=((11, 12, 0, 1),))
    segments = crack_segments(mesh, [True])
    assert len(segments) == 2
    assert segments[0][0, 0] == pytest.approx(lon[0])
    assert segments[-1][-1, 0] == pytest.approx(lon[-1])
    assert segments[0][-1, 0] == pytest.approx(np.pi)
    assert segments[1][0, 0] == pytest.approx(-np.pi)
    assert all(np.abs(np.diff(segment[:, 0])).max() < np.deg2rad(3) for segment in segments)
    assert all(np.isfinite(segment).all() for segment in segments)
    assert crack_segments(mesh, [False]) == []
    with pytest.raises(ValueError):
        crack_segments(mesh, [])


def test_four_maps_render_and_animation_bounds_frame_count(tmp_path):
    mesh = build_icosphere(1)
    fields = {"temperature_k": np.linspace(400, 2300, mesh.cell_count),
              "lid_thickness_km": np.linspace(0, 40, mesh.cell_count),
              "tensile_stress_mpa": np.linspace(0, 70, mesh.cell_count),
              "damage": np.linspace(0, 1, mesh.cell_count),
              "failed_edges": np.arange(len(mesh.shared_edges)) % 5 == 0}
    snapshot = save_shell_snapshot(mesh, fields, tmp_path / "map.png", 1.5,
                                   {"first_fracture_time_myr": 1.1, "intact_region_count": 3})
    with Image.open(snapshot) as image:
        assert image.width >= 1000 and image.height >= 600
    frames = []
    for i in range(7):
        frame = tmp_path / f"frame_{i}.png"
        Image.new("RGB", (20, 10), (i * 30, 0, 0)).save(frame)
        frames.append(frame)
    gif = save_shell_animation(frames, tmp_path / "movie.gif", max_frames=3)
    with Image.open(gif) as image:
        assert image.n_frames == 3
        image.seek(0)
        assert image.convert("RGB").getpixel((0, 0)) == (0, 0, 0)
        image.seek(2)
        assert image.convert("RGB").getpixel((0, 0)) == (180, 0, 0)
