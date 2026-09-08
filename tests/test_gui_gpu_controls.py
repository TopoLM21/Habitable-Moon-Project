"""GPU GUI contracts: no CUDA initialization or real model work required."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys

import pytest

from moon_gui.backend import RunSpec


@pytest.fixture
def gpu_window(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    import moon_gui.app as gui

    application = QApplication.instance() or QApplication([])
    monkeypatch.setattr(gui, "install_application_font", lambda _: None)
    window = gui.MoonWindow()
    yield window
    window.controller._set_state("Stopped")
    window.close()
    application.processEvents()


def test_default_mode_remains_optimized_cpu_with_new_options_off(gpu_window):
    window = gpu_window
    assert window.cpu_mode.currentIndex() == 1
    assert window.cpu_mode.itemData(0) is False
    assert window.cpu_mode.itemData(1) is True
    assert window.cpu_mode.itemData(2) == "gpu_surface"
    spec = window._make_spec()
    assert spec.cpu_optimized
    assert not spec.gpu_surface
    assert not spec.assignment_columns
    assert not spec.boundary_forces
    assert window.gpu_device.isHidden()
    assert window.gpu_device_label.isHidden()
    assert not window.gpu_device.isEnabled()
    assert window.assignment_columns.isEnabled()
    assert window.boundary_forces.isEnabled()


@pytest.mark.parametrize("mode", [1, 2])
@pytest.mark.parametrize("assignment,boundary", [(False, False), (True, False), (False, True), (True, True)])
def test_gpu_and_cpu_options_are_independent(gpu_window, mode, assignment, boundary):
    window = gpu_window
    window.cpu_mode.setCurrentIndex(mode)
    window.gpu_device.setValue(3)
    window.assignment_columns.setChecked(assignment)
    window.boundary_forces.setChecked(boundary)
    spec = window._make_spec()
    assert spec.cpu_optimized
    assert spec.gpu_surface is (mode == 2)
    assert spec.gpu_device == (3 if mode == 2 else 0)
    assert spec.assignment_columns is assignment
    assert spec.boundary_forces is boundary
    assert spec.render_workers == 4
    assert spec.cell_kernels
    assert window.gpu_device.isHidden() is (mode != 2)
    assert window.gpu_device.isEnabled() is (mode == 2)
    assert window.gpu_device.minimum() == 0
    assert window.gpu_device.maximum() == 31


def test_reference_cpu_does_not_receive_retained_optimization_selections(gpu_window):
    window = gpu_window
    window.cpu_mode.setCurrentIndex(2)
    window.assignment_columns.setChecked(True)
    window.boundary_forces.setChecked(True)
    window.gpu_device.setValue(2)
    window.cpu_mode.setCurrentIndex(0)
    spec = window._make_spec()
    assert not spec.cpu_optimized
    assert not spec.gpu_surface
    assert spec.gpu_device == 0
    assert not spec.assignment_columns
    assert not spec.boundary_forces
    assert not spec.cell_kernels
    assert spec.render_workers == 1
    assert spec.process_priority == "normal"
    assert not window.assignment_columns.isEnabled()
    assert not window.boundary_forces.isEnabled()
    assert not window.gpu_device.isEnabled()
    assert window.gpu_device.isHidden()


@pytest.mark.parametrize("state", ["Preparing", "Running", "Pausing", "Paused", "Stopping"])
def test_new_options_locked_for_the_entire_run(gpu_window, state):
    window = gpu_window
    window.cpu_mode.setCurrentIndex(2)
    window.controller._set_state(state)
    for widget in (
        window.cpu_mode, window.gpu_device, window.assignment_columns, window.boundary_forces,
    ):
        assert not widget.isEnabled()
    window.controller._set_state("Stopped")
    for widget in (
        window.cpu_mode, window.gpu_device, window.assignment_columns, window.boundary_forces,
    ):
        assert widget.isEnabled()


def _spec(tmp_path):
    return RunSpec(
        tmp_path / "project", tmp_path / "config.yaml", tmp_path / "results",
        cpu_optimized=True, gpu_surface=True,
    )


def test_gpu_child_uses_per_run_caches_and_its_own_interpreter_environment(tmp_path, monkeypatch):
    from moon_gui.app import simulation_environment

    for variable in ("CUPY_CACHE_DIR", "CUDA_CACHE_PATH", "MPLCONFIGDIR", "PYTHONPATH"):
        monkeypatch.delenv(variable, raising=False)
    spec = _spec(tmp_path)
    environment = simulation_environment(spec)
    for variable, folder in (
        ("CUPY_CACHE_DIR", "cupy"), ("CUDA_CACHE_PATH", "cuda"), ("MPLCONFIGDIR", "matplotlib"),
    ):
        cache = spec.output_dir / ".cache" / folder
        assert environment.value(variable) == str(cache)
        assert cache.is_dir()
        assert variable not in os.environ
    assert environment.value("PYTHONPATH") == str(spec.project_root)
    assert environment.value("PYTHONUNBUFFERED") == "1"
    assert environment.value("MPLBACKEND") == "Agg"


def test_explicit_cache_and_pythonpath_settings_are_preserved(tmp_path, monkeypatch):
    from moon_gui.app import simulation_environment

    explicit = {
        "CUPY_CACHE_DIR": str(tmp_path / "my_cupy"),
        "CUDA_CACHE_PATH": str(tmp_path / "my_cuda"),
        "MPLCONFIGDIR": str(tmp_path / "my_mpl"),
        "PYTHONPATH": str(tmp_path / "user_modules"),
    }
    for variable, value in explicit.items():
        monkeypatch.setenv(variable, value)
    spec = _spec(tmp_path)
    environment = simulation_environment(spec)
    for variable in ("CUPY_CACHE_DIR", "CUDA_CACHE_PATH", "MPLCONFIGDIR"):
        assert environment.value(variable) == explicit[variable]
    assert environment.value("PYTHONPATH") == str(spec.project_root) + os.pathsep + explicit["PYTHONPATH"]
    assert not spec.output_dir.exists()


def test_cpu_does_not_allocate_gpu_caches(tmp_path, monkeypatch):
    from moon_gui.app import simulation_environment

    for variable in ("CUPY_CACHE_DIR", "CUDA_CACHE_PATH", "MPLCONFIGDIR"):
        monkeypatch.delenv(variable, raising=False)
    spec = replace(_spec(tmp_path), gpu_surface=False)
    environment = simulation_environment(spec)
    assert not spec.output_dir.exists()
    assert not environment.contains("CUPY_CACHE_DIR")
    assert not environment.contains("CUDA_CACHE_PATH")


def test_gpu_child_error_is_logged_and_not_restarted_on_cpu(gpu_window, tmp_path, monkeypatch):
    from moon_gui.app import QProcess, SimulationController

    class FailedProcess:
        def readAllStandardOutput(self):
            return b"CUDA device unavailable\n"

    controller = SimulationController()
    controller.spec = _spec(tmp_path)
    controller.process = FailedProcess()
    logs = []
    failures = []
    restarts = []
    controller.log_line.connect(logs.append)
    controller.run_failed.connect(failures.append)
    monkeypatch.setattr(controller, "_start_next_segment", lambda: restarts.append(True))
    controller._process_finished(1, QProcess.ExitStatus.NormalExit)
    assert controller.state == "Error"
    assert logs == ["CUDA device unavailable"]
    assert len(failures) == 1
    assert "Журнал" in failures[0]
    assert not restarts
    assert controller.spec.gpu_surface


def test_opening_gpu_controls_does_not_import_cupy_or_the_model():
    project = Path(__file__).resolve().parents[1]
    script = r'''
import builtins
original_import = builtins.__import__
attempted_imports = []
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'cupy', 'tectonics', 'run_long_evolution_v131_gpu'}:
        attempted_imports.append(name)
        raise AssertionError('GUI must not import model/CUDA: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from PySide6.QtWidgets import QApplication
from moon_gui.app import MoonWindow
application = QApplication([])
window = MoonWindow()
window.cpu_mode.setCurrentIndex(2)
assert window._make_spec().gpu_surface
assert not attempted_imports, attempted_imports
window.close()
application.processEvents()
'''
    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=project, env=environment,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
