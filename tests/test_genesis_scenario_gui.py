"""Main scenario routing must never silently start the wrong model."""
import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
import pytest

from moon_gui.app import MATURE_SCENARIO_ID, MoonWindow
from moon_gui.genesis_schema import SatelliteOrigin


@pytest.fixture
def window():
    app = QApplication.instance() or QApplication([])
    result = MoonWindow()
    yield result
    result.controller._set_state("Stopped")
    result.close()
    app.processEvents()


def select(window, identifier):
    index = window.scenario.findData(identifier)
    assert index >= 0
    window.scenario.setCurrentIndex(index)


def test_only_mature_and_disk_quiet_are_available_with_stable_ids(window):
    assert [window.scenario.itemData(i) for i in range(window.scenario.count())] == [
        MATURE_SCENARIO_ID, *(origin.value for origin in SatelliteOrigin)]
    assert [window.scenario.model().item(i).isEnabled() for i in range(4)] == [True, True, False, False]
    assert window.scenario.currentData() == MATURE_SCENARIO_ID
    assert "(стартер)" in window.scenario.itemText(1)
    tip_a = window.scenario.itemData(1, Qt.ItemDataRole.ToolTipRole)
    assert "Аккреция в диске не рассчитывается" in tip_a
    tip_b = window.scenario.itemData(2, Qt.ItemDataRole.ToolTipRole)
    tip_c = window.scenario.itemData(3, Qt.ItemDataRole.ToolTipRole)
    assert "импакта" in tip_b and "циркуляризации" in tip_c
    assert tip_b != tip_c


def test_selection_only_changes_controls_without_launching(window, monkeypatch):
    calls = []
    monkeypatch.setattr(window, "_open_genesis_starter", lambda: calls.append("starter"))
    monkeypatch.setattr(window.controller, "start", lambda spec: calls.append("mature"))
    old_values = (window.end_time.value(), window.dt.value(), window.subdivisions.currentText())
    select(window, SatelliteOrigin.DISK_QUIET.value)
    assert not calls
    assert window.start_button.text() == "Открыть стартер…"
    assert window.start_button.isEnabled()
    assert not window.end_time.isEnabled() and not window.dt.isEnabled()
    assert not window.cpu_mode.isEnabled() and not window.resume_field.isEnabled()
    assert not window.config_field.isEnabled() and not window.output_field.isEnabled()
    assert "отдельном окне" in window.scenario_explanation.text()
    assert window.genesis_button.isEnabled() and window.genesis_starter_button.isEnabled()
    select(window, MATURE_SCENARIO_ID)
    assert not calls
    assert window.start_button.text() == "Запустить прогон"
    assert window.end_time.isEnabled() and window.dt.isEnabled()
    assert window.cpu_mode.isEnabled() and window.resume_field.isEnabled()
    assert old_values == (window.end_time.value(), window.dt.value(), window.subdivisions.currentText())


def test_disk_quiet_start_routes_before_mature_spec_or_checkpoint_validation(window, monkeypatch):
    calls = []
    monkeypatch.setattr(window, "_open_genesis_starter", lambda: calls.append("starter"))
    monkeypatch.setattr(window, "_make_spec", lambda: pytest.fail("Starter must not construct a mature RunSpec"))
    monkeypatch.setattr(window.controller, "start", lambda spec: pytest.fail("Starter cannot start the mature controller"))
    window.resume_field.set_path("missing-mature-checkpoint")
    window.config_field.set_path("missing-mature-config.yaml")
    select(window, SatelliteOrigin.DISK_QUIET.value)
    window.start_button.click()
    assert calls == ["starter"]


@pytest.mark.parametrize("origin", [SatelliteOrigin.DISK_IMPACT, SatelliteOrigin.CAPTURE_CIRCULARIZATION])
def test_unimplemented_origins_never_fall_through_to_mature_even_programmatically(window, monkeypatch, origin):
    monkeypatch.setattr(window, "_make_spec", lambda: pytest.fail("Planned scenario cannot construct a mature RunSpec"))
    monkeypatch.setattr(window, "_open_genesis_starter", lambda: pytest.fail("Planned origin cannot silently substitute scenario A"))
    monkeypatch.setattr(window.controller, "start", lambda spec: pytest.fail("Planned scenario cannot run"))
    select(window, origin.value)
    assert not window.start_button.isEnabled()
    assert "пока в плане" in window.scenario_explanation.text()
    window._start_run()


@pytest.mark.parametrize("state", ["Preparing", "Running", "Pausing", "Paused", "Stopping"])
def test_running_and_paused_runs_lock_scenario_and_reject_programmatic_switch(window, monkeypatch, state):
    monkeypatch.setattr(window, "_make_spec", lambda: pytest.fail("Locked run must not be restarted"))
    monkeypatch.setattr(window, "_open_genesis_starter", lambda: pytest.fail("Locked run cannot open starter"))
    window.controller._set_state(state)
    assert not window.scenario.isEnabled()
    assert not window.start_button.isEnabled()
    select(window, SatelliteOrigin.DISK_QUIET.value)
    assert window.scenario.currentData() == MATURE_SCENARIO_ID
    window._start_run()
    window.controller._set_state("Stopped")
    assert window.scenario.isEnabled()
    assert window.start_button.text() == "Запустить прогон"
    assert window.start_button.isEnabled()


def test_mature_start_retains_its_existing_dispatch(window, monkeypatch, tmp_path):
    calls = []
    spec = SimpleNamespace(assignment_optimized=False, resume_checkpoint=None,
                           output_dir=tmp_path / "fresh", subdivisions=3)
    spec.normalized = lambda: spec
    monkeypatch.setattr(window, "_make_spec", lambda: spec)
    monkeypatch.setattr(window, "_open_genesis_starter", lambda: pytest.fail("Mature option cannot open starter"))
    monkeypatch.setattr(window.controller, "start", lambda value: calls.append(value))
    window.start_button.click()
    assert calls == [spec]
