from __future__ import annotations

from moon_gui.genesis_schema import (
    GENESIS_PHASE_ORDER,
    ORIGIN_LABELS_RU,
    GenesisExperiment,
    GenesisPhase,
    PlateOnsetMode,
    SatelliteOrigin,
    experiment_matrix,
)


def test_genesis_axes_form_explicit_three_by_five_matrix() -> None:
    experiments = experiment_matrix()
    assert len(experiments) == 15
    assert len({item.key for item in experiments}) == 15
    assert {item.origin for item in experiments} == set(SatelliteOrigin)
    assert {item.plate_onset for item in experiments} == set(PlateOnsetMode)


def test_genesis_identifiers_and_phase_order_are_stable() -> None:
    experiment = GenesisExperiment(
        SatelliteOrigin.CAPTURE_CIRCULARIZATION,
        PlateOnsetMode.TIDE_ASSISTED,
    )
    assert experiment.key == "capture_circularization__tide_assisted"
    assert ORIGIN_LABELS_RU[SatelliteOrigin.DISK_QUIET].startswith("Генезис A")
    assert GENESIS_PHASE_ORDER[0] is GenesisPhase.INITIAL_CONDITIONS
    assert GENESIS_PHASE_ORDER[-1] is GenesisPhase.V031_HANDOFF
