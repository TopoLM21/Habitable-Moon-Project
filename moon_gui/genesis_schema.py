"""Stable identifiers for the planned pre-v0.31 genesis experiments.

This module deliberately contains no genesis physics.  It keeps satellite
origin and plate-onset mechanism as independent experimental axes, so the GUI
and future runners can share identifiers without coupling either axis to a
particular numerical implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product


class SatelliteOrigin(str, Enum):
    DISK_QUIET = "disk_quiet"
    DISK_IMPACT = "disk_impact"
    CAPTURE_CIRCULARIZATION = "capture_circularization"


class PlateOnsetMode(str, Enum):
    STAGNANT_LID_CONTROL = "stagnant_lid_control"
    CONVECTIVE_OVERSTRESS = "convective_overstress"
    IMPACT_TRIGGERED = "impact_triggered"
    TIDE_ASSISTED = "tide_assisted"
    HYBRID_DAMAGE = "hybrid_damage"


class GenesisPhase(str, Enum):
    INITIAL_CONDITIONS = "initial_conditions"
    SPIN_ORBIT_EVOLUTION = "spin_orbit_evolution"
    MAGMA_OCEAN_COOLING = "magma_ocean_cooling"
    LID_FORMATION = "lid_formation"
    PLATE_ONSET = "plate_onset"
    V031_HANDOFF = "v031_handoff"


ORIGIN_LABELS_RU = {
    SatelliteOrigin.DISK_QUIET: "Генезис A — диск / спокойно",
    SatelliteOrigin.DISK_IMPACT: "Генезис B — диск / импакт",
    SatelliteOrigin.CAPTURE_CIRCULARIZATION: "Генезис C — захват / циркуляризация",
}

PLATE_ONSET_LABELS_RU = {
    PlateOnsetMode.STAGNANT_LID_CONTROL: "застойная покрышка (контроль)",
    PlateOnsetMode.CONVECTIVE_OVERSTRESS: "конвективное разрушение",
    PlateOnsetMode.IMPACT_TRIGGERED: "запуск импактом",
    PlateOnsetMode.TIDE_ASSISTED: "приливно-облегчённый запуск",
    PlateOnsetMode.HYBRID_DAMAGE: "гибридное накопление повреждений",
}


@dataclass(frozen=True, slots=True)
class GenesisExperiment:
    origin: SatelliteOrigin
    plate_onset: PlateOnsetMode

    @property
    def key(self) -> str:
        return f"{self.origin.value}__{self.plate_onset.value}"


def experiment_matrix() -> tuple[GenesisExperiment, ...]:
    """Return the explicit 3 x 5 hypothesis matrix (15 experiments)."""

    return tuple(
        GenesisExperiment(origin, onset)
        for origin, onset in product(SatelliteOrigin, PlateOnsetMode)
    )


GENESIS_PHASE_ORDER = tuple(GenesisPhase)
