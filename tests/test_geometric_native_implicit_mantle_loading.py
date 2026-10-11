"""Implicit donor5 geometry, material correlation, and stable physical loading."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path

import pytest

from tectonics.geometric_native_first_entry_material import native_first_entry_material_from_dict
from tectonics.geometric_native_implicit_material_epoch import (
    initialize_native_implicit_material_epoch, native_implicit_material_epoch_from_dict,
    native_implicit_material_epoch_to_dict, transport_native_implicit_material_epoch,
)
from tectonics.geometric_native_implicit_mantle_loading import (
    _operators, assemble_native_implicit_physical_spatial_mantle_load,
    native_implicit_spatial_mantle_load_from_dict, native_implicit_spatial_mantle_load_to_dict,
)
from tectonics.geometric_native_pose import FractionInterval as I
from tectonics.geometric_spatial_mantle_loading import PrescribedSpatialMantleSource
from tectonics.mesh import build_icosphere


@pytest.fixture(scope="module")
def actual_case():
    root = Path(__file__).resolve().parents[1]
    directory = root/"analysis/geometric_native_first_entry_validation/control_release_02"
    initial = json.loads((directory/"original_force/initial.json").read_bytes())
    material = native_first_entry_material_from_dict(json.loads((directory/"material_state.json").read_bytes()),
        expected_original_mechanics_payload=initial["native_underthrust_motion"]["initial_mechanics"])
    epoch = initialize_native_implicit_material_epoch(material)
    raw_source = json.loads((directory/"original_force/native_source.json").read_bytes())["source"]
    raw_source["angular_velocity_rad_s"] = tuple(raw_source["angular_velocity_rad_s"])
    raw_source["amplitude_gradient"] = tuple(raw_source["amplitude_gradient"])
    source = PrescribedSpatialMantleSource(**raw_source)
    mesh = build_icosphere(0)
    load = assemble_native_implicit_physical_spatial_mantle_load(mesh, epoch, source)
    return mesh, material, epoch, source, load


def test_all_originals_hot_births_and_tiny_donor_five_preserved(actual_case):
    _, material, epoch, _, load = actual_case
    originals = tuple(row for row in epoch.components if row.material_kind == "original_live")
    hot = tuple(row for row in epoch.components if row.material_kind == "hot_birth")
    assert len(originals) == 8
    assert len(hot) == sum(row.newborn is not None for row in material.endpoint_vacancies)
    assert epoch.original_parent_allocations == material.parent_allocations
    assert len(epoch.paired_corrections) == 1
    correction = epoch.paired_corrections[0]
    assert correction.area_bounds_sr.lo > 0
    assert float(correction.area_bounds_sr.hi) < 1e-70
    assert len(load.correction_cell_measures) == 1
    local = load.correction_cell_measures[0]
    assert local.area_bounds_sr == correction.area_bounds_sr
    original = next(row for row in load.material_cell_measures
        if row.component_id == local.negative_component_id and row.cell == local.cell)
    newborn = next(row for row in load.material_cell_measures
        if row.component_id == local.positive_component_id and row.cell == local.cell)
    assert original.signed_correction_bounds_sr == -newborn.signed_correction_bounds_sr
    assert original.owner == newborn.owner == local.owner
    assert original.correlation_ids == newborn.correlation_ids == (local.correction_id,)


def test_every_owner_cell_measure_covers_complete_native_cell(actual_case):
    mesh, _, epoch, _, load = actual_case
    assert load.plate_ids == (0, 1)
    for cell in range(len(mesh.faces)):
        total = sum((area for index, _, area in load.owner_cell_area_bounds_m2 if index == cell), I(0))
        expected = float(mesh.areas_unit_sphere[cell])*(epoch.radius_km*1000.)**2
        assert abs(float(total.midpoint)-expected) <= expected*2e-14
    assert all(bound.lo <= Fraction.from_float(value) <= bound.hi
               for value, bound in zip(load.torque_nm, load.torque_bounds_nm))


def test_exact_operator_evaluation_does_not_create_transverse_cancellation_torque():
    source = PrescribedSpatialMantleSource((0., 0., 1e-12), 1e14, 1., (0., 0., 0.), "test")
    positions = ((.6, 0., .8), (-.6, 0., .8))
    _, torque, _, bounds = _operators((0,), ((0, 0, I(10**11)), (1, 0, I(10**11))),
        source, 100000., positions)
    assert torque[0] == torque[1] == 0.
    assert bounds[0] == bounds[1] == I(0)
    assert torque[2] > 0.


def test_epoch_and_load_replay_reject_rehashed_material_and_literal_type_changes(actual_case):
    mesh, material, epoch, source, load = actual_case
    epoch_raw = native_implicit_material_epoch_to_dict(epoch, include_source=False)
    assert native_implicit_material_epoch_from_dict(epoch_raw,
        expected_source_material_state=material) == epoch
    forged = deepcopy(epoch_raw)
    forged["components"][0]["owner"] = False
    with pytest.raises(ValueError):
        native_implicit_material_epoch_from_dict(forged, expected_source_material_state=material)
    load_raw = native_implicit_spatial_mantle_load_to_dict(mesh, epoch, load)
    assert native_implicit_spatial_mantle_load_from_dict(load_raw, mesh, epoch,
        expected_source=source) == load
    forged = deepcopy(load_raw)
    forged["correction_cell_measures"].clear()
    with pytest.raises(ValueError):
        native_implicit_spatial_mantle_load_from_dict(forged, mesh, epoch, expected_source=source)


def test_differential_so3_trial_requires_new_archive_and_vacancy_closure(actual_case):
    mesh, _, epoch, source, _ = actual_case
    moved = transport_native_implicit_material_epoch(epoch,
        {0:(1e-7, 2e-7, 3e-7), 1:(-1e-7, 2e-7, 3e-7)}, .001)
    assert moved.phase == "unresolved_free_transport"
    assert all(len(history) == 1 for _, history in moved.additional_owner_histories)
    with pytest.raises(ValueError, match="archive/vacancy closure"):
        assemble_native_implicit_physical_spatial_mantle_load(mesh, moved, source)


def test_common_so3_pose_keeps_complete_ordered_history_and_material_state(actual_case):
    _, material, epoch, _, _ = actual_case
    moved = transport_native_implicit_material_epoch(epoch,
        {0:(1e-7, 2e-7, 3e-7), 1:(1e-7, 2e-7, 3e-7)}, .001)
    assert moved.phase == "partition"
    assert moved.source_material_state == material
    assert moved.paired_corrections[0].containing_chart.pose[-1].omega_rad_per_myr == (1e-7, 2e-7, 3e-7)
    assert moved.time_myr > epoch.time_myr
    assert moved.force_admission_authorized is False
