"""Passive reference views retain exact material measure and mechanical anchors."""
from dataclasses import FrozenInstanceError, asdict, replace
from functools import lru_cache
import copy
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tests.test_geometric_bending_contact_dynamics import prepared
from tectonics.geometric_bending_contact_dynamics import _bending_signature
from tectonics.geometric_bending_reference import (
    current_bending_reference, validate_reference_rotation,
    bending_reference_to_dict, bending_reference_from_dict, _hash,
)
from tectonics.geometric_boundary import surface_digest
from tectonics.geometric_fault_motion_support import validate_reference_bending_bank_support
from tectonics.geometric_fault_traces import declare_fault_trace_set
from tectonics.geometric_surface import totals, audit_partition


@lru_cache(maxsize=1)
def fixture():
    surface, _, _, mechanics = prepared(mixed=True)
    identity = current_bending_reference(surface, mechanics)
    matrix = Rotation.from_rotvec((.21, -.37, .62)).as_matrix()
    return surface, mechanics, identity, current_bending_reference(surface, mechanics, matrix)


def _transform(view):
    dimension = len(view.stiffness)
    result = np.eye(dimension)
    for index in range(len(view.plate_ids)):
        result[3*index:3*index+3, 3*index:3*index+3] = view.rotation
    return result


def test_identity_view_is_original_geometry_with_zero_physical_duration():
    surface, mechanics, view, _ = fixture()
    assert view.surface is surface
    assert view.points == view.reference_points
    assert view.reference_surface_sha256 == view.current_surface_sha256 == surface_digest(surface)
    assert view.physical_dt_s == 0.
    assert view.plate_ids == mechanics.plate_ids
    assert view.reference_trace_sha256 == mechanics.fault_traces.trace_sha256
    assert view.reference_bending_sha256 == mechanics.bending_source_sha256


def test_rotated_material_coordinates_keep_time_age_ids_and_all_reservoirs_exact():
    surface, mechanics, original, current = fixture()
    before = asdict(surface), asdict(mechanics)
    assert current.surface.time_myr == surface.time_myr
    assert current.surface.radius_km == surface.radius_km
    assert totals(current.surface.fragments) == totals(surface.fragments)
    assert current.surface.known_material_ids == surface.known_material_ids
    for old, new in zip(surface.fragments, current.surface.fragments):
        assert old.fragment_id == new.fragment_id and old.parent_fragment_id == new.parent_fragment_id
        assert old.parcel == new.parcel
    assert current.current_surface_sha256 != current.reference_surface_sha256
    audit_partition(current.surface)
    assert (asdict(surface), asdict(mechanics)) == before


def test_rotated_quadrature_carries_original_labels_weights_areas_and_beam_stiffness_exactly():
    _, _, original, current = fixture()
    assert current.areas == original.areas
    assert current.stiffness == original.stiffness
    assert current.frames == original.frames
    assert current.offsets == original.offsets
    assert current.materials == original.materials
    assert current.reference_points == original.reference_points
    rotation = np.asarray(current.rotation)
    for old, new in zip(original.points, current.points):
        assert old.point_id == new.point_id
        assert old.line_weight_m == new.line_weight_m and old.area_m2 == new.area_m2
        for name in ("position", "horizontal", "normal", "down_dip"):
            np.testing.assert_array_equal(getattr(new, name), rotation@np.asarray(getattr(old, name)))


def test_anchor_covariance_and_virtual_work_preserve_original_configuration():
    _, _, original, current = fixture()
    transform = _transform(current)
    coordinates = np.linspace(-.002, .003, len(original.stiffness))
    rotated = transform@coordinates
    old_anchors = np.asarray(original.offsets)+np.asarray(original.operators)@coordinates
    new_anchors = np.asarray(current.offsets)+np.asarray(current.operators)@rotated
    np.testing.assert_allclose(new_anchors, old_anchors, rtol=3e-15, atol=2e-13)
    np.testing.assert_array_equal(current.rows, np.asarray(original.rows)@transform.T)
    np.testing.assert_array_equal(current.operators, np.asarray(original.operators)@transform.T)
    torque = np.linspace(-7e15, 11e15, len(original.stiffness))
    assert (transform@torque)@rotated == pytest.approx(torque@coordinates, rel=3e-15)
    old_energy = .5*coordinates@np.asarray(original.stiffness)@coordinates
    new_energy = .5*rotated@np.asarray(current.stiffness)@rotated
    assert old_energy == new_energy


def test_current_rows_match_independent_rotated_levers_and_beam_stiffness_oracle():
    surface, mechanics, _, current = fixture()
    radius = surface.radius_km*1000.
    count = 3*len(current.plate_ids)
    materials = {trace.trace_id: trace for trace in current.materials}
    for point, row in zip(current.points, current.rows):
        material = materials[point.trace_id]
        lever = radius*np.cross(point.position, point.horizontal)
        expected = np.zeros(len(row))
        sub, over = current.plate_ids.index(material.subducting_plate), current.plate_ids.index(material.overriding_plate)
        expected[3*sub:3*sub+3], expected[3*over:3*over+3] = -lever, lever
        np.testing.assert_allclose(row, expected, rtol=3e-15, atol=3e-11)
    stiffness = np.asarray(current.stiffness)
    for index, parameters in enumerate(mechanics.bending_parameters):
        # Original quarter-circle length is exact physical input of this fixture.
        length = radius*math.pi/2.
        for side, bank in enumerate((parameters.sub, parameters.over)):
            expected = radius**2*length*bank.young_modulus_pa*bank.elastic_thickness_m**3/(4.*(1.-bank.poisson_ratio**2)*bank.width_m**3)
            assert stiffness[count+2*index+side, count+2*index+side] == pytest.approx(expected, rel=4e-15)


def test_common_rotation_retains_complete_geographic_bank_support():
    _, mechanics, _, current = fixture()
    matrix = np.asarray(current.rotation)
    moved = tuple(replace(trace, start=tuple(matrix@trace.start), end=tuple(matrix@trace.end))
                  for trace in mechanics.fault_traces.traces)
    traces = declare_fault_trace_set(current.surface, moved, mechanics.fault_traces.quadrature)
    support = validate_reference_bending_bank_support(current.surface, traces, mechanics.bending_parameters)
    assert len(support.supports) == 2*len(current.points)
    assert all(item.fragment_intervals for item in support.supports)


def test_view_arrays_are_immutable_and_frame_change_cannot_write_original_history():
    _, _, _, current = fixture()
    with pytest.raises(FrozenInstanceError):
        current.physical_dt_s = 1.
    with pytest.raises(TypeError):
        current.areas[0] = 0.
    with pytest.raises(TypeError):
        current.operators[0][0][0] = 0.
    with pytest.raises(TypeError):
        current.stiffness[0][0] = 0.


@pytest.mark.parametrize("matrix", [
    (1., 0., 0.),
    np.diag((1., 1., -1.)),
    np.diag((1., 2., 1.)),
    [[True, 0., 0.], [0., 1., 0.], [0., 0., 1.]],
    [[float("nan"), 0., 0.], [0., 1., 0.], [0., 0., 1.]],
    [["1", "0", "0"], ["0", "1", "0"], ["0", "0", "1"]],
])
def test_rotation_requires_explicit_finite_proper_so3(matrix):
    with pytest.raises(ValueError, match="proper SO3"):
        validate_reference_rotation(matrix)


def test_reference_receipt_rebuilds_same_bits_without_renormalizing_saved_points():
    surface, mechanics, _, current = fixture()
    raw = bending_reference_to_dict(surface, mechanics, current)
    restored = bending_reference_from_dict(copy.deepcopy(raw), surface, mechanics)
    assert restored == current
    assert bending_reference_to_dict(surface, mechanics, restored) == raw


@pytest.mark.parametrize("mutate", [
    lambda raw: raw.update(physical_dt_s=1.),
    lambda raw: raw.update(physical_dt_s=False),
    lambda raw: raw.update(interpretation="physical_material_transport"),
    lambda raw: raw.update(reference_surface_sha256="0"*64),
    lambda raw: raw.update(reference_trace_sha256="0"*64),
    lambda raw: raw.update(reference_bending_sha256="0"*64),
    lambda raw: raw.update(current_surface_sha256="0"*64),
    lambda raw: raw.update(extra="unrecognized"),
])
def test_rehashed_reference_receipt_cannot_change_origin_schema_or_clock(mutate):
    surface, mechanics, _, current = fixture()
    raw = bending_reference_to_dict(surface, mechanics, current)
    mutate(raw)
    raw["payload_sha256"] = _hash({key: value for key, value in raw.items() if key != "payload_sha256"})
    with pytest.raises(ValueError):
        bending_reference_from_dict(raw, surface, mechanics)


def test_reference_rejects_changed_original_bank_parameters_and_point_area():
    surface, mechanics, _, _ = fixture()
    changed = tuple(replace(parameters, sub=replace(parameters.sub, width_m=20000.))
                    for parameters in mechanics.bending_parameters)
    with pytest.raises(ValueError, match="parameters"):
        current_bending_reference(surface, replace(mechanics, bending_parameters=changed))
    records = list(mechanics.states)
    identity, local = records[0]
    records[0] = (identity, replace(local, area_m2=2.*local.area_m2))
    with pytest.raises(ValueError):
        current_bending_reference(surface, replace(mechanics, states=tuple(records)))


def test_sparse_owner_view_uses_only_compact_actual_owner_coordinates():
    surface, mechanics, _, _ = fixture()
    mapping = {0: 7, 1: 10**9}
    sparse = replace(surface, fragments=tuple(replace(fragment,
        parcel=replace(fragment.parcel, plate=mapping[fragment.parcel.plate]))
        for fragment in surface.fragments))
    declared = tuple(replace(trace, subducting_plate=mapping[trace.subducting_plate],
        overriding_plate=mapping[trace.overriding_plate]) for trace in mechanics.fault_traces.traces)
    traces = declare_fault_trace_set(sparse, declared, mechanics.fault_traces.quadrature)
    bounds = tuple(getattr(mechanics, name) for name in (
        "max_rotation_radians", "max_tip_slope", "max_bending_strain", "max_deflection_radius_ratio"))
    seed = replace(mechanics, surface_sha256=surface_digest(sparse), plate_ids=(7, 10**9),
        fault_traces=traces, bending_source_sha256=_bending_signature(sparse, traces,
            mechanics.bending_parameters, bounds))
    view = current_bending_reference(sparse, seed, Rotation.from_rotvec((.3, -.1, .2)).as_matrix())
    assert view.plate_ids == (7, 10**9)
    assert len(view.stiffness) == 3*2+2*len(traces.traces)


def test_direct_view_and_owner_fields_reject_bool_aliases_of_zero():
    surface, mechanics, _, current = fixture()
    with pytest.raises(ValueError, match="immutable BendingReference"):
        bending_reference_to_dict(surface, mechanics, replace(current, physical_dt_s=False))
    with pytest.raises(ValueError, match="actual material owners"):
        current_bending_reference(surface, replace(mechanics, plate_ids=(False, 1)))


@lru_cache(maxsize=1)
def accepted_epoch():
    from tests.test_geometric_distributed_contact_invariants import world
    from tectonics.geometric_transport import advance_geometric_surface
    mesh, _, _ = world()
    surface, mechanics, _, _ = fixture()
    current = surface
    cumulative = np.eye(3)
    durations = (.001, .013)
    vectors = ((.23, -.41, .17), (-.37, .12, .46))
    for dt, vector in zip(durations, vectors):
        omega = np.tile(vector, (2, 1))
        result = advance_geometric_surface(mesh, current, omega, dt,
            birth_factory=lambda *args: (_ for _ in ()).throw(AssertionError("Common transport creates no births")))
        assert not result.births and not result.losses
        current = result.state
        cumulative = Rotation.from_rotvec(np.asarray(vector)*dt).as_matrix()@cumulative
    return current, cumulative, math.fsum(durations)


def test_supplied_accepted_epoch_carries_actual_iterative_geometry_ages_and_clock_bits():
    surface, mechanics, original, _ = fixture()
    accepted, rotation, elapsed = accepted_epoch()
    view = current_bending_reference(surface, mechanics, rotation,
        reference_elapsed_time_myr=elapsed, reference_surface=accepted, reference_roundoff_steps=2)
    assert view.surface is accepted
    assert view.reference_elapsed_time_myr == elapsed and view.reference_roundoff_steps == 2
    assert view.physical_dt_s == 0. and view.uses_supplied_reference_surface
    assert view.areas == original.areas and view.stiffness == original.stiffness
    assert view.interpretation != original.interpretation
    raw = bending_reference_to_dict(surface, mechanics, view)
    restored = bending_reference_from_dict(copy.deepcopy(raw), surface, mechanics)
    assert restored == view
    assert restored.surface == accepted
    assert raw["reference_surface"]["time_myr"] == accepted.time_myr
    assert bending_reference_to_dict(surface, mechanics, restored) == raw


@pytest.mark.parametrize("alter", ["rotation", "clock", "age", "reservoir", "id"])
def test_supplied_epoch_cannot_change_rigid_geometry_or_material_history(alter):
    surface, mechanics, _, _ = fixture()
    accepted, rotation, elapsed = accepted_epoch()
    if alter == "rotation":
        rotation = Rotation.from_rotvec((.1, -.2, .3)).as_matrix()
    elif alter == "clock":
        accepted = replace(accepted, time_myr=accepted.time_myr+.001)
    else:
        fragments = list(accepted.fragments)
        fragment = fragments[0]
        if alter == "age":
            fragment = replace(fragment, parcel=replace(fragment.parcel, age_myr=fragment.parcel.age_myr+1.))
        elif alter == "reservoir":
            fragment = replace(fragment, parcel=replace(fragment.parcel,
                density_excess_mass_kg=1.01*fragment.parcel.density_excess_mass_kg, specific_properties=()))
        else:
            fragment = replace(fragment, fragment_id="invented:id")
        fragments[0] = fragment
        accepted = replace(accepted, fragments=tuple(fragments))
    with pytest.raises(ValueError):
        current_bending_reference(surface, mechanics, rotation,
            reference_elapsed_time_myr=elapsed, reference_surface=accepted, reference_roundoff_steps=2)


@pytest.mark.parametrize("steps", [True, -1, 2.5, 10**18])
def test_reference_roundoff_budget_requires_bounded_explicit_iteration_count(steps):
    surface, mechanics, _, _ = fixture()
    with pytest.raises(ValueError):
        current_bending_reference(surface, mechanics, reference_roundoff_steps=steps)


def test_rehashed_supplied_surface_schema_is_strict_and_retains_original_source_hash():
    surface, mechanics, _, _ = fixture()
    accepted, rotation, elapsed = accepted_epoch()
    view = current_bending_reference(surface, mechanics, rotation,
        reference_elapsed_time_myr=elapsed, reference_surface=accepted, reference_roundoff_steps=2)
    raw = bending_reference_to_dict(surface, mechanics, view)
    raw["reference_surface"]["fragments"][0]["unexpected"] = 0.
    raw["payload_sha256"] = _hash({key: value for key, value in raw.items() if key != "payload_sha256"})
    with pytest.raises(ValueError, match="fragment schema"):
        bending_reference_from_dict(raw, surface, mechanics)
