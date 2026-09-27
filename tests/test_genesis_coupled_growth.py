"""Material-front activation and topology contracts for growing contacts."""
from copy import deepcopy
from dataclasses import fields, is_dataclass, replace

import numpy as np
import pytest

from tectonics.genesis import ENERGY_SCALE, mantle_enthalpy, surface_enthalpy
from tectonics.genesis_contact_growth import aggregate_cohorts, evaluate_cohorts
from tectonics.genesis_coupled import CoupledModel, CoupledParameters
from tectonics.genesis_faults import save_fault_checkpoint
from tectonics.genesis_material import face_frames
from tectonics.genesis_mobile import _RetryStep
from tectonics.genesis_shell import rock_enthalpy, rock_temperature
from test_genesis_contact import _source_bytes


@pytest.fixture(scope="module")
def partial_source(tmp_path_factory):
    """Conserved linear geotherm with a genuine half-solid material column."""
    path = tmp_path_factory.mktemp("coupled_growth_source")
    _, source, (state, thermal, orbit), _ = _source_bytes(path)
    ts, tm = source.thermal.solidus_k-400., source.thermal.solidus_k+400.
    energy = [mantle_enthalpy(tm, source.thermal)/ENERGY_SCALE,
              surface_enthalpy(ts, source.thermal)/ENERGY_SCALE, 0., 0.]
    energy[3] = thermal.initial_total_energy-sum(energy[:3])
    thermal = replace(thermal, energy=energy)
    z = (np.arange(source.p.column_layers)+.5)/source.p.column_layers
    h = np.broadcast_to(rock_enthalpy(ts+(tm-ts)*z, source.thermal), state.column_enthalpy.shape).copy()
    state = replace(state, column_enthalpy=h,
                    boundary_energy_j=float(np.sum(state.layer_mass_kg*h))-state.initial_column_energy_j)
    target = path/"partial_fault.npz"
    save_fault_checkpoint(target, source, state, thermal, orbit, {}, {"test": "partial growth source"})
    return target.read_bytes()


def _same(a, b, omit=()):
    for item in fields(a):
        if item.name in omit:
            continue
        x, y = getattr(a, item.name), getattr(b, item.name)
        if isinstance(x, np.ndarray):
            np.testing.assert_array_equal(x, y, err_msg=item.name)
        elif is_dataclass(x):
            _same(x, y)
        else:
            assert x == y, item.name


def _front_fraction(model, delta_m=0.):
    # Every face starts with a half-solid column. Increment the same material
    # depth on both sides, independent of displacement and face surface area.
    return .5+delta_m/model.reference_column_depth_m


def _prescribed_history(model, state, normal=15.):
    geometry = model._geometry(state.cut_edges)
    # Crack-tip traces can share a single material node, giving a zero jump
    # operator. Prescribe displacement on an actually independent bank pair.
    normal_rows = geometry.jump_operator[::2]
    index = int(np.argmax(np.asarray(normal_rows.multiply(normal_rows).sum(axis=1)).ravel()))
    row = geometry.jump_operator.getrow(2*index).toarray().ravel()
    shear = geometry.jump_operator.getrow(2*index+1).toarray().ravel()
    q = normal*row+4.*shear
    gap, jump = (geometry.jump_operator@q).reshape(-1, 2).T
    cohorts, _, _ = evaluate_cohorts(state.cohorts, gap, jump, 1e9,
                                    np.full(len(gap), .35), model.law_parameters)
    contact = replace(state.contact, displacement_m=q,
                      **aggregate_cohorts(cohorts, len(gap)))
    return replace(state, contact=contact, cohorts=cohorts)


def _new_cut_state(model, state):
    old = set(map(tuple, state.cut_edges))
    edge = next(row for row in np.asarray(model.original_mesh.shared_edges) if tuple(row[2:]) not in old)
    a, b, u, v = edge
    direction = np.cross(model.original_mesh.vertices[u], model.original_mesh.vertices[v])
    direction /= np.linalg.norm(direction)
    frame = face_frames(model.original_mesh)
    normals = state.plane_normal.copy()
    for face in (a, b):
        local = direction@frame[face]
        normals[face] = local/np.linalg.norm(local)
    damage = state.damage.copy()
    damage[[a, b]] = .9
    active = state.fault_active.copy()
    active[[a, b]] = True
    return replace(state, plane_normal=normals, damage=damage, fault_active=active)


def test_initial_reference_contacts_match_actual_half_solid_material(partial_source):
    model = CoupledModel(partial_source)
    state = model.initial()
    geometry = model._geometry(state.cut_edges)
    _, _, fraction, _, _, depth = model._phase_fields(state, model.source.thermal_state)
    np.testing.assert_allclose(fraction, .5, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(state.interface_depth_ref_m, 20000., rtol=1e-14)
    assert len(state.cohorts.trace_index) == 2*len(state.cut_edges)
    np.testing.assert_allclose(state.interface_birth_area_m2,
        np.repeat(geometry.edge_length_m, 2)*state.interface_depth_ref_m/2)
    assert model._interface_area_change(geometry, fraction, depth) == pytest.approx(0., abs=1e-14)


@pytest.mark.parametrize("radial_fraction", [-.001, .001])
def test_geometric_thickening_or_thinning_never_creates_or_retires_solid_cohorts(partial_source, radial_fraction):
    model = CoupledModel(partial_source)
    state = model.initial()
    q = state.contact.displacement_m.copy()
    q[-1] = model.radius_m*radial_fraction
    moved = replace(state, contact=replace(state.contact, displacement_m=q))
    copied = deepcopy(moved)
    _, _, fraction, _, _, physical_depth = model._phase_fields(moved, model.source.thermal_state)
    assert np.max(np.abs(physical_depth*1000-model.reference_column_depth_m)) > 10.
    assert model._interface_area_change(model._geometry(state.cut_edges), fraction, physical_depth) > .001
    after = model._grow_interfaces(moved, fraction, state.time_myr+1e-6)
    _same(after, copied)
    np.testing.assert_array_equal(model.layer_mass_kg, model.source.layer_mass_kg)


def test_subquantum_front_stays_pending_then_activates_its_full_material_interval(partial_source):
    model = CoupledModel(partial_source, CoupledParameters(growth_layer_m=1.))
    original = model.initial()
    waiting = model._grow_interfaces(original, _front_fraction(model, .4), original.time_myr+1e-6)
    _same(waiting, original)
    grown = model._grow_interfaces(waiting, _front_fraction(model, 1.2), original.time_myr+2e-6)
    count = len(original.cohorts.trace_index)
    assert len(grown.cohorts.trace_index) == 2*count
    np.testing.assert_allclose(grown.cohorts.z_hi_ref_m[count:]-grown.cohorts.z_lo_ref_m[count:], 1.2, atol=1e-10)
    np.testing.assert_array_equal(grown.cohorts.z_lo_ref_m[count:], original.interface_depth_ref_m)
    np.testing.assert_allclose(grown.interface_depth_ref_m-original.interface_depth_ref_m, 1.2, atol=1e-10)
    expected_new_area = np.repeat(model._geometry(original.cut_edges).edge_length_m, 2)*1.2/2
    np.testing.assert_allclose(grown.cohorts.area_ref_m2[count:], expected_new_area, rtol=1e-10)
    assert np.all(grown.cohorts.bonded[count:])
    assert grown.interface_birth_energy_j == 0
    for name in ("column_enthalpy", "elastic_strain", "damage", "water_access"):
        np.testing.assert_array_equal(getattr(grown, name), getattr(original, name))
    same = model._grow_interfaces(grown, _front_fraction(model, 1.2), original.time_myr+3e-6)
    _same(same, grown)


def test_addition_keeps_each_old_history_and_joule_bitwise_and_open_banks_stay_unbonded(partial_source):
    model = CoupledModel(partial_source)
    state = _prescribed_history(model, model.initial())
    original = deepcopy(state)
    geometry = model._geometry(state.cut_edges)
    gap, jump = (geometry.jump_operator@state.contact.displacement_m).reshape(-1, 2).T
    assert np.max(gap) > model.law_parameters.failure_opening_m
    assert state.cohorts.fracture_work_j.sum() > 0
    count = len(state.cohorts.trace_index)
    grown = model._grow_interfaces(state, _front_fraction(model, 2.), state.time_myr+1e-6)
    for item in fields(state.cohorts):
        np.testing.assert_array_equal(getattr(grown.cohorts, item.name)[:count], getattr(state.cohorts, item.name))
    _same(state, original)
    np.testing.assert_array_equal(grown.cohorts.birth_jump_m[count:], jump)
    np.testing.assert_array_equal(grown.cohorts.birth_gap_m[count:], np.maximum(gap, 0))
    open_trace = gap > model.parameters.bonding_gap_tolerance_m
    assert np.any(open_trace)
    assert not np.any(grown.cohorts.bonded[count:][open_trace])
    np.testing.assert_array_equal(grown.cohorts.traction_pa[count:][open_trace], 0.)
    np.testing.assert_array_equal(grown.cohorts.fracture_work_j[count:], 0.)
    np.testing.assert_array_equal(grown.cohorts.cumulative_slip_m[count:], 0.)
    for name in ("friction_work_cell_j", "viscous_work_cell_j", "fracture_work_cell_j", "shear_remainder_cell_j"):
        np.testing.assert_array_equal(getattr(grown.contact, name), getattr(state.contact, name))


def test_new_compression_energy_is_in_birth_ledger_not_friction_or_fracture(partial_source):
    model = CoupledModel(partial_source)
    state = _prescribed_history(model, model.initial(), normal=-.25)
    geometry = model._geometry(state.cut_edges)
    gap = (geometry.jump_operator@state.contact.displacement_m).reshape(-1, 2)[:, 0]
    assert np.any(gap < 0)
    count = len(state.cohorts.trace_index)
    grown = model._grow_interfaces(state, _front_fraction(model, 2.), state.time_myr+1e-6)
    expected = .5*model.law_parameters.normal_stiffness_pa_m*float(
        np.dot(grown.cohorts.area_ref_m2[count:], np.minimum(gap, 0)**2))
    assert grown.interface_birth_energy_j-state.interface_birth_energy_j == pytest.approx(expected, rel=2e-11)
    assert grown.cohorts.fracture_work_j.sum() == state.cohorts.fracture_work_j.sum()
    assert grown.cohorts.friction_work_j.sum() == state.cohorts.friction_work_j.sum()
    assert grown.interface_birth_energy_j > 0


def test_remelting_of_activated_contact_is_rejected_without_mutating_any_history(partial_source):
    model = CoupledModel(partial_source)
    state = _prescribed_history(model, model.initial())
    copied = deepcopy(state)
    with pytest.raises(_RetryStep, match="remelting"):
        model._grow_interfaces(state, _front_fraction(model, -.001), state.time_myr+1e-6)
    _same(state, copied)


def test_storage_limit_rejects_whole_growth_batch_without_partial_activation(partial_source):
    model = CoupledModel(partial_source)
    state = model.initial()
    copied = deepcopy(state)
    model.parameters = replace(model.parameters, max_cohort_count=len(state.cohorts.trace_index)+1)
    with pytest.raises(_RetryStep, match="cohort_count"):
        model._grow_interfaces(state, _front_fraction(model, 2.), state.time_myr+1e-6)
    _same(state, copied)


def test_new_cuts_preserve_material_geometry_and_map_every_cohort_by_edge_identity(partial_source):
    model = CoupledModel(partial_source)
    state = _prescribed_history(model, model.initial())
    # Two independent generations per trace must survive a change in the
    # sorted global edge order; preserving only the first record is incorrect.
    state = model._grow_interfaces(state, _front_fraction(model, 2.), state.time_myr)
    state = _new_cut_state(model, state)
    copied = deepcopy(state)
    grown = model._grow_cuts(state, model.source.thermal_state)
    assert len(grown.cut_edges) > len(state.cut_edges)
    _same(state, copied)
    old_geometry, new_geometry = model._geometry(state.cut_edges), model._geometry(grown.cut_edges)
    old_mesh, new_mesh = model.mesh_for(state), model.mesh_for(grown)
    np.testing.assert_array_equal(old_mesh.vertices[old_mesh.faces], new_mesh.vertices[new_mesh.faces])
    count = len(state.cohorts.trace_index)
    index = {tuple(edge): j for j, edge in enumerate(grown.cut_edges)}
    for k, old_trace in enumerate(state.cohorts.trace_index):
        old_edge = tuple(state.cut_edges[old_trace//2])
        expected = 2*index[old_edge]+old_trace % 2
        assert grown.cohorts.trace_index[k] == expected
    for item in fields(state.cohorts):
        if item.name != "trace_index":
            np.testing.assert_array_equal(getattr(grown.cohorts, item.name)[:count], getattr(state.cohorts, item.name))
    newly_born = grown.cohorts.trace_index[count:]
    assert len(newly_born) == 2*(len(grown.cut_edges)-len(state.cut_edges))
    assert np.all(grown.cohorts.bonded[count:])
    assert np.max(np.abs(grown.cohorts.birth_jump_m[count:])) < 1e-10
    np.testing.assert_array_equal(grown.cohorts.fracture_work_j[count:], 0.)
    for name in ("column_enthalpy", "elastic_strain", "water_access"):
        np.testing.assert_array_equal(getattr(grown, name), getattr(state, name))
    for name in ("friction_work_j", "viscous_work_j", "fracture_work_j", "shear_remainder_j"):
        assert np.sum(getattr(grown.cohorts, name)) == np.sum(getattr(state.cohorts, name))
    # Every previous trace is still attached to the same face pair.
    for old_edge, new_index in index.items():
        old_matches = np.flatnonzero(np.all(state.cut_edges == old_edge, axis=1))
        if len(old_matches):
            np.testing.assert_array_equal(new_geometry.topology.seam_faces[new_index],
                                          old_geometry.topology.seam_faces[old_matches[0]])


def test_storage_limit_also_rolls_back_new_topological_cuts(partial_source):
    model = CoupledModel(partial_source)
    state = _new_cut_state(model, model.initial())
    copied = deepcopy(state)
    model.parameters = replace(model.parameters, max_cohort_count=len(state.cohorts.trace_index))
    with pytest.raises(_RetryStep, match="cohort_count"):
        model._grow_cuts(state, model.source.thermal_state)
    _same(state, copied)


def test_real_thermal_mechanical_step_activates_growth_and_preserves_budgets(partial_source):
    model = CoupledModel(partial_source, CoupledParameters(growth_layer_m=1e-7))
    state = model.initial()
    # A linear geotherm has no conductive front motion before boundary changes
    # arrive. A cooler layer below the crossing gives a resolved local heat
    # sink while leaving the initial solid front exactly at half the column.
    temperature = rock_temperature(state.column_enthalpy, model.source_model.thermal)
    layer = np.flatnonzero(temperature[0] > model.source_model.thermal.solidus_k)[1]
    temperature[:, layer] -= 20.
    h = rock_enthalpy(temperature, model.source_model.thermal)
    state = replace(state, column_enthalpy=h,
                    boundary_energy_j=float(np.sum(model.layer_mass_kg*h))-state.initial_column_energy_j)
    initial = deepcopy(state)
    following, thermal, orbit, rows = model.step(state, model.source.thermal_state,
                                                model.source.orbit, state.time_myr+1/1e6)
    assert following.stopped_reason is None
    assert following.time_myr == thermal.time_myr == orbit.time_myr == state.time_myr+1/1e6
    assert following.accepted_steps > 0
    assert len(following.cohorts.trace_index) > len(state.cohorts.trace_index)
    assert np.all(following.interface_depth_ref_m > state.interface_depth_ref_m)
    assert np.linalg.norm(following.contact.displacement_m) > 0
    _same(state, initial)
    assert rows[-1]["relative_mass_residual"] == 0
    assert abs(rows[-1]["relative_column_energy_residual"]) < 1e-12
    assert abs(rows[-1]["relative_global_energy_residual"]) < 1e-12
    assert following.contact.equilibrium_residual <= model.contact_parameters.equilibrium_tolerance
