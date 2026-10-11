"""Work-conjugate inertia and basal tests against independent quadrature."""
from dataclasses import replace
from decimal import Decimal, localcontext
import json
import math
from pathlib import Path

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from tectonics.genesis_contact import ContactParameters
from tectonics.genesis_volume_basal import (
    DRAG_CLOSURE, LOAD_CLOSURE, assemble_genesis_volume_basal,
    conical_interval_mass_matrix, local_basal_drag_and_force,
    local_consistent_mass_matrix, radial_displacement_moments,
    recover_source_last_accepted_vertex_rate,
)
from tectonics.genesis_volume_operator import assemble_genesis_volume_operator
from tectonics.mesh import build_icosphere


def _triangle():
    mesh = build_icosphere(1)
    return mesh.vertices[mesh.faces[0]]*1e6


ANGULAR_POINTS = np.array(((2/3, 1/6, 1/6), (1/6, 2/3, 1/6), (1/6, 1/6, 2/3)))


def _radial_quadrature(outer, inner):
    x, weight = leggauss(3)
    points = .5*(outer+inner)+.5*(outer-inner)*x
    weight = .5*(outer-inner)*weight*points**2
    weight /= (outer**3-inner**3)/3
    return points, weight


@pytest.mark.parametrize("a,b", [(1., .2), (1., .99999), (.93, .91), (1., 1.-1e-12)])
def test_stable_radial_moments_match_high_precision_material_measure(a, b):
    with localcontext() as context:
        context.prec = 80
        aa, bb = Decimal.from_float(a), Decimal.from_float(b)
        mean_lambda = Decimal(3)*(aa**4-bb**4)/(Decimal(4)*(aa**3-bb**3))
        second_lambda = Decimal(3)*(aa**5-bb**5)/(Decimal(5)*(aa**3-bb**3))
        mean, variance = radial_displacement_moments(a, b)
        assert mean == pytest.approx(float(mean_lambda-1), rel=3e-15, abs=1e-28)
        assert variance == pytest.approx(float(second_lambda-mean_lambda**2), rel=3e-15)
    assert variance > 0


def test_consistent_mass_integrates_the_full_18_dof_kinetic_field():
    mass, outer, inner = 4.3e13, .99, .975
    velocity = np.random.default_rng(713).normal(size=(3, 2, 3))
    matrix = local_consistent_mass_matrix(mass, outer, inner)
    radial, weight = _radial_quadrature(outer, inner)
    independent = 0.
    for lam, w in zip(radial, weight):
        for shape in ANGULAR_POINTS:
            v = shape@(velocity[:, 0]+(lam-1)*velocity[:, 1])
            independent += mass*w*np.dot(v, v)/3
    vector = velocity.ravel()
    assert vector@matrix@vector == pytest.approx(independent, rel=2e-14)
    np.testing.assert_array_equal(matrix, matrix.T)


def test_uniform_translation_has_actual_mass_without_slope_floor():
    mass, outer, inner = 4.3e13, .99, .975
    velocity = np.array([.4, -.3, .2])
    field = np.zeros((3, 2, 3))
    field[:, 0] = velocity
    matrix = local_consistent_mass_matrix(mass, outer, inner)
    assert field.ravel()@matrix@field.ravel() == pytest.approx(mass*np.dot(velocity, velocity), rel=2e-15)
    # The smaller radial mode is a genuine finite-thickness mass, not a
    # diagonal floor chosen to cure a solver nullspace.
    assert np.linalg.eigvalsh(matrix).min() > 0


def test_rigid_rotation_inertia_uses_actual_conical_material_positions():
    triangle, mass, outer, inner = _triangle(), 4.3e13, .99, .975
    omega = np.array([2e-7, -1e-7, 3e-7])
    top_velocity = np.cross(omega, triangle)
    field = np.stack((top_velocity, top_velocity), axis=1)
    matrix = local_consistent_mass_matrix(mass, outer, inner)
    radial, weight = _radial_quadrature(outer, inner)
    independent = 0.
    for lam, w in zip(radial, weight):
        for shape in ANGULAR_POINTS:
            point = lam*(shape@triangle)
            v = np.cross(omega, point)
            independent += mass*w*np.dot(v, v)/3
    assert field.ravel()@matrix@field.ravel() == pytest.approx(independent, rel=2e-14)


def test_actual_mass_q_band_additivity_and_direct_removed_energy():
    triangle, mass, rho = _triangle(), 3e17, 3000.
    whole = conical_interval_mass_matrix(triangle, mass, rho, .2, .7)
    retained = conical_interval_mass_matrix(triangle, mass, rho, .2, .699)
    removed = conical_interval_mass_matrix(triangle, mass, rho, .699, .7)
    np.testing.assert_allclose(whole, retained+removed, rtol=2e-12, atol=2e-2)
    velocity = np.random.default_rng(23).normal(size=18)
    assert .5*velocity@removed@velocity > 0
    with pytest.raises(ValueError, match="actual positive"):
        conical_interval_mass_matrix(triangle, mass, rho, .7, .7)


def test_reference_basal_drag_and_traction_are_work_conjugate_to_the_same_field():
    triangle, bottom, beta = _triangle(), .97, 1e14
    traction = np.random.default_rng(41).normal(size=(3, 3))*2e4
    field = np.random.default_rng(52).normal(size=(3, 2, 3))
    drag, force, area = local_basal_drag_and_force(triangle, bottom, traction, beta)
    normal = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])
    normal /= np.linalg.norm(normal)
    expected_area = .5*np.linalg.norm(np.cross(bottom*(triangle[1]-triangle[0]),
                                              bottom*(triangle[2]-triangle[0])))
    power, loss = 0., 0.
    for shape in ANGULAR_POINTS:
        velocity = shape@(field[:, 0]+(bottom-1)*field[:, 1])
        tangent_velocity = velocity-normal*np.dot(normal, velocity)
        power += area*np.dot(shape@traction, velocity)/3
        loss += beta*area*np.dot(tangent_velocity, tangent_velocity)/3
    vector = field.ravel()
    assert area == pytest.approx(expected_area, rel=2e-15)
    assert force@vector == pytest.approx(power, rel=3e-15)
    assert vector@drag@vector == pytest.approx(loss, rel=3e-15)
    np.testing.assert_array_equal(drag, drag.T)


def test_basal_null_mode_has_zero_drag_and_load_work_but_positive_physical_inertia():
    triangle, bottom, beta = _triangle(), .97, 1e14
    traction = np.random.default_rng(11).normal(size=(3, 3))*2e4
    slope = np.random.default_rng(12).normal(size=(3, 3))
    field = np.stack((-(bottom-1)*slope, slope), axis=1).ravel()
    drag, force, _ = local_basal_drag_and_force(triangle, bottom, traction, beta)
    scale = np.linalg.norm(drag)*np.linalg.norm(field)
    assert np.linalg.norm(drag@field) <= 3e-17*scale
    assert abs(force@field) <= np.linalg.norm(force)*np.linalg.norm(field)*2e-17
    mass = local_consistent_mass_matrix(4.3e13, 1., bottom)
    assert field@mass@field > 0


def test_no_normal_drag_is_added_to_the_source_tangential_closure():
    triangle, bottom, beta = _triangle(), .97, 1e14
    normal = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])
    normal /= np.linalg.norm(normal)
    field = np.zeros((3, 2, 3))
    field[:, 0] = normal
    drag, _, area = local_basal_drag_and_force(triangle, bottom, np.zeros((3, 3)), beta)
    assert np.linalg.norm(drag@field.ravel()) < 3e-16*beta*area


@pytest.fixture(scope="module")
def actual_assembly():
    path = Path(__file__).resolve().parents[1]/"analysis/genesis_cohort_mechanics_validation/evidence/control/material_export.json"
    packet = json.loads(path.read_text())
    operator = assemble_genesis_volume_operator(packet)
    return packet, operator, assemble_genesis_volume_basal(operator)


def test_actual_source_consistent_inertia_has_exact_material_mass_and_real_basal_geometry(actual_assembly):
    packet, operator, basal = actual_assembly
    assert basal.source_checkpoint_sha256 == packet["source"]["checkpoint_sha256"]
    assert basal.source_identity == packet["birth_source_identity"]
    assert basal.time_myr == packet["time_myr"]
    assert basal.ndof == operator.ndof == 252
    assert basal.mass_matrix.shape == basal.basal_drag_matrix.shape == (252, 252)
    assert basal.cohort_mass_matrices.shape == (480, 18, 18)
    assert basal.face_basal_drag_matrices.shape == (80, 18, 18)
    assert basal.drag_closure == DRAG_CLOSURE and basal.load_closure == LOAD_CLOSURE
    assert not basal.normal_basal_drag_added
    assert not basal.arbitrary_diagonal_mass_or_slope_drag_added
    assert not basal.full_continuous_velocity_history_inherited
    assert not basal.physical_force_work_admission
    velocity = np.array([1., .2, -.4])
    field = np.zeros((42, 2, 3))
    field[:, 0] = velocity
    assert field.ravel()@basal.mass_matrix@field.ravel() == pytest.approx(
        basal.total_live_mass_kg*np.dot(velocity, velocity), rel=3e-15)
    expected_mass = sum(c.mass_kg for c in operator.geometry.cohorts)
    assert basal.total_live_mass_kg == expected_mass
    triangles = np.asarray(operator.geometry.vertices_m)[np.asarray(operator.geometry.faces)]
    independent_areas = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                               triangles[:, 2]-triangles[:, 0]), axis=1)/2
    independent_areas *= basal.basal_front_scale**2
    np.testing.assert_allclose(basal.basal_face_area_m2, independent_areas, rtol=2e-15)
    assert basal.basal_area_m2 == float(sum(basal.basal_face_area_m2))
    assert np.max(np.abs(basal.actual_conical_thickness_km-basal.source_thincolumn_thickness_km)) > .001
    assert np.max(np.abs(basal.basal_traction_at_vertices_pa
                         -basal.original_thincolumn_traction_at_vertices_pa)) > 1.


def test_actual_generalized_mass_drag_and_force_match_independent_face_cohort_quadrature(actual_assembly):
    _, operator, basal = actual_assembly
    field = np.random.default_rng(92).normal(size=operator.ndof)
    kinetic_terms = []
    for c, dofs in zip(operator.geometry.cohorts, operator.dofs):
        local = field[dofs].reshape(3, 2, 3)
        radial, weight = _radial_quadrature(c.outer_scale, c.inner_scale)
        # Normalize the independent quadrature by its literal summed weights
        # for the very thin actual newborn intervals (the cube difference is
        # only a floating reference normalization, not a material source).
        weight /= weight.sum()
        for lam, w in zip(radial, weight):
            for shape in ANGULAR_POINTS:
                velocity = shape@(local[:, 0]+(lam-1)*local[:, 1])
                kinetic_terms.append(c.mass_kg*w*np.dot(velocity, velocity)/3)
    kinetic = math.fsum(kinetic_terms)
    assert field@basal.mass_matrix@field == pytest.approx(kinetic, rel=4e-15)
    power, loss = 0., 0.
    geometry = operator.geometry
    beta = json.loads(basal.contact_parameters_json)["basal_drag_pa_s_m"]
    for face, dofs in enumerate(basal.face_dofs):
        tri = np.asarray(geometry.vertices_m)[np.asarray(geometry.faces[face])]
        normal = np.cross(tri[1]-tri[0], tri[2]-tri[0]); normal /= np.linalg.norm(normal)
        local = field[dofs].reshape(3, 2, 3)
        traction = basal.basal_traction_at_vertices_pa[np.asarray(geometry.faces[face])]
        for shape in ANGULAR_POINTS:
            velocity = shape@(local[:, 0]+(basal.basal_front_scale[face]-1)*local[:, 1])
            tangent = velocity-normal*np.dot(normal, velocity)
            power += basal.basal_face_area_m2[face]*np.dot(shape@traction, velocity)/3
            loss += beta*basal.basal_face_area_m2[face]*np.dot(tangent, tangent)/3
    assert basal.external_force_n@field == pytest.approx(power, rel=5e-15)
    assert field@basal.basal_drag_matrix@field == pytest.approx(loss, rel=5e-15)


def test_contact_drag_parameter_is_explicit_continuation_data(actual_assembly):
    _, operator, original = actual_assembly
    parameters = replace(ContactParameters(), basal_drag_pa_s_m=2e14)
    changed = assemble_genesis_volume_basal(operator, parameters)
    np.testing.assert_array_equal(changed.mass_matrix.toarray(), original.mass_matrix.toarray())
    np.testing.assert_array_equal(changed.external_force_n, original.external_force_n)
    np.testing.assert_array_equal(changed.basal_drag_matrix.toarray(), 2*original.basal_drag_matrix.toarray())
    assert json.loads(changed.contact_parameters_json)["basal_drag_pa_s_m"] == 2e14


def test_true_continued_births_add_direct_mass_without_false_retirement(actual_assembly):
    from tectonics.genesis_bulk_birth import bulk_birth_from_dict
    from tectonics.genesis_volume_geometry import build_updated_genesis_volume_geometry
    from tectonics.genesis_volume_operator import assemble_updated_volume_operator

    packet, source_operator, source_basal = actual_assembly
    path = Path(__file__).resolve().parents[1]/"analysis/genesis_cohort_mechanics_validation/evidence/control/fault_direct_continued.npz"
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        damage = archive["damage"].copy()
    history = bulk_birth_from_dict(metadata["state"]["bulk_birth_history"])
    geometry = build_updated_genesis_volume_geometry(source_operator.geometry, history)
    operator = assemble_updated_volume_operator(source_operator, geometry, damage)
    continued = assemble_genesis_volume_basal(operator)
    assert continued.total_live_mass_kg > source_basal.total_live_mass_kg
    assert len(geometry.cohorts) == 640
    assert np.all(continued.basal_front_scale < source_basal.basal_front_scale)
    direct_born_mass = np.zeros((252, 252))
    source_ids = set(source_operator.cohort_ids)
    for c, dofs in zip(geometry.cohorts, operator.dofs):
        if c.id in source_ids:
            continue
        local = conical_interval_mass_matrix(c.top_triangle_m,
            geometry.column_mass_kg[c.face_index], geometry.density_kg_m3, c.q_lo, c.live_q_hi)
        direct_born_mass[np.ix_(dofs, dofs)] += local
    difference = (continued.mass_matrix-source_basal.mass_matrix).toarray()
    np.testing.assert_allclose(difference, direct_born_mass, rtol=2e-10, atol=3e3)
    field = np.random.default_rng(827).normal(size=252)
    assert field@direct_born_mass@field > 0
    # The immutable older source still contains its own clock and 480 IDs.
    assert packet["time_myr"] == source_operator.time_myr < continued.time_myr
    assert len(packet["instantaneous_cohort_plane_stress_3d"]["source_cohort_ids"]) == 480


def test_last_accepted_chord_rate_matches_independent_actual_producer_replay(actual_assembly):
    from tectonics.genesis import SECONDS_PER_MYR
    from tectonics.genesis_faults import load_fault_checkpoint

    packet, _, _ = actual_assembly
    rate = recover_source_last_accepted_vertex_rate(packet)
    root = Path(__file__).resolve().parents[1]/"analysis/genesis_cohort_mechanics_validation/evidence/control"
    model, state, thermal, orbit, _ = load_fault_checkpoint(root/"precursor_fault_checkpoint.npz")
    # Re-execute the actual saved accepted intervals from the real precursor
    # to the source's immediately preceding accepted state. This tests the
    # reconstructed chart against independent producer geometry, rather than
    # merely comparing another copy of the Gram/frame inversion formula.
    for interval in packet["bulk_birth_history"]["intervals"]:
        target = interval["after_time_myr"]
        if state.time_myr < target <= rate.before_time_myr:
            state, thermal, orbit, _ = model.step(state, thermal, orbit, target, .01)
            assert state.stopped_reason is None
    assert state.time_myr == rate.before_time_myr
    assert rate.after_time_myr == packet["time_myr"]
    expected_before = state.vertices*1000*state.radius_km
    source_radius_m = 1000*packet["current_geometry"]["radius_km"]
    independent_resolution = 32*np.finfo(float).eps*source_radius_m
    np.testing.assert_allclose(rate.before_vertices_m, expected_before,
                               rtol=0, atol=independent_resolution)
    expected_current = np.asarray(packet["current_geometry"]["vertices_unit_xyz"])*source_radius_m
    duration_s = (rate.after_time_myr-rate.before_time_myr)*SECONDS_PER_MYR
    expected_rate = (expected_current-expected_before)/duration_s
    np.testing.assert_allclose(rate.velocity_m_s, expected_rate, rtol=0,
                               atol=independent_resolution/duration_s)
    assert rate.max_shared_node_mismatch_m < rate.coordinate_roundoff_bound_m
    assert np.max(np.linalg.norm(rate.velocity_m_s, axis=1)) > 1e-8
    assert not rate.continuous_instantaneous_velocity_history_recovered
    assert rate.semantics == "last_accepted_chord_finite_difference_surface_rate"
    with pytest.raises(ValueError):
        rate.velocity_m_s[0, 0] = 0.


def test_missing_last_interval_birth_coverage_is_refused_without_zero_fill(tmp_path):
    from tectonics.genesis import GenesisParameters
    from tectonics.genesis_bulk_export import export_genesis_bulk_material
    from tectonics.genesis_faults import FaultModel, save_fault_checkpoint
    from tectonics.genesis_onset import OnsetParameters
    from tectonics.genesis_shell import ShellParameters
    from tectonics.genesis_tides import TidalParameters

    model = FaultModel(ShellParameters(subdivisions=1), GenesisParameters(),
                       OnsetParameters(), TidalParameters(enabled=False))
    state, thermal, orbit = model.initial()
    state, thermal, orbit, _ = model.step(state, thermal, orbit, 1e-6, .01)
    assert state.time_myr > 0 and state.bulk_birth_history.intervals
    assert not state.bulk_birth_history.cohorts
    path = tmp_path/"actual_still_molten_interval.npz"
    save_fault_checkpoint(path, model, state, thermal, orbit, {},
                          {"purpose": "actual interval without solid births"})
    packet = export_genesis_bulk_material(path.read_bytes(), {"purpose": "refuse missing coverage"})
    with pytest.raises(ValueError, match="do not cover every source material face"):
        recover_source_last_accepted_vertex_rate(packet)


@pytest.mark.parametrize("a,b", [(1., 1.), (.9, .95), (1., 0.), (float("nan"), .9)])
def test_bad_radial_mass_interval_is_rejected(a, b):
    with pytest.raises(ValueError):
        local_consistent_mass_matrix(1., a, b)
