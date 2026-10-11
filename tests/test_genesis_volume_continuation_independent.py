"""Meaningful controls for the independently derived continuation audit."""
import copy
import base64
import hashlib
import io
import json
from fractions import Fraction

import numpy as np
import pytest

from analysis.genesis_volume_continuation_validation.independent_audit import (
    angular_rule, basal_drag_and_force, conical_scales, dynamic_balance,
    endpoint_clock_binding, geometry_binding, independent_strain_operators, isotropic_matrix,
    mass_matrix, quadrature_cohort, source_dto_binding, temporal_factors,
)
from tectonics.genesis_volume_temporal import certify_cone_homothetic_path, cone_homothetic_to_dict


TRIANGLE = np.eye(3)*1e5
FACES = np.array(((0, 1, 2),))
COLUMN_MASS = np.array((1e12,))
DENSITY = 3000.
COHORT = {"id": 1, "face_index": 0, "q_lo": .01, "q_hi": .02}


def test_angular_rule_integrates_independent_triangle_moments():
    N, w = angular_rule()
    np.testing.assert_allclose(w.sum(), 1., rtol=0, atol=3e-16)
    np.testing.assert_allclose(w@N, np.full(3, 1/3), rtol=0, atol=2e-16)
    np.testing.assert_allclose(np.einsum("p,pi,pj->ij", w, N, N),
        (np.ones((3, 3))+np.eye(3))/12, rtol=0, atol=1e-16)


def test_mass_constant_translation_exact_total_physical_mass():
    M = mass_matrix(TRIANGLE, FACES, COLUMN_MASS, DENSITY, [COHORT])
    rate = np.zeros((3, 6)); rate[:, :3] = (2., -1., 3.)
    velocity = rate.ravel()
    kinetic = .5*velocity@M@velocity
    assert kinetic == pytest.approx(.5*COLUMN_MASS[0]*.01*14, rel=2e-15)
    np.testing.assert_allclose(M, M.T, rtol=1e-15, atol=0)


def test_mass_retention_birth_remelt_matrices_partition_by_actual_intervals():
    old = [dict(COHORT, q_lo=0., q_hi=.1)]
    retained = [dict(COHORT, q_lo=0., q_hi=.08)]
    removed = [dict(COHORT, q_lo=.08, q_hi=.1)]
    matrices = [mass_matrix(TRIANGLE, FACES, COLUMN_MASS, DENSITY, entries)
                for entries in (old, retained, removed)]
    np.testing.assert_allclose(matrices[0], matrices[1]+matrices[2], rtol=2e-14, atol=1e-10)


def test_tangent_basal_drag_does_not_invent_normal_drag():
    normal = np.ones(3)/np.sqrt(3)
    D, F = basal_drag_and_force(TRIANGLE, FACES, COLUMN_MASS, DENSITY,
        [COHORT], 2e6, [[1e4, -1e4, 0.]])
    rates = np.zeros((3, 6)); rates[:, :3] = normal
    assert np.linalg.norm(D@rates.ravel()) <= 2e-15*np.linalg.norm(D)*np.linalg.norm(rates)
    lam, lm1 = conical_scales(TRIANGLE, COLUMN_MASS[0], DENSITY, COHORT["q_hi"])
    area = np.linalg.norm(np.cross(TRIANGLE[1]-TRIANGLE[0], TRIANGLE[2]-TRIANGLE[0]))/2*lam**2
    forces = F.reshape(3, 6)
    np.testing.assert_allclose(forces[:, :3].sum(axis=0), area*np.array((1e4, -1e4, 0.)), rtol=2e-15)
    np.testing.assert_allclose(forces[:, 3:].sum(axis=0), lm1*forces[:, :3].sum(axis=0), rtol=2e-15)


def test_cartesian_strain_operators_match_finite_differences_of_actual_field():
    N = np.array(((.2, .3, .5),))
    lam = .9
    increments = np.arange(18, dtype=float).reshape(3, 6)/100
    inverse = np.linalg.inv(TRIANGLE.T)
    x = lam*TRIANGLE.T@N[0]

    def displacement(point):
        raw = inverse@point
        radial = raw.sum()
        return (raw/radial)@(increments[:, :3]+(radial-1)*increments[:, 3:])

    h = .01
    gradient = np.column_stack([(displacement(x+h*np.eye(3)[i])-displacement(x-h*np.eye(3)[i]))/(2*h)
                                for i in range(3)])
    expected = (gradient[0, 0], gradient[1, 1], gradient[2, 2],
        gradient[0, 1]+gradient[1, 0], gradient[0, 2]+gradient[2, 0], gradient[1, 2]+gradient[2, 1])
    B0, B1 = independent_strain_operators(TRIANGLE, np.eye(3), N)
    np.testing.assert_allclose((B0[0]+B1[0]/lam)@increments.ravel(), expected, rtol=4e-8, atol=2e-15)


@pytest.mark.parametrize("ratio", [0., 1e-8, .1, 4., 1000.])
def test_independent_maxwell_time_radial_angular_work_identity(ratio):
    old0 = np.tile((1e-4, -2e-4, 5e-5, 3e-5, 1e-5, -1e-5), (3, 1))
    old1 = old0*.1
    thermal = np.tile((2e-5, 2e-5, 2e-5, 0, 0, 0), (3, 1))
    C = isotropic_matrix(6e10, .25, .1, .02)
    result = quadrature_cohort(TRIANGLE, np.eye(3), COLUMN_MASS[0], DENSITY, COHORT,
        np.arange(18)*.3, old0, old1, thermal, ratio, C)
    assert result["relative_balance_residual"] < 3e-13
    assert result["viscous_dissipation_j"] >= 0
    assert result["average_force_n"]@(np.arange(18)*.3) == pytest.approx(result["mechanical_work_j"], rel=1e-12)


def test_zero_ratio_time_integrals_equal_elastic_linear_driving_limits():
    np.testing.assert_allclose(temporal_factors(0.), (1., .5, 0., 0., 0.), rtol=0, atol=2e-16)


def midpoint_control(old_mass, new_mass, born_mass, removed_mass):
    M0, M1 = np.eye(2)*old_mass, np.eye(2)*new_mass
    Min, Mout = np.eye(2)*born_mass, np.eye(2)*removed_mass
    D, K, F = np.eye(2)*.3, np.diag((2., 3.)), np.array((1., 2.))
    q0, v0, dt = np.zeros(2), np.array((.1, -.2)), 2.
    dq = np.linalg.solve(2*M1/dt**2+D/dt+.5*K, F+2*M1@v0/dt)
    v1, Fav = 2*dq/dt-v0, .5*K@dq
    return dynamic_balance(M0, M1, Min, Mout, D, F, Fav, q0, dq, v0, v1, dt, Fav@dq)


def test_global_midpoint_balance_includes_positive_newborn_kinetic_inflow():
    result = midpoint_control(2., 3., 1., 0.)
    assert result["kinetic_birth_inflow_j"] == pytest.approx(.025)
    assert result["kinetic_remelt_outflow_j"] == 0.
    assert result["relative_global_balance_residual"] < 2e-15
    assert not result["continuous_velocity_history_certified"]


def test_global_midpoint_balance_uses_removed_mass_energy_directly():
    result = midpoint_control(3., 2., 0., 1.)
    assert result["kinetic_birth_inflow_j"] == 0.
    assert result["kinetic_remelt_outflow_j"] == pytest.approx(.025)
    assert result["relative_global_balance_residual"] < 2e-15


def test_fake_phase_flux_fails_mass_partition_even_if_other_balance_is_valid():
    with pytest.raises(ValueError, match="partition"):
        midpoint_control(2., 3., 0., 0.)


def test_exact_geometry_proof_is_bound_to_actual_total_fields_and_live_material():
    vertices = np.array(((1., 0, 0), (-1., 0, 0), (0, 1., 0), (0, -1., 0), (0, 0, 1.), (0, 0, -1.)))
    from itertools import product
    faces = []
    for face in product((0, 1), (2, 3), (4, 5)):
        faces.append(face if np.linalg.det(vertices[list(face)]) > 0 else (face[0], face[2], face[1]))
    q0, q1 = np.zeros((6, 6)), np.zeros((6, 6))
    q1[:, :3] = q1[:, 3:] = .01*vertices
    certificate = cone_homothetic_to_dict(certify_cone_homothetic_path(vertices, faces, q1[:, :3], q1[:, 3:],
        lambda_min=Fraction(1, 2), source_identity="a"*64, before_time_myr=2., after_time_myr=3.))
    result = geometry_binding(certificate, vertices, faces, q0.ravel(), q1.ravel(),
        [dict(COHORT, q_lo=.01, q_hi=.02)], np.full(8, .1), 1., 2., 3.)
    assert result["actual_before_after_fields_bound"]
    q1[0, 0] += 1e-6
    with pytest.raises(ValueError, match="not bound"):
        geometry_binding(certificate, vertices, faces, q0.ravel(), q1.ravel(),
            [dict(COHORT, q_lo=.01, q_hi=.02)], np.full(8, .1), 1., 2., 3.)


def test_exact_massq_domain_refuses_sub_float_tolerance_missing_live_band():
    vertices = np.array(((1., 0, 0), (-1., 0, 0), (0, 1., 0), (0, -1., 0), (0, 0, 1.), (0, 0, -1.)))
    from itertools import product
    faces = [face if np.linalg.det(vertices[list(face)]) > 0 else (face[0], face[2], face[1])
             for face in product((0, 1), (2, 3), (4, 5))]
    fields = np.zeros((6, 6))
    certificate = cone_homothetic_to_dict(certify_cone_homothetic_path(vertices, faces,
        fields[:, :3], fields[:, 3:], lambda_min=1., source_identity="a"*64,
        before_time_myr=2., after_time_myr=3.))
    # The live band is about 1e-17 radially below lambda=1 and rounds to
    # lambda=1 in floating arithmetic. Its exact mass-q domain still lies out.
    with pytest.raises(ValueError, match="exact cone proof radial domain"):
        geometry_binding(certificate, vertices, faces, fields.ravel(), fields.ravel(),
            [dict(COHORT, q_lo=0., q_hi=.02)], np.full(8, 1e-16), 1., 2., 3.)


def test_independent_shared_clock_guard_refuses_resealed_endpoint_clock_drift():
    state = {"geometry": {"time_myr": 3.}, "history": {"time_myr": 3.},
        "mechanics": {"state": {"time_myr": 3.}}, "thermal": {"time_myr": 3.,
            "global_state": {"time_myr": 3.}, "orbit": {"time_myr": 3.}}}
    endpoint_clock_binding(state, 3.)
    for field in ("geometry", "history", "mechanics", "thermal", "global", "orbit"):
        mutated = copy.deepcopy(state)
        target = (mutated["mechanics"]["state"] if field == "mechanics" else
            mutated["thermal"]["global_state"] if field == "global" else
            mutated["thermal"]["orbit"] if field == "orbit" else mutated[field])
        target["time_myr"] += 1e-6
        with pytest.raises(ValueError, match="one actual shared clock"):
            endpoint_clock_binding(mutated, 3.)


def test_source_dto_changes_are_refused_against_literal_embedded_npz():
    history = {"source_identity": "a"*64, "initial_faces": [[0, 1, 2]]}
    metadata = {"parameters": {"shell": {"density_kg_m3": 3000.}}, "parameter_hash": "b"*64,
        "state": {"bulk_birth_history": history, "cohort_mechanics": {"state": {"cohorts": []}},
                  "time_myr": 2., "radius_km": 5.}}
    arrays = {"vertices": np.eye(3), "layer_mass_kg": np.ones((1, 2)),
        "column_enthalpy": np.full((1, 2), 4.), "damage": np.zeros(1),
        "elastic_strain": np.zeros((1, 3)), "velocity_km_myr": np.ones((1, 3))}
    stream = io.BytesIO()
    np.savez(stream, metadata=json.dumps(metadata), **arrays)
    digest = hashlib.sha256(stream.getvalue()).hexdigest()
    packet = {"source_checkpoint_sha256": digest, "source_packet": {
        "source": {"checkpoint_base64": base64.b64encode(stream.getvalue()).decode(),
            "checkpoint_sha256": digest, "parameters": metadata["parameters"], "parameter_hash": "b"*64},
        "bulk_birth_history": history, "cohort_mechanics": metadata["state"]["cohort_mechanics"],
        "current_geometry": {"vertices_unit_xyz": arrays["vertices"].tolist(), "radius_km": 5., "faces": [[0, 1, 2]]},
        "current_columns": {"layer_mass_kg": arrays["layer_mass_kg"].tolist(),
            "specific_enthalpy_j_kg": arrays["column_enthalpy"].tolist()},
        "current_shell_memory": {"damage": [0.], "elastic_strain_xx_yy_gamma_xy": [[0., 0., 0.]],
            "velocity_km_myr_world": [[1., 1., 1.]]}, "time_myr": 2., "birth_source_identity": "a"*64}}
    assert source_dto_binding(packet)["embedded_npz_arrays_and_histories_bound"]
    changed = copy.deepcopy(packet)
    changed["source_packet"]["current_columns"]["layer_mass_kg"][0][0] += .1
    with pytest.raises(ValueError, match="literal source DTO layer_mass_kg"):
        source_dto_binding(changed)
    changed = copy.deepcopy(packet)
    changed["source_packet"]["source"]["parameters"]["shell"]["density_kg_m3"] += 1.
    with pytest.raises(ValueError, match="embedded source parameters"):
        source_dto_binding(changed)
