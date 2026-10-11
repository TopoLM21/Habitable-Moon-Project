"""Independent spatial/energetic checks of the inherited conical volume FE."""
from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import pytest

from tectonics.genesis_volume_operator import (
    ANGULAR_BARYCENTRIC, STRAIN_COMPONENTS, algorithmic_stiffness,
    assemble_genesis_volume_operator, assemble_updated_volume_operator, conical_radial_moments,
    conical_strain_operators, engineering_strain_to_tensor,
    evaluate_genesis_volume_operator, isotropic_elasticity,
    tensor_to_engineering_strain,
)


@pytest.fixture(scope="module")
def source_packet():
    path = Path(__file__).resolve().parents[1]/"analysis/genesis_cohort_mechanics_validation/evidence/control/material_export.json"
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def actual_operator(source_packet):
    return assemble_genesis_volume_operator(source_packet)


def test_engineering_shear_and_hooke_energy_independently():
    assert STRAIN_COMPONENTS == ("xx", "yy", "zz", "gamma_xy", "gamma_xz", "gamma_yz")
    strain = np.array((.01, -.02, .005, .03, -.015, .012))
    tensor = engineering_strain_to_tensor(strain)
    np.testing.assert_array_equal(tensor_to_engineering_strain(tensor), strain)
    E, nu = 70e9, .27
    mu, lam = E/(2*(1+nu)), E*nu/((1+nu)*(1-2*nu))
    stress = 2*mu*tensor+lam*np.trace(tensor)*np.eye(3)
    expected = .5*np.sum(stress*tensor)
    assert .5*strain@isotropic_elasticity(E, nu)@strain == pytest.approx(expected, rel=2e-15)
    for bad in ((0., .25), (1., .5), (1., -1.)):
        with pytest.raises(ValueError):
            isotropic_elasticity(*bad)


@pytest.mark.parametrize("component", range(6))
def test_all_six_affine_strains_are_represented(component):
    X = np.array(((0., 0., 10.), (2., 0., 10.), (0., 2., 10.)))
    expected = np.eye(6)[component]
    G = engineering_strain_to_tensor(expected)
    nodal = np.column_stack((X@G.T, X@G.T)).ravel()
    B0, B1 = conical_strain_operators(X, np.eye(3), ANGULAR_BARYCENTRIC)
    for scale in (.4, .8, 1.):
        actual = np.einsum("qai,i->qa", B0+B1/scale, nodal)
        np.testing.assert_allclose(actual, np.broadcast_to(expected, actual.shape), rtol=0, atol=8e-16)


def test_physical_cartesian_differentiation_matches_conical_B():
    X = np.array(((0., 0., 10.), (2., 0., 10.), (0., 2., 10.)))
    rng = np.random.default_rng(7648)
    dofs = rng.normal(size=(3, 6))
    N, scale = np.array((.3, .2, .5)), .63
    point = scale*(N@X)

    def displacement(x):
        lam = x[2]/10
        top = x[:2]/lam
        bary = np.array((1-top.sum()/2, top[0]/2, top[1]/2))
        return bary@dofs[:, :3]+(lam-1)*(bary@dofs[:, 3:])

    step = 1e-5
    derivative = np.column_stack([(displacement(point+step*axis)-displacement(point-step*axis))/(2*step)
                                  for axis in np.eye(3)])
    expected = tensor_to_engineering_strain((derivative+derivative.T)/2)
    B0, B1 = conical_strain_operators(X, np.eye(3), N[None])
    actual = (B0[0]+B1[0]/scale)@dofs.ravel()
    np.testing.assert_allclose(actual, expected, rtol=8e-10, atol=1e-10)


def test_director_variation_creates_bending_and_transverse_shear():
    X = np.array(((0., 0., 10.), (2., 0., 10.), (0., 2., 10.)))
    B0, B1 = conical_strain_operators(X, np.eye(3), np.array(((.2, .3, .5),)))
    q = np.zeros((3, 6))
    q[:, 3] = X[:, 0]
    top = (B0[0]+B1[0])@q.ravel()
    bottom = (B0[0]+B1[0]/.8)@q.ravel()
    assert top[0] == pytest.approx(0., abs=1e-14)
    assert bottom[0] != pytest.approx(0., abs=1e-14)
    assert top[4] != pytest.approx(0., abs=1e-14)
    q[:] = 0
    q[:, 2] = X[:, 0]
    assert ((B0[0]+B1[0])@q.ravel())[4] == pytest.approx(1.)


@pytest.mark.parametrize("outer,inner", ((1., .4), (.9, .899999999), (1., .999999)))
def test_radial_moments_match_independent_volume_quadrature(outer, inner):
    nodes, weights = np.polynomial.legendre.leggauss(30)
    lam = .5*(outer+inner)+.5*(outer-inner)*nodes
    physical = weights*lam**2
    physical /= physical.sum()
    mean, variance = conical_radial_moments(outer, inner)
    assert mean == pytest.approx(np.dot(physical, 1/lam), rel=3e-15)
    # Centred quadrature avoids the same s2-s1^2 cancellation as production.
    expected = np.dot(physical, (1/lam-mean)**2)
    assert variance == pytest.approx(expected, rel=2e-6, abs=5e-32)


def test_source_cohort_energy_and_prestress_are_inherited(actual_operator, source_packet):
    a = actual_operator
    zero = evaluate_genesis_volume_operator(a, np.zeros(a.ndof))
    expected = source_packet["instantaneous_cohort_plane_stress_3d"]["elastic_energy_j"]
    assert a.source_elastic_energy_j == source_packet["cohort_mechanics"]["state"]["elastic_energy_j"]
    assert a.source_elastic_energy_j == pytest.approx(expected, rel=3e-15)
    assert zero.elastic_energy_j == pytest.approx(expected, rel=3e-15)
    assert len(a.cohort_ids) == 480
    assert a.dofs.dtype.kind == "i"
    np.testing.assert_allclose(zero.force, a.force, rtol=3e-14, atol=.1)
    assert not a.full_historical_three_dimensional_memory_recovered
    assert not a.physical_force_work_admission
    face_mean_energy = source_packet["instantaneous_plane_stress_3d"]["elastic_energy_of_face_mean_j"]
    assert a.source_elastic_energy_j > 1.11*face_mean_energy


def test_complete_cell_rigid_motion_and_immutable_coefficients(actual_operator):
    a = actual_operator
    X = np.asarray(a.geometry.vertices_m)
    rotation = np.cross(np.array((1e-5, -2e-5, 3e-5)), X)
    q = np.column_stack((rotation, rotation))
    q[:, :3] += (20., -10., 5.)
    r = evaluate_genesis_volume_operator(a, q.ravel())
    np.testing.assert_allclose(r.memory0, a.memory0, rtol=5e-16, atol=1e-19)
    np.testing.assert_allclose(r.memory1, a.memory1, rtol=0, atol=1e-19)
    assert r.energy_j == pytest.approx(a.source_elastic_energy_j, rel=3e-15)
    scale = np.linalg.norm(a.K.data)*np.linalg.norm(q)
    assert np.linalg.norm(a.K@q.ravel()) < 2e-16*scale
    for value in (a.B0, a.C, a.dofs, a.K.data):
        with pytest.raises(ValueError):
            value.setflags(write=True)


def test_actual_energy_force_and_tangent_against_radial_quadrature(actual_operator):
    a = actual_operator
    rng = np.random.default_rng(58303)
    displacement = rng.normal(size=a.ndof)*.02
    r = evaluate_genesis_volume_operator(a, displacement)
    local = displacement[a.dofs]
    nodes, weights = np.polynomial.legendre.leggauss(8)
    lam = .5*(a.outer_scale[:, None]+a.inner_scale[:, None])+.5*(a.outer_scale-a.inner_scale)[:, None]*nodes
    measure = .5*weights[None, :]*lam**2*3/(a.outer_scale**2+a.outer_scale*a.inner_scale+a.inner_scale**2)[:, None]
    B = a.B0[:, :, None, :, :]+a.B1[:, :, None, :, :]/lam[:, None, :, None, None]
    elastic = a.source_elastic_strain[:, None, None, :]+np.einsum("cqrai,ci->cqra", B, local)
    stress = np.einsum("cab,cqrb->cqra", a.C, elastic)
    energy = .5*np.einsum("c,cr,cqra,cqra->", a.volume, measure, elastic, stress)/3
    local_f = np.einsum("c,cr,cqrai,cqra->ci", a.volume, measure, B, stress)/3
    force = np.zeros(a.ndof)
    np.add.at(force, a.dofs.ravel(), local_f.ravel())
    local_k = np.einsum("c,cr,cqrai,cab,cqrbj->cij", a.volume, measure, B, a.C, B)/3
    predicted = np.einsum("cij,cj->ci", local_k, local)
    global_predicted = np.zeros(a.ndof)
    np.add.at(global_predicted, a.dofs.ravel(), predicted.ravel())
    assert r.energy_j == pytest.approx(energy, rel=4e-15)
    np.testing.assert_allclose(r.force, force, rtol=2e-13, atol=1.)
    np.testing.assert_allclose(a.K@displacement, global_predicted, rtol=2e-13, atol=.1)


def test_physical_maxwell_force_tangent_uses_cohort_beta(actual_operator):
    a = actual_operator
    rng = np.random.default_rng(6831)
    beta = rng.uniform(.2, 1., len(a.cohort_ids))
    q, direction = rng.normal(size=a.ndof)*.01, rng.normal(size=a.ndof)*.002
    r0 = evaluate_genesis_volume_operator(a, q, strain_factors=beta)
    r1 = evaluate_genesis_volume_operator(a, q+direction, strain_factors=beta)
    np.testing.assert_allclose(r1.force-r0.force, algorithmic_stiffness(a, beta)@direction,
                               rtol=3e-7, atol=15.)


def test_old_membrane_virtual_work_pullback_keeps_actual_cohort_prestress(actual_operator, source_packet):
    from tectonics.genesis_faults import load_fault_checkpoint
    from tectonics.genesis_shell import Membrane
    import base64
    import io
    a = actual_operator
    model, state, _, _, _ = load_fault_checkpoint(io.BytesIO(base64.b64decode(
        source_packet["source"]["checkpoint_base64"])))
    membrane = Membrane(model.mesh_for(state), model.p.poisson_ratio)
    rng = np.random.default_rng(7232)
    old = rng.normal(size=membrane.ndof)*1e-7
    radius = state.radius_km*1000
    u = radius*np.einsum("vij,vj->vi", membrane.vertex_basis, old[:-1].reshape(-1, 2))
    u += old[-1]*np.asarray(a.geometry.vertices_m)
    full = np.column_stack((u, u)).ravel()
    strain = np.einsum("fai,fi->fa", membrane.b, old[membrane.dofs])
    expected = np.sum(a.volume*np.einsum("ci,ci->c", a.source_stress_pa[:, (0, 1, 3)], strain[a.face_indices]))
    assert np.dot(a.force, full) == pytest.approx(expected, rel=2e-13)


def test_resealed_fake_source_material_is_refused(source_packet):
    packet = deepcopy(source_packet)
    packet["instantaneous_cohort_plane_stress_3d"]["elastic_strain_local"][0][0][0] += .01
    with pytest.raises(ValueError):
        assemble_genesis_volume_operator(packet)


def test_updated_birth_material_requires_actual_prospective_memory(actual_operator, source_packet):
    from tectonics.genesis_bulk_birth import bulk_birth_from_dict
    from tectonics.genesis_volume_geometry import build_updated_genesis_volume_geometry
    a = actual_operator
    path = Path(__file__).resolve().parents[1]/"analysis/genesis_cohort_mechanics_validation/evidence/control/bulk_history_coupled.json"
    history = bulk_birth_from_dict(json.loads(path.read_text()))
    geometry = build_updated_genesis_volume_geometry(a.geometry, history)
    updated = assemble_updated_volume_operator(a, geometry,
        source_packet["current_shell_memory"]["damage"])
    assert len(updated.cohort_ids) == 640
    assert updated.requires_explicit_memory
    assert updated.source_elastic_energy_j is None
    assert updated.source_elastic_strain is None
    assert updated.source_stress_pa is None
    assert updated.initial_force is None
    with pytest.raises(ValueError, match="prospectively calculated"):
        evaluate_genesis_volume_operator(updated, np.zeros(updated.ndof))
    with pytest.raises(ValueError, match="prospectively calculated"):
        _ = updated.memory0
    # This test supplies a declared stress-free birth activation; no history
    # is assigned by the operator. Existing cohort tensors remain inherited.
    memory0, memory1 = np.zeros((640, 3, 6)), np.zeros((640, 3, 6))
    old = {identity: index for index, identity in enumerate(a.cohort_ids)}
    for index, identity in enumerate(updated.cohort_ids):
        if identity in old:
            memory0[index] = a.memory0[old[identity]]
    r = evaluate_genesis_volume_operator(updated, np.zeros(updated.ndof),
        elastic_constant=memory0, elastic_inverse_scale=memory1)
    assert r.energy_j == pytest.approx(a.source_elastic_energy_j, rel=3e-15)
    assert updated.geometry.source_packet_json == a.geometry.source_packet_json
    assert updated.geometry.vertices_m == a.geometry.vertices_m
    # A subsequent integration keeps the already accepted runtime prefix;
    # even a same-clock constitutive reassembly must retain that authority.
    repeated = assemble_updated_volume_operator(updated, geometry,
        source_packet["current_shell_memory"]["damage"])
    np.testing.assert_array_equal(repeated.K.toarray(), updated.K.toarray())
    assert repeated.requires_explicit_memory
