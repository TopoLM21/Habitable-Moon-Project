"""Declared mechanical material fixtures, never real planetary fault activation.

The controls integrate full angular/radial/time fields and differentiate the
actual physical average-force operator. Their prescribed active planes only
exercise the constitutive kernel; they do not initialize a mature planet.
"""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tectonics.genesis import SECONDS_PER_MYR
from tectonics.genesis_fault_law import WeakPlaneParameters
from tectonics.genesis_volume_fault_slip import (
    ORIENTATION_CLOSURE, TRACTION_CLOSURE, VolumeSlipIncrementBoundary,
    assemble_volume_fault_average_force_tangent, evaluate_volume_fault_slip,
    prepare_volume_fault_slip,
)
from tectonics.genesis_volume_loading import (
    RadialMoments, VolumeCohortMemory, VolumeLoading, VolumeMechanicsState,
    average_volume_loading_coefficients, coefficient_inner_product,
    evaluate_volume_loading, isotropic_elasticity,
)
from tectonics.genesis_volume_operator import (
    _construct_operator, algorithmic_stiffness, engineering_strain_to_tensor,
    tensor_to_engineering_strain,
)


def fixture(*, planes=True, parameters=None, ratios=(.6, 1.8), basis=None):
    """Two adjacent conical intervals with deliberately nonzero field variance."""
    count = len(ratios)
    p = (WeakPlaneParameters(cohesion_pa=1e5, viscosity_pa_s=3e18)
         if parameters is None else parameters)
    triangle = np.array(((0., 0., 10.), (2., 0., 10.), (0., 2., 10.)))
    outer = np.linspace(.95, .8, count)
    inner = outer-.12
    volume = np.linalg.det(triangle)/3*(outer**3-inner**3)
    mean = 1.5*(outer+inner)/(outer**2+outer*inner+inner**2)
    variance = .75*((outer-inner)/(outer**2+outer*inner+inner**2))**2
    moments = RadialMoments(tuple(volume), tuple(mean), tuple(variance))
    frames = np.repeat(np.eye(3)[None], count, axis=0) if basis is None else np.asarray(basis)
    rng = np.random.default_rng(98017)
    world0 = rng.normal(size=(count, 3, 6))*.00012
    world0[..., :3] += (-.0007, -.0006, -.0008)
    world0[..., 3] += .006
    world1 = rng.normal(size=(count, 3, 6))*.00008
    old0 = tensor_to_engineering_strain(frames[:, None].transpose(0, 1, 3, 2)
        @engineering_strain_to_tensor(world0)@frames[:, None])
    old1 = tensor_to_engineering_strain(frames[:, None].transpose(0, 1, 3, 2)
        @engineering_strain_to_tensor(world1)@frames[:, None])
    damage, water = np.array((.8,)), np.array((.2,))
    c = np.repeat((isotropic_elasticity(1e9, .25)
        *(.02+.98*(1-damage[0])**2))[None], count, axis=0)
    energy = .5*coefficient_inner_product(old0, old1, old0, old1, c, moments).sum()
    state = VolumeMechanicsState("a"*64, 1., 0,
        tuple(VolumeCohortMemory(i+1, 1000., old0[i], old1[i]) for i in range(count)),
        float(energy), float(energy))
    ratio = np.asarray(ratios, float)
    beta = np.divide(-np.expm1(-ratio), ratio, out=np.ones(count), where=ratio != 0)
    thermal = np.zeros_like(old0)
    thermal[..., :3] = -.00003
    loading = VolumeLoading(state, tuple(range(1, count+1)), 1.0001,
        np.full(count, 997.), moments, old0, old1, thermal,
        np.exp(-ratio)[:, None, None]*old0-beta[:, None, None]*thermal,
        np.exp(-ratio)[:, None, None]*old1, ratio, beta, c, 0., 0.)
    cohorts = tuple(SimpleNamespace(id=i+1, face_index=0, volume_m3=volume[i],
        top_triangle_m=triangle, outer_scale=outer[i], inner_scale=inner[i]) for i in range(count))
    geometry = SimpleNamespace(cohorts=cohorts, faces=((0, 1, 2),),
        vertices_m=tuple(map(tuple, triangle)), source_checkpoint_sha256="b"*64,
        source_identity="a"*64, time_myr=1.0001)
    assembly = _construct_operator(geometry, c, frames)
    prepared = prepare_volume_fault_slip(loading,
        face_indices=np.zeros(count, dtype=int), basis_xyz=frames,
        plane_normal_world=np.array(((1., 0., 0.),)),
        slip_direction_world=np.array(((0., 1., 0.),)),
        active=np.array((planes,), bool), damage=damage, water=water, parameters=p)
    return prepared, assembly, outer, inner


def increments(prepared, seed=891):
    rng = np.random.default_rng(seed)
    shape = prepared.loading.old0.shape
    return rng.normal(size=shape)*.00004, rng.normal(size=shape)*.00003


def prepare_again(prepared, **changes):
    values = dict(face_indices=prepared.face_indices, basis_xyz=prepared.basis_xyz,
        plane_normal_world=prepared.plane_normal_world,
        slip_direction_world=prepared.slip_direction_world, active=prepared.active,
        damage=prepared.damage, water=prepared.water, parameters=prepared.parameters)
    return prepare_volume_fault_slip(changes.pop("loading", prepared.loading), **(values | changes))


def test_closures_are_explicit_mean_traction_and_frozen_world_orientation():
    assert TRACTION_CLOSURE == "exact_interval_average_constant_rate_Maxwell_path"
    assert ORIENTATION_CLOSURE == "accepted_start_world_plane_frozen_over_small_strain_interval"


@pytest.mark.parametrize("planes,enabled", ((False, True), (True, False)))
def test_inactive_or_disabled_kernel_keeps_full_base_response_exact(planes, enabled):
    prepared, _, _, _ = fixture(planes=planes,
        parameters=WeakPlaneParameters(enabled=enabled))
    inc0, inc1 = increments(prepared)
    actual = evaluate_volume_fault_slip(prepared, inc0, inc1)
    expected = evaluate_volume_loading(prepared.loading, inc0, inc1)
    for name in expected.__dataclass_fields__:
        assert np.array_equal(getattr(actual.loading_response, name), getattr(expected, name))
    expected_avg = average_volume_loading_coefficients(prepared.loading, inc0, inc1)
    np.testing.assert_array_equal(actual.average_strain0, expected_avg[0])
    np.testing.assert_array_equal(actual.average_strain1, expected_avg[1])
    assert actual.plastic_work_j == actual.friction_work_j == actual.viscous_fault_work_j == 0.
    assert actual.physical_mechanical_work_j == expected.mechanical_work_j


def test_signed_yield_has_exact_nonnegative_work_split_and_normal_invariance():
    prepared, _, _, _ = fixture()
    inc0, inc1 = increments(prepared)
    result = evaluate_volume_fault_slip(prepared, inc0, inc1)
    assert result.yielding[0]
    assert result.shear_increment[0] > 0
    expected = result.yield_strength_pa+prepared.viscous_modulus_pa*result.shear_increment
    np.testing.assert_allclose(result.shear_stress_pa, expected, rtol=2e-14)
    assert result.friction_work_j > 0 and result.viscous_fault_work_j > 0
    assert result.plastic_work_j == result.friction_work_j+result.viscous_fault_work_j
    assert result.physical_mechanical_work_j == pytest.approx(
        result.loading_response.mechanical_work_j+result.plastic_work_j, rel=2e-13)
    assert abs(result.balance_residual_j) < 5e-12*prepared.loading.before_state.elastic_energy_j
    for c, p, q in zip(prepared.loading.elasticity_pa, prepared.normal_projector, prepared.flow_direction):
        assert p@c@q == pytest.approx(0., abs=1e-10)
        assert q@c@q == pytest.approx(c[3, 3], rel=1e-15)


def test_reversing_tangent_reverses_gamma_without_changing_physical_memory_or_work():
    prepared, _, _, _ = fixture()
    inc0, inc1 = increments(prepared)
    first = evaluate_volume_fault_slip(prepared, inc0, inc1)
    reversed_plane = prepare_again(prepared, slip_direction_world=-prepared.slip_direction_world)
    second = evaluate_volume_fault_slip(reversed_plane, inc0, inc1)
    np.testing.assert_array_equal(second.shear_increment, -first.shear_increment)
    np.testing.assert_array_equal(second.plastic_increment0, first.plastic_increment0)
    np.testing.assert_array_equal(second.loading_response.strain0, first.loading_response.strain0)
    assert second.friction_work_j == first.friction_work_j
    assert second.viscous_fault_work_j == first.viscous_fault_work_j


def test_plane_sign_has_no_physical_effect_when_both_vectors_reverse():
    prepared, _, _, _ = fixture()
    inc0, inc1 = increments(prepared)
    first = evaluate_volume_fault_slip(prepared, inc0, inc1)
    second = evaluate_volume_fault_slip(prepare_again(prepared,
        plane_normal_world=-prepared.plane_normal_world,
        slip_direction_world=-prepared.slip_direction_world), inc0, inc1)
    np.testing.assert_array_equal(second.loading_response.strain0, first.loading_response.strain0)
    np.testing.assert_array_equal(second.shear_increment, first.shear_increment)
    assert second.plastic_work_j == first.plastic_work_j


def test_all_cohort_beta_and_angular_radial_memory_survive_common_face_slip():
    prepared, _, _, _ = fixture()
    inc0, inc1 = increments(prepared)
    result = evaluate_volume_fault_slip(prepared, inc0, inc1)
    loading = prepared.loading
    np.testing.assert_allclose(result.loading_response.strain0,
        loading.memory0+loading.beta[:, None, None]*(inc0-result.plastic_increment0), rtol=1e-15)
    np.testing.assert_array_equal(result.loading_response.strain1,
        loading.memory1+loading.beta[:, None, None]*inc1)
    assert loading.beta[0] != loading.beta[1]
    assert prepared.theta[0] != prepared.theta[1]
    endpoint_diff = result.loading_response.strain0[:, 1]-result.loading_response.strain0[:, 0]
    expected_diff = (loading.memory0[:, 1]-loading.memory0[:, 0]
        +loading.beta[:, None]*(inc0[:, 1]-inc0[:, 0]))
    np.testing.assert_allclose(endpoint_diff, expected_diff, rtol=2e-14, atol=1e-18)
    volume, variance = map(np.asarray, (loading.radial_moments.volume_m3,
                                      loading.radial_moments.inverse_radius_variance))
    e1 = result.loading_response.strain1
    radial_variance_energy = .5*np.sum(volume*variance
        *np.einsum("cqi,cij,cqj->cq", e1, loading.elasticity_pa, e1).mean(axis=1))
    assert radial_variance_energy > 0


def test_newborn_zero_ratio_retains_half_average_and_full_endpoint_current_slip():
    prepared, _, _, _ = fixture(ratios=(.6, 0.))
    loading = prepared.loading
    old0, old1 = loading.old0.copy(), loading.old1.copy()
    old0[1] = old1[1] = 0.
    energy = float(.5*coefficient_inner_product(old0, old1, old0, old1,
        loading.elasticity_pa, loading.radial_moments).sum())
    state = replace(loading.before_state,
        cohorts=tuple(VolumeCohortMemory(i+1, 1000., old0[i], old1[i]) for i in range(2)),
        initial_energy_j=energy, elastic_energy_j=energy)
    loading = replace(loading, before_state=state, old0=old0, old1=old1,
        memory0=np.exp(-loading.ratio)[:, None, None]*old0-loading.beta[:, None, None]*loading.thermal0,
        memory1=np.exp(-loading.ratio)[:, None, None]*old1)
    prepared = prepare_again(prepared, loading=loading)
    inc0, inc1 = increments(prepared)
    result = evaluate_volume_fault_slip(prepared, inc0, inc1)
    assert prepared.theta[1] == .5
    np.testing.assert_array_equal(result.loading_response.strain0[1],
        inc0[1]-loading.thermal0[1]-result.plastic_increment0[1])
    np.testing.assert_array_equal(result.average_strain0[1],
        .5*(inc0[1]-loading.thermal0[1]-result.plastic_increment0[1]))


def test_full_work_and_maxwell_losses_match_independent_time_and_radial_quadrature():
    prepared, _, outer, inner = fixture()
    inc0, inc1 = increments(prepared)
    result = evaluate_volume_fault_slip(prepared, inc0, inc1)
    nodes, weights = np.polynomial.legendre.leggauss(64)
    times, time_weights = (nodes+1)/2, weights/2
    totals = np.zeros(5)  # physical, thermal, plastic, Maxwell, endpoint energy
    for i, (a, b) in enumerate(zip(outer, inner)):
        radius = b+(a-b)*(nodes+1)/2
        volume_weights = weights*(a-b)/2*radius**2
        volume_weights *= prepared.loading.radial_moments.volume_m3[i]/((a**3-b**3)/3)
        old = prepared.loading.old0[i, :, None]+prepared.loading.old1[i, :, None]/radius[None, :, None]
        total_inc = inc0[i, :, None]+inc1[i, :, None]/radius[None, :, None]
        thermal = prepared.loading.thermal0[i, :, None]
        plastic = result.plastic_increment0[i, :, None]
        driver = total_inc-thermal-plastic
        x = prepared.loading.ratio[i]
        relaxation = np.exp(-x*times)
        driven = -np.expm1(-x*times)/x
        eps = (relaxation[None, None, :, None]*old[:, :, None]
            +driven[None, None, :, None]*driver[:, :, None])
        stress = np.einsum("ij,artj->arti", prepared.loading.elasticity_pa[i], eps)
        for k, driving in enumerate((total_inc, -thermal, plastic)):
            totals[k] += np.einsum("arj,artj,r,t->", driving, stress,
                volume_weights, time_weights)/3
        totals[3] += x*np.einsum("artj,artj,r,t->", eps, stress,
            volume_weights, time_weights)/3
        endpoint = np.exp(-x)*old+(-np.expm1(-x)/x)*driver
        totals[4] += .5*np.einsum("arj,ij,ari,r->", endpoint,
            prepared.loading.elasticity_pa[i], endpoint, volume_weights)/3
    expected = (result.physical_mechanical_work_j, result.loading_response.thermal_work_j,
        result.plastic_work_j, result.loading_response.viscous_dissipation_j,
        result.loading_response.elastic_energy_j)
    np.testing.assert_allclose(totals, expected, rtol=5e-13, atol=1e-10)


def test_rigid_world_rotation_changes_no_constitutive_scalar_or_local_memory():
    prepared, _, _, _ = fixture()
    inc0, inc1 = increments(prepared)
    angle = .72
    rotation = np.array(((np.cos(angle), 0., np.sin(angle)), (0., 1., 0.),
                         (-np.sin(angle), 0., np.cos(angle))))
    rotated = prepare_again(prepared, basis_xyz=rotation@prepared.basis_xyz,
        plane_normal_world=prepared.plane_normal_world@rotation.T,
        slip_direction_world=prepared.slip_direction_world@rotation.T)
    first = evaluate_volume_fault_slip(prepared, inc0, inc1)
    second = evaluate_volume_fault_slip(rotated, inc0, inc1)
    np.testing.assert_allclose(second.loading_response.strain0, first.loading_response.strain0,
        rtol=1e-14, atol=1e-18)
    assert second.plastic_work_j == pytest.approx(first.plastic_work_j, rel=2e-14)
    assert second.loading_response.elastic_energy_j == pytest.approx(first.loading_response.elastic_energy_j, rel=2e-14)


def test_different_cohort_bases_preserve_same_world_plane_traction_and_energy():
    prepared, _, _, _ = fixture()
    angle = .53
    rotation = np.array(((np.cos(angle), -np.sin(angle), 0.),
                         (np.sin(angle), np.cos(angle), 0.), (0., 0., 1.)))
    frames = np.stack((np.eye(3), rotation))
    rotated, _, _, _ = fixture(basis=frames)
    inc0, inc1 = increments(prepared)
    rotated_inc = [tensor_to_engineering_strain(frames[:, None].transpose(0, 1, 3, 2)
        @engineering_strain_to_tensor(inc)@frames[:, None]) for inc in (inc0, inc1)]
    first = evaluate_volume_fault_slip(prepared, inc0, inc1)
    second = evaluate_volume_fault_slip(rotated, *rotated_inc)
    assert not np.array_equal(rotated.flow_direction[0], rotated.flow_direction[1])
    np.testing.assert_allclose(second.shear_increment, first.shear_increment, rtol=2e-14)
    np.testing.assert_allclose(second.normal_stress_pa, first.normal_stress_pa, rtol=2e-14)
    assert second.loading_response.elastic_energy_j == pytest.approx(first.loading_response.elastic_energy_j, rel=2e-14)


def test_exact_mean_force_tangent_matches_all_physical_dofs_and_is_nonsymmetric():
    prepared, assembly, _, _ = fixture()
    dq = np.random.default_rng(119).normal(size=assembly.ndof)*.00002
    def evaluate(q):
        local = q[assembly.dofs]
        inc0 = np.einsum("cqai,ci->cqa", assembly.B0, local)
        inc1 = np.einsum("cqai,ci->cqa", assembly.B1, local)
        response = evaluate_volume_fault_slip(prepared, inc0, inc1)
        return response, assemble_volume_fault_average_force_tangent(assembly, prepared, response)
    response, force = evaluate(dq)
    assert response.yielding[0] and response.normal_stress_pa[0] < 0
    assert force.force_n@dq == pytest.approx(response.physical_mechanical_work_j, rel=5e-13)
    dense = force.tangent.toarray()
    assert np.linalg.norm(dense-dense.T) > 1e-5*np.linalg.norm(dense)
    numerical = np.zeros_like(dense)
    epsilon = 1e-7
    for j in range(assembly.ndof):
        direction = np.zeros(assembly.ndof); direction[j] = epsilon
        numerical[:, j] = (evaluate(dq+direction)[1].force_n-evaluate(dq-direction)[1].force_n)/(2*epsilon)
    np.testing.assert_allclose(dense, numerical, rtol=2e-7, atol=3e-3)
    endpoint_tangent = algorithmic_stiffness(assembly, prepared.loading.beta).toarray()
    assert np.linalg.norm(dense-endpoint_tangent) > .1*np.linalg.norm(dense)


def test_inactive_mean_force_tangent_is_exact_theta_stiffness():
    prepared, assembly, _, _ = fixture(planes=False)
    inc0, inc1 = increments(prepared)
    result = evaluate_volume_fault_slip(prepared, inc0, inc1)
    force = assemble_volume_fault_average_force_tangent(assembly, prepared, result)
    np.testing.assert_array_equal(force.tangent.toarray(),
        algorithmic_stiffness(assembly, prepared.theta).toarray())


def test_limit_rejects_candidate_without_clipping_or_mutating_old_memory():
    prepared, _, _, _ = fixture(parameters=WeakPlaneParameters(
        cohesion_pa=0., friction_dry=0., friction_wet=0., viscosity_pa_s=1e10))
    old0, old1 = prepared.loading.old0.copy(), prepared.loading.old1.copy()
    with pytest.raises(VolumeSlipIncrementBoundary) as caught:
        evaluate_volume_fault_slip(prepared, np.zeros_like(old0), np.zeros_like(old1))
    assert np.max(np.abs(caught.value.shear_increment)) > caught.value.limit
    np.testing.assert_array_equal(prepared.loading.old0, old0)
    np.testing.assert_array_equal(prepared.loading.old1, old1)


@pytest.mark.parametrize("change", (
    {"plane_normal_world": ((2., 0., 0.),)},
    {"slip_direction_world": ((1., 0., 0.),)},
    {"plane_normal_world": ((np.nan, 0., 0.),)},
    {"active": (1,)},
    {"face_indices": (0., 0.)},
    {"water": (1.1,)},
    {"damage": (-.1,)},
    {"basis_xyz": np.repeat(np.diag((1., 1., -1.))[None], 2, axis=0)},
))
def test_invalid_planes_masks_strengths_and_bases_are_rejected(change):
    prepared, _, _, _ = fixture()
    with pytest.raises(ValueError):
        prepare_again(prepared, **change)


def test_anisotropic_matrix_is_rejected_instead_of_assuming_normal_invariance():
    prepared, _, _, _ = fixture()
    matrix = prepared.loading.elasticity_pa.copy()
    matrix[:, 0, 3] += 1e5; matrix[:, 3, 0] += 1e5
    with pytest.raises(ValueError, match="isotropic"):
        prepare_again(prepared, loading=replace(prepared.loading, elasticity_pa=matrix))


def test_force_helper_rejects_unbound_cohort_or_radial_operator():
    prepared, assembly, _, _ = fixture()
    result = evaluate_volume_fault_slip(prepared, *increments(prepared))
    with pytest.raises(ValueError, match="bindings"):
        assemble_volume_fault_average_force_tangent(
            replace(assembly, inverse_scale_mean=assembly.inverse_scale_mean+1e-4), prepared, result)


def test_force_helper_rejects_response_from_another_frozen_water_preparation():
    prepared, assembly, _, _ = fixture()
    other = prepare_again(prepared, water=(1.,))
    response = evaluate_volume_fault_slip(other, *increments(other))
    with pytest.raises(ValueError, match="different frozen preparation"):
        assemble_volume_fault_average_force_tangent(assembly, prepared, response)


def test_preparations_and_results_are_immutable_across_repeated_private_trials():
    prepared, _, _, _ = fixture()
    first = evaluate_volume_fault_slip(prepared, *increments(prepared))
    evaluate_volume_fault_slip(prepared, *increments(prepared, seed=701))
    second = evaluate_volume_fault_slip(prepared, *increments(prepared))
    np.testing.assert_array_equal(first.loading_response.strain0, second.loading_response.strain0)
    for array in (prepared.flow_direction, prepared.theta, prepared.face_indices,
                  prepared.loading.old0, prepared.loading.elasticity_pa,
                  first.loading_response.strain0, first.loading_response.strain1,
                  first.shear_increment, first.average_strain0, first.plastic_increment0):
        with pytest.raises(ValueError):
            array.flat[0] = 0.
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_preparation_detaches_from_mutable_outer_loading_arrays():
    prepared, _, _, _ = fixture()
    old = prepared.loading
    mutable = replace(old, old0=old.old0.copy(), memory0=old.memory0.copy())
    detached = prepare_again(prepared, loading=mutable)
    before = evaluate_volume_fault_slip(detached, *increments(detached))
    mutable.old0.setflags(write=True); mutable.old0.fill(.01)
    mutable.memory0.setflags(write=True); mutable.memory0.fill(.02)
    after = evaluate_volume_fault_slip(detached, *increments(detached))
    np.testing.assert_array_equal(after.loading_response.strain0, before.loading_response.strain0)
    assert after.physical_mechanical_work_j == before.physical_mechanical_work_j


@pytest.mark.parametrize("water", (0., .5, 1.))
def test_original_strength_parameters_and_water_interpolation_are_retained(water):
    prepared, _, _, _ = fixture(parameters=WeakPlaneParameters())
    prepared = prepare_again(prepared, water=(water,))
    p = prepared.parameters
    assert prepared.friction[0] == pytest.approx(p.friction_dry+(p.friction_wet-p.friction_dry)*water)
    assert prepared.cohesion_pa[0] == pytest.approx(p.cohesion_pa*(p.residual_cohesion_fraction
        +(1-p.residual_cohesion_fraction)*(1-prepared.damage[0]))
        *(1-(1-p.wet_cohesion_fraction)*water))
    assert p.activation_damage == .65 and p.activation_persistence_myr == .05
    assert p.band_width_km == 100. and p.max_shear_increment == .002
