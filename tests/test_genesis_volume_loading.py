"""Independent analytic/integral controls for prospective 3D cohort memory."""
import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tectonics.genesis import SECONDS_PER_MYR
from tectonics.genesis_volume_loading import (
    RadialMoments, VolumeCohortMemory, VolumeMechanicsState,
    accept_volume_loading, average_volume_loading_coefficients,
    average_volume_loading_stiffness_factors,
    coefficient_inner_product, evaluate_volume_loading,
    initial_volume_mechanics, isotropic_elasticity, prepare_volume_loading,
    volume_mechanics_energy, volume_mechanics_from_dict, volume_mechanics_to_dict,
)


def parameters(**changes):
    values = dict(young_modulus_pa=1000., poisson_ratio=.25,
        residual_stiffness=.02, linear_expansion_per_k=.00001,
        viscosity_reference_pa_s=3e15, viscosity_reference_temperature_k=1000.,
        activation_energy_j_mol=0., viscosity_min_pa_s=1., viscosity_max_pa_s=1e28)
    return SimpleNamespace(**{**values, **changes})


def moments(volume=(2.,), mean=(1.2,), variance=(.08,)):
    return RadialMoments(volume, mean, variance)


def source(p=None, ids=(1,), strain=None, temperature=None, radial=None):
    p = parameters() if p is None else p
    n = len(ids)
    strain = np.tile((.001, -.002, .0001, .0004, -.0005, .0006), (n, 1)) if strain is None else strain
    temperature = np.full(n, 1000.) if temperature is None else temperature
    radial = moments(tuple(np.full(n, 2.)), tuple(np.full(n, 1.2)), tuple(np.full(n, .08))) if radial is None else radial
    return initial_volume_mechanics("a"*64, 1., ids, temperature, strain, radial, np.zeros(n), p)


def prepare(state, x=1., *, p=None, radial=None, after_ids=None, temperatures=None,
            before=None, removed=None, before_damage=None, after_damage=None,
            old_profile_temperatures=None):
    p = parameters() if p is None else p
    ids = tuple(c.id for c in state.cohorts) if after_ids is None else after_ids
    n, m = len(ids), len(state.cohorts)
    radial = moments(tuple(np.full(n, 2.)), tuple(np.full(n, 1.2)), tuple(np.full(n, .08))) if radial is None else radial
    before = moments(tuple(np.full(m, 2.)), tuple(np.full(m, 1.2)), tuple(np.full(m, .08))) if before is None else before
    removed = moments(tuple(np.zeros(m)), tuple(np.zeros(m)), tuple(np.zeros(m))) if removed is None else removed
    temp = np.full(n, 1000.) if temperatures is None else temperatures
    return prepare_volume_loading(state, after_ids=ids, after_temperature_k=temp,
        birth_temperature_k=1400., radial_moments=radial, before_radial_moments=before,
        removed_radial_moments=removed, before_damage=np.zeros(m) if before_damage is None else before_damage,
        after_damage=np.zeros(n) if after_damage is None else after_damage, parameters=p,
        after_time_myr=state.time_myr+x*p.viscosity_reference_pa_s/p.young_modulus_pa/SECONDS_PER_MYR,
        before_temperature_k=old_profile_temperatures)


def reseal(raw):
    body = {k: v for k, v in raw.items() if k != "packet_sha256"}
    raw["packet_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def band_moments(lo, hi, coefficient=.6, column_volume=4.):
    a, b = (1-coefficient*lo)**(1/3), (1-coefficient*hi)**(1/3)
    denominator = a*a+a*b+b*b
    return moments((column_volume*(hi-lo),), (1.5*(a+b)/denominator,),
                   (.75*((a-b)/denominator)**2,))


def test_source_import_inherits_nonzero_plane_stress_energy_and_all_shears():
    p = parameters()
    eps = np.array([[.002, -.001, -p.poisson_ratio/(1-p.poisson_ratio)*.001, .0002, 0., 0.]])
    state = source(p, strain=eps)
    c = isotropic_elasticity(p.young_modulus_pa, p.poisson_ratio)
    stress = c@eps[0]
    assert stress[2] == pytest.approx(0., abs=1e-15)
    expected = .5*2*np.dot(eps[0], stress)
    assert state.initial_energy_j == pytest.approx(expected)
    assert state.elastic_energy_j == pytest.approx(expected, rel=1e-15)
    assert np.array_equal(np.asarray(state.cohorts[0].strain0), np.repeat(eps, 3, axis=0))
    assert not np.any(state.cohorts[0].strain1)


def test_radial_and_angular_variance_retain_zero_mean_bending_energy():
    c = isotropic_elasticity(1000., .25)
    m = moments()
    e1 = np.zeros((1, 3, 6)); e1[0, :, 0] = (.003, -.001, .002)
    e0 = -1.2*e1
    actual = coefficient_inner_product(e0, e1, e0, e1, c, m)[0]
    expected = 2*.08*c[0, 0]*(.003**2+.001**2+.002**2)/3
    assert actual == pytest.approx(expected, rel=1e-15)
    assert actual > 0


def test_coefficient_inner_product_matches_independent_q_angular_quadrature():
    rng = np.random.default_rng(518)
    e0, e1, f0, f1 = [rng.normal(size=(1, 3, 6))*.001 for _ in range(4)]
    m = band_moments(.1, .8)
    c = isotropic_elasticity(1000., .25)
    nodes, weights = np.polynomial.legendre.leggauss(64)
    q = .1+(.8-.1)*(nodes+1)/2
    scale = (1-.6*q)**(1/3)
    elastic = e0[0, :, None, :]+e1[0, :, None, :]/scale[None, :, None]
    other = f0[0, :, None, :]+f1[0, :, None, :]/scale[None, :, None]
    independent = 4*(.8-.1)/6*np.einsum("aqi,ij,aqj,q->", elastic, c, other, weights)
    assert coefficient_inner_product(e0, e1, f0, f1, c, m)[0] == pytest.approx(independent, rel=2e-14)


@pytest.mark.parametrize("x", (1e-10, 1e-4, .1, 1., 20., 100.))
def test_pure_relaxation_exact_energy_and_positive_dissipation(x):
    state = source()
    loading = prepare(state, x)
    actual_x = loading.ratio[0]
    response = evaluate_volume_loading(loading, np.zeros((1, 3, 6)), np.zeros((1, 3, 6)))
    expected = state.elastic_energy_j*np.exp(-2*actual_x)
    assert response.elastic_energy_j == pytest.approx(expected, rel=2e-14)
    assert response.viscous_dissipation_j == pytest.approx(state.elastic_energy_j*(-np.expm1(-2*actual_x)), rel=2e-12)
    assert response.mechanical_work_j == response.thermal_work_j == 0
    assert response.viscous_dissipation_j >= 0
    accepted = accept_volume_loading(loading, response)
    assert accepted.elastic_energy_j == response.elastic_energy_j
    assert accepted.initial_energy_j == state.initial_energy_j


def test_cooling_and_strain_work_match_independent_time_quadrature():
    state = source()
    loading = prepare(state, 1.5, temperatures=np.array([980.]))
    rng = np.random.default_rng(92)
    inc0, inc1 = [rng.normal(size=(1, 3, 6))*.001 for _ in range(2)]
    response = evaluate_volume_loading(loading, inc0, inc1)
    times, weights = np.polynomial.legendre.leggauss(80)
    times, weights = (times+1)/2, weights/2
    x, c = loading.ratio[0], loading.elasticity_pa[0]
    thermal = np.zeros((3, 6)); thermal[:, :3] = -.0002
    driver = inc0[0]-thermal
    e = (np.exp(-x*times)[None, :, None]*np.asarray(state.cohorts[0].strain0)[:, None, :]
         +(1-np.exp(-x*times))[None, :, None]/x*driver[:, None, :])
    e1 = (1-np.exp(-x*times))[None, :, None]/x*inc1[0, :, None, :]
    mean = e+1.2*e1
    mechanical = 2/3*np.einsum("ai,ij,atj,t->", inc0[0]+1.2*inc1[0], c, mean, weights)
    mechanical += 2*.08/3*np.einsum("ai,ij,atj,t->", inc1[0], c, e1, weights)
    thermal_work = -2/3*np.einsum("ai,ij,atj,t->", thermal, c, mean, weights)
    dissipation = 2*x/3*np.einsum("ati,ij,atj,t->", mean, c, mean, weights)
    dissipation += 2*x*.08/3*np.einsum("ati,ij,atj,t->", e1, c, e1, weights)
    assert response.mechanical_work_j == pytest.approx(mechanical, rel=2e-14)
    assert response.thermal_work_j == pytest.approx(thermal_work, rel=2e-14)
    assert response.viscous_dissipation_j == pytest.approx(dissipation, rel=2e-14)
    assert abs(response.balance_residual_j) < 1e-16


def test_temperature_contrast_gives_cohort_local_viscosity():
    p = parameters(activation_energy_j_mol=150000., viscosity_reference_pa_s=1e20,
                   viscosity_max_pa_s=1e28, viscosity_min_pa_s=1e10)
    state = source(p, ids=(1, 2), temperature=np.array([900., 1200.]))
    loading = prepare(state, 1., p=p, temperatures=np.array([900., 1200.]))
    assert loading.ratio[1] > 100*loading.ratio[0]
    assert loading.beta[1] < loading.beta[0]


def test_birth_only_current_increment_and_solidus_cooling():
    state = source()
    loading = prepare(state, after_ids=(1, 2), temperatures=np.array([1000., 1350.]))
    assert loading.ratio[1] == 0 and loading.beta[1] == 1
    assert not np.any(loading.old0[1]) and not np.any(loading.old1[1])
    inc0, inc1 = np.full((2, 3, 6), .0003), np.full((2, 3, 6), .0002)
    response = evaluate_volume_loading(loading, inc0, inc1)
    expected = inc0[1].copy(); expected[:, :3] += .0005
    np.testing.assert_array_equal(response.strain0[1], expected)
    np.testing.assert_array_equal(response.strain1[1], inc1[1])
    assert response.viscous_dissipation_j >= 0


def test_heterogeneous_cohort_tangent_is_beta_times_own_full_elasticity():
    p = parameters(activation_energy_j_mol=150000., viscosity_reference_pa_s=1e20,
                   viscosity_max_pa_s=1e28, viscosity_min_pa_s=1e10)
    state = source(p, ids=(1, 2), temperature=np.array([900., 1200.]))
    loading = prepare(state, 1., p=p, temperatures=np.array([900., 1200.]))
    zero = np.zeros((2, 3, 6))
    reference = evaluate_volume_loading(loading, zero, zero)
    direction = np.random.default_rng(66).normal(size=(2, 3, 6))*.001
    step = 1e-4
    perturbed = evaluate_volume_loading(loading, step*direction, step*direction)
    endpoint_stress0 = np.einsum("cij,cqj->cqi", loading.elasticity_pa, perturbed.strain0-reference.strain0)/step
    expected = loading.beta[:, None, None]*np.einsum("cij,cqj->cqi", loading.elasticity_pa, direction)
    np.testing.assert_allclose(endpoint_stress0, expected, rtol=1e-10, atol=1e-13)
    assert loading.beta[0] != loading.beta[1]


def test_prepared_arrays_are_immutable_between_newton_retries():
    loading = prepare(source())
    for name in ("memory0", "memory1", "old0", "old1", "beta", "elasticity_pa"):
        with pytest.raises(ValueError):
            getattr(loading, name).flat[0] = 1.


@pytest.mark.parametrize("x", (1e-12, 1e-6, .0009, .0011, .099, .101, .3, 2., 30.))
def test_average_coefficients_match_independent_high_order_time_quadrature(x):
    state = source()
    # Make a prospective, nonzero radial memory instead of importing an
    # unavailable birth field. The second step averages both old terms.
    first = prepare(state, .4)
    rng = np.random.default_rng(910)
    old_inc0, old_inc1, inc0, inc1 = [rng.normal(size=(1, 3, 6))*.001 for _ in range(4)]
    state = accept_volume_loading(first, evaluate_volume_loading(first, old_inc0, old_inc1))
    loading = prepare(state, x, temperatures=np.array([987.]))
    mean0, mean1 = average_volume_loading_coefficients(loading, inc0, inc1)
    nodes, weights = np.polynomial.legendre.leggauss(160)
    times, weights = (nodes+1)/2, weights/2
    ratio = loading.ratio[0]
    relaxation = np.exp(-ratio*times)
    driven = -np.expm1(-ratio*times)/ratio
    explicit0 = relaxation[None, :, None]*loading.old0[0, :, None, :]
    explicit0 += driven[None, :, None]*(inc0-loading.thermal0)[0, :, None, :]
    explicit1 = relaxation[None, :, None]*loading.old1[0, :, None, :]
    explicit1 += driven[None, :, None]*inc1[0, :, None, :]
    independent0 = np.einsum("atj,t->aj", explicit0, weights)
    independent1 = np.einsum("atj,t->aj", explicit1, weights)
    # Angular components can almost cancel the relaxed and driven fields;
    # bound quadrature roundoff by the contributing field amplitudes.
    amplitude = max(np.max(np.abs(loading.old0)), np.max(np.abs(loading.old1)),
                    np.max(np.abs(inc0-loading.thermal0)), np.max(np.abs(inc1)))
    atol = 32*np.finfo(float).eps*amplitude
    np.testing.assert_allclose(mean0[0], independent0, rtol=2e-13, atol=atol)
    np.testing.assert_allclose(mean1[0], independent1, rtol=2e-13, atol=atol)
    theta = average_volume_loading_stiffness_factors(loading)[0]
    assert theta == pytest.approx(np.dot(driven, weights), rel=2e-14)


def test_average_newborn_path_uses_half_current_increment_and_solidus_cooling():
    state = source()
    loading = prepare(state, after_ids=(1, 2), temperatures=np.array([1000., 1350.]))
    rng = np.random.default_rng(146)
    inc0, inc1 = [rng.normal(size=(2, 3, 6))*.001 for _ in range(2)]
    mean0, mean1 = average_volume_loading_coefficients(loading, inc0, inc1)
    assert average_volume_loading_stiffness_factors(loading)[1] == .5
    np.testing.assert_array_equal(mean0[1], .5*(inc0-loading.thermal0)[1])
    np.testing.assert_array_equal(mean1[1], .5*inc1[1])


def test_average_force_tangent_affine_and_work_equals_its_contraction():
    state = source()
    loading = prepare(state, .35, temperatures=np.array([995.]))
    rng = np.random.default_rng(310)
    inc0, inc1, delta0, delta1 = [rng.normal(size=(1, 3, 6))*.001 for _ in range(4)]
    mean0, mean1 = average_volume_loading_coefficients(loading, inc0, inc1)
    shifted0, shifted1 = average_volume_loading_coefficients(loading, inc0+delta0, inc1+delta1)
    theta = average_volume_loading_stiffness_factors(loading)[:, None, None]
    np.testing.assert_allclose(shifted0-mean0, theta*delta0, rtol=2e-14, atol=1e-18)
    np.testing.assert_allclose(shifted1-mean1, theta*delta1, rtol=2e-14, atol=1e-18)
    response = evaluate_volume_loading(loading, inc0, inc1)
    expected_work = coefficient_inner_product(inc0, inc1, mean0, mean1,
        loading.elasticity_pa, loading.radial_moments).sum()
    assert response.mechanical_work_j == expected_work
    assert average_volume_loading_stiffness_factors(loading)[0] != loading.beta[0]


def test_actual_retired_band_energy_preserves_radial_profile():
    p = parameters()
    whole, retained, removed = band_moments(.1, .8), band_moments(.1, .45), band_moments(.45, .8)
    rng = np.random.default_rng(413)
    e0, e1 = [rng.normal(size=(1, 3, 6))*.001 for _ in range(2)]
    c = isotropic_elasticity(1000., .25)
    initial = .5*coefficient_inner_product(e0, e1, e0, e1, c, whole).sum()
    state = VolumeMechanicsState("a"*64, 1., 0, (VolumeCohortMemory(1, 1000., e0[0], e1[0]),), initial, initial)
    loading = prepare(state, radial=retained, before=whole, removed=removed)
    expected_removed = .5*coefficient_inner_product(e0, e1, e0, e1, c, removed).sum()
    assert loading.remelt_energy_removed_j == expected_removed
    response = evaluate_volume_loading(loading, np.zeros_like(e0), np.zeros_like(e1))
    accepted = accept_volume_loading(loading, response)
    assert accepted.remelt_energy_removed_j == expected_removed
    assert accepted.balance_residual_j == pytest.approx(0., abs=2e-18)


def test_clipped_band_uses_old_profile_on_actual_retained_interval():
    whole, retained, removed = band_moments(.1, .8), band_moments(.1, .45), band_moments(.45, .8)
    state = source(radial=whole)
    # The old whole-band mean was 1000 K, but the surviving shallow part was
    # already 1030 K and has not changed. Merely retiring deeper material
    # must not create a spurious 30 K expansion in retained material.
    loading = prepare(state, radial=retained, before=whole, removed=removed,
        temperatures=np.array([1030.]), old_profile_temperatures=np.array([1030.]))
    assert not np.any(loading.thermal0)
    response = evaluate_volume_loading(loading, np.zeros((1, 3, 6)), np.zeros((1, 3, 6)))
    assert response.thermal_work_j == 0
    default = prepare(state, radial=retained, before=whole, removed=removed,
        temperatures=np.array([1030.]))
    np.testing.assert_allclose(default.thermal0[0, :, :3], np.full((3, 3), .0003), rtol=3e-16, atol=0.)


def test_old_profile_input_never_changes_newborn_solidus_activation():
    state = source()
    loading = prepare(state, after_ids=(1, 2), temperatures=np.array([1000., 1350.]),
        old_profile_temperatures=np.array([1000., 2300.]))
    np.testing.assert_array_equal(loading.thermal0[1, :, :3], np.full((3, 3), -.0005))


def test_damage_change_separate_from_relaxation_dissipation():
    state = source()
    loading = prepare(state, after_damage=np.array([.5]))
    factor = .02+.98*.5**2
    assert loading.damage_energy_change_j == pytest.approx((factor-1)*state.elastic_energy_j)
    response = evaluate_volume_loading(loading, np.zeros((1, 3, 6)), np.zeros((1, 3, 6)))
    assert response.elastic_energy_j == pytest.approx(state.elastic_energy_j*factor*np.exp(-2*loading.ratio[0]))
    assert response.damage_energy_change_j < 0
    assert response.viscous_dissipation_j > 0


def test_restart_typed_memory_and_next_response_exact():
    state = source()
    loading = prepare(state)
    inc0 = np.full((1, 3, 6), .0001); inc1 = np.full_like(inc0, -.00002)
    state = accept_volume_loading(loading, evaluate_volume_loading(loading, inc0, inc1))
    archive = json.loads(json.dumps(volume_mechanics_to_dict(state)))
    restored = volume_mechanics_from_dict(archive)
    assert restored == state
    assert copy.deepcopy(state) is state
    left, right = prepare(state), prepare(restored)
    result_l, result_r = evaluate_volume_loading(left, inc0, inc1), evaluate_volume_loading(right, inc0, inc1)
    np.testing.assert_array_equal(result_l.strain0, result_r.strain0)
    np.testing.assert_array_equal(result_l.strain1, result_r.strain1)
    assert accept_volume_loading(left, result_l) == accept_volume_loading(right, result_r)


@pytest.mark.parametrize("mutation", ("unsealed", "ledger", "extra", "missing", "order", "temperature"))
def test_strict_archive_rejects_invalid_fields_or_work_identity(mutation):
    raw = json.loads(json.dumps(volume_mechanics_to_dict(source())))
    if mutation == "unsealed":
        raw["state"]["elastic_energy_j"] += 1
    elif mutation == "ledger":
        raw["state"]["mechanical_work_j"] += 1; reseal(raw)
    elif mutation == "extra":
        raw["state"]["invented_total_strain"] = 0; reseal(raw)
    elif mutation == "missing":
        del raw["state"]["cohorts"][0]["strain1"]; reseal(raw)
    elif mutation == "order":
        raw["voigt_order"][-2:] = raw["voigt_order"][-2:][::-1]; reseal(raw)
    else:
        raw["state"]["cohorts"][0]["temperature_k"] = -1; reseal(raw)
    with pytest.raises(ValueError):
        volume_mechanics_from_dict(raw)


def test_retry_is_pure_and_unknown_memory_is_refused():
    state = source()
    before = volume_mechanics_to_dict(state)
    loading = prepare(state)
    evaluate_volume_loading(loading, np.zeros((1, 3, 6)), np.zeros((1, 3, 6)))
    assert volume_mechanics_to_dict(state) == before
    with pytest.raises(ValueError, match="source import required"):
        prepare_volume_loading(None, after_ids=(), after_temperature_k=(),
            birth_temperature_k=1400., radial_moments=None, before_radial_moments=None,
            removed_radial_moments=None, before_damage=(), after_damage=(),
            parameters=parameters(), after_time_myr=2.)


def test_mass_and_profile_corruption_are_rejected():
    state = source()
    with pytest.raises(ValueError, match="conserve source mass"):
        prepare(state, radial=moments((1.,)), removed=moments((0.,), (0.,), (0.,)))
    # Constant producer memory cannot detect profile variance corruption;
    # a prospective radial gradient supplies the necessary independent check.
    loading = prepare(state)
    state = accept_volume_loading(loading, evaluate_volume_loading(loading,
        np.zeros((1, 3, 6)), np.full((1, 3, 6), .001)))
    with pytest.raises(ValueError, match="preserve inherited energy"):
        prepare(state, radial=moments(mean=(1.3,)))


def test_completely_remelted_state_keeps_actual_removed_energy():
    state = source()
    loading = prepare(state, after_ids=(), radial=moments((), (), ()),
        temperatures=np.empty(0), removed=moments())
    response = evaluate_volume_loading(loading, np.empty((0, 3, 6)), np.empty((0, 3, 6)))
    accepted = accept_volume_loading(loading, response)
    assert not accepted.cohorts
    assert accepted.elastic_energy_j == 0
    assert accepted.remelt_energy_removed_j == state.initial_energy_j
