"""Independent mechanics oracles for parallel material-depth cohorts.

Expected stresses, energies and time-path integrals are derived here without
the producer's aggregation or its Maxwell work helper.
"""
from dataclasses import replace

import numpy as np
import pytest

from tectonics.genesis_fault_law import WeakPlaneParameters, return_map
from tectonics.genesis import SECONDS_PER_MYR
from tectonics.genesis_bulk_birth import initial_bulk_birth, advance_bulk_birth
from tectonics.genesis_cohort_mechanics import (
    accept_cohort_loading, cohort_temperature, evaluate_cohort_loading,
    initial_cohort_mechanics, prepare_cohort_loading,
)
from tectonics.genesis_shell import ShellParameters
from tectonics.mesh import build_icosphere


def elasticity(young, nu, damage=0., residual=.02):
    degradation = residual+(1-residual)*(1-damage)**2
    return young*degradation/(1-nu**2)*np.array(
        [[1., nu, 0.], [nu, 1., 0.], [0., 0., (1-nu)/2]])


def test_parallel_cohort_weak_plane_correction_and_tangent_match_volume_forces():
    """Distinct beta_i must survive common fault slip, even at equal C."""
    weight = np.array([.13, .27, .60])
    beta = np.array([.08, .42, .99])
    strains = np.array([[.0001, -.0007, .0018],
                        [-.0002, -.0004, .0041],
                        [.0003, -.0001, .0004]])
    normal = np.array([np.cos(.71), np.sin(.71)])
    plane = np.array([-normal[1], normal[0]])
    flow = np.array([plane[0]*normal[0], plane[1]*normal[1],
                     plane[0]*normal[1]+plane[1]*normal[0]])
    c = elasticity(7e10, .27, .23)
    parameters = replace(WeakPlaneParameters(), cohesion_pa=2e6,
                         viscosity_pa_s=1e17, friction_dry=.35, friction_wet=.1)

    def response(increment):
        cohort_trial = strains+beta[:, None]*increment
        result = return_map((weight@cohort_trial)[None], normal[None],
            np.array([True]), np.array([.23]), np.array([.2]),
            np.array([weight@beta]), .003, 7e10, .27, parameters)
        corrected = cohort_trial-beta[:, None]*result["shear_increment"][0]*flow
        independent_stress = sum(w*(c@e) for w, e in zip(weight, corrected))
        return result, independent_stress, corrected

    result, stress, corrected = response(np.array([.0002, .0001, -.0004]))
    assert abs(result["shear_increment"][0]) > 0
    np.testing.assert_allclose(stress, result["stress_pa"][0], rtol=3e-15, atol=1e-8)
    np.testing.assert_allclose(weight@corrected, result["elastic_strain"][0],
                               rtol=2e-15, atol=1e-18)
    eps = 1e-8
    derivative = np.column_stack([
        (response(np.array([.0002, .0001, -.0004])+eps*np.eye(3)[j])[1]
         -response(np.array([.0002, .0001, -.0004])-eps*np.eye(3)[j])[1])/(2*eps)
        for j in range(3)])
    np.testing.assert_allclose(derivative, result["tangent_pa"][0], rtol=3e-9, atol=3.)
    cohort_energy = sum(w*.5*e@c@e for w, e in zip(weight, corrected))
    average_energy = .5*(weight@corrected)@c@(weight@corrected)
    assert cohort_energy > average_energy


def independent_maxwell_path(e0, driving, duration_s, tau_s, c):
    """Gauss quadrature of the ODE, not algebra copied from the producer."""
    nodes, weights = np.polynomial.legendre.leggauss(80)
    time = .5*duration_s*(nodes+1)
    ratio = np.exp(-time[:, None]/np.asarray(tau_s)[None])
    strain = (e0[None]*ratio[..., None]
              +(driving*np.asarray(tau_s)[:, None]/duration_s)[None]
              *(-np.expm1(-time[:, None]/np.asarray(tau_s)[None]))[..., None])
    average = .5*np.einsum("t,tci->ci", weights, strain)
    stress = np.einsum("ij,tcj->tci", c, strain)
    dissipation = .5*duration_s*np.einsum(
        "t,tci,tci,c->c", weights, strain, stress, 1/np.asarray(tau_s))
    endpoint_ratio = np.exp(-duration_s/np.asarray(tau_s))
    endpoint = (e0*endpoint_ratio[:, None]
        +driving*(-np.expm1(-duration_s/np.asarray(tau_s)))[:, None]
        /(duration_s/np.asarray(tau_s))[:, None])
    work = np.einsum("ci,ij,cj->c", driving, c, average)
    energy_change = .5*(np.einsum("ci,ij,cj->c", endpoint, c, endpoint)
                        -np.einsum("ci,ij,cj->c", e0, c, e0))
    np.testing.assert_allclose(energy_change, work-dissipation, rtol=2e-13, atol=1e-10)
    return endpoint, average, work, dissipation


def origin():
    mesh = build_icosphere(0)
    mass = np.tile([1., 3., 6.], (mesh.cell_count, 1))
    history = initial_bulk_birth(mesh, mass, 10., {"purpose": "independent cohort oracle"}, 1400.)
    params = replace(ShellParameters(), density_kg_m3=2., column_layers=3,
        viscosity_reference_pa_s=3e18, viscosity_reference_temperature_k=1000.,
        activation_energy_j_mol=90000., viscosity_min_pa_s=1e15,
        viscosity_max_pa_s=1e25)
    return mesh, history, params


def grow(history, mesh, time, front):
    return advance_bulk_birth(history, before_front_fraction=history.front_fraction,
        after_front_fraction=np.full(mesh.cell_count, front),
        before_time_myr=history.time_myr, after_time_myr=time, mesh=mesh,
        radius_km=10., solidus_k=1400.)


def loading(before, state, after, mesh, params, old, new, *, established=True):
    n = mesh.cell_count
    return prepare_cohort_loading(state, before, after,
        before_column_temperature=np.tile(old, (n, 1)),
        after_column_temperature=np.tile(new, (n, 1)),
        before_surface_temperature_k=450., after_surface_temperature_k=450.,
        before_mantle_temperature_k=1430., after_mantle_temperature_k=1430.,
        before_established=state.established, after_established=established,
        parameters=params, before_damage=np.zeros(n), activation_mesh=mesh,
        activation_radius_km=10.)


def independent_profile_mean(mass, temperature, surface, mantle, lo, hi):
    """Integrate by explicit segment clipping rather than a primitive table."""
    bounds = np.concatenate(([0.], np.cumsum(mass)/sum(mass)))
    nodes = np.concatenate(([0.], .5*(bounds[:-1]+bounds[1:]), [1.]))
    values = np.concatenate(([surface], temperature, [mantle]))
    total = 0.
    for a, b, ta, tb in zip(nodes[:-1], nodes[1:], values[:-1], values[1:]):
        left, right = max(a, lo), min(b, hi)
        if right > left:
            slope = (tb-ta)/(b-a)
            total += (right-left)*(ta+slope*((left+right)/2-a))
    return total/(hi-lo)


def test_subcell_thermal_means_follow_live_mass_overlap_after_remelting():
    mesh, history, _ = origin()
    frozen = grow(history, mesh, 1., .64)
    melted = grow(frozen, mesh, 2., .27)
    temperatures = np.tile([920., 1210., 1370.], (mesh.cell_count, 1))
    mean = cohort_temperature(melted, temperatures, 440., 1480.)
    live = [c for c in melted.cohorts if c.live_q_hi > c.q_lo]
    expected = [independent_profile_mean([1., 3., 6.], temperatures[c.face_index],
        440., 1480., c.q_lo, c.live_q_hi) for c in live]
    np.testing.assert_allclose(mean, expected, rtol=4e-15, atol=2e-12)
    assert mean[1] != pytest.approx(independent_profile_mean(
        [1., 3., 6.], temperatures[0], 440., 1480., .1, .4))


def test_first_membrane_activation_preserves_unknown_raft_history_then_new_birth_cools():
    mesh, history, params = origin()
    precursor = grow(history, mesh, .1, .05)
    pending = loading(history, initial_cohort_mechanics(history), precursor, mesh,
                      params, [1500., 1500., 1500.], [900., 1350., 1460.], established=False)
    response = evaluate_cohort_loading(pending, np.zeros((mesh.cell_count, 3)), None,
                                      np.zeros(mesh.cell_count))
    state = accept_cohort_loading(pending, response)
    assert all(c.elastic_strain is None and c.activation_time_myr is None for c in state.cohorts)
    gated = grow(precursor, mesh, .2, .45)
    pending = loading(precursor, state, gated, mesh, params,
                      [900., 1350., 1460.], [880., 1310., 1430.])
    np.testing.assert_array_equal(pending.memory, 0.)
    np.testing.assert_array_equal(pending.thermal_increment, 0.)
    np.testing.assert_array_equal(pending.beta, 1.)
    response = evaluate_cohort_loading(pending, np.zeros((mesh.cell_count, 3)), None,
                                      np.zeros(mesh.cell_count))
    state = accept_cohort_loading(pending, response)
    assert all(c.activation_temperature_k == c.temperature_k and c.activation_time_myr == .2
               for c in state.cohorts)
    later = grow(gated, mesh, .200001, .52)
    pending = loading(gated, state, later, mesh, params,
                      [880., 1310., 1430.], [870., 1300., 1420.])
    newborn = np.array([r.id > len(gated.cohorts) for r in pending.records])
    assert newborn.any()
    expected = params.linear_expansion_per_k*(pending.temperature_k[newborn]-1400.)
    np.testing.assert_allclose(pending.memory[newborn, :2], -expected[:, None]*np.ones((1, 2)))
    np.testing.assert_array_equal(pending.beta[newborn], 1.)
    assert all(r.activation_temperature_k == 1400. for r, flag in zip(pending.records, newborn) if flag)


@pytest.mark.parametrize("ratio", [5e-11, 1e-7, .002, .3, 2.7, 22.])
def test_maxwell_path_work_and_dissipation_match_independent_ode_quadrature(ratio):
    mesh, history, params = origin()
    frozen = grow(history, mesh, .2, .45)
    gate = loading(history, initial_cohort_mechanics(history), frozen, mesh, params,
                   [1500., 1500., 1500.], [880., 1310., 1430.])
    initial_increment = np.tile([.0007, -.0003, .0004], (mesh.cell_count, 1))
    initial_response = evaluate_cohort_loading(gate, initial_increment, None, np.zeros(mesh.cell_count))
    state = accept_cohort_loading(gate, initial_response)
    endpoint = grow(frozen, mesh, .200001, .45)
    pending = loading(frozen, state, endpoint, mesh, params,
                      [880., 1310., 1430.], [877., 1304., 1422.])
    # Set the ODE parameter explicitly so quadrature covers tiny and stiff
    # limits independently of the producer's Arrhenius input calculation.
    beta = -np.expm1(-ratio)/ratio
    pending = replace(pending, ratio=np.full(len(pending.records), ratio),
        beta=np.full(len(pending.records), beta),
        memory=np.exp(-ratio)*pending.old_elastic-beta*pending.thermal_increment,
        effective_b=np.full(mesh.cell_count, beta))
    increment = np.tile([-.0002, .0004, .0001], (mesh.cell_count, 1))
    plastic = np.tile([.00003, -.00001, .00002], (mesh.cell_count, 1))
    response = evaluate_cohort_loading(pending, increment, None, np.zeros(mesh.cell_count),
                                      plastic_increment=plastic)
    duration = (endpoint.time_myr-frozen.time_myr)*SECONDS_PER_MYR
    driver = increment[pending.face_index]-plastic[pending.face_index]-pending.thermal_increment
    c = elasticity(params.young_modulus_pa, params.poisson_ratio)
    e, average, work, dissipation = independent_maxwell_path(pending.old_elastic,
        driver, duration, np.full(len(driver), duration/ratio), c)
    np.testing.assert_allclose(response.elastic, e, rtol=2e-14, atol=2e-18)
    volume = pending.volume_m3
    expected_work = np.sum(volume*np.einsum("ci,ij,cj->c", increment[pending.face_index], c, average))
    expected_thermal = -np.sum(volume*np.einsum("ci,ij,cj->c", pending.thermal_increment, c, average))
    expected_plastic = np.sum(volume*np.einsum("ci,ij,cj->c", plastic[pending.face_index], c, average))
    assert response.mechanical_work_j == pytest.approx(expected_work, rel=8e-13, abs=1e-8)
    assert response.thermal_work_j == pytest.approx(expected_thermal, rel=8e-13, abs=1e-8)
    assert response.plastic_work_j == pytest.approx(expected_plastic, rel=8e-13, abs=1e-8)
    assert response.viscous_dissipation_j == pytest.approx(volume@dissipation, rel=2e-9, abs=1e-8)
    assert abs(response.balance_residual_j) < max(response.energy_j, state.elastic_energy_j, 1.)*3e-12


def rotate_engineering(value, rotation):
    result = []
    for e in np.asarray(value):
        tensor = np.array([[e[0], e[2]/2], [e[2]/2, e[1]]])
        current = rotation@tensor@rotation.T
        result.append([current[0, 0], current[1, 1], 2*current[0, 1]])
    return np.asarray(result)


def test_local_temperature_viscosity_weighted_tangent_and_rotated_plastic_flow():
    mesh, history, params = origin()
    frozen = grow(history, mesh, .2, .45)
    gate = loading(history, initial_cohort_mechanics(history), frozen, mesh, params,
                   [1500., 1500., 1500.], [880., 1310., 1430.])
    state = accept_cohort_loading(gate, evaluate_cohort_loading(gate,
        np.tile([.0004, -.0008, .003], (mesh.cell_count, 1)), None, np.zeros(mesh.cell_count)))
    endpoint = grow(frozen, mesh, .200001, .45)
    pending = loading(frozen, state, endpoint, mesh, params,
                      [880., 1310., 1430.], [870., 1300., 1420.])
    temperature = np.array([independent_profile_mean([1., 3., 6.],
        [870., 1300., 1420.], 450., 1430., c.q_lo, c.live_q_hi) for c in pending.records])
    eta = np.clip(params.viscosity_reference_pa_s*np.exp(np.clip(
        params.activation_energy_j_mol/8.314462618*(1/temperature
            -1/params.viscosity_reference_temperature_k), -60., 60.)),
        params.viscosity_min_pa_s, params.viscosity_max_pa_s)
    ratio = (endpoint.time_myr-frozen.time_myr)*SECONDS_PER_MYR*params.young_modulus_pa/eta
    beta = -np.expm1(-ratio)/ratio
    np.testing.assert_allclose(pending.ratio, ratio, rtol=3e-14)
    np.testing.assert_allclose(pending.beta, beta, rtol=3e-14)
    assert np.ptp(beta[:3]) > .01
    angle = .83
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotations = np.broadcast_to(rotation, (mesh.cell_count, 2, 2))
    increment = np.tile([.0002, .0001, -.0002], (mesh.cell_count, 1))
    trial = evaluate_cohort_loading(pending, increment, rotations, np.zeros(mesh.cell_count))
    # The plane normal and return-map result are in the CURRENT chart, but
    # increment and thermal memory are in the OLD chart. This transformation
    # checks the constitutive contract without using rotate_tensor.
    normal = np.array([np.cos(.42), np.sin(.42)])
    tangent_direction = np.array([-normal[1], normal[0]])
    current_flow = np.array([tangent_direction[0]*normal[0], tangent_direction[1]*normal[1],
        tangent_direction[0]*normal[1]+tangent_direction[1]*normal[0]])
    fault_parameters = replace(WeakPlaneParameters(), cohesion_pa=1e6, viscosity_pa_s=1e15)
    fault = return_map(trial.mean_elastic, np.broadcast_to(normal, (mesh.cell_count, 2)),
        np.ones(mesh.cell_count, dtype=bool), np.zeros(mesh.cell_count), np.zeros(mesh.cell_count),
        pending.effective_b, .000001, params.young_modulus_pa, params.poisson_ratio, fault_parameters)
    assert np.any(fault["shear_increment"])
    current_plastic = fault["shear_increment"][:, None]*current_flow
    old_plastic = rotate_engineering(current_plastic, rotation.T)
    corrected = evaluate_cohort_loading(pending, increment, rotations, np.zeros(mesh.cell_count),
                                       plastic_increment=old_plastic)
    np.testing.assert_allclose(corrected.mean_elastic, fault["elastic_strain"], rtol=4e-14, atol=1e-18)
    np.testing.assert_allclose(corrected.stress, fault["stress_pa"], rtol=4e-14, atol=1e-7)
    wrongly_framed = evaluate_cohort_loading(pending, increment, rotations, np.zeros(mesh.cell_count),
                                           plastic_increment=current_plastic)
    assert np.max(np.abs(wrongly_framed.mean_elastic-fault["elastic_strain"])) > 1e-5
    # Elastic tangent is with respect to a current-frame common engineering
    # increment; pull that probe into the old chart before FD evaluation.
    eps = 1e-8
    derivative = []
    for j in range(3):
        probe = rotate_engineering(np.eye(3)[[j]], rotation.T)[0]
        plus = evaluate_cohort_loading(pending, increment+eps*probe, rotations, np.zeros(mesh.cell_count))
        minus = evaluate_cohort_loading(pending, increment-eps*probe, rotations, np.zeros(mesh.cell_count))
        derivative.append((plus.stress-minus.stress)/(2*eps))
    np.testing.assert_allclose(np.stack(derivative, axis=2), trial.tangent, rtol=3e-9, atol=4.)


def test_remelt_energy_and_damage_change_preserve_independent_cohort_balance():
    mesh, history, params = origin()
    frozen = grow(history, mesh, .2, .64)
    gate = loading(history, initial_cohort_mechanics(history), frozen, mesh, params,
                   [1500., 1500., 1500.], [880., 1310., 1430.])
    initial = np.tile([.0004, -.0008, .003], (mesh.cell_count, 1))
    state = accept_cohort_loading(gate, evaluate_cohort_loading(gate, initial, None, np.zeros(mesh.cell_count)))
    melted = grow(frozen, mesh, .200001, .27)
    pending = loading(frozen, state, melted, mesh, params,
                      [880., 1310., 1430.], [875., 1302., 1420.])
    c_before = elasticity(params.young_modulus_pa, params.poisson_ratio)
    density_energy = .5*initial[0]@c_before@initial[0]
    independent_removed = mesh.cell_count*10.*(.64-.27)/params.density_kg_m3*density_energy
    assert pending.remelt_energy_removed_j == pytest.approx(independent_removed, rel=2e-15)
    damage = np.full(mesh.cell_count, .35)
    response = evaluate_cohort_loading(pending, np.zeros_like(initial), None, damage)
    c_after = elasticity(params.young_modulus_pa, params.poisson_ratio, .35)
    damage_change = sum(v*.5*e@(c_after-c_before)@e for v, e in zip(pending.volume_m3, pending.old_elastic))
    assert response.damage_energy_change_j == pytest.approx(damage_change, rel=3e-15)
    assert abs(response.balance_residual_j) < state.elastic_energy_j*3e-12
