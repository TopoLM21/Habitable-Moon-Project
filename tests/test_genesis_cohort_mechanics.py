"""Analytic material tests; these do not substitute for formation controls."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import hashlib
import json

import numpy as np
import pytest

from tectonics.genesis import SECONDS_PER_MYR
from tectonics.genesis_bulk_birth import initial_bulk_birth, advance_bulk_birth
from tectonics.genesis_cohort_mechanics import (
    BIRTH_ACTIVATION, GATE_ACTIVATION, accept_cohort_loading,
    cohort_mechanics_energy, cohort_mechanics_from_dict, cohort_mechanics_max_strain,
    cohort_mechanics_mean, cohort_mechanics_to_dict, cohort_temperature,
    evaluate_cohort_loading, initial_cohort_mechanics, prepare_cohort_loading,
    validate_cohort_mechanics,
)
from tectonics.genesis_shell import ShellParameters
from tectonics.mesh import build_icosphere


def setup(*, mass=None, parameters=None):
    mesh = build_icosphere(0)
    mass = np.tile([3000., 3000.], (mesh.cell_count, 1)) if mass is None else mass
    history = initial_bulk_birth(mesh, mass, 10., {"purpose": "analytic cohort law"}, 1400.)
    p = parameters or replace(ShellParameters(), activation_energy_j_mol=0.,
        viscosity_reference_pa_s=6e17, viscosity_min_pa_s=6e17, viscosity_max_pa_s=6e17)
    return mesh, history, initial_cohort_mechanics(history), p


def advance(history, mesh, front, dt=1e-6):
    return advance_bulk_birth(history, before_front_fraction=history.front_fraction,
        after_front_fraction=np.full(mesh.cell_count, front), before_time_myr=history.time_myr,
        after_time_myr=history.time_myr+dt, mesh=mesh, radius_km=10., solidus_k=1400.)


def loading(state, before, after, mesh, p, *, old=1000., new=1000., gate=True,
            before_damage=None, old_surface=None, new_surface=None):
    n = mesh.cell_count
    old = np.broadcast_to(np.asarray(old), (n, 2)).copy()
    new = np.broadcast_to(np.asarray(new), (n, 2)).copy()
    return prepare_cohort_loading(state, before, after,
        before_column_temperature=old, after_column_temperature=new,
        before_surface_temperature_k=old[0, 0] if old_surface is None else old_surface,
        after_surface_temperature_k=new[0, 0] if new_surface is None else new_surface,
        before_mantle_temperature_k=old[0, -1], after_mantle_temperature_k=new[0, -1],
        before_established=state.established, after_established=gate, parameters=p,
        before_damage=np.zeros(n) if before_damage is None else before_damage,
        activation_mesh=mesh, activation_radius_km=10.)


def evaluate(law, inc=(0., 0., 0.), *, damage=None, rotation=None, plastic=None):
    n = len(law.history.front_fraction)
    return evaluate_cohort_loading(law, np.broadcast_to(inc, (n, 3)).copy(), rotation,
        np.zeros(n) if damage is None else damage, plastic_increment=plastic)


def activated(*, front=1., inc=(.001, -.0003, .0004), p=None, profile=1000.):
    mesh, before, state, p = setup(parameters=p)
    after = advance(before, mesh, front)
    law = loading(state, before, after, mesh, p, old=profile, new=profile)
    response = evaluate(law, inc)
    return mesh, after, accept_cohort_loading(law, response), p


def test_origin_is_molten_and_missing_history_is_not_reconstructed():
    mesh, history, state, p = setup()
    assert state.cohorts == () and state.elastic_energy_j == 0
    assert deepcopy(state) is state
    with pytest.raises(FrozenInstanceError):
        state.time_myr = 1.
    later = advance(history, mesh, .1)
    with pytest.raises(ValueError, match="replay"):
        initial_cohort_mechanics(later)
    with pytest.raises(ValueError, match="replay"):
        validate_cohort_mechanics(None, later)


def test_precursor_records_temperature_but_elastic_and_raft_geometry_are_unknown():
    mesh, before, state, p = setup()
    after = advance(before, mesh, .1)
    law = loading(state, before, after, mesh, p, old=2300., new=1300., gate=False)
    response = evaluate(law)
    state = accept_cohort_loading(law, response)
    assert all(c.elastic_strain is None and c.activation_frame_xyz is None for c in state.cohorts)
    assert all(c.temperature_k == pytest.approx(1300.) for c in state.cohorts)
    assert state.elastic_energy_j == state.mechanical_work_j == 0
    with pytest.raises(ValueError, match="Unactivated"):
        evaluate(law, (.01, 0., 0.))


def test_gate_activation_uses_endpoint_temperature_without_inventing_old_cooling():
    mesh, before, state, p = setup()
    precursor = advance(before, mesh, .1)
    law = loading(state, before, precursor, mesh, p, old=2300., new=1300., gate=False)
    state = accept_cohort_loading(law, evaluate(law))
    after = advance(precursor, mesh, .6)
    gate = loading(state, precursor, after, mesh, p, old=1300., new=900.)
    assert np.array_equal(gate.memory, np.zeros_like(gate.memory))
    assert np.array_equal(gate.beta, np.ones_like(gate.beta))
    state = accept_cohort_loading(gate, evaluate(gate))
    assert all(c.activation_closure == GATE_ACTIVATION and c.activation_temperature_k == pytest.approx(900.) for c in state.cohorts)
    assert state.elastic_energy_j == state.thermal_work_j == 0


def test_existing_cohort_relaxation_and_viscous_energy_are_exact():
    mesh, before, state, p = activated()
    after = advance(before, mesh, 1.)
    law = loading(state, before, after, mesh, p)
    response = evaluate(law)
    x = 1e-6*SECONDS_PER_MYR*p.young_modulus_pa/p.viscosity_reference_pa_s
    old = np.array([c.elastic_strain for c in state.cohorts])
    assert response.elastic == pytest.approx(old*np.exp(-x))
    assert response.energy_j == pytest.approx(state.elastic_energy_j*np.exp(-2*x))
    assert response.viscous_dissipation_j == pytest.approx(state.elastic_energy_j-response.energy_j)
    assert response.remelt_energy_removed_j == 0.  # No retired q interval.
    assert abs(response.balance_residual_j) < state.elastic_energy_j*1e-13
    assert np.array_equal(law.relaxed_elastic, law.memory)


def test_cooling_at_fixed_geometry_and_constant_mechanical_driver_close_work():
    mesh, before, state, p = activated()
    after = advance(before, mesh, 1.)
    law = loading(state, before, after, mesh, p, old=1000., new=900.)
    response = evaluate(law, (.0002, -.0001, .0007))
    expected = law.old_elastic*np.exp(-law.ratio)[:, None]
    expected += law.beta[:, None]*(np.array([.0002, -.0001, .0007])-np.array([-.001, -.001, 0.]))
    assert response.elastic == pytest.approx(expected)
    assert response.thermal_work_j > 0
    assert response.remelt_energy_removed_j == 0.  # Cooling/growth do not retire mass.
    assert response.viscous_dissipation_j > 0
    assert abs(response.balance_residual_j) < max(response.energy_j, state.elastic_energy_j)*1e-13
    accepted = accept_cohort_loading(law, response)
    validate_cohort_mechanics(accepted, after, True, parameters=p, damage=np.zeros(mesh.cell_count))
    assert cohort_mechanics_energy(accepted, after, np.zeros(mesh.cell_count), p) == accepted.elastic_energy_j
    assert cohort_mechanics_mean(accepted, after) == pytest.approx(response.mean_elastic)


def test_later_newborn_solidus_cooling_contributes_stress_work_and_no_fake_age():
    mesh, before, state, p = activated(front=.2, inc=(0., 0., 0.))
    after = advance(before, mesh, .4)
    law = loading(state, before, after, mesh, p, old=1000., new=900.)
    response = evaluate(law)
    newborn = np.array([c.activation_closure == BIRTH_ACTIVATION for c in law.records])
    assert np.count_nonzero(newborn) == mesh.cell_count
    assert response.elastic[newborn, :2] == pytest.approx(np.full((mesh.cell_count, 2), .005))
    assert np.array_equal(law.ratio[newborn], np.zeros(mesh.cell_count))
    accepted = accept_cohort_loading(law, response)
    assert all(c.activation_time_myr == after.time_myr for c in accepted.cohorts if c.activation_closure == BIRTH_ACTIVATION)
    assert response.stress[:, 0].min() > 0
    assert response.thermal_work_j > 0
    assert abs(response.balance_residual_j) < response.energy_j*1e-13


def test_local_viscosities_produce_depth_contrast_and_parallel_tangent():
    p = replace(ShellParameters(), viscosity_reference_pa_s=6e17,
                viscosity_min_pa_s=1e15, viscosity_max_pa_s=1e22)
    mesh, before, state, p = activated(p=p, profile=(900., 1200.))
    after = advance(before, mesh, 1.)
    law = loading(state, before, after, mesh, p, old=(900., 1200.), new=(900., 1200.))
    response = evaluate(law, (.0002, -.0001, .0004))
    first_face = response.elastic[law.face_index == 0]
    assert not np.allclose(first_face[0], first_face[1], rtol=1e-5, atol=1e-10)
    assert law.beta[law.face_index == 0][0] > law.beta[law.face_index == 0][1]
    for component in range(3):
        inc = np.array([.0002, -.0001, .0004])
        inc[component] += 1e-8
        perturbed = evaluate(law, inc)
        numerical = (perturbed.stress-response.stress)/1e-8
        assert numerical == pytest.approx(response.tangent[:, :, component], rel=1e-9, abs=1.)
    assert abs(response.balance_residual_j) < response.energy_j*1e-12


def test_mean_strain_energy_does_not_replace_depth_energy():
    p = replace(ShellParameters(), viscosity_reference_pa_s=6e17,
                viscosity_min_pa_s=1e15, viscosity_max_pa_s=1e22)
    mesh, before, state, p = activated(p=p, profile=(900., 1200.))
    after = advance(before, mesh, 1.)
    law = loading(state, before, after, mesh, p, old=(900., 1200.), new=(900., 1200.))
    response = evaluate(law)
    volume = np.bincount(law.face_index, weights=law.volume_m3, minlength=mesh.cell_count)
    nu = p.poisson_ratio
    d = np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])/(1-nu**2)
    mean_energy = .5*p.young_modulus_pa*np.sum(volume*np.einsum("fi,ij,fj->f", response.mean_elastic, d, response.mean_elastic))
    assert response.energy_j > mean_energy
    accepted = accept_cohort_loading(law, response)
    assert cohort_mechanics_max_strain(accepted) == response.max_elastic_strain


def test_remelting_removes_actual_old_energy_without_reactivating_ids():
    mesh, before, state, p = activated()
    after = advance(before, mesh, .25)
    law = loading(state, before, after, mesh, p)
    response = evaluate(law)
    assert response.remelt_energy_removed_j == pytest.approx(.75*state.elastic_energy_j)
    assert response.energy_j == pytest.approx(.25*state.elastic_energy_j*np.exp(-2*law.ratio[0]))
    state2 = accept_cohort_loading(law, response)
    assert len(state2.cohorts) == mesh.cell_count
    refrozen = advance(after, mesh, .8)
    born = loading(state2, after, refrozen, mesh, p)
    assert all(c.id >= after.next_id for c in born.records if c.activation_closure == BIRTH_ACTIVATION)


def test_damage_change_is_signed_and_common_plastic_work_is_distinct():
    mesh, before, state, p = activated()
    after = advance(before, mesh, 1.)
    law = loading(state, before, after, mesh, p)
    plastic = np.tile([.0001, -.00005, .0002], (mesh.cell_count, 1))
    response = evaluate(law, (.0003, -.0001, .0005), damage=np.full(mesh.cell_count, .3), plastic=plastic)
    assert response.damage_energy_change_j < 0
    assert response.plastic_work_j > 0
    assert abs(response.balance_residual_j) < max(response.energy_j, state.elastic_energy_j)*1e-13
    accept_cohort_loading(law, response)


def test_objective_rotation_preserves_energy_and_work():
    mesh, before, state, p = activated()
    after = advance(before, mesh, 1.)
    law = loading(state, before, after, mesh, p, old=1000., new=950.)
    a = .37
    r = np.tile([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], (mesh.cell_count, 1, 1))
    original = evaluate(law, (.0001, -.0002, .0003))
    turned = evaluate(law, (.0001, -.0002, .0003), rotation=r)
    assert turned.energy_j == pytest.approx(original.energy_j, rel=2e-15)
    assert turned.mechanical_work_j == original.mechanical_work_j
    assert turned.thermal_work_j == original.thermal_work_j
    assert turned.viscous_dissipation_j == original.viscous_dissipation_j
    assert not np.allclose(turned.elastic, original.elastic)


def test_piecewise_linear_temperature_integrates_subcell_and_nonuniform_mass():
    mesh = build_icosphere(0)
    mesh, before, state, p = setup(mass=np.tile([3000., 9000.], (mesh.cell_count, 1)))
    after = advance(before, mesh, .2)
    temperature = cohort_temperature(after, np.tile([800., 1200.], (mesh.cell_count, 1)), 600., 1400.)
    # First center at q=.125: profile600→800, then slope800K/q to q=.625.
    integral = .125*700+.075*(800+860)/2
    assert temperature == pytest.approx(np.full(mesh.cell_count, integral/.2))


def test_checkpoint_roundtrip_seal_id_clock_and_missing_memory_rejection():
    mesh, history, state, p = activated()
    raw = cohort_mechanics_to_dict(state)
    assert cohort_mechanics_from_dict(json.loads(json.dumps(raw))) == state
    raw["state"]["cohorts"][0]["elastic_strain"] = [1., 0., 0.]
    with pytest.raises(ValueError, match="seal"):
        cohort_mechanics_from_dict(raw)
    with pytest.raises(ValueError, match="coverage"):
        validate_cohort_mechanics(replace(state, cohorts=state.cohorts[:-1]), history)
    with pytest.raises(ValueError, match="clock"):
        validate_cohort_mechanics(replace(state, time_myr=state.time_myr+1.), history)
    with pytest.raises(ValueError, match="energy"):
        validate_cohort_mechanics(replace(state, elastic_energy_j=state.elastic_energy_j*2), history,
                                  parameters=p, damage=np.zeros(mesh.cell_count))
    forged = replace(state, elastic_energy_j=state.elastic_energy_j*2, balance_residual_j=state.elastic_energy_j)
    with pytest.raises(ValueError, match="work identity"):
        validate_cohort_mechanics(forged, history)


@pytest.mark.parametrize("ratio", [1e-12, 5e-11, 1e-8, 1e-4, .1, 10., 1e5])
def test_work_identity_across_relaxation_scales(ratio):
    dt = 1e-6
    eta = dt*SECONDS_PER_MYR*6e10/ratio
    p = replace(ShellParameters(), activation_energy_j_mol=0., viscosity_reference_pa_s=eta,
                viscosity_min_pa_s=eta, viscosity_max_pa_s=eta)
    mesh, before, state, p = activated(p=p)
    after = advance(before, mesh, 1., dt)
    law = loading(state, before, after, mesh, p, old=1000., new=995.)
    response = evaluate(law, (.0002, .0003, -.0007))
    assert response.viscous_dissipation_j >= 0
    assert abs(response.balance_residual_j) < max(response.energy_j, state.elastic_energy_j)*2e-12
    accept_cohort_loading(law, response)
