"""Analytic friction, dissipation, frame-objectivity, and tangent checks."""
from dataclasses import replace

import numpy as np
import pytest

from tectonics.genesis import SECONDS_PER_MYR
from tectonics.genesis_fault_law import (
    WeakPlaneParameters, return_map, select_plane, weak_plane_parameters_from_config,
)
from tectonics.genesis_material import rotate_tensor


YOUNG = 6e10
POISSON = .25


def stiffness(damage, floor=.02):
    d = np.array([[1., POISSON, 0.], [POISSON, 1., 0.],
                  [0., 0., (1-POISSON)/2]])/(1-POISSON**2)
    return YOUNG*(floor+(1-floor)*(1-np.asarray(damage))**2)[..., None, None]*d


def evaluate(stress, *, normal=(1., 0.), damage=.7, water=0., b=.8,
             dt=.002, params=None, active=True, floor=.02):
    stress = np.atleast_2d(np.asarray(stress, dtype=float))
    count = len(stress)
    damage = np.broadcast_to(damage, (count,))
    strain = np.linalg.solve(stiffness(damage, floor), stress[..., None])[..., 0]
    arguments = dict(elastic_trial=strain, plane_normal=np.broadcast_to(normal, (count, 2)),
        active=np.broadcast_to(active, (count,)), damage=damage, water=np.broadcast_to(water, (count,)),
        effective_b=np.broadcast_to(b, (count,)), dt_myr=dt, young_pa=YOUNG,
        poisson_ratio=POISSON, params=params or WeakPlaneParameters(), residual_stiffness=floor)
    return return_map(**arguments), arguments


def test_analytic_simple_shear_viscous_return_and_work_partition():
    parameters = WeakPlaneParameters()
    result, args = evaluate([-40e6, -20e6, 40e6], params=parameters)
    strength = 4e6*(.1+.9*.3)+.6*40e6
    modulus = .8*stiffness(.7)[2, 2]
    viscosity = parameters.viscosity_pa_s/(.002*SECONDS_PER_MYR)
    expected_slip = (40e6-strength)/(modulus+viscosity)
    assert result["shear_increment"][0] == pytest.approx(expected_slip, rel=2e-15)
    assert result["shear_stress_pa"][0] == pytest.approx(strength+viscosity*expected_slip, rel=2e-15)
    np.testing.assert_allclose(result["stress_pa"][0, :2], [-40e6, -20e6], rtol=3e-16)
    assert result["yield_strength_pa"][0] == pytest.approx(strength, rel=2e-15)
    friction_work = result["friction_work_density_j_m3"][0]
    viscous_work = result["viscous_work_density_j_m3"][0]
    assert friction_work > 0 and viscous_work > 0
    assert friction_work+viscous_work == pytest.approx(result["shear_stress_pa"][0]*expected_slip, rel=3e-15)
    # Non-dilatant shear corrects engineering xy only for this plane.
    np.testing.assert_array_equal(result["elastic_strain"][:, :2], args["elastic_trial"][:, :2])


@pytest.mark.parametrize("sign", [-1., 1.])
def test_elastic_below_threshold_and_no_reverse_flow(sign):
    result, args = evaluate([-40e6, -20e6, sign*10e6])
    np.testing.assert_array_equal(result["elastic_strain"], args["elastic_trial"])
    np.testing.assert_array_equal(result["tangent_pa"], .8*stiffness(np.array([.7])))
    np.testing.assert_array_equal(result["shear_increment"], 0.)
    np.testing.assert_array_equal(result["friction_work_density_j_m3"], 0.)
    np.testing.assert_array_equal(result["viscous_work_density_j_m3"], 0.)


def test_unloading_keeps_irreversible_shear_and_reversal_changes_slip_sign():
    result, args = evaluate([0., 0., 30e6], damage=0., b=1.)
    accumulated = result["shear_increment"].copy()
    assert accumulated[0] > 0
    shear_modulus = stiffness(0.)[2, 2]
    unloading = np.zeros((1, 3))
    unloading[:, 2] = -result["shear_stress_pa"]/shear_modulus
    unloaded = return_map(**{**args, "elastic_trial": result["elastic_strain"]+unloading})
    np.testing.assert_allclose(unloaded["stress_pa"], 0., atol=2e-8)
    np.testing.assert_array_equal(unloaded["shear_increment"], 0.)
    # Total engineering strain minus elastic strain retains the previous slip.
    total_strain = args["elastic_trial"]+unloading
    assert (total_strain-unloaded["elastic_strain"])[0, 2] == pytest.approx(accumulated[0], rel=3e-15)
    reverse = np.array([[0., 0., -30e6/shear_modulus]])
    reversed_state = return_map(**{**args, "elastic_trial": unloaded["elastic_strain"]+reverse})
    assert reversed_state["shear_increment"][0] == pytest.approx(-accumulated[0], rel=3e-15)
    assert reversed_state["friction_work_density_j_m3"][0] > 0
    assert reversed_state["viscous_work_density_j_m3"][0] > 0


def test_compression_strengthens_while_tension_does_not_create_negative_friction():
    result, _ = evaluate([[-40e6, -20e6, 30e6], [0., -20e6, 30e6], [40e6, -20e6, 30e6]])
    assert result["yield_strength_pa"][0] > result["yield_strength_pa"][1]
    assert result["yield_strength_pa"][1] == result["yield_strength_pa"][2]
    assert 0 < result["shear_increment"][0] < result["shear_increment"][1]
    assert result["shear_increment"][1] == result["shear_increment"][2]


def test_water_weakens_cohesion_and_friction_without_pore_pressure():
    result, _ = evaluate(np.broadcast_to([-40e6, -20e6, 30e6], (3, 3)), water=[0., .5, 1.])
    assert np.all(np.diff(result["yield_strength_pa"]) < 0)
    assert np.all(np.diff(result["shear_increment"]) > 0)
    np.testing.assert_allclose(result["normal_stress_pa"], -40e6, rtol=3e-16)
    assert result["yield_strength_pa"][2] == pytest.approx(4e6*(.1+.9*.3)*.5+.2*40e6)


@pytest.mark.parametrize("enabled,active", [(False, True), (True, False)])
def test_disabled_and_inactive_are_exact_original_material(enabled, active):
    result, args = evaluate([-40e6, -20e6, 40e6], active=active,
                            params=replace(WeakPlaneParameters(), enabled=enabled), floor=.08)
    np.testing.assert_array_equal(result["elastic_strain"], args["elastic_trial"])
    np.testing.assert_array_equal(result["tangent_pa"], .8*stiffness(np.array([.7]), .08))
    np.testing.assert_array_equal(result["shear_increment"], 0.)


def test_zero_stress_with_zero_cohesion_never_activates_extra_slip():
    result, _ = evaluate([0., 0., 0.], params=replace(WeakPlaneParameters(), cohesion_pa=0.))
    for name in ("shear_increment", "stress_pa", "friction_work_density_j_m3", "viscous_work_density_j_m3"):
        np.testing.assert_array_equal(result[name], 0.)


def test_frame_objectivity_and_no_dilation_for_oblique_planes():
    stress = np.array([[-40e6, -80e6, 70e6]])
    normal = np.array([[.6, .8]])
    result, args = evaluate(stress, normal=normal)
    angle = .713
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated_args = {**args, "elastic_trial": rotate_tensor(args["elastic_trial"], rotation[None], engineering=True),
                    "plane_normal": normal@rotation.T}
    transformed = return_map(**rotated_args)
    np.testing.assert_allclose(transformed["stress_pa"], rotate_tensor(result["stress_pa"], rotation[None]), rtol=3e-15, atol=3e-8)
    np.testing.assert_allclose(transformed["elastic_strain"], rotate_tensor(result["elastic_strain"], rotation[None], engineering=True), rtol=4e-15, atol=3e-17)
    for name in ("shear_increment", "shear_stress_pa", "normal_stress_pa", "yield_strength_pa",
                 "friction_work_density_j_m3", "viscous_work_density_j_m3"):
        np.testing.assert_allclose(transformed[name], result[name], rtol=4e-15, atol=3e-8)
    correction = args["elastic_trial"]-result["elastic_strain"]
    np.testing.assert_allclose(correction[:, 0]+correction[:, 1], 0., atol=2e-18)


@pytest.mark.parametrize("stress", [[-40e6, -10e6, 80e6], [40e6, 10e6, -80e6], [-40e6, -10e6, 1e6]])
def test_consistent_tangent_matches_total_increment_finite_difference(stress):
    result, args = evaluate(stress, normal=[.8, .6], b=.43)
    epsilon = 1e-8
    for component in range(3):
        delta = np.zeros((1, 3))
        delta[:, component] = epsilon*args["effective_b"]
        plus = return_map(**{**args, "elastic_trial": args["elastic_trial"]+delta})["stress_pa"]
        minus = return_map(**{**args, "elastic_trial": args["elastic_trial"]-delta})["stress_pa"]
        numeric = (plus-minus)/(2*epsilon)
        np.testing.assert_allclose(result["tangent_pa"][:, :, component], numeric, rtol=4e-8, atol=3.)


def test_smaller_time_step_has_smaller_slip_and_correct_viscous_balance():
    coarse, _ = evaluate([0., 0., 30e6], dt=.002)
    fine, _ = evaluate([0., 0., 30e6], dt=.001)
    assert 0 < fine["shear_increment"][0] < coarse["shear_increment"][0]
    for dt, result in ((.002, coarse), (.001, fine)):
        resistance = result["yield_strength_pa"]+1e20/(dt*SECONDS_PER_MYR)*np.abs(result["shear_increment"])
        np.testing.assert_allclose(np.abs(result["shear_stress_pa"]), resistance, rtol=2e-15)


def test_plane_selection_maximizes_one_coulomb_conjugate_and_is_objective():
    stress = np.array([[-30e6, -90e6, 12e6]])
    friction = .6
    normal = select_plane(stress, friction)
    angles = np.linspace(0., np.pi, 100001)
    candidates = np.column_stack((np.cos(angles), np.sin(angles)))
    tensor = np.array([[-30e6, 12e6], [12e6, -90e6]])
    def excess(n):
        t = np.stack((-n[..., 1], n[..., 0]), axis=-1)
        normal_stress = np.einsum("ni,ij,nj->n", n, tensor, n)
        shear = np.einsum("ni,ij,nj->n", t, tensor, n)
        return np.abs(shear)+friction*normal_stress
    assert excess(normal)[0] >= np.max(excess(candidates))-1e-6
    angle = .9
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = select_plane(rotate_tensor(stress, rotation[None]), friction)
    expected = normal@rotation.T
    # A material plane is unoriented; eigenvector signs need not be identical.
    np.testing.assert_allclose(rotated[..., :, None]*rotated[..., None, :],
                               expected[..., :, None]*expected[..., None, :], atol=5e-16)


@pytest.mark.parametrize("principal,friction,angle", [
    ([30e6, 10e6, 0.], .6, np.pi/4),
    ([-10e6, -30e6, 0.], .6, np.pi/4-.5*np.arctan(.6)),
    ([10e6, -30e6, 0.], 1., np.pi/6),
    ([30e6, -10e6, 0.], 1., np.pi/4),
    ([10e6, -30e6, 0.], 0., np.pi/4),
])
def test_plane_selection_respects_clamped_pressure_in_tension_and_mixed_stress(principal, friction, angle):
    normal = select_plane(principal, friction)
    np.testing.assert_allclose(normal, [np.cos(angle), np.sin(angle)], atol=3e-16)
    stress = np.array([[principal[0], principal[2]], [principal[2], principal[1]]])
    candidates = np.stack((np.cos(np.linspace(0, np.pi, 100001)),
                           np.sin(np.linspace(0, np.pi, 100001))), axis=-1)
    def score(n):
        t = np.stack((-n[..., 1], n[..., 0]), axis=-1)
        normal_stress = np.einsum("...i,ij,...j->...", n, stress, n)
        shear = np.einsum("...i,ij,...j->...", t, stress, n)
        return np.abs(shear)-friction*np.maximum(-normal_stress, 0.)
    assert score(normal) >= float(np.max(score(candidates)))-1e-6


def test_input_arrays_are_not_mutated_and_unactivated_zero_normal_is_allowed():
    _, args = evaluate([-40e6, -20e6, 40e6])
    copies = {key: value.copy() for key, value in args.items() if isinstance(value, np.ndarray)}
    return_map(**args)
    for key, before in copies.items():
        np.testing.assert_array_equal(args[key], before)
    result = return_map(**{**args, "active": np.array([False]), "plane_normal": np.zeros((1, 2))})
    np.testing.assert_array_equal(result["shear_increment"], 0.)


@pytest.mark.parametrize("changes", [{"enabled": 1}, {"cohesion_pa": -1}, {"viscosity_pa_s": 0},
    {"band_width_km": 0}, {"activation_damage": 1.1}, {"activation_persistence_myr": -1},
    {"residual_cohesion_fraction": 1.1}, {"wet_cohesion_fraction": -.1},
    {"friction_wet": .7}, {"friction_dry": 3.}, {"max_shear_increment": 0},
    {"max_shear_increment": .1}, {"cohesion_pa": float("nan")}, {"cohesion_pa": "4e6"}])
def test_invalid_parameters_rejected(changes):
    with pytest.raises(ValueError):
        replace(WeakPlaneParameters(), **changes).validate()


def test_configuration_schema_and_unknown_keys():
    assert weak_plane_parameters_from_config({}) == WeakPlaneParameters()
    assert weak_plane_parameters_from_config({"genesis_faults": {"schema_version": 1, "enabled": False}}).enabled is False
    for section in ({"schema_version": 2}, {"schema_version": True}, {"unknown": 1}):
        with pytest.raises(ValueError):
            weak_plane_parameters_from_config({"genesis_faults": section})


@pytest.mark.parametrize("changes", [{"dt_myr": 0}, {"young_pa": -1}, {"poisson_ratio": .5},
    {"residual_stiffness": 0}, {"active": np.array([1])}, {"damage": np.array([1.1])},
    {"water": np.array([float("nan")])}, {"effective_b": np.array([0.])},
    {"plane_normal": np.array([[2., 0.]])}, {"elastic_trial": np.zeros((1, 2))}])
def test_invalid_local_state_rejected(changes):
    _, args = evaluate([0., 0., 30e6])
    with pytest.raises(ValueError):
        return_map(**{**args, **changes})
