"""Certified angular nullspace arithmetic without constraining plate motion."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tectonics.genesis_contact_law import ContactLawParameters
from tectonics.geometric_contact_dynamics import _equilibrate_contact_network
from tectonics.geometric_contact_mechanics import (
    ElasticColumnParameters, solve_compressed_contact,
)


def _problem(*, unloaded=False):
    columns = ElasticColumnParameters(1., 10., 12., 1., 14., 16.)
    law = ContactLawParameters(normal_stiffness_pa_m=4., tangential_stiffness_pa_m=6.,
        tensile_strength_pa=1., fracture_energy_j_m2=1., cohesion_pa=100.,
        friction_dry=0., friction_wet=0., viscosity_pa_s_m=1.)
    materials = tuple(SimpleNamespace(columns=columns, law=law, water=0.) for _ in range(2))
    operators = np.zeros((2, 2, 8))
    levers = np.asarray([[[.314159, .271828, .141421], [-.18233, .4234, .28782]],
                         [[-.3173, .27823, .4876], [.1357, .43912, -.28653]]])
    operators[:, :, :3], operators[:, :, 3:6] = -levers, levers
    operators[:, 0, 6:] = (-.8, .8)
    operators[:, 1, 6:] = (.6, -.6)
    areas = np.asarray((2., 1.))
    offsets = np.zeros((2, 2)) if unloaded else np.asarray(((-.125, .013), (-.2, -.005)))
    old_local = tuple(solve_compressed_contact(columns=columns, law=law,
        area_m2=float(area), gap_free_m=float(offset[0]), jump_free_m=float(offset[1]),
        dt_s=1., water=0.).state for area, offset in zip(areas, offsets))
    return dict(qold=np.zeros(8), drag=np.diag((7., 11., 13., 17., 19., 23., 0., 0.)),
        torque=np.asarray((.08, -.05, .07, -.02, .04, -.09, 0., 0.)),
        rows=np.zeros((2, 8)), areas=areas, frames=np.asarray(((.6, .8), (.6, .8))),
        materials=materials, offsets=offsets, old_local=old_local, dt=.25,
        max_rotation_radians=.25, anchor_operators=operators,
        elastic_stiffness=np.diag((0., 0., 0., 0., 0., 0., 29., 31.)),
        rotation_dof_count=6)


def _locked_oracle(problem):
    # Derive the energy Hessian directly from the two series spring compliances.
    columns, law = problem["materials"][0].columns, problem["materials"][0].law
    springs = np.asarray((1./(1./columns.normal_stiffness_pa_m+1./law.normal_stiffness_pa_m),
                          1./(1./columns.shear_stiffness_pa_m+1./law.tangential_stiffness_pa_m)))
    matrix = problem["drag"]/problem["dt"]+problem["elastic_stiffness"]
    rhs = problem["torque"]+problem["drag"]@problem["qold"]/problem["dt"]
    for area, operator, offset in zip(problem["areas"], problem["anchor_operators"], problem["offsets"]):
        matrix = matrix+area*operator.T@np.diag(springs)@operator
        rhs = rhs-area*operator.T@(springs*offset)
    return np.linalg.solve(matrix, rhs)


def test_common_rigid_initial_guess_has_no_artificial_opening_or_contact_work():
    problem = _problem(unloaded=True)
    common = np.asarray((.037, -.043, .029))
    guess = np.r_[common, common, 0., 0.]
    problem["torque"] = problem["drag"]@guess/problem["dt"]
    result = _equilibrate_contact_network(**problem, initial_guess=guess,
                                         common_rotation_gauge=True)
    np.testing.assert_array_equal(result.q, guess)
    assert result.diagnostics["iterations"] == 0
    for step in result.steps:
        assert step.state.gap_free_m == step.state.jump_free_m == 0.
        assert step.traction_pa == (0., 0.)
        assert step.external_work_j == step.friction_work_j == step.viscous_work_j == 0.
    np.testing.assert_array_equal(result.contact_force, np.zeros(8))
    assert result.diagnostics["source_work_j"] > 0.
    assert result.diagnostics["source_work_j"] == pytest.approx(
        guess@problem["drag"]@guess/problem["dt"], rel=2e-15)
    assert result.diagnostics["source_work_j"] == result.diagnostics["basal_dissipation_j"]


def test_differential_motion_still_matches_independent_locked_equilibrium_and_work():
    problem = _problem()
    expected = _locked_oracle(problem)
    result = _equilibrate_contact_network(**problem, common_rotation_gauge=True)
    np.testing.assert_allclose(result.q, expected, rtol=3e-12, atol=2e-15)
    assert np.linalg.norm(result.q[:3]-result.q[3:6]) > .01
    assert result.diagnostics["iterations"] > 0
    np.testing.assert_allclose(problem["drag"]@(result.q-problem["qold"])/problem["dt"]
        +problem["elastic_stiffness"]@result.q+result.contact_force,
        problem["torque"], rtol=1e-11, atol=3e-14)
    assert result.diagnostics["source_work_j"] == pytest.approx(
        problem["torque"]@(result.q-problem["qold"]), rel=2e-15)
    assert result.diagnostics["energy_balance_relative_error"] < 2e-13


def test_optional_defaults_preserve_original_result_and_receipt_bits():
    problem = _problem()
    before = _equilibrate_contact_network(**problem)
    after = _equilibrate_contact_network(**problem, initial_guess=None, common_rotation_gauge=False)
    np.testing.assert_array_equal(after.q, before.q)
    np.testing.assert_array_equal(after.omega, before.omega)
    assert after.steps == before.steps
    assert after.diagnostics == before.diagnostics


def test_gauge_and_full_source_work_are_covariant_under_owner_permutation():
    problem = _problem()
    original = _equilibrate_contact_network(**problem, common_rotation_gauge=True)
    permutation = np.asarray((3, 4, 5, 0, 1, 2, 6, 7))
    changed = dict(problem)
    for name in ("qold", "torque"):
        changed[name] = problem[name][permutation]
    for name in ("drag", "elastic_stiffness"):
        changed[name] = problem[name][np.ix_(permutation, permutation)]
    changed["anchor_operators"] = problem["anchor_operators"][:, :, permutation]
    result = _equilibrate_contact_network(**changed, common_rotation_gauge=True)
    np.testing.assert_allclose(result.q, original.q[permutation], rtol=3e-12, atol=2e-15)
    for name in ("source_work_j", "basal_dissipation_j", "contact_work_j"):
        assert result.diagnostics[name] == pytest.approx(original.diagnostics[name], rel=3e-12)


def test_gauge_is_covariant_under_passive_rotation_of_operators_and_source():
    problem = _problem()
    original = _equilibrate_contact_network(**problem, common_rotation_gauge=True)
    rotation = Rotation.from_rotvec((.31, -.27, .19)).as_matrix()
    transform = np.eye(8)
    transform[:3, :3], transform[3:6, 3:6] = rotation, rotation
    changed = dict(problem, qold=transform@problem["qold"], torque=transform@problem["torque"],
        drag=transform@problem["drag"]@transform.T,
        elastic_stiffness=transform@problem["elastic_stiffness"]@transform.T,
        anchor_operators=problem["anchor_operators"]@transform.T)
    result = _equilibrate_contact_network(**changed, common_rotation_gauge=True)
    np.testing.assert_allclose(result.q, transform@original.q, rtol=3e-12, atol=2e-15)
    assert result.diagnostics["source_work_j"] == pytest.approx(
        original.diagnostics["source_work_j"], rel=3e-12)


def test_bad_angular_nullspace_is_rejected_without_modifying_history():
    problem = _problem()
    problem["anchor_operators"][0, 0, 0] = np.nextafter(problem["anchor_operators"][0, 0, 0], np.inf)
    old = problem["old_local"]
    with pytest.raises(ValueError, match="annihilate"):
        _equilibrate_contact_network(**problem, common_rotation_gauge=True)
    assert problem["old_local"] == old


def test_nullspace_certificate_detects_a_small_coefficient_hidden_by_float_sum():
    problem = _problem()
    operators = np.zeros((2, 2, 11))
    operators[:, :, 9:] = problem["anchor_operators"][:, :, 6:]
    operators[0, 0, (0, 3, 6)] = (1e20, 1., -1e20)
    assert np.sum(operators[0, 0, :9].reshape((3, 3)), axis=0)[0] == 0.
    problem.update(qold=np.zeros(11), drag=np.eye(11), torque=np.zeros(11),
        rows=np.zeros((2, 11)), anchor_operators=operators,
        elastic_stiffness=np.zeros((11, 11)), rotation_dof_count=9)
    with pytest.raises(ValueError, match="annihilate"):
        _equilibrate_contact_network(**problem, common_rotation_gauge=True)


def test_changing_source_keeps_old_local_history_and_incremental_work():
    problem = _problem()
    first = _equilibrate_contact_network(**problem, common_rotation_gauge=True)
    changed = dict(problem, qold=first.q,
        old_local=tuple(step.state for step in first.steps),
        torque=np.asarray((-.04, .03, .02, .05, -.06, .07, 0., 0.)))
    second = _equilibrate_contact_network(**changed, common_rotation_gauge=True,
                                          initial_guess=first.q)
    np.testing.assert_allclose(second.q, _locked_oracle(changed), rtol=3e-12, atol=2e-15)
    assert all(step.state.step_count == 3 for step in second.steps)
    assert all(step.state.elapsed_time_s == 1.5 for step in second.steps)
    expected = changed["torque"]@(second.q-first.q)
    assert second.diagnostics["source_work_j"] == pytest.approx(expected, rel=2e-15)
    assert abs(expected-changed["torque"]@second.q) > 1e-4


@pytest.mark.parametrize("guess", [[0.]*7, np.full(8, np.nan), np.full(8, np.inf),
                                   [True]+[0.]*7, ["0"]*8, np.zeros((2, 4))])
def test_initial_guess_requires_a_finite_real_matching_coordinate_vector(guess):
    with pytest.raises(ValueError, match="Initial contact guess"):
        _equilibrate_contact_network(**_problem(), initial_guess=guess)


def test_initial_guess_is_subject_to_existing_domain_callback():
    seen = []

    def domain(coordinates):
        seen.append(coordinates.copy())
        if coordinates[0] > .01:
            raise ValueError("declared domain excludes this guess")

    guess = np.r_[.02, np.zeros(7)]
    with pytest.raises(ValueError, match="declared domain"):
        _equilibrate_contact_network(**_problem(), initial_guess=guess, domain_check=domain)
    np.testing.assert_array_equal(seen[0], guess)


@pytest.mark.parametrize("flag", [0, 1, None, "true", np.bool_(True)])
def test_gauge_flag_is_an_explicit_python_boolean(flag):
    with pytest.raises(ValueError, match="explicit boolean"):
        _equilibrate_contact_network(**_problem(), common_rotation_gauge=flag)


def test_gauge_requires_generalized_contact_operators():
    problem = _problem()
    problem.pop("anchor_operators")
    problem.pop("elastic_stiffness")
    problem.pop("rotation_dof_count")
    with pytest.raises(ValueError, match="generalized contact"):
        _equilibrate_contact_network(**problem, common_rotation_gauge=True)
