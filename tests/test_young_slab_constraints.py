"""Independent convex/KKT checks for no-eduction constraint reactions."""
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import LinearConstraint, minimize

from tectonics.young_slab_constraints import _polish_reaction_support, solve_no_eduction


def scalar_case():
    drag = np.eye(6)
    drag[:2, :2] = [[2., 1.], [1., 3.]]
    torque = np.array([-2., 1., 0., 0., 0., 0.])
    feed = np.array([[1., 0., 0., 0., 0., 0.]])
    return drag, torque, feed


def brute_force_active_sets(drag, torque, feed):
    """Enumerate all constraints, independent of the production dual solve."""
    best = None
    for count in range(len(feed)+1):
        for ids in combinations(range(len(feed)), count):
            rows = feed[list(ids)]
            matrix = np.block([[drag, -rows.T], [rows, np.zeros((count, count))]])
            answer = np.linalg.solve(matrix, np.r_[torque, np.zeros(count)])
            omega, force = answer[:len(torque)], answer[len(torque):]
            if np.min(feed@omega, initial=0.) < -1e-9 or np.min(force, initial=0.) < -1e-9:
                continue
            energy = .5*omega@drag@omega-torque@omega
            if best is None or energy < best[0]:
                best = energy, omega
    assert best is not None
    return best[1]


def test_analytic_active_constraint_has_correct_force_sign_and_zero_work():
    drag, torque, feed = scalar_case()
    answer = solve_no_eduction(drag, torque, feed)
    np.testing.assert_allclose(answer.omega_rad_s.ravel(), [0., 1./3., 0., 0., 0., 0.], atol=1e-15)
    np.testing.assert_allclose(answer.normal_forces_n, [7./3.], rtol=1e-15)
    assert answer.unconstrained_feed_m_s[0] < 0.
    assert answer.active.tolist() == [True]
    assert abs(answer.reaction_work_w) < 2e-15
    assert answer.relative_residual < 1e-15
    assert answer.complementarity_relative_error < 1e-15
    assert answer.primal_relative_violation < 1e-15


@pytest.mark.parametrize("seed", [23, 127, 541, 8147])
def test_random_problem_matches_independent_enumerated_active_set_oracle(seed):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(6, 6))
    drag = matrix.T@matrix+np.eye(6)
    torque = rng.normal(size=6)
    feed = rng.normal(size=(4, 6))
    expected = brute_force_active_sets(drag, torque, feed)
    answer = solve_no_eduction(drag*1e41, torque*1e25, feed*5e6)
    np.testing.assert_allclose(answer.omega_rad_s.ravel(), expected*1e-16, rtol=1e-10, atol=1e-25)
    omega = answer.omega_rad_s.ravel()
    assert omega@(drag*1e41)@omega == pytest.approx(omega@(torque*1e25), rel=5e-12)


def test_feasible_unconstrained_motion_keeps_exact_free_solution():
    drag, torque, feed = scalar_case()
    torque[0] = 2.
    expected = np.linalg.solve(drag, torque)
    answer = solve_no_eduction(drag, torque, feed)
    np.testing.assert_allclose(answer.omega_rad_s.ravel(), expected, rtol=1e-15, atol=2e-16)
    assert np.count_nonzero(answer.reaction_torque_nm) == 0
    assert np.count_nonzero(answer.active) == 0


def test_duplicate_sections_share_tension_by_cross_section_and_preserve_split_threshold():
    drag, torque, feed = scalar_case()
    whole = solve_no_eduction(drag, torque, feed, reaction_weights=np.array([4.]))
    split = solve_no_eduction(drag, torque, np.repeat(feed, 2, axis=0),
                              reaction_weights=np.array([1., 3.]))
    np.testing.assert_allclose(split.omega_rad_s, whole.omega_rad_s, atol=2e-15)
    np.testing.assert_allclose(split.reaction_torque_nm, whole.reaction_torque_nm, rtol=2e-15)
    np.testing.assert_allclose(split.normal_forces_n, whole.normal_forces_n[0]*np.array([.25, .75]), rtol=3e-15)
    np.testing.assert_allclose(split.normal_forces_n/np.array([1., 3.]),
                               whole.normal_forces_n[0]/4., rtol=3e-15)


def test_many_dependent_constraints_retain_unique_primal_and_finite_nonnegative_forces():
    rng = np.random.default_rng(346)
    drag = np.eye(6)
    torque = rng.normal(size=6)
    rows = rng.normal(size=(14, 6))
    feed = np.r_[rows, -rows]
    answer = solve_no_eduction(drag*1e40, torque*1e24, feed*1e6,
                              reaction_weights=np.linspace(1., 3., len(feed)))
    np.testing.assert_allclose(answer.omega_rad_s, 0., atol=2e-28)
    assert np.all(answer.normal_forces_n >= 0.)
    assert answer.relative_residual < 1e-13
    assert answer.primal_relative_violation < 1e-11


@pytest.mark.parametrize("spurious_positive", [0., 1e-11, .1])
def test_support_polish_blocks_false_bounds_and_releases_missing_optimal_force(spurious_positive):
    # Exact optimum is (1/2, 0, 1/2). With zero perturbation its third
    # coordinate must be released; otherwise the second must hit its bound.
    equality = np.array([[1., -1., 1.]])/np.sqrt(3.)
    value = np.array([1./np.sqrt(3.)])
    initial = np.array([1., spurious_positive, spurious_positive])
    answer = _polish_reaction_support(equality, value, initial)
    np.testing.assert_allclose(answer, [.5, 0., .5], atol=3e-16)


def test_spatial_rotation_changes_only_coordinates():
    drag, torque, feed = scalar_case()
    rotation, _ = np.linalg.qr(np.random.default_rng(73).normal(size=(3, 3)))
    transform = np.kron(np.eye(2), rotation)
    original = solve_no_eduction(drag, torque, feed)
    rotated = solve_no_eduction(transform@drag@transform.T, transform@torque, feed@transform.T)
    np.testing.assert_allclose(rotated.omega_rad_s.ravel(), transform@original.omega_rad_s.ravel(), atol=2e-15)
    np.testing.assert_allclose(rotated.normal_forces_n, original.normal_forces_n, rtol=3e-15)


@pytest.mark.parametrize("force_scale", [1e-20, 1., 1e40])
def test_common_force_units_do_not_change_kinematics(force_scale):
    drag, torque, feed = scalar_case()
    answer = solve_no_eduction(drag*force_scale, torque*force_scale, feed)
    np.testing.assert_allclose(answer.omega_rad_s.ravel(), [0., 1/3., 0., 0., 0., 0.], atol=1e-15)
    np.testing.assert_allclose(answer.normal_forces_n/force_scale, [7./3.], rtol=2e-15)


def test_empty_and_zero_constraints_have_no_effect():
    drag, torque, _ = scalar_case()
    empty = solve_no_eduction(drag, torque, np.zeros((0, 6)))
    zero = solve_no_eduction(drag, torque, np.zeros((2, 6)))
    np.testing.assert_array_equal(empty.omega_rad_s, zero.omega_rad_s)
    assert empty.normal_forces_n.shape == (0,)
    assert np.count_nonzero(zero.normal_forces_n) == 0
    assert empty.primal_relative_violation == 0.


@pytest.mark.parametrize("kind", ["asymmetric", "indefinite", "nonfinite", "badweight"])
def test_invalid_constraint_problem_is_rejected(kind):
    drag, torque, feed = scalar_case()
    weights = None
    if kind == "asymmetric":
        drag[0, 1] = .5
    elif kind == "indefinite":
        drag[0, 1] = drag[1, 0] = 10.
    elif kind == "nonfinite":
        torque[0] = np.nan
    else:
        weights = [-1.]
    with pytest.raises(ValueError):
        solve_no_eduction(drag, torque, feed, reaction_weights=weights)


_ACTUAL_CASES = ["young_slab_constraint_degenerate.json", "young_slab_constraint_support_polish.json",
                 "young_slab_constraint_noisy_support.json"]


def actual_case(name):
    data = json.loads((Path(__file__).parent/"fixtures"/name).read_text(encoding="utf-8"))
    return tuple(np.asarray(data[key], dtype=float) for key in
        ("drag_nm_s", "driving_torque_nm", "feed_matrix_m", "reaction_weights"))


@pytest.mark.parametrize("name", _ACTUAL_CASES)
def test_actual_rank_deficient_trench_problem_matches_independent_primal_qp(name):
    drag, torque, feed, weights = actual_case(name)
    answer = solve_no_eduction(drag, torque, feed, reaction_weights=weights)
    # The reference directly minimizes the primal velocity objective. It does
    # not use the production dual or its reaction-distribution algorithm.
    diagonal = np.sqrt(np.diag(drag))
    rhs = torque.ravel()/diagonal
    matrix = drag/diagonal[:, None]/diagonal[None, :]
    factor = np.linalg.cholesky(matrix)
    rhs = np.linalg.solve(factor, rhs)
    amplitude = np.linalg.norm(rhs)
    rhs /= amplitude
    rows = np.linalg.solve(factor, (feed/diagonal).T).T
    rows /= np.linalg.norm(rows, axis=1)[:, None]
    oracle = minimize(lambda x: .5*(x-rhs)@(x-rhs), np.zeros(len(rhs)),
        jac=lambda x: x-rhs, method="SLSQP",
        constraints=LinearConstraint(rows, 0., np.inf),
        options=dict(ftol=1e-14, maxiter=5000))
    assert oracle.success, oracle.message
    actual = factor.T@(answer.omega_rad_s.ravel()*diagonal/amplitude)
    np.testing.assert_allclose(actual, oracle.x, atol=2e-8, rtol=2e-7)
    assert answer.primal_relative_violation < 2e-11
    assert answer.complementarity_relative_error < 2e-11
    assert answer.relative_residual < 2e-13
    assert np.all(answer.normal_forces_n >= 0.)


@pytest.mark.parametrize("name", _ACTUAL_CASES)
def test_actual_problem_section_split_preserves_tensile_stress(name):
    drag, torque, feed, weights = actual_case(name)
    whole = solve_no_eduction(drag, torque, feed, reaction_weights=weights)
    split_index = int(np.argmax(whole.normal_forces_n))
    fraction = .37
    split_feed = np.vstack([feed, feed[split_index]])
    split_weights = np.r_[weights, weights[split_index]*(1.-fraction)]
    split_weights[split_index] *= fraction
    split = solve_no_eduction(drag, torque, split_feed, reaction_weights=split_weights)
    omega_scale = np.linalg.norm(whole.omega_rad_s)
    np.testing.assert_allclose(split.omega_rad_s, whole.omega_rad_s, atol=omega_scale*2e-10)
    expected = whole.normal_forces_n[split_index]/weights[split_index]
    np.testing.assert_allclose(split.normal_forces_n[[split_index, -1]]/split_weights[[split_index, -1]],
                               expected, rtol=2e-8)


@pytest.mark.parametrize("name", _ACTUAL_CASES)
def test_actual_problem_constraint_permutation_preserves_solution(name):
    drag, torque, feed, weights = actual_case(name)
    original = solve_no_eduction(drag, torque, feed, reaction_weights=weights)
    permutation = np.random.default_rng(756).permutation(len(feed))
    permuted = solve_no_eduction(drag, torque, feed[permutation], reaction_weights=weights[permutation])
    np.testing.assert_allclose(permuted.omega_rad_s, original.omega_rad_s,
                               atol=np.linalg.norm(original.omega_rad_s)*2e-10)
    np.testing.assert_allclose(permuted.normal_forces_n, original.normal_forces_n[permutation],
                               atol=np.max(original.normal_forces_n)*2e-9)
