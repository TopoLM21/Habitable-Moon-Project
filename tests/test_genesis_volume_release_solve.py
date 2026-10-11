"""Bounded linear source trajectories, not an accepted-planet initialization."""
from dataclasses import replace
import inspect
import json

import numpy as np
import pytest

from tectonics.genesis_volume_enriched_memory import (
    prepare_enriched_loading, evaluate_enriched_loading, enriched_memory_to_dict,
)
from tectonics.genesis_volume_release_solve import (
    VolumeReleaseSolveRefusal, _mean_tangent, _solve_scaled_system,
    solve_release_work, volume_release_solve_to_dict,
)
from tectonics.genesis_volume_release_work import (
    SECONDS_PER_MYR, evaluate_release_work, volume_release_work_response_to_dict,
)
from tectonics.genesis_volume_release_cohesive import VolumeReleaseCohesiveRefusal
from tectonics.genesis_volume_seed_memory import source_seed_memory_loading
from tectonics.genesis_volume_operator import evaluate_genesis_volume_operator


def test_coupled_linear_operator_solution_keeps_off_diagonal_work_and_component_residuals():
    matrix = np.array(((7., -2., .4), (-2., 5., .7), (.4, .7, 3.)))
    expected = np.array((.3, -.4, 1.7))
    rhs = matrix@expected
    result, diagnostics = _solve_scaled_system(matrix, rhs)
    np.testing.assert_allclose(result, expected, rtol=3e-15, atol=0.)
    assert diagnostics["maximum_component_backward_error"] <= diagnostics["component_backward_error_bound"]
    assert diagnostics["positive_diagonal_scaled_without_floor"]
    assert not diagnostics["external_force_invented"]
    assert not diagnostics["historical_tie_reaction_used_as_force"]


@pytest.mark.parametrize("root_scale", [(1e-90, 1e90), (1e90, 1e-90)])
def test_extreme_positive_diagonal_scaling_preserves_literal_small_entries_without_a_floor(root_scale):
    roots = np.asarray(root_scale)
    normalized = np.array(((2., .3), (.3, 1.)))
    matrix = roots[:, None]*normalized*roots[None, :]
    expected_scaled = np.array((.7, -.9))
    rhs = roots*(normalized@expected_scaled)
    result, diagnostics = _solve_scaled_system(matrix, rhs)
    np.testing.assert_allclose(result*roots, expected_scaled, rtol=5e-15, atol=0.)
    assert diagnostics["scaled_condition_number"] < 3
    assert diagnostics["maximum_component_backward_error"] < 1e-14


@pytest.mark.parametrize("matrix,message", [
    (np.diag((1., 0.)), "nonpositive diagonal"),
    (np.diag((1., -1.)), "nonpositive diagonal"),
    (np.array(((1., .1), (0., 1.))), "not symmetric"),
    (np.array(((1., 2.), (2., 1.))), "not positive definite"),
    (np.array(((1., 1.-1e-15), (1.-1e-15, 1.))), "conditioning is unresolved"),
])
def test_unresolved_linear_operator_is_refused_without_regularization(matrix, message):
    with pytest.raises(VolumeReleaseSolveRefusal, match=message):
        _solve_scaled_system(matrix, np.ones(2))


@pytest.mark.parametrize("matrix,rhs", [
    (np.eye(2, dtype=bool), np.ones(2)),
    (np.eye(2), [True, 1.]),
    (np.array(((np.nan, 0.), (0., 1.))), np.ones(2)),
    (np.eye(2), [np.inf, 1.]),
    (np.ones((2, 3)), np.ones(2)),
    (np.eye(2), np.ones(3)),
])
def test_malformed_operator_or_force_is_refused(matrix, rhs):
    with pytest.raises(VolumeReleaseSolveRefusal):
        _solve_scaled_system(matrix, rhs)


def test_zero_source_force_has_exact_zero_solution_without_an_absolute_residual_floor():
    result, diagnostics = _solve_scaled_system(np.array(((3., .2), (.2, 4.))), np.zeros(2))
    np.testing.assert_array_equal(result, np.zeros(2))
    assert diagnostics["maximum_component_backward_error"] == 0.
    assert diagnostics["refinement_iterations"] == 0


@pytest.fixture(scope="module")
def tensor():
    from test_genesis_volume_release_work import make_archived_tensor_context
    result = make_archived_tensor_context()
    context = result[0]
    before = enriched_memory_to_dict(context.birth.initial_memory)
    yield result
    assert enriched_memory_to_dict(context.birth.initial_memory) == before


def test_Maxwell_mean_tangent_matches_affine_full_field_force_in_both_old_and_added_modes(tensor):
    context, *_ = tensor
    p, memory = context.partition, context.birth.initial_memory
    after = p.time_myr+1e-7
    loading = source_seed_memory_loading(context.birth, memory, after)
    trial = prepare_enriched_loading(p, memory, loading)
    tangent, theta = _mean_tangent(p, loading)
    zero = evaluate_enriched_loading(p, trial, np.zeros(p.parent_ndof), np.zeros(18))
    # The two mode families are checked separately so the enormous old force
    # does not certify an erased, much smaller enrichment contribution.
    for old_only in (True, False):
        d = np.zeros(p.ndof)
        if old_only:
            d[:p.parent_ndof] = np.sin(np.arange(p.parent_ndof)+.7)*1e-5
        else:
            d[p.parent_ndof:] = context.modes.release_basis@np.cos(np.arange(context.modes.rank)+.2)*1e-5
        response = evaluate_enriched_loading(p, trial, d[:p.parent_ndof], d[p.parent_ndof:])
        # Check the isolated superposed incremental field directly.  Its
        # force is independent of the large inherited prestress subtraction.
        if old_only:
            clean_old = evaluate_genesis_volume_operator(p.parent_assembly, d[:p.parent_ndof],
                elastic_constant=np.zeros_like(zero.mean0), elastic_inverse_scale=np.zeros_like(zero.mean1),
                strain_factors=theta).force
            actual = response.mean_generalized_force_n-zero.mean_generalized_force_n
            actual[:p.parent_ndof] = clean_old
        else:
            actual = response.mean_generalized_force_n-zero.mean_generalized_force_n
        expected = tangent@d
        roundoff = 512*np.finfo(float).eps*(np.abs(response.mean_generalized_force_n)
            +np.abs(zero.mean_generalized_force_n))
        bound = 2e-8*np.abs(expected)+roundoff
        assert np.all(np.abs(actual-expected) <= bound), {
            "old_only": old_only, "bad_indices": np.flatnonzero(np.abs(actual-expected) > bound).tolist()}
    assert np.all(theta >= 0.)


def test_source_forced_float18_path_conservatively_refuses_unresolved_actual_endpoint(tensor, monkeypatch):
    import tectonics.genesis_volume_release_solve as solver_module
    context, *_ = tensor
    original = solver_module.evaluate_release_work
    observed = []
    def observe(*args, **kwargs):
        response = original(*args, **kwargs)
        observed.append(volume_release_work_response_to_dict(response))
        return response
    monkeypatch.setattr(solver_module, "evaluate_release_work", observe)
    before = enriched_memory_to_dict(context.birth.initial_memory)
    with pytest.raises((VolumeReleaseSolveRefusal, VolumeReleaseCohesiveRefusal)) as refusal:
        solve_release_work(context, context.birth.initial_memory,
            context.cohesive.seed.initial_maximum_separation_m, context.partition.time_myr+1e-7)
    assert enriched_memory_to_dict(context.birth.initial_memory) == before
    if not observed:
        # The same expanded source path may be refused earlier by the frozen
        # component force/work guard.  No incomplete response is admitted.
        assert isinstance(refusal.value, VolumeReleaseCohesiveRefusal)
        assert "assembled mean force/work failed component convergence or exact path duality" in str(refusal.value)
        return
    assert isinstance(refusal.value, VolumeReleaseSolveRefusal)
    assert "independent full momentum/work gate" in str(refusal.value)
    assert len(observed) == 1
    response = observed[0]
    assert not response["balanced_trajectory"] and not response["projected_momentum_passed"]
    assert response["discrete_work_balance_passed"] and response["work_identity_passed"]
    assert response["complete_jump_kernel_remains_tied"]
    assert response["fracture_dissipation_increment_j"] == 0.
    np.testing.assert_array_equal(response["external_force_n"], context.partition.external_force_n)
    assert response["physical_release_admitted"] is response["funding_admitted"] is False
    assert response["new_accepted_Genesis_time_myr"] == response["creation_work_debited_from_Genesis_j"] == 0.


def test_source_velocity_is_required_and_cannot_be_reset_to_supply_a_different_trajectory(tensor):
    context, *_ = tensor
    changed = np.array(context.initial_velocity_m_s); changed[0] += 1e-9
    with pytest.raises(VolumeReleaseSolveRefusal, match="trusted source trajectory"):
        solve_release_work(context, context.birth.initial_memory,
            context.cohesive.seed.initial_maximum_separation_m, context.partition.time_myr+1e-7,
            before_velocity_m_s=changed)


def test_preexisting_damaged_history_is_explicitly_outside_bounded_solver(tensor):
    context, *_ = tensor
    old = np.array(context.cohesive.seed.initial_maximum_separation_m)
    old[0] = np.nextafter(context.cohesive.seed.peak_separation_m, np.inf)
    with pytest.raises(VolumeReleaseSolveRefusal, match="damaged history or softening"):
        solve_release_work(context, context.birth.initial_memory, old, context.partition.time_myr+1e-7)


@pytest.mark.parametrize("after", [True, np.nan, np.inf])
def test_invalid_time_is_refused_before_source_solve(tensor, after):
    context, *_ = tensor
    with pytest.raises(VolumeReleaseSolveRefusal):
        solve_release_work(context, context.birth.initial_memory,
            context.cohesive.seed.initial_maximum_separation_m, after)


def test_solver_does_not_accept_external_work_or_archived_tie_force_inputs():
    names = set(inspect.signature(solve_release_work).parameters)
    assert names == {"context", "before_memory", "old_maximum_separation_m", "after_time_myr",
                     "before_velocity_m_s"}
