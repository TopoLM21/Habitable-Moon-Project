"""Observe actual evolution without substituting a static unload experiment."""
from copy import deepcopy
import numpy as np
import pytest

from analysis.genesis_coupled_balance_audit import advance_with_balance_audit, reconstruct_balance
from analysis.genesis_coupled_validation import _fingerprint
from tectonics.genesis_coupled import CoupledModel, _RetryStep
from test_genesis_contact import _source_bytes


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    return _source_bytes(tmp_path_factory.mktemp("physical_balance"))[0]


def test_observer_preserves_actual_evolution_and_closes_force_and_work(source):
    model = CoupledModel(source)
    before = model.initial(), model.source.thermal_state, model.source.orbit
    saved = _fingerprint(before)
    expected = model.step(*before, before[0].time_myr+10/1e6)
    result, balances = advance_with_balance_audit(model, *before, before[0].time_myr+10/1e6)
    assert _fingerprint(result) == _fingerprint(expected)
    assert _fingerprint(before) == saved
    assert len(balances) == result[0].accepted_steps-before[0].accepted_steps
    assert sum(b.summary["step_years"] for b in balances) == pytest.approx(10.)
    for balance in balances:
        row, a = balance.summary, balance.arrays
        assert row["force_balance_relative_residual"] <= model.contact_parameters.equilibrium_tolerance
        assert row["remainder_identity_relative_error"] < 1e-10
        for name in ("bulk", "external", "drag"):
            assert row[name+"_work_ledger_relative_error"] < 1e-10
        assert row["mechanical_remainder_ledger_relative_error"] < 1e-10
        np.testing.assert_array_equal(a["residual_force_n"],
            a["bulk_force_n"]+a["contact_force_n"]+a["drag_force_n"]-a["external_force_n"])
    # Basal drag is a physical term. Removing it does not evaluate equilibrium
    # of this accepted physical step, even though the state itself is valid.
    assert max(b.summary["static_relative_residual_without_drag"] for b in balances) > .01
    assert "_solve" not in model.__dict__ and "_trial" not in model.__dict__


def test_corrupted_final_elastic_state_fails_independent_balance(source, monkeypatch):
    model = CoupledModel(source)
    before = model.initial(), model.source.thermal_state, model.source.orbit
    original, calls = model._solve, []

    def capture(state, loading, damage):
        solved = original(state, loading, damage)
        calls.append((state, loading, damage, solved))
        return solved

    monkeypatch.setattr(model, "_solve", capture)
    model.step(*before, before[0].time_myr+1/1e6)
    state, loading, damage, solved = calls[-1]
    bad = list(deepcopy(solved))
    bad[1][:, :2] += .001
    assert reconstruct_balance(model, state, loading, damage, bad).summary["force_balance_relative_residual"] > .01


def test_failed_complete_trial_is_discarded_and_corrector_not_double_counted(source, monkeypatch):
    model = CoupledModel(source)
    before = model.initial(), model.source.thermal_state, model.source.orbit
    original, attempted = model._trial, []

    def fail_once(*args):
        result = original(*args)
        attempted.append(result[0].time_myr)
        if len(attempted) == 1:
            raise _RetryStep("test_post_mechanics_rollback")
        return result

    monkeypatch.setattr(model, "_trial", fail_once)
    result, balances = advance_with_balance_audit(model, *before, before[0].time_myr+10/1e6)
    assert result[0].stopped_reason is None
    assert result[0].rejected_steps >= 1
    assert len(balances) == result[0].accepted_steps-before[0].accepted_steps
    assert len(attempted) == len(balances)+1
    assert sum(b.summary["step_years"] for b in balances) == pytest.approx(10.)
    assert model._trial is fail_once


def test_audit_restores_model_methods_after_unexpected_failure(source, monkeypatch):
    model = CoupledModel(source)
    before = model.initial(), model.source.thermal_state, model.source.orbit

    def broken(*args):
        raise ValueError("unexpected failure")

    monkeypatch.setattr(model, "_trial", broken)
    with pytest.raises(ValueError, match="unexpected failure"):
        advance_with_balance_audit(model, *before, before[0].time_myr+1/1e6)
    assert model._trial is broken
    assert "_solve" not in model.__dict__
