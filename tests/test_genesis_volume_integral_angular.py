"""Strict own-component tolerances and explicit unmeasured-node refusals."""
from fractions import Fraction

import numpy as np
import pytest

from analysis.genesis_volume_integral_validation import angular_control as angular

LD = np.longdouble
NAMES = angular.VOLUME_NAMES + ("full.weak[0,0]",)


@pytest.fixture(autouse=True)
def one_candidate(monkeypatch):
    monkeypatch.setattr(angular, "read_source", lambda raw: object())
    monkeypatch.setattr(angular, "_angular_support", lambda source: (np.eye(3, dtype=LD),))
    monkeypatch.setattr(angular, "_cell_bounds", lambda source, vertices: {
        -1: (0., .5), 1: (0., .5)})


def run(evaluator, **options):
    return angular.integrate_angular({}, evaluator, NAMES,
        **{**dict(max_cells=1, max_line_evaluations=34), **options})


def test_large_source_uncertainty_does_not_enlarge_acceptance():
    result = run(lambda n: ([1., 1., 1.], [0., 0., .01]))
    report = result.to_dict()
    assert report["status"] == "OPEN"
    assert report["source_uncertainty_failed_component_count"] == 1
    assert report["source_bounds_added_to_acceptance_allowance"] is False
    assert result.combined_error[2] > result.allowance[2]
    assert result.allowance[2] == pytest.approx(1e-6)


def test_small_source_uncertainty_fits_original_tolerance():
    result = run(lambda n: ([1., 1., 1.], [0., 0., 1e-9]))
    assert result.to_dict()["status"] == "FINITE_AGREEMENT"
    assert result.combined_error[2] < result.allowance[2]
    assert result.to_dict()["angular_accuracy_certified"] is False


def test_zero_component_keeps_exact_zero_scale():
    result = run(lambda n: ([1., 1., 0.], [0., 0., 0.]))
    assert result.to_dict()["status"] == "FINITE_AGREEMENT"
    assert result.allowance[2] == 0
    assert result.combined_error[2] == 0
    assert result.to_dict()["positive_scale_floor_used"] is False


def test_zero_component_with_uncertainty_remains_open():
    result = run(lambda n: ([1., 1., 0.], [0., 0., 1e-40]))
    assert result.to_dict()["status"] == "OPEN"
    assert result.allowance[2] == 0
    assert result.combined_error[2] > 0
    assert result.to_dict()["worst_error_ratio_unbounded"] is True


def test_weak_component_cannot_borrow_volume_scale():
    result = run(lambda n: ([1e12, 1e12, 1e-20], [0., 0., 1e-24]))
    report = result.to_dict()
    assert report["status"] == "OPEN"
    assert report["failed_component_count"] == 1
    assert report["worst_component"] == NAMES[2]


def test_positive_unsampled_candidate_cannot_disappear():
    result = run(lambda n: ([0., 0., 1.], [0., 0., 0.]))
    report = result.to_dict()
    assert report["status"] == "OPEN"
    assert report["unresolved_positive_candidate_cells"] == 1
    assert report["unresolved_positive_candidate_volume_m3"] == [.5, .5]


def test_line_refusal_is_not_replaced_by_zero():
    calls = []
    def evaluator(n):
        calls.append(np.array(n, copy=True))
        if len(calls) == 3:
            raise angular.NumericalLineBoundRefusal("Positive source line has no native chart")
        return [1., 1., 1.], [0., 0., 0.]
    with pytest.raises(angular.NumericalLineBoundRefusal) as caught:
        run(evaluator)
    report = caught.value.to_dict()
    assert report["status"] == "OPEN"
    assert report["completed_source_bound_line_evaluations"] == 2
    assert report["attempted_line_evaluations"] == 3
    assert report["unmeasured_line_replaced_with_zero"] is False
    assert report["offending_barycentric_exact_binary_ratios"] == [
        [str(a), str(b)] for a,b in (LD(v).as_integer_ratio() for v in calls[-1])]


def test_insufficient_initial_work_cap_refuses_before_evaluation():
    calls = []
    def evaluator(n):
        calls.append(n)
        return [1., 1., 1.], [0., 0., 0.]
    with pytest.raises(angular.AngularControlRefusal, match="initial conservative support"):
        run(evaluator, max_line_evaluations=33)
    assert not calls


@pytest.mark.parametrize("values,bounds", [
    ([1., 1.], [0., 0.]), ([1., 1., 1.], [0., 0., -1.]),
    ([-1., 1., 1.], [0., 0., 0.]), ([1., 1., np.inf], [0., 0., 0.]),
])
def test_invalid_bound_protocol_refuses(values, bounds):
    with pytest.raises(angular.AngularControlRefusal, match="Invalid source-bound radial"):
        run(lambda n: (values, bounds))


def test_reduction_bounds_cover_exact_binary_weighted_source_values():
    observed = []
    source_shift = LD("1e-8")
    def evaluator(n):
        value = LD(1)+n[1]-n[2]
        observed.append(value)
        return [1., 1., value], [0., 0., source_shift]
    result = run(evaluator)
    offset = 0
    for order, native_value, error in ((3,result.coarse[2],result.coarse_roundoff[2]),
                                     (5,result.fine[2],result.fine_roundoff[2])):
        nodes, weights = np.polynomial.legendre.leggauss(order)
        nodes, weights = np.asarray((nodes+1)/2,LD), np.asarray(weights/2,LD)
        exact = Fraction()
        for s, sw in zip(nodes, weights):
            for tw in weights:
                v = observed[offset]
                offset += 1
                exact += (Fraction(*s.as_integer_ratio())*Fraction(*sw.as_integer_ratio())*
                          Fraction(*tw.as_integer_ratio())*
                          (Fraction(*v.as_integer_ratio())+Fraction(*source_shift.as_integer_ratio())))
        residual = abs(Fraction(*native_value.as_integer_ratio())-exact)
        assert residual <= Fraction(*error.as_integer_ratio())
    assert offset == 34


def test_refinement_preserves_source_uncertainty_and_work_cap():
    result = angular.integrate_angular({},
        lambda n: ([1.,1.,1.], [0.,0.,.01]), NAMES,
        max_cells=5, max_line_evaluations=170)
    report = result.to_dict()
    assert report["status"] == "OPEN"
    assert report["radial_line_evaluations"] == 170
    assert report["evaluated_angular_cells"] == 5
    assert report["retained_angular_cells"] == 4
    assert report["combined_observed_and_source_error_bounds"][2] > .009


def test_duplicate_component_names_refuse():
    with pytest.raises(angular.AngularControlRefusal, match="First components"):
        angular.integrate_angular({}, lambda n: None, angular.VOLUME_NAMES+(angular.VOLUME_NAMES[0],))
