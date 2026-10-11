"""Angular control goldens and false-zero/weak-component refusals."""
import numpy as np
import pytest

from analysis.genesis_volume_exact_lambda_validation.angular_control import (
    AngularControlRefusal, integrate_angular,
)


def source():
    return dict(geometry_source=dict(triangle_m=np.eye(3).tolist(),
                displacement_A_m=np.zeros((3, 3)).tolist(),
                slope_P_m=np.zeros((3, 3)).tolist(), lambda_bounds=[.2, 1.],
                normal_xyz=[1., 0., 0.], offset_m=100.),
                seed_center_world_m=[0., 0., 0.], seed_radius_m=10.)


NAMES = ("reference_volume_negative_m3", "reference_volume_positive_m3")
VOLUME = (np.longdouble(1)-np.longdouble(.2)**3)/3


def test_constant_radial_volume_and_angular_polynomial_golden():
    def evaluate(N):
        return np.array([VOLUME, 0, N[1]**2, N[1]*N[2]], np.longdouble), np.zeros(4, np.longdouble)
    result = integrate_angular(source(), evaluate, NAMES+("u_squared", "u_v"))
    assert result.to_dict()["finite_order_numerical_agreement"]
    np.testing.assert_allclose(result.fine, [VOLUME/2, 0, 1/12, 1/24], rtol=2e-14, atol=0)
    assert result.line_evaluations == 34
    assert not result.to_dict()["general_contraction_accuracy_certified"]


def test_unsampled_positive_candidate_never_becomes_zero_domain():
    result = integrate_angular(source(), lambda N: (np.zeros(2), np.zeros(2)),
                               NAMES, max_cells=1, max_line_evaluations=34)
    assert result.to_dict()["status"] == "OPEN"
    assert result.unresolved_volume[0] > 0
    assert result.embedded_error[0] > 0
    assert result.upper_volume[0] > 0
    assert result.to_dict()["worst_error_ratio_unbounded"]


def test_weak_component_cannot_borrow_large_neighbor_scale():
    def evaluate(N):
        return np.array([VOLUME, 0, 1e40, 1e-30*N[1]**10], np.longdouble), np.zeros(4)
    result = integrate_angular(source(), evaluate, NAMES+("large", "weak"),
                               max_cells=1, max_line_evaluations=34)
    assert result.embedded_error[3] > result.allowance[3]
    assert result.to_dict()["status"] == "OPEN"
    assert result.allowance[3] < 1e-34
    assert result.to_dict()["positive_scale_floor_used"] is False


def test_work_cap_is_enforced_before_child_evaluations():
    counter = [0]
    def evaluate(N):
        counter[0] += 1
        return np.array([VOLUME, 0, N[1]**10]), np.zeros(3)
    result = integrate_angular(source(), evaluate, NAMES+("higher_degree",),
                               max_cells=2000, max_line_evaluations=34)
    assert result.line_evaluations == counter[0] == 34
    assert result.reason == "Predeclared angular cell or radial-line work budget exhausted"


def test_coarse_reduction_budget_survives_fine_node_polynomial_zeros():
    high_nodes = (np.polynomial.legendre.leggauss(5)[0]+1)/2
    def evaluate(N):
        polynomial = np.prod((1-N[0]-high_nodes.astype(np.longdouble))**2)
        return np.array([VOLUME, 0, polynomial], np.longdouble), np.zeros(3)
    result = integrate_angular(source(), evaluate, NAMES+("coarse_only_samples",),
                               max_cells=1, max_line_evaluations=34)
    assert abs(result.coarse[2]) > 1e20*abs(result.fine[2])
    assert result.coarse_roundoff[2] >= 36*np.finfo(np.longdouble).eps*abs(result.coarse[2])
    assert result.to_dict()["status"] == "OPEN"


def test_global_component_error_priority_resolves_nonuniform_polynomial():
    def evaluate(N):
        return np.array([VOLUME, 0, 1e40, N[1]**10], np.longdouble), np.zeros(4)
    result = integrate_angular(source(), evaluate, NAMES+("irrelevant_large", "u_power_10"),
                               max_line_evaluations=6800)
    assert result.to_dict()["finite_order_numerical_agreement"]
    assert abs(result.fine[3]-np.longdouble(1)/132) < 2e-12
    assert result.maximum_depth < 10
    assert result.embedded_error[3] <= result.allowance[3]


@pytest.mark.parametrize("options", [dict(relative_tolerance=True), dict(max_cells=0),
    dict(max_depth=True), dict(angular_orders=(5, 3)), dict(max_line_evaluations=1)])
def test_invalid_or_initially_insufficient_budgets_refuse(options):
    with pytest.raises(AngularControlRefusal):
        integrate_angular(source(), lambda N: (np.array([VOLUME, 0]), np.zeros(2)), NAMES, **options)


@pytest.mark.parametrize("values,budgets", [([1, 0], [-1, 0]),
    ([np.nan, 0], [0, 0]), ([-1, 0], [0, 0]), ([1], [0])])
def test_invalid_radial_evidence_refuses(values, budgets):
    with pytest.raises(AngularControlRefusal):
        integrate_angular(source(), lambda N: (values, budgets), NAMES)


def test_component_schema_is_unambiguous():
    with pytest.raises(AngularControlRefusal):
        integrate_angular(source(), lambda N: ([1, 0], [0, 0]), ("volume", "volume"))
