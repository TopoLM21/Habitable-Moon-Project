"""Physical map checks use exact rational declarations, with no old controls."""
from dataclasses import replace
from fractions import Fraction as F

import pytest

from tectonics.geometric_native_pose import FractionInterval as I
from tectonics.geometric_native_common_cold_depth_preparation import (
    DeclaredColdDepthPreparation, NativeCommonColdDepthPreparation,
    bind_native_common_cold_depth_preparation,
    evaluate_declared_cold_depth_preparation,
    evaluate_native_common_cold_depth_preparation,
    native_common_cold_depth_preparation_to_dict,
)


X = (I(3), I(4), I(0))
IDENTITY = tuple(tuple(I(int(i == j)) for j in range(3)) for i in range(3))
DECLARATION = DeclaredColdDepthPreparation(F(10), F(1), (F(0), F(0), F(1, 1000)),
                                          "independent rational physical twist")


def mm(a, b):
    return tuple(tuple(sum((a[i][k]*b[k][j] for k in range(3)), I(0))
                       for j in range(3)) for i in range(3))


def contains(interval, exact):
    assert interval.lo <= exact <= interval.hi


def rational_sin_cos(angle):
    # Exact partial sums; the next alternating terms are far below 1e-200.
    sine, cosine = F(0), F(0)
    ts, tc = angle, F(1)
    for n in range(60):
        sine += ts; cosine += tc
        ts *= -angle*angle/F((2*n+2)*(2*n+3))
        tc *= -angle*angle/F((2*n+1)*(2*n+2))
    assert abs(ts) < F(1, 10**200) and abs(tc) < F(1, 10**200)
    return sine, cosine


def exact_declared_oracle(fraction):
    tau, rate = F(35, 12), F(1, 1000)
    sine, cosine = rational_sin_cos(fraction*rate*tau)
    Q = ((cosine, -sine, F(0)), (sine, cosine, F(0)), (F(0), F(0), F(1)))
    x = (F(3), F(4), F(0))
    y = tuple(sum((Q[i][k]*x[k] for k in range(3)), F(0)) for i in range(3))
    t = (F(-3, 20), F(-1, 5), F(0))
    w = (F(-4, 1000), F(3, 1000), F(0))
    D = tuple(tuple(Q[i][j]+fraction*sum((Q[i][k]*w[k] for k in range(3)), F(0))*t[j]
                    for j in range(3)) for i in range(3))
    return Q, y, D


def test_zero_fraction_keeps_positions_original_gradients_and_modes():
    modes = ((I(2), I(-1), I(3)),)
    gradients = (IDENTITY,)
    field = evaluate_declared_cold_depth_preparation(DECLARATION, X, preparation_fraction=F(0),
        original_gradient_bounds=IDENTITY, mode_displacement_columns=modes,
        mode_original_gradient_columns_per_m=gradients)
    assert field.position_bounds_m == X
    assert field.spatial_gradient_bounds == IDENTITY
    assert field.inverse_spatial_gradient_bounds == IDENTITY
    assert field.original_cartesian_gradient_bounds == IDENTITY
    assert field.mode_displacement_columns == modes
    assert field.mode_original_gradient_columns_per_m == gradients
    assert any(v.lo != 0 or v.hi != 0 for v in field.preparation_velocity_bounds_m)


def test_half_twist_has_cartesian_rank_one_gradient_and_unit_jacobian():
    field = evaluate_declared_cold_depth_preparation(DECLARATION, X, preparation_fraction=F(1, 2))
    Q, y, D = exact_declared_oracle(F(1, 2))
    for i in range(3):
        contains(field.position_bounds_m[i], y[i])
        for j in range(3):
            contains(field.rotation_bounds[i][j], Q[i][j])
            contains(field.spatial_gradient_bounds[i][j], D[i][j])
    assert field.map_volume_jacobian_bounds == I(1)
    assert field.spatial_gradient_bounds != field.rotation_bounds
    for i, row in enumerate(mm(field.spatial_gradient_bounds, field.inverse_spatial_gradient_bounds)):
        for j, entry in enumerate(row):
            contains(entry, F(int(i == j)))


def test_inverse_twist_and_group_composition_return_original_positions():
    half = evaluate_declared_cold_depth_preparation(DECLARATION, X, preparation_fraction=F(1, 2))
    twice = evaluate_declared_cold_depth_preparation(DECLARATION, half.position_bounds_m,
        preparation_fraction=F(1, 2))
    full = evaluate_declared_cold_depth_preparation(DECLARATION, X, preparation_fraction=F(1))
    back = evaluate_declared_cold_depth_preparation(DECLARATION, full.position_bounds_m,
        preparation_fraction=F(1), inverse=True)
    for old, restored, a, b in zip(X, back.position_bounds_m, twice.position_bounds_m, full.position_bounds_m):
        contains(restored, old.lo)
        assert max(a.lo, b.lo) <= min(a.hi, b.hi)


def test_physical_fraction_velocity_acceleration_and_original_gradient_rate():
    field = evaluate_declared_cold_depth_preparation(DECLARATION, X, preparation_fraction=F(1, 2),
                                                    original_gradient_bounds=IDENTITY)
    Q, y, _ = exact_declared_oracle(F(1, 2))
    tau, rate = F(35, 12), F(1, 1000)
    expected_velocity = (-rate*tau*y[1], rate*tau*y[0], F(0))
    expected_acceleration = (-rate*rate*tau*tau*y[0], -rate*rate*tau*tau*y[1], F(0))
    for actual, expected in zip(field.preparation_velocity_bounds_m, expected_velocity):
        contains(actual, expected)
    for actual, expected in zip(field.preparation_acceleration_bounds_m, expected_acceleration):
        contains(actual, expected)
    assert field.preparation_original_gradient_rate_bounds == field.preparation_spatial_gradient_rate_bounds
    # Radius invariance is also visible as y dot d_lambda y = 0.
    contains(sum((a*b for a,b in zip(field.position_bounds_m, field.preparation_velocity_bounds_m)), I(0)), F(0))


def test_second_spatial_derivative_and_mode_gradient_curvature_are_retained():
    field = evaluate_declared_cold_depth_preparation(DECLARATION, X, preparation_fraction=F(1, 2),
        original_gradient_bounds=IDENTITY, mode_displacement_columns=((I(0), I(0), I(1)),),
        mode_original_gradient_columns_per_m=(tuple((I(0),)*3 for _ in range(3)),))
    _, y, _ = exact_declared_oracle(F(1, 2))
    # At z=0, tau_z=0 but tau_zz=-|x|/R²=-1/20.
    # T_zz=Q ((fraction Omega×x) tau_zz).
    expected = (F(1, 40000)*y[1], -F(1, 40000)*y[0], F(0))
    for k in range(3):
        contains(field.spatial_hessian_bounds_per_m[k][2][2], expected[k])
        contains(field.mode_original_gradient_columns_per_m[0][k][2], expected[k])
    assert any(v.lo != 0 or v.hi != 0 for row in field.mode_original_gradient_columns_per_m[0] for v in row)
    for tensor in field.spatial_hessian_bounds_per_m:
        for i in range(3):
            for j in range(3):
                assert tensor[i][j] == tensor[j][i]


def test_complete_parameter_interval_encloses_both_endpoints_and_middle():
    whole = evaluate_declared_cold_depth_preparation(DECLARATION, X, preparation_fraction=I(0, 1))
    for a in (F(0), F(1, 2), F(1)):
        _, y, _ = exact_declared_oracle(a)
        for actual, exact in zip(whole.position_bounds_m, y):
            contains(actual, exact)


def test_fixed_owner_scale_enters_depth_and_derivative_with_its_full_cube():
    s = F(9, 10)
    declared = replace(DECLARATION, fixed_radial_scale=s)
    field = evaluate_declared_cold_depth_preparation(declared, tuple(s*v for v in X),
                                                    preparation_fraction=F(1))
    for actual, expected in zip(field.physical_volume_depth_gradient_bounds,
                                (F(-3, 20)/s, F(-1, 5)/s, F(0))):
        contains(actual, expected)
    contains(field.physical_volume_depth_bounds_m, F(35, 12))
    _, unscaled, _ = exact_declared_oracle(F(1))
    for actual, expected in zip(field.position_bounds_m, unscaled):
        contains(actual, s*expected)


def test_green_strain_is_quadratic_despite_nonlinear_physical_rotation():
    field = evaluate_declared_cold_depth_preparation(DECLARATION, X, preparation_fraction=F(2, 3))
    wx = (F(-1, 250), F(3, 1000), F(0))
    gradtau = (F(-3, 20), F(-1, 5), F(0))
    d = tuple(tuple(wx[i]*gradtau[j] for j in range(3)) for i in range(3))
    L = tuple(tuple((d[i][j]+d[j][i])/2 for j in range(3)) for i in range(3))
    Q = tuple(tuple(sum((d[k][i]*d[k][j] for k in range(3)), F(0))/2
                    for j in range(3)) for i in range(3))
    geometric = mm(tuple(zip(*field.spatial_gradient_bounds)), field.spatial_gradient_bounds)
    for i in range(3):
        for j in range(3):
            exact = F(2, 3)*L[i][j]+F(4, 9)*Q[i][j]
            contains(field.relative_green_strain_linear_coefficients[i][j], L[i][j])
            contains(field.relative_green_strain_quadratic_coefficients[i][j], Q[i][j])
            contains(field.relative_green_strain_bounds[i][j], exact)
            contains((geometric[i][j]-int(i == j))/2, exact)
    reverse = evaluate_declared_cold_depth_preparation(DECLARATION, X,
        preparation_fraction=F(2, 3), inverse=True)
    for i in range(3):
        for j in range(3):
            contains(reverse.relative_green_strain_linear_coefficients[i][j], -L[i][j])
            contains(reverse.relative_green_strain_quadratic_coefficients[i][j], Q[i][j])


def test_original_green_coefficients_change_old_strain_without_resetting_it():
    old_A = tuple(tuple(I(F((2, 3, 4)[i]*int(i == j))) for j in range(3)) for i in range(3))
    field = evaluate_declared_cold_depth_preparation(DECLARATION, X, preparation_fraction=F(1),
                                                    original_gradient_bounds=old_A)
    expected_L = mm(mm(tuple(zip(*old_A)), field.relative_green_strain_linear_coefficients), old_A)
    expected_Q = mm(mm(tuple(zip(*old_A)), field.relative_green_strain_quadratic_coefficients), old_A)
    for i in range(3):
        for j in range(3):
            for actual, expected in ((field.original_green_strain_linear_coefficients[i][j], expected_L[i][j]),
                                     (field.original_green_strain_quadratic_coefficients[i][j], expected_Q[i][j])):
                assert max(actual.lo, expected.lo) <= min(actual.hi, expected.hi)
    unchanged = evaluate_declared_cold_depth_preparation(DECLARATION, X, preparation_fraction=F(0),
                                                        original_gradient_bounds=old_A)
    assert unchanged.original_cartesian_gradient_bounds == old_A
    assert unchanged.relative_green_strain_bounds == tuple((I(0),)*3 for _ in range(3))
    assert unchanged.source_reference_or_rest_metric_reset is False


def test_physical_path_is_unpaid_and_does_not_advance_clocks_or_admission():
    field = evaluate_declared_cold_depth_preparation(DECLARATION, X, preparation_fraction=F(1))
    assert field.radius_preservation_identity and field.global_inverse_identity
    assert field.preparation_work_paid is False
    assert field.source_reference_or_rest_metric_reset is False
    assert field.physical_material_admission_authorized is False
    assert field.new_mixed_force_equilibrium_certified is False
    assert field.physical_elapsed_time_added is False
    assert field.accepted_new_native_intervals == 0


@pytest.mark.parametrize("bad", [0.5, True, 1, F(-1), F(2), I(-1, 1)])
def test_untyped_or_outside_fraction_is_rejected(bad):
    with pytest.raises(ValueError, match="Preparation fraction"):
        evaluate_declared_cold_depth_preparation(DECLARATION, X, preparation_fraction=bad)


def test_unresolved_origin_and_incomplete_mode_gradient_context_are_rejected():
    with pytest.raises(ValueError, match="origin"):
        evaluate_declared_cold_depth_preparation(DECLARATION, (I(-1, 1),)*3, preparation_fraction=F(1))
    with pytest.raises(ValueError, match="Mode original gradients"):
        evaluate_declared_cold_depth_preparation(DECLARATION, X, preparation_fraction=F(1),
                                                mode_original_gradient_columns_per_m=(IDENTITY,))


def test_float_declaration_and_unbound_native_authority_are_rejected():
    with pytest.raises(ValueError, match="literal Fractions"):
        replace(DECLARATION, radius_m=10.0)
    with pytest.raises(ValueError, match="literal Fractions"):
        replace(DECLARATION, angular_rate_per_physical_volume_depth_m=(F(0), F(0), 0.001))
    with pytest.raises(ValueError):
        bind_native_common_cold_depth_preparation(object(), object(), 1, provenance="unbound")
    fabricated = NativeCommonColdDepthPreparation(DECLARATION, "a"*64, "b"*64, "c"*64,
        1, ("octant:4",), (F(0),)*4, "d"*64)
    with pytest.raises(ValueError, match="complete source binding"):
        native_common_cold_depth_preparation_to_dict(fabricated)
    with pytest.raises(ValueError, match="complete source binding"):
        evaluate_native_common_cold_depth_preparation(fabricated, X, preparation_fraction=F(1))

