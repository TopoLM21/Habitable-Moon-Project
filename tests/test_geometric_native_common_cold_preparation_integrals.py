"""New physical preparation integration: exact invariants and whole measure."""
from fractions import Fraction as F

import pytest

from tectonics.geometric_native_pose import FractionInterval as I, interval_pi
from tectonics.geometric_native_common_cold_continuum_energy import _BoxAlgebra
from tectonics import geometric_native_common_cold_preparation_integrals as api


def contains(bound, value):
    assert bound.lo <= value <= bound.hi


def mm(a, b):
    return tuple(tuple(sum((a[i][k]*b[k][j] for k in range(3)), F(0))
                       for j in range(3)) for i in range(3))


def transpose(a):
    return tuple(tuple(a[j][i] for j in range(3)) for i in range(3))


def exact_inner(a, b, lam, mu):
    return lam*sum((a[i][i] for i in range(3)), F(0))*sum((b[i][i] for i in range(3)), F(0)) \
        +2*mu*sum((a[i][j]*b[i][j] for i in range(3) for j in range(3)), F(0))


N = (F(3, 5), F(4, 5), F(0))
OMEGA = (F(1, 10), F(-1, 5), F(3, 10))
H = ((F(2), F(3), F(5)), (F(3), F(7), F(11)), (F(5), F(11), F(13)))


def exact_green(n=N, omega=OMEGA, t=F(7)):
    w = (omega[1]*n[2]-omega[2]*n[1], omega[2]*n[0]-omega[0]*n[2],
         omega[0]*n[1]-omega[1]*n[0])
    d = tuple(tuple(-t*w[i]*n[j] for j in range(3)) for i in range(3))
    L = tuple(tuple((d[i][j]+d[j][i])/2 for j in range(3)) for i in range(3))
    Q = tuple(tuple(x/2 for x in row) for row in mm(transpose(d), d))
    return L, Q


@pytest.mark.parametrize('lame,shear', [(F(5), F(3)), (F(-1), F(3))])
def test_relative_green_projection_and_squared_moments(lame, shear):
    algebra = _BoxAlgebra(I(0), 96)
    typed_H = tuple(tuple(algebra.constant(v) for v in row) for row in H)
    r1, r2, S2, S3, S4, Lbounds, Qbounds, invariant = api._relative_twist_kernel(
        N, F(7), OMEGA, (typed_H,), lame, shear, algebra)
    L, Q = exact_green()
    contains(r1[0], exact_inner(H, L, lame, shear))
    contains(r2[0], exact_inner(H, Q, lame, shear))
    contains(S2, exact_inner(L, L, lame, shear)/2)
    contains(S4, exact_inner(Q, Q, lame, shear)/2)
    assert exact_inner(L, Q, lame, shear) == 0
    assert S3 == I(0)
    assert S2.lo > 0 and S4.lo > 0
    for i in range(3):
        for j in range(3):
            contains(Lbounds[i][j], L[i][j])
            contains(Qbounds[i][j], Q[i][j])
    contains(invariant, sum((x*x for x in OMEGA), F(0))
             -sum((a*b for a,b in zip(OMEGA, N)), F(0))**2)


def test_zero_tip_complement_still_has_physical_preparation_energy():
    result = api._relative_twist_kernel(N, F(7), OMEGA, (), F(5), F(3), _BoxAlgebra(I(0), 96))
    assert result[0] == result[1] == ()
    assert result[2].lo > 0 and result[4].lo > 0
    assert result[3] == I(0)


def test_zero_rate_keeps_every_new_physical_moment_exactly_zero():
    algebra = _BoxAlgebra(I(0), 96)
    typed_H = tuple(tuple(algebra.constant(v) for v in row) for row in H)
    result = api._relative_twist_kernel(N, F(7), (F(0),)*3, (typed_H,), F(5), F(3), algebra)
    assert result[:5] == ((I(0),), (I(0),), I(0), I(0), I(0))


def test_global_rotation_objectivity_of_relative_strain_moments():
    # Rational orthogonal quarter turn; this rotates H and Omega along with n.
    rotation = ((F(0), F(-1), F(0)), (F(1), F(0), F(0)), (F(0), F(0), F(1)))
    rotate = lambda x: tuple(sum((rotation[i][j]*x[j] for j in range(3)), F(0)) for i in range(3))
    rotated_H = mm(mm(rotation, H), transpose(rotation))
    algebra = _BoxAlgebra(I(0), 96)
    current = api._relative_twist_kernel(rotate(N), F(7), rotate(OMEGA),
        (tuple(tuple(algebra.constant(v) for v in row) for row in rotated_H),), F(5), F(3), algebra)
    L, Q = exact_green()
    for actual, exact in zip((current[0][0], current[1][0], current[2], current[3], current[4]),
            (exact_inner(H, L, F(5), F(3)), exact_inner(H, Q, F(5), F(3)),
             exact_inner(L, L, F(5), F(3))/2, F(0), exact_inner(Q, Q, F(5), F(3))/2)):
        contains(actual, exact)


def test_uniform_activation_collar_respects_relative_inverse_and_contraction():
    r1, r2, s2, s3, s4, hcaps, dcap = api._collar_channels((F(2), F(3)),
        fixed_scale=F(101, 100), carrier_deviation=F(1, 10),
        omega_norm_upper=F(2, 5), original_radius_cubed_over_R_squared_upper=F(7),
        lame_pa=F(5), shear_pa=F(3), precision=96)
    assert hcaps == (F(2)/(F(101, 100)*F(9, 10)), F(3)/(F(101, 100)*F(9, 10)))
    assert dcap == F(14, 5)
    assert all(v.lo < 0 < v.hi for v in r1+r2)
    contains(s2, F(3)*dcap**2/2)
    contains(s4, F(11)*dcap**4/8)
    assert s2.lo == s4.lo == 0 and s3 == I(0)


@pytest.mark.parametrize('deviation', [F(-1, 10), F(1), F(2)])
def test_collar_requires_proved_positive_full_reference_inverse(deviation):
    with pytest.raises(ValueError):
        api._collar_channels((F(1),), fixed_scale=F(1), carrier_deviation=deviation,
            omega_norm_upper=F(1), original_radius_cubed_over_R_squared_upper=F(1),
            lame_pa=F(5), shear_pa=F(3), precision=96)


def test_complete_parent_measure_keeps_collar_active_complement_and_pole_once():
    R, H, gamma, bcut, delta = F(100), F(2), F(1, 10000), F(1, 10), F(1, 100)
    pi2 = interval_pi(112)/2
    regions = ((I(0), I(delta)), (I(delta), pi2-I(delta)), (I(0), I(delta)))
    total = I(0)
    for theta_lo, theta_hi in regions:
        for beta_lo, beta_hi, pole in ((I(0), I(2*gamma), False),
                (I(2*gamma), I(bcut), False), (I(bcut), pi2, True)):
            total += api._cell_volume(R, theta_lo, theta_hi, beta_lo, beta_hi, I(0, H), 96, pole)
    expected = pi2*R**2*H
    assert total.lo <= expected.lo <= expected.hi <= total.hi
    active_only = api._cell_volume(R, I(0), pi2, I(0), I(bcut), I(0, H), 96)
    assert total.lo > 5*active_only.hi


@pytest.mark.parametrize('options', [dict(precision=True), dict(theta_cell_count=True),
    dict(beta_cell_count=0), dict(depth_cell_count=-1), dict(max_cells=1), dict(provenance='')])
def test_incomplete_or_typed_aliased_traversals_are_rejected_before_any_sources(options):
    with pytest.raises(ValueError):
        api.integrate_native_common_cold_preparation_moments(None, None, None, None, **options)
