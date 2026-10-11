from fractions import Fraction as F

import pytest

from tectonics.geometric_native_pose import FractionInterval as I
from tectonics.geometric_native_reference_density_transport import (
    ReferenceDensityTwoJet, pullback_density_to_modes, transport_reference_density,
)


def matrix(rows):
    return tuple(tuple(F(x) for x in row) for row in rows)


IDENTITY = matrix(((1, 0, 0), (0, 1, 0), (0, 0, 1)))
BRIDGE = matrix(((2, 1, 0), (0, 3, 1), (1, 0, 2)))  # nonorthogonal, J=13
OLD_F = matrix(((1, F(1, 5), 0), (0, 1, F(1, 7)), (F(1, 11), 0, 1)))


def density(n=11):
    # Anisotropic potential with nonzero prestress and F/internal cross terms.
    H = tuple(tuple(F((a+1)*(b+1), 113)+F(3 if a == b else 0)
                    for b in range(n)) for a in range(n))
    return ReferenceDensityTwoJet(F(987654321, 17),
        tuple(F(7*a-19, 23) for a in range(n)), H)


def mm(a, b):
    return tuple(tuple(sum((a[i][k]*b[k][j] for k in range(len(b))), F(0))
                       for j in range(len(b[0]))) for i in range(len(a)))


def source_channels(n=11, modes=4):
    D = tuple(tuple(F((a+1)*(i+2)-5, 31) for i in range(modes)) for a in range(n))
    Q = tuple(tuple(tuple(F((a+2)*(i+j+1), 37) for j in range(modes))
                    for i in range(modes)) for a in range(n))
    return D, Q


def transformed_channels(D, Q, inverse):
    n, d = len(D), len(D[0])
    Dnew = []
    Qnew = []
    for a in range(n):
        if a >= 9:
            Dnew.append(D[a]); Qnew.append(Q[a]); continue
        row, col = divmod(a, 3)
        Dnew.append(tuple(sum((D[3*row+k][i]*inverse[k][col]
                               for k in range(3)), F(0)) for i in range(d)))
        Qnew.append(tuple(tuple(sum((Q[3*row+k][i][j]*inverse[k][col]
                                     for k in range(3)), F(0)) for j in range(d))
                          for i in range(d)))
    return tuple(Dnew), tuple(Qnew)


def test_identity_preserves_literal_energy_prestress_and_tangent():
    old = density()
    t = transport_reference_density(IDENTITY, OLD_F, old)
    assert t.reference_volume_ratio == 1
    assert t.new_deformation_gradient == OLD_F
    assert t.new_density == old


def test_nonorthogonal_transport_preserves_complete_modal_work_and_curvature():
    old = density()
    t = transport_reference_density(BRIDGE, OLD_F, old)
    assert t.reference_volume_ratio == 13
    assert mm(t.new_deformation_gradient, BRIDGE) == OLD_F
    D, Q = source_channels()
    Dnew, Qnew = transformed_channels(D, Q, t.inverse_bridge)
    original = pullback_density_to_modes(old, D, Q, F(17))
    changed = pullback_density_to_modes(t.new_density, Dnew, Qnew, F(17)*13)
    assert original == changed
    zeros = tuple(tuple(tuple(F(0) for _ in row) for row in q) for q in Q)
    assert original[2] != pullback_density_to_modes(old, D, zeros, F(17))[2]


def test_inverse_reference_change_recovers_entire_density_twojet():
    old = density()
    t = transport_reference_density(BRIDGE, OLD_F, old)
    back = transport_reference_density(t.inverse_bridge, t.new_deformation_gradient, t.new_density)
    assert back.new_deformation_gradient == OLD_F
    assert back.new_density == old
    assert back.reference_volume_ratio*t.reference_volume_ratio == 1


def test_reference_area_traction_and_cauchy_stress_are_unchanged():
    old = density()
    t = transport_reference_density(BRIDGE, OLD_F, old)
    Pold = tuple(old.gradient[3*i:3*i+3] for i in range(3))
    Pnew = tuple(t.new_density.gradient[3*i:3*i+3] for i in range(3))
    # Cof B = J B^-T. Reference area vector is not a unit normal alone.
    cof = tuple(tuple(13*t.inverse_bridge[j][i] for j in range(3)) for i in range(3))
    area = ((F(2),), (F(-3),), (F(5),))
    assert mm(Pnew, mm(cof, area)) == mm(Pold, area)
    Ftold = tuple(zip(*OLD_F)); Ftnew = tuple(zip(*t.new_deformation_gradient))
    assert mm(Pold, Ftold) == tuple(tuple(13*v for v in row) for row in mm(Pnew, Ftnew))


def test_outward_interval_transport_contains_exact_result():
    t = transport_reference_density(BRIDGE, OLD_F, density())
    lift = lambda a: tuple(tuple(I(v) for v in row) for row in a)
    old = density()
    enclosure = ReferenceDensityTwoJet(I(old.energy_density), tuple(I(v) for v in old.gradient), lift(old.hessian))
    out = transport_reference_density(lift(BRIDGE), lift(OLD_F), enclosure)
    assert out.reference_volume_ratio.lo <= 13 <= out.reference_volume_ratio.hi
    for actual, exact in zip(out.new_density.gradient, t.new_density.gradient):
        assert actual.lo <= exact <= actual.hi
    for ar, er in zip(out.new_density.hessian, t.new_density.hessian):
        for actual, exact in zip(ar, er):
            assert actual.lo <= exact <= actual.hi


@pytest.mark.parametrize('bad', [
    matrix(((1, 0, 0), (0, 1, 0), (0, 0, -1))),
    matrix(((1, 0, 0), (0, 1, 0), (0, 0, 0))),
    ((I(1), I(0), I(0)), (I(0), I(1), I(0)), (I(0), I(0), I(-1, 1))),
])
def test_nonpositive_or_uncertified_reference_jacobian_is_rejected(bad):
    with pytest.raises(ValueError, match='strictly positive'):
        transport_reference_density(bad, OLD_F, density())


def test_float_and_asymmetric_source_jet_are_rejected():
    with pytest.raises(ValueError, match='exact fractions'):
        transport_reference_density(tuple(tuple(float(x) for x in row) for row in BRIDGE), OLD_F, density())
    old = density()
    H = tuple(tuple(v+F(1) if (a,b)==(0,1) else v for b,v in enumerate(row))
              for a,row in enumerate(old.hessian))
    with pytest.raises(ValueError, match='symmetric'):
        ReferenceDensityTwoJet(old.energy_density, old.gradient, H)
