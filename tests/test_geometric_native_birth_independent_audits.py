"""Independent oracle identities and refusal of incomplete clock evidence."""
from copy import deepcopy
from decimal import Decimal as D, localcontext
from fractions import Fraction as F

import pytest

from analysis.geometric_native_birth_front_validation.independent_continuum_audit import (
    integrate_product, classify_monotone_field, PRECISION)
from analysis.geometric_native_birth_front_validation.independent_pair_polynomial_audit import audit_vertex


def test_independent_angular_oracle_trigonometric_unit_identity():
    with localcontext() as context:
        context.prec=PRECISION
        parts=[(D("0.03125"),D("0.125"))]
        sin2=integrate_product((D(0),D(0),D(1)),(D(0),D(0),D(1)),parts)
        cos2=integrate_product((D(0),D(1),D(0)),(D(0),D(1),D(0)),parts)
        assert abs(sin2+cos2-D("0.09375"))<D("1e-230")


def test_independent_angular_oracle_retains_microscopic_contact_energy():
    with localcontext() as context:
        context.prec=PRECISION
        energy=integrate_product((D("-1e-120"),D(0),D(0)),
            (D("-1e-120"),D(0),D(0)),[(D(0),D("1e-40"))])/2
        assert energy==D("5e-281")


def test_monotone_contact_refuses_lost_negative_subarc():
    with localcontext() as context:
        context.prec=PRECISION
        with pytest.raises(ValueError,match="closed subarcs"):
            classify_monotone_field((D("-1"),D(0),D("-1")),D(0),D("0.1"),[],[])


def test_monotone_contact_refuses_unproved_positive_pocket():
    with localcontext() as context:
        context.prec=PRECISION
        with pytest.raises(ValueError,match="monotone whole arc"):
            classify_monotone_field((D("0.9"),D("-1"),D("0.05")),D(0),D("0.1"),[],[])


def _i(value):
    value=F(value)
    pair=[value.numerator,value.denominator]
    return dict(lo=pair,hi=pair)


def _vertex():
    tiny=F(1,10**90)
    return dict(order=12,precision=128,time_cells=1,
        exact_initial_incidence=True,divided_by_birth_clock=True,
        scalar_coefficients=[_i(0),_i(-tiny)]+[_i(0)]*11,
        scalar_remainder_bound=[0,1],
        cell_sign_bounds=[dict(time_fraction=dict(lo=[0,1],hi=[1,1]),sign_bounds=_i(-tiny))])


def test_exact_clock_factor_retains_tiny_nonpositive_scalar():
    proof=audit_vertex(_vertex())
    assert proof["birth_clock_division"] is True
    assert proof["cells"][0]["independently_contracted_upper"]==[-1,10**90]


def test_nonzero_tail_cannot_be_removed_at_exact_birth_incidence():
    vertex=_vertex()
    vertex["scalar_coefficients"]=[_i(0)]*13
    vertex["scalar_remainder_bound"]=[1,10**100]
    vertex["cell_sign_bounds"][0]["sign_bounds"]=_i(0)
    with pytest.raises(ValueError,match="not certified nonpositive"):
        audit_vertex(vertex)


def test_whole_clock_scalar_refuses_an_interior_positive_pocket():
    vertex=_vertex()
    vertex.update(exact_initial_incidence=False,divided_by_birth_clock=False,
        scalar_coefficients=[_i(F(-3,16)),_i(1),_i(-1)]+[_i(0)]*10)
    vertex["cell_sign_bounds"][0]["sign_bounds"]=_i(F(-3,16))
    with pytest.raises(ValueError,match="not certified nonpositive"):
        audit_vertex(vertex)


@pytest.mark.parametrize("change",["clock_gap","extra_coefficient","false_incidence","bool_resource"])
def test_scalar_proof_refuses_incomplete_or_aliased_evidence(change):
    vertex=deepcopy(_vertex())
    if change=="clock_gap":
        vertex["cell_sign_bounds"][0]["time_fraction"]["lo"]=[1,100]
    elif change=="extra_coefficient":
        vertex["scalar_coefficients"].append(_i(0))
    elif change=="false_incidence":
        vertex["scalar_coefficients"][0]=_i(F(1,10**100))
    else:
        vertex["time_cells"]=True
    with pytest.raises(ValueError):
        audit_vertex(vertex)
