from fractions import Fraction as F
from copy import deepcopy
import pytest
from analysis.genesis_regional_moment_validation.moments import enclose,coefficients
from analysis.genesis_regional_moment_validation.audit import audit,recover

BOX=((F(),F(1)),)*3
C=(0,0,0)

def data(q,g,k):return dict(sphere=q,gamma=g,density_without_H=k)


def test_inverse_transform_nonunit_signed_box():
    p={C:F(3,7),(2,1,0):F(-5,3),(0,0,3):F(2)}
    box=((F(-2),F(-1)),(F(1,3),F(2,3)),(F(2),F(4)))
    assert recover(coefficients(p,box),box)==p


def test_inside_exact_signed_cancellation():
    p=data({C:F(-1)},{C:F(1)},{C:F(-1),(1,0,0):F(2)})
    r=enclose(p,BOX);assert list(map(F,r['bounds']))==[0,0];audit(p,r)


def test_sphere_cut_negative_density_keeps_work():
    # x² <= 1/4 occupies x in [0,1/2]; H=+1/2, K=-2x.
    p=data({C:F(-1,4),(2,0,0):F(1)},{C:F(1)},{(1,0,0):F(-2)})
    r=enclose(p,BOX);lo,hi=map(F,r['bounds']);assert (lo,hi)==(F(-1,2),0)
    assert lo<=F(-1,8)<=hi;audit(p,r)


def test_joint_sphere_gamma_and_signed_density():
    # x<=3/4, Gamma=x-1/2, K=x-1/2; W=1/2 integral_0^.75 |x-.5| dx=5/64.
    p=data({C:F(-9,16),(2,0,0):F(1)},{C:F(-1,2),(1,0,0):F(1)}, {C:F(-1,2),(1,0,0):F(1)})
    r=enclose(p,BOX);lo,hi=map(F,r['bounds']);assert lo<=F(5,64)<=hi;audit(p,r)
    assert hi-lo==F(1,2)  # Includes unresolved geometry; no invented sign.


def test_partition_preserves_analytic_work():
    p=data({C:F(-1)},{C:F(1)},{C:F(-1),(1,0,0):F(3)})
    parent=enclose(p,BOX);children=[]
    for bounds in [(F(),F(1,2)),(F(1,2),F(1))]:
        r=enclose(p,(bounds,*BOX[1:]));audit(p,r);children.append(r)
    for end in (0,1):assert sum(F(r['bounds'][end]) for r in children)==F(parent['bounds'][end])==F(1,4)


def test_audit_rejects_coefficients_and_narrowed_bounds():
    p=data({C:F(-1)},{C:F(1)},{C:F(2)});r=enclose(p,BOX)
    bad=deepcopy(r);bad['polynomials']['density_without_H']['coefficients'][0]='3'
    with pytest.raises(AssertionError):audit(p,bad)
    bad=deepcopy(r);bad['bounds']=['0','0']
    with pytest.raises(AssertionError):audit(p,bad)


def test_gamma_identically_zero_refused():
    with pytest.raises(ValueError):enclose(data({C:F(-1)},{},{C:F(1)}),BOX)


def test_de_casteljau_preserves_polynomial_on_both_child_boxes():
    from analysis.genesis_regional_moment_validation.restrict import halve
    p={C:F(-3,7),(2,1,0):F(4),(0,0,3):F(-2)}
    cert=coefficients(p,BOX)
    for axis in range(3):
        for side,child in enumerate(halve(cert,axis)):
            box=list(BOX);box[axis]=(F(side,2),F(side+1,2))
            assert recover(child,box)==p


def test_bernstein_basis_envelope_beats_range_times_volume():
    # K=x^8>=0; unknown spherical cut. Integral envelope is 1/9, max is 1.
    p=data({C:F(-1,4),(2,0,0):F(1)},{C:F(1)},{(8,0,0):F(1)})
    r=enclose(p,BOX);audit(p,r)
    assert tuple(map(F,r['bounds']))==(0,F(1,18))


def test_independent_half_formula_and_bad_coefficient_digest():
    from analysis.genesis_regional_moment_validation.restrict import halve
    from analysis.genesis_regional_moment_validation.regional_audit import restricted,verify_bounds
    from analysis.genesis_regional_moment_validation.regional import digest
    p={C:F(1),(2,1,2):F(-2)};cert=coefficients(p,BOX)
    for axis in range(3):
        for side,child in enumerate(halve(cert,axis)):assert child==restricted(cert,axis,side)
    record=enclose(data({C:F(-1)},{C:F(1)},p),BOX)
    record['coefficient_sha256']=digest(record['polynomials']);verify_bounds(record,record['polynomials'])
    record['coefficient_sha256']='0'*64
    with pytest.raises(AssertionError):verify_bounds(record,record['polynomials'])
