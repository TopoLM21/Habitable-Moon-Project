from fractions import Fraction as F
from copy import deepcopy
import pytest
from analysis.genesis_volume_bernstein_validation.control import (
    coefficients,constant,add,scale,mul,integral,enclose,run,
)
from analysis.genesis_volume_bernstein_validation.audit import audit,bernstein_value

BOX=((F(),F(1)),)*3


def raw():
    return dict(section=dict(triangle_m=[[0,0,10],[4,0,10],[0,4,10]],slope_P_m=[[0]*3]*3,
        displacement_A_m=[[0]*3]*3,lambda_bounds=[.8,1.],normal_xyz=[0,0,1],offset_m=9.25),
        seed_center_world_m=[1,1,9],seed_radius_m=.25)


def test_affine_bernstein_identity_exact_without_float_rounding():
    p={(2,1,0):F(7,13),(0,0,2):F(-1,17),(0,0,0):F(1,10**60)}
    box=((F(1,11),F(2,9)),(F(-1,3),F(1,5)),(F(2),F(7,3)))
    d,c=coefficients(p,box);cert=dict(degree=d,coefficients=list(map(str,c.values())))
    from analysis.genesis_volume_bernstein_validation.control import evaluate
    u=(F(2,7),F(3,8),F(4,9));x=[lo+(hi-lo)*v for v,(lo,hi) in zip(u,box)]
    assert bernstein_value(cert,u)==evaluate(p,x)


def test_whole_inside_exact_polynomial_integral():
    p={(1,0,2):F(7,3)};r=enclose(dict(Q=constant(-1),G=constant(1),K=p),BOX)
    assert list(map(F,r['bounds'][1]))==[F(7,18)]*2
    assert list(map(F,r['bounds'][0]))==[0,0]


def test_positive_region_between_all_gauss_nodes_is_retained():
    # Extremely narrow slab near zero: all order-3/5 nodes lie outside it.
    x={(1,0,0):F(1)};lo=F(1,2**50);hi=lo+F(1,2**60)
    Q=mul(add(x,constant(-lo)),add(x,constant(-hi)))
    r=enclose(dict(Q=Q,G=constant(1),K=constant(1)),BOX)
    lower,upper=map(F,r['bounds'][1]);assert lower<=hi-lo<=upper and upper>0
    assert r['states'][1]=='unknown'


def test_polynomial_hidden_from_both_gauss_rules_has_positive_exact_integral():
    # Only literal node generation; no physical radial evaluator.
    import numpy as np
    nodes=set(F(float(x)) for n in (3,5) for x in (np.polynomial.legendre.leggauss(n)[0]+1)/2)
    p=constant(1);x={(1,0,0):F(1)}
    for node in nodes:p=mul(p,mul(add(x,constant(-node)),add(x,constant(-node))))
    r=enclose(dict(Q=constant(-1),G=constant(1),K=p),BOX)
    lo,hi=map(F,r['bounds'][1]);assert lo==hi==integral(p,BOX)>0


def test_unknown_signed_density_preserves_negative_contribution():
    x={(1,0,0):F(1)}
    r=enclose(dict(Q=add(x,constant(-F(1,2))),G=constant(1),K=constant(-2)),BOX)
    lo,hi=map(F,r['bounds'][1]);assert lo<=-1<=hi and lo<0


def test_identically_zero_gamma_refused():
    with pytest.raises(ValueError,match='ownership'):
        enclose(dict(Q=constant(-1),G={},K=constant(1)),BOX)


def test_double_root_contact_is_not_classified_as_empty_by_samples():
    x=add({(1,0,0):F(1)},constant(-F(1,3)))
    r=enclose(dict(Q=mul(x,x),G=constant(1),K=constant(1)),BOX)
    assert r['states'][1]=='unknown' and F(r['bounds'][1][1])>0


def test_small_budget_and_depth_produce_honest_intervals_and_independent_cover():
    r=run(raw(),[F(),F()],[F(),F()],max_blocks=17,max_depth=2)
    assert not r['certificate_gate_passed'] and r['processed_blocks']<=17 and r['maximum_depth']<=2
    a=audit(r);assert a['exact_cover'] and a['cost_verified']
    assert list(map(F,r['integral_bounds'][1]))==[0,0]  # Exact tangent separation.
    assert F(r['integral_bounds'][0][1])>0


@pytest.mark.parametrize('change',['coefficient','bound','cover','cost'])
def test_independent_audit_rejects_tampering(change):
    r=run(raw(),[F(),F()],[F(),F()],max_blocks=5,max_depth=1);r=deepcopy(r)
    if change=='coefficient':r['records'][0]['polynomials']['Q']['coefficients'][0]='-999'
    if change=='bound':r['records'][0]['bounds'][0][1]='0'
    if change=='cover':r['leaves']=r['leaves'][1:]
    if change=='cost':r['processed_blocks']+=1
    with pytest.raises(AssertionError):audit(r)


def test_caps_cannot_be_raised():
    with pytest.raises(ValueError):run(raw(),[0,0],[0,0],max_blocks=2001)
