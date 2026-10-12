from fractions import Fraction as F
from itertools import product
import pytest
from analysis.genesis_gamma_strip_validation.strips import partition,root_window
from analysis.genesis_regional_moment_validation.moments import enclose
from analysis.genesis_regional_moment_validation.audit import audit

C=(0,0,0);BOX=((F(),F(1)),)*3


@pytest.mark.parametrize('density_sign',[-1,1])
def test_symmetric_joint_integral_across_exact_plane(density_sign):
    gamma={C:F(-1,2),(1,0,0):F(1)};p=dict(sphere={C:F(-1)},gamma=gamma,density_without_H={k:density_sign*v for k,v in gamma.items()})
    plan=partition(gamma,BOX);totals=[F(),F()]
    for q in plan['quarters']:
        for b in q['slabs']:
            r=enclose(p,[tuple(map(F,v)) for v in b]);audit(p,r)
            for e in (0,1):totals[e]+=F(r['bounds'][e])
    assert totals==[density_sign*F(1,8)]*2


def test_tilted_multiaffine_gamma_root_window():
    # Gamma=x(1+y/2)-1/4-z/8, positive derivative throughout.
    g={(1,0,0):F(1),(1,1,0):F(1,2),C:F(-1,4),(0,0,1):F(-1,8)}
    r=root_window(g,BOX,0)
    assert list(map(F,r['normalized_root_bounds']))==[F(1,6),F(3,8)]
    assert r['slope_sign']==1


def test_gamma_with_zero_derivative_returns_no_axis():
    assert partition({C:F(1)},BOX)['axis'] is None


def test_exact_quarter_and_strip_cover():
    g={(1,0,0):F(1),(0,1,0):F(1,3),C:F(-1,2)}
    plan=partition(g,BOX);volume=F()
    for q in plan['quarters']:
        axis=plan['axis'];boxes=[[tuple(map(F,v)) for v in b] for b in q['slabs']]
        assert boxes[0][axis][0]==F(q['box'][axis][0]) and boxes[-1][axis][1]==F(q['box'][axis][1])
        for a,b in zip(boxes,boxes[1:]):assert a[axis][1]==b[axis][0]
        for b in boxes:
            v=F(1)
            for lo,hi in b:v*=hi-lo
            volume+=v
    assert volume==1


def test_nonmultiaffine_gamma_refused():
    with pytest.raises(ValueError):partition({(2,0,0):F(1)},BOX)


def test_independent_strip_cover_refuses_gap_and_duplicate_quarter():
    from copy import deepcopy
    from analysis.genesis_gamma_strip_validation.audit import strip_cover
    g={C:F(-1,2),(1,0,0):F(1),(0,1,0):F(1,3)}
    plan=partition(g,BOX);assert strip_cover(plan,[list(b) for b in BOX],g)
    bad=deepcopy(plan);bad['quarters'][0]['slabs'][0][plan['axis']][1]='1/100'
    with pytest.raises(AssertionError):strip_cover(bad,[list(b) for b in BOX],g)
    bad=deepcopy(plan);bad['quarters'][1]=bad['quarters'][0]
    with pytest.raises(AssertionError):strip_cover(bad,[list(b) for b in BOX],g)
