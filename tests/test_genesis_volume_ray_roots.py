from fractions import Fraction as F
import pytest
from tectonics.genesis_volume_ray_roots import (
    multiply,evaluate,isolate,partition,sturm,count_open,restrict_ray,RootIsolationRefusal,
)
from tectonics.genesis_volume_event_partition import Polynomial


def factors(roots):
    p=(F(1),)
    for r in roots:p=multiply(p,(-r,F(1)))
    return p


@pytest.mark.parametrize('roots',[
    [F(0),F(1)], [F(1,3)]*2, [F(1,7),F(2,5),F(3,5),F(6,7)],
    [F(1,2),F(1,2)+F(1,2**100)], [F(1,2**120),F(1)-F(1,2**120)],
])
def test_all_roots_including_tangencies_endpoints_and_tiny_intervals(roots):
    result=isolate(factors(roots))
    assert result['distinct_roots']==len(set(roots))
    assert all(sum(lo<=r<=hi for lo,hi in result['roots'])==1 for r in set(roots))
    assert all(sum(lo<=r<=hi for r in set(roots))==1 for lo,hi in result['roots'])
    assert all(lo==hi or hi-lo<=min(lo,1-hi)/2**40 for lo,hi in result['roots'])


def test_sturm_open_counts_exclude_exact_endpoint_roots():
    s=sturm(factors([F(0),F(1,2),F(1)]))
    assert count_open(s,F(0),F(1))==1
    assert count_open(s,F(0),F(1,2))==0
    assert count_open(s,F(1,2),F(1))==0


def test_partition_covers_everything_and_keeps_multiple_event_root_boxes():
    ps={'tangent':factors([F(1,3)]*2),'several':factors([F(1,7),F(1,3),F(6,7)]),
        'irrational':(F(-1,2),F(),F(1)),'identity':()}
    result=partition(ps)
    assert result['identically_zero_events']==('identity',)
    intervals=result['intervals']
    assert sum(b-a for a,b,_ in intervals)==1
    assert all(a[1]==b[0] for a,b in zip(intervals,intervals[1:]))
    for a,b,kind in intervals:
        if kind=='root_free':
            for p in ps.values():
                if p:assert count_open(sturm(p),a,b)==0
    assert any(kind=='root_box' for _,_,kind in intervals)


def test_small_positive_island_is_not_declared_empty():
    lo=F(1,2);hi=lo+F(1,2**100)
    p=tuple(-v for v in factors([lo,hi]));result=partition({'event':p})
    middle=(lo+hi)/2
    assert evaluate(p,middle)>0
    assert any(a<middle<b for a,b,_ in result['intervals'])
    assert any(kind=='root_free' and evaluate(p,(a+b)/2)>0 for a,b,kind in result['intervals'])


def test_unresolved_isolation_refuses_instead_of_dropping_roots():
    with pytest.raises(RootIsolationRefusal):isolate((F(-1,2),F(),F(1)),max_depth=2)


def test_ray_restriction_matches_exact_multivariate_polynomial():
    p=Polynomial(3,{(2,1,1):F(7,3),(0,2,0):F(-4),(0,0,0):F(1,9)})
    start=(F(1,3),)*3;end=(F(1,2),F(1,2),F())
    q=restrict_ray(p,start,end)
    for rho in (F(),F(1,7),F(1,2),F(1)):
        n=tuple(a+rho*(b-a) for a,b in zip(start,end))
        assert evaluate(q,rho)==p.evaluate(n)
