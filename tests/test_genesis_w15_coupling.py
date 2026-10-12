from fractions import Fraction as F
import pytest
from analysis.genesis_w15_coupling_validation.core import bounds,contract,observed,build

BOX=((F(0),F(1)),)*3


@pytest.mark.parametrize('gamma, expected', [((1,2),(1,2)),((-2,-1),(-2,-1)),((-1,1),(-2,2))])
def test_signed_gamma(gamma,expected):
    r={'sphere':(-2,-1),'gamma':gamma,'b_0':(F(2),F(4))}
    assert bounds(r,BOX)['b_0']==expected


def test_uncertain_sphere_keeps_zero_and_negative_part():
    r={'sphere':(-1,1),'gamma':(-2,-1),'b_0':(F(2),F(4))}
    assert bounds(r,BOX)['b_0']==(-2,0)
    r['sphere']=(1,2)
    assert bounds(r,BOX)['b_0']==(0,0)


def test_negative_q_reverses_interval_and_zeros_do_not_inflate():
    assert contract([(F(1),F(3))]+[(F(0),F(0))]*17, [F(-2)]+[F(1)]*17)==(-6,-2)


def test_sign_is_separate_from_accuracy():
    assert observed((F(1),F(100)))['sign']=='positive'
    assert not observed((F(1),F(100)))['relative_2e_minus_6_certified']
    assert observed((F(-1),F(1)))['sign']=='unresolved'


def test_wrong_region_refused_before_graph_construction():
    with pytest.raises(ValueError,match='100 m'):
        build({'seed_radius_m':4.024215510221975},{},(0,0,0))
