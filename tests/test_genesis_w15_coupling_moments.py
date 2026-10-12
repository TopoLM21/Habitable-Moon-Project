from fractions import Fraction as F
import pytest
from analysis.genesis_w15_coupling_moments.run import integral,intersection
from analysis.genesis_regional_moment_validation.moments import coefficients


def test_zero_is_valid_exact_moment_not_a_failed_sign():
    assert integral({(1,0,0):F(3)},[(-F(1),F(1)),(F(0),F(2)),(F(0),F(1))])==0


def test_signed_mixed_polynomial_matches_bernstein_mean():
    p={(2,1,0):F(7,3),(0,0,1):F(-5),(0,0,0):F(-1,7)}
    box=[(F(-1),F(2)),(F(1,3),F(2,3)),(F(2),F(3))]
    cert=coefficients(p,box)
    assert integral(p,box)==sum(map(F,cert['coefficients']))/len(cert['coefficients'])


def test_intersection_retains_negative_part_and_refuses_inconsistency():
    assert intersection((F(-3),F(5)),(F(-1),F(2)))==(-1,2)
    with pytest.raises(ValueError):intersection((F(0),F(1)),(F(2),F(3)))
