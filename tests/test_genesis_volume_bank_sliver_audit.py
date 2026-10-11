from fractions import Fraction as F
import numpy as np
import pytest
from analysis.genesis_volume_banks_validation.sliver_audit import endpoint_node_upper,weight_minimum
from tectonics.genesis_volume_event_partition import split_cell
from tectonics.genesis_volume_source_coverage import exact_node
from tectonics.genesis_volume_rational_fields import fraction


@pytest.mark.parametrize('values',[(F(1,2**50),F(-1),F(-2)),
    (F(1,2**50),F(1,2**51),F(-1)),(F(1,2**50),F(),F(-1))])
def test_midpoint_lattice_bound_contains_every_actual_node_through_depth_three(values):
    w=weight_minimum((3,5));bound=endpoint_node_upper(values,3,w)
    assert bound<0 and max(values)>0
    cells=[tuple(tuple(F(int(i==j)) for j in range(3)) for i in range(3))]
    for depth in range(4):
        for cell in cells:
            for order in (3,5):
                nodes=tuple(map(fraction,(np.polynomial.legendre.leggauss(order)[0]+1)/2))
                for s in nodes:
                    for t in nodes:
                        n=exact_node(cell,s,t)
                        assert sum(x*y for x,y in zip(values,n))<=bound
        cells=[child for c in cells for child in split_cell(c)]


def test_a_resolvable_positive_corner_is_not_declared_unreachable():
    values=(F(1,10),F(-1),F(-1))
    assert endpoint_node_upper(values,24,weight_minimum((3,5)))>0
    s=fraction((np.polynomial.legendre.leggauss(5)[0][0]+1)/2)
    assert values[0]*(1-s)+values[1]*s>0


def test_non_crossing_endpoint_does_not_claim_the_crossing_lattice_proof():
    assert endpoint_node_upper((F(1),F(2),F(3)),24,F(1,1000)) is None
    assert endpoint_node_upper((F(-1),F(-2),F(-3)),24,F(1,1000)) is None
