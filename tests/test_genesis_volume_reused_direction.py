from fractions import Fraction as F
import numpy as np
import pytest
from analysis.genesis_volume_reused_direction_validation.experiment import ReusedDirectionExperiment,fourth_stencil,tensor_indicators,Cell,BASE,fixture
from analysis.genesis_volume_ray_validation.experiment import gauss


@pytest.mark.parametrize('axis',['s','t'])
def test_existing_grid_identifies_single_direction_variation(axis):
    nodes,weights=zip(*gauss(5));stencil=fourth_stencil(nodes)
    grid=np.array([[[float(s**8 if axis=='s' else t**8)] for s in nodes] for t in nodes],np.longdouble)
    d=tensor_indicators(grid,stencil,weights)
    assert d[axis][0]>1000*d['t' if axis=='s' else 's'][0]


def test_indicator_only_reads_previously_evaluated_nodes():
    e=ReusedDirectionExperiment(fixture());c=e.compute(Cell(0),BASE)
    before=(e.nodes,e.requests,e.mixed_new_nodes,len(e.samples),len(e.planner.cache))
    first=e.existing_indicators(c);second=e.existing_indicators(c)
    assert before==(e.nodes,e.requests,e.mixed_new_nodes,len(e.samples),len(e.planner.cache))
    assert all(np.array_equal(first[0][k],second[0][k]) for k in ('s','t'))
    assert all(np.isfinite(first[0][k]).all() for k in ('s','t'))


def test_fourth_difference_is_not_an_error_certificate():
    # A quartic is integrated by both original Gauss rules. Its curvature
    # can nevertheless dominate this heuristic: never replace Q55-Q33.
    nodes,weights=zip(*gauss(5));stencil=fourth_stencil(nodes)
    grid=np.array([[[float(s**4+F(1,10000)*t**8)] for s in nodes] for t in nodes],np.longdouble)
    d=tensor_indicators(grid,stencil,weights)
    assert d['s'][0]>d['t'][0]
    s_error=abs(sum(w*x**4 for x,w in gauss(5))-sum(w*x**4 for x,w in gauss(3)))
    t_error=abs(sum(w*x**8 for x,w in gauss(5))-sum(w*x**8 for x,w in gauss(3)))/10000
    assert s_error<F(1,10**14) and t_error>1000*s_error


def test_five_node_data_can_hide_a_real_coarse_rule_error():
    nodes,_=zip(*gauss(5))
    def hidden(x):
        result=F(1)
        for n in nodes:result*=(x-n)**2
        return result
    assert all(hidden(n)==0 for n in nodes)
    assert sum(w*hidden(x) for x,w in gauss(3))>0
    # The unchanged 3x3 rule detects what any 5x5-only indicator cannot see.
