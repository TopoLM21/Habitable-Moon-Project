from fractions import Fraction as F
import pytest
from analysis.genesis_regional_work_validation.expression import Graph,interval_evaluate,block_bound
from analysis.genesis_regional_work_validation.audit import evaluate


def test_symbolic_derivative_exact_product_chain():
    g=Graph();x=g.node('x',0);p=g.mul(g.add(x,g.const(2)),g.mul(x,x));d=g.diff(p,0)
    graph=dict(nodes=g.nodes,outputs=dict(derivative=d))
    assert interval_evaluate(graph,((F(3),F(3)),)*3)['derivative']==(F(39),F(39))


def test_interval_graph_preserves_cancellation_uncertainty():
    g=Graph();x=g.node('x',0);y=g.sub(x,x);graph=dict(nodes=g.nodes,outputs=dict(y=y))
    box=((F(-1),F(2)),)*3
    assert interval_evaluate(graph,box)['y']==(-3,3)
    assert evaluate(g.nodes,box)[y]==(-3,3)


@pytest.mark.parametrize('gamma,density,expected',[(1,2,(1,1)),(-1,2,(-1,-1)),(1,-2,(-1,-1))])
def test_exact_signed_full_box_integral(gamma,density,expected):
    g=Graph();outputs=dict(sphere=g.const(-1),gamma=g.const(gamma),density_without_H=g.const(density))
    graph=dict(nodes=g.nodes,outputs=outputs)
    bounds,state,_=block_bound(graph,((F(),F(1)),)*3)
    assert bounds==expected and state=='inside'


def test_unknown_sphere_does_not_drop_negative_work():
    g=Graph();x=g.node('x',0);outputs=dict(sphere=g.sub(x,g.const(F(1,2))),gamma=g.const(1),density_without_H=g.const(-2))
    bounds,state,_=block_bound(dict(nodes=g.nodes,outputs=outputs),((F(),F(1)),)*3)
    assert bounds==(-1,0) and state=='unknown_sphere'


def test_unknown_gamma_encloses_both_signs():
    g=Graph();x=g.node('x',0);outputs=dict(sphere=g.const(-1),gamma=g.sub(x,g.const(F(1,2))),density_without_H=g.const(2))
    bounds,_,_=block_bound(dict(nodes=g.nodes,outputs=outputs),((F(),F(1)),)*3)
    assert bounds==(-1,1)


def test_independent_replay_refuses_forward_reference():
    with pytest.raises(AssertionError):evaluate([('+',0,1)],((F(),F(1)),)*3)


def test_symbolic_identity_rejects_a_changed_coefficient():
    from analysis.genesis_regional_work_validation.identity import check
    g=Graph();x=g.node('x',0);outputs=dict(value=g.mul(x,x))
    left=dict(nodes=list(g.nodes),outputs=outputs)
    right=dict(nodes=g.nodes+[('c','1'),('+',outputs['value'],len(g.nodes))],outputs=dict(value=len(g.nodes)+1))
    assert check(left,left)['all_output_polynomials_exactly_equal']
    with pytest.raises(AssertionError):check(left,right)


@pytest.mark.parametrize('direction',[-1,1])
def test_real_centered_density_matches_independent_source_at_other_N(direction):
    import json
    from pathlib import Path
    from analysis.genesis_regional_work_validation.run import point_checks
    from analysis.genesis_regional_work_validation.centered_expression import build
    r=json.loads((Path(__file__).resolve().parents[1]/'analysis/genesis_regional_work_validation/source_fixture.json').read_bytes())
    original_root=[tuple(map(F,b)) for b in r['root']]
    r['graph']=build(r['source'],r['probe'],tuple(sum(b)/2 for b in original_root))
    root=[tuple(map(F,b)) for b in r['root']]
    for axis in (0,1):
        lo,hi=root[axis];mid=(lo+hi)/2+direction*(hi-lo)/4
        root[axis]=(mid-(hi-lo)/8,mid+(hi-lo)/8)
    assert len(point_checks(r['source'],r['probe'],r['graph'],root))==3
