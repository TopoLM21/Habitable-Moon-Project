from fractions import Fraction as F
import numpy as np
import pytest
from analysis.genesis_volume_directional_validation.experiment import DirectionalExperiment,indicators,split,Cell,fixture,BASE,MIXED
from analysis.genesis_volume_ray_validation.experiment import gauss
from analysis.genesis_volume_ray_enclosure_validation.experiment import EnclosureExperiment
from analysis.genesis_volume_directional_validation.audit import audit_cost
from tectonics.genesis_volume_event_partition import ratio


@pytest.mark.parametrize('axis',['s','t'])
def test_mixed_rules_identify_actual_degree_eight_error_direction(axis):
    q={}
    for ns,nt in BASE+MIXED:
        value=sum(sw*tw*(s**8 if axis=='s' else t**8) for s,sw in gauss(ns) for t,tw in gauss(nt))
        q[ns,nt]=np.array([float(value)],np.longdouble)
    errors=indicators(q)
    assert errors[axis][0]>1000*errors['t' if axis=='s' else 's'][0]
    assert abs(q[5,5][0]-np.longdouble(1)/9)<1e-15
    assert abs(q[3,3][0]-np.longdouble(1)/9)>1e-5


def test_mixed_diagnosis_does_not_hide_opposite_direction_cancellation():
    q={}
    for ns,nt in BASE+MIXED:
        value=sum(sw*tw*(s**8-t**8) for s,sw in gauss(ns) for t,tw in gauss(nt))
        q[ns,nt]=np.array([float(value)],np.longdouble)
    assert q[3,3][0]==q[5,5][0]==0
    assert indicators(q)['s'][0]>1e-5 and indicators(q)['t'][0]>1e-5


@pytest.mark.parametrize('axis',['s','t'])
def test_directional_children_keep_parameter_measure_and_full_root_uncertainty(axis):
    experiment=DirectionalExperiment(fixture());parent=Cell(1,F(1,4),F(3,4),F(1,8),F(7,8),3)
    children=split(parent,axis)
    assert len(children)==2 and all(c.depth==4 for c in children)
    assert sum((c.s1-c.s0)*(c.t1-c.t0) for c in children)==(parent.s1-parent.s0)*(parent.t1-parent.t0)
    assert all(sum(experiment.root_upper(c)[i] for c in children)==experiment.root_upper(parent)[i] for i in (0,1))


def test_cost_includes_mixed_rules_and_only_exact_identical_nodes_are_reused():
    experiment=DirectionalExperiment(fixture());cell=Cell(0)
    base=experiment.cost([cell],BASE);total=experiment.cost([cell])
    assert total>base>0
    experiment.compute(cell,BASE);assert experiment.nodes==base
    assert experiment.cost([cell],BASE)==0
    remaining=experiment.cost([cell],MIXED);assert remaining==total-base
    experiment.compute(cell,MIXED);assert experiment.nodes==total
    assert experiment.mixed_new_nodes==remaining>0 and experiment.requests>experiment.nodes
    assert all(F(*v.as_integer_ratio())>=u for v,u in zip(cell.source_error,experiment.root_upper(cell)))


def test_exact_node_cache_preserves_old_coarse_fine_values_and_enclosures():
    prior=EnclosureExperiment(fixture());old=prior.compute(Cell(0))
    current=DirectionalExperiment(fixture());new=current.compute(Cell(0),BASE)
    assert current.nodes<prior.nodes
    for name in ('coarse','fine','source_error','observed'):
        assert np.array_equal(getattr(old,name),getattr(new,name))
    for name in ('coarse_enclosure','fine_enclosure'):
        assert np.array_equal(getattr(old,name).lower,getattr(new,name).lower)
        assert np.array_equal(getattr(old,name).upper,getattr(new,name).upper)


def test_independent_work_replay_rejects_unreported_diagnostic_evaluations():
    experiment=DirectionalExperiment(fixture());cell=experiment.compute(Cell(0))
    report=dict(mode='diagnosis',cell_records=[dict(triangle=0,s0=ratio(cell.s0),s1=ratio(cell.s1),
        t0=ratio(cell.t0),t1=ratio(cell.t1),diagnostic=True)],radial_line_evaluations=experiment.nodes,
        quadrature_node_requests=experiment.requests,exact_node_cache_hits=experiment.requests-experiment.nodes,
        mixed_rule_new_evaluations=experiment.mixed_new_nodes,scalar_bank_fiber_calls=2*experiment.nodes)
    rays=[]
    for (index,t),(a,b,bands,cert) in experiment.planner.cache.items():
        rays.append(dict(triangle=index,t=ratio(t),start=list(map(ratio,a)),end=list(map(ratio,b)),
            intervals=[dict(lo=ratio(lo),hi=ratio(hi),kind=kind,integrated=(lo,hi,kind) in bands) for lo,hi,kind in cert['intervals']]))
    assert audit_cost(report,rays)['all_mixed_work_charged']
    report['radial_line_evaluations']-=1
    with pytest.raises(AssertionError):audit_cost(report,rays)
