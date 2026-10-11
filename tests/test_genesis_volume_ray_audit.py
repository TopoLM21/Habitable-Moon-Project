from fractions import Fraction as F
from copy import deepcopy
import pytest
from analysis.genesis_volume_ray_validation.experiment import Experiment,fixture,EVENTS
from analysis.genesis_volume_ray_validation.audit import audit_ray
from tectonics.genesis_volume_event_partition import ratio
from tectonics.genesis_volume_ray_roots import restrict_ray


def record():
    experiment=Experiment(fixture());a,b,bands,cert=experiment.planner.ray(0,F(1,2))
    row=dict(start=list(map(ratio,a)),end=list(map(ratio,b)),
        polynomials={k:list(map(ratio,restrict_ray(experiment.planner.events[k],a,b))) for k in EVENTS},
        isolated={k:dict(identically_zero=v['identically_zero'],roots=[[ratio(lo),ratio(hi)] for lo,hi in v['roots']]) for k,v in cert['events'].items()},
        intervals=[dict(lo=ratio(lo),hi=ratio(hi),kind=kind,integrated=(lo,hi,kind) in bands) for lo,hi,kind in cert['intervals']])
    return row


def test_independent_quadratic_auditor_accepts_exact_partition():
    roots,checks=audit_ray(fixture(),record())
    assert roots>=2 and checks>0


def test_independent_auditor_rejects_a_missing_root():
    row=record();row['isolated']['Delta']['roots']=[]
    with pytest.raises(AssertionError):audit_ray(fixture(),row)


def test_independent_auditor_rejects_a_discarded_root_box():
    row=record();next(r for r in row['intervals'] if r['kind']=='root_box')['integrated']=False
    with pytest.raises(AssertionError):audit_ray(fixture(),row)
