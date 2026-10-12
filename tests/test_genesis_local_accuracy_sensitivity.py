from fractions import Fraction as F
from pathlib import Path
import json
import numpy as np
import pytest
from analysis.genesis_local_accuracy_sensitivity.study import primitive,oracle,Counter,POINT_CAP,GC,T,K,KC
from analysis.genesis_local_accuracy_sensitivity.audit import check
from tectonics.genesis_volume_creation_law import ProspectiveCreationLaw,evaluate_creation_law


def test_primitive_at_peak_and_complete_separation():
    assert primitive(F(T,K))==F(T*T,2*K)
    assert primitive(F(2*GC,T))==GC


def test_oracle_keeps_negative_unloading_work():
    force,work=oracle(F(-1,1000),F(-1,2000))
    assert force<0 and work<0


@pytest.mark.parametrize('old,new',[(0.,.002),(0.,.0123),(-.002,-.001)])
def test_independent_reference_matches_available_kernel(old,new):
    law=ProspectiveCreationLaw(GC,T,K,KC);r=evaluate_creation_law(law,[0,old,0],[0,new,0],abs(old))
    force,work=oracle(old,new)
    assert float(r.traction_pa[1])==pytest.approx(float(force),rel=2e-14)
    assert float(r.mechanical_work_j_m2)==pytest.approx(float(work),rel=2e-14)


def test_cap_refuses_before_extra_evaluation():
    c=Counter();c.points=POINT_CAP
    with pytest.raises(ValueError,match='cap'):c.evaluate(ProspectiveCreationLaw(GC,T,K,KC),0.,np.array([.001]),0)
    assert c.calls==0


def test_saved_study_matches_analytic_trapezoid_error_without_kernel_calls():
    p=Path(__file__).resolve().parents[1]/'analysis/genesis_local_accuracy_sensitivity/result.json'
    result=check(json.loads(p.read_bytes()))
    assert result['additional_constitutive_kernel_calls']==0
