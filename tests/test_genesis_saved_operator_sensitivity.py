from fractions import Fraction as F
import pytest
from analysis.genesis_saved_operator_sensitivity.compare import error,sign,digest
from analysis.genesis_saved_operator_sensitivity.audit import audit_rows


def row(value,lo,hi):
    return dict(name='synthetic',reference=[str(lo),str(hi)],old_value=str(value),corrected_value=str(value),
                old_error=error(value,lo,hi),corrected_error=error(value,lo,hi))


def test_error_bounds_preserve_sign_and_zero_crossing():
    r=row(F(3),F(1),F(2));assert audit_rows([r])==1
    assert r['old_error']['signed']==['1','2']
    r=row(F(0),F(-1),F(1));assert audit_rows([r])==1
    assert r['old_error']['absolute_lower']=='0' and not r['old_error']['passes_selected_radial_2e6']


def test_exact_zero_has_no_floor():
    r=row(F(),F(),F());assert audit_rows([r])==1
    assert r['old_error']['own_relative_allowance']=='0' and r['old_error']['passes_selected_radial_2e6']
    assert sign(F(-1),F(1))=='unresolved'


def test_auditor_rejects_understated_error():
    r=row(F(3),F(1),F(2));r['old_error']['absolute_upper']='1'
    with pytest.raises(AssertionError):audit_rows([r])


def test_input_binding_changes_with_probe_or_source():
    base=dict(raw=dict(radius=100.),N=[F(1,3)],probes=[dict(q=[.001])])
    assert digest(base)!=digest({**base,'probes':[dict(q=[.002])]})
    assert digest(base)!=digest({**base,'raw':dict(radius=101.)})


def test_component_limit_is_enforced():
    rows=[{**row(F(1),F(1),F(1)),'name':str(i)} for i in range(101)]
    with pytest.raises(AssertionError):audit_rows(rows)
