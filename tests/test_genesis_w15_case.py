from fractions import Fraction as F
from copy import deepcopy
from pathlib import Path
import json
import pytest
from analysis.genesis_w15_case_validation.preconditions import verify,distance_to_zero


def inputs():
    root=Path(__file__).resolve().parents[1]
    selection=json.loads((root/'analysis/genesis_w15_probe_selection/preflight.json').read_text())
    fixture=json.loads((root/selection['source_path']).read_text());c=fixture['cases'][0]
    return selection['candidate'],c['supports']['fixed_100m_clipped_support']['original_declared_added_basis'],c['source_inputs']


def test_fixed_candidate_preconditions():
    candidate,B,raw=inputs();r=verify(candidate,B,raw)
    assert r['exact_literal_membership'] and r['kernel_check_passed']


def test_modified_increment_rejected_before_integration():
    c,B,raw=inputs();c=deepcopy(c);c['enrichment_increment_m'][0]+=1e-6
    with pytest.raises(AssertionError):verify(c,B,raw)


@pytest.mark.parametrize('bounds,expected',[(['2','3'],F(2)),(['-3','-2'],F(2)),(['-2','3'],F()),(['0','3'],F())])
def test_sign_symmetric_cauchy_distance(bounds,expected):
    assert distance_to_zero(bounds)==expected
