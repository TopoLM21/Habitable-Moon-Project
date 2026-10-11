from fractions import Fraction as F
import copy
import numpy as np
import pytest
from test_genesis_volume_source_coverage import simple,angular_fixture
from tectonics.genesis_volume_source_coverage import source_coverage
from analysis.genesis_volume_mapped_validation.coverage_audit import audit
from analysis.genesis_volume_mapped_validation.angular import integrate_angular,CellGeometry,AngularControlRefusal


@pytest.mark.parametrize('translation',[0.,.125,-.375])
def test_independent_minor_and_polygon_audit_for_varying_birth_offsets(translation):
    raw=simple();raw['section']['displacement_A_m']=np.array([[translation,.0625,0],[0,translation,0],[0,0,translation]])
    _,certificate=source_coverage(raw)
    assert audit(raw,certificate)['initial_support_completeness_proof_verified']
    bad=copy.deepcopy(certificate);bad['determinant_coefficients'][0][0]=str(int(bad['determinant_coefficients'][0][0])+1)
    with pytest.raises(ValueError):audit(raw,bad)


def test_exact_empty_outer_region_and_singular_fallback_are_distinct():
    raw=simple();raw['seed_center_world_m']=(100.,100.,100.)
    raw['section']['lambda_bounds']=(.9,1.)
    cells,c=source_coverage(raw)
    assert not cells and not c['singular_interval_retains_full_simplex']
    assert audit(raw,c)['initial_support_completeness_proof_verified']
    raw=simple(np.zeros((3,3)));cells,c=source_coverage(raw)
    assert len(cells)==1 and c['singular_interval_retains_full_simplex']
    assert audit(raw,c)['initial_support_completeness_proof_verified']


def test_initial_partition_depth_counts_toward_original_total_cap():
    report=integrate_angular(angular_fixture(),lambda n:(np.zeros(2),np.zeros(2)),
        ('reference_volume_negative_m3','reference_volume_positive_m3'),max_depth=1)
    assert report['maximum_evaluated_depth']==1
    assert report['status']=='OPEN'
    assert 'maximum angular depth' in report['reason']


def test_event_lane_refuses_loss_of_extra_source_center_bits():
    raw=angular_fixture()
    raw['seed_center_world_m']=np.array([np.longdouble('.2')+np.longdouble(2)**-60,.3,.4],np.longdouble)
    with pytest.raises(AngularControlRefusal,match='binary64'):
        CellGeometry(raw)


@pytest.mark.parametrize('sign',[1,-1])
def test_nonsingular_cramer_enclosure_is_exercised_for_both_signs(sign):
    raw=simple(np.diag([sign,1,1]));raw['section']['lambda_bounds']=(.8,1.)
    raw['seed_center_world_m']=(sign*.2,.3,.4)
    _,c=source_coverage(raw)
    assert not c['singular_interval_retains_full_simplex']
    assert len(c['coordinate_witnesses'])==3
    assert audit(raw,c)['independent_determinant_and_minor_polynomials']


def test_total_reference_volume_upper_enclosure_is_outward():
    report=integrate_angular(angular_fixture(),lambda n:(np.array([0.,7/24]),np.zeros(2)),
        ('reference_volume_negative_m3','reference_volume_positive_m3'))
    # JSON float presentation is not the exact internal interval; compare with
    # one presentation ULP of margin. Internal aggregation is directed.
    upper=report['reference_side_volume_envelope_upper_m3'][1]
    assert upper>=float(F(7,48))
    assert report['reference_side_volume_envelope_upper_m3'][0]==0


def test_prepared_contractions_reject_mismatched_chart_and_source():
    from dataclasses import replace
    from test_genesis_volume_precision_integrals import fixture
    from tectonics.genesis_volume_mapped_integrals import PreparedMappedIntegrals
    raw,source,options,_=fixture()
    runtime=PreparedMappedIntegrals(source,raw,**options)
    n=(F(2,7),F(1,5),F(18,35));chart,geometry=runtime.geometry_at(n)
    with pytest.raises(ValueError,match='chart'):
        runtime.evaluate_prepared(n,replace(chart,lambda_mid=chart.lambda_mid+np.longdouble('.01')),geometry)
    with pytest.raises(ValueError,match='source'):
        runtime.evaluate_prepared(n,chart,replace(geometry,source_binding_sha256='wrong'))
