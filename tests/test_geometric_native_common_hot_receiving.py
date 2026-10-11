from dataclasses import replace
from fractions import Fraction as F
from functools import lru_cache
import hashlib,json
from pathlib import Path
import pytest

from tectonics import geometric_native_common_hot_receiving as api
I=api.I
ROOT=Path(__file__).resolve().parents[1]

@lru_cache(maxsize=1)
def packet():
    p=ROOT/'analysis/geometric_native_common_hot_mean_validation/release_validation_02/native_control/source_packet.json'
    raw=p.read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='0f8a0ee8ae082e3434b136bab7515944354eaf059db95965d0e87259f6791035'
    return json.loads(raw)

@lru_cache(maxsize=1)
def point():return api.evaluate_native_hot_receiving_point(packet(),packet()['evaluation_queries'][1]['label'],0)

def fixture():
    X=(I(2),I(3),I(4));A=tuple(tuple(I(int(i==j)) for j in range(3)) for i in range(3))
    J=tuple(tuple(I(int(i==j)) for j in range(6)) for i in range(6))
    hs=tuple(tuple(I(2 if i==j==1 else 0) for j in range(6)) for i in range(6))
    return api._field_twojet(X,A,F(10),I(0),(F(0),)*5,J,hs,96)

def test_chart_curvature_enters_positions_and_original_material_gradient():
    value,first,second=fixture()
    assert value[:3]==(I(2),I(3),I(4))
    for interval,exact in ((second[0][4][4],F(2,5)),(second[3][4][4],F(1,5)),
            (second[7][4][4],F(1,5)),(second[11][4][4],F(1,5))):
        assert interval.lo<=exact<=interval.hi

def test_fresh_owner_body_cross_terms_use_same_left_rotation():
    value,first,second=fixture()
    assert first[1][0]==I(-4) and first[2][0]==I(3)
    assert second[1][0][3].lo<=F(-2,5)<=second[1][0][3].hi
    assert second[1][3][0]==second[1][0][3]
    assert second[1][0][0]==I(-3)

def test_all_five_authentic_hot_bodies_all_four_saved_charts_and_all_fifteen_labels():
    p=packet();components=set();depths=set()
    for query in p['evaluation_queries']:
        for index in range(15):
            q=api.evaluate_native_hot_receiving_point(p,query['label'],index)
            assert len(q.spatial_values)==12 and all(len(row)==9 for row in q.spatial_first)
            assert all(len(matrix)==9 and all(len(row)==9 for row in matrix) for matrix in q.spatial_second)
            assert q.positive_material_jacobian_bounds.lo>0
            assert q.scope==api.SCOPE and q.finite_mean_chart_curvature_retained
            assert not q.physical_receiving_traction_law_defined and not q.complete_curved_receiving_face_covered
            assert not q.clocks_advanced and q.accepted_new_native_intervals==0
            assert q.active_coordinate_indices[:3]==tuple(range(3*q.owner,3*q.owner+3))
            components.add(q.component_id);depths.add(json.loads(q.source_point_descriptor_json)['label'])
    assert len(components)==5 and depths=={'original_roof','original_interior','original_true_base'}

def test_declared_current_field_matches_saved_material_label_position_enclosure():
    p=packet()
    # Saved903 material labels were evaluated at the full30 diagnostic query.
    for index,saved in enumerate(p['material_point_queries']):
        q=api.evaluate_native_hot_receiving_point(p,p['evaluation_queries'][1]['label'],index)
        for a,b in zip(q.spatial_values[:3],saved['deformed_position_bounds_m']):
            b=api._interval(b);assert a.lo<=b.hi and b.lo<=a.hi

def test_full_work_pullback_retains_spatial_cross_tangent_and_prestress_curvature():
    q=point();gradient=(I(1),)*12
    tangent=tuple(tuple(I(1 if i==j else F(1,10)) for j in range(12)) for i in range(12))
    result=api.pullback_native_hot_receiving_work(q,gradient,tangent)
    for a in range(9):
        expected=sum((q.spatial_first[i][a] for i in range(12)),I(0))
        assert result['gradient_bounds'][a].lo<=expected.lo<=expected.hi<=result['gradient_bounds'][a].hi
        for b in range(9):
            expected=sum((q.spatial_second[i][a][b] for i in range(12)),I(0))+sum((q.spatial_first[i][a]*tangent[i][j]*q.spatial_first[j][b] for i in range(12) for j in range(12)),I(0))
            actual=result['hessian_bounds'][a][b]
            assert actual.lo<=expected.lo<=expected.hi<=actual.hi
    assert result['orientation_hyperstress_work_included'] and result['force_times_chart_curvature_included']
    assert not result['physical_receiving_traction_law_inferred']

def test_live_binding_rejects_changed_material_state_and_geometry():
    q=point()
    with pytest.raises(ValueError):api.native_hot_receiving_point_to_dict(replace(q,owner=1-q.owner))
    with pytest.raises(ValueError):api.native_hot_receiving_point_to_dict(replace(q,spatial_values=(I(0),)*12))
    with pytest.raises(ValueError):api.native_hot_receiving_point_to_dict(replace(q,physical_receiving_traction_law_defined=True))

def test_complete_point_restart_requires_exact_original_query_and_label():
    p=packet();q=point();raw=api.native_hot_receiving_point_to_dict(q)
    restored=api.native_hot_receiving_point_from_dict(raw,p,query_label=q.query_label,point_index=0)
    assert api.native_hot_receiving_point_to_dict(restored)==raw
    with pytest.raises(ValueError):api.native_hot_receiving_point_from_dict(raw,p,query_label=q.query_label,point_index=1)

def test_resealed_packet_cannot_create_unmatched_field_or_history():
    p=packet();altered=dict(p,clocks_advanced=True)
    altered['packet_sha256']=api._sha({k:v for k,v in altered.items() if k!='packet_sha256'})
    with pytest.raises(ValueError):api.evaluate_native_hot_receiving_point(altered,p['evaluation_queries'][0]['label'],0)

def test_literal_kinds_and_unsupported_body_states_are_rejected():
    p=packet();label=p['evaluation_queries'][0]['label']
    for index in (True,F(0),-1,15):
        with pytest.raises(ValueError):api.evaluate_native_hot_receiving_point(p,label,index)
    with pytest.raises(ValueError):api.evaluate_native_hot_receiving_point(p,'arbitrary new equilibrium',0)
    with pytest.raises(ValueError):api.evaluate_native_hot_receiving_point(p,label,0,precision=True)
    with pytest.raises(ValueError):api.pullback_native_hot_receiving_work(point(),(I(0),)*11,tuple())

def test_hyperstress_only_work_includes_gradient_chart_second_variations():
    q=point();force=(I(0),)*3+(I(1),)*9;zero=tuple((I(0),)*12 for _ in range(12))
    result=api.pullback_native_hot_receiving_work(q,force,zero)
    for a in range(9):
        expected=sum((q.spatial_first[i][a] for i in range(3,12)),I(0))
        assert result['gradient_bounds'][a].lo<=expected.lo<=expected.hi<=result['gradient_bounds'][a].hi
    assert result['gradient_units']==('N*m',)*3+('N',)*6
