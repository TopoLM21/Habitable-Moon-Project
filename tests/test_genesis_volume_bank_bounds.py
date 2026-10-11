from fractions import Fraction as F
import numpy as np
import pytest
from tectonics.genesis_volume_bank_bounds import PreparedBankBounds
from tectonics.genesis_volume_event_mesh import event_cut,event_mesh
from tectonics.genesis_volume_event_partition import linear,constant,add,signed_double_area
from analysis.genesis_volume_operand_validation.independent_geometry_bounds import source_line,_source_interval
from tectonics.genesis_volume_rational_fields import fraction


def fixture(radius=100.,deformed=False):
    X=np.array([[0.,0.,10.],[4.,0.,10.],[0.,4.,10.]])
    A=np.array([[.01,0.,.02],[0.,-.01,-.01],[.02,0.,.01]]) if deformed else np.zeros((3,3))
    return dict(section=dict(triangle_m=X,slope_P_m=np.zeros((3,3)),displacement_A_m=A,
        lambda_bounds=[.8,1.],normal_xyz=[0.,0.,1.],offset_m=9.),seed_center_world_m=[1.,1.,9.],seed_radius_m=radius)


def simplex():return tuple(tuple(F(int(i==j)) for j in range(3)) for i in range(3))


def exact_fiber_bounds(raw,n,side):
    line=source_line(raw,np.array([np.longdouble(v.numerator)/np.longdouble(v.denominator) for v in n]))
    t=_source_interval(line,320)
    if t is None:return F(),F()
    lo=list(t['low']);hi=list(t['high']);g,h=side*line['slope'],side*line['intercept']
    if g>0:lo=[max(v,-h/g) for v in lo]
    elif g<0:hi=[min(v,-h/g) for v in hi]
    elif h<=0:return F(),F()
    det=F(160)
    return det*max(F(),hi[0]**3-lo[1]**3)/3,det*max(F(),hi[1]**3-lo[0]**3)/3


def test_full_sphere_horizontal_gamma_gives_exact_bank_measures():
    raw=fixture();g=PreparedBankBounds(raw);c=g.certificate(simplex())
    lo,hi=map(fraction,raw['section']['lambda_bounds']);root=F(9,10)
    expected={-1:F(80)*(root**3-lo**3)/3,1:F(80)*(hi**3-root**3)/3}
    for side in (-1,1):
        a,b=[F(*map(int,v)) for v in c['bank_measure_bounds'][str(side)]]
        assert a==b==expected[side]
    assert sum(expected.values())==F(80)*(hi**3-lo**3)/3


@pytest.mark.parametrize('radius,deformed',[(1.,False),(1.,True),(100.,True)])
def test_whole_cell_bank_bounds_contain_independent_fibers(radius,deformed):
    raw=fixture(radius,deformed);g=PreparedBankBounds(raw)
    cell=((F(1,2),F(1,4),F(1,4)),(F(3,8),F(3,8),F(1,4)),(F(3,8),F(1,4),F(3,8)))
    cert=g.certificate(cell);area=signed_double_area(cell)/2
    for a in range(9):
        for b in range(9-a):
            n=tuple((a*cell[0][j]+b*cell[1][j]+(8-a-b)*cell[2][j])/8 for j in range(3))
            for side in (-1,1):
                lower,upper=[F(*map(int,v)) for v in cert['bank_measure_bounds'][str(side)]]
                true_lo,true_hi=exact_fiber_bounds(raw,n,side)
                assert lower<=area*true_lo<=area*true_hi<=upper


def test_sphere_separated_from_gamma_excludes_only_the_opposite_bank():
    raw=fixture(.1);raw['seed_center_world_m']=[.95,.95,9.5]
    g=PreparedBankBounds(raw)
    cell=((F(9,16),F(7,32),F(7,32)),(F(7,16),F(11,32),F(7,32)),(F(7,16),F(7,32),F(11,32)))
    cert=g.certificate(cell)
    assert F(*map(int,cert['bank_measure_bounds']['-1'][1]))==0
    assert F(*map(int,cert['bank_measure_bounds']['1'][1]))>0
    assert cert['bank_reasons']['-1']=='sphere_Gamma_separation'


@pytest.mark.parametrize('weights',[(-1,1,0),(-1,2,3),(1,-2,-3),(0,0,0)])
def test_event_cuts_preserve_exact_area_and_all_test_points(weights):
    root=simplex();event=linear(tuple(map(F,weights)));children,record=event_cut(root,event)
    assert sum(signed_double_area(c) for c in children)==1
    assert all(signed_double_area(c)>0 for c in children)
    for a in range(17):
        for b in range(17-a):
            n=(F(a,16),F(b,16),1-F(a+b,16))
            assert any(all(signed_double_area((c[i],c[(i+1)%3],n))>=0 for i in range(3)) for c in children)
    for entry in record['edge_parameters']:
        assert F(*map(int,entry['parameter_error']))<=F(1,2**41)


def test_near_vertex_cut_retains_the_entire_unresolved_cell():
    root=simplex();event=linear((F(-1,2**100),F(1),F(1)))
    children,record=event_cut(root,event)
    assert children==(root,) and not record['split']


def test_two_event_mesh_preserves_each_root_measure():
    raw=fixture(deformed=True);roots=(simplex(),);children,c=event_mesh(raw,roots)
    assert sum(signed_double_area(v) for v in children)==1
    assert c['source_cells_removed']==0 and not c['exact_event_alignment']


def test_bank_certificate_rejects_forged_empty_measure():
    import copy
    from tectonics.genesis_volume_bank_bounds import verify_bank_certificate
    raw=fixture();c=PreparedBankBounds(raw).certificate(simplex())
    assert verify_bank_certificate(raw,c)
    altered=copy.deepcopy(c);altered['bank_measure_bounds']['1'][1]=['0','1']
    with pytest.raises(ValueError):verify_bank_certificate(raw,altered)


def test_bank_integrator_retains_unsampled_positive_source_and_exact_measure():
    from analysis.genesis_volume_banks_validation.angular import integrate_angular
    names=('reference_volume_negative_m3','reference_volume_positive_m3')
    r=integrate_angular(fixture(),lambda n:(np.zeros(2),np.zeros(2)),names,max_line_evaluations=136)
    assert r['status']=='OPEN' and r['unresolved_positive_candidate_cells']>0
    assert r['original_own_relative_component_allowances']==[0.,0.]
    cells=r['final_cells']
    area=sum(signed_double_area(tuple(tuple(F(*map(int,q)) for q in row) for row in c['vertices'])) for c in cells)
    assert area==1


def test_bank_integrator_constant_fiber_matches_known_total_banks():
    from analysis.genesis_volume_banks_validation.angular import integrate_angular
    raw=fixture();lo,hi=map(fraction,raw['section']['lambda_bounds']);mid=F(9,10)
    exact=(F(160)*(mid**3-lo**3)/3,F(160)*(hi**3-mid**3)/3)
    values=np.array([np.longdouble(v.numerator)/np.longdouble(v.denominator) for v in exact])
    def callback(n):
        return values,np.array([np.longdouble('1e-16')]*2)
    r=integrate_angular(raw,callback,('reference_volume_negative_m3','reference_volume_positive_m3'))
    assert r['status']=='FINITE_AGREEMENT'
    assert np.allclose(r['fine_values'],[float(v/2) for v in exact],rtol=1e-14)
    assert not r['angular_accuracy_certified']


def test_proved_empty_source_preserves_the_full_parameter_measure_ledger():
    from analysis.genesis_volume_banks_validation.angular import integrate_angular
    raw=fixture(.1);raw['seed_center_world_m']=[100.,100.,100.];raw['section']['lambda_bounds']=[.1,1.]
    def never(n):raise AssertionError('Proved empty source was sampled')
    r=integrate_angular(raw,never,('reference_volume_negative_m3','reference_volume_positive_m3'))
    assert r['radial_line_evaluations']==0
    assert F(*map(int,r['initial_double_area']))==1
    assert F(*map(int,r['discarded_double_area']))==1
    assert F(*map(int,r['retained_double_area']))==0
    assert r['discarded_cells']


def test_independent_bank_auditor_matches_full_sphere_known_banks():
    from analysis.genesis_volume_banks_validation.independent_bank import audit_certificates
    raw=fixture();cert=PreparedBankBounds(raw).certificate(simplex())
    result=audit_certificates(raw,[cert])
    assert result['independent_bank_fiber_checks']==8
    assert result['all_sampled_bank_fibers_contained']


def test_actual_false_candidate_preserves_the_previous_strict_radial_exclusion():
    import json
    from pathlib import Path
    from analysis.genesis_volume_mapped_validation.angular import CellGeometry
    path=Path(__file__).resolve().parents[1]/'analysis/genesis_volume_banks_validation/radial_exclusion_fixture.json'
    fixture=json.loads(path.read_text())
    vertices=tuple(tuple(F(*map(int,v)) for v in row) for row in fixture['vertices'])
    raw=fixture['raw']
    assert all(hi==0 for lo,hi in CellGeometry(raw).bounds(vertices).values())
    cert=PreparedBankBounds(raw).certificate(vertices)
    assert cert['radial_support_classification']=='EMPTY_CERTIFIED_ON_SUPPLIED_CELL'
    assert all(v=='sphere_cohort_Bernstein' for v in cert['bank_reasons'].values())
    assert all(F(*map(int,b))==0 for pair in cert['bank_measure_bounds'].values() for b in pair)


def test_event_guided_integration_of_a_known_sphere_contained_plane_cut():
    from analysis.genesis_volume_banks_validation.angular import integrate_angular
    from tectonics.genesis_volume_operand_bounds import rational_enclosure
    raw=fixture();raw['section']['normal_xyz']=[1.,0.,0.];raw['section']['offset_m']=1.
    lo,hi=map(fraction,raw['section']['lambda_bounds'])
    positive=F(80)*((hi-F(1,4))**3-(lo-F(1,4))**3)/3
    total=F(80)*(hi**3-lo**3)/3
    def callback(n):
        root=F(1)/(4*n[1]) if n[1] else hi
        a,b=max(lo,root),min(hi,root)
        values=(F(160)*max(F(),b**3-lo**3)/3,F(160)*max(F(),hi**3-a**3)/3)
        leaves=[rational_enclosure(v) for v in values]
        centers=np.array([(x.lower+x.upper)/2 for x in leaves],np.longdouble)
        errors=np.array([x.absolute_error_for(v) for x,v in zip(leaves,centers)],np.longdouble)
        return centers,errors
    report=integrate_angular(raw,callback,('reference_volume_negative_m3','reference_volume_positive_m3'))
    assert report['exact_initial_parameter_measure_preserved']
    for measured,truth in zip(report['fine_values'],(total-positive,positive)):
        assert abs(F(*measured.as_integer_ratio())-truth)<=F(1,500000)*abs(truth)
    assert not report['angular_accuracy_certified']
    assert report['radial_line_evaluations']<=3400
