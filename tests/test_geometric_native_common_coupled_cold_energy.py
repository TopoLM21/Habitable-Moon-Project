"""Independent composed-map derivatives, source volume and cold projections."""
from dataclasses import replace
from fractions import Fraction as F
from pathlib import Path
import json
import runpy

import mpmath as mp
import pytest

from tectonics import geometric_native_common_coupled_cold_energy as api
from tectonics.geometric_native_pose import FractionInterval as I


ROOT=Path(__file__).resolve().parents[1]
CONTROL=ROOT/'analysis/geometric_native_common_general_force_validation/native_control.txt'


def contains(bound,value):
    value=F(mp.nstr(value,75)) if isinstance(value,mp.mpf) else F(value)
    assert bound.lo<=value<=bound.hi


def overlap(a,b):
    assert max(a.lo,b.lo)<=min(a.hi,b.hi)


def saved(value):
    return I(F(*value['lo']),F(*value['hi'])) if type(value) is dict else I(F(*value))


BASE=(F(3),F(4),F(1))
PSI=((F(1,10),F(1,5),F(3,10)),(F(-1,5),F(3,10),F(1,10)))
G=(((F(1,100),F(1,50),F(0)),(F(0),F(1,100),F(1,50)),(F(3,100),F(0),F(-1,100))),
   ((F(0),F(1,100),F(1,50)),(F(3,100),F(-1,100),F(0)),(F(1,100),F(0),F(1,50))))
OMEGA=(F(1,30),F(-1,40),F(1,50))
STATE=(F(1,20),F(-1,30),F(2,5))


def numeric_green(*state,omega=OMEGA,base=BASE,psi=PSI,columns=G):
    """Independent high precision direct Jacobian, no native helpers."""
    m=lambda v:mp.mpf(v.numerator)/v.denominator if type(v) is F else mp.mpf(v)
    x=mp.matrix([m(v) for v in base]);modes=[mp.matrix([m(v) for v in p]) for p in psi]
    gs=[mp.matrix([[m(v) for v in row] for row in g]) for g in columns]
    y=x+sum((state[i]*modes[i] for i in range(len(modes))),mp.zeros(3,1))
    B=mp.eye(3)+sum((state[i]*gs[i] for i in range(len(gs))),mp.zeros(3))
    o=[m(v) for v in omega];w=mp.matrix([o[1]*y[2]-o[2]*y[1],o[2]*y[0]-o[0]*y[2],o[0]*y[1]-o[1]*y[0]])
    d=-mp.sqrt(mp.fdot(y,y))*w*y.T/(mp.mpf(2)**3*mp.mpf(10)**2)
    Ft=(mp.eye(3)+state[-1]*d)*B
    return (Ft.T*Ft-mp.eye(3))/2


def native_green(state=STATE,*,base=BASE,psi=PSI,columns=G,omega=OMEGA):
    a=api._BoxAlgebra(I(0),96)
    convert=lambda v:a.constant(v)
    return api._mixed_green_twojet(tuple(map(convert,base)),tuple(tuple(map(convert,p)) for p in psi),
        tuple(tuple(tuple(map(convert,row)) for row in g) for g in columns),state[:-1],state[-1],omega,F(2),F(10),a)


@pytest.mark.parametrize('state',[STATE,(F(0),F(0),F(0)),(F(-1,40),F(1,80),F(1))])
def test_joint_green_all_first_and_second_derivatives_against_independent_cartesian_differentiation(state):
    E,first,second=native_green(state)
    with mp.workdps(85):
        values=tuple(mp.mpf(v.numerator)/v.denominator for v in state)
        for a in range(3):
            for b in range(3):
                fn=lambda *x:numeric_green(*x)[a,b]
                contains(api._range(E[a][b]),fn(*values))
                for i in range(3):
                    deriv=[0]*3;deriv[i]=1
                    contains(api._range(first[i][a][b]),mp.diff(fn,values,tuple(deriv)))
                    for j in range(3):
                        dd=[0]*3;dd[i]+=1;dd[j]+=1
                        contains(api._range(second[i][j][a][b]),mp.diff(fn,values,tuple(dd)))
                assert all(second[i][j]==second[j][i] for i in range(3) for j in range(3))


def test_joint_map_retains_modal_preparation_cross_curvature_and_is_not_additive():
    E,_,second=native_green()
    assert any(not(api._range(v).lo<=0<=api._range(v).hi) for row in second[0][2] for v in row)
    with mp.workdps(80):
        c1,c2,lam=(mp.mpf(v.numerator)/v.denominator for v in STATE)
        density=lambda e:mp.mpf(5)*sum(e[i,i] for i in range(3))**2/2+mp.mpf(3)*sum(e[i,j]**2 for i in range(3) for j in range(3))
        actual=density(numeric_green(c1,c2,lam))
        additive=density(numeric_green(c1,c2,0))+density(numeric_green(0,0,lam))
        assert abs(actual-additive)>mp.mpf('1e-7')


def test_preparation_rigid_rotation_cancels_from_green_strain():
    with mp.workdps(80):
        state=tuple(mp.mpf(v.numerator)/v.denominator for v in STATE)
        E=numeric_green(*state)
        y=mp.matrix([mp.mpf(v.numerator)/v.denominator for v in BASE])+sum((state[i]*mp.matrix([mp.mpf(v.numerator)/v.denominator for v in p]) for i,p in enumerate(PSI)),mp.zeros(3,1))
        omega=mp.matrix([mp.mpf(v.numerator)/v.denominator for v in OMEGA])
        K=mp.matrix([[0,-omega[2],omega[1]],[omega[2],0,-omega[0]],[-omega[1],omega[0],0]])
        tau=(mp.mpf(10)**3-(mp.sqrt(mp.fdot(y,y))/2)**3)/(3*mp.mpf(10)**2)
        Q=mp.expm(state[-1]*tau*K)
        B=mp.eye(3)+sum((state[i]*mp.matrix([[mp.mpf(v.numerator)/v.denominator for v in row] for row in g]) for i,g in enumerate(G)),mp.zeros(3))
        d=-mp.sqrt(mp.fdot(y,y))*(K*y)*y.T/(mp.mpf(2)**3*mp.mpf(10)**2)
        physical=Q*(mp.eye(3)+state[-1]*d)*B
        assert max(abs(((physical.T*physical-mp.eye(3))/2-E)[i,j]) for i in range(3) for j in range(3))<mp.mpf('1e-75')


def test_true_reference_pullback_changes_joint_strain_and_cannot_be_omitted():
    raw=tuple(tuple(tuple(2*v for v in row) for row in g) for g in G)
    good=native_green()[0];bad=native_green(columns=raw)[0]
    assert any(api._range(good[i][j]).hi<api._range(bad[i][j]).lo or api._range(bad[i][j]).hi<api._range(good[i][j]).lo for i in range(3) for j in range(3))


def test_radial_chart_rejects_a_whole_box_containing_the_origin():
    a=api._BoxAlgebra(I(0),96);z=a.constant(0);zm=((z,)*3,)*3
    with pytest.raises(ValueError,match='positive current radial'):
        api._mixed_green_twojet((z,z,z),((z,z,z),),(zm,),(F(0),),F(1,2),OMEGA,F(2),F(10),a)


@pytest.fixture(scope='module')
def native_context():
    if not CONTROL.is_file(): pytest.skip('New935 native driver is not yet supplied')
    return runpy.run_path(str(CONTROL))['context']()


def ctx_system(ctx):
    return ctx.get('cold_energy_system',ctx.get('coupled_cold_system'))


@pytest.fixture(scope='module')
def native_responses(native_context):
    system=ctx_system(native_context)
    cold=native_context['cold_carrier'];c=tuple(v/4 for v in cold.bound_coordinate_increments)
    return system,api.evaluate_native_common_coupled_cold_energy(system,(F(0),)*56,F(0)),api.evaluate_native_common_coupled_cold_energy(system,c,F(1,2))


def test_authentic_all_eight_parent_volume_and_three_archive_cohorts_once(native_responses):
    system,zero,mixed=native_responses
    assert len(mixed.cell_receipts)==408
    assert mixed.covered_source_volume_bounds_m3.lo<=mixed.whole_source_volume_bounds_m3.lo<=mixed.whole_source_volume_bounds_m3.hi<=mixed.covered_source_volume_bounds_m3.hi
    assert len(system.original_parent_ids)==8 and len(system.nonempty_archive_domain_ids)==3
    assert sum(c['kind']=='whole_donor_activation_collar' for c in mixed.cell_receipts)==24
    assert sum(c['kind']=='zero_modal_complement_preparation' for c in mixed.cell_receipts)==192
    assert all(not c['activation_history_replaced_by_zero'] for c in mixed.cell_receipts)
    assert all(c['original_material_cartesian_gradient_pullback_retained'] for c in mixed.cell_receipts)


def test_authentic_inactive_complement_has_positive_work_and_exact_zero_modal_jets(native_responses):
    _,_,mixed=native_responses
    cells=[c for c in mixed.cell_receipts if c['kind']=='zero_modal_complement_preparation']
    assert sum((c['half_norm_bounds_j'].lo for c in cells),F(0))>0
    assert all(c['active_coordinate_indices']==() and c['local_coordinate_indices']==(56,) for c in cells)
    assert all(c['projection_bounds_n']==c['projection_first_bounds']==c['projection_second_bounds']==() for c in cells)


def test_authentic_zero_literal_reference_energy_and_full_mixed_prestress_operator(native_responses):
    system,zero,mixed=native_responses
    assert zero.energy_j==I(sum(system.literal_reference_energies_j,F(0)))
    assert len(mixed.gradient57)==len(mixed.hessian57)==57
    assert all(len(row)==57 for row in mixed.hessian57)
    assert all(mixed.hessian57[i][j]==mixed.hessian57[j][i] for i in range(57) for j in range(57))
    for index,force in zip(system.tip_coordinate_indices,system.literal_reference_forces_n): assert zero.gradient57[index]==I(force)
    assert len(mixed.tip_projection_m)==len(mixed.tip_projection_first)==len(mixed.tip_projection_second)==8
    assert mixed.source_beams_included_once and mixed.original_gram_and_inverse_reused_without_quadrature


def test_authentic_system_and_response_full_restart_and_corruption(native_context,native_responses):
    system,zero,_=native_responses;ctx=api.native_common_coupled_cold_energy_context(system)
    dto=api.native_common_coupled_cold_energy_to_dict(system)
    restored=api.native_common_coupled_cold_energy_from_dict(json.loads(json.dumps(dto)),ctx['source899'],ctx['cold_carrier'],ctx['continuum_system'],ctx['preparations'],source923=ctx['source923'],provenance=system.provenance)
    assert api.native_common_coupled_cold_energy_to_dict(restored)==dto
    rdto=api.native_common_coupled_cold_energy_response_to_dict(zero)
    rr=api.native_common_coupled_cold_energy_response_from_dict(json.loads(json.dumps(rdto)),system=system,expected_coordinate_increments56=(F(0),)*56,expected_preparation_fraction=F(0))
    assert api.native_common_coupled_cold_energy_response_to_dict(rr)==rdto
    forged_system=json.loads(json.dumps(dto));forged_system['source_clocks_s'][0][0]=False
    assert forged_system==dto
    with pytest.raises(ValueError,match='restart changed'):
        api.native_common_coupled_cold_energy_from_dict(forged_system,ctx['source899'],ctx['cold_carrier'],ctx['continuum_system'],ctx['preparations'],source923=ctx['source923'],provenance=system.provenance)
    forged_response=json.loads(json.dumps(rdto));forged_response['coordinate_increments56'][0][0]=False
    assert forged_response==rdto
    with pytest.raises(ValueError,match='restart changed'):
        api.native_common_coupled_cold_energy_response_from_dict(forged_response,system=system,expected_coordinate_increments56=(F(0),)*56,expected_preparation_fraction=F(0))
    rdto['cell_receipts'][0]['source_volume_bounds_m3']['lo']=[0,1]
    with pytest.raises(ValueError,match='restart changed'):
        api.native_common_coupled_cold_energy_response_from_dict(rdto,system=system,expected_coordinate_increments56=(F(0),)*56,expected_preparation_fraction=F(0))
    with pytest.raises(ValueError,match='unregistered'):
        api.validate_native_common_coupled_cold_energy_response(replace(zero,energy_j=I(0),binding_sha256=api._seal(replace(zero,energy_j=I(0)))))


@pytest.mark.parametrize('bad',[True,1.0,F(-1),F(2)])
def test_authentic_preparation_type_and_domain_are_fail_closed(native_context,bad):
    with pytest.raises(ValueError,match='literal Fraction'):
        api.evaluate_native_common_coupled_cold_energy(ctx_system(native_context),preparation_fraction=bad)


def test_authentic_material_envelope_budget_and_no_time_adoption(native_context,native_responses):
    system,_,mixed=native_responses
    with pytest.raises(ValueError,match='coordinate envelope'):
        api.evaluate_native_common_coupled_cold_energy(system,tuple(v+1 for v in system.coordinate_envelope_abs_m))
    with pytest.raises(ValueError,match='cell budget'):
        api.evaluate_native_common_coupled_cold_energy(system,max_cells=1)
    assert not mixed.mixed_physical_force_equilibrium_certified and not mixed.full_system_preparation_work_paid
    assert not mixed.whole_clock_all_pair_admission_authorized and not mixed.clocks_advanced
    assert mixed.accepted_new_native_intervals==0


def test_authentic_lambda_zero_full_nonzero_modal_reduction_to_saved899(native_context):
    system=ctx_system(native_context)
    endpoint=next(q['response'] for q in native_context['source899']['evaluation_queries'] if q['label']=='registered_cold_endpoint')
    coordinates=tuple(F(*v) for v in endpoint['coordinate_increments'])
    response=api.evaluate_native_common_coupled_cold_energy(system,coordinates,F(0))
    assert any(coordinates)
    overlap(response.energy_j,saved(endpoint['energy_j']))
    for a,b in zip(response.gradient57[:56],endpoint['gradient_n']): overlap(a,saved(b))
    for i in range(56):
        for j in range(56): overlap(response.hessian57[i][j],saved(endpoint['hessian_n_m'][i][j]))
    assert response.original_gram_and_inverse_reused_without_quadrature


def test_authentic_zero_modal_reduction_to_saved923_energy_and_lambda_jets(native_context):
    system=ctx_system(native_context)
    old=next(q['response'] for q in native_context['source923']['work_queries'] if q['parameter']==[1,2])
    response=api.evaluate_native_common_coupled_cold_energy(system,(F(0),)*56,F(1,2))
    overlap(response.energy_j,saved(old['energy_j']))
    overlap(response.gradient57[56],saved(old['potential_path_power_j']))
    overlap(response.hessian57[56][56],saved(old['potential_path_second_derivative_j']))
    for a,b in zip(response.tip_projection_m,old['tip_projection_m']): overlap(a,saved(b))


@pytest.mark.parametrize('parent,tag,layer',[('octant:0','interior','upper'),('octant:0','start_seam','lower'),
                                           ('octant:4','interior','lower'),('octant:4','end_seam','upper')])
def test_authentic_source_point_reconstructs_original_and_relative_gradient_frames(native_context,parent,tag,layer):
    system=ctx_system(native_context);ctx=api.native_common_coupled_cold_energy_context(system)
    material=next(b for b in ctx['continuum_system'].banks if b.original_parent_id==parent)
    bank=next(b for b in ctx['cold_carrier'].banks if b.original_parent_id==parent)
    depth=material.elastic_layer_areal_depth_m/2 if layer=='upper' else (material.elastic_layer_areal_depth_m+material.areal_material_thickness_m)/2
    angle=F(1,3) if tag=='interior' else ctx['cold_carrier'].whole_domain_certificate.seam_half_width_radians/2
    coordinates=tuple(v/4 for v in ctx['cold_carrier'].bound_coordinate_increments)
    point=api.evaluate_native_common_coupled_cold_point(system,parent,tag,angle,bank.beta_cut_radians/2,depth,coordinates,F(1,2),layer=layer)
    assert point['local_coordinate_indices'][-1]==56
    assert point['diagnostic_point_used_as_volume_weight'] is False
    assert len(point['green_strain_first'])==len(point['local_coordinate_indices'])
    assert len(point['mode_original_gradient_columns_per_m'])==len(point['relative_gradient_columns_per_m'])
    dto=api.native_common_coupled_cold_point_to_dict(point)
    assert dto['original_parent_id']==parent and dto['layer']==layer
    assert all(point['green_strain_second'][i][j]==point['green_strain_second'][j][i]
        for i in range(len(point['local_coordinate_indices'])) for j in range(len(point['local_coordinate_indices'])))
