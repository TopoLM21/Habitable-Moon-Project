import copy
from pathlib import Path
import numpy as np
import pytest
from analysis.genesis_adaptive_control.controller import integrate,load_config,fingerprint,time_limit
from analysis.genesis_adaptive_control.metrics import assess
from analysis.genesis_adaptive_control.backend import continued_root
from analysis.genesis_backward_euler_prototype.core import step

CONFIG=Path(__file__).resolve().parents[1]/'analysis/genesis_adaptive_control/config.json'


def config(**kw):
    c=load_config(CONFIG);c.update(horizon_years=1.,initial_step_years=1.,min_step_years=.01,max_step_years=1.,max_attempts=24,max_accepted_windows=20)
    c.update(kw);return c


def obs(x=0.,work=0.,stored=1.):
    return dict(ids=['a'],stress_pa=np.array([[x,0,0]]),velocity_m_s=np.zeros((1,6,3)),damage=np.array([.04]),temperature_k=np.array([1300.]),enthalpy_j_kg=np.array([1560000.]),local_work_j=np.array([work]),local_stored_j=np.array([stored]),physical_heat_j=np.array([0.]),numerical_kinetic_j=0.,global_stored_j=stored,physical_drag_j=0.,activation_damage=.65,gate_flags=False)


def audited(before,after):
    L=after['ledger'];assert abs(L['balance_residual_j'])<1e-12
    return {'status':'PASS'}


def advance(before,h,label,cap):
    # Independent SI BE drag example (synthetic controller time unit = second).
    r=step([[1]],[[1]],[0],[0],[before['v']],h)
    return dict(elapsed_years=before['elapsed_years']+h,v=float(r['velocity'][0]),ledger=r['ledger'])


def observation(s):
    o=obs(s['v']);o['numerical_kinetic_j']=s['ledger']['numerical_kinetic_j'];return o


def test_manufactured_accepts_audited_fine_path_without_extrapolation():
    initial={'elapsed_years':0.,'v':1.};before=copy.deepcopy(initial);events=[]
    r=integrate(initial,config(),advance,audited,observation,assess,lambda k,v:events.append(k))
    assert r['status']=='PASS' and r['trial_calls']==3
    assert r['accepted']['v']==pytest.approx(1/1.5**2)
    assert initial==before and events.index('accept')>max(i for i,k in enumerate(events) if k=='audited_trial')


def test_rejection_rolls_back_then_reduces_and_is_bounded():
    initial={'elapsed_years':0.,'v':1.};seen=[];events=[]
    def advancing(b,h,label,cap):seen.append((label,b['elapsed_years'],b['v'],h));return advance(b,h,label,cap)
    c=config(stress_atol_pa=.005,stress_rtol=.001,max_attempts=12)
    r=integrate(initial,c,advancing,audited,observation,assess,lambda k,v:events.append((k,v)))
    rejected=[e for k,e in events if k=='reject'];assert rejected and rejected[0]['accepted_unchanged']
    assert seen[0][1:3]==seen[3][1:3] and seen[3][3]<seen[0][3]
    assert r['attempts']<=12 and r['trial_calls']<=36
    assert r['history'][0]['base_sha256']==fingerprint(initial)


@pytest.mark.parametrize('failure',['audit','gate','negative','mutation','clock'])
def test_invalid_attempt_never_commits_or_blindly_refines(failure):
    initial={'elapsed_years':0.,'v':1.};events=[]
    def adv(b,h,label,cap):
        r=advance(b,h,label,cap)
        if failure=='mutation':b['v']=9
        if failure=='clock':r['elapsed_years']+=1
        return r
    def audit(b,a):return {'status':'FAIL'} if failure=='audit' else audited(b,a)
    def observe(s):
        o=observation(s)
        if failure=='gate':o['damage']=np.array([.6501])
        if failure=='negative':o['physical_heat_j']=np.array([-1e-20])
        return o
    r=integrate(initial,config(),adv,audit,observe,assess,lambda k,v:events.append(k))
    assert r['status']=='STOPPED' and r['accepted']==initial and 'accept' not in events
    assert r['attempts']==1


def test_second_half_failure_cannot_commit_first_half():
    initial={'elapsed_years':0.,'v':1.}
    def adv(b,h,label,cap):
        if label.endswith('half2'):raise ValueError('phase event')
        return advance(b,h,label,cap)
    r=integrate(initial,config(),adv,audited,observation,assess)
    assert r['accepted']==initial and r['accepted_substeps']==0


@pytest.mark.parametrize('limit', ['max_attempts','max_rejections_per_window','min_step_years','max_accepted_windows'])
def test_limits_stop_cleanly(limit):
    c=config();c[limit]=1 if limit!='min_step_years' else 1.
    def forced(*args):return dict(score=2.,next_factor=.5)
    if limit=='max_accepted_windows':c['horizon_years']=2.;forced=lambda *a:dict(score=0.,next_factor=1.)
    r=integrate({'elapsed_years':0.,'v':1.},c,advance,audited,observation,forced)
    assert r['status']=='STOPPED' and any(x in r['reason'] for x in ('LIMIT','MIN_STEP_BUDGET_UNMET'))


def test_runtime_after_audit_stops_before_acceptance():
    now=[0.]
    def audit(b,a):now[0]+=4;return audited(b,a)
    r=integrate({'elapsed_years':0.,'v':1.},config(runtime_seconds=10.),advance,audit,observation,assess,clock=lambda:now[0])
    assert r['status']=='STOPPED' and r['accepted_substeps']==0


def test_near_zero_uses_absolute_budget_and_local_defect_not_hidden():
    c=config(global_work_atol_j_per_year=1.,local_work_atol_j_per_year=.1)
    a=obs(stored=0);b=obs(stored=0);d=obs(stored=0)
    assert assess(a,b,d,1,c)['score']==0
    b['local_work_j'][0]=.2
    r=assess(a,b,d,1,c);assert r['checks']['numerical_work']['local_ratio']==pytest.approx(2)
    assert r['score']>1 and r['checks']['numerical_work']['global_ratio']<1


def test_damage_error_margin_rejects_before_gate_even_with_large_absolute_tolerance():
    c=config(damage_atol=1.);a=obs();b=obs();d=obs()
    a['damage'][0]=.649;d['damage'][0]=.648
    r=assess(a,b,d,1,c);assert r['score']>1 and r['checks']['damage_margin']['local_ratio']>1


def test_all_signs_include_discarded_coarse_step():
    a=obs();b=obs();d=obs();a['numerical_kinetic_j']=-1
    assert assess(a,b,d,1,config())['fatal']=='NEGATIVE_DISSIPATION_OR_STORED_ENERGY'


def test_physical_velocity_metric_is_vector_difference_not_speed_difference():
    a=obs();b=obs();d=obs();a['velocity_m_s'][:]=[1,0,0];d['velocity_m_s'][:]=[-1,0,0]
    assert assess(a,b,d,1,config())['checks']['velocity_m_s']['max_difference']==2


def test_fine_numerical_work_sums_both_halves_only():
    a=obs(work=1e9);b=obs(work=2);d=obs(work=3)
    r=assess(a,b,d,1,config());assert r['checks']['numerical_work']['total_j']==5


def test_nested_deadline_restores_outer_alarm():
    import signal
    with time_limit(10):
        with time_limit(1):assert signal.getitimer(signal.ITIMER_REAL)[0]<=1
        assert 8<signal.getitimer(signal.ITIMER_REAL)[0]<=10
    assert signal.getitimer(signal.ITIMER_REAL)[0]==0


def test_real_orbit_fields_carried_without_mutating_original():
    root={'runtime_state':{'thermal':{'orbit':{'time_myr':0.,'eccentricity':.3}},'geometry':{}}}
    p={'state':{'displacement_m':[],'velocity_m_s':[],'fault_history':{},'mechanics':{}},'thermal':{'time_myr':1.},'orbit_after':{'time_myr':1.,'eccentricity':.2,'dissipated_energy_j':10}}
    r=continued_root(root,p)
    assert r['runtime_state']['thermal']['orbit']==p['orbit_after']
    assert root['runtime_state']['thermal']['orbit']['eccentricity']==.3


def test_work_factor_respects_absolute_and_relative_scaling():
    from analysis.genesis_adaptive_control.metrics import work_budget
    o=obs(work=4.,stored=1.)
    c=config(local_work_atol_j_per_year=1.,local_work_rtol_stored=3.,global_work_atol_j_per_year=100.)
    w=work_budget([o],1,c);assert w['predicted_step_factor']==pytest.approx(1.)
    o['local_work_j'][0]=16.
    w=work_budget([o],1,c);r=w['predicted_step_factor']
    assert 16*r*r==pytest.approx(r+3)


def test_stress_comparison_uses_world_tensor_and_counts_shear_twice():
    from analysis.genesis_adaptive_control.metrics import projected_stress_world
    frames=np.eye(3)[None];world=projected_stress_world([[0,0,2]],frames)
    assert np.linalg.norm(world)==pytest.approx(np.sqrt(8))
    rotated=np.array([[[0,-1,0],[1,0,0],[0,0,1]]])
    np.testing.assert_allclose(projected_stress_world([[0,0,-2]],rotated),world)


@pytest.mark.parametrize('key,value',[('runtime_seconds',0),('stress_atol_pa',float('nan')),('max_attempts',1.5),('max_step_years',51),('horizon_years',101),('safety',1.1)])
def test_bad_config_rejected(tmp_path,key,value):
    import json
    c=load_config(CONFIG);c[key]=value;p=tmp_path/'config.json';p.write_text(json.dumps(c))
    with pytest.raises(ValueError):load_config(p)


def test_failed_durable_acceptance_leaves_in_memory_cursor_unchanged():
    initial={'elapsed_years':0.,'v':1.}
    def emit(k,v):
        if k=='accept':raise OSError('disk cap')
    r=integrate(initial,config(),advance,audited,observation,assess,emit)
    assert r['accepted']==initial and r['status']=='STOPPED'


def test_stop_record_disk_failure_still_returns_last_accepted_cursor():
    initial={'elapsed_years':0.,'v':1.}
    def emit(k,v):
        if k in ('accept','stop'):raise OSError('disk full')
    r=integrate(initial,config(),advance,audited,observation,assess,emit)
    assert r['accepted']==initial and 'STOP_RECORD_FAILED' in r['reason']


def test_stiff_drag_does_not_impose_relaxation_time_step():
    def adv(b,h,label,cap):
        x=step([[1]],[[1e18]],[0],[0],[b['v']],h)
        return dict(elapsed_years=b['elapsed_years']+h,v=float(x['velocity'][0]),ledger=x['ledger'])
    def observe(s):
        o=observation(s);o['stress_pa']*=0;o['velocity_m_s'][:]=[s['v'],0,0];return o
    r=integrate({'elapsed_years':0.,'v':1.},config(),adv,audited,observe,assess)
    assert r['status']=='PASS' and r['attempts']==1
    assert r['accepted']['v']<1e-30
