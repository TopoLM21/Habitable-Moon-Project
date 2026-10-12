import copy
import json
import subprocess
import sys
from pathlib import Path
import pytest
from analysis.genesis_adaptive_control.controller import integrate,load_config,fingerprint
from analysis.genesis_adaptive_control.events import Event,EventPolicy
from analysis.genesis_adaptive_control.event_replay import policy_from_config

CONFIG=Path('analysis/genesis_adaptive_control/config.json')
SHA='a'*64

def config(**changes):
    c=load_config(CONFIG);c.update(horizon_years=2.,initial_step_years=.25,min_step_years=.01,max_step_years=2.)
    c.update(changes);return c

def ec(rows):
    return dict(source_sha256=SHA,valid_years=[0.,3.],resolution_years=1/1024,max_evaluations=1000,seconds=10.,events=rows)

def row(kind='phase',center=.75,formula='decreasing',identity='a',transition='remelt',offset=0.):
    return dict(kind=kind,identity=identity,units='synthetic',transition=transition,formula=formula,center=center,offset=offset)

def initial():return dict(elapsed_years=0.,source_sha256=SHA,value=0.)
def advance(s,h,*_):return {**s,'elapsed_years':s['elapsed_years']+h,'value':s['value']+h}
def run(policy,c=None,**kwargs):
    return integrate(initial(),c or config(),advance,lambda *_:{'status':'PASS'},lambda s:s,
        kwargs.pop('assess',lambda *_:dict(score=0.,next_factor=1.6)),event_policy=policy,**kwargs)

@pytest.mark.parametrize('kind',['phase','birth','damage_gate'])
def test_interior_event_localizes_and_never_advances(kind):
    p=policy_from_config(ec([row(kind)]));r=run(p,config(initial_step_years=2.))
    b=p.blocked['bracket_years']
    assert b[0]<=.75<=b[1] and b[1]-b[0]<=1/1024
    assert r['accepted']==initial() and r['trial_calls']==0
    assert p.blocked['events'][0]['kind']==kind

def test_same_sign_endpoints_touch_detected():
    p=policy_from_config(ec([row(formula='touch')]));r=run(p,config(initial_step_years=2.))
    assert r['status']=='STOPPED' and p.blocked['bracket_years'][0]<=.75<=p.blocked['bracket_years'][1]

def test_multiple_events_earliest_and_simultaneous_group():
    rows=[row('phase',1.5),row('birth',.75),row('damage_gate',.75)]
    p=policy_from_config(ec(rows));run(p,config(initial_step_years=2.))
    assert {e['kind'] for e in p.blocked['events']}=={'birth','damage_gate'}

def test_no_event_grows_but_output_rejection_still_shrinks():
    rows=[row(formula='constant',offset=1.)]
    r=run(policy_from_config(ec(rows)))
    assert r['status']=='PASS' and r['history'][1]['step_years']>r['history'][0]['step_years']
    r=run(policy_from_config(ec(rows)),assess=lambda w,a,b,h,c:dict(score=2. if h>.2 else 0.,next_factor=.5 if h>.2 else 1.6))
    assert r['history'][0]['decision']=='REJECT'
    assert r['history'][1]['step_years']<r['history'][0]['step_years']

def test_remelt_rolls_back_to_last_accepted_not_trial_bracket():
    p=policy_from_config(ec([row()]));r=run(p)
    assert r['accepted']['elapsed_years']==pytest.approx(.65)
    assert r['accepted_windows']==2 and 'UNSUPPORTED' in r['reason']
    assert r['accepted']['elapsed_years']<p.blocked['bracket_years'][0]

@pytest.mark.parametrize('failure',['budget','missing','mutation','source','time'])
def test_fail_closed_and_restart_cannot_refresh_localization_budget(failure):
    cfg=ec([row()]);cfg['max_evaluations']=1 if failure=='budget' else 1000
    p=policy_from_config(cfg)
    if failure=='missing':p.enclose=lambda *a:None
    if failure=='mutation':
        def corrupt(s,*a):s['value']=99;return (1.,1.)
        p.enclose=corrupt
    if failure=='source':
        cfg['source_sha256']='b'*64;p=policy_from_config(cfg)
    if failure=='time':p.seconds_used=11.
    r=run(p,config(initial_step_years=2.));assert r['accepted']==initial() and p.blocked['status']=='UNRESOLVED'
    used=p.used
    rr=run(p,config(initial_step_years=2.),resume=r['controller_state'])
    assert rr['accepted']==initial() and p.used==used

def test_restart_matches_uninterrupted_and_checks_identity():
    cfg=ec([row(formula='constant',offset=1.)]);saved=[]
    full=run(policy_from_config(cfg),emit=lambda k,v:saved.append(copy.deepcopy(v)) if k=='controller_checkpoint' else None)
    checkpoint=json.loads(json.dumps(saved[0]))
    resumed=run(policy_from_config(cfg),resume=checkpoint)
    for k in ('accepted','history','attempts','trial_calls','accepted_windows'):
        assert resumed[k]==full[k]
    checkpoint['accepted']['value']=999
    with pytest.raises(ValueError,match='RESTART'):run(policy_from_config(cfg),resume=checkpoint)
    with pytest.raises(ValueError,match='RESTART'):run(policy_from_config(ec([row(center=1.)])),resume=saved[0])

def test_audit_failure_is_stop_not_localization_retry():
    p=policy_from_config(ec([row(formula='constant',offset=1.)]))
    r=integrate(initial(),config(),advance,lambda *_:{'status':'FAIL'},lambda s:s,lambda *_:dict(score=0.,next_factor=1.6),event_policy=p)
    assert 'AUDIT_FAILED' in r['reason'] and r['accepted']==initial() and r['trial_calls']==1

@pytest.mark.parametrize('key,value',[('resolution_years',0),('max_evaluations',True),('seconds',float('inf')),('valid_years',[2,1])])
def test_config_rejected(key,value):
    cfg=ec([row()]);cfg[key]=value
    with pytest.raises(ValueError):policy_from_config(cfg)

def test_cli_save_restart_and_no_overwrite(tmp_path):
    cp=tmp_path/'config.json';ep=tmp_path/'events.json';out=tmp_path/'result.json';res=tmp_path/'resumed.json'
    cp.write_text(json.dumps(config()));ep.write_text(json.dumps(ec([row()])))
    args=[sys.executable,'-m','analysis.genesis_adaptive_control.event_replay','--controller-config',str(cp),'--event-config',str(ep),'--output',str(out)]
    subprocess.run(args,check=True,capture_output=True)
    subprocess.run(args[:-1]+[str(res),'--resume',str(out)],check=True,capture_output=True)
    a=json.loads(out.read_text());b=json.loads(res.read_text())
    assert a['accepted']==b['accepted'] and a['controller_state']==b['controller_state']
    assert subprocess.run(args,capture_output=True).returncode!=0

def test_same_sign_endpoints_two_crossings_detected():
    # (t-1)^2 - 1/16: positive at both ends, negative in the interior.
    p=policy_from_config(ec([row(formula='touch',center=1.,offset=-1/16)]))
    r=run(p,config(initial_step_years=2.))
    assert r['trial_calls']==0
    assert p.blocked['bracket_years'][0]<=.75<=p.blocked['bracket_years'][1]

def test_hanging_provider_has_hard_deadline():
    import time
    cfg=ec([row()]);cfg['seconds']=.005;p=policy_from_config(cfg)
    def slow(*_):time.sleep(.05);return (1.,1.)
    p.enclose=slow;r=run(p)
    assert r['accepted']==initial() and 'TIME_LIMIT' in r['reason']

def test_missing_validity_and_invalid_enclosure_fail_closed():
    for bad in ('outside','bounds'):
        p=policy_from_config(ec([row()]))
        if bad=='outside':
            cfg=ec([row()]);cfg['valid_years']=[1.,3.];p=policy_from_config(cfg)
        else:p.enclose=lambda *_:(2.,1.)
        r=run(p);assert r['trial_calls']==0 and p.blocked['status']=='UNRESOLVED'

def test_event_at_start_and_boundary_is_not_accepted():
    for t in (0.,2.):
        p=policy_from_config(ec([row('birth',t)]));r=run(p,config(initial_step_years=2.))
        assert r['accepted']==initial() and r['status']=='STOPPED'

def test_checkpoint_write_failure_preserves_already_durable_acceptance():
    cfg=ec([row(formula='constant',offset=1.)]);log=[]
    def emit(k,v):
        if k=='controller_checkpoint':raise OSError('disk unavailable')
        log.append(k)
    r=run(policy_from_config(cfg),emit=emit)
    assert 'accept' in log and r['accepted']['elapsed_years']==.25 and r['status']=='STOPPED'
    rr=run(policy_from_config(cfg),resume=r['controller_state'])
    assert rr['status']=='PASS'

def test_restart_does_not_replenish_rejection_budget():
    cfg=ec([row(formula='constant',offset=1.)]);c=config(max_rejections_per_window=1)
    fail=lambda *_:dict(score=2.,next_factor=.5)
    r=run(policy_from_config(cfg),c,assess=fail)
    rr=run(policy_from_config(cfg),c,assess=fail,resume=r['controller_state'])
    assert rr['trial_calls']==r['trial_calls']==3 and 'REJECTION_LIMIT' in rr['reason']
