"""Transactional first-order step doubling. No production or solver imports."""
import copy
import hashlib
import json
import math
import time
from pathlib import Path


class StopRun(RuntimeError):
    pass


def fingerprint(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def load_config(path):
    c=json.loads(Path(path).read_text())
    for k,v in c.items():
        if k=='description':continue
        if isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) or v<=0:
            raise ValueError('Positive finite config: '+k)
    for k in ('max_attempts','max_accepted_windows','max_rejections_per_window','artifact_limit_bytes','free_reserve_bytes'):
        if not isinstance(c[k],int):raise ValueError('Integer config: '+k)
    if not 0<c['min_factor']<c['safety']<1<c['max_factor']:raise ValueError('Step factors')
    if not c['min_step_years']<=c['initial_step_years']<=c['max_step_years']<=50:raise ValueError('Supported 0<h<=50yr')
    if c['horizon_years']>100 or c['horizon_years']<c['min_step_years']:raise ValueError('Bounded horizon <=100yr')
    if not 0<c['damage_error_fraction_of_margin']<1:raise ValueError('Gate error margin')
    return c


def integrate(initial,c,advance,audit,observe,assess,emit=lambda *a:None,clock=time.monotonic,*,event_policy=None,resume=None):
    """Each callback gets isolated data. Only audited half1+half2 commit together.

    advance(snapshot,h_years,label,remaining_seconds) returns a snapshot with
    elapsed_years. audit(before,after) must independently return status PASS.
    Errors (including gate/audit/domain failure) STOP rather than refine blindly.
    Numerical/observable budget rejection alone proposes a smaller next h.
    """
    accepted=copy.deepcopy(initial);start=clock();initial_hash=fingerprint(initial)
    history=[];attempts=calls=windows=rejects=0;h=c['initial_step_years'];origin=initial['elapsed_years']
    status='STOPPED';reason=''
    binding=fingerprint(dict(config=c,initial=initial,controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        event_binding=event_policy.binding if event_policy else None))
    if resume is not None and event_policy is None:raise ValueError('EVENT_POLICY_REQUIRED_FOR_RESTART')
    if resume is not None:
        saved=copy.deepcopy(resume);sha=saved.pop('sha256')
        if fingerprint(saved)!=sha or saved['binding']!=binding or saved['version']!=1:
            raise ValueError('CONTROLLER_RESTART_BINDING')
        accepted=saved['accepted'];history=saved['history'];h=saved['next_step_years']
        attempts=saved['attempts'];calls=saved['trial_calls'];windows=saved['windows'];rejects=saved['rejects']
        if event_policy:event_policy.restore(saved['event_state'])
    def checkpoint():
        state=dict(version=1,binding=binding,accepted=copy.deepcopy(accepted),history=copy.deepcopy(history),
            next_step_years=h,attempts=attempts,trial_calls=calls,windows=windows,rejects=rejects,
            event_state=event_policy.state() if event_policy else None)
        state['sha256']=fingerprint(state)
        return state
    try:
        while accepted['elapsed_years']-origin<c['horizon_years']-1e-10:
            if clock()-start>=c['runtime_seconds']:raise StopRun('RUNTIME_LIMIT')
            if attempts>=c['max_attempts']:raise StopRun('ATTEMPT_LIMIT')
            if windows>=c['max_accepted_windows']:raise StopRun('ACCEPTED_WINDOW_LIMIT')
            if rejects>=c['max_rejections_per_window']:raise StopRun('REJECTION_LIMIT')
            remaining=c['horizon_years']-(accepted['elapsed_years']-origin)
            h=min(h,remaining,c['max_step_years'])
            if h<c['min_step_years']*(1-1e-12):raise StopRun('MIN_STEP_LIMIT')
            # Avoid an unusable final sliver without stepping below minimum.
            if 0<remaining-h<c['min_step_years']:
                h=remaining/2
                if h<c['min_step_years']:raise StopRun('MIN_STEP_FINAL_REMAINDER')
            if event_policy:
                proof=event_policy.check(copy.deepcopy(accepted),h)
                emit('event_check',proof)
                if proof['status']!='CLEAR':raise StopRun(proof.get('reason','EVENT_UNRESOLVED'))
            attempts+=1;base=copy.deepcopy(accepted);base_hash=fingerprint(base)
            event=dict(attempt=attempts,start_years=base['elapsed_years'],step_years=h,base_sha256=base_hash)
            emit('attempt',event)
            def trial(before,dt,label):
                nonlocal calls
                left=c['runtime_seconds']-(clock()-start)
                if left<=0:raise StopRun('RUNTIME_LIMIT')
                isolated=copy.deepcopy(before);before_hash=fingerprint(isolated);calls+=1
                after=advance(isolated,dt,label,min(left,c['trial_seconds']))
                if fingerprint(isolated)!=before_hash:raise StopRun('INPUT_MUTATED')
                if clock()-start>=c['runtime_seconds']:raise StopRun('RUNTIME_LIMIT')
                if not math.isclose(after['elapsed_years'],before['elapsed_years']+dt,rel_tol=0,abs_tol=1e-9):raise StopRun('CLOCK_MISMATCH')
                ah=fingerprint(after);audit_before=copy.deepcopy(before)
                proof=audit(audit_before,after)
                if fingerprint(after)!=ah or fingerprint(audit_before)!=fingerprint(before):raise StopRun('AUDIT_MUTATED_STATE')
                if proof.get('status')!='PASS':raise StopRun('AUDIT_FAILED')
                emit('audited_trial',dict(label=label,sha256=ah,audit=proof))
                return after,observe(after)
            ow=trial(base,h,f'{attempts:03d}_whole')[1]
            half1,o1=trial(base,h/2,f'{attempts:03d}_half1')
            half2,o2=trial(half1,h/2,f'{attempts:03d}_half2')
            result=assess(ow,o1,o2,h,c)
            if clock()-start>=c['runtime_seconds']:raise StopRun('RUNTIME_LIMIT')
            event.update(assessment=result,rollback_sha256=base_hash)
            if result.get('fatal'):raise StopRun(result['fatal'])
            score=result['score'];factor=result['next_factor']
            if not math.isfinite(score) or score<0 or not math.isfinite(factor) or factor<=0:raise StopRun('INVALID_ASSESSMENT')
            if score>1 and factor>=1:raise StopRun('INVALID_SHRINK_FACTOR')
            if fingerprint(base)!=base_hash or fingerprint(accepted)!=base_hash:raise StopRun('ROLLBACK_STATE_CHANGED')
            if score<=1:
                # Emit durable acceptance before updating the in-memory cursor.
                event.update(decision='ACCEPT',first_half_sha256=fingerprint(half1),second_half_sha256=fingerprint(half2))
                emit('accept',event);accepted=copy.deepcopy(half2);windows+=1;rejects=0
            else:
                rejects+=1;event.update(decision='REJECT',accepted_unchanged=True);emit('reject',event)
            history.append(event)
            if score>1 and rejects>=c['max_rejections_per_window']:raise StopRun('REJECTION_LIMIT')
            if score>1 and h<=c['min_step_years']*(1+1e-12):raise StopRun('MIN_STEP_BUDGET_UNMET')
            h=max(c['min_step_years'],h*min(c['max_factor'],max(c['min_factor'],factor)))
            if event_policy:emit('controller_checkpoint',checkpoint())
        status='PASS';reason='HORIZON_COMPLETE'
    except Exception as exc:
        reason=f'{type(exc).__name__}: {exc}'
        try:
            emit('stop',dict(reason=reason,accepted_sha256=fingerprint(accepted),accepted_elapsed_years=accepted['elapsed_years']))
        except Exception as log_error:
            reason+=f'; STOP_RECORD_FAILED: {type(log_error).__name__}: {log_error}'
    if fingerprint(initial)!=initial_hash:raise RuntimeError('Caller initial state changed')
    return dict(status=status,reason=reason,accepted=accepted,attempts=attempts,trial_calls=calls,
                accepted_windows=windows,accepted_substeps=2*windows,history=history,seconds=clock()-start,
                initial_unchanged=True,canonical_advanced=False,
                **({'controller_state':checkpoint()} if event_policy else {}))


# Unlike the legacy single-level alarm helper, preserve an enclosing deadline.
from contextlib import contextmanager
import signal


@contextmanager
def time_limit(seconds):
    if seconds<=0:raise TimeoutError('TIME_LIMIT')
    previous_handler=signal.getsignal(signal.SIGALRM)
    previous_timer=signal.getitimer(signal.ITIMER_REAL);started=time.monotonic()
    def expired(*_):raise TimeoutError('TIME_LIMIT')
    signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,min(seconds,previous_timer[0]) if previous_timer[0]>0 else seconds)
    try:yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous_handler)
        if previous_timer[0]>0:
            remaining=previous_timer[0]-(time.monotonic()-started)
            # Preserve an already-expired outer alarm too; do not silently cancel it.
            signal.setitimer(signal.ITIMER_REAL,max(remaining,1e-6),previous_timer[1])
