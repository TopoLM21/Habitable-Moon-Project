"""Read saved profiles; one late A/B continuation with an analytic oracle."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from .shear_diagnostic import PARAMETERS,setup,smooth,initial,advance

ROOT=Path(__file__).parent


def metrics(flow):
    flow=np.asarray(flow);n=len(flow);normal=flow/flow.mean()
    excess=np.maximum(flow-flow.mean(),0)
    # Fractional cell allocation makes the top-10%-area metric comparable across grids.
    ordered=np.sort(flow)[::-1];cells=.1*n;whole=int(cells)
    top=(ordered[:whole].sum()+(cells-whole)*ordered[whole])/flow.sum()
    return dict(min_over_mean=float(normal.min()),max_over_mean=float(normal.max()),
        active_fraction=float(np.mean(flow>0)),cv=float(normal.std()),
        total_participation_width_fraction=float(1/np.mean(normal**2)),
        top_10_percent_area_flow_share=float(top),
        positive_excess_integral_fraction=float(excess.mean()/flow.mean()),
        positive_excess_participation_width_fraction=float(excess.mean()**2/np.mean(excess**2)) if excess.max()>0 else None,
        positive_excess_half_max_area_fraction=float(np.mean(excess>=.5*excess.max())) if excess.max()>0 else None)


def profile(case):
    c=setup(case['n'],case['weakening']);k=np.asarray(case['final_kappa']);bar=smooth(c,k)
    strength=c['c0']*(.4+.6*np.maximum(1-bar/5,0)) if case['weakening'] else c['c0']
    flow=np.maximum(case['tau']-strength,0)/.1
    return dict(n=case['n'],weakening=case['weakening'],dt=case['dt_reference_units'],
        final_bar_kappa_min=float(bar.min()),final_bar_kappa_max=float(bar.max()),
        final_residual_fraction=float(np.mean(bar>=5)) if case['weakening'] else None,
        initial_harmonic_filter_factors=c['multiplier'][1:4].tolist(),
        peak=metrics(case['peak_plastic_rate']),final=metrics(flow),
        final_flow=flow.tolist())


def reconstruct(case):
    """Final JSON omitted gamma: reconstruct using the exact uniform elastic/viscous part.

    This is explicitly a reconstructed diagnostic state, not a planetary restart.
    Mean gamma equals imposed rate*time; gamma_i-kappa_i is spatially uniform.
    """
    c=setup(case['n'],case['weakening']);s=initial(c)
    k=np.asarray(case['final_kappa']);t=case['history'][-1]['time']
    s.update(time=t,tau=case['tau'],kappa=k.tolist(),gamma=(k+t-k.mean()).tolist(),
             history=case['history'],peak_cv=case['peak_cv'],peak_time=case['peak_time'],
             peak_plastic_rate=case['peak_plastic_rate'],peak_kappa_cv=case['peak_kappa_cv'],
             minimum_effective_width_fraction=case['minimum_effective_width_fraction'])
    s['ledger']={key:case['ledger'][key] for key in s['ledger']}
    return c,s


def continuation(case,deadline):
    c,s=reconstruct(case);start=json.loads(json.dumps(s));dt=case['dt_reference_units']
    ep=PARAMETERS['plastic_viscosity_Gt_units'];tm=PARAMETERS['maxwell_time_reference_units']
    yield_=c['c0']*(.4 if c['weakening'] else 1.)
    tau_eq=(yield_.mean()+ep)/(1+ep/tm)
    relaxation=1/(1/tm+1/ep)
    assert np.all(smooth(c,np.array(s['kappa']))>=5) or not c['weakening']
    assert tau_eq>yield_.max() # analytic all-active branch is justified, not assumed
    analytic_tau=s['tau'];analytic_k=np.array(s['kappa']);max_error=0.;snapshots=[]
    count=round(2/dt);saved=None
    for j in range(count):
        if time.monotonic()>deadline:raise TimeoutError('300 second budget')
        s=advance(c,s,dt)
        analytic_tau=(analytic_tau+dt*(1+yield_.mean()/ep))/(1+dt/tm+dt/ep)
        analytic_k+=dt*(analytic_tau-yield_)/ep
        max_error=max(max_error,abs(s['tau']-analytic_tau),float(np.max(abs(np.array(s['kappa'])-analytic_k))))
        if j+1==count//2:saved=json.loads(json.dumps(s))
        if j+1 in (count//2,count):
            snapshots.append(dict(time=s['time'],tau=s['tau'],flow=metrics((s['tau']-yield_)/ep),
                                  kappa_std=float(np.std(s['kappa']))))
    assert max_error<1e-11
    path=ROOT/('shear_late_restart_B.json' if c['weakening'] else 'shear_late_restart_A.json')
    path.write_text(json.dumps(dict(parameters=PARAMETERS,n=c['n'],weakening=c['weakening'],dt=dt,state=saved),indent=2)+'\n')
    loaded=json.loads(path.read_text());assert loaded['parameters']==PARAMETERS
    cc=setup(loaded['n'],loaded['weakening']);r=loaded['state']
    for _ in range(count//2,count):
        if time.monotonic()>deadline:raise TimeoutError('restart budget')
        r=advance(cc,r,dt)
    exact=json.dumps(r,sort_keys=True)==json.dumps(s,sort_keys=True);assert exact
    ledger={k:s['ledger'][k]-start['ledger'][k] for k in s['ledger']}
    ledger['elastic_change']=.5*(s['tau']**2-start['tau']**2)
    ledger['balance']=ledger['elastic_change']+ledger['maxwell']+ledger['plastic']+ledger['numerical']-ledger['external']
    assert abs(ledger['balance'])<1e-10
    return dict(weakening=c['weakening'],start_time=start['time'],end_time=s['time'],dt=dt,
        n=c['n'],analytic_tau_equilibrium=float(tau_eq),analytic_relaxation_time=float(relaxation),
        continuation_relaxation_times=2/relaxation,max_analytic_state_error=max_error,
        restart_exact=exact,ledger_increment=ledger,max_step_energy_residual=s['max_balance'],
        max_mean_strain_error=s['max_mean_error'],snapshots=snapshots,
        final_kappa=s['kappa'],final_flow=((s['tau']-yield_)/ep).tolist())


def main():
    start=time.monotonic();source=ROOT/'shear_diagnostic_result.json'
    protected=[source,ROOT/'shear_diagnostic_restart.json',
        Path('/workspace/genesis-moving-basal-source-01.json'),
        Path('/workspace/genesis-moving-basal-pilot-01/checkpoint.json')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    result=json.loads(source.read_text());assert result['parameters']==PARAMETERS
    profiles=[profile(c) for c in result['cases']]
    restart=json.loads((ROOT/'shear_diagnostic_restart.json').read_text())
    bar=smooth(setup(restart['n'],True),np.array(restart['state']['kappa']))
    late=[continuation(c,start+300) for c in result['cases'][:2]]
    for p in protected:assert hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)]
    output=dict(parameters=PARAMETERS,continuation_duration_reference_units=2.,seconds=time.monotonic()-start,
        saved_time6_residual_fraction=float(np.mean(bar>=5)),saved_time6_min_bar_kappa=float(bar.min()),
        profiles=profiles,late=late,protected_hashes=hashes)
    (ROOT/'shear_late_result.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(seconds=output['seconds'],profiles=[{k:v for k,v in c.items() if k!='final_flow'} for c in profiles],
         late=[{k:v for k,v in c.items() if k not in ('final_kappa','final_flow')} for c in late]),indent=2))


if __name__=='__main__':main()
