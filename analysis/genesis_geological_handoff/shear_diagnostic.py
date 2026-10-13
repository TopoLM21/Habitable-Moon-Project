"""Independent periodic simple-shear VEVP experiment; no planetary imports.

Units internally: stress=tau_ref, strain=tau_ref/G, time=t_ref.
Positive monotone shear only; kappa is accumulated engineering plastic shear.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

PARAMETERS = dict(length_m=128000., ell_m=8000., shear_modulus_pa=2e10,
                  shear_reference_pa=1e6, time_reference_s=31557600.,
                  maxwell_time_reference_units=100., plastic_viscosity_Gt_units=.1,
                  residual_strength_fraction=.4, critical_plastic_shear_units=5.,
                  strength_perturbation_fraction=.001, imposed_shear_rate_units=1.,
                  end_time_units=12.)


def setup(n, weakening):
    x=(np.arange(n)+.5)/n
    # Analytic broad perturbation, identical at cell centres on every grid.
    seed=(np.cos(2*np.pi*x)+.5*np.sin(4*np.pi*x)+.25*np.cos(6*np.pi*x))/1.75
    c0=1-PARAMETERS['strength_perturbation_fraction']*seed
    ell=PARAMETERS['ell_m']/PARAMETERS['length_m']
    # Periodic finite-volume Helmholtz: exact eigenvalues of discrete Laplacian.
    multiplier=1/(1+4*(ell*n)**2*np.sin(np.pi*np.fft.rfftfreq(n))**2)
    return dict(n=n,weakening=weakening,c0=c0,multiplier=multiplier,seed=seed)


def smooth(c, k):
    return np.fft.irfft(np.fft.rfft(k)*c['multiplier'],n=c['n'])


def initial(c):
    return dict(time=0.,tau=0.,kappa=np.zeros(c['n']).tolist(),
                gamma=np.zeros(c['n']).tolist(),
                ledger=dict(external=0.,maxwell=0.,plastic=0.,numerical=0.,
                            damage=0.,drag=0.,thermal_work=0.),
                peak_cv=0.,minimum_effective_width_fraction=1.,
                peak_kappa_cv=0.,peak_time=0.,max_balance=0.,max_mean_error=0.,
                max_iterations=0,history=[])


def advance(c,s,dt):
    old=np.asarray(s['kappa']); trial=old.copy();tau0=s['tau']
    tm=PARAMETERS['maxwell_time_reference_units']
    ep=PARAMETERS['plastic_viscosity_Gt_units'];rate=PARAMETERS['imposed_shear_rate_units']
    residual_fraction=PARAMETERS['residual_strength_fraction']
    critical=PARAMETERS['critical_plastic_shear_units']
    for iteration in range(100):
        bar=smooth(c,trial)
        fraction=np.maximum(1-bar/critical,0.) if c['weakening'] else np.ones_like(bar)
        strength=c['c0']*(residual_fraction+(1-residual_fraction)*fraction)
        def residual(tau):
            return (1+dt/tm)*tau-tau0-dt*rate+np.mean(dt/ep*np.maximum(tau-strength,0.))
        tau=brentq(residual,0.,tau0+dt*rate,xtol=1e-14)
        dp=dt/ep*np.maximum(tau-strength,0.)
        new=old+dp
        if np.max(abs(new-trial))<2e-13:break
        trial=new
    else:raise AssertionError('nonlinear convergence')
    # Independent re-evaluation at converged endpoint.
    bar=smooth(c,new)
    strength=c['c0']*(residual_fraction+(1-residual_fraction)*np.maximum(1-bar/critical,0.)) if c['weakening'] else c['c0']
    assert np.max(abs(dp-dt/ep*np.maximum(tau-strength,0.)))<1e-11
    elastic_delta=.5*(tau*tau-tau0*tau0)
    wm=dt*tau*tau/tm;wp=tau*float(dp.mean());wn=.5*(tau-tau0)**2
    wx=tau*dt*rate
    balance=elastic_delta+wm+wp+wn-wx
    assert min(wm,wp,wn)>=0
    # All plastic work becomes heat; strength and overstress split is diagnostic only.
    assert abs(wp-np.mean(strength*dp+ep*dp*dp/dt))<1e-11
    gamma=np.asarray(s['gamma'])+(tau-tau0)+dt*tau/tm+dp
    mean_error=abs(gamma.mean()-(s['time']+dt)*rate)
    mean_dp=float(dp.mean());cv=float(dp.std()/mean_dp) if mean_dp>1e-10 else 0.
    width=1/(1+cv*cv)
    kcv=float(new.std()/new.mean()) if new.mean()>1e-8 else 0.
    out=json.loads(json.dumps(s))
    out.update(time=s['time']+dt,tau=float(tau),kappa=new.tolist(),gamma=gamma.tolist())
    for key,value in [('external',wx),('maxwell',wm),('plastic',wp),('numerical',wn)]:
        out['ledger'][key]+=value
    # Width/CV of incipient flow at infinitesimal mean flow is misleading.
    if mean_dp/dt>.1:
        if cv>out['peak_cv']:
            out.update(peak_cv=cv,peak_time=out['time'],peak_plastic_rate=(dp/dt).tolist())
        out['minimum_effective_width_fraction']=min(out['minimum_effective_width_fraction'],width)
    out['peak_kappa_cv']=max(out['peak_kappa_cv'],kcv if new.mean()>.1 else 0.)
    out['max_balance']=max(out['max_balance'],abs(balance))
    out['max_mean_error']=max(out['max_mean_error'],float(mean_error))
    out['max_iterations']=max(out['max_iterations'],iteration+1)
    assert abs(balance)<1e-10 and mean_error<1e-9
    return out


def run(n,dt,weakening,deadline,restart=False):
    c=setup(n,weakening);s=initial(c);count=round(PARAMETERS['end_time_units']/dt)
    saved=None
    for j in range(count):
        if time.monotonic()>deadline:raise TimeoutError('300 second experiment budget')
        s=advance(c,s,dt)
        if (j+1)%round(.5/dt)==0:
            s['history'].append(dict(time=s['time'],tau=s['tau'],kappa_mean=float(np.mean(s['kappa'])),
                                     kappa_std=float(np.std(s['kappa'])),peak_cv=s['peak_cv']))
        if j+1==count//2:saved=json.dumps(s,sort_keys=True,allow_nan=False)
    exact=None
    if restart:
        restart_path=Path(__file__).with_name('shear_diagnostic_restart.json')
        restart_path.write_text(json.dumps(dict(schema=1,parameters=PARAMETERS,
            n=n,weakening=weakening,dt=dt,state=json.loads(saved)),sort_keys=True,indent=2)+'\n')
        payload=json.loads(restart_path.read_text())
        assert payload['parameters']==PARAMETERS and payload['dt']==dt
        c=setup(payload['n'],payload['weakening']);r=payload['state']
        for j in range(count//2,count):
            if time.monotonic()>deadline:raise TimeoutError('restart budget')
            r=advance(c,r,dt)
            if (j+1)%round(.5/dt)==0:
                r['history'].append(dict(time=r['time'],tau=r['tau'],kappa_mean=float(np.mean(r['kappa'])),
                                         kappa_std=float(np.std(r['kappa'])),peak_cv=r['peak_cv']))
        exact=json.dumps(r,sort_keys=True)==json.dumps(s,sort_keys=True)
        assert exact
    ledger=s['ledger'];ledger['elastic_final']=.5*s['tau']**2
    ledger['heat']=ledger['maxwell']+ledger['plastic']
    ledger['whole_balance']=ledger['elastic_final']+ledger['heat']+ledger['numerical']-ledger['external']
    assert abs(ledger['whole_balance'])<1e-9
    k=np.asarray(s['kappa']);seed=c['seed'];h=PARAMETERS['length_m']/n
    result=dict(n=n,dt_reference_units=dt,weakening=weakening,
                cells_per_ell=PARAMETERS['ell_m']/h,periodic_kernel_support_cells=n,
                kernel_mass_within_ell=float(np.sum(np.fft.irfft(c['multiplier'],n=n)[
                    np.minimum(np.arange(n),n-np.arange(n))*h<=PARAMETERS['ell_m']])) ,
                restart_exact=exact,kappa_mean=float(k.mean()),kappa_std=float(k.std()),
                kappa_projection_on_seed=float(np.mean((k-k.mean())*seed)/np.mean(seed*seed)),
                **{key:s[key] for key in ('tau','peak_cv','peak_time','peak_kappa_cv',
                    'minimum_effective_width_fraction','max_balance','max_mean_error','max_iterations','history')},
                ledger=ledger,final_kappa=k.tolist(),peak_plastic_rate=s.get('peak_plastic_rate'))
    return result


def main():
    start=time.monotonic();deadline=start+300
    frozen=[Path('/workspace/genesis-moving-basal-source-01.json'),
            Path('/workspace/genesis-moving-basal-pilot-01/checkpoint.json')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen}
    # One declared set, no parameter search: A/B, finer A/B, half-step B.
    cases=[run(128,.025,False,deadline),run(128,.025,True,deadline,True),
           run(256,.025,False,deadline),run(256,.025,True,deadline),run(128,.0125,True,deadline)]
    for p in frozen:assert hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)]
    a,b,af,bf,bt=cases
    def relative(x,y):return abs(x-y)/max(abs(y),1e-14)
    result=dict(parameters=PARAMETERS,seconds=time.monotonic()-start,cases=cases,
                preserved_checkpoints=hashes,
                comparisons=dict(B_over_A_final_kappa_std=b['kappa_std']/a['kappa_std'],
                    B_over_A_peak_cv=b['peak_cv']/a['peak_cv'],
                    mesh_relative_kappa_std=relative(bf['kappa_std'],b['kappa_std']),
                    timestep_relative_kappa_std=relative(bt['kappa_std'],b['kappa_std']),
                    mesh_relative_effective_width=relative(bf['minimum_effective_width_fraction'],b['minimum_effective_width_fraction']),
                    timestep_relative_effective_width=relative(bt['minimum_effective_width_fraction'],b['minimum_effective_width_fraction'])))
    destination=Path(__file__).with_name('shear_diagnostic_result.json')
    destination.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','preserved_checkpoints')},indent=2))
    print(json.dumps([{k:v for k,v in c.items() if k not in ('history','final_kappa','peak_plastic_rate')} for c in cases],indent=2))


if __name__=='__main__':main()
