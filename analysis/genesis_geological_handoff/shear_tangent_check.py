"""Independent Jacobian of the full implicit diagnostic update; no trajectory."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from .shear_diagnostic import PARAMETERS,setup,initial,advance

ROOT=Path(__file__).parent


def check():
    start=time.monotonic();n=128;dt=.025;c=setup(n,True)
    assert PARAMETERS['plastic_viscosity_Gt_units']==.1
    # A mathematical active/unsaturated reference, not a new evolved checkpoint.
    state=initial(c);state.update(time=4.,tau=.8,kappa=[2.5]*n,gamma=[4.]*n)
    original=json.dumps(state,sort_keys=True)
    base=advance(c,state,dt)
    # Independent real-space FV matrix: do not use producer FFT multiplier.
    ell_h=PARAMETERS['ell_m']/(PARAMETERS['length_m']/n)
    lap=2*np.eye(n)-np.roll(np.eye(n),1,axis=1)-np.roll(np.eye(n),-1,axis=1)
    H=np.linalg.solve(np.eye(n)+ell_h**2*lap,np.eye(n))
    a=(1-PARAMETERS['residual_strength_fraction'])*c['c0']/PARAMETERS['critical_plastic_shear_units']
    b=dt/PARAMETERS['plastic_viscosity_Gt_units'];z=dt/PARAMETERS['maxwell_time_reference_units']
    matrix=np.zeros((n+1,n+1));matrix[:n,:n]=np.eye(n)-b*a[:,None]*H
    matrix[:n,n]=-b;matrix[n,:n]=1/n;matrix[n,n]=1+z
    bar=H@np.asarray(base['kappa'])
    strength=c['c0']*(.4+.6*(1-bar/5))
    assert bar.min()>0 and bar.max()<5 and base['tau']>strength.max()
    rows=[]
    for mode in (1,2,3,8,16,32):
        direction=np.cos(2*np.pi*mode*(np.arange(n)+.5)/n)
        answer=np.linalg.solve(matrix,np.r_[direction,direction.mean()])
        exact=answer[:n];tau_derivative=answer[-1]
        gamma_derivative=exact+(1+z)*tau_derivative
        Hmode=1/(1+4*ell_h**2*np.sin(np.pi*mode/n)**2)
        continuous=1.2*Hmode;homogeneous_gain=1/(1-dt*continuous)
        probes=[]
        for eps in (1e-3,5e-4,2.5e-4):
            if time.monotonic()-start>120:raise TimeoutError('120 second budget')
            outputs=[]
            for sign in (1,-1):
                s=json.loads(original)
                s['kappa']=(2.5+sign*eps*direction).tolist()
                s['gamma']=(4.+sign*eps*direction).tolist()
                out=advance(c,s,dt);outputs.append(out)
                kk=H@np.asarray(out['kappa']);yy=c['c0']*(.4+.6*(1-kk/5))
                assert kk.min()>0 and kk.max()<5 and out['tau']>yy.max()
            derivative=lambda key:(np.asarray(outputs[0][key])-np.asarray(outputs[1][key]))/(2*eps)
            d=derivative('kappa');gain=float(direction@d/(direction@direction))
            error=float(np.max(abs(d-exact)));terr=float(abs(derivative('tau')-tau_derivative))
            gerr=float(np.max(abs(derivative('gamma')-gamma_derivative)))
            assert max(error,terr,gerr)<1e-8 and gain>1
            probes.append(dict(epsilon=eps,gain_projection=gain,kappa_max_error=error,
                tau_derivative_error=terr,gamma_max_error=gerr,
                inferred_continuous_rate=(1-1/gain)/dt))
        rows.append(dict(mode=mode,wavelength_m=PARAMETERS['length_m']/mode,
            filter_factor=Hmode,homogeneous_continuous_rate_per_year=continuous,
            homogeneous_BE_gain=homogeneous_gain,analytic_heterogeneous_gain=float(direction@exact/(direction@direction)),
            coupling_relative_norm=float(np.linalg.norm(exact-homogeneous_gain*direction)/np.linalg.norm(exact)),probes=probes))
    assert all(rows[i]['probes'][0]['gain_projection']>rows[i+1]['probes'][0]['gain_projection'] for i in range(len(rows)-1))
    # A's kappa-independent strength must have neutral memory perturbations.
    ca=setup(n,False);s=json.loads(original);s['tau']=1.1
    direction=np.cos(2*np.pi*(np.arange(n)+.5)/n);eps=1e-3;out=[]
    for sign in (1,-1):
        x=json.loads(json.dumps(s));x['kappa']=(2.5+sign*eps*direction).tolist();x['gamma']=(4.+sign*eps*direction).tolist()
        out.append(advance(ca,x,dt))
    neutral=(np.asarray(out[0]['kappa'])-np.asarray(out[1]['kappa']))/(2*eps)
    neutral_error=float(np.max(abs(neutral-direction)));assert neutral_error<1e-9
    assert json.dumps(state,sort_keys=True)==original
    return dict(seconds=time.monotonic()-start,parameters=PARAMETERS,n=n,dt=dt,
        reference=dict(time_label=4.,tau_units=.8,kappa_units=2.5,gamma_units=4.,
                       role='Mathematical probe, not a time-evolved state; original spatial c0 retained.'),
        base_unsaturated_margin=float(5-bar.max()),base_yield_margin=float(base['tau']-strength.max()),
        modes=rows,A_neutral_error=neutral_error,reference_unchanged=True)


def main():
    paths=list(ROOT.glob('shear_*result.json'))+list(ROOT.glob('shear_*restart*.json'))
    paths += [Path('/workspace/genesis-moving-basal-source-01.json'),Path('/workspace/genesis-moving-basal-pilot-01/checkpoint.json')]
    paths=[p for p in paths if p.name!='shear_tangent_result.json']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result=check()
    for p in paths:assert hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)]
    result['preserved_hashes']=hashes
    (ROOT/'shear_tangent_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(seconds=result['seconds'],A_neutral_error=result['A_neutral_error'],
        modes=[dict(mode=r['mode'],rate=r['homogeneous_continuous_rate_per_year'],gain=r['probes'][-1]['gain_projection'],
                    max_error=max(p['kappa_max_error'] for p in r['probes']),coupling=r['coupling_relative_norm']) for r in result['modes']]),indent=2))


if __name__=='__main__':main()
