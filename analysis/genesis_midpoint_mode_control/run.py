"""Bounded frozen basal spectrum and production mode control; no evolution."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.linalg import cholesky,eigh,solve_triangular
from analysis.genesis_kinetic_diagnostic.run import PARENT,PARENT_SHA,read,matrix
from analysis.genesis_volume_continuation_validation.independent_audit import basal_drag_and_force
from .control import run_mode


def run():
    started=time.monotonic();packet,pin=read(PARENT,PARENT_SHA)
    s=packet['runtime_state'];g=s['geometry'];inp=s['last_report']['audit_input']
    M=matrix(g,g['cohorts'])
    D,_=basal_drag_and_force(g['vertices_m'],g['faces'],g['column_mass_kg'],g['density_kg_m3'],g['cohorts'],inp['basal_drag_pa_s_m'],np.asarray(inp['basal_traction_at_vertices_pa'])[np.asarray(g['faces'])])
    scale=1/np.sqrt(np.diag(M));Ms=M*scale[:,None]*scale[None,:];Ds=D*scale[:,None]*scale[None,:]
    L=cholesky(Ms,lower=True);B=solve_triangular(L,Ds,lower=True)
    B=solve_triangular(L,B.T,lower=True).T
    asym=float(np.linalg.norm(B-B.T,2));B=(B+B.T)/2
    w,U=eigh(B);norm=float(np.linalg.norm(B,2));residual=float(np.linalg.norm(B@U-U*w,2))
    # Resolution indicator from actual eigen residual plus floating-point scale;
    # not a certified interval or a physical error/step acceptance tolerance.
    numerical_scale=max(residual,asym,len(w)*np.finfo(float).eps*norm)
    if w[0]<=numerical_scale:raise ValueError('Smallest rate is not numerically separated from zero')
    v=np.asarray(s['velocity_m_s']);coeff=U.T@(L.T@(v/scale));K=float(.5*np.dot(coeff,coeff))
    cases={}
    for z in [.01,1.,2.,10.,1e4]:cases['dimensionless_z_'+str(z)]=run_mode(1.,z)
    for label,gamma in [('slowest',float(w[0])),('fastest',float(w[-1]))]:
        for years in [25,50,100]:cases[f'{label}_{years}yr']=run_mode(gamma,years*365.25*86400)
    # Check producer recurrence/energy, not agreement with continuous decay at
    # unresolved dt. These are machine arithmetic regression thresholds only.
    for c in cases.values():
        for row in c['rows']:
            if abs(row['update_error_over_initial'])>1e-12:raise ValueError('Production differs from midpoint equation')
            if abs(row['normalized_balance_residual'])>1e-12:raise ValueError('Manufactured discrete work balance failed')
    frozen=[]
    for years in [25,50,100]:
        dt=years*365.25*86400;z=w*dt;r=(2-z)/(2+z)
        frozen.append(dict(years=years,min_gamma_dt=float(z[0]),max_gamma_dt=float(z[-1]),
            min_amplification=float(min(r)),max_amplification=float(max(r)),
            numerical_kinetic_fraction=float(np.dot(coeff*r,coeff*r)/(2*K)),
            exact_kinetic_fraction_upper_log10=float(-2*w[0]*dt/np.log(10))))
    if hashlib.sha256(PARENT.read_bytes()).hexdigest()!=PARENT_SHA:raise ValueError('Canonical bytes changed')
    return dict(status='PASS_PRODUCTION_MIDPOINT_CONTROL',base_code_sha='fae2f4af71786b704f1a32ff793cd553d3ecf4a7',
        checkpoint=pin,seconds=time.monotonic()-started,
        spectrum=dict(scope='fixed canonical reference M and basal D only, not full coupled Jacobian',ndof=len(w),
            scaled_mass_condition=float(np.linalg.cond(Ms)),rate_min_per_s=float(w[0]),rate_max_per_s=float(w[-1]),
            decay_time_min_s=float(1/w[-1]),decay_time_max_s=float(1/w[0]),
            all_modes_resolved_positive=True,eigen_residual_spectral_norm_per_s=residual,
            whitened_asymmetry_norm_per_s=asym,numerical_resolution_scale_per_s=numerical_scale,
            rate_quantiles_per_s=np.quantile(w,[0,.1,.5,.9,1]).tolist(),
            canonical_kinetic_j=K,kinetic_projection_difference_j=K-s['kinetic_energy_j'],
            basal_drag_pa_s_m=inp['basal_drag_pa_s_m'],
            exact_frozen_transient_displacement_mass_rms_m=float(np.sqrt(np.sum((coeff/w)**2)/sum(c['mass_kg'] for c in g['cohorts']))),
            all_modes_no_sign_flip_dt_bound_seconds=float(2/w[-1]),
            slowest_mode_no_sign_flip_dt_bound_seconds=float(2/w[0]),
            decay_rates_per_s=w.tolist(),modal_kinetic_j=(.5*coeff**2).tolist()),
        frozen_basal_predictions=frozen,production_controls=cases,
        pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('tectonics/genesis_volume_dynamics.py'),Path('tectonics/genesis_volume_slip_dynamics.py'),Path('tectonics/genesis_volume_basal.py'),Path(__file__),Path(__file__).with_name('control.py')]},
        genesis_evolution=False,integrator_changed=False,physical_tolerance_assigned=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=run()
    with a.output.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps({k:r[k] for k in ['status','seconds','frozen_basal_predictions']}))
