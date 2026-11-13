"""Same physical forcing at each resolution; isolates quadrature from dynamics.

The arc kernel calculation diagnoses existing code without changing its law.
These are operator controls, not global new-plate events.
"""
from dataclasses import replace
import json,math,sys
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tectonics.mesh import build_icosphere
from tectonics.lithosphere import LithosphereState
from tectonics.volcanic_arc import _paint_gaussian
from tectonics.continental import ContinentalCycleParameters,initialize_continental_cycle,advance_continental_cycle

R=5287.;center=np.array([.37,.54,.755]);center/=np.linalg.norm(center)
physical=2*math.pi*R**2*quad(lambda theta:math.sin(theta)*math.exp(-.5*(R*theta/140.)**2),0,320./R,epsabs=1e-13)[0]
rows=[]
for sub in (1,2,3,4):
    mesh=build_icosphere(sub);n=mesh.cell_count;a=mesh.physical_cell_areas_km2(R)
    state=LithosphereState(0.,np.zeros(n,np.int32),np.zeros(n,np.int8),np.zeros(n),np.full(n,7.),np.zeros(n))
    scale=math.sqrt(float(np.mean(a)));sigma=max(140.,.45*scale);outer=max(320.,1.05*scale)
    field=np.zeros(n)
    _paint_gaussian(mesh,cKDTree(mesh.centroids),center,0,state,R,sigma,outer,1.,field)
    broadened=2*math.pi*R**2*quad(lambda theta:math.sin(theta)*math.exp(-.5*(R*theta/sigma)**2),0,outer/R,epsabs=1e-13)[0]
    rows.append({'cells':n,'cell_scale_km':scale,'effective_sigma_km':sigma,'effective_outer_km':outer,
                 'physical_kernel_integral_km2':physical,'broadened_continuum_integral_km2':broadened,
                 'discrete_integral_km2':float(a@field),'discrete_to_physical_ratio':float(a@field)/physical})

events=[];params=ContinentalCycleParameters(analytic_maturation=True)
rate=params.arc_maturation_rate_per_myr;tau=params.arc_potential_decay_myr
exact_time=-tau*math.log((rate*tau-params.juvenile_threshold)/(rate*tau-.95))
for sub in (1,2,3):
    mesh=build_icosphere(sub);n=mesh.cell_count;a=mesh.physical_cell_areas_km2(R)
    for dt in (4.,2.,1.):
        s=LithosphereState(0.,np.zeros(n,np.int32),np.zeros(n,np.int8),np.zeros(n),np.full(n,7.),np.zeros(n))
        s.oceanic_volume_km3=a*7;cycle=initialize_continental_cycle(mesh);cycle.felsic_potential[:]=.95
        when=[]
        for _ in range(int(8/dt)):
            s.time_myr+=dt
            s,cycle,d=advance_continental_cycle(mesh,s,[],cycle,dt,R,params,volcanic_arc_forcing=np.ones(n))
            if d.first_juvenile_event_time_myr is not None:when.append(d.first_juvenile_event_time_myr)
        events.append({'cells':n,'dt_myr':dt,'event_time_myr':when[0],'time_error_myr':when[0]-exact_time,
                       'final_thickness_km':float(np.mean(s.crust_thickness_km)),
                       'generated_volume_km3':cycle.cumulative_generated_volume_km3})
result={'source':'existing point-sampled mesh-broadened arc kernel; no calibrated physical inference',
        'arc_kernel':rows,'same_state_maturation':events,'analytic_event_time_myr':exact_time}
path=Path(__file__).with_name('operator_diagnostics.json')
with path.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps(result,indent=2))
