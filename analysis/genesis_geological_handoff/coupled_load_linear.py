"""Frozen-coefficient load response, with zero perturbation history only.

This is a linear response calculation, not a reset of geological history.
"""
from copy import deepcopy
import json,time
import numpy as np
from tectonics.experimental_angular_coupled import AngularCoupled,mechanical
from tectonics.simplified_genesis_nodal import material_mass
from .coupled_load_audit import SOURCE,FINE,vertex_traction,prolongation,with_traction,rms
from .angular_coupled_check import world_stress
from .edge_spatial_audit import ROOT


def response(a,traction):
    m=deepcopy(a.model)
    # Homogeneous initial perturbation; all material/thermal coefficients remain
    # those of the actual initial state. No state is saved as a trajectory.
    m.state['angular_memory']=np.zeros_like(m.state['angular_memory']).tolist()
    m.state['memory']=np.zeros_like(m.state['memory']).tolist()
    m.state['nodal_velocity']=np.zeros_like(m.state['nodal_velocity']).tolist()
    T=m.temperature(m.state['energy']);dt=.0625*31557600.
    basal=with_traction(m,traction);M=material_mass(m)
    r=mechanical(m,T,np.array(m.state['damage']),T,dt,M,basal)
    m.state['angular_memory']=r['angular_memory'];m.state['memory']=np.asarray(r['memory']).reshape(m.nf,2,6).tolist()
    K=np.zeros_like(M)
    for block in r['blocks']:
        H,L,R=map(block.get,('H','L','R'));idx=block['dofs']
        S=np.linalg.inv(np.eye(len(R))+dt*R)
        K[np.ix_(idx,idx)]+=L.T@H@S@L
    return world_stress(m),r['velocity'],K,r['ledger']


def main():
    if (ROOT/'coupled_load_linear.json').exists():raise FileExistsError('Preserve previous diagnostic')
    start=time.monotonic();a=AngularCoupled.load(SOURCE);b=AngularCoupled.load(FINE)
    P=prolongation(a.model.mesh,b.model.mesh);P6=np.kron(P,np.eye(6))
    tc,_=vertex_traction(a.model);tf,_=vertex_traction(b.model)
    cs,cv,ck,cl=response(a,tc);fs,fv,fk,fl=response(b,P@tc);ns,nv,nk,nl=response(b,tf)
    V=a.model.G[:,0,0];restricted=lambda x:x.reshape(80,4,3,3).mean(1)
    value=dict(dt_years=.0625,coarse_response_rms_pa=rms(cs,V),shared_fine_response_rms_pa=rms(restricted(fs),V),
        native_fine_response_rms_pa=rms(restricted(ns),V),
        shared_response_grid_difference_relative=rms(cs-restricted(fs),V)/rms(restricted(fs),V),
        native_response_grid_difference_relative=rms(cs-restricted(ns),V)/rms(restricted(ns),V),
        relaxation_stiffness_restriction_relative=float(np.linalg.norm(P6.T@fk@P6-ck)/np.linalg.norm(ck)),
        coarse_ledger=cl,shared_fine_ledger=fl,native_fine_ledger=nl,seconds=time.monotonic()-start)
    (ROOT/'coupled_load_linear.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({k:v for k,v in value.items() if not k.endswith('ledger')},indent=2))

if __name__=='__main__':main()
