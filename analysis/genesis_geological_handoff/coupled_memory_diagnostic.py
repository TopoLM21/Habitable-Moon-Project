"""Read saved alternative memory and diagnose rejected zero-load control."""
from pathlib import Path
import json,time
import numpy as np
from tectonics.experimental_angular_coupled import AngularCoupled
from .coupled_load_audit import SOURCE,FINE,prescribed,basal,rms
from .coupled_spatial_check import field
from .edge_spatial_audit import ROOT


def main():
    if (ROOT/'coupled_memory_diagnostic.json').exists():raise FileExistsError('Preserve previous diagnostic')
    start=time.monotonic();a=AngularCoupled.load(FINE)
    alt=AngularCoupled.load('/workspace/genesis-coupled-spatial-01/fine_alternative_final.json')
    V=a.model.G[:,0,0];initial=field(a.model)['stress'];alternative_initial=field(alt.reference.m)['stress']
    saved=np.load('/workspace/genesis-coupled-spatial-01/fields.npz')
    delta0=alternative_initial-initial
    final=saved['fine_full_0.125_stress'];delta1=saved['fine_alternative_0.125_stress']-final
    info=dict(primary_initial_stress_rms_pa=rms(initial,V),alternative_initial_difference_rms_pa=rms(delta0,V),
        alternative_final_difference_rms_pa=rms(delta1,V),alternative_difference_evolution_rms_pa=rms(delta1-delta0,V),
        primary_stress_evolution_rms_pa=rms(final-initial,V),
        initial_alternative_mapping=alt.origin['mapping'])
    original=basal.exchange;failure={}
    def inspect(face,v,dt):
        try:return original(face,v,dt)
        except ValueError as e:
            v=np.asarray(v).reshape(3,6);vb=v[:,:3]+face['h']*v[:,3:]
            d=(vb-face['vm'])@face['P'];r=face['beta']*face['area']*(basal.W@d)
            Q=float(dt*np.einsum('ij,ij->',r,d));crust=float(-dt*np.einsum('ij,ij->',r,vb));mantle=float(dt*np.einsum('ij,ij->',r,face['vm']))
            failure.update(error=str(e),heat_j=Q,crust_work_j=crust,mantle_work_j=mantle,
                work_identity_residual_j=Q+crust+mantle,allowance_j=1e-12*max(abs(Q),abs(crust),abs(mantle),1.),
                velocity_norm=float(np.linalg.norm(vb)),tangent_velocity_norm=float(np.linalg.norm(d)))
            raise
    basal.exchange=inspect
    try:
        with prescribed(np.zeros((a.model.mesh.vertex_count,3))):
            for i in range(2):
                try:a.advance(.0625)
                except ValueError:
                    if not failure:raise
                    failure['step']=i+1;break
    finally:basal.exchange=original
    info['zero_load_guard']=failure;info['seconds']=time.monotonic()-start
    (ROOT/'coupled_memory_diagnostic.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info,indent=2))

if __name__=='__main__':main()
