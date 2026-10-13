"""Fixed-scale field and transfer sensitivity; independent vector-shell fixture."""
import json,time
from pathlib import Path
import numpy as np
from tectonics.simplified_genesis import SimplifiedGenesis
from tectonics.experimental_edge_weakening import face_damage
from .edge_weakening_controls import GridControl
from .nodal_retirement_check import SOURCE
ROOT=Path(__file__).parent

def main():
    start=time.monotonic();source=SimplifiedGenesis.load(SOURCE)
    # Continuous degree-2 representation with fixed coefficients from original field.
    def design(x):return np.column_stack((np.ones(len(x)),x,x[:,0]*x[:,1],x[:,0]*x[:,2],x[:,1]*x[:,2],x[:,0]**2-x[:,1]**2,3*x[:,2]**2-1))
    coeff=np.linalg.lstsq(design(source.mesh.centroids),source.state['damage'],rcond=None)[0]
    def field(x):return np.clip(design(x)@coeff,0,1)
    approximation=float(np.sqrt(np.mean((field(source.mesh.centroids)-source.state['damage'])**2)))
    runs=[];profiles=[];parent=None
    for level,mapping in [(2,'analytic'),(3,'analytic'),(3,'parent_hold')]:
        g=GridControl(level);old=field(g.mesh.centroids) if mapping=='analytic' else np.repeat(parent,4)
        if level==2:parent=old.copy()
        oldratio=.02+.98*(1-g.old)**2;newratio=.02+.98*(1-old)**2
        g.a.C*=np.repeat(newratio/oldratio,2)[:,None,None];g.old=old;g.base_scale=newratio
        s=g.initial();clock=time.monotonic();s,row=g.step(s,.5);benchmark=time.monotonic()-clock
        print(json.dumps(dict(faces=g.a.nf,mapping=mapping,first_step_seconds=benchmark,sparse_bytes=g.a.M.data.nbytes+g.a.M.indices.nbytes+g.a.M.indptr.nbytes)),flush=True)
        rows=[row]
        for _ in range(39):
            if time.monotonic()-start>120:raise RuntimeError('Predeclared120s spatial cap')
            s,row=g.step(s,.5);rows.append(row)
        d=face_damage(g.mesh,s['d']);profiles.append(d)
        runs.append(dict(faces=g.a.nf,mapping=mapping,max_edge_damage=max(s['d']),mean_face_damage=float(g.a.area@d),max_face_damage=float(d.max()),
            saturated=int(sum(np.array(s['d'])>=1)),reserve_j=s['reserve'],energy=max(r['energy_relative'] for r in rows),
            spatial_profile=d.tolist(),seconds=time.monotonic()-clock,first_step_seconds=benchmark))
    coarse=profiles[0];fine=profiles[1].reshape(-1,4).mean(axis=1);transfer=profiles[2]
    result=dict(scope='20year manufactured vector shell, fixed degree2 old-damage/traction length scale; not remapped real planetary evolution',
        horizon_years=20,dt_years=.5,original_field_representation_rms=approximation,coefficients=coeff.tolist(),runs=runs,
        restricted_fine_coarse_rms=float(np.sqrt(np.mean((fine-coarse)**2))),restricted_fine_coarse_relative_l2=float(np.linalg.norm(fine-coarse)/np.linalg.norm(fine)),
        same_fine_grid_transfer_relative_l2=float(np.linalg.norm(transfer-profiles[1])/np.linalg.norm(profiles[1])),
        max_projection_width='one incident face; halves with mesh refinement, no physical regularization length',seconds=time.monotonic()-start)
    from tectonics.mesh import build_icosphere
    areas=build_icosphere(3).areas_unit_sphere.reshape(-1,4);weights=areas.sum(axis=1);weights/=weights.sum()
    restricted=(profiles[1].reshape(-1,4)*areas).sum(axis=1)/areas.sum(axis=1)
    result['area_weighted_restricted_relative_l2']=float(np.sqrt(np.sum(weights*(restricted-coarse)**2)/np.sum(weights*restricted**2)))
    (ROOT/'edge_spatial_control.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='runs'}),flush=True)
if __name__=='__main__':main()
