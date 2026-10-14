"""Read-only audit of saved common-quadrature spherical trajectories."""
from pathlib import Path
import json,time
import numpy as np
from tectonics.genesis_transfer_bridge import digest
from .spherical_geometry_path import SphericalPath
from .local_spherical_drag_reference import quadrature
from .edge_spatial_audit import ROOT
OUT=Path('/workspace/genesis-spherical-geometry-analytic-common-01')


def tensor(s):
    t=np.zeros(s.shape[:-1]+(3,3));t[...,range(3),range(3)]=s[...,:3]
    for k,(i,j) in enumerate(((0,1),(0,2),(1,2)),3):t[...,i,j]=t[...,j,i]=s[...,k]
    return t


def decompose(T,n):
    normal=np.einsum('...i,...ij,...j->...',n,T,n)
    shear=np.einsum('...ij,...j->...i',T,n)-normal[...,None]*n
    tangential=T-normal[...,None,None]*n[..., :,None]*n[...,None,:]-shear[..., :,None]*n[...,None,:]-n[..., :,None]*shear[...,None,:]
    return normal**2,2*np.sum(shear**2,axis=-1),np.sum(tangential**2,axis=(-1,-2))


def main():
    path=ROOT/'spherical_practical_audit.json'
    if path.exists():raise FileExistsError('Preserve audit')
    start=time.monotonic();inputs={p.name:digest(p.read_bytes()) for p in OUT.glob('*.json')}
    c=SphericalPath.load(OUT/'coarse_final.json');f=SphericalPath.load(OUT/'fine_final.json')
    q=len(f.memory)//320//2
    cs=tensor(np.einsum('qij,qj->qi',c.Cs,c.memory)).reshape(80,2,4,q,3,3)
    fs=tensor(np.einsum('qij,qj->qi',f.Cs,f.memory)).reshape(80,4,2,q,3,3).transpose(0,2,1,3,4,5)
    cw=c.weights.reshape(80,2,4,q);fw=f.weights.reshape(80,4,2,q).transpose(0,2,1,3)
    assert np.max(abs(cw-fw)/cw)<1e-12
    normals=np.array([quadrature(f.X[ids],5)[1] for ids in f.mesh.faces]).reshape(80,4,q,3)
    normals=np.broadcast_to(normals[:,None],(80,2,4,q,3));weights=cw/cw.sum();delta=cs-fs
    norm=lambda T:float(np.sqrt(np.sum(weights*np.sum(T*T,axis=(-1,-2)))))
    decomposition={}
    for name,T in [('coarse',cs),('fine',fs),('difference',delta)]:
        parts=decompose(T,normals);values=[float(np.sum(weights*x)) for x in parts]
        decomposition[name]=dict(total_rms_pa=norm(T),radial_normal_rms_pa=np.sqrt(values[0]),mixed_shear_frobenius_rms_pa=np.sqrt(values[1]),tangential_tensor_rms_pa=np.sqrt(values[2]),squared_norm_fractions=(np.array(values)/sum(values)).tolist())
    point_sq=np.sum(delta**2,axis=(-1,-2));face_sq=np.sum(cw*point_sq,axis=(1,2,3))/c.volumes
    contributions=np.sum(weights*point_sq,axis=(1,2,3));order=np.argsort(contributions)[::-1]
    means_c=np.einsum('frak,frakij->fij',cw,cs)/c.volumes[:,None,None]
    means_f=np.einsum('frak,frakij->fij',cw,fs)/c.volumes[:,None,None]
    mean_delta=means_c-means_f
    face_principal=np.linalg.eigvalsh(mean_delta)
    point_max=float(np.sqrt(point_sq.max()));mean_sq=np.sum(mean_delta**2,axis=(1,2));V=c.volumes/c.volumes.sum()
    point_r=norm(delta)/max(norm(cs),norm(fs),1e-12)
    r=json.loads((ROOT/'spherical_geometry_check_analytic_common.json').read_text())
    rows_c=r['runs']['coarse'];rows_f=r['runs']['fine'];Wc=sum(x['external_work'] for x in rows_c);Wf=sum(x['external_work'] for x in rows_f)
    result=dict(source_code='537e37ec79dfc750e1c3c3670ba964afb8261116',read_only=True,inputs=inputs,
        pointwise_quadrature=decomposition,pointwise_symmetric_relative=point_r,
        mean_tensor_rms_difference_pa=float(np.sqrt(V@mean_sq)),
        unresolved_within_parent_difference_rms_pa=float(np.sqrt(max(0.,norm(delta)**2-V@mean_sq))),
        maximum_point_frobenius_difference_pa=point_max,maximum_parent_mean_frobenius_difference_pa=float(np.sqrt(mean_sq.max())),
        maximum_parent_mean_principal_difference_bound_pa=float(abs(face_principal).max()),
        top_eight_fraction=float(contributions[order[:8]].sum()/contributions.sum()),
        top_faces=[dict(face=int(i),rms_difference_pa=float(np.sqrt(face_sq[i])),fraction=float(contributions[i]/contributions.sum()),centroid=(c.X[c.mesh.faces[i]].mean(0)/np.linalg.norm(c.X[c.mesh.faces[i]].mean(0))).tolist()) for i in order[:8]],
        external_work=dict(coarse_j=Wc,fine_j=Wf,relative_difference=abs(Wc-Wf)/max(abs(Wc),abs(Wf))),
        fine_energy=dict(elastic_j=rows_f[-1]['elastic_after'],kinetic_j=rows_f[-1]['kinetic_after'],heat_buffer_j=f.heat,numerical_j=f.cumulative_numerical),
        cost=dict(previous_final_series_seconds=r['seconds'],step_seconds_fine=[x['seconds'] for x in rows_f],audit_seconds=time.monotonic()-start),
        convergence='Only80/320 pair, quadrature order5/7 and temporal pair; no spatial order or asymptotic convergence established.')
    assert all(digest((OUT/name).read_bytes())==sha for name,sha in inputs.items())
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
