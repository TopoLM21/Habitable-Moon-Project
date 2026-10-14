"""Exact tangential/normal split of the fixed load on actual physical facets."""
import json,time
import numpy as np
from tectonics.experimental_angular_coupled import AngularCoupled
from .coupled_load_audit import SOURCE,FINE,vertex_traction,prolongation,rms,with_traction,load_summary
from tectonics.simplified_genesis_basal import facet
from copy import deepcopy
from .linear_response_resolution import prepare,stiffness,scaled_solve,stress,YEAR
from .edge_spatial_audit import ROOT


def projected_resultants():
    a=AngularCoupled.load(SOURCE);t,_=vertex_traction(a.model);base=with_traction(a.model,t)
    result={'full':load_summary(base)}
    for name,normal in [('tangent',False),('normal',True)]:
        b=deepcopy(base);F=np.zeros_like(b['F'])
        for f,idx in enumerate(b['dofs']):
            old=b['faces'][f];Q=np.eye(3)-old['P'] if normal else old['P']
            row=facet(old['bottom'],old['h'],old['beta'],t[a.model.mesh.faces[f]]@Q,old['vm'])
            b['faces'][f]=row;np.add.at(F,idx,row['Ft'])
        b['F']=F;b['Ft']=F;result[name]=load_summary(b)
    return result


def main():
    out=ROOT/'linear_traction_components.json'
    if out.exists():raise FileExistsError('Preserve previous component diagnosis')
    start=time.monotonic();a=AngularCoupled.load(SOURCE);b=AngularCoupled.load(FINE)
    P0=prolongation(a.model.mesh,b.model.mesh);P=np.kron(P0,np.eye(6));tc,_=vertex_traction(a.model)
    pc=prepare(a,tc);pf=prepare(b,P0@tc);dt=.0625*YEAR;V=a.model.G[:,0,0]
    results={};fields={};known={}
    for name,p,t in [('coarse',pc,tc),('fine',pf,P0@tc)]:
        K,blocks=stiffness(p,dt);A=p['M']/dt+p['D']+dt*K
        velocity=np.zeros((p['m'].mesh.vertex_count,6));velocity[:,:3]=t/1e14;velocity=velocity.ravel()
        # Independent tangential equilibrium: beta*P*v = P*traction.
        Ft=p['D']@velocity;Fn=p['F']-Ft
        responses={};velocities={}
        for kind,F in [('tangent',Ft),('normal',Fn),('full',p['F'])]:
            v,residual=scaled_solve(A,F);velocities[kind]=v
            sigma=stress(p,blocks,v,dt)
            if name=='fine':sigma=sigma.reshape(80,4,3,3).mean(1)
            responses[kind]=sigma
            results[name+'_'+kind]=dict(stress_rms_pa=rms(sigma,V),power_w=float(F@v),force_norm_n=float(np.linalg.norm(F)),residual=residual)
        known[name]=dict(pure_drag_known_velocity_m_s_norm=float(np.linalg.norm(velocity)),
            full_system_tangent_velocity_error=float(np.linalg.norm(velocities['tangent']-velocity)/np.linalg.norm(velocity)),
            superposition_relative=float(np.linalg.norm(velocities['full']-velocities['normal']-velocities['tangent'])/np.linalg.norm(velocities['full'])))
        fields[name]=responses
    comparisons={}
    for kind in ('tangent','normal','full'):
        c=fields['coarse'][kind];f=fields['fine'][kind];delta=rms(c-f,V)
        comparisons[kind]=dict(absolute_difference_pa=delta,symmetric_relative=delta/max(rms(c,V),rms(f,V),1e-12))
    # Independent analytic affine strain patch (no matrix-generated expected strain).
    patch={};B=np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,0.]])*1e-10/np.mean(np.linalg.norm(np.asarray(pc['m'].config['nodal']['X']),axis=1))
    for name,p in [('coarse',pc),('fine',pf)]:
        X=np.asarray(p['m'].config['nodal']['X']);u=np.tile((X@B.T)[:,None,:],(1,2,1)).reshape(-1)
        frames=np.asarray(p['m'].config['nodal']['frames']);err=den=0.
        for j,bl in enumerate(p['blocks']):
            f=j//3;g=frames[f].T@B@frames[f]
            expected=np.r_[g[0,0],g[1,1],g[2,2],g[0,1]+g[1,0],g[0,2]+g[2,0],g[1,2]+g[2,1],np.zeros(6)]
            actual=bl['L']@u[bl['dofs']];err+=np.sum((actual-expected)**2);den+=np.sum(expected**2)
        patch[name]=dict(analytic_affine_strain_relative=float(np.sqrt(err/den)))
    result=dict(components=results,comparisons=comparisons,known_tangent_solution=known,analytic_patch=patch,seconds=time.monotonic()-start,
        interpretation='Exact force split on unchanged physical geometry, no rescaling or new trajectory; normal component was retained by original dead-traction closure.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
