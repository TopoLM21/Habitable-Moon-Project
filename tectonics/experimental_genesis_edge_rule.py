"""Explicit edge fatigue: incubation or opt-in competing vector-shell weakening."""
import numpy as np
from .experimental_edge_fracture import Parameters,certify
from .experimental_edge_weakening import MODE,enabled,scale as effective_scale,commit_damage


def update(model,previous,years,*,commit_solver=None):
    from .simplified_genesis_nodal import operators
    s=model.state;spec=model.config['experimental_geological_edge_fatigue']
    if spec not in [dict(shear_strength_pa=1e6,damage_time_years=100.,mode=mode) for mode in ('incubation_then_geological_retirement',MODE)]:
        raise ValueError('UNSUPPORTED_EXPERIMENTAL_EDGE_PARAMETERS')
    if not model.faults['enabled'] or not s['membrane_established']:
        raise ValueError('EXPERIMENTAL_EDGE_REQUIRES_ENABLED_ESTABLISHED_SHELL')
    sigma=[];scale=effective_scale(model,s['damage']);volume=model.weights.sum(axis=1).copy()
    for f in range(model.nf):
        e=model.P[f]@np.array(s['memory'][f]);stress=e@(model.C[f]*scale[f])
        sigma.append(np.sum(model.weights[f,:,None]*stress,axis=0))
    sigma=np.array(sigma)
    for x in s.get('born_material',[]):
        f=x['face'];sigma[f]+=x['volume_m3']*(model.C[f]*scale[f])@np.array(x['memory']);volume[f]+=x['volume_m3']
    sigma/=volume[:,None]
    tensor=np.zeros((model.nf,3,3));tensor[:,range(3),range(3)]=sigma[:,:3]
    for index,(i,j) in enumerate(((0,1),(0,2),(1,2)),3):tensor[:,i,j]=tensor[:,j,i]=sigma[:,index]
    frame=operators(model)['frames'];world=np.einsum('fai,fij,fbj->fab',frame,tensor,frame)
    stress=[]
    for f,g,i,j in model.mesh.shared_edges:
        a,b=model.mesh.vertices[[i,j]];radial=a+b;radial/=np.linalg.norm(radial)
        tangent=b-a;tangent/=np.linalg.norm(tangent);normal=np.cross(tangent,radial)
        traction=.5*(world[f]+world[g])@normal
        stress.append(float(np.linalg.norm(traction-normal*(traction@normal))))
    stress=np.array(stress);old=np.array(previous.get('experimental_edge_fatigue',{}).get('damage',np.zeros(len(stress))))
    if (old.shape!=stress.shape or not np.isfinite(old).all()
            or np.any((old<0)|(old>1)) or not np.isfinite(stress).all()):
        raise ValueError('EXPERIMENTAL_EDGE_INVALID_STATE')
    rate=np.maximum(stress/spec['shear_strength_pa']-1,0)**2/spec['damage_time_years']
    d=np.minimum(1,old+years*rate);owner,reason=certify(model.mesh,d>=1,Parameters())
    release=None
    if enabled(model):
        release=(commit_damage if commit_solver is None else commit_solver)(model,d)
        if 'stress_pa' in s:s['stress_pa']=(np.array(s['stress_pa'])*(release['scale_after']/release['scale_before'])[:,None]).tolist()
        ledger=s.get('last_ledger')
        if ledger is not None:
            ledger['edge_fracture_release_j']=release['release_j']
            ledger['edge_elastic_before_j']=release['elastic_before_j']
            ledger['edge_elastic_after_j']=release['elastic_after_j']
            ledger['elastic_after']-=release['release_j']
            ledger['edge_fracture_reserve_j']=s['experimental_edge_fracture_reserve_j']
            ledger['edge_damage_heat_j']=0.
            residual=release['elastic_after_j']-release['elastic_before_j']+release['release_j']
            if abs(residual)>1e-12*max(release['elastic_before_j'],1.):raise ValueError('EDGE_RELEASE_BALANCE')
            ledger['edge_substep_balance_j']=residual
            ledger['whole_balance_j']+=residual
            if abs(ledger['whole_balance_j'])>ledger['whole_allowance_j']:raise ValueError('EDGE_WHOLE_ENERGY_BALANCE')
    s['experimental_edge_fatigue']=dict(damage=d.tolist(),shear_traction_pa=stress.tolist(),
        reason=reason,partition_owner=None if owner is None else owner.tolist(),
        failed_edges=(d>=1).tolist(),mechanical_heat_j=0.,
        semantics=('competing vector-shell weakening; edge release archived in mechanical reserve, no fracture heat' if enabled(model) else 'irreversible incubation only; shared mechanics retired at accepted partition, no prefracture stiffness reduction'))
