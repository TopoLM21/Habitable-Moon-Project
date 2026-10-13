"""Frictionless hard contact on a filled, fixed spherical material partition.

Normal constraints have no stiffness, friction or tensile strength. Full coverage
and fixed areas imply zero normal separation flux: all boundary normal velocities
must vanish. This is a closed-packing consequence, not a tangential constraint.
Only motions with a whole-time geometric certificate are committed. General
noncommuting contact paths stop rather than silently sample their endpoints.
"""
import numpy as np
from scipy.linalg import block_diag,null_space
from scipy.optimize import nnls
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

ROUND=512*np.finfo(float).eps


class ContactPathUnsupported(ValueError):pass


def unilateral_projection(B,rhs,J):
    """Independent normal-only QP kernel: J v >= 0; opening/sliding stay free.

    B SPD, rhs and J are caller-scaled SI or consistent generalized quantities.
    Returns v and compressive multipliers, never tensile contact reactions.
    """
    B=np.asarray(B,float);rhs=np.asarray(rhs,float);J=np.asarray(J,float)
    L=np.linalg.cholesky(B);free=np.linalg.solve(B,rhs)
    if not len(J):return free,np.empty(0)
    A=np.linalg.solve(L,J.T)
    lam,_=nnls(A,-L.T@free,maxiter=max(100,20*len(J)))
    v=free+np.linalg.solve(B,J.T@lam)
    scale=max(np.linalg.norm(free),np.linalg.norm(v),1.)
    if np.min(J@v)<-1e-10*scale:raise ValueError('CONTACT_QP_FEASIBILITY')
    return v,lam


def reference_partition(mesh):
    """Convex hull facets around the origin give a nonoverlapping radial tiling."""
    X=mesh.vertices;F=mesh.faces;hull=ConvexHull(X)
    if {tuple(sorted(x)) for x in F}!={tuple(sorted(x)) for x in hull.simplices}:raise ContactPathUnsupported('REFERENCE_NOT_CONVEX_SPHERICAL_TILING')
    if np.any(hull.equations[:,3]>=0):raise ContactPathUnsupported('REFERENCE_ORIGIN_OUTSIDE')
    return True


def boundary_rows(mesh,owner,poses):
    count=len(poses);rows=[];contacts=[]
    for fa,fb,u,v in mesh.shared_edges:
        a=int(owner[fa]);b=int(owner[fb])
        if a==b:continue
        p,q=mesh.vertices[[u,v]];n=np.cross(p,q);n/=np.linalg.norm(n)
        if n@mesh.centroids[fb]<0:n=-n
        n=poses[a]@n
        for point in (p,q):
            x=poses[a]@point;r=np.cross(x,n);row=np.zeros(3*count);row[3*a:3*a+3]=-r;row[3*b:3*b+3]=r
            rows.append(row);contacts.append(dict(owners=[a,b],point_unit=x.tolist(),normal_a_to_b=n.tolist()))
    return np.asarray(rows).reshape(-1,3*count),contacts


def hemisphere_axis(mesh,owner):
    if len(np.unique(owner))!=2:return None
    indices={u for a,b,u,v in mesh.shared_edges if owner[a]!=owner[b]}|{v for a,b,u,v in mesh.shared_edges if owner[a]!=owner[b]}
    points=mesh.vertices[sorted(indices)];_,_,vh=np.linalg.svd(points,full_matrices=False);axis=vh[-1]
    if np.max(abs(points@axis))>ROUND:return None
    for k in (0,1):
        side=mesh.centroids[owner==k]@axis;sign=np.sign(side.sum())
        if sign==0 or np.min(sign*(mesh.vertices[np.unique(mesh.faces[owner==k])]@axis)) < -ROUND:return None
        if k==0:first_sign=sign
        elif first_sign==sign:return None
    return axis


def placement_certificate(mesh,owner,poses):
    reference_partition(mesh);poses=np.asarray(poses,float)
    if np.max(abs(poses@poses.transpose(0,2,1)-np.eye(3)))>ROUND or np.any(np.linalg.det(poses)<=0):raise ContactPathUnsupported('NONRIGID_POSE')
    if np.array_equal(poses,np.repeat(poses[:1],len(poses),axis=0)):
        return dict(kind='common_rotation_of_partition',axis=None)
    axis=hemisphere_axis(mesh,owner)
    if axis is not None and np.linalg.norm(poses[0].T@poses[1]@axis-axis)<=ROUND:
        return dict(kind='opposite_invariant_hemispheres',axis=(poses[0]@axis).tolist())
    raise ContactPathUnsupported('INITIAL_PLACEMENT_UNCERTIFIED_OR_OVERLAPPING')


def swept_certificate(mesh,owner,poses,omega,dt):
    """All t in [0,dt], not endpoint or time-sample collision testing.

    An isometry of the entire partition preserves every separation. Independent
    spins about the common plane normal preserve both closed hemispheres. These
    invariant regions exclude all inter-domain pairs for the whole interval.
    Other paths are declined even if their endpoints happen to be disjoint.
    """
    initial=placement_certificate(mesh,owner,poses);omega=np.asarray(omega)
    if np.array_equal(omega,np.repeat(omega[:1],len(omega),axis=0)):
        kind='common_isometry_all_pairs_all_time'
    else:
        axis=hemisphere_axis(mesh,owner)
        if axis is None:raise ContactPathUnsupported('NONPLANAR_RELATIVE_PATH_UNCERTIFIED')
        axis=np.asarray(poses[0])@axis
        scale=max(float(np.max(np.linalg.norm(omega,axis=1))),np.finfo(float).tiny)
        if np.max(np.linalg.norm(np.cross(omega,axis),axis=1))>ROUND*scale:raise ContactPathUnsupported('NONCOMMUTING_PATH_REQUIRES_EVENT_SOLVER')
        kind='invariant_opposite_hemispheres_all_pairs_all_time'
    end=np.array([Rotation.from_rotvec(dt*w).as_matrix()@R for w,R in zip(omega,poses)])
    placement_certificate(mesh,owner,end)
    return end,dict(kind=kind,time_interval_seconds=[0.,float(dt)],all_domain_pairs=True,endpoint_sampling=False,initial=initial)


def closed_partition_step(mesh,owner,poses,inertias,drags,torques,old_omega,dt,radius_m):
    owner=np.asarray(owner,int);poses=np.asarray(poses,float);I=np.asarray(inertias);D=np.asarray(drags);tau=np.asarray(torques);old=np.asarray(old_omega)
    placement_certificate(mesh,owner,poses)
    count=len(I);B=block_diag(*[x+dt*y for x,y in zip(I,D)]);rhs=np.concatenate([x@w+dt*f for x,w,f in zip(I,old,tau)])
    free=np.linalg.solve(B,rhs);J,contacts=boundary_rows(mesh,owner,poses)
    axis=hemisphere_axis(mesh,owner) if count==2 else None
    # Exact common coordinates remove roundoff drift between locked copies.
    if count==1 or (count==2 and axis is None):
        w=np.linalg.solve(I.sum(axis=0)+dt*D.sum(axis=0),np.sum(np.einsum('kij,kj->ki',I,old)+dt*tau,axis=0));new=np.repeat(w[None,:],count,axis=0)
    elif count==2:
        world_axis=poses[0]@axis;Z=np.vstack((np.column_stack((np.eye(3),np.zeros(3))),np.column_stack((np.eye(3),world_axis))))
        coeff=np.linalg.solve(Z.T@B@Z,Z.T@rhs);new=(Z@coeff).reshape(count,3)
        scale=max(np.linalg.norm(new),np.linalg.norm(free),np.finfo(float).tiny)
        if abs(coeff[-1])<=ROUND*scale:
            common=np.linalg.solve(I.sum(axis=0)+dt*D.sum(axis=0),np.sum(np.einsum('kij,kj->ki',I,old)+dt*tau,axis=0));new=np.repeat(common[None,:],count,axis=0)
        elif np.max(np.linalg.norm(np.cross(new,world_axis),axis=1))<=ROUND*scale:
            new=(new@world_axis)[:,None]*world_axis
    else:
        if not np.array_equal(poses,np.repeat(poses[:1],count,axis=0)):raise ContactPathUnsupported('MULTIDOMAIN_SLIDING_CONTACT_SEARCH_REQUIRED')
        Z=null_space(J,rcond=ROUND);candidate=(Z@np.linalg.solve(Z.T@B@Z,Z.T@rhs)).reshape(count,3)
        common=np.linalg.solve(I.sum(axis=0)+dt*D.sum(axis=0),np.sum(np.einsum('kij,kj->ki',I,old)+dt*tau,axis=0))
        scale=max(np.linalg.norm(candidate),np.linalg.norm(free),np.finfo(float).tiny)
        if np.max(abs(candidate-common))>1e-10*scale:raise ContactPathUnsupported('MULTIPLE_NONCOMMUTING_SLIDING_PATHS')
        new=np.repeat(common[None,:],count,axis=0)
    scale=max(np.linalg.norm(rhs),np.linalg.norm(B@new.ravel()),np.finfo(float).tiny)
    impulse=B@new.ravel()-rhs
    multipliers=np.empty(0)
    if len(J):
        multiplier_scaled,_=nnls(J.T,impulse/scale,maxiter=max(100,20*len(J)));multipliers=multiplier_scaled*scale
        error=np.linalg.norm(J.T@multipliers-impulse)/scale
        if error>1e-10:raise ContactPathUnsupported('NONCOMPRESSIVE_REACTION_REQUIRED')
    else:error=np.linalg.norm(impulse)/scale
    angular_pair_sum=np.sum((J.T@multipliers).reshape(count,3),axis=0) if len(J) else np.zeros(3)
    end,certificate=swept_certificate(mesh,owner,poses,new,dt)
    de=new-old;K0=float(.5*np.einsum('ki,kij,kj->',old,I,old));K1=float(.5*np.einsum('ki,kij,kj->',new,I,new));work=float(dt*np.sum(tau*new));drag=float(dt*np.einsum('ki,kij,kj->',new,D,new));N=float(.5*np.einsum('ki,kij,kj->',de,I,de));contact_work=float(multipliers@(J@new.ravel())) if len(J) else 0.
    balance=K1-K0-work+drag+N-contact_work
    if abs(balance)>1e-10*max(abs(K0)+abs(K1)+abs(work)+abs(drag)+N,1.):raise ValueError('CONTACT_WORK_BALANCE')
    for item,value in zip(contacts,multipliers):item['normal_impulse_n_s']=float(value/radius_m)
    ledger=dict(kinetic_before=K0,kinetic_after=K1,external_work=work,drag=drag,numerical_rotation=N,rotation_balance=balance,momentum_relative=float(error))
    proof=dict(law='unbonded_frictionless_hard_contact',normal_impulses=contacts,contact_work_j=contact_work,net_internal_angular_impulse=angular_pair_sum.tolist(),whole_step=certificate,free_relative_omega_norm=float(np.linalg.norm(free.reshape(count,3)-free.reshape(count,3)[0])),accepted_relative_omega_norm=float(np.linalg.norm(new-new[0])))
    return new,end,ledger,proof
