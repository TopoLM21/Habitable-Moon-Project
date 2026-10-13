"""Moving material-front basal facets and explicit relative-velocity drag.

The inherited Genesis source has prescribed traction, not a recovered fluid
velocity. Production uses an explicit stationary-material-mantle limit.
The phase-front speed is never a material velocity. Geometry is lagged over
each accepted BE step; it is refreshed after endpoint growth on the next step.
"""
from copy import deepcopy
import numpy as np
from .genesis_volume_basal import CONSISTENT_ANGULAR_MASS as W
from .basal_coupling import mantle_source_pattern,basal_coupling_fraction

MODE='moving_material_front_relative'
MANTLE='stationary_material_mantle_no_inherited_velocity'


def enable(model):
    """Explicit opt-in with no inventory, memory, damage or clock reset."""
    from .simplified_genesis import SimplifiedGenesis
    if model.config.get('mechanical_mode')!='shared_nodal_moments':raise ValueError('MOVING_BASAL_REQUIRES_NODAL')
    c=deepcopy(model.config)
    c['nodal']['moving_basal']=dict(mode=MODE,beta_pa_s_m=1e14,mantle_velocity_source=MANTLE,
        source_traction='existing seeded Genesis mantle potential, current thickness transmission',
        heat_destination='current basal load-bearing thermal node',geometry_time='accepted step beginning, refresh after endpoint growth')
    return SimplifiedGenesis(c,model.state)


def geometry(model,state=None):
    c=model.config;s=model.state if state is None else state
    X=np.asarray(c['nodal']['X']);faces=model.mesh.faces;U=np.asarray(s['nodal_displacement']).reshape(-1,6)
    fronts=np.zeros(model.nf);nodes=model.solid_nodes[:,1].copy()
    for x in c['source_material_catalogue']:fronts[x['face_index']]=max(fronts[x['face_index']],x['live_q_hi'])
    catalog={x['id']:x for x in c['phase_cells']}
    for x in s.get('born_material',[]):
        row=catalog[x['cell_id']];f=x['face']
        if row['q_hi']>fronts[f]:fronts[f]=row['q_hi'];nodes[f]=x['node']
    tri=X[faces];coefficient=6*np.asarray(c['nodal']['column_mass'])/(c['nodal']['density']*np.linalg.det(tri.transpose(0,2,1)))
    z=-coefficient*fronts
    if np.any(z<=-1):raise ValueError('BASAL_FRONT_REACHES_CENTRE')
    h=np.expm1(np.log1p(z)/3);lam=1+h
    top=tri+U[faces,:3]
    bottom=lam[:,None,None]*tri+U[faces,:3]+h[:,None,None]*U[faces,3:]
    cross=np.cross(bottom[:,1]-bottom[:,0],bottom[:,2]-bottom[:,0]);length=np.linalg.norm(cross,axis=1)
    if np.any(length<=0) or np.any(np.einsum('fi,fi->f',cross,bottom.mean(axis=1))<=0):raise ValueError('COLLAPSED_BASAL_GEOMETRY')
    normal=cross/length[:,None];area=length/2
    depth=np.einsum('fvi,fi->fv',top-bottom,normal).mean(axis=1)
    if np.any(depth<=0):raise ValueError('NONPOSITIVE_BASAL_THICKNESS')
    return dict(bottom=bottom,normal=normal,area=area,depth=depth,h=h,fronts=fronts,thermal_nodes=nodes)


def facet(bottom,h,beta,traction,mantle_velocity):
    """SI assembly at one current physical facet, without front-migration speed."""
    bottom=np.asarray(bottom);traction=np.asarray(traction);vm=np.asarray(mantle_velocity)
    cross=np.cross(bottom[1]-bottom[0],bottom[2]-bottom[0]);area=np.linalg.norm(cross)/2
    if area<=0 or beta<=0:raise ValueError('BASAL_FACET_PARAMETERS')
    n=cross/(2*area);P=np.eye(3)-np.outer(n,n)
    D=beta*area*np.kron(W,np.kron(np.array([[1,h],[h,h*h]]),P))
    lift=lambda a:np.stack((a,h*a),axis=1).reshape(18)
    Ft=lift(area*W@traction);Fm=lift(beta*area*(W@vm)@P)
    return dict(D=D,Ft=Ft,Fm=Fm,area=area,P=P,h=h,bottom=bottom,vm=vm,beta=beta)


def exchange(face,v,dt):
    v=np.asarray(v).reshape(3,6);vb=v[:,:3]+face['h']*v[:,3:]
    delta=(vb-face['vm'])@face['P'];reaction=face['beta']*face['area']*(W@delta)
    Q=float(dt*np.einsum('ij,ij->',reaction,delta))
    crust=float(-dt*np.einsum('ij,ij->',reaction,vb))
    mantle=float(dt*np.einsum('ij,ij->',reaction,face['vm']))
    scale=max(abs(Q),abs(crust),abs(mantle),1.)
    if Q < -1e-12*scale or abs(Q+crust+mantle)>1e-12*scale:raise ValueError('RELATIVE_BASAL_WORK_IDENTITY')
    return dict(heat_j=Q,crust_drag_work_j=crust,mantle_drag_work_j=mantle,
        mantle_reaction_impulse_ns=dt*reaction.sum(axis=0),
        mantle_reaction_angular_impulse_nms=dt*np.cross(face['bottom'],reaction).sum(axis=0))


def assemble(model):
    c=model.config;spec=c['nodal']['moving_basal']
    if spec['mode']!=MODE or spec['mantle_velocity_source']!=MANTLE or spec['beta_pa_s_m']!=1e14:
        raise ValueError('UNSUPPORTED_BASAL_SOURCE')
    g=geometry(model);faces=model.mesh.faces;ndof=6*model.mesh.vertex_count
    coupling=basal_coupling_fraction(g['depth']/1000,model.shell['traction_coupling_depth_km'])
    numerator=np.zeros(model.mesh.vertex_count);denominator=numerator.copy()
    np.add.at(numerator,faces.ravel(),np.repeat(g['area']*coupling,3));np.add.at(denominator,faces.ravel(),np.repeat(g['area'],3))
    # Reuse the existing source pattern/seed, not a newly invented flow field.
    pattern=mantle_source_pattern(model.mesh,model.shell['seed'])
    traction=model.shell['convective_traction_pa']*(numerator/denominator)[:,None]*pattern
    D=np.zeros((ndof,ndof));Ft=np.zeros(ndof);Fm=np.zeros(ndof);rows=[]
    dofs=(faces[:,:,None]*6+np.arange(6)).reshape(model.nf,18)
    for f,idx in enumerate(dofs):
        row=facet(g['bottom'][f],g['h'][f],spec['beta_pa_s_m'],traction[faces[f]],np.zeros((3,3)))
        D[np.ix_(idx,idx)]+=row['D'];np.add.at(Ft,idx,row['Ft']);np.add.at(Fm,idx,row['Fm']);rows.append(row)
    return dict(D=D,F=Ft+Fm,Ft=Ft,Fm=Fm,faces=rows,geometry=g,dofs=dofs)


def budget(basal,v,dt):
    rows=[exchange(face,v[idx],dt) for face,idx in zip(basal['faces'],basal['dofs'])]
    g=basal['geometry'];mantle_work=sum(x['mantle_drag_work_j'] for x in rows)
    return dict(rows=rows,heat_j=sum(x['heat_j'] for x in rows),mantle_drag_work_j=mantle_work,
        prescribed_traction_work_j=float(dt*basal['Ft']@v),mantle_driver_work_j=-mantle_work,
        physical_external_work_j=float(dt*basal['Ft']@v)-mantle_work,
        mantle_reaction_impulse_ns=sum((x['mantle_reaction_impulse_ns'] for x in rows),np.zeros(3)).tolist(),
        mantle_reaction_angular_impulse_nms=sum((x['mantle_reaction_angular_impulse_nms'] for x in rows),np.zeros(3)).tolist(),
        mantle_velocity_source=MANTLE,beta_pa_s_m=1e14,area_m2=float(g['area'].sum()),
        depth_min_m=float(g['depth'].min()),depth_max_m=float(g['depth'].max()),depth_mean_m=float(g['depth'].mean()),
        phase_front_velocity_used_as_mantle_velocity=False,geometry_time='accepted step beginning')
