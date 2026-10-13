"""Experimental connected fracture on a scalar elastic surface network.

Not a constitutive replacement for shared_nodal_moments. No planetary imports.
"""
from dataclasses import dataclass,asdict
import json
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from .mesh import build_icosphere,connected_components
from .plates import Plate,PlateSystem


@dataclass(frozen=True)
class Parameters:
    radius_km:float=1000.
    thickness_m:float=10000.
    shear_modulus_pa:float=3e10
    shear_strength_pa:float=1e6
    damage_time_years:float=100.
    drag_pa_s_m:float=1e10
    forcing_pa:float=4e4
    min_domain_area_fraction:float=.05
    max_failed_edge_fraction:float=.35


def partition(mesh,failed):
    """Actual graph components, no selected seeds, paths or requested count."""
    neighbors=[[] for _ in range(mesh.cell_count)]
    for broken,(f,g,_,_) in zip(failed,mesh.shared_edges):
        if not broken:neighbors[f].append(g);neighbors[g].append(f)
    parts=sorted(connected_components(range(mesh.cell_count),tuple(tuple(sorted(x)) for x in neighbors)),key=min)
    owner=np.zeros(mesh.cell_count,dtype=int)
    for label,part in enumerate(parts):owner[part]=label
    return owner


def certify(mesh,failed,params):
    owner=partition(mesh,failed);areas=mesh.areas_unit_sphere;areas=areas/areas.sum()
    if owner.max()==0:return None,'NOT_SEPARATING'
    if np.mean(failed)>params.max_failed_edge_fraction:return None,'DIFFUSE_NETWORK_FAILURE'
    if min(np.bincount(owner,weights=areas))<params.min_domain_area_fraction:return None,'FRAGMENT_TOO_SMALL'
    seam=[]
    for index,(f,g,u,v) in enumerate(mesh.shared_edges):
        if owner[f]!=owner[g]:
            assert failed[index];seam.append((u,v))
    degree=np.bincount(np.array(seam).ravel(),minlength=mesh.vertex_count)
    # Permit several disjoint closed boundaries, but no dangling ends or junctions.
    if np.any((degree>0)&(degree!=2)):return None,'NON_MANIFOLD_BOUNDARY'
    return owner,'CLOSED_CONNECTED_BOUNDARIES'


class EdgeFracture:
    def __init__(self,level,params=Parameters(),reverse=False):
        self.params=params;self.level=level;self.mesh=build_icosphere(level);mesh=self.mesh
        # Canonical sorting makes assembly and state independent of traversal order.
        edges=list(mesh.shared_edges)
        if reverse:edges=edges[::-1]
        edges=sorted(edges);self.edges=np.array(edges);self.f=self.edges[:,0];self.g=self.edges[:,1]
        self.canonical_to_mesh=np.array([list(mesh.shared_edges).index(tuple(e)) for e in edges])
        R=params.radius_km*1000
        arc=lambda a,b:np.arctan2(np.linalg.norm(np.cross(a,b),axis=1),np.einsum('ij,ij->i',a,b))
        self.length=R*arc(mesh.vertices[self.edges[:,2]],mesh.vertices[self.edges[:,3]])
        self.distance=R*arc(mesh.centroids[self.f],mesh.centroids[self.g])
        self.area=mesh.areas_unit_sphere*R**2;self.cross=params.thickness_m*self.length
        self.stiffness=params.shear_modulus_pa*self.cross/self.distance
        m=len(edges);self.incidence=sparse.coo_matrix((np.r_[np.ones(m),-np.ones(m)],(np.r_[np.arange(m),np.arange(m)],np.r_[self.f,self.g])),shape=(m,mesh.cell_count)).tocsr()
        self.state=dict(time_years=0.,u=np.zeros(mesh.cell_count).tolist(),damage=np.zeros(m).tolist(),
            mass_kg=(3300*params.thickness_m*self.area).tolist(),area_m2=self.area.tolist(),
            enthalpy_j=(3300*params.thickness_m*self.area*1e6).tolist(),fracture_reservoir_j=0.,drag_heat_reservoir_j=0.,
            cumulative_external_j=0.,numerical_j=0.,owner=np.zeros(mesh.cell_count,dtype=int).tolist(),events=[])

    def step(self,years,forcing=None):
        p=self.params;s=self.state;dt=years*31557600.;old=np.array(s['u']);d=np.array(s['damage'])
        force=p.forcing_pa*self.area*self.mesh.centroids[:,2] if forcing is None else np.asarray(forcing)*self.area
        k=self.stiffness*(1-d)**2;L=self.incidence.T@sparse.diags(k)@self.incidence
        drag=p.drag_pa_s_m*self.area;A=L+sparse.diags(drag/dt)
        u=spsolve(A,force+drag/dt*old);du=u-old;jump=self.incidence@u
        traction=k*jump/self.cross
        rate=np.maximum(abs(traction)/p.shear_strength_pa-1,0)**2/p.damage_time_years
        newd=np.minimum(1.,d+years*rate);newk=self.stiffness*(1-newd)**2
        U0=.5*np.dot(k,(self.incidence@old)**2);U1=.5*np.dot(newk,jump**2)
        fracture=.5*np.dot(k-newk,jump**2);heat=np.dot(drag/dt,du**2)
        numerical=.5*np.dot(k,(self.incidence@du)**2);work=float(force@du)
        residual=U1-U0+fracture+heat+numerical-work
        scale=max(U0,U1,abs(work),fracture,heat,1.)
        assert fracture>=0 and heat>=0 and abs(residual)/scale<1e-10
        failed=np.zeros(len(k),bool);failed[self.canonical_to_mesh]=newd>=1
        owner,reason=certify(self.mesh,failed,p)
        updated=json.loads(json.dumps(s));updated.update(time_years=s['time_years']+years,u=u.tolist(),damage=newd.tolist(),
            fracture_reservoir_j=s['fracture_reservoir_j']+fracture,drag_heat_reservoir_j=s['drag_heat_reservoir_j']+heat,
            cumulative_external_j=s['cumulative_external_j']+work,numerical_j=s['numerical_j']+numerical)
        if owner is not None and owner.tolist()!=s['owner']:
            updated['owner']=owner.tolist();updated['events'].append(dict(time_years=updated['time_years'],domains=int(owner.max()+1)))
        for key in ('mass_kg','area_m2','enthalpy_j'):assert updated[key]==s[key]
        self.state=updated
        return dict(time_years=updated['time_years'],max_damage=float(newd.max()),failed_edges=int(failed.sum()),
            domains=len(set(updated['owner'])),reason=reason,energy_relative=float(abs(residual)/scale),
            max_traction_pa=float(abs(traction).max()),fracture_j=float(fracture),external_j=work)

    def system(self):
        owner=np.asarray(self.state['owner'],dtype=np.int32)
        # No invented separating kick; scalar displacement cannot supply Euler velocities.
        plates=tuple(Plate(i,int(np.flatnonzero(owner==i)[0]),np.array([0.,0.,1.]),0.) for i in range(owner.max()+1))
        return PlateSystem(owner,plates)

    def save(self,path):
        with open(path,'w') as f:json.dump(dict(level=self.level,parameters=asdict(self.params),state=self.state),f)

    @classmethod
    def load(cls,path):
        with open(path) as f:data=json.load(f)
        m=cls(data['level'],Parameters(**data['parameters']));m.state=data['state'];return m
