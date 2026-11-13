"""Bounded conforming material surface and dimensional gravity/drag control.

One shared vertex position/velocity welds adjacent triangles. Region labels
are material labels, not independent rigid rotations. No slip, fracture,
creation/recycling, mantle convection or thermal-feedback closure is implied.
Constant gravity over finite spherical columns and basal drag form a separate
explicit mechanical contract. No legacy proxy force is converted into joules.
"""
from copy import deepcopy
from dataclasses import asdict,dataclass
import hashlib,json,math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .mesh import SphereMesh,_build_topology
from .thermal import SECONDS_PER_MYR
from .thermal_budget import _add


def normalize(x):
    norm=np.linalg.norm(x,axis=-1,keepdims=True)
    if np.any(norm<1e-12):raise ValueError('Degenerate sphere position')
    return x/norm


def spherical_measures(vertices,faces):
    """Signed exact minor-triangle area and its Cartesian derivative, unit R."""
    a,b,c=np.moveaxis(vertices[faces],1,0)
    cross=np.stack([np.cross(b,c),np.cross(c,a),np.cross(a,b)],axis=1)
    num=np.einsum('ij,ij->i',a,cross[:,0]);den=1+np.einsum('ij,ij->i',a,b)+np.einsum('ij,ij->i',b,c)+np.einsum('ij,ij->i',c,a)
    if np.any(num<=0) or np.any(den<=0):raise ValueError('Inverted or non-minor spherical material cell')
    area=2*np.arctan2(num,den)
    sums=np.stack([b+c,c+a,a+b],axis=1)
    grad=2*(den[:,None,None]*cross-num[:,None,None]*sums)/(den**2+num**2)[:,None,None]
    return area,grad


def column_potential(area_m2,volume_m3,density,gravity,radius_m):
    """Finite radial sectors in prescribed constant gravity; no isostasy.

    V=A*(h+h²/R+h³/(3R²)); U=rho*g*A*integral[z*(1+z/R)² dz].
    The derivative holds material volume fixed. In the thin limit U=rho*g*V²/(2A).
    """
    area=np.asarray(area_m2);volume=np.asarray(volume_m3)
    if np.any(area<0) or np.any(volume<0) or np.any((area==0)&(volume>0)):
        raise ValueError('Invalid occupied column area/volume')
    depth=np.divide(volume,area,out=np.zeros_like(volume),where=area>0)
    h=radius_m*np.expm1(np.log1p(3*depth/radius_m)/3)
    x=h/radius_m
    U=density*gravity*area*h*h*(.5+x*(2/3+x/4))
    derivative=-density*gravity*h*h*(.5+x*(1/3+x/12))
    return U,derivative,h


@dataclass(frozen=True)
class SurfaceParameters:
    radius_m:float=5_287_000.
    gravity_m_s2:float=7.12
    basal_drag_pa_s_m:float=1e17
    felsic_density_kg_m3:float=2800.
    basalt_density_kg_m3:float=2900.
    heat_capacity_j_kg_k:float=1200.
    min_area_fraction:float=1e-4
    max_height_radius:float=.02

    def __post_init__(self):
        if any(not math.isfinite(v) or v<=0 for v in asdict(self).values()):raise ValueError('Invalid surface parameter')
        if self.min_area_fraction>=1 or self.max_height_radius>=.1:raise ValueError('Outside bounded surface geometry')


@dataclass(frozen=True)
class SurfaceControl:
    max_step_myr:float=1.
    initial_step_myr:float=.25
    position_atol_rad:float=1e-5
    thickness_rtol:float=1e-3
    energy_rtol:float=1e-4
    energy_atol_fraction:float=1e-12
    max_trials:int=10000

    def __post_init__(self):
        if any(not math.isfinite(v) or v<=0 for v in asdict(self).values()):raise ValueError('Invalid surface controller')


class ConformingSurface:
    VERSION=1

    def __init__(self,mesh,arrays,params=SurfaceParameters(),control=SurfaceControl(),initial_heat_j=(0.,0.),provenance=None):
        self.params=params;self.control=control
        self.vertices=mesh.vertices.copy();self.faces=mesh.faces.copy()
        self.neighbors,self.edges=_build_topology(self.faces)
        self.arrays={k:np.array(v,copy=True) for k,v in arrays.items()}
        required={'owner','felsic_fraction','felsic_volume_m3','basalt_volume_m3','felsic_age_moment_m3_myr','basalt_age_moment_m3_myr','potential'}
        if required!=self.arrays.keys():raise ValueError('Incomplete conforming material inventory')
        n=len(self.faces)
        if any(v.shape!=(n,) or np.any(~np.isfinite(v)) or np.any(v<0) for v in self.arrays.values()):raise ValueError('Invalid material fields')
        if np.any(self.arrays['felsic_fraction']>1):raise ValueError('Invalid material footprint fraction')
        if np.any(self.arrays['owner']!=self.arrays['owner'].astype(int)):raise ValueError('Invalid region labels')
        self.reference_area=spherical_measures(self.vertices,self.faces)[0]*params.radius_m**2
        self.data=dict(version=self.VERSION,time_myr=0.,origin_time_myr=0.,next_step_myr=control.initial_step_myr,
            accepted_steps=0,rejected_steps=0,heat_generated_j=[0.,0.],initial_heat_j=list(initial_heat_j),
            energy_defect_j=[0.,0.],external_work_j=[0.,0.],history=[],provenance=deepcopy(provenance or {}))
        self.data['initial_potential_j']=self.evaluate(self.vertices)['potential_j']
        self.data['initial_volume_m3']=[math.fsum(self.arrays[k+'_volume_m3']) for k in ('felsic','basalt')]
        self.data['material_identity']=hashlib.sha256(self.faces.tobytes()+self.arrays['owner'].astype(np.int64).tobytes()).hexdigest()

    @classmethod
    def from_area_inventory(cls,inventory,params=SurfaceParameters(),control=SurfaceControl(),thermal_data=None,provenance=None):
        """New declared mechanical initialization, never repair an evolved gap."""
        q=inventory.quaternions
        if inventory.coarse.time_myr!=0 or np.max(abs(q[:,:3]))>1e-15 or np.max(abs(abs(q[:,3])-1))>1e-15:
            raise ValueError('Conforming initialization requires the shared zero-time partition; evolved rigid gaps cannot be restored')
        a=inventory.coarse.arrays;area=inventory.mesh.areas_unit_sphere*(params.radius_m/1000)**2
        if np.max(abs(area-a['area'])/area)>1e-12:raise ValueError('Material and mechanical radii disagree')
        arrays=dict(owner=a['owner'],felsic_fraction=a['felsic_area']/a['area'],potential=a['potential'])
        for k in ('felsic','basalt'):
            arrays[k+'_volume_m3']=a[k+'_volume']*1e9
            arrays[k+'_age_moment_m3_myr']=a[k+'_age_moment']*1e9
        source=deepcopy(provenance or {})
        source.update(initialization='new fixed-base gravity/drag control from shared material areas; no recovered mechanical memory',
                      prior_thermal_inventory=deepcopy(thermal_data))
        heat=(0.,0.) if thermal_data is None else thermal_data['energy_c_j']
        return cls(inventory.mesh,arrays,params,control,heat,source)

    def geometry(self,vertices):
        if np.max(abs(np.linalg.norm(vertices,axis=1)-1))>2e-13:raise ValueError('Nonunit shared material vertex')
        area,gradient=spherical_measures(vertices,self.faces);area*=self.params.radius_m**2
        if np.min(area/self.reference_area)<self.params.min_area_fraction:raise ValueError('Material cell approaching collapse; remeshing is not implemented')
        total=math.fsum(area);sphere=4*math.pi*self.params.radius_m**2
        if abs(total-sphere)>5e-12*sphere:raise ValueError('Shared material partition has wrong spherical degree/area')
        return area,gradient*self.params.radius_m**2

    def evaluate(self,vertices):
        p=self.params;a=self.arrays;area,gradient=self.geometry(vertices)
        fractions=np.stack([a['felsic_fraction'],1-a['felsic_fraction']],axis=1)
        volumes=np.stack([a['felsic_volume_m3'],a['basalt_volume_m3']],axis=1)
        density=np.array([p.felsic_density_kg_m3,p.basalt_density_kg_m3])
        potential,pressure,height=column_potential(area[:,None]*fractions,volumes,density,p.gravity_m_s2,p.radius_m)
        if np.max(height)/p.radius_m>p.max_height_radius:raise ValueError('Column exceeds fixed-base gravity/drag applicability')
        dU=np.sum(pressure*fractions,axis=1)
        # Uniform surface pressure does no tangential work because sum(A)=4*pi*R².
        # Subtract its gauge before assembly to avoid cancellation of large
        # hydrostatic terms; this is not removal of a residual net force.
        dU-=math.fsum(dU*area)/math.fsum(area)
        nodal=np.zeros_like(vertices)
        for k in range(3):np.add.at(nodal,self.faces[:,k],dU[:,None]*gradient[:,k])
        nodal-=vertices*np.sum(nodal*vertices,axis=1)[:,None]
        weights=np.bincount(self.faces.ravel(),weights=np.repeat(area/3,3),minlength=len(vertices))
        resistance=p.basal_drag_pa_s_m*weights*p.radius_m**2 # J*s for angular velocity
        velocity=-nodal/resistance[:,None] # unit-sphere displacement per second
        dissipation=math.fsum(np.sum(nodal*nodal,axis=1)/resistance) # W, independently positive
        return dict(potential_j=math.fsum(potential.ravel()),area_m2=area,height_m=height,
                    potential_cells_j=potential,gradient_j=nodal,velocity_per_s=velocity,dissipation_w=dissipation,weights_m2=weights)

    def _heun(self,vertices,dt_myr):
        seconds=dt_myr*SECONDS_PER_MYR;start=self.evaluate(vertices)
        trial=normalize(vertices+seconds*start['velocity_per_s']);predict=self.evaluate(trial)
        end=normalize(vertices+.5*seconds*(start['velocity_per_s']+predict['velocity_per_s']))
        diag=self.evaluate(end)
        heat=.5*seconds*(start['dissipation_w']+predict['dissipation_w'])
        # Heat is integrated drag dissipation, never an energy-balancing residual.
        defect=math.fsum([diag['potential_j'],-start['potential_j'],heat])
        return end,heat,defect,diag

    def _accepted(self,vertices,dt,heat,defect,diag,error_ratio):
        self.vertices=vertices;self.data['time_myr']+=dt;self.data['accepted_steps']+=1
        _add(self.data['heat_generated_j'],heat);_add(self.data['energy_defect_j'],defect)
        for k in ('felsic','basalt'):self.arrays[k+'_age_moment_m3_myr']+=self.arrays[k+'_volume_m3']*dt
        self.data['history'].append(dict(time_myr=self.data['time_myr'],dt_myr=dt,potential_j=diag['potential_j'],
            heat_j=heat,energy_defect_j=defect,error_ratio=error_ratio,max_height_m=float(np.max(diag['height_m'])),
            max_speed_mm_yr=float(np.max(np.linalg.norm(diag['velocity_per_s'],axis=1))*self.params.radius_m*SECONDS_PER_MYR/1000),
            min_area_fraction=float(np.min(diag['area_m2']/self.reference_area))))

    def step_fixed(self,dt):
        """Convergence control. No tolerance-based acceptance is implied."""
        if not math.isfinite(dt) or dt<=0:raise ValueError('Invalid time step')
        result=self._heun(self.vertices,dt)
        self._accepted(result[0],dt,*result[1:],error_ratio=None)

    def prescribed_step(self,omega_rad_myr,dt):
        """Kinematic comparison only: weighted welded projection, explicit work.

        The regional rotations are imposed, not obtained from this force law.
        Independent Gaussian work integration is not replaced by delta(U).
        """
        if not math.isfinite(dt) or dt<=0:raise ValueError('Invalid prescribed interval')
        omega=np.asarray(omega_rad_myr,float);owners=self.arrays['owner'].astype(int)
        if omega.ndim!=2 or omega.shape[1]!=3 or np.any(~np.isfinite(omega)) or np.max(owners)>=len(omega):raise ValueError('Invalid prescribed regional rotations')
        area,_=self.geometry(self.vertices);nodal=np.zeros_like(self.vertices);weights=np.zeros(len(nodal))
        for k in range(3):
            np.add.at(nodal,self.faces[:,k],omega[owners]*area[:,None]/3)
            np.add.at(weights,self.faces[:,k],area/3)
        nodal/=weights[:,None]
        start=self.evaluate(self.vertices);end=Rotation.from_rotvec(nodal*dt).apply(self.vertices);diag=self.evaluate(end)
        def work(order):
            nodes,coeff=np.polynomial.legendre.leggauss(order);values=[]
            for s,w in zip((nodes+1)/2,coeff/2):
                x=Rotation.from_rotvec(nodal*dt*s).apply(self.vertices)
                ev=self.evaluate(x);derivative=dt*np.cross(nodal,x)
                values.append(w*math.fsum((ev['gradient_j']*derivative).ravel()))
            return math.fsum(values)
        low=work(5);high=work(9);defect=math.fsum([diag['potential_j'],-start['potential_j'],-high])
        tolerance=1e-8*abs(high)+64*math.ulp(start['potential_j'])
        if abs(high-low)>tolerance or abs(defect)>tolerance:raise ValueError('Prescribed geometric work quadrature unresolved')
        _add(self.data['external_work_j'],high)
        self._accepted(end,dt,0.,defect,diag,error_ratio=None)
        self.data['history'][-1].update(mode='prescribed_welded_projection',external_work_j=high,work_order_difference_j=abs(high-low))

    def advance(self,target_myr):
        if not math.isfinite(target_myr) or target_myr<self.data['time_myr']:raise ValueError('Invalid final time')
        c=self.control;trials=0
        while self.data['time_myr']<target_myr:
            trials+=1
            if trials>c.max_trials:raise RuntimeError('Conforming controller trial budget exceeded')
            dt=min(self.data['next_step_myr'],c.max_step_myr,target_myr-self.data['time_myr'])
            if self.data['time_myr']+dt==self.data['time_myr']:raise ValueError('Conforming time interval below arithmetic resolution')
            try:
                coarse=self._heun(self.vertices,dt);first=self._heun(self.vertices,dt/2);second=self._heun(first[0],dt/2)
                heat=math.fsum([first[1],second[1]]);defect=math.fsum([first[2],second[2]])
                error=max(float(np.max(np.linalg.norm(second[0]-coarse[0],axis=1)))/(3*c.position_atol_rad),
                    float(np.max(abs(second[3]['height_m']-coarse[3]['height_m'])/np.maximum(second[3]['height_m'],1.)))/(3*c.thickness_rtol),
                    (abs(first[2])+abs(second[2]))/(c.energy_rtol*heat+c.energy_atol_fraction*self.data['initial_potential_j']))
                if second[3]['potential_j']>self.evaluate(self.vertices)['potential_j']+c.energy_atol_fraction*self.data['initial_potential_j']:error=max(error,2.)
            except ValueError:
                error=float('inf')
            if error>1:
                self.data['rejected_steps']+=1;self.data['next_step_myr']=dt*(.5 if not math.isfinite(error) else max(.1,.8*error**(-1/3)))
                continue
            self._accepted(second[0],dt,heat,defect,second[3],error)
            self.data['next_step_myr']=min(c.max_step_myr,dt*min(2.,.9*max(error,1e-12)**(-1/3)))

    def mesh(self):
        area,_=self.geometry(self.vertices)
        return SphereMesh(self.vertices.copy(),self.faces.copy(),normalize(self.vertices[self.faces].sum(axis=1)),
                          area/self.params.radius_m**2,self.neighbors,self.edges)

    def budget(self):
        ev=self.evaluate(self.vertices);p=self.params;a=self.arrays
        volumes=[math.fsum(a[k+'_volume_m3']) for k in ('felsic','basalt')]
        mass=math.fsum(np.array(volumes)*[p.felsic_density_kg_m3,p.basalt_density_kg_m3])
        heat=math.fsum(self.data['heat_generated_j'])
        return dict(time_myr=self.data['time_myr'],area_relative_error=math.fsum(ev['area_m2'])/(4*math.pi*p.radius_m**2)-1,
            volume_m3=volumes,mass_kg=mass,potential_j=ev['potential_j'],heat_generated_j=heat,
            external_work_j=math.fsum(self.data['external_work_j']),
            energy_residual_j=math.fsum([ev['potential_j'],-self.data['initial_potential_j'],*self.data['heat_generated_j'],*(-x for x in self.data['external_work_j'])]),
            integrated_defect_j=math.fsum(self.data['energy_defect_j']),crust_heat_increment_k=heat/(mass*p.heat_capacity_j_kg_k),
            max_height_m=float(np.max(ev['height_m'])),material_regions=len(np.unique(a['owner'])),vertices=len(self.vertices),cells=len(self.faces))

    def checkpoint(self):
        return dict(parameters=asdict(self.params),control=asdict(self.control),data=deepcopy(self.data),
            arrays={**{k:v.tolist() for k,v in self.arrays.items()},'vertices':self.vertices.tolist(),'faces':self.faces.tolist(),'reference_area':self.reference_area.tolist()})

    @classmethod
    def restore(cls,checkpoint):
        d=deepcopy(checkpoint);obj=cls.__new__(cls)
        if d['data']['version']!=cls.VERSION:raise ValueError('Unknown conforming checkpoint version')
        obj.params=SurfaceParameters(**d['parameters']);obj.control=SurfaceControl(**d['control']);obj.data=d['data']
        a=d['arrays'];obj.vertices=np.array(a.pop('vertices'));obj.faces=np.array(a.pop('faces'),dtype=np.int64)
        obj.reference_area=np.array(a.pop('reference_area'));obj.arrays={k:np.array(v) for k,v in a.items()}
        obj.neighbors,obj.edges=_build_topology(obj.faces);obj.evaluate(obj.vertices)
        identity=hashlib.sha256(obj.faces.tobytes()+obj.arrays['owner'].astype(np.int64).tobytes()).hexdigest()
        if identity!=obj.data['material_identity']:raise ValueError('Changed material identity')
        for value,initial in zip(obj.budget()['volume_m3'],obj.data['initial_volume_m3']):
            if value!=initial:raise ValueError('Changed material volume at restart')
        return obj

    def save(self,path):
        path=Path(path);path.mkdir(parents=True,exist_ok=False)
        cp=self.checkpoint();arrays=cp.pop('arrays')
        np.savez_compressed(path/'surface.npz',**{k:np.asarray(v) for k,v in arrays.items()})
        with (path/'meta.json').open('x') as f:json.dump(cp,f,indent=2)

    @classmethod
    def load(cls,path):
        path=Path(path);cp=json.loads((path/'meta.json').read_text())
        with np.load(path/'surface.npz') as z:cp['arrays']={k:z[k].tolist() for k in z.files}
        return cls.restore(cp)
