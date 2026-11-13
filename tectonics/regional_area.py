"""Persistent material-area quadrature, independent of the receiver map.

Refinement prolongs the existing piecewise-constant material history, splitting
extensive quantities by actual spherical area. It never creates new inventory
or coarsens away history. Each owning region still undergoes the old rigid
rotation: overlap/contact and dimensional mechanical work remain unresolved.
"""
from copy import deepcopy
from dataclasses import asdict,dataclass,replace
import math
import numpy as np
from scipy.spatial.transform import Rotation
from .mesh import build_icosphere
from .integral_arcs import SphereReceiver,strip_quadrature,cap_quadrature
from .regional_material import RegionalMaterialInventory


@dataclass(frozen=True)
class ActiveAreaParameters:
    max_edge_km:float=120.
    quadrature_order:int=16
    quadrature_spacing_km:float=100.
    max_leaves:int=131072
    strip_rule:str='source_points'
    source_budget_rtol:float=1e-7

    def __post_init__(self):
        if not 1<=self.max_edge_km<=3000:raise ValueError('Active-area edge size outside bounded numerical domain')
        if not 2<=self.quadrature_order<=32 or int(self.quadrature_order)!=self.quadrature_order:raise ValueError('Invalid area quadrature order')
        if not 1<=self.quadrature_spacing_km<=1000:raise ValueError('Invalid area quadrature spacing')
        if not 20<=self.max_leaves<=1048576 or int(self.max_leaves)!=self.max_leaves:raise ValueError('Invalid explicit numerical leaf budget')
        if self.strip_rule not in ('source_points','coarea'):raise ValueError('Invalid material strip integration rule')
        if not 0<self.source_budget_rtol<=1e-4:raise ValueError('Invalid material source integration tolerance')


def _centers(tri):
    x=np.sum(tri,axis=1);return x/np.linalg.norm(x,axis=1)[:,None]


def _areas(tri):
    a,b,c=np.moveaxis(tri,1,0)
    return 2*np.arctan2(np.abs(np.einsum('ij,ij->i',a,np.cross(b,c))),
                       1+np.einsum('ij,ij->i',a,b)+np.einsum('ij,ij->i',b,c)+np.einsum('ij,ij->i',c,a))


def _children(tri):
    a,b,c=np.moveaxis(tri,1,0)
    ab=a+b;ab/=np.linalg.norm(ab,axis=1)[:,None]
    bc=b+c;bc/=np.linalg.norm(bc,axis=1)[:,None]
    ca=c+a;ca/=np.linalg.norm(ca,axis=1)[:,None]
    return np.stack([np.stack([a,ab,ca],axis=1),np.stack([b,bc,ab],axis=1),
                     np.stack([c,ca,bc],axis=1),np.stack([ab,bc,ca],axis=1)],axis=1)


def _group_sum(index,values,count):
    result=np.zeros(count);order=np.argsort(index,kind='stable');index=index[order];values=values[order]
    cuts=np.r_[0,np.flatnonzero(np.diff(index))+1,len(index)]
    for first,last in zip(cuts[:-1],cuts[1:]):
        if first<last:result[index[first]]=math.fsum(values[first:last])
    return result


def _touches_sources(tri,owners,segments,caps,radius):
    """Conservative bounding-cap tests: can over-refine, cannot skip a strip."""
    c=_centers(tri)
    beta=np.max(np.arccos(np.clip(np.einsum('nij,nj->ni',tri,c),-1,1)),axis=1)
    chord=2*np.sin(beta/2)+2e-13
    active=np.zeros(len(tri),bool)
    for segment,sigma,outer in segments:
        if segment.amplitude==0:continue
        use=np.flatnonzero((owners==segment.overriding_plate)&~active)
        if not len(use):continue
        a,b=segment.start,segment.end;n=np.cross(a,b);n/=np.linalg.norm(n)
        if n@segment.toward<0:n=-n
        tangent=b-a*(a@b);tangent/=np.linalg.norm(tangent)
        angle=math.atan2(np.linalg.norm(np.cross(a,b)),a@b)
        end_tangent=-math.sin(angle)*a+math.cos(angle)*tangent
        lo=max(0.,segment.offset_km-outer);hi=segment.offset_km+outer
        if not 0<=lo<hi<math.pi*radius/2:raise ValueError('Active source outside oriented strip domain')
        dot=np.einsum('ij,j->i',c[use],n)
        inside=(dot+chord[use]>=math.sin(lo/radius))&(dot-chord[use]<=math.sin(hi/radius))
        inside&=np.einsum('ij,j->i',c[use],tangent)+chord[use]>=0
        inside&=np.einsum('ij,j->i',c[use],-end_tangent)+chord[use]>=0
        active[use[inside]]=True
    for center,plate,amplitude,sigma,outer in caps:
        if amplitude==0:continue
        use=np.flatnonzero((owners==plate)&~active)
        if len(use):active[use[np.arccos(np.clip(c[use]@center,-1,1))<=beta[use]+outer/radius+2e-13]]=True
    return active


class AreaMaterialInventory:
    def __init__(self,coarse,leaves,reference_triangles,roots,codes,quaternions,area_params,history=()):
        self.coarse=coarse;self.leaves=leaves;self.params=coarse.params;self.area_params=area_params
        self.reference_triangles=np.array(reference_triangles,copy=True)
        self.roots=np.array(roots,dtype=np.int64,copy=True);self.codes=np.array(codes,dtype=np.int64,copy=True)
        self.quaternions=np.array(quaternions,copy=True);self.history=list(history)
        self.mesh=build_icosphere(self.params.material_subdivisions);self.receiver=SphereReceiver(self.mesh)
        n=len(self.roots)
        if self.reference_triangles.shape!=(n,3,3) or len(leaves.arrays['area'])!=n or len(self.codes)!=n:raise ValueError('Incomplete material-area geometry')
        if not np.all(np.isfinite(self.reference_triangles)) or np.any(self.codes<1) or np.any(self.codes>=2**32):raise ValueError('Invalid material-area tree')
        if np.any(self.roots<0) or np.any(self.roots>=self.mesh.cell_count):raise ValueError('Invalid material roots')
        if len(np.unique(self.roots*2**32+self.codes))!=n:raise ValueError('Duplicate material leaf')
        if np.any(leaves.arrays['owner']!=coarse.arrays['owner'][self.roots]):raise ValueError('Area history owner mismatch')
        if self.quaternions.shape!=(int(max(coarse.arrays['owner']))+1,4) or not np.all(np.isfinite(self.quaternions)):raise ValueError('Invalid saved regional rotations')
        if np.max(abs(np.linalg.norm(self.quaternions,axis=1)-1))>2e-13:raise ValueError('Nonunit saved regional rotations')
        if np.max(abs(np.linalg.norm(self.reference_triangles,axis=2)-1))>2e-12:raise ValueError('Nonunit material triangle')
        expected=_areas(self.reference_triangles)/self.mesh.areas_unit_sphere[self.roots]*coarse.arrays['area'][self.roots]
        if np.max(abs(expected-leaves.arrays['area'])/leaves.arrays['area'])>1e-10:raise ValueError('Saved material area disagrees with geometry')
        total=_group_sum(self.roots,leaves.arrays['area'],self.mesh.cell_count)
        if np.max(abs(total-coarse.arrays['area'])/coarse.arrays['area'])>1e-12:raise ValueError('Incomplete material-area coverage')
        keys=set((self.roots*2**32+self.codes).tolist())
        for root,code in zip(self.roots,self.codes):
            parent=int(code)
            while parent>=4:
                parent//=4
                if int(root)*2**32+parent in keys:raise ValueError('Overlapping material tree ancestors')
            if parent!=1:raise ValueError('Invalid material tree path')

    @classmethod
    def initialize(cls,mesh,state,radius,params,area_params):
        coarse=RegionalMaterialInventory.initialize(mesh,state,radius,params)
        material=build_icosphere(params.material_subdivisions);n=material.cell_count
        q=np.zeros((int(max(coarse.arrays['owner']))+1,4));q[:,3]=1.
        return cls(coarse,deepcopy(coarse),material.vertices[material.faces],np.arange(n),np.ones(n,int),q,area_params)

    @classmethod
    def restore(cls,data,params,area_params):
        if data.get('version')!=2 or data['parameters']!=asdict(params) or asdict(ActiveAreaParameters(**data['area_parameters']))!=asdict(area_params):
            raise ValueError('Material-area restart mode/parameters/history mismatch')
        a=data['arrays'];coarse={k[7:]:v for k,v in a.items() if k.startswith('coarse_')};leaves={k[5:]:v for k,v in a.items() if k.startswith('leaf_')}
        return cls(RegionalMaterialInventory(coarse,params,data['time_myr']),RegionalMaterialInventory(leaves,params,data['time_myr']),
                   a['reference_triangles'],a['roots'],a['codes'],a['quaternions'],area_params,data['history'])

    def checkpoint(self):
        a={**{'coarse_'+k:v.tolist() for k,v in self.coarse.arrays.items()},**{'leaf_'+k:v.tolist() for k,v in self.leaves.arrays.items()}}
        a.update(reference_triangles=self.reference_triangles.tolist(),roots=self.roots.tolist(),codes=self.codes.tolist(),quaternions=self.quaternions.tolist())
        return dict(version=2,parameters=asdict(self.params),area_parameters=asdict(self.area_params),time_myr=self.leaves.time_myr,
                    history=deepcopy(self.history),arrays=a,scope='persistent spherical material-area refinement; regional rigid motion and nonlocal receiver mapping remain limited')

    def _world_triangles(self,tri=None,owners=None):
        if tri is None:tri=self.reference_triangles
        if owners is None:owners=self.leaves.arrays['owner'].astype(int)
        matrices=Rotation.from_quat(self.quaternions).as_matrix()[owners]
        return np.einsum('nij,nkj->nki',matrices,tri)

    def rotate(self,system,dt):
        from .kinematics import angular_velocity_vectors
        w=angular_velocity_vectors(system)
        if len(w)!=len(self.quaternions):raise ValueError('Area mode cannot silently change material topology')
        self.quaternions=(Rotation.from_rotvec(w*dt)*Rotation.from_quat(self.quaternions)).as_quat()
        matrices=Rotation.from_quat(self.quaternions).as_matrix()
        self.coarse.arrays['points']=np.einsum('nij,nj->ni',matrices[self.coarse.arrays['owner'].astype(int)],self.mesh.centroids)
        self.leaves.arrays['points']=_centers(self._world_triangles())

    def refine(self,segments,caps,radius,sigma,outer):
        """Area-conservative prolongation; retain all pre-existing memory."""
        a={k:v.copy() for k,v in self.leaves.arrays.items()};tri=self.reference_triangles.copy()
        roots=self.roots.copy();codes=self.codes.copy();initial=len(roots);max_geometry_error=0.
        extensive=['area','felsic_area','felsic_volume','felsic_age_moment','basalt_volume','basalt_age_moment']
        before={k:math.fsum(a[k]) for k in extensive}
        while True:
            world=self._world_triangles(tri,a['owner'].astype(int))
            active=_touches_sources(world,a['owner'],[(s,sigma,outer) for s in segments],caps,radius)
            edges=np.max(np.arctan2(np.linalg.norm(np.cross(tri,np.roll(tri,-1,axis=1)),axis=2),np.einsum('nij,nij->ni',tri,np.roll(tri,-1,axis=1))),axis=1)*radius
            split=active&(edges>self.area_params.max_edge_km)
            if not np.any(split):break
            count=int(sum(split))
            if len(roots)+3*count>self.area_params.max_leaves:raise RuntimeError('Active-area refinement exceeds explicit leaf budget; no refinement committed')
            if max(codes[split])>=2**30:raise RuntimeError('Active-area tree exceeds integer geometry domain')
            children=_children(tri[split]);areas=_areas(children.reshape(-1,3,3)).reshape(-1,4)
            parent_area=_areas(tri[split]);sums=areas.sum(axis=1)
            max_geometry_error=max(max_geometry_error,float(max(abs(sums-parent_area)/parent_area)))
            if max_geometry_error>1e-11:raise ValueError('Spherical subdivision area mismatch')
            weights=areas/sums[:,None];keep=~split
            for key,value in a.items():
                if key in extensive:
                    child=value[split,None]*weights
                    child[:,-1]=value[split]-np.sum(child[:,:-1],axis=1)
                    if np.any(child<0):raise ValueError('Negative conservative material child')
                else:child=np.repeat(value[split,None],4,axis=1)
                a[key]=np.concatenate([value[keep],child.reshape((-1,)+value.shape[1:])])
            tri=np.concatenate([tri[keep],children.reshape(-1,3,3)])
            roots=np.r_[roots[keep],np.repeat(roots[split],4)]
            codes=np.r_[codes[keep],(codes[split,None]*4+np.arange(4)).ravel()]
        a['points']=_centers(self._world_triangles(tri,a['owner'].astype(int)))
        errors={k:math.fsum(a[k])-before[k] for k in extensive}
        if any(abs(errors[k])>64*math.ulp(max(before[k],1.)) for k in extensive):raise ValueError('Refinement changed extensive material history')
        self.leaves.arrays=a;self.reference_triangles=tri;self.roots=roots;self.codes=codes
        return dict(leaves_before=initial,leaves_after=len(roots),max_subdivision_geometry_relative_error=max_geometry_error,
                    refinement_extensive_errors=errors)

    def locate(self,world_points,owner):
        """Inverse rigid rotation + exact root/child containment, no nearest label."""
        points=Rotation.from_quat(self.quaternions[int(owner)]).inv().apply(world_points)
        roots=self.receiver.locate(points);allowed=self.coarse.arrays['owner'][roots]==owner
        result=np.full(len(points),-1,int);indices=np.flatnonzero(allowed)
        if not len(indices):return result
        p=points[indices];r=roots[indices].astype(np.int64);codes=np.ones(len(r),np.int64)
        tri=self.mesh.vertices[self.mesh.faces[r]]
        leafkeys=self.roots*2**32+self.codes;order=np.argsort(leafkeys);sortedkeys=leafkeys[order]
        for _ in range(16):
            keys=r*2**32+codes;pos=np.searchsorted(sortedkeys,keys);found=(pos<len(sortedkeys))
            found&=sortedkeys[np.minimum(pos,len(sortedkeys)-1)]==keys
            result[indices[found]]=order[pos[found]]
            if np.all(found):return result
            keep=~found;indices=indices[keep];p=p[keep];r=r[keep];codes=codes[keep];tri=tri[keep]
            child=_children(tri);normals=np.cross(child,np.roll(child,-1,axis=2))
            centers=np.sum(child,axis=2);sign=np.where(np.einsum('nkij,nkj->nki',normals,centers)>=0,1.,-1.)
            dots=np.einsum('nkij,nj->nki',normals,p)*sign
            inside=np.all(dots>=-2e-13*np.linalg.norm(normals,axis=3),axis=2)
            if np.any(~np.any(inside,axis=1)):raise ValueError('Material point lost by spherical child containment')
            chosen=np.argmax(inside,axis=1);tri=child[np.arange(len(r)),chosen];codes=codes*4+chosen
        raise ValueError('Material quadrature failed to reach a saved leaf')

    def integrated_field(self,segments,caps,radius,sigma,outer):
        if self.area_params.strip_rule=='coarea':return self._coarea_field(segments,caps,radius,sigma,outer)
        indices=[];values=[];source=[];excluded=[];source_area=[];excluded_area=[];points_count=0
        ap=self.area_params
        streams=[]
        for s in segments:
            if s.amplitude==0:continue
            points,weights=strip_quadrature(s,radius,sigma,outer,ap.quadrature_spacing_km,ap.quadrature_order)
            n=np.cross(s.start,s.end);n/=np.linalg.norm(n)
            if n@s.toward<0:n=-n
            d=np.arcsin(np.clip(points@n,-1,1))*radius
            phi=s.amplitude*np.exp(-.5*((d-s.offset_km)/sigma)**2)
            streams.append((points,weights,weights/phi,s.overriding_plate))
        for center,owner,amplitude,csigma,couter in caps:
            if amplitude==0:continue
            points,weights=cap_quadrature(center,radius,csigma,couter,amplitude,ap.quadrature_spacing_km,ap.quadrature_order)
            d=np.arccos(np.clip(points@center,-1,1))*radius;phi=amplitude*np.exp(-.5*(d/csigma)**2)
            streams.append((points,weights,weights/phi,owner))
        for points,weights,area,owner in streams:
            cells=self.locate(points,owner);good=cells>=0
            indices.append(cells[good]);values.append(weights[good]);source.append(math.fsum(weights));excluded.append(math.fsum(weights[~good]))
            source_area.append(math.fsum(area));excluded_area.append(math.fsum(area[~good]));points_count+=len(points)
        n=len(self.roots)
        integral=_group_sum(np.concatenate(indices),np.concatenate(values),n) if indices else np.zeros(n)
        total=math.fsum(source);outside=math.fsum(excluded);error=math.fsum([*integral,outside,-total])
        if abs(error)>64*math.ulp(max(total,1.)):raise ValueError('Material strip integration budget failed')
        return integral/self.leaves.arrays['area'],dict(source_integral_km2=total,excluded_material_source_km2=outside,
            deposited_source_km2=math.fsum(integral),source_deposition_residual_km2=error,quadrature_points=points_count,
            source_supported_area_sum_km2=math.fsum(source_area),excluded_material_area_sum_km2=math.fsum(excluded_area),
            supported_area_scope='sum of source supports; overlaps count once per additive source, not union area')

    def _coarea_field(self,segments,caps,radius,sigma,outer):
        from .strip_intersection import strip_intersection_integrals
        integral=np.zeros(len(self.roots));sources=[];excluded=[];source_area=[];outside_area=[];inside_area=[];error_estimates=[];area_estimates=[];panels=0
        roots=self.mesh.vertices[self.mesh.faces];owner=self.leaves.arrays['owner'];rootowner=self.coarse.arrays['owner']
        for s in segments:
            if s.amplitude==0:continue
            inverse=Rotation.from_quat(self.quaternions[s.overriding_plate]).inv()
            local=replace(s,start=inverse.apply(s.start),end=inverse.apply(s.end),toward=inverse.apply(s.toward))
            take=_touches_sources(self.reference_triangles,owner,[(local,sigma,outer)],[],radius)
            # Independently integrate complementary material regions in this
            # frame. Exclusion is not defined as total minus deposited source.
            complement=_touches_sources(roots,np.full(len(roots),s.overriding_plate),[(local,sigma,outer)],[],radius)&(rootowner!=s.overriding_plate)
            tri=np.concatenate([self.reference_triangles[take],roots[complement]])
            q=self.area_params.quadrature_order
            low=strip_intersection_integrals(tri,local,radius,sigma,outer,q)
            high=strip_intersection_integrals(tri,local,radius,sigma,outer,2*q)
            count=int(sum(take));integral[take]+=high[:count,0];excluded.append(math.fsum(high[count:,0]));outside_area.append(math.fsum(high[count:,1]))
            inside_area.append(math.fsum(high[:count,1]));area_estimates.append(math.fsum(abs(high[:,1]-low[:,1])))
            error_estimates.append(math.fsum(abs(high[:,0]-low[:,0])));panels+=len(tri)
            _,weights=strip_quadrature(local,radius,sigma,outer,self.area_params.quadrature_spacing_km,32)
            sources.append(math.fsum(weights))
            length=math.atan2(np.linalg.norm(np.cross(local.start,local.end)),local.start@local.end)
            source_area.append(radius**2*length*(math.sin((local.offset_km+outer)/radius)-math.sin(max(0,local.offset_km-outer)/radius)))
        capdiag=None
        if caps:
            # Localized post-breakoff caps retain their explicitly reported
            # point quadrature; do not claim the strip proof applies to them.
            original=self.area_params
            try:
                self.area_params=replace(original,strip_rule='source_points')
                field,capdiag=self.integrated_field([],caps,radius,sigma,outer)
            finally:self.area_params=original
            integral+=field*self.leaves.arrays['area']
            sources.append(capdiag['source_integral_km2']);excluded.append(capdiag['excluded_material_source_km2'])
            source_area.append(capdiag['source_supported_area_sum_km2']);outside_area.append(capdiag['excluded_material_area_sum_km2'])
            inside_area.append(capdiag['source_supported_area_sum_km2']-capdiag['excluded_material_area_sum_km2'])
        total=math.fsum(sources);out=math.fsum(excluded);error=math.fsum([*integral,out,-total])
        total_area=math.fsum(source_area);area_error=math.fsum([*inside_area,*outside_area,-total_area])
        if max(abs(error),math.fsum(error_estimates))>self.area_params.source_budget_rtol*max(total,1.)+64*math.ulp(max(total,1.)):
            raise ValueError(f'Coarea source quadrature unresolved: residual {error} km2 / total {total}')
        if max(abs(area_error),math.fsum(area_estimates))>self.area_params.source_budget_rtol*max(total_area,1.)+64*math.ulp(max(total_area,1.)):
            raise ValueError(f'Coarea support quadrature unresolved: residual {area_error} km2 / area {total_area}')
        return integral/self.leaves.arrays['area'],dict(source_integral_km2=total,excluded_material_source_km2=out,
            deposited_source_km2=math.fsum(integral),source_deposition_residual_km2=error,
            coarea_order_difference_estimate_km2=math.fsum(error_estimates),coarea_triangle_pairs=panels,
            support_area_residual_km2=area_error,coarea_area_order_difference_estimate_km2=math.fsum(area_estimates),
            source_supported_area_sum_km2=math.fsum(source_area),excluded_material_area_sum_km2=math.fsum(outside_area),
            cap_quadrature_points=0 if capdiag is None else capdiag['quadrature_points'],
            supported_area_scope='physical strip support sums; overlapping sources additive; caps use point quadrature')

    def aggregate(self):
        a=self.leaves.arrays;b=self.coarse.arrays;n=len(b['area'])
        for k in ('area','felsic_area','felsic_volume','felsic_age_moment','basalt_volume','basalt_age_moment'):
            b[k]=_group_sum(self.roots,a[k],n)
        ocean=a['area']-a['felsic_area'];parent_ocean=b['area']-b['felsic_area']
        b['potential']=np.divide(_group_sum(self.roots,a['potential']*ocean,n),parent_ocean,out=np.zeros(n),where=parent_ocean>0)
        self.coarse.time_myr=self.leaves.time_myr

    def runtime_step(self,mesh,state,system,cycle,boundaries,memory,radius,dt,arc_params,cycle_params,reference_thickness):
        from .volcanic_arc import integrated_arc_sources
        from .lithosphere import LithosphereStepDiagnostics
        from .continental import ContinentalCycleDiagnostics
        self.rotate(system,dt/2)
        segments,caps,*_=integrated_arc_sources(mesh,memory,radius,arc_params,boundaries)
        refinement=self.refine(segments,caps,radius,arc_params.arc_half_width_km,arc_params.arc_outer_width_km)
        field,integration=self.integrated_field(segments,caps,radius,arc_params.arc_half_width_km,arc_params.arc_outer_width_km)
        self.aggregate()
        source,drain=self.coarse.boundary_rates(mesh,state,boundaries,radius,reference_thickness,arc_params.integration_spacing_km)
        a=self.leaves.arrays;b=self.coarse.arrays
        ocean=a['area']-a['felsic_area'];parent_ocean=b['area']-b['felsic_area']
        leaf_source=source[self.roots]*np.divide(ocean,parent_ocean[self.roots],out=np.zeros(len(ocean)),where=parent_ocean[self.roots]>0)
        reaction=self.leaves.react(dt,field,cycle_params,leaf_source,drain[self.roots])
        self.rotate(system,dt/2);self.aggregate()
        state,cycle,projection=self.coarse.project(mesh,state,cycle,radius)
        cycle.cumulative_generated_area_km2+=reaction['matured_area_km2'];cycle.cumulative_generated_volume_km3+=reaction['born_felsic_km3']
        cycle.cumulative_recycled_volume_km3+=reaction['returned_felsic_km3']
        fraction=math.fsum(a['felsic_area'])/math.fsum(a['area'])
        lith={k:0. for k in LithosphereStepDiagnostics.__dataclass_fields__}
        lith.update(time_myr=self.leaves.time_myr,dt_myr=dt,continental_area_fraction=fraction,visible_continental_area_fraction=fraction,
            oceanic_created_volume_km3=reaction['born_basalt_km3'],oceanic_recycled_volume_km3=reaction['returned_basalt_km3'],
            material_source_index=np.arange(mesh.cell_count,dtype=np.int32))
        cd={k:0. for k in ContinentalCycleDiagnostics.__dataclass_fields__}
        cd.update(time_myr=self.leaves.time_myr,dt_myr=dt,continental_area_fraction=fraction,continental_volume_km3=math.fsum(a['felsic_volume']),
            juvenile_arc_area_created_km2=reaction['matured_area_km2'],arc_thickening_volume_km3=reaction['arc_thickening_km3'],
            delaminated_volume_km3=reaction['returned_felsic_km3'],max_felsic_potential=float(max(a['potential'])))
        self.history.append(dict(time_myr=self.leaves.time_myr,**reaction,**projection,**refinement,**integration))
        return state,cycle,LithosphereStepDiagnostics(**lith),ContinentalCycleDiagnostics(**cd)
