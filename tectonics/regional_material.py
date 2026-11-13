"""Experimental moving regional columns, independent of the display mesh.

Areas are material quadrature weights, not resolved free-boundary polygons.
Conservative projection fills Eulerian receiver capacities and reports the
required displacement. It must not be interpreted as a contact-force solution.
Only sensible heat is owned by the runtime's existing finite inventory.
"""
from copy import deepcopy
from dataclasses import asdict,dataclass
import math
import numpy as np
from scipy.optimize import linprog,brentq
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from .boundary_exchange import frozen_exchange
from .integral_arcs import SphereReceiver
from .mesh import build_icosphere

_GL_NODES,_GL_WEIGHTS=np.polynomial.legendre.leggauss(16)


@dataclass(frozen=True)
class RegionalMaterialParameters:
    material_subdivisions:int=3
    projection_candidates:int=8
    projection_max_candidates:int=128
    projection_area_rtol:float=2e-10

    def __post_init__(self):
        if not 1<=self.material_subdivisions<=5:raise ValueError('Regional material resolution outside bounded experiment')
        if not 2<=self.projection_candidates<=self.projection_max_candidates<=256:raise ValueError('Invalid projection graph')
        if not 0<self.projection_area_rtol<=1e-8:raise ValueError('Invalid projection accuracy')


def conservative_projection(points,material_areas,mesh,radius_km,params):
    """Minimum angular-distance transport; no source/sink or history averaging.

    Normalize each material row after the linear solve to preserve extensive
    inventories to roundoff. The independently checked column-capacity error
    is NOT renormalized into matter. A graph that cannot represent the map fails.
    """
    area=np.asarray(material_areas,float);target=mesh.physical_cell_areas_km2(radius_km)
    if np.any(area<=0) or abs(math.fsum(area)-math.fsum(target))>1e-12*math.fsum(target):
        raise ValueError('Regional/display area mismatch')
    count=len(area);n=mesh.cell_count;scale=float(np.mean(area))
    tree=cKDTree(mesh.centroids);k=min(params.projection_candidates,n);solution=None
    while True:
        dist,cols=tree.query(points,k=k);rows=np.repeat(np.arange(count),k)
        columns=cols.ravel();cost=dist.ravel()**2
        variables=np.arange(len(rows))
        constraints=coo_matrix((np.ones(2*len(rows)),(np.r_[rows,count+columns],np.r_[variables,variables])),shape=(count+n,len(rows))).tocsr()
        result=linprog(cost,A_eq=constraints,b_eq=np.r_[area,target]/scale,bounds=(0,None),method='highs',
                       options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
        if result.success:solution=result.x*scale;break
        if result.status!=2 or k>=min(n,params.projection_max_candidates):
            raise ValueError(f'Regional projection graph failed at k={k}: {result.message}')
        k=min(k*2,n,params.projection_max_candidates)
    keep=solution>0;rows=rows[keep];columns=columns[keep];values=solution[keep]
    row_sum=np.bincount(rows,weights=values,minlength=count)
    if np.any(row_sum<=0):raise ValueError('Regional projection lost a material row')
    values*=area[rows]/row_sum[rows]
    got=np.bincount(columns,weights=values,minlength=n)
    relative=float(np.max(np.abs(got-target)/target))
    if relative>params.projection_area_rtol:raise ValueError(f'Regional receiver-capacity residual {relative}')
    weights=coo_matrix((values/area[rows],(columns,rows)),shape=(n,count)).tocsr()
    displacement=np.arccos(np.clip(np.sum(points[rows]*mesh.centroids[columns],axis=1),-1,1))*radius_km
    return weights,{'max_receiver_area_relative_error':relative,'rms_projection_distance_km':float(np.sqrt(np.sum(values*displacement**2)/sum(area))),
                    'max_projection_distance_km':float(max(displacement)),'projection_candidates':k}


def arc_field_at_points(points,owners,segments,radius_km,sigma_km,outer_km,caps=()):
    """Physical strip field before receiver-cell averaging. Sum actual sources."""
    field=np.zeros(len(points));support=np.zeros(len(points),bool)
    for s in segments:
        a=np.asarray(s.start);b=np.asarray(s.end);normal=np.cross(a,b);normal/=np.linalg.norm(normal)
        if normal@s.toward<0:normal=-normal
        tangent=b-a*(a@b);tangent/=np.linalg.norm(tangent)
        length=math.atan2(np.linalg.norm(np.cross(a,b)),a@b)
        along=np.arctan2(points@tangent,points@a);distance=np.arcsin(np.clip(points@normal,-1,1))*radius_km
        mask=(owners==s.overriding_plate)&(along>=0)&(along<length)&(distance>=0)&(abs(distance-s.offset_km)<=outer_km)
        field[mask]+=s.amplitude*np.exp(-.5*((distance[mask]-s.offset_km)/sigma_km)**2);support|=mask
    for center,plate,amplitude,sigma,outer in caps:
        d=np.arccos(np.clip(points@center,-1,1))*radius_km
        mask=(owners==plate)&(d<=outer);field[mask]+=amplitude*np.exp(-.5*(d[mask]/sigma)**2);support|=mask
    return field,support


def _pvalue(t,v,p,source,drain,forcing,decay):
    """Exact frozen linear mass/potential equations; births carry zero history."""
    t=np.asarray(t);k=np.asarray(drain);s=np.asarray(source);v=np.asarray(v);p=np.asarray(p);q=np.asarray(forcing)
    V=frozen_exchange(v,np.zeros_like(v),s,k,t)[0] if t.ndim==0 else np.exp(-k*t)*v+s*t*np.where(k*t==0,1.,-np.expm1(-k*t)/np.where(k*t==0,1.,k*t))
    ed=np.exp(-decay*t);ek=np.exp(-k*t);E=-np.expm1(-decay*t)/decay
    a=k+decay;Ea=-np.expm1(-a*t)/a
    incoming=np.zeros_like(v);small=(k*t<1e-3)&(s>0);regular=(k*t>=1e-3)&(s>0)
    incoming[regular]=(Ea[regular]-ek[regular]*E[regular] if E.ndim else Ea[regular]-ek[regular]*E)/k[regular]
    # Positive Gaussian quadrature avoids cancellation in a divided difference.
    # Only the small-k branch uses it; dt/tau is bounded by the runtime guard.
    if np.any(small):
        x=(_GL_NODES+1)/2
        tt=np.broadcast_to(t,v.shape)[small];kk=k[small];xx=kk[:,None]*tt[:,None]*x
        phi=np.ones_like(xx);np.divide(-np.expm1(-xx),xx,out=phi,where=xx!=0)
        incoming[small]=tt**2*np.sum(_GL_WEIGHTS[None,:]/2*np.exp(-(kk[:,None]+decay)*tt[:,None]*(1-x))*x*phi,axis=1)
    y=np.exp(-a*t)*v*p+q*(v*ek*E+s*incoming)
    return np.divide(y,V,out=np.zeros_like(v),where=V>0)


class RegionalMaterialInventory:
    def __init__(self,arrays,params,time_myr=0.,history=()):
        self.arrays={k:np.array(v,copy=True) for k,v in arrays.items()};self.params=params
        self.time_myr=float(time_myr);self.history=list(history)
        n=len(self.arrays['area'])
        if any(np.any(~np.isfinite(v)) or len(v)!=n for v in self.arrays.values()):raise ValueError('Invalid regional inventory')
        if np.any(self.arrays['area']<=0):raise ValueError('Nonpositive regional material area')
        for key,v in self.arrays.items():
            if key not in ('points','owner') and np.any(v<0):raise ValueError('Negative regional material field')
        if np.any(self.arrays['felsic_area']>self.arrays['area']):raise ValueError('Regional footprint exceeds its area')

    @classmethod
    def initialize(cls,mesh,state,radius_km,params):
        if state.time_myr!=0 or np.any(state.sediment_volume_km3):raise ValueError('Regional mode requires explicit zero-time sediment-free initialization')
        material=build_icosphere(params.material_subdivisions)
        if material.cell_count<mesh.cell_count:raise ValueError('Initial material quadrature cannot be coarser than initialization')
        parent=SphereReceiver(mesh).locate(material.centroids);a=material.physical_cell_areas_km2(radius_km);old=mesh.physical_cell_areas_km2(radius_km)
        if max(abs(np.bincount(parent,weights=a,minlength=mesh.cell_count)-old))>1e-7:raise ValueError('Initial geometry is not nested')
        ratio=a/old[parent];f=state.continental_fraction[parent];fv=state.continental_volume_km3[parent]*ratio
        ov=state.oceanic_volume_km3[parent]*ratio
        return cls({'points':material.centroids,'owner':state.cell_plate[parent],'area':a,'felsic_area':a*f,'felsic_volume':fv,
                    'felsic_age_moment':fv*state.crust_age_myr[parent],'basalt_volume':ov,
                    'basalt_age_moment':state.oceanic_age_moment_km3_myr[parent]*ratio,'potential':np.zeros(len(a))},params)

    def checkpoint(self):
        return {'version':1,'parameters':asdict(self.params),'time_myr':self.time_myr,'history':deepcopy(self.history),
                'arrays':{k:v.tolist() for k,v in self.arrays.items()},
                'scope':'new moving material quadrature; fixed column areas; projection is not a contact solution'}

    @classmethod
    def restore(cls,data,params):
        if data.get('version')!=1 or data['parameters']!=asdict(params):raise ValueError('Regional restart parameters/history mismatch')
        return cls(data['arrays'],params,data['time_myr'],data['history'])

    def rotate(self,system,dt):
        from .kinematics import angular_velocity_vectors
        a=self.arrays;w=angular_velocity_vectors(system)[a['owner'].astype(int)];angle=np.linalg.norm(w,axis=1)*dt
        axis=np.divide(w,np.linalg.norm(w,axis=1)[:,None],out=np.zeros_like(w),where=np.linalg.norm(w,axis=1)[:,None]>0)
        p=a['points'];a['points']=p*np.cos(angle)[:,None]+np.cross(axis,p)*np.sin(angle)[:,None]+axis*np.sum(axis*p,axis=1)[:,None]*(1-np.cos(angle))[:,None]
        a['points']/=np.linalg.norm(a['points'],axis=1)[:,None]

    def react(self,dt,field,cycle_params,basalt_source=None,basalt_drain=None):
        """Finite-column reaction, localized maturation, gross source/sink ledger."""
        if dt<=0 or dt>cycle_params.arc_potential_decay_myr:raise ValueError('Regional reaction step outside tested dt/tau<=1 range')
        a=self.arrays;n=len(a['area']);v=a['basalt_volume'].copy();p=a['potential'].copy()
        s=np.zeros(n) if basalt_source is None else np.asarray(basalt_source);k=np.zeros(n) if basalt_drain is None else np.asarray(basalt_drain)
        if any(x.shape!=(n,) or np.any(~np.isfinite(x)) or np.any(x<0) for x in (np.asarray(field),s,k)):
            raise ValueError('Invalid regional source field')
        q=np.asarray(field)*cycle_params.arc_maturation_rate_per_myr;d=1/cycle_params.arc_potential_decay_myr
        threshold=cycle_params.juvenile_threshold;times=np.full(n,dt);cross=np.zeros(n,bool)
        end=_pvalue(dt,v,p,s,k,q,d);ocean=a['felsic_area']<a['area']*(1-1e-14)
        # Removing the nonnegative dilution term gives a rigorous upper bound
        # over this frozen interval, including any possible transient maximum.
        upper=np.maximum(p,p*np.exp(-d*dt)+q*(-math.expm1(-d*dt))/d)
        candidates=ocean&(v+s*dt>0)&(upper>=threshold)
        for j in np.flatnonzero(candidates):
            def val(t):return float(_pvalue(float(t),v[j:j+1],p[j:j+1],s[j:j+1],k[j:j+1],q[j:j+1],d)[0]) if t else float(p[j])
            def deriv(t):
                vt=float(frozen_exchange(v[j:j+1],[0.],s[j:j+1],k[j:j+1],t)[0][0]) if t else v[j]
                pt=val(t)
                return q[j]-d*pt-s[j]*(pt/max(vt,1e-300))
            peak=dt
            if deriv(0)>0 and deriv(dt)<0:peak=brentq(deriv,0,dt,xtol=1e-11)
            if val(peak)>=threshold:
                times[j]=0. if p[j]>=threshold else brentq(lambda t:val(t)-threshold,0,peak,xtol=1e-11)
                cross[j]=True
        ov=np.empty(n);om=np.empty(n);born=np.empty(n);returned=np.empty(n)
        for t in np.unique(times):
            use=times==t
            if t==0:ov[use]=v[use];om[use]=a['basalt_age_moment'][use];born[use]=returned[use]=0.
            else:ov[use],om[use],born[use],returned[use]=frozen_exchange(v[use],a['basalt_age_moment'][use],s[use],k[use],float(t))
        old_fa=a['felsic_area'].copy();added_area=np.where(cross,a['area']-old_fa,0.)
        juvenile=added_area*cycle_params.juvenile_continental_thickness_km
        rate=np.asarray(field)*cycle_params.continental_arc_thickening_km_per_myr
        thick_old=old_fa*rate*dt;thick_new=added_area*rate*(dt-times)
        a['felsic_age_moment']+=a['felsic_volume']*dt+thick_old*dt/2+juvenile*(dt-times)+thick_new*(dt-times)/2
        a['felsic_volume']+=juvenile+thick_old+thick_new;a['felsic_area']+=added_area
        returned[cross]+=ov[cross];ov[cross]=0.;om[cross]=0.
        a['basalt_volume']=ov;a['basalt_age_moment']=om;a['potential']=np.where(cross,0.,end)
        thickness=np.divide(a['felsic_volume'],a['felsic_area'],out=np.zeros(n),where=a['felsic_area']>0)
        delam=np.where(thickness>cycle_params.delamination_threshold_km,
            a['felsic_area']*np.maximum(0,thickness-cycle_params.delamination_target_km)*(-math.expm1(-cycle_params.delamination_rate_per_myr*dt)),0.)
        keep=np.divide(a['felsic_volume']-delam,a['felsic_volume'],out=np.ones(n),where=a['felsic_volume']>0)
        a['felsic_age_moment']*=keep;a['felsic_volume']-=delam
        self.time_myr+=dt
        result={'born_felsic_km3':math.fsum(juvenile+thick_old+thick_new),'returned_felsic_km3':math.fsum(delam),
                'born_basalt_km3':math.fsum(born),'returned_basalt_km3':math.fsum(returned),
                'matured_area_km2':math.fsum(added_area),'arc_thickening_km3':math.fsum(thick_old+thick_new),
                'events':int(sum(cross)),'first_event_myr':self.time_myr-dt+float(min(times[cross])) if any(cross) else None}
        return result

    def boundary_rates(self,mesh,state,boundaries,radius_km,reference_thickness_km,spacing_km=200.):
        """Line-quadrature requests deposited to same-owner regional columns.

        Coarse columns mix arriving zero-age basalt explicitly. Their fixed
        quadrature area is NOT newly opened geometrical surface area.
        """
        from .kinematics import BoundaryType
        from .subduction_memory import choose_subducting_side
        a=self.arrays;n=len(a['area']);source=np.zeros(n);drain=np.zeros(n)
        trees={}
        for pid in np.unique(a['owner']):
            indices=np.flatnonzero(a['owner']==pid);trees[int(pid)]=(indices,cKDTree(a['points'][indices]))
        for b in boundaries:
            u,w=mesh.vertices[b.vertex_u],mesh.vertices[b.vertex_v];angle=math.atan2(np.linalg.norm(np.cross(u,w)),u@w)
            tangent=w-u*(u@w);tangent/=np.linalg.norm(tangent)
            panels=max(1,math.ceil(angle*radius_km/spacing_km));theta=(np.arange(panels)+.5)*angle/panels
            points=u*np.cos(theta)[:,None]+tangent*np.sin(theta)[:,None]
            if b.boundary_type==BoundaryType.DIVERGENT:
                for pid in (b.plate_a,b.plate_b):
                    indices,tree=trees[pid];targets=indices[tree.query(points)[1]]
                    amount=.5*max(b.normal_rate_km_per_myr,0.)*angle*radius_km/panels
                    np.add.at(source,targets,amount*reference_thickness_km*(1-a['felsic_area'][targets]/a['area'][targets]))
            elif b.boundary_type==BoundaryType.CONVERGENT:
                pid=choose_subducting_side(state,b)
                if pid is None:continue
                indices,tree=trees[pid];targets=indices[tree.query(points)[1]]
                amount=max(-b.normal_rate_km_per_myr,0.)*angle*radius_km/panels
                np.add.at(drain,targets,amount/a['area'][targets]*(a['felsic_area'][targets]<a['area'][targets]))
        return source,drain

    def runtime_step(self,mesh,state,system,cycle,boundaries,memory,radius_km,dt,arc_params,cycle_params,reference_thickness_km):
        from .volcanic_arc import integrated_arc_sources
        from .lithosphere import LithosphereStepDiagnostics
        from .continental import ContinentalCycleDiagnostics
        self.rotate(system,dt/2)
        segments,caps,*_=integrated_arc_sources(mesh,memory,radius_km,arc_params,boundaries)
        field,support=arc_field_at_points(self.arrays['points'],self.arrays['owner'],segments,radius_km,
                                          arc_params.arc_half_width_km,arc_params.arc_outer_width_km,caps)
        source,drain=self.boundary_rates(mesh,state,boundaries,radius_km,reference_thickness_km,arc_params.integration_spacing_km)
        reaction=self.react(dt,field,cycle_params,source,drain)
        self.rotate(system,dt/2)
        state,cycle,projection=self.project(mesh,state,cycle,radius_km)
        cycle.cumulative_generated_area_km2+=reaction['matured_area_km2']
        cycle.cumulative_generated_volume_km3+=reaction['born_felsic_km3']
        cycle.cumulative_recycled_volume_km3+=reaction['returned_felsic_km3']
        a=self.arrays;total=math.fsum(a['area']);cf=math.fsum(a['felsic_area'])/total
        lith={key:0. for key in LithosphereStepDiagnostics.__dataclass_fields__}
        lith.update(time_myr=self.time_myr,dt_myr=dt,continental_area_fraction=cf,visible_continental_area_fraction=cf,
            oceanic_created_volume_km3=reaction['born_basalt_km3'],oceanic_recycled_volume_km3=reaction['returned_basalt_km3'],
            material_source_index=np.arange(mesh.cell_count,dtype=np.int32))
        cd={key:0. for key in ContinentalCycleDiagnostics.__dataclass_fields__}
        cd.update(time_myr=self.time_myr,dt_myr=dt,continental_area_fraction=cf,continental_volume_km3=math.fsum(a['felsic_volume']),
            juvenile_arc_area_created_km2=reaction['matured_area_km2'],arc_thickening_volume_km3=reaction['arc_thickening_km3'],
            delaminated_volume_km3=reaction['returned_felsic_km3'],max_felsic_potential=float(max(a['potential'])))
        self.history.append({'time_myr':self.time_myr,**reaction,**projection,
            'physical_source_sample_integral_km2':math.fsum(a['area']*field),'active_material_area_km2':math.fsum(a['area'][support])})
        return state,cycle,LithosphereStepDiagnostics(**lith),ContinentalCycleDiagnostics(**cd)

    def project(self,mesh,state,cycle,radius_km):
        a=self.arrays;weights,diagnostics=conservative_projection(a['points'],a['area'],mesh,radius_km,self.params)
        result=deepcopy(state);areas=mesh.physical_cell_areas_km2(radius_km);mapped=weights@a['area']
        result.continental_fraction=(weights@a['felsic_area'])/mapped
        result.continental_volume_km3=weights@a['felsic_volume'];result.oceanic_volume_km3=weights@a['basalt_volume']
        result.oceanic_age_moment_km3_myr=weights@a['basalt_age_moment']
        result.crust_type=(result.continental_fraction>=.5).astype(np.int8)
        result.crust_thickness_km=np.where(result.crust_type,
            result.continental_volume_km3/np.maximum(areas*result.continental_fraction,1e-300),
            result.oceanic_volume_km3/np.maximum(areas*(1-result.continental_fraction),1e-300))
        result.crust_age_myr=np.where(result.crust_type,(weights@a['felsic_age_moment'])/np.maximum(result.continental_volume_km3,1e-300),
            result.oceanic_age_moment_km3_myr/np.maximum(result.oceanic_volume_km3,1e-300))
        owners=np.array([weights@np.where(a['owner']==i,a['area'],0.) for i in range(int(max(a['owner']))+1)])
        result.cell_plate=np.argmax(owners,axis=0).astype(np.int32);result.time_myr=self.time_myr
        cycle=deepcopy(cycle);cycle.felsic_potential=(weights@(a['potential']*(a['area']-a['felsic_area'])))/np.maximum(weights@(a['area']-a['felsic_area']),1e-300);cycle.time_myr=self.time_myr
        return result,cycle,diagnostics
