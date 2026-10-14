"""Opt-in geological Genesis: one shell, prescribed ocean, shared mature engine.

Phenomenological strain accumulation is not an SI fracture/work law. Detailed
Maxwell mechanics is archived, not equilibrated or silently turned into heat.
Ocean temperature is a declared liquid thermostat; its heat export is measured.
"""
from copy import deepcopy
from dataclasses import asdict,dataclass
from pathlib import Path
from types import SimpleNamespace
import json,math
import numpy as np

from .genesis_geological_handoff import GeologicalContinuation,DENSITY,sha
from .genesis_transfer_bridge import encoded
from .genesis_mature import build_experimental_genesis_mature_import
from .genesis_starter_topology import select_starter_cut,split_starter_band,canonicalize_plate_seeds
from .genesis_implicit_enthalpy import solve,invert,phase
from .experimental_geological_parcels import initialize,transport,sync_thermal
from .genesis import SECONDS_PER_MYR
from .mesh import build_icosphere,connected_components
from .plates import Plate,PlateSystem
from .thermal import ThermalState
from .dynamics import DynamicsParameters,angular_velocity_vectors
from .mantle import advance_mantle_flow,MantleFlowParameters
from .transport import remap_transport_state
from .topology import PlateTopologyParameters
from .kinematics import classify_boundaries
from .topography import TopographyParameters,advance_topography
from .hydrosphere import HydrosphereParameters,diagnose_hydrosphere,advance_hydrosphere
from .tides import G,M_JUPITER


@dataclass(frozen=True)
class GeologicalGenesisParameters:
    seed:int=20261014
    mantle_rms_km_per_myr:float=30. # 3cm/year, scenario forcing, not SI reconstruction
    critical_strain:float=.1
    rupture_damage:float=.65
    healing_myr:float=50.
    reference_thickness_km:float=7.
    min_strength_fraction:float=.1
    min_child_cells:int=6
    ocean_temperature_k:float=300.
    mantle_boundary_layer_km:float=100.
    conductivity_w_m_k:float=3.6
    strength_log_std:float=.2
    strength_wavenumber_min:float=2.
    strength_wavenumber_max:float=5.


def temperature(mass,energy):
    n=len(mass);cp=np.full(n,1200.);cp[-1]=4200.;latent=np.full(n,4e5);latent[-1]=0
    return invert(np.asarray(energy)/mass,cp,latent,np.full(n,1400.),np.full(n,2000.),np.zeros(n))


class UnifiedGeologicalGenesis(GeologicalContinuation):
    FORMAT='unified-geological-genesis/v1'

    def __init__(self,checkpoint,record,source):
        super().__init__(checkpoint,record,source)
        if 'topology_parameters' in record:
            self.checkpoint.manager.params=PlateTopologyParameters(**record['topology_parameters'])

    @classmethod
    def from_checkpoint(cls,path,parameters=None):
        p=parameters or GeologicalGenesisParameters()
        if p.critical_strain<=0 or p.healing_myr<=0 or not 0<p.rupture_damage<1 or not 273.16<p.ocean_temperature_k<373.15:
            raise ValueError('Invalid geological scenario parameters')
        if (not all(np.isfinite(v) for v in asdict(p).values()) or p.mantle_rms_km_per_myr<0
            or p.strength_log_std<0 or not 0<p.strength_wavenumber_min<=p.strength_wavenumber_max
            or p.min_child_cells<1 or p.reference_thickness_km<=0 or p.min_strength_fraction<=0
            or p.conductivity_w_m_k<=0 or p.mantle_boundary_layer_km<=0):raise ValueError('Invalid geological material parameters')
        blob=Path(path).read_bytes();body=json.loads(blob)['payload'];c,s=body['config'],body['state']
        if 'equilibrated_start' in body.get('origin',{}):raise ValueError('Weakened scenario cannot be the unified source')
        nf=len(c['solid_nodes']);level=0;count=nf
        while count>20:count//=4;level+=1
        mesh=build_icosphere(level);nodes=np.array(c['solid_nodes']).ravel()
        if mesh.cell_count!=nf or s.get('born_material'):raise ValueError('Use approved unadvanced single-shell source')
        mass0=np.array(c['mass']);H0=np.array(s['energy']);retained=set(nodes)
        other=np.array([i for i in range(len(mass0)-1) if i not in retained])
        mass=np.r_[mass0[nodes],math.fsum(mass0[other]),mass0[-1]]
        energy=np.r_[H0[nodes],math.fsum(H0[other]),mass[-1]*4200*p.ocean_temperature_k]
        bins=np.arange(len(nodes)).reshape(nf,-1);radius=c['radius_km'];areas=mesh.physical_cell_areas_km2(radius)
        time=float(s['time_myr']);T=temperature(mass,energy);face_mass=mass[bins].sum(1);faceH=energy[bins].sum(1)
        # A seeded smooth convection field, independent of any plate/continent map.
        rng=np.random.default_rng(p.seed);A=rng.normal(size=(3,3));A=(A+A.T)/2;A-=np.eye(3)*np.trace(A)/3
        n=mesh.centroids;grad=n@A;grad-=n*np.sum(n*grad,axis=1)[:,None]
        axis=rng.normal(size=3);axis/=np.linalg.norm(axis);dot=n@axis
        grad+=.35*(3*dot**2-1)[:,None]*(axis-dot[:,None]*n)
        grad*=p.mantle_rms_km_per_myr/np.sqrt(np.average(np.sum(grad**2,axis=1),weights=areas))
        mantle=np.cross(n,grad)/radius
        field=np.zeros(nf)
        for _ in range(8):
            axis=rng.normal(size=3);axis/=np.linalg.norm(axis)
            field+=np.cos(rng.uniform(p.strength_wavenumber_min,p.strength_wavenumber_max)*(n@axis)+rng.uniform(0,2*np.pi))
        heterogeneity=np.exp(p.strength_log_std*(field-field.mean())/field.std())
        owner=np.zeros(nf,dtype=np.int32);system=PlateSystem(owner,(Plate(0,0,np.array([0.,0.,1.]),0.),))
        thickness=face_mass/(DENSITY*1e9*areas)
        thermal=ThermalState(time,time,float(T[-2]),.1,1.,float(np.average(thickness,weights=areas)))
        water=mass[-1]/1000/1e9
        bundle=build_experimental_genesis_mature_import(mesh,radius_km=radius,system=system,
            crust_age_myr=np.zeros(nf),crust_thickness_km=thickness,tidal_damage=np.array(s['damage']),
            mantle_lithosphere_thickness_km=np.zeros(nf),mantle_lithosphere_density_anomaly_kg_m3=np.zeros(nf),
            thermal=thermal,elevation_m=np.zeros(nf),water_volume_km3=water,mantle_cell_omega_rad_per_myr=mantle,
            source_mass_kg=face_mass,source_enthalpy_j=faceH,next_plume_birth_time_myr=time+1e6,
            source_metadata=dict(mode=cls.FORMAT,composition='undifferentiated silicate, mafic reference carrier; no continents inferred'))
        compact=dict(config=dict(mass=mass.tolist(),solid_nodes=bins.tolist()),state=dict(energy=energy.tolist(),born_material=[]),
                     original_path=str(path),original_sha256=sha(blob),original_solid_node_ids=nodes.tolist(),
                     original_time_myr=time,aggregated_interior_nodes=other.tolist(),retired_mechanics='archived detailed checkpoint, no constitutive consumer or artificial heat')
        parcel=initialize(SimpleNamespace(config=compact['config'],state=compact['state']),bins.tolist())
        tide=c['tidal_parameters'];period=2*np.pi*np.sqrt((s['orbit']['semimajor_axis_km']*1000)**3/(G*tide['primary_mass_kg']))/3600
        record=dict(format=cls.FORMAT,subdivisions=level,radius_km=radius,parameters=asdict(p),
            topology_parameters=asdict(PlateTopologyParameters(min_plate_cells=p.min_child_cells,
                min_plate_area_km2=float(p.min_child_cells*areas.mean()),
                split_min_child_area_km2=float(p.min_child_cells*areas.mean()),
                split_min_rift_span_km=float(2*np.sqrt(areas.mean())),split_differential_speed_deg_per_myr=0.)),
            source_sha256=sha(encoded(compact)),steps=[],events=[],dynamics_parameters=asdict(DynamicsParameters(mantle_projection='velocity_least_squares')),
            gravity_m_s2=c['thermal_parameters']['surface_gravity_m_s2'],period_hours=float(period),eccentricity=s['orbit']['eccentricity'],
            primary_mass_jupiter=tide['primary_mass_kg']/M_JUPITER,love_h2=tide['love_h2'],
            orbit_approximation='source synchronous Kepler period and current eccentricity frozen; native degree-2 fatigue, no new tidal solver or orbit evolution',
            interior_reference_mass_kg=float(mass[-2]),thermal_inventory=dict(mass_kg=mass.tolist(),energy_j=energy.tolist()),
            initial_total_mass_kg=math.fsum(mass),initial_H_j=math.fsum(energy),water_mass_kg=float(mass[-1]),water_volume_km3=water,
            water_initialization=dict(temperature_k=p.ocean_temperature_k,liquid_kg=float(mass[-1]),vapor_kg=0.,ice_kg=0.,
                signed_H_change_from_original_j=float(energy[-1]-H0[-1]),condensation_history_reproduced=False,climate='fixed liquid thermostat, heat export recorded'),
            material_parcels=parcel,fracture_damage=np.zeros(nf).tolist(),material_strength=heterogeneity.tolist(),cumulative_exported_heat_j=0.,
            approximations=['phenomenological strain-to-fracture, no SI fracture energy','mature effective force proxies, no SI mechanical work closure',
                'canonical face footprint; original mass and silicate enthalpy retained','cohort thermal profiles remapped into fixed bins; other interior aggregated',
                'native newly generated 7km noncontinental carrier; not continental differentiation','ocean liquid initial condition, fixed 300K climate boundary'])
        engine=cls(bundle.checkpoint,record,compact)
        initial=engine.water_diagnostics()
        if initial['ocean_area_fraction']<1-1e-12:raise ValueError('Initial ocean must cover the whole sphere')
        engine.record['initial_water']=initial
        return engine

    def water_diagnostics(self):
        cp=self.checkpoint
        d=asdict(diagnose_hydrosphere(self.mesh,cp.state,cp.topo,cp.hydrosphere,self.record['radius_km'],HydrosphereParameters(),TopographyParameters()))
        d.update(liquid_mass_kg=self.record['water_mass_kg'],vapor_mass_kg=0.,ice_mass_kg=0.,
                 min_floor_m=float(cp.topo.elevation_m.min()),max_floor_m=float(cp.topo.elevation_m.max()))
        return d

    def rates(self):
        p=GeologicalGenesisParameters(**self.record['parameters']);cp=self.checkpoint;n=self.mesh.centroids;r=self.record['radius_km']
        v=np.cross(cp.mantle_flow.cell_omega_rad_per_myr,r*n)
        # Symmetric local velocity gradient; rigid rotation gives zero strain.
        strain=np.zeros(self.mesh.cell_count)
        for i,neighbors in enumerate(self.mesh.neighbors):
            neighbors=np.asarray(neighbors,dtype=int)
            dx=r*(n[neighbors]-n[i]);dv=v[neighbors]-v[i]
            gradient=np.linalg.lstsq(dx,dv,rcond=None)[0].T
            P=np.eye(3)-np.outer(n[i],n[i]);symmetric=P@(gradient+gradient.T)/2@P
            strain[i]=np.linalg.norm(symmetric)
        inv=self.record['thermal_inventory'];T=temperature(np.array(inv['mass_kg']),inv['energy_j']);nodes=np.array(self.source['config']['solid_nodes'])
        hot=np.clip((T[nodes].mean(1)-300)/1100,0,1)
        thickness=cp.state.oceanic_volume_km3/self.mesh.physical_cell_areas_km2(r)
        strength=np.maximum(p.min_strength_fraction,np.sqrt(thickness/p.reference_thickness_km))*(1-.7*hot)
        strength*=np.array(self.record['material_strength'])*(1-.5*cp.state.tidal_damage)
        return strain/(p.critical_strain*strength)

    def damage_at(self,dt,rates):
        p=GeologicalGenesisParameters(**self.record['parameters']);old=np.array(self.record['fracture_damage']);rr=rates+1/p.healing_myr
        equilibrium=rates/rr
        return equilibrium+(old-equilibrium)*np.exp(-rr*dt)

    def candidate(self,damage):
        p=GeologicalGenesisParameters(**self.record['parameters']);area=self.mesh.physical_cell_areas_km2(self.record['radius_km']).mean()
        system=canonicalize_plate_seeds(self.mesh,self.checkpoint.system)
        return select_starter_cut(self.mesh,system,np.asarray(damage)>=p.rupture_damage-1e-12,np.asarray(damage),
                                  self.record['radius_km'],p.min_child_cells*area,2*np.sqrt(area))

    def event_time(self,dt,rates):
        p=GeologicalGenesisParameters(**self.record['parameters']);old=np.array(self.record['fracture_damage']);rr=rates+1/p.healing_myr;eq=rates/rr
        with np.errstate(divide='ignore',invalid='ignore'):
            ratio=(p.rupture_damage-eq)/(old-eq);times=-np.log(ratio)/rr
        # Selector changes only at eligibility crossings with these frozen rates.
        for t in sorted(set(float(x) for x in times if np.isfinite(x) and 1e-9<x<=dt)):
            if self.candidate(self.damage_at(t,rates)) is not None:return t
        return dt

    def install_system(self,cp,system):
        cp.transport_state=remap_transport_state(cp.system,system,cp.transport_state)
        cp.system=system;cp.state.cell_plate=system.cell_plate.copy()

    def try_split(self):
        d=np.array(self.record['fracture_damage']);cut=self.candidate(d)
        if cut is None:return False
        p=GeologicalGenesisParameters(**self.record['parameters']);area=self.mesh.physical_cell_areas_km2(self.record['radius_km']).mean()
        old=canonicalize_plate_seeds(self.mesh,self.checkpoint.system)
        new,event=split_starter_band(self.mesh,old,cut,self.record['radius_km'],p.min_child_cells*area,2*np.sqrt(area),time_myr=self.checkpoint.state.time_myr)
        if new is None:return False
        self.checkpoint.manager._remap_collision_memory(old,new)
        self.install_system(self.checkpoint,new);d[cut]=0.;self.record['fracture_damage']=d.tolist()
        row=asdict(event);row.update(kind='geological_weak_band_split',cut=cut.tolist(),rule='connected eligible band; inherited velocities, no separating kick',damage_consumption='phenomenological release on cut, no fictitious heat')
        self.record['events'].append(row);self.checkpoint.events.append(deepcopy(row));return True

    def step(self,dt_myr):
        if not np.isfinite(dt_myr) or not 0<dt_myr<=2:raise ValueError('Geological macro step must be in (0,2] Myr')
        trial=type(self)(deepcopy(self.checkpoint),self.record,self.source);target=trial.checkpoint.state.time_myr+dt_myr
        count=0;first=len(trial.record['steps']);events=len(trial.record['events'])
        while target-trial.checkpoint.state.time_myr>1e-10:
            count+=1
            if count>128:raise ValueError('Bounded event budget exceeded')
            if trial.try_split():continue
            rates=trial.rates();dt=trial.event_time(target-trial.checkpoint.state.time_myr,rates)
            trial.record['fracture_damage']=trial.damage_at(dt,rates).tolist()
            GeologicalContinuation.step(trial,dt)
        trial.try_split()
        self.checkpoint,self.record=trial.checkpoint,trial.record
        return dict(time_myr=self.checkpoint.state.time_myr,accepted_substeps=len(self.record['steps'])-first,events=len(self.record['events'])-events,plates=len(self.checkpoint.system.plates))

    def _after_material_step(self,old,new,source_index,audit):
        p=GeologicalGenesisParameters(**self.record['parameters']);inv=self.record['thermal_inventory']
        mass=np.array(inv['mass_kg']);energy=np.array(inv['energy_j']);M0=math.fsum(mass);H0=math.fsum(energy)
        transfer=transport(self.record['material_parcels'],mass,energy,source_index,new.state.oceanic_volume_km3*DENSITY*1e9,new.state.time_myr)
        for actual,key in [(transfer['created_mass'],'generated_volume_km3'),(transfer['recycled_mass'],'recycled_volume_km3')]:
            if not math.isclose(actual,audit['material'][key]*DENSITY*1e9,rel_tol=2e-13,abs_tol=1.):raise ValueError('Native/parcel mass mismatch')
        mass_error=math.fsum(mass)-M0;transport_H=math.fsum(energy)-H0
        if abs(mass_error)>3e-14*M0 or abs(transport_H)>3e-14*max(abs(H0),1):raise ValueError('Material/enthalpy transport imbalance')
        if not math.isclose(mass[-2],audit['interior_reference_mass_kg'],rel_tol=3e-14):raise ValueError('Independent interior ledger mismatch')
        for name,new_value in [('fracture_damage',0.),('material_strength',1.)]:
            previous=np.array(self.record[name]);out=np.full(len(source_index),new_value);good=source_index>=0;out[good]=previous[source_index[good]];self.record[name]=out.tolist()
        bins=self.record['material_parcels']['nodes'];areas=self.mesh.physical_cell_areas_km2(self.record['radius_km'])*1e6
        thickness=new.state.oceanic_volume_km3/(areas/1e6)*1000;edges=[]
        for f,row in enumerate(bins):
            h=max(thickness[f],1.);G=p.conductivity_w_m_k*areas[f]
            edges.extend([(len(mass)-2,row[-1],G/(p.mantle_boundary_layer_km*1000)),(row[-1],row[0],G/(h/2)),(row[0],len(mass)-1,G/(h/4))])
        cp=np.full(len(mass),1200.);cp[-1]=4200.;latent=np.full(len(mass),4e5);latent[-1]=0
        dt=audit['dt_myr']*SECONDS_PER_MYR
        out=solve(mass,energy/mass,cp,latent,np.full(len(mass),1400.),np.full(len(mass),2000.),np.zeros(len(mass)),edges,dt,fixed_temperature={len(mass)-1:p.ocean_temperature_k})
        E=mass*out['specific_h'];export=math.fsum(float(q) for (i,j,_),q in zip(edges,out['exchange_j']) if j==len(mass)-1)
        residual=math.fsum([*E,*(-energy),export]);allowance=64*math.fsum(abs(np.spacing(x)) for x in np.r_[E,energy])+1e-9*abs(export)
        if abs(residual)>allowance:raise ValueError('Open thermal balance')
        if mass[-1]!=self.record['water_mass_kg']:raise ValueError('Water mass changed')
        if not np.allclose([sum(mass[row]) for row in bins],new.state.oceanic_volume_km3*DENSITY*1e9,rtol=2e-13):raise ValueError('Parcel carrier mass mismatch')
        self.record['thermal_inventory']=dict(mass_kg=mass.tolist(),energy_j=E.tolist());self.record['cumulative_exported_heat_j']+=export
        sync_thermal(self.record['material_parcels'],mass,E)
        new.thermal.time_myr=new.state.time_myr;new.thermal.system_age_myr+=audit['dt_myr'];new.thermal.mantle_temperature_k=float(out['temperature_k'][-2])
        new.mantle_flow,_=advance_mantle_flow(self.mesh,new.mantle_flow,audit['dt_myr'],1.,MantleFlowParameters())
        boundaries=classify_boundaries(self.mesh,new.system,self.record['radius_km'],5.,1.)
        system,topology,events=new.manager.update(self.mesh,new.state,new.system,boundaries,self.record['radius_km'],audit['dt_myr'])
        if topology.topology_changed:self.install_system(new,system)
        else:new.system=system
        new.topology_rows.append(asdict(topology))
        for event in events:
            row=asdict(event);self.record['events'].append(row);new.events.append(deepcopy(row))
        new.system=canonicalize_plate_seeds(self.mesh,new.system)
        boundaries=classify_boundaries(self.mesh,new.system,self.record['radius_km'],5.,1.)
        new.topo,relief,_=advance_topography(self.mesh,new.state,boundaries,old.topo,audit['dt_myr'],self.record['radius_km'],TopographyParameters(),previous_lithosphere=old.state,transport_source_index=source_index)
        new.hydrosphere,water=advance_hydrosphere(self.mesh,new.state,new.topo,old.hydrosphere,self.record['radius_km'],HydrosphereParameters(),TopographyParameters())
        if new.hydrosphere.water_volume_km3!=self.record['water_volume_km3'] or abs(water.relative_volume_error)>1e-6:raise ValueError('Water inventory/sea level balance')
        if np.any(new.state.continental_fraction) or np.any(new.state.continental_volume_km3):raise ValueError('Unexpected continental material')
        audit.update(coupled_enthalpy_closed=True,thermal=dict(transport_H_residual_j=transport_H,mass_residual_kg=mass_error,global_residual_j=residual,allowance_j=allowance,exported_heat_j=export,
            mantle_temperature_k=float(out['temperature_k'][-2]),ocean_temperature_k=float(out['temperature_k'][-1]),silicate_melt_mass_kg=float(mass[:-1]@out['melt_fraction'][:-1]),
            silicate_solid_mass_kg=float(mass[:-1]@(1-out['melt_fraction'][:-1])),created_H_j=transfer['created_H'],recycled_H_j=transfer['recycled_H']),
            water=asdict(water),relief=asdict(relief),plate_count=len(new.system.plates),max_geological_damage=max(self.record['fracture_damage']))
