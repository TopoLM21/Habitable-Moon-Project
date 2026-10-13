"""Active source enthalpy reservoirs attached to the geological raster.

Single undifferentiated silicate EOS; native noncontinental material carrier.
Inherited conductances form a FIXED Eulerian reference graph (not updated
moving-boundary conduction). Thermal parcels move; the graph does not.
No latent heat duplication, artificial kinetic heating, or chemical invention.
"""
from copy import deepcopy
from dataclasses import asdict, replace
import math
import numpy as np

from .genesis import GenesisParameters, SECONDS_PER_MYR, vapor_column_kg_m2
from .genesis_geological_handoff import GeologicalContinuation, DENSITY, kinematics, sha
from .genesis_implicit_enthalpy import solve, invert, phase
from .genesis_mature import build_experimental_genesis_mature_import
from .genesis_transfer_bridge import encoded
from .genesis_thermal_ports import apply_ports, water_temperature
from .genesis_tides import TidalParameters, TidalOrbitState, mean_motion_rad_s, validate_tidal_orbit
from .mesh import build_icosphere
from .plates import Plate, PlateSystem
from .thermal import ThermalState
from .dynamics import DynamicsParameters
from .tides import M_JUPITER


def topology_mesh(model):
    count = model.nf; level = 0
    while count > 20 and count % 4 == 0:
        level += 1; count //= 4
    mesh = build_icosphere(level)
    if mesh.faces.shape != model.mesh.faces.shape or not np.array_equal(mesh.faces,model.mesh.faces):
        raise ValueError('Face-ID mapping requires matching canonical connectivity')
    if mesh.neighbors != model.mesh.neighbors:
        raise ValueError('Face-ID mapping requires matching neighbor graph')
    return mesh, level


def eligibility(model, *, experimental_nodal_retirement=False, partition_owner=None, experimental_parcel_mapping=False):
    """Read-only readiness inspection; does not demand unknown chemical fractions."""
    c,s = model.config,model.state; reasons=[]
    if c.get('mechanical_mode')=='shared_nodal_moments' and not experimental_nodal_retirement:
        reasons.append('Shared nodal deformation and momentum require an explicit handoff map')
    owner=s['owner'] if partition_owner is None else partition_owner
    if len(np.unique(owner)) < 2: reasons.append('No multiple domains from the selector yet')
    if not np.allclose(s['rotation'],np.eye(3)[None],rtol=0,atol=1e-14):
        reasons.append('Finite domain rotations need a conservative spatial remap')
    if c.get('edge_strips'): reasons.append('Independent edge geometry needs a spatial remap')
    parcel_nodes=None
    if s.get('born_material') and not experimental_parcel_mapping: reasons.append('Variable-size newborn solid profiles need a parcel mapping')
    if experimental_parcel_mapping:
        try:
            from .experimental_geological_parcels import layout
            parcel_nodes=layout(model)
        except (ValueError,KeyError) as e:reasons.append(str(e))
    try:
        mesh,_ = topology_mesh(model)
        displacement = float(np.max(np.linalg.norm(mesh.centroids-model.mesh.centroids,axis=1))*c['radius_km'])
    except ValueError as e: reasons.append(str(e)); displacement=None
    nodes=np.asarray(c['solid_nodes']); silicate=np.arange(len(c['mass'])-1)
    for field in ['cp','latent','offset']:
        values=np.asarray(c[field])[silicate]
        if not np.all(values==values[0]): reasons.append('Mixed silicate EOS requires species-resolved mixing: '+field)
    if not np.allclose(np.asarray(c['mass'])[nodes].sum(axis=1),c['face_mass_kg'],rtol=1e-12,atol=0):
        reasons.append('Mechanical and mapped solid masses differ')
    if len(np.unique(nodes)) != nodes.size or np.any(nodes >= len(c['mass'])-2):
        reasons.append('Solid thermal node ownership is not disjoint from reservoirs')
    required = ['thermal_parameters','tidal_parameters']
    for name in required:
        if name not in c: reasons.append('Missing inherited '+name)
    if 'orbit' not in s: reasons.append('Missing inherited orbital state')
    if not any('Missing inherited' in x for x in reasons):
        p=GenesisParameters(**c['thermal_parameters']); tides=TidalParameters(**c['tidal_parameters'])
        try:
            p.validate()
            for field,expected in [('cp',p.silicate_heat_capacity_j_kg_k),('latent',p.silicate_latent_heat_j_kg),('offset',0.)]:
                if not np.all(np.asarray(c[field])[:-1]==expected):reasons.append('Source EOS differs from inherited Genesis '+field)
            if c['cp'][-1]!=p.water_heat_capacity_j_kg_k or c['latent'][-1]!=0. or c['offset'][-1]!=p.water_latent_heat_j_kg:
                reasons.append('Water EOS does not match inherited Genesis reservoir')
            validate_tidal_orbit(TidalOrbitState(**s['orbit']),tides)
            if s['orbit']['time_myr']!=s['time_myr']: reasons.append('Source and orbital clocks differ')
            if p.surface_layer_density_kg_m3 != DENSITY:
                reasons.append('Native reference-density budget must be generalized to source density')
            if not math.isclose(sum(c['mass'][:-1]),p.silicate_column_kg_m2*p.area_m2,rel_tol=3e-12):
                reasons.append('Source silicate mass does not match thermal ports')
            if not math.isclose(c['mass'][-1],p.water_column_kg_m2*p.area_m2,rel_tol=3e-12):
                reasons.append('Source water mass does not match thermal ports')
            if water_temperature(s['energy'][-1]/c['mass'][-1],p) < 273.16:
                reasons.append('Existing Genesis has no ice EOS')
        except ValueError as e: reasons.append(str(e))
    return dict(eligible=not reasons, reasons=reasons, domains=len(np.unique(owner)),
        time_myr=s['time_myr'], composition='undifferentiated silicate; elemental fractions unknown',
        continental_material_inferred=False, chemistry_required_for_this_mode=False,
        optional_future_fields=['elemental fractions','differentiation/depletion laws','continental generation'],
        canonical_face_relocation_max_km=displacement,
        geometry_contract='same face IDs/adjacency; preserve extensive mass/H, replace footprint geometry explicitly',
        source_thermal_node_count=len(c['mass']), mapped_solid_node_count=int(nodes.size if parcel_nodes is None else sum(map(len,parcel_nodes))),
        interior_nodes_retained=len(c['mass'])-(nodes.size if parcel_nodes is None else sum(map(len,parcel_nodes)))-1,
        new_physical_parameters_required=[],
        mantle_forcing='project existing face torque through existing face drag; not recovered 3-D fluid velocity')


def total(values):
    return math.fsum(float(x) for x in values)


class ThermalGeologicalContinuation(GeologicalContinuation):
    FORMAT = 'geological-source-enthalpy-2'

    @classmethod
    def from_source(cls, model, mantle_omega=None, *, experimental_nodal_retirement=False, fracture_edges=None, experimental_parcel_mapping=False):
        partition_owner=None;nodal_projection=None
        if experimental_nodal_retirement:
            from .experimental_nodal_handoff import prepare_nodal_retirement
            nodal_projection=prepare_nodal_retirement(model,fracture_edges)
            partition_owner=nodal_projection['owner']
        readiness=eligibility(model,experimental_nodal_retirement=experimental_nodal_retirement,partition_owner=partition_owner,experimental_parcel_mapping=experimental_parcel_mapping)
        if not readiness['eligible']: raise ValueError('; '.join(readiness['reasons']))
        c,s=model.config,model.state; mesh,level=topology_mesh(model)
        owner=np.array(s['owner'] if partition_owner is None else partition_owner,dtype=np.int32); plates=[]
        rotations=np.array(s['omega'] if nodal_projection is None else nodal_projection['omega_rad_s'])
        for i,w in enumerate(rotations*SECONDS_PER_MYR):
            speed=float(np.linalg.norm(w)); axis=w/speed if speed else np.array([0.,0.,1.])
            plates.append(Plate(i,int(np.flatnonzero(owner==i)[0]),axis,speed))
        mass=np.array(c['mass']); E=np.array(s['energy']); nodes=np.array(c['solid_nodes'])
        face_mass=mass[nodes].sum(axis=1); face_H=E[nodes].sum(axis=1); T=model.temperature(E)
        parcel_state=None
        if experimental_parcel_mapping:
            from .experimental_geological_parcels import layout,initialize
            parcel_nodes=layout(model);parcel_state=initialize(model,parcel_nodes)
            face_mass=np.array([total(mass[row]) for row in parcel_nodes]);face_H=np.array([total(E[row]) for row in parcel_nodes])
        p=GenesisParameters(**c['thermal_parameters']); tides=TidalParameters(**c['tidal_parameters'])
        T[-1]=water_temperature(E[-1]/mass[-1],p)
        forcing='explicit supplied scenario field'
        if mantle_omega is None:
            mantle_omega=np.linalg.solve(np.asarray(c['face_drag']),np.asarray(c['face_torque'])[...,None])[...,0]*SECONDS_PER_MYR
            forcing='per-face existing drag^-1 torque; effective zero-velocity basal equilibrium target'
        areas=mesh.physical_cell_areas_km2(c['radius_km'])
        thickness=face_mass/(DENSITY*areas*1e9)
        # Compatibility diagnostics, not a second mantle energy integrator.
        reference_flux=total(abs(g*(T[i]-T[j])) for i,j,g in c['edges'])/p.area_m2
        if reference_flux <= 0: raise ValueError('No nonzero source thermal flux for native diagnostic reference')
        thermal=ThermalState(s['time_myr'],p.system_age_at_start_myr+s['time_myr'],float(T[-2]),
            reference_flux,1.,float(np.average(thickness,weights=areas)))
        water=(p.water_column_kg_m2-vapor_column_kg_m2(T[-1],p))*p.area_m2/p.water_density_kg_m3/1e9
        bundle=build_experimental_genesis_mature_import(mesh,radius_km=c['radius_km'],
            system=PlateSystem(owner,tuple(plates)),crust_age_myr=np.full(model.nf,s['time_myr']),
            crust_thickness_km=thickness,tidal_damage=np.array(s['damage']),
            mantle_lithosphere_thickness_km=np.zeros(model.nf),mantle_lithosphere_density_anomaly_kg_m3=np.zeros(model.nf),
            thermal=thermal,elevation_m=np.zeros(model.nf),water_volume_km3=water,
            mantle_cell_omega_rad_per_myr=mantle_omega,source_mass_kg=face_mass,source_enthalpy_j=face_H,
            next_plume_birth_time_myr=s['time_myr']+1e6,
            source_metadata={'composition':'undifferentiated silicate; noncontinental native carrier, not chemical basalt proof',
                             'geometry':readiness['geometry_contract']})
        source=dict(config=deepcopy(c),state=deepcopy(s)); I=model.material_inertia(); w=rotations[owner]
        L=np.einsum('fij,fj->i',I,w); K=float(.5*np.einsum('fi,fij,fj->',w,I,w))
        P=None
        if 'face_linear_momentum_map' in c:
            P=np.einsum('fij,fj->i',np.asarray(c['face_linear_momentum_map']),w).tolist()
        proxy=kinematics(bundle.checkpoint,mesh,c['radius_km'])
        elastic=float(np.sum(model.constitutive(T,s['damage'],np.asarray(s['memory']),T,0)[4]))
        if nodal_projection is not None:
            L=np.array(nodal_projection['angular_momentum_kg_m2_s']);P=nodal_projection['linear_momentum_kg_m_s']
            K=nodal_projection['kinetic_j'];elastic=nodal_projection['elastic_j']
        record=dict(format=cls.FORMAT,subdivisions=level,radius_km=c['radius_km'],
            dynamics_parameters=asdict(DynamicsParameters()),gravity_m_s2=p.surface_gravity_m_s2,
            period_hours=2*np.pi/mean_motion_rad_s(tides)/3600,eccentricity=s['orbit']['eccentricity'],
            primary_mass_jupiter=tides.primary_mass_kg/M_JUPITER,love_h2=tides.love_h2,
            interior_reference_mass_kg=float(mass[-2]),inactive_surface_mass_kg=0.,
            source_sha256=sha(encoded(source)),steps=[],eligibility=readiness,mantle_forcing=forcing,
            thermal_inventory=dict(time_myr=s['time_myr'],mass_kg=mass.tolist(),energy_j=E.tolist(),orbit=deepcopy(s['orbit'])),
            thermal_geometry_approximation='fixed inherited Eulerian conductances; no evolving thickness reconstruction',
            native_thermal_diagnostics='reference flux=sum absolute source graph exchanges/area; activity=1; initial thickness=mass-weighted carrier depth; these do not add heat',
            mapping=dict(mass={'active_solid_kg':total(face_mass),'active_deep_kg':float(mass[-2]),
                              'active_other_interior_kg':total(mass)-total(face_mass)-mass[-2]-mass[-1],
                              'active_water_kg':float(mass[-1]),'total_kg':total(mass),'archive_double_counted':False},
                composition={'active':'undifferentiated silicate, noncontinental carrier',
                             'chemical_fractions':None,'continental_fraction':0.,'differentiation_active':False},
                H={'active_total_j':total(E),'mapping_change_j':0.,'model':'inherited EOS/ports/fixed reference graph',
                   'source_archive_is_not_active_inventory':True},
                momentum={'source_angular_kg_m2_s':L.tolist(),'source_linear_kg_m_s':P,
                    'mapped_shell_proxy':proxy,'mapping_angular_change_kg_m2_s':(np.array(proxy['angular_momentum_kg_m2_s'])-L).tolist(),
                    'mapping_linear_change_kg_m_s':None if P is None else (np.array(proxy['linear_momentum_kg_m_s'])-P).tolist(),
                    'approximation':'inertial constrained rotations -> effective quasi-static velocities; no SI impulse closure'},
                energy={'source_kinetic_j':K,'mapped_kinetic_proxy_j':proxy['kinetic_energy_j'],
                    'signed_model_change_kinetic_j':proxy['kinetic_energy_j']-K,'retired_elastic_j':elastic,
                    'source_passive_reserve_j':s['passive_reserve_j'],
                    'prior_passive_kinetic_projection_j':c.get('passive_kinetic_projection_j',0.),
                    'mapping_heat_j':0.,'total_mechanical_energy_conserved':False},
                Maxwellmemory={'status':'exact frozen source; active detailed constitutive memory retired','heat_j':0.}),
            inactive_modules=['continental differentiation','topography','plumes','topology changes'],import_report=bundle.report)
        if parcel_state is not None:record['material_parcels']=parcel_state
        if nodal_projection is not None:
            record['experimental_nodal_retirement']=nodal_projection
            record['mapping']['energy']['source_edge_fracture_reserve_j']=float(s.get('experimental_edge_fracture_reserve_j',0.))
            record['mapping']['momentum']['approximation']='shared nodal SI momenta -> fitted quasi-static region rotations; signed change recorded, no impulse conservation claim'
            record['mapping']['energy']['signed_active_mechanical_model_change_j']=proxy['kinetic_energy_j']-K-elastic-float(s['passive_reserve_j'])-float(s.get('experimental_edge_fracture_reserve_j',0.))
            record['mapping']['Maxwellmemory']['status']='full nodal and moment source archived; active mechanics retired at approved geological model change'
        return cls(bundle.checkpoint,record,source)

    def step(self,dt_myr):
        trial=type(self)(deepcopy(self.checkpoint),self.record,self.source)
        audit=GeologicalContinuation.step(trial,dt_myr)
        self.checkpoint,self.record=trial.checkpoint,trial.record
        return audit

    def _after_material_step(self,old,new,source_index,audit):
        c=self.source['config']; inv=self.record['thermal_inventory']
        nodes=np.asarray(c['solid_nodes']); mass=np.array(inv['mass_kg']); E=np.array(inv['energy_j'])
        oldmass=mass.copy(); oldE=E.copy(); inherited=source_index>=0; newborn=~inherited
        parcels=self.record.get('material_parcels')
        if parcels is not None:
            from .experimental_geological_parcels import transport
            generated=new.state.oceanic_volume_km3*DENSITY*1e9
            transfer=transport(parcels,mass,E,source_index,generated,new.state.time_myr)
            recycled_mass=transfer['recycled_mass'];recycled_H=transfer['recycled_H']
            created_mass=transfer['created_mass'];created_H=transfer['created_H'];donor_h=transfer['donor_h']
            face_nodes=parcels['nodes'];nodes=np.array([i for row in face_nodes for i in row])
        else:
            sources=source_index[inherited]
            if len(np.unique(sources)) != len(sources): raise ValueError('Thermal parcel duplication')
            survived=np.zeros(self.mesh.cell_count,bool); survived[sources]=True
            loss_nodes=nodes[~survived].ravel()
            recycled_mass=total(mass[loss_nodes]); recycled_H=total(E[loss_nodes])
            generated_mass=new.state.oceanic_volume_km3[newborn]*DENSITY*1e9
            # Draw first from the old donor. Recycling returns its own H, never donor temperature.
            donor_h=float(E[-2]/mass[-2]); created_mass=total(generated_mass)
            if created_mass >= mass[-2]: raise ValueError('Deep donor exhausted before recycling')
            created_H=created_mass*donor_h
            mass[nodes[inherited]]=oldmass[nodes[sources]]; E[nodes[inherited]]=oldE[nodes[sources]]
            shares=oldmass[nodes[newborn]]/oldmass[nodes[newborn]].sum(axis=1)[:,None]
            mass[nodes[newborn]]=generated_mass[:,None]*shares
            E[nodes[newborn]]=mass[nodes[newborn]]*donor_h
            mass[-2]=oldmass[-2]-created_mass+recycled_mass
            E[-2]=oldE[-2]-created_H+recycled_H
        # Independently compare donor transfers to native geological counters.
        budget=audit['material']
        for actual,expected in [(created_mass,budget['generated_volume_km3']*DENSITY*1e9),
                                (recycled_mass,budget['recycled_volume_km3']*DENSITY*1e9)]:
            if not math.isclose(actual,expected,rel_tol=2e-14,abs_tol=1.):
                raise ValueError('Native/thermal material transaction mismatch')
        mapped_face_mass=np.array([total(mass[row]) for row in face_nodes]) if parcels is not None else mass[nodes].sum(axis=1)
        if not np.allclose(mapped_face_mass,new.state.oceanic_volume_km3*DENSITY*1e9,rtol=2e-14,atol=1.):
            raise ValueError('Native/thermal face masses differ')
        # Node-local transport roundoff cannot hide behind the huge interior reservoir.
        if parcels is None:
            local_residual=E[nodes[inherited]]-oldE[nodes[sources]]
            if np.any(local_residual): raise ValueError('Inherited parcel H changed during transport')
        transport_residual=math.fsum([*E,*(-oldE)])
        transport_allowance=16*total(abs(np.spacing(x)) for x in np.r_[E,oldE])
        if abs(transport_residual)>transport_allowance: raise ValueError('Transport enthalpy balance')
        mass_residual=math.fsum([*mass,*(-oldmass)])
        mass_allowance=16*total(abs(np.spacing(x)) for x in np.r_[mass,oldmass])
        if abs(mass_residual)>mass_allowance: raise ValueError('Global thermal mass balance')
        if not math.isclose(mass[-2],audit['interior_reference_mass_kg'],rel_tol=2e-14):
            raise ValueError('Native and thermal interior ledgers disagree')
        p=GenesisParameters(**c['thermal_parameters']); tides=TidalParameters(**c['tidal_parameters'])
        target=new.state.time_myr; dt=(target-old.state.time_myr)*SECONDS_PER_MYR
        sourced,orbit,ports=apply_ports(dict(time_myr=old.state.time_myr,mass_kg=mass.tolist(),energy_j=E.tolist()),
            p,tides,TidalOrbitState(**inv['orbit']),target,allow_condensation=True)
        Es=np.array(sourced['energy_j']); cp=np.asarray(c['cp']); latent=np.asarray(c['latent']); offset=np.asarray(c['offset'])
        thermal=solve(mass,Es/mass,cp,latent,np.full(len(E),c['solidus']),np.full(len(E),c['liquidus']),offset,c['edges'],dt,water_parameters=p)
        final=mass*thermal['specific_h']; T=thermal['temperature_k']
        # Sum each node's exchange terms together: sequential subtraction from
        # a huge deep reservoir loses small face exchanges before the audit.
        terms=[[float(final[i]),-float(Es[i])] for i in range(len(E))]
        flux_magnitude=np.zeros(len(E))
        for (i,j,_),q in zip(c['edges'],thermal['exchange_j']):
            terms[i].append(float(q)); terms[j].append(-float(q))
            flux_magnitude[i]+=abs(q); flux_magnitude[j]+=abs(q)
        local=np.array([math.fsum(x) for x in terms])
        allowance=mass*cp*thermal['tolerance_k']+16*(abs(np.spacing(final))+abs(np.spacing(Es))+abs(np.spacing(flux_magnitude)))
        if np.any(abs(local)>allowance):
            bad=int(np.argmax(abs(local)/allowance))
            raise ValueError(f'Local thermal energy residual: node={bad}, residual={local[bad]}, allowance={allowance[bad]}, solver={thermal["residual_k"][bad]}, T={T[bad]}')
        net_residual=math.fsum([*final,*(-oldE)]+[-x['energy_into_thermal_j'] for x in ports['ports']])
        global_allowance=total(allowance)+transport_allowance+ports['endpoint_roundoff_allowance_j']
        if abs(net_residual)>global_allowance: raise ValueError('Global active enthalpy residual')
        oldT=invert(oldE/oldmass,cp,latent,c['solidus'],c['liquidus'],offset)
        oldT[-1]=water_temperature(oldE[-1]/oldmass[-1],p)
        thermal_row=dict(time_myr=target,active_enthalpy_before_j=total(oldE),active_enthalpy_after_j=total(final),
            created_mass_kg=created_mass,recycled_mass_kg=recycled_mass,
            created_enthalpy_j=created_H,recycled_enthalpy_j=recycled_H,
            donor_specific_enthalpy_j_kg=donor_h,transport_residual_j=transport_residual,
            transport_allowance_j=transport_allowance,ports=ports,
            local_max_residual_j=float(np.max(abs(local))),max_local_residual_over_allowance=float(np.max(abs(local)/allowance)),
            global_residual_j=net_residual,global_allowance_j=global_allowance,
            deep_before_k=float(oldT[-2]),deep_after_k=float(T[-2]),surface_before_k=float(oldT[-1]),surface_after_k=float(T[-1]),
            max_temperature_change_k=float(np.max(abs(T-oldT))),
            max_carrier_melt_fraction=float(np.max(phase(T[nodes],c['solidus'],c['liquidus']))),
            total_mass_residual_kg=mass_residual,mass_roundoff_allowance_kg=mass_allowance,mechanical_heat_j=0.,
            inactive_memory_heat_j=0.,conduction_iterations=thermal['iterations'])
        if parcels is not None:
            from .experimental_geological_parcels import sync_thermal
            sync_thermal(parcels,mass,final)
            thermal_row['parcel_transport']=transfer
        self.record['thermal_inventory']=dict(time_myr=target,mass_kg=mass.tolist(),energy_j=final.tolist(),orbit=asdict(orbit))
        # Match native scalar clock/temperature to the active source reservoir, not a second EOS.
        new.thermal=replace(new.thermal,time_myr=target,system_age_myr=p.system_age_at_start_myr+target,mantle_temperature_k=float(T[-2]))
        new.thermal_rows.append(thermal_row)
        new.hydrosphere.time_myr=target
        new.hydrosphere.water_volume_km3=(p.water_column_kg_m2-vapor_column_kg_m2(T[-1],p))*p.area_m2/p.water_density_kg_m3/1e9
        self.record['eccentricity']=orbit.eccentricity
        self.record['period_hours']=2*np.pi/mean_motion_rad_s(replace(tides,semimajor_axis_km=orbit.semimajor_axis_km))/3600
        audit.update(thermal=thermal_row,coupled_enthalpy_closed=True,
                     mechanical_energy_closed=False,thermal_source='active inherited reservoirs/ports; no archived H substitute')
