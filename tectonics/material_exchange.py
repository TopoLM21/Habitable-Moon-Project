"""Experimental coarse mantle/crust inventories with sensible enthalpy.

One well-mixed mantle and one well-mixed crust temperature; basalt and
continental/sediment rock-equivalent masses have separate provenance pools.
No latent heat, chemical differentiation, mechanical work or paid fracture is
claimed. Geological production is a request, never an unlimited mass source.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import math

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq

from .lithosphere import continental_material_fields
from .thermal import (_geometry, convective_state, diagnose_thermal_state,
                      radiogenic_power_w, tidal_heating_power_w, SECONDS_PER_MYR)
from .thermal_budget import _add


@dataclass(frozen=True)
class MaterialExchangeParameters:
    continental_rock_density_kg_m3: float = 2800.
    basalt_density_kg_m3: float = 2900.
    temperature_atol_k: float = 1.
    temperature_rtol: float = 1e-4
    exchange_rtol: float = 1e-3
    exchange_atol_j_per_myr: float = 1e19
    initial_step_myr: float = .25
    max_step_myr: float = 4.
    max_trials: int = 10000

    def __post_init__(self):
        for key, val in asdict(self).items():
            if not math.isfinite(val) or val <= 0:
                raise ValueError(f'Invalid material exchange parameter {key}')
        if isinstance(self.max_trials, bool) or int(self.max_trials) != self.max_trials:
            raise ValueError('max_trials must be an integer')


def surface_volumes(state):
    if state.oceanic_volume_km3 is None or state.continental_volume_km3 is None:
        raise ValueError('Material inventory requires explicit basalt and continental volumes')
    sediment = 0. if state.sediment_volume_km3 is None else math.fsum(state.sediment_volume_km3)
    f = math.fsum(state.continental_volume_km3) + sediment
    o = math.fsum(state.oceanic_volume_km3)
    if min(f, o) < 0 or not math.isfinite(f+o):
        raise ValueError('Invalid surface inventory')
    return np.array([f, o])


def linear_thermal_exchange(temperature, capacities, conductances, powers, surface_k, dt_myr):
    """Exact frozen-coefficient two-box solution and its flux integrals.

    powers = [radiogenic mantle, radiogenic crust, orbital], W.
    conductances = [mantle/crust, crust/surface], W/K. Returned integrals are
    [radiogenic total, orbital, surface loss, mantle->crust], J; the last may
    be negative. This same signed exchange is used on both sides.
    """
    cm, cc = capacities; gm, gs = conductances
    rm, rc, tide = powers
    matrix = np.zeros((5, 5))
    matrix[0, :3] = [-gm/cm, gm/cm, (rm+tide)/cm]
    matrix[1, :3] = [gm/cc, -(gm+gs)/cc, rc/cc]
    matrix[:2, :3] *= SECONDS_PER_MYR
    matrix[3, 0] = matrix[4, 1] = 1.
    y0 = np.array([temperature[0]-surface_k, temperature[1]-surface_k, 1., 0., 0.])
    y = expm(matrix * dt_myr) @ y0
    q = np.array([(rm+rc)*dt_myr, tide*dt_myr, gs*y[4], gm*(y[3]-y[4])]) * SECONDS_PER_MYR
    return y[:2]+surface_k, q, np.array([rm, rc])*dt_myr*SECONDS_PER_MYR


class MaterialExchangeBudget:
    VERSION = 1

    @staticmethod
    def _conductances(tm, volume_m3, area_m2, radius_km, gravity, params):
        h = volume_m3 / area_m2
        k = params.thermal_conductivity_w_m_k
        layer = convective_state(tm, radius_km, gravity, params)[4]*1000.
        if h <= 0 or k <= 0 or layer < h:
            raise ValueError('Two-box thermal domain: crust thickness must fit the effective thermal layer')
        # The old total boundary-layer resistance already includes the crust.
        # Split it at the crust midpoint, rather than adding another full lid.
        return np.array([k*area_m2/(layer-h/2), 2*k*area_m2/h])

    @classmethod
    def initialize(cls, mesh, state, thermal, mass_earth, radius_km, gravity, params,
                   options=MaterialExchangeParameters(), ocean_reference_km=7.):
        areas = mesh.physical_cell_areas_km2(radius_km)
        frac, fvol = continental_material_fields(state, areas)
        if state.oceanic_volume_km3 is not None:
            raise ValueError('Fresh inventory initialization cannot overwrite material history')
        candidate = deepcopy(state)
        candidate.continental_volume_km3 = fvol.copy()
        candidate.continental_fraction = frac.copy()
        candidate.oceanic_volume_km3 = areas*(1-frac)*np.where(state.crust_type == 0, state.crust_thickness_km, ocean_reference_km)
        volumes = surface_volumes(candidate)
        masses = volumes*1e9*np.array([options.continental_rock_density_kg_m3, options.basalt_density_kg_m3])
        _, total, area, _ = _geometry(mass_earth, radius_km, params)
        mc = float(np.sum(masses)); mm = total-mc; cp = params.mantle_heat_capacity_j_kg_k
        if mm <= 0 or mc <= 0:
            raise ValueError('Crust cannot be carved from the finite exchangeable inventory')
        volume = float(np.sum(volumes))*1e9
        old_t = thermal.mantle_temperature_k; ts = params.surface_temperature_k
        energy = total*cp*old_t

        def crust_t(tm):
            gm, gs = cls._conductances(tm, volume, area, radius_km, gravity, params)
            return (gm*tm+gs*ts)/(gm+gs)

        upper = (total*old_t-mc*ts)/mm
        tm = brentq(lambda t: mm*cp*t+mc*cp*crust_t(t)-energy, old_t, upper, xtol=1e-10)
        tc = crust_t(tm)
        energy_c = mc*cp*tc
        energy_m = [energy, 0.]; _add(energy_m, -energy_c)
        obj = cls.__new__(cls); obj.options = options
        obj.data = {
            'version': cls.VERSION, 'options': asdict(options), 'time_myr': thermal.time_myr,
            'origin_time_myr': thermal.time_myr, 'origin_system_age_myr': thermal.system_age_myr,
            'total_mass_kg': total, 'initial_energy_j': energy,
            'origin_mass_kg': [mm, float(masses[0]), float(masses[1])],
            'origin_hi': np.diag([mm,*masses]).tolist(), 'origin_lo': np.zeros((3,3)).tolist(),
            'energy_m_j': energy_m, 'energy_c_j': [energy_c, 0.],
            'surface_volumes_km3': volumes.tolist(), 'area_m2': area,
            'radiogenic_input_j': [0.,0.], 'orbital_input_j': [0.,0.],
            'surface_output_j': [0.,0.], 'basal_exchange_j': [0.,0.],
            'advected_out_j': [0.,0.], 'advected_return_j': [0.,0.],
            'created_kg': [0.,0.], 'recycled_kg': [0.,0.],
            'accepted_steps': 0, 'rejected_steps': 0, 'next_step_myr': options.initial_step_myr,
            'physical_signature': None, 'history': [],
            'initialization': 'new conservative partition of legacy exchangeable heat; not recovered crust history',
        }
        return obj, candidate, replace(thermal, mantle_temperature_k=obj.temperatures(params)[0])

    def checkpoint(self):
        return deepcopy(self.data)

    @classmethod
    def restore(cls, data, options):
        required = {'energy_m_j','energy_c_j','origin_hi','origin_lo','next_step_myr','history',
                    'surface_volumes_km3','physical_signature','created_kg','recycled_kg'}
        if data.get('version') != cls.VERSION or required-data.keys():
            raise ValueError('Incomplete/unsupported material exchange checkpoint')
        if data['options'] != asdict(options):
            raise ValueError('Material exchange parameters changed at restart')
        obj=cls.__new__(cls); obj.options=options; obj.data=deepcopy(data)
        for name,shape in (('origin_hi',(3,3)),('origin_lo',(3,3)),('energy_m_j',(2,)),
                           ('energy_c_j',(2,)),('surface_volumes_km3',(2,))):
            value=np.asarray(data[name],dtype=float)
            if value.shape!=shape or np.any(~np.isfinite(value)):
                raise ValueError(f'Invalid saved material inventory {name}')
        masses=obj.pool_masses()
        origins=np.asarray(data['origin_hi'])+np.asarray(data['origin_lo'])
        if np.any(origins<0) or masses[0]<=0 or sum(masses[1:])<=0 or min(data['surface_volumes_km3'])<0:
            raise ValueError('Invalid saved material masses')
        expected=np.asarray(data['surface_volumes_km3'])*1e9*np.array([
            options.continental_rock_density_kg_m3,options.basalt_density_kg_m3])
        if any(abs(a-b)>64*math.ulp(max(a,b,1.)) for a,b in zip(expected,masses[1:])):
            raise ValueError('Saved material geometry disagrees with masses')
        if any(not math.isfinite(data[key]) or data[key]<=0 for key in ('next_step_myr','total_mass_kg')):
            raise ValueError('Invalid saved material controller')
        if any(math.fsum(data[key])<=0 for key in ('energy_m_j','energy_c_j')):
            raise ValueError('Invalid saved material energy')
        return obj

    def pool_masses(self):
        return np.array([math.fsum([*hi,*lo]) for hi,lo in zip(self.data['origin_hi'],self.data['origin_lo'])])

    def temperatures(self, params):
        m=self.pool_masses(); cp=params.mantle_heat_capacity_j_kg_k
        return np.array([math.fsum(self.data['energy_m_j'])/(cp*m[0]),
                         math.fsum(self.data['energy_c_j'])/(cp*math.fsum(m[1:]))])

    def exchange(self, born_km3, returned_km3, target_volumes_km3, params):
        """Atomic endpoint process: creation first, then recycling of well-mixed pools.

        Separate basalt/felsic origin inventories survive label/topology changes.
        They are pool-averaged provenance, not spatial chemical differentiation.
        """
        density=np.array([self.options.continental_rock_density_kg_m3,self.options.basalt_density_kg_m3])*1e9
        born=np.asarray(born_km3,dtype=float)*density
        returned=np.asarray(returned_km3,dtype=float)*density
        target=np.asarray(target_volumes_km3,dtype=float)*density
        if any(a.shape!=(2,) or np.any(~np.isfinite(a)) or np.any(a<0) for a in (born,returned,target)):
            raise ValueError('Invalid material transfer request')
        work=deepcopy(self)
        d=work.data
        if math.fsum(born) >= work.pool_masses()[0]:
            raise ValueError('Finite mantle inventory exhausted; no transfer committed')
        available=work.pool_masses()[1:]+born
        # A geometry operator can explicitly exhaust a pool. Its independently
        # summed volume differs from the compensated mass inventory by a few
        # ulps. Drain the actual stored components; never leave/drop a phantom
        # remainder or relax the tolerance relative to that tiny remainder.
        exhausted=target==0.
        exhaustion_error=np.abs(returned-available)
        exhaustion_bound=np.array([64*math.ulp(max(float(a),float(b),1.)) for a,b in zip(available,returned)])
        if np.any(exhausted & (exhaustion_error>exhaustion_bound)):
            raise ValueError('Empty target disagrees with requested complete material transfer')
        if np.any((returned > available) & ~exhausted):
            raise ValueError('Recycling exceeds available crust; no transfer committed')
        if math.fsum(target)<=0 or math.fsum([*work.pool_masses()[1:],*born,*(-returned)])<=0:
            raise ValueError('Two-box domain requires a nonempty crust; no transfer committed')
        cp=params.mantle_heat_capacity_j_kg_k

        def transfer(source, destination, amount, complete=False):
            if amount == 0 and not complete:return 0.
            pools=work.pool_masses()
            ts=work.temperatures(params)[0 if source==0 else 1]
            if complete:
                amount=float(pools[source])
                for j in range(3):
                    pair=[d['origin_hi'][destination][j],d['origin_lo'][destination][j]]
                    for key in ('origin_hi','origin_lo'):_add(pair,d[key][source][j])
                    d['origin_hi'][destination][j],d['origin_lo'][destination][j]=pair
                    d['origin_hi'][source][j]=d['origin_lo'][source][j]=0.
            else:
                origins=np.array([math.fsum([d['origin_hi'][source][j],d['origin_lo'][source][j]]) for j in range(3)])
                moved=origins*(amount/pools[source])
                # Arithmetic remainder belongs to the largest provenance component,
                # not to an unrecorded source of mass.
                j=int(np.argmax(origins)); moved[j]=math.fsum([amount,*(-np.delete(moved,j))])
                for k,part in enumerate(moved):
                    for row,sign in ((source,-1.),(destination,1.)):
                        pair=[d['origin_hi'][row][k],d['origin_lo'][row][k]]
                        _add(pair,sign*float(part));d['origin_hi'][row][k],d['origin_lo'][row][k]=pair
            q=float(amount)*cp*ts
            if source==0:
                _add(d['energy_m_j'],-q);_add(d['energy_c_j'],q);_add(d['advected_out_j'],q)
            else:
                _add(d['energy_c_j'],-q);_add(d['energy_m_j'],q);_add(d['advected_return_j'],q)
            return amount

        for i in range(2): transfer(0,i+1,float(born[i]))
        for i in range(2): returned[i]=transfer(i+1,0,float(returned[i]),complete=bool(exhausted[i]))
        actual=work.pool_masses()[1:]
        if any(abs(a-b)>64*math.ulp(max(a,b,1.)) for a,b in zip(actual,target)):
            raise ValueError(f'Material requests disagree with geometric inventories: {actual-target} kg')
        for value in born:_add(d['created_kg'],float(value))
        for value in returned:_add(d['recycled_kg'],float(value))
        d['surface_volumes_km3']=np.asarray(target_volumes_km3,dtype=float).tolist()
        self.data=d

    def advance(self, state, dt_myr, mass_earth, radius_km, gravity, period_hours, primary_mass_jupiter, eccentricity, params):
        if dt_myr<=0 or not math.isfinite(dt_myr) or state.time_myr!=self.data['time_myr']:
            raise ValueError('Invalid two-box time increment')
        signature={'params':asdict(params),'mass_earth':mass_earth,'radius_km':radius_km,'gravity':gravity,
                   'period_hours':period_hours,'primary_mass_jupiter':primary_mass_jupiter,
                   'eccentricity_times':np.asarray(eccentricity.times_myr).tolist(),
                   'eccentricity_values':np.asarray(eccentricity.eccentricity).tolist()}
        if self.data['physical_signature'] is not None and self.data['physical_signature']!=signature:
            raise ValueError('Two-box physical configuration changed')
        work=deepcopy(self);d=work.data;d['physical_signature']=signature
        mass=work.pool_masses();cm,cc=params.mantle_heat_capacity_j_kg_k*np.array([mass[0],math.fsum(mass[1:])])
        volume=math.fsum(d['surface_volumes_km3'])*1e9
        time=state.time_myr;target=time+dt_myr
        if not math.isfinite(target) or target<=time: raise ValueError('Unrepresentable target time')
        temperature=work.temperatures(params)
        if abs(temperature[0]-state.mantle_temperature_k)>4*math.ulp(temperature[0]):
            raise ValueError('Thermal state disagrees with material enthalpy')

        def coefficients(t,temp):
            if np.any(~np.isfinite(temp)) or temp[0]<params.surface_temperature_k+1 or temp[1]<params.surface_temperature_k:
                raise ValueError('Two-box temperature outside thermal domain')
            conductance=self._conductances(temp[0],volume,d['area_m2'],radius_km,gravity,params)
            age=d['origin_system_age_myr']+t-d['origin_time_myr']
            powers=[radiogenic_power_w(age,mass[0],params),radiogenic_power_w(age,math.fsum(mass[1:]),params),
                    tidal_heating_power_w(eccentricity.at(t),radius_km,period_hours,primary_mass_jupiter,params)]
            return conductance,powers

        def step(t,temp,h):
            g,p=coefficients(t,temp)
            middle=linear_thermal_exchange(temp,(cm,cc),g,p,params.surface_temperature_k,h/2)[0]
            g,p=coefficients(t+h/2,middle)
            return linear_thermal_exchange(temp,(cm,cc),g,p,params.surface_temperature_k,h)

        coefficients(time,temperature)
        trials=0
        while time<target:
            trials+=1
            if trials>self.options.max_trials:raise RuntimeError('Two-box thermal trial budget exhausted')
            future=np.asarray(eccentricity.times_myr);future=future[(future>time)&(future<target)]
            endpoint=float(future[0]) if len(future) else target
            h=min(d['next_step_myr'],self.options.max_step_myr,endpoint-time)
            if time+h<=time:raise RuntimeError('Two-box time step underflow')
            try:
                coarse,cq,_=step(time,temperature,h)
                middle,q1,r1=step(time,temperature,h/2)
                fine,q2,r2=step(time+h/2,middle,h/2)
                q=q1+q2;rad=r1+r2
                tscale=self.options.temperature_atol_k+self.options.temperature_rtol*np.abs(fine)
                qscale=self.options.exchange_atol_j_per_myr*h+self.options.exchange_rtol*np.abs(q)
                error=max(float(np.max(abs(fine-coarse)/(3*tscale))),float(np.max(abs(q-cq)/(3*qscale))))
                coefficients(time+h,fine)
            except ValueError:
                error=math.inf
            if not math.isfinite(error) or error>1:
                d['rejected_steps']+=1
                d['next_step_myr']=h*(.5 if not math.isfinite(error) else max(.1,.8*error**(-1/3)))
                continue
            for value in (rad[0],q[1],-q[3]): _add(d['energy_m_j'],float(value))
            for value in (rad[1],q[3],-q[2]): _add(d['energy_c_j'],float(value))
            for value in rad: _add(d['radiogenic_input_j'],float(value))
            for key,value in zip(('orbital_input_j','surface_output_j','basal_exchange_j'),q[1:]):
                _add(d[key],float(value))
            time=endpoint if h==endpoint-time else time+h
            temperature=work.temperatures(params)
            d['accepted_steps']+=1
            d['next_step_myr']=min(self.options.max_step_myr,h*min(2.,.9*max(error,1e-12)**(-1/3)))
        d['time_myr']=target
        new=replace(state,time_myr=target,system_age_myr=d['origin_system_age_myr']+target-d['origin_time_myr'],
                    mantle_temperature_k=float(temperature[0]))
        self.data=d
        return self.thermal_state(new,mass_earth,radius_km,gravity,period_hours,primary_mass_jupiter,eccentricity,params)

    def thermal_state(self,state,mass_earth,radius_km,gravity,period_hours,primary_mass_jupiter,eccentricity,params):
        state=replace(state,mantle_temperature_k=float(self.temperatures(params)[0]))
        diag=diagnose_thermal_state(state,mass_earth,radius_km,gravity,period_hours,primary_mass_jupiter,eccentricity.at(state.time_myr),params)
        # Existing activity/thermal-thickness diagnostics use the same mantle T;
        # actual heat fluxes are the separately integrated two-box ledger.
        state.tectonic_activity_factor=diag.tectonic_activity_factor
        state.thermal_lithosphere_thickness_km=diag.thermal_lithosphere_thickness_km
        temp=self.temperatures(params)
        _,gs=self._conductances(temp[0],math.fsum(self.data['surface_volumes_km3'])*1e9,
                                self.data['area_m2'],radius_km,gravity,params)
        surface=gs*(temp[1]-params.surface_temperature_k)
        diag.convective_heat_flux_w_m2=surface/self.data['area_m2']
        diag.convective_power_tw=surface/1e12
        diag.net_heat_flux_w_m2=diag.radiogenic_heat_flux_w_m2+diag.tidal_heat_flux_w_m2-diag.convective_heat_flux_w_m2
        return state,diag

    def diagnostics(self,params):
        d=self.data;m=self.pool_masses();temp=self.temperatures(params)
        residual=math.fsum([*d['energy_m_j'],*d['energy_c_j'],-d['initial_energy_j'],
                           *(-x for x in d['radiogenic_input_j']),*(-x for x in d['orbital_input_j']),*d['surface_output_j']])
        origin_error=[math.fsum([*(d['origin_hi'][i][j] for i in range(3)),
                                 *(d['origin_lo'][i][j] for i in range(3)),-d['origin_mass_kg'][j]]) for j in range(3)]
        return {'time_myr':d['time_myr'],'mantle_temperature_k':float(temp[0]),'crust_temperature_k':float(temp[1]),
                'pool_masses_kg':m.tolist(),'mass_residual_kg':math.fsum([*m,-d['total_mass_kg']]),
                'origin_mass_residual_kg':origin_error,'energy_residual_j':residual,
                'created_kg':math.fsum(d['created_kg']),'recycled_kg':math.fsum(d['recycled_kg']),
                'advected_out_j':math.fsum(d['advected_out_j']),'advected_return_j':math.fsum(d['advected_return_j']),
                'surface_output_j':math.fsum(d['surface_output_j']),
                'radiogenic_input_j':math.fsum(d['radiogenic_input_j']),'orbital_input_j':math.fsum(d['orbital_input_j']),
                'accepted_steps':d['accepted_steps'],'rejected_steps':d['rejected_steps'],
                'latent_chemical_mechanical_energy_included':False}
