"""Explicit thermal/material coupling for the coarse deforming surface.

Sensible-heat physics is inherited, not upgraded to a melting model. Mechanical
heat has one owner. Prescribed material controls cannot turn donor heat into
unfunded emplacement work. No automatic production or fracture is supplied.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

from .conforming_surface import column_potential
from .regional_deformation import DeformingSurface
from .material_exchange import MaterialExchangeBudget, MaterialExchangeParameters
from .thermal import ThermalParameters, ThermalState
from .thermal_budget import _add
from .tides import EccentricityHistory


@dataclass(frozen=True)
class CoupledParameters:
    mass_earth: float = .5
    period_hours: float = 47.
    primary_mass_jupiter: float = 5.
    step_myr: float = 1.

    def __post_init__(self):
        if any(not math.isfinite(x) or x <= 0 for x in asdict(self).values()):
            raise ValueError('Invalid coupled configuration')


class DeformationExchange:
    VERSION = 1

    def __init__(self, surface, material, thermal, thermal_params, eccentricity,
                 options=CoupledParameters()):
        if surface.data['time_myr'] != 0 or thermal.time_myr != 0 or material.data['time_myr'] != 0:
            raise ValueError('Coupling requires explicit zero-time initialization')
        if math.fsum(surface.data['heat_generated_j']) != 0 or math.fsum(surface.data['external_work_j']) != 0:
            raise ValueError('Cannot reinterpret prior mechanical exchanges')
        self.surface=deepcopy(surface);self.material=deepcopy(material)
        self.thermal=deepcopy(thermal);self.thermal_params=thermal_params
        self.eccentricity=deepcopy(eccentricity);self.options=options
        self.data=dict(version=self.VERSION,mechanical_heat_deposited_j=[0.,0.],
            transfer_work_j=[0.,0.],transfer_heat_j=[0.,0.],history=[],
            initial_surface_potential_j=surface.data['initial_potential_j'])
        self._validate()

    def _validate(self):
        p=self.surface.params;tp=self.thermal_params;b=self.material
        if p.heat_capacity_j_kg_k != tp.mantle_heat_capacity_j_kg_k:
            raise ValueError('Mechanical and thermal heat capacities disagree')
        if (p.felsic_density_kg_m3 != b.options.continental_rock_density_kg_m3 or
            p.basalt_density_kg_m3 != b.options.basalt_density_kg_m3):
            raise ValueError('Mechanical and material densities disagree')
        volume=[math.fsum(self.surface.arrays[k+'_volume_m3'])/1e9 for k in ('felsic','basalt')]
        for a,z in zip(volume,b.data['surface_volumes_km3']):
            if abs(a-z)>64*math.ulp(max(a,z,1.)):
                raise ValueError('Mechanical and material volume inventories disagree')
        area=4*math.pi*p.radius_m**2
        if abs(area-b.data['area_m2'])>64*math.ulp(area):
            raise ValueError('Mechanical and thermal areas disagree')
        if self.surface.data['time_myr'] != self.thermal.time_myr or self.thermal.time_myr != b.data['time_myr']:
            raise ValueError('Coupled times disagree')
        heat=math.fsum([*self.surface.data['heat_generated_j'],
                       *(-x for x in self.data['mechanical_heat_deposited_j'])])
        if abs(heat)>32*math.ulp(max(abs(math.fsum(self.surface.data['heat_generated_j'])),1.)):
            raise ValueError('Mechanical heat was omitted or counted twice')

    def _thermal_step(self, dt):
        p=self.surface.params;o=self.options
        self.thermal=replace(self.thermal,mantle_temperature_k=float(self.material.temperatures(self.thermal_params)[0]))
        self.thermal,_=self.material.advance(self.thermal,dt,o.mass_earth,p.radius_m/1000,
            p.gravity_m_s2,o.period_hours,o.primary_mass_jupiter,self.eccentricity,self.thermal_params)

    def _deposit_heat(self, q):
        if q<0 or not math.isfinite(q):raise ValueError('Invalid mechanical heat')
        _add(self.material.data['energy_c_j'],q)
        _add(self.data['mechanical_heat_deposited_j'],q)

    def step(self, dt):
        """Atomic Strang thermal/mechanical split; coefficients remain explicit."""
        if not math.isfinite(dt) or dt<=0:raise ValueError('Invalid coupled time step')
        work=deepcopy(self);before=work.surface.data['heat_generated_j'].copy()
        target=work.thermal.time_myr+dt
        middle=work.thermal.time_myr+dt/2
        if target<=work.thermal.time_myr:raise ValueError('Unrepresentable coupled target')
        first=dt if middle in (work.thermal.time_myr,target) else middle-work.thermal.time_myr
        work._thermal_step(first)
        work.surface.advance(target)
        q=math.fsum([*work.surface.data['heat_generated_j'],*(-x for x in before)])
        work._deposit_heat(q)
        if work.thermal.time_myr<target:
            work._thermal_step(target-work.thermal.time_myr)
        work._validate()
        d=work.diagnostics();work.data['history'].append(dict(time_myr=target,dt_myr=dt,
            deposited_heat_j=q,crust_temperature_k=d['crust_temperature_k'],
            mantle_temperature_k=d['mantle_temperature_k'],energy_residual_j=d['combined_energy_residual_j']))
        self.__dict__.update(work.__dict__)

    def advance(self, target):
        if not math.isfinite(target) or target<self.thermal.time_myr:raise ValueError('Invalid coupled target')
        while self.thermal.time_myr<target:
            self.step(min(self.options.step_myr,target-self.thermal.time_myr))

    def material_request(self, born_m3, returned_m3, *, external_work_available_j=0.):
        """Prescribed fixed-area transfer, with explicit mechanical funding.

        The effective donor/receiver is at the support radius. This is a
        controlled exchange test, not a melt supply or subduction mechanism.
        No whole-pool thermal energy is implicitly available as pump work.
        """
        shape=(len(self.surface.faces),2)
        born=np.asarray(born_m3,dtype=float);returned=np.asarray(returned_m3,dtype=float)
        if any(x.shape!=shape or np.any(~np.isfinite(x)) or np.any(x<0) for x in (born,returned)):
            raise ValueError('Invalid spatial material request')
        if not math.isfinite(external_work_available_j) or external_work_available_j<0:
            raise ValueError('Invalid explicit work budget')
        work=deepcopy(self);s=work.surface;p=s.params
        old=np.stack([s.arrays[k+'_volume_m3'] for k in ('felsic','basalt')],axis=1)
        # Returns remove pre-existing material; no instantaneous birth/recycle loop.
        if np.any(returned>old):raise ValueError('Return exceeds local existing material')
        target=(old-returned)+born
        if np.any((returned>0)&(old-returned==old)) or np.any((born>0)&(old-returned+born==old-returned)):
            raise ValueError('Material request below local volume representation')
        area=s.geometry(s.vertices)[0]
        fraction=np.stack([s.arrays['felsic_fraction'],1-s.arrays['felsic_fraction']],axis=1)
        footprints=area[:,None]*fraction
        if np.any((target>0)&(footprints<=0)):
            raise ValueError('A new material footprint needs an explicit area operator')
        rho=np.array([p.felsic_density_kg_m3,p.basalt_density_kg_m3])
        initial_potential=column_potential(footprints,old,rho,p.gravity_m_s2,p.radius_m)[0]
        final_potential=column_potential(footprints,target,rho,p.gravity_m_s2,p.radius_m)[0]
        # Integrate returns then births. This prevents negative net work from
        # disguising unpaid positive emplacement within a simultaneous request.
        def path_work(start,delta):
            def quadrature(order):
                nodes,weights=leggauss(order);values=[]
                for x,w in zip((nodes+1)/2,weights/2):
                    h=column_potential(footprints,start+x*delta,rho,p.gravity_m_s2,p.radius_m)[2]
                    values.append(float(w)*math.fsum((rho*p.gravity_m_s2*h*delta).ravel()))
                return math.fsum(values)
            coarse=quadrature(9);fine=quadrature(17)
            if abs(fine-coarse)>1e-10*max(abs(fine),1.):
                raise ValueError('Material placement work quadrature unresolved')
            return fine
        return_work=path_work(old,-returned) if np.any(returned) else 0.
        placement=path_work(old-returned,born) if np.any(born) else 0.
        if placement>external_work_available_j:
            raise ValueError(f'Unfunded emplacement work: required {placement} J')
        # Use the existing conservative enthalpy/provenance operator in its
        # defined order, with separate return and creation calls.
        return_totals=np.array([math.fsum(returned[:,i]) for i in range(2)])/1e9
        born_totals=np.array([math.fsum(born[:,i]) for i in range(2)])/1e9
        after_return=np.array([math.fsum((old-returned)[:,i]) for i in range(2)])/1e9
        final=np.array([math.fsum(target[:,i]) for i in range(2)])/1e9
        work.material.exchange([0.,0.],return_totals,after_return,work.thermal_params)
        work.material.exchange(born_totals,[0.,0.],final,work.thermal_params)
        for i,k in enumerate(('felsic','basalt')):
            keep=np.divide(old[:,i]-returned[:,i],old[:,i],out=np.ones(len(old)),where=old[:,i]>0)
            s.arrays[k+'_age_moment_m3_myr']*=keep
            s.arrays[k+'_volume_m3']=target[:,i]
            for key,value in [('material_created_m3',born_totals[i]*1e9),('material_returned_m3',return_totals[i]*1e9)]:
                if key not in s.data:s.data[key]=[[0.,0.],[0.,0.]]
                _add(s.data[key][i],float(value))
        s.evaluate(s.vertices)  # same physical applicability guard as motion
        q=-return_work
        _add(s.data['heat_generated_j'],q);_add(s.data['external_work_j'],placement)
        defect=math.fsum([*(final_potential-initial_potential).ravel(),q,-placement])
        _add(s.data['energy_defect_j'],defect)
        work._deposit_heat(q)
        _add(work.data['transfer_work_j'],placement);_add(work.data['transfer_heat_j'],q)
        work.thermal=replace(work.thermal,mantle_temperature_k=float(work.material.temperatures(work.thermal_params)[0]))
        work._validate()
        event=dict(time_myr=work.thermal.time_myr,born_m3=(born_totals*1e9).tolist(),
            returned_m3=(return_totals*1e9).tolist(),external_placement_work_j=placement,
            gravitational_return_heat_j=q,independent_work_defect_j=defect,
            scope='prescribed transfer at fixed footprints; no automatic crust source')
        work.data['history'].append(event);self.__dict__.update(work.__dict__)
        return event

    def diagnostics(self):
        s=self.surface.budget();d=self.material.diagnostics(self.thermal_params);b=self.material.data
        combined=math.fsum([*b['energy_m_j'],*b['energy_c_j'],-b['initial_energy_j'],
            *(-x for x in b['radiogenic_input_j']),*(-x for x in b['orbital_input_j']),*b['surface_output_j'],
            s['potential_j'],-self.data['initial_surface_potential_j'],
            *(-x for x in self.surface.data['external_work_j'])])
        thermal_error=math.fsum([*b['energy_m_j'],*b['energy_c_j'],-b['initial_energy_j'],
            *(-x for x in b['radiogenic_input_j']),*(-x for x in b['orbital_input_j']),*b['surface_output_j'],
            *(-x for x in self.data['mechanical_heat_deposited_j'])])
        return dict(**s,**{k:v for k,v in d.items() if k not in s and k!='latent_chemical_mechanical_energy_included'},
            combined_energy_residual_j=combined,thermal_energy_residual_j=thermal_error,
            mechanical_heat_deposited_j=math.fsum(self.data['mechanical_heat_deposited_j']),
            mechanical_energy_included=True,latent_heat_included=False,
            scope='welded fixed-base dilation and sensible heat; no automatic creation or fracture')

    def checkpoint(self):
        return dict(data=deepcopy(self.data),surface=self.surface.checkpoint(),
            material=self.material.checkpoint(),thermal=asdict(self.thermal),
            thermal_parameters=asdict(self.thermal_params),options=asdict(self.options),
            eccentricity=dict(times_myr=np.asarray(self.eccentricity.times_myr).tolist(),
                              eccentricity=np.asarray(self.eccentricity.eccentricity).tolist()))

    @classmethod
    def restore(cls, cp):
        d=deepcopy(cp);obj=cls.__new__(cls)
        if d['data']['version']!=cls.VERSION:raise ValueError('Unsupported deformation/exchange restart')
        obj.data=d['data'];obj.surface=DeformingSurface.restore(d['surface'])
        obj.material=MaterialExchangeBudget.restore(d['material'],MaterialExchangeParameters(**d['material']['options']))
        obj.thermal=ThermalState(**d['thermal']);obj.thermal_params=ThermalParameters(**d['thermal_parameters'])
        obj.options=CoupledParameters(**d['options'])
        obj.eccentricity=EccentricityHistory(**{k:np.array(v) for k,v in d['eccentricity'].items()})
        obj._validate();return obj

    def save(self, path):
        path=Path(path);path.mkdir(parents=True,exist_ok=False)
        cp=self.checkpoint();arrays=cp['surface'].pop('arrays')
        np.savez_compressed(path/'surface.npz',**{k:np.asarray(v) for k,v in arrays.items()})
        with (path/'meta.json').open('x') as f:json.dump(cp,f,indent=2,allow_nan=False)

    @classmethod
    def load(cls,path):
        path=Path(path);cp=json.loads((path/'meta.json').read_text())
        with np.load(path/'surface.npz') as z:cp['surface']['arrays']={k:z[k].tolist() for k in z.files}
        return cls.restore(cp)
