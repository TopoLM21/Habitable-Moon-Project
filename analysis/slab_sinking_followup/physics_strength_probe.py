"""Illustrative confined yield/creep response; never used by production dynamics.

This tests the interpretation of a surface strength applied to a whole cold
mantle hinge. Existing shell rheology/friction are reused without a fitted rate.
No cutoff, material state, force, or trajectory is changed.
"""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import erf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tectonics.genesis import SECONDS_PER_MYR
from tectonics.genesis_shell import rock_temperature
from tectonics.genesis_starter_continuation import load_starter_source, _load_cp, YoungWorldCoupling
from tectonics.genesis_starter_fracture import YoungShellFracture
from tectonics.simulation import load_config


def extensional_yield_pa(pressure_pa, tensile_pa, cohesion_pa, friction):
    """Normal-faulting Mohr-Coulomb envelope, capped by tensile opening.

    Compression is positive: sigma1=P and sigma3=P-differential_stress.
    No pore pressure is assumed. Friction therefore describes this idealized
    confined comparison, not a prediction of the hydrated slab's actual yield.
    """
    phi = np.arctan(friction)
    frictional = 2.*(cohesion_pa*np.cos(phi)+pressure_pa*np.sin(phi))/(1.+np.sin(phi))
    return np.minimum(tensile_pa+pressure_pa, frictional)


def extension_response(tension_n, width_m, weight_m, viscosity_pa_s, yield_pa):
    """Solve F=W integral min(4 eta strain_rate,Y) dz in plane strain.

    Uniform axial strain rate across the section; the rate is inferred from
    the supplied force, never chosen to obtain a target plate speed. Perfect
    plasticity supplies no unique finite strain rate at/above full yield.
    """
    weight = np.asarray(weight_m, float)
    eta = np.asarray(viscosity_pa_s, float)
    limit = np.asarray(yield_pa, float)
    capacity = float(width_m*np.dot(weight, limit))
    if tension_n <= 0.:
        return dict(status='no_tensile_extension', idealized_capacity_n=capacity, strain_rate_s=0., strain_time_myr=None)
    if tension_n >= capacity:
        return dict(status='at_or_above_idealized_plastic_capacity', idealized_capacity_n=capacity, strain_rate_s=None, strain_time_myr=None)
    lo, hi = 0., float(np.max(limit/(4.*eta)))
    for _ in range(180):
        mid = .5*(lo+hi)
        force = float(width_m*np.dot(weight, np.minimum(4.*eta*mid, limit)))
        if force < tension_n:
            lo = mid
        else:
            hi = mid
    rate = .5*(lo+hi)
    result_force = float(width_m*np.dot(weight, np.minimum(4.*eta*rate, limit)))
    return dict(status='finite_idealized_viscoplastic_response', idealized_capacity_n=capacity,
        strain_rate_s=rate, strain_time_myr=1./rate/SECONDS_PER_MYR,
        force_relative_residual=abs(result_force-tension_n)/tension_n,
        yielded_thickness_fraction=float(np.dot(weight, 4.*eta*rate >= limit)/np.sum(weight)))


def profile(model, sample, state, cfg, source_face, thickness_km, order):
    shell = model.shell
    crust = float(state.crust_thickness_km[source_face])
    reference_z = np.r_[0., (np.arange(shell.column_layers)+.5)*shell.column_depth_km/shell.column_layers, shell.column_depth_km]
    surface, mantle = (float(sample.thermal[k]) for k in ('surface_temperature_k','mantle_temperature_k'))
    reference_t = np.r_[surface, rock_temperature(sample.state.column_enthalpy, model.thermal), mantle]
    edges = np.unique(np.r_[crust, crust+thickness_km, reference_z])
    edges = edges[(edges >= crust) & (edges <= crust+thickness_km)]
    nodes, gauss = np.polynomial.legendre.leggauss(order)
    z = (.5*(edges[1:]+edges[:-1])[:,None]+.5*np.diff(edges)[:,None]*nodes).ravel()
    weights = (.5*np.diff(edges)[:,None]*gauss*1000.).ravel()
    temperature = np.interp(z, reference_z, reference_t)
    age = float(state.crust_age_myr[source_face])
    elapsed = float(state.time_myr)-float(cfg['young_shell']['origin_time_myr'])
    primordial = age >= elapsed-64.*np.finfo(float).eps*max(1.,abs(state.time_myr))
    if not primordial:
        kappa = float(cfg.get('mechanical_lithosphere',{}).get('thermal_diffusivity_m2_s',1e-6))
        length = 2.*np.sqrt(kappa*age*SECONDS_PER_MYR)/1000.
        cooling = surface+(mantle-surface)*erf(z/length) if length>0. else np.full(z.shape,mantle)
        temperature = np.maximum(temperature,cooling)
    viscosity = np.clip(shell.viscosity_reference_pa_s*np.exp(np.clip(
        shell.activation_energy_j_mol/8.314462618*(1./np.maximum(temperature,1.)-1./shell.viscosity_reference_temperature_k),-60.,60.)),
        shell.viscosity_min_pa_s,shell.viscosity_max_pa_s)
    return z,weights,temperature,viscosity,dict(material_age_myr=age,primordial=bool(primordial),crust_overburden_km=crust,
        reference_column_depth_km=shell.column_depth_km)


def main():
    input_path=Path(__file__).with_name('physics_failure_probe.json')
    probes=json.loads(input_path.read_text(encoding='utf-8'))
    rows=[]
    for probe in probes['probes']:
        if not probe['failures']:
            continue
        root=Path(probe['source'])
        before={name:sha256((root/name).read_bytes()).hexdigest() for name in probe['source_sha256']}
        assert before==probe['source_sha256']
        cfg=load_config(root/'mature_config.yaml')
        model,young,_=load_starter_source(root/'young_context/starter_checkpoint.npz')
        cp=_load_cp(root/'mature_checkpoint',cfg)
        fracture=YoungShellFracture.load(model,root/'young_context/fracture_memory.npz')
        coupling=YoungWorldCoupling(model,young,cfg,cp,fracture=fracture)
        coupling.mechanical_fields(cp.state)
        sample=model.loading.sample(young.thermal_context)
        for failure in probe['failures']:
            face=int(failure['source_face']); h=float(failure['cold_hinge_thickness_km']); width=float(failure['trench_length_km'])*1000.
            strength=float(failure['neck_strength_pa'])
            assert np.isclose(strength,fracture.memory.strength_pa[face],rtol=1e-12)
            incoming=float(cp.state.mantle_lithosphere_thickness_km[face])
            if not np.isclose(h,incoming,rtol=1e-12):
                rows.append(dict(elapsed_myr=probe['elapsed_myr'],contact_key=failure['contact_key'],status='skip_archived_hinge_without_matching_local_profile'))
                continue
            water=float(fracture.memory.water_access[face])
            mu=model.parameters.friction_dry+(model.parameters.friction_wet-model.parameters.friction_dry)*water
            cohesion=model.parameters.shear_cohesion_pa/model.shell.tensile_strength_pa*strength
            output=[]
            for order in (16,32):
                z,w,T,eta,context=profile(model,sample,cp.state,cfg,face,h,order)
                pressure=model.shell.density_kg_m3*probe['gravity_m_s2']*z*1000.
                envelope=extensional_yield_pa(pressure,strength,cohesion,mu)
                answer=extension_response(failure['tension_n'],width,w,eta,envelope)
                output.append(dict(**answer,**context,quadrature_order=order,
                    effective_confined_strength_mpa=answer['idealized_capacity_n']/(width*h*1000.)/1e6,
                    temperature_min_max_k=[float(T.min()),float(T.max())],
                    viscosity_min_max_pa_s=[float(eta.min()),float(eta.max())]))
            high=output[-1]; low=output[0]
            rows.append(dict(elapsed_myr=probe['elapsed_myr'],contact_key=failure['contact_key'],source_face=face,
                pressure_density_kg_m3=model.shell.density_kg_m3,gravity_m_s2=probe['gravity_m_s2'],water_access=water,friction=mu,
                weakened_cohesion_mpa=cohesion/1e6,old_strength_mpa=strength/1e6,cold_hinge_thickness_km=h,
                axial_stress_mpa=failure['tension_n']/(width*h*1000.)/1e6,
                idealized_capacity_ratio_to_old=high['idealized_capacity_n']/failure['capacity_n'],
                capacity_quadrature_relative_change=abs(high['idealized_capacity_n']/low['idealized_capacity_n']-1.),
                strain_time_quadrature_relative_change=(abs(high['strain_time_myr']/low['strain_time_myr']-1.) if high['strain_time_myr'] is not None else None),**high))
        assert {name:sha256((root/name).read_bytes()).hexdigest() for name in before}==before
    result=dict(interpretation='Diagnostic sensitivity only: no production cutoff/force/history changes; frozen0.4 examples, not original event history.',
        assumptions=['Existing shell density3000kg/m3 and actual planet gravity, zero pore pressure.',
            'Cold mantle alone; chemical crust supplies overburden but no neck strength.',
            'Horizontal plane-strain uniform axial strain rate, lithostatic vertical principal stress.',
            'Current age/reference-column temperature; existing Newtonian Shell Arrhenius viscosity.',
            'Existing wet/damage weakening is used for cohesion; hydrated creep, grain size, nonlinear olivine and localization are unresolved.',
            'Strain times are unit-strain times of an undeformed section, never breakoff times.'],
        input_probe_sha256=sha256(input_path.read_bytes()).hexdigest(),
        source_sha256={str(p['source']):p['source_sha256'] for p in probes['probes']},
        helper_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),rows=rows)
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([dict(age=r['elapsed_myr'],contact=r['contact_key'],stress=r.get('axial_stress_mpa'),confined=r.get('effective_confined_strength_mpa'),time_myr=r.get('strain_time_myr'),status=r['status']) for r in rows],indent=2))

if __name__=='__main__':
    main()
