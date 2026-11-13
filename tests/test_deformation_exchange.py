from copy import deepcopy
from dataclasses import replace
from decimal import Decimal, localcontext
import math

import numpy as np
import pytest

from tectonics.conforming_surface import ConformingSurface
from tectonics.regional_deformation import DeformingSurface,DilationParameters
from tectonics.deformation_exchange import DeformationExchange,CoupledParameters
from tectonics.mesh import build_icosphere
from tectonics.tides import constant_eccentricity
from test_material_exchange import initialized


def coupled(step=1.,length=1e6):
    budget,state,thermal,p,_=initialized()
    mesh=build_icosphere(1);n=mesh.cell_count
    arrays=dict(owner=(mesh.centroids[:,0]<0).astype(int),
        felsic_fraction=state.continental_fraction,
        felsic_volume_m3=state.continental_volume_km3*1e9,
        basalt_volume_m3=state.oceanic_volume_km3*1e9,
        felsic_age_moment_m3_myr=state.continental_volume_km3*1e9*40,
        basalt_age_moment_m3_myr=state.oceanic_volume_km3*1e9*40,
        potential=np.linspace(0,.2,n))
    surface=DeformingSurface.from_surface(ConformingSurface(mesh,arrays,
        initial_heat_j=budget.data['energy_c_j']),DilationParameters(length))
    return DeformationExchange(surface,budget,thermal,p,constant_eccentricity(0.),
        CoupledParameters(step_myr=step))


def inventories(obj):
    return np.stack([obj.surface.arrays[k+'_volume_m3'] for k in ('felsic','basalt')],axis=1)


def test_mechanical_heat_has_one_thermal_owner_and_combined_balance():
    obj=coupled();before=obj.checkpoint();obj.advance(4.);d=obj.diagnostics()
    assert d['heat_generated_j']==d['mechanical_heat_deposited_j']>0
    assert abs(d['thermal_energy_residual_j'])/d['heat_generated_j']<2e-12
    assert abs(d['combined_energy_residual_j']-d['energy_residual_j'])/d['heat_generated_j']<3e-12
    assert d['orbital_input_j']==0 and d['surface_output_j']>0
    assert np.array_equal(inventories(obj),inventories(DeformationExchange.restore(before)))
    assert d['mass_residual_kg']==pytest.approx(0.,abs=2e9)


def test_unfunded_material_addition_and_unrepresentable_exchange_are_atomic():
    obj=coupled();original=obj.checkpoint();vol=inventories(obj)
    with pytest.raises(ValueError,match='Unfunded'):obj.material_request(vol*.01,np.zeros_like(vol))
    assert obj.checkpoint()==original
    small=np.zeros_like(vol);small[vol>0]=1e-20
    with pytest.raises(ValueError,match='representation'):obj.material_request(small,small)
    assert obj.checkpoint()==original


def test_paid_addition_preserves_mass_provenance_age_and_donor_enthalpy():
    obj=coupled();vol=inventories(obj);tm,tc=obj.material.temperatures(obj.thermal_params)
    q0=deepcopy(obj.material.data['energy_c_j']);b0=obj.surface.budget();age=obj.surface.arrays['basalt_age_moment_m3_myr'].copy()
    event=obj.material_request(vol*.02,np.zeros_like(vol),external_work_available_j=1e28)
    d=obj.diagnostics();dm=np.sum(vol*.02,axis=0)@np.array([2800.,2900.])
    with localcontext() as ctx:
        ctx.prec=60
        delta=sum(map(Decimal,obj.material.data['energy_c_j']))-sum(map(Decimal,q0))
        assert float(delta)==pytest.approx(dm*1200*tm,rel=2e-15)
    assert d['external_work_j']==event['external_placement_work_j']>0
    assert d['heat_generated_j']==0
    assert d['potential_j']-b0['potential_j']==pytest.approx(event['external_placement_work_j'],rel=2e-13)
    assert np.array_equal(obj.surface.arrays['basalt_age_moment_m3_myr'],age)
    assert np.max(abs(np.array(d['origin_mass_residual_kg'])))<2e9
    assert np.array_equal(inventories(DeformationExchange.restore(obj.checkpoint())),inventories(obj))


def test_return_releases_independently_integrated_gravity_work_and_keeps_history():
    obj=coupled();vol=inventories(obj);b=obj.diagnostics();age=obj.surface.arrays['felsic_age_moment_m3_myr'].copy()
    event=obj.material_request(np.zeros_like(vol),vol*.1)
    d=obj.diagnostics()
    assert event['gravitational_return_heat_j']>0
    assert event['external_placement_work_j']==0
    assert d['potential_j']+d['heat_generated_j']==pytest.approx(b['potential_j'],rel=2e-15)
    assert abs(d['combined_energy_residual_j'])/d['heat_generated_j']<3e-12
    assert obj.surface.arrays['felsic_age_moment_m3_myr']==pytest.approx(age*.9,rel=2e-15)
    assert d['recycled_kg']>0 and d['created_kg']==0


def test_positive_placement_cannot_be_hidden_by_simultaneous_return():
    obj=coupled();vol=inventories(obj);before=obj.checkpoint()
    with pytest.raises(ValueError,match='Unfunded'):obj.material_request(vol*.01,vol*.01)
    assert obj.checkpoint()==before
    event=obj.material_request(vol*.01,vol*.01,external_work_available_j=1e28)
    assert event['external_placement_work_j']>0 and event['gravitational_return_heat_j']>0


def test_coupled_restart_includes_transfers_heat_controller_and_material_memory(tmp_path):
    obj=coupled();obj.advance(2.);vol=inventories(obj)
    obj.material_request(vol*.01,vol*.02,external_work_available_j=1e28)
    obj.save(tmp_path/'coupled');restored=DeformationExchange.load(tmp_path/'coupled')
    assert restored.checkpoint()==obj.checkpoint()
    obj.advance(4.);restored.advance(4.)
    assert restored.checkpoint()==obj.checkpoint()
    bad=obj.checkpoint();bad['data']['mechanical_heat_deposited_j'][0]*=2
    with pytest.raises(ValueError,match='counted twice'):DeformationExchange.restore(bad)
    bad=obj.checkpoint();bad['surface']['arrays']['basalt_volume_m3'][0]*=1.01
    # Choose a nonzero cell when testing the integrity ledger.
    index=int(np.flatnonzero(inventories(obj)[:,1])[0]);bad['surface']['arrays']['basalt_volume_m3'][index]*=1.01
    with pytest.raises(ValueError,match='volume'):DeformationExchange.restore(bad)


def test_coupled_time_refinement_and_tiny_final_interval():
    results=[]
    for step in (2.,1.,.5):
        obj=coupled(step);obj.advance(8.);results.append(obj.diagnostics())
    delta=[abs(results[i]['crust_temperature_k']-results[i+1]['crust_temperature_k']) for i in (0,1)]
    assert delta[1]<delta[0]/2
    obj=coupled();obj.advance(1.);target=float(np.nextafter(1.,2.));obj.advance(target)
    assert obj.thermal.time_myr==obj.surface.data['time_myr']==target
    assert obj.data['history'][-1]['dt_myr']==target-1.
