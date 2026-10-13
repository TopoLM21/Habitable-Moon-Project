"""Active H transport, existing condensation law, and model-change bookkeeping."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from tectonics.genesis import GenesisParameters
from tectonics.genesis_geological_thermal import ThermalGeologicalContinuation, eligibility
from tectonics.genesis_implicit_enthalpy import solve
from tectonics.genesis_thermal_ports import water_specific_enthalpy, water_temperature
from tectonics.genesis_transfer_bridge import encoded
from analysis.genesis_geological_handoff.thermal_experiment import thermal_source,run


@pytest.fixture(scope='module')
def source():
    return thermal_source()[0]


def test_active_thermal_end_to_end_restart(tmp_path):
    result=run(tmp_path/'run')
    assert result['deep_final_k']>result['deep_initial_k']
    assert result['surface_final_k']<647.096
    assert result['created_H_j']>0 and result['recycled_H_j']>0
    engine=ThermalGeologicalContinuation.load(tmp_path/'run/direct')
    assert engine.checkpoint.hydrosphere.water_volume_km3>0
    assert engine.checkpoint.thermal.time_myr==engine.checkpoint.state.time_myr
    assert engine.record['thermal_inventory']['time_myr']==engine.checkpoint.state.time_myr
    assert not engine.checkpoint.state.continental_fraction.any()
    for row in engine.record['steps']:
        t=row['thermal']
        assert abs(t['total_mass_residual_kg'])<=t['mass_roundoff_allowance_kg']
        assert t['mechanical_heat_j']==t['inactive_memory_heat_j']==0


def test_connected_water_silicate_graph_conserves_energy():
    p=GenesisParameters();mass=np.array([2e14,1e14]);cp=np.array([1200.,4200.])
    initial=np.array([1200.*1000.,water_specific_enthalpy(600.,p)])
    out=solve(mass,initial,cp,np.array([4e5,0.]),np.full(2,1400.),np.full(2,2000.),
              np.array([0.,p.water_latent_heat_j_kg]),[(0,1,1e12)],1e9,water_parameters=p)
    assert np.max(abs(out['residual_k'])/out['tolerance_k'])<=1
    np.testing.assert_allclose(np.sum(mass*out['specific_h']),np.sum(mass*initial),rtol=3e-14)
    assert out['temperature_k'][1]>600
    assert water_temperature(out['specific_h'][1],p)==pytest.approx(out['temperature_k'][1])


def test_unknown_chemistry_is_not_fabricated_or_a_gate(source):
    before=encoded(dict(config=source.config,state=source.state)); e=eligibility(source)
    assert e['eligible'] and not e['chemistry_required_for_this_mode']
    engine=ThermalGeologicalContinuation.from_source(source)
    assert engine.record['mapping']['composition']['chemical_fractions'] is None
    assert np.all(engine.checkpoint.state.continental_fraction==0)
    assert encoded(dict(config=source.config,state=source.state))==before


def test_nonzero_kinetic_mapping_change_is_not_heat(source):
    m=deepcopy(source);m.state['omega']=np.tile([0.,0.,1e-15],(3,1)).tolist()
    e=ThermalGeologicalContinuation.from_source(m).record['mapping']['energy']
    assert e['source_kinetic_j']>0
    assert e['signed_model_change_kinetic_j']==e['mapped_kinetic_proxy_j']-e['source_kinetic_j']
    assert e['signed_model_change_kinetic_j']!=0
    assert e['mapping_heat_j']==0 and not e['total_mechanical_energy_conserved']


def test_thermal_failure_rolls_back_native_transport(source,monkeypatch):
    engine=ThermalGeologicalContinuation.from_source(source)
    original=engine.checkpoint;record=encoded(engine.record)
    def fail(*args,**kwargs):raise ValueError('thermal failure')
    monkeypatch.setattr('tectonics.genesis_geological_thermal.apply_ports',fail)
    with pytest.raises(ValueError,match='thermal failure'):engine.step(10.)
    assert engine.checkpoint is original and encoded(engine.record)==record


def test_homogeneous_source_eos_required_for_mass_mixing(source):
    mixed=deepcopy(source);mixed.config['cp'][0]+=1
    assert not eligibility(mixed)['eligible']
    with pytest.raises(ValueError,match='Mixed silicate EOS'):
        ThermalGeologicalContinuation.from_source(mixed)


def test_face_id_geometry_mapping_keeps_extensive_inventories(source):
    m=deepcopy(source);m.mesh.centroids=m.mesh.centroids.copy();m.mesh.centroids[0]*=1.00001
    engine=ThermalGeologicalContinuation.from_source(m)
    assert engine.record['eligibility']['canonical_face_relocation_max_km']>0
    np.testing.assert_array_equal(engine.record['thermal_inventory']['energy_j'],m.state['energy'])
    np.testing.assert_array_equal(engine.record['thermal_inventory']['mass_kg'],m.config['mass'])
