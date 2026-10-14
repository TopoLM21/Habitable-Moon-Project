"""Shared mature engine invariants under a different, explicitly mafic start."""
import json
from copy import deepcopy
import numpy as np
import pytest
from tectonics.unified_geological_genesis import UnifiedGeologicalGenesis,GeologicalGenesisParameters
from tectonics.genesis_implicit_enthalpy import solve
from tectonics.genesis_transfer_bridge import encoded
from tectonics.mesh import build_icosphere
from analysis.genesis_geological_handoff.experiment import exact_saved_state

@pytest.fixture
def source(tmp_path):
    mesh=build_icosphere(2);a=mesh.physical_cell_areas_km2(5287.)
    m=np.r_[np.repeat(a*.07*3000*1e9/2,2),1.99e24,1.1267e21]
    H=np.r_[m[:-2]*1200*600,m[-2]*(1200*2300+4e5),m[-1]*4200*600]
    body=dict(config=dict(solid_nodes=np.arange(640).reshape(320,2).tolist(),mass=m.tolist(),radius_km=5287.,
        thermal_parameters=dict(surface_gravity_m_s2=7.12),tidal_parameters=dict(primary_mass_kg=9.49065e27,love_h2=.6)),
        state=dict(time_myr=.8684,energy=H.tolist(),damage=np.zeros(320).tolist(),born_material=[],
                   orbit=dict(semimajor_axis_km=773185.95,eccentricity=4.3803e-5)))
    path=tmp_path/'source.json';path.write_text(json.dumps(dict(payload=body)));return path

def test_initial_full_ocean_and_explicit_scenario_budget(source):
    e=UnifiedGeologicalGenesis.from_checkpoint(source);r=e.record
    assert len(e.checkpoint.system.plates)==1
    assert e.water_diagnostics()['ocean_area_fraction']==1.
    assert np.sum(e.checkpoint.state.continental_volume_km3)==0
    assert 47<r['period_hours']<48 and r['primary_mass_jupiter']==5.
    assert r['water_initialization']['signed_H_change_from_original_j']<0
    assert r['water_initialization']['condensation_history_reproduced'] is False

def test_rigid_mantle_rotation_cannot_drive_phenomenological_strain(source):
    e=UnifiedGeologicalGenesis.from_checkpoint(source)
    e.checkpoint.mantle_flow.cell_omega_rad_per_myr[:]=[.001,.002,-.003]
    assert max(e.rates())<1e-11
    assert e.event_time(1.,e.rates())==1.
    assert e.candidate(np.ones(320)) is None  # uniform saturation is not a separating crack

def test_transactional_bounds_and_zero_forcing_do_not_invent_plates(source):
    e=UnifiedGeologicalGenesis.from_checkpoint(source,GeologicalGenesisParameters(mantle_rms_km_per_myr=0))
    r=encoded(e.record);cp=e.checkpoint
    with pytest.raises(ValueError):e.step(3.)
    assert cp is e.checkpoint and encoded(e.record)==r
    e.step(1.);assert len(e.checkpoint.system.plates)==1
    assert e.checkpoint.transport_state.cumulative_commit_count==0
    assert e.water_diagnostics()['liquid_mass_kg']==e.record['water_mass_kg']

def test_exact_restart_keeps_native_rotation_and_topology_memory(source,tmp_path):
    e=UnifiedGeologicalGenesis.from_checkpoint(source)
    e.step(1.);e.step(1.);saved=e.save(tmp_path/'saved');resumed=UnifiedGeologicalGenesis.load(saved)
    assert resumed.checkpoint.manager.params==e.checkpoint.manager.params
    before=e.checkpoint.baseline
    e.step(1.);resumed.step(1.)
    np.testing.assert_array_equal(e.checkpoint.baseline.cell_plate,before.cell_plate)
    exact_saved_state(e.save(tmp_path/'direct'),resumed.save(tmp_path/'resumed'))

def test_fixed_ocean_temperature_exports_exactly_the_lost_heat():
    m=np.array([2.,3.]);cp=np.array([1200.,4200.]);T=np.array([1700.,300.]);H=m*cp*T
    out=solve(m,H/m,cp,np.zeros(2),np.full(2,1400.),np.full(2,2000.),np.zeros(2),[(0,1,10.)],1e8,fixed_temperature={1:300.})
    assert out['temperature_k'][1]==300.
    expected=(m[0]*cp[0]*T[0]+1e9*300)/(m[0]*cp[0]+1e9)
    assert out['temperature_k'][0]==pytest.approx(expected,abs=1e-12,rel=0)
    # Subtracting nearly equal temperatures then multiplying by G*dt has this
    # roundoff bound; absolute heat tolerance must include that conditioning.
    roundoff=32*np.spacing(max(out['temperature_k']))*1e9
    assert abs(np.sum(m*out['specific_h']-H)+out['exchange_j'][0])<=roundoff

def test_seeded_weakness_is_reproducible_and_parameters_explicit(source):
    a=UnifiedGeologicalGenesis.from_checkpoint(source);b=UnifiedGeologicalGenesis.from_checkpoint(source)
    assert a.record['material_strength']==b.record['material_strength']
    assert np.std(np.log(a.record['material_strength']))==pytest.approx(.2)
    with pytest.raises(ValueError):UnifiedGeologicalGenesis.from_checkpoint(source,GeologicalGenesisParameters(strength_log_std=-1))
