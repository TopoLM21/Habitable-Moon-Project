"""Fast velocity separation and exact preservation of the known stress history."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from analysis.genesis_geological_handoff.velocity_only_start import initialize,protected_state,terminal_fast_modes,mass_whitener,stress_inventory
from analysis.genesis_geological_handoff.spherical_coupled import SphericalCoupled
from analysis.genesis_geological_handoff.equilibrated_start import equilibrate
from tectonics.genesis_transfer_bridge import encoded


def test_only_fast_velocity_is_replaced_with_mass_metric():
    M=np.diag([2.,3.]);T=mass_whitener(M);Dbar=np.diag([.01,1e5]);Kbar=np.diag([.2,1.])
    force=np.array([4.,6.]);v0=np.array([2.,-3.])
    v,r=terminal_fast_modes(M,Dbar,Kbar,T,force,v0,100.)
    assert r['selected'].tolist()==[False,True]
    assert v[0]==pytest.approx(v0[0])
    assert v[1]==pytest.approx(force[1]/(3*1e5),abs=1e-15)
    assert r['cutoff']==100.
    np.testing.assert_allclose(T.T@M@T,np.eye(2),atol=1e-15)


@pytest.fixture(scope='module')
def prepared():
    path=Path('/workspace/spherical-coupled-preflight-initial.json')
    if not path.exists():pytest.skip('Full spherical source required')
    a=SphericalCoupled.load(path);before=deepcopy(a.model.state);inventory=stress_inventory(a);protected=protected_state(a)
    ledger=initialize(a)
    return a,before,inventory,protected,ledger


def test_real_known_memory_thresholds_and_energy_are_exact(prepared):
    a,before,inventory,protected,L=prepared
    assert protected_state(a)==protected
    assert encoded(stress_inventory(a))==encoded(inventory)
    for key in ('spherical_memory','spherical_born_memory','memory','nodal_displacement','damage','energy'):
        assert encoded(a.model.state[key])==encoded(before[key])
    assert inventory['faces_above_tensile']==84 and inventory['edges_above_shear']==418
    assert L['elastic_change_j']==0 and L['thermal_heat_j']==0 and L['time_advanced_years']==0
    assert L['selected_modes']==970 and L['preserved_velocity_modes']==2
    assert L['selected_rate_minmax_per_s'][0]>10*L['cutoff_rate_per_s']
    assert L['excluded_rate_max_per_s']<L['cutoff_rate_per_s']/10
    assert L['projector_dense_factored_error']<1e-7 and L['fast_force_relative']<1e-7
    assert L['surface_acceleration_after_max_m_s2']>1e-4  # residual is not hidden
    assert not L['equilibrium_claim']


def test_checkpoint_reload_and_one_time_operation(prepared,tmp_path):
    a,*_=prepared;p=tmp_path/'velocity.json';a.save(p);b=SphericalCoupled.load(p)
    assert encoded(a.model.state)==encoded(b.model.state)
    assert encoded(a.origin)==encoded(b.origin)
    with pytest.raises(ValueError,match='ALREADY_VELOCITY_PREPARED'):initialize(b)
    with pytest.raises(FileExistsError):a.save(p)


def test_weakened_scenario_needs_explicit_permission_and_is_rejected_as_source(prepared):
    a=deepcopy(prepared[0]);original=encoded(a.model.state)
    with pytest.raises(ValueError,match='EXPLICIT_WEAKENED_SCENARIO_REQUIRED'):equilibrate(a)
    a.origin['equilibrated_start']={'version':'weakened'}
    with pytest.raises(ValueError,match='WEAKENED_SCENARIO_NOT_A_PRIMARY_SOURCE'):initialize(a)
    assert encoded(a.model.state)==original


def test_failed_velocity_projection_rolls_back(prepared,monkeypatch):
    import analysis.genesis_geological_handoff.velocity_only_start as module
    a=deepcopy(prepared[0]);a.origin.pop('velocity_only_start');state=encoded(a.model.state);origin=encoded(a.origin)
    def fail(_):raise ValueError('UNRESOLVED_SPECTRUM')
    monkeypatch.setattr(module,'prepare',fail)
    with pytest.raises(ValueError,match='UNRESOLVED_SPECTRUM'):initialize(a)
    assert encoded(a.model.state)==state and encoded(a.origin)==origin
