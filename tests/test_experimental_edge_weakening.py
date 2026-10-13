"""Energetic and mechanical checks for the opt-in vector feedback."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from tectonics.simplified_genesis import SimplifiedGenesis
from tectonics.experimental_edge_weakening import MODE,commit_damage
from analysis.genesis_geological_handoff.nodal_retirement_check import OPTION
from analysis.genesis_geological_handoff.edge_weakening_controls import local_controls


@pytest.fixture
def source():
    p=Path('/workspace/genesis-moving-basal-source-01.json')
    if not p.is_file():pytest.skip('Local authenticated source only; no fetch')
    m=SimplifiedGenesis.load(p);m.config['experimental_geological_edge_fatigue']={**OPTION,'mode':MODE}
    return m


def test_vector_redistribution_inventory_and_restart():
    if not Path('/workspace/genesis-moving-basal-source-01.json').is_file():pytest.skip('Local source absent')
    result=local_controls()
    assert max(result['patch_stress_ratio'])<1
    assert min(result['adjacent_stress_change_pa'])>0


def test_independent_whole_energy_includes_reserve_change(source):
    for _ in range(2):
        prior=deepcopy(source.state);source.advance(25.);s=source.state;L=s['last_ledger']
        deltaH=np.sum(np.asarray(s['energy'],np.longdouble)-np.asarray(prior['energy'],np.longdouble))
        deltaR=s['experimental_edge_fracture_reserve_j']-prior.get('experimental_edge_fracture_reserve_j',0.)
        residual=deltaH+L['elastic_after']-L['elastic_before']+L['kinetic_after']-L['kinetic_before']+deltaR-L['source_energy']-L['external_work']+L['numerical_memory']+L['numerical_kinetic']
        assert abs(residual)<=L['whole_allowance_j']
        assert deltaR==pytest.approx(L['edge_fracture_release_j'],rel=1e-13)
        assert L['edge_damage_heat_j']==0 and deltaR>=0


def test_invalid_reserve_and_irreversible_edge_state(source):
    source.state['experimental_edge_fracture_reserve_j']=-1
    with pytest.raises(ValueError,match='EDGE_RESERVE_INVALID'):source.validate()
    source.state['experimental_edge_fracture_reserve_j']=0
    d=np.full(len(source.mesh.shared_edges),.8);commit_damage(source,d)
    with pytest.raises(ValueError,match='EDGE_HEALING'):commit_damage(source,d*.5)


def test_uniform_damage_changes_vector_response_under_same_load(source):
    from tectonics.simplified_genesis_nodal import mechanical,material_mass
    from tectonics.experimental_edge_weakening import scale
    T=source.temperature(source.state['energy']);D=np.array(source.state['damage'])
    a=mechanical(source,T,D,T,25*31557600.,material_mass(source))
    commit_damage(source,np.full(len(source.mesh.shared_edges),.8))
    b=mechanical(source,T,D,T,25*31557600.,material_mass(source))
    np.testing.assert_allclose(scale(source,D),np.full(source.nf,.02+.98*.2**2),rtol=1e-14,atol=0)
    assert np.linalg.norm(b['stress'])<np.linalg.norm(a['stress'])
    assert not np.array_equal(a['velocity'],b['velocity'])
    assert b['ledger']['balance_relative']<5e-10


def test_handoff_archives_fracture_reserve_without_heat(source):
    from tectonics.genesis_geological_thermal import ThermalGeologicalContinuation
    from tests.test_experimental_nodal_handoff import boundary
    d=np.zeros(len(source.mesh.shared_edges));d[0]=.8;commit_damage(source,d)
    g=ThermalGeologicalContinuation.from_source(source,experimental_nodal_retirement=True,fracture_edges=boundary(source))
    e=g.record['mapping']['energy'];reserve=source.state['experimental_edge_fracture_reserve_j']
    assert e['source_edge_fracture_reserve_j']==reserve and reserve>0
    assert g.source['state']['experimental_edge_fracture_reserve_j']==reserve
    assert e['signed_active_mechanical_model_change_j']==pytest.approx(e['mapped_kinetic_proxy_j']-e['source_kinetic_j']-e['retired_elastic_j']-e['source_passive_reserve_j']-reserve)
    assert e['mapping_heat_j']==0
