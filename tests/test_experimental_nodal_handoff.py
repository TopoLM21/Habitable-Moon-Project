"""Local real-source adapter checks; never retrieve archives or continue a pilot."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from tectonics.simplified_genesis import SimplifiedGenesis
from tectonics.genesis_geological_thermal import ThermalGeologicalContinuation, eligibility
from tectonics.genesis_transfer_bridge import encoded
from tectonics.experimental_genesis_edge_rule import update
from tectonics.simplified_genesis_nodal import material_mass


@pytest.fixture
def source():
    path=Path('/workspace/genesis-moving-basal-source-01.json')
    if not path.is_file():pytest.skip('Authenticated local source required; no fetch')
    return SimplifiedGenesis.load(path)


def boundary(m):
    side=m.mesh.centroids[:,2]>0
    return np.array([side[f]!=side[g] for f,g,_,_ in m.mesh.shared_edges])


def test_explicit_mapping_archives_source_and_accounts_actual_nodal_energy(source):
    before=encoded(dict(config=source.config,state=source.state))
    g=ThermalGeologicalContinuation.from_source(source,experimental_nodal_retirement=True,fracture_edges=boundary(source))
    assert encoded(g.source)==before
    assert encoded(dict(config=source.config,state=source.state))==before
    np.testing.assert_array_equal(g.record['thermal_inventory']['mass_kg'],source.config['mass'])
    np.testing.assert_array_equal(g.record['thermal_inventory']['energy_j'],source.state['energy'])
    v=np.array(source.state['nodal_velocity']);e=g.record['mapping']['energy']
    assert e['source_kinetic_j']==pytest.approx(.5*v@material_mass(source)@v)
    assert e['signed_active_mechanical_model_change_j']==pytest.approx(e['mapped_kinetic_proxy_j']-e['source_kinetic_j']-e['retired_elastic_j']-e['source_passive_reserve_j'])
    assert e['mapping_heat_j']==0 and not e['total_mechanical_energy_conserved']


def test_default_open_seam_and_diffuse_guards(source):
    assert not eligibility(source)['eligible']
    for mask,reason in [(None,'CERTIFIED_FRACTURE'),(np.ones(len(source.mesh.shared_edges),bool),'DIFFUSE_NETWORK')]:
        with pytest.raises(ValueError,match=reason):
            ThermalGeologicalContinuation.from_source(source,experimental_nodal_retirement=True,fracture_edges=mask)
    mask=boundary(source);mask[np.flatnonzero(mask)[0]]=False
    with pytest.raises(ValueError,match='NOT_SEPARATING'):
        ThermalGeologicalContinuation.from_source(source,experimental_nodal_retirement=True,fracture_edges=mask)


def test_newborn_geometry_and_clock_guards_retained(source):
    for mutate,reason in [(lambda m:m.state.update(born_material=[{}]),'newborn'),
                          (lambda m:m.state['rotation'][0][0].__setitem__(0,.9),'Finite domain rotations'),
                          (lambda m:m.state['orbit'].update(time_myr=0.),'clocks differ')]:
        m=deepcopy(source);mutate(m)
        result=eligibility(m,experimental_nodal_retirement=True,partition_owner=(m.mesh.centroids[:,2]>0).astype(int))
        assert not result['eligible'] and any(reason in x for x in result['reasons'])


def test_experimental_incubation_preserves_activation_guards(source):
    from analysis.genesis_geological_handoff.nodal_retirement_check import OPTION
    source.config['experimental_geological_edge_fatigue']=OPTION
    source.state['membrane_established']=False
    with pytest.raises(ValueError,match='ENABLED_ESTABLISHED_SHELL'):update(source,source.state,25.)
    source.state['membrane_established']=True
    previous=deepcopy(source.state);previous['experimental_edge_fatigue']={'damage':[float('nan')]}
    with pytest.raises(ValueError,match='INVALID_STATE'):update(source,previous,25.)


def test_pending_handoff_prevents_another_nodal_step(source):
    source.state['experimental_edge_fatigue']={'partition_owner':[0]}
    before=encoded(source.state)
    with pytest.raises(ValueError,match='HANDOFF_PENDING'):source.advance(25.)
    assert encoded(source.state)==before


def test_rigid_nodal_velocity_is_recovered_without_an_added_kick(source):
    from tectonics.simplified_genesis_nodal import operators
    from tectonics.experimental_nodal_handoff import prepare_nodal_retirement
    omega=np.array([2e-15,-3e-15,1e-15])
    velocity=np.cross(omega,operators(source)['X'])
    source.state['nodal_velocity']=np.concatenate([velocity,velocity],axis=1).ravel().tolist()
    mapped=prepare_nodal_retirement(source,boundary(source))
    np.testing.assert_allclose(mapped['omega_rad_s'],np.tile(omega,(2,1)),rtol=2e-14,atol=1e-29)
    assert max(mapped['fit_velocity_rms_error_m_s'])<1e-22
