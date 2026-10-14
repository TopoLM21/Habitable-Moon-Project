"""Inventory, thermal-network and tensor invariants of nested refinement."""
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from tectonics.experimental_angular_coupled import AngularCoupled
from tectonics.genesis_transfer_bridge import encoded
from analysis.genesis_geological_handoff.coupled_refine import refine, rotate_strain
from analysis.genesis_geological_handoff.coupled_spatial_check import energy, field, restrict_fields


@pytest.fixture(scope='module')
def pair():
    p = Path('/workspace/genesis-angular-coupled-01/initial.json')
    if not p.exists():
        pytest.skip('Authenticated source checkpoint required')
    a = AngularCoupled.load(p)
    before = encoded(dict(config=a.model.config, state=a.model.state))
    b, P, audit = refine(a)
    assert encoded(dict(config=a.model.config, state=a.model.state)) == before
    return a, b, P, audit


def test_inventory_and_thermal_network_restrict_exactly(pair):
    a,b,_,_ = pair; c=a.model.config; d=b.model.config; n=len(c['mass'])
    for name in ('mass',):
        np.testing.assert_array_equal(np.asarray(d[name])[:-2].reshape(n-2,4).sum(1), c[name][:-2])
        assert d[name][-2:] == c[name][-2:]
    np.testing.assert_array_equal(np.asarray(b.model.state['energy'])[:-2].reshape(n-2,4).sum(1),a.model.state['energy'][:-2])
    parent = lambda i: int(i)//4 if i<4*(n-2) else n-2+int(i)-4*(n-2)
    old=defaultdict(float); new=defaultdict(float)
    for i,j,g in c['edges']: old[tuple(sorted((int(i),int(j))))]+=g
    for i,j,g in d['edges']: new[tuple(sorted((parent(i),parent(j))))]+=g
    assert old == new
    for catalogue, parent_key, id_key in [('source_material_catalogue','parent_material_id','material_id'),('phase_cells','parent_cell_id','id')]:
        groups=defaultdict(list)
        for row in d[catalogue]: groups[row[parent_key]].append(row)
        assert len({r[id_key] for r in d[catalogue]})==len(d[catalogue])
        for row in c[catalogue]:
            children=groups[row[id_key]]
            assert len(children)==4 and sum(x['mass_kg'] for x in children)==row['mass_kg']
            assert all(x['q_lo']==row['q_lo'] for x in children)
            hi='live_q_hi' if catalogue=='source_material_catalogue' else 'q_hi'
            assert all(x[hi]==row[hi] for x in children)


def test_fe_mass_energy_and_physical_stress(pair):
    a,b,P,audit=pair
    assert audit['mass_matrix_restriction_relative']<1e-12
    assert audit['kinetic_after_j']==pytest.approx(audit['kinetic_before_j'],rel=1e-13)
    assert energy(b.model)==pytest.approx(energy(a.model),rel=1e-13)
    np.testing.assert_allclose(restrict_fields(field(b.model))['stress'],field(a.model)['stress'],rtol=1e-12,atol=1e-7)
    # Every coarse FE translation is reproduced by the refined geometry.
    X=np.asarray(a.model.config['nodal']['X']); Y=np.asarray(b.model.config['nodal']['X'])
    np.testing.assert_allclose(P[::6,::6]@X,Y,rtol=0,atol=0)


def test_engineering_shear_rotation_preserves_tensor_norm():
    theta=.713; Q=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
    e=np.array([1.,2.,3.,4.,5.,6.]); rotated=rotate_strain(e,np.eye(3),Q)
    norm=lambda x: x[:3]@x[:3]+.5*(x[3:]@x[3:])
    assert norm(rotated)==pytest.approx(norm(e),rel=1e-14)
    np.testing.assert_allclose(rotate_strain(rotated,Q,np.eye(3)),e,rtol=1e-14)


def test_refinement_refuses_existing_edge_damage(pair):
    a=deepcopy(pair[0]);a.model.state['experimental_edge_fatigue']['damage'][0]=.01
    with pytest.raises(ValueError,match='ZERO_EDGE_SOURCE'): refine(a)
