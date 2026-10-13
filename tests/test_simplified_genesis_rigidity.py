"""Negative mobility controls: these must not be reported as moving plates."""
import numpy as np
from scipy.spatial.transform import Rotation
from analysis.simplified_genesis_validation.synthetic import (
    irregular_three_domain_synthetic,hemisphere_synthetic)
from tectonics.simplified_genesis import SimplifiedGenesis
from tectonics.simplified_genesis_contact import packing_rigidity,boundary_rows
from tectonics.genesis_transfer_bridge import encoded


def test_damage_generated_irregular_three_domains_are_geometrically_locked(tmp_path):
    model=irregular_three_domain_synthetic()
    assert len(model.state['omega'])==1 and max(model.state['damage'])==0
    model.advance(25.);model.advance(25.)
    assert len(model.state['omega'])==3 and len(model.state['events'])==2
    s=model.state;owner=np.array(s['owner']);poses=np.array(s['rotation'])
    proof=packing_rigidity(model.mesh,owner,poses)
    assert proof['common_rotation_only'] and proof['relative_dof']==0
    assert not proof['subdivision_can_unlock'] and len(proof['witnesses'])==2
    # Independent numerical rank cross-check, not the witness-graph algorithm.
    J,_=boundary_rows(model.mesh,owner,poses)
    singular=np.linalg.svd(J,compute_uv=False)
    assert singular[5]>5. and singular[6]<1e-12
    assert min(x['sine_between_planes'] for x in proof['witnesses'])>.99
    assert np.linalg.norm(s['omega'][0])>1e-14  # No global zero-velocity clamp.
    model.save(tmp_path/'three.json');restart=SimplifiedGenesis.load(tmp_path/'three.json')
    model.advance(25.);restart.advance(25.)
    assert encoded(model.state)==encoded(restart.state)
    p=model.state['contact'];L=model.state['last_ledger']
    assert p['accepted_relative_omega_norm']==0 and p['free_relative_omega_norm']>1e-16
    assert p['whole_step']['all_domain_pairs'] and not p['whole_step']['endpoint_sampling']
    assert L['momentum_relative']<1e-10
    assert abs(L['whole_balance_j'])<=L['whole_allowance_j']


def test_subdivision_cannot_produce_a_contact_free_prefix():
    model=irregular_three_domain_synthetic();model.advance(25.);model.advance(25.)
    owner=np.array(model.state['owner']);old=np.array(model.state['omega'])
    I=np.array([model.config['face_inertia']])[0]
    J,_=boundary_rows(model.mesh,owner,np.array(model.state['rotation']))
    # Only one candidate per depth, not 2**depth executed planetary steps.
    # Negative gap velocity on an already touching arc implies penetration
    # arbitrarily close to t=0; safe endpoints cannot change that derivative.
    for depth in [0,4,12,24,40]:
        dt=25*365.25*86400/(2**depth)
        free=np.array([np.linalg.solve(I[owner==k].sum(axis=0)+dt*model.face_drag[owner==k].sum(axis=0),
            I[owner==k].sum(axis=0)@old[k]+dt*model.face_torque[owner==k].sum(axis=0)) for k in range(3)])
        gaps=J@free.ravel()
        assert np.min(gaps)<-1e-20 and np.max(gaps)>1e-20


def test_lock_proof_is_covariant_but_does_not_lock_hemisphere_slide():
    model=irregular_three_domain_synthetic();model.advance(25.);model.advance(25.)
    owner=np.array(model.state['owner'])
    poses=np.repeat(Rotation.from_rotvec([.2,.3,-.1]).as_matrix()[None],3,axis=0)
    assert packing_rigidity(model.mesh,owner,poses)['common_rotation_only']
    h=hemisphere_synthetic();h.advance(25.)
    proof=packing_rigidity(h.mesh,np.array(h.state['owner']),np.array(h.state['rotation']))
    assert not proof['common_rotation_only'] and proof['relative_dof'] is None
    h.advance(25.)
    assert h.state['contact']['accepted_relative_omega_norm']>1e-16
