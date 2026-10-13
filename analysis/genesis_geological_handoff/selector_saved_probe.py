"""Read-only selector-input probe. Never undo or accept the rejected split."""
import argparse
import json
from pathlib import Path
import numpy as np
from tectonics.simplified_genesis import SimplifiedGenesis
from tectonics.genesis_starter_topology import select_starter_cut
from tectonics.plates import Plate,PlateSystem
from analysis.genesis_geological_handoff.real_progression import sha


def probe(source,candidate):
    paths=[Path(source),Path(candidate)];hashes=[sha(p) for p in paths]
    safe,rejected=[SimplifiedGenesis.load(p) for p in paths]
    assert len(safe.state['omega'])==1 and not safe.state['events']
    assert safe.config==rejected.config
    owner=np.array(safe.state['owner']);damage=np.array(rejected.state['damage'])
    age=np.array(rejected.state['candidate_age']);faults=safe.faults
    # Only selector arguments use the pre-split parent and consumption mask.
    # The saved rejected energy, memory and domain state are never reversed.
    eligible=(damage>=faults['activation_damage'])&(age>=faults['activation_persistence_myr'])&~np.array(safe.state['consumed'])
    system=PlateSystem(owner,(Plate(0,0,np.array([0.,0.,1.]),0.),))
    cut=select_starter_cut(safe.mesh,system,eligible,damage,**safe.config['selector'])
    assert cut is not None
    event=rejected.state['events'][-1]
    assert np.array_equal(cut,event['cut'])
    try:safe.validate_cut_localization(owner,damage,cut)
    except ValueError as error:
        assert str(error)=='DIFFUSE_DAMAGE_NOT_A_LOCALIZED_BELT'
    else:raise AssertionError('Rejected real candidate admitted')
    assert hashes==[sha(p) for p in paths]
    return dict(status='PASS',probe_scope='saved damage/age with pre-split parent/consumption selector arguments; no inverse evolution',
        source=str(source),candidate=str(candidate),source_sha256=hashes[0],candidate_sha256=hashes[1],
        raw_cut_matches_saved_event=True,guard_rejected=True,files_unchanged=True,
        all_faces_above_threshold=bool(np.all(damage>=faults['activation_damage'])),
        damage_range=[float(damage.min()),float(damage.max())],cut_faces=len(cut),total_faces=safe.nf)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--candidate',required=True)
    args=p.parse_args();print(json.dumps(probe(args.source,args.candidate),indent=2))
