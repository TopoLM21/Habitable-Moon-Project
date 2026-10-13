"""Bounded synthetic end-to-end hard-contact replay; no real planetary run."""
import json,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from tectonics.simplified_genesis import SimplifiedGenesis
from tectonics.simplified_genesis_contact import placement_certificate,swept_certificate,ContactPathUnsupported
from tectonics.spherical_polygons import intersect_convex,polygon_area
from tectonics.genesis_transfer_bridge import encoded,digest
from .synthetic import synthetic,hemisphere_synthetic

ROOT=Path(__file__).resolve().parents[2];OUT=Path('/workspace/simplified-genesis-hard-contact-01')


def overlap(model,a,b):
    s=model.state;X=model.mesh.vertices;F=model.mesh.faces;owner=np.array(s['owner']);R=np.array(s['rotation'])
    polygon=intersect_convex(X[F[a]]@R[owner[a]].T,X[F[b]]@R[owner[b]].T)
    return (polygon_area(polygon)*model.config['radius_km']**2 if len(polygon) else 0.)


def main():
    start=time.perf_counter();OUT.mkdir(exist_ok=False)
    oldpath=Path('/workspace/simplified-genesis-synthetic-01/final.json');pin=digest(oldpath.read_bytes());old=SimplifiedGenesis.load(oldpath)
    m=synthetic();assert encoded(m.config)==encoded(old.config)
    m.advance(25.);m.advance(25.);m.save(OUT/'corrected_original_synthetic.json')
    oldarea=overlap(old,122,123);newarea=overlap(m,122,123);assert oldarea>29 and newarea==0
    worst=0.;checked=0
    for a,b,_,_ in m.mesh.shared_edges:
        if m.state['owner'][a]==m.state['owner'][b]:continue
        worst=max(worst,overlap(m,a,b));checked+=1
    assert worst==0
    try:placement_certificate(old.mesh,np.array(old.state['owner']),np.array(old.state['rotation']))
    except ContactPathUnsupported:legacy_rejected=True
    else:legacy_rejected=False
    assert legacy_rejected
    slide=hemisphere_synthetic();initial_mass=sum(slide.mass);slide.advance(25.);assert len(slide.state['omega'])==2
    slide.advance(25.);slide.save(OUT/'sliding.json');restored=SimplifiedGenesis.load(OUT/'sliding.json');slide.advance(25.);restored.advance(25.);assert encoded(slide.state)==encoded(restored.state)
    slide.save(OUT/'sliding_restart_endpoint.json');assert sum(slide.mass)==initial_mass
    maxarea=0.
    for a,b,_,_ in slide.mesh.shared_edges:
        if slide.state['owner'][a]!=slide.state['owner'][b]:maxarea=max(maxarea,overlap(slide,a,b))
    assert maxarea==0
    poses=np.repeat(np.eye(3)[None],2,axis=0);omega=np.array([[2*np.pi,0.,0.],[0.,0.,0.]])
    placement_certificate(slide.mesh,np.array(slide.state['owner']),np.array([Rotation.from_rotvec(w).as_matrix() for w in omega]))
    try:swept_certificate(slide.mesh,np.array(slide.state['owner']),poses,omega,1.)
    except ContactPathUnsupported:midstep_rejected=True
    else:midstep_rejected=False
    assert midstep_rejected
    result=dict(status='HARD_CONTACT_SUPPORTED_PATHS_PASS',new_real_intervals=0,source_synthetic_sha256=pin,source_config_unchanged=True,corrected_original=dict(pair=[122,123],old_overlap_km2=oldarea,new_overlap_km2=newarea,neighbor_pairs_checked=checked,max_neighbor_overlap_km2=worst,free_relative_omega=m.state['contact']['free_relative_omega_norm'],accepted_relative_omega=m.state['contact']['accepted_relative_omega_norm'],contact_work_j=m.state['contact']['contact_work_j'],momentum_relative=m.state['last_ledger']['momentum_relative'],certificate=m.state['contact']['whole_step'],invalid_legacy_placement_rejected=legacy_rejected),sliding=dict(initial_owner_count=1,initial_damage=0.,first_split_myr=slide.state['events'][0]['time_myr'],owner_count=len(slide.state['omega']),relative_omega_rad_s=float(np.linalg.norm(np.diff(slide.state['omega'],axis=0))),max_boundary_pair_overlap_km2=maxarea,total_mass_unchanged=True,restart_next_step_exact=True,contact_work_j=slide.state['contact']['contact_work_j'],rotation_balance_j=slide.state['last_ledger']['rotation_balance'],certificate=slide.state['contact']['whole_step']),safe_endpoints_unsafe_middle_rejected=midstep_rejected,scope='filled spherical partition: arbitrary common rotation or two invariant hemispheres; other finite paths rejected',seconds=time.perf_counter()-start,output_bytes=sum(p.stat().st_size for p in OUT.iterdir()),artifacts={p.name:digest(p.read_bytes()) for p in OUT.iterdir()})
    assert digest(oldpath.read_bytes())==pin
    with (ROOT/'analysis/simplified_genesis_validation/hard_contact_result.json').open('xb') as f:f.write(encoded(result))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
