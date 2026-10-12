"""Saved-state topology adapter, not a native StarterState or evolution model."""
from pathlib import Path
import json
import numpy as np
from tectonics.mesh import SphereMesh,_normalize,_spherical_triangle_area,_build_topology
from tectonics.plates import Plate,PlateSystem
from tectonics.genesis_starter_topology import select_starter_cut
from tectonics.genesis_transfer_bridge import encoded,digest


def eligible(history,parameters):
    """Source persistent weak-plane readiness; no stress-only instant fracture."""
    d=np.asarray(history['damage'],float);age=np.asarray(history['fault_candidate_age_myr'],float);active=np.asarray(history['fault_active'])
    if d.ndim!=1 or age.shape!=d.shape or active.shape!=d.shape or active.dtype.kind!='b':raise ValueError('FAULT_SHAPES')
    if not np.isfinite(d).all() or not np.isfinite(age).all() or np.any((d<0)|(d>1)) or np.any(age<0):raise ValueError('FAULT_VALUES')
    if not parameters['enabled'] or not history['membrane_established']:return np.zeros_like(active)
    return active|((d>=parameters['activation_damage'])&(age>=parameters['activation_persistence_myr']))


def mesh_view(vertices,faces):
    # Spherical directional topology for the existing selector; volume geometry
    # and nodal motion are retained verbatim separately, not replaced by this view.
    X=_normalize(np.asarray(vertices,float));F=np.asarray(faces,int);tri=X[F]
    area=np.array([_spherical_triangle_area(*t) for t in tri]);neighbors,edges=_build_topology(F)
    return SphereMesh(X,F,_normalize(tri.sum(axis=1)),area,neighbors,edges)


def select(payload):
    g=payload['geometry'];s=payload['state'];params=payload['selector_parameters']
    mesh=mesh_view(np.array(g['vertices_m'])+np.array(s['displacement_m']).reshape(-1,6)[:,:3],g['faces'])
    ids=list(dict.fromkeys(g['owner_ids_per_face']));owner=np.array([ids.index(i) for i in g['owner_ids_per_face']],np.int32)
    # PlateSystem is solely an ownership view. These zero Euler placeholders
    # are NEVER used to replace the saved nonrigid nodal velocities/momentum.
    system=PlateSystem(owner,tuple(Plate(i,int(np.flatnonzero(owner==i)[0]),np.array([0.,0.,1.]),0.) for i in range(len(ids))))
    mask=eligible(s['fault_history'],s['source_parameters']['faults'])
    cut=select_starter_cut(mesh,system,mask,np.array(s['fault_history']['damage']),**params)
    return dict(status='NO_CANDIDATE' if cut is None else 'CANDIDATE_REQUIRES_PHYSICAL_CUT_MODEL',eligible_faces=np.flatnonzero(mask).tolist(),cut=None if cut is None else cut.tolist(),owner_ids=ids,time_myr=s['time_myr'],physical_evolution_admitted=False,native_starter_advance_supported=False)


def save(path,payload):
    body=encoded(payload);envelope=dict(format='genesis-starter-selector-v1',payload_sha256=digest(body),payload=payload)
    with Path(path).open('xb') as f:f.write(encoded(envelope))


def load(path):
    obj=json.loads(Path(path).read_bytes())
    if obj['format']!='genesis-starter-selector-v1' or digest(encoded(obj['payload']))!=obj['payload_sha256']:raise ValueError('RESTART_CHECKSUM')
    p=obj['payload']
    if p['state']['time_myr']!=p['state']['fault_history']['time_myr']:raise ValueError('RESTART_CLOCK')
    return p
