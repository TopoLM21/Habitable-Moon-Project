"""Read saved physical positions independently; no evolution or source mutation."""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation
from tectonics.simplified_genesis import SimplifiedGenesis
from tectonics.simplified_genesis_edges import swept_mesh_certificate
from tectonics.genesis_material import rebuild_material_mesh
from tectonics.genesis_transfer_bridge import encoded,digest
from tectonics.spherical_polygons import intersect_convex,polygon_area


def main():
    root=Path('/workspace/simplified-genesis-edges-01')
    first=SimplifiedGenesis.load(root/'edge_step_1.json');second=SimplifiedGenesis.load(root/'edge_step_2.json')
    pinned=json.loads(Path(__file__).with_name('edge_strips_result.json').read_text())
    for name,sha in pinned['artifacts'].items():assert digest((root/name).read_bytes())==sha
    a=np.asarray(first.state['edge_material']['vertices']);b=np.asarray(second.state['edge_material']['vertices'])
    observed=[]
    dt=.01*365.25*86400
    for vertices in first.config['edge_strips']['core_vertices']:
        u,_,vt=np.linalg.svd(a[vertices].T@b[vertices]);R=vt.T@u.T
        observed.append(Rotation.from_matrix(R).as_rotvec()/dt)
    physical=float(np.linalg.norm(np.diff(observed,axis=0)))
    declared=second.state['edge_ledger']['relative_core_omega_rad_s']
    assert physical>1e-15 and abs(physical/declared-1)<1e-5
    oldmesh=rebuild_material_mesh(first.mesh,a)
    delta=b/np.sum(a*b,axis=1)[:,None]-a
    path=swept_mesh_certificate(oldmesh,delta)
    worst=0.;pairs=0
    for fa,fb,_,_ in first.mesh.shared_edges:
        if first.state['owner'][fa]==first.state['owner'][fb]:continue
        polygon=intersect_convex(b[first.mesh.faces[fa]],b[first.mesh.faces[fb]])
        area=polygon_area(polygon)*first.config['radius_km']**2 if len(polygon) else 0.
        worst=max(worst,area);pairs+=1
    assert worst==0
    L=second.state['edge_ledger']
    scale=dt*np.linalg.norm(np.sum(second.config['face_torque'],axis=0))
    result=dict(status='SAVED_PHYSICAL_GEOMETRY_CHECKED',hashes_verified=True,neighbor_pairs=pairs,
        max_neighbor_overlap_km2=worst,relative_omega_from_fitted_positions_rad_s=physical,
        relative_omega_from_solver_rad_s=declared,whole_interval_from_saved_positions=path,
        angular_defect_relative_to_external_impulse=float(np.linalg.norm(L['angular_momentum_defect'])/scale),
        angular_defect_relative_to_net_momentum=L['angular_momentum_relative'],new_steps=0)
    with Path(__file__).with_name('edge_strips_geometry_readback.json').open('xb') as f:f.write(encoded(result))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
