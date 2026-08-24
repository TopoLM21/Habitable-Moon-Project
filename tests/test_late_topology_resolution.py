import numpy as np
from tectonics.mesh import build_icosphere
from tectonics.late_tectonics import _dijkstra_path,_path_length_km,_seam_spread_kernel
R=5287.0

def test_physical_dijkstra_path_length_converges_across_resolutions():
    lengths=[]
    for s in (3,4,5):
        m=build_icosphere(s); owner=np.zeros(m.cell_count,dtype=np.int32); pref=np.zeros(m.cell_count)
        start=int(np.argmax(m.centroids[:,2])); goal=int(np.argmin(m.centroids[:,2]))
        path=_dijkstra_path(m,owner,0,start,goal,pref,m.cell_count,R)
        lengths.append(_path_length_km(m,path,R))
    assert max(lengths)-min(lengths) < 1000.0


def test_unresolved_seam_area_integral_is_approximately_resolution_stable():
    integrals=[]
    for s in (3,4,5):
        m=build_icosphere(s); owner=np.zeros(m.cell_count,dtype=np.int32)
        source=int(np.argmax(m.centroids[:,2])); areas=m.physical_cell_areas_km2(R)
        kernel=_seam_spread_kernel(m,owner,{source},R,160.0)
        integrals.append(float(np.sum(kernel*areas)))
    # A single narrow fault is sub-grid at s3/s4; dilution should nevertheless
    # keep its integrated weakening within ~25% across the sweep.
    assert max(integrals)/min(integrals) < 1.25
