import numpy as np
from dataclasses import replace
from tectonics.mesh import build_icosphere
from tectonics.simulation import build_prototype, load_config
from tectonics.lithosphere import CrustType, initialize_lithosphere
from tectonics.continental import ContinentalCycleParameters, _gravitational_collapse


def test_gravitational_collapse_conserves_volume():
    cfg=load_config('configs/canonical_moon.yaml')
    cfg['mesh']['subdivisions']=1
    proto=build_prototype(cfg)
    st=initialize_lithosphere(proto.mesh, proto.plates, 0.4, 2, 7.0, 35.0, 300.0)
    cont=np.flatnonzero(st.crust_type==int(CrustType.CONTINENTAL))
    assert len(cont)>2
    src=int(cont[0])
    # Ensure at least one same-plate continental neighbour can receive material.
    same=[int(n) for n in proto.mesh.neighbors[src] if st.crust_type[n]==int(CrustType.CONTINENTAL) and st.cell_plate[n]==st.cell_plate[src]]
    if not same:
        # Make a direct neighbour continental for this unit test.
        dst=int(proto.mesh.neighbors[src][0]); st.crust_type[dst]=int(CrustType.CONTINENTAL); st.cell_plate[dst]=st.cell_plate[src]; st.crust_thickness_km[dst]=30.0
    st.crust_thickness_km[src]=70.0
    areas=proto.mesh.physical_cell_areas_km2(float(cfg['moon']['radius_km']))
    mask=st.crust_type==int(CrustType.CONTINENTAL)
    before=float(np.sum(areas[mask]*st.crust_thickness_km[mask]))
    p=ContinentalCycleParameters(gravitational_collapse_enabled=True, gravitational_collapse_rate_per_myr=1.0)
    moved=_gravitational_collapse(proto.mesh, st, areas, p, 4.0)
    after=float(np.sum(areas[mask]*st.crust_thickness_km[mask]))
    assert moved >= 0.0
    assert abs(after-before) <= max(1e-6, abs(before)*1e-12)
