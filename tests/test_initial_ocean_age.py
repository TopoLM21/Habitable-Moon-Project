import numpy as np
from tectonics.mesh import build_icosphere
from tectonics.plates import random_plate_system
from tectonics.lithosphere import CrustType, initialize_lithosphere, initialize_oceanic_crust_ages, boundary_records_for_state


def test_initial_ocean_age_is_mature_and_ridges_are_young():
    mesh=build_icosphere(2)
    ps=random_plate_system(mesh,8,11,.25,.15,.65)
    state=initialize_lithosphere(mesh,ps,.28,5,7.,35.,500.,radius_km=5287.)
    boundaries=boundary_records_for_state(mesh,state,ps,5287.,4.,1.)
    initialize_oceanic_crust_ages(mesh,state,boundaries,5287.,spreading_rate_km_per_myr=30.,max_age_myr=160.,unseeded_age_myr=120.)
    ocean=state.crust_type==int(CrustType.OCEANIC)
    assert np.any(state.crust_age_myr[ocean] > 20.0)
    assert np.max(state.crust_age_myr[ocean]) <= 160.0+1e-12
    # If the random state has oceanic divergent ridge cells, at least one must
    # be initialized exactly young.
    ridge=[]
    for b in boundaries:
        if str(b.boundary_type).endswith('DIVERGENT'):
            ridge += [i for i in (b.face_a,b.face_b) if ocean[i]]
    if ridge:
        assert np.min(state.crust_age_myr[np.asarray(ridge,int)]) == 0.0
