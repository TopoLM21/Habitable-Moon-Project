import numpy as np

from tectonics.late_tectonics import LateTectonicsParameters, advance_late_tectonics
from tectonics.lithosphere import CrustType, initialize_lithosphere
from tectonics.mesh import build_icosphere
from tectonics.plates import random_plate_system


def test_giant_plate_accumulates_intraplate_stress():
    mesh=build_icosphere(2)
    system=random_plate_system(mesh,1,42,0.0,0.1,0.1)
    state=initialize_lithosphere(mesh,system,continental_fraction=0.0,continental_nuclei=1)
    d=advance_late_tectonics(mesh,state,system,[],20.0,5287.0,0.7,LateTectonicsParameters(),last_split_time_myr=0.0)
    assert d.largest_plate_fraction > 0.999
    assert np.mean(state.intraplate_stress) > 0.0


def test_supercontinent_lid_builds_heat_memory():
    mesh=build_icosphere(2)
    system=random_plate_system(mesh,1,7,0.0,0.1,0.1)
    state=initialize_lithosphere(mesh,system,continental_fraction=0.5,continental_nuclei=1)
    advance_late_tectonics(mesh,state,system,[],40.0,5287.0,0.8,LateTectonicsParameters(),last_split_time_myr=0.0)
    cont=state.crust_type==int(CrustType.CONTINENTAL)
    assert np.mean(state.supercontinent_heat[cont]) > 0.0
    assert np.mean(state.supercontinent_heat[~cont]) == 0.0


def test_old_ocean_gets_extra_failure_stress():
    mesh=build_icosphere(2)
    system=random_plate_system(mesh,2,9,0.1,0.1,0.2)
    state=initialize_lithosphere(mesh,system,continental_fraction=0.0,continental_nuclei=1)
    state.crust_age_myr[:]=500.0
    advance_late_tectonics(mesh,state,system,[],20.0,5287.0,0.6,LateTectonicsParameters(),last_split_time_myr=0.0)
    assert np.mean(state.intraplate_stress) > 0.0
