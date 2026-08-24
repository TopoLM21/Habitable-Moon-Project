import numpy as np

from tectonics.dynamics import DynamicsParameters, angular_velocity_vectors, update_plate_dynamics
from tectonics.lithosphere import initialize_lithosphere
from tectonics.mesh import build_icosphere
from tectonics.plates import PlateSystem, random_plate_system


def _setup():
    mesh=build_icosphere(2)
    plates=random_plate_system(mesh,6,20260817,0.2,0.15,0.7)
    state=initialize_lithosphere(mesh,plates,continental_fraction=0.25,continental_nuclei=3)
    current=PlateSystem(cell_plate=state.cell_plate.copy(),plates=plates.plates)
    return mesh,plates,state,current


def test_dynamic_update_is_deterministic():
    mesh,base,state,current=_setup(); p=DynamicsParameters()
    a,da,_,_=update_plate_dynamics(mesh,state,current,base,5287.0,2.0,4.0,1.0,p)
    b,db,_,_=update_plate_dynamics(mesh,state,current,base,5287.0,2.0,4.0,1.0,p)
    assert np.allclose(angular_velocity_vectors(a),angular_velocity_vectors(b))
    assert da.mean_speed_deg_per_myr == db.mean_speed_deg_per_myr


def test_dynamic_update_changes_some_angular_velocities():
    mesh,base,state,current=_setup(); p=DynamicsParameters()
    before=angular_velocity_vectors(current)
    after,_,_,_=update_plate_dynamics(mesh,state,current,base,5287.0,2.0,4.0,1.0,p)
    assert np.max(np.linalg.norm(angular_velocity_vectors(after)-before,axis=1)) > 1e-8


def test_speed_cap_is_respected():
    mesh,base,state,current=_setup(); p=DynamicsParameters(force_speed_scale_deg_per_myr=50.0,max_speed_deg_per_myr=0.5,velocity_relaxation_myr=0.01)
    after,_,_,_=update_plate_dynamics(mesh,state,current,base,5287.0,2.0,4.0,1.0,p)
    speeds=np.rad2deg(np.linalg.norm(angular_velocity_vectors(after),axis=1))
    assert np.max(speeds) <= 0.5000001


def test_net_rotation_is_removed_area_weighted():
    mesh,base,state,current=_setup(); p=DynamicsParameters(remove_net_rotation=True,min_active_speed_deg_per_myr=0.0)
    after,_,_,_=update_plate_dynamics(mesh,state,current,base,5287.0,2.0,4.0,1.0,p)
    areas=mesh.physical_cell_areas_km2(5287.0)
    weights=np.bincount(state.cell_plate,weights=areas,minlength=len(after.plates))
    omega=angular_velocity_vectors(after)
    mean=np.sum(omega*weights[:,None],axis=0)/np.sum(weights)
    assert np.linalg.norm(mean) < 1e-12
