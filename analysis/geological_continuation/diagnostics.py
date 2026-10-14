"""Identical observables for current and unmodified historical kernels."""
import numpy as np
from tectonics.mesh import connected_components
from tectonics.dynamics import angular_velocity_vectors

def snapshot(mesh,state,system,topo,water,radius):
    a=mesh.physical_cell_areas_km2(radius);owner=state.cell_plate
    velocity=np.cross(angular_velocity_vectors(system)[owner],mesh.centroids*radius)
    speed=np.linalg.norm(velocity,axis=1);vol=state.continental_volume_km3
    return dict(time_myr=float(state.time_myr),plates=len(system.plates),
        mean_speed_km_per_myr=float(np.average(speed,weights=a)),moving_area_fraction=float(a[speed>1].sum()/a.sum()),
        largest_plate_fraction=float(np.bincount(owner,weights=a).max()/a.sum()),
        owner_components=sum(len(connected_components(np.flatnonzero(owner==p),mesh.neighbors)) for p in range(len(system.plates))),
        continental_volume_km3=float(vol.sum()),continental_area_fraction=float(np.average(state.continental_fraction,weights=a)),
        land_area_fraction=float(water.land_area_fraction),sea_level_m=float(water.sea_level_m),
        water_volume_km3=float(water.water_volume_km3),water_relative_volume_error=float(water.relative_volume_error),
        mean_floor_m=float(np.average(topo.elevation_m,weights=a)),min_floor_m=float(min(topo.elevation_m)),max_floor_m=float(max(topo.elevation_m)))
