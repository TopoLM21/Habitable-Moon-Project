import numpy as np
from tectonics.mesh import build_icosphere,connected_components
from tectonics.transport import _optimal_assignment,SubgridTransportParameters,rotate_by_quaternion,quaternion_from_axis_angle,_median_cell_spacing_rad

def test_connected_raster_projection_is_local_bijective_and_deterministic():
    mesh=build_icosphere(2);source=np.flatnonzero(mesh.centroids[:,0]>.2)
    q=quaternion_from_axis_angle(np.array([.2,.5,.7])/np.linalg.norm([.2,.5,.7]),.24)
    desired=rotate_by_quaternion(mesh.centroids[source],q)
    p=SubgridTransportParameters(connected_footprint=True)
    target=_optimal_assignment(mesh,desired,p)
    np.testing.assert_array_equal(target,_optimal_assignment(mesh,desired,p))
    assert len(np.unique(target))==len(source)
    assert len(connected_components(target,mesh.neighbors))==1
    assert np.max(np.arccos(np.clip(np.sum(desired*mesh.centroids[target],axis=1),-1,1)))<=1.5*_median_cell_spacing_rad(mesh)
    np.testing.assert_allclose(desired@desired.T,mesh.centroids[source]@mesh.centroids[source].T,atol=1e-14)

def test_roundoff_coverage_repair_does_not_delete_volume_or_hide_real_overflow():
    from tectonics.lithosphere import _redistribute_continental_footprint_overflow
    mesh=build_icosphere(0);a=np.ones(20);fraction=np.zeros(20);fraction[0]=np.nextafter(1.,2.)
    volume=fraction*26.;before=volume.copy()
    _redistribute_continental_footprint_overflow(mesh,a,fraction,volume,np.array([],dtype=int))
    assert fraction[0]==1.;np.testing.assert_array_equal(volume,before)
    fraction[:]=1.;fraction[0]=1.1;volume=fraction*26.
    _redistribute_continental_footprint_overflow(mesh,a,fraction,volume,np.array([],dtype=int))
    assert fraction.max()>1.01  # real global capacity exhaustion stays visible
