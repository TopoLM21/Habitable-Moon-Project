"""Independent guards for the spatial diagnostic's comparison conventions."""
from types import SimpleNamespace
import numpy as np
from tectonics.mesh import build_icosphere
from analysis.genesis_geological_handoff.edge_spatial_audit import metric, restrict, world_stress
from analysis.genesis_geological_handoff.edge_spatial_geometry_audit import local_stress
from analysis.genesis_geological_handoff.edge_weakening_controls import GridControl


def test_restriction_preserves_area_integral_and_constant_tensor():
    coarse, fine = build_icosphere(1), build_icosphere(2)
    np.testing.assert_allclose(fine.areas_unit_sphere.reshape(-1,4).sum(axis=1), coarse.areas_unit_sphere, rtol=1e-13)
    field = np.arange(fine.cell_count, dtype=float)**2
    mapped = restrict(field, fine.areas_unit_sphere)
    np.testing.assert_allclose(coarse.areas_unit_sphere@mapped, fine.areas_unit_sphere@field, rtol=1e-13)
    tensor = np.broadcast_to(np.arange(9).reshape(3,3), (fine.cell_count,3,3))
    answer = metric(tensor[::4], tensor, fine.areas_unit_sphere)
    assert answer['relative_l2'] < 1e-14


def test_tensor_rotation_and_shear_use_stress_not_engineering_strain():
    mesh = build_icosphere(0)
    angle = .37
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0,0,1.]])
    g = SimpleNamespace(mesh=mesh, a=SimpleNamespace(nf=mesh.cell_count), frames=np.tile(rotation,(mesh.cell_count,1,1)))
    tensor = np.array([[2.,1.,.5],[1.,-1.,.3],[.5,.3,-1.]])*1e6
    stress = local_stress(g,tensor)
    np.testing.assert_allclose(world_stress(g,stress), np.broadcast_to(tensor,(mesh.cell_count,3,3)), atol=1e-8)
    expected = []
    for _,_,i,j in mesh.shared_edges:
        normal = np.cross(mesh.vertices[i],mesh.vertices[j]);normal /= np.linalg.norm(normal)
        traction = tensor@normal
        expected.append(np.linalg.norm(traction-normal*np.dot(normal,traction)))
    np.testing.assert_allclose(GridControl.shear(g,stress),expected,rtol=1e-13)
    assert np.max(GridControl.shear(g,local_stress(g,1e6*np.eye(3)))) < 1e-8


def test_metric_denominator_is_restricted_fine_field():
    area = np.array([1.,2.,3.,4., 4.,3.,2.,1.])
    fine = np.array([1.,1.,1.,1., 3.,3.,3.,3.])
    result = metric(np.array([0.,2.]),fine,area)
    assert result['absolute_rms'] == 1.
    np.testing.assert_allclose(result['reference_rms'],np.sqrt(5))
    np.testing.assert_allclose(result['relative_l2'],1/np.sqrt(5))
