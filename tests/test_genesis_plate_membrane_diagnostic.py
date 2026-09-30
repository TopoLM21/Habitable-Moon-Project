"""Independent free domains, SI work, and rigid-mode separation for diagnostics."""
from dataclasses import replace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tectonics.basal_coupling import mantle_source_pattern
from tectonics.genesis_plate_membrane_diagnostic import solve_plate_basal_membrane
from tectonics.mesh import build_icosphere


@pytest.fixture(scope="module")
def mesh():
    return build_icosphere(2)


def pattern(mesh):
    return 1000.*mantle_source_pattern(mesh,19)[mesh.faces]


def solve(mesh,owner,load,**options):
    return solve_plate_basal_membrane(mesh,owner,load,np.full(mesh.cell_count,35.),
        radius_km=5287.,**options)


def test_zero_and_rigid_slip_produce_no_deforming_stress(mesh):
    owner=(mesh.centroids@np.array([1.,2.,3.])>0).astype(int)
    zero=solve(mesh,owner,np.zeros((mesh.cell_count,3,3)))
    assert not np.any(zero.stress_pa)
    assert zero.elastic_energy_j == 0.
    axes=np.array([[1300.,-410.,850.],[-700.,1600.,230.]])
    rigid=np.cross(axes[owner,None,:],mesh.vertices[mesh.faces])
    result=solve(mesh,owner,rigid)
    assert np.max(np.abs(result.stress_pa)) < 1e-8
    assert len(result.components) == 2
    for row in result.components:
        assert row['retained_load_relative_norm'] < 1e-14
        assert np.linalg.norm(row['raw_basal_torque_nm']) > 1e22


def test_rigid_torque_addition_does_not_change_nonrigid_stress(mesh):
    owner=(mesh.centroids[:,0]>0).astype(int)
    base=solve(mesh,owner,pattern(mesh))
    axes=np.array([[1300.,-410.,850.],[-700.,1600.,230.]])
    rigid=np.cross(axes[owner,None,:],mesh.vertices[mesh.faces])
    altered=solve(mesh,owner,pattern(mesh)+rigid)
    np.testing.assert_allclose(altered.stress_pa,base.stress_pa,atol=1e-8,rtol=1e-12)
    np.testing.assert_allclose(altered.face_vertex_displacement_m,base.face_vertex_displacement_m,atol=1e-10,rtol=1e-12)


def test_boundary_vertices_are_not_welded_between_plates(mesh):
    owner=(mesh.centroids[:,0]>0).astype(int)
    load=pattern(mesh);load[owner==0]=0.
    result=solve(mesh,owner,load)
    np.testing.assert_array_equal(result.stress_pa[owner==0],0.)
    np.testing.assert_array_equal(result.face_vertex_displacement_m[owner==0],0.)
    assert np.max(np.abs(result.stress_pa[owner==1])) > 1e3
    assert sum(r['vertex_count'] for r in result.components) > mesh.vertex_count


def test_virtual_work_and_torque_units_match_independent_surface_quadrature(mesh):
    owner=np.zeros(mesh.cell_count,dtype=int)
    load=pattern(mesh)+np.cross([0.,2300.,0.],mesh.vertices[mesh.faces])
    result=solve(mesh,owner,load)
    assert result.elastic_energy_j > 0.
    assert result.balanced_external_work_j == pytest.approx(2*result.elastic_energy_j,rel=1e-12)
    expected=(5287e3)**3*np.sum(mesh.areas_unit_sphere[:,None,None]
        *np.cross(mesh.vertices[mesh.faces],load)/3.,axis=(0,1))
    assert np.linalg.norm(np.asarray(result.components[0]['raw_basal_torque_nm'])-expected) < 1e-13*np.linalg.norm(expected)
    assert result.components[0]['equilibrium_relative_residual'] < 1e-12
    assert result.components[0]['gauge_residual'] < 1e-15
    assert np.linalg.norm(result.components[0]['balanced_torque_nm']) < 1e-13*np.linalg.norm(expected)


def test_si_modulus_and_thickness_scaling(mesh):
    owner=(mesh.centroids[:,0]>0).astype(int);load=pattern(mesh)
    baseline=solve(mesh,owner,load)
    stiff=solve(mesh,owner,load,young_modulus_pa=12e10)
    np.testing.assert_allclose(stiff.stress_pa,baseline.stress_pa,atol=1e-8,rtol=1e-12)
    np.testing.assert_allclose(stiff.strain,.5*baseline.strain,atol=1e-18,rtol=1e-12)
    thick=solve_plate_basal_membrane(mesh,owner,load,np.full(mesh.cell_count,70.),radius_km=5287.)
    np.testing.assert_allclose(thick.stress_pa,.5*baseline.stress_pa,atol=1e-8,rtol=1e-12)


def test_relabeling_and_cartesian_rotation_preserve_physical_result(mesh):
    owner=(mesh.centroids[:,0]>0).astype(int);load=pattern(mesh)
    result=solve(mesh,owner,load)
    relabeled=solve(mesh,np.where(owner==0,41,8),load)
    np.testing.assert_array_equal(relabeled.stress_pa,result.stress_pa)
    rotation=Rotation.from_rotvec([.2,-.4,.7]).as_matrix()
    rotated=replace(mesh,vertices=mesh.vertices@rotation.T,centroids=mesh.centroids@rotation.T)
    transformed=solve(rotated,owner,load@rotation.T)
    # Face-local axes rotate with the geometry; stress components are unchanged.
    np.testing.assert_allclose(transformed.stress_pa,result.stress_pa,atol=1e-8,rtol=1e-11)
    np.testing.assert_allclose(transformed.face_vertex_displacement_m,
        result.face_vertex_displacement_m@rotation.T,atol=1e-10,rtol=1e-11)


def test_disconnected_support_receives_separate_gauges(mesh):
    owner=np.zeros(mesh.cell_count,dtype=int);load=pattern(mesh)
    active=np.abs(mesh.centroids[:,2])>.6
    load[~active]=0.
    h=np.where(active,35.,0.)
    result=solve_plate_basal_membrane(mesh,owner,load,h,radius_km=5287.)
    assert len(result.components)==2
    assert all(row['plate_id']==0 for row in result.components)
    np.testing.assert_array_equal(result.stress_pa[~active],0.)
    assert all(row['equilibrium_relative_residual'] < 1e-10 for row in result.components)


def test_no_floor_can_make_unsupported_faces_carry_load(mesh):
    with pytest.raises(ValueError,match='zero-thickness'):
        solve_plate_basal_membrane(mesh,np.zeros(mesh.cell_count,dtype=int),pattern(mesh),
            np.zeros(mesh.cell_count),radius_km=5287.)


def test_closed_sphere_solution_converges_to_analytic_degree_two_stress():
    # For f=r^T S r (trace(S)=0), Delta_s f=-6f and
    # div_s Hess_s f=-5 grad_s f. The tangent membrane balance gives
    # sigma=tau*R/[H*(5+nu)]*((1-nu)*Hess_s f + nu*Delta_s f*I).
    tensor=np.diag([1.,-.5,-.5]);nu=.25;errors=[]
    for subdivision in (1,2,3):
        mesh=build_icosphere(subdivision);x=mesh.vertices
        potential=np.einsum('vi,ij,vj->v',x,tensor,x)
        traction=1000.*2.*(x@tensor.T-potential[:,None]*x)
        result=solve(mesh,np.zeros(mesh.cell_count,dtype=int),traction[mesh.faces])
        points=mesh.vertices[mesh.faces]
        e1=points[:,1]-points[:,0];e1/=np.linalg.norm(e1,axis=1)[:,None]
        normal=np.cross(points[:,1]-points[:,0],points[:,2]-points[:,0])
        normal/=np.linalg.norm(normal,axis=1)[:,None]
        e2=np.cross(normal,e1);frame=np.stack([e1,e2],axis=2)
        f=np.einsum('fi,ij,fj->f',normal,tensor,normal)
        hessian=2*(np.einsum('fia,ij,fjb->fab',frame,tensor,frame)-f[:,None,None]*np.eye(2))
        exact_tensor=1000.*5287./(35.*(5.+nu))*((1.-nu)*hessian-6*nu*f[:,None,None]*np.eye(2))
        exact=np.column_stack([exact_tensor[:,0,0],exact_tensor[:,1,1],exact_tensor[:,0,1]])
        difference=result.stress_pa-exact
        norm=lambda value: np.sqrt(np.sum(mesh.areas_unit_sphere[:,None]*value*value*np.array([1,1,2])))
        errors.append(norm(difference)/norm(exact))
    # Constant element stress from linear displacements is first-order in h.
    assert errors[1] < .6*errors[0]
    assert errors[2] < .6*errors[1]
    assert errors[2] < .05
