"""Independent force, finite-motion and continuum-limit checks for tied shells."""
import numpy as np
import pytest
from scipy import sparse
from scipy.linalg import eigh
from scipy.spatial.transform import Rotation

from tectonics.genesis_material import rebuild_material_mesh
from tectonics.genesis_crack_path import ReferenceCrackPath
from tectonics.genesis_moving_path_diagnostics import observe_material_path
from tectonics.genesis_moving_support import MaterialPathSupport
from tectonics.genesis_moving_tied import MovingTiedLoading, MovingTiedMechanics
from tectonics.genesis_path_mesh import insert_crack_path
from tectonics.genesis_shell import Membrane, maxwell_factors
from tectonics.mesh import build_icosphere


YEAR = 365.25*86400.


def _fixture(mesh=None):
    mesh = build_icosphere(1) if mesh is None else mesh
    model = MovingTiedMechanics(mesh, 5.3e6)
    volume = mesh.areas_unit_sphere*model.initial_radius_m**2*1e4
    young = np.full(mesh.cell_count, 60e9)
    return model, volume, young


def _zero_force(mesh, radius, depth, membrane):
    return np.zeros(membrane.ndof)


def _area(mesh, radius):
    result = np.zeros(mesh.vertex_count)
    for face, unit_area in zip(mesh.faces, mesh.areas_unit_sphere):
        result[face] += unit_area*radius**2/3
    return result


def _directional_force(axis, amplitude):
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)

    def callback(mesh, radius, depth, membrane):
        world = amplitude*(axis-mesh.vertices*(mesh.vertices@axis)[:, None])
        nodal = np.einsum("vij,vi->vj", membrane.vertex_basis, world)
        return np.r_[(nodal*_area(mesh, radius)[:, None]).ravel(), 0.]
    return callback


@pytest.mark.parametrize("thermal_strain,beta", [(4e-4, .6), (-3e-4, .9)])
def test_uniform_free_thermal_radius_has_exact_finite_logarithmic_solution(thermal_strain, beta):
    model, volume, young = _fixture()
    # Heterogeneous material weight cannot change the exactly stress-free
    # solution of uniform thermal strain on a free closed sphere.
    volume *= np.linspace(.8, 1.2, len(volume))
    young *= np.linspace(.7, 1.3, len(young))
    initial = model.initial(np.zeros((len(volume), 3)))
    memory = np.zeros_like(initial.elastic_strain)
    memory[:, :2] = thermal_strain
    result = model.trial(initial, MovingTiedLoading(200., volume, young,
        memory, np.full(len(volume), beta), _zero_force))
    expected_radius = initial.radius_m*np.exp(-thermal_strain/beta)
    assert result.radius_m == pytest.approx(expected_radius, rel=3e-13)
    np.testing.assert_allclose(result.elastic_strain, 0., rtol=0, atol=3e-12)
    np.testing.assert_allclose(result.vertices, initial.vertices, rtol=0, atol=3e-12)
    np.testing.assert_array_equal(result.last_volume_m3, volume)
    # Constant density means thickness must compensate the changed area.
    new_mesh = model.mesh_for(result)
    depth = volume/(new_mesh.areas_unit_sphere*result.radius_m**2)
    old_depth = volume/(model.template_mesh.areas_unit_sphere*initial.radius_m**2)
    np.testing.assert_allclose(depth/old_depth, np.exp(2*thermal_strain/beta),
        rtol=2e-12, atol=0)


def test_arrival_geometry_force_balance_and_drag_are_independently_reconstructed():
    model, volume, young = _fixture()
    memory = np.random.default_rng(791).normal(size=(len(volume), 3))*2e-5
    initial = model.initial(memory)
    dt = 2000.
    external = _directional_force([.2, -.4, .7], 20000.)
    result = model.trial(initial, MovingTiedLoading(dt, volume, young,
        memory, np.full(len(volume), .8), external))
    mesh = model.mesh_for(result)
    membrane = Membrane(mesh, model.poisson_ratio)
    stress = (result.elastic_strain@membrane.d.T)*young[:, None]
    internal = np.zeros(membrane.ndof)
    for face in range(mesh.cell_count):
        internal[membrane.dofs[face]] += (
            membrane.b[face].T@stress[face])*volume[face]/result.radius_m
    # Obtain the geodesic arrival tangent directly from the two world meshes,
    # without consulting last_increment_current_m or the production helper.
    earlier, now = initial.vertices, result.vertices
    dot = np.clip(np.einsum("vi,vi->v", earlier, now), -1., 1.)
    cross = np.linalg.norm(np.cross(earlier, now), axis=1)
    angle = np.arctan2(cross, dot)
    tangent = dot[:, None]*now-earlier
    distance = np.linalg.norm(tangent, axis=1)
    direction = np.divide(tangent, distance[:, None], out=np.zeros_like(tangent),
        where=distance[:, None] > 0)
    velocity = direction*(result.radius_m*angle/(dt*YEAR))[:, None]
    world_drag = velocity*(_area(mesh, result.radius_m)
        *model.parameters.basal_drag_pa_s_m)[:, None]
    drag = np.r_[np.einsum("vij,vi->vj", membrane.vertex_basis, world_drag).ravel(), 0.]
    force = external(mesh, result.radius_m,
        volume/(mesh.areas_unit_sphere*result.radius_m**2), membrane)
    scale = max(np.linalg.norm(internal), np.linalg.norm(force))
    assert np.linalg.norm(internal+drag-force)/scale < 2e-8
    assert np.linalg.norm(internal-force)/scale > .1
    np.testing.assert_allclose(result.last_drag_force_n, drag, rtol=3e-8, atol=1e3)
    assert result.last_drag_force_n[-1] == 0.
    step_drag_work = np.sum(world_drag*velocity)*(dt*YEAR)
    assert result.drag_work_j == pytest.approx(step_drag_work, rel=3e-8)
    assert result.drag_work_j > 0


def test_loaded_motion_is_covariant_under_a_finite_world_rotation():
    model, volume, young = _fixture()
    rotation = Rotation.from_rotvec(np.array([.3, -.4, .2])).as_matrix()
    rotated_mesh = rebuild_material_mesh(model.template_mesh,
        model.template_mesh.vertices@rotation.T)
    rotated_model, rotated_volume, rotated_young = _fixture(rotated_mesh)
    # Face frames follow the material first edge, so their tensor components
    # remain identical under a common three-dimensional world rotation.
    memory = np.random.default_rng(404).normal(size=(len(volume), 3))*3e-5
    beta = np.full(len(volume), .7)
    axis = np.array([.2, -.4, .7])
    result = model.trial(model.initial(memory), MovingTiedLoading(2000., volume,
        young, memory, beta, _directional_force(axis, 17000.)))
    rotated = rotated_model.trial(rotated_model.initial(memory), MovingTiedLoading(
        2000., rotated_volume, rotated_young, memory, beta,
        _directional_force(rotation@axis, 17000.)))
    np.testing.assert_allclose(rotated.vertices, result.vertices@rotation.T,
        rtol=0, atol=2e-11)
    assert rotated.radius_m == pytest.approx(result.radius_m, rel=2e-12)
    np.testing.assert_allclose(rotated.elastic_strain, result.elastic_strain,
        rtol=2e-7, atol=3e-13)
    assert rotated.drag_work_j == pytest.approx(result.drag_work_j, rel=2e-7)
    assert rotated.bulk_work_j == pytest.approx(result.bulk_work_j, rel=2e-7)
    first_membrane = Membrane(model.mesh_for(result), model.poisson_ratio)
    second_membrane = Membrane(rotated_model.mesh_for(rotated), model.poisson_ratio)
    first_drag = np.einsum("vij,vj->vi", first_membrane.vertex_basis,
        result.last_drag_force_n[:-1].reshape(-1, 2))
    second_drag = np.einsum("vij,vj->vi", second_membrane.vertex_basis,
        rotated.last_drag_force_n[:-1].reshape(-1, 2))
    np.testing.assert_allclose(second_drag, first_drag@rotation.T, rtol=1e-6, atol=5e5)


def test_small_motion_approaches_independent_maxwell_drag_continuum_mode():
    model, volume, young = _fixture()
    mesh, radius = model.template_mesh, model.initial_radius_m
    membrane = Membrane(mesh, model.poisson_ratio)
    local = np.einsum("fai,ab,fbj,f->fij", membrane.b, 60e9*membrane.d,
        membrane.b, volume/radius**2)
    stiffness = sparse.coo_matrix((local.ravel(), (membrane.rr, membrane.cc)),
        shape=(membrane.ndof, membrane.ndof)).toarray()
    drag = np.repeat(_area(mesh, radius), 2)*model.parameters.basal_drag_pa_s_m
    effective = stiffness[:-1, :-1]-np.outer(stiffness[:-1, -1],
        stiffness[-1, :-1])/stiffness[-1, -1]
    rates, vectors = eigh(effective, np.diag(drag))
    selected = np.flatnonzero(rates > rates[-1]*1e-6)[-8]
    mode = np.r_[vectors[:, selected],
        -stiffness[-1, :-1]@vectors[:, selected]/stiffness[-1, -1]]
    initial_strain = np.einsum("fai,fi->fa", membrane.b, mode[membrane.dofs])/radius
    initial_strain *= 1e-7/np.max(np.abs(initial_strain))
    duration, eta = 100., 1e21
    exact = np.exp(-(60e9/eta+rates[selected])*duration*YEAR)
    errors = []
    for dt in (20., 10., 5.):
        state = model.initial(initial_strain)
        r, beta = maxwell_factors(dt*YEAR, np.full(len(volume), eta/60e9))
        for _ in range(round(duration/dt)):
            state = model.trial(state, MovingTiedLoading(dt, volume, young,
                r[:, None]*state.elastic_strain, beta, _zero_force))
        amplitude = np.sum(state.elastic_strain*initial_strain)/np.sum(initial_strain**2)
        np.testing.assert_allclose(state.elastic_strain, amplitude*initial_strain,
            rtol=0, atol=2e-12)
        errors.append(abs(amplitude-exact))
    assert errors[-1] < 7e-5
    assert all(.4 < finer/coarser < .6 for coarser, finer in zip(errors, errors[1:]))


def test_current_path_recovered_tractions_close_virtual_work_with_moving_bulk_and_drag():
    model, volume, young = _fixture()
    mesh = model.template_mesh
    points = np.array([[1., .12, .23], [1., .42, .49], [1., .72, .53]])
    points /= np.linalg.norm(points, axis=1)[:, None]
    path = ReferenceCrackPath(points, model.initial_radius_m/1000.)
    insertion = insert_crack_path(mesh, path)
    support = MaterialPathSupport(mesh, insertion, model.initial_radius_m, .25)
    memory = np.random.default_rng(428).normal(size=(len(volume), 3))*2e-5
    forcing = _directional_force([.2, -.4, .7], 20000.)
    state = model.trial(model.initial(memory), MovingTiedLoading(2000., volume,
        young, memory, np.full(len(volume), .8), forcing))
    before = {name: value.copy() for name, value in vars(state).items()
        if isinstance(value, np.ndarray)}
    observed = observe_material_path(model, state, support,
        np.zeros(mesh.cell_count), forcing)
    basis = observed.model.basis
    child_volume = support.extensive(volume)
    # Reassemble bulk virtual force directly from stress and the strain
    # operator; no bulk() or diagnostic internal-force routine is used here.
    internal = basis.strain_operator.T@(
        child_volume[:, None]*observed.stress_pa).ravel()/state.radius_m
    fine_mesh = basis.subdivision.mesh
    fine_membrane = Membrane(fine_mesh, .25)
    fine_force = forcing(fine_mesh, state.radius_m,
        child_volume/(fine_mesh.areas_unit_sphere*state.radius_m**2), fine_membrane)
    ancestors = basis.topology.parent_vertex
    split_area = _area(basis.topology.mesh, state.radius_m)
    unsplit_area = _area(fine_mesh, state.radius_m)
    split_force = np.r_[(fine_force[:-1].reshape(-1, 2)[ancestors]
        *(split_area/unsplit_area[ancestors])[:, None]).ravel(), fine_force[-1]]
    external = np.r_[state.last_external_force_n,
        np.asarray(basis.W.T@split_force).ravel()]
    background = basis.parent_displacement_operator@state.last_increment_current_m
    drag_fine = np.r_[np.repeat(split_area, 2), 0.]*background
    drag_fine *= model.parameters.basal_drag_pa_s_m/(state.last_step_years*YEAR)
    drag = np.r_[state.last_drag_force_n, np.asarray(basis.W.T@drag_fine).ravel()]
    traction = observed.recovery.traction_pa
    interface = observed.model.jump_operator.T@(
        observed.model.geometry.interface_area_m2[:, None]*traction).ravel()
    residual = internal+drag+interface-external
    scale = max(np.linalg.norm(internal), np.linalg.norm(drag), np.linalg.norm(external))
    assert np.linalg.norm(residual)/scale < 2e-8
    rng = np.random.default_rng(615)
    virtual = rng.normal(size=basis.ndof)
    assert abs(residual@virtual)/(scale*np.linalg.norm(virtual)) < 2e-8
    assert np.linalg.norm(internal+interface-external)/scale > .1
    assert observed.parent_force_relative_error < 2e-14
    assert observed.parent_energy_relative_error < 2e-14
    np.testing.assert_array_equal(observed.state.displacement_m, 0.)
    assert observed.state.active_interval is None
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(state, name), value)
