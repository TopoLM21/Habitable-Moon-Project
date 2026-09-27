"""Energy, material and admissible-space controls for frozen shell extension."""
from hashlib import sha256

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from analysis.genesis_shell_release_fixture import build_shell_release_fixture
from tectonics.genesis_contact import ContactModel
from tectonics.genesis_shell_release import FrozenShell
from test_genesis_contact import _source_bytes


def _shell(fixture, *, amplitude=None, strain=None, **kwargs):
    traction = fixture.traction_xyz_pa if amplitude is None else fixture.traction(
        fixture.mesh.vertices, amplitude_pa=amplitude)
    return FrozenShell.from_uniform(fixture.mesh, fixture.radius_m,
        fixture.depth_m, 6e10, .25, traction, elastic_strain=strain, **kwargs)


@pytest.fixture(scope="module")
def fixture():
    return build_shell_release_fixture(2)


@pytest.fixture(scope="module")
def loaded(fixture):
    model = _shell(fixture)
    return model, model.compare_extension(fixture.seed_cuts, fixture.trial_cuts)


def test_unloaded_extension_has_no_available_energy(fixture):
    result = _shell(fixture, amplitude=0).compare_extension(fixture.seed_cuts, fixture.trial_cuts)
    assert result.release_j == result.mean_release_j_m2 == 0
    assert result.added_area_m2 > 0
    assert result.admissible_open_crack
    np.testing.assert_array_equal(result.after.displacement_m, 0)


def test_same_topology_cannot_release_energy(fixture, loaded):
    result = loaded[0].compare_extension(fixture.seed_cuts, fixture.seed_cuts)
    assert result.release_j == result.added_area_m2 == result.mean_release_j_m2 == 0
    np.testing.assert_array_equal(result.before.displacement_m, result.after.displacement_m)


def test_single_edge_has_no_independent_banks_or_release(fixture, loaded):
    result = loaded[0].compare_extension(fixture.cuts(0), fixture.cuts(1))
    assert result.added_area_m2 > 0
    assert result.after.membrane.ndof == result.before.membrane.ndof
    assert result.release_j == 0
    np.testing.assert_array_equal(result.after.bulk_matrix.toarray(), result.before.bulk_matrix.toarray())


def test_dead_load_release_requires_external_potential_work(fixture, loaded):
    _, result = loaded
    assert result.release_j > 0
    assert result.admissible_open_crack
    # With no eigenstrain and fixed forces, a more compliant shell stores MORE
    # elastic energy. Stored energy alone would report the wrong release sign.
    delta_stored = result.after.stored_energy_j-result.before.stored_energy_j
    delta_work = result.after.external_potential_work_j-result.before.external_potential_work_j
    assert delta_stored > 0
    assert delta_work == pytest.approx(2*delta_stored, rel=2e-11)
    assert result.release_j == pytest.approx(delta_work-delta_stored, rel=2e-11)
    assert result.release_j == pytest.approx(result.potential_difference_j, rel=2e-11)
    assert result.release_j == pytest.approx(result.relaxation_energy_j, rel=2e-11)
    assert result.added_area_m2 == pytest.approx(
        (fixture.length_m(3)-fixture.length_m(2))*fixture.depth_m, rel=1e-14)


@pytest.mark.parametrize("factor", [0.25, 2., 7.])
def test_release_scales_quadratically_with_dead_load(factor, fixture, loaded):
    result = _shell(fixture, amplitude=factor).compare_extension(fixture.seed_cuts, fixture.trial_cuts)
    assert result.release_j == pytest.approx(loaded[1].release_j*factor**2, rel=3e-11)
    np.testing.assert_allclose(result.after.displacement_m,
        loaded[1].after.displacement_m*factor, rtol=2e-10, atol=1e-12)


def test_existing_bank_displacements_prolong_by_material_corners(fixture):
    strain = np.zeros((fixture.mesh.cell_count, 3))
    strain[:, 0] = 1e-7*fixture.mesh.centroids[:, 0]
    strain[:, 2] = 1e-7*fixture.mesh.centroids[:, 1]
    model = _shell(fixture, strain=strain)
    result = model.compare_extension(fixture.seed_cuts, fixture.trial_cuts)
    old, new = result.before, result.after
    assert old.topology.mesh.vertex_count > model.mesh.vertex_count
    assert np.max(np.abs(old.normal_gap_m))+np.max(np.abs(old.tangential_jump_m)) > 0
    p = model._prolongation(old, new)
    lifted = p@old.displacement_m
    np.testing.assert_array_equal(lifted[new.membrane.dofs], old.displacement_m[old.membrane.dofs])
    np.testing.assert_allclose((p.T@new.bulk_matrix@p).toarray(), old.bulk_matrix.toarray(), rtol=2e-14, atol=2.)
    np.testing.assert_allclose(p.T@new.external_force, old.external_force, rtol=2e-14, atol=1e-4)
    np.testing.assert_allclose(p.T@new.initial_bulk_force, old.initial_bulk_force, rtol=2e-14, atol=.05)
    np.testing.assert_allclose((new.constraints@p).toarray(), old.constraints.toarray(), rtol=2e-14, atol=1e-16)
    old_strain = np.einsum("fai,fi->fa", old.membrane.b, old.displacement_m[old.membrane.dofs])
    lifted_strain = np.einsum("fai,fi->fa", new.membrane.b, lifted[new.membrane.dofs])
    np.testing.assert_array_equal(lifted_strain, old_strain)
    for name in ("force_pullback_relative_error", "prestress_pullback_relative_error",
                 "stiffness_pullback_relative_error", "gauge_pullback_relative_error"):
        assert getattr(result, name) < 1e-14
    assert result.material_area_relative_error == result.material_volume_relative_error == 0


def test_uniform_prestress_relaxes_by_common_radial_contraction(fixture):
    expansion = 2e-5
    strain = np.tile([expansion, expansion, 0.], (fixture.mesh.cell_count, 1))
    model = _shell(fixture, amplitude=0, strain=strain)
    result = model.compare_extension(fixture.seed_cuts, fixture.trial_cuts)
    for state in (result.before, result.after):
        assert state.displacement_m[-1] == pytest.approx(-expansion*fixture.radius_m, rel=2e-13)
        assert np.max(np.abs(state.displacement_m[:-1])) < 1e-10
        assert state.stored_energy_j < model.initial_energy_j*1e-25
    assert abs(result.release_j) < model.initial_energy_j*1e-24


def test_healing_and_free_fragments_are_refused(fixture, loaded):
    with pytest.raises(ValueError, match="contain every existing cut"):
        loaded[0].compare_extension(fixture.trial_cuts, fixture.seed_cuts)
    triangle = fixture.mesh.faces[0]
    loop = np.column_stack((triangle, np.roll(triangle, -1)))
    with pytest.raises(ValueError, match="Detached material"):
        loaded[0].solve(loop)


def test_compressive_release_is_rejected_for_interpenetration(fixture, loaded):
    result = _shell(fixture, amplitude=-1).compare_extension(fixture.seed_cuts, fixture.trial_cuts)
    assert result.release_j == pytest.approx(loaded[1].release_j, rel=3e-11)
    assert result.after.min_gap_m < -1e-6
    assert not result.admissible_open_crack
    assert "free_banks_interpenetrate" in result.rejection_reasons


def test_non_tangent_forces_and_unbalanced_torque_are_refused(fixture):
    with pytest.raises(ValueError, match="tangent"):
        FrozenShell.from_uniform(fixture.mesh, fixture.radius_m, fixture.depth_m,
            6e10, .25, fixture.mesh.vertices)
    torque_traction = np.cross([0., 0., 1.], fixture.mesh.vertices)
    model = FrozenShell.from_uniform(fixture.mesh, fixture.radius_m, fixture.depth_m,
        6e10, .25, torque_traction)
    with pytest.raises(ValueError, match="rigid-rotation load"):
        model.solve(fixture.seed_cuts)


def _balanced_prestress_shell(fixture, amplitude, *, torque_fraction=0.):
    strain = amplitude*np.random.default_rng(17).normal(size=(fixture.mesh.cell_count, 3))
    source = _shell(fixture, amplitude=0., strain=strain)
    equilibrium = source.solve([])
    force = equilibrium.initial_bulk_force
    world = np.einsum("vij,vj->vi", equilibrium.membrane.vertex_basis,
                      force[:-1].reshape(-1, 2))
    torque = np.cross([0., 0., 1.], fixture.mesh.vertices)
    world += torque_fraction*np.linalg.norm(world)/np.linalg.norm(torque)*torque
    return FrozenShell(fixture.mesh, fixture.radius_m, source.depth_m,
        source.elasticity_pa, strain, world, force[-1])


@pytest.mark.parametrize("amplitude", [1e-12, 1e-7, 1e-3])
def test_balanced_prestress_is_not_rejected_for_cancellation_torque(fixture, amplitude):
    # The external reaction exactly supports inherited stress. Cartesian /
    # tangent force conversion leaves a tiny residual with arbitrary direction;
    # its direction alone must not require an artificial rotational support.
    model = _balanced_prestress_shell(fixture, amplitude)
    state = model.solve([])
    force_scale = np.linalg.norm(state.initial_bulk_force)
    assert np.linalg.norm(state.external_force-state.initial_bulk_force)/force_scale < 1e-14
    assert state.equilibrium_residual < 1e-13
    assert state.max_added_strain < amplitude*1e-12
    assert state.admissible_open_crack


def test_resolved_torque_on_nearly_balanced_prestress_is_still_refused(fixture):
    model = _balanced_prestress_shell(fixture, 1e-7, torque_fraction=1e-5)
    with pytest.raises(ValueError, match="rigid-rotation load"):
        model.solve([])


@pytest.mark.parametrize("bad_tensor", ["asymmetric", "indefinite", "zero", "nan"])
def test_invalid_elasticity_is_refused(bad_tensor, fixture, loaded):
    source = loaded[0]
    tensor = source.elasticity_pa.copy()
    if bad_tensor == "asymmetric":
        tensor[0, 0, 1] *= 2
    elif bad_tensor == "indefinite":
        tensor[0, 0, 0] = -1
    elif bad_tensor == "zero":
        tensor[0] = 0
    else:
        tensor[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        FrozenShell(fixture.mesh, fixture.radius_m, source.depth_m, tensor,
            source.elastic_strain, source.vertex_force_xyz_n)


def test_source_arrays_are_owned_and_cannot_be_made_writeable(fixture, loaded):
    source = loaded[0]
    arrays = [source.depth_m.copy(), source.elasticity_pa.copy(), source.elastic_strain.copy(),
              source.vertex_force_xyz_n.copy()]
    model = FrozenShell(fixture.mesh, fixture.radius_m, *arrays)
    fingerprint = model.fingerprint
    reference = model.compare_extension(fixture.seed_cuts, fixture.trial_cuts)
    for array in arrays:
        array[...] = 42
    repeat = model.compare_extension(fixture.seed_cuts, fixture.trial_cuts)
    assert model.fingerprint == fingerprint
    assert repeat.release_j == reference.release_j
    for array in (model.depth_m, model.elasticity_pa, model.elastic_strain,
                  model.vertex_force_xyz_n, model.reference_volume_m3, model.mesh.vertices,
                  model.mesh.faces, model.mesh.centroids, model.mesh.areas_unit_sphere):
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_cut_order_and_endpoint_order_do_not_change_result(fixture, loaded):
    model, reference = loaded
    reordered = model.compare_extension(fixture.seed_cuts[::-1, ::-1], fixture.trial_cuts[::-1, ::-1])
    assert reordered.release_j == reference.release_j
    np.testing.assert_array_equal(reordered.after.displacement_m, reference.after.displacement_m)
    np.testing.assert_array_equal(reordered.after.topology.cut_edges, reference.after.topology.cut_edges)


def test_rigid_rotation_of_mesh_and_load_preserves_energy_and_world_motion(fixture, loaded):
    rotation = Rotation.from_rotvec([.31, -.52, .17]).as_matrix()
    rotated_fixture = build_shell_release_fixture(2, rotation=rotation)
    rotated = _shell(rotated_fixture).compare_extension(rotated_fixture.seed_cuts, rotated_fixture.trial_cuts)
    reference = loaded[1]
    assert rotated.release_j == pytest.approx(reference.release_j, rel=1e-10)
    assert rotated.added_area_m2 == pytest.approx(reference.added_area_m2, rel=1e-14)
    for a, b in ((reference.before, rotated.before), (reference.after, rotated.after)):
        def world(state):
            return np.einsum("vij,vj->vi", state.membrane.vertex_basis,
                state.displacement_m[:-1].reshape(-1, 2))
        np.testing.assert_allclose(world(b), world(a)@rotation.T, rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(b.normal_gap_m, a.normal_gap_m, rtol=1e-9, atol=1e-11)


def test_fault_adapter_preserves_checkpoint_and_existing_contact_bulk_assembly(tmp_path, monkeypatch):
    data, source_model, _, path = _source_bytes(tmp_path)
    production_before = ContactModel(data)
    expected_cuts = production_before.topology.cut_edges.copy()
    expected_hash = sha256(data).hexdigest()
    # The new adapter must not call or alter the historical seam selector.
    import tectonics.genesis_contact as contact_module
    def forbidden_selector(*args, **kwargs):
        raise AssertionError("Diagnostic adapter called the legacy seam selector")
    with monkeypatch.context() as patch:
        patch.setattr(contact_module, "select_seams", forbidden_selector)
        frozen = FrozenShell.from_fault_checkpoint(path)
    triangle = source_model.mesh.faces[0]
    cuts = np.array([[triangle[0], triangle[1]], [triangle[1], triangle[2]]])
    inherited = ContactModel(data, reference_cuts=cuts)
    diagnostic = frozen.solve(cuts)
    np.testing.assert_array_equal(diagnostic.bulk_matrix.toarray(), inherited.bulk_matrix.toarray())
    np.testing.assert_array_equal(diagnostic.initial_bulk_force, inherited.initial_bulk_force)
    np.testing.assert_allclose(diagnostic.external_force, inherited.external_force, rtol=2e-14, atol=.01)
    # ContactModel's nested sum and this oracle's fused einsum use different
    # summation orders even though every material entry is identical.
    assert frozen.initial_energy_j == pytest.approx(inherited.initial_elastic_energy_j, rel=2e-14)
    assert frozen.source_hash == inherited.source_hash == expected_hash
    assert path.read_bytes() == data
    production_after = ContactModel(data)
    np.testing.assert_array_equal(production_after.topology.cut_edges, expected_cuts)
    np.testing.assert_array_equal(production_after.bulk_matrix.toarray(), production_before.bulk_matrix.toarray())
    np.testing.assert_array_equal(production_after.source_state.elastic_strain,
        production_before.source_state.elastic_strain)
    assert production_after.source_hash == expected_hash
