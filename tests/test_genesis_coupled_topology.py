"""Additional cracks release connectivity without rewriting material history."""
from copy import deepcopy
from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np
import pytest

from tectonics.genesis_contact import ContactModel
from tectonics.genesis_coupled_topology import transfer_contact_history
from tectonics.genesis_seams import split_mesh
from tectonics.genesis_shell import Membrane
from tectonics.mesh import build_icosphere


@pytest.fixture(scope="module")
def mesh():
    return build_icosphere(1)


def _boundary(mesh, face):
    return [(u, v) for a, b, u, v in mesh.shared_edges if face in (a, b)]


def _geometry(mesh, cuts):
    topology = split_mesh(mesh, cuts)
    geometry = SimpleNamespace(topology=topology, source_hash="same-original-checkpoint",
        radius_m=5e6, membrane=Membrane(topology.mesh, .25),
        depth_m=np.full(mesh.cell_count, 10000.),
        source_state=SimpleNamespace(water_access=np.zeros(mesh.cell_count)))
    ContactModel._build_interfaces(geometry, mesh)
    return geometry


def _state(geometry):
    state = ContactModel.initial(geometry)
    rng = np.random.default_rng(913)
    state.displacement_m[:] = rng.normal(size=len(state.displacement_m))
    for field in fields(state):
        value = getattr(state, field.name)
        if isinstance(value, np.ndarray) and field.name != "displacement_m":
            value[:] = np.arange(value.size).reshape(value.shape)+1
    return replace(state, elapsed_years=200., last_step_years=10., accepted_steps=20,
                   rejected_steps=3, drag_work_j=17., external_work_j=91.)


def _assert_equal(left, right):
    for field in fields(left):
        a, b = getattr(left, field.name), getattr(right, field.name)
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b, err_msg=field.name)
        else:
            assert a == b, field.name


def test_cut_preserves_every_material_corner_position_and_radial_motion(mesh):
    cuts = _boundary(mesh, 30)
    old = _geometry(mesh, cuts)
    new = _geometry(mesh, list(set(cuts+_boundary(mesh, 0))))
    state = _state(old)
    moved = transfer_contact_history(old, new, state)
    old_mesh = ContactModel.mesh_for(old, state)
    new_mesh = ContactModel.mesh_for(new, moved)
    np.testing.assert_array_equal(old_mesh.vertices[old_mesh.faces], new_mesh.vertices[new_mesh.faces])
    np.testing.assert_array_equal(old_mesh.areas_unit_sphere, new_mesh.areas_unit_sphere)
    np.testing.assert_array_equal(old_mesh.centroids, new_mesh.centroids)
    assert moved.displacement_m[-1] == state.displacement_m[-1]
    # A parent ID is ambiguous once a prior cut already has two moving banks.
    duplicates = np.flatnonzero(old.topology.parent_vertex >= 0)[mesh.vertex_count:]
    assert len(duplicates) > 0
    assert any(not np.array_equal(state.displacement_m[2*v:2*v+2],
                                 state.displacement_m[2*old.topology.parent_vertex[v]:2*old.topology.parent_vertex[v]+2])
               for v in duplicates)


def test_existing_endpoints_keep_history_by_edge_identity_and_new_ones_are_zero(mesh):
    cuts = _boundary(mesh, 30)
    old = _geometry(mesh, cuts)
    new = _geometry(mesh, list(set(cuts+_boundary(mesh, 0))))
    state = _state(old)
    moved = transfer_contact_history(old, new, state)
    old_index = {tuple(edge): i for i, edge in enumerate(old.topology.cut_edges)}
    new_jump = (new.jump_operator@moved.displacement_m).reshape(-1, 2)
    old_jump = (old.jump_operator@state.displacement_m).reshape(-1, 2)
    actual = ContactModel._geometric_jump(new, moved, ContactModel.mesh_for(new, moved))
    for index, edge in enumerate(new.topology.cut_edges):
        key = tuple(edge)
        endpoints = slice(2*index, 2*index+2)
        for field in fields(state):
            old_value = getattr(state, field.name)
            if not isinstance(old_value, np.ndarray) or field.name == "displacement_m":
                continue
            expected = old_value[2*old_index[key]:2*old_index[key]+2] if key in old_index else 0.
            np.testing.assert_array_equal(getattr(moved, field.name)[endpoints], expected)
        if key in old_index:
            np.testing.assert_array_equal(new_jump[endpoints], old_jump[2*old_index[key]:2*old_index[key]+2])
        else:
            np.testing.assert_allclose(new_jump[endpoints], 0., atol=2e-15, rtol=0.)
            np.testing.assert_array_equal(actual[endpoints], 0.)
    for name in ("friction_work_cell_j", "viscous_work_cell_j", "fracture_work_cell_j", "shear_remainder_cell_j"):
        assert getattr(moved, name).sum() == getattr(state, name).sum()


def test_transfer_keeps_scalars_and_does_not_alias_or_mutate_source(mesh):
    old = _geometry(mesh, _boundary(mesh, 0))
    new = _geometry(mesh, list(set(_boundary(mesh, 0)+_boundary(mesh, 1))))
    state = _state(old)
    before = deepcopy(state)
    moved = transfer_contact_history(old, new, state)
    for field in fields(state):
        a, b = getattr(state, field.name), getattr(moved, field.name)
        if isinstance(a, np.ndarray):
            assert not np.shares_memory(a, b)
        else:
            assert a == b
    moved.displacement_m[:] = 123.
    moved.fracture_work_cell_j[:] = 456.
    _assert_equal(state, before)


def test_identity_transfer_is_exact_independent_copy(mesh):
    geometry = _geometry(mesh, _boundary(mesh, 0))
    state = _state(geometry)
    moved = transfer_contact_history(geometry, geometry, state)
    _assert_equal(state, moved)
    assert not np.shares_memory(state.displacement_m, moved.displacement_m)


def test_transfer_from_no_contacts_initializes_all_endpoint_history(mesh):
    old = _geometry(mesh, [])
    new = _geometry(mesh, _boundary(mesh, 0))
    state = _state(old)
    moved = transfer_contact_history(old, new, state)
    np.testing.assert_allclose(new.jump_operator@moved.displacement_m, 0., atol=2e-15, rtol=0.)
    for field in fields(moved):
        value = getattr(moved, field.name)
        if isinstance(value, np.ndarray) and field.name != "displacement_m":
            np.testing.assert_array_equal(value, 0.)


def test_contact_row_order_does_not_define_history_identity(mesh):
    old = _geometry(mesh, _boundary(mesh, 0))
    new = _geometry(mesh, _boundary(mesh, 0))
    order = np.arange(new.topology.seam_count)[::-1]
    for name in ("cut_edges", "seam_faces", "bank_vertices"):
        setattr(new.topology, name, getattr(new.topology, name)[order].copy())
    ContactModel._build_interfaces(new, mesh)
    state = _state(old)
    moved = transfer_contact_history(old, new, state)
    np.testing.assert_array_equal(moved.fracture_work_cell_j.reshape(-1, 2),
                                  state.fracture_work_cell_j.reshape(-1, 2)[order])
    np.testing.assert_array_equal(moved.traction_pa.reshape(-1, 2, 2),
                                  state.traction_pa.reshape(-1, 2, 2)[order])


@pytest.mark.parametrize("corruption,match", [
    ("source", "source hash"), ("radius", "reference mesh"),
    ("frame", "coordinates or frames"), ("parent", "corner map"),
    ("orientation", "bank orientation"), ("bank", "material faces"),
])
def test_invalid_reference_or_orientation_is_rejected(mesh, corruption, match):
    old = _geometry(mesh, _boundary(mesh, 0))
    new = _geometry(mesh, _boundary(mesh, 0))
    if corruption == "source":
        new.source_hash = "another-checkpoint"
    elif corruption == "radius":
        new.radius_m += 1.
    elif corruption == "frame":
        new.membrane.vertex_basis[0] *= -1
    elif corruption == "parent":
        new.topology.parent_vertex[0] = 1
    elif corruption == "orientation":
        new.topology.bank_vertices = new.topology.bank_vertices[:, ::-1].copy()
        new.topology.seam_faces = new.topology.seam_faces[:, ::-1].copy()
    elif corruption == "bank":
        new.topology.seam_faces[0, 0] = (int(new.topology.seam_faces[0, 0])+20) % mesh.cell_count
    with pytest.raises(ValueError, match=match):
        transfer_contact_history(old, new, _state(old))


def test_existing_cuts_cannot_be_removed(mesh):
    old = _geometry(mesh, _boundary(mesh, 0))
    new = _geometry(mesh, _boundary(mesh, 0)[:-1])
    with pytest.raises(ValueError, match="remove existing cuts"):
        transfer_contact_history(old, new, _state(old))


@pytest.mark.parametrize("field", ["displacement_m", "plastic_slip_m", "traction_pa", "fracture_work_cell_j"])
@pytest.mark.parametrize("corruption", ["shape", "nan"])
def test_corrupt_state_arrays_cannot_be_transferred(mesh, field, corruption):
    geometry = _geometry(mesh, _boundary(mesh, 0))
    state = _state(geometry)
    if corruption == "shape":
        setattr(state, field, getattr(state, field)[:-1])
    else:
        getattr(state, field).flat[0] = np.nan
    with pytest.raises(ValueError, match="does not match"):
        transfer_contact_history(geometry, geometry, state)
