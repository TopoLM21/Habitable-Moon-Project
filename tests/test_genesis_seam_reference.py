"""A smooth unlocalized field must not be mistaken for physical fragment birth."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from analysis.genesis_seam_reference import (
    cut_graph_summary, evaluate_reference, reference_axes, uniform_cap_state,
)
from tectonics.genesis_contact import ContactParameters, select_seams
from tectonics.genesis_material import rebuild_material_mesh
from tectonics.genesis_seams import split_mesh
from tectonics.mesh import build_icosphere, connected_components


def test_uniform_field_exposes_two_family_aliasing_under_refinement():
    """The pathology exists without physics, localization, or loading changes."""
    counts = []
    for subdivision in (2, 3, 4):
        mesh = build_icosphere(subdivision)
        overlap, state, _ = evaluate_reference(mesh, 0.)
        single_family, _, _ = evaluate_reference(mesh, 30.)
        assert np.ptp(state.damage) == 0
        assert not np.any(state.cumulative_shear)
        assert single_family["component_count"] == 1
        assert overlap["component_count"] == overlap["two_cell_component_count"] + 1
        counts.append(overlap["two_cell_component_count"])
    assert 0 < counts[0] < counts[1] < counts[2]


@pytest.mark.parametrize("angle", [0., 30.])
def test_graph_counter_agrees_with_real_displacement_topology(angle):
    mesh = build_icosphere(3)
    report, _, cuts = evaluate_reference(mesh, angle)
    topology = split_mesh(mesh, cuts)
    actual = connected_components(range(mesh.cell_count), topology.mesh.neighbors)
    assert report["component_sizes"] == sorted(map(len, actual), reverse=True)
    np.testing.assert_array_equal(topology.mesh.areas_unit_sphere, mesh.areas_unit_sphere)


def test_reference_field_is_objective_under_joint_rigid_rotation():
    mesh = build_icosphere(3)
    rotation = Rotation.from_rotvec([.31, -.62, .27]).as_matrix()
    moved = rebuild_material_mesh(mesh, mesh.vertices @ rotation.T)
    center, tangent = reference_axes()
    original = uniform_cap_state(mesh, 7.)
    rotated = uniform_cap_state(moved, 7., center=rotation @ center, tangent=rotation @ tangent)
    np.testing.assert_array_equal(original.fault_active, rotated.fault_active)
    np.testing.assert_allclose(original.plane_normal, rotated.plane_normal, atol=1e-14)
    np.testing.assert_array_equal(select_seams(mesh, original, ContactParameters()),
                                  select_seams(moved, rotated, ContactParameters()))


@pytest.mark.parametrize("disable", ["activation", "damage"])
def test_no_eligible_weak_planes_gives_no_cuts(disable):
    mesh = build_icosphere(3)
    state = uniform_cap_state(mesh, 0.)
    if disable == "activation":
        state.fault_active[:] = False
    else:
        state.damage[:] = 0.
    cuts = select_seams(mesh, state, ContactParameters())
    report = cut_graph_summary(mesh, cuts)
    assert report["cut_count"] == 0
    assert report["component_count"] == 1
    assert report["cut_length_unit_sphere"] == 0.


def test_stricter_angle_demonstrates_gaps_not_a_localization_solution():
    mesh = build_icosphere(3)
    state = uniform_cap_state(mesh, 0.)
    # At the bisector neither family is eligible. Removing the diamonds this
    # way supplies no missing crack location or continuous propagation rule.
    cuts = select_seams(mesh, state, replace(ContactParameters(), alignment_degrees=20.))
    assert len(cuts) == 0
