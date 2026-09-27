"""Geometric cuts, shared DOFs and surviving material bonds are distinct."""
from copy import deepcopy
from dataclasses import fields

import numpy as np
import pytest

from tectonics.genesis_contact_growth import aggregate_cohorts, append_cohorts, empty_cohorts
from tectonics.genesis_contact_law import ContactLawParameters
from tectonics.genesis_seam_diagnostics import cohort_bonded_traces, seam_connectivity
from tectonics.genesis_seams import split_mesh
from tectonics.mesh import build_icosphere


@pytest.fixture(scope="module")
def mesh():
    return build_icosphere(1)


def _patch(mesh):
    patch = set(mesh.shared_edges[0][:2])
    cuts = [(u, v) for a, b, u, v in mesh.shared_edges if (a in patch) != (b in patch)]
    return split_mesh(mesh, cuts), patch


def test_closed_two_face_patch_is_not_detached_while_contacts_are_bonded(mesh):
    topology, patch = _patch(mesh)
    report = seam_connectivity(topology, np.ones(2*topology.seam_count, dtype=bool))
    assert report["cut_component_count"] == 2
    assert report["two_cell_component_count"] == 1
    assert report["cohesive_component_count"] == 1
    assert report["bonded_seam_count"] == topology.seam_count
    assert report["fully_decohered_seam_count"] == 0
    assert report["shared_vertex_link_count"] == report["shared_vertex_trace_count"] == 0
    expected = 1-mesh.areas_unit_sphere[list(patch)].sum()/mesh.areas_unit_sphere.sum()
    assert report["largest_cut_component_area_fraction"] == pytest.approx(expected)


def test_closed_patch_disconnects_after_every_cohesive_bond_is_lost(mesh):
    topology, _ = _patch(mesh)
    report = seam_connectivity(topology, np.zeros(2*topology.seam_count, dtype=bool))
    assert report["cut_component_count"] == report["cohesive_component_count"] == 2
    assert report["bonded_seam_count"] == 0
    assert report["fully_decohered_seam_count"] == topology.seam_count


@pytest.mark.parametrize("endpoint", [0, 1])
def test_a_single_remaining_endpoint_bond_keeps_the_patch_connected(mesh, endpoint):
    topology, _ = _patch(mesh)
    damage = np.ones(2*topology.seam_count)
    damage[endpoint] = np.nextafter(1., 0.)
    report = seam_connectivity(topology, damage < 1.)
    assert report["cohesive_component_count"] == 1
    assert report["bonded_seam_count"] == 1
    assert report["fully_decohered_seam_count"] == topology.seam_count-1


def test_uncut_mesh_accepts_empty_boolean_mask(mesh):
    report = seam_connectivity(split_mesh(mesh, []), np.empty(0, dtype=bool))
    assert report == {
        "cut_component_count": 1, "two_cell_component_count": 0,
        "largest_cut_component_area_fraction": 1., "cohesive_component_count": 1,
        "shared_vertex_link_count": 0, "shared_vertex_trace_count": 0,
        "bonded_seam_count": 0, "fully_decohered_seam_count": 0,
    }


@pytest.mark.parametrize("segments,shared_traces", [(1, 2), (2, 2)])
def test_open_cracks_retain_shared_tip_dofs_even_without_bonds(mesh, segments, shared_traces):
    a, b, c = map(int, mesh.faces[0])
    topology = split_mesh(mesh, [(a, b), (b, c)][:segments])
    report = seam_connectivity(topology, np.zeros(2*segments, dtype=bool))
    assert report["cut_component_count"] == report["cohesive_component_count"] == 1
    assert report["shared_vertex_link_count"] == segments
    assert report["shared_vertex_trace_count"] == shared_traces
    assert report["fully_decohered_seam_count"] == segments


def test_coincident_but_separate_vertices_do_not_join_broken_faces(mesh):
    topology = split_mesh(mesh, [edge[2:] for edge in mesh.shared_edges])
    report = seam_connectivity(topology, np.zeros(2*topology.seam_count, dtype=bool))
    assert report["cut_component_count"] == report["cohesive_component_count"] == mesh.cell_count
    assert report["shared_vertex_link_count"] == 0
    mask = np.zeros(2*topology.seam_count, dtype=bool)
    mask[1] = True
    assert seam_connectivity(topology, mask)["cohesive_component_count"] == mesh.cell_count-1


def test_connectivity_does_not_mutate_topology_or_mask(mesh):
    topology, _ = _patch(mesh)
    before = deepcopy(topology)
    mask = np.ones(2*topology.seam_count, dtype=bool)
    seam_connectivity(topology, mask)
    for name in ("parent_vertex", "original_faces", "cut_edges", "seam_faces", "bank_vertices"):
        np.testing.assert_array_equal(getattr(topology, name), getattr(before, name))
    for name in ("vertices", "faces", "areas_unit_sphere", "centroids"):
        np.testing.assert_array_equal(getattr(topology.mesh, name), getattr(before.mesh, name))
    assert topology.intact_shared_edges == before.intact_shared_edges
    np.testing.assert_array_equal(mask, True)


@pytest.mark.parametrize("kind", ["integer", "float", "short", "matrix", "scalar"])
def test_invalid_trace_masks_are_rejected(mesh, kind):
    topology, _ = _patch(mesh)
    count = 2*topology.seam_count
    mask = {"integer": np.ones(count, dtype=int), "float": np.ones(count),
            "short": np.ones(count-1, dtype=bool), "matrix": np.ones((count//2, 2), dtype=bool),
            "scalar": True}[kind]
    with pytest.raises(ValueError, match="Bonded trace mask"):
        seam_connectivity(topology, mask)


def _cohorts():
    return append_cohorts(empty_cohorts(), np.array([0, 0, 1]), np.zeros(3), np.ones(3),
                          np.array([2., 2e-18, 2.]), np.zeros(3), np.zeros(3), 1.4,
                          ContactLawParameters(), 1e-9)


def test_tiny_weak_cohort_is_still_a_bond_when_averaged_damage_rounds_to_one():
    cohorts = _cohorts()
    cohorts.damage[:] = [1., np.nextafter(1., 0.), 1.]
    before = deepcopy(cohorts)
    assert aggregate_cohorts(cohorts, 2)["interface_damage"][0] == 1.
    mask = cohort_bonded_traces(cohorts, 2)
    np.testing.assert_array_equal(mask, [True, False])
    for field in fields(cohorts):
        np.testing.assert_array_equal(getattr(cohorts, field.name), getattr(before, field.name))
    mask[:] = False
    assert cohort_bonded_traces(cohorts, 2)[0]


def test_unbonded_cohort_cannot_supply_cohesion_and_empty_traces_are_false():
    cohorts = _cohorts()
    cohorts.bonded[:] = False
    np.testing.assert_array_equal(cohort_bonded_traces(cohorts, 3), False)
    np.testing.assert_array_equal(cohort_bonded_traces(empty_cohorts(), 3), False)
    assert cohort_bonded_traces(empty_cohorts(), 0).shape == (0,)


@pytest.mark.parametrize("kind", ["index_type", "index_range", "bonded_type", "damage_nan", "damage_range", "damage_shape"])
def test_invalid_cohort_connectivity_fields_are_rejected(kind):
    cohorts = _cohorts()
    if kind == "index_type": cohorts.trace_index = cohorts.trace_index.astype(float)
    if kind == "index_range": cohorts.trace_index[0] = 2
    if kind == "bonded_type": cohorts.bonded = cohorts.bonded.astype(int)
    if kind == "damage_nan": cohorts.damage[0] = np.nan
    if kind == "damage_range": cohorts.damage[0] = 1.01
    if kind == "damage_shape": cohorts.damage = cohorts.damage[:-1]
    with pytest.raises(ValueError):
        cohort_bonded_traces(cohorts, 2)


@pytest.mark.parametrize("ntraces", [-1, 1., True])
def test_invalid_trace_counts_are_rejected(ntraces):
    with pytest.raises(ValueError):
        cohort_bonded_traces(empty_cohorts(), ntraces)
