"""Preserve material motion and contact history when more edges are cut.

This transfer only refines displacement connectivity on an unchanged reference
mesh. It neither advances time nor creates separation, material or mechanical
work. Interface birth areas are owned by the coupled integrator, not inferred
from the current lid thickness here.
"""
from __future__ import annotations

from dataclasses import fields, replace

import numpy as np


def _edge_keys(topology):
    edges = np.asarray(topology.cut_edges)
    if (edges.ndim != 2 or edges.shape[1] != 2
            or not np.issubdtype(edges.dtype, np.integer)
            or np.any(edges[:, 0] >= edges[:, 1])):
        raise ValueError("Contact cuts require ordered original vertex pairs")
    keys = [tuple(int(item) for item in edge) for edge in edges]
    if len(set(keys)) != len(keys):
        raise ValueError("Contact cuts must have unique original edge identities")
    return keys


def _validate_reference(topology):
    mesh = topology.mesh
    faces = np.asarray(mesh.faces)
    parent = np.asarray(topology.parent_vertex)
    original = np.asarray(topology.original_faces)
    if (faces.ndim != 2 or faces.shape[1] != 3
            or not np.issubdtype(faces.dtype, np.integer)
            or parent.shape != (mesh.vertex_count,)
            or not np.issubdtype(parent.dtype, np.integer)
            or np.any(parent < 0) or original.shape != faces.shape
            or np.any(faces < 0) or np.any(faces >= mesh.vertex_count)
            or len(np.unique(faces)) != mesh.vertex_count
            or not np.array_equal(parent[faces], original)):
        raise ValueError("Contact topology has an inconsistent material corner map")
    banks = np.asarray(topology.bank_vertices)
    seam_faces = np.asarray(topology.seam_faces)
    count = len(topology.cut_edges)
    if (banks.shape != (count, 2, 2)
            or not np.issubdtype(banks.dtype, np.integer)
            or np.any(banks < 0) or np.any(banks >= mesh.vertex_count)
            or seam_faces.shape != (count, 2)
            or not np.issubdtype(seam_faces.dtype, np.integer)
            or np.any(seam_faces < 0) or np.any(seam_faces >= mesh.cell_count)
            or not np.array_equal(parent[banks], np.broadcast_to(
                topology.cut_edges[:, None, :], banks.shape))):
        raise ValueError("Contact topology has inconsistent bank identities")
    for bank in range(2):
        for endpoint in range(2):
            belongs = np.any(faces[seam_faces[:, bank]] == banks[:, bank, endpoint, None], axis=1)
            if not np.all(belongs):
                raise ValueError("Contact banks do not belong to their material faces")


def transfer_contact_history(oldgeometry, newgeometry, old_contact_state):
    """Copy a ContactState through a monotone refinement of its cuts.

    Geometries must originate from the same source checkpoint and unchanged
    reference face frames. Every new vertex has exactly one predecessor, found
    through material face corners; original vertex IDs alone are insufficient
    after the first split. Existing two-endpoint histories are matched by their
    original edge IDs. New endpoints start with zero history because the edge
    was still shared immediately before cutting.

    All scalar state fields are retained, including elapsed time and work.
    Every returned array owns new storage. No equilibrium solve is performed:
    the next coupled trial must equilibrate the newly released connectivity.
    """
    source = getattr(oldgeometry, "source_hash", None)
    if not source or source != getattr(newgeometry, "source_hash", None):
        raise ValueError("Contact topology transfer requires the same source hash")
    old, new = oldgeometry.topology, newgeometry.topology
    old_keys, new_keys = _edge_keys(old), _edge_keys(new)
    _validate_reference(old)
    _validate_reference(new)
    if (not np.array_equal(old.original_faces, new.original_faces)
            or old.original_shared_edges != new.original_shared_edges
            or oldgeometry.radius_m != newgeometry.radius_m):
        raise ValueError("Contact topology transfer requires the same material reference mesh")
    new_index = {key: index for index, key in enumerate(new_keys)}
    if not set(old_keys).issubset(new_index):
        raise ValueError("Contact topology transfer cannot remove existing cuts")

    # Repeated assignments are checked: a new node may split an old fan, but
    # must never join nodes that already moved independently.
    predecessor = np.full(new.mesh.vertex_count, -1, dtype=np.int64)
    for new_vertex, old_vertex in zip(new.mesh.faces.ravel(), old.mesh.faces.ravel()):
        previous = predecessor[new_vertex]
        if previous >= 0 and previous != old_vertex:
            raise ValueError("Contact topology transfer cannot merge previous vertex fans")
        predecessor[new_vertex] = old_vertex
    if (np.any(predecessor < 0)
            or not np.array_equal(new.parent_vertex, old.parent_vertex[predecessor])
            or not np.array_equal(new.mesh.vertices, old.mesh.vertices[predecessor])
            or not np.array_equal(newgeometry.membrane.vertex_basis,
                                  oldgeometry.membrane.vertex_basis[predecessor])):
        raise ValueError("Contact topology transfer changed reference vertex coordinates or frames")

    old_index = {key: index for index, key in enumerate(old_keys)}
    old_points, new_points = [], []
    for key, index in new_index.items():
        mapped_banks = predecessor[new.bank_vertices[index]]
        if key in old_index:
            prior = old_index[key]
            if (not np.array_equal(new.seam_faces[index], old.seam_faces[prior])
                    or not np.array_equal(mapped_banks, old.bank_vertices[prior])):
                raise ValueError("Contact topology transfer changed existing bank orientation")
            old_points.extend((2*prior, 2*prior+1))
            new_points.extend((2*index, 2*index+1))
        elif not np.array_equal(mapped_banks[0], mapped_banks[1]):
            raise ValueError("A new contact edge must have shared previous bank vertices")

    displacement = np.asarray(old_contact_state.displacement_m)
    if (displacement.shape != (2*old.mesh.vertex_count+1,)
            or displacement.dtype.kind not in "fiu" or not np.isfinite(displacement).all()):
        raise ValueError("Contact displacement does not match the previous topology")
    moved = np.empty(2*new.mesh.vertex_count+1, dtype=displacement.dtype)
    moved[:-1] = displacement[:-1].reshape(-1, 2)[predecessor].ravel()
    moved[-1] = displacement[-1]
    changes = {"displacement_m": moved}
    for field in fields(old_contact_state):
        value = getattr(old_contact_state, field.name)
        if field.name == "displacement_m" or not isinstance(value, np.ndarray):
            continue
        if (value.ndim == 0 or value.shape[0] != 2*len(old_keys)
                or value.dtype.kind not in "fiu" or not np.isfinite(value).all()):
            raise ValueError(f"Contact history {field.name} does not match the previous endpoints")
        copied = np.zeros((2*len(new_keys),)+value.shape[1:], dtype=value.dtype)
        copied[new_points] = value[old_points]
        changes[field.name] = copied
    return replace(old_contact_state, **changes)
