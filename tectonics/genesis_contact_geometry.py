"""Reference geometry of paired contact traces, independent of a checkpoint.

The supplied closed material mesh and its ``SeamTopology`` define endpoint
identity and bank orientation. This helper only assembles geometry and the
linear displacement-jump operator; it does not create constitutive history,
choose cuts, prescribe water, or assign traction to a newly inserted surface.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class InterfaceGeometry:
    """Two quadrature traces per cut, in seam then endpoint order.

    Rows of ``jump_operator`` alternate normal opening and tangential jump.
    Columns are two tangent displacement components per split vertex followed
    by the common radial displacement; that radial column is identically zero.
    Each trace area is half the edge length times the smaller solid bank depth.
    Forces from ``J.T @ (area * traction)`` are conjugate to displacement in m.
    """

    edge_length_m: np.ndarray
    interface_area_m2: np.ndarray
    interface_normal: np.ndarray
    interface_tangent: np.ndarray
    jump_operator: sparse.csr_matrix
    contact_rows: np.ndarray
    contact_columns: np.ndarray


def build_interface_geometry(original_mesh, topology, radius_m, depth_m, membrane):
    """Assemble paired traces on any supplied closed reference subdivision.

    ``depth_m`` is positive solid thickness per face of ``original_mesh``.
    ``membrane`` supplies the split-vertex tangent basis and standard DOF count.
    For an enriched displacement space with split displacement ``q = Q u``,
    the owner composes the returned operator as ``J @ Q``. No bulk stiffness,
    mechanical forcing, or history is inferred from that composition.
    """
    if (isinstance(radius_m, bool) or not isinstance(radius_m, Real)
            or not np.isfinite(radius_m) or radius_m <= 0):
        raise ValueError("Contact reference radius must be finite and positive")
    depth_m = np.asarray(depth_m, dtype=float)
    if (depth_m.shape != (original_mesh.cell_count,)
            or not np.isfinite(depth_m).all() or np.any(depth_m <= 0)):
        raise ValueError("Contact solid depth must be finite and positive per reference face")
    if (not np.array_equal(topology.original_faces, original_mesh.faces)
            or not np.array_equal(topology.mesh.vertices,
                                  original_mesh.vertices[topology.parent_vertex])):
        raise ValueError("Contact topology does not match its supplied reference mesh")
    basis = np.asarray(membrane.vertex_basis)
    if (basis.shape != (topology.mesh.vertex_count, 3, 2)
            or not np.isfinite(basis).all()
            or membrane.ndof != 2*topology.mesh.vertex_count+1):
        raise ValueError("Contact membrane does not use the split reference vertex DOFs")

    topo = topology
    xyz = original_mesh.vertices[topo.cut_edges]
    normal = np.cross(xyz[:, 0], xyz[:, 1])
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    direction = original_mesh.centroids[topo.seam_faces[:, 1]]-original_mesh.centroids[topo.seam_faces[:, 0]]
    normal *= np.where(np.einsum("fi,fi->f", normal, direction) >= 0, 1., -1.)[:, None]
    edge_length_m = np.arctan2(np.linalg.norm(np.cross(xyz[:, 0], xyz[:, 1]), axis=1),
                               np.einsum("fi,fi->f", xyz[:, 0], xyz[:, 1]))*radius_m
    interface_area_m2 = np.repeat(edge_length_m*np.min(depth_m[topo.seam_faces], axis=1)/2, 2)
    interface_normal = np.repeat(normal, 2, axis=0)
    tangent = np.cross(normal[:, None, :], xyz)
    interface_tangent = tangent.reshape(-1, 3)
    row, column, values = [], [], []
    for seam, banks in enumerate(topo.bank_vertices):
        for endpoint in range(2):
            point = 2*seam+endpoint
            for component, axis in enumerate((normal[seam], tangent[seam, endpoint])):
                for bank, sign in ((0, -1), (1, 1)):
                    vertex = banks[bank, endpoint]
                    projection = axis@basis[vertex]
                    for local in range(2):
                        row.append(2*point+component)
                        column.append(2*vertex+local)
                        values.append(sign*projection[local])
    jump_operator = sparse.coo_matrix((values, (row, column)),
        shape=(4*len(topo.cut_edges), membrane.ndof)).tocsr()
    jump_operator.eliminate_zeros()
    block_dofs = np.arange(2*len(interface_area_m2)).reshape(-1, 2)
    contact_rows = np.repeat(block_dofs, 2, axis=1).ravel()
    contact_columns = np.tile(block_dofs, (1, 2)).ravel()
    return InterfaceGeometry(edge_length_m, interface_area_m2, interface_normal,
        interface_tangent, jump_operator, contact_rows, contact_columns)
