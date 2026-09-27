"""A prescribed diagnostic support carried by a fully tied material shell.

The parent mesh owns the evolution. This helper only transports one existing
subdivision by its original chord barycentric ancestry and rebuilds the tied
observation basis. It does not remesh, advance a state, transport active contact
history, choose a crack, or reset elastic memory.
"""
from __future__ import annotations

from dataclasses import replace
from numbers import Real

import numpy as np

from .genesis_crack_path import ReferenceCrackPath
from .genesis_material import rebuild_material_mesh
from .genesis_path_basis import EmbeddedPathBasis
from .genesis_path_material import _owned
from .mesh import SphereMesh


class MaterialPathSupport:
    """Fixed material ancestry and extensive shares for a moving observation.

    ``basis_at`` uses current geometry to project parent tensors into child
    frames. Its ``basis.subdivision`` therefore contains *current geometric*
    area shares. Material quantities must instead use this owner's
    ``extensive`` method, which always uses the initial shares. Rebuilding an
    observation must not redistribute mass or any other extensive history.

    Initial mesh and ancestry arrays have immutable owned byte storage. Each
    returned basis owns its data and may be discarded after the observation.
    There is deliberately no operation here for continuing an active crack.
    """

    def __init__(self, parent_mesh, insertion, radius_m, poisson_ratio):
        if not isinstance(parent_mesh, SphereMesh):
            raise ValueError("Material support requires a SphereMesh parent")
        # Validate the full subdivision, path, interpolation, radius and elastic
        # parameter contract once through the same basis used for observation.
        initial = EmbeddedPathBasis(parent_mesh, insertion, radius_m, poisson_ratio)
        self.parent_mesh = initial.subdivision.parent_mesh
        self.insertion = replace(initial.insertion, **{
            name: _owned(getattr(initial.insertion, name)) for name in
            ("parent_face", "path_vertex_ids", "path_arclength_m",
             "vertex_parent_face", "vertex_barycentric")})
        self.radius_m = initial.radius_m
        self.poisson_ratio = float(poisson_ratio)
        self.parent_face = initial.subdivision.parent_face
        self.area_fraction = initial.subdivision.area_fraction
        self._largest_child = initial.subdivision._largest_child

    def basis_at(self, parent_mesh, radius_m):
        """Rebuild a tied observation with identical material IDs/connectivity.

        All ordered support vertices become path control points: a moved
        original straight segment need not remain one great-circle arc.
        Arclength is recomputed at the current radius, never used as a new
        material label. Contact intervals/history cannot be moved this way.
        """
        if (isinstance(radius_m, (bool, np.bool_)) or not isinstance(radius_m, Real)
                or not np.isfinite(radius_m) or radius_m <= 0):
            raise ValueError("Current radius_m must be positive and finite")
        old = self.parent_mesh
        if (not isinstance(parent_mesh, SphereMesh)
                or parent_mesh.vertex_count != old.vertex_count
                or not np.array_equal(parent_mesh.faces, old.faces)
                or parent_mesh.neighbors != old.neighbors
                or parent_mesh.shared_edges != old.shared_edges):
            raise ValueError("Moving support requires identical parent material topology")
        # Recompute geometric metadata from the actual vertices. Input meshes
        # and the stored reference remain untouched, even if validation fails.
        current = rebuild_material_mesh(old, parent_mesh.vertices)
        insertion = self.insertion
        corners = current.faces[insertion.vertex_parent_face]
        weights = insertion.vertex_barycentric
        vertices = np.einsum("vi,vij->vj", weights, current.vertices[corners])
        length = np.linalg.norm(vertices, axis=1)
        if np.any(length <= 0) or not np.isfinite(length).all():
            raise ValueError("Moving support ancestry produced a collapsed vertex")
        vertices /= length[:, None]
        # Preserve original material vertices bit for bit, avoiding needless
        # renormalization of an already validated current parent coordinate.
        one_hot = (np.count_nonzero(weights, axis=1) == 1) & (weights.max(axis=1) == 1.)
        ids = np.flatnonzero(one_hot)
        vertices[ids] = current.vertices[corners[ids, np.argmax(weights[ids], axis=1)]]
        child = rebuild_material_mesh(insertion.mesh, vertices)
        path = ReferenceCrackPath(vertices[insertion.path_vertex_ids], radius_m / 1000.)
        transported = replace(insertion, mesh=child, path=path,
                              path_arclength_m=path.arclength_m)
        return EmbeddedPathBasis(current, transported, radius_m, self.poisson_ratio)

    def extensive(self, parent_values):
        """Partition parent amounts by fixed initial shares, including arrays.

        The same input gives bitwise identical child amounts at every geometry.
        A per-parent correction places summation roundoff in its largest
        original child, so a tiny child never absorbs that residual.
        """
        values = np.asarray(parent_values)
        if (values.ndim < 1 or len(values) != self.parent_mesh.cell_count
                or values.dtype.kind not in "biuf" or not np.isfinite(values).all()):
            raise ValueError("Material amounts must be finite numeric values per parent face")
        values = np.asarray(values, dtype=float)
        weights = self.area_fraction.reshape((-1,) + (1,) * (values.ndim - 1))
        result = values[self.parent_face] * weights
        summed = np.zeros_like(values)
        np.add.at(summed, self.parent_face, result)
        result[self._largest_child] += values - summed
        if not np.isfinite(result).all():
            raise ValueError("Fixed material subdivision overflowed an extensive field")
        if np.any((values[self.parent_face] >= 0) & (result < 0)):
            raise ValueError("Subdivision roundoff would create a negative material amount")
        return result
