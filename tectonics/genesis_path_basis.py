"""Hierarchical, assumed-strain mechanics on a prescribed crack support.

The unchanged parent membrane supplies background strains. Independent bank
motion supplies compatible refined-element increments. This deliberately
preserves the parent problem when banks are tied; it is NOT the compatible
strain of the displayed refined displacement. Both strain measures remain
available. Releasing bank DOFs requires a separately justified interface law
or an explicitly prescribed existing crack, not merely support insertion.
"""
from __future__ import annotations

from dataclasses import replace
from numbers import Real

import numpy as np
from scipy import sparse

from .genesis_material import rotate_tensor
from .genesis_path_material import ConservativeSubdivision
from .genesis_path_mesh import PathMeshInsertion
from .genesis_seams import split_mesh
from .genesis_shell import Membrane


def tangent_prolongation(parent_mesh, insertion):
    """Differentiate normalized chord interpolation, including radial DOF.

    Unlike unnormalized barycentric displacement, this preserves infinitesimal
    rigid rotations exactly. Original unchanged nodes retain identity blocks.
    """
    child = insertion.mesh
    owner = np.asarray(insertion.vertex_parent_face)
    weights = np.asarray(insertion.vertex_barycentric)
    if (owner.shape != (child.vertex_count,) or owner.dtype.kind not in "iu"
            or np.any(owner < 0) or np.any(owner >= parent_mesh.cell_count)
            or weights.shape != (child.vertex_count, 3)
            or weights.dtype.kind not in "fiu" or not np.isfinite(weights).all()
            or np.any(weights < 0)
            or not np.allclose(weights.sum(axis=1), 1., rtol=0, atol=2e-12)):
        raise ValueError("Invalid normalized chord interpolation ancestry")
    old, new = Membrane(parent_mesh, .25), Membrane(child, .25)
    corners = parent_mesh.faces[owner]
    chord = np.einsum("vi,vij->vj", weights, parent_mesh.vertices[corners])
    length = np.linalg.norm(chord, axis=1)
    if (np.any(length <= 0) or not np.allclose(chord/length[:, None],
            child.vertices, rtol=0, atol=2e-12)):
        raise ValueError("Interpolation ancestry does not reconstruct child vertices")
    blocks = np.einsum("vji,vkjl->vkil", new.vertex_basis,
                       old.vertex_basis[corners])
    blocks *= (weights/length[:, None])[:, :, None, None]
    sole = np.argmax(weights, axis=1)
    unchanged = ((np.count_nonzero(weights, axis=1) == 1)
        & (np.max(weights, axis=1) == 1.)
        & np.all(child.vertices == parent_mesh.vertices[
            corners[np.arange(child.vertex_count), sole]], axis=1))
    blocks[unchanged] = 0.
    indices = np.flatnonzero(unchanged)
    blocks[indices, sole[indices]] = np.eye(2)
    rows = np.broadcast_to((2*np.arange(child.vertex_count))[:, None, None, None]
            + np.arange(2)[None, None, :, None], blocks.shape)
    cols = np.broadcast_to(2*corners[:, :, None, None]
            + np.arange(2)[None, None, None, :], blocks.shape)
    result = sparse.coo_matrix((np.r_[blocks.ravel(), 1.],
        (np.r_[rows.ravel(), new.ndof-1], np.r_[cols.ravel(), old.ndof-1])),
        shape=(new.ndof, old.ndof)).tocsr()
    result.eliminate_zeros()
    return result


def _operator(local, dofs, ndof):
    rows = np.broadcast_to(np.arange(local.shape[0]*3).reshape(-1, 3, 1), local.shape)
    columns = np.broadcast_to(dofs[:, None, :], local.shape)
    matrix = sparse.coo_matrix((local.ravel(), (rows.ravel(), columns.ravel())),
                              shape=(local.shape[0]*3, ndof)).tocsr()
    matrix.eliminate_zeros()
    return matrix


def _array(value, shape, name):
    value = np.asarray(value)
    if (value.shape != shape or value.dtype.kind not in "fiu"
            or not np.isfinite(value).all()):
        raise ValueError(f"{name} must be finite real values with shape {shape}")
    return np.asarray(value, dtype=float)


def _areas(mesh, radius_m):
    area = np.zeros(mesh.vertex_count)
    np.add.at(area, mesh.faces.ravel(), np.repeat(mesh.areas_unit_sphere/3, 3))
    return area*radius_m**2


class EmbeddedPathBasis:
    """Parent displacement plus area-balanced relative motion of split banks.

    ``nparent`` includes the parent's common radial DOF. Generalized DOFs are
    ``[parent displacement, two tangent components per enrichment vertex]``.
    The full path is split once; ``free_dofs`` ties all inactive sections and
    active interval tips. No physical nucleation or propagation rule is used.

    Background drag retains parent quadrature. Relative drag uses fine nodal
    areas and is orthogonal to the background in that fine metric. The radial
    DOF has zero basal drag, as in the original physical-time solver.
    """

    def __init__(self, parent_mesh, insertion, radius_m, poisson_ratio):
        for name, value in (("radius_m", radius_m), ("poisson_ratio", poisson_ratio)):
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                    or not np.isfinite(value)):
                raise ValueError(f"{name} must be a finite real number")
        if radius_m <= 0 or not -1 < poisson_ratio < .5:
            raise ValueError("Positive radius and physical Poisson ratio are required")
        if not isinstance(insertion, PathMeshInsertion):
            raise ValueError("An explicit PathMeshInsertion is required")
        if not np.isclose(radius_m, insertion.path.radius_km*1000., rtol=2e-14, atol=0):
            raise ValueError("Basis and crack support must use the same physical radius")
        self.radius_m = float(radius_m)
        self.subdivision = ConservativeSubdivision(parent_mesh, insertion.mesh, insertion.parent_face)
        # Own the arrays used by interval selection and interpolation.
        self.insertion = replace(insertion, mesh=self.subdivision.mesh,
            **{name: np.array(getattr(insertion, name), copy=True) for name in
               ("parent_face", "path_vertex_ids", "path_arclength_m",
                "vertex_parent_face", "vertex_barycentric")})
        support = self.insertion.path_vertex_ids
        if (support.ndim != 1 or support.dtype.kind not in "iu" or len(support) < 2
                or np.any(support < 0) or np.any(support >= insertion.mesh.vertex_count)
                or len(np.unique(support)) != len(support)):
            raise ValueError("Support vertices must form a simple ordered path")
        arclength = _array(self.insertion.path_arclength_m, support.shape, "Support arclength")
        if (arclength[0] != 0 or arclength[-1] != insertion.path.length_m
                or np.any(np.diff(arclength) <= 0)):
            raise ValueError("Support arclength must increase from zero to path length")
        self.parent_membrane = Membrane(self.subdivision.parent_mesh, poisson_ratio)
        self.topology = split_mesh(self.subdivision.mesh, self.insertion.cut_edges)
        self.membrane = Membrane(self.topology.mesh, poisson_ratio)
        self.nparent = self.parent_membrane.ndof
        parent_vertex = self.topology.parent_vertex
        copies = [[] for _ in range(self.subdivision.mesh.vertex_count)]
        for vertex, parent in enumerate(parent_vertex):
            copies[parent].append(vertex)
        if any(len(items) > 2 for items in copies):
            raise ValueError("Embedded support permits only two banks at a vertex")
        self.enrichment_vertices = np.array([i for i, items in enumerate(copies)
                                             if len(items) == 2], dtype=np.int64)
        if not np.isin(self.enrichment_vertices, support[1:-1]).all():
            raise ValueError("Only interior support vertices may have independent banks")
        self.ndof = self.nparent+2*len(self.enrichment_vertices)
        fine_area = _areas(self.topology.mesh, self.radius_m)
        self.fine_drag_area_m2 = np.r_[np.repeat(fine_area, 2), 0.]
        rows, columns, entries = [], [], []
        for index, vertex in enumerate(self.enrichment_vertices):
            first, second = copies[vertex]
            a, b = fine_area[[first, second]]
            if min(a, b) <= 0 or not np.isfinite(a+b):
                raise ValueError("Each enrichment bank needs finite positive drag area")
            for bank, value in ((first, b/(a+b)), (second, -a/(a+b))):
                for component in range(2):
                    rows.append(2*bank+component)
                    columns.append(2*index+component)
                    entries.append(value)
        self.W = sparse.coo_matrix((entries, (rows, columns)),
            shape=(self.membrane.ndof, self.ndof-self.nparent)).tocsr()
        prolongation = tangent_prolongation(self.subdivision.parent_mesh, self.insertion)
        fine_rows = np.r_[(2*parent_vertex[:, None]+np.arange(2)).ravel(), prolongation.shape[0]-1]
        self.parent_displacement_operator = prolongation[fine_rows]
        self.displacement_operator = sparse.hstack((self.parent_displacement_operator, self.W), format="csr")
        parent = self.subdivision.parent_face
        rotation = self.subdivision.frame_rotation
        transform = np.stack([rotate_tensor(np.broadcast_to(unit, (len(parent), 3)),
            rotation, engineering=True) for unit in np.eye(3)], axis=2)
        background = np.einsum("fij,fjk->fik", transform, self.parent_membrane.b[parent])
        inherited = _operator(background, self.parent_membrane.dofs[parent], self.nparent)
        self._geometric_operator = _operator(self.membrane.b, self.membrane.dofs, self.membrane.ndof)
        self.strain_operator = sparse.hstack((inherited, self._geometric_operator@self.W), format="csr")
        parent_area = _areas(self.subdivision.parent_mesh, self.radius_m)
        relative_drag = np.asarray(self.W.power(2).T@self.fine_drag_area_m2).ravel()
        self.drag_area_m2 = np.r_[np.repeat(parent_area, 2), 0., relative_drag]
        if not np.isfinite(self.drag_area_m2).all() or np.any(relative_drag <= 0):
            raise ValueError("Embedded drag areas must be finite and relative drag positive")
        self._incident_edges = {int(v): set() for v in self.enrichment_vertices}
        for edge in self.insertion.cut_edges:
            key = tuple(edge)
            for vertex in edge:
                if int(vertex) in self._incident_edges:
                    self._incident_edges[int(vertex)].add(key)

    def free_dofs(self, interval=None):
        """Return parent DOFs and vertices strictly inside the active interval.

        ``None`` means all banks tied. Fronts must already exist in the common
        support geometry; no geometry or state changes occur during selection.
        """
        if interval is None:
            return np.arange(self.nparent, dtype=np.int64)
        active = {tuple(edge) for edge in self.insertion.cuts_for(interval)}
        indices = [i for i, vertex in enumerate(self.enrichment_vertices)
                   if self._incident_edges[int(vertex)].issubset(active)]
        return np.r_[np.arange(self.nparent),
            (self.nparent+2*np.asarray(indices, dtype=np.int64)[:, None]+np.arange(2)).ravel()]

    def strain(self, z):
        """Constitutive engineering strain of generalized displacement in m."""
        z = _array(z, (self.ndof,), "Generalized displacement")
        value = np.asarray(self.strain_operator@z).reshape(-1, 3)/self.radius_m
        if not np.isfinite(value).all():
            raise ValueError("Embedded constitutive strain overflowed")
        return value

    def geometric_strain(self, z):
        """Compatible fine strain, separately from the assumed-strain model."""
        z = _array(z, (self.ndof,), "Generalized displacement")
        value = np.asarray(self._geometric_operator@(self.displacement_operator@z)).reshape(-1, 3)/self.radius_m
        if not np.isfinite(value).all():
            raise ValueError("Embedded geometric strain overflowed")
        return value

    def bulk(self, volume_m3, elasticity_pa, memory):
        """Return physical K [N/m] and prestress g [N] from one strain basis."""
        count = self.topology.mesh.cell_count
        volume = _array(volume_m3, (count,), "Material volume")
        elasticity = _array(elasticity_pa, (count, 3, 3), "Elasticity")
        strain = _array(memory, (count, 3), "Elastic memory")
        if np.any(volume <= 0):
            raise ValueError("Material volumes must be positive")
        if (not np.allclose(elasticity, elasticity.transpose(0, 2, 1), rtol=1e-12, atol=0)
                or np.any(np.linalg.eigvalsh(elasticity) <= 0)):
            raise ValueError("Elasticity must be symmetric positive definite")
        indices = np.arange(3*count).reshape(-1, 3)
        weighting = sparse.coo_matrix(((elasticity*volume[:, None, None]).ravel(),
            (np.broadcast_to(indices[:, :, None], elasticity.shape).ravel(),
             np.broadcast_to(indices[:, None, :], elasticity.shape).ravel())),
            shape=(3*count, 3*count)).tocsr()
        h = self.strain_operator/self.radius_m
        matrix = (h.T@weighting@h).tocsr()
        force = np.asarray(h.T@(weighting@strain.ravel())).ravel()
        if not np.isfinite(matrix.data).all() or not np.isfinite(force).all():
            raise ValueError("Embedded bulk assembly overflowed")
        return matrix, force

    def external(self, parent_force, fine_force):
        """Preserve old external work, adding only explicit relative work."""
        parent = _array(parent_force, (self.nparent,), "Parent force")
        fine = _array(fine_force, (self.membrane.ndof,), "Fine force")
        force = np.r_[parent, self.W.T@fine]
        if not np.isfinite(force).all():
            raise ValueError("Embedded external force overflowed")
        return force
