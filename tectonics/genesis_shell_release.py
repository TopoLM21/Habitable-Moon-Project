"""Reversible virtual extension of explicit edge cracks in the genesis membrane.

Geometry, material, prestress and dead loads are frozen. Free banks carry no
cohesion, friction or viscosity. Penetrating/large-motion solutions are retained
as rejected diagnostics, never admitted as physical growth. No time advances,
cuts are not selected automatically, and no production checkpoint is written.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import warnings

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import MatrixRankWarning, norm as sparse_norm, spsolve

from .genesis_shell import Membrane, maximum_total_strain
from .genesis_seams import split_mesh
from .mesh import SphereMesh, connected_components


VERSION = "genesis-shell-release-0.1"


def _array(value, shape, name):
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
        raise ValueError(f"{name} must be a finite real array of shape {shape}")
    return np.frombuffer(np.asarray(raw, dtype="<f8").tobytes(), dtype="<f8").reshape(shape)


def _scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a positive finite number")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def _relative(difference, reference):
    return float(np.linalg.norm(difference)/max(np.linalg.norm(reference), np.finfo(float).tiny))


@dataclass(frozen=True)
class ShellEquilibrium:
    topology: object
    membrane: Membrane
    bulk_matrix: object
    constraints: object
    initial_bulk_force: np.ndarray
    external_force: np.ndarray
    displacement_m: np.ndarray
    stored_energy_j: float
    external_potential_work_j: float
    potential_energy_j: float
    reduced_potential_j: float
    normal_gap_m: np.ndarray
    tangential_jump_m: np.ndarray
    equilibrium_residual: float
    constraint_residual: float
    max_added_strain: float
    max_motion_edge_fraction: float
    min_gap_m: float
    admissible_open_crack: bool
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class ShellRelease:
    before: ShellEquilibrium
    after: ShellEquilibrium
    release_j: float
    added_area_m2: float
    mean_release_j_m2: float
    potential_difference_j: float
    relaxation_energy_j: float
    embedding_energy_error_j: float
    force_pullback_relative_error: float
    prestress_pullback_relative_error: float
    stiffness_pullback_relative_error: float
    gauge_pullback_relative_error: float
    relative_release_identity_error: float
    material_area_relative_error: float
    material_volume_relative_error: float
    admissible_open_crack: bool
    rejection_reasons: tuple[str, ...]


class FrozenShell:
    """Physical SI data on unchanged material faces; loads belong to that material.

    Vertex forces are tangent Cartesian vectors on the original mesh. They are
    distributed to split copies by incident face area, exactly as in ContactModel.
    A positive stiffness tensor uses engineering shear, as does Membrane.b.
    Global rigid rotation is removed with material-area-weighted constraints.
    """

    def __init__(self, mesh, radius_m, depth_m, elasticity_pa, elastic_strain,
                 vertex_force_xyz_n, radial_force_n=0., *, source_hash=None,
                 equilibrium_tolerance=1e-9, max_strain=.005,
                 max_motion_edge_fraction=.02, penetration_tolerance_m=1e-6):
        # Validating and copying through split_mesh also rejects stale geometry.
        self.mesh = split_mesh(mesh, []).mesh
        for name in ("vertices", "faces", "centroids", "areas_unit_sphere"):
            array = getattr(self.mesh, name)
            copied = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
            setattr(self.mesh, name, copied)
        self.radius_m = _scalar(radius_m, "radius_m")
        count = self.mesh.cell_count
        self.depth_m = _array(depth_m, (count,), "depth_m")
        if np.any(self.depth_m <= 0):
            raise ValueError("Every material face needs positive solid depth")
        self.elasticity_pa = _array(elasticity_pa, (count, 3, 3), "elasticity_pa")
        if (not np.allclose(self.elasticity_pa, self.elasticity_pa.transpose(0, 2, 1), rtol=1e-13, atol=0)
                or np.any(np.linalg.eigvalsh(self.elasticity_pa) <= 0)):
            raise ValueError("Reversible elasticity must be symmetric positive definite")
        self.elastic_strain = _array(elastic_strain, (count, 3), "elastic_strain")
        self.vertex_force_xyz_n = _array(vertex_force_xyz_n, (self.mesh.vertex_count, 3), "vertex_force_xyz_n")
        radial_component = np.einsum("vi,vi->v", self.vertex_force_xyz_n, self.mesh.vertices)
        if np.linalg.norm(radial_component) > 1e-12*max(np.linalg.norm(self.vertex_force_xyz_n), np.finfo(float).tiny):
            raise ValueError("Vertex forces must be tangent to the reference sphere")
        if isinstance(radial_force_n, (bool, np.bool_)) or not np.isfinite(radial_force_n):
            raise ValueError("Common radial force must be finite")
        self.radial_force_n = float(radial_force_n)
        self.equilibrium_tolerance = _scalar(equilibrium_tolerance, "equilibrium_tolerance")
        self.max_strain = _scalar(max_strain, "max_strain")
        self.max_motion_edge_fraction = _scalar(max_motion_edge_fraction, "max_motion_edge_fraction")
        self.penetration_tolerance_m = _scalar(penetration_tolerance_m, "penetration_tolerance_m")
        if self.equilibrium_tolerance > 1e-5 or self.max_strain > .01 or self.max_motion_edge_fraction > .05:
            raise ValueError("Controls exceed the small-strain diagnostic limits")
        self.source_hash = source_hash
        self.reference_volume_m3 = _array(self.mesh.areas_unit_sphere*self.radius_m**2*self.depth_m,
                                           (count,), "reference_volume_m3")
        self.initial_energy_j = .5*float(np.einsum("fi,fij,fj,f->", self.elastic_strain,
                    self.elasticity_pa, self.elastic_strain, self.reference_volume_m3))
        self.vertex_area = self._vertex_areas(self.mesh)
        original = Membrane(self.mesh, .25)  # Only b/bases are used, never this d.
        self.original_membrane = original
        self.original_force = np.r_[np.einsum("vij,vi->vj", original.vertex_basis,
                                               self.vertex_force_xyz_n).ravel(), self.radial_force_n]
        self._gauge_norm = np.linalg.norm(self._raw_constraints(original, self.vertex_area), axis=1)
        identity = hashlib.sha256(VERSION.encode())
        for array in (self.mesh.vertices, self.mesh.faces, self.mesh.areas_unit_sphere,
                      self.depth_m, self.elasticity_pa, self.elastic_strain,
                      self.vertex_force_xyz_n, np.asarray([self.radius_m, self.radial_force_n])):
            identity.update(np.ascontiguousarray(array).tobytes())
        self.fingerprint = identity.hexdigest()

    @classmethod
    def from_uniform(cls, mesh, radius_m, depth_m, young_pa, poisson_ratio,
                     traction_xyz_pa, elastic_strain=None, **kwargs):
        young = _scalar(young_pa, "young_pa")
        nu = _scalar(poisson_ratio, "poisson_ratio")
        if nu >= .49:
            raise ValueError("poisson_ratio must lie between zero and .49")
        membrane = Membrane(mesh, nu)
        depth = np.full(mesh.cell_count, _scalar(depth_m, "depth_m"))
        traction = _array(traction_xyz_pa, (mesh.vertex_count, 3), "traction_xyz_pa")
        force = traction*cls._vertex_areas(mesh)[:, None]*float(radius_m)**2
        return cls(mesh, radius_m, depth, np.broadcast_to(young*membrane.d, (mesh.cell_count, 3, 3)),
                   np.zeros((mesh.cell_count, 3)) if elastic_strain is None else elastic_strain,
                   force, **kwargs)

    @classmethod
    def from_fault_checkpoint(cls, filename, **kwargs):
        # Explicit empty cuts bypass the historical selector. Never save this
        # temporary assembly as a production ContactModel checkpoint.
        from .genesis_contact import ContactModel
        data = Path(filename).read_bytes()
        inherited = ContactModel(data, source_path=str(Path(filename).resolve()), reference_cuts=[])
        force = np.einsum("vij,vj->vi", inherited.membrane.vertex_basis,
                          inherited.external_force[:-1].reshape(-1, 2))
        return cls(inherited.topology.mesh, inherited.radius_m, inherited.depth_m,
                   inherited.elasticity, inherited.source_state.elastic_strain, force,
                   inherited.external_force[-1], source_hash=hashlib.sha256(data).hexdigest(), **kwargs)

    @staticmethod
    def _vertex_areas(mesh):
        area = np.zeros(mesh.vertex_count)
        np.add.at(area, mesh.faces.ravel(), np.repeat(mesh.areas_unit_sphere/3, 3))
        return area

    @staticmethod
    def _rotations(membrane):
        values = []
        for axis in np.eye(3):
            rot = np.cross(axis, membrane.mesh.vertices)
            values.append(np.r_[np.einsum("vij,vi->vj", membrane.vertex_basis, rot).ravel(), 0.])
        return np.asarray(values)

    def _raw_constraints(self, membrane, area):
        return self._rotations(membrane)*np.r_[np.repeat(area, 2), 0.]

    def _gaps(self, topology, membrane, q):
        edges = topology.cut_edges
        if not len(edges):
            return np.empty(0), np.empty(0)
        xyz = self.mesh.vertices[edges]
        normal = np.cross(xyz[:, 0], xyz[:, 1])
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        across = self.mesh.centroids[topology.seam_faces[:, 1]]-self.mesh.centroids[topology.seam_faces[:, 0]]
        normal *= np.where(np.einsum("ij,ij->i", normal, across) >= 0, 1., -1.)[:, None]
        displacement = np.einsum("vij,vj->vi", membrane.vertex_basis, q[:-1].reshape(-1, 2))
        banks = displacement[topology.bank_vertices]
        jump = banks[:, 1]-banks[:, 0]
        tangent = np.cross(normal[:, None, :], xyz)
        return np.einsum("svi,si->sv", jump, normal).ravel(), np.sum(jump*tangent, axis=2).ravel()

    def solve(self, cuts):
        topology = split_mesh(self.mesh, cuts)
        mesh = topology.mesh
        if len(connected_components(range(mesh.cell_count), mesh.neighbors)) != 1:
            raise ValueError("Detached material adds unconstrained rigid modes; free-fragment solve unsupported")
        membrane = Membrane(mesh, .25)
        weight = mesh.areas_unit_sphere*self.depth_m
        local_k = np.einsum("fai,fab,fbj,f->fij", membrane.b, self.elasticity_pa, membrane.b, weight)
        k = sparse.coo_matrix((local_k.ravel(), (membrane.rr, membrane.cc)),
                              shape=(membrane.ndof, membrane.ndof)).tocsr()
        stress = np.einsum("fij,fj->fi", self.elasticity_pa, self.elastic_strain)
        local_g = np.einsum("fai,fa,f->fi", membrane.b, stress, weight*self.radius_m)
        g = np.zeros(membrane.ndof)
        np.add.at(g, membrane.dofs.ravel(), local_g.ravel())
        area = self._vertex_areas(mesh)
        share = area/self.vertex_area[topology.parent_vertex]
        f = np.r_[(self.original_force[:-1].reshape(-1, 2)[topology.parent_vertex]*share[:, None]).ravel(),
                   self.radial_force_n]
        c = sparse.csr_matrix(self._raw_constraints(membrane, area)/self._gauge_norm[:, None])
        rhs = f-g
        rigid = self._rotations(membrane)
        # Near an inherited equilibrium, f-g may contain only cancellation
        # roundoff. Measure its torque against the forces being balanced, as
        # in the final force residual; dividing by that tiny difference would
        # turn roundoff into a false unsupported rigid-rotation load.
        load_scale = max(np.linalg.norm(f), np.linalg.norm(g), np.finfo(float).tiny)
        torque_error = float(np.linalg.norm(rigid@rhs)/(load_scale*np.linalg.norm(rigid)))
        if torque_error > self.equilibrium_tolerance:
            raise ValueError("Unbalanced rigid-rotation load; gauge would introduce artificial support reactions")
        if np.any(k.diagonal() <= 0):
            raise ValueError("Elastic matrix has unsupported zero-stiffness degrees of freedom")
        scaling = 1/np.sqrt(k.diagonal())
        s = sparse.diags(scaling)
        cs = c@s
        row_scale = 1/np.sqrt(np.asarray(cs.multiply(cs).sum(axis=1)).ravel())
        cs = sparse.diags(row_scale)@cs
        system = sparse.bmat([[s@k@s, cs.T], [cs, None]], format="csc")
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            try:
                answer = spsolve(system, np.r_[scaling*rhs, np.zeros(3)])
            except (MatrixRankWarning, RuntimeError) as exc:
                raise ValueError("Elastic equilibrium is singular or failed") from exc
        if not np.isfinite(answer).all():
            raise ValueError("Elastic equilibrium is nonfinite")
        q = scaling*answer[:membrane.ndof]
        force_scale = max(np.linalg.norm(k@q), np.linalg.norm(g), np.linalg.norm(f), np.finfo(float).tiny)
        residual = float(np.linalg.norm(k@q+g-f)/force_scale)
        gauge_residual = float(np.linalg.norm(c@q)/max(np.linalg.norm(q), np.finfo(float).tiny))
        if residual > self.equilibrium_tolerance or gauge_residual > self.equilibrium_tolerance:
            raise ValueError("Elastic equilibrium failed the unconstrained force/gauge residual checks")
        inc = np.einsum("fai,fi->fa", membrane.b, q[membrane.dofs])/self.radius_m
        strain = self.elastic_strain+inc
        stored = .5*float(np.einsum("fi,fij,fj,f->", strain, self.elasticity_pa, strain, self.reference_volume_m3))
        load_work = float(f@q)
        reduced = .5*float(q@(k@q))+float((g-f)@q)
        gaps, jumps = self._gaps(topology, membrane, q)
        edge = self.mesh.vertices[np.asarray(self.mesh.shared_edges)[:, 2:]]
        lengths = np.arctan2(np.linalg.norm(np.cross(edge[:, 0], edge[:, 1]), axis=1),
                             np.sum(edge[:, 0]*edge[:, 1], axis=1))*self.radius_m
        strain_max = float(maximum_total_strain(inc))
        motion = float(np.linalg.norm(q[:-1].reshape(-1, 2), axis=1).max()/lengths.min())
        min_gap = float(gaps.min()) if len(gaps) else 0.
        reasons = []
        if min_gap < -self.penetration_tolerance_m:
            reasons.append("free_banks_interpenetrate")
        if strain_max > self.max_strain or abs(q[-1])/self.radius_m > self.max_strain:
            reasons.append("reference_strain_limit")
        if motion > self.max_motion_edge_fraction:
            reasons.append("reference_motion_limit")
        return ShellEquilibrium(topology, membrane, k, c, g, f, q, stored, load_work,
                                self.initial_energy_j+reduced, reduced, gaps, jumps,
                                residual, gauge_residual, strain_max, motion, min_gap,
                                not reasons, tuple(reasons))

    @staticmethod
    def _prolongation(old, new):
        """Map via material corners, preserving the identities of existing banks."""
        ancestry = np.full(new.topology.mesh.vertex_count, -1, dtype=int)
        for old_vertex, new_vertex in zip(old.topology.mesh.faces.ravel(), new.topology.mesh.faces.ravel()):
            if ancestry[new_vertex] not in (-1, old_vertex):
                raise ValueError("Trial topology heals or merges existing banks")
            ancestry[new_vertex] = old_vertex
        rows = np.arange(new.membrane.ndof)
        cols = np.r_[(2*ancestry[:, None]+np.arange(2)).ravel(), old.membrane.ndof-1]
        return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(new.membrane.ndof, old.membrane.ndof))

    def compare_extension(self, seed_cuts, trial_cuts):
        old = self.solve(seed_cuts)
        old_set = {tuple(edge) for edge in old.topology.cut_edges}
        trial_topology = split_mesh(self.mesh, trial_cuts)
        new_set = {tuple(edge) for edge in trial_topology.cut_edges}
        if not old_set <= new_set:
            raise ValueError("Trial cuts must contain every existing cut")
        new = self.solve(trial_cuts)
        p = self._prolongation(old, new)
        lifted = p@old.displacement_m
        dk = p.T@new.bulk_matrix@p-old.bulk_matrix
        k_error = float(sparse_norm(dk)/sparse_norm(old.bulk_matrix))
        f_error = _relative(p.T@new.external_force-old.external_force, old.external_force)
        g_error = _relative(p.T@new.initial_bulk_force-old.initial_bulk_force, old.initial_bulk_force)
        c_error = float(sparse_norm(new.constraints@p-old.constraints)/sparse_norm(old.constraints))
        if max(k_error, f_error, g_error, c_error) > 1e-11:
            raise ValueError("Virtual extension changed inherited material, load or rotation gauge")
        lifted_potential = .5*float(lifted@(new.bulk_matrix@lifted))+float((new.initial_bulk_force-new.external_force)@lifted)
        embedding_error = lifted_potential-old.reduced_potential_j
        difference = old.reduced_potential_j-new.reduced_potential_j
        delta = new.displacement_m-lifted
        relaxation = .5*float(delta@(new.bulk_matrix@delta))
        residual_work = float(delta@(new.bulk_matrix@new.displacement_m+new.initial_bulk_force-new.external_force))
        released = relaxation-residual_work
        energy_scale = max(abs(old.reduced_potential_j), abs(new.reduced_potential_j),
                           abs(self.initial_energy_j), np.finfo(float).tiny)
        roundoff = 200*np.finfo(float).eps*energy_scale
        error = abs(difference-released)
        if error > max(roundoff, self.equilibrium_tolerance*abs(released)) or released < -roundoff:
            raise ValueError("Virtual extension energy identity failed")
        added_area = 0.
        for edge, faces in zip(new.topology.cut_edges, new.topology.seam_faces):
            if tuple(edge) in old_set:
                continue
            a, b = self.mesh.vertices[edge]
            length = np.arctan2(np.linalg.norm(np.cross(a, b)), float(a@b))*self.radius_m
            added_area += length*float(np.min(self.depth_m[faces]))
        # Zero added area is useful as a same-topology negative control.
        mean_release = released/added_area if added_area else 0.
        reasons = tuple(sorted(set(old.rejection_reasons+new.rejection_reasons)))
        area_error = float(np.max(np.abs(new.topology.mesh.areas_unit_sphere-self.mesh.areas_unit_sphere)))
        volume_error = float(np.max(np.abs(new.topology.mesh.areas_unit_sphere*self.radius_m**2*self.depth_m-self.reference_volume_m3)))
        return ShellRelease(old, new, released, added_area, mean_release, difference, relaxation,
                            embedding_error, f_error, g_error, k_error, c_error,
                            error/max(abs(released), roundoff, np.finfo(float).tiny),
                            area_error/max(float(self.mesh.areas_unit_sphere.max()), np.finfo(float).tiny),
                            volume_error/max(float(self.reference_volume_m3.max()), np.finfo(float).tiny),
                            not reasons, reasons)
