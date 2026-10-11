"""Prospective, fully tied banks for the source-bound six-DOF volume field.

Supplied cut edges are computational supports, not calculated fracture
surfaces.  The unchanged ``split_mesh`` partitions corner fans; each whole
cohort retains its parent face, placement, ID, constitutive tensor and all
angular/radial strain coefficients.  Only its nodal assembly indices change.

The parent GenesisVolumeGeometry is never split or assigned new owner IDs.
R copies the six Cartesian (A_xyz, P_xyz) amplitudes to every corner copy.
J measures one independent duplicate-minus-parent jump per copied DOF.
These banks remain tied: this module neither creates a physical crack nor
admits contact, release, fracture work, or a planetary bank trajectory.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json

import numpy as np
from scipy import sparse

from .genesis_volume_operator import (
    ANGULAR_BARYCENTRIC, GenesisVolumeOperator, _assemble_matrix, _construct_operator,
    _immutable, engineering_strain_to_tensor, isotropic_elasticity,
)


FORMAT = "genesis-prospective-tied-volume-banks-0.1"
SCOPE = "source_bound_parent_cohorts_on_prospective_edge_split_six_dof_support"


def _frozen_csr(matrix):
    result = sparse.csr_matrix(matrix, copy=True)
    result.sum_duplicates()
    result.sort_indices()
    result.data = _immutable(result.data)
    result.indices = _immutable(result.indices, result.indices.dtype)
    result.indptr = _immutable(result.indptr, result.indptr.dtype)
    return result


@dataclass(frozen=True, slots=True)
class ProspectiveBankMesh:
    vertices: np.ndarray
    faces: np.ndarray
    centroids: np.ndarray
    areas_unit_sphere: np.ndarray
    neighbors: tuple
    shared_edges: tuple

    @property
    def vertex_count(self):
        return len(self.vertices)

    @property
    def cell_count(self):
        return len(self.faces)


@dataclass(frozen=True, slots=True)
class ProspectiveVolumeBankTopology:
    mesh: ProspectiveBankMesh
    parent_vertex: np.ndarray
    original_faces: np.ndarray
    cut_edges: np.ndarray
    seam_faces: np.ndarray
    bank_vertices: np.ndarray
    original_shared_edges: tuple
    intact_shared_edges: tuple

    @property
    def original_vertex_count(self):
        return int(self.parent_vertex.max())+1

    @property
    def seam_count(self):
        return len(self.cut_edges)

    @property
    def calculated_fracture_surface(self):
        return False


@dataclass(frozen=True, slots=True)
class GenesisVolumeBankBasal:
    parent_basal: object
    mass_matrix: object
    basal_drag_matrix: object
    external_force_n: np.ndarray
    cohort_mass_matrices: np.ndarray
    face_basal_drag_matrices: np.ndarray
    face_external_force_n: np.ndarray
    face_dofs: np.ndarray
    total_live_mass_kg: float


@dataclass(frozen=True, slots=True)
class GenesisVolumeBanks:
    parent_assembly: GenesisVolumeOperator
    topology: ProspectiveVolumeBankTopology
    ndof: int
    dofs: np.ndarray
    face_dofs: np.ndarray
    prolongation: object
    jump_operator: object
    stiffness: object
    basal: GenesisVolumeBankBasal | None
    binding_sha256: str
    format: str = FORMAT
    scope: str = SCOPE

    @property
    def R(self):
        return self.prolongation

    @property
    def J(self):
        return self.jump_operator

    @property
    def K(self):
        return self.stiffness

    @property
    def physical_bank_creation_admitted(self):
        return False

    @property
    def physical_contact_release_admitted(self):
        return False

    @property
    def physical_force_work_admission(self):
        return False


@dataclass(frozen=True, slots=True)
class GenesisVolumeBankResponse:
    elastic_constant: np.ndarray
    elastic_inverse_scale: np.ndarray
    cohort_volume_mean_elastic: np.ndarray
    cohort_volume_mean_stress_pa: np.ndarray
    elastic_energy_j: float
    generalized_force_n: np.ndarray
    stiffness: object
    maximum_elastic_principal_strain: float

    @property
    def energy_j(self):
        return self.elastic_energy_j

    @property
    def force(self):
        return self.generalized_force_n

    @property
    def memory0(self):
        return self.elastic_constant

    @property
    def memory1(self):
        return self.elastic_inverse_scale


def _hash_value(digest, value):
    """Bind original arrays and sparse structure, including zero-size shapes."""
    if isinstance(value, np.ndarray):
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    elif sparse.issparse(value):
        digest.update(str(value.shape).encode())
        for array in (value.data, value.indices, value.indptr):
            _hash_value(digest, array)
    elif hasattr(value, "__dataclass_fields__"):
        for field in fields(value):
            if field.name == "binding_sha256":
                continue
            digest.update(field.name.encode())
            _hash_value(digest, getattr(value, field.name))
    else:
        digest.update(repr(value).encode())
    digest.update(b"\x00")


def _binding(banks):
    digest = hashlib.sha256()
    _hash_value(digest, banks)
    return digest.hexdigest()


def _same_matrix(actual, expected):
    return (sparse.isspmatrix_csr(actual) and actual.shape == expected.shape
            and all(type(getattr(actual, name)) is np.ndarray
                    and getattr(actual, name).dtype == getattr(expected, name).dtype
                    for name in ("indptr", "indices", "data"))
            and np.array_equal(actual.indptr, expected.indptr)
            and np.array_equal(actual.indices, expected.indices)
            and np.array_equal(actual.data, expected.data))


def _real_values(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "ifu":
        raise ValueError(f"{name} requires real numeric values, not logical or complex fields")
    return np.asarray(raw, float)


def _check_parent(assembly):
    """Check geometry by replay and every local/global operator by reconstruction.

    Current damage is the caller's accepted-state binding.  Here its elastic
    degradation must lie in the authentic source law's allowed range and be
    constant among live cohorts of a face; no new constitutive law is fitted.
    """
    from .genesis_volume_geometry import validate_genesis_volume_geometry
    if type(assembly) is not GenesisVolumeOperator:
        raise ValueError("Volume banks require the genuine source-bound parent operator")
    geometry = validate_genesis_volume_geometry(assembly.geometry)
    if (assembly.source_checkpoint_sha256 != geometry.source_checkpoint_sha256
            or assembly.source_identity != geometry.source_identity
            or assembly.time_myr != geometry.time_myr):
        raise ValueError("Volume bank parent source or material clock differs from its geometry")
    packet = json.loads(geometry.source_packet_json)
    ncohort = len(geometry.cohorts)
    face = np.array([c.face_index for c in geometry.cohorts], int)
    shell = packet["source"]["parameters"]["shell"]
    C0 = isotropic_elasticity(shell["young_modulus_pa"], shell["poisson_ratio"])
    C = np.asarray(assembly.elasticity_pa)
    if C.shape != (ncohort, 6, 6) or C.dtype != np.dtype(float) or not np.isfinite(C).all():
        raise ValueError("Volume bank parent has malformed cohort elasticity")
    factors = C[:, 0, 0]/C0[0, 0]
    if (np.any(factors < shell["residual_stiffness"]-3e-15)
            or np.any(factors > 1+3e-15)
            or not np.allclose(C, factors[:, None, None]*C0, rtol=3e-15, atol=0)):
        raise ValueError("Volume bank elasticity is outside its source damage law")
    for index in np.unique(face):
        if not np.array_equal(factors[face == index], np.full(np.count_nonzero(face == index), factors[face == index][0])):
            raise ValueError("Volume bank parent damage differs within one material face")
    face_basis = np.asarray(packet["current_geometry"]["face_frames_xyz"], float)
    face_basis = np.concatenate((face_basis, np.cross(face_basis[:, :, 0], face_basis[:, :, 1])[:, :, None]), axis=2)
    basis = face_basis[face]
    if not np.array_equal(assembly.basis_xyz, basis):
        raise ValueError("Volume bank parent changed inherited cohort frames")
    elastic = assembly.source_elastic_strain
    if elastic is not None:
        from .genesis_volume_operator import tensor_to_engineering_strain
        source = packet["instantaneous_cohort_plane_stress_3d"]
        if (type(elastic) is not np.ndarray or elastic.dtype != np.dtype(float)
                or geometry.runtime_birth_history_json is not None
                or source is None
                or not np.array_equal(assembly.cohort_ids, source["source_cohort_ids"])
                or not np.array_equal(elastic, tensor_to_engineering_strain(source["elastic_strain_local"]))):
            raise ValueError("Volume bank parent supplied an invented inherited elastic memory")
    expected = _construct_operator(geometry, C, basis, elastic)
    for field in fields(GenesisVolumeOperator):
        name = field.name
        actual, rebuilt = getattr(assembly, name), getattr(expected, name)
        if name == "geometry":
            continue
        if name == "source_elastic_energy_j" and elastic is not None:
            # The parent keeps the exact persisted source constitutive ledger.
            if actual != packet["cohort_mechanics"]["state"]["elastic_energy_j"]:
                raise ValueError("Volume bank parent changed the inherited energy ledger")
        elif sparse.issparse(rebuilt):
            if not _same_matrix(actual, rebuilt):
                raise ValueError("Volume bank parent global stiffness differs from its cohort locals")
        elif isinstance(rebuilt, np.ndarray):
            if (type(actual) is not np.ndarray or actual.dtype != rebuilt.dtype
                    or not np.array_equal(actual, rebuilt)):
                raise ValueError(f"Volume bank parent {name} differs from its source-bound reconstruction")
        elif actual != rebuilt:
            raise ValueError(f"Volume bank parent {name} differs from its source-bound reconstruction")
    return geometry, packet


def _check_basal(assembly, basal):
    from .genesis_contact import ContactParameters
    from .genesis_volume_basal import GenesisVolumeBasalAssembly, assemble_genesis_volume_basal
    if type(basal) is not GenesisVolumeBasalAssembly:
        raise ValueError("Volume banks require a genuine source-bound basal assembly")
    try:
        parameters = ContactParameters(**json.loads(basal.contact_parameters_json))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Volume bank basal parameters are malformed") from exc
    rebuilt = assemble_genesis_volume_basal(assembly, parameters)
    for field in fields(GenesisVolumeBasalAssembly):
        actual, expected = getattr(basal, field.name), getattr(rebuilt, field.name)
        if sparse.issparse(expected):
            same = _same_matrix(actual, expected)
        elif isinstance(expected, np.ndarray):
            same = (type(actual) is np.ndarray and actual.dtype == expected.dtype
                    and np.array_equal(actual, expected))
        else:
            same = actual == expected
        if not same:
            raise ValueError(f"Volume bank basal {field.name} differs from its parent source")


def prepare_volume_banks(parent_assembly, cut_edges, *, basal=None):
    """Assemble prospective bank locals without changing source material.

    ``cut_edges`` uses ORIGINAL reference vertex IDs.  An isolated cut tip
    can remain shared; a support with no duplicated vertex has no free jump
    DOF.  No physical fracture-location claim follows from supplying edges.
    ``basal`` optionally retains the exact parent consistent mass, basal drag
    and reference dead traction by reassembling their unchanged local terms.
    """
    from .genesis_material import rebuild_material_mesh
    from .genesis_seams import split_mesh
    from .mesh import build_icosphere
    geometry, packet = _check_parent(parent_assembly)
    shell = packet["source"]["parameters"]["shell"]
    radius_m = 1000*packet["current_geometry"]["radius_km"]
    mesh = rebuild_material_mesh(build_icosphere(shell["subdivisions"]),
                                 np.asarray(geometry.vertices_m)/radius_m)
    if not np.array_equal(mesh.faces, geometry.faces):
        raise ValueError("Volume bank support changed original reference faces")
    split = split_mesh(mesh, cut_edges)
    frozen_mesh = ProspectiveBankMesh(*(_immutable(getattr(split.mesh, name),
        int if name == "faces" else float) for name in
        ("vertices", "faces", "centroids", "areas_unit_sphere")),
        split.mesh.neighbors, split.mesh.shared_edges)
    topology = ProspectiveVolumeBankTopology(frozen_mesh,
        *(_immutable(getattr(split, name), int) for name in
          ("parent_vertex", "original_faces", "cut_edges", "seam_faces", "bank_vertices")),
        split.original_shared_edges, split.intact_shared_edges)
    faces = topology.mesh.faces
    face_dofs = _immutable((6*faces[:, :, None]+np.arange(6)).reshape(-1, 18), int)
    dofs = _immutable(face_dofs[parent_assembly.face_indices], int)
    ndof = 6*len(topology.parent_vertex)
    parent_ndof = parent_assembly.ndof
    columns = (6*topology.parent_vertex[:, None]+np.arange(6)).ravel()
    R = _frozen_csr(sparse.coo_matrix((np.ones(ndof), (np.arange(ndof), columns)),
                                     shape=(ndof, parent_ndof)))
    duplicate = np.arange(parent_ndof, ndof)
    jump_count = len(duplicate)
    J = _frozen_csr(sparse.coo_matrix((np.tile((-1., 1.), jump_count),
        (np.repeat(np.arange(jump_count), 2), np.column_stack((columns[duplicate], duplicate)).ravel())),
        shape=(jump_count, ndof)))
    K = _assemble_matrix(dofs, parent_assembly.cohort_stiffness, ndof)
    bank_basal = None
    if basal is not None:
        _check_basal(parent_assembly, basal)
        M = _assemble_matrix(dofs, basal.cohort_mass_matrices, ndof)
        D = _assemble_matrix(face_dofs, basal.face_basal_drag_matrices, ndof)
        F = np.zeros(ndof)
        np.add.at(F, face_dofs.ravel(), basal.face_external_force_n.ravel())
        bank_basal = GenesisVolumeBankBasal(basal, M, D, _immutable(F),
            basal.cohort_mass_matrices, basal.face_basal_drag_matrices,
            basal.face_external_force_n, face_dofs, basal.total_live_mass_kg)
    banks = GenesisVolumeBanks(parent_assembly, topology, ndof, dofs, face_dofs,
        R, J, K, bank_basal, "")
    from dataclasses import replace
    return replace(banks, binding_sha256=_binding(banks))


def validate_volume_banks(banks, *, parent_assembly=None):
    """Reject changed topology, locals, source, or CSR buffers after preparation.

    This in-memory binding is not an untrusted checkpoint serializer.  A
    caller loading a new support must rebuild it through ``prepare``.
    """
    if (type(banks) is not GenesisVolumeBanks or banks.format != FORMAT
            or banks.scope != SCOPE or banks.binding_sha256 != _binding(banks)):
        raise ValueError("Volume bank support differs from its prepared source binding")
    if parent_assembly is not None:
        a, b = hashlib.sha256(), hashlib.sha256()
        _hash_value(a, parent_assembly)
        _hash_value(b, banks.parent_assembly)
        if type(parent_assembly) is not GenesisVolumeOperator or a.digest() != b.digest():
            raise ValueError("Volume bank support belongs to a different parent operator")
    return banks


def evaluate_volume_banks(banks, displacement_increment_m, *,
                          elastic_constant=None, elastic_inverse_scale=None,
                          strain_factors=None):
    """Evaluate a NEW bank increment against complete accepted strain memory.

    Transferred absolute displacement is a kinematic ledger and MUST NOT be
    supplied again as this increment at the transition.  Evaluating zero
    increment retains every already accumulated angular/radial coefficient.
    """
    validate_volume_banks(banks)
    parent = banks.parent_assembly
    u = _real_values(displacement_increment_m, "Volume bank increment")
    if u.shape != (banks.ndof,) or not np.isfinite(u).all():
        raise ValueError("Volume bank increment requires all finite physical metre DOFs")
    shape = (len(parent.cohort_ids), 3, 6)
    if elastic_constant is None or elastic_inverse_scale is None:
        raise ValueError("Volume banks require both complete accepted elastic memory coefficients")
    e0 = _real_values(elastic_constant, "Volume bank constant strain memory")
    e1 = _real_values(elastic_inverse_scale, "Volume bank inverse-scale strain memory")
    if e0.shape != shape or e1.shape != shape or not np.isfinite(e0).all() or not np.isfinite(e1).all():
        raise ValueError("Volume bank elastic memory coefficient shape/finite mismatch")
    beta = np.ones(len(parent.cohort_ids)) if strain_factors is None else _real_values(
        strain_factors, "Volume bank strain factors")
    if beta.shape != parent.volume_m3.shape or not np.isfinite(beta).all() or np.any(beta < 0):
        raise ValueError("Volume bank strain factors must be finite nonnegative per cohort")
    K = banks.stiffness if np.array_equal(beta, np.ones_like(beta)) else _assemble_matrix(
        banks.dofs, beta[:, None, None]*parent.cohort_stiffness, banks.ndof)
    local = u[banks.dofs]
    e0 = e0+beta[:, None, None]*np.einsum("cqai,ci->cqa", parent.B0, local)
    e1 = e1+beta[:, None, None]*np.einsum("cqai,ci->cqa", parent.B1, local)
    mean = e0+parent.inverse_scale_mean[:, None, None]*e1
    C, variance = parent.elasticity_pa, parent.inverse_scale_variance
    stress = np.einsum("cab,cqb->cqa", C, mean)
    inverse_stress = np.einsum("cab,cqb->cqa", C, e1)
    density = .5*(np.einsum("cqa,cqa->cq", mean, stress)
        +variance[:, None]*np.einsum("cqa,cqa->cq", e1, inverse_stress))
    energy = float(np.sum(parent.volume_m3[:, None]*density)/3)
    Bmean = parent.B0+parent.inverse_scale_mean[:, None, None, None]*parent.B1
    local_force = (np.einsum("cqai,cqa->ci", Bmean, stress)
        +variance[:, None]*np.einsum("cqai,cqa->ci", parent.B1, inverse_stress))*parent.volume_m3[:, None]/3
    force = np.zeros(banks.ndof)
    np.add.at(force, banks.dofs.ravel(), local_force.ravel())
    inverse_angular = np.linalg.inv(ANGULAR_BARYCENTRIC)
    corner0 = np.einsum("vq,cqa->cva", inverse_angular, e0)
    corner1 = np.einsum("vq,cqa->cva", inverse_angular, e1)
    endpoints = np.stack((parent.outer_scale, parent.inner_scale), axis=1)
    corners = corner0[:, None, :, :]+corner1[:, None, :, :]/endpoints[:, :, None, None]
    maximum = float(np.max(np.abs(np.linalg.eigvalsh(engineering_strain_to_tensor(corners))), initial=0.))
    return GenesisVolumeBankResponse(*map(_immutable, (e0, e1, mean.mean(axis=1),
        stress.mean(axis=1))), energy, _immutable(force), K, maximum)
