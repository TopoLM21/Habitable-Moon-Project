"""Prospective six-strain volume mechanics of the actual Genesis cohorts.

The current mass-exact conical placement is an explicit new geometric
closure.  Every source elastic tensor, prestress and volume is inherited;
neither stress-free replacement nor historical 3-D recovery is performed.

On a top material triangle X=sum(N_a X_a), the displacement is
u=sum(N_a (u_a+(lambda-1)*p_a)), with six Cartesian metre DOFs per vertex.
Its engineering strain operator is B0+B1/lambda.  Three degree-two angular
points and analytic radial moments integrate its quadratic energy exactly.
This small-strain operator alone does not certify finite motion, contact,
thermal/work closure, material admission or a complete tectonic model.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json

import numpy as np
from scipy import sparse


STRAIN_COMPONENTS = ("xx", "yy", "zz", "gamma_xy", "gamma_xz", "gamma_yz")
ANGULAR_BARYCENTRIC = np.array(((2/3, 1/6, 1/6), (1/6, 2/3, 1/6),
                               (1/6, 1/6, 2/3)), dtype=float)
FORMAT = "genesis-inherited-conical-volume-operator-0.1"


def _immutable(value, dtype=float):
    a = np.asarray(value, dtype=dtype)
    # A bytes owner, unlike setflags(False) alone, cannot be made writable.
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def isotropic_elasticity(young_modulus_pa, poisson_ratio):
    """Stress [xx,yy,zz,xy,xz,yz] conjugate to six engineering strains."""
    E, nu = float(young_modulus_pa), float(poisson_ratio)
    if not np.isfinite((E, nu)).all() or E <= 0 or not -1 < nu < .5:
        raise ValueError("Volume elasticity requires a stable finite isotropic tensor")
    mu, lam = E/(2*(1+nu)), E*nu/((1+nu)*(1-2*nu))
    C = np.zeros((6, 6))
    C[:3, :3] = lam
    C[np.arange(3), np.arange(3)] += 2*mu
    C[3:, 3:] = np.eye(3)*mu
    return C


def tensor_to_engineering_strain(tensor):
    a = np.asarray(tensor, float)
    if a.shape[-2:] != (3, 3) or not np.isfinite(a).all():
        raise ValueError("Volume elastic strain requires finite symmetric 3-D tensors")
    if not np.allclose(a, a.swapaxes(-1, -2), rtol=0, atol=2e-13):
        raise ValueError("Volume elastic strain tensor is not symmetric")
    return np.stack((a[..., 0, 0], a[..., 1, 1], a[..., 2, 2],
                     2*a[..., 0, 1], 2*a[..., 0, 2], 2*a[..., 1, 2]), axis=-1)


def engineering_strain_to_tensor(strain):
    a = np.asarray(strain, float)
    if a.shape[-1:] != (6,) or not np.isfinite(a).all():
        raise ValueError("Volume strain requires six finite engineering components")
    result = np.zeros(a.shape[:-1]+(3, 3))
    result[..., 0, 0], result[..., 1, 1], result[..., 2, 2] = a[..., 0], a[..., 1], a[..., 2]
    for component, i, j in ((3, 0, 1), (4, 0, 2), (5, 1, 2)):
        result[..., i, j] = result[..., j, i] = a[..., component]/2
    return result


def _sym_gradient_column(vector, gradient):
    """Engineering sym(vector outer gradient), with no engineering shear loss."""
    v, g = np.asarray(vector), np.asarray(gradient)
    return np.array((v[0]*g[0], v[1]*g[1], v[2]*g[2],
                     v[0]*g[1]+v[1]*g[0], v[0]*g[2]+v[2]*g[0],
                     v[1]*g[2]+v[2]*g[1]))


def conical_strain_operators(top_triangle_m, basis_xyz, barycentric):
    """Return B0,B1 for literal physical u,p Cartesian metre amplitudes.

    Local strains use the inherited proper source face basis.  Coordinates
    measure from the planet centre; the normal plane offset h is necessary
    for the derivative of the true conical map.
    """
    X, Q, N = (np.asarray(value, float) for value in
               (top_triangle_m, basis_xyz, barycentric))
    if X.shape != (3, 3) or Q.shape != (3, 3) or N.ndim != 2 or N.shape[1] != 3:
        raise ValueError("Conical strain operator geometry/basis/barycentric shape mismatch")
    if not np.isfinite(X).all() or not np.isfinite(Q).all() or not np.isfinite(N).all():
        raise ValueError("Conical strain operator geometry must be finite")
    if not np.allclose(Q.T@Q, np.eye(3), rtol=0, atol=2e-12) or np.linalg.det(Q) <= 0:
        raise ValueError("Conical strain operator requires a proper orthonormal basis")
    if np.any(N < 0) or not np.allclose(N.sum(axis=1), 1., rtol=0, atol=2e-14):
        raise ValueError("Conical angular points must lie in the material triangle")
    local = X@Q
    h = float(np.mean(local[:, 2]))
    if h <= 0 or not np.allclose(local[:, 2], h, rtol=2e-12, atol=2e-9):
        raise ValueError("Conical top plane must have a positive consistent normal offset")
    design = np.column_stack((np.ones(3), local[:, :2]))
    if np.linalg.det(design) <= 0:
        raise ValueError("Conical top triangle must retain outward orientation")
    gradients = np.linalg.inv(design)[1:]
    angular_xy = N@local[:, :2]
    B0, B1 = np.zeros((len(N), 6, 18)), np.zeros((len(N), 6, 18))
    for point in range(len(N)):
        for vertex in range(3):
            g = np.r_[gradients[:, vertex], -angular_xy[point]@gradients[:, vertex]/h]
            normal_gradient = np.array((0., 0., N[point, vertex]/h))
            for axis in range(3):
                local_vector = Q[axis, :]
                column = _sym_gradient_column(local_vector, g)
                B1[point, :, 6*vertex+axis] = column
                B0[point, :, 6*vertex+3+axis] = column+_sym_gradient_column(local_vector, normal_gradient)
                B1[point, :, 6*vertex+3+axis] = -column
    return B0, B1


def conical_radial_moments(outer_scale, inner_scale):
    """Mean(1/lambda), variance(1/lambda), including thin-cell stability."""
    a, b = float(outer_scale), float(inner_scale)
    if not np.isfinite((a, b)).all() or not 0 < b < a <= 1:
        raise ValueError("Conical live material interval needs positive ordered scales")
    denominator = a*a+a*b+b*b
    mean = 1.5*(a+b)/denominator
    variance = .75*((a-b)/denominator)**2
    return mean, variance


@dataclass(frozen=True, slots=True)
class GenesisVolumeOperator:
    geometry: object
    source_checkpoint_sha256: str
    source_identity: str
    time_myr: float
    ndof: int
    dofs: np.ndarray
    cohort_ids: np.ndarray
    face_indices: np.ndarray
    volume_m3: np.ndarray
    basis_xyz: np.ndarray
    elasticity_pa: np.ndarray
    source_elastic_strain: np.ndarray | None
    source_stress_pa: np.ndarray | None
    B0: np.ndarray
    B1: np.ndarray
    vertex_B0: np.ndarray
    vertex_B1: np.ndarray
    inverse_scale_mean: np.ndarray
    inverse_scale_variance: np.ndarray
    outer_scale: np.ndarray
    inner_scale: np.ndarray
    cohort_stiffness: np.ndarray
    stiffness: object
    initial_force: np.ndarray | None
    source_elastic_energy_j: float | None

    @property
    def full_historical_three_dimensional_memory_recovered(self):
        return False

    @property
    def physical_force_work_admission(self):
        return False

    @property
    def source_geometry(self):
        return self.geometry

    @property
    def K(self):
        return self.stiffness

    @property
    def force(self):
        return self.initial_force

    @property
    def C(self):
        return self.elasticity_pa

    @property
    def volume(self):
        return self.volume_m3

    @property
    def s1(self):
        return self.inverse_scale_mean

    @property
    def s2(self):
        return _immutable(self.inverse_scale_mean**2+self.inverse_scale_variance)

    @property
    def radial_variance(self):
        return self.inverse_scale_variance

    @property
    def memory0(self):
        if self.source_elastic_strain is None:
            raise ValueError("Updated material geometry requires its prospectively calculated memory")
        return _immutable(np.broadcast_to(self.source_elastic_strain[:, None, :],
                          (len(self.cohort_ids), 3, 6)))

    @property
    def memory1(self):
        if self.source_elastic_strain is None:
            raise ValueError("Updated material geometry requires its prospectively calculated memory")
        return _immutable(np.zeros((len(self.cohort_ids), 3, 6)))

    @property
    def requires_explicit_memory(self):
        return self.source_elastic_strain is None


def _assemble_matrix(dofs, local, ndof):
    width = dofs.shape[1]
    row = np.repeat(dofs, width, axis=1).ravel()
    col = np.tile(dofs, (1, width)).ravel()
    K = sparse.coo_matrix((local.ravel(), (row, col)), shape=(ndof, ndof)).tocsr()
    K.sum_duplicates()
    K.data = _immutable(K.data)
    K.indices = _immutable(K.indices, K.indices.dtype)
    K.indptr = _immutable(K.indptr, K.indptr.dtype)
    return K


def assemble_genesis_volume_operator(source_packet, geometry=None):
    """Build the volume operator only after replay of the actual source bytes."""
    from .genesis_bulk_export import validate_genesis_bulk_material
    from .genesis_volume_geometry import (build_genesis_volume_geometry,
                                         validate_genesis_volume_geometry)
    packet = validate_genesis_bulk_material(source_packet)
    if packet["instantaneous_cohort_plane_stress_3d"] is None:
        raise ValueError("Volume continuation requires mechanically activated source cohort memory")
    geometry = (build_genesis_volume_geometry(packet) if geometry is None else
                validate_genesis_volume_geometry(geometry, packet))
    data = packet["instantaneous_cohort_plane_stress_3d"]
    ids = np.asarray(data["source_cohort_ids"], dtype=int)
    cohorts = geometry.cohorts
    if tuple(ids) != tuple(c.id for c in cohorts):
        raise ValueError("Volume geometry and source elastic memory have different cohort IDs")
    face = np.asarray(data["original_face_indices"], dtype=int)
    volume = np.asarray([c.volume_m3 for c in cohorts])
    if not np.allclose(volume, data["live_volume_m3"], rtol=2e-13, atol=0):
        raise ValueError("Volume geometry changed inherited material volumes")
    shell = packet["source"]["parameters"]["shell"]
    damage = np.asarray(packet["current_shell_memory"]["damage"])[face]
    residual = shell["residual_stiffness"]
    factor = residual+(1-residual)*(1-damage)**2
    C = factor[:, None, None]*isotropic_elasticity(shell["young_modulus_pa"], shell["poisson_ratio"])
    elastic = tensor_to_engineering_strain(data["elastic_strain_local"])
    stress = np.einsum("cij,cj->ci", C, elastic)
    expected_stress = np.asarray(data["stress_local_pa"])
    stress_tensor = engineering_strain_to_tensor(stress)
    stress_tensor[:, 0, 1] *= 2; stress_tensor[:, 1, 0] *= 2
    stress_tensor[:, 0, 2] *= 2; stress_tensor[:, 2, 0] *= 2
    stress_tensor[:, 1, 2] *= 2; stress_tensor[:, 2, 1] *= 2
    if not np.allclose(stress_tensor, expected_stress, rtol=3e-12, atol=1e-5):
        raise ValueError("Inherited volume prestress differs from the real constitutive source")
    frames = np.asarray(data["basis_xyz"])
    assembly = _construct_operator(geometry, C, frames, elastic)
    if not np.isclose(assembly.source_elastic_energy_j, data["elastic_energy_j"], rtol=3e-12, atol=1e-6):
        raise ValueError("Volume transfer changed inherited cohort elastic energy")
    # Keep the actual persisted constitutive ledger, rather than a rebuilt
    # exporter tensor reduction. Both independently checked 3-D contractions
    # differ from this only by floating summation.
    persisted_energy = float(packet["cohort_mechanics"]["state"]["elastic_energy_j"])
    if not np.isclose(assembly.source_elastic_energy_j, persisted_energy, rtol=3e-12, atol=1e-6):
        raise ValueError("Volume transfer changed the persisted constitutive energy")
    return replace(assembly, source_elastic_energy_j=persisted_energy)


def _construct_operator(geometry, C, frames, elastic=None):
    cohorts = geometry.cohorts
    ids = np.asarray([c.id for c in cohorts], int)
    face = np.asarray([c.face_index for c in cohorts], int)
    volume = np.asarray([c.volume_m3 for c in cohorts])
    stress = None if elastic is None else np.einsum("cij,cj->ci", C, elastic)
    B0, B1, V0, V1, radial = [], [], [], [], []
    for c, Q in zip(cohorts, frames):
        b0, b1 = conical_strain_operators(c.top_triangle_m, Q, ANGULAR_BARYCENTRIC)
        v0, v1 = conical_strain_operators(c.top_triangle_m, Q, np.eye(3))
        B0.append(b0); B1.append(b1); V0.append(v0); V1.append(v1)
        radial.append(conical_radial_moments(c.outer_scale, c.inner_scale))
    B0, B1, V0, V1, radial = map(np.asarray, (B0, B1, V0, V1, radial))
    mean, variance = radial.T
    Bmean = B0+mean[:, None, None, None]*B1
    local_k = (np.einsum("cqai,cab,cqbj->cij", Bmean, C, Bmean)
               +variance[:, None, None]*np.einsum("cqai,cab,cqbj->cij", B1, C, B1))*volume[:, None, None]/3
    faces = np.asarray(geometry.faces, dtype=int)
    dofs = (6*faces[face, :, None]+np.arange(6)).reshape(-1, 18)
    ndof = 6*len(geometry.vertices_m)
    force, energy = None, None
    if elastic is not None:
        local_f = np.einsum("cqai,ca->ci", Bmean, stress)*volume[:, None]/3
        force = np.zeros(ndof)
        np.add.at(force, dofs.ravel(), local_f.ravel())
        force = _immutable(force)
        energy = float(.5*np.einsum("c,ci,cij,cj->", volume, elastic, C, elastic))
    return GenesisVolumeOperator(geometry, geometry.source_checkpoint_sha256,
        geometry.source_identity, geometry.time_myr, ndof,
        *(_immutable(a, int) for a in (dofs, ids, face)),
        *map(_immutable, (volume, frames, C)),
        None if elastic is None else _immutable(elastic),
        None if stress is None else _immutable(stress),
        *map(_immutable, (B0, B1, V0, V1, mean, variance,
                         [c.outer_scale for c in cohorts], [c.inner_scale for c in cohorts], local_k)),
        _assemble_matrix(dofs, local_k, ndof), force, energy)


def assemble_updated_volume_operator(source_assembly, new_geometry, damage):
    """Update live material integration without inventing the newborn memory.

    Geometry must replay its complete accepted bulk-birth continuation and
    retain the authentic current reference vertices, material topology and
    source.  The caller subsequently supplies the complete calculated
    six-strain memory of retained and stress-free newly activated material.
    """
    from .genesis_volume_geometry import validate_genesis_volume_geometry
    if type(source_assembly) is not GenesisVolumeOperator:
        raise ValueError("Updated volume mechanics requires its genuine source operator")
    geometry = validate_genesis_volume_geometry(new_geometry)
    source = source_assembly.geometry
    for name in ("source_checkpoint_sha256", "source_identity", "vertices_m", "faces",
                 "column_mass_kg", "density_kg_m3", "source_packet_json"):
        if getattr(source, name) != getattr(geometry, name):
            raise ValueError("Updated volume geometry changed the original reference or source")
    if geometry.time_myr < source.time_myr:
        raise ValueError("Updated volume geometry cannot move its material clock backwards")
    if source.runtime_birth_history_json is not None:
        from .genesis_bulk_birth import bulk_birth_from_dict, validate_bulk_birth_continuation
        if geometry.runtime_birth_history_json is None:
            raise ValueError("Updated volume mechanics cannot discard its accepted birth continuation")
        validate_bulk_birth_continuation(
            bulk_birth_from_dict(json.loads(geometry.runtime_birth_history_json)),
            bulk_birth_from_dict(json.loads(source.runtime_birth_history_json)))
    d = np.asarray(damage, float)
    if d.shape != (len(geometry.faces),) or not np.isfinite(d).all() or np.any((d < 0) | (d > 1)):
        raise ValueError("Updated volume mechanics requires finite damage on every material face")
    packet = json.loads(geometry.source_packet_json)
    shell = packet["source"]["parameters"]["shell"]
    original_basis = packet["current_geometry"]["face_frames_xyz"]
    face_basis = np.asarray(original_basis, float)
    face_basis = np.concatenate((face_basis, np.cross(face_basis[:, :, 0], face_basis[:, :, 1])[:, :, None]), axis=2)
    face = np.asarray([c.face_index for c in geometry.cohorts], int)
    factor = shell["residual_stiffness"]+(1-shell["residual_stiffness"])*(1-d[face])**2
    C = factor[:, None, None]*isotropic_elasticity(shell["young_modulus_pa"], shell["poisson_ratio"])
    return _construct_operator(geometry, C, face_basis[face])


@dataclass(frozen=True, slots=True)
class GenesisVolumeResponse:
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


def algorithmic_stiffness(assembly, strain_factors):
    """Consistent physical-force tangent for uniform-per-cohort Maxwell beta."""
    beta = np.asarray(strain_factors, float)
    if beta.shape != assembly.volume_m3.shape or not np.isfinite(beta).all() or np.any(beta < 0):
        raise ValueError("Volume algorithmic factors must be finite nonnegative per cohort")
    if np.array_equal(beta, np.ones_like(beta)):
        return assembly.stiffness
    return _assemble_matrix(assembly.dofs, beta[:, None, None]*assembly.cohort_stiffness,
                            assembly.ndof)


def evaluate_genesis_volume_operator(assembly, displacement_m, *,
                                    elastic_constant=None, elastic_inverse_scale=None,
                                    strain_factors=None):
    """Evaluate inherited or prospectively retained affine strain memory.

    Optional memory coefficients describe e=e_constant+e_inverse/lambda at
    the three angular points.  Their affine angular variation is retained,
    allowing subsequent uniform-per-cohort Maxwell relaxation without
    projecting the detailed state back to a depth average.
    """
    if type(assembly) is not GenesisVolumeOperator:
        raise ValueError("An assembled actual Genesis volume operator is required")
    u = np.asarray(displacement_m, float)
    if u.shape != (assembly.ndof,) or not np.isfinite(u).all():
        raise ValueError("Volume displacement must contain every finite physical metre DOF")
    shape = (len(assembly.cohort_ids), 3, 6)
    if assembly.requires_explicit_memory and (elastic_constant is None or elastic_inverse_scale is None):
        raise ValueError("Updated live volume requires complete prospectively calculated elastic memory")
    e0 = (np.broadcast_to(assembly.source_elastic_strain[:, None, :], shape).copy()
          if elastic_constant is None else np.asarray(elastic_constant, float))
    e1 = np.zeros(shape) if elastic_inverse_scale is None else np.asarray(elastic_inverse_scale, float)
    if e0.shape != shape or e1.shape != shape or not np.isfinite(e0).all() or not np.isfinite(e1).all():
        raise ValueError("Volume elastic memory coefficient shape/finite mismatch")
    beta = np.ones(len(assembly.cohort_ids)) if strain_factors is None else np.asarray(strain_factors, float)
    tangent = algorithmic_stiffness(assembly, beta)
    local = u[assembly.dofs]
    e0 = e0+beta[:, None, None]*np.einsum("cqai,ci->cqa", assembly.B0, local)
    e1 = e1+beta[:, None, None]*np.einsum("cqai,ci->cqa", assembly.B1, local)
    mean = e0+assembly.inverse_scale_mean[:, None, None]*e1
    C, variance = assembly.elasticity_pa, assembly.inverse_scale_variance
    stress = np.einsum("cab,cqb->cqa", C, mean)
    inverse_stress = np.einsum("cab,cqb->cqa", C, e1)
    density = .5*(np.einsum("cqa,cqa->cq", mean, stress)
        +variance[:, None]*np.einsum("cqa,cqa->cq", e1, inverse_stress))
    energy = float(np.sum(assembly.volume_m3[:, None]*density)/3)
    Bmean = assembly.B0+assembly.inverse_scale_mean[:, None, None, None]*assembly.B1
    local_f = (np.einsum("cqai,cqa->ci", Bmean, stress)
        +variance[:, None]*np.einsum("cqai,cqa->ci", assembly.B1, inverse_stress))*assembly.volume_m3[:, None]/3
    force = np.zeros(assembly.ndof)
    np.add.at(force, assembly.dofs.ravel(), local_f.ravel())
    # Affine angular fields are reconstructed at triangle vertices.  The
    # spectral norm of strain is separately convex in N and 1/lambda, so all
    # six angular/radial corners bound the complete cell, not only quadrature.
    inverse_angular = np.linalg.inv(ANGULAR_BARYCENTRIC)
    corner0 = np.einsum("vq,cqa->cva", inverse_angular, e0)
    corner1 = np.einsum("vq,cqa->cva", inverse_angular, e1)
    endpoints = np.stack((assembly.outer_scale, assembly.inner_scale), axis=1)
    corners = corner0[:, None, :, :]+corner1[:, None, :, :]/endpoints[:, :, None, None]
    maximum = float(np.max(np.abs(np.linalg.eigvalsh(engineering_strain_to_tensor(corners))), initial=0.))
    return GenesisVolumeResponse(*map(_immutable, (e0, e1, mean.mean(axis=1),
        stress.mean(axis=1))), energy, _immutable(force), tangent, maximum)


# Short aliases keep orchestration code readable without hiding the source.
assembly_global = assemble_genesis_volume_operator
response = evaluate_genesis_volume_operator
