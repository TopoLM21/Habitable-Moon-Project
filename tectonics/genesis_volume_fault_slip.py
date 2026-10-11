"""Work-consistent finite-width weak-plane slip in the Genesis volume chart.

This prospective closure uses the exact interval-average traction of the
constant-driving Maxwell path. The original membrane law instead tests its
endpoint traction; their tangents and work must not be identified. A signed
shear increment is common to a face, while every cohort retains its own
Maxwell factors and full angular/radial elastic coefficients.

The caller supplies accepted-start physical plane and slip vectors. They are
frozen throughout the interval and all private mechanical iterations. This
small-strain material law neither transports a finite-deformation plane nor
chooses a dip, splits banks, solves contact, or declares a planet fault.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import math

import numpy as np
from scipy import sparse

from .genesis import SECONDS_PER_MYR
from .genesis_fault_law import WeakPlaneParameters
from .genesis_volume_loading import (
    VolumeLoading, VolumeResponse, average_volume_loading_coefficients,
    average_volume_loading_stiffness_factors, coefficient_inner_product,
    evaluate_volume_loading,
)


FORMAT = "genesis-volume-average-traction-weak-plane-slip-0.1"
TRACTION_CLOSURE = "exact_interval_average_constant_rate_Maxwell_path"
ORIENTATION_CLOSURE = "accepted_start_world_plane_frozen_over_small_strain_interval"
WORK_SCOPE = "physical_total_work_minus_finite_width_slip_and_Maxwell_losses"


def _immutable(value, dtype=float):
    a = np.asarray(value, dtype=dtype)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _array(value, shape, name):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"Volume slip {name} shape or finite values mismatch")
    return a


def _masked_plane_vectors(normal, tangent, active):
    nf = len(active)
    n = _array(normal, (nf, 3), "world plane normal")
    m = _array(tangent, (nf, 3), "world slip direction")
    if (not np.allclose(np.sum(n[active]**2, axis=1), 1., rtol=0, atol=2e-12)
            or not np.allclose(np.sum(m[active]**2, axis=1), 1., rtol=0, atol=2e-12)
            or not np.allclose(np.sum(n[active]*m[active], axis=1), 0., rtol=0, atol=2e-12)):
        raise ValueError("Active volume slip requires orthonormal world plane/slip vectors")
    return n, m


def _isotropic_matrix_check(c):
    """Normal invariance is a mathematical requirement, not an assumption."""
    count = len(c)
    mu, lam = c[:, 3, 3], c[:, 0, 1]
    expected = np.zeros((count, 6, 6))
    expected[:, :3, :3] = lam[:, None, None]
    expected[:, np.arange(3), np.arange(3)] += 2*mu[:, None]
    expected[:, np.arange(3, 6), np.arange(3, 6)] = mu[:, None]
    scale = np.maximum(np.max(np.abs(c), axis=(1, 2)), 1.)
    if (np.any(mu <= 0) or np.any(3*lam+2*mu <= 0)
            or np.any(np.max(np.abs(c-expected), axis=(1, 2)) > 3e-12*scale)):
        raise ValueError("Volume slip requires stable full isotropic cohort elasticity")


class VolumeSlipIncrementBoundary(ValueError):
    """A private candidate exceeds the unchanged source shear increment limit."""

    def __init__(self, shear_increment, limit):
        self.shear_increment = _immutable(shear_increment)
        self.limit = float(limit)
        super().__init__("Volume weak-plane shear increment exceeds the source limit; "
                         "the candidate must be rejected, not clipped")


@dataclass(frozen=True, slots=True)
class VolumeSlipLoading:
    loading: VolumeLoading
    parameters: WeakPlaneParameters
    face_indices: np.ndarray
    basis_xyz: np.ndarray
    plane_normal_world: np.ndarray
    slip_direction_world: np.ndarray
    active: np.ndarray
    damage: np.ndarray
    water: np.ndarray
    flow_direction: np.ndarray
    normal_projector: np.ndarray
    theta: np.ndarray
    face_volume_m3: np.ndarray
    friction: np.ndarray
    cohesion_pa: np.ndarray
    slip_modulus_pa: np.ndarray
    viscous_modulus_pa: float

    def __post_init__(self):
        for name in ("basis_xyz", "plane_normal_world", "slip_direction_world",
                     "damage", "water", "flow_direction", "normal_projector",
                     "theta", "face_volume_m3", "friction", "cohesion_pa",
                     "slip_modulus_pa"):
            object.__setattr__(self, name, _immutable(getattr(self, name)))
        object.__setattr__(self, "face_indices", _immutable(self.face_indices, int))
        object.__setattr__(self, "active", _immutable(self.active, bool))


@dataclass(frozen=True, slots=True)
class VolumeSlipResponse:
    # Existing base mechanics records NET driver work. The surrounding slip
    # state must explicitly retain the separate physical/slip work ledgers.
    loading_response: VolumeResponse
    average_strain0: np.ndarray
    average_strain1: np.ndarray
    plastic_increment0: np.ndarray
    shear_increment: np.ndarray
    trial_shear_stress_pa: np.ndarray
    shear_stress_pa: np.ndarray
    normal_stress_pa: np.ndarray
    yield_strength_pa: np.ndarray
    yielding: np.ndarray
    friction_work_cell_j: np.ndarray
    viscous_work_cell_j: np.ndarray
    friction_work_j: float
    viscous_fault_work_j: float
    plastic_work_j: float
    physical_mechanical_work_j: float
    balance_residual_j: float
    prepared_loading: VolumeSlipLoading

    def __post_init__(self):
        local = self.loading_response
        if type(local) is not VolumeResponse or type(self.prepared_loading) is not VolumeSlipLoading:
            raise ValueError("Volume slip requires a typed net loading response")
        object.__setattr__(self, "loading_response", replace(local,
            strain0=_immutable(local.strain0), strain1=_immutable(local.strain1)))
        for name in ("average_strain0", "average_strain1", "plastic_increment0",
                     "shear_increment", "trial_shear_stress_pa", "shear_stress_pa",
                     "normal_stress_pa", "yield_strength_pa", "friction_work_cell_j",
                     "viscous_work_cell_j"):
            object.__setattr__(self, name, _immutable(getattr(self, name)))
        object.__setattr__(self, "yielding", _immutable(self.yielding, bool))


@dataclass(frozen=True, slots=True)
class VolumeSlipForceTangent:
    force_n: np.ndarray
    tangent: object

    def __post_init__(self):
        object.__setattr__(self, "force_n", _immutable(self.force_n))
        matrix = self.tangent.copy().tocsr()
        matrix.sum_duplicates()
        matrix.data = _immutable(matrix.data)
        matrix.indices = _immutable(matrix.indices, matrix.indices.dtype)
        matrix.indptr = _immutable(matrix.indptr, matrix.indptr.dtype)
        object.__setattr__(self, "tangent", matrix)


def prepare_volume_fault_slip(loading, *, face_indices, basis_xyz,
                              plane_normal_world, slip_direction_world,
                              active, damage, water, parameters):
    """Prepare immutable source-parameter slip data for private trial solves.

    Unit vectors are physical Cartesian vectors, not face-angle components.
    Every cohort may have a different proper fixed orthonormal strain basis.
    The kernel never activates an inactive plane or modifies source parameters.
    """
    if type(loading) is not VolumeLoading or type(parameters) is not WeakPlaneParameters:
        raise ValueError("Volume slip requires prepared loading and source weak-plane parameters")
    parameters.validate()
    # VolumeLoading already isolates producer trials, but its owned arrays
    # permit setflags(write=True). A slip preparation owns a further literal
    # snapshot with immutable bytes backing, so later caller/Newton mutations
    # cannot change the frozen interval or invalidate its scalar/tangent data.
    loading = VolumeLoading(**{field.name: getattr(loading, field.name)
        for field in fields(VolumeLoading)})
    for name in ("temperatures_k", "old0", "old1", "thermal0", "memory0",
                 "memory1", "ratio", "beta", "elasticity_pa"):
        object.__setattr__(loading, name, _immutable(getattr(loading, name)))
    count = len(loading.after_ids)
    enabled = np.asarray(active)
    if enabled.ndim != 1 or enabled.dtype.kind != "b":
        raise ValueError("Volume slip active mask must be boolean [face]")
    nf = len(enabled)
    faces = np.asarray(face_indices)
    if (faces.shape != (count,) or faces.dtype.kind not in "iu"
            or np.any(faces < 0) or np.any(faces >= nf)):
        raise ValueError("Volume slip face indices must be integer [cohort] within face coverage")
    basis = _array(basis_xyz, (count, 3, 3), "cohort basis")
    if (not np.allclose(basis.transpose(0, 2, 1)@basis, np.eye(3), rtol=0, atol=2e-12)
            or np.any(np.linalg.det(basis) <= 0)):
        raise ValueError("Volume slip requires proper orthonormal cohort bases")
    normal, tangent = _masked_plane_vectors(plane_normal_world, slip_direction_world, enabled)
    d, w = (_array(v, (nf,), name) for v, name in ((damage, "damage"), (water, "water")))
    if np.any((d < 0) | (d > 1)) or np.any((w < 0) | (w > 1)):
        raise ValueError("Volume slip damage/water must be within [0, 1]")
    volume = _array(loading.radial_moments.volume_m3, (count,), "cohort volume")
    c = _array(loading.elasticity_pa, (count, 6, 6), "elasticity")
    if np.any(volume <= 0):
        raise ValueError("Volume slip live cohorts require positive volume")
    _isotropic_matrix_check(c)
    dt = loading.after_time_myr-loading.before_state.time_myr
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("Volume slip requires a positive shared interval")
    theta = average_volume_loading_stiffness_factors(loading)
    if not np.isfinite(theta).all() or np.any(theta <= 0) or np.any(theta > .5+1e-14):
        raise ValueError("Volume slip average Maxwell factors outside their physical range")
    local_n = np.einsum("cji,cj->ci", basis, normal[faces])
    local_m = np.einsum("cji,cj->ci", basis, tangent[faces])
    q = np.column_stack((local_m[:, 0]*local_n[:, 0],
                         local_m[:, 1]*local_n[:, 1],
                         local_m[:, 2]*local_n[:, 2],
                         local_m[:, 0]*local_n[:, 1]+local_m[:, 1]*local_n[:, 0],
                         local_m[:, 0]*local_n[:, 2]+local_m[:, 2]*local_n[:, 0],
                         local_m[:, 1]*local_n[:, 2]+local_m[:, 2]*local_n[:, 1]))
    p = np.column_stack((local_n[:, 0]**2, local_n[:, 1]**2, local_n[:, 2]**2,
                         2*local_n[:, 0]*local_n[:, 1],
                         2*local_n[:, 0]*local_n[:, 2],
                         2*local_n[:, 1]*local_n[:, 2]))
    q[~enabled[faces]] = 0.
    p[~enabled[faces]] = 0.
    face_volume = np.bincount(faces, weights=volume, minlength=nf)
    if np.any(enabled & (face_volume <= 0)):
        raise ValueError("An active volume slip plane requires live material")
    hardening = np.bincount(faces, weights=volume*theta*np.einsum("ci,cij,cj->c", q, c, q), minlength=nf)
    hardening = np.divide(hardening, face_volume, out=np.zeros(nf), where=face_volume > 0)
    friction = parameters.friction_dry+(parameters.friction_wet-parameters.friction_dry)*w
    cohesion = (parameters.cohesion_pa*(parameters.residual_cohesion_fraction
        +(1-parameters.residual_cohesion_fraction)*(1-d))
        *(1-(1-parameters.wet_cohesion_fraction)*w))
    zeta = parameters.viscosity_pa_s/(dt*SECONDS_PER_MYR)
    if not math.isfinite(zeta) or zeta <= 0:
        raise ValueError("Volume slip viscosity/time modulus must be positive and finite")
    return VolumeSlipLoading(loading, parameters, faces, basis, normal, tangent,
        enabled, d, w, q, p, theta, face_volume, friction, cohesion, hardening, zeta)


def _face_traction(prepared, mean0, mean1, projector):
    loading = prepared.loading
    mean = mean0.mean(axis=1)+np.asarray(loading.radial_moments.inverse_radius_mean)[:, None]*mean1.mean(axis=1)
    stress = np.einsum("cij,cj->ci", loading.elasticity_pa, mean)
    volume = np.asarray(loading.radial_moments.volume_m3)
    total = np.bincount(prepared.face_indices,
        weights=volume*np.einsum("ci,ci->c", projector, stress),
        minlength=len(prepared.active))
    return np.divide(total, prepared.face_volume_m3, out=np.zeros_like(total),
                     where=prepared.face_volume_m3 > 0)


def evaluate_volume_fault_slip(prepared, increment0, increment1):
    """Evaluate exact averaged-traction viscous/Coulomb slip without mutation.

    The strength uses modeled time-average compression. Water only changes
    cohesion/friction, never a synthesized pore pressure. Non-dilatant slip
    leaves normal traction invariant under the required isotropic elasticity.
    Maximal source shear increments are rejected, never silently clipped.
    """
    if type(prepared) is not VolumeSlipLoading:
        raise ValueError("Volume slip evaluation requires its typed preparation")
    loading = prepared.loading
    shape = (len(loading.after_ids), 3, 6)
    inc0, inc1 = (_array(value, shape, name) for value, name in (
        (increment0, "mechanical increment0"), (increment1, "mechanical increment1")))
    trial0, trial1 = average_volume_loading_coefficients(loading, inc0, inc1)
    tau_trial = _face_traction(prepared, trial0, trial1, prepared.flow_direction)
    normal = _face_traction(prepared, trial0, trial1, prepared.normal_projector)
    strength = prepared.cohesion_pa+prepared.friction*np.maximum(-normal, 0.)
    overstress = np.maximum(np.abs(tau_trial)-strength, 0.)
    yielding = prepared.active & prepared.parameters.enabled & (overstress > 0)
    denominator = prepared.slip_modulus_pa+prepared.viscous_modulus_pa
    gamma = np.zeros(len(prepared.active))
    gamma[yielding] = np.sign(tau_trial[yielding])*overstress[yielding]/denominator[yielding]
    if np.any(np.abs(gamma) > prepared.parameters.max_shear_increment):
        raise VolumeSlipIncrementBoundary(gamma, prepared.parameters.max_shear_increment)
    plastic0 = np.broadcast_to(gamma[prepared.face_indices, None, None]
        *prepared.flow_direction[:, None, :], shape)
    effective0 = inc0-plastic0
    local = evaluate_volume_loading(loading, effective0, inc1)
    mean0, mean1 = average_volume_loading_coefficients(loading, effective0, inc1)
    tau = _face_traction(prepared, mean0, mean1, prepared.flow_direction)
    final_normal = _face_traction(prepared, mean0, mean1, prepared.normal_projector)
    traction_scale = np.maximum.reduce((np.abs(normal), np.abs(tau_trial), strength, np.ones_like(normal)))
    if np.any(np.abs(final_normal-normal) > 8e-12*traction_scale):
        raise ValueError("Volume nondilatant slip changed isotropic normal traction")
    friction_cell = prepared.face_volume_m3*strength*np.abs(gamma)
    viscous_cell = prepared.face_volume_m3*prepared.viscous_modulus_pa*gamma**2
    friction_work, viscous_work = float(friction_cell.sum()), float(viscous_cell.sum())
    plastic_work = friction_work+viscous_work
    traction_work = float(np.sum(prepared.face_volume_m3*gamma*tau))
    physical_work = float(coefficient_inner_product(inc0, inc1, mean0, mean1,
        loading.elasticity_pa, loading.radial_moments).sum())
    scale = max(local.elastic_energy_j, loading.before_state.elastic_energy_j,
                abs(physical_work), plastic_work, local.viscous_dissipation_j, 1.)
    slip_scale = max(abs(traction_work), plastic_work, 1.)
    work_scale = max(abs(physical_work), abs(local.mechanical_work_j), plastic_work, 1.)
    if (abs(traction_work-plastic_work) > 5e-12*slip_scale
            or abs(physical_work-local.mechanical_work_j-plastic_work) > 5e-12*work_scale):
        raise ValueError("Volume weak-plane average traction/work split mismatch")
    residual = local.elastic_energy_j-loading.before_state.elastic_energy_j-(
        physical_work+local.thermal_work_j-local.viscous_dissipation_j-plastic_work
        +local.damage_energy_change_j-local.remelt_energy_removed_j)
    if abs(residual) > 5e-12*scale:
        raise ValueError("Volume full Maxwell/weak-plane work identity mismatch")
    return VolumeSlipResponse(local, mean0, mean1, plastic0, gamma, tau_trial, tau,
        normal, strength, yielding, friction_cell, viscous_cell, friction_work,
        viscous_work, plastic_work, physical_work, residual, prepared)


def assemble_volume_fault_average_force_tangent(assembly, prepared, response):
    """Return actual average-stress force and its nonassociated sparse tangent.

    The base tangent uses theta, not endpoint beta. On yielding compressed
    planes the Coulomb normal-stress derivative makes the tangent nonsymmetric;
    an orchestration layer must not use a positive-definite symmetric solve.
    """
    from .genesis_volume_operator import (
        GenesisVolumeOperator, algorithmic_stiffness, evaluate_genesis_volume_operator,
    )
    if (type(assembly) is not GenesisVolumeOperator
            or type(prepared) is not VolumeSlipLoading
            or type(response) is not VolumeSlipResponse):
        raise ValueError("Volume slip force requires typed operator, loading and response")
    if response.prepared_loading is not prepared:
        raise ValueError("Volume slip response is bound to a different frozen preparation")
    loading = prepared.loading
    if (tuple(int(i) for i in assembly.cohort_ids) != loading.after_ids
            or not np.array_equal(assembly.face_indices, prepared.face_indices)
            or not np.array_equal(assembly.basis_xyz, prepared.basis_xyz)
            or not np.array_equal(assembly.volume_m3, loading.radial_moments.volume_m3)
            or not np.array_equal(assembly.inverse_scale_mean, loading.radial_moments.inverse_radius_mean)
            or not np.array_equal(assembly.inverse_scale_variance, loading.radial_moments.inverse_radius_variance)
            or not np.array_equal(assembly.elasticity_pa, loading.elasticity_pa)):
        raise ValueError("Volume slip average force lost exact operator/loading bindings")
    mean_response = evaluate_genesis_volume_operator(assembly, np.zeros(assembly.ndof),
        elastic_constant=response.average_strain0, elastic_inverse_scale=response.average_strain1)
    base = algorithmic_stiffness(assembly, prepared.theta)
    bmean = assembly.B0+assembly.inverse_scale_mean[:, None, None, None]*assembly.B1
    cq = np.einsum("cij,cj->ci", loading.elasticity_pa, prepared.flow_direction)
    cp = np.einsum("cij,cj->ci", loading.elasticity_pa, prepared.normal_projector)
    weight = assembly.volume_m3*prepared.theta/3
    a_local = np.einsum("cqai,ca->ci", bmean, cq)*weight[:, None]
    n_local = np.einsum("cqai,ca->ci", bmean, cp)*weight[:, None]
    rows = np.repeat(prepared.face_indices, assembly.dofs.shape[1])
    shape = (len(prepared.active), assembly.ndof)
    a = sparse.coo_matrix((a_local.ravel(), (rows, assembly.dofs.ravel())), shape=shape).tocsr()
    n = sparse.coo_matrix((n_local.ravel(), (rows, assembly.dofs.ravel())), shape=shape).tocsr()
    denominator = prepared.face_volume_m3*(prepared.slip_modulus_pa+prepared.viscous_modulus_pa)
    inverse = np.divide(response.yielding.astype(float), denominator,
        out=np.zeros_like(denominator), where=denominator > 0)
    nonassociation = (np.sign(response.shear_increment)*prepared.friction
        *(response.normal_stress_pa < 0)*response.yielding)
    derivative = a+sparse.diags(nonassociation)@n
    tangent = base-a.T@sparse.diags(inverse)@derivative
    return VolumeSlipForceTangent(mean_response.force, tangent)
