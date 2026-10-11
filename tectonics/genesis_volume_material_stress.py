"""Observe calculated Genesis stress at explicit material coordinates.

The accepted elastic memory is an affine angular field with an inverse-radius
term.  It already contains every accepted mechanical increment: absolute
displacement is used for placement only, never added to this memory again.
This observer neither selects a fracture location nor supplies missing birth
chronology, a dip, toughness, a future birth traction or a release decision.
An untrusted planetary checkpoint must first pass the public slip replay.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import hashlib

import numpy as np

from .genesis_volume_checkpoint import _canonical
from .genesis_volume_operator import ANGULAR_BARYCENTRIC, engineering_strain_to_tensor
from .genesis_volume_slip_checkpoint import volume_slip_checkpoint_to_dict
from .genesis_volume_slip_dynamics import VolumeSlipModel, VolumeSlipState


OBSERVER_FORMAT = "genesis-volume-material-stress-observer-0.1"
FORMAT = "genesis-volume-material-stress-samples-0.1"
SCOPE = "accepted_endpoint_elastic_memory_at_explicit_material_coordinates_no_fracture_location"
STRAIN_INTERPOLATION = "affine_material_barycentric_and_inverse_reference_radial_scale"
PLACEMENT = "accepted_conical_A_P_displacement_on_original_material_mass_chart"
_FALSE_CLAIMS = ("history_reinitialized", "clock_advanced", "activation_accepted",
    "physical_creation_admitted", "physical_release_admitted",
    "physical_fault_contact_admission_claimed", "historical_three_dimensional_memory_recovered",
    "instantaneous_future_birth_traction_claimed")


class VolumeMaterialStressRefusal(ValueError):
    """A sample or snapshot is outside its accepted material context."""


def _numeric(value, name, *, integer=False):
    # Reject nested mixed booleans before numpy silently promotes their dtype.
    def boolean(v):
        return isinstance(v, (bool, np.bool_)) or (
            isinstance(v, (list, tuple)) and any(boolean(x) for x in v))
    if boolean(value):
        raise VolumeMaterialStressRefusal(f"Material stress {name} requires real numbers")
    try:
        a = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise VolumeMaterialStressRefusal(f"Material stress {name} requires a numeric array") from error
    if a.dtype.kind not in ("iu" if integer else "iuf"):
        raise VolumeMaterialStressRefusal(f"Material stress {name} has invalid numeric type")
    if integer and a.dtype.kind == "u" and np.any(a > np.iinfo(np.int64).max):
        raise VolumeMaterialStressRefusal(f"Material stress {name} is outside the integer ID range")
    a = np.asarray(a, dtype=np.int64 if integer else float)
    if not np.isfinite(a).all():
        raise VolumeMaterialStressRefusal(f"Material stress {name} requires finite values")
    return a


def _owned(value, dtype=float):
    a = np.asarray(value, dtype=dtype)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _digest(body):
    return hashlib.sha256(_canonical(body)).hexdigest()


def _geometry_payload(geometry):
    # The original source is identified by the full parent checkpoint seal.
    # Do not duplicate its embedded formation archive in a spatial observer.
    return {f.name: getattr(geometry, f.name) for f in fields(geometry)
        if f.name not in ("source_packet_json", "runtime_birth_history_json", "cohorts")} | {
        "cohorts": [{f.name: getattr(c, f.name) for f in fields(c)}
                    for c in geometry.cohorts]}


@dataclass(frozen=True, slots=True)
class BoundVolumeMaterialStress:
    parent_checkpoint_sha256: str
    formation_checkpoint_sha256: str
    source_identity: str
    time_myr: float
    accepted_steps: int
    accepted_step_sizes_myr: tuple
    geometry: object
    displacement_m: np.ndarray
    velocity_m_s: np.ndarray
    cohort_ids: np.ndarray
    face_indices: np.ndarray
    face_damage: np.ndarray
    basis_xyz: np.ndarray
    elasticity_pa: np.ndarray
    elastic_constant: np.ndarray
    elastic_inverse_scale: np.ndarray
    binding_sha256: str

    def __post_init__(self):
        for name in ("displacement_m", "velocity_m_s", "face_damage", "basis_xyz",
                     "elasticity_pa", "elastic_constant", "elastic_inverse_scale"):
            object.__setattr__(self, name, _owned(getattr(self, name)))
        for name in ("cohort_ids", "face_indices"):
            object.__setattr__(self, name, _owned(getattr(self, name), np.int64))
        object.__setattr__(self, "accepted_step_sizes_myr", tuple(self.accepted_step_sizes_myr))

    @property
    def e0(self):
        return self.elastic_constant

    @property
    def e1(self):
        return self.elastic_inverse_scale

    def __deepcopy__(self, memo):
        return self


def _observer_body(observer):
    return dict(format=OBSERVER_FORMAT, scope=SCOPE,
        parent_checkpoint_sha256=observer.parent_checkpoint_sha256,
        formation_checkpoint_sha256=observer.formation_checkpoint_sha256,
        source_identity=observer.source_identity, time_myr=observer.time_myr,
        accepted_steps=observer.accepted_steps,
        accepted_step_sizes_myr=list(observer.accepted_step_sizes_myr),
        geometry=_geometry_payload(observer.geometry),
        **{name: getattr(observer, name).tolist() for name in (
            "displacement_m", "velocity_m_s", "cohort_ids", "face_indices",
            "face_damage", "basis_xyz", "elasticity_pa", "elastic_constant",
            "elastic_inverse_scale")},
        strain_interpolation_closure=STRAIN_INTERPOLATION, placement_closure=PLACEMENT,
        stress_scope="accepted_endpoint_not_interval_average_or_future_birth",
        selected_sharp_material_surface=None, source_creation_energy_law=None,
        creation_work_debited_from_Genesis_j=0.,
        **{name: False for name in _FALSE_CLAIMS})


def validate_volume_material_stress_observer(observer):
    """Validate immutable snapshot integrity; a seal alone is not source replay."""
    if type(observer) is not BoundVolumeMaterialStress:
        raise VolumeMaterialStressRefusal("An immutable accepted material stress observer is required")
    if _digest(_observer_body(observer)) != observer.binding_sha256:
        raise VolumeMaterialStressRefusal("Material stress observer immutable binding differs")
    return observer


def bind_volume_material_stress(model, state):
    """Bind once to a trusted, complete accepted Genesis runtime state."""
    if type(model) is not VolumeSlipModel or type(state) is not VolumeSlipState:
        raise VolumeMaterialStressRefusal("Material stress requires accepted Genesis slip model/state")
    parent = volume_slip_checkpoint_to_dict(model, state)
    assembly, _ = model._assembly(state.geometry, state.fault_history.damage)
    memories = state.mechanics.cohorts
    if tuple(c.id for c in memories) != tuple(int(i) for i in assembly.cohort_ids):
        raise VolumeMaterialStressRefusal("Accepted material stress cohort identities differ")
    observer = BoundVolumeMaterialStress(
        parent["checkpoint_sha256"], parent["source_checkpoint_sha256"],
        state.geometry.source_identity, state.time_myr, state.mechanics.accepted_steps,
        tuple(state.accepted_step_sizes_myr), state.geometry,
        state.displacement_m, state.velocity_m_s, assembly.cohort_ids,
        assembly.face_indices, state.fault_history.damage, assembly.basis_xyz,
        assembly.elasticity_pa, np.asarray([c.strain0 for c in memories]),
        np.asarray([c.strain1 for c in memories]), "")
    return replace(observer, binding_sha256=_digest(_observer_body(observer)))


def volume_material_stress_observer_to_dict(observer):
    validate_volume_material_stress_observer(observer)
    return {**_observer_body(observer), "binding_sha256": observer.binding_sha256}


def validate_volume_material_stress_observer_payload(raw, *, expected_observer):
    """Require the full recomputed runtime context, not an editable new seal."""
    expected = volume_material_stress_observer_to_dict(expected_observer)
    try:
        equal = type(raw) is dict and _canonical(raw) == _canonical(expected)
    except (TypeError, ValueError):
        equal = False
    if not equal:
        raise VolumeMaterialStressRefusal("Material stress observer differs from full accepted-source context")
    return expected


def _stress_tensor(stress):
    t = np.zeros(stress.shape[:-1]+(3, 3))
    t[..., 0, 0], t[..., 1, 1], t[..., 2, 2] = stress[..., 0], stress[..., 1], stress[..., 2]
    for component, i, j in ((3, 0, 1), (4, 0, 2), (5, 1, 2)):
        t[..., i, j] = t[..., j, i] = stress[..., component]
    return t


def _pointwise_fields(e0, e1, elasticity, basis, barycentric, radial_scale):
    weights = barycentric@np.linalg.inv(ANGULAR_BARYCENTRIC)
    elastic = np.einsum("nq,nqa->na", weights, e0+e1/radial_scale[:, None, None])
    stress = np.einsum("nab,nb->na", elasticity, elastic)
    strain_tensor, stress_tensor = engineering_strain_to_tensor(elastic), _stress_tensor(stress)
    world_strain = np.einsum("nij,njk,nlk->nil", basis, strain_tensor, basis)
    world_stress = np.einsum("nij,njk,nlk->nil", basis, stress_tensor, basis)
    return elastic, strain_tensor, world_strain, stress, stress_tensor, world_stress


@dataclass(frozen=True, slots=True)
class VolumeMaterialStressSamples:
    parent_checkpoint_sha256: str
    formation_checkpoint_sha256: str
    source_identity: str
    observer_binding_sha256: str
    time_myr: float
    cohort_ids: np.ndarray
    face_indices: np.ndarray
    owner_ids: tuple
    material_ids: tuple
    angular_barycentric: np.ndarray
    mass_coordinate_q: np.ndarray
    radial_scale: np.ndarray
    reference_position_m: np.ndarray
    current_position_m: np.ndarray
    elastic_strain_local6: np.ndarray
    elastic_strain_local_tensor: np.ndarray
    elastic_strain_world_tensor: np.ndarray
    stress_local6: np.ndarray
    stress_local_tensor: np.ndarray
    stress_world_tensor: np.ndarray
    elastic_energy_density_j_m3: np.ndarray
    normal_world: np.ndarray | None
    traction_world_pa: np.ndarray | None
    sample_sha256: str

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, np.ndarray):
                object.__setattr__(self, f.name, _owned(v, v.dtype))
        object.__setattr__(self, "owner_ids", tuple(self.owner_ids))
        object.__setattr__(self, "material_ids", tuple(self.material_ids))

    def __deepcopy__(self, memo):
        return self


def _sample_body(samples):
    body = {f.name: (getattr(samples, f.name).tolist()
        if isinstance(getattr(samples, f.name), np.ndarray) else getattr(samples, f.name))
        for f in fields(samples) if f.name != "sample_sha256"}
    body.update(format=FORMAT, scope=SCOPE, strain_interpolation_closure=STRAIN_INTERPOLATION,
        placement_closure=PLACEMENT,
        stress_scope="accepted_endpoint_not_interval_average_or_future_birth",
        supplied_normal_is_fracture_orientation=False,
        selected_sharp_material_surface=None, source_creation_energy_law=None,
        creation_work_debited_from_Genesis_j=0., **{name: False for name in _FALSE_CLAIMS})
    return body


def evaluate_volume_material_stress(observer, cohort_ids, angular_barycentric,
        mass_coordinate_q, *, normal_world=None):
    """Evaluate explicit points; scalar IDs/q or normals may broadcast.

    Angular coordinates are source-triangle material barycentric coordinates,
    not coordinates found by projecting a moved world point. q is the original
    material column mass coordinate, not equal depth or a mechanical DOF.
    Shared cohort boundaries keep the caller's explicit one-sided identity.
    An optional normal is a unit Cartesian probe; it does not select a fault.
    """
    validate_volume_material_stress_observer(observer)
    N = _numeric(angular_barycentric, "angular barycentric coordinates")
    if N.shape == (3,):
        N = N[None, :]
    if N.ndim != 2 or N.shape[1] != 3 or not len(N):
        raise VolumeMaterialStressRefusal("Material stress angular coordinates require nonempty (n,3)")
    if np.any(N < 0) or np.any(N > 1) or not np.allclose(N.sum(axis=1), 1., rtol=0, atol=32*np.finfo(float).eps):
        raise VolumeMaterialStressRefusal("Material stress angular coordinates leave their material triangle")
    n = len(N)
    ids = _numeric(cohort_ids, "cohort IDs", integer=True)
    q = _numeric(mass_coordinate_q, "mass coordinate q")
    if ids.ndim == 0:
        ids = np.repeat(ids, n)
    if q.ndim == 0:
        q = np.repeat(q, n)
    if ids.shape != (n,) or q.shape != (n,):
        raise VolumeMaterialStressRefusal("Material stress sample IDs and q must match angular sample count")
    indices = np.searchsorted(observer.cohort_ids, ids)
    if np.any(indices == len(observer.cohort_ids)) or np.any(observer.cohort_ids[indices] != ids):
        raise VolumeMaterialStressRefusal("Material stress requires explicit live cohort IDs")
    cohorts = [observer.geometry.cohorts[int(i)] for i in indices]
    lo, hi = np.array([c.q_lo for c in cohorts]), np.array([c.live_q_hi for c in cohorts])
    if np.any(q < lo) or np.any(q > hi):
        raise VolumeMaterialStressRefusal("Material stress mass coordinate q leaves its live cohort")
    triangle = np.asarray([c.top_triangle_m for c in cohorts])
    face = observer.face_indices[indices]
    mass = np.asarray(observer.geometry.column_mass_kg)[face]
    coefficient = 6*mass/(observer.geometry.density_kg_m3*np.linalg.det(triangle))
    cube = 1-coefficient*q
    if np.any(cube <= 0) or not np.isfinite(cube).all():
        raise VolumeMaterialStressRefusal("Material stress radial coordinate is outside its physical cone")
    radial = np.cbrt(cube)
    # Exact endpoints retain the same declared chart values as geometry.
    radial = np.where(q == lo, np.array([c.outer_scale for c in cohorts]), radial)
    radial = np.where(q == hi, np.array([c.inner_scale for c in cohorts]), radial)
    nodes = observer.displacement_m.reshape(-1, 6)[np.asarray(observer.geometry.faces)[face]]
    A, P = nodes[..., :3], nodes[..., 3:]
    reference = radial[:, None]*np.einsum("ni,nij->nj", N, triangle)
    current = (radial[:, None]*np.einsum("ni,nij->nj", N, triangle+P)
               +np.einsum("ni,nij->nj", N, A-P))
    values = _pointwise_fields(observer.e0[indices], observer.e1[indices],
        observer.elasticity_pa[indices], observer.basis_xyz[indices], N, radial)
    normals = traction = None
    if normal_world is not None:
        normals = _numeric(normal_world, "normal world")
        if normals.shape == (3,):
            normals = np.repeat(normals[None, :], n, axis=0)
        if normals.shape != (n, 3) or not np.allclose(np.linalg.norm(normals, axis=1), 1., rtol=0, atol=32*np.finfo(float).eps):
            raise VolumeMaterialStressRefusal("Material stress probe normals require explicit unit world vectors")
        traction = np.einsum("nij,nj->ni", values[5], normals)
    samples = VolumeMaterialStressSamples(
        observer.parent_checkpoint_sha256, observer.formation_checkpoint_sha256,
        observer.source_identity, observer.binding_sha256, observer.time_myr,
        ids, face, tuple(c.owner_id for c in cohorts), tuple(c.material_id for c in cohorts),
        N, q, radial, reference, current, *values,
        .5*np.einsum("ni,ni->n", values[0], values[3]), normals, traction, "")
    if not all(np.isfinite(getattr(samples, f.name)).all() for f in fields(samples)
               if isinstance(getattr(samples, f.name), np.ndarray)):
        raise VolumeMaterialStressRefusal("Material stress pointwise reconstruction overflowed")
    return replace(samples, sample_sha256=_digest(_sample_body(samples)))


def volume_material_stress_to_dict(samples):
    if type(samples) is not VolumeMaterialStressSamples:
        raise VolumeMaterialStressRefusal("Immutable material stress samples are required")
    body = _sample_body(samples)
    if _digest(body) != samples.sample_sha256:
        raise VolumeMaterialStressRefusal("Material stress sample immutable binding differs")
    return {**body, "sample_sha256": samples.sample_sha256}


def volume_material_stress_from_dict(raw, *, observer):
    """Recompute every sample from the trusted accepted observer context."""
    if type(raw) is not dict:
        raise VolumeMaterialStressRefusal("Material stress archive requires full accepted-source context")
    try:
        expected = evaluate_volume_material_stress(observer, raw["cohort_ids"],
            raw["angular_barycentric"], raw["mass_coordinate_q"],
            normal_world=raw["normal_world"])
        equal = _canonical(raw) == _canonical(volume_material_stress_to_dict(expected))
    except (KeyError, TypeError, ValueError, OverflowError):
        equal = False
    if not equal:
        raise VolumeMaterialStressRefusal("Material stress archive differs from full accepted-source reconstruction")
    return expected
