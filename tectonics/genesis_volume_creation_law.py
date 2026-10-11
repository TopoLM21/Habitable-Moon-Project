"""Prospective three-component cohesive creation kernel, per one area of Γ.

This explicitly parameterised law is a constitutive kernel, not a calibrated
Genesis material law.  It does not locate a fracture, insert a material
surface, replace prestress by interface traction, or release tied banks.  In
particular, the parameters are not inferred from finite-width slip energy or
from the legacy contact defaults.

The normal gap is positive in opening.  Both orthonormal tangent components
enter the same isotropic effective separation.  Compression has a separate
reversible penalty; there is no friction or viscosity in this kernel.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np


class CreationLawRefusal(ValueError):
    """An input cannot define a finite, admissible prospective kernel."""


class CreationLawSofteningRefusal(CreationLawRefusal):
    """The specified toughness does not admit the requested bilinear law."""


def _contains_boolean(value):
    # np.asarray([False, 1.]) silently promotes the boolean to float.  Reject
    # that mixed sequence before conversion, as well as ordinary bool arrays.
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, (list, tuple)):
        return any(_contains_boolean(item) for item in value)
    return False


def _numeric(value, name):
    if _contains_boolean(value):
        raise CreationLawRefusal(f"Creation law {name} requires real numeric values, not booleans")
    try:
        original = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise CreationLawRefusal(f"Creation law {name} requires a real numeric array") from error
    if original.dtype.kind not in "iuf":
        raise CreationLawRefusal(f"Creation law {name} requires a real numeric dtype")
    with np.errstate(over="ignore", invalid="ignore"):
        array = original.astype(float)
    if not np.isfinite(array).all():
        raise CreationLawRefusal(f"Creation law {name} requires finite values")
    return array


def _positive_scalar(value, name):
    array = _numeric(value, name)
    if array.shape != () or array <= 0:
        raise CreationLawRefusal(f"Creation law {name} must be a positive real scalar")
    return float(array)


def _immutable(value):
    array = np.asarray(value, dtype=float)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True, slots=True)
class ProspectiveCreationLaw:
    """Required explicit parameters; none is a fact derived from Genesis.

    ``fracture_energy_j_m2`` counts both created banks once per original
    material surface area Γ.  A complete monotone opening or shear separation
    dissipates exactly that value, independently of the mode direction.
    """

    fracture_energy_j_m2: float
    peak_traction_pa: float
    cohesive_stiffness_pa_m: float
    compression_stiffness_pa_m: float

    def __post_init__(self):
        for name in ("fracture_energy_j_m2", "peak_traction_pa",
                     "cohesive_stiffness_pa_m", "compression_stiffness_pa_m"):
            object.__setattr__(self, name, _positive_scalar(getattr(self, name), name))
        delta0 = self.peak_traction_pa / self.cohesive_stiffness_pa_m
        deltaf = 2. * (self.fracture_energy_j_m2 / self.peak_traction_pa)
        if not (math.isfinite(delta0) and math.isfinite(deltaf)
                and delta0 > 0 and deltaf > 0):
            raise CreationLawRefusal("Creation law separation thresholds must be finite and positive")
        if deltaf <= delta0:
            raise CreationLawSofteningRefusal(
                "Creation law requires final separation 2 Gc / T greater than peak separation T / k")
        # This ratio is used in both damage and its tangent.  An arbitrarily
        # close pair of floating-point thresholds need not be resolvable.
        if not math.isfinite(deltaf / (deltaf - delta0)):
            raise CreationLawRefusal("Creation law softening interval is not numerically resolvable")

    @property
    def peak_separation_m(self):
        return self.peak_traction_pa / self.cohesive_stiffness_pa_m

    @property
    def final_separation_m(self):
        return 2. * (self.fracture_energy_j_m2 / self.peak_traction_pa)

    @property
    def kernel_only(self):
        return True

    @property
    def physical_release_admitted(self):
        return False


@dataclass(frozen=True, slots=True)
class CreationLawResponse:
    traction_pa: np.ndarray
    tangent_pa_m: np.ndarray
    effective_separation_m: np.ndarray
    maximum_separation_m: np.ndarray
    damage: np.ndarray
    stored_energy_j_m2: np.ndarray
    old_stored_energy_j_m2: np.ndarray
    fracture_dissipation_j_m2: np.ndarray
    old_fracture_dissipation_j_m2: np.ndarray
    fracture_dissipation_increment_j_m2: np.ndarray
    mechanical_work_j_m2: np.ndarray
    kernel_only: bool = field(init=False, default=True)
    physical_release_admitted: bool = field(init=False, default=False)

    def __post_init__(self):
        names = ("traction_pa", "tangent_pa_m", "effective_separation_m",
                 "maximum_separation_m", "damage", "stored_energy_j_m2",
                 "old_stored_energy_j_m2", "fracture_dissipation_j_m2",
                 "old_fracture_dissipation_j_m2", "fracture_dissipation_increment_j_m2",
                 "mechanical_work_j_m2")
        for name in names:
            value = _numeric(getattr(self, name), name)
            object.__setattr__(self, name, _immutable(value))


def _effective_gap(gap):
    effective = gap.copy()
    effective[..., 0] = np.maximum(effective[..., 0], 0.)
    with np.errstate(over="ignore", invalid="ignore"):
        radius = np.hypot.reduce(effective, axis=-1)
    return effective, radius


def _damage(law, maximum):
    delta0, deltaf = law.peak_separation_m, law.final_separation_m
    damage = np.zeros(maximum.shape)
    softening = (maximum > delta0) & (maximum < deltaf)
    damage[softening] = (deltaf / (deltaf - delta0)
                         * (1. - delta0 / maximum[softening]))
    damage[maximum >= deltaf] = 1.
    # Exact d lies in [0, 1]; keep endpoint roundoff from producing negative
    # stored energy or an attractive traction beyond complete separation.
    return np.clip(damage, 0., 1.)


def _energy(law, gap, radius, damage):
    compression = np.minimum(gap[..., 0], 0.)
    return (.5 * law.cohesive_stiffness_pa_m * (1. - damage) * radius * radius
            + .5 * law.compression_stiffness_pa_m * compression * compression)


def evaluate_creation_law(law, old_gap_m, new_gap_m, old_maximum_separation_m):
    """Evaluate a straight gap segment with irreversible maximum separation.

    Gaps have shape ``(..., 3)``; history has broadcastable shape ``(...)``.
    Mechanical work is the exact straight-segment traction integral, computed
    as Δ stored energy + Δ fracture dissipation.  Effective separation is
    convex along a straight segment, so its maximum is attained at an endpoint
    and no unobserved intermediate peak is discarded.  This statement does
    not apply to an unspecified curved gap trajectory.

    At a previously attained maximum the tangent uses the unloading side.
    At zero normal gap it uses the opening side, at the peak threshold the
    elastic side, and at the final threshold the fully separated side.  These
    kink conventions do not alter the continuous traction or segment work.
    History smaller than the old effective separation is rejected; a mismatch
    within 32 machine epsilons of the maximum of those lengths and δ0 is
    snapped upward to the old effective separation.
    """
    if type(law) is not ProspectiveCreationLaw:
        raise CreationLawRefusal("Creation law requires explicit ProspectiveCreationLaw parameters")
    old_gap = _numeric(old_gap_m, "old gap")
    new_gap = _numeric(new_gap_m, "new gap")
    old_maximum = _numeric(old_maximum_separation_m, "old maximum separation")
    if old_gap.ndim < 1 or new_gap.ndim < 1 or old_gap.shape[-1] != 3 or new_gap.shape[-1] != 3:
        raise CreationLawRefusal("Creation law gap arrays must have final dimension three")
    try:
        batch_shape = np.broadcast_shapes(old_gap.shape[:-1], new_gap.shape[:-1], old_maximum.shape)
        old_gap = np.broadcast_to(old_gap, batch_shape + (3,))
        new_gap = np.broadcast_to(new_gap, batch_shape + (3,))
        old_maximum = np.broadcast_to(old_maximum, batch_shape)
    except ValueError as error:
        raise CreationLawRefusal("Creation law gaps and history have incompatible broadcast shapes") from error
    if np.any(old_maximum < 0):
        raise CreationLawRefusal("Creation law maximum separation cannot be negative")
    old_effective, old_radius = _effective_gap(old_gap)
    effective, radius = _effective_gap(new_gap)
    del old_effective
    if not (np.isfinite(old_radius).all() and np.isfinite(radius).all()):
        raise CreationLawRefusal("Creation law effective separation is not finite")
    tolerance = 32. * np.finfo(float).eps * np.maximum(
        np.maximum(old_maximum, old_radius), law.peak_separation_m)
    if np.any(old_maximum < old_radius - tolerance):
        raise CreationLawRefusal("Creation law old history is smaller than old effective separation")
    old_maximum = np.maximum(old_maximum, old_radius)
    maximum = np.maximum(old_maximum, radius)
    damage = _damage(law, maximum)
    old_damage = _damage(law, old_maximum)
    delta0, deltaf = law.peak_separation_m, law.final_separation_m
    # Clamp before division: a finite, already separated very large gap
    # should not overflow an irrelevant unbounded softening fraction.
    dissipation = law.fracture_energy_j_m2 * (
        (np.clip(maximum, delta0, deltaf) - delta0) / (deltaf - delta0))
    old_dissipation = law.fracture_energy_j_m2 * (
        (np.clip(old_maximum, delta0, deltaf) - delta0) / (deltaf - delta0))
    derivative = np.zeros(batch_shape)
    loading = (radius > old_maximum) & (radius > delta0) & (radius < deltaf)
    derivative[loading] = (deltaf / (deltaf - delta0)
                           * (delta0 / radius[loading]) / radius[loading])
    active_projection = np.ones(batch_shape + (3,))
    active_projection[..., 0] = new_gap[..., 0] >= 0.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        traction = law.cohesive_stiffness_pa_m * (1. - damage)[..., None] * effective
        traction[..., 0] += law.compression_stiffness_pa_m * np.minimum(new_gap[..., 0], 0.)
        tangent = (law.cohesive_stiffness_pa_m * (1. - damage)[..., None, None]
                   * active_projection[..., :, None] * np.eye(3))
        # Construct the softening term only where r > δ0.  Forming outer(z,z)
        # in already separated cells would needlessly overflow at huge gaps.
        if np.any(loading):
            loading_effective = effective[loading]
            tangent[loading] -= (law.cohesive_stiffness_pa_m
                * derivative[loading, None, None]
                * loading_effective[:, :, None]
                * (loading_effective / radius[loading, None])[:, None, :])
        tangent[..., 0, 0] += law.compression_stiffness_pa_m * (new_gap[..., 0] < 0.)
        energy = _energy(law, new_gap, radius, damage)
        old_energy = _energy(law, old_gap, old_radius, old_damage)
        dissipation_increment = dissipation - old_dissipation
        work = energy - old_energy + dissipation_increment
    return CreationLawResponse(
        traction_pa=traction, tangent_pa_m=tangent,
        effective_separation_m=radius, maximum_separation_m=maximum, damage=damage,
        stored_energy_j_m2=energy, old_stored_energy_j_m2=old_energy,
        fracture_dissipation_j_m2=dissipation,
        old_fracture_dissipation_j_m2=old_dissipation,
        fracture_dissipation_increment_j_m2=dissipation_increment,
        mechanical_work_j_m2=work)
