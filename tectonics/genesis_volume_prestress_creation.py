"""Prospective traction-compatible creation work with an explicit reference.

This computational kernel retains a supplied initial traction at zero actual
gap increment.  The offset is an internal constitutive reference, not a
calculated material gap, a Genesis strain history, or an inserted Γ.  In
particular, a compression offset does not satisfy unilateral geometric contact
and must not be used as evidence of physical contact admission.

Merely subtracting a constant from a shifted cohesive potential would leave
an artificial negative reference energy after damage.  Here the cohesive
reference is subtracted with the same remaining stiffness factor.  The final
separation and dissipation are changed consistently so that complete cohesive
failure dissipates exactly the caller's Gc once per material area Γ.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .genesis_volume_creation_law import (
    CreationLawRefusal, CreationLawSofteningRefusal, ProspectiveCreationLaw,
    _effective_gap, _immutable, _numeric)


class PrestressCreationRefusal(CreationLawRefusal):
    """Supplied data cannot define a virgin, traction-compatible reference."""


@dataclass(frozen=True, slots=True)
class PrestressCreationSeed:
    law: ProspectiveCreationLaw
    initial_traction_pa: np.ndarray
    reference_traction_pa: np.ndarray = field(init=False)
    reference_peak_snap_factor: np.ndarray = field(init=False)
    reference_offset_m: np.ndarray = field(init=False)
    initial_maximum_separation_m: np.ndarray = field(init=False)
    reference_cohesive_energy_j_m2: np.ndarray = field(init=False)
    reference_compression_energy_j_m2: np.ndarray = field(init=False)
    auxiliary_fracture_energy_j_m2: np.ndarray = field(init=False)
    final_separation_m: np.ndarray = field(init=False)
    kernel_only: bool = field(init=False, default=True)
    physical_release_admitted: bool = field(init=False, default=False)
    normal_contact_admitted: bool = field(init=False, default=False)

    def __post_init__(self):
        if type(self.law) is not ProspectiveCreationLaw:
            raise PrestressCreationRefusal("Prestress creation requires an explicit ProspectiveCreationLaw")
        # Recheck parameters rather than trusting the nominal frozen type.
        # The seed owns a distinct law, so an explicit constructor bypass on
        # the caller's object cannot silently change the prepared reference.
        law = ProspectiveCreationLaw(**asdict(self.law))
        object.__setattr__(self, "law", law)
        traction = _numeric(self.initial_traction_pa, "initial traction")
        if traction.ndim < 1 or traction.shape[-1] != 3:
            raise PrestressCreationRefusal("Prestress creation traction must have final dimension three")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            offset = traction / law.cohesive_stiffness_pa_m
            offset[..., 0] = traction[..., 0] / np.where(
                traction[..., 0] >= 0., law.cohesive_stiffness_pa_m,
                law.compression_stiffness_pa_m)
        _, radius = _effective_gap(offset)
        if not (np.isfinite(offset).all() and np.isfinite(radius).all()):
            raise PrestressCreationRefusal("Prestress creation reference separation must be finite")
        delta0 = law.peak_separation_m
        peak_tolerance = 32. * np.finfo(float).eps * np.maximum(radius, delta0)
        if np.any(radius - delta0 > peak_tolerance):
            raise PrestressCreationRefusal(
                "Prestress creation requires a virgin initial cohesive traction no greater than peak strength")
        # A rotation of an exactly peak-strength traction can put its computed
        # norm a few ulps above δ0.  Snap only that machine-sized excess inward,
        # retaining both the supplied and reconstructed tractions in the seed.
        # Four epsilons leave room for the three-component norm rounding; the
        # The 32-epsilon gate bounds the input norm excess; this additional
        # four-epsilon inward guard is explicitly part of the recorded factor.
        peak_snap_factor = np.ones(radius.shape)
        snap = radius > delta0
        peak_snap_factor[snap] = (delta0 * (1. - 4. * np.finfo(float).eps)) / radius[snap]
        offset[..., 1:] *= peak_snap_factor[..., None]
        offset[..., 0] = np.where(offset[..., 0] >= 0.,
            offset[..., 0] * peak_snap_factor, offset[..., 0])
        _, radius = _effective_gap(offset)
        if np.any(radius > delta0):
            raise PrestressCreationRefusal("Prestress creation peak reference cannot be resolved within machine tolerance")
        with np.errstate(over="ignore", invalid="ignore"):
            reference_traction = law.cohesive_stiffness_pa_m * offset
            reference_traction[..., 0] = np.where(offset[..., 0] >= 0.,
                law.cohesive_stiffness_pa_m, law.compression_stiffness_pa_m) * offset[..., 0]
        if not np.isfinite(reference_traction).all():
            raise PrestressCreationRefusal("Prestress creation reconstructed reference traction must be finite")
        compression = np.minimum(offset[..., 0], 0.)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            cohesive_energy = .5 * law.cohesive_stiffness_pa_m * radius * radius
            compression_energy = .5 * law.compression_stiffness_pa_m * compression * compression
            auxiliary = law.fracture_energy_j_m2 + cohesive_energy
            final = 2. * (auxiliary / law.peak_traction_pa)
        for name, value in (("reference cohesive energy", cohesive_energy),
                            ("reference compression energy", compression_energy),
                            ("auxiliary fracture energy", auxiliary),
                            ("final separation", final)):
            if not np.isfinite(value).all():
                raise PrestressCreationRefusal(f"Prestress creation {name} must be finite")
        if np.any(final <= law.peak_separation_m):
            raise CreationLawSofteningRefusal("Prestress creation final separation must exceed peak separation")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            softening_ratio = final / (final - law.peak_separation_m)
        if not np.isfinite(softening_ratio).all():
            raise PrestressCreationRefusal("Prestress creation softening interval is not numerically resolvable")
        values = dict(initial_traction_pa=traction, reference_traction_pa=reference_traction,
                      reference_peak_snap_factor=peak_snap_factor, reference_offset_m=offset,
                      initial_maximum_separation_m=radius,
                      reference_cohesive_energy_j_m2=cohesive_energy,
                      reference_compression_energy_j_m2=compression_energy,
                      auxiliary_fracture_energy_j_m2=auxiliary,
                      final_separation_m=final)
        for name, value in values.items():
            object.__setattr__(self, name, _immutable(value))

    @property
    def peak_separation_m(self):
        return self.law.peak_separation_m


@dataclass(frozen=True, slots=True)
class PrestressCreationResponse:
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
    normal_contact_admitted: bool = field(init=False, default=False)

    def __post_init__(self):
        for name in ("traction_pa", "tangent_pa_m", "effective_separation_m",
                     "maximum_separation_m", "damage", "stored_energy_j_m2",
                     "old_stored_energy_j_m2", "fracture_dissipation_j_m2",
                     "old_fracture_dissipation_j_m2", "fracture_dissipation_increment_j_m2",
                     "mechanical_work_j_m2"):
            object.__setattr__(self, name, _immutable(_numeric(getattr(self, name), name)))


def prepare_prestress_creation(law, initial_traction_pa):
    """Prepare immutable, per-point reference data from explicitly supplied t0.

    The supplied traction has no automatic interpretation as endpoint stress,
    dynamic bank reaction, historical traction, or actual Genesis activation.
    A separate source-bound assembly must establish its force interpretation.
    The initial effective cohesive radius must be on the virgin elastic
    branch, including the peak itself.  An overstrength or already damaged
    reference cannot be made admissible by inventing an initial damage value.
    Only a norm excess within 32 machine epsilons is snapped inward at peak:
    ``initial_traction_pa`` retains the supplied values, whereas
    ``reference_traction_pa`` and ``reference_peak_snap_factor`` record the
    resulting reference and its rounding correction.  No damaged history is
    assigned to compensate for that norm rounding.
    """
    return PrestressCreationSeed(law, initial_traction_pa)


def _damage(maximum, delta0, deltaf):
    damage = np.zeros(maximum.shape)
    softening = (maximum > delta0) & (maximum < deltaf)
    damage[softening] = (deltaf[softening] / (deltaf[softening] - delta0)
                         * (1. - delta0 / maximum[softening]))
    damage[maximum >= deltaf] = 1.
    return np.clip(damage, 0., 1.)


def _dissipation(law, maximum, initial_radius, auxiliary, deltaf):
    """Stable form of (Gc+B*) s(κ) - B* d(κ), which is nonnegative.

    Above δ0 its derivative is (.5 k κ²-B*) d'(κ) >= 0 because the
    reference is virgin and initial_radius <= δ0.  At failure it is Gc.
    """
    result = np.zeros(maximum.shape)
    softening = (maximum > law.peak_separation_m) & (maximum < deltaf)
    maximum_s = maximum[softening]
    reference_s = initial_radius[softening]
    onset = law.peak_separation_m
    fraction = (maximum_s - onset) / (deltaf[softening] - onset)
    # 1-r*²/(δ0 κ), written as a sum of nonnegative terms.  It avoids
    # subtracting nearly equal energies when the initial traction is at peak.
    factor = ((onset - reference_s) / onset
              + (reference_s / onset) * ((maximum_s - reference_s) / maximum_s))
    result[softening] = auxiliary[softening] * fraction * factor
    result[maximum >= deltaf] = law.fracture_energy_j_m2
    return np.clip(result, 0., law.fracture_energy_j_m2)


def _relative_energy(law, gap, radius, initial_radius, reference_offset, damage):
    energy = np.zeros(radius.shape)
    remaining = damage < 1.
    # Fully separated cells need no square of a potentially huge separation.
    energy[remaining] = (.5 * law.cohesive_stiffness_pa_m
        * (1. - damage[remaining]) * (radius[remaining] - initial_radius[remaining])
        * (radius[remaining] + initial_radius[remaining]))
    compression = np.minimum(gap[..., 0], 0.)
    reference_compression = np.minimum(reference_offset[..., 0], 0.)
    energy += (.5 * law.compression_stiffness_pa_m
               * (compression - reference_compression) * (compression + reference_compression))
    return energy


def evaluate_prestress_creation(seed, old_gap_m, new_gap_m, old_maximum_separation_m):
    """Evaluate straight actual-gap increments relative to the prepared seed.

    At birth both actual gaps are zero, history is the seed's initial maximum,
    traction is t0, and relative stored energy and dissipation are zero.
    Relative energy can be negative on unloading; it is a signed change from
    the prestressed reference, not newly assigned historical energy.

    With h=h*+g, B=.5k|[max(hn,0),ht1,ht2]|² and C=.5kc min(hn,0)²,
    Ψ=(1-d)(B-B*)+(C-C*).  The auxiliary endpoint energy Gc+B* is used only
    to set δf=2(Gc+B*)/T; actual fracture dissipation is (Gc+B*)s-B*d.
    Full cohesive failure therefore leaves no cohesive reference energy and
    dissipates exactly the original Gc.  Compression work remains reversible.

    Tangent conventions match the zero-prestress kernel: unloading at an old
    maximum, opening side at hn=0, elastic side at δ0, separated side at δf.
    The input history must retain both the initial and old effective radius;
    a 32-machine-epsilon roundoff mismatch is snapped upward.  Work is exact
    only for the declared straight gap segment, not an unspecified trajectory.
    """
    if type(seed) is not PrestressCreationSeed:
        raise PrestressCreationRefusal("Prestress creation requires a prepared typed seed")
    old_gap = _numeric(old_gap_m, "old actual gap")
    new_gap = _numeric(new_gap_m, "new actual gap")
    old_maximum = _numeric(old_maximum_separation_m, "old maximum separation")
    if old_gap.ndim < 1 or new_gap.ndim < 1 or old_gap.shape[-1] != 3 or new_gap.shape[-1] != 3:
        raise PrestressCreationRefusal("Prestress creation gaps must have final dimension three")
    try:
        batch_shape = np.broadcast_shapes(seed.initial_maximum_separation_m.shape,
            old_gap.shape[:-1], new_gap.shape[:-1], old_maximum.shape)
        old_gap = np.broadcast_to(old_gap, batch_shape + (3,))
        new_gap = np.broadcast_to(new_gap, batch_shape + (3,))
        old_maximum = np.broadcast_to(old_maximum, batch_shape)
        reference = np.broadcast_to(seed.reference_offset_m, batch_shape + (3,))
        initial_radius = np.broadcast_to(seed.initial_maximum_separation_m, batch_shape)
        auxiliary = np.broadcast_to(seed.auxiliary_fracture_energy_j_m2, batch_shape)
        final = np.broadcast_to(seed.final_separation_m, batch_shape)
    except ValueError as error:
        raise PrestressCreationRefusal("Prestress creation seed, gaps and history have incompatible broadcast shapes") from error
    if np.any(old_maximum < 0):
        raise PrestressCreationRefusal("Prestress creation maximum separation cannot be negative")
    with np.errstate(over="ignore", invalid="ignore"):
        old_internal_gap = reference + old_gap
        internal_gap = reference + new_gap
    if not (np.isfinite(old_internal_gap).all() and np.isfinite(internal_gap).all()):
        raise PrestressCreationRefusal("Prestress creation internal reference gap must be finite")
    _, old_radius = _effective_gap(old_internal_gap)
    effective, radius = _effective_gap(internal_gap)
    if not (np.isfinite(old_radius).all() and np.isfinite(radius).all()):
        raise PrestressCreationRefusal("Prestress creation effective separation must be finite")
    required_history = np.maximum(initial_radius, old_radius)
    tolerance = 32. * np.finfo(float).eps * np.maximum(
        np.maximum(required_history, old_maximum), seed.peak_separation_m)
    if np.any(old_maximum < required_history - tolerance):
        raise PrestressCreationRefusal("Prestress creation old history must retain its initial and old effective separation")
    old_maximum = np.maximum(old_maximum, required_history)
    maximum = np.maximum(old_maximum, radius)
    law = seed.law
    damage = _damage(maximum, law.peak_separation_m, final)
    old_damage = _damage(old_maximum, law.peak_separation_m, final)
    dissipation = _dissipation(law, maximum, initial_radius, auxiliary, final)
    old_dissipation = _dissipation(law, old_maximum, initial_radius, auxiliary, final)
    loading = (radius > old_maximum) & (radius > law.peak_separation_m) & (radius < final)
    projection = np.ones(batch_shape + (3,))
    projection[..., 0] = internal_gap[..., 0] >= 0.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        traction = law.cohesive_stiffness_pa_m * (1. - damage)[..., None] * effective
        traction[..., 0] += law.compression_stiffness_pa_m * np.minimum(internal_gap[..., 0], 0.)
        tangent = (law.cohesive_stiffness_pa_m * (1. - damage)[..., None, None]
                   * projection[..., :, None] * np.eye(3))
        if np.any(loading):
            loading_radius = radius[loading]
            unit_gap = effective[loading] / loading_radius[:, None]
            coefficient = ((law.peak_traction_pa / loading_radius)
                           * (final[loading] / (final[loading] - law.peak_separation_m)))
            tangent[loading] -= coefficient[:, None, None] * unit_gap[:, :, None] * unit_gap[:, None, :]
        tangent[..., 0, 0] += law.compression_stiffness_pa_m * (internal_gap[..., 0] < 0.)
        energy = _relative_energy(law, internal_gap, radius, initial_radius, reference, damage)
        old_energy = _relative_energy(law, old_internal_gap, old_radius, initial_radius, reference, old_damage)
        increment = dissipation - old_dissipation
        work = energy - old_energy + increment
    return PrestressCreationResponse(
        traction_pa=traction, tangent_pa_m=tangent,
        effective_separation_m=radius, maximum_separation_m=maximum, damage=damage,
        stored_energy_j_m2=energy, old_stored_energy_j_m2=old_energy,
        fracture_dissipation_j_m2=dissipation,
        old_fracture_dissipation_j_m2=old_dissipation,
        fracture_dissipation_increment_j_m2=increment,
        mechanical_work_j_m2=work)
