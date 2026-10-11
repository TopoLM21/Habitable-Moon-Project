"""Birth history of the existing, spatially uniform Genesis surface reservoir.

The thermal owner supplies accepted endpoint samples. Within each interval we
interpolate surface temperature linearly, and integrate the existing linear
solidus/liquidus phase law. Birth is uniform in time only on the clipped phase
crossing, so its mass-weighted birth time is that crossing's midpoint. This is
an explicit time-discretization closure, not a claim to resolve the thermal
solver's internal trajectory or a depth-dependent solidification front.

Heating removes each solid cohort in proportion to its retained mass: the
same well-mixed, scalar reservoir interpretation as the uniform phase view.
Re-solidification creates a new cohort. No age is inferred from mechanical
lid depth, and neither material nor a second heat reservoir is introduced.
The records describe mass and birth time, not 3D stress or chemical evolution.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import hashlib
import json
import math
from numbers import Real

from .genesis import GenesisParameters, melt_fraction


MODEL_VERSION = "genesis-surface-solidification-0.1"
FORMAT = MODEL_VERSION
TIME_CLOSURE = "linear_endpoint_surface_temperature_phase_crossing"
REMELT_CLOSURE = "proportional_well_mixed_surface_reservoir"
_TRUSTED_HISTORIES = []  # At most four strong identity references, never an archive flag.


def _real(value, name, *, nonnegative=False, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite real values")
    try:
        value = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite real values") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite real values")
    if (nonnegative and value < 0) or (positive and value <= 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@lru_cache(maxsize=4)
def _validated_thermal_hash(thermal):
    # GenesisParameters is frozen and contains only scalar numeric fields.
    thermal.validate()
    return hashlib.sha256(_canonical(asdict(thermal)).encode()).hexdigest()


def _thermal_hash(thermal):
    if not isinstance(thermal, GenesisParameters):
        raise ValueError("Solidification requires Genesis thermal parameters")
    return _validated_thermal_hash(thermal)


def _is_trusted(history, pin):
    return any(saved is history and digest == pin for saved, digest in _TRUSTED_HISTORIES)


def _remember(history, pin):
    _TRUSTED_HISTORIES[:] = [(saved, digest) for saved, digest in _TRUSTED_HISTORIES
        if saved is not history]+[(history, pin)]
    del _TRUSTED_HISTORIES[:-4]


def _close(first, second, *, scale=1.):
    return math.isclose(first, second, rel_tol=2e-12, abs_tol=2e-14*scale)


@dataclass(frozen=True, slots=True)
class SolidificationCohort:
    sequence: int
    birth_start_myr: float
    birth_end_myr: float
    formed_mass_kg: float
    retained_mass_kg: float

    def __post_init__(self):
        if type(self.sequence) is not int or self.sequence < 1:
            raise ValueError("A solid cohort needs a positive sequence")
        for name in ("birth_start_myr", "birth_end_myr", "retained_mass_kg"):
            object.__setattr__(self, name, _real(getattr(self, name), name, nonnegative=True))
        object.__setattr__(self, "formed_mass_kg", _real(self.formed_mass_kg, "formed_mass_kg", positive=True))
        if self.birth_end_myr <= self.birth_start_myr:
            raise ValueError("A cohort requires a nonempty birth interval")
        if self.retained_mass_kg > self.formed_mass_kg:
            raise ValueError("A cohort cannot retain more mass than it formed")

    @property
    def mean_birth_time_myr(self):
        return self.birth_start_myr+.5*(self.birth_end_myr-self.birth_start_myr)


@dataclass(frozen=True, slots=True)
class SolidificationInterval:
    before_time_myr: float
    after_time_myr: float
    before_surface_temperature_k: float
    after_surface_temperature_k: float

    def __post_init__(self):
        for name in ("before_time_myr", "after_time_myr"):
            object.__setattr__(self, name, _real(getattr(self, name), name, nonnegative=True))
        for name in ("before_surface_temperature_k", "after_surface_temperature_k"):
            object.__setattr__(self, name, _real(getattr(self, name), name, positive=True))
        if self.after_time_myr <= self.before_time_myr:
            raise ValueError("Accepted solidification intervals must advance time")


@dataclass(frozen=True, slots=True)
class SolidificationHistory:
    time_myr: float
    surface_temperature_k: float
    surface_reservoir_mass_kg: float
    parameters_sha256: str
    cohorts: tuple[SolidificationCohort, ...] = ()
    intervals: tuple[SolidificationInterval, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "time_myr", _real(self.time_myr, "time_myr", nonnegative=True))
        for name in ("surface_temperature_k", "surface_reservoir_mass_kg"):
            object.__setattr__(self, name, _real(getattr(self, name), name, positive=True))
        if (type(self.parameters_sha256) is not str or len(self.parameters_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.parameters_sha256)):
            raise ValueError("Solidification history needs its thermal parameter hash")
        object.__setattr__(self, "cohorts", tuple(self.cohorts))
        object.__setattr__(self, "intervals", tuple(self.intervals))
        if (any(type(c) is not SolidificationCohort for c in self.cohorts)
                or any(type(i) is not SolidificationInterval for i in self.intervals)):
            raise ValueError("Solidification records must be immutable typed cohorts and intervals")

    def __deepcopy__(self, memo):
        # Every nested record and scalar is immutable. Starter's trial-state
        # deepcopy must retain this identity rather than recopy its entire past.
        memo[id(self)] = self
        return self

    @property
    def accepted_intervals(self):
        return len(self.intervals)

    @property
    def solid_mass_kg(self):
        return math.fsum(c.retained_mass_kg for c in self.cohorts)

    @property
    def formed_mass_kg(self):
        return math.fsum(c.formed_mass_kg for c in self.cohorts)

    @property
    def remelted_mass_kg(self):
        return math.fsum(c.formed_mass_kg-c.retained_mass_kg for c in self.cohorts)

    @property
    def birth_time_mass_moment_kg_myr(self):
        return math.fsum(c.retained_mass_kg*c.mean_birth_time_myr for c in self.cohorts)

    @property
    def mean_birth_time_myr(self):
        mass = self.solid_mass_kg
        return self.birth_time_mass_moment_kg_myr/mass if mass else None

    @property
    def mean_age_myr(self):
        born = self.mean_birth_time_myr
        return self.time_myr-born if born is not None else 0.


def _sample(sample, thermal):
    if not isinstance(sample, Mapping):
        raise ValueError("Solidification samples must be thermal diagnostic mappings")
    try:
        time = _real(sample["time_myr"], "sample time_myr", nonnegative=True)
        temperature = _real(sample["surface_temperature_k"], "sample surface_temperature_k", positive=True)
        phi = _real(sample["surface_melt_fraction"], "sample surface_melt_fraction", nonnegative=True)
    except KeyError as exc:
        raise ValueError("Solidification needs time, surface temperature and melt fraction") from exc
    actual = melt_fraction(temperature, thermal)
    if phi > 1 or not _close(phi, actual):
        raise ValueError("Surface melt fraction disagrees with the existing thermal phase law")
    return time, temperature, 1-actual


def _birth_interval(interval, thermal):
    t0, t1 = interval.before_time_myr, interval.after_time_myr
    a, b = interval.before_surface_temperature_k, interval.after_surface_temperature_k
    if b >= a:
        raise ValueError("Birth requires a cooling phase crossing")
    hot, cold = min(a, thermal.liquidus_k), max(b, thermal.solidus_k)
    if hot <= cold:
        raise ValueError("Birth requires a nonempty solidus/liquidus crossing")
    start = t0+(a-hot)/(a-b)*(t1-t0)
    end = t0+(a-cold)/(a-b)*(t1-t0)
    return max(t0, min(t1, start)), max(t0, min(t1, end))


def initial_solidification_history(thermal):
    """Create history only at the physical, fully molten t=0 start."""
    pin = _thermal_hash(thermal)
    if melt_fraction(thermal.initial_surface_temperature_k, thermal) != 1.:
        raise ValueError("Solidification history must start from fully molten material")
    history = SolidificationHistory(0., thermal.initial_surface_temperature_k,
        thermal.area_m2*thermal.surface_column_kg_m2, pin)
    return validate_solidification_history(history, thermal)


def _validate_current(history, thermal, sample):
    mass = thermal.area_m2*thermal.surface_column_kg_m2
    if history.surface_reservoir_mass_kg != mass:
        raise ValueError("Solidification surface inventory changed")
    if any(not math.isfinite(value) for value in (history.solid_mass_kg,
            history.formed_mass_kg, history.remelted_mass_kg,
            history.birth_time_mass_moment_kg_myr)):
        raise ValueError("Solidification history contains overflowing mass or birth moment")
    target = mass*(1-melt_fraction(history.surface_temperature_k, thermal))
    if not _close(history.solid_mass_kg, target, scale=mass):
        raise ValueError("Solidification mass disagrees with the thermal surface reservoir")
    if sample is not None:
        t, temp, solid = _sample(sample, thermal)
        if t != history.time_myr or not _close(temp, history.surface_temperature_k, scale=thermal.liquidus_k):
            raise ValueError("Solidification history and current thermal sample disagree")
        if not _close(history.solid_mass_kg, mass*solid, scale=mass):
            raise ValueError("Solidification mass and current phase sample disagree")


def _continued_history(parent, interval, cohorts):
    """Append locally checked records to an already validated immutable parent.

    The public constructor checks every supplied record. Repeating that scan
    for each accepted endpoint would still replay the growing thermal past.
    This private path uses only the unchanged validated parent, the checked
    interval and cohorts produced above; external constructors never gain it.
    """
    result = object.__new__(SolidificationHistory)
    for name, value in (("time_myr", interval.after_time_myr),
            ("surface_temperature_k", interval.after_surface_temperature_k),
            ("surface_reservoir_mass_kg", parent.surface_reservoir_mass_kg),
            ("parameters_sha256", parent.parameters_sha256), ("cohorts", cohorts),
            ("intervals", parent.intervals+(interval,))):
        object.__setattr__(result, name, value)
    return result


def validate_solidification_history(history, thermal, sample=None):
    """Check continuity and replay all cohort mass/moments from thermal records.

    No missing history or earlier material birth is synthesized. A reverse
    traversal accumulates subsequent proportional remelting in linear time.
    """
    if type(history) is not SolidificationHistory:
        raise ValueError("Missing solidification history; recalculate from the molten birth")
    pin = _thermal_hash(thermal)
    if history.parameters_sha256 != pin:
        raise ValueError("Solidification thermal parameters changed; recalculate from birth")
    if _is_trusted(history, pin):
        _validate_current(history, thermal, sample)
        return history
    mass = thermal.area_m2*thermal.surface_column_kg_m2
    if history.surface_reservoir_mass_kg != mass:
        raise ValueError("Solidification surface inventory changed")
    time, temperature = 0., thermal.initial_surface_temperature_k
    for interval in history.intervals:
        if interval.before_time_myr != time or interval.before_surface_temperature_k != temperature:
            raise ValueError("Solidification thermal history contains a gap or changed endpoint")
        time, temperature = interval.after_time_myr, interval.after_surface_temperature_k
    if history.time_myr != time or history.surface_temperature_k != temperature:
        raise ValueError("Solidification history does not end at its stored thermal state")
    expected = []
    retention = 1.
    for interval in reversed(history.intervals):
        before = 1-melt_fraction(interval.before_surface_temperature_k, thermal)
        after = 1-melt_fraction(interval.after_surface_temperature_k, thermal)
        if after > before:
            start, end = _birth_interval(interval, thermal)
            formed = mass*(after-before)
            expected.append((start, end, formed, formed*retention))
        elif after < before:
            retention *= after/before
    expected.reverse()
    if len(history.cohorts) != len(expected):
        raise ValueError("Solidification cohorts disagree with accepted phase crossings")
    for sequence, (cohort, actual) in enumerate(zip(history.cohorts, expected), 1):
        start, end, formed, retained = actual
        if (cohort.sequence != sequence or cohort.birth_start_myr != start
                or cohort.birth_end_myr != end
                or not _close(cohort.formed_mass_kg, formed, scale=mass)
                or not _close(cohort.retained_mass_kg, retained, scale=mass)):
            raise ValueError("Solidification cohort birth or remelting history changed")
    _validate_current(history, thermal, sample)
    _remember(history, pin)
    return history


def advance_solidification_history(history, before_sample, after_sample, thermal):
    """Advance only from supplied, consecutive accepted thermal endpoints."""
    validate_solidification_history(history, thermal, before_sample)
    t0, a, before = _sample(before_sample, thermal)
    t1, b, after = _sample(after_sample, thermal)
    interval = SolidificationInterval(t0, t1, a, b)
    if a != history.surface_temperature_k:
        raise ValueError("Solidification advance changed the preceding temperature endpoint")
    mass = history.surface_reservoir_mass_kg
    target = mass*after
    cohorts = history.cohorts
    if after > before:
        start, end = _birth_interval(interval, thermal)
        cohort = SolidificationCohort(len(cohorts)+1, start, end,
            mass*(after-before), mass*(after-before))
        cohorts += (cohort,)
    elif after < before:
        ratio = after/before
        cohorts = tuple(replace(c, retained_mass_kg=c.retained_mass_kg*ratio) for c in cohorts)
    # Correct summation roundoff in one retained cohort; never fabricate mass
    # in an empty molten history, nor change cohort birth mass or time.
    if cohorts and target:
        correction = target-math.fsum(c.retained_mass_kg for c in cohorts)
        if correction:
            index = max(range(len(cohorts)), key=lambda i: cohorts[i].retained_mass_kg)
            changed = cohorts[index].retained_mass_kg+correction
            if changed > cohorts[index].formed_mass_kg and _close(changed, cohorts[index].formed_mass_kg, scale=mass):
                changed = cohorts[index].formed_mass_kg
            cohorts = cohorts[:index]+(replace(cohorts[index], retained_mass_kg=changed),)+cohorts[index+1:]
    result = _continued_history(history, interval, cohorts)
    # The parent was validated and every new record above was checked locally.
    # Only these internally produced immutable objects receive incremental
    # trust. Constructed/deserialized histories always replay the full record.
    _validate_current(result, thermal, after_sample)
    _remember(result, result.parameters_sha256)
    return result


def solidification_history_to_dict(history):
    if type(history) is not SolidificationHistory:
        raise ValueError("Cannot save missing solidification history")
    body = {"format": FORMAT, "time_closure": TIME_CLOSURE, "remelt_closure": REMELT_CLOSURE,
        **asdict(history)}
    # JSON containers are fresh and cannot mutate the immutable history.
    body = json.loads(_canonical(body))
    return {**body, "sha256": hashlib.sha256(_canonical(body).encode()).hexdigest()}


def solidification_history_from_dict(raw, thermal):
    required = {"format", "time_closure", "remelt_closure", "time_myr",
        "surface_temperature_k", "surface_reservoir_mass_kg", "parameters_sha256",
        "cohorts", "intervals", "sha256"}
    if type(raw) is not dict or set(raw) != required:
        raise ValueError("Missing or malformed solidification history; recalculate from molten birth")
    try:
        body = {key: value for key, value in raw.items() if key != "sha256"}
        if (raw["format"] != FORMAT or raw["time_closure"] != TIME_CLOSURE
                or raw["remelt_closure"] != REMELT_CLOSURE
                or raw["sha256"] != hashlib.sha256(_canonical(body).encode()).hexdigest()):
            raise ValueError("Solidification version, closure or archive hash changed")
        if type(raw["cohorts"]) is not list or type(raw["intervals"]) is not list:
            raise ValueError("Solidification records must be archive lists")
        cohorts = tuple(SolidificationCohort(**record) for record in raw["cohorts"])
        intervals = tuple(SolidificationInterval(**record) for record in raw["intervals"])
        history = SolidificationHistory(raw["time_myr"], raw["surface_temperature_k"],
            raw["surface_reservoir_mass_kg"], raw["parameters_sha256"], cohorts, intervals)
        return validate_solidification_history(history, thermal)
    except (TypeError, OverflowError) as exc:
        raise ValueError("Malformed solidification history") from exc
