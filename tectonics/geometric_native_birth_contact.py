"""Explicit solid HOT contacts and independently owned hydrostatic foundations.

The interface is permanently unbonded, frictionless and unilateral. Two
declared elastic columns and the compression penalty act in series; neither
elastic thickness nor a cold cantilever is inferred. A negative actual birth
gap inserts recoverable energy paid by the activation ledger. Only shear has
a virgin birth reference. Opening and reclosure do not erase normal preload.

Each Winkler foundation has a separate reference, state and journal. A HOT
component shared by several interfaces therefore owns its foundation once.
Its basal reference area is independent of every inclined interface area.
These local models confer no temporal material or global force admission.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from fractions import Fraction
import hashlib
import json
import math

from .genesis_contact_law import ContactLawParameters
from .geometric_contact_mechanics import ElasticColumnParameters
from .geometric_intrinsic_geometry import RotationIncrement
from .geometric_native_polygons import NativePlane, native_plane_from_dict, native_plane_to_dict


CONTACT_FORMAT = "native-solid-hot-unbonded-unilateral-contact-1"
FOUNDATION_FORMAT = "native-solid-hot-hydrostatic-winkler-foundation-1"


def _number(value, name, *, positive=False):
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be an explicit finite real number")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be an explicit finite real number") from error
    if not math.isfinite(value):
        raise ValueError(f"{name} must be an explicit finite real number")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _text(value, name):
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be an explicit nonempty string")
    return value


def _digest(value, name):
    if (type(value) is not str or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(f"{name} must be a literal lowercase SHA-256")
    return value


def _owner(value):
    if type(value) is not int or value < 0:
        raise ValueError("Birth owners must retain literal nonnegative integer identities")
    return value


def _clock(value):
    if type(value) is not Fraction or value < 0:
        raise ValueError("Birth clock must retain an exact nonnegative Fraction")
    return value


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()


def _hash(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _json(value):
    if value is None or type(value) in (str, int, bool):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for child in value:
            _json(child)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for child in value.values():
            _json(child)
        return
    raise ValueError("Birth DTO must retain literal finite JSON types")


def _same_json(actual, expected, message):
    _json(actual)
    if _encoded(actual) != _encoded(expected):
        raise ValueError(message)


def _schema(raw, names):
    if type(raw) is not dict or set(raw) != set(names):
        raise ValueError("Invalid explicit birth declaration schema")


def _dataclass_floats(value, kind):
    if type(value) is not kind:
        raise ValueError("Birth physics requires the explicit declared parameter class")
    return {field.name: _number(getattr(value, field.name), field.name)
        for field in fields(kind)}


@dataclass(frozen=True, slots=True)
class BornContactParameters:
    columns: ElasticColumnParameters
    law: ContactLawParameters
    provenance: str

    def __post_init__(self):
        columns = ElasticColumnParameters(**_dataclass_floats(self.columns, ElasticColumnParameters))
        law = ContactLawParameters(**_dataclass_floats(self.law, ContactLawParameters))
        law.validate()
        if any(getattr(law, name) != 0 for name in
                ("friction_dry", "friction_wet", "cohesion_pa", "viscosity_pa_s_m")):
            raise ValueError("This HOT law requires explicitly zero friction, cohesion and viscosity")
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "law", law)
        _text(self.provenance, "Contact parameter provenance")
        effective_normal_stiffness(self)


def effective_normal_stiffness(parameters):
    """Condensed series stiffness of declared columns and compression penalty."""
    if type(parameters) is not BornContactParameters:
        raise ValueError("Explicit BornContactParameters are required")
    column = parameters.columns.normal_stiffness_pa_m
    penalty = parameters.law.normal_stiffness_pa_m
    value = 1. / (1. / column + 1. / penalty)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Declared HOT compliance must give finite positive stiffness")
    return value


@dataclass(frozen=True, slots=True)
class BornFoundationParameters:
    density_kg_m3: float
    gravity_m_s2: float
    provenance: str

    def __post_init__(self):
        for name in ("density_kg_m3", "gravity_m_s2"):
            object.__setattr__(self, name, _number(getattr(self, name), name, positive=True))
        _text(self.provenance, "Foundation parameter provenance")
        if not math.isfinite(self.stiffness_pa_m) or self.stiffness_pa_m <= 0:
            raise ValueError("Hydrostatic density and gravity must give finite positive stiffness")

    @property
    def stiffness_pa_m(self):
        return self.density_kg_m3 * self.gravity_m_s2


@dataclass(frozen=True, slots=True)
class BornContactReference:
    family_id: str
    source_sha256: str
    source_epoch_sha256: str
    donor_cohort_id: str
    receiver_cohort_id: str
    donor_material_id: str
    receiver_material_id: str
    donor_parent_id: str
    receiver_parent_id: str
    donor_owner: int
    receiver_owner: int
    plane: NativePlane
    birth_time_myr: Fraction
    interface_area_m2: float
    birth_gap_free_m: float
    birth_jump_free_m: float
    provenance: str

    def __post_init__(self):
        for name in ("family_id", "donor_cohort_id", "receiver_cohort_id",
                "donor_material_id", "receiver_material_id", "donor_parent_id",
                "receiver_parent_id", "provenance"):
            _text(getattr(self, name), name)
        for name in ("source_sha256", "source_epoch_sha256"):
            _digest(getattr(self, name), name)
        _owner(self.donor_owner)
        _owner(self.receiver_owner)
        if self.donor_owner == self.receiver_owner:
            raise ValueError("A birth contact requires two different material owners")
        if type(self.plane) is not NativePlane:
            raise ValueError("Birth contact must name an explicit native weak plane")
        object.__setattr__(self, "plane", native_plane_from_dict(native_plane_to_dict(self.plane)))
        _clock(self.birth_time_myr)
        for name in ("interface_area_m2", "birth_gap_free_m", "birth_jump_free_m"):
            object.__setattr__(self, name, _number(getattr(self, name), name,
                positive=name == "interface_area_m2"))


@dataclass(frozen=True, slots=True)
class BornFoundationReference:
    patch_id: str
    source_sha256: str
    source_epoch_sha256: str
    cohort_id: str
    material_id: str
    parent_id: str
    owner: int
    birth_time_myr: Fraction
    basal_area_m2: float
    radial_birth_m: float
    provenance: str

    def __post_init__(self):
        for name in ("patch_id", "cohort_id", "material_id", "parent_id", "provenance"):
            _text(getattr(self, name), name)
        for name in ("source_sha256", "source_epoch_sha256"):
            _digest(getattr(self, name), name)
        _owner(self.owner)
        _clock(self.birth_time_myr)
        for name in ("basal_area_m2", "radial_birth_m"):
            object.__setattr__(self, name, _number(getattr(self, name), name,
                positive=name == "basal_area_m2"))


def _parameter_body(value):
    if type(value) is BornContactParameters:
        return dict(columns={f.name: getattr(value.columns, f.name) for f in fields(value.columns)},
            law={f.name: getattr(value.law, f.name) for f in fields(value.law)},
            provenance=value.provenance)
    if type(value) is BornFoundationParameters:
        return {f.name: getattr(value, f.name) for f in fields(value)}
    raise ValueError("Unknown explicit birth parameter class")


def _reference_body(value):
    if type(value) not in (BornContactReference, BornFoundationReference):
        raise ValueError("Unknown explicit birth reference class")
    result = {f.name: getattr(value, f.name) for f in fields(value)}
    result["birth_time_myr"] = [value.birth_time_myr.numerator, value.birth_time_myr.denominator]
    if type(value) is BornContactReference:
        result["plane"] = native_plane_to_dict(value.plane)
    return result


def _validate_parameters(value, kind):
    if type(value) is not kind:
        raise ValueError("Birth parameter class changed")
    if kind is BornContactParameters:
        for section, cls in ((value.columns, ElasticColumnParameters), (value.law, ContactLawParameters)):
            if type(section) is not cls or any(type(getattr(section, f.name)) is not float
                    for f in fields(cls)):
                raise ValueError("Stored HOT constitutive parameters changed literal type")
    else:
        if any(type(getattr(value, name)) is not float for name in ("density_kg_m3", "gravity_m_s2")):
            raise ValueError("Stored foundation parameters changed literal type")
    rebuilt = kind(**{f.name: getattr(value, f.name) for f in fields(kind)})
    if _encoded(_parameter_body(value)) != _encoded(_parameter_body(rebuilt)):
        raise ValueError("Stored birth parameters lost their canonical declaration")
    return value


def _validate_reference(value, kind):
    if type(value) is not kind:
        raise ValueError("Birth reference class changed")
    numeric = (("interface_area_m2", "birth_gap_free_m", "birth_jump_free_m")
        if kind is BornContactReference else ("basal_area_m2", "radial_birth_m"))
    if any(type(getattr(value, name)) is not float for name in numeric):
        raise ValueError("Stored birth reference changed literal numeric type")
    if kind is BornContactReference:
        plane = value.plane
        if (type(plane) is not NativePlane or type(plane.normal) is not tuple
                or any(type(x) is not int for x in plane.normal)
                or type(plane.rotations) is not tuple or type(plane.label) is not str):
            raise ValueError("Stored native birth plane changed literal type")
        for step in plane.rotations:
            if (type(step) is not RotationIncrement or type(step.omega_rad_per_myr) is not tuple
                    or any(type(x) is not float for x in step.omega_rad_per_myr)
                    or type(step.dt_myr) is not float):
                raise ValueError("Stored native birth plane history changed literal type")
    rebuilt = kind(**{f.name: getattr(value, f.name) for f in fields(kind)})
    if kind is BornContactReference and value.plane != rebuilt.plane:
        raise ValueError("Stored native birth plane lost its raw canonical declaration")
    if _encoded(_reference_body(value)) != _encoded(_reference_body(rebuilt)):
        raise ValueError("Stored birth reference lost its full canonical declaration")
    return value


def born_contact_parameters_to_dict(parameters):
    return _parameter_body(_validate_parameters(parameters, BornContactParameters))


def born_foundation_parameters_to_dict(parameters):
    return _parameter_body(_validate_parameters(parameters, BornFoundationParameters))


def born_contact_reference_to_dict(reference):
    return _reference_body(_validate_reference(reference, BornContactReference))


def born_foundation_reference_to_dict(reference):
    return _reference_body(_validate_reference(reference, BornFoundationReference))


def _read_reference(raw, kind):
    _schema(raw, (f.name for f in fields(kind)))
    values = dict(raw)
    clock = values["birth_time_myr"]
    if (type(clock) is not list or len(clock) != 2 or any(type(x) is not int for x in clock)
            or clock[1] <= 0):
        raise ValueError("Birth DTO clock must be an exact canonical rational pair")
    values["birth_time_myr"] = Fraction(*clock)
    if kind is BornContactReference:
        values["plane"] = native_plane_from_dict(values["plane"])
    result = kind(**values)
    _same_json(raw, _reference_body(result), "Birth DTO reference changed exact fields or literal types")
    return result


def born_contact_reference_from_dict(raw):
    return _read_reference(raw, BornContactReference)


def born_foundation_reference_from_dict(raw):
    return _read_reference(raw, BornFoundationReference)


def born_contact_parameters_from_dict(raw):
    _schema(raw, ("columns", "law", "provenance"))
    _schema(raw["columns"], (f.name for f in fields(ElasticColumnParameters)))
    _schema(raw["law"], (f.name for f in fields(ContactLawParameters)))
    result = BornContactParameters(ElasticColumnParameters(**raw["columns"]),
        ContactLawParameters(**raw["law"]), raw["provenance"])
    _same_json(raw, _parameter_body(result), "Birth parameter DTO changed exact fields or literal types")
    return result


def born_foundation_parameters_from_dict(raw):
    _schema(raw, (f.name for f in fields(BornFoundationParameters)))
    result = BornFoundationParameters(**raw)
    _same_json(raw, _parameter_body(result), "Foundation DTO changed exact fields or literal types")
    return result


@dataclass(frozen=True, slots=True)
class BornContactIncrement:
    gap_free_m: float
    jump_free_m: float
    dt_s: float


@dataclass(frozen=True, slots=True)
class BornFoundationIncrement:
    radial_displacement_m: float
    dt_s: float


@dataclass(frozen=True, slots=True)
class BornContactState:
    """Own virgin interface history and condensed column-plus-penalty energy.

    ``normal_contact_energy_j`` includes both supported elastic columns and
    the compression penalty. It is the reduced normal potential, rather than
    penalty energy alone. No foundation energy is owned by this interface.
    """
    parameters: BornContactParameters
    reference: BornContactReference
    history: tuple[BornContactIncrement, ...]
    gap_free_m: float
    jump_free_m: float
    normal_traction_pa: float
    plastic_slip_m: float
    cumulative_slip_m: float
    normal_contact_energy_j: float
    activation_work_j: float
    cumulative_external_work_j: float
    cumulative_numerical_dissipation_j: float
    elapsed_time_s: float
    step_count: int
    state_sha256: str

    @property
    def shear_traction_pa(self):
        return 0.


@dataclass(frozen=True, slots=True)
class BornFoundationState:
    parameters: BornFoundationParameters
    reference: BornFoundationReference
    history: tuple[BornFoundationIncrement, ...]
    radial_displacement_m: float
    foundation_energy_j: float
    cumulative_external_work_j: float
    cumulative_numerical_dissipation_j: float
    elapsed_time_s: float
    step_count: int
    state_sha256: str


@dataclass(frozen=True, slots=True)
class BornContactStep:
    state: BornContactState
    normal_traction_pa: float
    normal_tangent_pa_m: float
    energy_gradient_n: tuple[float, float]
    restoring_force_n: tuple[float, float]
    tangent_n_m: tuple[tuple[float, float], tuple[float, float]]
    signed_slip_increment_m: float
    external_work_j: float
    stored_energy_increment_j: float
    numerical_dissipation_j: float
    energy_balance_residual_j: float
    energy_balance_relative_error: float


@dataclass(frozen=True, slots=True)
class BornFoundationStep:
    state: BornFoundationState
    restoring_force_n: float
    energy_gradient_n: float
    tangent_n_m: float
    external_work_j: float
    stored_energy_increment_j: float
    numerical_dissipation_j: float
    energy_balance_residual_j: float
    energy_balance_relative_error: float


def _state_body(state):
    result = {f.name: getattr(state, f.name) for f in fields(state)
        if f.name not in ("parameters", "reference", "history", "state_sha256")}
    result.update(format=CONTACT_FORMAT if type(state) is BornContactState else FOUNDATION_FORMAT,
        parameters=_parameter_body(state.parameters), reference=_reference_body(state.reference),
        history=[{f.name: getattr(row, f.name) for f in fields(row)} for row in state.history])
    return result


def _finite(value):
    result = float(value)
    if not math.isfinite(result) or (value != 0 and result == 0):
        raise ValueError("Birth calculation exceeded the finite representable range")
    return result


def _contact_increment(k, old_gap, new_gap):
    old, new = Fraction.from_float(old_gap), Fraction.from_float(new_gap)
    old_closed, new_closed = min(old, 0), min(new, 0)
    old_energy, new_energy = k * old_closed**2 / 2, k * new_closed**2 / 2
    work = k * new_closed * (new-old)
    remainder = work - (new_energy-old_energy)
    if remainder < 0:
        raise RuntimeError("Unilateral convex backward-Euler remainder became negative")
    return new_energy, work, remainder


def _contact_state(parameters, reference, history):
    _validate_parameters(parameters, BornContactParameters)
    _validate_reference(reference, BornContactReference)
    if type(history) is not tuple:
        raise ValueError("Contact history must retain immutable increments")
    k = Fraction.from_float(reference.interface_area_m2)*Fraction.from_float(effective_normal_stiffness(parameters))
    gap, jump = reference.birth_gap_free_m, reference.birth_jump_free_m
    energy = k * min(Fraction.from_float(gap), 0)**2 / 2
    activation = energy
    work = remainder = slip = elapsed = Fraction(0)
    for row in history:
        if type(row) is not BornContactIncrement or any(type(getattr(row, f.name)) is not float for f in fields(row)):
            raise ValueError("Contact history changed exact increment types")
        new_gap = _number(row.gap_free_m, "Contact gap")
        new_jump = _number(row.jump_free_m, "Contact jump")
        _number(row.dt_s, "Contact duration", positive=True)
        energy, dw, dr = _contact_increment(k, gap, new_gap)
        work += dw
        remainder += dr
        slip += abs(Fraction.from_float(new_jump)-Fraction.from_float(jump))
        elapsed += Fraction.from_float(row.dt_s)
        gap, jump = new_gap, new_jump
    traction = _finite(Fraction.from_float(effective_normal_stiffness(parameters))
        *min(Fraction.from_float(gap), 0))
    values = (gap, jump, traction, _finite(Fraction.from_float(jump)-Fraction.from_float(reference.birth_jump_free_m)),
        _finite(slip), _finite(energy), _finite(activation), _finite(work), _finite(remainder),
        _finite(elapsed), len(history))
    state = BornContactState(parameters, reference, history, *values, "")
    return BornContactState(parameters, reference, history, *values, _hash(_state_body(state)))


def foundation_stiffness_n_m(parameters, reference):
    """One independently owned radial patch stiffness, never a contact area."""
    _validate_parameters(parameters, BornFoundationParameters)
    _validate_reference(reference, BornFoundationReference)
    return _finite(Fraction.from_float(parameters.stiffness_pa_m)*Fraction.from_float(reference.basal_area_m2))


def _foundation_state(parameters, reference, history):
    _validate_parameters(parameters, BornFoundationParameters)
    _validate_reference(reference, BornFoundationReference)
    if type(history) is not tuple:
        raise ValueError("Foundation history must retain immutable increments")
    k = Fraction.from_float(parameters.stiffness_pa_m)*Fraction.from_float(reference.basal_area_m2)
    birth = Fraction.from_float(reference.radial_birth_m)
    radial = reference.radial_birth_m
    energy = work = remainder = elapsed = Fraction(0)
    for row in history:
        if type(row) is not BornFoundationIncrement or any(type(getattr(row, f.name)) is not float for f in fields(row)):
            raise ValueError("Foundation history changed exact increment types")
        _number(row.radial_displacement_m, "Foundation radial displacement")
        _number(row.dt_s, "Foundation duration", positive=True)
        old, new = Fraction.from_float(radial)-birth, Fraction.from_float(row.radial_displacement_m)-birth
        work += k*new*(new-old)
        remainder += k*(new-old)**2/2
        energy = k*new**2/2
        elapsed += Fraction.from_float(row.dt_s)
        radial = row.radial_displacement_m
    values = (radial, _finite(energy), _finite(work), _finite(remainder), _finite(elapsed), len(history))
    state = BornFoundationState(parameters, reference, history, *values, "")
    return BornFoundationState(parameters, reference, history, *values, _hash(_state_body(state)))


def activate_born_contact(parameters, reference):
    """Zero-duration virgin birth; actual negative gap is paid, not rebased."""
    return _contact_state(parameters, reference, ())


def activate_born_foundation(parameters, reference):
    """Create exactly one unstressed radial foundation at its declared birth."""
    return _foundation_state(parameters, reference, ())


def _validate_state(state, parameters, reference, kind):
    if type(state) is not kind:
        raise ValueError("Birth state class changed")
    parameter_kind = BornContactParameters if kind is BornContactState else BornFoundationParameters
    reference_kind = BornContactReference if kind is BornContactState else BornFoundationReference
    _validate_parameters(parameters, parameter_kind)
    _validate_reference(reference, reference_kind)
    _validate_parameters(state.parameters, parameter_kind)
    _validate_reference(state.reference, reference_kind)
    if (_encoded(_parameter_body(state.parameters)) != _encoded(_parameter_body(parameters))
            or _encoded(_reference_body(state.reference)) != _encoded(_reference_body(reference))):
        raise ValueError("Birth state differs from independent complete expected physics/reference")
    if type(state.step_count) is not int or type(state.state_sha256) is not str:
        raise ValueError("Birth state changed literal count or seal type")
    excluded = {"parameters", "reference", "history", "step_count", "state_sha256"}
    if any(type(getattr(state, f.name)) is not float for f in fields(state) if f.name not in excluded):
        raise ValueError("Birth state changed literal scalar type")
    rebuilt = (_contact_state if kind is BornContactState else _foundation_state)(parameters, reference, state.history)
    if _encoded(_state_body(state)) != _encoded(_state_body(rebuilt)) or state.state_sha256 != rebuilt.state_sha256:
        raise ValueError("Birth state differs from its complete physical replay")
    return state


def validate_born_contact_state(state, *, expected_parameters, expected_reference):
    return _validate_state(state, expected_parameters, expected_reference, BornContactState)


def validate_born_foundation_state(state, *, expected_parameters, expected_reference):
    return _validate_state(state, expected_parameters, expected_reference, BornFoundationState)


def _balance(work, increment, remainder, old_energy, new_energy):
    residual = math.fsum((work, -increment, -remainder))
    scale = max(abs(work), abs(increment)+remainder, old_energy+new_energy,
        float.fromhex("0x0.0000000000001p-1022"))
    relative = abs(residual)/scale
    if relative > 2e-12:
        raise RuntimeError("Born local model failed its independent work balance")
    return residual, relative


def advance_born_contact(previous, parameters, reference, *, gap_free_m, jump_free_m, dt_s):
    """Pure contact trial; signed slip is virgin shear motion and has no work."""
    validate_born_contact_state(previous, expected_parameters=parameters, expected_reference=reference)
    gap = _number(gap_free_m, "Contact gap")
    jump = _number(jump_free_m, "Contact jump")
    dt = _number(dt_s, "Contact duration", positive=True)
    state = _contact_state(parameters, reference, previous.history+(BornContactIncrement(gap, jump, dt),))
    k = Fraction.from_float(reference.interface_area_m2)*Fraction.from_float(effective_normal_stiffness(parameters))
    energy, work, remainder = _contact_increment(k, previous.gap_free_m, gap)
    old_energy = k*min(Fraction.from_float(previous.gap_free_m), 0)**2/2
    external, increment, numerical = _finite(work), _finite(energy-old_energy), _finite(remainder)
    residual, relative = _balance(external, increment, numerical, previous.normal_contact_energy_j, state.normal_contact_energy_j)
    gradient = _finite(k*min(Fraction.from_float(gap), 0))
    tangent = effective_normal_stiffness(parameters) if gap < 0 else 0.
    integrated_tangent = _finite(k) if gap < 0 else 0.
    signed_slip = _finite(Fraction.from_float(jump)-Fraction.from_float(previous.jump_free_m))
    return BornContactStep(state, state.normal_traction_pa, tangent, (gradient, 0.),
        (-gradient, 0.), ((integrated_tangent, 0.), (0., 0.)), signed_slip,
        external, increment, numerical, residual, relative)


def advance_born_foundation(previous, parameters, reference, *, radial_displacement_m, dt_s):
    """Pure trial of an independently owned hydrostatic radial spring."""
    validate_born_foundation_state(previous, expected_parameters=parameters, expected_reference=reference)
    radial = _number(radial_displacement_m, "Foundation radial displacement")
    dt = _number(dt_s, "Foundation duration", positive=True)
    state = _foundation_state(parameters, reference,
        previous.history+(BornFoundationIncrement(radial, dt),))
    k = Fraction.from_float(parameters.stiffness_pa_m)*Fraction.from_float(reference.basal_area_m2)
    birth = Fraction.from_float(reference.radial_birth_m)
    old, new = Fraction.from_float(previous.radial_displacement_m)-birth, Fraction.from_float(radial)-birth
    external, increment, numerical = (_finite(k*new*(new-old)),
        _finite(k*(new**2-old**2)/2), _finite(k*(new-old)**2/2))
    residual, relative = _balance(external, increment, numerical, previous.foundation_energy_j, state.foundation_energy_j)
    gradient = _finite(k*new)
    return BornFoundationStep(state, -gradient, gradient, _finite(k), external,
        increment, numerical, residual, relative)


def _state_to_dict(state, *, expected_parameters, expected_reference, kind):
    _validate_state(state, expected_parameters, expected_reference, kind)
    return dict(_state_body(state), state_sha256=state.state_sha256)


def born_contact_state_to_dict(state, *, expected_parameters, expected_reference):
    return _state_to_dict(state, expected_parameters=expected_parameters,
        expected_reference=expected_reference, kind=BornContactState)


def born_foundation_state_to_dict(state, *, expected_parameters, expected_reference):
    return _state_to_dict(state, expected_parameters=expected_parameters,
        expected_reference=expected_reference, kind=BornFoundationState)


def _state_from_dict(raw, *, expected_parameters, expected_reference, kind):
    _json(raw)
    names = [f.name for f in fields(kind) if f.name not in ("parameters", "reference", "history")]
    _schema(raw, names+["format", "parameters", "reference", "history"])
    body = {name: value for name, value in raw.items() if name != "state_sha256"}
    if type(raw["state_sha256"]) is not str or raw["state_sha256"] != _hash(body):
        raise ValueError("Birth state DTO selfseal differs")
    contact = kind is BornContactState
    params = (born_contact_parameters_from_dict if contact else born_foundation_parameters_from_dict)(raw["parameters"])
    reference = (born_contact_reference_from_dict if contact else born_foundation_reference_from_dict)(raw["reference"])
    parameter_kind = BornContactParameters if contact else BornFoundationParameters
    reference_kind = BornContactReference if contact else BornFoundationReference
    _validate_parameters(expected_parameters, parameter_kind)
    _validate_reference(expected_reference, reference_kind)
    _same_json(raw["parameters"], _parameter_body(expected_parameters), "Independent full birth parameters differ")
    _same_json(raw["reference"], _reference_body(expected_reference), "Independent full birth reference differs")
    if type(raw["history"]) is not list:
        raise ValueError("Birth DTO history must retain an explicit JSON list")
    increment_kind = BornContactIncrement if contact else BornFoundationIncrement
    history = []
    for row in raw["history"]:
        _schema(row, (f.name for f in fields(increment_kind)))
        if any(type(value) is not float for value in row.values()):
            raise ValueError("Birth DTO increments changed exact literal numeric types")
        history.append(increment_kind(**row))
    state = (_contact_state if contact else _foundation_state)(params, reference, tuple(history))
    _same_json(raw, dict(_state_body(state), state_sha256=state.state_sha256),
        "Birth state DTO differs from complete physical replay")
    return state


def born_contact_state_from_dict(raw, *, expected_parameters, expected_reference):
    return _state_from_dict(raw, expected_parameters=expected_parameters,
        expected_reference=expected_reference, kind=BornContactState)


def born_foundation_state_from_dict(raw, *, expected_parameters, expected_reference):
    return _state_from_dict(raw, expected_parameters=expected_parameters,
        expected_reference=expected_reference, kind=BornFoundationState)
