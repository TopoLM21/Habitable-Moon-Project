"""Canonical initial radial coordinates from the saved physical banks.

The source-bound birth builder measures radial motion in ``w / radius``.
Rounding that quotient through binary64 changes the physical reference used
by a newly born contact even though its saved donor bank has not moved.  This
additive adapter divides the exact binary physical values in an explicitly
scoped Decimal context and records the remaining quotient error.  It changes
no anchor operator, stiffness, birth reference, paid activation or history.

This is an initial-coordinate adapter, not a continuation-state converter.
An existing canonical mechanical coordinate that differs from the quotient
of a rounded physical bank is refused rather than silently rebased.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
from fractions import Fraction
import math


FORMAT = "native-birth-exact-binary-radial-before-1"


def _binary(value, name, *, positive=False):
    if type(value) is not float or not math.isfinite(value) or (positive and value <= 0.):
        raise ValueError(f"{name} must be a finite literal float" + (" and positive" if positive else ""))
    return Fraction.from_float(value)


def _precision(value):
    if type(value) is not int or value < 100:
        raise ValueError("Exact radial normalization needs at least 100 declared Decimal digits")
    return value


@dataclass(frozen=True, slots=True)
class ExactRadialBeforeCoordinate:
    coordinate_index: int
    bank_id: str
    physical_before_m: Fraction
    coordinate_decimal: str
    normalized_quotient_error: Fraction
    physical_reconstruction_error_m: Fraction
    physical_quotient_error_bound_m: Fraction
    binary64_reconstruction_error_m: Fraction


@dataclass(frozen=True, slots=True)
class ExactRadialBeforeNormalization:
    radius_m: Fraction
    decimal_precision: int
    coordinates: tuple[ExactRadialBeforeCoordinate, ...]
    format: str = FORMAT

    @property
    def maximum_physical_quotient_error_bound_m(self):
        return max((item.physical_quotient_error_bound_m for item in self.coordinates),
            default=Fraction(0))

    @property
    def material_admission_authorized(self):
        return False


def exact_radial_before_coordinate(physical_before_m, radius_m, *, coordinate_index=0,
        bank_id="physical-bank", decimal_precision=160):
    """Divide exact binary physical inputs and retain the signed physical error.

    The returned string is accepted directly by the existing composed contact
    solvers.  The error bound is an exact Fraction, including nonterminating
    quotients; it is not inferred from a printed number of decimal places.
    """
    precision = _precision(decimal_precision)
    physical = _binary(physical_before_m, "Saved physical radial displacement")
    radius = _binary(radius_m, "Physical radius", positive=True)
    if type(coordinate_index) is not int or coordinate_index < 0:
        raise ValueError("The radial coordinate index must be a nonnegative integer")
    if type(bank_id) is not str or not bank_id:
        raise ValueError("The physical bank needs an explicit nonempty identity")
    quotient = physical/radius
    binary_quotient = physical_before_m/radius_m
    if not math.isfinite(binary_quotient):
        raise ValueError("The initial coordinate must have a finite binary64 representative")
    with localcontext(Context(prec=precision, rounding=ROUND_HALF_EVEN)):
        # Decimal.from_float alone is exact, but dividing the Fraction's
        # numerator/denominator states the source and avoids an intervening
        # float quotient on every platform.
        value = Decimal(quotient.numerator)/Decimal(quotient.denominator)
        canonical = str(value)
    normalized_error = Fraction(value)-quotient
    physical_error = normalized_error*radius
    alias = Fraction.from_float(binary_quotient)*radius-physical
    return ExactRadialBeforeCoordinate(coordinate_index, bank_id, physical, canonical,
        normalized_error, physical_error, abs(physical_error), alias)


def normalize_native_birth_radial_before(inputs, previous, *, decimal_precision=None):
    """Copy already-built native inputs, replacing only radial before values.

    The genuine builder authenticates the accepted source.  This helper also
    checks the canonical physical trace order and that the supplied initial
    coordinates belong to these exact saved banks.  It is inexpensive and
    does not rebuild or advance that source.  Angular coordinates and every
    other input retain their original objects, including the already radius
    normalized stiffness block.
    """
    from .geometric_bending_contact_dynamics import _transverse_coordinates

    if type(inputs) is not dict:
        raise ValueError("Already-built native inputs must be a dictionary")
    precision = _precision(inputs["decimal_precision"] if decimal_precision is None else decimal_precision)
    radius_float = inputs["radius_m"]
    radius = _binary(radius_float, "Physical radius", positive=True)
    angular_count = inputs["rotation_dof_count"]
    old_dimension = len(inputs["old_stiffness"])
    foundations = inputs["born_foundations"]
    before = inputs["coordinates_before"]
    if (type(angular_count) is not int or angular_count < 0 or angular_count % 3
            or angular_count > old_dimension or type(before) is not tuple
            or len(before) != old_dimension+len(foundations)):
        raise ValueError("Initial coordinate dimensions must cover every physical bank")
    trace_set = previous.initial_mechanics.fault_traces
    radial = _transverse_coordinates(trace_set, previous.transverse_displacement_m)
    if radial.size != old_dimension-angular_count:
        raise ValueError("Every original radial coordinate needs its unchanged physical trace bank")
    indexed = []
    for trace_index, (trace, row) in enumerate(zip(trace_set.traces, radial)):
        for bank_index, (label, physical) in enumerate(zip(("donor", "receiver"), row)):
            index = angular_count+2*trace_index+bank_index
            indexed.append((index, trace.trace_id+":"+label, float(physical)))
    foundation_indices = [item.radial_dof_index for item in foundations]
    if sorted(foundation_indices) != list(range(old_dimension, len(before))):
        raise ValueError("Each independent foundation must register one distinct extra radial coordinate")
    indexed.extend((item.radial_dof_index, item.reference.patch_id,
        item.previous_state.radial_displacement_m) for item in foundations)
    values, records = list(before), []
    for index, bank_id, physical in indexed:
        record = exact_radial_before_coordinate(physical, radius_float,
            coordinate_index=index, bank_id=bank_id, decimal_precision=precision)
        old_value = before[index]
        if type(old_value) is str:
            if old_value != record.coordinate_decimal:
                raise ValueError("An existing canonical mechanical history cannot be renormalized from a rounded physical bank")
        elif type(old_value) is not float or old_value != physical/radius_float:
            raise ValueError("Initial radial coordinate does not belong to its saved physical bank")
        values[index] = record.coordinate_decimal
        records.append(record)
    normalized = dict(inputs)
    normalized["coordinates_before"] = tuple(values)
    if decimal_precision is not None:
        normalized["decimal_precision"] = precision
    return normalized, ExactRadialBeforeNormalization(radius, precision, tuple(records))


def build_native_birth_exact_radial_inputs(mesh, previous, load, support, dt_s, *,
        decimal_precision=160):
    """Authenticate the genuine source, then normalize its saved banks exactly."""
    from .geometric_native_birth_continuum_motion import build_native_birth_continuum_inputs

    inputs, frames = build_native_birth_continuum_inputs(mesh, previous, load, support, dt_s,
        decimal_precision=decimal_precision)
    inputs, normalization = normalize_native_birth_radial_before(inputs, previous)
    return inputs, frames, normalization


def radial_before_normalization_to_dict(normalization):
    """Expose exact rational error evidence without changing solver inputs."""
    if type(normalization) is not ExactRadialBeforeNormalization:
        raise ValueError("An exact radial normalization certificate is required")
    def rational(value):
        return [value.numerator, value.denominator]
    return dict(format=normalization.format, radius_m=rational(normalization.radius_m),
        decimal_precision=normalization.decimal_precision,
        material_admission_authorized=False,
        maximum_physical_quotient_error_bound_m=rational(normalization.maximum_physical_quotient_error_bound_m),
        coordinates=[dict(coordinate_index=item.coordinate_index, bank_id=item.bank_id,
            physical_before_m=rational(item.physical_before_m), coordinate_decimal=item.coordinate_decimal,
            normalized_quotient_error=rational(item.normalized_quotient_error),
            physical_reconstruction_error_m=rational(item.physical_reconstruction_error_m),
            physical_quotient_error_bound_m=rational(item.physical_quotient_error_bound_m),
            binary64_reconstruction_error_m=rational(item.binary64_reconstruction_error_m))
            for item in normalization.coordinates])
