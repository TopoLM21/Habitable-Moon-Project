"""Explicit elastic cantilevers attached to frozen physical fault traces.

Each bank is an independent locally planar strip with outward radial trench
tip displacement ``w``. Distance ``x`` runs inward from the trench to a clamp
at ``width_m``. The free trench rotation and zero end moment give the exact
tip-load shape ``phi(x/L) = 1 - 3*x/(2*L) + x**3/(2*L**3)``. The strip is
uniform along the physical trace, so its stiffness is ``3*D*trace_length/L**3``.

This module supplies reversible bending only. It infers no elastic thickness,
clamp distance, gravity foundation, viscosity, or existing slab hinge drag.
Tangential column strain, along-strike curvature, and spherical membrane
strain are distinct mechanisms owned by their explicitly declared models.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from numbers import Real

import numpy as np

from .flexure import FlexureParameters, flexural_rigidity_nm


FORMAT = "geometric-cantilever-bending-1"


def _number(value, name, *, positive=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(float(value))):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if positive and value <= 0.:
        raise ValueError(f"{name} must be positive")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be an explicit nonempty string")
    return value


@dataclass(frozen=True, slots=True)
class ElasticBendingBankParameters:
    """Physical clamp distance and isotropic thin-strip elastic material.

    ``width_m`` is the inward distance from trench to clamp, never a material
    fragment or numerical cell size. ``elastic_thickness_m`` must be supplied
    separately from any normal/shear column length or cold slab thickness.
    The admissible Poisson range is the full isotropic three-dimensional range.
    """
    width_m: float
    elastic_thickness_m: float
    young_modulus_pa: float
    poisson_ratio: float
    provenance: str

    def __post_init__(self):
        for name in ("width_m", "elastic_thickness_m", "young_modulus_pa"):
            object.__setattr__(self, name, _number(getattr(self, name), name, positive=True))
        nu = _number(self.poisson_ratio, "poisson_ratio")
        if not -1. < nu < .5:
            raise ValueError("poisson_ratio must lie strictly between -1 and .5")
        object.__setattr__(self, "poisson_ratio", nu)
        _text(self.provenance, "provenance")
        if self.elastic_thickness_m >= self.width_m:
            raise ValueError("elastic_thickness_m must be smaller than the declared clamp distance")
        bending_bank_rigidity_nm(self)


@dataclass(frozen=True, slots=True)
class FaultBendingParameters:
    trace_id: str
    sub: ElasticBendingBankParameters
    over: ElasticBendingBankParameters

    def __post_init__(self):
        _text(self.trace_id, "trace_id")
        _bank(self.sub)
        _bank(self.over)


def _bank(bank):
    if not isinstance(bank, ElasticBendingBankParameters):
        raise ValueError("Explicit ElasticBendingBankParameters are required")
    return bank


def bending_bank_rigidity_nm(bank):
    """Return ``E*H**3/[12*(1-nu**2)]`` in N m without material inference.

    Reuse the established rigidity law with explicitly supplied material.
    Undo its diagnostic-only denominator floor for valid auxetic materials
    near nu=-1: the new mechanical model uses the exact declared denominator.
    """
    _bank(bank)
    denominator = 1.-bank.poisson_ratio**2
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        rigidity = float(flexural_rigidity_nm(bank.elastic_thickness_m/1000.,
            FlexureParameters(young_modulus_pa=bank.young_modulus_pa,
                              poisson_ratio=bank.poisson_ratio)))
        if denominator < 1e-12:
            rigidity *= 1e-12/denominator
    if not math.isfinite(rigidity) or rigidity <= 0.:
        raise ValueError("Declared elastic material must give finite positive bending rigidity")
    return rigidity


def _coordinate(x_over_width):
    array = np.asarray(x_over_width)
    original = np.asarray(x_over_width, dtype=object)
    if (array.dtype.kind not in "iuf" or not np.isfinite(array).all()
            or any(isinstance(value, (bool, np.bool_)) for value in original.flat)):
        raise ValueError("Cantilever coordinate must be finite and real")
    array = np.asarray(array, dtype=float)
    if np.any(array < 0.) or np.any(array > 1.):
        raise ValueError("Cantilever coordinate x/width must lie in [0, 1]")
    return array


def _value(array):
    return float(array) if array.ndim == 0 else array


def cantilever_shape(x_over_width):
    """Unit trench displacement, zero trench moment, and clamped far bank."""
    x = _coordinate(x_over_width)
    return _value(.5*(1.-x)**2*(x+2.))


def cantilever_shape_slope(x_over_width):
    """Dimensionless d phi / d(x/L); physical slope is w/L times this."""
    x = _coordinate(x_over_width)
    return _value(-1.5*(1.-x)*(1.+x))


def cantilever_shape_curvature(x_over_width):
    """Dimensionless d² phi / d(x/L)²; physical curvature is w/L² times this."""
    x = _coordinate(x_over_width)
    return _value(3.*x)


def bending_strip_stiffness_n_m(bank, trace_length_m):
    """Positive tip stiffness of a declared whole-trace cantilever [N/m]."""
    length = _number(trace_length_m, "trace_length_m", positive=True)
    _bank(bank)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        stiffness = float(3.*np.float64(bending_bank_rigidity_nm(bank))*length
                          /np.float64(bank.width_m)**3)
    if not math.isfinite(stiffness) or stiffness <= 0.:
        raise ValueError("Physical strip dimensions must give finite positive bending stiffness")
    return stiffness


def bank_energy_j(bank, trace_length_m, tip_displacement_m):
    displacement = _number(tip_displacement_m, "tip_displacement_m")
    stiffness = bending_strip_stiffness_n_m(bank, trace_length_m)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        energy = float(.5*stiffness*np.float64(displacement)**2)
    if not math.isfinite(energy):
        raise ValueError("Bending energy overflowed")
    return energy


def bank_restoring_force_n(bank, trace_length_m, tip_displacement_m):
    """Beam force on the trench tip, positive outward; equals -dU/dw."""
    displacement = _number(tip_displacement_m, "tip_displacement_m")
    force = -bending_strip_stiffness_n_m(bank, trace_length_m)*displacement
    if not math.isfinite(force):
        raise ValueError("Bending restoring force overflowed")
    return force


def bank_max_tip_slope(bank, tip_displacement_m):
    _bank(bank)
    displacement = _number(tip_displacement_m, "tip_displacement_m")
    slope = 1.5*(abs(displacement)/bank.width_m)
    if not math.isfinite(slope):
        raise ValueError("Bending tip slope overflowed")
    return slope


def bank_max_surface_strain(bank, tip_displacement_m):
    """Maximum fibre strain H/2 times maximum curvature [dimensionless]."""
    _bank(bank)
    return (bank.elastic_thickness_m/bank.width_m
            *bank_max_tip_slope(bank, tip_displacement_m))


def canonical_bending_parameters(parameters, *, trace_ids=None):
    """Canonical unique physical trace binding, optionally requiring exact coverage."""
    try:
        parameters = tuple(parameters)
    except TypeError as error:
        raise ValueError("Explicit fault bending parameters are required") from error
    if any(not isinstance(parameter, FaultBendingParameters) for parameter in parameters):
        raise ValueError("Every physical trace needs FaultBendingParameters")
    result = tuple(sorted(parameters, key=lambda parameter: parameter.trace_id))
    identities = tuple(parameter.trace_id for parameter in result)
    if len(set(identities)) != len(identities):
        raise ValueError("Physical trace bending parameters cannot repeat")
    if trace_ids is not None:
        expected = tuple(trace_ids)
        if (any(not isinstance(identity, str) or not identity.strip() for identity in expected)
                or len(set(expected)) != len(expected) or set(identities) != set(expected)):
            raise ValueError("Bending parameters must cover every physical trace exactly once")
    return result


def bending_parameters_to_dict(parameters):
    """Strict JSON-compatible canonical parameter list, including provenance."""
    return [asdict(parameter) for parameter in canonical_bending_parameters(parameters)]


def bending_parameters_from_dict(raw):
    if not isinstance(raw, list):
        raise ValueError("Bending parameters must be an explicit canonical list")
    bank_names = {field.name for field in fields(ElasticBendingBankParameters)}
    fault_names = {field.name for field in fields(FaultBendingParameters)}
    result = []
    for item in raw:
        if not isinstance(item, dict) or set(item) != fault_names:
            raise ValueError("Invalid fault bending parameter schema")
        banks = []
        for name in ("sub", "over"):
            if not isinstance(item[name], dict) or set(item[name]) != bank_names:
                raise ValueError("Invalid elastic bending bank schema")
            banks.append(ElasticBendingBankParameters(**item[name]))
        result.append(FaultBendingParameters(item["trace_id"], *banks))
    canonical = canonical_bending_parameters(result)
    if tuple(result) != canonical:
        raise ValueError("Bending parameter list must be in canonical physical trace order")
    return canonical


def bending_parameters_sha256(parameters):
    """Bind restart parameters to exact declared physics and provenance."""
    payload = {"format": FORMAT, "traces": bending_parameters_to_dict(parameters)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
