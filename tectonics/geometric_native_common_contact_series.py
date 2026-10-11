"""Explicit reduced serial carrier driven by authentic common body traces.

This declares a new anchor interpretation.  Contact and the two source
columns are statically condensed in series, rather than added in parallel to
the contact-only853 response.  Literal scalar preload and variable mechanical
energy are retained; physical 3D endpoint/material, area and old world-force
handover are deliberately not inferred.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from fractions import Fraction as F
import hashlib
import json

from .geometric_native_pose import FractionInterval as I
from .geometric_native_common_reference import validate_native_common_reference
from .geometric_native_common_contact_ports import validate_native_common_contact_port_response
from .geometric_native_common_contact_state import validate_native_common_contact_state_observation
from .geometric_native_common_contact_constitutive import (
    FRAME, BRANCHES, _payload, _encoded, _seal, _round, _square, _absolute,
    evaluate_native_common_contact_scalar_kinematics,
    validate_native_common_contact_scalar_kinematics,
)


FORMAT = "native-common-explicit-reduced-contact-column-series-1"
SCOPE = "DECLARED_REDUCED_SERIAL_CARRIER_PER_REFERENCE_AREA"
ANCHOR_INTERPRETATION = "common_body_trace_increment_drives_eliminated_serial_carrier"
_REGISTRY = {}
_DEPENDENCIES = {}


def _remember(value, dependencies):
    result = replace(value, binding_sha256=_seal(value))
    _REGISTRY[type(result), result.binding_sha256] = _encoded(result)
    _DEPENDENCIES[type(result), result.binding_sha256] = dependencies
    return result


def _registered(value, kind):
    if (type(value) is not kind or value.binding_sha256 != _seal(value)
            or _REGISTRY.get((kind, value.binding_sha256)) != _encoded(value)):
        raise ValueError("An altered or re-sealed serial carrier lacks authentic explicit declaration/replay")
    return _DEPENDENCIES[kind, value.binding_sha256]


def _inputs(reference, port_response, source_observation):
    reference = validate_native_common_reference(reference)
    port = validate_native_common_contact_port_response(port_response, reference=reference)
    if port.receiver_fields.material_kind != "original_live" or port.port.cut.receiving_trace_id is None:
        raise ValueError("The actual HOT receiver has no source serial-carrier contact law")
    state = validate_native_common_contact_state_observation(source_observation,
        reference=reference, port_response=port)
    if (not state.compressed_source_observation or not state.source_relaxed_friction_free_branch
            or state.shear_traction_bounds_pa != I(0)):
        raise ValueError("The declared serial carrier needs the genuine compressed zero-shear source branch")
    return reference, port, state


def _literal(value):
    if type(value) is not float:
        raise ValueError("Serial columns must use complete literal binary64 source parameters")
    return F.from_float(value)


def _exact_coefficients(kn, kt, columns):
    """Exact compliance sum, with no rounded source harmonic substituted."""
    Ls, Es, Gs, Lo, Eo, Go = columns
    if any(type(x) is not F or x <= 0 for x in (kn, kt)+columns):
        raise ValueError("Positive source contact stiffness and individual column literals are required")
    normal, shear = (Ls/Es, Lo/Eo), (Ls/Gs, Lo/Go)
    return normal, shear, 1/(1/kn+sum(normal)), 1/(1/kt+sum(shear))


class _SeriesScope:
    @property
    def scope(self):
        return SCOPE

    @property
    def source_scalar_prestress_preserved(self):
        return True

    @property
    def variable_mechanical_energy_and_derivatives_defined(self):
        return True

    @property
    def additive_to_853_contact_only_energy_authorized(self):
        return False

    @property
    def old_rod_three_dimensional_material_map_certified(self):
        return False

    @property
    def old_world_or_generalized_force_handover_certified(self):
        return False

    @property
    def physical_interface_area_or_memory_allocation_certified(self):
        return False

    @property
    def all_sixteen_source_carriers_physically_transferred(self):
        return False

    @property
    def old_rod_energy_added_to_existing_common_bulk_prestrain(self):
        return False

    @property
    def force_equilibrium_certified(self):
        return False

    @property
    def physical_material_admission_authorized(self):
        return False

    @property
    def source_history_or_thermal_state_advanced(self):
        return False


@dataclass(frozen=True, slots=True)
class NativeCommonContactSeriesCarrierDeclaration(_SeriesScope):
    source_snapshot_sha256: str
    reference_snapshot_sha256: str
    cut_snapshot_sha256: str
    geometry_bounds_sha256: str
    port_binding_sha256: str
    source_observation_binding_sha256: str
    receiving_trace_id: str
    provenance: str
    anchor_interpretation: str
    receiver_frame: str
    source_trace_json_sha256: str
    individual_column_literal_parameters: tuple
    normal_contact_stiffness_pa_m: F
    tangential_contact_stiffness_pa_m: F
    individual_normal_compliances_m_pa: tuple
    individual_shear_compliances_m_pa: tuple
    exact_normal_series_stiffness_pa_m: F
    exact_shear_series_stiffness_pa_m: F
    source_binary64_normal_series_stiffness_pa_m: F
    source_binary64_shear_series_stiffness_pa_m: F
    exact_minus_source_normal_series_stiffness_pa_m: F
    exact_minus_source_shear_series_stiffness_pa_m: F
    exact_minus_source_normal_column_stiffness_pa_m: F
    exact_minus_source_shear_column_stiffness_pa_m: F
    normal_anchor_reference_bounds_m: I
    shear_anchor_reference_bounds_m: I
    normal_anchor_minus_source_free_gap_bounds_m: I
    shear_anchor_minus_source_free_jump_bounds_m: I
    binding_sha256: str
    anchor_coordinate_origin: str = "ZERO_MODAL_INCREMENT_IN_NEW_COMMON_REFERENCE"
    assembly_role: str = "REPLACEMENT_LOCAL_SERIES_COMPONENT_NOT_PARALLEL_CONTACT853"


def declare_native_common_contact_series_carrier(reference, port_response, source_observation, *,
        provenance, anchor_interpretation):
    reference, port, state = _inputs(reference, port_response, source_observation)
    if type(provenance) is not str or not provenance.strip():
        raise ValueError("A serial anchor interpretation requires explicit nonempty provenance")
    if anchor_interpretation != ANCHOR_INTERPRETATION:
        raise ValueError("The eliminated serial-carrier anchor interpretation must be explicit")
    trace = json.loads(state.source_trace_json)
    c = trace["columns"]
    columns = tuple(_literal(c[key]) for key in (
        "sub_length_m", "sub_normal_modulus_pa", "sub_shear_modulus_pa",
        "over_length_m", "over_normal_modulus_pa", "over_shear_modulus_pa"))
    kn, kt = state.normal_stiffness_pa_m, state.tangential_stiffness_pa_m
    normal, shear, Kn, Kt = _exact_coefficients(kn, kt, columns)
    anchor = -state.normal_pressure_bounds_pa/Kn
    shear_anchor = state.plastic_slip_bounds_m
    value = NativeCommonContactSeriesCarrierDeclaration(
        state.source_snapshot_sha256, reference.reference_snapshot_sha256,
        state.cut_snapshot_sha256, state.geometry_bounds_sha256, port.port.binding_sha256,
        state.binding_sha256, state.receiving_trace_id, provenance, anchor_interpretation, FRAME,
        hashlib.sha256(state.source_trace_json.encode()).hexdigest(), columns, kn, kt,
        normal, shear, Kn, Kt, state.original_series_normal_stiffness_pa_m,
        state.original_series_shear_stiffness_pa_m,
        Kn-state.original_series_normal_stiffness_pa_m,
        Kt-state.original_series_shear_stiffness_pa_m,
        1/sum(normal)-state.original_column_normal_stiffness_pa_m,
        1/sum(shear)-state.original_column_shear_stiffness_pa_m,
        anchor, shear_anchor, anchor-state.free_gap_bounds_m,
        shear_anchor-state.free_jump_bounds_m, "")
    return _remember(value, (reference, port, state))


def validate_native_common_contact_series_carrier_declaration(value, *, reference=None,
        port_response=None, source_observation=None):
    old_reference, old_port, old_state = _registered(value, NativeCommonContactSeriesCarrierDeclaration)
    reference, port, state = _inputs(old_reference if reference is None else reference,
        old_port if port_response is None else port_response,
        old_state if source_observation is None else source_observation)
    if (value.reference_snapshot_sha256 != reference.reference_snapshot_sha256
            or value.port_binding_sha256 != port.port.binding_sha256
            or value.geometry_bounds_sha256 != port.geometry.bounds_snapshot_sha256
            or value.source_observation_binding_sha256 != state.binding_sha256):
        raise ValueError("The serial carrier declaration belongs to another authentic source/anchor query")
    return value


@dataclass(frozen=True, slots=True)
class NativeCommonContactSeriesResponse(_SeriesScope):
    declaration_binding_sha256: str
    source_observation_binding_sha256: str
    port_response_binding_sha256: str
    kinematics: object
    branch: str
    far_anchor_normal_bounds_m: I
    far_anchor_shear_bounds_m: I
    traction_bounds_pa: tuple
    frozen_shear_trial_traction_bounds_pa: I
    local_tangent_bounds_pa_m: tuple
    normal_contact_strain_bounds_m: I
    elastic_contact_shear_strain_bounds_m: I
    contact_jump_bounds_m: I
    individual_rod_normal_shortening_bounds_m: tuple
    individual_rod_shear_extension_bounds_m: tuple
    energy_component_order: tuple
    reference_energy_component_bounds_j_m2: tuple
    reference_total_energy_bounds_j_m2: I
    energy_component_bounds_j_m2: tuple
    total_energy_bounds_j_m2: I
    residual_bounds_per_reference_m2: tuple
    hessian_bounds_per_reference_m2: tuple
    proposed_plastic_increment_bounds_m: I
    proposed_plastic_slip_bounds_m: I
    proposed_cumulative_slip_bounds_m: I
    precision: int
    binding_sha256: str
    source_memory_proposal_applied: bool = False
    symmetric_left_chart_and_receiver_frame_terms_included: bool = True
    frozen_predictor_is_actual_relaxed_shear_law: bool = False


def _energy_split(normal, shear, declaration, precision):
    n2, t2 = _square(normal), _square(shear)
    return (_round(n2/(2*declaration.normal_contact_stiffness_pa_m)
                   +t2/(2*declaration.tangential_contact_stiffness_pa_m), precision),
        *tuple(_round(n2*cn/2+t2*ct/2, precision) for cn, ct in zip(
            declaration.individual_normal_compliances_m_pa,
            declaration.individual_shear_compliances_m_pa)))


def evaluate_native_common_contact_series_carrier(reference, port_response, source_observation,
        declaration, *, branch="relaxed_source", precision=160):
    reference, port, state = _inputs(reference, port_response, source_observation)
    declaration = validate_native_common_contact_series_carrier_declaration(declaration,
        reference=reference, port_response=port, source_observation=state)
    if branch not in BRANCHES:
        raise ValueError("Choose the relaxed source or the explicitly frozen-memory serial predictor")
    kinematics = evaluate_native_common_contact_scalar_kinematics(reference, port, precision=precision)
    xi_n, xi_t, _ = kinematics.scalar_increment_bounds_m
    Kn, Kt = declaration.exact_normal_series_stiffness_pa_m, declaration.exact_shear_series_stiffness_pa_m
    normal = -state.normal_pressure_bounds_pa if xi_n == I(0) else _round(
        -state.normal_pressure_bounds_pa+Kn*xi_n, precision)
    if normal.hi >= 0:
        raise ValueError("Split or reject the serial query before contact opening")
    trial = _round(Kt*xi_t, precision)
    relaxed = branch == "relaxed_source"
    shear = I(0) if relaxed else trial
    stiffness = I(Kn), I(0) if relaxed else I(Kt)
    tractions = normal, shear
    components = _energy_split(normal, shear, declaration, precision)
    reference_components = _energy_split(-state.normal_pressure_bounds_pa, I(0), declaration, precision)
    first, second = kinematics.scalar_first_derivative_bounds, kinematics.scalar_second_derivative_bounds
    count = len(kinematics.coordinate_order)
    residual = tuple(_round(sum((tractions[a]*first[a][i] for a in range(2)), I(0)), precision)
        for i in range(count))
    hessian = [[I(0) for _ in range(count)] for _ in range(count)]
    for i in range(count):
        for j in range(i, count):
            value = _round(sum((stiffness[a]*first[a][i]*first[a][j]
                +tractions[a]*second[a][i][j] for a in range(2)), I(0)), precision)
            hessian[i][j] = hessian[j][i] = value
    increment = xi_t if relaxed else I(0)
    plastic = _round(state.plastic_slip_bounds_m+increment, precision)
    cumulative = _round(state.cumulative_slip_bounds_m+_absolute(increment), precision)
    contact_jump = plastic if relaxed else _round(state.plastic_slip_bounds_m
        +shear/declaration.tangential_contact_stiffness_pa_m, precision)
    value = NativeCommonContactSeriesResponse(declaration.binding_sha256,
        state.binding_sha256, port.binding_sha256, kinematics, branch,
        declaration.normal_anchor_reference_bounds_m+xi_n,
        declaration.shear_anchor_reference_bounds_m+xi_t,
        tractions, trial, ((stiffness[0], I(0)), (I(0), stiffness[1])),
        normal/declaration.normal_contact_stiffness_pa_m,
        shear/declaration.tangential_contact_stiffness_pa_m, contact_jump,
        tuple(-normal*c for c in declaration.individual_normal_compliances_m_pa),
        tuple(-shear*c for c in declaration.individual_shear_compliances_m_pa),
        ("contact", "sub_column", "over_column"), reference_components,
        sum(reference_components, I(0)), components, sum(components, I(0)),
        residual, tuple(tuple(row) for row in hessian), increment, plastic, cumulative, precision, "")
    return _remember(value, (reference, port, state, declaration))


def validate_native_common_contact_series_response(value, *, reference=None, port_response=None,
        source_observation=None, declaration=None):
    old_reference, old_port, old_state, old_declaration = _registered(value, NativeCommonContactSeriesResponse)
    reference, port, state = _inputs(old_reference if reference is None else reference,
        old_port if port_response is None else port_response,
        old_state if source_observation is None else source_observation)
    declaration = validate_native_common_contact_series_carrier_declaration(
        old_declaration if declaration is None else declaration,
        reference=reference, port_response=port, source_observation=state)
    validate_native_common_contact_scalar_kinematics(value.kinematics, reference=reference, port_response=port)
    if (value.declaration_binding_sha256 != declaration.binding_sha256
            or value.source_observation_binding_sha256 != state.binding_sha256
            or value.port_response_binding_sha256 != port.binding_sha256):
        raise ValueError("Serial response has different declared authentic dependencies")
    return value


def _export(value):
    return dict(format=FORMAT, scope=SCOPE, **_payload(value),
        source_scalar_prestress_preserved=True, variable_mechanical_energy_and_derivatives_defined=True,
        additive_to_853_contact_only_energy_authorized=False,
        old_rod_three_dimensional_material_map_certified=False,
        old_world_or_generalized_force_handover_certified=False,
        physical_interface_area_or_memory_allocation_certified=False,
        all_sixteen_source_carriers_physically_transferred=False,
        old_rod_energy_added_to_existing_common_bulk_prestrain=False,
        force_equilibrium_certified=False, physical_material_admission_authorized=False,
        source_history_or_thermal_state_advanced=False)


def native_common_contact_series_carrier_declaration_to_dict(value):
    return _export(validate_native_common_contact_series_carrier_declaration(value))


def native_common_contact_series_carrier_declaration_from_dict(raw, *, reference, port_response, source_observation):
    if type(raw) is not dict:
        raise ValueError("A serial carrier declaration export must be a mapping")
    value = declare_native_common_contact_series_carrier(reference, port_response, source_observation,
        provenance=raw["provenance"], anchor_interpretation=raw["anchor_interpretation"])
    if native_common_contact_series_carrier_declaration_to_dict(value) != raw:
        raise ValueError("Serial carrier declaration differs from complete authentic replay")
    return value


def native_common_contact_series_response_to_dict(value):
    return _export(validate_native_common_contact_series_response(value))


def native_common_contact_series_response_from_dict(raw, *, reference, port_response, source_observation, declaration):
    if type(raw) is not dict:
        raise ValueError("A serial response export must be a mapping")
    value = evaluate_native_common_contact_series_carrier(reference, port_response, source_observation,
        declaration, branch=raw["branch"], precision=raw["precision"])
    if native_common_contact_series_response_to_dict(value) != raw:
        raise ValueError("Serial response differs from complete source/declaration/port replay")
    return value
