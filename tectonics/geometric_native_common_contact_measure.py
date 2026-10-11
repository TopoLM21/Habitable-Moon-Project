"""Explicit NEW conditional depth measure on an authentic common cold cut.

The retained source measure C d(theta_computational) is distributed through
the declared physical-volume depth by a normalized nonnegative Bernstein
profile. This conditional profile is a new model. It does not recover an old
three-dimensional memory density, assign old Gauss areas, or transfer rods.
The genuine mapped coarea is retained separately from this reference measure.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from fractions import Fraction as F
import hashlib
import json
from math import comb

from .geometric_native_pose import FractionInterval as I, interval_pi, _rounded_interval
from .geometric_native_common_reference import validate_native_common_reference
from .geometric_native_common_contact_ports import (
    _payload, _encoded, _read_fraction,
    validate_native_common_contact_port_response,
)
from .geometric_native_common_contact_state import (
    _source, validate_native_common_contact_state_observation,
)
from .geometric_native_common_contact_cuts import _RUNTIME, _inverse_map

FORMAT = "native-common-contact-reference-depth-measure-1"
MODEL = "new_normalized_source_reference_area_depth_profile"
SCOPE = "DECLARED_POINT_REFERENCE_MEASURE_NO_GLOBAL_TRANSFER"
_REGISTRY = {}


def _seal(value):
    return hashlib.sha256(_encoded({f.name: getattr(value, f.name)
        for f in fields(value) if f.name != "binding_sha256"})).hexdigest()


def _remember(value):
    value = replace(value, binding_sha256=_seal(value))
    _REGISTRY[type(value), value.binding_sha256] = _encoded(value)
    return value


def _registered(value, kind):
    if (type(value) is not kind or value.binding_sha256 != _seal(value)
            or _REGISTRY.get((kind, value.binding_sha256)) != _encoded(value)):
        raise ValueError("An altered or re-sealed reference measure lacks authentic replay")
    return value


def _profile(coefficients):
    if (type(coefficients) is not tuple or not 1 <= len(coefficients) <= 33
            or any(type(c) is not F or c < 0 for c in coefficients)
            or sum(coefficients, F(0)) != len(coefficients)):
        raise ValueError("The Bernstein profile needs nonnegative exact coefficients with integral one")
    return coefficients


def _unit_interval(value):
    if type(value) is not I or not 0 <= value.lo <= value.hi <= 1:
        raise ValueError("Normalized depth must lie entirely in [0,1]")
    return value


def _profile_power_coefficients(coefficients):
    """Exact polynomial coefficients of sum c_k binom(n,k) z^k(1-z)^(n-k)."""
    coefficients = _profile(coefficients)
    n = len(coefficients)-1
    result = [F(0)]*(n+1)
    for k, c in enumerate(coefficients):
        for j in range(n-k+1):
            result[k+j] += c*comb(n,k)*comb(n-k,j)*(-1)**j
    return tuple(result)


def _profile_integral(coefficients, normalized_depth_bounds):
    z = _unit_interval(normalized_depth_bounds)
    power = _profile_power_coefficients(coefficients)
    result = sum((c*(z.hi**(k+1)-z.lo**(k+1))/(k+1)
                  for k,c in enumerate(power)), F(0))
    if result < 0:
        raise ValueError("A nonnegative normalized profile cannot have negative rectangle mass")
    return result


def _profile_bounds(coefficients, z):
    coefficients, z = _profile(coefficients), _unit_interval(z)
    if len(set(coefficients)) == 1:
        return I(coefficients[0])
    n = len(coefficients)-1
    return sum((c*comb(n,k)*I(z.lo**k,z.hi**k)
        *I((1-z.hi)**(n-k),(1-z.lo)**(n-k))
        for k,c in enumerate(coefficients)), I(0))


def _square(value):
    return I(0 if value.lo <= 0 <= value.hi else min(value.lo**2,value.hi**2),
             max(value.lo**2,value.hi**2))


def _dot(a,b):
    return sum((x*y for x,y in zip(a,b)), I(0))


class _MeasureScope:
    @property
    def scope(self):
        return SCOPE

    @property
    def active_point_depth_profile_model_declared(self):
        return True

    @property
    def inherited_three_dimensional_source_depth_profile(self):
        return False

    @property
    def full_face_memory_measure_transfer_certified(self):
        return False

    @property
    def global_receiver_source_allocation_certified(self):
        return False

    @property
    def old_mechanical_atom_area_assigned(self):
        return False

    @property
    def old_rod_energy_or_prestress_transferred(self):
        return False

    @property
    def source_state_or_thermal_state_advanced(self):
        return False

    @property
    def force_equilibrium_certified(self):
        return False

    @property
    def physical_material_admission_authorized(self):
        return False


@dataclass(frozen=True, slots=True)
class NativeCommonContactReferenceMeasureDeclaration(_MeasureScope):
    source_snapshot_sha256: str
    source_epoch_sha256: str
    reference_snapshot_sha256: str
    cut_snapshot_sha256: str
    port_binding_sha256: str
    receiving_trace_id: str
    receiving_family_id: str
    source_history_binding_sha256: str
    source_current_memory_sha256: str
    source_current_atom_states_sha256: str
    source_current_memory_trace_sha256: str
    source_current_node_epoch: str
    source_native_elapsed_time_s: F
    source_memory_elapsed_time_s: F
    source_memory_stage_count: int
    source_contact_elapsed_time_s: F
    source_contact_step_count: int
    source_radius_m: F
    source_interface_height_m: F
    source_literal_original_frame_sine: F
    source_interface_area_density_m2_per_computational_radian: F
    source_computational_extent_radians: F
    source_reference_interface_area_m2: F
    common_physical_radius_cubed_bounds_m3: I
    physical_volume_depth_bounds_m: I
    physical_volume_depth_span_m: F
    depth_profile_coefficients: tuple
    depth_profile_integral_exact: F
    provenance: str
    precision: int
    binding_sha256: str


def _inputs(reference,port_response,state):
    reference = validate_native_common_reference(reference)
    response = validate_native_common_contact_port_response(port_response,reference=reference)
    if response.port.cut.receiving_trace_id is None or response.port.cut.receiver_material_kind != "original_live":
        raise ValueError("The HOT curve has no source receiver interface measure")
    state = validate_native_common_contact_state_observation(state,
        reference=reference,port_response=response)
    if not response.geometry.genuine_query_support_certified:
        raise ValueError("An unresolved material band has no point reference measure")
    return reference,response,state


def declare_native_common_contact_reference_measure(reference,port_response,state,*,
        provenance,depth_profile_coefficients=(F(1),)):
    """Declare a NEW normalized conditional profile with authentic 1D source C."""
    if type(provenance) is not str or not provenance.strip():
        raise ValueError("The new conditional depth model needs explicit nonempty provenance")
    coefficients = _profile(depth_profile_coefficients)
    reference,response,state = _inputs(reference,port_response,state)
    before,history,path = _source(reference.source_snapshot_sha256)
    tid = response.port.cut.receiving_trace_id
    atoms = [a for a in history["quadrature_histories"] if a["trace_id"] == tid]
    if len(atoms) != 4 or any(a["original_frame"] != atoms[0]["original_frame"] for a in atoms):
        raise ValueError("The authentic receiver must retain its four original frame declarations")
    R = F.from_float(history["origin_surface"]["radius_km"])*1000
    height = F.from_float(atoms[0]["original_material"]["contact_height_m"])
    sine = F.from_float(atoms[0]["original_frame"][0])
    if R != reference.radius_m or min(R,height,sine) <= 0 or height != state.source_interface_height_m:
        raise ValueError("Original interface radius, height or literal frame sine differs from source")
    current = json.loads(state.source_current_memory_trace_json)
    extent = F.from_float(current["extent_radians"])
    cube = response.port.cut.common_physical_radius_cubed_bounds_m3
    tau = I((R**3-cube.hi)/(3*R**2),(R**3-cube.lo)/(3*R**2))
    span = tau.hi-tau.lo
    if span <= 0 or extent <= 0:
        raise ValueError("The authentic cut and source angle must have positive complete spans")
    C = R*height/sine
    value = NativeCommonContactReferenceMeasureDeclaration(
        reference.source_snapshot_sha256,reference.source_epoch_sha256,reference.reference_snapshot_sha256,
        response.port.cut.cut_snapshot_sha256,response.port.binding_sha256,tid,response.port.cut.receiving_family_id,
        state.source_history_binding_sha256,state.source_current_memory_sha256,state.source_current_atom_states_sha256,
        hashlib.sha256(state.source_current_memory_trace_json.encode()).hexdigest(),state.source_current_node_epoch,
        state.source_native_elapsed_time_s,state.source_memory_elapsed_time_s,state.source_memory_stage_count,
        state.source_contact_elapsed_time_s,state.source_contact_step_count,R,height,sine,C,extent,C*extent,
        cube,tau,span,coefficients,_profile_integral(coefficients,I(0,1)),provenance,state.precision,"")
    return _remember(value)


def validate_native_common_contact_reference_measure_declaration(value,*,reference=None,port_response=None,state=None):
    value = _registered(value,NativeCommonContactReferenceMeasureDeclaration)
    if reference is not None and value.reference_snapshot_sha256 != validate_native_common_reference(reference).reference_snapshot_sha256:
        raise ValueError("The depth declaration belongs to another common reference")
    if port_response is not None:
        response = validate_native_common_contact_port_response(port_response,reference=reference)
        if (value.cut_snapshot_sha256 != response.port.cut.cut_snapshot_sha256
                or value.port_binding_sha256 != response.port.binding_sha256):
            raise ValueError("A depth declaration cannot borrow another material cut")
    if state is not None:
        state = validate_native_common_contact_state_observation(state,reference=reference,port_response=port_response)
        if (value.cut_snapshot_sha256 != state.cut_snapshot_sha256
                or value.port_binding_sha256 != state.port_binding_sha256
                or value.reference_snapshot_sha256 != state.reference_snapshot_sha256
                or value.receiving_trace_id != state.receiving_trace_id
                or value.source_current_memory_sha256 != state.source_current_memory_sha256
                or value.source_current_atom_states_sha256 != state.source_current_atom_states_sha256
                or value.source_history_binding_sha256 != state.source_history_binding_sha256
                or value.source_current_memory_trace_sha256 != hashlib.sha256(state.source_current_memory_trace_json.encode()).hexdigest()
                or (value.source_native_elapsed_time_s,value.source_memory_elapsed_time_s,value.source_memory_stage_count,
                    value.source_contact_elapsed_time_s,value.source_contact_step_count)
                    != (state.source_native_elapsed_time_s,state.source_memory_elapsed_time_s,state.source_memory_stage_count,
                        state.source_contact_elapsed_time_s,state.source_contact_step_count)):
            raise ValueError("The declared measure cannot substitute current source memory or clocks")
    return value


def source_reference_rectangle_area(declaration,*,computational_angle_bounds,normalized_depth_bounds):
    """Exact C*theta-width*profile integral: a SOURCE budget, not cut support."""
    declaration = validate_native_common_contact_reference_measure_declaration(declaration)
    theta = computational_angle_bounds
    if type(theta) is not I or not 0 <= theta.lo <= theta.hi <= declaration.source_computational_extent_radians:
        raise ValueError("A source reference rectangle leaves the original computational extent")
    return (declaration.source_interface_area_density_m2_per_computational_radian
        *(theta.hi-theta.lo)*_profile_integral(declaration.depth_profile_coefficients,normalized_depth_bounds))


@dataclass(frozen=True, slots=True)
class NativeCommonContactReferenceMeasureResponse(_MeasureScope):
    declaration_binding_sha256: str
    source_snapshot_sha256: str
    reference_snapshot_sha256: str
    cut_snapshot_sha256: str
    port_binding_sha256: str
    geometry_bounds_sha256: str
    source_state_observation_binding_sha256: str
    receiving_trace_id: str
    computational_angle_bounds: I
    receiver_projection_coordinate_bounds: tuple
    receiver_projection_coordinate_parameter_derivative_bounds: tuple
    projection_squared_radius_bounds: I
    receiving_native_angle_parameter_derivative_bounds: I
    receiving_computational_angle_parameter_derivative_bounds: I
    receiving_angle_orientation: int
    normalized_depth_bounds: I
    depth_profile_value_bounds: I
    reference_area_density_bounds_m2_per_parameter_per_physical_volume_depth_m: I
    physical_area_density_bounds_m2_per_parameter_per_physical_volume_depth_m: I
    reference_to_physical_area_density_ratio_bounds: I
    source_plastic_slip_bounds_m: I
    source_cumulative_slip_bounds_m: I
    source_current_memory_sha256: str
    precision: int
    binding_sha256: str


def evaluate_native_common_contact_reference_measure(reference,port_response,state,declaration,*,precision=160):
    """Reference density and its ratio to genuine physical coarea at one query."""
    if type(precision) is not int or not 96 <= precision <= 512:
        raise ValueError("Reference measure precision must be an integer from 96 to 512")
    reference,response,state = _inputs(reference,port_response,state)
    declaration = validate_native_common_contact_reference_measure_declaration(declaration,
        reference=reference,port_response=response,state=state)
    geometry,cut = response.geometry,response.port.cut
    runtime = _RUNTIME[cut.cut_snapshot_sha256]
    owner = response.port.receiver_body.embedding_bank.owner
    q = geometry.receiver_owner_local_unit_direction_bounds
    qp = _inverse_map(runtime["poses"][owner],geometry.source_unit_world_direction_parameter_derivative_bounds)
    trace = json.loads(state.source_trace_json)
    S,T = (tuple(I(F.from_float(float(x))) for x in trace[k]) for k in ("start","end"))
    x,y,xp,yp = _dot(S,q),_dot(T,q),_dot(S,qp),_dot(T,qp)
    denominator = _square(x)+_square(y)
    if denominator.lo <= 0:
        raise ValueError("The receiver angle projection is not strictly regular")
    native_derivative = (x*yp-y*xp)/denominator
    if native_derivative.lo <= 0 <= native_derivative.hi:
        raise ValueError("The complete receiver angle derivative must have a strict orientation")
    derivative = native_derivative*declaration.source_computational_extent_radians/(interval_pi(precision+16)/2)
    orientation = 1 if derivative.lo > 0 else -1
    absolute_derivative = derivative if orientation == 1 else -derivative
    zeta = (geometry.common_physical_volume_depth_bounds_m-declaration.physical_volume_depth_bounds_m.lo)/declaration.physical_volume_depth_span_m
    _unit_interval(zeta)
    profile = _profile_bounds(declaration.depth_profile_coefficients,zeta)
    density = declaration.source_interface_area_density_m2_per_computational_radian*absolute_derivative*profile/declaration.physical_volume_depth_span_m
    physical = geometry.area_density_bounds_m2_per_parameter_per_material_m
    if physical.lo <= 0 or density.lo < 0:
        raise ValueError("The declared reference and genuine physical measures must be nonnegative/regular")
    rounded = lambda value:_rounded_interval(value,precision+16)
    return _remember(NativeCommonContactReferenceMeasureResponse(declaration.binding_sha256,
        reference.source_snapshot_sha256,reference.reference_snapshot_sha256,cut.cut_snapshot_sha256,
        response.port.binding_sha256,geometry.bounds_snapshot_sha256,state.binding_sha256,state.receiving_trace_id,
        state.computational_angle_bounds,(x,y),(xp,yp),rounded(denominator),rounded(native_derivative),rounded(derivative),
        orientation,zeta,rounded(profile),rounded(density),physical,rounded(density/physical),
        state.plastic_slip_bounds_m,state.cumulative_slip_bounds_m,state.source_current_memory_sha256,precision,""))


def validate_native_common_contact_reference_measure_response(value,*,reference=None,port_response=None,state=None,declaration=None):
    value = _registered(value,NativeCommonContactReferenceMeasureResponse)
    if reference is not None and value.reference_snapshot_sha256 != validate_native_common_reference(reference).reference_snapshot_sha256:
        raise ValueError("A measure response belongs to another common reference")
    if port_response is not None:
        response = validate_native_common_contact_port_response(port_response,reference=reference)
        if value.geometry_bounds_sha256 != response.geometry.bounds_snapshot_sha256 or value.port_binding_sha256 != response.port.binding_sha256:
            raise ValueError("A measure response belongs to another physical material query")
    if state is not None and value.source_state_observation_binding_sha256 != validate_native_common_contact_state_observation(
            state,reference=reference,port_response=port_response).binding_sha256:
        raise ValueError("A measure response substituted its current source observation")
    if declaration is not None and value.declaration_binding_sha256 != validate_native_common_contact_reference_measure_declaration(
            declaration,reference=reference,port_response=port_response,state=state).binding_sha256:
        raise ValueError("A measure response substituted its conditional depth declaration")
    return value


_SCOPE_FLAGS = ("active_point_depth_profile_model_declared","inherited_three_dimensional_source_depth_profile",
    "full_face_memory_measure_transfer_certified","global_receiver_source_allocation_certified",
    "old_mechanical_atom_area_assigned","old_rod_energy_or_prestress_transferred",
    "source_state_or_thermal_state_advanced","force_equilibrium_certified","physical_material_admission_authorized")


def _export(value):
    return dict(format=FORMAT,model=MODEL,scope=SCOPE,**_payload(value),
                **{name:getattr(value,name) for name in _SCOPE_FLAGS})


def native_common_contact_reference_measure_declaration_to_dict(value):
    return _export(validate_native_common_contact_reference_measure_declaration(value))


def native_common_contact_reference_measure_response_to_dict(value):
    return _export(validate_native_common_contact_reference_measure_response(value))


def native_common_contact_reference_measure_declaration_from_dict(raw,*,reference,port_response,state):
    if type(raw) is not dict or raw.get("format") != FORMAT:
        raise ValueError("Expected a complete declared reference-depth measure")
    value = declare_native_common_contact_reference_measure(reference,port_response,state,
        provenance=raw["provenance"],depth_profile_coefficients=tuple(_read_fraction(x) for x in raw["depth_profile_coefficients"]))
    if _encoded(raw) != _encoded(native_common_contact_reference_measure_declaration_to_dict(value)):
        raise ValueError("Serialized reference measure differs from source, model or clocks")
    return value


def native_common_contact_reference_measure_response_from_dict(raw,*,reference,port_response,state,declaration):
    if type(raw) is not dict or raw.get("format") != FORMAT or type(raw.get("precision")) is not int:
        raise ValueError("Expected a complete reference measure response")
    value = evaluate_native_common_contact_reference_measure(reference,port_response,state,declaration,precision=raw["precision"])
    if _encoded(raw) != _encoded(native_common_contact_reference_measure_response_to_dict(value)):
        raise ValueError("Serialized reference density differs from authentic geometry/source replay")
    return value
