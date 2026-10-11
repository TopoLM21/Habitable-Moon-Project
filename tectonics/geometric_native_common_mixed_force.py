"""Declared COMMON100 endpoint force assembly on original source atoms.

Whole cold899 and HOT903 operators are reused at their authenticated saved
queries.  Eight residual coordinates shift the existing cold beam driver
once.  The current owner load is a NEW frozen increment potential.  Contacts
are NEW original-source-atom laws on current cold895 material points, not a
complete born-face or physical whole-clock mixed equilibrium.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from fractions import Fraction as F
import hashlib
import json

from .geometric_native_pose import FractionInterval as I, interval_sqrt
from .geometric_native_common_carrier_source_work import (
    _payload, _encoded, _digest, _typed_guard, _fraction, _interval,
    _vector, _matrix, _round, _packet, _multiply, _inverse,
)
from .geometric_native_common_cold_material_carrier import (
    _registered as _registered_cold_carrier,
    evaluate_native_common_cold_material_point,
    native_common_cold_material_point_to_dict,
    _saved_json_kinds,
)

FORMAT = "native-common-hundred-mixed-endpoint-system-1"
RESPONSE_FORMAT = "native-common-hundred-mixed-endpoint-response-1"
SCOPE = "DECLARED_ORIGINAL_SOURCE_ATOM_HUNDRED_COORDINATE_ENDPOINT_ASSEMBLY_ONLY"
OWNER_LOAD_MODEL = "NEW_FROZEN_OWNER_LOAD_DRAG_INCREMENT_POTENTIAL"
CONTACT_MODEL = "NEW_CURRENT_UNIT_NANSON_NORMAL_ORIGINAL_MATERIAL_ATOM_MEASURE_ONCE"
INTERNAL_MODEL = "EIGHT_RESIDUALS_SHIFT_EXISTING_COLD_CONTINUUM_BEAM_DRIVER_ONCE"
AUTHORITY_PINS = (
    ("903", "0f8a0ee8ae082e3434b136bab7515944354eaf059db95965d0e87259f6791035",
     "c53e4cd75219328eeac974f3b9982aa33ad78eaf2e07d823feafb9f07b289e6c"),
    ("899", "4a5c059176cef063bf12ff7bd98df830395504d6c12051f269a177fcbc375fec",
     "c863be05c9d641025666a1a4c3929701c7d472f9d574085926d64a2d08b4c439"),
    ("891", "b45d5d4d80c74c509bfac033c4a60c5b96c8327704e0c727fcf680406d82e57e",
     "1f1b07b977bbb092e08281d8d200a26f8bf9ab85e2a18f001b2061275f742c40"),
    ("879", "f2884284f663564d304537f91614efbb01e5b7df5ebdab86ac7eb6e77287ee64",
     "368a524a954ac5f2ba3a6b92c182d34d1c3567ba7df1314c0afd02b2cff2c23e"),
)
REMAINING_CONDITIONS = (
    "Whole-source physical material and old-force/prestress/work handover, HOT law/mean chart, loads/drag and NEW mixed equilibrium",
    "Whole-clock all-pair admission including reverse C5 to C1",
    "Accepted continuation with independent path work, balances and joint restart",
)
_SYSTEMS, _CONTEXTS, _RESPONSES, _EVALUATIONS = {}, {}, {}, {}
_PROJECTION_JETS, _CONTACT_CACHE, _CONTACT_AUTHORITIES = {}, {}, {}


def _json(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)


def _raw_digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _body(value):
    return {f.name:getattr(value,f.name) for f in fields(value) if f.name!="binding_sha256"}


def _remember(value,registry):
    value=replace(value,binding_sha256=_digest(_body(value)))
    registry[value.binding_sha256]=_typed_guard(value)
    return value


def _validate(value,cls,registry):
    if (type(value) is not cls or _digest(_body(value))!=value.binding_sha256
            or registry.get(value.binding_sha256)!=_typed_guard(value)):
        raise ValueError("The mixed endpoint changed its complete authentic typed binding")
    return value


def _precision(value):
    if type(value) is not int or not 96<=value<=512:
        raise ValueError("The mixed endpoint precision is a literal integer in [96,512]")
    return value


def _square(value):
    value=value if type(value) is I else I(value)
    return I(F(0) if value.lo<=0<=value.hi else min(value.lo**2,value.hi**2),
             max(value.lo**2,value.hi**2))


def _hessian_units(i,j):
    if i<6:
        return "N*m/rad" if j<6 else "N"
    return "N/rad" if j<6 else "N/m"


def _saved_case(packet,coordinates,key):
    matches=[row for row in packet["evaluation_queries"]
             if tuple(_fraction(v) for v in row["response"]["coordinate_increments"])==coordinates]
    if len(matches)!=1:
        raise ValueError("The complete endpoint body coordinates require exactly one authenticated saved "+key+" operator")
    return matches[0]


def _nested_interval_matrix(raw,n,m):
    if (type(raw) is not list or len(raw)!=n
            or any(type(row) is not list or len(row)!=m for row in raw)):
        raise ValueError("A saved endpoint operator retains every literal matrix row")
    return tuple(tuple(_interval(x) for x in row) for row in raw)


@dataclass(frozen=True,slots=True)
class NativeMixedForceInternalBeam:
    original_parent_id: str
    component_id: str
    trace_id: str
    side: str
    global_coordinate_index: int
    cold_bank_index: int
    cold_tip_coordinate_index: int
    stiffness_exact_n_m: F
    literal_reference_force_n: F
    literal_reference_energy_j: F
    complete_original_beam_json: str


@dataclass(frozen=True,slots=True)
class NativeCommonMixedForceSystem:
    coordinate_order: tuple
    original_coordinate_order: tuple
    coordinate_units: tuple
    gradient_units: tuple
    hessian_units: tuple
    bound_coordinate_increments: tuple
    original_bound_coordinate_increments: tuple
    authentic_saved_cold_coordinate_cases: tuple
    authentic_saved_hot_coordinate_cases: tuple
    saved_cold_operator_bindings: tuple
    saved_hot_operator_bindings: tuple
    source_packet_seals: tuple
    source_authority_raw_sha256: tuple
    cold_continuum_system_binding_sha256: str
    hot_mean_system_binding_sha256: str
    cold_material_carrier_binding_sha256: str
    source_work_system_binding_sha256: str
    source_atoms_complete_sha256: str
    complete_original_material_state_json: str
    complete_original_source_clock_work_history_json: str
    complete_source_normal_material_json: str
    complete_saved_joint_reference_field_json: str
    complete_current_load_json: str
    internal_beams: tuple
    owner_torque_bounds_nm: tuple
    owner_drag_bounds_nm_s: tuple
    owner_drag_time_scale_s: F
    owner_drag_time_scale_is_new_declared_calibration: bool
    owner_load_model: str
    source_contact_model: str
    internal_beam_model: str
    precision: int
    provenance: str
    scope: str
    hot_constitutive_law_defined: bool
    finite_mean_chart_defined: bool
    finite_mean_equilibrium_certified: bool
    physical_source_force_prestress_work_handover_certified: bool
    historical_archived_three_dimensional_prestress_recovered: bool
    prepared_reference_full_system_work_certified: bool
    born_hot_receiving_contact_law_certified: bool
    whole_physical_mixed_operator_certified: bool
    new_mixed_force_equilibrium_certified: bool
    physical_material_admission_authorized: bool
    whole_clock_all_pair_admission_authorized: bool
    full_system_temporal_path_work_certified: bool
    accepted_original_native_intervals: int
    accepted_new_native_intervals: int
    source_geometry_elapsed_time_s: F
    contact_geometry_elapsed_time_s: F
    memory_elapsed_time_s: F
    material_geometry_elapsed_time_s: F
    clocks_advanced: bool
    original_point_1_completed: bool
    original_point_1_remaining_work_blocks: int
    remaining_physical_conditions: tuple
    binding_sha256: str


def bind_native_common_mixed_force(hot903,cold899,work891,transport879,*,cold_carrier,
        provenance,precision=96,owner_drag_time_scale_s=F(1000000)):
    _precision(precision)
    if type(provenance) is not str or not provenance.strip():
        raise ValueError("The NEW original-source-atom endpoint assembly requires provenance")
    if type(owner_drag_time_scale_s) is not F or owner_drag_time_scale_s<=0:
        raise ValueError("The NEW owner drag calibration requires a positive literal Fraction second scale")
    packets=(hot903,cold899,work891,transport879)
    for packet,(_,_,seal) in zip(packets,AUTHORITY_PINS):
        _packet(packet,seal)
    _registered_cold_carrier(cold_carrier)
    hot=hot903["hot_mean_system"];cold=cold899["cold_continuum_energy_system"]
    work=work891["carrier_source_work_system"];seed=json.loads(work["seed_json"])
    if (len(hot903["evaluation_queries"])!=4 or len(cold899["evaluation_queries"])!=2
            or hot903["complete_hot_material_constitutive_operator_certified"] is not True
            or hot903["complete_declared_hot_mean_restart_verified"] is not True
            or cold899["complete_cold_continuum_operator_certified"] is not True
            or cold899["complete_declared_cold_continuum_restart_verified"] is not True):
        raise ValueError("The full two cold and four HOT authenticated component operators are required")
    if (hot["cold_continuum_component_binding_sha256"]!=cold["binding_sha256"]
            or hot["cold_material_carrier_binding_sha256"]!=cold["cold_material_carrier_binding_sha256"]
            or cold_carrier.binding_sha256!=cold["cold_material_carrier_binding_sha256"]):
        raise ValueError("The exact HOT/cold material reference association changed")
    order=tuple(tuple(x) for x in work["coordinate_order"])
    original=tuple(_fraction(x) for x in work["bound_coordinate_increments"])
    hotorder=tuple(tuple(x) for x in hot["coordinate_order"])
    coldorder=tuple(tuple(x) for x in cold["coordinate_order"])
    if (len(order)!=100 or len(original)!=100 or len(hotorder)!=30 or len(coldorder)!=56
            or tuple(tuple(x) for x in hot["complete_original_100_coordinate_order"])!=order
            or tuple(_fraction(x) for x in hot["complete_original_100_coordinate_values"])!=original
            or order[:6]!=tuple(("owner",owner,axis) for owner in (0,1) for axis in ("x","y","z"))):
        raise ValueError("The authentic full COMMON100 catalogue and exact HOT replacement are required")
    if tuple(order[6+i][1] for i in range(30))!=tuple(row[0] for row in hotorder):
        raise ValueError("HOT finite means must replace the same original thirty body coordinates")
    if order[36:92]!=coldorder:
        raise ValueError("Cold56 must be injected by its complete authentic catalogue, not a bank-position shortcut")
    neworder=order[:6]+tuple(("body",cid,name) for cid,name in hotorder)+order[36:]
    coldcoords=tuple(_fraction(x) for x in cold["bound_coordinate_increments"])
    hotcoords=tuple(_fraction(x) for x in hot["bound_coordinate_increments"])
    if hotcoords!=(F(0),)*30 or len(coldcoords)!=56:
        raise ValueError("The authentic HOT source is zero and the complete registered cold source must remain explicit")
    bound=(F(0),)*6+hotcoords+coldcoords+(F(0),)*8
    coldcases=tuple(tuple(_fraction(x) for x in row["response"]["coordinate_increments"])
                    for row in cold899["evaluation_queries"])
    hotcases=tuple(tuple(_fraction(x) for x in row["response"]["coordinate_increments"])
                   for row in hot903["evaluation_queries"])
    if len(set(coldcases))!=2 or len(set(hotcases))!=4:
        raise ValueError("Distinct complete saved component queries cannot be collapsed or substituted")
    beammap={b["original_parent_id"]:b for b in seed["beams"]}
    expected={"octant:0":99,"octant:1":93,"octant:2":95,"octant:3":97,
              "octant:4":98,"octant:5":92,"octant:6":94,"octant:7":96}
    if set(beammap)!=set(expected) or len(seed["beams"])!=8:
        raise ValueError("All eight authentic source internal beam identities are required")
    internals=[]
    for k,bank in enumerate(cold["banks"]):
        beam=beammap[bank["original_parent_id"]]
        if (_json(beam)!=bank["complete_source_beam_json"] or beam["global_coordinate_index"]!=expected[bank["original_parent_id"]]
                or _fraction(beam["stiffness_exact_n_m"])!=_fraction(bank["source_old_tip_stiffness_n_m"])
                or _fraction(beam["literal_reduced_beam_binary_force_evaluation_n"])!=_fraction(bank["source_reference_force_n"])
                or _fraction(beam["literal_reduced_beam_binary_energy_evaluation_j"])!=_fraction(bank["source_reference_energy_j"])):
            raise ValueError("The authentic physical-parent beam/internal injection or literal calibration changed")
        gi=beam["global_coordinate_index"]
        if order[gi]!=("source_internal",beam["trace_id"],beam["side"]):
            raise ValueError("A residual coordinate lost its original source trace and bank identity")
        internals.append(NativeMixedForceInternalBeam(bank["original_parent_id"],bank["component_id"],beam["trace_id"],
            beam["side"],gi,k,bank["tip_coordinate_index"],_fraction(bank["source_old_tip_stiffness_n_m"]),
            _fraction(bank["source_reference_force_n"]),_fraction(bank["source_reference_energy_j"]),_json(beam)))
    if len(internals)!=8 or {x.global_coordinate_index for x in internals}!=set(range(92,100)):
        raise ValueError("The declared eight residuals must complete COMMON100 exactly")
    load=json.loads(hot["complete_current_load_json"])
    if load.get("geometric_measure_midpoint_is_material_debit") is not False:
        raise ValueError("The frozen owner loading must retain its actual full-source measure authority")
    torque=tuple(_interval(x) for x in load["torque_bounds_nm"])
    drag=_nested_interval_matrix(load["drag_bounds_nm_s"],6,6)
    if len(torque)!=6:
        raise ValueError("The actual source epoch load retains six complete owner torque components")
    if hot["complete_original_material_state_json"]!=transport879["material_transfer"]["complete_original_material_state_json"]:
        raise ValueError("The complete HOT/cold source material epoch changed")
    result=NativeCommonMixedForceSystem(neworder,order,("rad",)*6+("m",)*94,("N*m",)*6+("N",)*94,
        tuple(tuple(_hessian_units(i,j) for j in range(100)) for i in range(100)),bound,original,coldcases,hotcases,
        tuple(row["response"]["binding_sha256"] for row in cold899["evaluation_queries"]),
        tuple(row["response"]["binding_sha256"] for row in hot903["evaluation_queries"]),
        tuple(pin[2] for pin in AUTHORITY_PINS),tuple(pin[1] for pin in AUTHORITY_PINS),
        cold["binding_sha256"],hot["binding_sha256"],cold_carrier.binding_sha256,work["binding_sha256"],
        _raw_digest(seed["source_atoms"]),hot["complete_original_material_state_json"],
        hot["complete_original_source_clock_work_history_json"],hot["complete_source_normal_material_json"],
        hot["complete_saved_joint_reference_field_json"],hot["complete_current_load_json"],tuple(internals),torque,drag,
        owner_drag_time_scale_s,True,OWNER_LOAD_MODEL,CONTACT_MODEL,INTERNAL_MODEL,precision,provenance,SCOPE,
        True,True,False,False,False,False,False,False,False,False,False,False,1,0,F(0),F(2000000),
        F(1000000),F(1000000),False,False,1,REMAINING_CONDITIONS,"")
    result=_remember(result,_SYSTEMS)
    _CONTEXTS[result.binding_sha256]=dict(packets=packets,cold_carrier=cold_carrier,source_seed=seed,
        cold_cases=cold899["evaluation_queries"],hot_cases=hot903["evaluation_queries"],
        cold_banks=cold["banks"])
    return result


def validate_native_common_mixed_force(system):
    _validate(system,NativeCommonMixedForceSystem,_SYSTEMS)
    context=_CONTEXTS.get(system.binding_sha256)
    if context is None:
        raise ValueError("The declared endpoint lacks its complete immutable source context")
    for packet,(_,_,seal) in zip(context["packets"],AUTHORITY_PINS):
        _packet(packet,seal)
    _registered_cold_carrier(context["cold_carrier"])
    return system


def native_common_mixed_force_to_dict(system):
    return dict(format=FORMAT,**_payload(validate_native_common_mixed_force(system)))


def native_common_mixed_force_from_dict(raw,hot903,cold899,work891,transport879,*,cold_carrier):
    _saved_json_kinds(raw)
    if type(raw) is not dict or raw.get("format")!=FORMAT:
        raise ValueError("A complete declared COMMON100 endpoint DTO is required")
    value=bind_native_common_mixed_force(hot903,cold899,work891,transport879,cold_carrier=cold_carrier,
        provenance=raw.get("provenance"),precision=raw.get("precision"),
        owner_drag_time_scale_s=_fraction(raw["owner_drag_time_scale_s"]))
    if _encoded(native_common_mixed_force_to_dict(value))!=_encoded(raw):
        raise ValueError("A resealed mixed endpoint cannot change its complete source, units or frozen query context")
    return value


def _coordinates(system,value):
    value=system.bound_coordinate_increments if value is None else value
    if type(value) is not tuple or len(value)!=100 or any(type(x) is not F for x in value):
        raise ValueError("The endpoint requires all 100 literal Fraction coordinates")
    if any(value[:6]):
        raise ValueError("The declared owner chart evaluates exactly zero owner increments with full first/second jets")
    if value[6:36] not in system.authentic_saved_hot_coordinate_cases:
        raise ValueError("Unsupported HOT coordinates require a newly certified component operator")
    if value[36:92] not in system.authentic_saved_cold_coordinate_cases:
        raise ValueError("Unsupported cold coordinates require a newly certified component operator")
    for beam in system.internal_beams:
        radius=_fraction(_CONTEXTS[system.binding_sha256]["cold_banks"][beam.cold_bank_index]["radius_m"])
        if radius+value[beam.global_coordinate_index]<=0:
            raise ValueError("An internal material arm must retain a positive original radius plus residual")
    return value


def _internal_beam_correction_twojet(p,dp,d2p,source_force,stiffness,residual,*,precision=96):
    """Shift one already calibrated beam by r, retaining all prestress terms."""
    _precision(precision)
    if (type(dp) is not tuple or type(d2p) is not tuple or len(dp)!=len(d2p)
            or any(type(row) is not tuple or len(row)!=len(dp) for row in d2p)
            or type(source_force) is not F or type(stiffness) is not F or stiffness<=0
            or type(residual) is not F or type(p) not in (F,I)
            or any(type(x) not in (F,I) for x in dp)
            or any(type(x) not in (F,I) for row in d2p for x in row)):
        raise ValueError("A residual beam correction needs complete literal calibrated projection twojets")
    force=I(source_force)+stiffness*p
    energy=_round(force*residual+stiffness*residual*residual/2,precision)
    return dict(energy_bounds_j=energy,
        cold_gradient=tuple(_round(stiffness*residual*x,precision) for x in dp),
        internal_gradient=_round(force+stiffness*residual,precision),
        cold_hessian=tuple(tuple(_round(stiffness*residual*x,precision) for x in row) for row in d2p),
        cold_internal_hessian=tuple(_round(stiffness*x,precision) for x in dp),
        internal_hessian=I(stiffness))


def _owner_load_twojet(theta,drag,torque,time_scale,*,precision=96):
    """NEW frozen owner increment potential; no physical loading path inferred."""
    _precision(precision)
    if (type(theta) is not tuple or len(theta)!=6 or any(type(x) is not F for x in theta)
            or type(drag) is not tuple or len(drag)!=6 or any(type(row) is not tuple or len(row)!=6 for row in drag)
            or type(torque) is not tuple or len(torque)!=6 or type(time_scale) is not F or time_scale<=0
            or any(type(x) not in (F,I) for row in drag for x in row)
            or any(type(x) not in (F,I) for x in torque)):
        raise ValueError("The frozen owner potential retains six torques, complete drag and its explicit scale")
    symmetric=tuple(tuple((drag[i][j]+drag[j][i])/2 for j in range(6)) for i in range(6))
    H=tuple(tuple(_round(x/time_scale,precision) for x in row) for row in symmetric)
    g=tuple(_round(sum((H[i][j]*theta[j] for j in range(6)),I(0))-torque[i],precision) for i in range(6))
    energy=_round(sum((theta[i]*H[i][j]*theta[j]/2 for i in range(6) for j in range(6)),I(0))
                  -sum((torque[i]*theta[i] for i in range(6)),I(0)),precision)
    return dict(energy_bounds_j=energy,gradient_bounds=g,hessian_bounds=H)


def _projection_twojet(context,cold_response,precision):
    key=(cold_response["binding_sha256"],precision)
    if key in _PROJECTION_JETS:
        return _PROJECTION_JETS[key]
    A=_nested_interval_matrix(cold_response["gram_inverse_certificate"]["inverse_gram_bounds_m_n"],8,8)
    raw=cold_response["integrals"]
    gradient=_nested_interval_matrix(raw["projection_remainder_gradient_n_m"],8,56)
    hess=tuple(_nested_interval_matrix(block,56,56) for block in raw["projection_hessian_n_m2"])
    if len(hess)!=8:
        raise ValueError("Every saved cold global projection Hessian must remain complete")
    dp=tuple(tuple(_round(int(i==context["cold_banks"][k]["tip_coordinate_index"])
        +sum((A[k][l]*gradient[l][i] for l in range(8)),I(0)),precision) for i in range(56)) for k in range(8))
    d2p=tuple(tuple(tuple(_round(sum((A[k][l]*hess[l][i][j] for l in range(8)),I(0)),precision)
        for j in range(56)) for i in range(56)) for k in range(8))
    p=tuple(_interval(x) for x in cold_response["tip_projection_increment_m"])
    result=(p,dp,d2p);_PROJECTION_JETS[key]=result
    return result


@dataclass(frozen=True,slots=True)
class NativeCommonMixedForceResponse:
    system_binding_sha256: str
    coordinate_order: tuple
    coordinate_increments: tuple
    coordinate_units: tuple
    gradient_units: tuple
    hessian_units: tuple
    energy_bounds_j: I
    gradient_bounds: tuple
    hessian_bounds: tuple
    energy_partitions_json: str
    internal_beam_corrections_json: tuple
    contact_atoms_json: tuple
    source_point_receipts_json: tuple
    contact_authority_binding_sha256: str
    cold_operator_response_binding_sha256: str
    hot_operator_response_binding_sha256: str
    precision: int
    whole_cold_parent_count: int
    whole_hot_body_count: int
    source_atom_count: int
    source_material_point_count: int
    source_internal_beam_count: int
    complete_declared_endpoint_force_assembly_certified: bool
    cold_beams_included_once: bool
    internal_residuals_shift_existing_beams_once: bool
    old891_complete_beam_bulk_operator_added_again: bool
    born_hot_receiving_contact_law_certified: bool
    whole_physical_mixed_operator_certified: bool
    new_mixed_force_equilibrium_certified: bool
    finite_mean_equilibrium_certified: bool
    physical_source_force_prestress_work_handover_certified: bool
    historical_archived_three_dimensional_prestress_recovered: bool
    prepared_reference_full_system_work_certified: bool
    physical_material_admission_authorized: bool
    whole_clock_all_pair_admission_authorized: bool
    full_system_temporal_path_work_certified: bool
    accepted_original_native_intervals: int
    accepted_new_native_intervals: int
    source_geometry_elapsed_time_s: F
    contact_geometry_elapsed_time_s: F
    memory_elapsed_time_s: F
    material_geometry_elapsed_time_s: F
    clocks_advanced: bool
    original_point_1_completed: bool
    original_point_1_remaining_work_blocks: int
    remaining_physical_conditions: tuple
    binding_sha256: str
    scope: str = SCOPE
    hot_constitutive_law_defined: bool = True
    finite_mean_chart_defined: bool = True
    endpoint_owner_chart_is_exact_zero: bool = True
    source_body_operator_queries_reexecuted: bool = False
    original_material_atom_measure_used_once: bool = True
    current_unit_nanson_normal_follower_component_defined: bool = True
    owner_angles_radians_are_dimensionless_in_energy_derivatives: bool = True


def _contact_result(system,cold_coordinates,internal_coordinates,precision):
    context=_CONTEXTS[system.binding_sha256]
    key=(system.binding_sha256,cold_coordinates,internal_coordinates,precision)
    if key not in _CONTACT_CACHE:
        if "_mf_contacts" not in globals():
            raise ValueError("The complete NEW source-contact twojet has not yet been integrated")
        _CONTACT_CACHE[key]=_mf_contacts(context["cold_carrier"],context["packets"][2],
            cold_coordinates,internal_coordinates,precision=precision)
    return _CONTACT_CACHE[key]


def native_common_mixed_force_contact_authority_to_dict(system,cold_coordinate_increments=None,*,precision=96):
    """Shared full original32 material attachment geometry, exported once."""
    validate_native_common_mixed_force(system);_precision(precision)
    cold=system.bound_coordinate_increments[36:92] if cold_coordinate_increments is None else cold_coordinate_increments
    if type(cold) is not tuple or any(type(x) is not F for x in cold) or cold not in system.authentic_saved_cold_coordinate_cases:
        raise ValueError("Contact geometry requires an authenticated complete saved cold56 query")
    key=(system.binding_sha256,cold,precision)
    if key not in _CONTACT_AUTHORITIES:
        context=_CONTEXTS[system.binding_sha256]
        if "_mf_contact_authority" in globals():
            authority=_mf_contact_authority(context["cold_carrier"],context["packets"][2],cold,precision=precision)
            points=authority["point_receipts"]
        else:
            contact=_contact_result(system,cold,(F(0),)*8,precision)
            points=contact["point_receipts"]
        if len(points)!=32:
            raise ValueError("All thirty-two original source material point receipts must be preserved")
        raw=dict(authority) if "_mf_contact_authority" in globals() else {}
        raw.update(dict(format="native-common-mixed-original-source-contact-authority-1",
            contact_geometry_authority_format=authority.get("format") if "_mf_contact_authority" in globals() else None,
            system_binding_sha256=system.binding_sha256,cold_coordinate_increments=_payload(cold),
            source_packet_seals=list(system.source_packet_seals),
            source_original_atom_area_measure_used_once=True,current_director_is_unit_nanson_normal=True,
            point_receipts=points,precision=precision))
        raw["binding_sha256"]=_raw_digest(raw)
        _CONTACT_AUTHORITIES[key]=raw
    # Copy export containers so a caller cannot mutate the shared context.
    return json.loads(_json(_CONTACT_AUTHORITIES[key]))


def evaluate_native_common_mixed_force(system,coordinate_increments=None,*,precision=96):
    validate_native_common_mixed_force(system);_precision(precision)
    coords=_coordinates(system,coordinate_increments)
    key=(system.binding_sha256,coords,precision)
    if key in _EVALUATIONS:
        return _validate(_EVALUATIONS[key],NativeCommonMixedForceResponse,_RESPONSES)
    context=_CONTEXTS[system.binding_sha256]
    hotrow=_saved_case(context["packets"][0],coords[6:36],"HOT30")
    coldrow=_saved_case(context["packets"][1],coords[36:92],"cold56")
    hot=hotrow["response"];cold=coldrow["response"]
    hotenergy=_interval(hot["energy_bounds_j"]);coldenergy=_interval(cold["energy_j"])
    g=[I(0)]*100;H=[[I(0) for _ in range(100)] for _ in range(100)]
    for start,rawg,rawH,count in ((6,hot["gradient_bounds_n"],hot["hessian_bounds_n_m"],30),
            (36,cold["gradient_n"],cold["hessian_n_m"],56)):
        if len(rawg)!=count:
            raise ValueError("Every inherited component force row must be present")
        matrix=_nested_interval_matrix(rawH,count,count)
        for i in range(count):
            g[start+i]=_interval(rawg[i])
            for j in range(count):H[start+i][start+j]=matrix[i][j]
    p,dp,d2p=_projection_twojet(context,cold,precision)
    internalenergy=I(0);beamreceipts=[]
    for beam in system.internal_beams:
        k=beam.cold_bank_index;index=beam.global_coordinate_index
        correction=_internal_beam_correction_twojet(p[k],dp[k],d2p[k],beam.literal_reference_force_n,
                                                   beam.stiffness_exact_n_m,coords[index],precision=precision)
        internalenergy+=correction["energy_bounds_j"]
        g[index]+=correction["internal_gradient"]
        H[index][index]+=correction["internal_hessian"]
        for i in range(56):
            gi=36+i;g[gi]+=correction["cold_gradient"][i]
            H[gi][index]+=correction["cold_internal_hessian"][i]
            H[index][gi]+=correction["cold_internal_hessian"][i]
            for j in range(56):H[gi][36+j]+=correction["cold_hessian"][i][j]
        beamreceipts.append(_json(dict(original_parent_id=beam.original_parent_id,trace_id=beam.trace_id,
            source_bank=beam.side,global_internal_coordinate_index=index,cold_bank_index=k,
            source_stiffness_exact_n_m=_payload(beam.stiffness_exact_n_m),
            source_literal_reference_force_n=_payload(beam.literal_reference_force_n),
            inherited_cold_projection_increment_bounds_m=_payload(p[k]),
            internal_residual_coordinate_m=_payload(coords[index]),
            energy_increment_bounds_j=_payload(correction["energy_bounds_j"]),
            projection_first_sha256=_digest(dp[k]),projection_second_sha256=_digest(d2p[k]),
            old_reference_energy_added_again=False,nonlinear_projection_curvature_included=True)))
    load=_owner_load_twojet(coords[:6],system.owner_drag_bounds_nm_s,system.owner_torque_bounds_nm,
                           system.owner_drag_time_scale_s,precision=precision)
    for i in range(6):
        g[i]+=load["gradient_bounds"][i]
        for j in range(6):H[i][j]+=load["hessian_bounds"][i][j]
    contact=_contact_result(system,coords[36:92],coords[92:100],precision)
    if (len(contact["gradient_bounds"])!=100 or len(contact["hessian_bounds"])!=100
            or any(len(row)!=100 for row in contact["hessian_bounds"])
            or len(contact["atoms"])!=16 or len(contact["point_receipts"])!=32):
        raise ValueError("The complete original source sixteen-atom/32-point force twojet is required")
    if (contact["original_material_nanson_follower_contact_component_defined"] is not True
            or contact["prepared_zero_source_traction_calibration"] is not True
            or contact["clocks_advanced"] is not False or contact["accepted_new_native_intervals"]!=0
            or contact["prepared_reference_full_system_work_certified"] is not False
            or contact["physical_source_force_prestress_work_handover_certified"] is not False
            or contact["historical_normal_fracture_memory_recovered"] is not False
            or contact["current_continuous_face_contact_law_defined"] is not False
            or contact["born_hot_receiving_law_inferred"] is not False):
        raise ValueError("The declared atom endpoint cannot authorize unrecovered physical laws, work or history")
    for i in range(100):
        g[i]+=contact["gradient_bounds"][i]
        for j in range(100):H[i][j]+=contact["hessian_bounds"][i][j]
    contactauthority=native_common_mixed_force_contact_authority_to_dict(system,coords[36:92],precision=precision)
    pointreferences=tuple(_json(dict(point_id=x["point_id"],side=x["side"],
        complete_point_receipt_sha256=_raw_digest(x),
        current_point_binding_sha256=x["current"]["binding_sha256"],
        prepared_zero_point_binding_sha256=x["prepared_zero"]["binding_sha256"],
        contact_authority_binding_sha256=contactauthority["binding_sha256"])) for x in contact["point_receipts"])
    energy=_round(coldenergy+hotenergy+internalenergy+load["energy_bounds_j"]+contact["energy_bounds_j"],precision)
    partitions=dict(cold_whole_material_and_eight_beams_bounds_j=_payload(coldenergy),
        hot_whole_material_and_finite_radial_base_bounds_j=_payload(hotenergy),
        eight_internal_beam_driver_correction_bounds_j=_payload(_round(internalenergy,precision)),
        frozen_owner_load_drag_potential_bounds_j=_payload(load["energy_bounds_j"]),
        original_source_atom_loading_potential_bounds_j=_payload(contact["energy_bounds_j"]),
        original_source_atom_recoverable_energy_bounds_j=_payload(contact["recoverable_bounds_j"]),
        original_source_atom_fracture_dissipation_bounds_j=_payload(contact["fracture_bounds_j"]),
        total_bounds_j=_payload(energy),contact_loading_potential_includes_fracture_dissipation=True,
        old891_complete_beam_bulk_operator_added_again=False,old_scalar_hot_foundation_added_again=False,
        owner_load_is_new_frozen_increment_potential=True,physical_loading_work_recovered=False)
    # Hessian intervals are retained with an exact symmetric hull; equality of
    # mixed partials is a property of each declared twojet, not a midpoint rule.
    fullH=tuple(tuple(H[i][j] if 6<=i<36 and 6<=j<36 else
                         _round(I(min(H[i][j].lo,H[j][i].lo),max(H[i][j].hi,H[j][i].hi)),precision)
                         for j in range(100)) for i in range(100))
    value=NativeCommonMixedForceResponse(system.binding_sha256,system.coordinate_order,coords,
        system.coordinate_units,system.gradient_units,system.hessian_units,energy,
        tuple(x if 6<=i<36 else _round(x,precision) for i,x in enumerate(g)),fullH,_json(partitions),tuple(beamreceipts),
        tuple(_json(x) for x in contact["atoms"]),pointreferences,
        contactauthority["binding_sha256"],cold["binding_sha256"],hot["binding_sha256"],precision,
        8,5,16,32,8,True,True,True,False,False,False,False,False,False,False,False,False,False,False,
        1,0,F(0),F(2000000),F(1000000),F(1000000),False,False,1,REMAINING_CONDITIONS,"")
    value=_remember(value,_RESPONSES);_EVALUATIONS[key]=value
    return value


def native_common_mixed_force_response_to_dict(response):
    return dict(format=RESPONSE_FORMAT,**_payload(_validate(response,NativeCommonMixedForceResponse,_RESPONSES)))


def native_common_mixed_force_response_from_dict(raw,*,system,expected_coordinate_increments=None,precision=96):
    _saved_json_kinds(raw)
    if type(raw) is not dict or raw.get("format")!=RESPONSE_FORMAT:
        raise ValueError("A complete declared COMMON100 force response DTO is required")
    value=evaluate_native_common_mixed_force(system,expected_coordinate_increments,precision=precision)
    if _encoded(native_common_mixed_force_response_to_dict(value))!=_encoded(raw):
        raise ValueError("A resealed mixed endpoint response cannot change its full literal query or operator")
    return value


# NEW original-material follower contacts on the whole prepared 895 field.
# This fragment is pasted once into the additive mixed-force core.  It never
# imports or reexecutes a historical solve, control, integrator or work ledger.
# The declared scalar potential is:
#   Y_i = x_895(c) + A_895(c) A_old_i^-1 (p_old_i-X_old_i+r_i v_old_i)
#   n(c) = cof(A_895,over(c)) N_source / |cof(A_895,over(c)) N_source|
#   A_old = B_saved Q_owner C_roof (COMPLETE original-input chart)
#   N_source = A_old,over^T n_archival
#   delta = (R_over n).(R_over Y_over-R_sub Y_sub) - n_0.(Y_0over-Y_0sub)
#   E_a = literal_E_a - A0_a Phi(tau_old/K) + A0_a Phi(tau_old/K+delta).
# A0 is the ORIGINAL literal atom measure, once.  The cofactor norm is a
# normal transport proof, not a second current-area energy multiplier.
# This NEW prepared-zero source-traction calibration is not reference work,
# recovered historical prestress or a current continuous-face/HOT law.

_MF_CONTACT_POINT_CACHE = {}


def _mf_square(value):
    value = value if type(value) is I else I(value)
    return I(F(0) if value.lo <= 0 <= value.hi else min(value.lo**2, value.hi**2),
             max(value.lo**2, value.hi**2))


def _mf_pair(i, j):
    return (i, j) if i <= j else (j, i)


class _MFContactJet:
    """Outward-rounded sparse scalar twojet on named global coordinates."""
    __slots__ = ("value", "gradient", "hessian", "precision")

    def __init__(self, value, gradient=None, hessian=None, precision=96):
        if type(precision) is not int or not 96 <= precision <= 512:
            raise ValueError("A follower contact needs an explicit interval precision")
        self.precision = precision
        self.value = _round(value if type(value) is I else I(value), precision)
        self.gradient = {}
        self.hessian = {}
        for index, item in ({} if gradient is None else gradient).items():
            if type(index) is not int or not 0 <= index < 100:
                raise ValueError("A contact derivative must name an actual global coordinate")
            item = _round(item if type(item) is I else I(item), precision)
            if item != I(0):
                self.gradient[index] = item
        for pair, item in ({} if hessian is None else hessian).items():
            if (type(pair) is not tuple or len(pair) != 2
                    or any(type(index) is not int or not 0 <= index < 100 for index in pair)
                    or pair[0] > pair[1]):
                raise ValueError("Sparse contact Hessian keys must be ordered global pairs")
            item = _round(item if type(item) is I else I(item), precision)
            if item != I(0):
                self.hessian[pair] = item

    @classmethod
    def variable(cls, value, index, precision=96):
        return cls(value, {index: I(1)}, precision=precision)

    def _coerce(self, other):
        if type(other) is _MFContactJet:
            if other.precision != self.precision:
                raise ValueError("One contact twojet cannot mix enclosure precisions")
            return other
        return _MFContactJet(other, precision=self.precision)

    def __add__(self, other):
        other = self._coerce(other)
        keys = self.gradient.keys() | other.gradient.keys()
        pairs = self.hessian.keys() | other.hessian.keys()
        return _MFContactJet(self.value+other.value,
            {i: self.gradient.get(i, I(0))+other.gradient.get(i, I(0)) for i in keys},
            {p: self.hessian.get(p, I(0))+other.hessian.get(p, I(0)) for p in pairs},
            self.precision)

    __radd__ = __add__

    def __neg__(self):
        return _MFContactJet(-self.value, {i: -v for i, v in self.gradient.items()},
            {p: -v for p, v in self.hessian.items()}, self.precision)

    def __sub__(self, other):
        return self+(-self._coerce(other))

    def __rsub__(self, other):
        return self._coerce(other)+(-self)

    def __mul__(self, other):
        other = self._coerce(other)
        indices = sorted(self.gradient.keys() | other.gradient.keys())
        pairs = self.hessian.keys() | other.hessian.keys()
        pairs = set(pairs) | {(i, j) for a, i in enumerate(indices) for j in indices[a:]}
        return _MFContactJet(self.value*other.value,
            {i: self.gradient.get(i, I(0))*other.value+self.value*other.gradient.get(i, I(0))
                for i in indices},
            {p: self.hessian.get(p, I(0))*other.value+self.value*other.hessian.get(p, I(0))
                +self.gradient.get(p[0], I(0))*other.gradient.get(p[1], I(0))
                +self.gradient.get(p[1], I(0))*other.gradient.get(p[0], I(0)) for p in pairs},
            self.precision)

    __rmul__ = __mul__

    def compose(self, value, first, second):
        indices = sorted(self.gradient)
        pairs = set(self.hessian) | {(i, j) for a, i in enumerate(indices) for j in indices[a:]}
        return _MFContactJet(value, {i: first*v for i, v in self.gradient.items()},
            {p: first*self.hessian.get(p, I(0))
                +second*self.gradient.get(p[0], I(0))*self.gradient.get(p[1], I(0))
                for p in pairs}, self.precision)

    def square(self):
        return self.compose(_mf_square(self.value), 2*self.value, I(2))

    def sqrt(self):
        if self.value.lo <= 0:
            raise ValueError("A normalized material normal needs a strictly positive whole-box norm")
        root = interval_sqrt(self.value, self.precision)
        return self.compose(root, 1/(2*root), -1/(4*root*root*root))

    def inverse(self):
        if self.value.lo <= 0 <= self.value.hi:
            raise ValueError("A follower twojet reciprocal cannot cross zero")
        reciprocal = 1/self.value
        return self.compose(reciprocal, -reciprocal*reciprocal,
            2*reciprocal*reciprocal*reciprocal)

    def __truediv__(self, other):
        return self*self._coerce(other).inverse()

    def __rtruediv__(self, other):
        return self._coerce(other)*self.inverse()


def _mf_jet_dot(a, b):
    return sum((x*y for x, y in zip(a, b)), _MFContactJet(0, precision=a[0].precision))


def _mf_jet_matrix_vector(matrix, vector):
    return tuple(_mf_jet_dot(row, vector) for row in matrix)


def _mf_jet_export(jet):
    return dict(value_bounds=_payload(jet.value),
        first_derivative_bounds=[[i, _payload(v)] for i, v in sorted(jet.gradient.items())],
        second_derivative_bounds=[[i, j, _payload(v)] for (i, j), v in sorted(jet.hessian.items())])


def _mf_nanson_director(original_gradient, mode_original_gradients_by_global_index,
                        material_normal, *, precision=96):
    """Unit Nanson normal and complete material twojet, including cofactor curvature.

    This transports the ORIGINAL material covector, not a common Cartesian
    old normal used in the wrong input basis. Positive material determinant
    is checked by the authenticated 895 material-point API separately.
    """
    if (type(original_gradient) is not tuple or len(original_gradient) != 3
            or any(type(row) is not tuple or len(row) != 3 for row in original_gradient)
            or type(material_normal) is not tuple or len(material_normal) != 3
            or type(mode_original_gradients_by_global_index) is not dict):
        raise ValueError("A Nanson twojet needs its complete original Cartesian basis")
    A = tuple(tuple(_MFContactJet(original_gradient[i][j],
        {index: G[i][j] for index, G in mode_original_gradients_by_global_index.items()},
        precision=precision) for j in range(3)) for i in range(3))
    cofactor = []
    for i in range(3):
        row = []
        for j in range(3):
            rows, columns = [k for k in range(3) if k != i], [k for k in range(3) if k != j]
            minor = A[rows[0]][columns[0]]*A[rows[1]][columns[1]]-A[rows[0]][columns[1]]*A[rows[1]][columns[0]]
            row.append(minor if (i+j) % 2 == 0 else -minor)
        cofactor.append(tuple(row))
    N = tuple(_MFContactJet(v, precision=precision) for v in material_normal)
    covector = _mf_jet_matrix_vector(tuple(cofactor), N)
    norm2 = sum((v.square() for v in covector), _MFContactJet(0, precision=precision))
    norm = norm2.sqrt()
    normal = tuple(v/norm for v in covector)
    unit_residual = _round(sum((_mf_square(v.value) for v in normal), I(0))-1, precision)
    if not unit_residual.lo <= 0 <= unit_residual.hi:
        raise ValueError("The complete Nanson unit-direction bounds fail normalization")
    return normal, dict(original_material_normal=_payload(material_normal),
        cofactor_original_gradient_bounds=_payload(tuple(tuple(v.value for v in row) for row in cofactor)),
        transported_cofactor_normal_twojet=[_mf_jet_export(v) for v in covector],
        transported_cofactor_normal_norm_squared_twojet=_mf_jet_export(norm2),
        transported_cofactor_normal_norm_bounds=_payload(norm.value),
        unit_normal_twojet=[_mf_jet_export(v) for v in normal],
        unit_normal_norm_squared_residual_bounds=_payload(unit_residual),
        whole_box_normal_norm_strictly_positive=True,
        normalized_direction_only_no_second_current_area_energy_multiplier=True,
        full_polynomial_cofactor_second_derivatives_included=True)


def _mf_cross(axis, vector):
    return (axis[1]*vector[2]-axis[2]*vector[1],
            axis[2]*vector[0]-axis[0]*vector[2],
            axis[0]*vector[1]-axis[1]*vector[0])


def _mf_axis(index):
    return tuple(I(int(k == index)) for k in range(3))


def _mf_interval_dot(a, b):
    return sum((x*y for x, y in zip(a, b)), I(0))


def _mf_symmetric_spin(a, b, vector):
    ab, ba = _mf_cross(a, _mf_cross(b, vector)), _mf_cross(b, _mf_cross(a, vector))
    return tuple((x+y)/2 for x, y in zip(ab, ba))


def _mf_contact_scalar_twojet(subY, overY, nmaterial, sub_owner_indices,
                             receiver_owner_indices, baseline_scalar, *, precision=96):
    """Full follower scalar in the fresh owner exponential chart at zero.

    At every material/internal value it is n(c).(Yover-Rover^T Rsub Ysub).
    The receiver-side owner rotation cancels before interval arithmetic.
    This retains Nanson material derivatives and the source prestress
    curvature. It does not pretend these fresh-chart formulas hold at a
    nonzero value of the same exponential coordinates.
    """
    if (len(subY) != 3 or len(overY) != 3 or len(nmaterial) != 3
            or len(sub_owner_indices) != 3 or len(receiver_owner_indices) != 3
            or any(type(x) is not _MFContactJet for x in subY+overY+nmaterial)):
        raise ValueError("A complete material follower scalar needs two actual sides and a normal twojet")
    base = _mf_jet_dot(nmaterial, tuple(b-a for a, b in zip(subY, overY)))-baseline_scalar
    if sub_owner_indices == receiver_owner_indices:
        return base
    if set(sub_owner_indices) & set(receiver_owner_indices):
        raise ValueError("Distinct source owners cannot alias a rotation coordinate")
    gradient, hessian = dict(base.gradient), dict(base.hessian)
    n = tuple(v.value for v in nmaterial)
    Y = tuple(v.value for v in subY)
    material_indices = sorted(set().union(*(v.gradient.keys() for v in subY+overY+nmaterial)))
    for axis_index, (sub_index, over_index) in enumerate(zip(sub_owner_indices, receiver_owner_indices)):
        axis = _mf_axis(axis_index)
        spinY = _mf_cross(axis, Y)
        derivative = _round(-_mf_interval_dot(n, spinY), precision)
        gradient[sub_index], gradient[over_index] = derivative, -derivative
        for index in material_indices:
            dn = tuple(v.gradient.get(index, I(0)) for v in nmaterial)
            dY = tuple(v.gradient.get(index, I(0)) for v in subY)
            value = _round(-_mf_interval_dot(dn, spinY)-_mf_interval_dot(n, _mf_cross(axis, dY)), precision)
            hessian[_mf_pair(sub_index, index)] = value
            hessian[_mf_pair(over_index, index)] = -value
    for i, sub_i in enumerate(sub_owner_indices):
        for j in range(i, 3):
            same = _round(-_mf_interval_dot(n, _mf_symmetric_spin(_mf_axis(i), _mf_axis(j), Y)), precision)
            hessian[_mf_pair(sub_i, sub_owner_indices[j])] = same
            hessian[_mf_pair(receiver_owner_indices[i], receiver_owner_indices[j])] = same
        for j, over_j in enumerate(receiver_owner_indices):
            mixed = _round(_mf_interval_dot(n, _mf_cross(_mf_axis(j), _mf_cross(_mf_axis(i), Y))), precision)
            hessian[_mf_pair(sub_i, over_j)] = mixed
    return _MFContactJet(base.value, gradient, hessian, precision)


def _mf_cohesive_series(free_gap, kn, compliance, tensile_strength, fracture_energy, *, precision=96):
    """NEW virgin monotone Mode-I source-column/contact series potential.

    Phi is a quadratic intact law, bilinear softening loading potential,
    then the literal fracture-energy plateau. This is loading potential =
    recoverable energy + fracture work. It is not a recovered old damage
    state or a potential for arbitrary unloading after proposed failure.
    """
    if (type(free_gap) is not I or any(type(v) is not F for v in
            (kn, compliance, tensile_strength, fracture_energy))
            or min(kn, tensile_strength, fracture_energy) <= 0 or compliance < 0):
        raise ValueError("The declared cohesive series law needs positive exact source coefficients")
    K = 1/(1/kn+compliance)
    onset_contact, failure = tensile_strength/kn, 2*fracture_energy/tensile_strength
    onset = onset_contact+compliance*tensile_strength
    if failure <= onset or failure <= onset_contact:
        raise ValueError("This positive material source has no monotone softening branch")
    if free_gap.hi < onset:
        branch = "intact_or_compressed"
        traction, tangent = K*free_gap, I(K)
        contact_gap = free_gap/(1+compliance*kn)
        damage, fracture = I(0), I(0)
        potential = K*_mf_square(free_gap)/2
        recoverable = kn*_mf_square(contact_gap)/2+compliance*_mf_square(traction)/2
    elif free_gap.lo > failure:
        branch = "fully_failed_open"
        contact_gap, traction, tangent = free_gap, I(0), I(0)
        damage, fracture = I(1), I(fracture_energy)
        potential, recoverable = I(fracture_energy), I(0)
    elif onset < free_gap.lo and free_gap.hi < failure:
        branch = "monotone_softening"
        contact_gap = I(onset_contact)+(free_gap-onset)*(failure-onset_contact)/(failure-onset)
        traction, tangent = tensile_strength*(failure-free_gap)/(failure-onset), I(-tensile_strength/(failure-onset))
        damage = failure/(failure-onset_contact)*(1-onset_contact/contact_gap)
        fracture = fracture_energy*(contact_gap-onset_contact)/(failure-onset_contact)
        increment = free_gap-onset
        potential = I(tensile_strength*onset/2)+tensile_strength*increment-tensile_strength*_mf_square(increment)/(2*(failure-onset))
        recoverable = kn*(1-damage)*_mf_square(contact_gap)/2+compliance*_mf_square(traction)/2
    else:
        raise ValueError("Split a cohesive threshold enclosure: it has no unique classical Hessian")
    values = dict(free_gap_bounds_m=free_gap, contact_gap_bounds_m=contact_gap,
        traction_bounds_pa=traction, tangent_bounds_pa_m=tangent,
        loading_potential_density_bounds_j_m2=potential,
        recoverable_energy_density_bounds_j_m2=recoverable,
        fracture_work_density_bounds_j_m2=fracture, proposed_damage_bounds=damage,
        proposed_max_opening_bounds_m=I(max(F(0), contact_gap.lo), max(F(0), contact_gap.hi)),
        energy_partition_residual_bounds_j_m2=potential-recoverable-fracture)
    values = {k: _round(v, precision) for k, v in values.items()}
    residual = values["energy_partition_residual_bounds_j_m2"]
    if not residual.lo <= 0 <= residual.hi:
        raise ValueError("The declared loading potential misses recoverable or fracture work")
    return dict(branch=branch, series_stiffness_exact_pa_m=K,
        onset_free_gap_m=onset, failure_free_gap_m=failure, **values)


def _mf_source_fraction(value):
    if type(value) is float:
        return F.from_float(value)
    if type(value) is int:
        return F(value)
    raise ValueError("A source law keeps its literal binary64/integer scalar kind")


def _mf_source_point_receipts(cold_carrier, seed, cold_coordinates, precision):
    key = (cold_carrier.binding_sha256, cold_coordinates, precision,
           hashlib.sha256(json.dumps(seed["attachments"], sort_keys=True, separators=(",", ":"),
               ensure_ascii=False, allow_nan=False).encode()).hexdigest())
    if key in _MF_CONTACT_POINT_CACHE:
        return _MF_CONTACT_POINT_CACHE[key]
    pointmap, baseline = {}, {}
    fields = seed["attachments"]
    if len(fields) != 32 or len({(v["field"]["point_id"], v["field"]["side"]) for v in fields}) != 32:
        raise ValueError("Every one of the thirty-two original source material sides must occur once")
    counts = {"LIVE_ORIGINAL_MATERIAL": 0, "ARCHIVED_ORIGINAL_MATERIAL": 0}
    parent_allocations = {v["fragment_id"]: v for v in json.loads(
        seed["local"]["complete_original_source_material_state_json"])["parent_allocations"]}
    for item in fields:
        field = item["field"]
        membership = field["original_label_membership"]
        if membership not in counts:
            raise ValueError("A source material attachment cannot become HOT or an unknown allocation")
        counts[membership] += 1
        archived = membership == "ARCHIVED_ORIGINAL_MATERIAL"
        parent_cohort = parent_allocations[field["original_parent_id"]]["cohort_id"]
        # Public887/895 chronology reports the ORIGINAL PARENT cohort for
        # live and archived points. The saved source field instead annotates
        # a point-specific archived attachment, None for live labels.
        expected_point_cohort = parent_cohort if archived else None
        if field["archive_cohort_id"] != expected_point_cohort or (archived and parent_cohort is None):
            raise ValueError("The source point-specific archive annotation changed its parent ancestry")
        ray = tuple(_fraction(v) for v in item["original_ray"])
        depth = _interval(field["material_depth_bounds_m"])
        if depth != I(0):
            raise ValueError("The preserved source attachment is an exact original depth-zero label")
        args = (cold_carrier, field["original_parent_id"], ray, depth)
        point = evaluate_native_common_cold_material_point(*args, coordinate_increments=cold_coordinates,
            time_fraction=F(1), precision=precision)
        zero = (F(0),)*56
        point0 = point if cold_coordinates == zero else evaluate_native_common_cold_material_point(*args,
            coordinate_increments=zero, time_fraction=F(1), precision=precision)
        for actual in (point, point0):
            if (actual.original_material_id != field["original_material_id"]
                    or actual.original_parent_id != field["original_parent_id"]
                    or actual.owner != field["owner"] or actual.cohort_id != parent_cohort
                    or not actual.material_membership_authenticated or actual.volume_jacobian_bounds.lo <= 0
                    or actual.coordinate_order != cold_carrier.coordinate_order
                    or actual.allocation != ("archived" if archived else "live")
                    or actual.archive_debits != int(archived)):
                raise ValueError("A new point query changed the authentic source material/owner/allocation")
            native_common_cold_material_point_to_dict(actual)
        name = field["point_id"], field["side"]
        pointmap[name], baseline[name] = point, point0
    if counts != {"LIVE_ORIGINAL_MATERIAL": 24, "ARCHIVED_ORIGINAL_MATERIAL": 8}:
        raise ValueError("The full source has exactly twenty-four live and eight archived material sides")
    result = pointmap, baseline
    _MF_CONTACT_POINT_CACHE[key] = result
    return result


def _mf_roof_chart_bridge(original_ray, radius, roof, *, precision=96):
    """D(r_old*q)/D(r_original*q), at the exact original depth-zero label.

    The legacy BQ source derivative takes a roof-adjusted original-owner
    Cartesian chart as input. This bridge converts it to the SAME complete
    ORIGINAL_PARENT_CARTESIAN input used by the full895 field. The roof is
    a genuine literal source parameter; it is never silently set to zero.
    Its tangential/radial eigenvalues are alpha,alpha,beta with
    alpha=(R+roof)/R and beta=R^2/(R+roof)^2, so det(C)=1 exactly.
    """
    if (type(original_ray) is not tuple or len(original_ray) != 3
            or any(type(v) is not F for v in original_ray)
            or type(radius) is not F or type(roof) is not F
            or min(radius, radius+roof) <= 0):
        raise ValueError("The original source roof chart needs a nonzero exact ray and positive radii")
    raynorm = interval_sqrt(I(sum((v*v for v in original_ray), F(0))), precision)
    if raynorm.lo <= 0:
        raise ValueError("The original source ray has no positive homogeneous norm")
    q = tuple(_round(I(v)/raynorm, precision) for v in original_ray)
    unit_residual = _round(sum((_mf_square(v) for v in q), I(0))-1, precision)
    if not unit_residual.lo <= 0 <= unit_residual.hi:
        raise ValueError("The original roof correction lacks its complete unit-ray enclosure")
    alpha, beta = (radius+roof)/radius, radius*radius/(radius+roof)**2
    if roof == 0:
        C = tuple(tuple(I(int(i == j)) for j in range(3)) for i in range(3))
    else:
        C = tuple(tuple(_round(I(alpha*int(i == j))+(beta-alpha)*q[i]*q[j], precision)
            for j in range(3)) for i in range(3))
    return C, dict(original_ray_literal=_payload(original_ray),
        original_ray_norm_bounds=_payload(raynorm), original_unit_ray_bounds=_payload(q),
        original_unit_ray_norm_squared_residual_bounds=_payload(unit_residual),
        original_material_radius_m=_payload(radius), authenticated_source_roof_m=_payload(roof),
        original_source_chart_radius_m=_payload(radius+roof),
        roof_chart_tangential_coefficient_exact=_payload(alpha), roof_chart_radial_coefficient_exact=_payload(beta),
        roof_chart_correction_bounds=_payload(C), roof_chart_determinant_exact=_payload(F(1)),
        complete_original_parent_cartesian_input_chart_retained=True,
        positive_roof_chart_eigenvalues_certified=True,
        zero_roof_identity_preserved_symbolically=roof == 0)


def _mf_original_source_gradient(item, cold_carrier, *, precision=96):
    field = item["field"]
    bank = next((v for v in cold_carrier.banks if v.original_parent_id == field["original_parent_id"]), None)
    if (bank is None or bank.component_id != field["component_id"]
            or bank.original_material_id != field["original_material_id"] or bank.owner != field["owner"]):
        raise ValueError("The source original input chart needs its authentic complete cold parent")
    B, Q = _matrix(item["old_reference_gradient"]), _matrix(item["material_owner_pose"])
    legacy = _multiply(B, Q)
    ray = tuple(_fraction(v) for v in item["original_ray"])
    C, receipt = _mf_roof_chart_bridge(ray, bank.radius_m, bank.roof_radial_displacement_m, precision=precision)
    complete = tuple(tuple(_round(v, precision) for v in row) for row in _multiply(legacy, C))
    return complete, dict(**receipt,
        source_parent_id=bank.original_parent_id, source_component_id=bank.component_id,
        legacy_source_common_gradient_bounds=_payload(B), legacy_source_owner_pose_bounds=_payload(Q),
        legacy_source_common_owner_cartesian_gradient_bounds=_payload(legacy),
        complete_old_original_cartesian_gradient_bounds=_payload(complete),
        legacy_source_input_frame="OWNER_CURRENT_CARTESIAN_INPUT_TO_COMMON_PHI",
        complete_source_input_frame="ORIGINAL_PARENT_CARTESIAN_INPUT",
        original_roof_input_bridge_applied_once=True,
        old_owner_pose_applied_once_before_original_roof_bridge=True)


def _mf_contact_authority(cold_carrier, work891_packet, cold_coordinates, *, precision=96):
    """Shared source32 point authority, exported once per actual cold query.

    This preserves all original descriptors and the complete authenticated
    current/prepared-zero full895 fields; no HOT/internal context is hidden
    in an arbitrary selected force response.
    """
    if (type(cold_coordinates) is not tuple or len(cold_coordinates) != 56
            or any(type(v) is not F for v in cold_coordinates)):
        raise ValueError("A contact field authority needs the exact complete cold query")
    source = work891_packet["carrier_source_work_system"]
    seed = json.loads(source["seed_json"])
    pointmap, baseline = _mf_source_point_receipts(cold_carrier, seed, cold_coordinates, precision)
    itemmap = {(v["field"]["point_id"], v["field"]["side"]): v for v in seed["attachments"]}
    points = tuple(dict(point_id=name[0], side=name[1],
        current=native_common_cold_material_point_to_dict(pointmap[name]),
        prepared_zero=native_common_cold_material_point_to_dict(baseline[name]),
        original_saved_attachment=itemmap[name]) for name in sorted(pointmap))
    atoms = []
    for atom in seed["source_atoms"]:
        item = itemmap[atom["point_id"], "over"]
        oldA, bridge = _mf_original_source_gradient(item, cold_carrier, precision=precision)
        archival_n = _vector(atom["archival_old_reference"]["receiver_director_bounds"][0])
        Nsource = tuple(sum((oldA[j][i]*archival_n[j] for j in range(3)), I(0)) for i in range(3))
        atoms.append(dict(point_id=atom["point_id"], original_atom=atom,
            original_receiver_input_frame_bridge=bridge,
            original_receiver_material_gradient_bounds=_payload(oldA),
            original_receiver_archival_normal_bounds=_payload(archival_n),
            original_receiver_material_normal_bounds=_payload(Nsource)))
    return dict(format="new-mixed-force-original16-contact-field-authority-1",
        cold_carrier_binding_sha256=cold_carrier.binding_sha256,
        source_work_component_binding_sha256=source["binding_sha256"],
        cold_coordinate_order=_payload(cold_carrier.coordinate_order),
        cold_coordinate_increments=_payload(cold_coordinates), precision=precision,
        original_material_point_count=32, original_live_attachment_count=24,
        original_archived_attachment_count=8, original_atom_count=16,
        point_receipts=points, original_atom_authorities=tuple(atoms),
        prepared_zero_calibration_coordinates=_payload((F(0),)*56),
        old_work_or_source_memory_advanced=False,
        current_continuous_face_contact_law_defined=False,
        physical_material_admission_authorized=False)


def _mf_normal_orientation_hyperstress(A, material_normal, unit_normal, normal_norm, D, *, precision=96):
    """Derivative of the unit-normal pairing with respect to ORIGINAL A.

    It evaluates polynomial cofactor derivatives, without an inverse,
    and keeps the normalization derivative. The result S_ij satisfies
    d(n.D)=S:dA for a pure normal-orientation variation. Multiplication
    by A0*tau gives the physical point hyperstress in N*m.
    """
    if normal_norm.lo <= 0:
        raise ValueError("A material orientation hyperstress needs a proved nonzero cofactor normal")
    nd = _mf_interval_dot(unit_normal, D)
    tangent_D = tuple(d-n*nd for d, n in zip(D, unit_normal))
    result = []
    for a in range(3):
        row = []
        for b in range(3):
            dw = [I(0)]*3
            for i in range(3):
                for j in range(3):
                    rr, cc = [k for k in range(3) if k != i], [k for k in range(3) if k != j]
                    derivative = (int((a, b) == (rr[0], cc[0]))*A[rr[1]][cc[1]]
                        +A[rr[0]][cc[0]]*int((a, b) == (rr[1], cc[1]))
                        -int((a, b) == (rr[0], cc[1]))*A[rr[1]][cc[0]]
                        -A[rr[0]][cc[1]]*int((a, b) == (rr[1], cc[0])))
                    dw[i] += (-1 if (i+j) % 2 else 1)*derivative*material_normal[j]
            row.append(_round(_mf_interval_dot(tangent_D, tuple(dw))/normal_norm, precision))
        result.append(tuple(row))
    return tuple(result)


def _mf_material_side(item, point, internal_coordinate, global_cold_indices, *, cold_carrier, precision=96):
    field = item["field"]
    old_gradient, bridge = _mf_original_source_gradient(item, cold_carrier, precision=precision)
    old_inverse = _inverse(old_gradient, precision)
    identity = _multiply(old_gradient, old_inverse)
    if any(not (identity[i][j]-int(i == j)).lo <= 0 <= (identity[i][j]-int(i == j)).hi
            for i in range(3) for j in range(3)):
        raise ValueError("The original material source basis lacks its complete certified inverse")
    l0 = tuple(a-b for a, b in zip(_vector(field["source_reference_position_bounds_m"]),
                                  _vector(field["material_reference_position_bounds_m"])))
    v = _vector(field["source_virtual_radial_direction_bounds"])
    residual_index = field["residual_coordinate_index"]
    if type(residual_index) is not int or not 92 <= residual_index < 100:
        raise ValueError("An original source arm must name its genuine internal radial coordinate")
    r = _MFContactJet.variable(internal_coordinate, residual_index, precision)
    arm = tuple(r*x+y for x, y in zip(v, l0))
    positions = tuple(_MFContactJet(point.position_bounds_m[i],
        {index: point.mode_displacement_columns[k][i] for k, index in enumerate(global_cold_indices)},
        precision=precision) for i in range(3))
    original_gradient = tuple(tuple(_MFContactJet(point.original_cartesian_gradient_bounds[i][j],
        {index: point.mode_original_gradient_columns_per_m[k][i][j]
            for k, index in enumerate(global_cold_indices)}, precision=precision)
        for j in range(3)) for i in range(3))
    transport = tuple(tuple(sum((original_gradient[i][k]*old_inverse[k][j] for k in range(3)),
        _MFContactJet(0, precision=precision)) for j in range(3)) for i in range(3))
    convected = _mf_jet_matrix_vector(transport, arm)
    Y = tuple(a+b for a, b in zip(positions, convected))
    return Y, dict(old_original_cartesian_gradient_bounds=_payload(old_gradient),
        complete_original_source_input_frame_bridge=bridge,
        old_original_cartesian_inverse_bounds=_payload(old_inverse),
        complete_old_material_inverse_identity_bounds=_payload(identity),
        original_internal_arm_bounds_m=_payload(l0), original_virtual_radial_direction_bounds=_payload(v),
        internal_radial_coordinate_index=residual_index, internal_radial_increment_m=_payload(internal_coordinate),
        original_material_point_binding_sha256=point.binding_sha256,
        total_attachment_position_twojet=[_mf_jet_export(jet) for jet in Y],
        original_material_transport_twojet=[[_mf_jet_export(jet) for jet in row] for row in transport],
        full_fifty_six_seam_blended_material_columns_retained=True,
        original_input_basis_and_virtual_arm_retained=True,
        body_internal_arm_cross_derivatives_retained=True,
        internal_radial_arm_does_not_change_original_body_gradient=True)


def _mf_contacts(cold_carrier, work891_packet, cold_coordinates, internal_coordinates, *, precision=96):
    """Complete declared source16 material-follower endpoint E/G/H on COMMON100.

    Owner values are the fresh chart zero; all owner first/second variations
    are retained. This law owns only the original16 literal source atoms.
    It owns neither beams/bulk twice nor newly born interfaces/HOT contacts.
    """
    if (type(cold_coordinates) is not tuple or len(cold_coordinates) != 56
            or type(internal_coordinates) is not tuple or len(internal_coordinates) != 8
            or any(type(v) is not F for v in cold_coordinates+internal_coordinates)):
        raise ValueError("The follower contact query needs all fifty-six cold and eight internal coordinates")
    system891 = work891_packet["carrier_source_work_system"]
    seed = json.loads(system891["seed_json"])
    order = tuple(tuple(v) for v in system891["coordinate_order"])
    if len(order) != 100 or len(set(order)) != 100:
        raise ValueError("The contact assembly needs its authentic complete COMMON100 catalogue")
    global_cold_indices = tuple(order.index(tuple(v)) for v in cold_carrier.coordinate_order)
    if len(global_cold_indices) != 56 or set(global_cold_indices) != set(range(36, 92)):
        raise ValueError("The full cold source must inject once into the genuine COMMON56 slots")
    pointmap, baseline_points = _mf_source_point_receipts(cold_carrier, seed, cold_coordinates, precision)
    itemmap = {(v["field"]["point_id"], v["field"]["side"]): v for v in seed["attachments"]}
    atoms = seed["source_atoms"]
    if len(atoms) != 16 or len({v["point_id"] for v in atoms}) != 16:
        raise ValueError("The original sixteen positive material source atoms must occur exactly once")
    G, H = [I(0)]*100, [[I(0) for _ in range(100)] for _ in range(100)]
    total, recoverable, fracture = I(0), I(0), I(0)
    receipts = []
    exact_zero = not any(cold_coordinates+internal_coordinates)
    for atom in atoms:
        point_id = atom["point_id"]
        subitem, overitem = itemmap[point_id, "sub"], itemmap[point_id, "over"]
        subpoint, overpoint = pointmap[point_id, "sub"], pointmap[point_id, "over"]
        sub0, over0 = baseline_points[point_id, "sub"], baseline_points[point_id, "over"]
        sf, of = subitem["field"], overitem["field"]
        subY, subsaved = _mf_material_side(subitem, subpoint,
            internal_coordinates[sf["residual_coordinate_index"]-92], global_cold_indices,
            cold_carrier=cold_carrier, precision=precision)
        overY, oversaved = _mf_material_side(overitem, overpoint,
            internal_coordinates[of["residual_coordinate_index"]-92], global_cold_indices,
            cold_carrier=cold_carrier, precision=precision)
        if exact_zero:
            Ysub0, Yover0 = subY, overY
        else:
            Ysub0, _ = _mf_material_side(subitem, sub0, F(0), global_cold_indices,
                cold_carrier=cold_carrier, precision=precision)
            Yover0, _ = _mf_material_side(overitem, over0, F(0), global_cold_indices,
                cold_carrier=cold_carrier, precision=precision)
        old_over_gradient, _ = _mf_original_source_gradient(overitem, cold_carrier, precision=precision)
        n_archival = _vector(atom["archival_old_reference"]["receiver_director_bounds"][0])
        Nsource = tuple(sum((old_over_gradient[j][i]*n_archival[j] for j in range(3)), I(0)) for i in range(3))
        modegrad = {index: overpoint.mode_original_gradient_columns_per_m[k]
                    for k, index in enumerate(global_cold_indices)
                    if any(v != I(0) for row in overpoint.mode_original_gradient_columns_per_m[k] for v in row)}
        nmaterial, normal_receipt = _mf_nanson_director(overpoint.original_cartesian_gradient_bounds,
            modegrad, Nsource, precision=precision)
        if exact_zero:
            n0 = nmaterial
        else:
            n0, _ = _mf_nanson_director(over0.original_cartesian_gradient_bounds, {}, Nsource, precision=precision)
        baseline_scalar = _round(_mf_interval_dot(tuple(v.value for v in n0),
            tuple(b.value-a.value for a, b in zip(Ysub0, Yover0))), precision)
        owners_sub, owners_over = tuple(sf["owner_coordinate_indices"]), tuple(of["owner_coordinate_indices"])
        expected_sub = tuple(order.index(("owner", sf["owner"], axis)) for axis in ("x", "y", "z"))
        expected_over = tuple(order.index(("owner", of["owner"], axis)) for axis in ("x", "y", "z"))
        if owners_sub != expected_sub or owners_over != expected_over:
            raise ValueError("The source point substituted its actual material owner rotation chart")
        scalar = _mf_contact_scalar_twojet(subY, overY, nmaterial, owners_sub, owners_over,
            baseline_scalar, precision=precision)
        if exact_zero:
            # The current and calibration expressions are the SAME source
            # expression, so preserve their exact identity before intervals.
            scalar = _MFContactJet(I(0), scalar.gradient, scalar.hessian, precision)
        source_history = json.loads(atom["complete_original_atom_history_json"])
        source_material = source_history["original_material"]
        law, columns = source_material["law"], source_material["columns"]
        kn, strength, gamma = (_mf_source_fraction(law[name]) for name in
            ("normal_stiffness_pa_m", "tensile_strength_pa", "fracture_energy_j_m2"))
        C = (_mf_source_fraction(columns["sub_length_m"])/_mf_source_fraction(columns["sub_normal_modulus_pa"])
            +_mf_source_fraction(columns["over_length_m"])/_mf_source_fraction(columns["over_normal_modulus_pa"]))
        K = 1/(1/kn+C)
        if K != _fraction(atom["series_stiffness_exact_pa_m"][0]):
            raise ValueError("The full original material source series stiffness changed")
        if any(_mf_source_fraction(law[name]) != 0 for name in
            ("cohesion_pa", "friction_dry", "friction_wet", "viscosity_pa_s_m")):
            raise ValueError("This authentic relaxed shear source does not supply a nonzero shear/viscous law")
        tau_old = _fraction(atom["literal_source_tractions_pa"][0])
        shear_old = _fraction(atom["literal_source_tractions_pa"][1])
        state = json.loads(atom["source_local_state_json"])
        if tau_old >= 0 or shear_old != 0 or state["gap_m"] >= 0:
            raise ValueError("The prepared-zero calibration keeps the authentic compressed relaxed source")
        if any(state.get(name, 0.0) != 0.0 for name in ("max_opening_m", "damage")):
            raise ValueError("The declared virgin normal seed cannot recover a nonvirgin old normal memory")
        anchor = tau_old/K
        initial = _mf_cohesive_series(I(anchor), kn, C, strength, gamma, precision=precision)
        loading = _mf_cohesive_series(I(anchor)+scalar.value, kn, C, strength, gamma, precision=precision)
        area = _fraction(atom["original_area_m2"])
        literal_energy = sum((_fraction(v) for v in atom["stored_source_energy_components_j"]), F(0))
        if area <= 0:
            raise ValueError("An original positive source atom cannot be removed or assigned a chart area")
        offset = _round(I(literal_energy)-area*initial["loading_potential_density_bounds_j_m2"], precision)
        energy = _round(offset+area*loading["loading_potential_density_bounds_j_m2"], precision)
        stored = _round(offset+area*loading["recoverable_energy_density_bounds_j_m2"], precision)
        dissipated = _round(area*loading["fracture_work_density_bounds_j_m2"], precision)
        if exact_zero:
            energy = stored = I(literal_energy)
            dissipated = I(0)
        traction, tangent = loading["traction_bounds_pa"], loading["tangent_bounds_pa_m"]
        contactE = scalar.compose(energy, area*traction, area*tangent)
        active = sorted(set(contactE.gradient) | {i for p in contactE.hessian for i in p})
        localg = tuple(contactE.gradient.get(i, I(0)) for i in active)
        localh = tuple(tuple(contactE.hessian.get(_mf_pair(i, j), I(0)) for j in active) for i in active)
        # Independent vector/tensor virtual work at the material attachment:
        # tau n.d(Yover-Ysub) + tau dn.(Yover-Ysub). The second term is the
        # orientation contribution (hyperstress/couple); it is not dropped.
        D = tuple(b.value-a.value for a, b in zip(subY, overY))
        normal = tuple(v.value for v in nmaterial)
        normal_norm = _interval(normal_receipt["transported_cofactor_normal_norm_bounds"])
        orientation_scalar_gradient = _mf_normal_orientation_hyperstress(
            overpoint.original_cartesian_gradient_bounds, Nsource, normal, normal_norm, D, precision=precision)
        potential_orientation_hyperstress = tuple(tuple(_round(area*traction*v, precision)
            for v in row) for row in orientation_scalar_gradient)
        material_work = {}
        orientation_work = {}
        virtual_residual = {}
        for index in active:
            if index < 6:
                axis_index = (owners_sub.index(index) if index in owners_sub
                    else owners_over.index(index) if index in owners_over else None)
                if axis_index is None:
                    physical = orientation = I(0)
                else:
                    axis = _mf_axis(axis_index)
                    sub_spin = _mf_cross(axis, tuple(v.value for v in subY)) if index in owners_sub else (I(0),)*3
                    over_spin = _mf_cross(axis, tuple(v.value for v in overY)) if index in owners_over else (I(0),)*3
                    physical = _mf_interval_dot(normal, tuple(b-a for a, b in zip(sub_spin, over_spin)))
                    orientation = _mf_interval_dot(_mf_cross(axis, normal), D) if index in owners_over else I(0)
            else:
                dD = tuple(b.gradient.get(index, I(0))-a.gradient.get(index, I(0))
                            for a, b in zip(subY, overY))
                physical = _mf_interval_dot(normal, dD)
                if index in global_cold_indices:
                    local_index = global_cold_indices.index(index)
                    Gnormal = overpoint.mode_original_gradient_columns_per_m[local_index]
                    orientation = sum((orientation_scalar_gradient[i][j]*Gnormal[i][j]
                        for i in range(3) for j in range(3)), I(0))
                else:
                    orientation = I(0)
            pw = _round(area*traction*physical, precision)
            ow = _round(area*traction*orientation, precision)
            residual = _round(contactE.gradient.get(index, I(0))-pw-ow, precision)
            if not residual.lo <= 0 <= residual.hi:
                raise ValueError("The material follower normal lost attachment or orientation virtual work")
            material_work[index], orientation_work[index], virtual_residual[index] = pw, ow, residual
        force_on_over = tuple(_round(-area*traction*v, precision) for v in normal)
        force_on_sub = tuple(-v for v in force_on_over)
        orientation_couple = tuple(_round(-area*traction*v, precision) for v in _mf_cross(normal, D))
        sub_torque = tuple(_round(v, precision) for v in _mf_cross(tuple(v.value for v in subY), force_on_sub))
        over_torque = tuple(_round(a+b, precision) for a, b in zip(
            _mf_cross(tuple(v.value for v in overY), force_on_over), orientation_couple))
        moment_residual = tuple(_round(a+b, precision) for a, b in zip(sub_torque, over_torque))
        if any(not v.lo <= 0 <= v.hi for v in moment_residual):
            raise ValueError("The complete follower attachment forces and director couple fail global moment closure")
        for i, gi in zip(active, localg):
            G[i] += gi
        for i in active:
            for j in active:
                H[i][j] += contactE.hessian.get(_mf_pair(i, j), I(0))
        total += energy
        recoverable += stored
        fracture += dissipated
        receipts.append(dict(point_id=point_id, trace_id=atom["trace_id"],
            original_area_m2=_payload(area), source_work_lineage_id=atom["original_source_work_lineage_id"],
            original_atom_source_binding_json=json.dumps(atom, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False),
            original_source_state_json=atom["source_local_state_json"],
            original_material_law_json=json.dumps(source_material, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False),
            sub=subsaved, over=oversaved, receiver_nanson_normal=normal_receipt,
            prepared_zero_normal_bounds=_payload(tuple(v.value for v in n0)),
            prepared_zero_attachment_scalar_bounds_m=_payload(baseline_scalar),
            source_free_gap_anchor_m=_payload(anchor), source_traction_bounds_pa=_payload(I(tau_old)),
            contact_stiffness_exact_pa_m=_payload(kn), original_column_compliance_exact_m_pa=_payload(C),
            tensile_strength_exact_pa=_payload(strength), fracture_energy_exact_j_m2=_payload(gamma),
            literal_source_recoverable_energy_j=_payload(literal_energy), literal_energy_offset_bounds_j=_payload(offset),
            scalar_increment_twojet=_mf_jet_export(scalar), loading=_payload(loading),
            energy_bounds_j=_payload(energy), recoverable_energy_bounds_j=_payload(stored),
            proposed_fracture_work_bounds_j=_payload(dissipated),
            active_coordinate_indices=active, gradient_bounds=_payload(localg), hessian_bounds=_payload(localh),
            prestress_curvature_hessian_bounds=_payload(tuple(tuple(area*traction*scalar.hessian.get(_mf_pair(i, j), I(0))
                for j in active) for i in active)),
            force_on_sub_bounds_n=_payload(force_on_sub), force_on_over_bounds_n=_payload(force_on_over),
            action_reaction_residual_bounds_n=_payload((I(0),)*3),
            potential_original_gradient_orientation_hyperstress_bounds_nm=_payload(potential_orientation_hyperstress),
            physical_original_gradient_orientation_hyperstress_bounds_nm=_payload(tuple(tuple(-v for v in row)
                for row in potential_orientation_hyperstress)),
            receiver_normal_orientation_couple_bounds_nm=_payload(orientation_couple),
            total_attachment_torque_on_sub_bounds_nm=_payload(sub_torque),
            total_attachment_torque_on_over_bounds_nm=_payload(over_torque),
            global_moment_residual_bounds_nm=_payload(moment_residual),
            material_attachment_virtual_work_bounds=[[i, _payload(v)] for i, v in sorted(material_work.items())],
            nanson_orientation_virtual_work_bounds=[[i, _payload(v)] for i, v in sorted(orientation_work.items())],
            virtual_work_residual_bounds=[[i, _payload(v)] for i, v in sorted(virtual_residual.items())],
            receiver_owner_coordinate_indices=list(owners_over), sub_owner_coordinate_indices=list(owners_sub),
            owner_chart="FRESH_LEFT_EXPONENTIAL_AT_EXACT_ZERO_WITH_ALL_SIX_FIRST_SECOND_VARIATIONS",
            original_material_nanson_normal_law_defined=True,
            normalized_cofactor_direction_and_full_second_derivatives_included=True,
            orientation_hyperstress_virtual_work_included=True,
            simultaneous_fresh_owner_rotation_objective_by_relative_rotation_identity=True,
            prepared_zero_source_traction_calibration=True,
            exact_prepared_zero_scalar_energy_identity_used=exact_zero,
            original_source_area_included_once=True,
            current_area_energy_multiplier_added=False,
            current_continuous_face_contact_law_defined=False,
            born_hot_receiving_law_inferred=False,
            source_normal_history_seed="NEW_DECLARED_VIRGIN_MODE_I_PREPARATION_NOT_RECOVERED_OLD_FRACTURE_MEMORY",
            proposed_normal_damage_or_fracture_work_applied=False,
            source_state_or_work_advanced=False))
    if any(G[i] != I(0) or any(H[i][j] != I(0) or H[j][i] != I(0) for j in range(100)) for i in range(6, 36)):
        raise ValueError("Original COLD source contacts cannot acquire an invented HOT receiving law")
    points = _mf_contact_authority(cold_carrier, work891_packet, cold_coordinates, precision=precision)["point_receipts"]
    return dict(energy_bounds_j=_round(total, precision), recoverable_bounds_j=_round(recoverable, precision),
        fracture_bounds_j=_round(fracture, precision),
        gradient_bounds=tuple(_round(v, precision) for v in G),
        hessian_bounds=tuple(tuple(_round(v, precision) for v in row) for row in H),
        atoms=tuple(receipts), point_receipts=points,
        original_atom_count=16, original_live_attachment_count=24, original_archived_attachment_count=8,
        original_material_nanson_follower_contact_component_defined=True,
        prepared_zero_source_traction_calibration=True,
        prepared_reference_full_system_work_certified=False,
        physical_source_force_prestress_work_handover_certified=False,
        historical_normal_fracture_memory_recovered=False,
        current_continuous_face_contact_law_defined=False,
        born_hot_receiving_law_inferred=False,
        clocks_advanced=False, accepted_new_native_intervals=0)
