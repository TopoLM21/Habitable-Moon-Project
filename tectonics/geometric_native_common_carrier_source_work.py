"""Literal cohesive source preparation on the declared whole-material carrier.

This proposal keeps the registered COMMON body values, original beams and
bulk remainder.  It supplies no new bulk operator, follower contact search,
HOT law, physical interval acceptance or history/clock commit.  The scalar
loading potential is recoverable energy PLUS Mode-I dissipation; it is not
called a conservative stored energy after fracture.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from fractions import Fraction as F
import hashlib
import json

from .geometric_native_pose import FractionInterval as I
from .geometric_native_archive_material_carrier import (
    NativeArchiveMaterialCarrier, native_archive_material_carrier_to_dict,
    evaluate_native_archive_material_point, native_archive_material_point_to_dict,
)
from .geometric_native_common_local_source_work import (
    _local_source_jet, _second_spin, _axis, _scatter,
)
from .geometric_native_common_source_carrier import _round, _dot, _cross, _map

FORMAT = "native-common-carrier-cohesive-source-preparation-1"
RESPONSE_FORMAT = "native-common-carrier-cohesive-source-preparation-response-1"
SCOPE = "ACTUAL_CARRIER_ATTACHMENT_AND_AUTHENTIC_COHESIVE_PREPARATION_COMPONENT_ONLY"
BRANCH = "monotone_cohesive_loading"
_AUTHORITY_SEALS = (
    "faca2daf3a40dc39e6ae7aaca7fcd91c887756fb50eed3c37b09b8183d920469",
    "368a524a954ac5f2ba3a6b92c182d34d1c3567ba7df1314c0afd02b2cff2c23e",
    "a46fe79aeabef70ec01672ad4c33c737bfc06d5323a112edcfeb2616f7b148a3",
)
_REGISTRY = {}
_GEOMETRY_CACHE = {}
_RESPONSE_GUARDS = {}
_ZERO = (I(0),) * 3
_IDENTITY = tuple(tuple(I(int(i == j)) for j in range(3)) for i in range(3))


def _payload(value):
    if type(value) is F:
        return [value.numerator, value.denominator]
    if type(value) is I:
        return dict(lo=_payload(value.lo), hi=_payload(value.hi))
    if type(value) is tuple:
        return [_payload(v) for v in value]
    if hasattr(type(value), "__dataclass_fields__"):
        return {f.name: _payload(getattr(value, f.name)) for f in fields(value)}
    if type(value) in (str, int, bool) or value is None:
        return value
    if type(value) is dict:
        return {k: _payload(v) for k, v in value.items()}
    if type(value) is list:
        return [_payload(v) for v in value]
    raise ValueError("A source preparation payload has an undeclared scalar/container kind")


def _encoded(value):
    return json.dumps(_payload(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _typed_guard(value):
    """Include literal Python kinds, which the exported JSON cannot distinguish."""
    if type(value) is F:
        result = ["Fraction",value.numerator,value.denominator]
    elif type(value) is I:
        result = ["FractionInterval",_typed_guard(value.lo),_typed_guard(value.hi)]
    elif hasattr(type(value),"__dataclass_fields__"):
        result = [type(value).__module__,type(value).__qualname__,
                  [[f.name,_typed_guard(getattr(value,f.name))] for f in fields(value)]]
    elif type(value) in (tuple,list):
        result = [type(value).__name__,[_typed_guard(v) for v in value]]
    elif type(value) is dict:
        if any(type(k) is not str for k in value):
            raise ValueError("A typed preparation receipt retains literal string keys")
        result = ["dict",[[k,_typed_guard(value[k])] for k in sorted(value)]]
    elif type(value) in (str,int,bool) or value is None:
        result = [type(value).__name__,value]
    else:
        raise ValueError("A preparation receipt contains an undeclared typed value")
    return hashlib.sha256(json.dumps(result,sort_keys=True,separators=(",",":"),
                                    ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def _fraction(raw):
    if (type(raw) is not list or len(raw) != 2 or
            any(type(v) is not int for v in raw) or raw[1] <= 0):
        raise ValueError("Saved exact fractions retain both literal integer kinds")
    result = F(*raw)
    if [result.numerator, result.denominator] != raw:
        raise ValueError("A saved fraction is not canonical")
    return result


def _interval(raw):
    if type(raw) is not dict or set(raw) != {"lo", "hi"}:
        raise ValueError("A saved interval needs its complete bounds")
    return I(_fraction(raw["lo"]), _fraction(raw["hi"]))


def _vector(raw):
    if type(raw) is not list or len(raw) != 3:
        raise ValueError("A saved Cartesian vector has exactly three entries")
    return tuple(_interval(v) for v in raw)


def _matrix(raw):
    if type(raw) is not list or len(raw) != 3:
        raise ValueError("A saved Cartesian matrix has exactly three rows")
    return tuple(_vector(v) for v in raw)


def _precision(value):
    if type(value) is not int or not 96 <= value <= 512:
        raise ValueError("Source preparation uses an integer precision from 96 to 512")
    return value


def _exact_float(value):
    if type(value) is not float:
        raise ValueError("Constitutive binary64 authority retains its original scalar kind")
    return F.from_float(value)


def _vadd(a, b):
    return tuple(x + y for x, y in zip(a, b))


def _vsub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _transpose(a):
    return tuple(tuple(a[j][i] for j in range(3)) for i in range(3))


def _multiply(a, b):
    return tuple(tuple(sum((a[i][k] * b[k][j] for k in range(3)), I(0))
                       for j in range(3)) for i in range(3))


def _determinant(a):
    return (a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])
            -a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])
            +a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]))


def _inverse(a, precision):
    determinant = _round(_determinant(a),precision)
    if determinant.lo <= 0:
        raise ValueError("The old complete Cartesian chart inverse lacks a positive determinant")
    cofactors = []
    for i in range(3):
        row = []
        for j in range(3):
            rows,cols = tuple(k for k in range(3) if k != i),tuple(k for k in range(3) if k != j)
            minor = a[rows[0]][cols[0]]*a[rows[1]][cols[1]]-a[rows[0]][cols[1]]*a[rows[1]][cols[0]]
            row.append(_round((-1 if (i+j)%2 else 1)*minor,precision))
        cofactors.append(tuple(row))
    return tuple(tuple(_round(cofactors[j][i]/determinant,precision) for j in range(3)) for i in range(3))


def _material_transport_pullback(new_original_gradient, old_common_gradient,
        old_material_owner_pose, *, precision=96):
    """T=A(BQ)^-1, retaining the complete old roof/depth shear and owner pose."""
    _precision(precision)
    inverse = _multiply(_transpose(old_material_owner_pose),_inverse(old_common_gradient,precision))
    return tuple(tuple(_round(v,precision) for v in row)
                 for row in _multiply(new_original_gradient,inverse))


def _outer(a, b):
    return tuple(tuple(x * y for y in b) for x in a)


def _contract(a, b):
    return sum((a[i][j] * b[i][j] for i in range(3) for j in range(3)), I(0))


def _square(a):
    lo = F(0) if a.lo <= 0 <= a.hi else min(a.lo*a.lo, a.hi*a.hi)
    return I(lo, max(a.lo*a.lo, a.hi*a.hi))


def _max_zero(a):
    return I(max(F(0), a.lo), max(F(0), a.hi))


@dataclass(frozen=True, slots=True)
class NativeCohesiveSeriesLoading:
    branch: str
    free_gap_bounds_m: I
    contact_gap_bounds_m: I
    traction_bounds_pa: I
    tangent_bounds_pa_m: I
    loading_potential_density_bounds_j_m2: I
    recoverable_energy_density_bounds_j_m2: I
    fracture_work_density_bounds_j_m2: I
    proposed_damage_bounds: I
    proposed_max_opening_bounds_m: I
    onset_free_gap_m: F
    failure_free_gap_m: F
    energy_partition_residual_bounds_j_m2: I


def evaluate_native_cohesive_series_loading(free_gap_bounds_m, *,
        contact_stiffness_pa_m, column_compliance_m_pa, tensile_strength_pa,
        fracture_energy_j_m2, precision=96):
    """Exact monotone series solution of the literal bilinear Mode-I law.

    The contact gap differs from the total free gap by C*traction.  Reject
    branch-corner boxes: there is no single classical Hessian there.
    """
    _precision(precision)
    d = free_gap_bounds_m
    kn, C, T, gamma = (contact_stiffness_pa_m, column_compliance_m_pa,
                      tensile_strength_pa, fracture_energy_j_m2)
    if type(d) is not I or any(type(v) is not F for v in (kn, C, T, gamma)):
        raise ValueError("The cohesive series law requires complete exact inputs")
    if min(kn, T, gamma) <= 0 or C < 0:
        raise ValueError("The actual contact and source columns need positive moduli")
    g0, gf = T / kn, 2 * gamma / T
    onset, failure = g0 + C*T, gf
    if gf <= g0 or failure <= onset:
        raise ValueError("The cohesive series path has snapback or no softening interval")
    K = 1 / (1/kn + C)
    if d.hi < onset:
        branch = "intact_or_compressed"
        traction, tangent = K*d, I(K)
        gap = d / (1 + C*kn)
        damage, fracture = I(0), I(0)
        loading = K * _square(d) / 2
        recoverable = kn * _square(gap) / 2 + C * _square(traction) / 2
    elif d.lo > failure:
        branch = "fully_failed_open"
        gap, traction, tangent = d, I(0), I(0)
        damage, fracture = I(1), I(gamma)
        loading, recoverable = I(gamma), I(0)
    elif onset < d.lo and d.hi < failure:
        branch = "monotone_softening"
        gap = I(g0) + (d-onset)*(gf-g0)/(failure-onset)
        traction, tangent = T*(failure-d)/(failure-onset), I(-T/(failure-onset))
        damage = gf/(gf-g0)*(1-g0/gap)
        fracture = gamma*(gap-g0)/(gf-g0)
        delta = d-onset
        loading = I(T*onset/2) + T*delta - T*_square(delta)/(2*(failure-onset))
        recoverable = kn*(1-damage)*_square(gap)/2 + C*_square(traction)/2
    else:
        raise ValueError("Cohesive branch threshold box has no unique classical twojet")
    values = tuple(_round(v, precision) for v in (gap, traction, tangent,
        loading, recoverable, fracture, damage, _max_zero(gap), loading-recoverable-fracture))
    if not values[-1].lo <= 0 <= values[-1].hi:
        raise ValueError("Cohesive recoverable and fracture energy partition does not close")
    return NativeCohesiveSeriesLoading(branch, d, *values[:-1], onset, failure, values[-1])


def _carrier_side_jet(p0, Xold, chi, transport, psi, gradients, coordinates,
        residual, radial_direction, owner_indices, body_indices, residual_index,
        *, preparation_fraction, precision=96):
    """Local twojet of χλ+u+(Tλ+Du)(l+r*v), in the OLD material basis."""
    _precision(precision)
    lam = preparation_fraction
    if type(lam) is not F or not 0 <= lam <= 1:
        raise ValueError("Preparation is a separate exact fraction, not an owner clock")
    if (any(type(v) is not tuple or len(v) != 3 for v in (p0, Xold, chi, radial_direction))
            or type(coordinates) is not tuple or any(type(v) is not F for v in coordinates)
            or type(residual) is not F or len(psi) != len(coordinates)
            or len(gradients) != len(coordinates) or len(body_indices) != len(coordinates)
            or len(owner_indices) != 3):
        raise ValueError("A material side twojet needs every typed sparse driver")
    active = owner_indices + body_indices + (residual_index,)
    if len(active) != len(set(active)):
        raise ValueError("Owner, body and material residual drivers cannot alias")
    base = tuple(Xold[i]+lam*(chi[i]-Xold[i]) for i in range(3))
    T = tuple(tuple(_IDENTITY[i][j]+lam*(transport[i][j]-_IDENTITY[i][j])
                    for j in range(3)) for i in range(3))
    u = tuple(sum((a*p[i] for a, p in zip(coordinates, psi)), I(0)) for i in range(3))
    D = tuple(tuple(sum((a*g[i][j] for a, g in zip(coordinates, gradients)), I(0))
                    for j in range(3)) for i in range(3))
    total = tuple(tuple(_round(T[i][j]+D[i][j], precision) for j in range(3)) for i in range(3))
    arm0 = _vsub(p0, Xold)
    arm = tuple(l+residual*v for l, v in zip(arm0, radial_direction))
    x = tuple(_round(a+b, precision) for a, b in zip(base, u))
    Farm = tuple(_round(v, precision) for v in _map(total, arm))
    y = tuple(_round(a+b, precision) for a, b in zip(x, Farm))
    body = tuple(tuple(_round(a+b, precision) for a,b in zip(p, _map(g, arm)))
                 for p,g in zip(psi,gradients))
    radial = tuple(_round(v, precision) for v in _map(total, radial_direction))
    first = {index: tuple(_round(v, precision) for v in _cross(_axis(i), y))
             for i,index in enumerate(owner_indices)}
    first.update(zip(body_indices,body)); first[residual_index] = radial
    second = {}
    for i,ia in enumerate(owner_indices):
        for j,ib in enumerate(owner_indices):
            second[ia,ib] = tuple(_round(v,precision) for v in _second_spin(_axis(i),_axis(j),y))
        for index in body_indices+(residual_index,):
            value = tuple(_round(v,precision) for v in _cross(_axis(i),first[index]))
            second[ia,index] = second[index,ia] = value
    for index,g in zip(body_indices,gradients):
        value = tuple(_round(v,precision) for v in _map(g,radial_direction))
        second[index,residual_index] = second[residual_index,index] = value
    path = tuple(_round(a+b,precision) for a,b in zip(_vsub(chi,Xold),
        _map(tuple(tuple(transport[i][j]-_IDENTITY[i][j] for j in range(3)) for i in range(3)),arm)))
    return dict(x=x,y=y,delta_y=_vsub(y,p0),arm0=arm0,arm=arm,Farm=Farm,F=total,
        u=u,virtual_radial_driver=radial,body_drivers=body,first=first,second=second,
        owner_indices=owner_indices,body_indices=body_indices,residual_index=residual_index,
        mode_displacement_columns=psi,mode_spatial_gradient_columns_per_m=gradients,
        preparation_driver=path,reference_position=Xold,carrier_position=chi,
        carrier_reference_transport=transport,preparation_fraction=lam)


@dataclass(frozen=True, slots=True)
class NativeCommonCarrierSourceWorkSystem:
    seed_json: str
    source_packet_seals: tuple
    coordinate_order: tuple
    bound_coordinate_increments: tuple
    archive_carrier_model_sha256: str
    provenance: str
    precision: int
    binding_sha256: str
    scope: str = SCOPE
    tensile_history_seed: str = "NEW_DECLARED_VIRGIN_MODE_I_PREPARATION_NOT_RECOVERED_OLD_FRACTURE_MEMORY"
    old_normal_fracture_memory_recovered: bool = False


@dataclass(frozen=True, slots=True)
class NativeCarrierSourceSideResponse:
    point_id: str
    side: str
    original_parent_id: str
    original_material_id: str
    owner: int
    original_ray: tuple
    original_label_membership: str
    archive_cohort_id: str | None
    material_depth_bounds_m: I
    actual_material_position_bounds_m: tuple
    carrier_reference_position_bounds_m: tuple
    old_material_reference_position_bounds_m: tuple
    virtual_source_position_bounds_m: tuple
    virtual_source_displacement_bounds_m: tuple
    internal_arm_bounds_m: tuple
    convected_internal_arm_bounds_m: tuple
    deformation_gradient_bounds: tuple
    total_original_material_deformation_gradient_bounds: tuple
    local_material_volume_jacobian_bounds: I
    carrier_reference_transport_bounds: tuple
    old_original_reference_gradient_bounds: tuple
    frame_pullback_identity_residual_bounds: tuple
    virtual_radial_driver_bounds: tuple
    body_driver_columns: tuple
    carrier_point_json: str


@dataclass(frozen=True, slots=True)
class NativeCarrierSourceAtomResponse:
    point_id: str
    trace_id: str
    original_area_m2: F
    source_work_lineage_id: tuple
    sub: NativeCarrierSourceSideResponse
    over: NativeCarrierSourceSideResponse
    active_coordinate_indices: tuple
    receiver_director_bounds: tuple
    scalar_increment_bounds_m: tuple
    scalar_first_derivative_bounds: tuple
    scalar_second_derivative_bounds: tuple
    source_free_gap_anchor_m: F
    preparation_normal_driver_bounds_m: I
    before_free_gap_bounds_m: I
    current_free_gap_bounds_m: I
    loading_response: NativeCohesiveSeriesLoading
    before_loading_response: NativeCohesiveSeriesLoading
    literal_energy_rounding_offset_bounds_j: I
    loading_potential_bounds_j: I
    recoverable_energy_bounds_j: I
    proposed_fracture_work_bounds_j: I
    potential_gradient_bounds: tuple
    potential_hessian_bounds: tuple
    prestress_curvature_hessian_bounds: tuple
    force_on_sub_n: tuple
    force_on_over_n: tuple
    sub_material_hyperstress_bounds_nm: tuple
    over_material_hyperstress_bounds_nm: tuple
    complete_attachment_torque_on_sub_nm: tuple
    complete_attachment_torque_on_over_nm: tuple
    action_reaction_residual_bounds_n: tuple
    world_moment_residual_bounds_nm: tuple
    virtual_work_residual_bounds: tuple
    preparation_potential_work_bounds_j: I
    preparation_physical_reaction_work_bounds_j: I
    preparation_recoverable_energy_change_bounds_j: I
    preparation_fracture_work_increment_bounds_j: I
    preparation_energy_balance_residual_bounds_j: I
    original_source_state_json: str
    proposed_plastic_slip_bounds_m: I
    proposed_cumulative_slip_bounds_m: I


@dataclass(frozen=True, slots=True)
class NativeCommonCarrierSourceWorkResponse:
    system_binding_sha256: str
    time_fraction: F
    preparation_fraction: F
    coordinate_order: tuple
    coordinate_increments: tuple
    branch: str
    source_atom_responses: tuple
    source_beam_responses: tuple
    source_loading_potential_bounds_j: I
    registered_orthogonal_energy_bounds_j: I
    loading_potential_bounds_j: I
    recoverable_energy_bounds_j: I
    proposed_fracture_work_bounds_j: I
    potential_gradient_bounds: tuple
    potential_hessian_bounds: tuple
    preparation_potential_work_bounds_j: I
    preparation_physical_reaction_work_bounds_j: I
    preparation_recoverable_energy_change_bounds_j: I
    preparation_fracture_work_increment_bounds_j: I
    preparation_energy_balance_residual_bounds_j: I
    source_contact_elapsed_time_s: F
    source_memory_elapsed_time_s: F
    source_geometry_elapsed_time_s: F
    material_geometry_elapsed_time_s: F
    precision: int
    binding_sha256: str
    scope: str = SCOPE
    source_state_or_work_advanced: bool = False
    whole_support_nonzero_common_material_admission_certified: bool = False
    current_follower_contact_law_defined: bool = False
    hot_constitutive_law_defined: bool = False
    new_mixed_equilibrium_certified: bool = False
    accepted_new_native_intervals: int = 0
    tensile_history_seed: str = "NEW_DECLARED_VIRGIN_MODE_I_PREPARATION_NOT_RECOVERED_OLD_FRACTURE_MEMORY"
    old_normal_fracture_memory_recovered: bool = False


def _body(value):
    return {f.name: getattr(value,f.name) for f in fields(value) if f.name != "binding_sha256"}


def _packet(packet, expected):
    if type(packet) is not dict or packet.get("packet_sha256") != expected:
        raise ValueError("A genuine immutable source packet seal is required")
    # Packet JSON contains literal floats; do not coerce them through _payload.
    raw = json.dumps({k:v for k,v in packet.items() if k != "packet_sha256"},
        sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("A complete source packet changed behind its saved seal")


def bind_native_common_carrier_source_work(local_source_packet,
        material_transport_packet, source_carrier_packet, archive_carrier, *,
        provenance, precision=96, initial_max_opening_m=F(0), initial_damage=F(0)):
    _precision(precision)
    if type(provenance) is not str or not provenance.strip():
        raise ValueError("Declare the new geometry preparation and its tensile history seed")
    if (type(initial_max_opening_m) is not F or type(initial_damage) is not F
            or initial_max_opening_m != 0 or initial_damage != 0):
        raise ValueError("This new declared virgin preparation cannot represent nonvirgin fracture history")
    if type(archive_carrier) is not NativeArchiveMaterialCarrier:
        raise ValueError("Actual material attachment requires its authenticated whole carrier")
    carrier_dto = native_archive_material_carrier_to_dict(archive_carrier)
    for p,s in zip((local_source_packet,material_transport_packet,source_carrier_packet),_AUTHORITY_SEALS):
        _packet(p,s)
    local = local_source_packet["local_source_work_system"]
    transfer = material_transport_packet["material_transfer"]
    source = source_carrier_packet["carrier_system"]
    material_states = tuple(json.loads(v) for v in (
        archive_carrier.source_material_state_json,
        transfer["complete_original_material_state_json"],
        local["complete_original_source_material_state_json"]))
    canonical_states = tuple(json.dumps(v,sort_keys=True,separators=(",",":"),
        ensure_ascii=False,allow_nan=False) for v in material_states)
    if (len(set(canonical_states)) != 1
            or material_states[0]["state_sha256"] != archive_carrier.source_material_state_sha256):
        raise ValueError("The carrier and all source terms must share the complete authentic original material state")
    fields_by_key = {(v["point_id"],v["side"]):v for v in local["material_point_fields"]}
    tops = {}
    for row in material_transport_packet["original_material_queries"]:
        r = row["response"]
        if r["material_depth_bounds_m"] == {"lo":[0,1],"hi":[0,1]}:
            key = r["point_id"],r["side"]
            if key in tops:
                raise ValueError("A literal material side cannot duplicate its preparation seed")
            tops[key] = r
    if len(fields_by_key) != 32 or set(fields_by_key) != set(tops):
        raise ValueError("All 32 authentic original material attachments must occur once")
    saved = []
    for key,f in fields_by_key.items():
        r = tops[key]
        checks = (("material_reference_position_bounds_m","reference_position_bounds_m"),
                  ("mode_displacement_columns","mode_displacement_columns"),
                  ("mode_spatial_gradient_columns_per_m","mode_spatial_gradient_columns_per_m"))
        if any(f[a] != r[b] for a,b in checks) or any(f[n] != r[n] for n in (
                "original_parent_id","original_material_id","owner","archive_cohort_id")):
            raise ValueError("Saved material fields and old local source work do not share one label/basis")
        if r["reference_gradient_input_frame"] != "OWNER_CURRENT_CARTESIAN_INPUT_TO_COMMON_PHI":
            raise ValueError("The old material gradient has a different Cartesian input frame")
        saved.append(dict(field=f,original_ray=r["original_direction_literal"],
            old_reference_gradient=r["reference_gradient_bounds"],
            material_owner_pose=r["material_owner_pose_bounds"]))
    if len(source["carriers"]) != 16 or len(transfer["beams"]) != 8 or len(local["projection_blocks"]) != 8:
        raise ValueError("The original single potential has exactly 16+8+8 terms")
    if local["complete_current_source_states_json"] != source["complete_current_source_states_json"]:
        raise ValueError("Cohesive preparation needs the same literal old source states")
    for atom in source["carriers"]:
        history = json.loads(atom["complete_original_atom_history_json"])
        material = history["original_material"]
        law = material["law"]
        if any(law[n] != 0.0 for n in ("cohesion_pa","friction_dry","friction_wet","viscosity_pa_s_m")):
            raise ValueError("This authentic relaxed shear preparation requires zero friction/cohesion/viscosity")
        state = json.loads(atom["source_local_state_json"])
        if state["gap_m"] >= 0 or _fraction(atom["literal_source_tractions_pa"][1]) != 0:
            raise ValueError("Virgin tensile loading is seeded only by the retained compressed relaxed source")
        if any(state.get(n,0.0) != 0.0 for n in ("max_opening_m","damage")):
            raise ValueError("The actual source explicitly declares nonvirgin normal fracture history")
    seed = dict(local=local,attachments=saved,source_atoms=source["carriers"],
                beams=transfer["beams"],
                tensile_history_initialization="NEW_DECLARED_VIRGIN_MODE_I_PREPARATION_NOT_RECOVERED_OLD_FRACTURE_MEMORY",
                old_normal_memory_contains_only_shear_fields=True,
                fixed_archival_mechanical_directors=True,
                unchanged_source_clocks_and_work_ledger_json=True)
    # Save only authenticated immutable fields, not mutable external dictionaries.
    seed_json = json.dumps(seed,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)
    order = tuple(tuple(v) for v in local["coordinate_order"])
    coordinates = tuple(_fraction(v) for v in local["bound_coordinate_increments"])
    if len(order) != 100 or len(coordinates) != 100 or coordinates[:6] != (F(0),)*6:
        raise ValueError("The authentic fixed COMMON catalogue contains 100 coordinates")
    system = NativeCommonCarrierSourceWorkSystem(seed_json,_AUTHORITY_SEALS,order,coordinates,
        archive_carrier.model_sha256,provenance,precision,"")
    system = replace(system,binding_sha256=_digest(_body(system)))
    _REGISTRY[system.binding_sha256] = (system,json.loads(seed_json),archive_carrier)
    return system


def _registered(system):
    if type(system) is not NativeCommonCarrierSourceWorkSystem:
        raise ValueError("The source preparation needs its exact typed registered system")
    value = _REGISTRY.get(system.binding_sha256)
    if value is None or value[0] != system or _digest(_body(system)) != system.binding_sha256:
        raise ValueError("The complete source preparation system changed")
    native_archive_material_carrier_to_dict(value[2])
    return value[1],value[2]


def _geometry(system,time,precision):
    key = system.binding_sha256,time,precision
    if key in _GEOMETRY_CACHE:
        return _GEOMETRY_CACHE[key]
    seed,carrier = _registered(system)
    result = {}
    parent_allocations = {v["fragment_id"]:v for v in json.loads(
        seed["local"]["complete_original_source_material_state_json"])["parent_allocations"]}
    for item in seed["attachments"]:
        field = item["field"]
        ray = tuple(_fraction(v) for v in item["original_ray"])
        point = evaluate_native_archive_material_point(carrier,field["original_parent_id"],ray,I(0),time,precision=precision)
        if point.material_id != field["original_material_id"] or point.owner != field["owner"]:
            raise ValueError("A physical source point changed its original material identity or owner")
        # The genuine public chronology reports the ORIGINAL PARENT's cohort
        # on live as well as archived points.  The old source field's cohort
        # is instead its POINT-specific archived-attachment annotation.
        parent_cohort = parent_allocations[field["original_parent_id"]]["cohort_id"]
        membership = field["original_label_membership"]
        if membership not in ("LIVE_ORIGINAL_MATERIAL","ARCHIVED_ORIGINAL_MATERIAL"):
            raise ValueError("The source endpoint has an unknown original material membership enum")
        archived = membership == "ARCHIVED_ORIGINAL_MATERIAL"
        expected_point_cohort = parent_cohort if archived else None
        if (point.cohort_id != parent_cohort
                or field["archive_cohort_id"] != expected_point_cohort
                or point.allocation != ("archived" if archived else "live")
                or point.archive_debits != int(archived)
                or (archived and parent_cohort is None)):
            raise ValueError("Actual endpoint source material allocation differs from the genuine retained ancestry")
        B,Q = _matrix(item["old_reference_gradient"]),_matrix(item["material_owner_pose"])
        old = _multiply(B,Q)
        # Saved roofs need not be zero.  Invert the COMPLETE authenticated
        # old Cartesian derivative, including its finite depth rotation.
        inverse = _multiply(_transpose(Q),_inverse(B,precision))
        residual = _multiply(old,inverse)
        if any(not (residual[i][j]-int(i==j)).lo <= 0 <= (residual[i][j]-int(i==j)).hi
               for i in range(3) for j in range(3)):
            raise ValueError("The authenticated old material chart inverse does not enclose identity")
        transport = _material_transport_pullback(point.deformation_gradient_bounds,B,Q,precision=precision)
        pull = _multiply(transport,old)
        identity = tuple(tuple(_round(pull[i][j]-point.deformation_gradient_bounds[i][j],precision)
                               for j in range(3)) for i in range(3))
        if any(not v.lo <= 0 <= v.hi for row in identity for v in row):
            raise ValueError("New and old material Cartesian bases fail the complete pullback")
        result[field["point_id"],field["side"]] = (point,transport,old,identity)
    _GEOMETRY_CACHE[key] = result
    return result


def _side(field,item,geometry,q,lam,precision,carrier):
    point,transport,old,identity = geometry
    psi = tuple(_vector(v) for v in field["mode_displacement_columns"])
    gradients = tuple(_matrix(v) for v in field["mode_spatial_gradient_columns_per_m"])
    indices = tuple(field["body_coordinate_indices"])
    jet = _carrier_side_jet(_vector(field["source_reference_position_bounds_m"]),
        _vector(field["material_reference_position_bounds_m"]),point.position_bounds_m,transport,
        psi,gradients,tuple(q[i] for i in indices),q[field["residual_coordinate_index"]],
        _vector(field["source_virtual_radial_direction_bounds"]),tuple(field["owner_coordinate_indices"]),
        indices,field["residual_coordinate_index"],preparation_fraction=lam,precision=precision)
    original_total = tuple(tuple(_round(v,precision) for v in row) for row in _multiply(jet["F"],old))
    jacobian = _round(_determinant(original_total),precision)
    if jacobian.lo <= 0:
        raise ValueError("The actual local material source attachment has no positive full Cartesian Jacobian")
    response = NativeCarrierSourceSideResponse(field["point_id"],field["side"],
        field["original_parent_id"],field["original_material_id"],field["owner"],
        tuple(_fraction(v) for v in item["original_ray"]),field["original_label_membership"],
        field["archive_cohort_id"],I(0),jet["x"],point.position_bounds_m,
        jet["reference_position"],jet["y"],jet["delta_y"],jet["arm"],jet["Farm"],jet["F"],original_total,jacobian,
        transport,old,identity,jet["virtual_radial_driver"],jet["body_drivers"],
        json.dumps(native_archive_material_point_to_dict(point,system=carrier),sort_keys=True,separators=(",",":")))
    return response,jet


def _atom(atom,sides,items,geometry,q,lam,precision,carrier):
    key = atom["point_id"]
    sf,of = sides[key,"sub"],sides[key,"over"]
    sub,sj = _side(sf,items[key,"sub"],geometry[key,"sub"],q,lam,precision,carrier)
    over,oj = _side(of,items[key,"over"],geometry[key,"over"],q,lam,precision,carrier)
    frames = tuple(_vector(v) for v in atom["archival_old_reference"]["receiver_director_bounds"])
    jet = _local_source_jet(sj,oj,frames,tuple(of["owner_coordinate_indices"]),precision=precision)
    material = json.loads(atom["complete_original_atom_history_json"])["original_material"]
    law,columns = material["law"],material["columns"]
    kn,T,gamma = (_exact_float(law[n]) for n in ("normal_stiffness_pa_m","tensile_strength_pa","fracture_energy_j_m2"))
    C = (_exact_float(columns["sub_length_m"])/_exact_float(columns["sub_normal_modulus_pa"])
         +_exact_float(columns["over_length_m"])/_exact_float(columns["over_normal_modulus_pa"]))
    K = 1/(1/kn+C)
    if K != _fraction(atom["series_stiffness_exact_pa_m"][0]):
        raise ValueError("The authentic contact and source column compliance changed")
    anchor = _fraction(atom["literal_source_tractions_pa"][0])/K
    driver = _round(_dot(frames[0],_vsub(oj["preparation_driver"],sj["preparation_driver"])),precision)
    if driver.lo <= 0:
        raise ValueError("The whole straight preparation path is not strictly monotone opening")
    eta = jet["scalar_increment_bounds_m"][0]
    before_gap = _round(I(anchor)+eta-lam*driver,precision)
    current_gap = _round(I(anchor)+eta,precision)
    kwargs = dict(contact_stiffness_pa_m=kn,column_compliance_m_pa=C,
                  tensile_strength_pa=T,fracture_energy_j_m2=gamma,precision=precision)
    before = evaluate_native_cohesive_series_loading(before_gap,**kwargs)
    loading = evaluate_native_cohesive_series_loading(current_gap,**kwargs)
    initial = evaluate_native_cohesive_series_loading(I(anchor),**kwargs)
    area = _fraction(atom["original_area_m2"])
    literal = sum((_fraction(v) for v in atom["stored_source_energy_components_j"]),F(0))
    offset = _round(I(literal)-area*initial.loading_potential_density_bounds_j_m2,precision)
    energy = _round(offset+area*loading.loading_potential_density_bounds_j_m2,precision)
    recoverable = _round(offset+area*loading.recoverable_energy_density_bounds_j_m2,precision)
    fracture = _round(area*loading.fracture_work_density_bounds_j_m2,precision)
    active = jet["active_coordinate_indices"]
    B,H = jet["scalar_first_derivative_bounds"][0],jet["scalar_second_derivative_bounds"][0]
    tau,tangent = loading.traction_bounds_pa,loading.tangent_bounds_pa_m
    gradient = tuple(_round(area*tau*b,precision) for b in B)
    curvature = tuple(tuple(_round(area*tau*h,precision) for h in row) for row in H)
    hessian = tuple(tuple(_round(area*(tangent*B[i]*B[j]+tau*H[i][j]),precision)
                          for j in range(len(active))) for i in range(len(active)))
    force = tuple(_round(area*tau*v,precision) for v in frames[0])
    opposite = tuple(-v for v in force)
    Ps,Po = _outer(force,sub.internal_arm_bounds_m),_outer(opposite,over.internal_arm_bounds_m)
    ts,to = _cross(sj["y"],force),_cross(oj["y"],opposite)
    to = _vadd(to,_cross(_vsub(oj["y"],sj["y"]),force))
    moment = tuple(_round(v,precision) for v in _vadd(ts,to))
    power = []
    for index in active:
        value = I(0)
        for side,local,f,P,torque in ((sub,sj,force,Ps,ts),(over,oj,opposite,Po,to)):
            if index in local["owner_indices"]:
                value += torque[local["owner_indices"].index(index)]
            elif index in local["body_indices"]:
                j = local["body_indices"].index(index)
                value += _dot(f,local["mode_displacement_columns"][j])+_contract(P,local["mode_spatial_gradient_columns_per_m"][j])
            elif index == local["residual_index"]:
                value += _dot(f,local["virtual_radial_driver"])
        power.append(_round(value,precision))
    virtual = tuple(_round(g+w,precision) for g,w in zip(gradient,power))
    if any(not v.lo <= 0 <= v.hi for v in moment+virtual):
        raise ValueError("Complete material point/arm/director physical work fails closure")
    delta = _round(area*(loading.loading_potential_density_bounds_j_m2-before.loading_potential_density_bounds_j_m2),precision)
    delta_rec = _round(area*(loading.recoverable_energy_density_bounds_j_m2-before.recoverable_energy_density_bounds_j_m2),precision)
    delta_fracture = _round(area*(loading.fracture_work_density_bounds_j_m2-before.fracture_work_density_bounds_j_m2),precision)
    balance = _round(delta-delta_rec-delta_fracture,precision)
    if not balance.lo <= 0 <= balance.hi:
        raise ValueError("Authentic cohesive preparation work misses recoverable or fracture energy")
    state = json.loads(atom["source_local_state_json"])
    # Zero friction/cohesion/viscosity means shear is relaxed at every proposed
    # preparation point; free relative motion changes slip without heat.
    slip0,cumulative0 = _exact_float(state["plastic_slip_m"]),_exact_float(state["cumulative_slip_m"])
    jump = jet["scalar_increment_bounds_m"][1]
    slip = _round(I(slip0)+jump,precision)
    absjump = I(F(0) if jump.lo <= 0 <= jump.hi else min(abs(jump.lo),abs(jump.hi)),max(abs(jump.lo),abs(jump.hi)))
    shear_driver = _round(_dot(frames[1],_vsub(oj["preparation_driver"],sj["preparation_driver"])),precision)
    before_jump = jump-lam*shear_driver
    before_absolute = I(F(0) if before_jump.lo <= 0 <= before_jump.hi else min(abs(before_jump.lo),abs(before_jump.hi)),
                        max(abs(before_jump.lo),abs(before_jump.hi)))
    driver_absolute = I(F(0) if shear_driver.lo <= 0 <= shear_driver.hi else min(abs(shear_driver.lo),abs(shear_driver.hi)),
                        max(abs(shear_driver.lo),abs(shear_driver.hi)))
    cumulative = _round(I(cumulative0)+before_absolute+lam*driver_absolute,precision)
    return NativeCarrierSourceAtomResponse(key,atom["trace_id"],area,
        tuple(atom["original_source_work_lineage_id"]),sub,over,active,frames,
        jet["scalar_increment_bounds_m"],jet["scalar_first_derivative_bounds"],jet["scalar_second_derivative_bounds"],
        anchor,driver,before_gap,current_gap,loading,before,offset,energy,recoverable,fracture,
        gradient,hessian,curvature,force,opposite,Ps,Po,ts,to,
        tuple(_round(a+b,precision) for a,b in zip(force,opposite)),moment,virtual,
        delta,-delta,delta_rec,delta_fracture,balance,atom["source_local_state_json"],slip,cumulative)


def evaluate_native_common_carrier_source_work(system, *, time_fraction=F(1),
        preparation_fraction=F(1), coordinate_increments=None, branch=BRANCH,precision=96):
    seed,carrier = _registered(system)
    _precision(precision)
    if type(time_fraction) is not F or time_fraction != 1:
        raise ValueError("The retained COMMON fields and source ancestry authenticate only the exact journal endpoint")
    if type(preparation_fraction) is not F or not 0 <= preparation_fraction <= 1 or branch != BRANCH:
        raise ValueError("Only authentic monotone relaxed cohesive preparation is supported")
    q = system.bound_coordinate_increments if coordinate_increments is None else coordinate_increments
    if type(q) is not tuple or len(q) != 100 or any(type(v) is not F for v in q) or q[:92] != system.bound_coordinate_increments[:92]:
        raise ValueError("The first92 genuine NONZERO body values stay registered and fixed")
    if any(carrier.radius_m+r <= 0 for r in q[92:]):
        raise ValueError("Authentic internal source residuals preserve positive original virtual radial support")
    geometry = _geometry(system,time_fraction,precision)
    items = {(v["field"]["point_id"],v["field"]["side"]):v for v in seed["attachments"]}
    sides = {k:v["field"] for k,v in items.items()}
    atoms = tuple(_atom(a,sides,items,geometry,q,preparation_fraction,precision,carrier)
                  for a in seed["source_atoms"])
    G,H = [I(0)]*100,[[I(0) for _ in range(100)] for _ in range(100)]
    for atom in atoms:
        _scatter(G,H,atom.active_coordinate_indices,atom.potential_gradient_bounds,atom.potential_hessian_bounds)
    local = seed["local"]
    blocks = {b["source_coordinate_index"]:b for b in local["projection_blocks"]}
    beams = []
    for beam in seed["beams"]:
        b = blocks[beam["source_coordinate_index"]]
        index = system.coordinate_order.index(("source_internal",beam["trace_id"],beam["side"]))
        driver = _interval(b["projection_increment_bounds_m"])+q[index]
        K,reference = _fraction(beam["stiffness_exact_n_m"]),_fraction(beam["reference_displacement_m"])
        first = tuple(_interval(v) for v in b["projection_gradient_bounds"])+(I(1),)
        second = tuple(tuple(_interval(b["projection_hessian_bounds"][i][j]) if i<7 and j<7 else I(0)
                             for j in range(8)) for i in range(8))
        traction = K*(reference+driver)
        gradient = tuple(_round(traction*v,precision) for v in first)
        hessian = tuple(tuple(_round(K*first[i]*first[j]+traction*second[i][j],precision) for j in range(8)) for i in range(8))
        indices = tuple(b["global_coordinate_indices"])+(index,)
        energy = _round(K*(reference*driver+_square(driver)/2),precision)
        _scatter(G,H,indices,gradient,hessian)
        beams.append(dict(trace_id=beam["trace_id"],side=beam["side"],active_coordinate_indices=indices,
            driver_increment_bounds_m=driver,potential_increment_bounds_j=energy,
            potential_gradient_bounds=gradient,potential_hessian_bounds=hessian))
    remainder = I(0)
    for b in local["projection_blocks"]:
        remainder += _interval(b["orthogonal_energy_bounds_j"])
        _scatter(G,H,tuple(b["global_coordinate_indices"]),tuple(_interval(v) for v in b["orthogonal_gradient_bounds_n"]),
                 tuple(tuple(_interval(v) for v in row) for row in b["orthogonal_hessian_bounds_n_m"]))
    beam_energy = I(_fraction(local["source_beam_literal_reference_energy_j"]))+sum((b["potential_increment_bounds_j"] for b in beams),I(0))
    atom_energy = sum((a.loading_potential_bounds_j for a in atoms),I(0))
    recoverable = beam_energy+remainder+sum((a.recoverable_energy_bounds_j for a in atoms),I(0))
    fracture = sum((a.proposed_fracture_work_bounds_j for a in atoms),I(0))
    changes = tuple(_round(sum((getattr(a,n) for a in atoms),I(0)),precision) for n in (
        "preparation_potential_work_bounds_j","preparation_physical_reaction_work_bounds_j",
        "preparation_recoverable_energy_change_bounds_j","preparation_fracture_work_increment_bounds_j",
        "preparation_energy_balance_residual_bounds_j"))
    result = NativeCommonCarrierSourceWorkResponse(system.binding_sha256,time_fraction,preparation_fraction,
        system.coordinate_order,q,branch,atoms,tuple(beams),_round(beam_energy+atom_energy,precision),
        _round(remainder,precision),_round(beam_energy+atom_energy+remainder,precision),
        _round(recoverable,precision),_round(fracture,precision),tuple(_round(v,precision) for v in G),
        tuple(tuple(_round(v,precision) for v in row) for row in H),*changes,
        *(_fraction(local[n]) for n in ("source_contact_elapsed_time_s","source_memory_elapsed_time_s",
           "source_geometry_elapsed_time_s","material_geometry_elapsed_time_s")),precision,"")
    result = replace(result,binding_sha256=_digest(_body(result)))
    _RESPONSE_GUARDS[result.binding_sha256] = _typed_guard(result)
    return result


def native_common_carrier_source_work_system_to_dict(system):
    _registered(system)
    return dict(format=FORMAT,**_payload(system))


def native_common_carrier_source_work_system_from_dict(raw, *, local_source_packet,
        material_transport_packet, source_carrier_packet, archive_carrier):
    if type(raw) is not dict or raw.get("format") != FORMAT:
        raise ValueError("A complete source preparation system DTO is required")
    rebuilt = bind_native_common_carrier_source_work(local_source_packet,material_transport_packet,
        source_carrier_packet,archive_carrier,provenance=raw.get("provenance"),precision=raw.get("precision"))
    if _encoded(raw) != _encoded(native_common_carrier_source_work_system_to_dict(rebuilt)):
        raise ValueError("The preparation restart changes an authentic field, context or scalar kind")
    return rebuilt


def native_common_carrier_source_work_response_to_dict(response):
    if (type(response) is not NativeCommonCarrierSourceWorkResponse
            or _digest(_body(response)) != response.binding_sha256
            or _RESPONSE_GUARDS.get(response.binding_sha256) != _typed_guard(response)):
        raise ValueError("The complete prepared source response changed")
    return dict(format=RESPONSE_FORMAT,**_payload(response))


def native_common_carrier_source_work_response_from_dict(raw, *, system,
        expected_time_fraction, expected_preparation_fraction, expected_coordinate_increments,
        precision=96):
    if type(raw) is not dict or raw.get("format") != RESPONSE_FORMAT:
        raise ValueError("A complete prepared source response DTO is required")
    rebuilt = evaluate_native_common_carrier_source_work(system,time_fraction=expected_time_fraction,
        preparation_fraction=expected_preparation_fraction,coordinate_increments=expected_coordinate_increments,precision=precision)
    if _encoded(raw) != _encoded(native_common_carrier_source_work_response_to_dict(rebuilt)):
        raise ValueError("A source preparation response was transplanted or altered")
    return rebuilt
