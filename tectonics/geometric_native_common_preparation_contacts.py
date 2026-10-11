"""Actual cold depth-twist preparation of the original sixteen source atoms.

The active twojet has six fresh owner rotations, eight original internal
arms and the physical preparation fraction. Cold56 and HOT30 are fixed;
this is not a complete 100/101-coordinate mixed equilibrium operator.
Queries name an exact preparation point. Positive-width lambda intervals
are refused; temporal branch history and whole-path admission are open.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from fractions import Fraction as F
import hashlib
import json
from math import factorial

from .geometric_native_pose import FractionInterval as I
from .geometric_native_common_carrier_source_work import (
    _payload, _encoded, _digest, _typed_guard, _fraction, _interval,
    _vector, _matrix, _multiply, _inverse, _round, _packet,
    evaluate_native_cohesive_series_loading,
)
from .geometric_native_common_cold_depth_preparation import (
    _validate_bound, evaluate_native_common_cold_depth_preparation,
)
from .geometric_native_common_mixed_force import (
    _MFContactJet as J, _mf_jet_dot, _mf_jet_export, _mf_contact_scalar_twojet,
    _mf_roof_chart_bridge, _mf_cross, _mf_interval_dot, _mf_axis,
    _mf_normal_orientation_hyperstress,
)

FORMAT = "native-common-preparation-original-contact-system-1"
RESPONSE_FORMAT = "native-common-preparation-original-contact-response-1"
SCOPE = "ACTUAL_COLD_TWIST_ORIGINAL_SIXTEEN_CONTACT_ACTIVE_FIFTEEN_COORDINATE_TWOJET"
SOURCE_PINS = (
    ("907", "5e5555712e57fd69156ee475fe5feb161baef3cb96dd911a443228d160f53005", "bba7f1f4913d6bc47f5f41beff0a279be3c203993ddd0b0b06979c719abeda1a"),
    ("891", "b45d5d4d80c74c509bfac033c4a60c5b96c8327704e0c727fcf680406d82e57e", "1f1b07b977bbb092e08281d8d200a26f8bf9ab85e2a18f001b2061275f742c40"),
    ("895", "fbafd092a4eb98472e60f79665e1e0fe8e9e1c2c2b9d3679344a32e59593cc89", "27cbc05b687509fa22c00aa4339dd66d461c4def73ed37d4eae3acd62847da70"),
)
_SYSTEMS, _CONTEXTS, _RESPONSES, _EVALUATIONS = {}, {}, {}, {}
_POINT_BASE_FIELDS, _POINT_MAP_FIELDS = {}, {}
_ZERO = (I(0),)*3
_IDENTITY = tuple(tuple(I(int(i == j)) for j in range(3)) for i in range(3))


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _body(value):
    return {f.name: getattr(value, f.name) for f in fields(value) if f.name != "binding_sha256"}


def _remember(value, registry):
    value = replace(value, binding_sha256=_digest(_body(value)))
    registry[value.binding_sha256] = _typed_guard(value)
    return value


def _validate(value, cls, registry):
    if (type(value) is not cls or _digest(_body(value)) != value.binding_sha256
            or registry.get(value.binding_sha256) != _typed_guard(value)):
        raise ValueError("An altered or resealed preparation contact binding lacks its complete source")
    return value


def _precision(value):
    if type(value) is not int or not 96 <= value <= 512:
        raise ValueError("Preparation contact precision is an explicit integer in [96,512]")


def _fraction_parameter(value):
    value = I(value) if type(value) is F else value
    if type(value) is not I or not 0 <= value.lo <= value.hi <= 1:
        raise ValueError("The physical preparation fraction is an exact Fraction or singleton interval in [0,1]")
    if value.lo != value.hi:
        raise ValueError("The active preparation point-state twojet refuses positive-width preparation intervals")
    return value


def _literal(value):
    if type(value) is float:
        return F.from_float(value)
    if type(value) is int:
        return F(value)
    raise ValueError("A literal source material coefficient cannot change its numeric kind")


def _map(a, x):
    return tuple(_mf_interval_dot(row, x) for row in a)


def _cofactor(a):
    result = []
    for i in range(3):
        row = []
        for j in range(3):
            rr, cc = [k for k in range(3) if k != i], [k for k in range(3) if k != j]
            row.append((-1 if (i+j) % 2 else 1)*(a[rr[0]][cc[0]]*a[rr[1]][cc[1]]-a[rr[0]][cc[1]]*a[rr[1]][cc[0]]))
        result.append(tuple(row))
    return tuple(result)


def _source_law(atom):
    material = json.loads(atom["complete_original_atom_history_json"])["original_material"]
    law, columns = material["law"], material["columns"]
    kn, strength, gamma = (_literal(law[k]) for k in ("normal_stiffness_pa_m", "tensile_strength_pa", "fracture_energy_j_m2"))
    C = (_literal(columns["sub_length_m"])/_literal(columns["sub_normal_modulus_pa"])
        +_literal(columns["over_length_m"])/_literal(columns["over_normal_modulus_pa"]))
    K = 1/(1/kn+C)
    if (K != _fraction(atom["series_stiffness_exact_pa_m"][0]) or any(_literal(law[k]) != 0 for k in
            ("cohesion_pa", "friction_dry", "friction_wet", "viscosity_pa_s_m"))):
        raise ValueError("The complete original series law changed its stiffness or relaxed shear source")
    state = json.loads(atom["source_local_state_json"])
    tau = _fraction(atom["literal_source_tractions_pa"][0])
    if (tau >= 0 or _fraction(atom["literal_source_tractions_pa"][1]) != 0 or state["gap_m"] >= 0
            or any(state.get(k, 0.0) != 0.0 for k in ("damage", "max_opening_m"))):
        raise ValueError("A source contact cannot invent a recovered nonvirgin normal history")
    return material, kn, C, strength, gamma, K, tau/K


@dataclass(frozen=True, slots=True)
class NativeCommonPreparationContacts:
    coordinate_order: tuple
    original_coordinate_order: tuple
    active_original100_indices: tuple
    lambda_coordinate_index: int
    cold_coordinate_increments: tuple
    hot_coordinate_increments: tuple
    source907_raw_sha256: str
    source907_packet_sha256: str
    source891_raw_sha256: str
    source891_packet_sha256: str
    source895_raw_sha256: str
    source895_packet_sha256: str
    zero_contact_authority_binding_sha256: str
    cold_material_carrier_binding_sha256: str
    source_work_system_binding_sha256: str
    mixed_system_binding_sha256: str
    preparation_binding_sha256s: tuple
    preparation_reference_snapshot_sha256: str
    literal_source_energies_j: tuple
    source_contact_relaxation_lower_bound_j: F
    original_source_area_total_m2: F
    source_clocks_s: tuple
    provenance: str
    scope: str
    binding_sha256: str


def bind_native_common_preparation_contacts(source907, source891, preparations, *, source895, provenance):
    """Authenticate complete immutable source packets and the actual two owners.

    Complete packets are kept as immutable canonical JSON in the binding
    context. Restarts require those same whole authorities, never a selected
    geometry row substituted for the source packet.
    """
    if type(provenance) is not str or not provenance.strip():
        raise ValueError("Actual preparation contacts require explicit provenance")
    if type(preparations) is not tuple or len(preparations) != 2:
        raise ValueError("Both authentic owner preparations are required in owner order")
    preparations = tuple(_validate_bound(v) for v in preparations)
    if tuple(v.owner for v in preparations) != (0, 1):
        raise ValueError("The complete physical preparation owners cannot be reordered")
    packet_texts = []
    for packet, (_, rawpin, seal) in zip((source907, source891, source895), SOURCE_PINS):
        _packet(packet, seal)
        saved = _json(packet)
        if hashlib.sha256((saved+"\n").encode()).hexdigest() != rawpin:
            raise ValueError("The complete raw preparation source packet changed")
        packet_texts.append(saved)
    work = source891["carrier_source_work_system"]
    seed = json.loads(work["seed_json"])
    mixed = source907["mixed_force_system"]
    cold = source895["cold_material_carrier_system"]
    authorities = [row["authority"] for row in source907["contact_field_authorities"]
                   if row["label"] == "prepared_zero_cold_reference"]
    if len(authorities) != 1:
        raise ValueError("Exactly one complete frozen zero material authority is required")
    authority = authorities[0]
    if (_digest({k:v for k,v in authority.items() if k != "binding_sha256"}) != authority["binding_sha256"]
            or authority["cold_coordinate_increments"] != [[0,1]]*56
            or authority["prepared_zero_calibration_coordinates"] != [[0,1]]*56
            or authority["cold_carrier_binding_sha256"] != cold["binding_sha256"]
            or authority["source_work_component_binding_sha256"] != work["binding_sha256"]
            or authority["system_binding_sha256"] != mixed["binding_sha256"]
            or any(v.system_binding_sha256 != cold["binding_sha256"] for v in preparations)
            or any(v.reference_snapshot_sha256 != preparations[0].reference_snapshot_sha256 for v in preparations)):
        raise ValueError("The original complete zero authority changed its material/source association")
    clocks = tuple(F(x) for x in (0,2000000,1000000,1000000))
    if any(v.source_clocks_s != clocks for v in preparations):
        raise ValueError("Physical preparation cannot advance an original source clock")
    order = tuple(tuple(v) for v in mixed["coordinate_order"])
    if (len(order) != 100 or len(set(order)) != 100 or tuple(tuple(v) for v in work["coordinate_order"]) != tuple(tuple(v) for v in mixed["original_coordinate_order"])
            or mixed["source_work_system_binding_sha256"] != work["binding_sha256"]
            or mixed["cold_material_carrier_binding_sha256"] != cold["binding_sha256"]):
        raise ValueError("The complete COMMON100 catalogue/source graph changed")
    active = tuple(range(6))+tuple(range(92,100))
    active_order = tuple(order[i] for i in active)+(("preparation", "cold_depth_twist", "lambda"),)
    points = authority["point_receipts"]
    atoms = authority["original_atom_authorities"]
    if (len(points) != 32 or len(atoms) != 16
            or len({(v["point_id"],v["side"]) for v in points}) != 32
            or len({v["point_id"] for v in atoms}) != 16
            or sorted(_json(v["original_saved_attachment"]) for v in points) != sorted(_json(v) for v in seed["attachments"])
            or sorted(_json(v["original_atom"]) for v in atoms) != sorted(_json(v) for v in seed["source_atoms"])):
        raise ValueError("All sixteen original atoms and thirty-two original sides must occur once")
    counts = {"live":0, "archived":0}
    for point in points:
        cur = point["current"]
        if (point["current"] != point["prepared_zero"] or cur["bound_coordinate_increments"] != [[0,1]]*56
                or cur["system_binding_sha256"] != cold["binding_sha256"]
                or _digest({k:v for k,v in cur.items() if k not in ("format","binding_sha256")}) != cur["binding_sha256"]
                or cur["material_membership_authenticated"] is not True
                or _interval(cur["volume_jacobian_bounds"]).lo <= 0 or cur["allocation"] not in counts
                or cur["archive_debits"] != int(cur["allocation"] == "archived")
                or cur["material_depth_bounds_m"] != _payload(I(0))):
            raise ValueError("A frozen source point changed its authenticated complete material identity")
        counts[cur["allocation"]] += 1
    if counts != {"live":24, "archived":8}:
        raise ValueError("All twenty-four live and eight archived attachments must be retained")
    energies, cap, area_total = [], F(0), F(0)
    for row in atoms:
        atom = row["original_atom"]
        _, kn,C,T,gamma,K,anchor = _source_law(atom)
        area = _fraction(atom["original_area_m2"])
        if area <= 0:
            raise ValueError("Every original source atom requires its positive original measure")
        energies.append(sum((_fraction(v) for v in atom["stored_source_energy_components_j"]), F(0)))
        cap -= area*K*anchor*anchor/2
        area_total += area
    result = NativeCommonPreparationContacts(active_order,order,active,14,(F(0),)*56,(F(0),)*30,
        SOURCE_PINS[0][1],SOURCE_PINS[0][2],SOURCE_PINS[1][1],SOURCE_PINS[1][2],SOURCE_PINS[2][1],SOURCE_PINS[2][2],
        authority["binding_sha256"],cold["binding_sha256"],work["binding_sha256"],mixed["binding_sha256"],
        tuple(v.binding_sha256 for v in preparations),preparations[0].reference_snapshot_sha256,
        tuple(energies),cap,area_total,clocks,provenance,SCOPE,"")
    result = _remember(result,_SYSTEMS)
    # Canonical immutable copies prevent callers from mutating the authentic
    # source behind an existing bound object. Only compact actual geometry is
    # decoded for each query; the complete authorities remain retained.
    _CONTEXTS[result.binding_sha256] = dict(packet_texts=tuple(packet_texts),
        authority_json=_json(authority),banks_json=_json([{k:b[k] for k in
            ("original_parent_id","component_id","original_material_id","owner","radius_m","roof_radial_displacement_m")}
            for b in cold["banks"]]),preparations=preparations)
    return result


def validate_native_common_preparation_contacts(system):
    _validate(system,NativeCommonPreparationContacts,_SYSTEMS)
    if system.binding_sha256 not in _CONTEXTS:
        raise ValueError("The complete immutable preparation source context is missing")
    return system


def native_common_preparation_contacts_to_dict(system):
    return dict(format=FORMAT,**_payload(validate_native_common_preparation_contacts(system)))


def native_common_preparation_contacts_from_dict(raw, source907, source891, preparations, *, source895, provenance):
    system = bind_native_common_preparation_contacts(source907,source891,preparations,source895=source895,provenance=provenance)
    if _encoded(raw) != _encoded(native_common_preparation_contacts_to_dict(system)):
        raise ValueError("A complete preparation contact restart changed source parameters or provenance")
    return system



def _matrix_round_product(a,b,precision):
    return tuple(tuple(_round(sum((a[i][k]*b[k][j] for k in range(3)),I(0)),precision)
        for j in range(3)) for i in range(3))


def _matrix_round_map(a,x,precision):
    return tuple(_round(sum((a[i][j]*x[j] for j in range(3)),I(0)),precision) for i in range(3))


def _preparation_matrix_exponential(generator,*,precision):
    """Rigorous matrix exponential with an induced infinity-norm tail.

    The actual generator remains inside its outward interval matrix. Each
    power/addition encloses the corresponding exact Taylor expression.
    For b >= ||K||inf the omitted tail after term N is bounded by
    b**(N+1)/(N+1)!/(1-b/(N+2)). Exact power-of-two scaling and outward
    squaring retain exp(K)=exp(K/2**s)**(2**s).
    """
    _precision(precision)
    work=precision+32
    K=tuple(tuple(_round(v,work) for v in row) for row in generator)
    original=K
    b0=max(sum((max(abs(v.lo),abs(v.hi)) for v in row),F(0)) for row in K)
    b=b0;squarings=0
    while b>F(1,2):
        K=tuple(tuple(v/2 for v in row) for row in K)
        b/=2;squarings+=1
    target=F(1,10**(precision+8))
    budget=max(128,precision+64)
    term,total=_IDENTITY,_IDENTITY
    for n in range(1,budget+1):
        term=tuple(tuple(_round(v/n,work) for v in row)
            for row in _matrix_round_product(term,K,work))
        total=tuple(tuple(_round(total[i][j]+term[i][j],work) for j in range(3)) for i in range(3))
        ratio=b/F(n+2)
        if ratio>=1:
            continue
        exact_tail=(b**(n+1)/factorial(n+1))/(1-ratio)
        tail=_round(I(exact_tail),work).hi
        if tail<=target:
            break
    else:
        raise ValueError("The rigorous point rotation exhausted its declared Taylor budget")
    total=tuple(tuple(_round(v+I(-tail,tail),work) for v in row) for row in total)
    for _ in range(squarings):
        total=_matrix_round_product(total,total,work)
    result=tuple(tuple(_round(v,precision) for v in row) for row in total)
    certificate=dict(method="OUTWARD_INTERVAL_MATRIX_TAYLOR_SCALING_SQUARING",
        unscaled_generator_bounds=original,unscaled_generator_infinity_norm_upper=b0,
        scaled_generator_infinity_norm_upper=b,scaling_squaring_count=squarings,
        taylor_term_count=n,tail_scaled_matrix_upper=tail,tail_ratio=ratio,
        tail_target_upper=target,maximum_taylor_terms=budget,
        output_precision_decimal_digits=precision,work_precision_decimal_digits=work,
        guard_precision_decimal_digits=32,tail_theorem="b^(N+1)/(N+1)!/(1-b/(N+2))",
        tail_bound_outward_compressed=True,midpoint_inputs_used=False,
        rigorous_outward_rational_matrix_arithmetic=True)
    return result,certificate


@dataclass(frozen=True,slots=True)
class _PreparationPointFields:
    preparation_fraction_bounds: I
    physical_volume_depth_bounds_m: I
    physical_volume_depth_gradient_bounds: tuple
    rotation_bounds: tuple
    position_bounds_m: tuple
    spatial_gradient_bounds: tuple
    preparation_velocity_bounds_m: tuple
    preparation_acceleration_bounds_m: tuple
    preparation_spatial_gradient_rate_bounds: tuple
    preparation_spatial_gradient_second_bounds: tuple
    original_cartesian_gradient_bounds: tuple
    preparation_original_gradient_rate_bounds: tuple
    preparation_original_gradient_second_bounds: tuple


def _preparation_point_fields(preparation,x,A0,lam,point_binding,*,precision):
    """Actual frozen-material point geometry, independent of the arm residual."""
    preparation=_validate_bound(preparation)
    basekey=(point_binding,preparation.binding_sha256,precision)
    key=basekey+(lam.lo,)
    if key in _POINT_MAP_FIELDS:
        return _POINT_MAP_FIELDS[key]
    base=_POINT_BASE_FIELDS.get(basekey)
    if base is None:
        base=evaluate_native_common_cold_depth_preparation(preparation,x,preparation_fraction=F(0),
            original_gradient_bounds=A0,precision=precision)
        _POINT_BASE_FIELDS[basekey]=base
    work=precision+32
    omega=tuple(_round(I(v),work) for v in preparation.declaration.angular_rate_per_physical_volume_depth_m)
    W=tuple(tuple(_mf_cross(omega,_mf_axis(j))[i] for j in range(3)) for i in range(3))
    tau,g=base.physical_volume_depth_bounds_m,base.physical_volume_depth_gradient_bounds
    wx=_matrix_round_map(W,x,work)
    W2=_matrix_round_product(W,W,work)
    d=tuple(tuple(_round(wx[i]*g[j],work) for j in range(3)) for i in range(3))
    K=tuple(tuple(_round(lam*tau*W[i][j],work) for j in range(3)) for i in range(3))
    Q,certificate=_preparation_matrix_exponential(K,precision=precision)
    Z=tuple(tuple(_round(_IDENTITY[i][j]+lam*d[i][j],work) for j in range(3)) for i in range(3))
    DT=_matrix_round_product(Q,Z,precision)
    WZ,W2Z,Wd=(_matrix_round_product(W,Z,work),_matrix_round_product(W2,Z,work),
        _matrix_round_product(W,d,work))
    first0=tuple(tuple(_round(tau*WZ[i][j]+d[i][j],work) for j in range(3)) for i in range(3))
    second0=tuple(tuple(_round(tau*tau*W2Z[i][j]+2*tau*Wd[i][j],work) for j in range(3)) for i in range(3))
    DT1,DT2=_matrix_round_product(Q,first0,precision),_matrix_round_product(Q,second0,precision)
    position=_matrix_round_map(Q,x,precision)
    velocity=tuple(_round(tau*v,precision) for v in _matrix_round_map(Q,wx,work))
    acceleration=tuple(_round(tau*tau*v,precision)
        for v in _matrix_round_map(Q,_matrix_round_map(W2,x,work),work))
    result=_PreparationPointFields(lam,tau,g,Q,position,DT,velocity,acceleration,DT1,DT2,
        _matrix_round_product(DT,A0,precision),_matrix_round_product(DT1,A0,precision),
        _matrix_round_product(DT2,A0,precision))
    certificate.update(preparation_fraction_exact=lam.lo,
        original_material_point_binding_sha256=point_binding,
        preparation_binding_sha256=preparation.binding_sha256)
    # No caller-visible mutable certificate aliases a cached geometry proof.
    certificate_json=_json(_payload(certificate))
    _POINT_MAP_FIELDS[key]=(result,certificate_json)
    return result,certificate_json

def _geometry_side(point, bank, preparation, lam, internal, *, precision):
    cur = point["current"]
    item, field = point["original_saved_attachment"],point["original_saved_attachment"]["field"]
    x,A0 = _vector(cur["position_bounds_m"]),_matrix(cur["original_cartesian_gradient_bounds"])
    B,Qold = _matrix(item["old_reference_gradient"]),_matrix(item["material_owner_pose"])
    roofC,bridge = _mf_roof_chart_bridge(tuple(_fraction(v) for v in item["original_ray"]),
        _fraction(bank["radius_m"]),_fraction(bank["roof_radial_displacement_m"]),precision=precision)
    Aold = tuple(tuple(_round(v,precision) for v in row) for row in _multiply(_multiply(B,Qold),roofC))
    invold = _inverse(Aold,precision)
    identity = _multiply(Aold,invold)
    if any(not (identity[i][j]-int(i==j)).lo <= 0 <= (identity[i][j]-int(i==j)).hi for i in range(3) for j in range(3)):
        raise ValueError("The complete original material bridge lacks its inverse certificate")
    mapfield,rotation_certificate_json = _preparation_point_fields(preparation,x,A0,lam,
        cur["binding_sha256"],precision=precision)
    omega = tuple(I(v) for v in preparation.declaration.angular_rate_per_physical_volume_depth_m)
    W = tuple(tuple(_mf_cross(omega,_mf_axis(j))[i] for j in range(3)) for i in range(3))
    wx = _mf_cross(omega,x)
    d = tuple(tuple(_round(wx[i]*mapfield.physical_volume_depth_gradient_bounds[j],precision) for j in range(3)) for i in range(3))
    tau = mapfield.physical_volume_depth_bounds_m
    D2 = mapfield.preparation_spatial_gradient_second_bounds
    l0 = tuple(a-b for a,b in zip(_vector(field["source_reference_position_bounds_m"]),_vector(field["material_reference_position_bounds_m"])))
    v = _vector(field["source_virtual_radial_direction_bounds"])
    oldglobal = field["residual_coordinate_index"]
    index = oldglobal-92+6
    if not 6 <= index < 14:
        raise ValueError("The source arm lost its authentic active internal coordinate")
    T0 = _multiply(A0,invold)
    arm0,armv = _map(T0,l0),_map(T0,v)
    arm = tuple(a+internal*b for a,b in zip(arm0,armv))
    DT,DT1 = mapfield.spatial_gradient_bounds,mapfield.preparation_spatial_gradient_rate_bounds
    pos = tuple(a+b for a,b in zip(mapfield.position_bounds_m,_map(DT,arm)))
    first = tuple(a+b for a,b in zip(mapfield.preparation_velocity_bounds_m,_map(DT1,arm)))
    second = tuple(a+b for a,b in zip(mapfield.preparation_acceleration_bounds_m,_map(D2,arm)))
    irate,icross = _map(DT,armv),_map(DT1,armv)
    Y = tuple(J(pos[i],{14:first[i],index:irate[i]}, {(14,14):second[i],(index,14):icross[i]},precision) for i in range(3))
    receipt = dict(point_id=point["point_id"],side=point["side"],owner=cur["owner"],
        original_parent_id=cur["original_parent_id"],original_material_id=cur["original_material_id"],
        allocation=cur["allocation"],archive_debits=cur["archive_debits"],
        original_material_point_binding_sha256=cur["binding_sha256"],
        complete_original_source_input_frame_bridge=bridge,old_original_cartesian_gradient_bounds=_payload(Aold),
        old_original_cartesian_inverse_bounds=_payload(invold),complete_old_material_inverse_identity_bounds=_payload(identity),
        original_internal_arm_bounds_m=_payload(l0),original_virtual_radial_direction_bounds=_payload(v),
        original_internal_coordinate_index=oldglobal,active_internal_coordinate_index=index,
        internal_coordinate_increment_m=_payload(internal),original_position_bounds_m=_payload(x),
        original_cartesian_gradient_bounds=_payload(A0),preparation_map_position_bounds_m=_payload(mapfield.position_bounds_m),
        preparation_map_spatial_gradient_bounds=_payload(DT),preparation_map_spatial_gradient_first_bounds=_payload(DT1),
        preparation_map_spatial_gradient_second_bounds=_payload(D2),preparation_rotation_bounds=_payload(mapfield.rotation_bounds),
        rotation_certificate=json.loads(rotation_certificate_json),
        physical_volume_depth_bounds_m=_payload(tau),physical_volume_depth_gradient_bounds=_payload(mapfield.physical_volume_depth_gradient_bounds),
        rank_one_twist_gradient_bounds=_payload(d),prepared_original_gradient_bounds=_payload(mapfield.original_cartesian_gradient_bounds),
        prepared_original_gradient_first_bounds=_payload(mapfield.preparation_original_gradient_rate_bounds),
        prepared_original_gradient_second_bounds=_payload(mapfield.preparation_original_gradient_second_bounds),
        attachment_position_twojet=tuple(_mf_jet_export(v) for v in Y),
        original_measure_not_current_nanson_measure_used=True,map_jacobian_exact=_payload(F(1)))
    return Y,receipt,(A0,mapfield,d,W,tau)


def _normal_twojet(data,material_normal,precision):
    A0,mapfield,d,W,tau = data
    w0 = _map(_cofactor(A0),material_normal)
    dtw0 = tuple(sum((d[j][i]*w0[j] for j in range(3)),I(0)) for i in range(3))
    lam = mapfield.preparation_fraction_bounds
    z = tuple(a-lam*b for a,b in zip(w0,dtw0))
    first0 = tuple(tau*a-b for a,b in zip(_map(W,z),dtw0))
    second0 = tuple(tau*tau*a-2*tau*b for a,b in zip(_map(_multiply(W,W),z),_map(W,dtw0)))
    w,w1,w2 = _map(mapfield.rotation_bounds,z),_map(mapfield.rotation_bounds,first0),_map(mapfield.rotation_bounds,second0)
    jets = tuple(J(w[i],{14:w1[i]},{(14,14):w2[i]},precision) for i in range(3))
    norm2 = _mf_jet_dot(jets,jets)
    norm = norm2.sqrt()
    normal = tuple(v/norm for v in jets)
    return normal,dict(original_material_normal_bounds=_payload(material_normal),
        original_prepared_cofactor_normal_bounds=_payload(w0),transported_cofactor_normal_twojet=tuple(_mf_jet_export(v) for v in jets),
        transported_cofactor_normal_norm_squared_twojet=_mf_jet_export(norm2),
        transported_cofactor_normal_norm_bounds=_payload(norm.value),unit_normal_twojet=tuple(_mf_jet_export(v) for v in normal),
        whole_box_normal_norm_strictly_positive=True,second_preparation_original_gradient_curvature_included=True,
        map_cofactor_formula="Q_lambda*(I-lambda*d_transpose)*cofactor(A0)*Nsource",
        current_area_energy_multiplier_added=False)


@dataclass(frozen=True, slots=True)
class NativeCommonPreparationContactsResponse:
    system_binding_sha256: str
    coordinate_order: tuple
    active_original100_indices: tuple
    lambda_coordinate_index: int
    preparation_fraction_bounds: I
    internal_coordinates: tuple
    energy_bounds_j: I
    recoverable_energy_bounds_j: I
    fracture_work_bounds_j: I
    gradient_bounds: tuple
    hessian_bounds: tuple
    atoms_json: tuple
    point_receipts_json: tuple
    simultaneous_owner_rotation_directional_first_derivative_bounds: tuple
    source_contact_relaxation_lower_bound_j: F
    original_atom_count: int
    original_material_point_count: int
    original_live_attachment_count: int
    original_archived_attachment_count: int
    precision: int
    scope: str
    full_general_state_mixed_operator_certified: bool
    new_mixed_force_equilibrium_certified: bool
    reference_preparation_work_paid: bool
    full_system_preparation_work_certified: bool
    historical_normal_fracture_memory_recovered: bool
    proposed_normal_damage_or_fracture_work_applied: bool
    full_system_temporal_path_work_certified: bool
    clocks_advanced: bool
    accepted_new_native_intervals: int
    binding_sha256: str


def evaluate_native_common_preparation_contacts(system,preparation_fraction=F(0),internal_coordinates=None,*,precision=96):
    system = validate_native_common_preparation_contacts(system)
    _precision(precision)
    lam = _fraction_parameter(preparation_fraction)
    internal = (F(0),)*8 if internal_coordinates is None else internal_coordinates
    if type(internal) is not tuple or len(internal) != 8 or any(type(v) is not F for v in internal):
        raise ValueError("All eight authentic internal arm coordinates are literal Fractions")
    query_key = (system.binding_sha256,lam.lo,lam.hi,internal,precision)
    if query_key in _EVALUATIONS:
        return _validate(_EVALUATIONS[query_key],NativeCommonPreparationContactsResponse,_RESPONSES)
    context = _CONTEXTS[system.binding_sha256]
    authority = json.loads(context["authority_json"])
    banks = {v["original_parent_id"]:v for v in json.loads(context["banks_json"])}
    points = {(v["point_id"],v["side"]):v for v in authority["point_receipts"]}
    side_cache,zero_cache = {},{}
    for point in points.values():
        field=point["original_saved_attachment"]["field"]
        if _fraction(banks[field["original_parent_id"]]["radius_m"])+internal[field["residual_coordinate_index"]-92] <= 0:
            raise ValueError("An original internal material arm must retain positive radius plus residual")
    for name,point in points.items():
        field = point["original_saved_attachment"]["field"]
        index = field["residual_coordinate_index"]-92
        args = (point,banks[field["original_parent_id"]],context["preparations"][field["owner"]])
        side_cache[name] = _geometry_side(*args,lam,internal[index],precision=precision)
        zero_cache[name] = side_cache[name] if lam == I(0) and internal[index] == 0 else _geometry_side(*args,I(0),F(0),precision=precision)
    G,H = [I(0)]*15,[[I(0)]*15 for _ in range(15)]
    E,stored,fracture = I(0),I(0),I(0)
    rows = []
    exact_zero = lam == I(0) and not any(internal)
    for original in authority["original_atom_authorities"]:
        atom,aid = original["original_atom"],original["point_id"]
        sy,sreceipt,_ = side_cache[aid,"sub"]
        oy,oreceipt,odata = side_cache[aid,"over"]
        N = _vector(original["original_receiver_material_normal_bounds"])
        normal,nreceipt = _normal_twojet(odata,N,precision)
        s0,_,_ = zero_cache[aid,"sub"]
        o0,_,od0 = zero_cache[aid,"over"]
        n0 = normal if exact_zero else _normal_twojet(od0,N,precision)[0]
        baseline = _mf_interval_dot(tuple(v.value for v in n0),tuple(b.value-a.value for a,b in zip(s0,o0)))
        sf,of = points[aid,"sub"]["original_saved_attachment"]["field"],points[aid,"over"]["original_saved_attachment"]["field"]
        subowner,overowner = tuple(sf["owner_coordinate_indices"]),tuple(of["owner_coordinate_indices"])
        scalar = _mf_contact_scalar_twojet(sy,oy,normal,subowner,overowner,baseline,precision=precision)
        if exact_zero:
            scalar = J(I(0),scalar.gradient,scalar.hessian,precision)
        material,kn,C,T,gamma,K,anchor = _source_law(atom)
        lawargs = dict(contact_stiffness_pa_m=kn,column_compliance_m_pa=C,tensile_strength_pa=T,fracture_energy_j_m2=gamma,precision=precision)
        loading = evaluate_native_cohesive_series_loading(I(anchor)+scalar.value,**lawargs)
        initial = evaluate_native_cohesive_series_loading(I(anchor),**lawargs)
        area = _fraction(atom["original_area_m2"])
        literal = sum((_fraction(v) for v in atom["stored_source_energy_components_j"]),F(0))
        offset = I(literal)-area*initial.loading_potential_density_bounds_j_m2
        energy = offset+area*loading.loading_potential_density_bounds_j_m2
        recoverable = offset+area*loading.recoverable_energy_density_bounds_j_m2
        dissipated = area*loading.fracture_work_density_bounds_j_m2
        if exact_zero:
            energy=recoverable=I(literal);dissipated=I(0)
        ej = scalar.compose(energy,area*loading.traction_bounds_pa,area*loading.tangent_bounds_pa_m)
        D = tuple(b.value-a.value for a,b in zip(sy,oy))
        n = tuple(v.value for v in normal)
        traction = loading.traction_bounds_pa
        force_over = tuple(-area*traction*v for v in n)
        force_sub = tuple(-v for v in force_over)
        couple = tuple(-area*traction*v for v in _mf_cross(n,D))
        moment = tuple(a+b+c for a,b,c in zip(_mf_cross(tuple(v.value for v in sy),force_sub),_mf_cross(tuple(v.value for v in oy),force_over),couple))
        if any(not v.lo <= 0 <= v.hi for v in moment):
            raise ValueError("Actual preparation attachment force/director couple lost moment closure")
        virtual = []
        for i in range(15):
            if i < 6:
                axisindex = subowner.index(i) if i in subowner else overowner.index(i) if i in overowner else None
                if axisindex is None:
                    translation=orientation=I(0)
                else:
                    axis=_mf_axis(axisindex)
                    sv=_mf_cross(axis,tuple(v.value for v in sy)) if i in subowner else _ZERO
                    ov=_mf_cross(axis,tuple(v.value for v in oy)) if i in overowner else _ZERO
                    translation=_mf_interval_dot(n,tuple(b-a for a,b in zip(sv,ov)))
                    orientation=_mf_interval_dot(_mf_cross(axis,n),D) if i in overowner else I(0)
            else:
                translation=_mf_interval_dot(n,tuple(b.gradient.get(i,I(0))-a.gradient.get(i,I(0)) for a,b in zip(sy,oy)))
                orientation=_mf_interval_dot(tuple(v.gradient.get(i,I(0)) for v in normal),D)
            pw,ow=area*traction*translation,area*traction*orientation
            residual=ej.gradient.get(i,I(0))-pw-ow
            if not residual.lo <= 0 <= residual.hi:
                raise ValueError("Actual preparation omitted material attachment or normal orientation work")
            virtual.append(dict(active_coordinate_index=i,material_attachment_work_bounds=_payload(_round(pw,precision)),
                nanson_orientation_work_bounds=_payload(_round(ow,precision)),residual_bounds=_payload(_round(residual,precision))))
        for i in range(15):
            G[i]+=ej.gradient.get(i,I(0))
            for j in range(15):H[i][j]+=ej.hessian.get((min(i,j),max(i,j)),I(0))
        E+=energy;stored+=recoverable;fracture+=dissipated
        rows.append(_json(dict(point_id=aid,trace_id=atom["trace_id"],original_atom=atom,
            original_area_m2=_payload(area),original_material_law=material,
            source_free_gap_anchor_m=_payload(anchor),literal_source_energy_j=_payload(literal),literal_energy_offset_bounds_j=_payload(offset),
            scalar_increment_twojet=_mf_jet_export(scalar),loading=_payload(loading),
            energy_twojet=_mf_jet_export(ej),energy_bounds_j=_payload(_round(energy,precision)),
            recoverable_energy_bounds_j=_payload(_round(recoverable,precision)),fracture_work_bounds_j=_payload(_round(dissipated,precision)),
            sub=sreceipt,over=oreceipt,receiver_nanson_normal=nreceipt,
            prepared_zero_attachment_scalar_bounds_m=_payload(_round(baseline,precision)),
            force_on_sub_bounds_n=_payload(tuple(_round(v,precision) for v in force_sub)),
            force_on_over_bounds_n=_payload(tuple(_round(v,precision) for v in force_over)),
            receiver_normal_orientation_couple_bounds_nm=_payload(tuple(_round(v,precision) for v in couple)),
            global_moment_residual_bounds_nm=_payload(tuple(_round(v,precision) for v in moment)),
            virtual_work=virtual,source_original_area_included_once=True,current_area_energy_multiplier_added=False,
            exact_zero_literal_energy_identity_preserved=exact_zero,
            original_material_nanson_follower_contact_component_defined=True,
            simultaneous_owner_rotation_objective_by_relative_rotation_identity=True,
            proposed_normal_damage_or_fracture_work_applied=False,physical_material_admission_authorized=False)))
    result = NativeCommonPreparationContactsResponse(system.binding_sha256,system.coordinate_order,
        system.active_original100_indices,14,lam,internal,_round(E,precision),_round(stored,precision),_round(fracture,precision),
        tuple(_round(v,precision) for v in G),tuple(tuple(_round(v,precision) for v in row) for row in H),
        tuple(rows),tuple(_json(v) for v in authority["point_receipts"]),_ZERO,
        system.source_contact_relaxation_lower_bound_j,16,32,24,8,precision,SCOPE,
        False,False,False,False,False,False,False,False,0,"")
    result = _remember(result,_RESPONSES)
    _EVALUATIONS[query_key] = result
    return result


def native_common_preparation_contacts_response_to_dict(response):
    return dict(format=RESPONSE_FORMAT,**_payload(_validate(response,NativeCommonPreparationContactsResponse,_RESPONSES)))


def native_common_preparation_contacts_response_from_dict(raw,*,system,expected_preparation_fraction,expected_internal_coordinates=None,precision=96):
    rebuilt = evaluate_native_common_preparation_contacts(system,expected_preparation_fraction,expected_internal_coordinates,precision=precision)
    if _encoded(raw) != _encoded(native_common_preparation_contacts_response_to_dict(rebuilt)):
        raise ValueError("A preparation contact response restart changed its complete geometry/source/branch query")
    return rebuilt
