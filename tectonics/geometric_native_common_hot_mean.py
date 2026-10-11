"""Whole authentic HOT material and its finite actual-base mean chart.

The five source masks are retained without an area floor.  This is a NEW
prepared reference, associated with the fixed owner scales of cold component
899.  It establishes constitutive components and an implicit chart; it does
not advance a clock, solve mixed equilibrium, or certify material admission.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from fractions import Fraction as F
import hashlib
import json

from .geometric_native_pose import FractionInterval as I, interval_sqrt, certify_rotation_history
from .geometric_native_spherical_solid import interval_cbrt
from .geometric_native_common_carrier_source_work import (
    _payload, _encoded, _digest, _typed_guard, _fraction, _interval, _packet, _round,
)
from .geometric_intrinsic_geometry import rotation_increment_from_dict
from .geometric_native_joint_spherical_field import rotate_joint_physical_depth
from .geometric_native_polygons import native_polygon_from_dict
from .geometric_native_affine_body import _normalized_native_vertex_bounds, _unit_cone_bounds

FORMAT = "native-common-hot-finite-mean-system-1"
RESPONSE_FORMAT = "native-common-hot-finite-mean-response-1"
SCOPE = "DECLARED_COMPLETE_HOT_FINITE_BASE_MEAN_ENDPOINT_ONLY"
REFERENCE = "NEW_FIXED_COLD895_OWNER_SCALE_SOURCE_WORLD_COMMON879_DEPTH_HOT_REFERENCE"
MEASURE = "AUTHENTIC_ORIGINAL_R_SQUARED_DOMEGA_DZ_ONCE"
BASAL_LAW = "NEW_OBJECTIVE_FINITE_RADIAL_BASE_DISPLACEMENT_WITH_FULL_VARIANCE"
TRANSPORT_SEAL = "368a524a954ac5f2ba3a6b92c182d34d1c3567ba7df1314c0afd02b2cff2c23e"
WORK_SEAL = "1f1b07b977bbb092e08281d8d200a26f8bf9ab85e2a18f001b2061275f742c40"
COLD_SEAL = "c863be05c9d641025666a1a4c3929701c7d472f9d574085926d64a2d08b4c439"
COMMON_HOT_SEAL = "60259e6e56bb7335fa994988adfbbac5ac933b524ca164b0298c2e0a16a40446"
NORMAL_SEAL = "e2db27aa164161482117993f4f23010293eae45100c91d3a52ff0508075688ee"
CURVED_SEAL = "f192cb4a9fbdcfd83ab4bad69e1c5c12307bfe4df78626aa514deb3d4f84cccd"
AFFINE_SEAL = "afbc10eb17cdc173177a32507be37f77f936827fe83e64764165ae702f892697"
_SYSTEMS, _RESPONSES, _EVALUATIONS, _BODY_GUARDS = {}, {}, {}, {}
_IDENTITY = tuple(tuple(F(i == j) for j in range(3)) for i in range(3))
GENERATORS = (
    _IDENTITY,
    ((F(1), F(0), F(0)), (F(0), F(-1), F(0)), (F(0), F(0), F(0))),
    ((F(1), F(0), F(0)), (F(0), F(1), F(0)), (F(0), F(0), F(-2))),
    ((F(0), F(1), F(0)), (F(1), F(0), F(0)), (F(0), F(0), F(0))),
    ((F(0), F(0), F(1)), (F(0), F(0), F(0)), (F(1), F(0), F(0))),
    ((F(0), F(0), F(0)), (F(0), F(0), F(1)), (F(0), F(1), F(0))),
)
COORDINATE_NAMES = ("actual_base_mean", "deviatoric_xx_minus_yy", "deviatoric_xx_plus_yy_minus_2zz", "xy", "xz", "yz")
_C = ((1,1,1,0,0,0),(1,-1,1,0,0,0),(1,0,-2,0,0,0),
      (0,0,0,1,0,0),(0,0,0,0,1,0),(0,0,0,0,0,1))


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _body(value):
    return {f.name:getattr(value, f.name) for f in fields(value) if f.name != "binding_sha256"}


def _remember(value, registry):
    value = replace(value, binding_sha256=_digest(_body(value)))
    registry[value.binding_sha256] = _typed_guard(value)
    return value


def _precision(value):
    if type(value) is not int or not 96 <= value <= 512:
        raise ValueError("HOT precision must be a literal integer in [96,512]")
    return value


def _validate(value, cls, registry):
    if (type(value) is not cls or registry.get(value.binding_sha256) != _typed_guard(value)
            or _digest(_body(value)) != value.binding_sha256):
        raise ValueError("A HOT component changed its complete typed source binding")
    return value


def _dot(a,b):
    return sum((x*y for x,y in zip(a,b)), I(0))


def _mv(a,b):
    return tuple(_dot(row,b) for row in a)


def _mm(a,b):
    return tuple(tuple(sum((a[i][k]*b[k][j] for k in range(len(b))), I(0))
                       for j in range(len(b[0]))) for i in range(len(a)))


def _tr(a):
    return tuple(zip(*a))


def _sym(a):
    return tuple(tuple((a[i][j]+a[j][i])/2 for j in range(3)) for i in range(3))


def _square(x):
    x = x if type(x) is I else I(x)
    return I(F(0) if x.lo <= 0 <= x.hi else min(x.lo*x.lo,x.hi*x.hi),
             max(x.lo*x.lo,x.hi*x.hi))


def _hull(rows):
    return tuple(I(min(row[j].lo for row in rows),max(row[j].hi for row in rows)) for j in range(3))


def _intersection(a,b):
    low,high = max(a.lo,b.lo),min(a.hi,b.hi)
    if low > high:
        raise ValueError("Independent rigorous HOT enclosures are inconsistent")
    return I(low,high)


def _literal_float(value):
    if type(value) not in (float,int) or type(value) is bool:
        raise ValueError("Source material scalars retain their literal finite numeric kind")
    return F.from_float(value) if type(value) is float else F(value)


@dataclass(frozen=True, slots=True)
class NativeCommonHotMeanBody:
    component_id: str
    material_id: str
    original_parent_id: str
    owner: int
    radius_m: F
    areal_material_thickness_m: F
    original_base_radius_bounds_m: I
    prepared_base_radius_bounds_m: I
    fixed_owner_scale: F
    owner_pose_bounds: tuple
    angular_rate_per_physical_volume_depth_m: tuple
    actual_area_bounds_sr: I
    original_reference_volume_bounds_m3: I
    prepared_reference_volume_bounds_m3: I
    young_modulus_pa: F
    shear_modulus_pa: F
    lame_modulus_pa: F
    density_kg_m3: F
    gravity_m_s2: F
    source_birth_radial_displacement_m: F
    mask_cells: tuple
    complete_common_bank_json: str
    complete_original_common_hot_component_json: str
    complete_original_birth_json: str
    complete_foundation_source_json: str
    complete_conditional_old19_step_json: str
    conditional_old19_coordinate_decimal: str
    authentic_old100_coordinate_indices: tuple
    authentic_old100_coordinate_values: tuple
    saved_nonzero_probe_old_cartesian_amplitudes_m: tuple
    source_bank_snapshot_sha256: str
    source_snapshot_sha256: str
    source_support_sha256: str
    source_epoch_sha256: str
    conditional_old19_coordinate_index: int = -1
    conditional_old19_radial_displacement_literal_json: str = ""


@dataclass(frozen=True, slots=True)
class NativeCommonHotMeanSystem:
    bodies: tuple
    coordinate_order: tuple
    bound_coordinate_increments: tuple
    complete_original_100_coordinate_order: tuple
    complete_original_100_coordinate_values: tuple
    complete_original_material_state_json: str
    complete_original_source_clock_work_history_json: str
    complete_source_normal_material_json: str
    complete_source_foundation_interface_support_json: str
    complete_source_hot_face_traces_json: str
    complete_saved_joint_reference_field_json: str
    complete_current_load_json: str
    cold_continuum_component_binding_sha256: str
    cold_continuum_response_bindings_sha256: tuple
    cold_material_carrier_binding_sha256: str
    source_packet_seals: tuple
    precision: int
    provenance: str
    scope: str
    reference_model: str
    material_reference_measure: str
    basal_law: str
    original_point_1_completed: bool
    accepted_original_native_intervals: int
    accepted_new_native_intervals: int
    source_geometry_elapsed_time_s: F
    contact_geometry_elapsed_time_s: F
    memory_elapsed_time_s: F
    material_geometry_elapsed_time_s: F
    clocks_advanced: bool
    physical_source_force_prestress_work_handover_certified: bool
    historical_archived_three_dimensional_prestress_recovered: bool
    prepared_reference_full_system_work_certified: bool
    physical_material_admission_authorized: bool
    whole_clock_all_pair_admission_authorized: bool
    full_system_temporal_path_work_certified: bool
    new_mixed_force_equilibrium_certified: bool
    finite_mean_equilibrium_certified: bool
    hot_constitutive_law_defined: bool
    finite_mean_chart_defined: bool
    binding_sha256: str


def _mask_cells(bank, owner_pose, precision):
    """Range of every complete source polygon, and each positive correction once.

    Native polygons are actual disjoint reference pieces.  A correction's
    containing chart bounds its integrand, whereas its own delta area is the
    material measure.  Chart area is never assigned to curved material.
    """
    mask = json.loads(bank["complete_mask_source_json"])
    rate = tuple(_fraction(x) for x in bank["angular_rate_per_physical_volume_depth_m"])
    H = _fraction(bank["areal_material_thickness_m"])
    pieces=[]
    for index,raw in enumerate(mask["proxy_geometries"]):
        poly=native_polygon_from_dict(raw)
        vertices=_normalized_native_vertex_bounds(raw,precision)
        vertices=tuple(rotate_joint_physical_depth(v,rate,I(H),
                               precision=precision,order=32) for v in vertices)
        cone=_unit_cone_bounds(vertices,precision)
        area=poly.area_bounds()
        if area.lo <= 0:
            raise ValueError("A complete positive HOT source piece cannot be floored or discarded")
        pieces.append(dict(kind="AUTHENTIC_DISJOINT_NATIVE_SOURCE_POLYGON",source_index=index,
            source_sha256=hashlib.sha256(_json(raw).encode()).hexdigest(),
            area_bounds_sr=_payload(area),direction_bounds=_payload(cone),
            material_area_is_containing_chart_area=False,
            source_polygon_pose_applied_once=True,additional_owner_rotation_applied=False,
            source_owner_history_preserved_as_metadata=True))
    for index,encoded in enumerate(bank["paired_corrections_source_json"]):
        raw=json.loads(encoded)
        if raw["positive_component_id"] != bank["component_id"]:
            raise ValueError("HOT correction must preserve its authentic positive material identity")
        vertices=_normalized_native_vertex_bounds(raw["containing_chart"],precision)
        vertices=tuple(rotate_joint_physical_depth(v,rate,I(H),
                               precision=precision,order=32) for v in vertices)
        cone=_unit_cone_bounds(vertices,precision)
        area=_interval(raw["area_bounds_sr"])
        if area.lo <= 0 or raw.get("chart_is_archive_domain") is not False:
            raise ValueError("The actual positive curved material must retain its own positive measure")
        pieces.append(dict(kind="AUTHENTIC_POSITIVE_PAIRED_CURVED_CORRECTION_ONCE",source_index=index,
            source_sha256=hashlib.sha256(encoded.encode()).hexdigest(),
            area_bounds_sr=_payload(area),direction_bounds=_payload(cone),
            material_area_is_containing_chart_area=False,
            source_polygon_pose_applied_once=True,additional_owner_rotation_applied=False,
            source_owner_history_preserved_as_metadata=True))
    if not pieces:
        raise ValueError("A HOT body needs its entire authentic nonempty mask")
    _intersection(sum((_interval(x["area_bounds_sr"]) for x in pieces),I(0)),
                  _interval(bank["actual_area_bounds_sr"]))
    return tuple(_json(piece) for piece in pieces)


def bind_native_common_hot_mean(transport879,work891,cold899,*,common_hot_packet,
        normal_support_packet,curved19_packet,affine_hot_packet=None,provenance,precision=96):
    _precision(precision)
    if type(provenance) is not str or not provenance.strip():
        raise ValueError("The HOT prepared reference and new constitutive extension need provenance")
    for packet,seal in ((transport879,TRANSPORT_SEAL),(work891,WORK_SEAL),(cold899,COLD_SEAL),
            (common_hot_packet,COMMON_HOT_SEAL),(normal_support_packet,NORMAL_SEAL),
            (curved19_packet,CURVED_SEAL)):
        _packet(packet,seal)
    if affine_hot_packet is not None:
        _packet(affine_hot_packet,AFFINE_SEAL)
    common=transport879["common_reference"]
    hot=tuple(sorted((b for b in common["reference"]["banks"] if b["material_kind"]=="hot_birth"),
                     key=lambda b:b["component_id"]))
    if len(hot)!=5:
        raise ValueError("All five complete authentic HOT masks are required")
    source=work891["carrier_source_work_system"]
    order=tuple(tuple(x) for x in source["coordinate_order"])
    values=tuple(_fraction(x) for x in source["bound_coordinate_increments"])
    if len(order)!=100 or len(values)!=100:
        raise ValueError("The unchanged complete COMMON100 source catalogue must be retained")
    c=cold899["cold_continuum_energy_system"]
    cert=json.loads(c["complete_source_carrier_certificate_json"])
    if (cold899["complete_cold_continuum_operator_certified"] is not True
            or cold899["complete_declared_cold_continuum_restart_verified"] is not True):
        raise ValueError("The unchanged whole cold component must retain its existing authority")
    foundations=common_hot_packet["full_reference_face_birth_support"]["foundations"]
    comps={row["component_id"]:row for row in common_hot_packet["hot_components"]}
    # Curved19 is a saved, explicitly unaccepted diagnostic lineage.
    trial=curved19_packet["actual_curved_force_trial"]
    conditional=trial["corrected_foundation_steps"]
    bodies=[]
    for bank in hot:
        cid=bank["component_id"]; comp=comps[cid]
        matches=[f for f in foundations if f["declaration"]["component_id"]==cid]
        if len(matches)!=1:
            raise ValueError("HOT body lost its complete unique foundation authority")
        foundation=matches[0]
        oldbody=comp["body"]["original_body"]
        if (oldbody["coordinate_order"]!=["xx","yy","zz","xy","xz","yz"]
                or comp["assembly"]["bulk"]["coefficient_kernel"]["body"]["coordinate_order"]
                    !=["xx","yy","zz","xy","xz","yz"]):
            raise ValueError("The complete historical HOT Cartesian mode labels changed")
        for key in ("component_id","material_id","original_parent_id","owner","actual_area_bounds_sr",
                "complete_mask_source_json","paired_corrections_source_json","source_snapshot_sha256",
                "source_support_sha256","source_epoch_sha256"):
            if oldbody[key]!=bank[key]:
                raise ValueError("A HOT common binding changed authentic source "+key)
        closure=comp["assembly"]["bulk"]["coefficient_kernel"]["closure"]
        cols=closure["source_columns"]
        if (closure["bank_source_id"]!=cid or closure["source_bank"]!="over"
                or closure["source_support_sha256"]!=bank["source_support_sha256"]):
            raise ValueError("The isotropic HOT law must preserve its complete authentic material bank binding")
        E=_literal_float(cols["over_normal_modulus_pa"]); mu=_literal_float(cols["over_shear_modulus_pa"])
        if E!=F(10**9) or mu!=F(4*10**8) or closure["lateral_closure"]!="laterally_free_standard_isotropic":
            raise ValueError("The authentic explicit isotropic HOT E/G extension changed")
        nu=E/(2*mu)-1; lame=2*mu*nu/(1-2*nu)
        params=foundation["declaration"]["parameters"]
        rho=_literal_float(params["density_kg_m3"]);gravity=_literal_float(params["gravity_m_s2"])
        birth=_literal_float(foundation["declaration"]["radial_birth_m"])
        R=_fraction(bank["radius_m"]);H=_fraction(bank["areal_material_thickness_m"])
        if R!=100000 or H!=2000 or _fraction(bank["roof_radial_displacement_m"])!=0 or birth!=0:
            raise ValueError("Authentic complete HOT source roofs, radius, thickness and birth changed")
        area=_interval(bank["actual_area_bounds_sr"]);volume=_interval(bank["reference_volume_bounds_m3"])
        if area.lo<=0 or volume.lo<=0:
            raise ValueError("Tiny authentic HOT areas and volumes must remain strictly positive")
        _intersection(volume,area*R*R*H)
        scale=_fraction(cert["donor" if bank["owner"]==1 else "receiver"]["fixed_radial_scale"])
        history=json.loads(bank["source_owner_history_json"])
        pose=certify_rotation_history(tuple(rotation_increment_from_dict(x) for x in history),precision).intervals
        rb=interval_cbrt(I(R**3-3*R*R*H),precision)
        indices=tuple(i for i,x in enumerate(order) if x[0]=="body" and x[1]==cid)
        if (len(indices)!=6 or tuple(order[i] for i in indices)!=tuple(("body",cid,name)
                for name in ("xx","yy","zz","xy","xz","yz"))
                or any(values[i]!=0 for i in indices)):
            raise ValueError("The bound authentic HOT30 amplitudes are exactly zero")
        old19=[x for x in conditional if x["state"]["reference"]["cohort_id"]==cid]
        if len(old19)!=1:
            raise ValueError("HOT birth lost its complete unique unaccepted old19 diagnostic lineage")
        oldstep=old19[0]
        old19index=14+conditional.index(oldstep)
        if len(trial["coordinates_decimal"])!=19:
            raise ValueError("The full conditional old19 coordinate catalogue changed")
        bodies.append(NativeCommonHotMeanBody(cid,bank["material_id"],bank["original_parent_id"],
            bank["owner"],R,H,rb,_round(rb*scale,precision),scale,pose,
            tuple(_fraction(x) for x in bank["angular_rate_per_physical_volume_depth_m"]),area,volume,
            _round(volume*scale**3,precision),E,mu,lame,rho,gravity,birth,
            _mask_cells(bank,pose,precision),_json(bank),_json(comp),
            bank["complete_parent_allocation_or_birth_json"],_json(foundation),_json(oldstep),
            trial["coordinates_decimal"][old19index],indices,
            tuple(values[i] for i in indices),tuple(_fraction(x) for x in comp["probe_coordinates_m"]),
            bank["bank_snapshot_sha256"],bank["source_snapshot_sha256"],bank["source_support_sha256"],
            bank["source_epoch_sha256"],old19index,
            _json(oldstep["state"]["radial_displacement_m"])))
    bodies=tuple(bodies)
    for body in bodies:
        _BODY_GUARDS[_digest(body)] = _typed_guard(body)
    material=transport879["material_transfer"]["complete_original_material_state_json"]
    history=_json({k:transport879["material_transfer"][k] for k in
        ("complete_before_work_ledgers_json","complete_current_source_memory_json",
         "complete_current_source_states_json","complete_unaccepted_candidate_work_ledgers_json",
         "original_source_work_lineages")})
    normal=_json(normal_support_packet["bulk_closures"])
    result=NativeCommonHotMeanSystem(bodies,tuple((b.component_id,name) for b in bodies for name in COORDINATE_NAMES),
        (F(0),)*30,order,values,material,history,normal,
        _json(common_hot_packet["full_reference_face_birth_support"]),
        _json(common_hot_packet["hot_source_face_traces"]),_json(common_hot_packet["saved_joint_reference_field"]),
        _json(common_hot_packet["current_load"]),c["binding_sha256"],
        tuple(row["response"]["binding_sha256"] for row in cold899["evaluation_queries"]),
        c["cold_material_carrier_binding_sha256"],
        (TRANSPORT_SEAL,WORK_SEAL,COLD_SEAL,COMMON_HOT_SEAL,NORMAL_SEAL,CURVED_SEAL),precision,provenance,
        SCOPE,REFERENCE,MEASURE,BASAL_LAW,False,1,0,F(0),F(2000000),F(1000000),F(1000000),False,
        False,False,False,False,False,False,False,False,True,True,"")
    return _remember(result,_SYSTEMS)


def validate_native_common_hot_mean(system):
    return _validate(system,NativeCommonHotMeanSystem,_SYSTEMS)


def native_common_hot_mean_to_dict(system):
    return dict(format=FORMAT,**_payload(validate_native_common_hot_mean(system)))


def native_common_hot_mean_from_dict(raw,transport879,work891,cold899,*,common_hot_packet,
        normal_support_packet,curved19_packet,affine_hot_packet=None):
    if type(raw) is not dict or raw.get("format")!=FORMAT:
        raise ValueError("A complete authentic HOT system DTO is required")
    value=bind_native_common_hot_mean(transport879,work891,cold899,common_hot_packet=common_hot_packet,
        normal_support_packet=normal_support_packet,curved19_packet=curved19_packet,
        affine_hot_packet=affine_hot_packet,provenance=raw.get("provenance"),precision=raw.get("precision"))
    if native_common_hot_mean_to_dict(value)!=raw:
        raise ValueError("A resealed HOT system cannot alter its authoritative source or declared reference")
    return value


def _coordinates(system,coordinates):
    coordinates=system.bound_coordinate_increments if coordinates is None else coordinates
    if type(coordinates) is not tuple or len(coordinates)!=30 or any(type(x) is not F for x in coordinates):
        raise ValueError("The finite chart requires all 30 literal Fraction metre coordinates")
    # This is an explicitly declared chart domain, not an adopted source state.
    if any(abs(x)>100 for x in coordinates):
        raise ValueError("The finite HOT chart has a declared absolute amplitude bound of 100 metres")
    return coordinates


def _deviator(b):
    return tuple(tuple(sum((b[k]*GENERATORS[k+1][i][j] for k in range(5)),F(0))
                       for j in range(3)) for i in range(3))


def _eigen_enclosure(h,b,R):
    T=_deviator(b)
    rho=max(sum((abs(x) for x in row),F(0)) for row in T)
    return I(1+(h.lo-rho)/R,1+(h.hi+rho)/R),rho


def _radial_twojet(h,b,R,rb,v,precision):
    """Full physical (h,b) mean integrand jet using the exact unit-ray identity."""
    t=I(1)+h/R;T=_deviator(b);Tv=_mv(T,v);a=_dot(v,Tv)
    delta=2*t*a/R+_dot(Tv,Tv)/(R*R)
    eigen,_=_eigen_enclosure(h,b,R)
    if eigen.lo<=0:
        raise ValueError("The complete finite HOT root bracket must remain SPD")
    s=interval_sqrt(_intersection(t*t+delta,I(eigen.lo**2,eigen.hi**2)),precision)
    # Preserve cancellation at small amplitudes without treating a saved ray
    # midpoint as physical material.  t^2-1 is evaluated in factored form.
    d=rb*((h/R)*(2+h/R)+delta)/(s+1)
    kv=tuple(_mv(K,v) for K in GENERATORS)
    y=tuple(t*v[i]+Tv[i]/R for i in range(3))
    products=tuple((t+a/R) if k==0 else _dot(y,kv[k]) for k in range(6))
    first=tuple(_round(rb*products[k]/(R*s),precision) for k in range(6))
    second=[]
    isotropic=all(x==0 for x in b)
    for i in range(6):
        row=[]
        for j in range(6):
            if isotropic and (i==0 or j==0):
                value=I(0)
            else:
                dot=I(1) if i==j==0 else _dot(kv[i],kv[j])
                value=rb/(R*R)*(dot/s-products[i]*products[j]/(s*s*s))
            row.append(_round(value,precision))
        second.append(tuple(row))
    if isotropic:
        d=rb*h/R
        first=(rb/R,)+tuple(_round(rb*_dot(v,kv[k])/R,precision) for k in range(1,6))
    return _round(d,precision),first,tuple(second)


def _radial_value(h,b,R,rb,v,precision):
    if all(x==0 for x in b):
        return _round(rb*h/R,precision)
    t=I(1)+h/R;Tv=_mv(_deviator(b),v)
    delta=2*t*_dot(v,Tv)/R+_dot(Tv,Tv)/(R*R)
    eigen,_=_eigen_enclosure(h,b,R)
    if eigen.lo<=0:
        raise ValueError("The whole root bracket must remain positive definite")
    s=interval_sqrt(_intersection(t*t+delta,I(eigen.lo**2,eigen.hi**2)),precision)
    return _round(rb*((h/R)*(2+h/R)+delta)/(s+1),precision)


def _mask_integral(body,function,precision):
    values=[]; receipts=[]
    for encoded in body.mask_cells:
        cell=json.loads(encoded);area=_interval(cell["area_bounds_sr"])
        v=tuple(_interval(x) for x in cell["direction_bounds"])
        vector=function(v)
        values.append(tuple(_round(area*x,precision) for x in vector))
        receipts.append(dict(source_sha256=cell["source_sha256"],kind=cell["kind"],
            area_bounds_sr=cell["area_bounds_sr"],full_direction_bounds=cell["direction_bounds"],
            integrand_range_sha256=_digest(vector),whole_cell_range_certified=True,
            containing_chart_area_assigned_as_material=False))
    n=len(values[0])
    return tuple(_round(sum((x[i] for x in values),I(0)),precision) for i in range(n)),tuple(receipts)


def _mean_jet(body,h,b,precision):
    def integrand(v):
        d,g,H=_radial_twojet(h,b,body.radius_m,body.prepared_base_radius_bounds_m,v,precision)
        return (d,)+g+tuple(x for row in H for x in row)
    integral,receipts=_mask_integral(body,integrand,precision)
    values=tuple(_round(x/body.actual_area_bounds_sr,precision) for x in integral)
    mean=values[0];g=values[1:7];H=tuple(values[7+6*i:13+6*i] for i in range(6))
    eigen,rho=_eigen_enclosure(h,b,body.radius_m)
    lower=body.prepared_base_radius_bounds_m.lo/body.radius_m*eigen.lo/eigen.hi
    upper=body.prepared_base_radius_bounds_m.hi/body.radius_m
    g=( _intersection(g[0],I(lower,upper)),)+g[1:]
    if all(x==0 for x in b):
        mean=body.prepared_base_radius_bounds_m*h/body.radius_m
        g=(body.prepared_base_radius_bounds_m/body.radius_m,)+g[1:]
    return mean,g,H,receipts


def _solve_finite_mean_body(body,q,b,*,precision=96,root_iterations=12):
    """Unique exact full-mask root enclosure, with its implicit first/twojet."""
    _precision(precision)
    if (type(body) is not NativeCommonHotMeanBody or type(q) is not F or type(b) is not tuple
            or len(b)!=5 or any(type(x) is not F for x in b)
            or type(root_iterations) is not int or not 0<=root_iterations<=64):
        raise ValueError("The finite mean root requires a complete authentic body and literal query")
    R=body.radius_m;rb=body.prepared_base_radius_bounds_m
    h0=_round(I(R*q)/rb,precision)
    _,rho=_eigen_enclosure(I(0),b,R)
    if R+h0.lo<=2*rho:
        raise ValueError("The full finite mean bracket lacks the strict SPD uniqueness domain")
    initial=I(h0.lo-rho,h0.hi+rho)
    h=initial;history=[]
    if all(x==0 for x in b):
        h=h0
    else:
        for _ in range(root_iterations):
            centre=h.midpoint
            values,_=_mask_integral(body,lambda v:(_radial_value(I(centre),b,R,rb,v,precision),),precision)
            M=_round(values[0]/body.actual_area_bounds_sr,precision)
            eigen,_=_eigen_enclosure(h,b,R)
            gh=I(rb.lo/R*eigen.lo/eigen.hi,rb.hi/R)
            # Outward compression keeps a shrinking bracket exact without
            # accumulating unrelated enormous rational denominators.
            new=_intersection(h,_round(I(centre)-(M-q)/gh,precision))
            history.append(dict(bracket_before_m=_payload(h),whole_mean_at_trial_bounds_m=_payload(M),
                                mean_h_derivative_bounds=_payload(gh),bracket_after_m=_payload(new)))
            if new.width>=h.width or new.width==0:
                h=new;break
            h=new
    mean,g,H,receipts=_mean_jet(body,h,b,precision)
    if not mean.lo<=q<=mean.hi:
        raise ValueError("The complete actual-mask root enclosure failed to contain its prescribed mean")
    jac=(I(1)/g[0],)+tuple(-g[i]/g[0] for i in range(1,6))
    J=tuple(tuple(jac[a] if i==0 else I(int(i==a)) for a in range(6)) for i in range(6))
    second=tuple(tuple(_round(-sum((J[i][a]*H[i][j]*J[j][bb]
                        for i in range(6) for j in range(6)),I(0))/g[0],precision)
                        for bb in range(6)) for a in range(6))
    eigen,_=_eigen_enclosure(h,b,R)
    cert=dict(format="whole-actual-base-implicit-mean-root-1",component_id=body.component_id,
        actual_source_area_bounds_sr=_payload(body.actual_area_bounds_sr),
        prepared_base_radius_bounds_m=_payload(rb),prescribed_mean_m=_payload(q),
        five_tracefree_amplitudes_m=_payload(b),initial_global_bracket_m=_payload(initial),
        scalar_affine_amplitude_bounds_m=_payload(h),actual_mean_bounds_m=_payload(mean),
        scalar_affine_amplitude_enclosure_width_m=_payload(h.width),
        complete_mask_mean_h_derivative_positive_bounds=_payload(g[0]),
        whole_bracket_deformation_eigenvalue_bounds=_payload(eigen),
        whole_mask_unique_root_certified=True,exact_isotropic_branch=all(x==0 for x in b),
        literal_root_replaced_by_midpoint=False,interval_newton_history=history,
        complete_mask_cells=receipts,positive_curve_count=sum(json.loads(x)["kind"]==
            "AUTHENTIC_POSITIVE_PAIRED_CURVED_CORRECTION_ONCE" for x in body.mask_cells),
        true_area_normalization=True,positive_source_area_floor_used=False,
        summed_complete_source_piece_area_bounds_sr=_payload(sum(
            (_interval(json.loads(x)["area_bounds_sr"]) for x in body.mask_cells),I(0))),
        complete_source_piece_area_matches_authentic_total=True,
        source_polygon_pose_applied_once=True,
        additional_owner_rotation_applied=False,
        source_owner_history_preserved_as_metadata=True,
        historical_hot_pose_equivalence_claimed=False,
        individual_complete_body_positive_jacobian_certified=True,
        individual_complete_body_global_injectivity_certified=True,
        prepared_reference_original_material_jacobian_exact=_payload(body.fixed_owner_scale**3),
        deformed_original_material_jacobian_bounds=_payload(I(eigen.lo**3,eigen.hi**3)*body.fixed_owner_scale**3),
        whole_body_physical_density_bounds_kg_m3=_payload(I(body.density_kg_m3)/
            (I(eigen.lo**3,eigen.hi**3)*body.fixed_owner_scale**3)),
        prepared_reference_inverse_recipe="z=(R^3-(norm(X0)/s)^3)/(3R^2); u_source_world=Qdepth(z)^T*X0/norm(X0)",
        deformed_reference_inverse_recipe="apply constant SPD F inverse, then complete prepared reference inverse",
        source_density_pushforward_model="rho_source/(fixed_scale^3*det(F))",
        physical_all_pair_admission_certified=False,reference_preparation_work_certified=False)
    return h,J,second,cert


def solve_native_common_hot_finite_mean(body,q,b,*,precision=96,root_iterations=12):
    """Authenticate the complete material body before evaluating its chart."""
    if (type(body) is not NativeCommonHotMeanBody
            or _BODY_GUARDS.get(_digest(body)) != _typed_guard(body)):
        raise ValueError("The finite mean root cannot substitute an unbound authentic material body")
    return _solve_finite_mean_body(body,q,b,precision=precision,root_iterations=root_iterations)


def _bulk_twojet(body,h,b,precision):
    R=body.radius_m;t=I(1)+h/R;T=_deviator(b)
    Fmat=tuple(tuple(t*int(i==j)+T[i][j]/R for j in range(3)) for i in range(3))
    FTF=_mm(_tr(Fmat),Fmat)
    strain=tuple(tuple((FTF[i][j]-int(i==j))/2 for j in range(3)) for i in range(3))
    derivatives=tuple(_sym(_mm(_tr(Fmat),tuple(tuple(x/R for x in row) for row in K))) for K in GENERATORS)
    second=tuple(tuple(_sym(_mm(_tr(Ki),Kj)) for Kj in GENERATORS) for Ki in GENERATORS)
    def inner(a,bb):
        return 2*body.shear_modulus_pa*sum((a[i][j]*bb[i][j] for i in range(3) for j in range(3)),I(0)) \
            +body.lame_modulus_pa*sum((a[i][i] for i in range(3)),I(0))*sum((bb[i][i] for i in range(3)),I(0))
    V=body.original_reference_volume_bounds_m3
    energy=_round(V*(body.shear_modulus_pa*sum((_square(strain[i][j])
        for i in range(3) for j in range(3)),I(0))+body.lame_modulus_pa/2*
        _square(sum((strain[i][i] for i in range(3)),I(0)))),precision)
    grad=tuple(_round(V*inner(strain,d),precision) for d in derivatives)
    H=tuple(tuple(_round(V*(inner(derivatives[i],derivatives[j])+inner(strain,second[i][j])/(R*R)),precision)
                  for j in range(6)) for i in range(6))
    return energy,grad,H


def _basal_twojet(body,h,b,precision):
    def integrand(v):
        d,g,H=_radial_twojet(h,b,body.radius_m,body.prepared_base_radius_bounds_m,v,precision)
        d=d-body.source_birth_radial_displacement_m
        return (_square(d)/2,)+tuple(d*x for x in g)+tuple(g[i]*g[j]+d*H[i][j]
               for i in range(6) for j in range(6))
    values,receipts=_mask_integral(body,integrand,precision)
    k=body.density_kg_m3*body.gravity_m_s2*body.radius_m**2
    values=tuple(_round(k*x,precision) for x in values)
    return values[0],values[1:7],tuple(values[7+6*i:13+6*i] for i in range(6)),receipts


def _chain_twojet(energy,grad,H,J,hsecond,precision):
    newgrad=tuple(_round(sum((grad[i]*J[i][a] for i in range(6)),I(0)),precision) for a in range(6))
    newH=tuple(tuple(_round(sum((J[i][a]*H[i][j]*J[j][b]
                        for i in range(6) for j in range(6)),I(0))+grad[0]*hsecond[a][b],precision)
                        for b in range(6)) for a in range(6))
    # The physical (h,b) force conjugate to h equals f_xx+f_yy+f_zz.
    return energy,newgrad,newH


@dataclass(frozen=True, slots=True)
class NativeCommonHotMeanResponse:
    system_binding_sha256: str
    coordinate_increments: tuple
    energy_bounds_j: I
    gradient_bounds_n: tuple
    hessian_bounds_n_m: tuple
    body_responses_json: tuple
    precision: int
    root_iterations: int
    whole_original_hot_material_count: int
    positive_paired_curved_correction_count: int
    complete_hot30_operator_certified: bool
    finite_actual_base_mean_chart_certified: bool
    source_scalar_foundation_added_again: bool
    physical_material_admission_authorized: bool
    whole_clock_all_pair_admission_authorized: bool
    physical_source_force_prestress_work_handover_certified: bool
    prepared_reference_full_system_work_certified: bool
    new_mixed_force_equilibrium_certified: bool
    finite_mean_equilibrium_certified: bool
    accepted_new_native_intervals: int
    clocks_advanced: bool
    binding_sha256: str


def evaluate_native_common_hot_mean(system,coordinate_increments=None,*,precision=96,root_iterations=12):
    system=validate_native_common_hot_mean(system);_precision(precision)
    z=_coordinates(system,coordinate_increments)
    if type(root_iterations) is not int or not 0<=root_iterations<=64:
        raise ValueError("The implicit finite-mean root budget must be a literal integer")
    key=(system.binding_sha256,z,precision,root_iterations)
    if key in _EVALUATIONS:
        value=_EVALUATIONS[key]
        return _validate(value,NativeCommonHotMeanResponse,_RESPONSES)
    energy=I(0);gradient=[I(0)]*30;Hessian=[[I(0) for _ in range(30)] for _ in range(30)];rows=[]
    for index,body in enumerate(system.bodies):
        q=z[6*index];b=z[6*index+1:6*index+6]
        h,J,hs,certificate=solve_native_common_hot_finite_mean(body,q,b,precision=precision,
                                                             root_iterations=root_iterations)
        bulk=_bulk_twojet(body,h,b,precision);basal=_basal_twojet(body,h,b,precision)
        physical=(bulk[0]+basal[0],tuple(bulk[1][i]+basal[1][i] for i in range(6)),
                  tuple(tuple(bulk[2][i][j]+basal[2][i][j] for j in range(6)) for i in range(6)))
        E,g,H=_chain_twojet(*physical,J,hs,precision)
        if q==0 and all(x==0 for x in b):
            E=I(0);g=(I(0),)*6
        energy+=E
        for i in range(6):
            gradient[6*index+i]=g[i]
            for j in range(i,6):
                symmetric=_intersection(H[i][j],H[j][i])
                Hessian[6*index+i][6*index+j]=symmetric
                Hessian[6*index+j][6*index+i]=symmetric
        rows.append(_json(dict(component_id=body.component_id,finite_mean_root_certificate=certificate,
            implicit_physical_coordinate_jacobian=_payload(J),implicit_scalar_coordinate_hessian=_payload(hs),
            full_physical_cartesian_amplitude_bounds_m=_payload(tuple(sum((_C[i][j]*((h,)+b)[j]
                for j in range(6)),I(0)) for i in range(6))),
            bulk_energy_bounds_j=_payload(bulk[0]),basal_energy_bounds_j=_payload(basal[0]),
            basal_whole_mask_range_receipts=basal[3],full_same_field_basal_variance_retained=True,
            chart_force_curvature_term_included=True,energy_bounds_j=_payload(E),
            gradient_bounds_n=_payload(g),hessian_bounds_n_m=_payload(H),
            physical_scalar_force_bounds_n=_payload(physical[1][0]),
            physical_source_historical_prestress_recovered=False)))
    result=NativeCommonHotMeanResponse(system.binding_sha256,z,_round(energy,precision),tuple(gradient),
        tuple(tuple(row) for row in Hessian),tuple(rows),precision,root_iterations,5,
        sum(sum(json.loads(x)["kind"]=="AUTHENTIC_POSITIVE_PAIRED_CURVED_CORRECTION_ONCE"
                for x in b.mask_cells) for b in system.bodies),True,True,False,False,False,False,False,False,False,0,False,"")
    result=_remember(result,_RESPONSES);_EVALUATIONS[key]=result
    return result


def native_common_hot_mean_response_to_dict(response):
    return dict(format=RESPONSE_FORMAT,**_payload(_validate(response,NativeCommonHotMeanResponse,_RESPONSES)))


def native_common_hot_mean_response_from_dict(raw,*,system,expected_coordinate_increments=None,
        precision=96,root_iterations=12):
    if type(raw) is not dict or raw.get("format")!=RESPONSE_FORMAT:
        raise ValueError("A complete HOT30 response DTO is required")
    result=evaluate_native_common_hot_mean(system,expected_coordinate_increments,
                                        precision=precision,root_iterations=root_iterations)
    if native_common_hot_mean_response_to_dict(result)!=raw:
        raise ValueError("A resealed HOT response cannot change its authenticated full query or operator")
    return result
