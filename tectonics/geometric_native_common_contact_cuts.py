"""Actual retained contact boundary traces in the NEW common reference.

Five cold/cold cuts are rebuilt from genuine native source edges.  The C6/C7
edge is restricted by the exact terminal envelope support, and the separate
C6/HOT C1 curve is the complete analytic first-entry boundary, never a polygon
proxy.  Boundary traces refer to both material closures, not positive-volume
interior points.  Unknown native endpoints and the implicit turning-time band
are retained as enclosures and distinguished from genuine parameter queries.

Historical spring atoms, free anchors, constitutive directors/measure and rest
gap are untouched.  This module supplies geometry and body-port prerequisites;
it does not allocate atom identities or certify a contact law or force interval.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from fractions import Fraction as F
from functools import lru_cache
from pathlib import Path
import hashlib
import json

from .geometric_intrinsic_geometry import rotation_increment_from_dict
from .geometric_native_pose import (
    FractionInterval as I, _rounded_interval, certify_rotation_history,
    interval_pi, interval_sqrt,
)
from .geometric_native_polygons import (
    native_polygon_from_dict, native_shared_boundary_arcs, native_arc_to_dict,
    _ray_values, _plane_values, _atan2_positive,
)
from .geometric_native_spherical_solid import _arc_frame, interval_cbrt
from .geometric_native_first_entry_domains import _normal_series, _composites
from .geometric_native_normal_support import (
    load_native_normal_support_source, _encoded, _payload, _read_interval, _read_fraction,
)
from .geometric_native_common_reference import (
    validate_native_common_reference, bind_native_common_reference_bank,
    bound_native_common_reference_bank_point, _rotate_common,
)

FORMAT = "native-common-contact-cuts-1"
SCOPE = "ACTUAL_COMMON_CONTACT_BOUNDARY_GEOMETRY_ONLY"
_ARCHIVE = Path(__file__).resolve().parents[1]/"analysis/geometric_native_common_reference_validation"
AUDIT_RAW_SHA256 = "7ac3cd74013a20d7a06e95eccec4e6ceb3db412fb197ec3ed559c5af43108200"
AUDIT_SOURCE_RAW_SHA256 = "4f7647fe1e2957fb2d1a37cf923d175dc880c167fd960907555bf845902b1a48"
_SETS = {}
_CUTS = {}
_RUNTIME = {}
_BOUNDS = {}


def _dot(a, b):
    return sum((x*y for x, y in zip(a, b)), I(0))


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _map(M, x):
    return tuple(_dot(row, x) for row in M)


def _inverse_map(M, x):
    return tuple(_dot(tuple(M[j][i] for j in range(3)), x) for i in range(3))


def _square(x):
    return I(0 if x.lo <= 0 <= x.hi else min(x.lo*x.lo, x.hi*x.hi),
             max(x.lo*x.lo, x.hi*x.hi))


def _round(x, precision):
    return _rounded_interval(x if type(x) is I else I(x), precision)


def _budget(precision, order=28):
    if type(precision) is not int or not 96 <= precision <= 512:
        raise ValueError("Contact geometry precision must be an integer from96 to512")
    if type(order) is not int or not 8 <= order <= 64:
        raise ValueError("Contact geometry rotation order must be an integer from8 to64")


def _record(value):
    return _encoded(_payload(value))


def _seal(value, key):
    return hashlib.sha256(_encoded({f.name: _payload(getattr(value, f.name))
        for f in fields(value) if f.name != key})).hexdigest()


@lru_cache(maxsize=1)
def _archive():
    raw=(_ARCHIVE/"retained_original_contact_cut_audit.json").read_bytes()
    script=(_ARCHIVE/"retained_original_contact_cut_audit_source.txt").read_bytes()
    if (hashlib.sha256(raw).hexdigest()!=AUDIT_RAW_SHA256 or
            hashlib.sha256(script).hexdigest()!=AUDIT_SOURCE_RAW_SHA256):
        raise ValueError("The independently source-bound retained-cut audit changed")
    audit=json.loads(raw)
    body=dict(audit)
    if body.pop("audit_sha256")!=hashlib.sha256(_encoded(body)).hexdigest():
        raise ValueError("Retained-cut audit selfseal changed")
    return audit


@dataclass(frozen=True, slots=True)
class NativeCommonContactCut:
    cut_id: str
    kind: str
    donor_component_id: str
    receiver_component_id: str
    donor_original_parent_id: str
    receiver_original_parent_id: str
    incoming_donor_trace_id: str
    receiving_trace_id: str | None
    receiving_family_id: str
    source_snapshot_sha256: str
    source_epoch_sha256: str
    reference_snapshot_sha256: str
    donor_bank_snapshot_sha256: str
    receiver_bank_snapshot_sha256: str
    common_physical_radius_cubed_bounds_m3: I
    parameter_bounds: I
    parameter_endpoint_bounds: I
    source_parameter_expression: str
    start_unit_bounds: tuple | None
    tangent_unit_bounds: tuple | None
    receiving_normal_orientation: int
    genuine_native_arc_json: str | None
    complete_donor_mask_source_json: str
    complete_receiver_mask_source_json: str
    complete_source_sweep_json: str
    complete_source_history_json: str
    complete_source_owner_histories_json: str
    complete_source_trace_bindings_json: str
    complete_paired_correction_source_json: str | None
    exact_terminal_slope_expression: str | None
    terminal_slope_bounds: I | None
    maximum_ratio_time_fraction_bounds: I | None
    maximum_ratio_bounds: I | None
    native_untrimmed_angle_endpoint_bounds: I | None
    source_live_cut_endpoint_margins: tuple
    geometry_enclosure_precision: int
    cut_snapshot_sha256: str

    @property
    def parameter_guaranteed_bounds(self):
        return (I(self.maximum_ratio_time_fraction_bounds.hi,1) if self.kind=="true_curved_envelope"
                else I(0,self.parameter_endpoint_bounds.lo))

    @property
    def donor_material_kind(self):
        return "original_live"

    @property
    def receiver_material_kind(self):
        return "hot_birth" if self.kind=="true_curved_envelope" else "original_live"

    @property
    def true_source_boundary_of_both_actual_masks_certified(self):
        return True

    @property
    def old_mechanical_atom_ids_assigned(self):
        return False

    @property
    def constitutive_rest_contact_reference_recovered(self):
        return False

    @property
    def physical_material_admission_authorized(self):
        return False


@dataclass(frozen=True, slots=True)
class NativeCommonContactCutSet:
    reference_snapshot_sha256: str
    source_snapshot_sha256: str
    source_epoch_sha256: str
    complete_source_dto_json: tuple
    complete_historical_saved_field_json: str
    archived_audit_raw_sha256: str
    archived_audit_source_raw_sha256: str
    cuts: tuple
    geometry_enclosure_precision: int
    set_snapshot_sha256: str

    @property
    def physical_material_admission_authorized(self):
        return False


def _analytic_source(source, sweep):
    before=json.loads(dict(source.complete_source_dto_json)["before"])
    owners=dict(before["snapshot"]["current_material_epoch"]["source_material_state"]["journal"]["omega_rad_per_myr"])
    journal=before["snapshot"]["current_material_epoch"]["source_material_state"]["journal"]
    C,Y,Z,_=_normal_series(owners[1],owners[0],journal["dt_myr"],sweep["order"],sweep["precision"])
    ap,bp,curvature,height=_composites(C,Y,Z)
    named=(("C",C),("Y",Y),("Z",Z),("A1",ap),("B1",bp),
           ("curvature_numerator",curvature),("height_numerator",height),
           ("area_integrand_numerator",-height*curvature))
    if len(sweep["analytic_polynomials"])!=len(named):
        raise ValueError("The complete curved normal-series source changed")
    for row,(name,poly) in zip(sweep["analytic_polynomials"],named):
        if (row["name"]!=name or tuple(_read_fraction(x) for x in row["coefficients"])!=poly.coefficients
                or tuple(_read_fraction(x) for x in row["derivative_tails"])!=poly.tails):
            raise ValueError("Complete exact curved coefficients/tails failed source replay")
    root=_read_interval(sweep["envelope"]["maximum_ratio_time_fraction"])
    whole=I(0,1)
    if (not 0<root.lo<=root.hi<1 or C.bounds(whole).lo<=0 or ap.bounds(whole).lo<=0
            or bp.bounds(whole).lo<=0 or curvature.bounds(whole,1).hi>=0
            or curvature.bounds(I(root.lo)).hi>=0 or curvature.bounds(I(root.lo,1)).hi>=0):
        raise ValueError("The true continuous envelope lost strict source hypotheses")
    k=bp.bounds(I(1))/ap.bounds(I(1))
    q=Z.bounds(I(1))/Y.bounds(I(1))
    qm=_read_interval(sweep["envelope"]["maximum_ratio_bounds"])
    if not 0<k.lo<=k.hi<q.lo<=q.hi<qm.lo:
        raise ValueError("Terminal support and true curved branch order are unresolved")
    if not (_read_interval(sweep["envelope"]["terminal_slope_bounds"]).lo<=k.lo<=k.hi<=
            _read_interval(sweep["envelope"]["terminal_slope_bounds"]).hi):
        raise ValueError("The exact terminal slope no longer binds its source enclosure")
    return C,Y,Z,ap,bp,curvature,height,root,k,qm


def _curve_geometry(runtime, parameter, precision, endpoint=None):
    C,Y,Z,ap,bp,curvature,height,root,k1,qm=runtime["analytic"]
    rounded=lambda x:_round(x,precision+24)
    c,y,z=(rounded(p.bounds(parameter)) for p in (C,Y,Z))
    ac=rounded(ap.bounds(parameter))
    a=rounded(y/c)
    k=rounded(rounded(bp.bounds(parameter))/ac)
    h=rounded(rounded(height.bounds(parameter))/ac)
    if endpoint=="start":
        # At the unique turning root, h=0 and k=q_max exactly.  This is a
        # correlated source identity, not replacement by a root midpoint.
        h,k=I(0),qm
    speed_factor=rounded(-rounded(curvature.bounds(parameter))/_square(ac))
    if speed_factor.lo<=0:
        raise ValueError("The true curved source boundary lost strict regularity")
    q=(h,-k,I(1))
    s2=rounded(I(1)+_square(h)+_square(k))
    s=interval_sqrt(s2,precision)
    b=rounded(z/c)
    d=interval_sqrt(rounded(I(1)+_square(a)+_square(b)),precision)
    v=tuple(rounded(x/s) for x in q)
    qp=(rounded(speed_factor*a),speed_factor,I(0))
    projection=rounded(_dot(v,qp))
    derivative=tuple(rounded((x-u*projection)/s) for x,u in zip(qp,v))
    # q cross q'=(-k')(-1,a,b), directed into true archive/HOT C1.
    N=tuple(rounded(x/d) for x in (I(-1),a,b))
    speed=rounded(speed_factor*d/s2)
    pose=runtime["poses"][1]
    U=tuple(rounded(x) for x in _map(pose,v))
    Up=tuple(rounded(x) for x in _map(pose,derivative))
    normal=tuple(rounded(x) for x in _map(pose,N))
    return U,Up,normal,speed


def _common_radius(reference, donor, receiver):
    a=bind_native_common_reference_bank(reference,donor)
    b=bind_native_common_reference_bank(reference,receiver)
    lo=max(a.layer.layer.bottom_radius_cubed_m3,b.layer.layer.bottom_radius_cubed_m3)
    hi=min(a.layer.layer.top_radius_m**3,b.layer.layer.top_radius_m**3)
    if not 0<lo<hi:
        raise ValueError("Actual bank radii have no positive common material depth")
    return a,b,I(lo,hi)


def bind_common_contact_cuts(reference, *, precision=160):
    """Rebuild five real current cuts and the genuine separate curved boundary."""
    reference=validate_native_common_reference(reference)
    _budget(precision)
    key=(reference.reference_snapshot_sha256,precision)
    if key in _SETS:
        return validate_common_contact_cuts(_SETS[key],reference=reference)
    source=load_native_normal_support_source()
    if source.source_snapshot_sha256!=reference.source_snapshot_sha256:
        raise ValueError("Contact cuts require the same complete authenticated source")
    audit=_archive()
    if (audit["source_snapshot_sha256"]!=source.source_snapshot_sha256 or
            any(audit["complete_source_dto_sha256"][k]!=hashlib.sha256(v.encode()).hexdigest()
                for k,v in source.complete_source_dto_json)):
        raise ValueError("Retained-cut audit differs from the literal current source DTOs")
    before=json.loads(dict(source.complete_source_dto_json)["before"])
    epoch=before["snapshot"]["current_material_epoch"]
    material=epoch["source_material_state"]
    sweep=material["sweep_certificate"]
    components={c["component_id"]:c for c in epoch["components"]}
    cold={c["original_parent_id"]:c for c in components.values() if c["material_kind"]=="original_live"}
    if len(cold)!=8 or len(components)!=13 or dict(epoch["additional_owner_histories"])!={0:[],1:[]}:
        raise ValueError("The genuine initial source masks or owner histories changed")
    frame_precision=max(precision,320)
    poses={o:certify_rotation_history(tuple(rotation_increment_from_dict(s) for s in h),frame_precision).intervals
           for o,h in before["snapshot"]["owner_histories"]}
    bindings=material["trace_bindings"]
    donor_trace={b["donor_fragment_id"]:b["trace_id"] for b in bindings}
    receiver_trace={b["receiver_fragment_id"]:b["trace_id"] for b in bindings}
    if len(donor_trace)!=4 or len(receiver_trace)!=4:
        raise ValueError("Current cuts need all four authentic physical source families")
    analytic=_analytic_source(source,sweep)
    runtime=dict(reference=reference,source=source,before=before,epoch=epoch,
                 poses=poses,analytic=analytic,cold=cold)
    common=dict(source_snapshot_sha256=source.source_snapshot_sha256,source_epoch_sha256=source.source_epoch_sha256,
        reference_snapshot_sha256=reference.reference_snapshot_sha256,
        complete_source_sweep_json=_encoded(sweep).decode(),
        complete_source_history_json=_encoded(before["original_physical_history"]).decode(),
        complete_source_owner_histories_json=_encoded(before["snapshot"]["owner_histories"]).decode(),
        complete_source_trace_bindings_json=_encoded(bindings).decode(),geometry_enclosure_precision=frame_precision,
        cut_snapshot_sha256="")
    cuts=[]
    # The independently complete arrangement audit identifies the five pairs;
    # each literal native arc is independently rebuilt from the actual masks.
    rows=audit["true_source_endpoint_cold_boundary_arcs"]
    if len(rows)!=5:
        raise ValueError("The complete authenticated retained-cut arrangement changed")
    for row in rows:
        dp,rp=row["donor_parent"],row["receiver_parent"]
        donor,receiver=cold[dp],cold[rp]
        genuine=[]
        for aa in donor["proxy_geometries"]:
            for bb in receiver["proxy_geometries"]:
                genuine.extend(native_shared_boundary_arcs(native_polygon_from_dict(aa),native_polygon_from_dict(bb)))
        if len(genuine)!=1 or _encoded(native_arc_to_dict(genuine[0]))!=_encoded(row["native_arc"]):
            raise ValueError("A retained source cut is not its actual unique native shared boundary")
        arc=genuine[0]
        S,T,_,L=_arc_frame(arc,frame_precision)
        declared_end=_read_interval(row["native_angle_bounds_radians"])
        if not declared_end.lo<=L.lo<=L.hi<=declared_end.hi:
            raise ValueError("A native endpoint escaped its complete original audit uncertainty")
        N=_cross(S,T)
        # Source arc plane belongs to donor; its opposite is receiver-inward.
        inward=tuple(-x for x in _plane_values(arc.plane,frame_precision)[1])
        orientation=_dot(N,inward)
        if orientation.lo<=0<=orientation.hi:
            raise ValueError("The real receiver side of a native cut is unresolved")
        sign=1 if orientation.lo>0 else -1
        margins=()
        kind="native_arc"
        end=declared_end
        expr="native_minor_arc(start,tangent,theta), 0<=theta<=true_native_end"
        slope=None
        if dp=="octant:5":
            k=analytic[8]
            ends=tuple(_inverse_map(poses[1],_ray_values(ray,frame_precision)[1]) for ray in (arc.start,arc.end))
            margins=tuple(q[1]+k*q[2] for q in ends)
            if rp=="octant:1":
                if not (margins[0].lo>0 and margins[1].hi<0 and declared_end.hi<interval_pi(frame_precision).lo):
                    raise ValueError("The true C6 terminal cut requires unique strict implicit trimming")
                Uj,_,_,_=_curve_geometry(runtime,I(1),frame_precision)
                end=_atan2_positive(interval_sqrt(_dot(_cross(S,Uj),_cross(S,Uj)),frame_precision),_dot(S,Uj),frame_precision)
                if not 0<end.lo<=end.hi<declared_end.lo:
                    raise ValueError("The true envelope join did not strictly trim the archived proxy endpoint")
                kind="implicit_terminal_trim"
                slope=k
                expr="native_minor_arc restricted by original_donor_y+(B1(1)/A1(1))*original_donor_z>=0; terminal_join=(b1-a1*k1,-k1,1)"
            elif not all(m.lo>0 for m in margins):
                raise ValueError("The microscopic C6 cross-pair cut is not wholly live")
        elif donor.get("component_id") in {c["negative_component_id"] for c in epoch["paired_corrections"]}:
            raise ValueError("A true curved correction cannot be ignored by a polygon cut")
        db,rb,cubes=_common_radius(reference,donor["component_id"],receiver["component_id"])
        cut=NativeCommonContactCut(cut_id=f"retained-contact:{dp}:{rp}",kind=kind,
            donor_component_id=db.component_id,receiver_component_id=rb.component_id,
            donor_original_parent_id=dp,receiver_original_parent_id=rp,
            incoming_donor_trace_id=donor_trace[dp],receiving_trace_id=receiver_trace[rp],
            receiving_family_id="continuous-source-receiver:"+receiver_trace[rp],
            donor_bank_snapshot_sha256=db.bank_snapshot_sha256,receiver_bank_snapshot_sha256=rb.bank_snapshot_sha256,
            common_physical_radius_cubed_bounds_m3=cubes,parameter_bounds=I(0,end.hi),parameter_endpoint_bounds=end,
            source_parameter_expression=expr,start_unit_bounds=S,tangent_unit_bounds=T,receiving_normal_orientation=sign,
            genuine_native_arc_json=_encoded(native_arc_to_dict(arc)).decode(),
            complete_donor_mask_source_json=db.complete_mask_source_json,complete_receiver_mask_source_json=rb.complete_mask_source_json,
            complete_paired_correction_source_json=None,exact_terminal_slope_expression="B1(1)/A1(1)" if slope else None,
            terminal_slope_bounds=slope,maximum_ratio_time_fraction_bounds=None,maximum_ratio_bounds=None,
            native_untrimmed_angle_endpoint_bounds=declared_end,source_live_cut_endpoint_margins=margins,**common)
        cut=replace(cut,cut_snapshot_sha256=_seal(cut,"cut_snapshot_sha256"))
        cuts.append(cut)
    corrections=epoch["paired_corrections"]
    if len(corrections)!=1:
        raise ValueError("The complete true curved subset pairing changed")
    correction=corrections[0]
    if (correction["negative_component_id"]!=cold["octant:5"]["component_id"] or
            components[correction["positive_component_id"]]["material_kind"]!="hot_birth" or
            components[correction["positive_component_id"]]["original_parent_id"]!="octant:5" or
            _read_interval(correction["area_bounds_sr"]).lo<=0):
        raise ValueError("The true positive curved material subset cannot be replaced or clipped")
    db,rb,cubes=_common_radius(reference,correction["negative_component_id"],correction["positive_component_id"])
    root,k,qm=analytic[7],analytic[8],analytic[9]
    curved=NativeCommonContactCut(cut_id="retained-contact:octant:5:hot-curved-extra",kind="true_curved_envelope",
        donor_component_id=db.component_id,receiver_component_id=rb.component_id,
        donor_original_parent_id="octant:5",receiver_original_parent_id="octant:5",
        incoming_donor_trace_id=donor_trace["octant:5"],receiving_trace_id=None,
        receiving_family_id="true-curved-hot-boundary:"+rb.component_id,
        donor_bank_snapshot_sha256=db.bank_snapshot_sha256,receiver_bank_snapshot_sha256=rb.bank_snapshot_sha256,
        common_physical_radius_cubed_bounds_m3=cubes,parameter_bounds=I(root.lo,1),parameter_endpoint_bounds=I(1),
        source_parameter_expression="u_m<=u<=1; original_donor_q=(b(u)-a(u)*k(u),-k(u),1); k=B1/A1; source_owner1_endpoint_pose",
        start_unit_bounds=None,tangent_unit_bounds=None,receiving_normal_orientation=1,genuine_native_arc_json=None,
        complete_donor_mask_source_json=db.complete_mask_source_json,complete_receiver_mask_source_json=rb.complete_mask_source_json,
        complete_paired_correction_source_json=_encoded(correction).decode(),exact_terminal_slope_expression="B1(1)/A1(1)",
        terminal_slope_bounds=k,maximum_ratio_time_fraction_bounds=root,maximum_ratio_bounds=qm,
        native_untrimmed_angle_endpoint_bounds=None,source_live_cut_endpoint_margins=(),**common)
    curved=replace(curved,cut_snapshot_sha256=_seal(curved,"cut_snapshot_sha256"))
    cuts.append(curved)
    result=NativeCommonContactCutSet(reference.reference_snapshot_sha256,source.source_snapshot_sha256,
        source.source_epoch_sha256,source.complete_source_dto_json,source.saved_field_json,AUDIT_RAW_SHA256,
        AUDIT_SOURCE_RAW_SHA256,tuple(cuts),precision,"")
    result=replace(result,set_snapshot_sha256=_seal(result,"set_snapshot_sha256"))
    _SETS[key]=result
    for cut in result.cuts:
        _CUTS[cut.cut_snapshot_sha256]=_record(cut)
        _RUNTIME[cut.cut_snapshot_sha256]=runtime
    return result


def validate_common_contact_cuts(value, *, reference=None):
    if type(value) is not NativeCommonContactCutSet:
        raise ValueError("Expected an authenticated immutable current-cut collection")
    if reference is not None:
        validate_native_common_reference(reference)
        if value.reference_snapshot_sha256!=reference.reference_snapshot_sha256:
            raise ValueError("Current cuts belong to another common reference")
    expected=_SETS.get((value.reference_snapshot_sha256,value.geometry_enclosure_precision))
    if expected is None or _record(expected)!=_record(value) or value.set_snapshot_sha256!=_seal(value,"set_snapshot_sha256"):
        raise ValueError("Current cuts were not built by complete source/mask/history replay")
    for cut in value.cuts:
        validate_common_contact_cut(cut,reference=reference)
    return value


def validate_common_contact_cut(cut, *, reference=None):
    if type(cut) is not NativeCommonContactCut or _CUTS.get(cut.cut_snapshot_sha256)!=_record(cut):
        raise ValueError("Current cut was not built from its authenticated true source masks")
    if cut.cut_snapshot_sha256!=_seal(cut,"cut_snapshot_sha256"):
        raise ValueError("The complete typed current-cut receipt changed")
    if reference is not None:
        validate_native_common_reference(reference)
        if cut.reference_snapshot_sha256!=reference.reference_snapshot_sha256:
            raise ValueError("Current cut belongs to another common reference")
    return cut


def common_contact_cuts_to_dict(value):
    validate_common_contact_cuts(value)
    return dict(format=FORMAT,scope=SCOPE,**_payload(value))


def common_contact_cut_to_dict(cut):
    validate_common_contact_cut(cut)
    return dict(format=FORMAT,scope=SCOPE,**_payload(cut))


def common_contact_cuts_from_dict(raw, *, reference):
    if type(raw) is not dict or type(raw.get("geometry_enclosure_precision")) is not int:
        raise ValueError("Current-cut loader requires complete literal source-bound JSON")
    expected=bind_common_contact_cuts(reference,precision=raw["geometry_enclosure_precision"])
    if _encoded(raw)!=_encoded(common_contact_cuts_to_dict(expected)):
        raise ValueError("Saved current cuts differ from independently rebuilt source/masks/history")
    return expected


def common_contact_cut_from_dict(raw, *, reference):
    if (type(raw) is not dict or type(raw.get("cut_id")) is not str or
            type(raw.get("geometry_enclosure_precision")) is not int):
        raise ValueError("Current-cut loader requires a literal complete source receipt")
    # Frames use a source-required floor of320 digits independently of a
    # collection's requested budget. All requests below that floor rebuild
    # exactly the same cut receipt; use the ordinary default for that case.
    frame_precision=raw["geometry_enclosure_precision"]
    candidates=bind_common_contact_cuts(reference,precision=160 if frame_precision==320 else frame_precision).cuts
    expected=next((c for c in candidates if c.cut_id==raw["cut_id"]),None)
    if expected is None or _encoded(raw)!=_encoded(common_contact_cut_to_dict(expected)):
        raise ValueError("Saved current cut differs from independently rebuilt source/masks/history")
    return expected


@dataclass(frozen=True, slots=True)
class NativeCommonContactCutBounds:
    cut: NativeCommonContactCut
    parameter_bounds: I
    parameter_expression: str
    source_unit_world_direction_bounds: tuple
    source_unit_world_direction_parameter_derivative_bounds: tuple
    source_angular_speed_bounds: I
    donor_original_unit_direction_bounds: tuple
    receiver_original_unit_direction_bounds: tuple | None
    receiver_owner_local_unit_direction_bounds: tuple
    donor_material_depth_bounds_m: I
    receiver_material_depth_bounds_m: I
    common_physical_radius_cubed_bounds_m3: I
    physical_radius_bounds_m: I
    common_physical_volume_depth_bounds_m: I
    world_position_bounds_m: tuple
    donor_world_position_bounds_m: tuple
    receiver_world_position_bounds_m: tuple
    parameter_tangent_bounds_m: tuple
    material_depth_tangent_bounds: tuple
    angular_unit_tangent_bounds: tuple
    receiver_inward_unit_normal_bounds: tuple
    inward_unit_down_direction_bounds: tuple
    area_density_bounds_m2_per_parameter_per_material_m: I
    receiving_native_angle_bounds_radians: I | None
    receiving_computational_angle_bounds_radians: I | None
    genuine_query_support_certified: bool
    every_point_of_parameter_enclosure_is_material: bool
    original_quarter_constraints_refined_from_true_cut_proof: bool
    geometry_enclosure_precision: int
    rotation_order: int
    bounds_snapshot_sha256: str

    @property
    def reference_faces_match_at_every_common_physical_radius(self):
        return self.donor_world_position_bounds_m==self.receiver_world_position_bounds_m

    @property
    def constitutive_rest_contact_reference_recovered(self):
        return False

    @property
    def old_mechanical_atom_ids_assigned(self):
        return False

    @property
    def physical_material_admission_authorized(self):
        return False


def _refine_original_quarter(runtime, parent_id, q):
    source=runtime["before"]["snapshot"]["current_material_epoch"]["source_material_state"]["journal"]["source_surface"]
    parent=next(p for p in source["fragments"] if p["fragment_id"]==parent_id)
    result=list(q)
    for plane in parent["geometry"]["supports"]:
        axis=next(i for i,x in enumerate(plane["normal"]) if x)
        sign=plane["normal"][axis]
        margin=sign*result[axis]
        if margin.hi<0:
            raise ValueError("A purported actual source cut lies outside its original material quarter")
        margin=I(max(F(0),margin.lo),margin.hi)
        result[axis]=sign*margin
    return tuple(result)


def _receiver_coordinate(runtime, cut, q, precision):
    from .geometric_native_physical_pair_time import certify_native_quarter_carrier_coordinate
    trace=next(t for t in runtime["before"]["snapshot"]["current_material_epoch"]["source_material_state"]
        ["original_mechanics_payload"]["fault_traces"]["traces"] if t["trace_id"]==cut.receiving_trace_id)
    start,end=tuple(F.from_float(float(x)) for x in trace["start"]),tuple(F.from_float(float(x)) for x in trace["end"])
    cosine,sine=_dot(tuple(I(x) for x in start),q),_dot(tuple(I(x) for x in end),q)
    if cosine.lo==sine.lo==0:
        # A whole quarter box contains the zero rectangle corner, whereas the
        # authenticated source direction is a unit carrier. Keep the complete
        # quarter enclosure instead of refusing that correlated identity.
        theta=I(0,interval_pi(precision+8).hi/2)
    else:
        theta=certify_native_quarter_carrier_coordinate(start,end,q,precision=precision)
    # The public coordinate converter needs the authenticated time-path object.
    # Its source formula is exact linear rescaling of the declared original
    # computational extent by the native half-pi, not a Gauss-label replacement.
    path=runtime["before"]["original_physical_history"]["contact_time_path"]
    memory=path["step"]["previous_memory"]["traces"]
    extent=F.from_float(next(m["extent_radians"] for m in memory if m["trace_id"]==cut.receiving_trace_id))
    half=interval_pi(precision+8)/2
    computational=theta*extent/half
    return theta,I(max(F(0),computational.lo),min(extent,computational.hi))


def evaluate_common_contact_cut(reference, cut, parameter_bounds=None,
        physical_radius_cubed_bounds_m3=None, *, precision=160, order=28):
    """Enclose actual full-depth boundary geometry and both material labels.

    ``None`` encloses the entire actual cut, including uncertain endpoint bands.
    An interval wholly inside the certified parameter core is a genuine query.
    ``'start'``/``'end'`` bind the exact correlated source endpoint expression;
    its interval enclosure is never claimed to consist entirely of material.
    """
    reference=validate_native_common_reference(reference)
    _budget(precision,order)
    cut=validate_common_contact_cut(cut,reference=reference)
    runtime=_RUNTIME[cut.cut_snapshot_sha256]
    endpoint=parameter_bounds if type(parameter_bounds) is str else None
    if endpoint not in (None,"start","end"):
        raise ValueError("A cut endpoint query must name its actual source start or end")
    if endpoint is not None:
        parameter=(cut.maximum_ratio_time_fraction_bounds if endpoint=="start" and cut.kind=="true_curved_envelope"
                   else I(0) if endpoint=="start" else cut.parameter_endpoint_bounds)
    else:
        parameter=cut.parameter_bounds if parameter_bounds is None else parameter_bounds
    if (type(parameter) is not I or not cut.parameter_bounds.lo<=parameter.lo<=parameter.hi<=cut.parameter_bounds.hi):
        raise ValueError("A query lies outside the complete actual cut parameter enclosure")
    cubes=cut.common_physical_radius_cubed_bounds_m3 if physical_radius_cubed_bounds_m3 is None else physical_radius_cubed_bounds_m3
    whole=cut.common_physical_radius_cubed_bounds_m3
    if type(cubes) is not I or not whole.lo<=cubes.lo<=cubes.hi<=whole.hi:
        raise ValueError("A cut query lies outside both genuine physical material depths")
    if cut.kind=="true_curved_envelope":
        genuine=endpoint is not None or parameter.lo>=cut.maximum_ratio_time_fraction_bounds.hi
        U,Up,N,speed=_curve_geometry(runtime,parameter,precision,endpoint=endpoint)
    else:
        from .geometric_native_active_contact_line_work import _trig_ranges_signed
        sine,cosine=_trig_ranges_signed(parameter.lo,parameter.hi,precision)
        S,T=cut.start_unit_bounds,cut.tangent_unit_bounds
        U=tuple(s*cosine+t*sine for s,t in zip(S,T))
        Up=tuple(-s*sine+t*cosine for s,t in zip(S,T))
        N=_cross(S,T)
        speed=I(1)
        genuine=endpoint is not None or parameter.hi<=cut.parameter_endpoint_bounds.lo
        if endpoint=="end":
            if cut.kind=="implicit_terminal_trim":
                U,_,_,_=_curve_geometry(runtime,I(1),precision)
            else:
                arc=json.loads(cut.genuine_native_arc_json)
                from .geometric_native_polygons import native_ray_from_dict
                v=_ray_values(native_ray_from_dict(arc["end"]),cut.geometry_enclosure_precision)[1]
                norm=interval_sqrt(_dot(v,v),precision)
                U=tuple(x/norm for x in v)
            # At the exact native endpoint use its correlated plane direction.
            Up=_cross(N,U)
    db=bind_native_common_reference_bank(reference,cut.donor_component_id)
    rb=bind_native_common_reference_bank(reference,cut.receiver_component_id)
    qd=_inverse_map(runtime["poses"][db.owner],U)
    qr=_inverse_map(runtime["poses"][rb.owner],U)
    if genuine:
        qd=_refine_original_quarter(runtime,db.original_parent_id,qd)
        if rb.material_kind=="original_live":
            qr=_refine_original_quarter(runtime,rb.original_parent_id,qr)
    R,A=reference.radius_m,reference.angular_rate_per_physical_volume_depth_m
    r=interval_cbrt(cubes,precision)
    tau=(I(R**3)-cubes)/(3*R**2)
    # N cross U is the unit tangent analytically. This retains the correlation
    # when curved speed varies by many orders of magnitude across a whole box.
    T=_cross(N,U)
    c=cubes/R**2*_dot(tuple(I(a) for a in A),T)
    metric=interval_sqrt(I(1)+_square(c),precision)
    J=speed*R**2/r*metric
    if J.lo<=0:
        raise ValueError("The actual mapped boundary metric lost strict positive regularity")
    rotate=lambda x:_rotate_common(x,A,tau,precision,order)
    X=tuple(_round(r*x,precision) for x in rotate(U))
    Xp=tuple(_round(r*x,precision) for x in rotate(Up))
    Xt=tuple(_round(x,precision) for x in rotate(tuple(-R**2/(r*r)*u+r*a for u,a in zip(U,_cross(tuple(I(a) for a in A),U)))))
    normal=tuple(_round(x/metric,precision) for x in rotate(tuple(cut.receiving_normal_orientation*(n-c*u) for n,u in zip(N,U))))
    down=tuple(_round(x/metric,precision) for x in rotate(tuple(-u-c*n for u,n in zip(U,N))))
    angular=tuple(_round(cut.receiving_normal_orientation*x,precision) for x in rotate(T))
    zd=(I((R+db.roof_radial_displacement_m)**3)-cubes)/(3*R**2)
    zr=(I((R+rb.roof_radial_displacement_m)**3)-cubes)/(3*R**2)
    pd=bound_native_common_reference_bank_point(db,U,zd,precision=precision,order=order)
    pr=bound_native_common_reference_bank_point(rb,U,zr,precision=precision,order=order)
    if pd.world_position_bounds_m!=pr.world_position_bounds_m or pd.world_position_bounds_m!=X:
        raise ValueError("Actual source cut failed common-radius physical-depth gluing")
    theta=computational=None
    if cut.receiving_trace_id is not None and genuine:
        theta,computational=_receiver_coordinate(runtime,cut,qr,precision)
    result=NativeCommonContactCutBounds(cut,parameter,"source_"+endpoint if endpoint else "interval",
        tuple(_round(x,precision) for x in U),tuple(_round(x,precision) for x in Up),_round(speed,precision),
        tuple(_round(x,precision) for x in qd),tuple(_round(x,precision) for x in qr) if rb.material_kind=="original_live" else None,
        tuple(_round(x,precision) for x in qr),zd,zr,cubes,r,tau,X,X,X,Xp,Xt,angular,normal,down,_round(J,precision),
        theta,computational,genuine,endpoint is None and genuine,genuine,precision,order,"")
    result=replace(result,bounds_snapshot_sha256=_seal(result,"bounds_snapshot_sha256"))
    _BOUNDS[result.bounds_snapshot_sha256]=_record(result)
    return result


def validate_common_contact_cut_bounds(value, *, reference=None):
    if (type(value) is not NativeCommonContactCutBounds or
            _BOUNDS.get(value.bounds_snapshot_sha256)!=_record(value) or
            value.bounds_snapshot_sha256!=_seal(value,"bounds_snapshot_sha256")):
        raise ValueError("Evaluated cut geometry was not built from an authenticated parameter/source expression")
    validate_common_contact_cut(value.cut,reference=reference)
    return value


def common_contact_cut_bounds_to_dict(value):
    validate_common_contact_cut_bounds(value)
    return dict(format=FORMAT,scope=SCOPE,**_payload(value))


def common_contact_cut_bounds_from_dict(raw, *, reference):
    if type(raw) is not dict or type(raw.get("cut")) is not dict:
        raise ValueError("Evaluated cut geometry needs a complete literal source receipt")
    cut_raw=dict(format=FORMAT,scope=SCOPE,**raw["cut"])
    cut=common_contact_cut_from_dict(cut_raw,reference=reference)
    expression=raw.get("parameter_expression")
    if expression not in ("interval","source_start","source_end"):
        raise ValueError("Saved geometry changed its exact source parameter expression")
    parameter=expression[7:] if expression.startswith("source_") else _read_interval(raw["parameter_bounds"])
    expected=evaluate_common_contact_cut(reference,cut,parameter,
        _read_interval(raw["common_physical_radius_cubed_bounds_m3"]),
        precision=raw["geometry_enclosure_precision"],order=raw["rotation_order"])
    if _encoded(raw)!=_encoded(common_contact_cut_bounds_to_dict(expected)):
        raise ValueError("Evaluated geometry differs from independently rebuilt common map and true source cut")
    return expected
