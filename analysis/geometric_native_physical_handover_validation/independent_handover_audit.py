"""Standard-library audit of saved, unaccepted material-port handover evidence.

The prior source histories and actual force candidate are external authorities.
Exact rational checks cover source-shape and reference-transport algebra;
independent Decimal geometry checks are explicitly numerical point checks.
Nothing here grants native support, adopts candidate memory or solves force.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import product
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
PRECISION = 360
PHYSICAL = ROOT / "analysis/geometric_native_physical_binding_validation"
MATERIAL = ROOT / "analysis/geometric_native_first_entry_validation"
BASE = Path(__file__).resolve().parent
PRIOR_PRIMARY_SHA256 = {
    "analysis/geometric_native_physical_binding_validation/release_validation_01/validation_summary.json":
        "71ca5f59bdbd8ed52ed5eb70aac65db573040df08cd4a971efbdc0f5404dac7f",
    "analysis/geometric_native_physical_binding_validation/release_validation_01/artifact_manifest.json":
        "e04a6df8cdd7be689f45dcb6510a56bb743a93ff296605f07e09579982199204",
    "analysis/geometric_native_first_entry_validation/release_validation_02/validation_summary.json":
        "2f077013f0a514c8788d0210d708ef6189f460cb6c3978cb83bfe9eb926b99cc",
    "analysis/geometric_native_first_entry_validation/release_validation_02/artifact_manifest.json":
        "e871a378b3cf4c0a30d0e5e89c4efd04f9b6c21fa4b3ec828c1630e5e9217d7a",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def keys(raw, names, label):
    require(type(raw) is dict and set(raw) == set(names.split()), label + " schema changed")


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(path.read_bytes(), object_pairs_hook=unique, parse_constant=nonfinite)


def same(first, second, label):
    require(encoded(first) == encoded(second), label + " changed literal source payload")


def seal(raw, key, label):
    require(type(raw) is dict and key in raw, label + " lacks " + key)
    body = {name:value for name,value in raw.items() if name != key}
    require(digest(body) == raw[key], label + " selfseal mismatch")
    return body


def fraction(raw):
    require(type(raw) is list and len(raw) == 2 and all(type(v) is int for v in raw)
            and raw[1] > 0, "Expected literal rational pair")
    value = Fraction(*raw)
    require([value.numerator,value.denominator] == raw, "Noncanonical rational pair")
    return value


def exact_float(value):
    require(type(value) in (int,float) and math.isfinite(value), "Nonfinite or nonliteral source number")
    return Fraction(value)


def rational(value):
    return [value.numerator,value.denominator]


def interval(raw):
    require(type(raw) is dict and set(raw) == {"lo","hi"}, "Expected rational interval")
    low, high = fraction(raw["lo"]),fraction(raw["hi"])
    require(low <= high, "Reversed rational interval")
    return low, high


def vector(raw, size=3):
    require(type(raw) is list and len(raw) == size, "Wrong rational vector shape")
    return tuple(fraction(v) for v in raw)


def ivalue(raw):
    if type(raw) is dict:
        return interval(raw)
    value = fraction(raw)
    return value,value


def iadd(first, second):
    return first[0]+second[0],first[1]+second[1]


def ineg(value):
    return -value[1],-value[0]


def isub(first, second):
    return iadd(first,ineg(second))


def imul(first, second):
    values = [a*b for a in first for b in second]
    return min(values),max(values)


def idiv(first, second):
    require(not second[0] <= 0 <= second[1], "Interval division by an unresolved zero")
    return imul(first,(1/second[1],1/second[0]))


def iscale(value, scalar):
    return imul(value,(scalar,scalar))


def isquare(value):
    squares=[v*v for v in value]
    return (Fraction(0) if value[0] <= 0 <= value[1] else min(squares)),max(squares)


def encloses(raw, expected, label):
    low, high = interval(raw)
    require(low <= expected[0] <= expected[1] <= high, label + " fails independent exact enclosure")


def dec(value):
    if type(value) is Fraction:
        return Decimal(value.numerator)/Decimal(value.denominator)
    if type(value) is Decimal:
        return value
    return Decimal(value)


def dot(first, second):
    return sum((a*b for a,b in zip(first,second)),0)


def cross(a,b):
    return a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]


def sin_cos(value):
    require(abs(value) <= 4, "Independent trig budget exceeded")
    threshold = Decimal(10)**-(PRECISION-16)
    result=[]
    for sine in (True,False):
        term = value if sine else Decimal(1)
        total=term
        for index in range(10000):
            n=2*index+(2 if sine else 1)
            term=-term*value*value/Decimal(n*(n+1));total+=term
            if abs(term) < threshold:
                break
        else:
            raise ValueError("Independent trig series did not converge")
        result.append(total)
    return tuple(result)


def rotate(point, axis, instant):
    angle_vector=tuple(v*instant for v in axis)
    magnitude=dot(angle_vector,angle_vector).sqrt()
    if not magnitude:
        return point
    sine,cosine=sin_cos(magnitude)
    direction=tuple(v/magnitude for v in angle_vector)
    perpendicular,projection=cross(direction,point),dot(direction,point)
    return tuple(v*cosine+t*sine+a*projection*(1-cosine)
                 for v,t,a in zip(point,perpendicular,direction))


def atan(value):
    sign=-1 if value < 0 else 1;value=abs(value);doubles=0
    while value > Decimal("0.2"):
        value=value/(1+(1+value*value).sqrt());doubles+=1
    term,total=value,value;threshold=Decimal(10)**-(PRECISION-16)
    for index in range(1,10000):
        term*=-value*value;addition=term/Decimal(2*index+1);total+=addition
        if abs(addition) < threshold:
            return sign*total*(2**doubles)
    raise ValueError("Independent atan series did not converge")


def half_pi():
    return (16*atan(Decimal(1)/5)-4*atan(Decimal(1)/239))/2


def atan2_positive(sine,cosine):
    require(sine >= 0 and cosine >= 0 and sine+cosine > 0, "Unresolved chart quadrant")
    return atan(sine/cosine) if cosine >= sine else half_pi()-atan(cosine/sine)


def contains(raw,value,label):
    low,high=interval(raw)
    epsilon=Decimal(10)**-(PRECISION-40)*max(Decimal(1),abs(value))
    require(dec(low)-epsilon <= value <= dec(high)+epsilon,
            label + " fails independent numerical containment")


def check_vector(raw,values,label):
    require(type(raw) is list and len(raw) == len(values), label + " vector shape changed")
    for index,(bound,value) in enumerate(zip(raw,values)):
        contains(bound,value,label + " component " + str(index))


def check_chart(raw, expected_source):
    keys(raw,"format model scope source source_sha256 inward material_labels_normalized native_bank_support_certified "
         "physical_pair_admission_authorized candidate_mechanics_adopted source_shape_is_continuum_interior_contact_solution "
         "accepted_native_force_intervals accepted_work_increment_j start_projection end_projection inward_projection "
         "in_plane_norm along_strike_angle_rad inward_angle_rad inward_distance_m x_over_width shape shape_slope "
         "shape_curvature shape_third_derivative displacement_m slope curvature_per_m third_derivative_per_m2 chart_sha256",
         "Source beam chart")
    seal(raw,"chart_sha256","Source beam chart")
    source=raw["source"]
    same(source,expected_source,"Authenticated source beam chart")
    require(raw["source_sha256"] == digest(source),"Beam chart source seal mismatch")
    require(raw["format"] == "native-source-beam-material-chart-1"
            and raw["model"] == "unchanged_source_cantilever_one_mode_Ritz_interior_coordinate"
            and raw["scope"] == "source_shape_only_no_support_or_material_handover_admission",
            "Source beam chart model/scope changed")
    for name in ("material_labels_normalized","native_bank_support_certified",
                 "physical_pair_admission_authorized","candidate_mechanics_adopted",
                 "source_shape_is_continuum_interior_contact_solution"):
        require(raw[name] is False,"Beam chart grants " + name)
    require(type(raw["accepted_native_force_intervals"]) is int and raw["accepted_native_force_intervals"] == 0
            and fraction(raw["accepted_work_increment_j"]) == 0,
            "Beam chart adopts native force/work")
    start,end=vector(source["start"]),vector(source["end"])
    require(dot(start,start) == dot(end,end) == 1 and dot(start,end) == 0,
            "Source bank is not the authenticated exact quarter trace")
    sign=-1 if source["bank"] == "sub" else 1
    require(source["bank"] in ("sub","over"),"Invalid source bank")
    inward=tuple(sign*v for v in cross(start,end))
    same(raw["inward"],[rational(v) for v in inward],"Source bank inward direction")
    points=source["reference_coordinates"]
    require(type(points) is list and len(points) == 3,"Source chart label shape changed")
    coordinates=tuple(ivalue(v) for v in points)
    require(source["reference_coordinates_kind"] in ("fraction","interval")
            and all(type(v) is (list if source["reference_coordinates_kind"] == "fraction" else dict)
                    for v in points),"Source chart label kind changed")
    def projection(axis):
        result=(Fraction(0),Fraction(0))
        for value,component in zip(axis,coordinates):
            result=iadd(result,iscale(component,value))
        return result
    projections=[projection(axis) for axis in (start,end,inward)]
    for name,expected in zip(("start_projection","end_projection","inward_projection"),projections):
        encloses(raw[name],expected,name)
    require(all(value[0] >= 0 for value in projections),"Source chart clips incompatible signs")
    radius,width=fraction(source["radius_m"]),fraction(source["width_m"])
    require(radius > 0 and width > 0 and dec(width/radius) < half_pi(),"Invalid source beam hemisphere/width")
    low,high=interval(raw["x_over_width"])
    require(0 <= low <= high <= 1,"Whole source chart coordinate lies outside original beam")
    x=interval(raw["inward_distance_m"])
    require(0 <= x[0] <= x[1] <= width,"Source chart narrowed an invalid source strip")
    encloses(raw["inward_distance_m"],iscale(interval(raw["inward_angle_rad"]),radius),"Source distance algebra")
    encloses(raw["x_over_width"],iscale(x,1/width),"Source dimensionless distance algebra")
    phi=lambda z:(1-z)*(1-z)*(z+2)/2
    derivative=lambda z:-Fraction(3,2)*(1-z)*(1+z)
    expected_shapes={"shape":(phi(high),phi(low)),"shape_slope":(derivative(low),derivative(high)),
        "shape_curvature":(3*low,3*high),"shape_third_derivative":(Fraction(3),Fraction(3))}
    for name,expected in expected_shapes.items():
        encloses(raw[name],expected,"Source cubic " + name)
    tip=interval(source["tip_displacement_m"])
    for name,shape,factor in (("displacement_m","shape",Fraction(1)),
            ("slope","shape_slope",1/width),("curvature_per_m","shape_curvature",1/width**2),
            ("third_derivative_per_m2","shape_third_derivative",1/width**3)):
        encloses(raw[name],iscale(imul(interval(raw[shape]),tip),factor),"Source cubic " + name)
    checks=0
    for corner in product(*[tuple(set(value)) for value in coordinates]):
        p=tuple(dec(v) for v in corner)
        a,b,c=(dot(tuple(dec(v) for v in axis),p) for axis in (start,end,inward))
        norm=(a*a+b*b).sqrt()
        require(norm > 0,"Source chart lacks independent in-plane norm")
        theta=atan2_positive(b,a);beta=atan(c/norm);distance=dec(radius)*beta;z=distance/dec(width)
        for name,value in (("in_plane_norm",norm),("along_strike_angle_rad",theta),
                ("inward_angle_rad",beta),("inward_distance_m",distance),("x_over_width",z),
                ("shape",1-Decimal("1.5")*z+z**3/2),
                ("shape_slope",Decimal("1.5")*(z*z-1)),("shape_curvature",3*z),
                ("shape_third_derivative",Decimal(3))):
            contains(raw[name],value,"Beam chart " + name)
        for amplitude in set(tip):
            w=dec(amplitude)
            for name,value in (("displacement_m",w*(1-Decimal("1.5")*z+z**3/2)),
                    ("slope",w*Decimal("1.5")*(z*z-1)/dec(width)),
                    ("curvature_per_m",w*3*z/dec(width)**2),
                    ("third_derivative_per_m2",3*w/dec(width)**3)):
                contains(raw[name],value,"Beam chart amplitude " + name)
        checks+=1
    return checks


def interval_cross(first,second):
    return (isub(imul(first[1],second[2]),imul(first[2],second[1])),
            isub(imul(first[2],second[0]),imul(first[0],second[2])),
            isub(imul(first[0],second[1]),imul(first[1],second[0])))


def check_operator(raw,position,frame,reference_dot,shape):
    keys(raw,"sub_angular over_angular sub_radial over_radial angular_coordinate_units radial_coordinate_units "
         "receiver_frame_couple_included","Material port operator")
    require(raw["angular_coordinate_units"] == "radian" and raw["radial_coordinate_units"] == "metre"
            and raw["receiver_frame_couple_included"] is True,
            "Material port operator dropped units or receiver-frame couple")
    lever=interval_cross(tuple(interval(v) for v in position),tuple(interval(v) for v in frame))
    for name,expected in (("sub_angular",tuple(ineg(v) for v in lever)),("over_angular",lever)):
        require(type(raw[name]) is list and len(raw[name]) == 3,"Material angular operator shape changed")
        for bound,value in zip(raw[name],expected):
            encloses(bound,value,"Material angular virtual-power operator")
    encloses(raw["sub_radial"],iscale(interval(shape),-reference_dot),"Own source beam radial coupling")
    encloses(raw["over_radial"],(reference_dot,reference_dot),"Original receiver radial coupling")


def check_virtual_power(port,values,local):
    """Check the finite-frame couple and source Ritz radial virtual power."""
    sub,over,normal,down,direction,shape=values
    tn,ts=dec(exact_float(local["traction_normal_pa"])),dec(exact_float(local["traction_shear_pa"]))
    force=tuple(tn*a+ts*b for a,b in zip(normal,down))
    separation=tuple(a-b for a,b in zip(over,sub))
    couple=cross(force,separation)
    motions=[((Fraction(1,7),Fraction(-2,9),Fraction(3,11)),
              (Fraction(-4,13),Fraction(5,17),Fraction(-6,19)),Fraction(7,23),Fraction(-8,29)),
             ((Fraction(0),Fraction(0),Fraction(1)),(Fraction(0),Fraction(0),Fraction(0)),Fraction(0),Fraction(0)),
             ((Fraction(0),Fraction(0),Fraction(0)),(Fraction(0),Fraction(0),Fraction(1)),Fraction(1),Fraction(2))]
    area=dec(exact_float(local["area_m2"]))
    for sub_w,over_w,sub_rate,over_rate in motions:
        ws,wo=tuple(dec(v) for v in sub_w),tuple(dec(v) for v in over_w)
        vs=tuple(a+shape*dec(sub_rate)*b for a,b in zip(cross(ws,sub),direction))
        vo=tuple(a+dec(over_rate)*b for a,b in zip(cross(wo,over),direction))
        contact_power=(dot(force,tuple(a-b for a,b in zip(vo,vs)))+dot(couple,wo))*area
        normal_power=dot(normal,tuple(a-b for a,b in zip(vo,vs)))+dot(separation,cross(wo,normal))
        down_power=dot(down,tuple(a-b for a,b in zip(vo,vs)))+dot(separation,cross(wo,down))
        expected=(tn*normal_power+ts*down_power)*area
        epsilon=Decimal(10)**-(PRECISION-40)*max(Decimal(1),abs(expected))
        require(abs(contact_power-expected) <= epsilon,"Material-port frame couple violates virtual power")
        # Check the saved covectors against the same independently computed
        # virtual motion. Containment retains production interval uncertainty.
        for raw,velocity,label in ((port["normal_operator"],normal_power,"normal"),
                                   (port["down_operator"],down_power,"down")):
            total=(Fraction(0),Fraction(0))
            for bounds,rates in ((raw["sub_angular"],sub_w),(raw["over_angular"],over_w)):
                for bound,rate in zip(bounds,rates):
                    total=iadd(total,iscale(interval(bound),rate))
            total=iadd(total,iscale(interval(raw["sub_radial"]),sub_rate))
            total=iadd(total,iscale(interval(raw["over_radial"]),over_rate))
            contains({"lo":rational(total[0]),"hi":rational(total[1])},velocity,"Saved " + label + " virtual power")
    return len(motions)


def tagged_geometry(raw):
    require(type(raw) is list and len(raw) == 2,"Projected contact geometry shape changed")
    result=[]
    for value in raw:
        require(type(value) is dict,"Projected contact geometry type tag missing")
        if value.get("kind") == "fraction" and set(value) == {"kind","value"}:
            item=fraction(value["value"]);result.append((item,item))
        elif value.get("kind") == "interval" and set(value) == {"kind","lo","hi"}:
            result.append(interval({"lo":value["lo"],"hi":value["hi"]}))
        else:
            raise ValueError("Projected contact geometry literal type changed")
    return tuple(result)


def tagged_intervals(values):
    return [{"kind":"interval","lo":rational(low),"hi":rational(high)} for low,high in values]


def check_transport(state,history,node,binding,law,old_geometry,new_geometry,context):
    keys(state,"format law source_history point_id trace_id source_epoch original_local_state original_columns "
         "original_contact_law original_water original_area_m2 original_banks initial_binding_id initial_geometry "
         "initial_geometry_context active_binding_id active_banks reference_geometry reference_geometry_context "
         "receipts revision consumed_binding_ids offset_expression anchor_expression receiver_frame_model "
         "geometric_correspondence_authenticated active_history_count work_lineage_id component_attachment_committed "
         "global_force_candidate_adopted nodal_work_assigned_to_material_volume consumption_scope state_sha256",
         "Contact reference transport")
    seal(state,"state_sha256","Contact reference transport")
    require(state["format"] == "native-contact-reference-transport-1","Unsupported reference transport format")
    same(state["source_history"],history,"Transport external source history")
    same(state["law"],law,"Explicit contact reference transport law")
    point_id=node["point_id"];local=node["unaccepted_candidate_state"]
    require(state["point_id"] == point_id and state["trace_id"] == node["trace_id"]
            and state["source_epoch"] == "unaccepted_candidate","Transport source identity/epoch changed")
    same(state["original_local_state"],local,"Every source LocalContactState field")
    material=node["original_material"]
    for name,expected in (("original_columns",material["columns"]),("original_contact_law",material["law"]),
            ("original_water",material["water"]),("original_area_m2",node["original_area_m2"])):
        same(state[name],expected,"Original transported parameter " + name)
    original_banks=dict(sub_fragment_id=binding["donor_fragment_id"],sub_material_id=binding["donor_material_id"],
        sub_owner=binding["subducting_plate"],over_fragment_id=binding["receiver_fragment_id"],
        over_material_id=binding["receiver_material_id"],over_owner=binding["overriding_plate"],
        correspondence=dict(model="original_authenticated_pair",trace_id=node["trace_id"],source_point_id=point_id,
                            receiver_axes="unchanged_original_receiver_material_axes"))
    target_banks=dict(original_banks,correspondence=context)
    same(state["original_banks"],original_banks,"Transport original banks")
    same(state["active_banks"],target_banks,"Transport target banks")
    source_id,target_id=point_id+":original",point_id+":incoming:1"
    require(state["initial_binding_id"] == source_id and state["active_binding_id"] == target_id
            and type(state["revision"]) is int and state["revision"] == 1
            and state["consumed_binding_ids"] == [source_id],
            "Transport source was duplicated or binding was reused")
    same(state["initial_geometry"],tagged_intervals(old_geometry),"Original projected tip geometry")
    same(state["reference_geometry"],tagged_intervals(new_geometry),"New projected material-port geometry")
    same(state["initial_geometry_context"],dict(context,attachment="original_pair"),"Initial geometry expression context")
    same(state["reference_geometry_context"],context,"Active geometry expression context")
    require(state["offset_expression"] == "a_new = original_free_anchor - h_new"
            and state["anchor_expression"] == "free_current = original_free_anchor + (h_current - h_new)"
            and state["receiver_frame_model"] == "unchanged_original_receiver_material_axes",
            "Transport reference/receiver-frame law changed")
    require(type(state["active_history_count"]) is int and state["active_history_count"] == 1
            and state["work_lineage_id"] == [history["original_checkpoint_sha256"],point_id]
            and state["component_attachment_committed"] is True
            and state["geometric_correspondence_authenticated"] is False
            and state["global_force_candidate_adopted"] is False
            and state["nodal_work_assigned_to_material_volume"] is False,
            "Reference transport reset, cloned or adopted its source history")
    require(state["consumption_scope"] == "one immutable lineage; assembled ledgers must deduplicate source histories",
            "Reference transport weakened the once-only ledger contract")
    require(type(state["receipts"]) is list and len(state["receipts"]) == 1,
            "Candidate attachment must contain exactly one transition")
    receipt=state["receipts"][0]
    expected=dict(revision=1,source_binding_id=source_id,target_binding_id=target_id,
        source_banks=original_banks,target_banks=target_banks,
        source_geometry=tagged_intervals(old_geometry),target_geometry=tagged_intervals(new_geometry),
        source_geometry_context=dict(context,attachment="original_pair"),target_geometry_context=context,
        original_state_sha256=digest(local),signed_external_work_j=[0,1],dissipation_j=[0,1],energy_change_j=[0,1],
        strain_identity="contact(g,j-p), column(g-g_free,j-j_free), p,c unchanged",
        source_binding_consumed=True,target_binding_active=True,original_work_lineage_count=1)
    same(receipt,expected,"Complete source/reference transition receipt")
    # At the declared event, h_current and h_new are the same authenticated
    # expression, so their cancellation is symbolic. A numerical interval
    # midpoint would not preserve this identity.
    free=tuple(exact_float(local[name]) for name in ("gap_free_m","jump_free_m"))
    require(all(type(value) is Fraction for value in free),"Source free anchors were re-evaluated")
    return tuple(state["work_lineage_id"])


def check_bundle(bundle,history,law,prior_queries):
    keys(bundle,"format model precision source_history transport_law time_fraction source_epoch ports retained_point_ids "
         "unchanged_global_beam_state original_interface_measure_preserved outgoing_active_attachments_replaced_once "
         "finite_measure_assigned_to_first_entry_point transported_candidate_is_force_equilibrium "
         "native_full_bank_and_clamp_support_certified candidate_mechanics_adopted physical_pair_admission_authorized "
         "transfer_external_work_j transfer_dissipation_j accepted_loaded_native_intervals accepted_source_work_j handover_sha256",
         "Candidate material-port handover")
    seal(bundle,"handover_sha256","Candidate material-port handover")
    require(bundle["format"] == "native-candidate-material-port-handover-1"
            and bundle["model"] == "receiver_attached_loaded_configuration_source_beam_Ritz_handover",
            "Unsupported candidate handover model")
    require(type(bundle["precision"]) is int and bundle["precision"] == 128
            and fraction(bundle["time_fraction"]) == 1 and bundle["source_epoch"] == "unaccepted_candidate",
            "Handover changed the literal candidate endpoint or proof budget")
    same(bundle["source_history"],history,"Handover external full source history")
    same(bundle["transport_law"],law,"Handover explicitly declared transport law")
    require(law["model"] == "receiver_attached_complete_lumped_contact_configuration"
            and type(law["provenance"]) is str and law["provenance"].strip() == law["provenance"]
            and bool(law["provenance"]),"Handover transport law lacks explicit provenance")
    for name in ("unchanged_global_beam_state","original_interface_measure_preserved",
                 "outgoing_active_attachments_replaced_once"):
        require(bundle[name] is True,"Handover lost " + name)
    for name in ("finite_measure_assigned_to_first_entry_point","transported_candidate_is_force_equilibrium",
                 "native_full_bank_and_clamp_support_certified","candidate_mechanics_adopted",
                 "physical_pair_admission_authorized"):
        require(bundle[name] is False,"Handover incorrectly grants " + name)
    for name in ("transfer_external_work_j","transfer_dissipation_j","accepted_loaded_native_intervals","accepted_source_work_j"):
        require(type(bundle[name]) is int and bundle[name] == 0,"Handover adopts work/force: " + name)
    nodes={node["point_id"]:node for node in history["quadrature_histories"]}
    require(len(nodes) == len(history["quadrature_histories"]) == 16,"Source original node cardinality changed")
    bindings={row["trace_id"]:row for row in history["material_state"]["trace_bindings"]}
    fragments={row["fragment_id"]:row for row in history["material_state"]["journal"]["source_surface"]["fragments"]}
    parameters={row["trace_id"]:row for row in history["accepted_before"]["initial_mechanics"]["bending_parameters"]}
    candidate=history["actual_candidate"]
    radial={row[0]:row[1:] for row in candidate["transverse_displacement_m"]}
    dt=exact_float(history["literal_dt_myr"])
    omega={owner:tuple(exact_float(v) for v in values) for owner,values in candidate["omega_rad_per_myr"]}
    axes={owner:tuple(dec(v*dt) for v in values) for owner,values in omega.items()}
    radius=exact_float(history["origin_surface"]["radius_km"])*1000
    endpoint_maps={record["point_id"]:record["query"]["tip_maps"] for record in prior_queries
                   if record["time_fraction"] == [1,1]}
    eligible=[];retained=[];numerical={}
    for identity,node in nodes.items():
        binding=bindings[node["trace_id"]];point=node["original_point"];material=node["original_material"]
        q,n,d=(tuple(exact_float(v) for v in point[name]) for name in ("position","normal","down_dip"))
        relative=rotate(rotate(tuple(dec(v) for v in q),axes[binding["overriding_plate"]],Decimal(1)),
                        tuple(-v for v in axes[binding["subducting_plate"]]),Decimal(1))
        start,end=(tuple(exact_float(v) for v in material[name]) for name in ("start","end"))
        inward=tuple(-v for v in cross(start,end));signed=dot(tuple(dec(v) for v in inward),relative)
        require(abs(signed) > Decimal(10)**-(PRECISION-40),"Handover direction is numerically unresolved")
        (eligible if signed > 0 else retained).append(identity)
        numerical[identity]=(q,n,d,relative,start,end,inward)
    require(len(eligible) == len(retained) == 8
            and len({nodes[identity]["trace_id"] for identity in eligible}) == 2,
            "Candidate handover eligibility differs from original full-vector geometry")
    require(type(bundle["ports"]) is list and [p["point_id"] for p in bundle["ports"]] == eligible
            and bundle["retained_point_ids"] == retained,"Candidate attachments lost once-only source coverage")
    lineages=set();chart_checks=power_checks=0
    for port in bundle["ports"]:
        keys(port,"point_id trace_id target_reference_coordinates target_material_margins target_chart source_geometry "
             "world_direction sub_position_m over_position_m normal down_direction normal_operator down_operator transport_state",
             "Candidate material port")
        identity=port["point_id"];node=nodes[identity];binding=bindings[node["trace_id"]]
        require(port["trace_id"] == node["trace_id"],"Handover port trace was reassigned")
        q,n,d,relative,start,end,inward=numerical[identity]
        check_vector(port["target_reference_coordinates"],relative,"Full Ω source-material pullback")
        coordinates=tuple(interval(v) for v in port["target_reference_coordinates"])
        distance=(Fraction(0),Fraction(0))
        for value,component in zip(inward,coordinates):distance=iadd(distance,iscale(component,value))
        require(distance[0] > 0,"Incoming port source-bank membership was narrowed or clipped")
        donor=fragments[binding["donor_fragment_id"]]
        require(donor["geometry"]["pose"] == [] and all(plane["rotations"] == [] for plane in donor["geometry"]["supports"]),
                "Original source bank geometry was posed or replaced")
        margins=[]
        for plane in donor["geometry"]["supports"]:
            value=(Fraction(0),Fraction(0))
            for a,b in zip(plane["normal"],coordinates):value=iadd(value,iscale(b,Fraction(a)))
            require(value[0] >= 0,"Incoming port lacks complete original source material support")
            margins.append(value)
        require(len(port["target_material_margins"]) == len(margins),"Source material margin count changed")
        for raw,value in zip(port["target_material_margins"],margins):encloses(raw,value,"Source fragment membership margin")
        receiver=fragments[binding["receiver_fragment_id"]]
        require(receiver["geometry"]["pose"] == []
                and all(plane["rotations"] == [] for plane in receiver["geometry"]["supports"]),
                "Original receiver bank geometry was posed or replaced")
        require(all(dot(tuple(Fraction(v) for v in plane["normal"]),q) >= 0
                    for plane in receiver["geometry"]["supports"]),"Original receiver label lacks source support")
        w_sub,w_over=(exact_float(v) for v in radial[node["trace_id"]])
        chart_source=dict(trace_id=node["trace_id"],bank="sub",start=[rational(v) for v in start],
            end=[rational(v) for v in end],reference_coordinates_kind="interval",
            reference_coordinates=port["target_reference_coordinates"],radius_m=rational(radius),
            width_m=rational(exact_float(parameters[node["trace_id"]]["sub"]["width_m"])),
            tip_displacement_m={"lo":rational(w_sub),"hi":rational(w_sub)},precision=128)
        chart_checks+=check_chart(port["target_chart"],chart_source)
        same(port["source_geometry"],endpoint_maps[identity],"Authenticated original endpoint tip geometry")
        seal(port["source_geometry"],"maps_sha256","Original source tip map")
        old_geometry=tuple(interval(port["source_geometry"][name]) for name in ("gap_m","jump_m"))
        world=rotate(tuple(dec(v) for v in q),axes[binding["overriding_plate"]],Decimal(1))
        normal=rotate(tuple(dec(v) for v in n),axes[binding["overriding_plate"]],Decimal(1))
        down=rotate(tuple(dec(v) for v in d),axes[binding["overriding_plate"]],Decimal(1))
        phi_low,phi_high=interval(port["target_chart"]["shape"])
        beta=atan(dot(tuple(dec(v) for v in inward),relative)/(
            dot(tuple(dec(v) for v in start),relative)**2+dot(tuple(dec(v) for v in end),relative)**2).sqrt())
        z=dec(radius)*beta/dec(fraction(chart_source["width_m"]))
        phi=1-Decimal("1.5")*z+z**3/2
        sub=tuple((dec(radius)+phi*dec(w_sub))*v for v in world)
        over=tuple((dec(radius)+dec(w_over))*v for v in world)
        for name,values in (("world_direction",world),("normal",normal),("down_direction",down),
                           ("sub_position_m",sub),("over_position_m",over)):
            check_vector(port[name],values,"Material port " + name)
        displacement=interval(port["target_chart"]["displacement_m"])
        difference=isub((w_over,w_over),displacement)
        new_geometry=(iscale(difference,dot(q,n)),iscale(difference,dot(q,d)))
        check_operator(port["normal_operator"],port["sub_position_m"],port["normal"],dot(q,n),port["target_chart"]["shape"])
        check_operator(port["down_operator"],port["sub_position_m"],port["down_direction"],dot(q,d),port["target_chart"]["shape"])
        context=dict(coordinate_model="exact_analytic_owner_pullback_expression",time_fraction=[1,1],
            original_receiver_label=[rational(v) for v in q],donor_owner=binding["subducting_plate"],
            receiver_owner=binding["overriding_plate"],owner_motion=[[owner,[rational(v) for v in values]] for owner,values in omega.items()],
            duration_myr=rational(dt),target_label_expression="Q_sub(1)^-1 Q_over(1) original_receiver_label",
            target_label_evaluated_once_at_transfer=True,material_label_frozen_during_subsequent_force_variations=True,
            target_label_enclosure=port["target_reference_coordinates"],source_beam_chart=port["target_chart"],
            original_interface_point_id=identity,original_interface_area_m2=node["original_area_m2"],
            source_epoch="unaccepted_candidate",candidate_mechanics_adopted=False)
        lineage=check_transport(port["transport_state"],history,node,binding,law,old_geometry,new_geometry,context)
        require(lineage not in lineages,"Candidate handover cloned a source energy/work lineage")
        lineages.add(lineage)
        power_checks+=check_virtual_power(port,(sub,over,normal,down,world,phi),node["unaccepted_candidate_state"])
    # Every active or retained attachment uses exactly one old interface area,
    # actual LocalContactState and source beam basis. No area is attributed to
    # an off-quadrature first-entry point or invented from incoming volume.
    active=[p["transport_state"]["original_local_state"] for p in bundle["ports"]]
    active.extend(nodes[identity]["unaccepted_candidate_state"] for identity in retained)
    source_states=[node["unaccepted_candidate_state"] for node in nodes.values()]
    for key in ("area_m2","bulk_energy_j","contact_energy_j","cumulative_external_work_j",
                "cumulative_friction_work_j","cumulative_viscous_work_j","cumulative_numerical_dissipation_j"):
        require(sum((exact_float(s[key]) for s in active),Fraction(0))
                == sum((exact_float(s[key]) for s in source_states),Fraction(0)),
                "Assembled handover ledger duplicated or lost " + key)
    return dict(original_node_count=16,transported_port_count=8,retained_port_count=8,
        eligible_trace_count=2,numerical_pullback_checks=16,numerical_chart_corner_checks=chart_checks,
        virtual_power_checks=power_checks,exact_local_state_and_once_only_work_ledger_preserved=True,
        source_nonzero_plastic_count=sum(s["plastic_slip_m"] != 0 for s in source_states),
        transported_nonzero_plastic_count=sum(p["transport_state"]["original_local_state"]["plastic_slip_m"] != 0 for p in bundle["ports"]),
        source_positive_cumulative_slip_count=sum(s["cumulative_slip_m"] > 0 for s in source_states))


def check_once_ledger(raw,history,bundle):
    keys(raw,"format original_node_count transported_port_count retained_port_count eligible_source_fragment_pairs "
         "quadrature_histories total_original_area_m2 transported_area_m2 retained_area_m2 original_local_energy_work_totals_j "
         "local_energy_work_counted_once source_work_before unaccepted_candidate_work beam_configuration_before beam_configuration_after "
         "unchanged_global_beam_dofs_and_stiffness positive_cumulative_slip_count nonzero_signed_plastic_slip_count "
         "positive_compressive_pressure_count signed_plastic_memory_preserved transfer_external_work_j transfer_dissipation_j "
         "accepted_loaded_native_intervals accepted_source_work_j ledger_sha256","Once-only quadrature ledger")
    require(raw["format"] == "native-material-port-handover-once-only-ledger-1",
            "Unsupported handover once-only ledger format")
    nodes={node["point_id"]:node for node in history["quadrature_histories"]}
    ports={port["point_id"]:port for port in bundle["ports"]}
    bindings={row["trace_id"]:row for row in history["material_state"]["trace_bindings"]}
    require(raw["original_node_count"] == 16 and raw["transported_port_count"] == raw["retained_port_count"] == 8,
            "Once-only interface ledger cardinality changed")
    pairs=sorted({(bindings[port["trace_id"]]["donor_fragment_id"],bindings[port["trace_id"]]["receiver_fragment_id"])
                  for port in ports.values()})
    require(pairs == [("octant:6","octant:2"),("octant:7","octant:3")]
            and raw["eligible_source_fragment_pairs"] == [list(pair) for pair in pairs],
            "Candidate handover changed original bank pairs")
    require(type(raw["quadrature_histories"]) is list
            and [row["point_id"] for row in raw["quadrature_histories"]] == list(nodes),
            "Once-only ledger lost original interface ordering/coverage")
    total=transported=retained=Fraction(0);aggregates={}
    for row in raw["quadrature_histories"]:
        identity=row["point_id"];node=nodes[identity];local=node["unaccepted_candidate_state"]
        binding=bindings[node["trace_id"]];area=exact_float(node["original_area_m2"])
        require(row["trace_id"] == node["trace_id"]
                and row["disposition"] == ("transported_once" if identity in ports else "retained_once")
                and row["source_fragment_pair"] == [binding["donor_fragment_id"],binding["receiver_fragment_id"]]
                and fraction(row["original_area_m2"]) == area
                and row["local_state_fields_identical"] is True,
                "Once-only ledger source identity/measure/disposition changed")
        same(row["source_local_state"],local,"Ledger original LocalState")
        same(row["transported_local_state"],local,"Ledger retained complete LocalState")
        require(local["area_m2"] == node["original_area_m2"]
                and local["plastic_slip_m"] != 0 and local["cumulative_slip_m"] > 0
                and local["traction_normal_pa"] < 0,
                "Loaded original plastic/preparation history was reset")
        total+=area
        if identity in ports:transported+=area
        else:retained+=area
        for name,value in local.items():
            if name.endswith("_j"):aggregates[name]=aggregates.get(name,Fraction(0))+exact_float(value)
    require(total == transported+retained
            and fraction(raw["total_original_area_m2"]) == total
            and fraction(raw["transported_area_m2"]) == transported
            and fraction(raw["retained_area_m2"]) == retained,"Original interface measure was duplicated/lost")
    same(raw["original_local_energy_work_totals_j"],{key:rational(value) for key,value in aggregates.items()},
         "Complete once-only original energy/work totals")
    same(raw["source_work_before"],history["before_work_ledgers"],"Source prepared global work ledger")
    same(raw["unaccepted_candidate_work"],history["unaccepted_candidate_work_ledgers"],"Source unaccepted global work ledger")
    beam=dict(parameters=history["accepted_before"]["initial_mechanics"]["bending_parameters"],
              candidate_transverse_displacement_m=history["actual_candidate"]["transverse_displacement_m"],
              original_stiffness=history["original_reference"]["stiffness"])
    same(raw["beam_configuration_before"],beam,"Original global beam DOFs/stiffness")
    same(raw["beam_configuration_after"],beam,"Unchanged global beam DOFs/stiffness")
    for name in ("local_energy_work_counted_once","unchanged_global_beam_dofs_and_stiffness","signed_plastic_memory_preserved"):
        require(raw[name] is True,"Once-only ledger lost " + name)
    for name in ("positive_cumulative_slip_count","nonzero_signed_plastic_slip_count","positive_compressive_pressure_count"):
        require(type(raw[name]) is int and raw[name] == 16,"Original loaded history count changed: " + name)
    for name in ("transfer_external_work_j","transfer_dissipation_j","accepted_loaded_native_intervals","accepted_source_work_j"):
        require(type(raw[name]) is int and raw[name] == 0,"Once-only ledger adopts work/native interval")
    return len(aggregates)


def relative_file(root,name):
    require(type(name) is str and bool(name) and not Path(name).is_absolute(),"Artifact path is not relative")
    path=(root/name).resolve()
    require(path.is_relative_to(root.resolve()) and path.is_file(),"Artifact escaped its evidence root or is missing: " + name)
    require(path.relative_to(root.resolve()).as_posix() == name,"Noncanonical artifact path: " + name)
    return path


def verify_prior(raw):
    seal(raw,"receipt_sha256","Handover source validation")
    keys(raw,"format status original_source_count prior_physical_artifact_count prior_material_artifact_count "
         "prior_physical_tests_passed prior_material_tests_passed prior_tests_reexecuted source_artifacts receipt_sha256",
         "Handover source validation")
    require(raw["format"] == "native-material-port-handover-source-validation-1" and raw["status"] == "PASS"
            and raw["prior_tests_reexecuted"] is False,"Prior test/source scope changed")
    for name,value in (("original_source_count",664),("prior_physical_artifact_count",56),
            ("prior_material_artifact_count",83),("prior_physical_tests_passed",253),("prior_material_tests_passed",770)):
        require(type(raw[name]) is int and raw[name] == value,"Prior count changed: " + name)
    paths=(PHYSICAL/"release_validation_01/validation_summary.json",PHYSICAL/"release_validation_01/artifact_manifest.json",
        PHYSICAL/"release_gates_01/receipt.json",PHYSICAL/"control_release_01/report.json",
        PHYSICAL/"control_release_01/independent_binding_receipt.json",PHYSICAL/"control_release_01/original_history.json",
        MATERIAL/"release_validation_02/validation_summary.json",MATERIAL/"release_validation_02/artifact_manifest.json",
        MATERIAL/"control_release_02/original_force/initial.json",MATERIAL/"control_release_02/original_force/native_source.json",
        MATERIAL/"control_release_02/original_force/force_candidate.json",MATERIAL/"control_release_02/material_state.json",
        MATERIAL/"control_release_02/point_certificates.json",BASE/"starting_sources.json")
    require(set(raw["source_artifacts"]) == {path.relative_to(ROOT).as_posix() for path in paths},
            "Complete independently authenticated source reference set changed")
    for name,record in raw["source_artifacts"].items():
        keys(record,"sha256 bytes","Prior source artifact reference")
        path=relative_file(ROOT,name)
        require(type(record["bytes"]) is int and path.stat().st_size == record["bytes"]
                and file_sha(path) == record["sha256"],"Prior source bytes/hash differ: " + name)
    verified=0;prior_hashes={}
    for directory,count,scope in ((PHYSICAL/"release_validation_01",56,"physical"),
                                   (MATERIAL/"release_validation_02",83,"material")):
        summary_path=directory/"validation_summary.json";manifest_path=directory/"artifact_manifest.json"
        for path in (summary_path,manifest_path):
            require(file_sha(path) == PRIOR_PRIMARY_SHA256[path.relative_to(ROOT).as_posix()],
                    "Previously independently verified primary authority bytes changed")
        summary=read(summary_path);manifest=read(manifest_path)
        seal(summary,"summary_sha256",scope + " prior summary")
        seal(manifest,"manifest_sha256",scope + " prior manifest")
        require(summary["status"] == "PASS" and type(manifest["count"]) is int
                and manifest["count"] == len(manifest["artifacts"]) == count,"Prior authoritative release count/status changed")
        same(summary["artifact_manifest"],dict(path=manifest_path.relative_to(ROOT).as_posix(),sha256=file_sha(manifest_path),count=count),
             "Prior summary manifest reference")
        for name,record in manifest["artifacts"].items():
            keys(record,"sha256 bytes","Prior manifest record")
            path=relative_file(ROOT,name)
            require(type(record["bytes"]) is int and record["bytes"] == path.stat().st_size
                    and record["sha256"] == file_sha(path),"Prior manifest artifact changed: " + name)
            verified+=1
        reference_keys=("fresh_scoped_gate","actual_control","independent_audit") if scope == "physical" else (
                       "release_gate","actual_control","independent_audit")
        for name in reference_keys:
            reference=summary[name];path=relative_file(ROOT,reference["path"])
            require(file_sha(path) == reference["sha256"],"Prior summary evidence hash changed: " + name)
            value=read(path)
            # The archived material gate predates receipt selfseals. Its raw
            # bytes are authenticated by both the summary and full manifest.
            if not (scope == "material" and name == "release_gate"):
                seal(value,"report_sha256" if name == "actual_control" else "receipt_sha256",scope + " " + name)
            require(value["status"] == "PASS","Prior evidence status changed")
        for path,value,seal_name in ((summary_path,summary,"summary_sha256"),(manifest_path,manifest,"manifest_sha256")):
            prior_hashes[path.relative_to(ROOT).as_posix()]=dict(sha256=file_sha(path),selfseal=value[seal_name],bytes=path.stat().st_size)
        if scope == "physical":
            gate=read(PHYSICAL/"release_gates_01/receipt.json")
            require(summary["source_count"] == 664 and summary["fresh_scoped_gate"]["tests_passed"] == 253
                    and summary["fresh_scoped_gate"]["test_files"] == 5
                    and gate["counts"] == {"passed":253} and len(gate["test_paths"]) == 5
                    and gate["original_770_tests_reexecuted"] is False and gate["is_full_profile_execution"] is False
                    and summary["prior_authoritative_validation"]["tests_passed"] == 770
                    and summary["prior_authoritative_validation"]["tests_reexecuted"] is False,
                    "Prior 253 fresh/770 archived-only test ancestry changed")
            archived=summary["prior_authoritative_validation"]
            require(relative_file(ROOT,archived["path"]) == MATERIAL/"release_validation_02/validation_summary.json"
                    and file_sha(ROOT/archived["path"]) == archived["sha256"],"Prior archived material authority changed")
        else:
            gate=read(MATERIAL/"release_gates_02/receipt.json")
            require(summary["release_gate"]["tests_passed"] == 770 and gate["counts"] == {"passed":770}
                    and len(gate["test_paths"]) == 20 and gate["is_full_profile_execution"] is False,
                    "Prior archived material gate count/scope changed")
    return dict(prior_artifacts_verified=verified,prior_authoritative_primary_artifacts=prior_hashes,
                prior_physical_tests_passed=253,prior_material_tests_passed=770,prior_tests_reexecuted=False)


def check_observations(raw,bundle):
    keys(raw,"format count observations receipt_sha256","Transfer reference observations")
    seal(raw,"receipt_sha256","Transfer reference observations")
    require(raw["format"] == "native-material-port-transfer-reference-observations-1"
            and type(raw["count"]) is int and raw["count"] == 8
            and type(raw["observations"]) is list and len(raw["observations"]) == 8,
            "Transfer reference observation count changed")
    for row,port in zip(raw["observations"],bundle["ports"]):
        local=port["transport_state"]["original_local_state"]
        anchors=[rational(exact_float(local[name])) for name in ("gap_free_m","jump_free_m")]
        expected=dict(point_id=port["point_id"],source_free_anchor=anchors,
            anchors_at_transfer=[dict(lo=value,hi=value) for value in anchors],
            source_reference_geometry_context=port["transport_state"]["reference_geometry_context"],
            correlated_geometry_cancelled_symbolically=True,time_fraction=[1,1],source_epoch="unaccepted_candidate")
        same(row,expected,"Exact symbolic cancellation of identical projected geometry expressions")
    return 8


def check_mutation_receipts(raw):
    keys(raw,"format count mutations receipt_sha256","Recorded mutation refusals")
    seal(raw,"receipt_sha256","Recorded mutation refusals")
    cases=(("tiny_owner_axis_removed","source_history.actual_candidate.omega_rad_per_myr"),
        ("source_candidate_memory_changed","source_history.actual_candidate.states"),
        ("original_measure_doubled","ports[0].transport_state.original_area_m2"),
        ("source_beam_width_changed","ports[0].target_chart.source.width_m"),
        ("source_beam_amplitude_changed","ports[0].target_chart.source.tip_displacement_m"),
        ("source_beam_shape_changed","ports[0].target_chart.shape"),
        ("receiver_frame_couple_removed","ports[0].normal_operator.receiver_frame_couple_included"),
        ("receiver_angular_couple_changed","ports[0].normal_operator.over_angular"),
        ("duplicate_port","ports"),("duplicate_retained_history","retained_point_ids"),
        ("candidate_plastic_reset","ports[0].transport_state.original_local_state.plastic_slip_m"),
        ("candidate_work_reset","ports[0].transport_state.original_local_state.cumulative_external_work_j"),
        ("transport_law_provenance_changed","transport_law.provenance"),
        ("independent_complete_source_changed","expected_source_history_dto.actual_candidate.states"))
    require(raw["format"] == "native-material-port-handover-mutation-refusals-1" and type(raw["count"]) is int
            and raw["count"] == 14 and type(raw["mutations"]) is list and len(raw["mutations"]) == 14,
            "Recorded source/law/operator mutation coverage changed")
    for row,(label,path) in zip(raw["mutations"],cases):
        checksum="valid_rehashed_source_checksum" if label == "independent_complete_source_changed" else "valid_rehashed_handover_checksum"
        keys(row,"label changed_path rejected " + checksum + " exception_type message mutated_dto_sha256","Recorded mutation refusal")
        require(row["label"] == label and row["changed_path"] == path and row["rejected"] is True
                and row[checksum] is True and row["exception_type"] in ("TypeError","ValueError")
                and type(row["message"]) is str and bool(row["message"])
                and type(row["mutated_dto_sha256"]) is str and len(row["mutated_dto_sha256"]) == 64
                and all(c in "0123456789abcdef" for c in row["mutated_dto_sha256"]),
                "Recorded rehashed mutation was not refused: " + label)
    return 14


def run_audit(control):
    started=time.perf_counter()
    require(control.is_dir(),"Control evidence directory missing")
    artifacts={name:file_sha(relative_file(control,name)) for name in (
        "sources_before.json","sources_after.json","source_validation_receipt.json",
        "source_history_before.json","source_history_after.json","source_checkpoint_before.json","source_checkpoint_after.json",
        "force_candidate_before.json","force_candidate_after.json","declared_handover_law.json",
        "candidate_handover.json","candidate_handover_replayed.json","quadrature_transport_ledger.json",
        "port_reference_observations.json","mutation_refusals.json")}
    report=read(control/"report.json");seal(report,"report_sha256","Handover control report")
    require(report["format"] == "native-candidate-material-port-handover-control-1" and report["status"] == "PASS",
            "Handover control failed or changed format")
    same(report["artifacts_sha256"],artifacts,"Complete input artifact hashes")
    prior=verify_prior(read(control/"source_validation_receipt.json"))
    baseline=read(BASE/"starting_sources.json")
    require(type(baseline) is dict and len(baseline) == 664,"Original source profile count changed")
    same(baseline,read(PHYSICAL/"release_gates_01/sources_after.json"),"Original gate 664-source baseline")
    same(baseline,read(PHYSICAL/"control_release_01/sources_after.json"),"Original control 664-source baseline")
    before=read(control/"sources_before.json");after=read(control/"sources_after.json")
    same(before,after,"Handover source freeze")
    current={path.relative_to(ROOT).as_posix():file_sha(path) for directory in ("tectonics","tests","scripts","analysis")
             for path in sorted((ROOT/directory).rglob("*.py"))}
    same(current,after,"Exhaustive current Python source enumeration")
    for name,expected in baseline.items():
        require(current.get(name) == expected,"Frozen original Python source changed: " + name)
    require(report["source_count"] == len(current) and report["original_source_count"] == 664
            and report["added_python_sources"] == sorted(set(current)-set(baseline))
            and report["existing_sources_changed"] == [],"Control source ancestry changed")
    pairs=(("source_history_before.json","source_history_after.json"),
        ("source_checkpoint_before.json","source_checkpoint_after.json"),
        ("force_candidate_before.json","force_candidate_after.json"),
        ("candidate_handover.json","candidate_handover_replayed.json"))
    same(report["byte_pairs"],[list(pair) for pair in pairs],"Four exact source/replay bytepair references")
    byte_pairs=[]
    for first,second in pairs:
        a,b=control/first,control/second
        require(a.read_bytes() == b.read_bytes(),"Source/replay bytes changed: " + first)
        byte_pairs.append(dict(first=first,second=second,bytes=a.stat().st_size,sha256=file_sha(a)))
    for name,source in (("source_history_before.json",PHYSICAL/"control_release_01/original_history.json"),
        ("source_checkpoint_before.json",MATERIAL/"control_release_02/original_force/initial.json"),
        ("force_candidate_before.json",MATERIAL/"control_release_02/original_force/force_candidate.json")):
        require((control/name).read_bytes() == source.read_bytes(),"External authoritative source bytes changed: " + name)
    history=read(control/"source_history_before.json");seal(history,"binding_sha256","Authenticated original history")
    candidate=read(control/"force_candidate_before.json")
    require(type(candidate) is dict and len(candidate) == 14,"All fourteen actual force candidate fields required")
    same(history["actual_candidate"],candidate,"Original complete actual force candidate")
    checkpoint=read(control/"source_checkpoint_before.json")
    same(history["accepted_before"],checkpoint["native_underthrust_motion"],"Entire original typed accepted checkpoint")
    require(history["original_checkpoint_sha256"] == file_sha(control/"source_checkpoint_before.json"),
            "Original checkpoint byte authentication changed")
    law=read(control/"declared_handover_law.json");keys(law,"model provenance","Declared constitutive transport law")
    same(law,report["law"],"Explicit control law reference")
    bundle=read(control/"candidate_handover.json")
    results=check_bundle(bundle,history,law,read(PHYSICAL/"control_release_01/original_nodal_queries.json"))
    ledger=read(control/"quadrature_transport_ledger.json");seal(ledger,"ledger_sha256","Once-only quadrature ledger")
    results["original_energy_work_field_count"]=check_once_ledger(ledger,history,bundle)
    results["port_reference_observation_count"]=check_observations(read(control/"port_reference_observations.json"),bundle)
    results["mutation_refusal_count"]=check_mutation_receipts(read(control/"mutation_refusals.json"))
    for name in ("source_before_after_unchanged","original_typed_checkpoint_loaded","original_force_re_solved",
            "history_builder_original_force_re_solved","prior_material_state_replayed","original_candidate_all_fields_identical",
            "original_local_states_preserved","source_history_bytes_unchanged","source_checkpoint_bytes_unchanged",
            "source_candidate_bytes_unchanged","source_quad_measure_work_counted_once",
            "correlated_geometry_cancelled_symbolically","unchanged_global_beam_dofs_and_stiffness"):
        require(report[name] is True,"Control lacks source/transport guarantee: " + name)
    for name in ("transported_candidate_is_force_equilibrium","native_full_bank_and_clamp_support_certified",
            "finite_measure_assigned_to_first_entry_point","candidate_mechanics_adopted","physical_pair_admission_authorized",
            "global_native_inventory_updated"):
        require(report[name] is False,"Control incorrectly grants " + name)
    for name in ("transfer_external_work_j","transfer_dissipation_j","accepted_loaded_native_intervals","accepted_source_work_j"):
        require(type(report[name]) is int and report[name] == 0,"Control accepts native force/work")
    counts={"original_candidate_field_count":14,"original_node_count":16,"transported_port_count":8,"retained_port_count":8,
        "transported_source_trace_count":2,"positive_cumulative_slip_count":16,"nonzero_signed_plastic_slip_count":16,
        "positive_compressive_pressure_count":16,"port_reference_observation_count":8,"mutation_refusal_count":14}
    for name,value in counts.items():
        require(type(report[name]) is int and report[name] == value,"Control count changed: " + name)
    same(report["time_fraction"],[1,1],"Control exact endpoint fraction")
    require(report["source_epoch"] == "unaccepted_candidate","Control adopted a candidate epoch")
    artifacts["report.json"]=file_sha(control/"report.json")
    return dict(format="native-material-port-handover-independent-audit-1",status="PASS",
        seconds=time.perf_counter()-started,decimal_precision=PRECISION,numerical_point_checks_only=True,
        numerical_scope="Independent Decimal point evaluation of saved analytic endpoint expressions and chart box corners; no independent whole-interval proof",
        exact_algebra_scope="Original cubic source basis, reference offset/cancellation, complete LocalState and once-only quadrature measure/work, unchanged beam DOFs/K, frozen-label covectors and receiver-frame virtual power",
        standard_library_only=True,production_modules_imported=False,independent_force_solve_performed=False,
        independent_whole_interval_certification=False,recorded_mutation_refusals_verified=True,
        mutation_tests_independently_reexecuted=False,source_count=len(current),original_source_count=664,
        existing_sources_changed=[],source_before_after_unchanged=True,exhaustive_current_source_map_verified=True,
        all_original_candidate_fields_identical=True,original_candidate_field_count=14,
        external_original_history_checkpoint_candidate_bytes_preserved=True,
        original_local_states_preserved=True,source_quad_measure_work_counted_once=True,
        correlated_geometry_cancelled_symbolically=True,unchanged_global_beam_dofs_and_stiffness=True,
        work_lineage_scope="original_checkpoint_sha256 plus point_id; before/candidate epochs cannot create independent stores",
        byte_pairs=byte_pairs,transported_candidate_is_force_equilibrium=False,native_full_bank_and_clamp_support_certified=False,
        finite_measure_assigned_to_first_entry_point=False,candidate_mechanics_adopted=False,
        physical_pair_admission_authorized=False,global_native_inventory_updated=False,
        transfer_external_work_j=0,transfer_dissipation_j=0,accepted_loaded_native_intervals=0,accepted_source_work_j=0,
        audit_source_sha256=file_sha(Path(__file__)),input_artifact_sha256=artifacts,**prior,**results)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    with localcontext() as context:
        context.prec=PRECISION
        receipt=run_audit(args.control.resolve())
    receipt["receipt_sha256"]=digest(receipt)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("xb") as stream:
        stream.write(encoded(receipt)+b"\n")
    print(json.dumps({name:receipt[name] for name in ("status","original_node_count","transported_port_count",
        "retained_port_count","virtual_power_checks","source_count")},sort_keys=True))


if __name__ == "__main__":
    main()
