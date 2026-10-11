"""Declared material-point lift of the complete frozen903 source-cell rule.

The 68 positive reference-area weights are used once.  A cell centroid is
never substituted for a material label.  Each deterministic witness is inside
an actual source proxy polygon and the physical source cell; cold witnesses
also retain an authenticated LIVE first-entry allocation.  The rule is a NEW
discretization: it has no continuum quadrature or whole-bottom-support proof.
Owner rotations are queried at zero, with their true first and second jets.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from fractions import Fraction as F
import hashlib
import json
from math import comb, factorial

from .geometric_native_pose import FractionInterval as I, interval_sqrt, interval_atan, certify_rotation_history
from .geometric_native_spherical_solid import interval_cbrt
from .geometric_native_affine_body import AFFINE_SYMMETRIC_GENERATORS
from .geometric_intrinsic_geometry import rotation_increment_from_dict
from .geometric_native_common_carrier_source_work import (
    _payload, _digest, _typed_guard, _fraction, _interval, _packet, _round,
)
from .geometric_native_common_cold_material_carrier import (
    native_common_cold_material_carrier_to_dict, _registered as _cold_archive,
    evaluate_native_common_cold_material_point, _seam_partition,
)
from .geometric_native_archive_material_carrier import (
    _registered as _archive_state, evaluate_native_archive_material_point,
    native_archive_material_point_to_dict, _sin_cos,
)
from .geometric_native_common_hot_mean import (
    native_common_hot_mean_to_dict, validate_native_common_hot_mean,
    solve_native_common_hot_finite_mean, GENERATORS,
)
from .geometric_native_common_cold_depth_preparation import native_common_cold_depth_preparation_to_dict
from .geometric_native_common_preparation_contacts import _preparation_matrix_exponential
from .geometric_native_polygons import (
    NativeRay, native_polygon_from_dict, native_polygon_to_dict,
    native_intersection, native_ray_to_dict, _side, _ray_values,
)
from .geometric_native_mantle_loading import _mesh_cells
from .geometric_basal_loading import mesh_fingerprint
from .mesh import build_icosphere
from .geometric_native_first_entry_material import (
    certify_native_first_entry_material_point, evaluate_native_first_entry_material_point,
    native_first_entry_material_point_chronology_to_dict,
)

FORMAT = "native-common-basal-material-trace-lift-1"
RESPONSE_FORMAT = "native-common-basal-material-trace-response-1"
SCOPE = "NEW_DECLARED_REFERENCE_SOURCE_CELL_MATERIAL_TRACE_LIFT"
SOURCE903_SEAL = "c53e4cd75219328eeac974f3b9982aa33ad78eaf2e07d823feafb9f07b289e6c"
SOURCE895_SEAL = "27cbc05b687509fa22c00aa4339dd66d461c4def73ed37d4eae3acd62847da70"
SOURCE907_SEAL = "bba7f1f4913d6bc47f5f41beff0a279be3c203993ddd0b0b06979c719abeda1a"
SOURCE_MESH = "a333aa866eee6828f9a9fd7c84547ba681fe011f881f5e7ab98ec6e20f82ee66"
_SYSTEMS, _RESPONSES, _CONTEXTS, _EVALUATIONS, _BINDINGS = {}, {}, {}, {}, {}
_CHRONOLOGIES, _SOURCE_MODE_CACHE, _DEPTH_ROTATIONS, _LIVE_PROOFS = {}, {}, {}, {}
_ZERO = (I(0),)*3
_ID = tuple(tuple(I(int(i==j)) for j in range(3)) for i in range(3))


def _json(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)


def _body(value):
    return {f.name:getattr(value,f.name) for f in fields(value) if f.name!="binding_sha256"}


def _remember(value,registry):
    value=replace(value,binding_sha256=_digest(_body(value)))
    registry[value.binding_sha256]=_typed_guard(value)
    return value


def _validate(value,cls,registry):
    if (type(value) is not cls or registry.get(value.binding_sha256)!=_typed_guard(value)
            or _digest(_body(value))!=value.binding_sha256):
        raise ValueError("An altered or resealed basal trace lacks its complete registered source")
    return value


def _precision(value):
    if type(value) is not int or not 96<=value<=512:
        raise ValueError("Basal trace precision must be a literal integer in [96,512]")
    return value


def _mv(a,x):
    return tuple(_round(sum((a[i][j]*x[j] for j in range(3)),I(0)),96) for i in range(3))


def _map(a,x,precision):
    return tuple(_round(sum((a[i][j]*x[j] for j in range(3)),I(0)),precision) for i in range(3))


def _dot(a,b):
    return sum((x*y for x,y in zip(a,b)),I(0))


def _cross(a,b):
    return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])


def _square(x):
    return I(F(0) if x.lo<=0<=x.hi else min(x.lo*x.lo,x.hi*x.hi),max(x.lo*x.lo,x.hi*x.hi))


def _normalized_vertices(polygon,precision):
    result=[]
    for ray in polygon.vertices:
        v=_ray_values(ray.rotated(polygon.pose),precision)[1]
        n=interval_sqrt(sum((_square(x) for x in v),I(0)),precision+16)
        result.append(tuple(_round(x/n,precision) for x in v))
    return tuple(result)


def _fractional_candidate(values,precision=160):
    # A rational candidate is subsequently proved by native predicates.  Its
    # numerical construction grants no geometric/material membership.
    den=10**(precision-40)
    return tuple(F(round(x*den),den) for x in values)


def _exact_gap_coefficients(ray,donor_axis,receiver_axis,order):
    """Exact coefficients of -(exp(Kr*t)ex).(exp(Kd*t)ray)."""
    receiver=[(F(1),F(0),F(0))];donor=[ray]
    for n in range(1,order+1):
        receiver.append(tuple(v/n for v in _cross(receiver_axis,receiver[-1])))
        donor.append(tuple(v/n for v in _cross(donor_axis,donor[-1])))
    return tuple(-sum((sum((receiver[i][k]*donor[n-i][k] for k in range(3)),F(0))
        for i in range(n+1)),F(0)) for n in range(order+1))


def _exact_bernstein(coefficients,lo=F(0),hi=F(1)):
    order=len(coefficients)-1
    if lo==0 and hi==1:
        translated=coefficients
    else:
        width=hi-lo
        translated=tuple(width**n*sum((coefficients[j]*comb(j,n)*lo**(j-n)
            for j in range(n,order+1)),F(0)) for n in range(order+1))
    return tuple(sum((translated[n]*F(comb(k,n),comb(order,n)) for n in range(k+1)),F(0))
        for k in range(order+1))


def _strict_negative_upper(value,precision):
    # The inherited carrier rounder is relative to the nonzero magnitude and
    # has no binary64 underflow.  Check the persisted strict sign explicitly.
    bound=_round(I(value),precision+16).hi
    if value>=0 or bound>=0:
        raise ValueError("An all-clock proof must retain a strictly negative persisted upper bound")
    return bound


def _no_entry_gap_certificate(ray,donor_axis,receiver_axis,*,precision=96,maximum_cells=256):
    """Prove gap<0 over the complete clock, or return no certificate.

    Every coefficient, translation and Bernstein bound is evaluated exactly.
    Only the saved receipt is outward compressed.  A negative endpoint or
    representative sample never establishes this theorem.
    """
    _precision(precision)
    if (any(type(v) is not tuple or len(v)!=3 or any(type(x) is not F for x in v)
            for v in (ray,donor_axis,receiver_axis)) or not any(ray)
            or type(maximum_cells) is not int or not 1<=maximum_cells<=4096):
        raise ValueError("The all-clock material proof requires exact ray/axes and a literal bounded cell budget")
    gamma=sum(map(abs,donor_axis),F(0))+sum(map(abs,receiver_axis),F(0))
    if gamma>=F(1,4):
        raise ValueError("The authenticated historical relative-rotation proof exceeds its small-motion domain")
    size=sum(map(abs,ray),F(0))
    deviation=gamma/(1-gamma)
    simple_upper=-ray[0]+deviation*size
    if simple_upper<0:
        return dict(format="whole-clock-negative-relative-material-gap-1",
            method="EXACT_RELATIVE_ROTATION_L1_MAJORANT",original_ray=ray,
            donor_axis=donor_axis,receiver_axis=receiver_axis,clock_fraction_bounds=I(0,1),
            axis_l1_sum=gamma,ray_l1_norm=size,relative_pose_deviation_upper=deviation,
            whole_clock_gap_upper=_strict_negative_upper(simple_upper,precision),
            strict_whole_clock_gap_negative=True,all_clock_live_material_allocation_proved=True,
            historical_archive_activation_beta_identically_zero=True,endpoint_only_proof=False,
            receipt_precision_decimal_digits=precision,exact_arithmetic_before_receipt_compression=True)
    # Bernstein convex-hull bounds retain all components of both authentic
    # axes.  The sum-norm tail bounds the product of the two exponential series.
    for order in (24,48,64):
        coefficients=_exact_gap_coefficients(ray,donor_axis,receiver_axis,order)
        tail=size*gamma**(order+1)/F(factorial(order+1))/(1-gamma/F(order+2))
        stack=[(F(0),F(1),0)];cells=[];unproved=False
        while stack:
            lo,hi,depth=stack.pop()
            bernstein=_exact_bernstein(coefficients,lo,hi)
            upper=max(bernstein)+tail
            if upper<0:
                cells.append(dict(clock_fraction_bounds=I(lo,hi),
                    bernstein_coefficient_bounds=tuple(_round(I(x),precision+16) for x in bernstein),
                    whole_cell_gap_upper=_strict_negative_upper(upper,precision)))
                continue
            # A rigorously positive point rules out this theorem immediately;
            # it grants no alternative material allocation or endpoint shortcut.
            def value(t):
                result=F(0)
                for a in reversed(coefficients): result=result*t+a
                return result
            if any(value(t)-tail>=0 for t in (lo,(lo+hi)/2,hi)):
                return None
            if depth>=16 or len(cells)+len(stack)+2>maximum_cells:
                unproved=True;break
            mid=(lo+hi)/2
            stack.append((mid,hi,depth+1));stack.append((lo,mid,depth+1))
        if not unproved:
            return dict(format="whole-clock-negative-relative-material-gap-1",
                method="EXACT_RATIONAL_TAYLOR_BERNSTEIN_WITH_PRODUCT_NORM_TAIL",
                original_ray=ray,donor_axis=donor_axis,receiver_axis=receiver_axis,
                clock_fraction_bounds=I(0,1),axis_l1_sum=gamma,ray_l1_norm=size,taylor_order=order,
                power_coefficient_bounds=tuple(_round(I(x),precision+16) for x in coefficients),
                product_exponential_tail_upper=_round(I(tail),precision+16).hi,
                tail_theorem="||ray||1*gamma^(N+1)/(N+1)!/(1-gamma/(N+2))",
                cells=tuple(sorted(cells,key=lambda c:c["clock_fraction_bounds"].lo)),
                maximum_cell_count=maximum_cells,strict_whole_clock_gap_negative=True,
                all_clock_live_material_allocation_proved=True,
                historical_archive_activation_beta_identically_zero=True,endpoint_only_proof=False,
                receipt_precision_decimal_digits=precision,exact_arithmetic_before_receipt_compression=True)
    return None


def _verify_no_entry_gap_certificate(certificate,*,original_ray,donor_axis,receiver_axis):
    if type(certificate) is not dict:
        raise ValueError("A complete all-clock no-entry proof receipt is required")
    expected=_no_entry_gap_certificate(original_ray,donor_axis,receiver_axis,
        precision=certificate.get("receipt_precision_decimal_digits"),
        maximum_cells=certificate.get("maximum_cell_count",256))
    if expected is None or _typed_guard(expected)!=_typed_guard(certificate):
        raise ValueError("A changed all-clock no-entry proof cannot authorize material loading")
    return expected


def _live_witness_proof(state,archive,bank,ray,precision):
    key=state.state_sha256,archive.model_sha256,bank.original_parent_id,ray,precision
    if key in _LIVE_PROOFS: return _LIVE_PROOFS[key]
    motions=dict(state.journal.omega_rad_per_myr);duration=F.from_float(state.journal.dt_myr)
    journal_donor=tuple(F.from_float(x)*duration for x in motions[archive.donor_owner])
    journal_receiver=tuple(F.from_float(x)*duration for x in motions[archive.receiver_owner])
    if (journal_donor!=archive.donor_axis or journal_receiver!=archive.receiver_axis
            or archive.duration_myr!=duration or archive.donor_owner!=state.sweep_certificate.donor_owner
            or archive.receiver_owner!=state.sweep_certificate.receiver_owner):
        raise ValueError("The actual first-entry clock and archive-activation path must retain identical authentic axes")
    source=json.loads(bank.source_context_json)["embedding_bank"]
    history=tuple(rotation_increment_from_dict(x) for x in json.loads(source["source_owner_history_json"]))
    if (len(history)!=1 or history[0].dt_myr!=state.journal.dt_myr
            or history[0].omega_rad_per_myr!=motions[bank.owner]):
        raise ValueError("A lifted original-material owner pose differs from the authentic journal and archive histories")
    allocation=next(a for a in state.parent_allocations if a.fragment_id==bank.original_parent_id)
    if allocation.owner!=bank.owner or allocation.material_id!=bank.original_material_id:
        raise ValueError("The original material parent was transplanted onto another source owner or identity")
    parent=next(p for p in state.journal.source_surface.fragments if p.fragment_id==bank.original_parent_id)
    if any(_side(NativeRay("seed",ray),support)<0 for support in parent.geometry.support_planes):
        raise ValueError("The complete original material ray lies outside its authentic source parent")
    if bank.owner==archive.receiver_owner:
        if allocation.archived_area_bounds_sr!=I(0):
            raise ValueError("The true receiving original material unexpectedly has a selected archive")
        gap=dict(format="whole-clock-live-receiving-original-material-1",
            method="AUTHENTIC_RECEIVING_ORIGINAL_ZERO_ARCHIVE_AND_ZERO_ACTIVATION_POLICY",
            original_ray=ray,clock_fraction_bounds=I(0,1),archived_source_area_bounds_sr=I(0),
            all_clock_live_material_allocation_proved=True,
            historical_archive_activation_beta_identically_zero=True,endpoint_only_proof=False)
    elif bank.owner==archive.donor_owner:
        gap=_no_entry_gap_certificate(ray,journal_donor,journal_receiver,precision=precision)
        if gap is None: return None
    else:
        raise ValueError("A basal original material owner lies outside the authentic paired history")
    result=dict(format="source-bound-live-original-material-whole-clock-proof-1",
        material_state_sha256=state.state_sha256,archive_model_sha256=archive.model_sha256,
        original_parent_id=bank.original_parent_id,material_id=bank.original_material_id,owner=bank.owner,
        original_ray=ray,actual_journal_donor_axis=journal_donor,actual_journal_receiver_axis=journal_receiver,
        archive_model_donor_axis=archive.donor_axis,archive_model_receiver_axis=archive.receiver_axis,
        journal_and_archive_axes_identical=True,actual_owner_history_json=source["source_owner_history_json"],
        original_parent_payload_sha256=allocation.parent_payload_sha256,
        complete_old_clock_fraction_bounds=I(0,1),gap_proof=gap,
        all_clock_live_material_allocation_proved=True,historical_archive_activation_beta_identically_zero=True,
        archived_material_receives_basal_loading=False,whole_clock_all_pair_admission_authorized=False,
        preparation_clock_or_work_advanced=False)
    _LIVE_PROOFS[key]=result
    return result


def _source_candidate(overlap,history,kind,bank,precision):
    vertices=_normalized_vertices(overlap,precision)
    centre=tuple(sum((v[i].midpoint for v in vertices),F(0))/len(vertices) for i in range(3))
    Q=certify_rotation_history(history,precision).matrix
    def label(v):
        return tuple(sum((Q[j][i]*v[j] for j in range(3)),F(0)) for i in range(3)) if kind=="original_live" else v
    if kind=="original_live":
        inward=bank.original_quarter_inward
        original=tuple(label(tuple(v.midpoint for v in vertex)) for vertex in vertices)
        # Prefer an interior point near the authentic active material clamp
        # when the intersection reaches that strip.  Strict membership and
        # true live allocation still decide whether any candidate is usable.
        def latitude(v):
            norm=interval_sqrt(I(sum((x*x for x in v),F(0))),precision+16)
            return sum((a*x for a,x in zip(inward,v)),F(0))/norm.midpoint
        centre_latitude=latitude(label(centre))
        for index in sorted(range(len(vertices)),key=lambda k:latitude(original[k])):
            vertex_latitude=latitude(original[index])
            if vertex_latitude>=bank.beta_cut_radians:
                continue
            vertex=tuple(x.midpoint for x in vertices[index])
            separation=centre_latitude-vertex_latitude
            preferred=min(F(1,2),(bank.beta_cut_radians-vertex_latitude)/(2*separation)) if separation>0 else F(1,2)
            for power in (0,2,4,8):
                eps=preferred/F(2**power)
                yield _fractional_candidate(label(tuple((1-eps)*vertex[i]+eps*centre[i] for i in range(3))),precision)
    yield _fractional_candidate(label(centre),precision)
    for index in range(len(vertices)):
        vertex=tuple(x.midpoint for x in vertices[index])
        yield _fractional_candidate(label(tuple((vertex[i]+centre[i])/2 for i in range(3))),precision)


def _select_witness(row,bank,mask,cell,state,precision,*,archive):
    history=tuple(rotation_increment_from_dict(x) for x in json.loads(mask["source_owner_history_json"]))
    support=json.loads(mask["complete_mask_source_json"])
    overlaps=[]
    for index,raw in enumerate(support["proxy_geometries"]):
        polygon=native_polygon_from_dict(raw)
        overlap=native_intersection(polygon,cell)
        if overlap is not None:
            overlaps.append((overlap.area_bounds().lo,index,polygon,overlap))
    allocation_failures=[]
    for _,index,polygon,overlap in sorted(overlaps,key=lambda item:item[0],reverse=True):
        for original in _source_candidate(overlap,history,row["material_kind"],bank,precision):
            ray=NativeRay("seed",original)
            world=ray.rotated(history) if row["material_kind"]=="original_live" else ray
            sides=tuple(_side(world,p) for p in overlap.support_planes)
            if min(sides)<=0:
                continue
            chronology=None
            if row["material_kind"]=="original_live":
                live_proof=_live_witness_proof(state,archive,bank,original,96)
                if live_proof is None:
                    chronology=certify_native_first_entry_material_point(state,row["original_parent_id"],original)
                    query=evaluate_native_first_entry_material_point(chronology,F(1))
                    if query.material_allocation!="live":
                        allocation_failures.append(query.material_allocation)
                        continue
                allocation="AUTHENTIC_ENDPOINT_LIVE_ORIGINAL_MATERIAL"
                proof=None if chronology is None else native_first_entry_material_point_chronology_to_dict(chronology)
                if chronology is not None:
                    _CHRONOLOGIES[state.state_sha256,row["original_parent_id"],original]=chronology
            else:
                # The genuine source proxy is a subset of the actual vacancy;
                # its extra positive curved material is retained in the weight
                # but no containing-chart point is labelled a birth point.
                allocation="AUTHENTIC_HOT_SOURCE_PROXY_INTERIOR"
                proof=None;live_proof=None
            return dict(original_ray=original,source_proxy_index=index,
                source_proxy_polygon_json=_json(native_polygon_to_dict(polygon)),
                source_intersection_polygon_json=_json(native_polygon_to_dict(overlap)),
                source_world_ray_json=_json(native_ray_to_dict(world)),
                source_intersection_strict_support_signs=sides,
                endpoint_allocation=allocation,cold_point_chronology_json=None if proof is None else _json(proof),
                cold_live_whole_clock_proof_json=None if live_proof is None else _json(_payload(live_proof)),
                material_live_proof_is_endpoint_only=False if live_proof is not None else None,
                cold_historical_activation_beta_zero_proved=live_proof is not None,
                point_is_material_cell_centroid=False,positive_curved_subset_point_claimed=False,
                material_point_selection_rule="STRICT_RATIONAL_NATIVE_INTERSECTION_INTERIOR_ACTIVE_CLAMP_PREFERENCE_V1")
    raise ValueError("Unproved positive material/cell trace: "+_json(dict(component=row["component_id"],
        cell=row["cell"],positive_proxy_intersections=len(overlaps),rejected_allocations=allocation_failures)))


@dataclass(frozen=True,slots=True)
class NativeCommonBasalTrace:
    coordinate_order: tuple
    coordinate_units: tuple
    source_packet_seals: tuple
    complete903_source_binding_sha256: str
    source903_load_sha256: str
    source_epoch_sha256: str
    source_mesh_sha256: str
    source_mantle_json: str
    source_radius_m: F
    complete_source903_load_json: str
    source_row_witnesses_json: tuple
    cold_carrier_binding_sha256: str
    hot_system_binding_sha256: str
    preparation_bindings_sha256: tuple
    internal_contact_arm_only_coordinate_indices: tuple
    positive_source_material_cell_row_count: int
    original_live_source_material_cell_row_count: int
    hot_birth_source_material_cell_row_count: int
    paired_correction_metadata_count: int
    precision: int
    provenance: str
    scope: str
    continuum_basal_quadrature_certified: bool
    complete_actual_bottom_support_certified: bool
    owner_rotation_query_is_zero_slice: bool
    full_general_state_mixed_operator_certified: bool
    physical_source_force_prestress_work_handover_certified: bool
    whole_clock_all_pair_admission_authorized: bool
    physical_material_admission_authorized: bool
    full_system_temporal_path_work_certified: bool
    clocks_advanced: bool
    accepted_original_native_intervals: int
    accepted_new_native_intervals: int
    original_point_1_completed: bool
    binding_sha256: str


def bind_native_common_basal_trace(source903,source895,source907,*,cold_carrier,hot_system,
        preparations,provenance,precision=96):
    _precision(precision)
    if type(provenance) is not str or not provenance.strip():
        raise ValueError("The NEW basal trace discretization requires explicit provenance")
    for packet,pin in ((source903,SOURCE903_SEAL),(source895,SOURCE895_SEAL),(source907,SOURCE907_SEAL)):
        _packet(packet,pin)
    cold=native_common_cold_material_carrier_to_dict(cold_carrier)
    hot=native_common_hot_mean_to_dict(hot_system)
    if cold!=source895["cold_material_carrier_system"] or hot!=source903["hot_mean_system"]:
        raise ValueError("The registered whole895/903 material fields must equal the complete genuine sources")
    if (type(preparations) is not tuple or len(preparations)!=2
            or tuple(p.owner for p in preparations)!=(0,1)):
        raise ValueError("The basal trace requires both authentic owner-wide T_lambda maps")
    for prep in preparations:
        native_common_cold_depth_preparation_to_dict(prep)
        if prep.system_binding_sha256!=cold_carrier.binding_sha256:
            raise ValueError("A basal preparation changed its complete cold895 material binding")
    bindkey=((SOURCE903_SEAL,SOURCE895_SEAL,SOURCE907_SEAL),cold_carrier.binding_sha256,hot_system.binding_sha256,
        tuple(p.binding_sha256 for p in preparations),provenance,precision)
    mixed=source907["mixed_force_system"]
    order=tuple(tuple(x) for x in mixed["coordinate_order"])
    if len(order)!=100 or tuple(mixed["coordinate_units"])!=("rad",)*6+("m",)*94:
        raise ValueError("The complete unchanged mixed100 source catalog is required")
    if hot_system.complete_current_load_json!=mixed["complete_current_load_json"]:
        raise ValueError("The physical mantle source changed between full903 and full907")
    beams=mixed["internal_beams"]
    if (len(beams)!=8 or tuple(sorted(b["global_coordinate_index"] for b in beams))!=tuple(range(92,100))
            or any(order[b["global_coordinate_index"]][0]!="source_internal" for b in beams)):
        raise ValueError("The eight residuals must retain their authentic contact-arm-only source law")
    load=json.loads(hot_system.complete_current_load_json)
    if (load["archived_material_receives_basal_loading"] is not False or load["mesh_sha256"]!=SOURCE_MESH
            or load["geometric_measure_midpoint_is_material_debit"] is not False):
        raise ValueError("The actual source cell rule must preserve reference measures and archive exclusion")
    rows=load["material_cell_measures"]
    if (len(rows)!=68 or sum(r["material_kind"]=="original_live" for r in rows)!=59
            or sum(r["material_kind"]=="hot_birth" for r in rows)!=9
            or len(load["correction_cell_measures"])!=1):
        raise ValueError("The basal trace cannot silently drop a positive source material/cell row")
    if bindkey in _BINDINGS:
        return validate_native_common_basal_trace(_BINDINGS[bindkey])
    archive=_cold_archive(cold_carrier);state=_archive_state(archive)
    if hot_system.complete_original_material_state_json!=_json(json.loads(hot_system.complete_original_material_state_json)):
        raise ValueError("The entire material-state authority must remain canonical")
    if json.loads(hot_system.complete_original_material_state_json)["state_sha256"]!=state.state_sha256:
        raise ValueError("The basal source903 material state differs from the complete registered895 archive state")
    mesh=build_icosphere(0)
    if mesh_fingerprint(mesh)!=load["mesh_sha256"]:
        raise ValueError("The exact physical source mesh changed")
    cells=_mesh_cells(mesh)
    cold_banks={b.component_id:b for b in cold_carrier.banks}
    hot_bodies={b.component_id:b for b in hot_system.bodies}
    witnesses=[]
    for index,row in enumerate(rows):
        area=_interval(row["area_bounds_m2"])
        if area.lo<=0 or row["material_kind"] not in ("original_live","hot_birth"):
            raise ValueError("Every authentic positive source row must retain its full positive weight")
        if row["material_kind"]=="original_live":
            bank=cold_banks[row["component_id"]]
            support=json.loads(bank.source_context_json)["embedding_bank"]
        else:
            bank=hot_bodies[row["component_id"]]
            support=json.loads(bank.complete_common_bank_json)
        if bank.original_parent_id!=row["original_parent_id"] or bank.owner!=row["owner"]:
            raise ValueError("A source material/cell row was transplanted onto another physical body")
        witness=_select_witness(row,bank,support,cells[row["cell"]],state,max(160,precision),archive=archive)
        witness.update(source_material_cell_index=index,complete_source_material_cell_row=row,
            source_binding_sha256=_digest(dict(packet903=SOURCE903_SEAL,load=load["load_sha256"],row=row)),
            material_depth_m=bank.areal_material_thickness_m,
            source_material_id=row["material_id"],source_component_id=row["component_id"],
            source_original_parent_id=row["original_parent_id"],source_cell=row["cell"],
            material_kind=row["material_kind"],owner=row["owner"],
            source_weight_bounds_m2=area,corrected_source_weight_included_once=True,
            paired_correction_added_as_extra_weight=False)
        witnesses.append(_json(_payload(witness)))
    radius=tuple(b.radius_m for b in cold_carrier.banks)+tuple(b.radius_m for b in hot_system.bodies)
    if set(radius)!={F(100000)}:
        raise ValueError("The actual full source radius must retain its authentic physical meaning")
    result=NativeCommonBasalTrace(order+(("preparation","cold_depth_twist","lambda"),),
        ("rad",)*6+("m",)*94+("1",),(SOURCE903_SEAL,SOURCE895_SEAL,SOURCE907_SEAL),SOURCE903_SEAL,
        load["load_sha256"],load["epoch_sha256"],load["mesh_sha256"],_json(load["source"]),F(100000),
        hot_system.complete_current_load_json,tuple(witnesses),cold_carrier.binding_sha256,hot_system.binding_sha256,
        tuple(p.binding_sha256 for p in preparations),tuple(range(92,100)),68,59,9,1,precision,provenance,SCOPE,
        False,False,True,False,False,False,False,False,False,1,0,False,"")
    result=_remember(result,_SYSTEMS)
    _CONTEXTS[result.binding_sha256]=dict(cold=cold_carrier,hot=hot_system,preparations=preparations,
        archive=archive,state=state,cold_banks=cold_banks,hot_bodies=hot_bodies)
    _BINDINGS[bindkey]=result
    return result


def validate_native_common_basal_trace(system):
    _validate(system,NativeCommonBasalTrace,_SYSTEMS)
    ctx=_CONTEXTS.get(system.binding_sha256)
    if ctx is None:
        raise ValueError("The complete basal source lost its registered material geometry context")
    native_common_cold_material_carrier_to_dict(ctx["cold"])
    validate_native_common_hot_mean(ctx["hot"])
    for preparation in ctx["preparations"]:
        native_common_cold_depth_preparation_to_dict(preparation)
    return system


def native_common_basal_trace_to_dict(system):
    return dict(format=FORMAT,**_payload(validate_native_common_basal_trace(system)))


def native_common_basal_trace_from_dict(raw,source903,source895,source907,*,cold_carrier,hot_system,preparations):
    if type(raw) is not dict or raw.get("format")!=FORMAT:
        raise ValueError("A complete registered basal trace DTO is required")
    expected=bind_native_common_basal_trace(source903,source895,source907,cold_carrier=cold_carrier,
        hot_system=hot_system,preparations=preparations,provenance=raw.get("provenance"),precision=raw.get("precision"))
    if native_common_basal_trace_to_dict(expected)!=raw:
        raise ValueError("A basal restart changed its exact material witnesses, positive measures or source")
    return expected


def _coordinates(system,coordinates):
    coordinates=(F(0),)*101 if coordinates is None else coordinates
    if type(coordinates) is not tuple or len(coordinates)!=101 or any(type(x) is not F for x in coordinates):
        raise ValueError("The basal trace requires all 101 literal Fraction coordinate values")
    if any(coordinates[:6]):
        raise ValueError("The basal trace owner query is restricted to zero rotations; its owner jets are retained")
    if not 0<=coordinates[100]<=1:
        raise ValueError("The physical preparation parameter is restricted to [0,1]")
    if any(abs(x)>100 for x in coordinates[6:36]):
        raise ValueError("The actual HOT30 finite chart retains its 100 metre domain")
    cold=_CONTEXTS[system.binding_sha256]["cold"]
    if any(abs(x)>bound for x,bound in zip(coordinates[36:92],cold.coordinate_envelope_abs_m)):
        raise ValueError("The actual cold56 trace leaves its fixed895 support envelope")
    return coordinates


@dataclass(frozen=True,slots=True)
class _ClosedLiveColdPoint:
    original_parent_id: str
    original_ray: tuple
    material_depth_bounds_m: I
    time_fraction: F
    model_sha256: str
    material_id: str
    owner: int
    allocation: str
    archive_debits: int
    beta_bounds: I
    position_bounds_m: tuple
    reference_radius_bounds_m: I
    radial_support_bounds_m: I
    live_whole_clock_proof_json: str
    source_owner_rotation_certificate_json: str
    response_sha256: str


def _closed_live_source_point(state,archive,bank,ray,proof,precision):
    if (proof["material_state_sha256"]!=state.state_sha256 or proof["archive_model_sha256"]!=archive.model_sha256
            or proof["original_parent_id"]!=bank.original_parent_id or proof["original_ray"]!=ray
            or proof["all_clock_live_material_allocation_proved"] is not True
            or proof["historical_archive_activation_beta_identically_zero"] is not True):
        raise ValueError("A closed original material geometry requires both genuine all-clock LIVE and beta-zero proofs")
    R=archive.radius_m;depth=bank.areal_material_thickness_m
    height=dict(archive.whole_domain_certificate.original_parent_thicknesses_m)[bank.original_parent_id]
    if depth!=height:
        raise ValueError("The basal label must use its authentic full original-parent material thickness once")
    r0=interval_cbrt(I(R**3-3*R**2*depth),precision)
    radius=r0 if bank.owner==archive.donor_owner else I(R)
    axis=archive.donor_axis if bank.owner==archive.donor_owner else archive.receiver_axis
    omega=tuple(I(v) for v in axis)
    W=tuple(tuple(_cross(omega,tuple(I(int(i==j)) for i in range(3)))[k] for j in range(3)) for k in range(3))
    Q,certificate=_preparation_matrix_exponential(W,precision=precision)
    norm=interval_sqrt(I(sum((x*x for x in ray),F(0))),precision+16)
    q=tuple(_round(I(v)/norm,precision) for v in ray)
    position=tuple(_round(radius*v,precision) for v in _map(Q,q,precision))
    point=_ClosedLiveColdPoint(bank.original_parent_id,ray,I(depth),F(1),archive.model_sha256,
        bank.original_material_id,bank.owner,"live",0,I(0),position,r0,radius,
        _json(_payload(proof)),_json(_payload(certificate)),"")
    seal=_digest({f.name:getattr(point,f.name) for f in fields(point) if f.name!="response_sha256"})
    return replace(point,response_sha256=seal)


def _source_point_json(point,ctx):
    if type(point) is _ClosedLiveColdPoint:
        if point.response_sha256!=_digest({f.name:getattr(point,f.name) for f in fields(point) if f.name!="response_sha256"}):
            raise ValueError("The closed authentic source geometry receipt changed")
        return _json(dict(format="new-source-bound-closed-live-archive-model-point-1",
            scope="PROVEN_AUTHENTIC_OLD_LIVE_BETA_ZERO_POSITION_RECIPE_ONLY",response=_payload(point),
            old_archive_point_api_reexecuted=False,full_original_clock_live_material_allocation_proved=True,
            archive_model_activation_beta_identically_zero=True,complete_cartesian_material_gradient_computed=False,
            physical_material_admission_authorized=False,whole_clock_all_pair_admission_authorized=False))
    return _json(native_archive_material_point_to_dict(point,system=ctx["archive"]))


def _cold_source_modes(ctx,bank,ray,precision):
    """The literal895 position/basis formulas, with an equivalent fast rotor.

    Original material allocation and beta-zero source geometry are proved on
    the complete authentic original clock.  The original carrier API is used
    only when that closed proof is unresolved.  Clamp/seam fields are kept;
    no mathematical bank, receiving/HOT field or archive debit is invented.
    """
    cold,archive,state=ctx["cold"],ctx["archive"],ctx["state"]
    key=(cold.binding_sha256,bank.original_parent_id,ray,precision)
    if key in _SOURCE_MODE_CACHE: return _SOURCE_MODE_CACHE[key]
    live_proof=_live_witness_proof(state,archive,bank,ray,precision)
    if live_proof is not None:
        point=_closed_live_source_point(state,archive,bank,ray,live_proof,precision)
    else:
        chronology=_CHRONOLOGIES.get((state.state_sha256,bank.original_parent_id,ray))
        if chronology is None:
            raise ValueError("A cold basal witness lost its genuine live first-entry chronology")
        point=evaluate_native_archive_material_point(archive,bank.original_parent_id,ray,
            I(bank.areal_material_thickness_m),F(1),precision=precision,chronology=chronology)
    if (point.allocation!="live" or point.material_id!=bank.original_material_id or point.owner!=bank.owner):
        raise ValueError("An archived or transplanted original material point cannot receive basal loading")
    norm=interval_sqrt(I(sum((x*x for x in ray),F(0))),precision+16)
    q=tuple(_round(I(v)/norm,precision) for v in ray)
    certificate=cold.whole_domain_certificate.donor if bank.owner==archive.donor_owner else cold.whole_domain_certificate.receiver
    partitions=_seam_partition(cold,bank.owner,ray,point.reference_radius_bounds_m,precision)
    if ray[1]==ray[2]==0:
        partitions=tuple((b.original_parent_id,I(F(1,4)),_ZERO) for b in cold.banks if b.owner==bank.owner)
    indices={cid:tuple(i for i,row in enumerate(cold.coordinate_order) if row[1]==cid) for cid in ctx["cold_banks"]}
    columns=[_ZERO]*56;receipts=[]
    for parent,weight,_ in partitions:
        if weight==I(0): continue
        source=next(b for b in cold.banks if b.original_parent_id==parent)
        beta_label=I(sum((a*x for a,x in zip(source.original_quarter_inward,ray)),F(0)))/norm
        if beta_label.lo<0:
            raise ValueError("A seam-blended cold field crossed its authentic original hemisphere")
        sine=_sin_cos(I(source.beta_cut_radians),precision)[0]
        inactive=beta_label.lo>=sine.hi
        h=interval_sqrt(I(sum((ray[i]*ray[i] for i in (1,2)),F(0)))/I(sum((x*x for x in ray),F(0))),precision+16)
        if inactive or h.hi==0:
            beta=I(source.beta_cut_radians,F(2));chi=I(0);modes=(_ZERO,)*7;rotation=None
        else:
            if h.lo<=0: raise ValueError("A cold source chart requires a split away from its active pole")
            beta=interval_atan(beta_label/h,precision)
            if beta.lo<source.beta_cut_radians<beta.hi:
                raise ValueError("A cold material trace straddles its authentic source clamp")
            active=beta.lo<source.beta_cut_radians
            t=beta/source.beta_cut_radians
            chi=_round((I(1)-t)*(I(1)-t)*(t+2)/2,precision) if active else I(0)
            if chi==I(0): modes=(_ZERO,)*7;rotation=None
            else:
                R=source.radius_m;top=R+source.roof_radial_displacement_m;depth=source.areal_material_thickness_m
                rold=interval_cbrt(I(top**3-3*R**2*depth),precision)
                tau=depth+(R**3-top**3)/(3*R**2)
                rkey=(source.angular_rate_per_physical_volume_depth_m,tau,precision)
                if rkey not in _DEPTH_ROTATIONS:
                    omega=tuple(I(v) for v in source.angular_rate_per_physical_volume_depth_m)
                    W=tuple(tuple(_cross(omega,tuple(I(int(i==j)) for i in range(3)))[k] for j in range(3)) for k in range(3))
                    _DEPTH_ROTATIONS[rkey]=_preparation_matrix_exponential(tuple(tuple(tau*v for v in row) for row in W),precision=precision)
                Qdepth,rotation=_DEPTH_ROTATIONS[rkey]
                e=_map(Qdepth,_map(source.material_owner_pose_bounds,q,precision),precision)
                X=tuple(_round(rold*v,precision) for v in e)
                modes=(tuple(_round(chi*v,precision) for v in e),)+tuple(
                    tuple(_round(chi*v/R,precision) for v in _map(tuple(tuple(I(x) for x in row) for row in E),X,precision))
                    for E in AFFINE_SYMMETRIC_GENERATORS)
        for i,v in zip(indices[source.component_id],modes): columns[i]=tuple(_round(weight*x,precision) for x in v)
        receipts.append(dict(original_parent_id=parent,component_id=source.component_id,seam_weight_bounds=_payload(weight),
            source_beta_bounds_radians=_payload(beta),source_clamp_shape_bounds=_payload(chi),
            original_clamp_inactive=chi==I(0),rotation_certificate=None if rotation is None else _payload(rotation),
            source_mode_columns=_payload(modes)))
    base=tuple(_round(certificate.fixed_radial_scale*v,precision) for v in point.position_bounds_m)
    result=(base,tuple(columns),point,tuple(receipts))
    _SOURCE_MODE_CACHE[key]=result
    return result


def _prepared_cold_twojet(preparation,x,columns,lam,precision):
    declaration=preparation.declaration;R,s=declaration.radius_m,declaration.fixed_radial_scale
    omega=tuple(I(v) for v in declaration.angular_rate_per_physical_volume_depth_m)
    norm=interval_sqrt(sum((_square(v) for v in x),I(0)),precision+16)
    if norm.lo<=0:
        raise ValueError("A physical basal trace cannot cross an unresolved origin")
    tau=_round((I(R**3)-norm*norm*norm/s**3)/(3*R**2),precision)
    denominator=s**3*R**2
    g=tuple(_round(-norm*v/denominator,precision) for v in x)
    ht=tuple(tuple(_round(-(norm*int(i==j)+x[i]*x[j]/norm)/denominator,precision)
        for j in range(3)) for i in range(3))
    W=tuple(tuple(_cross(omega,tuple(I(int(i==j)) for i in range(3)))[k]
        for j in range(3)) for k in range(3))
    Q,cert=_preparation_matrix_exponential(tuple(tuple(_round(lam*tau*v,precision) for v in row) for row in W),precision=precision)
    wx,wwx=_cross(omega,x),_cross(omega,_cross(omega,x))
    DT=tuple(tuple(_round(v,precision) for v in row) for row in
        tuple(tuple(sum((Q[i][k]*(I(int(k==j))+lam*wx[k]*g[j]) for k in range(3)),I(0))
            for j in range(3)) for i in range(3)))
    rate=tuple(tuple(sum((Q[i][k]*(tau*W[k][j]+wx[k]*g[j]+lam*tau*wwx[k]*g[j])
            for k in range(3)),I(0)) for j in range(3)) for i in range(3))
    first=tuple(_map(DT,v,precision) for v in columns)+(_map(Q,tuple(tau*v for v in wx),precision),)
    modal_second=[]
    for a in columns:
        ga=_dot(g,a);wa=_cross(omega,a);row=[]
        for b in columns:
            gb=_dot(g,b);wb=_cross(omega,b)
            hab=sum((a[i]*ht[i][j]*b[j] for i in range(3) for j in range(3)),I(0))
            before=tuple(lam*(wa[i]*gb+wb[i]*ga+wx[i]*hab)+lam*lam*wwx[i]*ga*gb for i in range(3))
            row.append(_map(Q,before,precision))
        row.append(_map(rate,a,precision));modal_second.append(tuple(row))
    modal_second.append(tuple(_map(rate,a,precision) for a in columns)+(_map(Q,tuple(tau*tau*v for v in wwx),precision),))
    if lam==0:
        # These mathematical zeros are exact; no missing load is substituted.
        modal_second=tuple(tuple(_ZERO if i<len(columns) and j<len(columns) else v
            for j,v in enumerate(row)) for i,row in enumerate(modal_second))
    return _map(Q,x,precision),first,tuple(modal_second),dict(rotation_certificate=_payload(cert),
        physical_volume_depth_bounds_m=_payload(tau),nonlinear_body_body_curvature_included=True,
        mixed_body_preparation_curvature_included=True,preparation_second_derivative_included=True)


def _owner_lift(x,indices,first,second,owner,precision):
    own=tuple(3*owner+i for i in range(3));allindices=tuple(sorted(own+indices));lookup={v:i for i,v in enumerate(indices)}
    axes=tuple(tuple(I(int(i==j)) for i in range(3)) for j in range(3))
    f={idx:vector for idx,vector in zip(indices,first)}
    for axis,index in zip(axes,own): f[index]=tuple(_round(v,precision) for v in _cross(axis,x))
    h={}
    for i,a in enumerate(indices):
        for j,b in enumerate(indices[i:],i):
            symmetric=tuple(I(max(second[i][j][axis].lo,second[j][i][axis].lo),
                min(second[i][j][axis].hi,second[j][i][axis].hi)) for axis in range(3))
            h[a,b]=h[b,a]=symmetric
    for i,a in enumerate(own):
        for j,b in enumerate(own):
            h[a,b]=tuple(_round((u+v)/2,precision) for u,v in zip(_cross(axes[i],_cross(axes[j],x)),_cross(axes[j],_cross(axes[i],x))))
        for b in indices:
            h[a,b]=h[b,a]=tuple(_round(v,precision) for v in _cross(axes[i],f[b]))
    position_first=tuple(tuple(f[index][axis] for index in allindices) for axis in range(3))
    position_second=tuple(tuple(tuple(h[a,b][axis] for b in allindices) for a in allindices) for axis in range(3))
    return allindices,position_first,position_second


def _saved(raw):
    if type(raw) is list:
        if len(raw)==2 and all(type(x) is int for x in raw): return _fraction(raw)
        return tuple(_saved(x) for x in raw)
    if type(raw) is dict:
        if set(raw)=={"lo","hi"}: return _interval(raw)
        return {k:_saved(v) for k,v in raw.items()}
    return raw


@dataclass(frozen=True,slots=True)
class NativeCommonBasalTraceResponse:
    system_binding_sha256: str
    coordinate_values: tuple
    trace_records_json: tuple
    positive_source_material_cell_row_count: int
    original_live_material_trace_count: int
    hot_birth_material_trace_count: int
    active_cold_coordinate_indices: tuple
    active_hot_coordinate_indices: tuple
    precision: int
    root_iterations: int
    scope: str
    all_positive_source_cell_weights_used_once: bool
    paired_correction_added_again: bool
    aggregate_owner6_load_reused: bool
    archived_material_receives_basal_loading: bool
    internal_contact_arm_basal_derivatives_exactly_zero: bool
    all_local_first_second_position_derivatives_computed: bool
    continuum_basal_quadrature_certified: bool
    complete_actual_bottom_support_certified: bool
    full_general_state_mixed_operator_certified: bool
    physical_source_force_prestress_work_handover_certified: bool
    whole_clock_all_pair_admission_authorized: bool
    physical_material_admission_authorized: bool
    full_system_temporal_path_work_certified: bool
    new_mixed_force_equilibrium_certified: bool
    clocks_advanced: bool
    accepted_new_native_intervals: int
    original_point_1_completed: bool
    binding_sha256: str


def evaluate_native_common_basal_trace(system,coordinates101=None,*,precision=96,root_iterations=12):
    system=validate_native_common_basal_trace(system);_precision(precision)
    coordinates=_coordinates(system,coordinates101)
    if type(root_iterations) is not int or not 0<=root_iterations<=64:
        raise ValueError("The finite HOT mean root budget must be a literal integer in [0,64]")
    key=(system.binding_sha256,coordinates,precision,root_iterations)
    if key in _EVALUATIONS: return validate_native_common_basal_trace_response(_EVALUATIONS[key],system=system)
    ctx=_CONTEXTS[system.binding_sha256];cold=ctx["cold"];hot=ctx["hot"]
    coldindices={b.component_id:tuple(i for i,row in enumerate(system.coordinate_order[:100]) if row[0]=="body" and row[1]==b.component_id) for b in cold.banks}
    hotindices={b.component_id:tuple(i for i,row in enumerate(system.coordinate_order[:100]) if row[0]=="body" and row[1]==b.component_id) for b in hot.bodies}
    roots={};records=[];activecold=set();activehot=set()
    for encoded in system.source_row_witnesses_json:
        saved=json.loads(encoded);w=_saved(saved);ray=w["original_ray"];cid=w["source_component_id"]
        if w["material_kind"]=="original_live":
            bank=ctx["cold_banks"][cid]
            base,allcolumns,point,source_fields=_cold_source_modes(ctx,bank,ray,precision)
            current=tuple(_round(base[i]+sum((a*v[i] for a,v in zip(coordinates[36:92],allcolumns)),I(0)),precision) for i in range(3))
            # Preserve every actual seam-blended 895 column, including zero
            # columns justified by the original material clamp.
            pairs=tuple((36+i,v) for i,v in enumerate(allcolumns) if any(x!=I(0) for x in v))
            ids=tuple(i for i,_ in pairs);columns=tuple(v for _,v in pairs)
            x,f,H,geometry=_prepared_cold_twojet(ctx["preparations"][point.owner],current,columns,coordinates[100],precision)
            ids=ids+(100,)
            pointbinding=_digest(dict(cold895=cold.binding_sha256,source_archive_point=point.response_sha256,
                original_ray=ray,actual_coordinate_values=coordinates[36:92],actual_position=current,
                exact_same_source_mode_columns=allcolumns,precision=precision))
            geometry.update(actual_cold895_field_recipe_binding_sha256=pointbinding,
                source_archive_point_json=_source_point_json(point,ctx),
                old_archive_point_api_reexecuted=type(point) is not _ClosedLiveColdPoint,
                full_original_clock_live_material_allocation_proved=type(point) is _ClosedLiveColdPoint,
                cold_preparation_input_position_bounds_m=_payload(current),
                whole_cold56_source_mode_displacement_columns=_payload(allcolumns),
                source_bank_field_receipts=source_fields,
                complete_registered895_position_and_mode_recipe_preserved=True,actual_seam_blended_cold56_fields=True,
                original_cartesian_material_gradient_computed_in_this_trace=False)
            activecold.update(i for i,v in zip(ids,f) if 36<=i<92 and any(x!=I(0) for x in v))
        else:
            body=ctx["hot_bodies"][cid];ids=hotindices[cid]
            index=hot.bodies.index(body);q=coordinates[6+6*index];b=coordinates[7+6*index:12+6*index]
            if cid not in roots: roots[cid]=solve_native_common_hot_finite_mean(body,q,b,precision=precision,root_iterations=root_iterations)
            h,J,hs,certificate=roots[cid]
            qray=tuple(I(v) for v in ray);norm=interval_sqrt(sum((_square(x) for x in qray),I(0)),precision+16)
            direction=tuple(_round(v/norm,precision) for v in qray)
            omega=tuple(I(v) for v in body.angular_rate_per_physical_volume_depth_m)
            W=tuple(tuple(_cross(omega,tuple(I(int(i==j)) for i in range(3)))[k] for j in range(3)) for k in range(3))
            Q,rotcert=_preparation_matrix_exponential(tuple(tuple(body.areal_material_thickness_m*v for v in row) for row in W),precision=precision)
            base=tuple(_round(body.prepared_base_radius_bounds_m*v,precision) for v in _map(Q,direction,precision))
            physical=tuple(_map(tuple(tuple(I(v)/body.radius_m for v in row) for row in K),base,precision) for K in GENERATORS)
            x=tuple(_round(base[i]+h*physical[0][i]+sum((b[k]*physical[k+1][i] for k in range(5)),I(0)),precision) for i in range(3))
            f=tuple(tuple(_round(sum((physical[k][i]*J[k][a] for k in range(6)),I(0)),precision) for i in range(3)) for a in range(6))
            H=tuple(tuple(tuple(_round(physical[0][i]*hs[a][bb],precision) for i in range(3)) for bb in range(6)) for a in range(6))
            pointbinding=_digest(dict(source_body=body.source_bank_snapshot_sha256,source_witness=w["source_binding_sha256"],
                original_ray=ray,material_depth=w["material_depth_m"],coordinates=coordinates[6+6*index:12+6*index],precision=precision))
            geometry=dict(actual_hot903_source_body_binding_sha256=body.source_bank_snapshot_sha256,
                finite_mean_root_certificate=certificate,implicit_chart_jacobian=_payload(J),implicit_scalar_hessian=_payload(hs),
                physical_base_rotation_certificate=_payload(rotcert),source_polygon_pose_applied_once=True,
                additional_historical_owner_pose_applied=False,chart_force_curvature_term_included=True)
            activehot.update(i for i,v in zip(ids,f) if any(x!=I(0) for x in v))
        ids,first,second=_owner_lift(x,ids,f,H,w["owner"],precision)
        record=dict(w,position_bounds_m=x,global_coordinate_indices=ids,position_first_bounds=first,
            position_second_bounds=second,actual_point_binding_sha256=pointbinding,geometry_receipt=geometry,
            owner_rotation_query_is_zero_slice=True,owner_rotation_first_and_second_derivatives_retained=True,
            internal_contact_arm_basal_fields_exact_zero=True,source_reference_area_used_once=True,
            physical_time_schedule_declared=False,continuum_basal_quadrature_certified=False,
            complete_actual_bottom_support_certified=False)
        records.append(_json(_payload(record)))
    result=NativeCommonBasalTraceResponse(system.binding_sha256,coordinates,tuple(records),68,59,9,
        tuple(sorted(activecold)),tuple(sorted(activehot)),precision,root_iterations,SCOPE,
        True,False,False,False,True,True,False,False,False,False,False,False,False,False,False,0,False,"")
    result=_remember(result,_RESPONSES);_EVALUATIONS[key]=result
    return result


def validate_native_common_basal_trace_response(response,*,system):
    system=validate_native_common_basal_trace(system)
    _validate(response,NativeCommonBasalTraceResponse,_RESPONSES)
    if response.system_binding_sha256!=system.binding_sha256:
        raise ValueError("A basal trace response was transplanted onto another complete source")
    return response


def native_common_basal_trace_response_to_dict(response):
    _validate(response,NativeCommonBasalTraceResponse,_RESPONSES)
    return dict(format=RESPONSE_FORMAT,**_payload(response))


def native_common_basal_trace_response_from_dict(raw,*,system,expected_coordinates101=None,precision=96,root_iterations=12):
    if type(raw) is not dict or raw.get("format")!=RESPONSE_FORMAT:
        raise ValueError("A complete exact-query basal trace response DTO is required")
    value=evaluate_native_common_basal_trace(system,expected_coordinates101,precision=precision,root_iterations=root_iterations)
    if native_common_basal_trace_response_to_dict(value)!=raw:
        raise ValueError("A basal response restart changed its exact expected query, witnesses or geometry")
    return value


def native_common_basal_trace_loading_context(system,response=None):
    system=validate_native_common_basal_trace(system)
    source=json.loads(system.source_mantle_json)
    literal=lambda v:F.from_float(v) if type(v) is float else F(v)
    result=dict(source=dict(radius_m=system.source_radius_m,
        basal_drag_pa_s_m=literal(source["basal_drag_pa_s_m"]),
        angular_velocity_rad_s=tuple(literal(x) for x in source["angular_velocity_rad_s"]),
        amplitude_constant=literal(source["amplitude_constant"]),
        amplitude_gradient=tuple(literal(x) for x in source["amplitude_gradient"])),
        coordinate_order=system.coordinate_order,coordinate_units=system.coordinate_units,
        internal_contact_arm_only_coordinate_indices=system.internal_contact_arm_only_coordinate_indices,
        complete903_source_binding_sha256=system.complete903_source_binding_sha256,
        owner_rotation_query_is_zero_slice=True,full_general_state_mixed_operator_certified=False,
        continuum_basal_quadrature_certified=False,complete_actual_bottom_support_certified=False)
    if response is not None:
        validate_native_common_basal_trace_response(response,system=system)
        result.update(coordinate_values=response.coordinate_values,
            trace_records=tuple(_saved(json.loads(row)) for row in response.trace_records_json))
    return result
