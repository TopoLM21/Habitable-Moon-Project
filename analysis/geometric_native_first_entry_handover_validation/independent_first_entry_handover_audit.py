"""Standard-library audit of implicit first-entry contact transport.

Archived exact chronology kernels recheck the complete owner and material
receiver event partitions.  New Decimal point checks independently evaluate
the original continuum return-map family, the donor's own source beam, and
the saved carrier-map Jacobian and objective covectors.  Point observations
inside an enclosure are numerical checks, never selected physical roots or
an independent proof of whole material-bank force admission.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import importlib.util
from itertools import product
import json
import math
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
PRECISION = 360
PHYSICAL = ROOT / "analysis/geometric_native_physical_binding_validation"
MATERIAL = ROOT / "analysis/geometric_native_first_entry_validation"
HANDOVER = ROOT / "analysis/geometric_native_physical_handover_validation"


def load_helper(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


old = load_helper("independent_prior_material_port_handover_audit",
                  HANDOVER / "independent_handover_audit.py")
require, keys, encoded, digest = old.require, old.keys, old.encoded, old.digest
file_sha, read, same, seal = old.file_sha, old.read, old.same, old.seal
fraction, rational, interval, exact_float = old.fraction, old.rational, old.interval, old.exact_float
dec, contains, check_vector = old.dec, old.contains, old.check_vector
dot, cross, sin_cos, rotate = old.dot, old.cross, old.sin_cos, old.rotate
atan, atan2_positive, half_pi = old.atan, old.atan2_positive, old.half_pi
iadd, isub, imul, iscale, ineg = old.iadd, old.isub, old.imul, old.iscale, old.ineg
encloses, relative_file = old.encloses, old.relative_file


PRIOR_AUTHORITIES = {
    "analysis/geometric_native_physical_binding_validation/release_validation_01": (
        "71ca5f59bdbd8ed52ed5eb70aac65db573040df08cd4a971efbdc0f5404dac7f",
        "e04a6df8cdd7be689f45dcb6510a56bb743a93ff296605f07e09579982199204", 56),
    "analysis/geometric_native_first_entry_validation/release_validation_02": (
        "2f077013f0a514c8788d0210d708ef6189f460cb6c3978cb83bfe9eb926b99cc",
        "e871a378b3cf4c0a30d0e5e89c4efd04f9b6c21fa4b3ec828c1630e5e9217d7a", 83),
    "analysis/geometric_native_physical_handover_validation/release_validation_02": (
        "14dfb3532f3d3803a1700817751533ee255fca8536b30c2655b4cf03437c9825",
        "cf294df6fc8ee6e240529ae4ce9d1d6f23a3d0bbc7af0085aff8967f60c860e6", 46),
}


def verify_prior():
    verified = 0
    authorities = {}
    for name, (summary_sha, manifest_sha, count) in PRIOR_AUTHORITIES.items():
        summary_path = ROOT / name / "validation_summary.json"
        manifest_path = ROOT / name / "artifact_manifest.json"
        require(file_sha(summary_path) == summary_sha and file_sha(manifest_path) == manifest_sha,
                "Previously verified release authority bytes changed: " + name)
        summary, manifest = read(summary_path), read(manifest_path)
        seal(summary, "summary_sha256", name + " summary")
        seal(manifest, "manifest_sha256", name + " manifest")
        require(summary["status"] == "PASS" and manifest["count"] == count
                and len(manifest["artifacts"]) == count, "Archived release scope/count changed")
        same(summary["artifact_manifest"], dict(path=manifest_path.relative_to(ROOT).as_posix(),
             sha256=manifest_sha, count=count), "Archived summary manifest reference")
        for artifact, record in manifest["artifacts"].items():
            keys(record, "sha256 bytes", "Archived artifact record")
            path = relative_file(ROOT, artifact)
            require(type(record["bytes"]) is int and path.stat().st_size == record["bytes"]
                    and file_sha(path) == record["sha256"], "Archived evidence changed: " + artifact)
            verified += 1
        authorities[name] = dict(summary_sha256=summary_sha, manifest_sha256=manifest_sha, count=count)
    require(verified == 185, "Incomplete prior source/evidence ancestry")
    return dict(prior_artifacts_verified=verified, prior_authorities=authorities,
                prior_tests_reexecuted=False, prior_tests_passed=[770, 253, 178])


def field_value(raw, angle):
    values = raw["coefficients"]
    require(type(values) is list and len(values) == 5, "Original degree-two Fourier field changed")
    values = tuple(dec(exact_float(value)) for value in values)
    sine, cosine = sin_cos(angle)
    double_sine, double_cosine = sin_cos(2 * angle)
    return values[0] + values[1] * cosine + values[2] * sine + values[3] * double_cosine + values[4] * double_sine


def source_column_coefficients(columns, law, water):
    # These expressions deliberately evaluate the original binary64 series
    # constants before lifting them to exact rationals.  Replacing them with
    # a rational compliance evaluation would change the original dense law.
    normal = 1. / (columns["sub_length_m"] / columns["sub_normal_modulus_pa"]
                   + columns["over_length_m"] / columns["over_normal_modulus_pa"])
    shear = 1. / (columns["sub_length_m"] / columns["sub_shear_modulus_pa"]
                  + columns["over_length_m"] / columns["over_shear_modulus_pa"])
    kn = 1. / (1. / normal + 1. / law["normal_stiffness_pa_m"])
    kt = 1. / (1. / shear + 1. / law["tangential_stiffness_pa_m"])
    mu = law["friction_dry"] + (law["friction_wet"] - law["friction_dry"]) * water
    cohesion = law["cohesion_pa"] * (1. - (1. - law["wet_cohesion_fraction"]) * water)
    return tuple(dec(exact_float(value)) for value in (normal, shear,
        law["normal_stiffness_pa_m"], law["tangential_stiffness_pa_m"], kn, kt, mu, cohesion))


FIELD_NAMES = (
    "free_gap_bounds_m", "free_jump_bounds_m", "previous_plastic_slip_bounds_m",
    "previous_cumulative_slip_bounds_m", "plastic_increment_bounds_m", "plastic_slip_bounds_m",
    "cumulative_slip_bounds_m", "normal_pressure_bounds_pa", "shear_traction_bounds_pa",
    "contact_gap_bounds_m", "contact_jump_bounds_m", "bulk_normal_strain_bounds_m",
    "bulk_shear_strain_bounds_m", "contact_elastic_shear_strain_bounds_m",
    "bulk_energy_density_bounds_j_m2", "contact_energy_density_bounds_j_m2", "total_energy_density_bounds_j_m2",
)


def check_continuous_field(raw, history, trace_id, native_angle, time_window, expression_context):
    seal(raw, "query_sha256", "Continuous contact-field query")
    require(raw["format"] == "native-original-continuous-contact-field-query-1"
            and raw["model"] == "original_inviscid_dense_return_map_entire_piecewise_history"
            and raw["trace_id"] == trace_id, "Continuous field used a neighbouring trace or different law")
    same(raw["source_history"], history, "Full independent continuous-field source")
    same(raw["source_expression_context"], expression_context, "Implicit root/receiver source expression")
    same(raw["time_fraction_bounds"], time_window, "Unselected first-entry root window")
    require(raw["piece_selection_model"] == "all_closed_original_piece_intersections_no_node_or_midpoint_selection"
            and raw["energy_measure"] == "joules_per_square_metre_of_original_interface_area",
            "Continuous field measure or closed-piece selection changed")
    for name in ("finite_point_area_m2", "finite_point_work_j", "cumulative_work_density_j_m2"):
        require(raw[name] is None, "Continuous field invented finite point measure/work: " + name)
    for name in ("candidate_mechanics_adopted", "incoming_pair_handover_authorized",
                 "geometry_expression_authenticated_by_this_component"):
        require(raw[name] is False, "Continuous field incorrectly grants " + name)
    require(raw["original_pair_history_authenticated"] is True, "Original continuum history was not retained")
    path = history["contact_time_path"]
    traces = {trace["trace_id"]: trace for trace in path["initial_mechanics"]["fault_traces"]["traces"]}
    previous = {trace["trace_id"]: trace for trace in path["step"]["previous_memory"]["traces"]}[trace_id]
    candidate = {trace["trace_id"]: trace for trace in path["step"]["memory"]["traces"]}[trace_id]
    trace = traces[trace_id]
    require(trace["law"]["viscosity_pa_s_m"] == 0, "Fractional viscous endpoints cannot provide the retained continuum")
    angle = interval(raw["computational_angle_bounds"])
    extent = exact_float(previous["extent_radians"])
    require(0 <= angle[0] <= angle[1] <= extent, "Continuous query left its original material quarter")
    for native_end in set(interval(native_angle)):
        native_coordinate = min(dec(extent), max(Decimal(0), dec(native_end) * dec(extent) / half_pi()))
        contains(raw["computational_angle_bounds"], native_coordinate,
                 "Native-to-original material-angle conversion")
    for name, memory in (("previous_memory_intersections", previous),
                         ("unaccepted_candidate_memory_intersections", candidate)):
        expected = []
        for index, piece in enumerate(memory["pieces"]):
            low = max(angle[0], exact_float(piece["low_radians"]))
            high = min(angle[1], exact_float(piece["high_radians"]))
            if low <= high:
                expected.append(dict(piece_index=index,
                    computational_angle_bounds=dict(lo=rational(low), hi=rational(high)), original_piece=piece))
        same(raw[name], expected, "Complete unchanged original memory intersections")
        require(expected and interval(expected[0]["computational_angle_bounds"])[0] == angle[0]
                and interval(expected[-1]["computational_angle_bounds"])[1] == angle[1],
                "Continuous source skipped an original memory interval")
    support = next(row for row in history["whole_time_support"]["certificates"] if row["trace_id"] == trace_id)
    same(raw["whole_source_constitutive_support"], support, "Whole original source constitutive predicate")
    require(raw["constitutive_status"] == support["status"] == "inviscid_contact_time_constitutive_admissible",
            "Actual continuous field lost its authenticated inviscid down-dip branch")
    old_anchor = next(row for row in path["previous_anchor_fields"] if row[0] == trace_id)
    new_anchor = next(row for row in path["step"]["anchor_fields"] if row[0] == trace_id)
    kncol, ktcol, kncontact, ktcontact, kn, kt, mu, cohesion = source_column_coefficients(
        trace["columns"], trace["law"], trace["water"])
    checks = 0
    require(len(raw["pieces"]) == len(raw["previous_memory_intersections"]), "Continuous branch-piece count changed")
    for piece, intersection in zip(raw["pieces"], raw["previous_memory_intersections"]):
        require(piece["previous_piece_index"] == intersection["piece_index"], "Continuous query reordered memory pieces")
        same(piece["computational_angle_bounds"], intersection["computational_angle_bounds"],
             "Continuous return-map source piece domain")
        for theta, instant in product(set(interval(piece["computational_angle_bounds"])), set(interval(time_window))):
            theta, instant = dec(theta), dec(instant)
            gap0, jump0 = field_value(old_anchor[1], theta), field_value(old_anchor[2], theta)
            gap1, jump1 = field_value(new_anchor[1], theta), field_value(new_anchor[2], theta)
            gap, jump = gap0 + instant * (gap1-gap0), jump0 + instant * (jump1-jump0)
            old_p = field_value(intersection["original_piece"]["plastic_slip"], theta)
            old_c = field_value(intersection["original_piece"]["cumulative_slip"], theta)
            pressure = -kn * gap
            strength = cohesion + mu * pressure
            increment, plastic = jump-old_p+strength/kt, jump+strength/kt
            cumulative, shear = old_c-increment, -strength
            contact_gap, elastic_shear = -pressure/kncontact, shear/ktcontact
            contact_jump = plastic+elastic_shear
            bulk_normal, bulk_shear = gap*(kn/kncontact-1), shear*(1/ktcontact-1/kt)
            bulk_energy = (kncol*bulk_normal**2+ktcol*bulk_shear**2)/2
            contact_energy = (kncontact*contact_gap**2+ktcontact*elastic_shear**2)/2
            values = (gap, jump, old_p, old_c, increment, plastic, cumulative, pressure, shear,
                contact_gap, contact_jump, bulk_normal, bulk_shear, elastic_shear,
                bulk_energy, contact_energy, bulk_energy+contact_energy)
            for name, value in zip(FIELD_NAMES, values):
                contains(piece[name], value, "Independent dense piece " + name)
                contains(raw[name], value, "Independent dense union " + name)
            require(pressure > 0 and cumulative >= 0 and plastic < 0,
                    "Actual first-root field reset its loaded compressed plastic history")
            checks += 1
    previous_nodes = [node["previous_state"] for node in history["quadrature_histories"] if node["trace_id"] == trace_id]
    require(previous_nodes and all(node["elapsed_time_s"] == previous_nodes[0]["elapsed_time_s"] for node in previous_nodes),
            "Original continuum source has inconsistent retained local clocks")
    duration, native_duration = exact_float(history["literal_dt_s"]), exact_float(history["literal_dt_myr"])
    elapsed = exact_float(previous_nodes[0]["elapsed_time_s"])
    native_elapsed = exact_float(history["accepted_before"]["snapshot"]["elapsed_time_s"])
    physical = exact_float(history["origin_surface"]["time_myr"])
    for end in set(interval(time_window)):
        contains(raw["elapsed_time_s_bounds"], dec(elapsed+duration*end), "Retained local continuum clock")
        contains(raw["native_elapsed_time_s_bounds"], dec(native_elapsed+duration*end), "Unadopted native clock")
        contains(raw["physical_time_myr_bounds"], dec(physical+native_duration*end), "Original physical clock")
    return checks


POINT_LABELS = (
    "main_plane_reentry", "transverse_masked_main_entry", "ordinary_entry", "no_entry",
    "swept_only_owner_entry_exit", "empty_donor_four", "retained_overriding_material",
)
ENTRY_LABELS = (POINT_LABELS[0], POINT_LABELS[1], POINT_LABELS[2], POINT_LABELS[4])


def independently_audit_chronologies(records, history, checkpoint, candidate):
    require([row["label"] for row in records] == list(POINT_LABELS), "Seven original point contexts changed")
    same(records, read(MATERIAL / "control_release_02/point_certificates.json"), "Original full point contexts")
    material, journal = history["material_state"], history["material_state"]["journal"]
    material_audit = load_helper("independent_material_first_entry_source_audit",
        MATERIAL / "independent_first_entry_audit.py")
    point_audit = load_helper("independent_material_first_entry_point_audit",
        MATERIAL / "independent_first_entry_point_audit.py")
    binding = material_audit.audit_source_binding(journal,
        read(MATERIAL / "control_release_02/original_source_surface.json"), checkpoint, candidate,
        read(MATERIAL / "control_release_02/original_force/constitutive_time_support.json"),
        initial_checkpoint_sha256=file_sha(MATERIAL / "control_release_02/original_force/initial.json"))
    require(len(binding["parents"]) == 8, "Original eight-parent material authority changed")
    results = {}
    for row in records:
        proof = point_audit.audit_point_context(row["certificate"], journal,
            material["sweep_certificate"], material, binding)
        results[row["label"]] = proof
    require(results["transverse_masked_main_entry"]["first_receivers"] == ["octant:3"]
            and results["main_plane_reentry"]["first_receivers"] == ["octant:2"],
            "First owner root was replaced by a later original-fragment crossing")
    return results


def numerical_owner_root(point, donor_axis, receiver_axis, window):
    """Observe the proved unique owner root; do not construct a model state."""
    magnitude = dot(point, point).sqrt()
    require(magnitude > 0, "Independent numerical owner root lacks an original direction")
    point = tuple(v/magnitude for v in point)
    def relative(instant):
        return rotate(rotate(point, donor_axis, instant), tuple(-v for v in receiver_axis), instant)
    low, high = (dec(value) for value in interval(window))
    left, right = relative(low)[0], relative(high)[0]
    require(left >= 0 and right <= 0, "Actual first owner-entry bracket lacks independent opposite entry signs")
    for _ in range(1200):
        if high-low < Decimal(10)**-315:
            break
        middle = (low+high)/2
        value = relative(middle)[0]
        if value > 0:
            low = middle
        elif value < 0:
            high = middle
        else:
            low = high = middle
            break
    else:
        raise ValueError("Independent numerical root observation did not converge")
    observed = (low+high)/2
    return observed, relative(observed)


def bank_attachment(binding, correspondence):
    return dict(sub_fragment_id=binding["donor_fragment_id"], sub_material_id=binding["donor_material_id"],
        sub_owner=binding["subducting_plate"], over_fragment_id=binding["receiver_fragment_id"],
        over_material_id=binding["receiver_material_id"], over_owner=binding["overriding_plate"],
        correspondence=correspondence)


def check_symbol_family(raw, history, receiver_binding, donor_binding, law, context):
    keys(raw, "format law source_history trace_id original_banks initial_binding_id initial_geometry_context "
        "active_binding_id active_banks reference_geometry_context receipts revision consumed_binding_ids original_columns "
        "original_contact_law original_water continuous_interface_measure original_contact_time_path_sha256 memory_identity "
        "strain_identity offset_expression anchor_expression receiver_frame_model geometric_correspondence_authenticated "
        "finite_point_area_m2 finite_point_work_j field_scope global_force_candidate_adopted "
        "nodal_work_assigned_to_material_volume state_sha256", "First-root transported symbol family")
    seal(raw, "state_sha256", "First-root symbol family")
    require(raw["format"] == "native-contact-reference-transport-symbol-family-1"
            and raw["trace_id"] == receiver_binding["trace_id"], "Transport family used the wrong source field")
    same(raw["source_history"], history, "Full original symbol-family source")
    same(raw["law"], law, "Declared symbol-family transport law")
    trace = next(trace for trace in history["contact_time_path"]["initial_mechanics"]["fault_traces"]["traces"]
                 if trace["trace_id"] == receiver_binding["trace_id"])
    original = bank_attachment(receiver_binding, dict(model="original_authenticated_pair",
        trace_id=receiver_binding["trace_id"], source_point_id="continuous_original_material_coordinate",
        receiver_axes="unchanged_original_receiver_material_axes"))
    target = bank_attachment(dict(donor_binding, receiver_fragment_id=receiver_binding["receiver_fragment_id"],
        receiver_material_id=receiver_binding["receiver_material_id"], overriding_plate=receiver_binding["overriding_plate"]), context)
    same(raw["original_banks"], original, "Original receiving contact-bank pair")
    same(raw["active_banks"], target, "Changed physical donor and retained first receiver")
    prefix = "first-entry:" + context["point_chronology_sha256"] + ":" + receiver_binding["trace_id"]
    source_id, target_id = prefix+":original", prefix+":incoming:1"
    require(raw["initial_binding_id"] == source_id and raw["active_binding_id"] == target_id
            and type(raw["revision"]) is int and raw["revision"] == 1
            and raw["consumed_binding_ids"] == [source_id], "First-root family cloned or failed to replace its source")
    initial_context = dict(context, attachment="original_receiver_contact_field")
    same(raw["initial_geometry_context"], initial_context, "Original implicit geometry expression")
    same(raw["reference_geometry_context"], context, "New frozen implicit geometry reference")
    expected_receipt = dict(revision=1, source_binding_id=source_id, target_binding_id=target_id,
        source_banks=original, target_banks=target, source_geometry_context=initial_context,
        target_geometry_context=context, signed_transfer_external_work_identity="0", transfer_dissipation_identity="0",
        strain_energy_change_identity="0 at every common material coordinate and exact clock",
        source_binding_consumed=True, finite_point_area_m2=None, finite_point_work_j=None)
    same(raw["receipts"], [expected_receipt], "Complete once-only implicit first-root transition")
    for name, expected in (("original_columns",trace["columns"]),("original_contact_law",trace["law"]),
                           ("original_water",trace["water"])):
        same(raw[name], expected, "Unchanged transport parameters " + name)
    require(raw["continuous_interface_measure"] == "original R * contact_height / sin(dip) * dtheta; not a finite point area"
            and raw["original_contact_time_path_sha256"] == history["contact_time_path"]["path_sha256"]
            and raw["memory_identity"] == "original continuous p(theta,u), c(theta,u) retained in the declared dense path"
            and raw["strain_identity"] == "g,j-p,g-g_free,j-j_free retained at the same original theta,u"
            and raw["offset_expression"] == "a_new = original_free_anchor - h_new(theta,u)"
            and raw["anchor_expression"] == "free_current = original_free_anchor + (h_current - h_new)(theta,u)"
            and raw["receiver_frame_model"] == "unchanged_original_receiver_material_axes"
            and raw["field_scope"] == "implicit exact-clock reference identity; no midpoint or finite nodal state",
            "Symbol-family source measure, memory or reference identity changed")
    for name in ("geometric_correspondence_authenticated", "global_force_candidate_adopted",
                 "nodal_work_assigned_to_material_volume"):
        require(raw[name] is False, "Symbol transport incorrectly grants " + name)
    require(raw["finite_point_area_m2"] is None and raw["finite_point_work_j"] is None,
            "Symbol transport assigned an isolated material point a nodal reservoir")
    return source_id, target_id


def check_handover(raw, binding, chronology, proof, history, law):
    keys(raw, "format model status precision order explicit_law original_physical_binding complete_original_source_history "
        "attachments first_root_midpoint_used additional_handover_work_j additional_handover_dissipation_j "
        "original_nodal_areas_states_work_and_beam_dofs_unchanged source_field_frozen_at_implicit_first_entry "
        "whole_first_entry_domain_integration_adopted whole_spatial_bank_and_clamp_support_certified "
        "physical_pair_admission_authorized candidate_mechanics_adopted accepted_loaded_native_intervals "
        "accepted_source_work_j handover_sha256", "Implicit first-root contact handover")
    seal(raw, "handover_sha256", "Implicit first-root contact handover")
    require(raw["format"] == "native-first-entry-continuous-contact-handover-1"
            and raw["model"] == "receiver_attached_original_field_frozen_at_implicit_first_entry"
            and raw["precision"] == 256 and raw["order"] == 32,
            "Implicit first-root handover changed its model/proof budget")
    same(raw["explicit_law"], law, "Explicit first-root constitutive law")
    same(raw["original_physical_binding"], binding, "Authenticated original physical banks")
    same(raw["complete_original_source_history"], history, "Complete original handover source")
    for name in ("first_root_midpoint_used", "whole_first_entry_domain_integration_adopted",
        "whole_spatial_bank_and_clamp_support_certified", "physical_pair_admission_authorized", "candidate_mechanics_adopted"):
        require(raw[name] is False, "First-root handover incorrectly grants " + name)
    require(raw["original_nodal_areas_states_work_and_beam_dofs_unchanged"] is True
            and raw["source_field_frozen_at_implicit_first_entry"] is True, "First-root transfer lost its fixed source/clock")
    for name in ("additional_handover_work_j", "additional_handover_dissipation_j", "accepted_source_work_j"):
        require(fraction(raw[name]) == 0, "First-root transfer adopted unpaid work/dissipation")
    require(type(raw["accepted_loaded_native_intervals"]) is int and raw["accepted_loaded_native_intervals"] == 0,
            "First-root transfer adopted force clock")
    if binding["time_fraction_bounds"] is None:
        require(raw["status"] == "no_first_entry" and raw["attachments"] == [], "Negative context acquired an attachment")
        return dict(field_point_checks=0, measure_point_checks=0, source_chart_corner_checks=0, numerical_owner_roots=0)
    require(raw["status"] == "first_entry_field_attachment_committed" and len(raw["attachments"]) == 1
            and len(binding["rows"]) == 1, "True first-root material correspondence was not uniquely committed")
    row = binding["rows"][0]
    attachment = raw["attachments"][0]
    keys(attachment, "original_donor_trace_id original_receiver_trace_id desired_fragment_pair original_donor_trace_pair "
        "original_receiver_trace_pair source_field_query sub_material_chart interface_measure transport_family expression_context "
        "sub_tip_displacement_bounds_m over_tip_displacement_bounds_m receiver_world_direction receiver_world_normal "
        "receiver_world_down_dip sub_position_bounds_m over_position_bounds_m reference_gap_bounds reference_jump_bounds "
        "transported_free_gap_bounds transported_free_jump_bounds attachment_committed additional_handover_work_j "
        "additional_handover_dissipation_j finite_point_area_m2 finite_point_work_j "
        "plastic_and_energy_density_preserved_by_source_expression_identity", "First-root material attachment")
    for name, expected in (("original_donor_trace_id",row["donor_trace_id"]),
        ("original_receiver_trace_id",row["receiver_trace_id"]),("desired_fragment_pair",row["desired_fragment_pair"]),
        ("original_donor_trace_pair",row["original_donor_trace_pair"]),("original_receiver_trace_pair",row["original_receiver_trace_pair"])):
        same(attachment[name], expected, "First-root actual bank identity " + name)
    require(row["desired_fragment_pair"] == [chronology["original_parent_id"], proof["first_receivers"][0]]
            and proof["ancestry_status"] == binding["first_receiver_ancestry_status"] == "proved"
            and row["receiver_coordinate_status"] == "first_owner_root_receiver_carrier_coordinate_proved"
            and row["incoming_original_label_on_donor_carrier"] is False,
            "First-root correspondence used an original saved tip or a later receiver crossing")
    window, theta = binding["time_fraction_bounds"], row["receiver_native_angle_bounds"]
    require(interval(window) == proof["owner"]["first_window"], "Transfer clock is not the first independently proved owner-entry root")
    banks = {bank["trace_id"]: bank for bank in history["material_state"]["trace_bindings"]}
    donor_binding, receiver_binding = banks[row["donor_trace_id"]], banks[row["receiver_trace_id"]]
    field = attachment["source_field_query"]
    context = dict(model=raw["model"], point_chronology_sha256=chronology["point_chronology_sha256"],
        first_physical_binding_sha256=binding["binding_sha256"], original_parent_id=chronology["original_parent_id"],
        original_bulk_label=chronology["original_point"], first_entry_time_fraction_bounds=window,
        receiver_native_angle_bounds=theta, original_computational_angle_bounds=field["computational_angle_bounds"],
        source_trace_id=row["receiver_trace_id"], incoming_own_beam_trace_id=row["donor_trace_id"],
        target_fragment_pair=row["desired_fragment_pair"], source_expression="original_receiver_field(theta(first_root), first_root)",
        reference_expression="incoming_own_beam_and_first_receiver_at_same_implicit_root",
        source_expression_frozen_for_later_geometry_variations=True,
        incoming_bulk_direction_normalization="exact_unit_carrier_identity_at_proved_first_root",
        original_gauss_labels_unchanged=True, same_point_and_time_on_both_sides=True,
        material_label_frozen_during_subsequent_force_variations=True, first_root_midpoint_used=False)
    same(attachment["expression_context"], context, "Complete frozen first-root expression")
    field_checks = check_continuous_field(field, history, row["receiver_trace_id"], theta, window, context)
    measure_checks = check_measure(attachment["interface_measure"], history, donor_binding, receiver_binding, theta, window)
    check_symbol_family(attachment["transport_family"], history, receiver_binding, donor_binding, law, context)
    require(attachment["attachment_committed"] is True
            and attachment["plastic_and_energy_density_preserved_by_source_expression_identity"] is True,
            "First-root binding retained a diagnostic instead of changing the active attachment")
    for name in ("additional_handover_work_j", "additional_handover_dissipation_j"):
        require(fraction(attachment[name]) == 0, "First-root attachment created unpaid transfer work")
    require(attachment["finite_point_area_m2"] is None and attachment["finite_point_work_j"] is None,
            "First-root observation received a finite reservoir or nodal work")
    same(attachment["transported_free_gap_bounds"], field["free_gap_bounds_m"], "Exact unchanged loaded free normal reference")
    same(attachment["transported_free_jump_bounds"], field["free_jump_bounds_m"], "Exact unchanged loaded free shear reference")
    # Original global beam amplitudes are retained at the same implicit clock.
    before_radial = {item[0]:item[1:] for item in history["accepted_before"]["snapshot"]["transverse_displacement_m"]}
    candidate_radial = {item[0]:item[1:] for item in history["actual_candidate"]["transverse_displacement_m"]}
    def tip(trace_id, bank):
        a, b = exact_float(before_radial[trace_id][bank]), exact_float(candidate_radial[trace_id][bank])
        return iadd((a,a), iscale(interval(window), b-a))
    sub_tip, over_tip = tip(row["donor_trace_id"],0), tip(row["receiver_trace_id"],1)
    encloses(attachment["sub_tip_displacement_bounds_m"], sub_tip, "Own donor original beam amplitude at first root")
    encloses(attachment["over_tip_displacement_bounds_m"], over_tip, "First receiver original beam amplitude at first root")
    radius = exact_float(history["origin_surface"]["radius_km"])*1000
    donor_trace = next(trace for trace in history["contact_time_path"]["initial_mechanics"]["fault_traces"]["traces"]
                       if trace["trace_id"] == row["donor_trace_id"])
    receiver_trace = next(trace for trace in history["contact_time_path"]["initial_mechanics"]["fault_traces"]["traces"]
                          if trace["trace_id"] == row["receiver_trace_id"])
    for parameter in ("columns", "law", "water"):
        same(donor_trace[parameter], receiver_trace[parameter], "Unchanged target/source transport parameter " + parameter)
    parameters = next(parameter for parameter in history["contact_time_path"]["initial_mechanics"]["bending_parameters"]
                      if parameter["trace_id"] == row["donor_trace_id"])
    chart_source = dict(trace_id=row["donor_trace_id"], bank="sub",
        start=[rational(exact_float(value)) for value in donor_trace["start"]],
        end=[rational(exact_float(value)) for value in donor_trace["end"]], reference_coordinates_kind="fraction",
        reference_coordinates=chronology["original_point"], radius_m=rational(radius),
        width_m=rational(exact_float(parameters["sub"]["width_m"])),
        tip_displacement_m=attachment["sub_tip_displacement_bounds_m"], precision=256)
    chart_checks = old.check_chart(attachment["sub_material_chart"], chart_source)
    source_nodes = [node for node in history["quadrature_histories"] if node["trace_id"] == row["receiver_trace_id"]]
    require(source_nodes and all(node["original_frame"] == source_nodes[0]["original_frame"]
            and node["original_point"]["horizontal"] == source_nodes[0]["original_point"]["horizontal"] for node in source_nodes),
            "First receiver has inconsistent literal source material axes")
    sine, cosine = (exact_float(value) for value in source_nodes[0]["original_frame"])
    measure_source = attachment["interface_measure"]["source"]
    extent = exact_float(next(memory["extent_radians"] for memory in history["contact_time_path"]["step"]["previous_memory"]["traces"]
                              if memory["trace_id"] == row["receiver_trace_id"]))
    require(fraction(measure_source["source_extent_radians"]) == extent
            and fraction(measure_source["source_interface_area_density_m2_per_radian"])
                == radius*exact_float(receiver_trace["contact_height_m"])/sine,
            "Reference continuum measure was inferred from quadrature atoms or recomputed dip")
    delta = isub(interval(attachment["over_tip_displacement_bounds_m"]), interval(attachment["sub_material_chart"]["displacement_m"]))
    # Outward compression may marginally widen this natural interval image.
    encloses(attachment["reference_gap_bounds"], iscale(delta,cosine), "True-carrier reference normal projection")
    encloses(attachment["reference_jump_bounds"], iscale(delta,-sine), "True-carrier reference shear projection")
    dt = exact_float(history["literal_dt_myr"])
    axes = {owner: tuple(dec(exact_float(value)*dt) for value in values)
            for owner,values in history["actual_candidate"]["omega_rad_per_myr"]}
    donor_axis, receiver_axis = axes[donor_binding["subducting_plate"]], axes[receiver_binding["overriding_plate"]]
    point = tuple(dec(fraction(value)) for value in chronology["original_point"])
    observed_time, relative = numerical_owner_root(point, donor_axis, receiver_axis, window)
    start, end = (tuple(dec(exact_float(value)) for value in receiver_trace[name]) for name in ("start","end"))
    observed_angle = atan2_positive(dot(end,relative),dot(start,relative))
    contains(theta, observed_angle, "Numerically observed true first-root carrier angle")
    sin_theta, cos_theta = sin_cos(observed_angle)
    carrier = tuple(a*cos_theta+b*sin_theta for a,b in zip(start,end))
    world = rotate(carrier,receiver_axis,observed_time)
    horizontal = tuple(dec(exact_float(value)) for value in source_nodes[0]["original_point"]["horizontal"])
    world_horizontal = rotate(horizontal,receiver_axis,observed_time)
    normal = tuple(dec(sine)*h+dec(cosine)*q for h,q in zip(world_horizontal,world))
    down = tuple(dec(cosine)*h-dec(sine)*q for h,q in zip(world_horizontal,world))
    incoming_unit = rotate(tuple(value/dot(point,point).sqrt() for value in point),donor_axis,observed_time)
    require(max(abs(a-b) for a,b in zip(world,incoming_unit)) < Decimal(10)**-300,
            "Implicit receiver carrier and normalized unchanged incoming label disagree at true owner root")
    own_start, own_end = (tuple(dec(exact_float(value)) for value in donor_trace[name]) for name in ("start","end"))
    inward = tuple(-value for value in cross(own_start,own_end))
    beta = atan(dot(inward,point)/(dot(own_start,point)**2+dot(own_end,point)**2).sqrt())
    z = dec(radius)*beta/dec(fraction(chart_source["width_m"]))
    phi = 1-Decimal("1.5")*z+z**3/2
    def observed_tip(trace_id, bank):
        a, b = exact_float(before_radial[trace_id][bank]), exact_float(candidate_radial[trace_id][bank])
        return dec(a)+observed_time*dec(b-a)
    sub_displacement = phi*observed_tip(row["donor_trace_id"],0)
    over_displacement = observed_tip(row["receiver_trace_id"],1)
    sub_position = tuple((dec(radius)+sub_displacement)*q for q in world)
    over_position = tuple((dec(radius)+over_displacement)*q for q in world)
    for name,value in (("receiver_world_direction",world),("receiver_world_normal",normal),
        ("receiver_world_down_dip",down),("sub_position_bounds_m",sub_position),("over_position_bounds_m",over_position)):
        check_vector(attachment[name],value,"True numerical first-root " + name)
    contains(attachment["reference_gap_bounds"],dec(cosine)*(over_displacement-sub_displacement),"True numerical reference gap")
    contains(attachment["reference_jump_bounds"],-dec(sine)*(over_displacement-sub_displacement),"True numerical reference jump")
    return dict(field_point_checks=field_checks, measure_point_checks=measure_checks,
                source_chart_corner_checks=chart_checks, numerical_owner_roots=1)


def check_measure(raw, history, donor_binding, receiver_binding, theta, window):
    seal(raw, "measure_sha256", "Carrier material/contact measure")
    require(raw["format"] == "native-first-entry-carrier-contact-measure-1"
            and raw["model"] == "full_SO3_carrier_pullback_Jacobian_retained_original_interface_measure",
            "First-entry measure law changed")
    source = raw["source"]
    require(source["trace_id"] == receiver_binding["trace_id"]
            and source["donor_owner"] == donor_binding["subducting_plate"]
            and source["receiver_owner"] == receiver_binding["overriding_plate"],
            "Carrier measure used original neighbouring bank pair")
    same(source["native_angle_bounds"], theta, "Exact implicit native carrier-angle envelope")
    same(source["time_fraction_bounds"], window, "Exact implicit root time envelope")
    require(raw["original_interface_measure_preserved"] is True
            and raw["source_density_interpretation"] == "original_dimensional_continuum_reference_measure"
            and raw["solid_angle_metric"] == "reference_unit_sphere_not_deformed_radial_beam_surface",
            "Carrier measure confused continuum interface and deformed swept material area")
    for name in ("literal_quadrature_atoms_reweighted", "continuum_total_equals_literal_nodal_area_asserted",
            "interface_measure_is_swept_material_measure", "first_entry_chronology_certified",
            "rectangle_is_once_only_archive_area", "source_material_support_certified",
            "nodal_work_or_energy_density_assigned", "physical_pair_admission_authorized", "candidate_mechanics_adopted"):
        require(raw[name] is False, "Measure kernel incorrectly grants " + name)
    require(raw["accepted_native_force_intervals"] == 0 and fraction(raw["accepted_work_increment_j"]) == 0,
            "Measure kernel adopts native clock/work")
    start, end = old.vector(source["start"]), old.vector(source["end"])
    require(dot(start,start) == dot(end,end) == 1 and dot(start,end) == 0, "Carrier is not an original axis quarter")
    dt = fraction(source["dt_myr"])
    require(dt == exact_float(history["literal_dt_myr"]), "Measure changed source duration")
    omega = {owner: old.vector(axis) for owner,axis in source["omega_records"]}
    same(source["omega_records"], [[owner,[rational(exact_float(value)) for value in axis]]
         for owner,axis in history["actual_candidate"]["omega_rad_per_myr"]], "Complete original owner angular velocities")
    axes = {owner:tuple(dec(value*dt) for value in axis) for owner,axis in omega.items()}
    donor_axis, receiver_axis = axes[source["donor_owner"]], axes[source["receiver_owner"]]
    difference = tuple(a-b for a,b in zip(donor_axis, receiver_axis))
    extent, density = fraction(source["source_extent_radians"]), fraction(source["source_interface_area_density_m2_per_radian"])
    require(extent > 0 and density > 0, "Original dimensional continuum measure lost its positive source coefficient")
    scale = dec(extent)/half_pi()
    contains(raw["source_angle_scale"], scale, "Retained quarter-angle scale")
    contains(raw["interface_area_density_per_native_radian_m2"], dec(density)*scale, "Original continuum area density")
    raw_angle_bounds, time_bounds = interval(theta), interval(window)
    angle_bounds = max(Decimal(0),dec(raw_angle_bounds[0])), min(half_pi(),dec(raw_angle_bounds[1]))
    require(angle_bounds[0] <= angle_bounds[1]
            and raw["angle_domain"] == "supplied_enclosure_intersect_declared_native_quarter",
            "Measure discarded or widened its declared quarter intersection")
    for endpoint in angle_bounds:
        contains(raw["native_quarter_angle_enclosure"],endpoint,"Declared analytic quarter intersection")
    span = angle_bounds[1]-angle_bounds[0]
    contains(raw["native_quarter_angle_span_rad"],span,"Analytic quarter intersection span")
    contains(raw["rectangle_interface_area_m2"], dec(density)*scale*span,
             "Interface rectangle measure retained independently of time width")
    checks = 0
    for angle, instant in product(set(angle_bounds), set(time_bounds)):
        angle, instant = dec(angle), dec(instant)
        sine,cosine = sin_cos(angle)
        carrier = tuple(dec(a)*cosine+dec(b)*sine for a,b in zip(start,end))
        tangent = tuple(-dec(a)*sine+dec(b)*cosine for a,b in zip(start,end))
        world, world_tangent = rotate(carrier,receiver_axis,instant), rotate(tangent,receiver_axis,instant)
        source_q = rotate(world,tuple(-v for v in donor_axis),instant)
        source_theta = rotate(world_tangent,tuple(-v for v in donor_axis),instant)
        source_time = rotate(cross(tuple(-v for v in difference),world),tuple(-v for v in donor_axis),instant)
        signed = dot(difference,world_tangent)
        direct = dot(source_q,cross(source_theta,source_time))
        epsilon = Decimal(10)**-(PRECISION-40)*max(Decimal(1),abs(signed))
        require(abs(signed-direct) <= epsilon, "Carrier pullback Jacobian violates direct oriented determinant identity")
        for name,value in (("carrier_direction",carrier),("carrier_tangent",tangent),
            ("receiver_world_direction",world),("receiver_world_tangent",world_tangent),
            ("source_material_direction",source_q),("source_theta_derivative",source_theta),
            ("source_time_derivative",source_time)):
            check_vector(raw[name],value,"Independent carrier-map " + name)
        contains(raw["signed_solid_angle_jacobian"],signed,"Signed material pullback Jacobian")
        contains(raw["unsigned_solid_angle_jacobian"],abs(signed),"Unsigned material pullback Jacobian")
        contains(raw["rectangle_multiplicity_solid_angle_sr"],
            abs(signed)*span*dec(time_bounds[1]-time_bounds[0]),
            "Swept rectangle multiplicity measure")
        contains(raw["computational_angle_bounds"],angle*scale,"Measure computational quarter coordinate")
        checks += 1
    signed_low,signed_high = interval(raw["signed_solid_angle_jacobian"])
    sign = 1 if signed_low > 0 else -1 if signed_high < 0 else 0 if signed_low == signed_high == 0 else None
    require(raw["jacobian_sign"] == sign and (source["required_jacobian_sign"] is None
            or source["required_jacobian_sign"] == sign), "Carrier Jacobian sign was inferred from a point sample")
    return checks


def check_original_ledger(raw, history):
    keys(raw, "format original_node_count quadrature_histories total_original_area_m2 original_energy_work_totals_j "
        "source_work_before unaccepted_candidate_work beam_configuration_before beam_configuration_after "
        "local_energy_work_counted_once unchanged_global_beam_dofs_and_stiffness original_atoms_assigned_to_first_entry_points "
        "finite_measure_assigned_to_first_entry_point candidate_mechanics_adopted accepted_loaded_native_intervals "
        "accepted_source_work_j transfer_external_work_j transfer_dissipation_j ledger_sha256", "Original once-only nodal ledger")
    seal(raw,"ledger_sha256","Original once-only quadrature ledger")
    require(raw["format"] == "native-first-entry-handover-original-once-only-ledger-1"
            and raw["original_node_count"] == 16 and len(raw["quadrature_histories"]) == 16,
            "First-root source lost original sixteen finite histories")
    nodes = history["quadrature_histories"]
    require(len(nodes) == 16 and len({node["point_id"] for node in nodes}) == 16,
            "Original quadrature cardinality/identity changed")
    area, totals = Fraction(0), {}
    for row,node in zip(raw["quadrature_histories"],nodes):
        local = node["unaccepted_candidate_state"]
        expected = dict(point_id=node["point_id"], trace_id=node["trace_id"],
            original_area_m2=rational(exact_float(node["original_area_m2"])),
            disposition="retained_once_in_original_quadrature_lineage", local_state_before=local, local_state_after=local)
        same(row,expected,"Original full loaded LocalContactState and once-only measure")
        require(local["area_m2"] == node["original_area_m2"] and local["plastic_slip_m"] < 0
                and local["cumulative_slip_m"] > 0 and local["traction_normal_pa"] < 0,
                "Original finite state lost its loaded compressed plastic configuration")
        area += exact_float(node["original_area_m2"])
        for name,value in local.items():
            if name.endswith("_j"):
                totals[name] = totals.get(name,Fraction(0)) + exact_float(value)
    require(fraction(raw["total_original_area_m2"]) == area,"Original atomic interface area changed")
    same(raw["original_energy_work_totals_j"], {name:rational(value) for name,value in totals.items()},
         "Complete exact once-only finite energy/work totals")
    same(raw["source_work_before"],history["before_work_ledgers"],"Original accepted work ledger")
    same(raw["unaccepted_candidate_work"],history["unaccepted_candidate_work_ledgers"],"Original unaccepted candidate work ledger")
    beam = dict(parameters=history["contact_time_path"]["initial_mechanics"]["bending_parameters"],
        candidate_transverse_displacement_m=history["actual_candidate"]["transverse_displacement_m"],
        original_stiffness=history["original_reference"]["stiffness"])
    same(raw["beam_configuration_before"],beam,"Original beam configuration and stiffness")
    same(raw["beam_configuration_after"],beam,"Unchanged beam configuration and stiffness")
    require(raw["local_energy_work_counted_once"] is True and raw["unchanged_global_beam_dofs_and_stiffness"] is True,
            "First-root source copied or lost finite energy/work or beam basis")
    for name in ("original_atoms_assigned_to_first_entry_points", "finite_measure_assigned_to_first_entry_point", "candidate_mechanics_adopted"):
        require(raw[name] is False,"First-root source ledger incorrectly grants " + name)
    for name in ("accepted_loaded_native_intervals", "accepted_source_work_j", "transfer_external_work_j", "transfer_dissipation_j"):
        require(type(raw[name]) is int and raw[name] == 0,"First-root source ledger adopts native clock/work")
    return len(totals)


MUTATIONS = (
    ("first_root_midpoint_claimed", "first_root_midpoint_used"),
    ("first_root_window_replaced_by_midpoint", "original_physical_binding.time_fraction_bounds"),
    ("receiving_trace_replaced_by_donor", "attachments[0].source_field_query.trace_id"),
    ("changed_first_receiver_material", "attachments[0].desired_fragment_pair[1]"),
    ("source_beam_width_changed", "attachments[0].sub_material_chart.source.width_m"),
    ("incoming_source_beam_shape_changed", "attachments[0].sub_material_chart.shape"),
    ("incoming_own_beam_amplitude_changed", "attachments[0].sub_tip_displacement_bounds_m"),
    ("receiving_frame_changed", "attachments[0].receiver_world_normal[0]"),
    ("original_plastic_memory_changed", "attachments[0].source_field_query.plastic_slip_bounds_m"),
    ("original_cumulative_memory_changed", "attachments[0].source_field_query.cumulative_slip_bounds_m"),
    ("original_energy_density_changed", "attachments[0].source_field_query.total_energy_density_bounds_j_m2"),
    ("original_reference_measure_changed", "attachments[0].interface_measure.source.source_interface_area_density_m2_per_radian"),
    ("swept_material_jacobian_changed", "attachments[0].interface_measure.signed_solid_angle_jacobian"),
    ("attachment_commit_removed", "attachments[0].attachment_committed"),
    ("finite_area_assigned_to_point", "attachments[0].finite_point_area_m2"),
    ("nodal_work_assigned_to_point", "attachments[0].finite_point_work_j"),
    ("fabricated_cumulative_work_density", "attachments[0].source_field_query.cumulative_work_density_j_m2"),
    ("once_only_attachment_receipt_duplicated", "attachments[0].transport_family.receipts"),
    ("unpaid_transfer_work_created", "additional_handover_work_j"),
    ("original_source_transverse_axis_removed", "complete_original_source_history.actual_candidate.omega_rad_per_myr"),
    ("transport_law_provenance_changed", "explicit_law.provenance"),
    ("independent_complete_source_changed", "expected_source_history_dto.actual_candidate.states"),
)


def check_mutations(raw):
    seal(raw,"receipt_sha256","Recorded first-entry mutation refusals")
    keys(raw,"format count mutations receipt_sha256","Recorded first-entry mutation refusals")
    require(raw["format"] == "native-first-entry-handover-mutation-refusals-1" and raw["count"] == 30
            and len(raw["mutations"]) == 30,"First-entry recorded mutation coverage changed")
    expected = []
    for label in ENTRY_LABELS:
        expected.extend((label,name,path) for name,path in (
            ("unknown_complete_schema_field","unknown_field"),("top_level_literal_type_changed","precision")))
        if label == "transverse_masked_main_entry":
            expected.extend((label,name,path) for name,path in MUTATIONS)
    for row,(context,label,path) in zip(raw["mutations"],expected):
        checksum = "valid_rehashed_source_checksum" if label == "independent_complete_source_changed" else "valid_rehashed_handover_checksum"
        keys(row,"context_label label changed_path rejected " + checksum + " exception_type message mutated_dto_sha256",
             "First-entry recorded mutation refusal")
        require(row["context_label"] == context and row["label"] == label and row["changed_path"] == path
                and row["rejected"] is True and row[checksum] is True
                and row["exception_type"] in ("ValueError","TypeError") and type(row["message"]) is str and row["message"]
                and type(row["mutated_dto_sha256"]) is str and len(row["mutated_dto_sha256"]) == 64
                and all(letter in "0123456789abcdef" for letter in row["mutated_dto_sha256"]),
                "Rehashed first-entry mutation was not refused: " + label)
    return len(expected)


def check_observations(raw, handovers):
    seal(raw,"receipt_sha256","Implicit first-root reference observations")
    keys(raw,"format count observations receipt_sha256","Implicit first-root reference observations")
    require(raw["format"] == "native-first-entry-handover-reference-observations-1" and raw["count"] == 4
            and len(raw["observations"]) == 4,"Implicit first-root observation count changed")
    for row,record in zip(raw["observations"],handovers):
        attachment = record["handover"]["attachments"][0]
        field = attachment["source_field_query"]
        expected = dict(label=record["label"], desired_fragment_pair=attachment["desired_fragment_pair"],
            source_receiver_trace_id=attachment["original_receiver_trace_id"], own_donor_beam_trace_id=attachment["original_donor_trace_id"],
            source_free_gap_bounds=field["free_gap_bounds_m"], transported_free_gap_bounds=field["free_gap_bounds_m"],
            source_free_jump_bounds=field["free_jump_bounds_m"], transported_free_jump_bounds=field["free_jump_bounds_m"],
            implicit_root_reference_cancels_symbolically=True, attachment_revision=1,
            finite_point_area_m2=None, finite_point_work_j=None)
        same(row,expected,"Exact frozen source and correlated first-event geometry cancellation")
    return 4


def _run_audit(control):
    """Audit sealed saved evidence; return an unsealed standard-library receipt."""
    started = time.perf_counter()
    control = Path(control).resolve()
    require(control.is_dir(),"First-entry handover control evidence missing")
    names = (
        "sources_before.json","sources_after.json","source_validation_receipt.json",
        "source_history_before.json","source_history_after.json","source_checkpoint_before.json","source_checkpoint_after.json",
        "force_candidate_before.json","force_candidate_after.json","declared_handover_law.json","point_certificates.json",
        "original_first_entry_bindings.json","first_entry_handovers.json","first_entry_handovers_replayed.json",
        "negative_contexts.json","first_root_reference_observations.json","original_quadrature_ledger.json","mutation_refusals.json",
    )
    artifacts = {name:file_sha(relative_file(control,name)) for name in names}
    report = read(control/"report.json")
    seal(report,"report_sha256","First-entry handover control report")
    require(report["format"] == "native-first-entry-contact-handover-control-1" and report["status"] == "PASS",
            "First-entry handover control failed or changed format")
    same(report["artifacts_sha256"],artifacts,"Complete saved input artifact inventory")
    prior = verify_prior()
    authority = read(control/"source_validation_receipt.json")
    seal(authority,"receipt_sha256","Verified prior source/evidence references")
    require(authority["format"] == "native-first-entry-handover-source-validation-1" and authority["status"] == "PASS"
            and authority["original_source_count"] == 674 and authority["verified_prior_manifest_records"] == 185
            and authority["prior_tests_reexecuted"] is False,"Prior source ancestry changed")
    for name,record in authority["source_artifacts"].items():
        path = relative_file(ROOT,name)
        require(path.stat().st_size == record["bytes"] and file_sha(path) == record["sha256"],"Original source artifact changed: " + name)
    baseline = read(BASE/"starting_sources.json")
    require(type(baseline) is dict and len(baseline) == 674,"Original674Python baseline changed")
    for directory in ("release_gates_02","control_release_02"):
        for stage in ("before","after"):
            same(baseline,read(HANDOVER/directory/("sources_"+stage+".json")),"Original674source release freeze")
    before,after = read(control/"sources_before.json"),read(control/"sources_after.json")
    same(before,after,"First-entry handover source freeze")
    current = {path.relative_to(ROOT).as_posix():file_sha(path) for directory in ("tectonics","tests","scripts","analysis")
               for path in sorted((ROOT/directory).rglob("*.py"))}
    same(current,after,"Exhaustive current Python-source enumeration")
    require(all(current.get(name) == expected for name,expected in baseline.items()),"A frozen original Python source changed")
    require(report["original_source_count"] == 674 and report["source_count"] == len(current)
            and report["added_python_sources"] == sorted(set(current)-set(baseline))
            and report["existing_sources_changed"] == [],"Current source count/ancestry changed")
    pairs = (("source_history_before.json","source_history_after.json"),
        ("source_checkpoint_before.json","source_checkpoint_after.json"),
        ("force_candidate_before.json","force_candidate_after.json"),
        ("first_entry_handovers.json","first_entry_handovers_replayed.json"))
    same(report["byte_pairs"],[list(pair) for pair in pairs],"Source/replay exact bytepair references")
    byte_pairs = []
    for first,second in pairs:
        require((control/first).read_bytes() == (control/second).read_bytes(),"Original source or strict restart replay bytes changed")
        byte_pairs.append(dict(first=first,second=second,bytes=(control/first).stat().st_size,sha256=file_sha(control/first)))
    for name,source in (("source_history_before.json",PHYSICAL/"control_release_01/original_history.json"),
        ("source_checkpoint_before.json",MATERIAL/"control_release_02/original_force/initial.json"),
        ("force_candidate_before.json",MATERIAL/"control_release_02/original_force/force_candidate.json")):
        require((control/name).read_bytes() == source.read_bytes(),"Full independent source bytes changed: " + name)
    history = read(control/"source_history_before.json")
    seal(history,"binding_sha256","Original complete physical history")
    checkpoint,candidate = read(control/"source_checkpoint_before.json"),read(control/"force_candidate_before.json")
    require(type(candidate) is dict and len(candidate) == 14,"Fourteen original force candidate fields required")
    same(history["actual_candidate"],candidate,"Full original actual force candidate")
    same(history["accepted_before"],checkpoint["native_underthrust_motion"],"Full authentic original force checkpoint")
    require(history["original_checkpoint_sha256"] == file_sha(control/"source_checkpoint_before.json"),"Original source checkpoint byte identity changed")
    law = read(control/"declared_handover_law.json")
    keys(law,"model provenance","Declared constitutive transport law")
    require(law["model"] == "receiver_attached_complete_lumped_contact_configuration"
            and type(law["provenance"]) is str and law["provenance"].strip() == law["provenance"] and law["provenance"],
            "Transport law lacks explicit authoritative provenance")
    same(law,report["law"],"Full explicit control transport law")
    records = read(control/"point_certificates.json")
    proofs = independently_audit_chronologies(records,history,checkpoint,candidate)
    chronologies = {row["label"]:row["certificate"] for row in records}
    bindings = read(control/"original_first_entry_bindings.json")
    same(bindings,read(PHYSICAL/"control_release_01/incoming_first_entry_bindings.json"),"Full original first-entry physical bindings")
    bindings = {row["label"]:row["binding"] for row in bindings}
    handovers = read(control/"first_entry_handovers.json")
    require([row["label"] for row in handovers] == list(ENTRY_LABELS),"Four genuine first-root contexts changed")
    results = {name:0 for name in ("field_point_checks","measure_point_checks","source_chart_corner_checks","numerical_owner_roots")}
    for record in handovers:
        keys(record,"label handover","Positive first-root record")
        label = record["label"]
        checked = check_handover(record["handover"],bindings[label],chronologies[label],proofs[label],history,law)
        for name,count in checked.items():
            results[name] += count
    cross_attachment = next(row["handover"]["attachments"][0] for row in handovers if row["label"] == "transverse_masked_main_entry")
    require(cross_attachment["desired_fragment_pair"] == ["octant:6","octant:3"]
            and cross_attachment["original_donor_trace_pair"] == ["octant:6","octant:2"]
            and cross_attachment["original_receiver_trace_pair"] == ["octant:7","octant:3"]
            and cross_attachment["original_donor_trace_id"] != cross_attachment["original_receiver_trace_id"],
            "Actual six-to-three crossing borrowed its donor spring history")
    negatives = read(control/"negative_contexts.json")
    require([row["label"] for row in negatives] == [label for label in POINT_LABELS if label not in ENTRY_LABELS],
            "Three negative first-entry controls changed")
    for record in negatives:
        keys(record,"label binding handover_created handover replayed_handover","Negative first-entry record")
        label = record["label"]
        same(record["binding"],bindings[label],"Negative original physical context")
        same(record["handover"],record["replayed_handover"],"Negative first-root restart")
        require(record["handover_created"] is False and not proofs[label]["first_receivers"],"Negative context acquired a contact")
        check_handover(record["handover"],bindings[label],chronologies[label],proofs[label],history,law)
    energy_fields = check_original_ledger(read(control/"original_quadrature_ledger.json"),history)
    observations = check_observations(read(control/"first_root_reference_observations.json"),handovers)
    mutations = check_mutations(read(control/"mutation_refusals.json"))
    counts = dict(original_candidate_field_count=14,original_node_count=16,original_context_count=7,
        first_root_handover_count=4,negative_context_count=3,first_root_reference_observation_count=4,mutation_refusal_count=30)
    for name,value in counts.items():
        require(type(report[name]) is int and report[name] == value,"Control count changed: " + name)
    for name in ("source_before_after_unchanged","original_typed_checkpoint_loaded","original_force_re_solved",
        "history_builder_original_force_re_solved","prior_material_state_replayed","original_candidate_all_fields_identical",
        "first_receiver_pair_6_to_3_committed","cross_pair_receiving_history_and_donor_beam_distinct",
        "source_field_frozen_at_implicit_first_entry","implicit_root_geometry_cancelled_symbolically",
        "source_history_bytes_unchanged","source_checkpoint_bytes_unchanged","source_candidate_bytes_unchanged",
        "original_quadrature_measure_work_counted_once","unchanged_global_beam_dofs_and_stiffness"):
        require(report[name] is True,"Control lacks source or transport guarantee: " + name)
    for name in ("first_root_midpoint_used","transported_candidate_is_force_equilibrium",
        "native_full_bank_and_clamp_support_certified","finite_measure_assigned_to_first_entry_point",
        "nodal_work_assigned_to_first_entry_point","whole_first_entry_interface_domain_certified",
        "candidate_mechanics_adopted","physical_pair_admission_authorized","global_native_inventory_updated"):
        require(report[name] is False,"Control incorrectly grants " + name)
    for name in ("transfer_external_work_j","transfer_dissipation_j","accepted_loaded_native_intervals","accepted_source_work_j"):
        require(type(report[name]) is int and report[name] == 0,"Control adopted force interval/work")
    artifacts["report.json"] = file_sha(control/"report.json")
    require(not any(name == "tectonics" or name.startswith("tectonics.") for name in sys.modules),
            "Independent audit imported a production module")
    return dict(format="native-first-entry-contact-handover-independent-audit-1",status="PASS",
        seconds=time.perf_counter()-started,decimal_precision=PRECISION,standard_library_only=True,production_modules_imported=False,
        independent_original_owner_and_all_receiver_root_partitions_rechecked=True,
        independent_whole_new_spatial_bank_or_force_admission_proof=False,
        numerical_geometry_and_continuum_points_only=True,
        numerical_root_observations_used_only_for_audit=True,production_transfer_clock_selected=False,
        numerical_scope="Decimal owner-equation root observation within exact bracket; independent continuum field and carrier-map corner values; no point observation grants a whole new domain proof",
        exact_scope="Complete original owner/all-receiver event partitions, unchanged Fourier pieces and clocks, explicit reference/strain identities, original finite area/energy/work counted once, unchanged global beam DOFs and stiffness",
        force_covectors_or_equilibrium_audited=False,first_root_midpoint_used=False,
        first_receiver_pair_6_to_3_committed=True,cross_pair_receiving_history_and_donor_beam_distinct=True,
        source_field_frozen_at_implicit_first_entry=True,implicit_root_geometry_cancelled_symbolically=True,
        original_quadrature_measure_work_counted_once=True,unchanged_global_beam_dofs_and_stiffness=True,
        original_energy_work_field_count=energy_fields,first_root_reference_observation_count=observations,
        mutation_refusal_count=mutations,recorded_mutation_refusals_verified=True,mutation_tests_independently_reexecuted=False,
        original_material_parent_count=8,original_context_count=7,first_root_handover_count=4,negative_context_count=3,original_node_count=16,
        original_source_count=674,source_count=len(current),existing_sources_changed=[],source_before_after_unchanged=True,
        exhaustive_current_source_map_verified=True,byte_pairs=byte_pairs,
        chronology_leaf_count=sum(proof["leaf_count"] for proof in proofs.values()),
        chronology_root_count=sum(proof["root_count"] for proof in proofs.values()),
        whole_first_entry_interface_domain_certified=False,native_full_bank_and_clamp_support_certified=False,
        finite_measure_assigned_to_first_entry_point=False,nodal_work_assigned_to_first_entry_point=False,
        candidate_mechanics_adopted=False,physical_pair_admission_authorized=False,global_native_inventory_updated=False,
        transfer_external_work_j=0,transfer_dissipation_j=0,accepted_loaded_native_intervals=0,accepted_source_work_j=0,
        audit_source_sha256=file_sha(Path(__file__)),input_artifact_sha256=artifacts,**prior,**results)


def run_audit(control):
    """Run with an isolated Decimal context even for direct library callers."""
    with localcontext() as context:
        context.prec = PRECISION
        return _run_audit(control)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    with localcontext() as context:
        context.prec = PRECISION
        receipt = run_audit(args.control)
    receipt["receipt_sha256"] = digest(receipt)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("xb") as stream:
        stream.write(encoded(receipt)+b"\n")
    print(json.dumps({name:receipt[name] for name in ("status","seconds","source_count",
        "first_root_handover_count","negative_context_count","original_node_count","numerical_owner_roots")},sort_keys=True))


if __name__ == "__main__":
    main()
