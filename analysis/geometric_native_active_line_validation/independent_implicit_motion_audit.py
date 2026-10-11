"""Stdlib-only audit of first adoption, fresh refusal, and exact incoming flux.

The force equations are checked by the separate authentic production solve.
This audit checks exported source links, the adopted fields and ledger, and
independently encloses the continuous incoming edge flux with rational SO(3).
"""
from __future__ import annotations

from fractions import Fraction as F
from math import factorial, isqrt, isfinite
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.geometric_native_active_line_validation.independent_active_line_audit import interval, rational
from analysis.geometric_native_active_line_validation.independent_line_oracle import (
    cross_matrix, exponential_coefficients, sum_matrices,
)
from analysis.geometric_native_active_line_validation.verify_baseline import (
    BASE, PRIOR, digest, read, sealed, sources, value_sha, verify_baseline, write,
)


def scalar(value):
    if type(value) is int:
        return F(value)
    if type(value) is float and isfinite(value):
        return F.from_float(value)
    raise ValueError("Independent source algebra requires literal finite numbers")


def plus(a, b):
    return a[0]+b[0], a[1]+b[1]


def times(a, b):
    values = tuple(x*y for x in a for y in b)
    return min(values), max(values)


def negative(a):
    return -a[1], -a[0]


def dot(a, b):
    result = F(0), F(0)
    for x, y in zip(a, b):
        result = plus(result, times(x, y))
    return result


def cross(a, b):
    return tuple(plus(times(a[i], b[j]), negative(times(a[j], b[i])))
        for i, j in ((1, 2), (2, 0), (0, 1)))


def rotation_matrix(step, degree=18):
    axis = tuple(scalar(value)*scalar(step["dt_myr"]) for value in step["omega_rad_per_myr"])
    norm = sum(map(abs, axis), F(0))
    if norm > F(1, 2):
        raise ValueError("Independent rotation tail requires this saved small-step domain")
    center = sum_matrices(exponential_coefficients(cross_matrix(axis), degree))
    error = 2*norm**(degree+1)/factorial(degree+1)
    return tuple(tuple((x-error, x+error) for x in row) for row in center)


def rotate(vector, steps):
    for step in steps:
        vector = tuple(dot(row, vector) for row in rotation_matrix(step))
    return vector


def plane_vector(plane, extra=()):
    scale = max(map(abs, plane["normal"]))
    if not scale:
        raise ValueError("Native plane has a zero normal")
    vector = tuple((F(x, scale), F(x, scale)) for x in plane["normal"])
    return rotate(vector, (*plane["rotations"], *extra))


def ray_vector(ray):
    if ray["kind"] == "seed":
        scale = max(map(abs, ray["reference"]))
        vector = tuple((F(x, scale), F(x, scale)) for x in ray["reference"])
        return rotate(vector, ray["rotations"])
    if ray["kind"] != "cross" or len(ray["planes"]) != 2:
        raise ValueError("Unknown exact native ray expression")
    result = cross(*(plane_vector(row) for row in ray["planes"]))
    return result if ray["sign"] == 1 else tuple(map(negative, result))


def unit_ray(ray, bits=512):
    vector = ray_vector(ray)
    norm2 = dot(vector, vector)
    if norm2[0] <= 0:
        raise ValueError("Independent native ray norm is not positive")
    scale = 1 << bits
    lower = F(isqrt((norm2[0].numerator << (2*bits))//norm2[0].denominator), scale)
    upper = F(isqrt((norm2[1].numerator << (2*bits))//norm2[1].denominator)+1, scale)
    return tuple(times(value, (1/upper, 1/lower)) for value in vector)


def overlapping(first, second):
    if max(first[0], second[0]) > min(first[1], second[1]):
        raise AssertionError("Independent exact enclosures do not overlap")


def audit_incoming_geometry(geometry, candidate, epoch):
    body = dict(geometry)
    seal = body.pop("geometry_sha256")
    if value_sha(body) != seal or geometry["source_epoch_sha256"] != epoch["epoch_sha256"]:
        raise AssertionError("Incoming geometry source or full-body seal differs")
    if geometry["omega_rad_per_myr"] != candidate["omega_rad_per_myr"]:
        raise AssertionError("Newborn witness uses different Euler axes from the genuine fresh solve")
    if geometry["dt_myr"] != candidate["dt_s"]/(365.25*86400.*1e6):
        raise AssertionError("Incoming geometry literal clock differs from its force candidate")
    if rational(geometry["exact_literal_dt_myr"]) != scalar(geometry["dt_myr"]):
        raise AssertionError("Incoming geometry exact clock differs")
    if (geometry["force_admission_authorized"] or geometry["complete_all_pairs_exclusion_proved"]
            or not geometry["positive_newborn_contact_proved"]
            or not geometry["immediate_unsupported_incoming_proved"]):
        raise AssertionError("Newborn geometry scope differs")
    omega = {owner: tuple((scalar(x), scalar(x)) for x in vector)
        for owner, vector in candidate["omega_rad_per_myr"]}
    contacts, edges, observations = geometry["positive_contacts"], 0, []
    if len(contacts) != 4:
        raise AssertionError("Actual newborn contact witnesses were lost")
    for contact in contacts:
        area = interval(contact["endpoint_intersection_area_bounds_sr"])
        if (area[0] <= 0 or contact["initial_positive_area_intersection"]
                or contact["donor_owner"] == contact["receiver_owner"]
                or "hot_birth" not in (contact["donor_material_kind"], contact["receiver_material_kind"])):
            raise AssertionError("A positive newborn witness lacks its actual source distinction")
        difference = tuple(plus(a, negative(b)) for a, b in zip(
            omega[contact["donor_owner"]], omega[contact["receiver_owner"]]))
        for edge in contact["immediate_incoming_edges"]:
            edges += 1
            arc = edge["exact_initial_shared_edge"]
            if arc["length_radians"] <= 0 or edge["safe_positive_duration_with_same_euler_vectors_exists"]:
                raise AssertionError("Immediate edge has an invalid length or truncation claim")
            points = tuple(unit_ray(arc[name]) for name in ("start", "end"))
            normal = plane_vector(edge["receiver_inward_plane"])
            fluxes = tuple(dot(normal, cross(difference, point)) for point in points)
            if any(value[0] <= 0 for value in fluxes):
                raise AssertionError("Independent continuous-edge endpoint flux is not positive")
            for actual, exported in zip(fluxes, edge["endpoint_incoming_flux_bounds_per_myr"]):
                overlapping(actual, interval(exported))
            interior = tuple(times(plus(a, b), (F(1, 2), F(1, 2))) for a, b in zip(*points))
            flux = dot(normal, cross(difference, interior))
            if flux[0] <= 0:
                raise AssertionError("Independent analytic interior incoming flux is not positive")
            overlapping(flux, interval(edge["interior_incoming_flux_bounds_per_myr"]))
            for bank in ("donor", "receiver"):
                polygon = contact["initial_"+bank+"_geometry"]
                for row in edge[bank+"_other_support_bounds"]:
                    support = plane_vector(polygon["supports"][row["index"]], polygon["pose"])
                    bound = dot(support, interior)
                    if bound[0] <= 0:
                        raise AssertionError("Independent interior other support is not positive")
                    overlapping(bound, interval(row["bounds"]))
            observations.append(dict(donor=contact["donor_component_id"],
                receiver=contact["receiver_component_id"],
                independently_enclosed_interior_flux_lower_per_myr=float(flux[0]),
                endpoint_flux_lowers_per_myr=[float(row[0]) for row in fluxes]))
    if edges != 2:
        raise AssertionError("The actual two immediate positive-entry edges were lost")
    return observations


def audit(out):
    started, out = time.perf_counter(), Path(out)
    baseline = verify_baseline()
    frozen = read(BASE/"frozen_release_sources.json")
    if sources() != frozen:
        raise ValueError("Independent motion audit requires the complete frozen source map")
    report = sealed(out/"motion_control_report.json", "report_sha256")
    required = ("native_force_admission_proved", "full_first_interval_balances_passed",
        "subsequent_candidate_solved", "subsequent_candidate_refused", "subsequent_force_balances_passed",
        "refusal_preserved_before_bytes", "joint_restart_passed", "joint_restart_repeats_original_genuine_force")
    if (report["status"] != "PASS" or report["accepted_loaded_native_intervals"] != 1
            or report["native_force_admission_scope"] != "first original physical interval only"
            or any(report[name] is not True for name in required)
            or report["subsequent_interval_accepted"] or report["entire_point_1_completed"]
            or report["physical_inputs_invented"] or report["continuum_work_added_to_Gauss_ledger"]):
        raise AssertionError("First acceptance and subsequent refusal scope differs")
    source = read(out/"source_history_before.json")
    original = read(PRIOR/"control_release_02/source_history_before.json")
    if source != original:
        raise AssertionError("First interval differs from its independently sealed physical source")
    motion = read(out/"accepted_motion_before.json")
    snapshot, candidate = motion["snapshot"], source["actual_candidate"]
    if motion["original_physical_history"] != source or value_sha(snapshot) != motion["state_sha256"]:
        raise AssertionError("Adopted snapshot or complete original physical source differs")
    for name in ("states", "transverse_displacement_m"):
        if snapshot[name] != candidate[name]:
            raise AssertionError("The accepted "+name+" differs from the genuine first force result")
    if snapshot["spatial_memory"] != candidate["spatial_step"]["memory"]:
        raise AssertionError("The accepted exact spatial constitutive memory differs")
    fields = [[row[0], *[field["coefficients"] for field in row[1:]]]
        for row in candidate["spatial_step"]["anchor_fields"]]
    if snapshot["free_anchor_fields"] != fields:
        raise AssertionError("Accepted exact continuous anchor coefficients differ")
    owners = [[owner, [dict(dt_myr=source["literal_dt_myr"], omega_rad_per_myr=omega)]]
        for owner, omega in candidate["omega_rad_per_myr"]]
    if snapshot["owner_histories"] != owners or snapshot["elapsed_time_s"] != candidate["dt_s"]:
        raise AssertionError("Accepted ordered SO(3) or physical clock differs")
    if len(snapshot["receipts"]) != 1 or snapshot["accepted_loaded_native_intervals"] != 1:
        raise AssertionError("Adopted interval count differs")
    if snapshot["preparation_work_j"] != source["accepted_before"]["initial_mechanics"]["preparation_work_j"]:
        raise AssertionError("The original physical preparation work was substituted")
    for ledger, diagnostic in (("cumulative_source_work_j", "source_work_j"),
            ("cumulative_basal_dissipation_j", "basal_dissipation_j"),
            ("cumulative_bending_numerical_dissipation_j", "bending_numerical_dissipation_j")):
        if snapshot[ledger] != candidate["diagnostics"][diagnostic]:
            raise AssertionError("A first interval mechanical ledger was substituted")
    if snapshot["continuous_contact_work_added_to_mechanical_ledger"]:
        raise AssertionError("Continuum admission work was added to the original Gauss ledger")
    radius = scalar(source["origin_surface"]["radius_km"])*1000
    coordinates = [scalar(x) for row in candidate["rotation_increment_radians"] for x in row]
    coordinates += [scalar(x)/radius for row in candidate["transverse_displacement_m"] for x in row[1:]]
    stiffness = source["original_reference"]["stiffness"]
    bending = sum((coordinates[i]*scalar(value)*coordinates[j]/2
        for i, row in enumerate(stiffness) for j, value in enumerate(row)), F(0))
    accounted = sum((scalar(local["cumulative_external_work_j"]) for _, local in snapshot["states"]), F(0))
    accounted += scalar(snapshot["cumulative_basal_dissipation_j"])+bending
    accounted += scalar(snapshot["cumulative_bending_numerical_dissipation_j"])
    supplied = scalar(snapshot["preparation_work_j"])+scalar(snapshot["cumulative_source_work_j"])
    energy_error = abs(accounted-supplied)/max(abs(accounted), abs(supplied))
    if energy_error >= F(5, 10**10):
        raise AssertionError("Independent exact first cumulative mechanical energy does not balance")
    load = read(out/"subsequent_source_load.json")
    next_candidate = read(out/"subsequent_force_candidate.json")
    load_body = dict(load)
    load_sha = load_body.pop("load_sha256")
    if value_sha(load_body) != load_sha or load["epoch_sha256"] != snapshot["current_material_epoch"]["epoch_sha256"]:
        raise AssertionError("The subsequent load differs from the accepted material partition")
    if (next_candidate["before_state_sha256"] != motion["state_sha256"]
            or next_candidate["current_load_sha256"] != load_sha
            or next_candidate["before_accepted_intervals"] != 1
            or next_candidate["candidate_mechanics_adopted"]
            or not next_candidate["new_temporal_material_admission_required"]):
        raise AssertionError("The genuine fresh force candidate links or admission scope differ")
    source_error = read(out/"subsequent_source_operator_error.json")
    if (source_error["load_sha256"] != load_sha or source_error["material_midpoint_debit"]
            or rational(source_error["relative_residual_norm_upper"]) >= F(5, 10**10)):
        raise AssertionError("Fresh operator enclosure or material debit scope differs")
    refusal = sealed(out/"subsequent_temporal_refusal.json", "refusal_sha256")
    if (refusal["previous_state_sha256"] != motion["state_sha256"]
            or refusal["current_load_sha256"] != load_sha or refusal["dt_s"] != next_candidate["dt_s"]
            or refusal["physical_pair_admission_authorized"] or refusal["candidate_mechanics_adopted"]
            or refusal["material_candidate_committed"] or refusal["previous_force_memory_work_and_parent_reservoirs_changed"]):
        raise AssertionError("The material refusal does not preserve the accepted force state")
    incoming = audit_incoming_geometry(refusal["newborn_geometry"], next_candidate,
        snapshot["current_material_epoch"])
    refusals = read(out/"semantic_refusals.json")
    if len(refusals) != 5 or any(row["rejected"] is not True
            or row["root_state_checksum_resealed_only"] is not True
            or row["nested_proof_checksums_resealed"] is not False for row in refusals):
        raise AssertionError("Semantic substitutions lack the declared limited reseal evidence")
    pairs = report["byte_pairs"]
    if len(pairs) != 6:
        raise AssertionError("Joint replay/refusal preservation byte pairs are incomplete")
    for row in pairs:
        a, b = out/row["first"], out/row["second"]
        if a.read_bytes() != b.read_bytes() or digest(a) != row["sha256"] or a.stat().st_size != row["bytes"]:
            raise AssertionError("Joint replay/refusal preservation bytes differ")
    if read(out/"sources_before.json") != frozen or read(out/"sources_after.json") != frozen or sources() != frozen:
        raise AssertionError("Sources changed during the physical control or audit")
    receipt = dict(format="native-implicit-motion-independent-export-audit-1", status="PASS",
        scope="exact adopted source fields, first mechanical ledger, genuine next-source links, rational all-edge incoming flux and preservation bytes",
        production_force_equations_independently_resolved_here=False,
        native_endpoint_intersection_area_reproved_here=False,
        endpoint_area_witness_checked_against_production_rational_lower_bounds=True,
        accepted_loaded_native_intervals=1, subsequent_interval_accepted=False,
        entire_point_1_completed=False, new_physical_inputs_invented=False,
        independently_enclosed_first_energy_relative_error=float(energy_error),
        original_gauss_atom_count=len(snapshot["states"]), positive_newborn_contact_count=4,
        independently_positive_continuous_edge_count=len(incoming), incoming_edge_observations=incoming,
        semantic_refusal_count=len(refusals), byte_pairs_verified=len(pairs),
        source_count=len(frozen), prior_704_sources_unchanged=True,
        verified_inherited_manifest_records=baseline["verified_manifest_records"],
        seconds=time.perf_counter()-started)
    return write(out/"independent_motion_receipt.json", receipt, "receipt_sha256")


if __name__ == "__main__":
    print(audit(Path(sys.argv[1]))["status"])
