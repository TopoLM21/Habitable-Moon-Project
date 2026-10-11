"""Stdlib-only independent audit of exported whole-line physical evidence."""
from __future__ import annotations

from decimal import localcontext
from fractions import Fraction as F
from math import factorial
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.geometric_native_active_line_validation.independent_line_oracle import (
    independent_frozen_potential_integral, joint_pullback_coefficients,
    original_measure_and_coefficients, polynomial_at, polynomial_bounds,
    turning_numerator_coefficients,
)
from analysis.geometric_native_active_line_validation.verify_baseline import (
    BASE, PRIOR, digest, encoded, read, sealed, sources, verify_baseline, write,
)


BUDGETS = ("source_work_bounds_j", "friction_dissipation_bounds_j",
    "bulk_energy_increment_bounds_j", "contact_energy_increment_bounds_j",
    "total_energy_increment_bounds_j", "compliance_balance_residual_bounds_j")


def rational(value):
    if type(value) is not list or len(value) != 2 or any(type(x) is not int for x in value):
        raise ValueError("Independent evidence needs literal exact rational pairs")
    result = F(*value)
    if value != [result.numerator, result.denominator]:
        raise ValueError("Independent evidence rational pair is not canonical")
    return result


def interval(value):
    if type(value) is not dict or set(value) != {"lo", "hi"}:
        raise ValueError("Independent evidence needs explicit exact interval bounds")
    result = rational(value["lo"]), rational(value["hi"])
    if result[0] > result[1]:
        raise ValueError("Independent evidence interval is reversed")
    return result


def add(first, second):
    return first[0]+second[0], first[1]+second[1]


def multiply(bounds, coefficient):
    values = bounds[0]*coefficient, bounds[1]*coefficient
    return min(values), max(values)


def contains(outer, inner):
    if not outer[0] <= inner[0] <= inner[1] <= outer[1]:
        raise AssertionError("Independent scalar interval is outside the published enclosure")


def audit(out):
    started = time.perf_counter()
    out = Path(out)
    baseline = verify_baseline()
    frozen = read(BASE / "frozen_release_sources.json")
    if sources() != frozen:
        raise ValueError("Independent audit requires the complete frozen source map")
    report = sealed(out / "report.json", "report_sha256")
    if report["status"] != "PASS" or report["accepted_native_force_intervals"] != 0:
        raise ValueError("The line-only physical control status or scope differs")
    source = read(out / "source_history_before.json")
    if source != read(PRIOR / "control_release_02/source_history_before.json"):
        raise ValueError("Independent line source is not the authoritative actual force history")
    certificate = sealed(out / "line_certificate.json", "certificate_sha256")
    work = sealed(out / "weighted_pair_work.json", "integral_sha256")
    support = sealed(out / "full_bank_clamp_support.json", "support_sha256")
    if certificate["source_history"] != source or support["source_history"] != source:
        raise ValueError("The line/support exported source changed")
    independent, norm = joint_pullback_coefficients(source, certificate["order"])
    coefficient_count = 0
    for column in certificate["columns"]:
        axis = column["initial_coordinate_index"]
        if rational(column["rate_bound"]) != norm:
            raise AssertionError("The saved joint matrix norm differs from the actual axes")
        for degree, row in enumerate(column["coefficients"]):
            for coordinate, scalar in enumerate(row):
                if rational(scalar) != independent[degree][coordinate][axis]:
                    raise AssertionError("The saved joint matrix series differs from independent convolution")
                coefficient_count += 1
    turn_count = 0
    for axis in certificate["axes"]:
        polynomial, error = turning_numerator_coefficients(source, axis["axis_coordinate_index"])
        derivative = polynomial_bounds(polynomial, 1)
        saved_derivative = interval(axis["turning_numerator_derivative_bounds"])
        increasing = saved_derivative[0] > 0
        if not (derivative[0]-40*error > 0 if increasing else derivative[1]+40*error < 0):
            raise AssertionError("Independent whole-clock ratio turning is not strict")
        low, high = interval(axis["unique_turning_clock_bounds"])
        first, last = polynomial_at(polynomial, low), polynomial_at(polynomial, high)
        if not (first+error < 0 < last-error if increasing else last+error < 0 < first-error):
            raise AssertionError("Independent ratio turn lies outside its published root bracket")
        if interval(axis["positive_axis_other_support_bounds"])[0] <= 0:
            raise AssertionError("The other original donor support was not proved on all seams")
        if min(interval(axis["positive_axis_owner_depth_over_clock_bounds"])[0],
                interval(axis["positive_axis_fresh_owner_derivative_bounds"])[0]) <= 0:
            raise AssertionError("The positive seam lacks its complete fresh-owner proof")
        turn_count += 1
    if (certificate["whole_line_selection_proved"] is not True
            or certificate["geometric_proof_from_point_samples"] is not False
            or certificate["tiny_cross_axis_strips_discarded"] is not False
            or certificate["inconclusive_owner_entry_proof_retained_as_if_negative"] is not False
            or certificate["negative_axis_seams"] != "proved_negative_owner_entry_retain_current_frozen_binding"):
        raise AssertionError("The saved whole-angle proof/retention semantics differ")
    quarter_by_trace = {row["receiver_trace_id"]: row for row in certificate["quarters"]}
    if len(quarter_by_trace) != 4:
        raise AssertionError("A receiving quarter was lost")
    source_banks = source["material_state"]["trace_bindings"]
    expected_pairs = {(donor["donor_fragment_id"], receiver["receiver_fragment_id"])
        for donor in source_banks for receiver in source_banks}
    actual_pairs = {(row["donor_fragment_id"], row["receiver_fragment_id"])
        for row in work["receiving_pair_budgets"]}
    if len(work["receiving_pair_budgets"]) != 16 or actual_pairs != expected_pairs:
        raise AssertionError("The independent full original pair product differs")
    original_area = F(0)
    positive_strip_area = F(0)
    for row in work["quarters"]:
        quarter = quarter_by_trace[row["receiver_trace_id"]]
        start, core, end = (interval(quarter[name]) for name in (
            "start_strip_computational_angle_bounds", "core_computational_angle_bounds", "end_strip_computational_angle_bounds"))
        extent = rational(quarter["computational_extent_radians"])
        if not start[0] == 0 < start[1] == core[0] < core[1] == end[0] < end[1] == extent:
            raise AssertionError("The whole exact receiving line is not fully partitioned")
        coefficient, _ = original_measure_and_coefficients(source, row["receiver_trace_id"])
        area = coefficient*extent
        strip_area = coefficient*(start[1]-start[0]+end[1]-end[0])
        if (rational(row["whole_original_interface_area_m2"]) != area
                or rational(row["strip_original_interface_area_m2"]) != strip_area
                or strip_area <= 0):
            raise AssertionError("The original weighted contact measure or tiny strips changed")
        original_area += area
        positive_strip_area += strip_area
        branch = row["branch_proof"]
        if (interval(branch["whole_line_gap_bounds_m"])[1] >= 0
                or interval(branch["whole_line_gap_rate_bounds_m_per_fraction"])[0] <= 0
                or interval(branch["whole_line_jump_rate_bounds_m_per_fraction"])[1] >= 0):
            raise AssertionError("The whole-line carried constitutive branch is not strict")
    if rational(work["original_interface_area_m2"]) != original_area:
        raise AssertionError("Original contact measure does not close")
    if rational(work["explicitly_enclosed_strip_interface_area_m2"]) != positive_strip_area:
        raise AssertionError("The positive-width strips were not all retained")
    total_bounds = {name: (F(0), F(0)) for name in BUDGETS}
    scalar_identity_count = 0
    independent_core_count = 0
    for row in work["receiving_pair_budgets"]:
        _, coefficients = original_measure_and_coefficients(source, row["receiver_trace_id"])
        if tuple(map(rational, row["literal_normal_coefficients_pa_m"])) != coefficients:
            raise AssertionError("The pair replaced its literal binary64 compliance")
        kn, bulk, contact = coefficients
        if kn-bulk-contact == 0:
            raise AssertionError("The actual literal compliance residual was erased")
        potential = interval(row["potential_increment_integral_bounds_m4"])
        for name, coefficient in zip(BUDGETS, (kn, F(0), bulk, contact, bulk+contact, kn-bulk-contact)):
            actual = interval(row[name])
            contains(actual, multiply(potential, coefficient))
            total_bounds[name] = add(total_bounds[name], actual)
            scalar_identity_count += 1
        quarter = quarter_by_trace[row["receiver_trace_id"]]
        if row["donor_trace_id"] == quarter["original_donor_trace_id"]:
            low, high = interval(quarter["core_computational_angle_bounds"])
            with localcontext() as context:
                context.prec = 240
                value = independent_frozen_potential_integral(source, row["receiver_trace_id"],
                    low, high, rational(quarter["computational_extent_radians"]))
            contains(interval(row["core_potential_increment_integral_bounds_m4"]), (F(value), F(value)))
            independent_core_count += 1
    for name, total in total_bounds.items():
        contains(interval(work[name]), total)
    if interval(work["friction_dissipation_bounds_j"]) != (F(0), F(0)):
        raise AssertionError("The original inviscid source gained friction dissipation")
    if interval(work["source_work_bounds_j"])[1] >= 0:
        raise AssertionError("The actual full-rotation preload unloading lost its signed work")
    if interval(work["compliance_balance_residual_bounds_j"])[0] <= 0:
        raise AssertionError("The independently derived signed unloading residual is not retained")
    if len(support["parent_supports"]) != 8 or len(support["live_clamps"]) != 8:
        raise AssertionError("Complete original bank/clamp support was lost")
    refusals = read(out / "semantic_refusals.json")
    if len(refusals) != report["semantic_refusal_count"] or not all(row["rejected"] for row in refusals):
        raise AssertionError("Semantic refusal evidence differs")
    for row in report["byte_pairs"]:
        first, second = out / row["first"], out / row["second"]
        if (first.read_bytes() != second.read_bytes() or first.stat().st_size != row["bytes"]
                or digest(first) != row["sha256"]):
            raise AssertionError("A complete source/restart byte pair differs")
    if read(out / "sources_before.json") != frozen or read(out / "sources_after.json") != frozen:
        raise AssertionError("The physical line control ran on different Python sources")
    receipt = dict(format="native-active-line-independent-audit-1", status="PASS",
        seconds=time.perf_counter()-started, imports_production_modules=False,
        verified_prior_manifest_records=baseline["verified_manifest_records"],
        immutable_prior_python_sources=704, current_frozen_python_sources=len(frozen),
        exact_joint_matrix_coefficient_observations=coefficient_count,
        independent_whole_clock_unique_ratio_turns=turn_count,
        independent_exact_measure_quarters=4, exact_receiving_pair_count=16,
        independent_common_potential_scalar_identity_checks=scalar_identity_count,
        independent_240_digit_real_harmonic_core_integrals=independent_core_count,
        strict_negative_seam_retention_and_positive_fresh_gate_checked=True,
        positive_width_strips_retained=True, signed_literal_compliance_residual_retained=True,
        verified_original_parent_supports=8, verified_live_clamp_records=8,
        verified_semantic_refusals=len(refusals), verified_byte_pairs=len(report["byte_pairs"]),
        scope="Independent line selection geometry, original contact measure, weighted core work/common-potential coefficient balances and immutable source/restart evidence; support record coverage only, not a fresh force-admission audit",
        accepted_native_force_intervals=0, entire_point_1_completed=False)
    saved = write(out / "independent_active_line_receipt.json", receipt, "receipt_sha256")
    print(encoded({key: saved[key] for key in ("status", "seconds", "exact_joint_matrix_coefficient_observations",
        "independent_common_potential_scalar_identity_checks", "accepted_native_force_intervals")}).decode(), flush=True)
    return saved


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    audit(parser.parse_args().control)
