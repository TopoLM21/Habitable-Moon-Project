"""Fixed original Genesis supports: exact radial fields, bounded angular work.

The declared work cap is a computational bound, not a relaxed numerical
tolerance.  Every resource refusal remains OPEN and all physical gates remain
false.  Source history is read and authenticated, never restored or advanced.
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import time
import numpy as np

from analysis.genesis_volume_clipped_validation.actual_control import canonical, seal, pin
from .source_control import load_authentic_source_fixture, support_source_inputs
from .angular_control import integrate_angular
from tectonics.genesis_volume_exact_lambda import reference_side_volumes_m3
from tectonics.genesis_volume_exact_lambda_bulk import (
    prepare_source, integrate_line, prepare_line_evaluator, source_to_dict, line_to_dict, parent_probe_work,
)

LD = np.longdouble
CONFIGURATION = dict(relative_tolerance=2e-6, max_cells=2000, max_depth=24,
                     angular_orders=(3, 5), max_line_evaluations=3400)


def binary_ratios(values):
    return [str(numerator)+"/"+str(denominator) for numerator, denominator in
            (LD(value).as_integer_ratio() for value in values)]


def volumes(line):
    if line.interval is None:
        return np.zeros(2, LD), np.zeros(2, LD)
    sides = reference_side_volumes_m3(line.interval)
    values = np.asarray([sides[LD(-.5)], sides[LD(.5)]], LD)
    # This covers represented arithmetic only, not geometry-root propagation.
    return values, 256*np.finfo(LD).eps*np.abs(values)


def parent_work_archive(source, probe):
    result = parent_probe_work(source, probe)
    value, budget = result["value"], result["absolute_roundoff_budget"]
    return {**result, "value":float(value),
            "absolute_roundoff_budget":float(budget+np.finfo(float).eps*abs(value))}


def run(output):
    output = Path(output).resolve()
    root = Path(__file__).resolve().parents[2]
    if not output.is_relative_to(Path("/workspace")) or output.is_relative_to(root):
        raise ValueError("Control output must be fresh and external")
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    # Persist the limits before examining any genuine numerical result.
    request = seal(dict(format="genesis-exact-lambda-predeclared-request-0.1",
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        configuration=CONFIGURATION, cohort_ids=[1, 41, 80], declared_radius_m=100.,
        original_contained_radius_preserved=True, source_history_reinitialized=False,
        physical_creation_admitted=False, physical_release_admitted=False,
        automated_goal_enabled=False), "request_sha256")
    (output/"request.json").write_bytes(canonical(request)+b"\n")
    fixture = load_authentic_source_fixture()
    (output/"source_fixture.json").write_bytes(canonical(fixture)+b"\n")
    packets = []
    for case_index, case in enumerate(fixture["cases"]):
        for support_name, support in case["supports"].items():
            raw = support_source_inputs(case, support_name)
            source = prepare_source(raw)
            old_local = raw["previous_local_bulk_blocks"]
            if (not np.array_equal(source.parent_stiffness, old_local["parent_stiffness"])
                    or not np.array_equal(source.parent_mass, old_local["parent_mass"])):
                raise ValueError("Whole original parent K/M changed in the exact-radial source")
            options = dict(added_basis=support["original_declared_added_basis"],
                           parent_basis=support["original_declared_parent_basis"],
                           probes=support["declared_probes"])
            N = np.ones(3, LD)/3
            witness = integrate_line(source, N, **options)
            prepared = prepare_line_evaluator(source, **options)
            if witness.interval is None:
                raise ValueError("Previously declared original midpoint line is unexpectedly empty")
            names = ("reference_volume_negative_m3", "reference_volume_positive_m3")+witness.names
            def evaluate(barycentric):
                line = prepared(barycentric)
                v, b = volumes(line)
                if line.names != witness.names:
                    raise ValueError("Exact radial component schema changed during angular control")
                return np.r_[v, line.values], np.r_[b, line.budgets]
            numerical_started = time.perf_counter()
            result = integrate_angular(raw, evaluate, names, **CONFIGURATION)
            angular = result.to_dict()
            record = seal(dict(format="genesis-source-bound-exact-lambda-support-0.1",
                cohort_id=case["original_cohort_id"], support=support_name,
                source_case_sha256=case["source_case_sha256"],
                source_support_sha256=support["support_sha256"],
                source_fixture_sha256=fixture["source_fixture_sha256"],
                source=source_to_dict(source), original_computational_support=support,
                line_witness_barycentric_exact_binary_ratios=binary_ratios(N),
                line_witness=line_to_dict(witness),
                unchanged_whole_original_cohort_parent_probe_work={probe["name"]:parent_work_archive(source, probe)
                    for probe in support["declared_probes"]},
                angular_control=angular, numerical_seconds=time.perf_counter()-numerical_started,
                angular_work_cap_counts_selected_witness=False,
                selected_witness_line_evaluations=1,
                total_radial_line_evaluations=angular["radial_line_evaluations"]+1,
                original_whole_parent_K_M_byte_equal=True,
                exact_radial_polynomial_integration_used=True,
                whole_parent_work_integrated_over_clipped_support=False,
                geometry_operand_accuracy_certified=False,
                physical_source_selected=False, physical_creation_admitted=False,
                physical_release_admitted=False, available_for_creation_funding=False,
                source_history_reinitialized=False, accepted_planet_memory_changed=False,
                accepted_steps=0, new_material_cohorts=0, funding_debit_j=0.,
                automated_goal_enabled=False, point_one_complete=False), "support_report_sha256")
            filename=f"case_{case_index:03d}_{support_name}.json"
            (output/filename).write_bytes(canonical(record)+b"\n")
            descriptor=dict(filename=filename, **pin(output/filename), cohort_id=case["original_cohort_id"],
                support=support_name, support_report_sha256=record["support_report_sha256"],
                status=angular["status"], failed_component_count=angular["failed_component_count"],
                component_count=angular["component_count"], evaluated_angular_cells=angular["evaluated_angular_cells"],
                radial_line_evaluations=angular["radial_line_evaluations"], worst_component=angular["worst_component"],
                total_radial_line_evaluations=angular["radial_line_evaluations"]+1,
                worst_error_ratio=angular["worst_error_ratio"], reason=angular["reason"],
                numerical_seconds=record["numerical_seconds"])
            packets.append(descriptor)
            print(json.dumps(descriptor, sort_keys=True), flush=True)
    for source_pin in fixture["source_pins"].values():
        if pin(source_pin["path"]) != {key:source_pin[key] for key in ("bytes", "raw_sha256")}:
            raise ValueError("Authoritative source changed during bounded numerical control")
    report = seal(dict(format="genesis-source-bound-exact-lambda-control-report-0.1",
        status="COMPLETED" if all(item["status"] == "FINITE_AGREEMENT" for item in packets)
            else "COMPLETED_WITH_NUMERICAL_REFUSALS",
        cases=packets, request_sha256=request["request_sha256"], configuration=CONFIGURATION,
        source_fixture_sha256=fixture["source_fixture_sha256"],
        source_fixture_pin=pin(output/"source_fixture.json"),
        source_binding_status="PASS", numerical_convergence_status="FINITE_AGREEMENT"
            if all(item["status"] == "FINITE_AGREEMENT" for item in packets) else "OPEN",
        source_time_myr=fixture["source_time_myr"], cohort_count=800,
        accepted_interval_count=4, source_history_reinitialized=False,
        accepted_planet_memory_changed=False, old_parent_K_M_D_unchanged=True,
        all_800_birth_memories_unchanged=True, source_selected_prospective_law=None,
        physical_source_selected=False, geometry_operand_accuracy_certified=False,
        physical_contraction_certified=False, physical_creation_admitted=False,
        physical_release_admitted=False, available_for_creation_funding=False,
        accepted_steps=0, new_material_cohorts=0, funding_debit_j=0.,
        automated_goal_enabled=False, point_one_complete=False,
        elapsed_seconds=time.perf_counter()-started), "report_sha256")
    (output/"report.json").write_bytes(canonical(report)+b"\n")
    print(json.dumps(dict(status=report["status"], report_sha256=report["report_sha256"],
          elapsed_seconds=report["elapsed_seconds"]), sort_keys=True), flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
