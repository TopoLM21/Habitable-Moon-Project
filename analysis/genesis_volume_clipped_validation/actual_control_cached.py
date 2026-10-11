"""Source-identical clipped control with once-computed point contractions.

The frozen first control and its interrupted capture remain intact.  This
new driver reuses its source authentication and archive formats while keeping
only native small matrix/probe results for neighboring order comparisons.
Every public point contraction and projected block is computed once per
order before caching; no full Gram matrix substitutes for point projection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.genesis_volume_clipped_validation.actual_control import (
    PARENT, PREVIOUS, RELEASE_REPORT, MEMORY_READBACK, RELEASE_READBACK,
    PARENT_RAW, PARENT_SELF, FORMATION, PREVIOUS_RAW, PREVIOUS_SELF,
    RELEASE_REPORT_SELF, MEMORY_READBACK_SELF, RELEASE_READBACK_SELF,
    MEMORY_READBACK_RAW, RELEASE_READBACK_RAW, FORMAT, SCOPE,
    ORDERS, BASELINE_ORDERS, RADIUS_M, RELATIVE_TOLERANCE, MAX_POINTS,
    canonical, pin, require, seal, check_seal, plain, exact_array,
    prepare_virtual_surface_basis, declared_probes, authenticate_inputs,
    bind_case,
)
from analysis.genesis_volume_clipped_validation.cached_comparison import (
    prepare_cached_comparison_operand, compare_cached_clipped_orders,
)
from tectonics.genesis_volume_clipped_domain import (
    clipped_domain_to_dict, prepare_cohort_clipped_domain,
)
from tectonics.genesis_volume_clipped_bulk import (
    clipped_bulk_contractions, clipped_bulk_operator_to_dict,
    clipped_bulk_projected_blocks, prepare_clipped_bulk_operator,
)


def projected_blocks_archive_from_live(live, added_basis):
    """Archive the exact native point-projected result used by comparison."""
    rounding = {name: (np.finfo(float).eps*np.abs(np.asarray(value, dtype=np.longdouble))).astype(float).tolist()
                for name, value in live.items() if isinstance(value, np.ndarray)}
    names = ("stiffness_cross", "stiffness_enriched", "mass_cross", "mass_enriched", "enrichment_force_n")
    return seal(dict(blocks={name: plain(live[name]) for name in names},
        roundoff_component_bounds={name: plain(live[name+"_budget"]) for name in names},
        added_basis=plain(added_basis), parent_basis=np.eye(18).tolist(), projection_before_Gram=True,
        physical_contraction_certified=False,
        literal_float64_archive_rounding_bounds=rounding,
        archive_rounding_is_arithmetic_diagnostic_only=True,
        live_finite_order_comparisons_include_only_live_operand_budgets=True,
        available_for_creation_funding=False), "archive_sha256")


def integrate_sequence(section, C, Q, e0, e1, center, radius, orders, probes, label, *, old_local):
    W, virtual_basis = prepare_virtual_surface_basis(section, center, radius)
    previous_operand, records, comparisons = None, [], []
    for order in orders:
        domain = prepare_cohort_clipped_domain(section, center, radius,
            radial_order=order, angular_order=order, ray_order=order,
            max_points=MAX_POINTS, require_both_sides=True)
        operator = prepare_clipped_bulk_operator(domain, C, Q, e0, e1)
        exact_array(operator.parent_stiffness, old_local["parent_stiffness"],
                    "Clipped integration changed the old selected full-cohort stiffness")
        exact_array(operator.parent_mass, old_local["parent_mass"],
                    "Clipped integration changed the old selected full-cohort mass")
        contractions = []
        for probe in probes:
            options = {k: probe[k] for k in ("mean_e0", "mean_e1", "mean_z_m") if k in probe}
            contractions.append(dict(name=probe["name"], contractions=clipped_bulk_contractions(operator,
                probe["parent_increment_m"], probe["enrichment_increment_m"], **options)))
        projected_live = clipped_bulk_projected_blocks(operator, W)
        operand = prepare_cached_comparison_operand(operator, projected_live, contractions,
            added_basis=W, probes=probes)
        records.append(dict(order=order, domain=clipped_domain_to_dict(domain),
            bulk_operator=clipped_bulk_operator_to_dict(operator, include_points=(order == orders[-1])),
            declared_probe_contractions=contractions,
            declared_projected_blocks=projected_blocks_archive_from_live(projected_live, W),
            original_full_cohort_parent_stiffness_exact=True,
            original_full_cohort_parent_mass_exact=True,
            literal_point_operator_tables_included=(order == orders[-1])))
        if previous_operand is not None:
            comparisons.append(compare_cached_clipped_orders((previous_operand, operand),
                relative_tolerance=RELATIVE_TOLERANCE, probes=probes, added_basis=W))
        # Only the bound native small operand remains for the next pair.
        # All point-derived evidence stays in the literal archive record.
        previous_operand = operand
        del operator, projected_live
        print(json.dumps(dict(status="ORDER_COMPLETE", label=label, order=order,
            cohort_id=section.cohort_id, seed_radius_m=radius, point_count=domain.point_count)), flush=True)
    pair_results = [comparison["pair_results"][0] for comparison in comparisons]
    comparison = seal(dict(format="genesis-clipped-bulk-control-sequential-pair-comparison-0.1",
        scope="neighbor_bound_native_results_all_literal_pair_packets_retained_no_convergence_certificate",
        comparison_packets=comparisons, pair_results=pair_results,
        finite_order_agreement=bool(pair_results) and all(p["finite_order_agreement"] for p in pair_results),
        qualification_available=(len(orders) >= 2), relative_tolerance=RELATIVE_TOLERANCE,
        operative_basis_virtual_sha256=virtual_basis["virtual_basis_sha256"],
        full_18_coordinate_components_and_declared_15_coordinate_point_contractions_retained=True,
        physical_contraction_certified=False, available_for_creation_funding=False), "comparison_sha256")
    decay = []
    for left, right in zip(comparison["pair_results"], comparison["pair_results"][1:]):
        ratios = {}
        for name, before in left["components"].items():
            previous_difference = before["maximum_absolute_difference"]
            current_difference = right["components"][name]["maximum_absolute_difference"]
            ratios[name] = dict(previous_maximum_absolute_difference=previous_difference,
                current_maximum_absolute_difference=current_difference,
                difference_ratio=None if current_difference == 0 else previous_difference/current_difference,
                current_difference_exact_zero=(current_difference == 0),
                rigorous_convergence_rate_claimed=False)
        decay.append(dict(previous_low_orders=left["low_orders"], previous_high_orders=left["high_orders"],
            current_low_orders=right["low_orders"], current_high_orders=right["high_orders"],
            block_difference_decay=ratios, scope="diagnostic_finite_order_difference_decay_not_error_certificate"))
    return dict(label=label, seed_radius_m=radius, fixed_orders=list(orders), orders=records,
        virtual_surface_basis=virtual_basis,
        all_order_diagnostic_comparison=comparison,
        required_final_order_pair=None if not pair_results else pair_results[-1],
        diagnostic_convergence_difference_decay=decay,
        required_final_orders=list(orders[-2:]),
        lower_order_agreement_is_diagnostic_only=True, radius_unchanged_for_all_orders=True,
        relative_tolerance_fixed_before_execution=RELATIVE_TOLERANCE,
        highest_order_literal_point_tables_retained=True,
        lower_order_geometry_arrays_and_all_operator_blocks_retained=True)


def run(output, *, smoke_first_case_order4=False):
    output = output.resolve()
    require(output.is_relative_to(Path("/workspace")) and not output.is_relative_to(ROOT),
            "Control output must be fresh and external inside /workspace")
    output.mkdir(exist_ok=False)
    require(pin(PARENT)["raw_sha256"] == PARENT_RAW, "Accepted parent raw pin differs")
    parent_bytes = PARENT.read_bytes()
    parent = check_seal(json.loads(parent_bytes), "checkpoint_sha256", PARENT_SELF)
    require(pin(PREVIOUS)["raw_sha256"] == PREVIOUS_RAW, "Authenticated birth-memory fixture raw pin differs")
    previous = check_seal(json.loads(PREVIOUS.read_bytes()), "fixture_sha256", PREVIOUS_SELF)
    release_report = check_seal(json.loads(RELEASE_REPORT.read_bytes()), "report_sha256", RELEASE_REPORT_SELF)
    require(pin(MEMORY_READBACK)["raw_sha256"] == MEMORY_READBACK_RAW
            and pin(RELEASE_READBACK)["raw_sha256"] == RELEASE_READBACK_RAW,
            "Previously validated final readback raw pin differs")
    memory_readback = check_seal(json.loads(MEMORY_READBACK.read_bytes()), "sha256", MEMORY_READBACK_SELF)
    release_readback = check_seal(json.loads(RELEASE_READBACK.read_bytes()), "sha256", RELEASE_READBACK_SELF)
    require(memory_readback["status"] == release_readback["status"] == release_report["status"] == "PASS",
            "Prior source reconstruction validation is not PASS")
    require(release_report["parent_raw_sha256"] == PARENT_RAW
            and release_report["previous_memory_fixture_pin"]["raw_sha256"] == PREVIOUS_RAW
            and "accepted_Genesis_has_no_new_source_activation" in release_report["physical_refusals"]
            and "no_source_selected_prospective_nucleation_law" in release_report["physical_refusals"],
            "Accepted-source physical refusal was replaced")
    runtime, geometry = authenticate_inputs(parent, previous)
    # The private two-step numerical trajectories in the old portable fixture
    # are already authenticated by its raw pin and sealed readback.  This new
    # integration needs only the literal birth/operator records, so discard
    # parsed copies of the unrelated private trajectory tables to bound RAM.
    prior_cases = [{k: case[k] for k in ("name", "seed_definition", "embedded_partition",
                   "accepted_interval_reaction", "source_memory_birth", "birth_memory")}
                   for case in previous["cases"]]
    del previous
    (output / "parent_before.json").write_bytes(parent_bytes)
    probes = declared_probes()
    if smoke_first_case_order4:
        prior = prior_cases[0]
        section, C, Q, e0, e1, inputs = bind_case(parent, prior)
        primary = integrate_sequence(section, C, Q, e0, e1,
            prior["seed_definition"]["seed_center_world_m"], RADIUS_M, (4,), probes,
            "PRELIMINARY_first_case_order4_API_smoke", old_local=inputs["previous_local_bulk_blocks"])
        (output / "parent_after.json").write_bytes(PARENT.read_bytes())
        require((output / "parent_after.json").read_bytes() == parent_bytes,
                "Source changed during preliminary API smoke")
        smoke = seal(dict(format=FORMAT + "-PRELIMINARY-smoke", scope="PRELIMINARY_API_SMOKE_NO_CONVERGENCE_QUALIFICATION",
            status="PRELIMINARY_COMPLETED_NO_CONVERGENCE_QUALIFICATION",
            name=prior["name"], source_inputs=inputs, declared_probes=list(probes),
            fixed_100m_clipped_support=primary, numerical_convergence_status="NOT_EVALUATED",
            parent_raw_sha256=PARENT_RAW, previous_memory_fixture_raw_sha256=PREVIOUS_RAW,
            public_planetary_restore_count=0, accepted_planet_memory_changed=False,
            physical_release_admitted=False, physical_creation_admitted=False,
            funding_admitted=False, automated_goal_enabled=False,
            declared_radius_m=RADIUS_M, new_Genesis_time_steps=0, new_material_cohorts=0), "smoke_sha256")
        (output / "smoke_fixture.json").write_bytes(canonical(smoke) + b"\n")
        print(json.dumps(dict(status=smoke["status"], fixture=pin(output / "smoke_fixture.json"),
            point_count=len(primary["orders"][0]["domain"]["barycentric"]),
            smoke_sha256=smoke["smoke_sha256"])), flush=True)
        return
    cases, summaries = [], []
    for index, prior in enumerate(prior_cases):
        section, C, Q, e0, e1, inputs = bind_case(parent, prior)
        center = prior["seed_definition"]["seed_center_world_m"]
        case_probes = probes + (dict(name="saved_accepted_interval_mean",
            parent_increment_m=probes[0]["parent_increment_m"],
            enrichment_increment_m=probes[0]["enrichment_increment_m"],
            mean_e0=inputs["accepted_interval_mean_e0"], mean_e1=inputs["accepted_interval_mean_e1"]),)
        primary = integrate_sequence(section, C, Q, e0, e1, center, RADIUS_M,
            ORDERS, case_probes, "fixed_100m_clipped_support",
            old_local=inputs["previous_local_bulk_blocks"])
        baseline = None
        if index == 0:
            baseline = integrate_sequence(section, C, Q, e0, e1, center,
                prior["seed_definition"]["seed_radius_m"], BASELINE_ORDERS, case_probes,
                "original_contained_computational_radius_baseline",
                old_local=inputs["previous_local_bulk_blocks"])
        case = dict(name=prior["name"], source_inputs=inputs, declared_probes=list(case_probes),
            fixed_100m_clipped_support=primary, original_radius_baseline=baseline)
        summary = dict(name=prior["name"], original_cohort_id=section.cohort_id,
            original_face_index=section.face_index, seed_radius_m=RADIUS_M,
            orders=list(ORDERS), point_counts=[len(v["domain"]["barycentric"]) for v in primary["orders"]],
            required_final_orders=list(ORDERS[-2:]),
            finite_order_agreement=primary["required_final_order_pair"]["finite_order_agreement"],
            original_radius_baseline_finite_order_agreement=None if baseline is None
                else baseline["required_final_order_pair"]["finite_order_agreement"])
        summaries.append(summary)
        case_packet = seal(dict(format=FORMAT + "-case", scope=SCOPE, **case), "case_sha256")
        case_path = output / f"case_{index:03d}_cohort_{section.cohort_id}.json"
        case_path.write_bytes(canonical(case_packet) + b"\n")
        cases.append(dict(name=prior["name"], original_cohort_id=section.cohort_id,
            case_file=case_path.name, case_pin=pin(case_path), case_sha256=case_packet["case_sha256"],
            numerical_summary=summary))
        print(json.dumps(dict(status="CASE_ARCHIVED", name=prior["name"], case_file=case_path.name,
            case_pin=cases[-1]["case_pin"], finite_order_agreement=summary["finite_order_agreement"])), flush=True)
        del primary, baseline, case, case_packet, inputs
    (output / "parent_after.json").write_bytes(PARENT.read_bytes())
    require((output / "parent_before.json").read_bytes() == (output / "parent_after.json").read_bytes()
            == parent_bytes and pin(PARENT)["raw_sha256"] == PARENT_RAW
            and pin(PREVIOUS)["raw_sha256"] == PREVIOUS_RAW,
            "Original Genesis or authenticated source archive changed")
    agreement = all(s["finite_order_agreement"] and s["original_radius_baseline_finite_order_agreement"] is not False
                    for s in summaries)
    refusals = [dict(case=s["name"], support="fixed_100m_clipped_support")
                for s in summaries if not s["finite_order_agreement"]]
    refusals += [dict(case=s["name"], support="original_radius_baseline")
                 for s in summaries if s["original_radius_baseline_finite_order_agreement"] is False]
    flags = dict(physical_creation_admitted=False, physical_release_admitted=False,
        physical_contraction_certified=False, physical_force_work_admission=False,
        funding_admitted=False, available_for_creation_funding=False,
        history_reinitialized=False, accepted_planet_memory_changed=False,
        actual_dynamic_reaction_replaced=False, physical_Gamma_selected=False,
        automated_goal_enabled=False, point_one_complete=False,
        new_Genesis_time_steps=0, new_material_cohorts=0,
        creation_work_debited_from_Genesis_j=0.0)
    source_pins = dict(parent=pin(PARENT), previous_memory_fixture=pin(PREVIOUS),
        previous_release_report=pin(RELEASE_REPORT), previous_memory_final_readback=pin(MEMORY_READBACK),
        previous_release_final_readback=pin(RELEASE_READBACK))
    fixture = seal(dict(format=FORMAT, scope=SCOPE, once_computed_native_comparison_cache=True, parent_raw_sha256=PARENT_RAW,
        parent_checkpoint_sha256=PARENT_SELF, formation_checkpoint_sha256=FORMATION,
        previous_memory_fixture_sha256=PREVIOUS_SELF, source_pins=source_pins,
        source_time_myr=geometry["time_myr"], cohort_count=800, accepted_interval_count=4,
        public_planetary_restore_count=0,
        source_binding_method="already_publicly_reconstructed_immutable_birth_archive_plus_direct_raw_parent_checks",
        physical_refusals=release_report["physical_refusals"], declared_fixed_radius_m=RADIUS_M,
        declared_orders=list(ORDERS), declared_relative_tolerance=RELATIVE_TOLERANCE,
        declared_max_points_per_order=MAX_POINTS, cases=cases,
        numerical_convergence_status="PASS" if agreement else "OPEN",
        finite_order_numerical_refusals=refusals, **flags), "fixture_sha256")
    (output / "control_fixture.json").write_bytes(canonical(fixture) + b"\n")
    report = seal(dict(format=FORMAT + "-report", scope=SCOPE, once_computed_native_comparison_cache=True,
        status="COMPLETED" if agreement else "COMPLETED_WITH_NUMERICAL_REFUSALS",
        structure_and_source_binding_status="PASS", numerical_convergence_status="PASS" if agreement else "OPEN",
        parent_raw_sha256=PARENT_RAW, parent_checkpoint_sha256=PARENT_SELF,
        formation_checkpoint_sha256=FORMATION, fixture_sha256=fixture["fixture_sha256"],
        source_pins=source_pins, public_planetary_restore_count=0,
        source_time_myr=geometry["time_myr"], cohort_count=800, accepted_interval_count=4,
        whole_parent_before_after_byte_identical=True, all_800_birth_memories_preserved=True,
        original_parent_K_M_D_preserved_without_reassembly=True,
        source_activation_none_preserved=True, source_selected_prospective_law=None,
        declared_fixed_radius_m=RADIUS_M, declared_orders=list(ORDERS),
        declared_relative_tolerance=RELATIVE_TOLERANCE, finite_order_numerical_refusals=refusals,
        physical_refusals=release_report["physical_refusals"], cases=summaries, **flags), "report_sha256")
    (output / "report.json").write_bytes(canonical(report) + b"\n")
    print(json.dumps(dict(status=report["status"], numerical_convergence_status=report["numerical_convergence_status"],
        report=pin(output / "report.json"), report_sha256=report["report_sha256"],
        fixture=pin(output / "control_fixture.json"), fixture_sha256=fixture["fixture_sha256"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-first-case-order4", action="store_true",
                        help="Retain a preliminary first-case one-order API smoke without any convergence qualification")
    args = parser.parse_args()
    run(args.output, smoke_first_case_order4=args.smoke_first_case_order4)
