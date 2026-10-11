"""Validate domain contact transport without adopting a native force interval.

The source release, its literal quadrature atoms and its original unaccepted
candidate remain immutable. Continuum integration is a separately declared
measure; it cannot silently replace the sixteen original quadrature measures.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
PREVIOUS = ROOT / "analysis/geometric_native_first_entry_handover_validation"
sys.path.insert(0, str(ROOT))
from analysis.geometric_native_first_entry_handover_validation.run_first_entry_handover_control import (
    ENTRY_LABELS, POINT_LABELS, PHYSICAL, SOURCE, LAW_PROVENANCE, digest, encoded,
    get_physical_case, original_quadrature_ledger, rational, read,
    rehash_declared_seals, sealed, sources, value_sha, write,
    verify_previous_sources as verify_older_sources,
)


def guard_sources():
    current, starting = sources(), read(BASE / "starting_sources.json")
    prior_maps = [PREVIOUS / directory / ("sources_" + stage + ".json")
        for directory in ("release_gates_01", "control_release_01")
        for stage in ("before", "after")]
    if len(starting) != 684 or any(read(path) != starting for path in prior_maps):
        raise ValueError("Domain transport must preserve the authenticated 684-source release")
    changed = sorted(name for name, expected in starting.items() if current.get(name) != expected)
    if changed:
        raise ValueError("A frozen original Python source changed: " + repr(changed))
    return current, starting


def verify_previous_sources():
    older = verify_older_sources()
    directory = PREVIOUS / "release_validation_01"
    summary_path, manifest_path = directory / "validation_summary.json", directory / "artifact_manifest.json"
    summary, manifest = sealed(summary_path, "summary_sha256"), sealed(manifest_path, "manifest_sha256")
    if (summary["status"] != "PASS" or summary["source_count"] != 684
            or summary["fresh_scoped_gate"]["tests_passed"] != 265
            or manifest["count"] != 48 or len(manifest["artifacts"]) != 48
            or summary["artifact_manifest"]["sha256"] != digest(manifest_path)):
        raise ValueError("The prior four-root handover release differs")
    for name, record in manifest["artifacts"].items():
        path = ROOT / name
        if not path.is_file() or path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            raise ValueError("Prior four-root handover artifact differs: " + name)
    for name in ("fresh_scoped_gate", "actual_control", "independent_audit"):
        reference = summary[name]
        if digest(ROOT / reference["path"]) != reference["sha256"]:
            raise ValueError("Prior four-root handover evidence differs: " + name)
    ancestry = [dict(path=str(summary_path.relative_to(ROOT)), sha256=digest(summary_path),
        verified_manifest_records=48, tests_passed=265, tests_reexecuted=False), *older["ancestry"]]
    paths = set(older["source_artifacts"])
    paths.update(str(path.relative_to(ROOT)) for path in (
        summary_path, manifest_path, PREVIOUS / "release_gates_01/receipt.json",
        PREVIOUS / "control_release_01/report.json",
        PREVIOUS / "control_release_01/independent_first_entry_handover_receipt.json",
        PREVIOUS / "control_release_01/first_entry_handovers.json",
        PREVIOUS / "control_release_01/original_quadrature_ledger.json",
        PREVIOUS / "control_release_01/original_first_entry_bindings.json",
        BASE / "starting_sources.json"))
    return dict(format="native-domain-contact-handover-source-validation-1", status="PASS",
        original_source_count=684, verified_prior_manifest_records=233,
        prior_tests_reexecuted=False, ancestry=ancestry,
        source_artifacts={name:dict(sha256=digest(ROOT / name), bytes=(ROOT / name).stat().st_size)
            for name in sorted(paths)})


def original_continuum_work_rows(history, source_raw):
    """Keep the time one-form distinct from every literal endpoint atom."""
    from tectonics.geometric_native_contact_field_work import (
        query_native_contact_field_work, integrate_native_contact_field_work,
        native_contact_field_work_to_dict, native_contact_field_work_from_dict,
        native_contact_field_work_integral_to_dict, native_contact_field_work_integral_from_dict,
        integrate_native_contact_field_work_over_original_interface,
        native_contact_field_work_interface_integral_to_dict,
        native_contact_field_work_interface_integral_from_dict,
    )
    from tectonics.geometric_native_pose import FractionInterval as I
    rows, replayed = [], []
    for trace in history.previous_memory.traces:
        angle, time_box = I(Fraction(0), Fraction.from_float(trace.extent_radians)), I(Fraction(0), Fraction(1))
        context = dict(model="complete_original_interface_angle_and_clock_domain", trace_id=trace.trace_id,
            original_nodal_work_ledger_replaced=False, first_root_midpoint_used=False)
        arguments = dict(history=history, trace_id=trace.trace_id, computational_angle_bounds=angle,
            expected_source_history_dto=source_raw, source_expression_context=context, precision=256)
        query = query_native_contact_field_work(time_fraction_bounds=time_box, **arguments)
        integral = integrate_native_contact_field_work(initial_time_fraction=Fraction(0),
            final_time_fraction=Fraction(1), **arguments)
        raw = native_contact_field_work_to_dict(query, history=history, expected_source_history_dto=source_raw)
        integral_raw = native_contact_field_work_integral_to_dict(integral, history=history,
            expected_source_history_dto=source_raw)
        query_replay = native_contact_field_work_from_dict(deepcopy(raw), time_fraction_bounds=time_box, **arguments)
        integral_replay = native_contact_field_work_integral_from_dict(deepcopy(integral_raw),
            initial_time_fraction=Fraction(0), final_time_fraction=Fraction(1), **arguments)
        interface_arguments = dict(history=history, trace_id=trace.trace_id,
            initial_computational_angle=angle.lo, final_computational_angle=angle.hi,
            initial_time_fraction=Fraction(0), final_time_fraction=Fraction(1),
            expected_source_history_dto=source_raw, source_expression_context=context, precision=256)
        interface_integral = integrate_native_contact_field_work_over_original_interface(**interface_arguments)
        interface_raw = native_contact_field_work_interface_integral_to_dict(interface_integral,
            history=history, expected_source_history_dto=source_raw)
        interface_replay = native_contact_field_work_interface_integral_from_dict(deepcopy(interface_raw),
            **interface_arguments)
        rows.append(dict(trace_id=trace.trace_id, whole_clock_rate_query=raw,
            exact_clock_density_integral=integral_raw, whole_original_interface_integral=interface_raw))
        replayed.append(dict(trace_id=trace.trace_id,
            whole_clock_rate_query=native_contact_field_work_to_dict(query_replay,
                history=history, expected_source_history_dto=source_raw),
            exact_clock_density_integral=native_contact_field_work_integral_to_dict(integral_replay,
                history=history, expected_source_history_dto=source_raw),
            whole_original_interface_integral=native_contact_field_work_interface_integral_to_dict(interface_replay,
                history=history, expected_source_history_dto=source_raw)))
    if len(rows) != 4 or len({row["trace_id"] for row in rows}) != 4 or encoded(rows) != encoded(replayed):
        raise AssertionError("The four original continuum work families lost strict source replay")
    return rows, replayed


def check_rehashed_mutation(raw, *, mutate, loader, label, changed_path):
    """A saved selfseal never substitutes for independent complete source inputs."""
    forged = deepcopy(raw)
    mutate(forged)
    # New nested components use additional integrity seals; rehash each
    # derived payload from the bottom up before asking its independent loader.
    def new_seals(value):
        if type(value) is list:
            for item in value:
                new_seals(item)
        elif type(value) is dict:
            for item in value.values():
                new_seals(item)
            if value.get("format") == "native-source-beam-material-chart-1":
                value["source_sha256"] = value_sha(value["source"])
            for key in ("binding_sha256", "chart_sha256", "measure_sha256", "handover_sha256",
                    "field_sha256", "query_sha256", "state_sha256", "family_sha256",
                    "integral_sha256", "work_sha256", "domain_sha256"):
                if key in value:
                    value[key] = value_sha({name:item for name, item in value.items() if name != key})
    new_seals(forged)
    try:
        loader(forged)
    except (ValueError, TypeError) as error:
        return dict(label=label, changed_path=changed_path, rejected=True,
            valid_rehashed_declared_checksum=True, exception_type=type(error).__name__,
            message=str(error), mutated_dto_sha256=value_sha(forged))
    raise AssertionError("A validly rehashed domain/work mutation was accepted: " + label)


def changed_at(path, value):
    def mutate(raw):
        current = raw
        for key in path[:-1]:
            current = current[key]
        current[path[-1]] = value
    return mutate


def shifted_bounds(path, amount=Fraction(1, 64)):
    def mutate(raw):
        current = raw
        for key in path:
            current = current[key]
        current["lo"] = rational(Fraction(*current["lo"]) + amount)
        current["hi"] = rational(Fraction(*current["hi"]) + amount)
    return mutate


def domain_mutation_refusals(registry_raw, integral_raw, work_rows, *,
        registry_loader, integral_loader, history, source_raw):
    from tectonics.geometric_native_contact_field_work import native_contact_field_work_interface_integral_from_dict
    result = []
    def check(raw, loader, label, path, mutation):
        result.append(check_rehashed_mutation(raw, mutate=mutation, loader=loader,
            label=label, changed_path=path))
    check(registry_raw, registry_loader, "unknown_domain_schema_field", "unknown_field",
        lambda raw: raw.update(unknown_field=1))
    if "precision" in registry_raw:
        check(registry_raw, registry_loader, "domain_precision_literal_type_changed", "precision",
            lambda raw: raw.update(precision=float(raw["precision"])))
    registry_cases = (
        ("receiving_history_family_duplicated", "families",
            lambda raw: raw["families"].append(deepcopy(raw["families"][0]))),
        ("receiving_family_original_history_count_changed", "families[0].original_history_count",
            changed_at(("families", 0, "original_history_count"), 2)),
        ("receiving_family_consumed_lineage_duplicated", "families[0].consumed_binding_ids",
            lambda raw: raw["families"][0]["consumed_binding_ids"].append(raw["families"][0]["original_binding_id"])),
        ("causal_branch_source_family_changed", "branches[0].source_family_id",
            changed_at(("branches", 0, "source_family_id"), registry_raw["families"][1]["family_id"])),
        ("causal_branch_finite_history_fabricated", "branches[0].new_finite_history_created",
            changed_at(("branches", 0, "new_finite_history_created"), True)),
        ("whole_donor_beam_width_changed", "source_beam_supports[0].width_m",
            changed_at(("source_beam_supports", 0, "width_m"), [10001, 1])),
        ("whole_donor_beam_shape_changed", "source_beam_supports[0].shape_bounds",
            shifted_bounds(("source_beam_supports", 0, "shape_bounds"))),
        ("whole_donor_native_clamp_support_fabricated", "source_beam_supports[0].native_force_and_clamp_support_certified",
            changed_at(("source_beam_supports", 0, "native_force_and_clamp_support_certified"), True)),
        ("domain_source_work_budget_changed", "source_work_budgets[0].source_work_bounds_j",
            shifted_bounds(("source_work_budgets", 0, "source_work_bounds_j"))),
        ("original_atom_lineage_duplicated", "original_atom_lineages",
            lambda raw: raw["original_atom_lineages"].append(deepcopy(raw["original_atom_lineages"][0]))),
        ("global_unpaid_transfer_work_fabricated", "exact_global_transfer_balances.external_work_j",
            changed_at(("exact_global_transfer_balances", "external_work_j"), [1, 1])),
        ("material_contact_work_density_fabricated", "interface_energy_or_work_density_on_archived_bulk_declared",
            changed_at(("interface_energy_or_work_density_on_archived_bulk_declared",), True)),
        ("literal_atom_and_continuum_work_summed", "contactline_work_and_literal_atom_work_are_summed",
            changed_at(("contactline_work_and_literal_atom_work_are_summed",), True)),
    )
    for label, path, mutate in registry_cases:
        check(registry_raw, registry_loader, label, path, mutate)
    integral_cases = (
        ("unknown_integral_schema_field", "unknown_field", lambda raw: raw.update(unknown_field=1)),
        ("integral_precision_literal_type_changed", "precision", lambda raw: raw.update(precision=float(raw["precision"]))),
        ("carrier_mask_omitted", "carrier_masks", lambda raw: raw["carrier_masks"].pop()),
        ("carrier_mask_duplicated", "carrier_masks", lambda raw: raw["carrier_masks"].append(deepcopy(raw["carrier_masks"][0]))),
        ("receiving_parent_changed", "carrier_masks[0].original_receiver_fragment_id",
            changed_at(("carrier_masks", 0, "original_receiver_fragment_id"),
                "octant:2" if integral_raw["carrier_masks"][0]["original_receiver_fragment_id"] != "octant:2" else "octant:3")),
        ("receiving_support_changed", "carrier_masks[0].receiver_supports[0][0]",
            changed_at(("carrier_masks", 0, "receiver_supports", 0, 0), [17, 1])),
        ("archive_parent_omitted", "parent_measures", lambda raw: raw["parent_measures"].pop()),
        ("archive_area_changed", "parent_measures[0].archive_surface_area_bounds_m2",
            shifted_bounds(("parent_measures", 0, "archive_surface_area_bounds_m2"))),
        ("continuum_interface_density_changed", "original_interfaces[0].area_density_m2_per_radian",
            changed_at(("original_interfaces", 0, "area_density_m2_per_radian"), [1, 1])),
        ("literal_quadrature_area_changed", "original_interfaces[0].quadrature_atoms[0].literal_area_m2",
            changed_at(("original_interfaces", 0, "quadrature_atoms", 0, "literal_area_m2"), [1, 1])),
        ("initial_event_map_injectivity_claim_changed", "coarea_proof.first_event_map_injective",
            changed_at(("coarea_proof", "first_event_map_injective"), False)),
        ("receiver_boundary_incomplete", "coarea_proof.original_receiver_boundary_complete",
            changed_at(("coarea_proof", "original_receiver_boundary_complete"), False)),
        ("owner_regular_axis_projection_changed", "coarea_proof.owner_boundary_regular_axis_projection_bounds",
            shifted_bounds(("coarea_proof", "owner_boundary_regular_axis_projection_bounds"))),
        ("rectangle_multiplicity_claimed_as_archive_area", "coarea_proof.rectangle_multiplicity_used_as_archive_area",
            changed_at(("coarea_proof", "rectangle_multiplicity_used_as_archive_area"), True)),
        ("contact_density_inferred_from_material_area", "contact_interface_to_swept_material_density_declared",
            changed_at(("contact_interface_to_swept_material_density_declared",), True)),
        ("numerical_receiving_pair_integration_fabricated", "receiver_pair_integrals_numerically_enclosed",
            changed_at(("receiver_pair_integrals_numerically_enclosed",), True)),
        ("nodal_work_reassigned", "source_nodal_work_reassigned",
            changed_at(("source_nodal_work_reassigned",), True)),
    )
    for label, path, mutate in integral_cases:
        check(integral_raw, integral_loader, label, path, mutate)
    representative = work_rows[0]["whole_original_interface_integral"]
    source = representative["source_time_integral"]["source_field_query"]
    angle = source["computational_angle_bounds"]
    kwargs = dict(history=history, trace_id=work_rows[0]["trace_id"],
        initial_computational_angle=Fraction(*angle["lo"]), final_computational_angle=Fraction(*angle["hi"]),
        initial_time_fraction=Fraction(0), final_time_fraction=Fraction(1),
        expected_source_history_dto=source_raw, source_expression_context=source["source_expression_context"],
        precision=source["precision"])
    work_loader = lambda raw: native_contact_field_work_interface_integral_from_dict(raw, **kwargs)
    for label, path, mutation in (
        ("unknown_work_schema_field", "unknown_field", lambda raw: raw.update(unknown_field=1)),
        ("whole_interface_source_work_changed", "source_work_bounds_j", shifted_bounds(("source_work_bounds_j",))),
        ("whole_interface_dissipation_changed", "friction_dissipation_bounds_j", shifted_bounds(("friction_dissipation_bounds_j",))),
        ("whole_interface_initial_energy_changed", "initial_total_energy_bounds_j", shifted_bounds(("initial_total_energy_bounds_j",))),
        ("whole_interface_final_energy_changed", "final_total_energy_bounds_j", shifted_bounds(("final_total_energy_bounds_j",))),
        ("literal_compliance_residual_suppressed", "signed_compliance_residual_suppressed",
            changed_at(("signed_compliance_residual_suppressed",), True)),
        ("continuum_measure_replaced_by_quadrature_measure", "continuum_measure_is_original_nodal_gauss_measure",
            changed_at(("continuum_measure_is_original_nodal_gauss_measure",), True)),
        ("continuum_work_replaced_by_backward_euler_work", "continuum_work_is_original_endpoint_backward_euler_work",
            changed_at(("continuum_work_is_original_endpoint_backward_euler_work",), True)),
        ("original_interface_area_density_changed", "original_interface_area_density_m2_per_computational_radian",
            changed_at(("original_interface_area_density_m2_per_computational_radian",), [1, 1])),
        ("material_swept_area_substituted_into_work", "swept_archive_measure_integrated",
            changed_at(("swept_archive_measure_integrated",), True)),
        ("whole_interface_invented_native_force_acceptance", "native_candidate_mechanics_adopted",
            changed_at(("native_candidate_mechanics_adopted",), True)),
        ("source_fraction_interval_changed", "source_time_integral.final_time_fraction",
            changed_at(("source_time_integral", "final_time_fraction"), [1, 2])),
    ):
        check(representative, work_loader, label, path, mutation)
    if len(result) < 25:
        raise AssertionError("Whole-domain scientific mutation coverage is incomplete")
    return result


def run_control(out):
    from tectonics.geometric_native_contact_reference_transport import ContactHistoryTransportLaw, TRANSPORT_LAW_MODEL
    from tectonics.geometric_native_first_entry_material import (
        native_first_entry_material_point_chronology_from_dict, native_first_entry_material_point_chronology_to_dict,
    )
    from tectonics.geometric_native_first_entry_physical_binding import (
        certify_native_first_entry_physical_binding, native_first_entry_physical_binding_to_dict,
    )
    from tectonics.geometric_native_first_entry_interface_integral import (
        native_first_entry_interface_integral_to_dict, native_first_entry_interface_integral_from_dict,
        evaluate_native_first_entry_interface_carrier_mask,
    )
    from tectonics.geometric_native_first_entry_contact_domain_handover import (
        certify_native_first_entry_contact_domain_handover,
        native_first_entry_contact_domain_handover_to_dict,
        native_first_entry_contact_domain_handover_from_dict,
        evaluate_native_first_entry_contact_domain_point,
        native_first_entry_contact_domain_point_to_dict,
        native_first_entry_contact_domain_point_from_dict,
    )
    from tectonics.geometric_native_first_entry_contact_handover import native_first_entry_contact_handover_to_dict
    from tectonics.geometric_native_physical_history import _candidate_payload, native_physical_history_binding_to_dict
    from tectonics.geometric_native_underthrust_motion_io import save_native_underthrust_motion_checkpoint

    started = time.perf_counter()
    before_sources, baseline = guard_sources()
    authority = verify_previous_sources()
    out.mkdir(parents=True, exist_ok=False)
    write(out / "sources_before.json", before_sources)
    write(out / "source_validation_receipt.json", authority, seal="receipt_sha256")
    case = get_physical_case()
    history = case["history"]
    source_raw = native_physical_history_binding_to_dict(history, mesh=case["mesh"])
    write(out / "source_history_before.json", source_raw)
    if (out / "source_history_before.json").read_bytes() != (PHYSICAL / "control_release_01/original_history.json").read_bytes():
        raise AssertionError("The complete original physical source changed")
    checkpoint = save_native_underthrust_motion_checkpoint(out / "source_checkpoint_before.json",
        case["mesh"], case["origin"], case["before"], provenance=case["provenance"])
    if checkpoint.read_bytes() != case["original_files"][SOURCE / "original_force/initial.json"]:
        raise AssertionError("The original typed force checkpoint bytes changed")
    write(out / "force_candidate_before.json", case["candidate_raw"])
    law = ContactHistoryTransportLaw(model=TRANSPORT_LAW_MODEL, provenance=LAW_PROVENANCE)
    write(out / "declared_handover_law.json", asdict(law))
    print("Committing the complete implicit contact-domain registry with four once-only source families", flush=True)
    registry = certify_native_first_entry_contact_domain_handover(history, law=law,
        expected_source_history_dto=source_raw, precision=256, order=32)
    if (not registry.whole_domain_field_transfer_committed
            or registry.whole_domain_handover_work_j != 0
            or registry.whole_domain_handover_dissipation_j != 0
            or registry.whole_domain_handover_energy_change_j != 0
            or registry.physical_pair_admission_authorized or registry.candidate_mechanics_adopted):
        raise AssertionError("The committed complete field registry altered work, energy or force admission")
    registry_raw = native_first_entry_contact_domain_handover_to_dict(registry)
    registry_loader = lambda raw: native_first_entry_contact_domain_handover_from_dict(raw,
        expected_history=history, expected_source_history_dto=source_raw, expected_law=law)
    registry_replay = registry_loader(deepcopy(registry_raw))
    write(out / "domain_contact_handover.json", registry_raw)
    write(out / "domain_contact_handover_replayed.json",
        native_first_entry_contact_domain_handover_to_dict(registry_replay))
    integral = registry.interface_integral
    integral_raw = native_first_entry_interface_integral_to_dict(integral, expected_source_history_dto=source_raw)
    integral_loader = lambda raw: native_first_entry_interface_integral_from_dict(raw,
        expected_history=history, expected_source_history_dto=source_raw)
    integral_replay = integral_loader(deepcopy(integral_raw))
    write(out / "first_entry_interface_integral.json", integral_raw)
    write(out / "first_entry_interface_integral_replayed.json",
        native_first_entry_interface_integral_to_dict(integral_replay, expected_source_history_dto=source_raw))
    print("Integrating the original four continuum-interface work and energy budgets analytically", flush=True)
    work_rows, work_replayed = original_continuum_work_rows(history, source_raw)
    write(out / "original_continuum_work.json", work_rows)
    write(out / "original_continuum_work_replayed.json", work_replayed)

    old_bindings = {row["label"]:row["binding"] for row in read(
        PHYSICAL / "control_release_01/incoming_first_entry_bindings.json")}
    old_handovers = {row["label"]:row["handover"] for row in read(
        PREVIOUS / "control_release_01/first_entry_handovers.json")}
    records = read(SOURCE / "point_certificates.json")
    if tuple(row["label"] for row in records) != POINT_LABELS:
        raise AssertionError("The complete original seven-label context changed")
    point_rows, point_replayed, chronologies, bank_rows, observations, primitives = [], [], [], [], [], []
    first_count = negative_count = 0
    for record in records:
        label = record["label"]
        print("Evaluating complete domain registry on original implicit chronology: " + label, flush=True)
        chronology = native_first_entry_material_point_chronology_from_dict(deepcopy(record["certificate"]),
            state=case["material_state"])
        chronology_raw = native_first_entry_material_point_chronology_to_dict(chronology)
        if encoded(chronology_raw) != encoded(record["certificate"]):
            raise AssertionError("The original chronology changed: " + label)
        binding = certify_native_first_entry_physical_binding(chronology, history, precision=256)
        binding_raw = native_first_entry_physical_binding_to_dict(binding)
        if encoded(binding_raw) != encoded(old_bindings[label]):
            raise AssertionError("The original physical bank source changed: " + label)
        point = evaluate_native_first_entry_contact_domain_point(registry, binding,
            expected_source_history_dto=source_raw)
        point_raw = native_first_entry_contact_domain_point_to_dict(point)
        point_replay = native_first_entry_contact_domain_point_from_dict(deepcopy(point_raw),
            expected_registry=registry, expected_binding=binding, expected_source_history_dto=source_raw)
        chronologies.append(dict(label=label, certificate=chronology_raw))
        bank_rows.append(dict(label=label, binding=binding_raw))
        point_rows.append(dict(label=label, query=point_raw))
        point_replayed.append(dict(label=label, query=native_first_entry_contact_domain_point_to_dict(point_replay)))
        handover = point.point_handover
        mask_values = [evaluate_native_first_entry_interface_carrier_mask(mask, chronology)
            for mask in integral.carrier_masks]
        observations.append(dict(label=label, masks=[dict(
            original_donor_fragment_id=mask.original_donor_fragment_id,
            original_receiver_fragment_id=mask.original_receiver_fragment_id, selected=value)
            for mask, value in zip(integral.carrier_masks, mask_values)],
            source_family_count_before=len(registry.families), source_family_count_after=len(registry.families),
            first_root_midpoint_used=False, finite_point_area_m2=None, finite_point_work_j=None))
        if binding.time_fraction_bounds is None:
            if label in ENTRY_LABELS or handover.attachments or any(value is True for value in mask_values):
                raise AssertionError("A non-entry context created a material contact")
            negative_count += 1
            continue
        if label not in ENTRY_LABELS or len(handover.attachments) != 1 or mask_values.count(True) != 1:
            raise AssertionError("The whole domain mask or positive first-root context differs")
        if encoded(native_first_entry_contact_handover_to_dict(handover)) != encoded(old_handovers[label]):
            raise AssertionError("The whole domain changed the conserved first-event operator: " + label)
        attachment, = handover.attachments
        if label == "transverse_masked_main_entry" and attachment.desired_fragment_pair != ("octant:6", "octant:3"):
            raise AssertionError("The actual first receiver 6→3 was replaced")
        first_count += 1
        for primitive in point.source_work_primitives:
            from tectonics.geometric_native_contact_field_work import native_contact_field_work_primitive_to_dict
            primitive_raw = native_contact_field_work_primitive_to_dict(primitive, history=history,
                expected_source_history_dto=source_raw)
            if primitive.final_time_fraction_bounds != binding.time_fraction_bounds:
                raise AssertionError("An implicit work primitive selected a root midpoint")
            primitives.append(dict(label=label, source_receiver_trace_id=attachment.original_receiver_trace_id,
                primitive=primitive_raw))
    if (len(point_rows) != 7 or first_count != 4 or negative_count != 3 or len(primitives) != 4
            or len(registry.families) != 4 or len(integral.parent_measures) != 8 or len(integral.carrier_masks) != 16):
        raise AssertionError("Whole source-domain, first-root or once-only work-family coverage differs")
    write(out / "point_certificates.json", chronologies)
    write(out / "original_first_entry_bindings.json", bank_rows)
    write(out / "domain_point_queries.json", point_rows)
    write(out / "domain_point_queries_replayed.json", point_replayed)
    write(out / "carrier_mask_observations.json", observations)
    write(out / "first_entry_work_primitives.json", primitives)
    ledger = original_quadrature_ledger(history)
    write(out / "original_quadrature_ledger.json", ledger, seal="ledger_sha256")
    if (out / "original_quadrature_ledger.json").read_bytes() != (
            PREVIOUS / "control_release_01/original_quadrature_ledger.json").read_bytes():
        raise AssertionError("The once-only original literal quadrature ledger changed")
    refusals = domain_mutation_refusals(registry_raw, integral_raw, work_rows,
        registry_loader=registry_loader, integral_loader=integral_loader,
        history=history, source_raw=source_raw)
    write(out / "mutation_refusals.json", dict(format="native-domain-contact-handover-mutation-refusals-1",
        count=len(refusals), mutations=refusals), seal="receipt_sha256")
    write(out / "source_history_after.json", native_physical_history_binding_to_dict(history, mesh=case["mesh"]))
    save_native_underthrust_motion_checkpoint(out / "source_checkpoint_after.json", case["mesh"],
        case["origin"], case["before"], provenance=case["provenance"])
    write(out / "force_candidate_after.json", _candidate_payload(case["candidate"], case["overlaps"]))
    pairs = (("source_history_before.json", "source_history_after.json"),
        ("source_checkpoint_before.json", "source_checkpoint_after.json"),
        ("force_candidate_before.json", "force_candidate_after.json"),
        ("domain_contact_handover.json", "domain_contact_handover_replayed.json"),
        ("first_entry_interface_integral.json", "first_entry_interface_integral_replayed.json"),
        ("original_continuum_work.json", "original_continuum_work_replayed.json"),
        ("domain_point_queries.json", "domain_point_queries_replayed.json"))
    if any((out / first).read_bytes() != (out / second).read_bytes() for first, second in pairs):
        raise AssertionError("Original source bytes or strict whole-domain restart differ")
    if (case["before"].receipts or case["before"].elapsed_time_s != 0
            or case["before"].cumulative_source_work_j != 0 or case["before"].current_inventory.records):
        raise AssertionError("Domain transport changed an accepted source trajectory")
    if any(path.read_bytes() != expected for path, expected in case["original_files"].items()):
        raise AssertionError("An original authoritative source artifact changed")
    after_sources, baseline_after = guard_sources()
    if before_sources != after_sources or baseline != baseline_after:
        raise AssertionError("Python sources changed during the physical control")
    write(out / "sources_after.json", after_sources)
    previous = sealed(PREVIOUS / "release_validation_01/validation_summary.json", "summary_sha256")
    plan, remaining = deepcopy(previous["original_six_point_plan"]), deepcopy(previous["remaining_work_blocks"])
    plan[0].update(completed_here="whole first-entry contact field registry with four once-only receiving histories, sixteen causal carrier masks and eight original parent measures; independent continuous original-interface work and energy budgets",
        remaining="new receiving-pair weighted contact/work integration, native full-bank/clamp force support and spatial admission, accepted intervals and joint restart")
    remaining[0].update(completed="оператор всей неявной области первого входа, сохраняемые семейства истории и мера материала; аналитические балансы работы и энергии на четырёх исходных интерфейсах",
        remaining="численные взвешенные интегралы работы и энергии по новым принимающим парам и замыкание их полного баланса")
    report = dict(format="native-domain-contact-handover-control-1", status="PASS",
        seconds=time.perf_counter() - started, original_source_count=684, source_count=len(before_sources),
        added_python_sources=sorted(set(before_sources) - set(baseline)), existing_sources_changed=[],
        source_before_after_unchanged=True, original_typed_checkpoint_loaded=True,
        original_force_re_solved=True, history_builder_original_force_re_solved=True,
        prior_material_state_replayed=True, original_candidate_all_fields_identical=True,
        original_candidate_field_count=14, original_node_count=16, original_context_count=7,
        first_root_handover_count=first_count, negative_context_count=negative_count,
        original_continuum_trace_count=4, domain_parent_count=8, causal_mask_count=16,
        first_root_midpoint_used=False, first_receiver_pair_6_to_3_committed=True,
        source_history_bytes_unchanged=True, source_checkpoint_bytes_unchanged=True,
        source_candidate_bytes_unchanged=True, original_quadrature_measure_work_counted_once=True,
        unchanged_global_beam_dofs_and_stiffness=True,
        continuum_and_original_quadrature_measures_distinct=True,
        continuum_work_first_law_retains_literal_compliance_residual=True,
        whole_first_entry_domain_registry_committed=True,
        whole_first_entry_interface_domain_numerically_integrated=False,
        whole_original_interface_continuum_work_integrated=True,
        per_pair_incoming_continuum_work_numerically_integrated=False,
        transported_candidate_is_force_equilibrium=False, native_full_bank_and_clamp_support_certified=False,
        finite_measure_assigned_to_first_entry_point=False, nodal_work_assigned_to_first_entry_point=False,
        candidate_mechanics_adopted=False, physical_pair_admission_authorized=False,
        global_native_inventory_updated=False, transfer_external_work_j=0, transfer_dissipation_j=0,
        accepted_loaded_native_intervals=0, accepted_source_work_j=0, mutation_refusal_count=len(refusals),
        byte_pairs=[list(pair) for pair in pairs], law=asdict(law), original_six_point_plan=plan,
        remaining_work_blocks=remaining, remaining_work_block_count=3,
        artifacts_sha256={str(path.relative_to(out)):digest(path) for path in sorted(out.rglob("*")) if path.is_file()})
    saved = write(out / "report.json", report, seal="report_sha256")
    print(json.dumps({name:saved[name] for name in ("status", "seconds", "source_count",
        "domain_parent_count", "causal_mask_count", "first_root_handover_count", "mutation_refusal_count",
        "accepted_loaded_native_intervals")}), flush=True)
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    run_control(parser.parse_args().out.resolve())
