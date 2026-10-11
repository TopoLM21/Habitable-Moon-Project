"""Replay real seam changes and actual frozen-label contact work once per path."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction as F
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from analysis.geometric_native_active_binding_validation.validation_support import (
    PREVIOUS, digest, encoded, guard_sources, read, refuse_mutation,
    verify_previous_sources, write,
)
from analysis.geometric_native_physical_handover_validation.run_handover_control import (
    SOURCE, get_physical_case,
)
from analysis.geometric_native_first_entry_handover_validation.run_first_entry_handover_control import (
    original_quadrature_ledger,
)


CARRIERS = (
    ("two_changes", "octant:3", (F(0), F(1), F(1, 10**55))),
    ("unchanged", "octant:2", (F(0), F(1), F(-1))),
)
BUDGET_FIELDS = (
    "source_work_density_bounds_j_m2", "friction_dissipation_density_bounds_j_m2",
    "bulk_energy_increment_density_bounds_j_m2", "contact_energy_increment_density_bounds_j_m2",
    "total_energy_increment_density_bounds_j_m2", "compliance_balance_residual_density_bounds_j_m2",
)
STATE_FIELDS = (
    "free_gap_bounds_m", "free_jump_bounds_m", "plastic_slip_bounds_m", "cumulative_slip_bounds_m",
    "normal_pressure_bounds_pa", "shear_traction_bounds_pa", "bulk_energy_density_bounds_j_m2",
    "contact_energy_density_bounds_j_m2", "total_energy_density_bounds_j_m2",
)


def rational(value):
    return [value.numerator, value.denominator]


def interval(value):
    return dict(lo=rational(value.lo), hi=rational(value.hi))


def changed_at(path, replacement):
    def mutate(raw):
        target = raw
        for item in path[:-1]:
            target = target[item]
        target[path[-1]] = replacement
    return mutate


def shifted_interval(path, amount=F(1, 64)):
    def mutate(raw):
        target = raw
        for item in path:
            target = target[item]
        for endpoint in ("lo", "hi"):
            target[endpoint] = rational(F(*target[endpoint]) + amount)
    return mutate


def reference_arguments(reference, source):
    return dict(expected_history=reference.history, expected_source_history_dto=source,
        expected_receiver_trace_id=reference.receiver_trace_id,
        expected_donor_trace_id=reference.donor_trace_id,
        expected_receiver_original_direction=reference.receiver_original_direction,
        expected_switch_time_fraction_bounds=reference.switch_time_fraction_bounds,
        expected_switch_chronology=reference.switch_chronology,
        expected_switch_event_sequence=reference.switch_event_sequence,
        expected_root_proof_context=json.loads(reference.root_proof_context_payload_json),
        expected_initial_state=reference.initial_state,
        precision=reference.precision, order=reference.order)


def mutation_refusals(path_raw, query_raw, reference_raw, *, path_loader,
        query_loader, reference_loader):
    rows = []
    def check(raw, loader, label, path, mutation):
        rows.append(refuse_mutation(raw, mutation=mutation, loader=loader,
            label=label, changed_path=path))
    path_cases = (
        ("unknown_path_field", "unknown_field", lambda raw: raw.update(unknown_field=1)),
        ("path_precision_literal_type_changed", "precision", lambda raw: raw.update(precision=float(raw["precision"]))),
        ("receiver_bank_replaced", "receiver_fragment_id", changed_at(("receiver_fragment_id",), "octant:2")),
        ("receiver_ray_replaced", "receiver_reference_coordinates[2]",
            changed_at(("receiver_reference_coordinates", 2), [1, 10**54])),
        ("donor_chronology_omitted", "donor_chronologies", lambda raw: raw["donor_chronologies"].pop()),
        ("transfer_event_omitted", "events", lambda raw: raw["events"].pop()),
        ("transfer_event_duplicated", "events", lambda raw: raw["events"].append(deepcopy(raw["events"][0]))),
        ("incoming_donor_changed", "events[0].new_donor_fragment_id",
            changed_at(("events", 0, "new_donor_fragment_id"), "octant:7")),
        ("seam_material_coordinate_shifted", "events[0].donor_reference_coordinates_bounds[2]",
            shifted_interval(("events", 0, "donor_reference_coordinates_bounds", 2))),
        ("fresh_owner_derivative_changed", "events[0].fresh_owner_depth_derivative_bounds",
            shifted_interval(("events", 0, "fresh_owner_depth_derivative_bounds"))),
        ("preceding_binding_consumed_again", "events[1].consumed_binding_id",
            changed_at(("events", 1, "consumed_binding_id"), path_raw["events"][0]["consumed_binding_id"])),
        ("material_reentry_fabricated", "events[1].same_material_reentry",
            changed_at(("events", 1, "same_material_reentry"), True)),
        ("frozen_bank_reprojected", "receiving_carrier_reprojected_after_switch",
            changed_at(("receiving_carrier_reprojected_after_switch",), True)),
        ("root_midpoint_fabricated", "root_midpoint_selected", changed_at(("root_midpoint_selected",), True)),
        ("all_angles_schedule_fabricated", "all_receiver_angles_scheduled",
            changed_at(("all_receiver_angles_scheduled",), True)),
        ("native_force_acceptance_fabricated", "accepted_native_force_intervals",
            changed_at(("accepted_native_force_intervals",), 1)),
    )
    for label, path, mutation in path_cases:
        check(path_raw, path_loader, label, path, mutation)
    query_cases = (
        ("unknown_query_field", "unknown_field", lambda raw: raw.update(unknown_field=1)),
        ("query_clock_replaced", "time_fraction_bounds", shifted_interval(("time_fraction_bounds",))),
        ("actual_prefix_segment_omitted", "segments", lambda raw: raw["segments"].pop(0)),
        ("actual_segment_duplicated", "segments", lambda raw: raw["segments"].append(deepcopy(raw["segments"][-1]))),
        ("active_binding_replaced", "active_binding_id",
            changed_at(("active_binding_id",), path_raw["segments"][0]["active_binding_id"])),
        ("consumed_lineage_duplicated", "consumed_binding_ids",
            lambda raw: raw["consumed_binding_ids"].append(raw["consumed_binding_ids"][0])),
        ("actual_work_budget_changed", "budgets.source_work_density_bounds_j_m2",
            shifted_interval(("budgets", "source_work_density_bounds_j_m2"))),
        ("signed_compliance_residual_changed", "budgets.compliance_balance_residual_density_bounds_j_m2",
            shifted_interval(("budgets", "compliance_balance_residual_density_bounds_j_m2"))),
        ("plastic_memory_changed", "segments[-1].plastic_slip_bounds_m",
            shifted_interval(("segments", -1, "plastic_slip_bounds_m"))),
        ("cumulative_memory_changed", "segments[-1].cumulative_slip_bounds_m",
            shifted_interval(("segments", -1, "cumulative_slip_bounds_m"))),
        ("stored_energy_changed", "segments[-1].total_energy_density_bounds_j_m2",
            shifted_interval(("segments", -1, "total_energy_density_bounds_j_m2"))),
        ("affine_source_work_prefix_fabricated", "original_affine_source_work_included",
            changed_at(("original_affine_source_work_included",), True)),
        ("literal_and_continuum_ledgers_summed", "original_nodal_ledgers_summed_with_continuous_work",
            changed_at(("original_nodal_ledgers_summed_with_continuous_work",), True)),
        ("point_area_fabricated", "finite_point_area_m2", changed_at(("finite_point_area_m2",), [1, 1])),
        ("transfer_work_fabricated", "transfer_work_density_j_m2",
            changed_at(("transfer_work_density_j_m2",), [1, 1])),
    )
    for label, path, mutation in query_cases:
        check(query_raw, query_loader, label, path, mutation)
    reference_cases = (
        ("unknown_reference_field", "unknown_field", lambda raw: raw.update(unknown_field=1)),
        ("source_field_replayed_at_later_switch", "initial_state_kind",
            changed_at(("initial_state_kind",), "original_dense_source_at_zero")),
        ("previous_actual_state_replaced", "initial_state_sha256",
            changed_at(("initial_state_sha256",), "0"*64)),
        ("reference_frozen_material_shifted", "frozen_original_material_coordinates[0]",
            shifted_interval(("frozen_original_material_coordinates", 0))),
        ("reference_own_beam_shape_changed", "frozen_shape_bounds",
            shifted_interval(("frozen_shape_bounds",))),
        ("reference_free_gap_changed", "source_gap_at_switch_bounds_m",
            shifted_interval(("source_gap_at_switch_bounds_m",))),
        ("reference_plastic_memory_changed", "source_plastic_slip_at_switch_bounds_m",
            shifted_interval(("source_plastic_slip_at_switch_bounds_m",))),
        ("reference_cumulative_memory_changed", "source_cumulative_slip_at_switch_bounds_m",
            shifted_interval(("source_cumulative_slip_at_switch_bounds_m",))),
        ("reference_current_material_reprojected", "frozen_material_label_reprojected_during_segment",
            changed_at(("frozen_material_label_reprojected_during_segment",), True)),
    )
    for label, path, mutation in reference_cases:
        check(reference_raw, reference_loader, label, path, mutation)
    return rows


def run_control(out):
    from tectonics.geometric_native_contact_reference_transport import ContactHistoryTransportLaw, TRANSPORT_LAW_MODEL
    from tectonics.geometric_native_active_contact_binding import (
        certify_native_active_contact_binding, native_active_contact_binding_to_dict,
        native_active_contact_binding_from_dict,
    )
    from tectonics.geometric_native_active_contact_path_work import (
        query_native_active_contact_path_work, native_active_contact_path_work_to_dict,
        native_active_contact_path_work_from_dict,
    )
    from tectonics.geometric_native_active_contact_work import (
        native_active_contact_segment_reference_to_dict,
        native_active_contact_segment_reference_from_dict,
    )
    from tectonics.geometric_native_physical_history import (
        _candidate_payload, native_physical_history_binding_to_dict,
    )
    from tectonics.geometric_native_pose import FractionInterval as I
    from tectonics.geometric_native_underthrust_motion_io import save_native_underthrust_motion_checkpoint

    started = time.perf_counter()
    before_sources, original_sources = guard_sources(require_frozen=True)
    authority = verify_previous_sources()
    out.mkdir(parents=True, exist_ok=False)
    write(out / "sources_before.json", before_sources)
    write(out / "source_validation_receipt.json", authority, seal="receipt_sha256")
    case = get_physical_case()
    history = case["history"]
    source_raw = native_physical_history_binding_to_dict(history, mesh=case["mesh"])
    write(out / "source_history_before.json", source_raw)
    if (out / "source_history_before.json").read_bytes() != (
            PREVIOUS / "control_release_01/source_history_before.json").read_bytes():
        raise AssertionError("The complete original physical source changed")
    save_native_underthrust_motion_checkpoint(out / "source_checkpoint_before.json",
        case["mesh"], case["origin"], case["before"], provenance=case["provenance"])
    if (out / "source_checkpoint_before.json").read_bytes() != case["original_files"][SOURCE / "original_force/initial.json"]:
        raise AssertionError("The original typed source checkpoint changed")
    write(out / "force_candidate_before.json", case["candidate_raw"])
    law = ContactHistoryTransportLaw(TRANSPORT_LAW_MODEL,
        "Explicit receiver-attached complete configuration; bank changes only at proved quarter seams")
    write(out / "declared_handover_law.json", asdict(law))
    write(out / "receiver_carriers.json", [dict(label=label, receiver_fragment_id=receiver,
        receiver_reference_coordinates=[rational(value) for value in carrier])
        for label, receiver, carrier in CARRIERS])
    path_rows, replay_rows, paths, path_loaders = [], [], {}, {}
    for label, receiver, carrier in CARRIERS:
        print("Certifying complete fixed-receiver path: " + label, flush=True)
        path = certify_native_active_contact_binding(history, receiver, carrier,
            law=law, expected_source_history_dto=source_raw)
        arguments = dict(expected_history=history, expected_law=law,
            expected_source_history_dto=source_raw, expected_receiver_fragment_id=receiver,
            expected_receiver_reference_coordinates=carrier)
        loader = lambda raw, args=arguments: native_active_contact_binding_from_dict(raw, **args)
        raw = native_active_contact_binding_to_dict(path)
        replay = loader(deepcopy(raw))
        path_rows.append(dict(label=label, path=raw))
        replay_rows.append(dict(label=label, path=native_active_contact_binding_to_dict(replay)))
        paths[label], path_loaders[label] = path, loader
    changing, retained = paths["two_changes"], paths["unchanged"]
    if ([(event.old_donor_fragment_id, event.new_donor_fragment_id) for event in changing.events]
            != [("octant:7", "octant:6"), ("octant:6", "octant:7")]
            or len(changing.segments) != 3 or retained.events or len(retained.segments) != 1):
        raise AssertionError("The genuine two-change and unchanged receiving paths differ")
    if any(event.material_reentry or not event.first_entry_proved for event in changing.events):
        raise AssertionError("A quarter change reallocated an already-entered material label")
    if changing.events[0].donor_reference_coordinates_bounds == changing.events[1].donor_reference_coordinates_bounds:
        raise AssertionError("The returning donor reused the preceding original material label")
    write(out / "active_paths.json", path_rows)
    write(out / "active_paths_replayed.json", replay_rows)
    first, second = changing.events
    clocks = {
        "two_changes": (("initial", I(F(0))), ("before_first", I(F(1, 10**13))),
            ("first", first.time_fraction_bounds), ("between", I(F(1, 10**11))),
            ("second", second.time_fraction_bounds), ("after_second", I(F(1, 10**10))),
            ("final", I(F(1)))),
        "unchanged": (("initial", I(F(0))), ("final", I(F(1)))),
    }
    query_rows, query_replayed, queries, query_loaders = [], [], {}, {}
    reference_rows, reference_replayed, references, reference_loaders = [], [], {}, {}
    for label, path in paths.items():
        for clock_label, clock in clocks[label]:
            print("Evaluating actual frozen-label work: " + label + ":" + clock_label, flush=True)
            query = query_native_active_contact_path_work(path, clock,
                expected_source_history_dto=source_raw)
            arguments = dict(expected_binding=path, expected_time_fraction_bounds=clock,
                expected_source_history_dto=source_raw)
            loader = lambda raw, args=arguments: native_active_contact_path_work_from_dict(raw, **args)
            raw = native_active_contact_path_work_to_dict(query, expected_source_history_dto=source_raw)
            replay = loader(deepcopy(raw))
            query_rows.append(dict(label=label, clock_label=clock_label, clock=interval(clock), query=raw))
            query_replayed.append(dict(label=label, clock_label=clock_label, clock=interval(clock),
                query=native_active_contact_path_work_to_dict(replay, expected_source_history_dto=source_raw)))
            queries[label, clock_label], query_loaders[label, clock_label] = query, loader
            if query.finite_point_area_m2 is not None or query.finite_point_work_j is not None:
                raise AssertionError("An isolated receiving carrier acquired finite contact area or work")
            if len(set(query.consumed_binding_ids)) != len(query.consumed_binding_ids):
                raise AssertionError("A preceding actual binding was consumed more than once")
            if query.active_binding_id in query.consumed_binding_ids:
                raise AssertionError("The current attachment remains active after it was consumed")
            if query.candidate_mechanics_adopted or query.accepted_native_force_intervals:
                raise AssertionError("Prescribed active geometry adopted the refused physical candidate")
            for segment in query.segments:
                reference = segment.reference
                if reference.reference_sha256 in references:
                    continue
                reference_raw = native_active_contact_segment_reference_to_dict(reference)
                reference_args = reference_arguments(reference, source_raw)
                reference_loader = lambda raw, args=reference_args: native_active_contact_segment_reference_from_dict(raw, **args)
                reference_replay = reference_loader(deepcopy(reference_raw))
                sequence = json.loads(reference.root_proof_context_payload_json)["segment_sequence"]
                reference_rows.append(dict(label=label, segment_sequence=sequence, reference=reference_raw))
                reference_replayed.append(dict(label=label, segment_sequence=sequence,
                    reference=native_active_contact_segment_reference_to_dict(reference_replay)))
                references[reference.reference_sha256] = reference
                reference_loaders[reference.reference_sha256] = reference_loader
    if len(reference_rows) != 4:
        raise AssertionError("The active histories must retain precisely three changing references and one unchanged reference")
    for clock_label, revision in (("initial", 0), ("before_first", 0), ("first", 1),
            ("between", 1), ("second", 2), ("after_second", 2), ("final", 2)):
        if queries["two_changes", clock_label].revision != revision:
            raise AssertionError("An actual clock selected the wrong unique active binding")
    for clock_label in ("first", "second"):
        query = queries["two_changes", clock_label]
        previous, current = query.segments[-2:]
        if any(getattr(current, name) != I(F(0)) for name in BUDGET_FIELDS):
            raise AssertionError("A common switch expression created work, dissipation or energy")
        for name in STATE_FIELDS:
            if getattr(previous, name) != getattr(current, name):
                raise AssertionError("A transfer changed the evolved preceding actual state: " + name)
        if current.reference.initial_state != previous:
            raise AssertionError("A later switch reloaded the affine source instead of the preceding active state")
    write(out / "active_queries.json", query_rows)
    write(out / "active_queries_replayed.json", query_replayed)
    write(out / "active_segment_references.json", reference_rows)
    write(out / "active_segment_references_replayed.json", reference_replayed)
    ledger = original_quadrature_ledger(history)
    write(out / "original_quadrature_ledger.json", ledger, seal="ledger_sha256")
    if (out / "original_quadrature_ledger.json").read_bytes() != (
            PREVIOUS / "control_release_01/original_quadrature_ledger.json").read_bytes():
        raise AssertionError("The original sixteen literal quadrature histories changed")
    final_query = queries["two_changes", "final"]
    last_reference = final_query.current_state.reference
    refusals = mutation_refusals(path_rows[0]["path"], query_rows[6]["query"],
        native_active_contact_segment_reference_to_dict(last_reference),
        path_loader=path_loaders["two_changes"], query_loader=query_loaders["two_changes", "final"],
        reference_loader=reference_loaders[last_reference.reference_sha256])
    write(out / "mutation_refusals.json", dict(format="native-active-binding-mutation-refusals-1",
        count=len(refusals), mutations=refusals), seal="receipt_sha256")
    write(out / "source_history_after.json", native_physical_history_binding_to_dict(history, mesh=case["mesh"]))
    save_native_underthrust_motion_checkpoint(out / "source_checkpoint_after.json",
        case["mesh"], case["origin"], case["before"], provenance=case["provenance"])
    write(out / "force_candidate_after.json", _candidate_payload(case["candidate"], case["overlaps"]))
    if any(path.read_bytes() != expected for path, expected in case["original_files"].items()):
        raise AssertionError("A complete original actual-source artifact changed")
    if (case["before"].receipts or case["before"].elapsed_time_s != 0
            or case["before"].cumulative_source_work_j != 0 or case["before"].current_inventory.records):
        raise AssertionError("The active attachment changed an accepted source trajectory")
    pairs = (("source_history_before.json", "source_history_after.json"),
        ("source_checkpoint_before.json", "source_checkpoint_after.json"),
        ("force_candidate_before.json", "force_candidate_after.json"),
        ("active_paths.json", "active_paths_replayed.json"),
        ("active_queries.json", "active_queries_replayed.json"),
        ("active_segment_references.json", "active_segment_references_replayed.json"))
    if any((out / first_name).read_bytes() != (out / second_name).read_bytes()
            for first_name, second_name in pairs):
        raise AssertionError("Original source bytes or strict active-history restart differ")
    after_sources, original_after = guard_sources(require_frozen=True)
    if before_sources != after_sources or original_sources != original_after:
        raise AssertionError("Python sources changed during the fresh physical control")
    write(out / "sources_after.json", after_sources)
    report = dict(format="native-active-binding-control-1", status="PASS",
        seconds=time.perf_counter()-started, original_source_count=694, source_count=len(before_sources),
        added_python_sources=sorted(set(before_sources)-set(original_sources)), existing_sources_changed=[],
        source_before_after_unchanged=True, original_typed_checkpoint_loaded=True,
        original_force_re_solved=True, history_builder_original_force_re_solved=True,
        original_candidate_field_count=14, original_literal_quadrature_ledger_unchanged=True,
        unique_active_binding_per_receiving_carrier=True, event_transfer_increments_zero=True,
        frozen_original_material_label_used_for_later_geometry=True,
        first_switch_uses_evolved_initial_segment=True, second_switch_uses_evolved_previous_segment=True,
        original_affine_source_work_included=False, receiving_carrier_count=2,
        active_transfer_event_count=2, actual_path_query_count=len(query_rows),
        frozen_segment_reference_count=len(reference_rows), mutation_refusal_count=len(refusals),
        source_history_bytes_unchanged=True, source_checkpoint_bytes_unchanged=True,
        source_candidate_bytes_unchanged=True, all_receiver_angles_scheduled=False,
        per_receiving_pair_weighted_contact_integrals_evaluated=False,
        physical_pair_admission_authorized=False, candidate_mechanics_adopted=False,
        accepted_loaded_native_intervals=0, accepted_source_work_j=0,
        completed_scope="unique certified fixed-receiver quarter changes with frozen material labels, actual full-rotation constitutive/work continuation from exact zero, once-only evolved-state transfer and strict restart",
        completed_scope_ru="единственная действующая привязка выбранного принимающего носителя, смена исходных банков только на доказанных швах; полный SO(3) путь замороженных материальных меток, эволюция памяти и локальной работы от u=0, сохраняемый последовательный перенос и restart",
        byte_pairs=[list(pair) for pair in pairs], law=asdict(law),
        artifacts_sha256={str(path.relative_to(out)): digest(path)
            for path in sorted(out.rglob("*")) if path.is_file()})
    saved = write(out / "report.json", report, seal="report_sha256")
    print(encoded({name: saved[name] for name in ("status", "seconds", "source_count",
        "active_transfer_event_count", "actual_path_query_count", "mutation_refusal_count",
        "accepted_loaded_native_intervals")}).decode(), flush=True)
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    run_control(parser.parse_args().out.resolve())
