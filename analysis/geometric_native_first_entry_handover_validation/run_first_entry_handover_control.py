"""Replay physical first-root handovers against the complete original source.

The four explicit entering material labels retain their implicit root windows.
Their receiving continuum history and original donor beam remain distinct.
No finite contact measure or quadrature work is assigned to an isolated label.
The original sixteen quadrature histories remain a separate once-only ledger.
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
PREVIOUS = ROOT / "analysis/geometric_native_physical_handover_validation"
PHYSICAL = ROOT / "analysis/geometric_native_physical_binding_validation"
MATERIAL = ROOT / "analysis/geometric_native_first_entry_validation"
SOURCE = MATERIAL / "control_release_02"
sys.path.insert(0, str(ROOT))
from analysis.geometric_native_physical_handover_validation.run_handover_control import (
    LAW_PROVENANCE, digest, encoded, get_physical_case, rational, read, sealed,
    sources, value_sha, write,
)

POINT_LABELS = (
    "main_plane_reentry", "transverse_masked_main_entry", "ordinary_entry", "no_entry",
    "swept_only_owner_entry_exit", "empty_donor_four", "retained_overriding_material",
)
ENTRY_LABELS = (POINT_LABELS[0], POINT_LABELS[1], POINT_LABELS[2], POINT_LABELS[4])


def guard_sources():
    current, starting = sources(), read(BASE / "starting_sources.json")
    prior_maps = [PREVIOUS / directory / ("sources_" + stage + ".json")
        for directory in ("release_gates_02", "control_release_02")
        for stage in ("before", "after")]
    if len(starting) != 674 or any(read(path) != starting for path in prior_maps):
        raise ValueError("First-entry handover must retain the authenticated 674-source release")
    changed = sorted(name for name, expected in starting.items() if current.get(name) != expected)
    if changed:
        raise ValueError("A frozen original Python source changed: " + repr(changed))
    return current, starting


def verify_previous_sources():
    releases = ((PREVIOUS / "release_validation_02", 46, 178),
        (PHYSICAL / "release_validation_01", 56, 253),
        (MATERIAL / "release_validation_02", 83, 770))
    ancestry, artifact_paths = [], set()
    for directory, count, tests in releases:
        summary_path, manifest_path = directory / "validation_summary.json", directory / "artifact_manifest.json"
        summary = sealed(summary_path, "summary_sha256")
        manifest = sealed(manifest_path, "manifest_sha256")
        if (summary["status"] != "PASS" or manifest["count"] != count
                or len(manifest["artifacts"]) != count
                or summary["artifact_manifest"]["sha256"] != digest(manifest_path)):
            raise ValueError("Prior authenticated release changed: " + str(directory))
        for name, record in manifest["artifacts"].items():
            path = ROOT / name
            if not path.is_file() or path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
                raise ValueError("Prior artifact differs: " + name)
            artifact_paths.add(path)
        for key in ("fresh_scoped_gate", "release_gate", "actual_control", "independent_audit"):
            if key in summary:
                reference = summary[key]
                if digest(ROOT / reference["path"]) != reference["sha256"]:
                    raise ValueError("Prior evidence reference differs: " + key)
        ancestry.append(dict(path=str(summary_path.relative_to(ROOT)), sha256=digest(summary_path),
            verified_manifest_records=count, tests_passed=tests, tests_reexecuted=False))
        artifact_paths.update((summary_path, manifest_path))
    previous = sealed(PREVIOUS / "release_validation_02/validation_summary.json", "summary_sha256")
    if previous["source_count"] != 674 or previous["fresh_scoped_gate"]["tests_passed"] != 178:
        raise ValueError("Prior source/test ancestry differs")
    source_paths = (
        PREVIOUS / "release_validation_02/validation_summary.json",
        PREVIOUS / "release_validation_02/artifact_manifest.json",
        PREVIOUS / "release_gates_02/receipt.json",
        PREVIOUS / "control_release_02/report.json",
        PREVIOUS / "control_release_02/independent_handover_receipt.json",
        PHYSICAL / "control_release_01/original_history.json",
        PHYSICAL / "control_release_01/incoming_first_entry_bindings.json",
        SOURCE / "original_force/initial.json", SOURCE / "original_force/native_source.json",
        SOURCE / "original_force/force_candidate.json", SOURCE / "material_state.json",
        SOURCE / "point_certificates.json", BASE / "starting_sources.json",
    )
    return dict(format="native-first-entry-handover-source-validation-1", status="PASS",
        original_source_count=674, verified_prior_manifest_records=185,
        verified_unique_prior_artifacts=len(artifact_paths), prior_tests_reexecuted=False,
        ancestry=ancestry, source_artifacts={str(path.relative_to(ROOT)):dict(
            sha256=digest(path), bytes=path.stat().st_size) for path in source_paths})


def original_quadrature_ledger(history):
    """Preserve the authenticated atoms without treating them as field densities."""
    records, totals = [], {}
    area = Fraction(0)
    for node in history.quadrature_histories:
        state = node.unaccepted_candidate_state
        raw = asdict(state)
        area += Fraction.from_float(node.original_area_m2)
        for name, value in raw.items():
            if name.endswith("_j"):
                totals[name] = totals.get(name, Fraction(0)) + Fraction.from_float(value)
        records.append(dict(point_id=node.point_id, trace_id=node.trace_id,
            original_area_m2=rational(Fraction.from_float(node.original_area_m2)),
            disposition="retained_once_in_original_quadrature_lineage", local_state_before=raw,
            local_state_after=deepcopy(raw)))
    if len(records) != 16 or len({row["point_id"] for row in records}) != 16:
        raise AssertionError("The original sixteen quadrature histories are not represented once")
    beam = dict(parameters=[asdict(row) for row in history.initial_mechanics.bending_parameters],
        candidate_transverse_displacement_m=history.actual_candidate.transverse_displacement_m,
        original_stiffness=history.original_reference.stiffness)
    return dict(format="native-first-entry-handover-original-once-only-ledger-1",
        original_node_count=16, quadrature_histories=records, total_original_area_m2=rational(area),
        original_energy_work_totals_j={name:rational(value) for name, value in totals.items()},
        source_work_before=history.before_work_ledgers,
        unaccepted_candidate_work=history.unaccepted_candidate_work_ledgers,
        beam_configuration_before=beam, beam_configuration_after=deepcopy(beam),
        local_energy_work_counted_once=True, unchanged_global_beam_dofs_and_stiffness=True,
        original_atoms_assigned_to_first_entry_points=False,
        finite_measure_assigned_to_first_entry_point=False,
        candidate_mechanics_adopted=False, accepted_loaded_native_intervals=0,
        accepted_source_work_j=0, transfer_external_work_j=0, transfer_dissipation_j=0)


def rehash_declared_seals(value):
    if type(value) is list:
        for item in value:
            rehash_declared_seals(item)
    elif type(value) is dict:
        for item in value.values():
            rehash_declared_seals(item)
        # Every declared nested DTO uses an external-source replay after this
        # valid selfseal is recomputed. Source identities are not authorities.
        mapping = {
            "native-original-physical-history-binding-1":"binding_sha256",
            "native-first-entry-original-physical-bank-binding-1":"binding_sha256",
            "native-source-beam-material-chart-1":"chart_sha256",
            "native-first-entry-carrier-contact-measure-1":"measure_sha256",
        }
        if value.get("format") == "native-source-beam-material-chart-1":
            value["source_sha256"] = value_sha(value["source"])
        key = mapping.get(value.get("format"))
        if key is None:
            seals = [name for name in value if name in (
                "handover_sha256", "field_sha256", "query_sha256", "state_sha256", "family_sha256")]
            key = seals[0] if len(seals) == 1 else None
        if key in value:
            value[key] = value_sha({name:item for name, item in value.items() if name != key})


def mutation_refusals(raw, *, binding, source_raw, law, scientific=False):
    """Reject validly rehashed scientific mutations against independent sources."""
    from tectonics.geometric_native_first_entry_contact_handover import (
        native_first_entry_contact_handover_from_dict,
    )
    result = []
    cases = [
        ("unknown_complete_schema_field", "unknown_field", lambda dto: dto.update(unknown_field=1)),
        ("top_level_literal_type_changed", "precision", lambda dto: dto.update(precision=float(dto["precision"]))),
    ]
    def field_change(path, value):
        def mutate(dto):
            target = dto
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
        return mutate
    def shifted(path):
        def mutate(dto):
            target = dto
            for key in path:
                target = target[key]
            for key in ("lo", "hi"):
                target[key] = rational(Fraction(*target[key]) + Fraction(1, 64))
        return mutate
    def midpoint_root(dto):
        bounds = dto["original_physical_binding"]["time_fraction_bounds"]
        middle = rational((Fraction(*bounds["lo"]) + Fraction(*bounds["hi"])) / 2)
        bounds.update(lo=middle, hi=middle)
    if scientific:
        cases.extend((
            ("first_root_midpoint_claimed", "first_root_midpoint_used",
                field_change(("first_root_midpoint_used",), True)),
            ("first_root_window_replaced_by_midpoint", "original_physical_binding.time_fraction_bounds", midpoint_root),
            ("receiving_trace_replaced_by_donor", "attachments[0].source_field_query.trace_id",
                field_change(("attachments", 0, "source_field_query", "trace_id"),
                    raw["attachments"][0]["original_donor_trace_id"])),
            ("changed_first_receiver_material", "attachments[0].desired_fragment_pair[1]",
                field_change(("attachments", 0, "desired_fragment_pair", 1), "octant:2")),
            ("source_beam_width_changed", "attachments[0].sub_material_chart.source.width_m",
                field_change(("attachments", 0, "sub_material_chart", "source", "width_m"), [10001, 1])),
            ("incoming_source_beam_shape_changed", "attachments[0].sub_material_chart.shape",
                shifted(("attachments", 0, "sub_material_chart", "shape"))),
            ("incoming_own_beam_amplitude_changed", "attachments[0].sub_tip_displacement_bounds_m",
                shifted(("attachments", 0, "sub_tip_displacement_bounds_m"))),
            ("receiving_frame_changed", "attachments[0].receiver_world_normal[0]",
                shifted(("attachments", 0, "receiver_world_normal", 0))),
            ("original_plastic_memory_changed", "attachments[0].source_field_query.plastic_slip_bounds_m",
                shifted(("attachments", 0, "source_field_query", "plastic_slip_bounds_m"))),
            ("original_cumulative_memory_changed", "attachments[0].source_field_query.cumulative_slip_bounds_m",
                shifted(("attachments", 0, "source_field_query", "cumulative_slip_bounds_m"))),
            ("original_energy_density_changed", "attachments[0].source_field_query.total_energy_density_bounds_j_m2",
                shifted(("attachments", 0, "source_field_query", "total_energy_density_bounds_j_m2"))),
            ("original_reference_measure_changed", "attachments[0].interface_measure.source.source_interface_area_density_m2_per_radian",
                field_change(("attachments", 0, "interface_measure", "source",
                    "source_interface_area_density_m2_per_radian"), [1, 1])),
            ("swept_material_jacobian_changed", "attachments[0].interface_measure.signed_solid_angle_jacobian",
                shifted(("attachments", 0, "interface_measure", "signed_solid_angle_jacobian"))),
            ("attachment_commit_removed", "attachments[0].attachment_committed",
                field_change(("attachments", 0, "attachment_committed"), False)),
            ("finite_area_assigned_to_point", "attachments[0].finite_point_area_m2",
                field_change(("attachments", 0, "finite_point_area_m2"), 1.0)),
            ("nodal_work_assigned_to_point", "attachments[0].finite_point_work_j",
                field_change(("attachments", 0, "finite_point_work_j"), 1.0)),
            ("fabricated_cumulative_work_density", "attachments[0].source_field_query.cumulative_work_density_j_m2",
                field_change(("attachments", 0, "source_field_query", "cumulative_work_density_j_m2"), 1.0)),
            ("once_only_attachment_receipt_duplicated", "attachments[0].transport_family.receipts",
                lambda dto: dto["attachments"][0]["transport_family"]["receipts"].append(
                    deepcopy(dto["attachments"][0]["transport_family"]["receipts"][0]))),
            ("unpaid_transfer_work_created", "additional_handover_work_j",
                field_change(("additional_handover_work_j",), [1, 1])),
            ("original_source_transverse_axis_removed", "complete_original_source_history.actual_candidate.omega_rad_per_myr",
                lambda dto: next(row for row in dto["complete_original_source_history"]["actual_candidate"]["omega_rad_per_myr"]
                    if row[0] == 1)[1].__setitem__(0, 0.0)),
            ("transport_law_provenance_changed", "explicit_law.provenance",
                field_change(("explicit_law", "provenance"), law.provenance + " changed")),
        ))
    for label, changed_path, mutate in cases:
        forged = deepcopy(raw)
        mutate(forged)
        rehash_declared_seals(forged)
        try:
            native_first_entry_contact_handover_from_dict(forged, expected_binding=binding,
                expected_source_history_dto=source_raw, expected_law=law)
        except (ValueError, TypeError) as error:
            result.append(dict(label=label, changed_path=changed_path, rejected=True,
                valid_rehashed_handover_checksum=True, exception_type=type(error).__name__,
                message=str(error), mutated_dto_sha256=value_sha(forged)))
        else:
            raise AssertionError("A validly rehashed first-entry mutation was accepted: " + label)
    if scientific:
        forged_source = deepcopy(source_raw)
        forged_source["actual_candidate"]["states"][0][1]["cumulative_slip_m"] += 0.001
        rehash_declared_seals(forged_source)
        try:
            native_first_entry_contact_handover_from_dict(raw, expected_binding=binding,
                expected_source_history_dto=forged_source, expected_law=law)
        except (ValueError, TypeError) as error:
            result.append(dict(label="independent_complete_source_changed",
                changed_path="expected_source_history_dto.actual_candidate.states", rejected=True,
                valid_rehashed_source_checksum=True, exception_type=type(error).__name__,
                message=str(error), mutated_dto_sha256=value_sha(forged_source)))
        else:
            raise AssertionError("A forged independently supplied original history was accepted")
    return result


def run_control(out):
    from tectonics.geometric_native_contact_reference_transport import ContactHistoryTransportLaw, TRANSPORT_LAW_MODEL
    from tectonics.geometric_native_first_entry_material import (
        native_first_entry_material_point_chronology_from_dict,
        native_first_entry_material_point_chronology_to_dict,
    )
    from tectonics.geometric_native_first_entry_physical_binding import (
        certify_native_first_entry_physical_binding, native_first_entry_physical_binding_to_dict,
    )
    from tectonics.geometric_native_first_entry_contact_handover import (
        certify_native_first_entry_contact_handover, native_first_entry_contact_handover_to_dict,
        native_first_entry_contact_handover_from_dict, evaluate_native_first_entry_contact_free_anchors,
    )
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
        raise AssertionError("The complete original physical history changed")
    checkpoint = save_native_underthrust_motion_checkpoint(out / "source_checkpoint_before.json",
        case["mesh"], case["origin"], case["before"], provenance=case["provenance"])
    if checkpoint.read_bytes() != case["original_files"][SOURCE / "original_force/initial.json"]:
        raise AssertionError("The typed source checkpoint did not preserve its original bytes")
    write(out / "force_candidate_before.json", case["candidate_raw"])
    law = ContactHistoryTransportLaw(model=TRANSPORT_LAW_MODEL, provenance=LAW_PROVENANCE)
    write(out / "declared_handover_law.json", asdict(law))
    original_bindings = {row["label"]:row["binding"] for row in read(
        PHYSICAL / "control_release_01/incoming_first_entry_bindings.json")}
    chronologies, binding_rows, handover_rows, replay_rows, negatives, refusals, observations = [], [], [], [], [], [], []
    records = read(SOURCE / "point_certificates.json")
    if tuple(row["label"] for row in records) != POINT_LABELS:
        raise AssertionError("The complete original seven-point context changed")
    for record in records:
        label = record["label"]
        print("Authenticating original first-entry chronology and physical banks: " + label, flush=True)
        chronology = native_first_entry_material_point_chronology_from_dict(deepcopy(record["certificate"]),
            state=case["material_state"])
        chronology_raw = native_first_entry_material_point_chronology_to_dict(chronology)
        if encoded(chronology_raw) != encoded(record["certificate"]):
            raise AssertionError("An authenticated incoming chronology changed: " + label)
        binding = certify_native_first_entry_physical_binding(chronology, history, precision=256)
        binding_raw = native_first_entry_physical_binding_to_dict(binding)
        if encoded(binding_raw) != encoded(original_bindings[label]):
            raise AssertionError("An original bank binding changed: " + label)
        chronologies.append(dict(label=label, certificate=chronology_raw))
        binding_rows.append(dict(label=label, binding=binding_raw))
        if binding.time_fraction_bounds is None:
            if label in ENTRY_LABELS:
                raise AssertionError("An original true first entry disappeared: " + label)
            no_entry = certify_native_first_entry_contact_handover(binding, law=law,
                expected_source_history_dto=source_raw, precision=256, order=32)
            no_entry_raw = native_first_entry_contact_handover_to_dict(no_entry)
            no_entry_replay = native_first_entry_contact_handover_from_dict(deepcopy(no_entry_raw),
                expected_binding=binding, expected_source_history_dto=source_raw, expected_law=law)
            no_entry_replayed_raw = native_first_entry_contact_handover_to_dict(no_entry_replay)
            if no_entry.attachments or encoded(no_entry_raw) != encoded(no_entry_replayed_raw):
                raise AssertionError("A no-entry context created an attachment or changed its strict replay")
            negatives.append(dict(label=label, binding=binding_raw, handover_created=False,
                handover=no_entry_raw, replayed_handover=no_entry_replayed_raw))
            continue
        if label not in ENTRY_LABELS:
            raise AssertionError("A non-entry context gained a material handover: " + label)
        print("Transporting receiving continuum configuration on its true implicit first root: " + label, flush=True)
        handover = certify_native_first_entry_contact_handover(binding, law=law,
            expected_source_history_dto=source_raw, precision=256, order=32)
        raw = native_first_entry_contact_handover_to_dict(handover)
        replay = native_first_entry_contact_handover_from_dict(deepcopy(raw), expected_binding=binding,
            expected_source_history_dto=source_raw, expected_law=law)
        handover_rows.append(dict(label=label, handover=raw))
        replay_rows.append(dict(label=label, handover=native_first_entry_contact_handover_to_dict(replay)))
        if len(handover.attachments) != 1:
            raise AssertionError("An explicit first root needs one proved first receiver: " + label)
        attachment = handover.attachments[0]
        if (attachment.source_field_query.trace_id != attachment.original_receiver_trace_id
                or attachment.sub_material_chart.trace_id != attachment.original_donor_trace_id
                or not attachment.attachment_committed or attachment.transport_family.revision != 1
                or len(attachment.transport_family.receipts) != 1
                or attachment.additional_handover_work_j != 0
                or attachment.finite_point_area_m2 is not None or attachment.finite_point_work_j is not None):
            raise AssertionError("The first-entry attachment lost its source/receiver or once-only lineage")
        if label == "transverse_masked_main_entry" and (
                attachment.desired_fragment_pair != ("octant:6", "octant:3")
                or attachment.original_donor_trace_pair != ("octant:6", "octant:2")
                or attachment.original_receiver_trace_pair != ("octant:7", "octant:3")
                or attachment.original_receiver_trace_id == attachment.original_donor_trace_id):
            raise AssertionError("The actual 6→3 handover reused the wrong donor or receiver history")
        anchors = evaluate_native_first_entry_contact_free_anchors(handover, 0,
            (attachment.reference_gap_bounds, attachment.reference_jump_bounds),
            geometry_context=attachment.expression_context)
        expected_anchors = (attachment.source_free_gap_bounds, attachment.source_free_jump_bounds)
        if anchors != expected_anchors:
            raise AssertionError("The implicit first-event geometry did not cancel by expression identity")
        observations.append(dict(label=label, desired_fragment_pair=list(attachment.desired_fragment_pair),
            source_receiver_trace_id=attachment.original_receiver_trace_id,
            own_donor_beam_trace_id=attachment.original_donor_trace_id,
            source_free_gap_bounds=raw["attachments"][0]["source_field_query"]["free_gap_bounds_m"],
            transported_free_gap_bounds=raw["attachments"][0]["transported_free_gap_bounds"],
            source_free_jump_bounds=raw["attachments"][0]["source_field_query"]["free_jump_bounds_m"],
            transported_free_jump_bounds=raw["attachments"][0]["transported_free_jump_bounds"],
            implicit_root_reference_cancels_symbolically=True, attachment_revision=1,
            finite_point_area_m2=None, finite_point_work_j=None))
        refusals.extend(dict(context_label=label, **row) for row in mutation_refusals(
            raw, binding=binding, source_raw=source_raw, law=law,
            scientific=label == "transverse_masked_main_entry"))
    if tuple(row["label"] for row in handover_rows) != ENTRY_LABELS or len(negatives) != 3:
        raise AssertionError("First-entry positive/negative coverage changed")
    write(out / "point_certificates.json", chronologies)
    write(out / "original_first_entry_bindings.json", binding_rows)
    write(out / "first_entry_handovers.json", handover_rows)
    write(out / "first_entry_handovers_replayed.json", replay_rows)
    write(out / "negative_contexts.json", negatives)
    write(out / "first_root_reference_observations.json", dict(
        format="native-first-entry-handover-reference-observations-1", count=len(observations),
        observations=observations), seal="receipt_sha256")
    ledger = original_quadrature_ledger(history)
    write(out / "original_quadrature_ledger.json", ledger, seal="ledger_sha256")
    write(out / "mutation_refusals.json", dict(format="native-first-entry-handover-mutation-refusals-1",
        count=len(refusals), mutations=refusals), seal="receipt_sha256")
    write(out / "source_history_after.json", native_physical_history_binding_to_dict(history, mesh=case["mesh"]))
    save_native_underthrust_motion_checkpoint(out / "source_checkpoint_after.json", case["mesh"],
        case["origin"], case["before"], provenance=case["provenance"])
    write(out / "force_candidate_after.json", _candidate_payload(case["candidate"], case["overlaps"]))
    pairs = (("source_history_before.json", "source_history_after.json"),
        ("source_checkpoint_before.json", "source_checkpoint_after.json"),
        ("force_candidate_before.json", "force_candidate_after.json"),
        ("first_entry_handovers.json", "first_entry_handovers_replayed.json"))
    if any((out / first).read_bytes() != (out / second).read_bytes() for first, second in pairs):
        raise AssertionError("The unchanged physical source or strict handover replay bytes differ")
    if (case["before"].receipts or case["before"].elapsed_time_s != 0
            or case["before"].cumulative_source_work_j != 0 or case["before"].current_inventory.records):
        raise AssertionError("A first-root handover changed an accepted trajectory or ledger")
    if any(path.read_bytes() != expected for path, expected in case["original_files"].items()):
        raise AssertionError("An authoritative original source artifact changed")
    after_sources, baseline_after = guard_sources()
    if before_sources != after_sources or baseline != baseline_after:
        raise AssertionError("Python sources changed during the physical control")
    write(out / "sources_after.json", after_sources)
    report = dict(format="native-first-entry-contact-handover-control-1", status="PASS",
        seconds=time.perf_counter() - started, original_source_count=674, source_count=len(before_sources),
        added_python_sources=sorted(set(before_sources) - set(baseline)), existing_sources_changed=[],
        source_before_after_unchanged=True, original_typed_checkpoint_loaded=True,
        original_force_re_solved=True, history_builder_original_force_re_solved=True,
        prior_material_state_replayed=True, original_candidate_all_fields_identical=True,
        original_candidate_field_count=14, original_node_count=16, original_context_count=7,
        first_root_handover_count=4, negative_context_count=3, first_root_midpoint_used=False,
        first_root_reference_observation_count=len(observations),
        first_receiver_pair_6_to_3_committed=True, cross_pair_receiving_history_and_donor_beam_distinct=True,
        source_field_frozen_at_implicit_first_entry=True, implicit_root_geometry_cancelled_symbolically=True,
        source_history_bytes_unchanged=True, source_checkpoint_bytes_unchanged=True,
        source_candidate_bytes_unchanged=True, original_quadrature_measure_work_counted_once=True,
        unchanged_global_beam_dofs_and_stiffness=True, mutation_refusal_count=len(refusals),
        byte_pairs=[list(pair) for pair in pairs], law=asdict(law),
        transported_candidate_is_force_equilibrium=False, native_full_bank_and_clamp_support_certified=False,
        finite_measure_assigned_to_first_entry_point=False, nodal_work_assigned_to_first_entry_point=False,
        whole_first_entry_interface_domain_certified=False,
        candidate_mechanics_adopted=False, physical_pair_admission_authorized=False,
        global_native_inventory_updated=False, transfer_external_work_j=0, transfer_dissipation_j=0,
        accepted_loaded_native_intervals=0, accepted_source_work_j=0,
        artifacts_sha256={str(path.relative_to(out)):digest(path) for path in sorted(out.rglob("*")) if path.is_file()})
    saved = write(out / "report.json", report, seal="report_sha256")
    print(json.dumps({name:saved[name] for name in ("status", "seconds", "source_count",
        "first_root_handover_count", "negative_context_count", "mutation_refusal_count",
        "accepted_loaded_native_intervals")}), flush=True)
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    run_control(parser.parse_args().out.resolve())
