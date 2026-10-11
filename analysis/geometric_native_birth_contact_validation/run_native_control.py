"""A genuine reduced native trial with explicit born supports and limits.

Two initial shared arcs use declared single-point quadrature and five whole
HOT masks own independent hydrostatic supports.  This experiment preserves
the accepted source while the positive vertex fronts stay unresolved.
"""
from __future__ import annotations

from decimal import Decimal, localcontext
from pathlib import Path
import time
import traceback

from analysis.geometric_native_birth_contact_validation.validation_support import (
    BASE, ROOT, digest, encoded, guard_sources, value_sha, write,
)


def _archive(out, current, immutable):
    directory = out / "development_additive_source_bodies"
    directory.mkdir(exist_ok=False)
    rows = {}
    for name in sorted(set(current)-set(immutable)):
        target = directory / (name.replace("/", "__") + ".txt")
        with target.open("xb") as stream:
            stream.write((ROOT / name).read_bytes())
        if digest(target) != current[name]:
            raise ValueError("A development source changed during its archival snapshot: " + name)
        rows[name] = dict(path=str(target.relative_to(ROOT)), sha256=digest(target), bytes=target.stat().st_size)
    return write(out / "development_source_archive.json", dict(format="native-born-development-source-archive-1",
        status="PRESERVED", count=len(rows), sources=rows), "archive_sha256")


def run_native_control(out, *, development=False, inventory=True):
    started = time.perf_counter()
    before, immutable = guard_sources(require_frozen=not development)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write(out / "sources_before.json", before)
    if development:
        _archive(out, before, immutable)
    from analysis.geometric_native_birth_contact_validation.native_case import build_native_birth_network_case
    from tectonics.geometric_native_birth_contact_network import (
        birth_contact_network_to_dict, solve_birth_contact_network,
    )
    from tectonics.geometric_native_birth_contact import (
        born_contact_state_to_dict, born_foundation_state_to_dict,
    )
    from tectonics.geometric_native_implicit_underthrust_motion import native_implicit_underthrust_motion_to_dict
    from tectonics.geometric_native_implicit_mantle_loading import native_implicit_spatial_mantle_load_to_dict
    from tectonics.geometric_native_underthrust_motion import SECONDS_PER_MYR
    from tests.test_geometric_native_implicit_underthrust_motion import get_implicit_second_trial_case

    trial_started = time.perf_counter()
    trial = get_implicit_second_trial_case()
    trial_seconds = time.perf_counter()-trial_started
    previous, mesh = trial["before"], trial["first"]["original"]["mesh"]
    accepted_before = native_implicit_underthrust_motion_to_dict(previous)
    write(out / "accepted_motion_before.json", accepted_before)
    write(out / "actual_source_load.json", native_implicit_spatial_mantle_load_to_dict(mesh,
        previous.current_surface, trial["load"]))
    case_started = time.perf_counter()
    case = build_native_birth_network_case(trial=trial)
    assembly_seconds = time.perf_counter()-case_started
    write(out / "fixed_birth_support.json", case["source_payload"])
    write(out / "native_point_audits.json", case["node_audits"])
    write(out / "scope_before_solve.json", case["scope"])
    inputs = case["network_inputs"]
    force_started = time.perf_counter()
    try:
        mixed = solve_birth_contact_network(**inputs)
        packet = birth_contact_network_to_dict(mixed, **inputs)
        write(out / "actual_mixed_network.json", packet)
        force_seconds = time.perf_counter()-force_started
        baseline_started = time.perf_counter()
        baseline_inputs = dict(inputs, born_attachments=())
        baseline = solve_birth_contact_network(**baseline_inputs)
        write(out / "same_engine_without_born_contacts.json", birth_contact_network_to_dict(baseline, **baseline_inputs))
        baseline_seconds = time.perf_counter()-baseline_started
        with localcontext() as context:
            context.prec = 200
            differences = [str(Decimal(now)-Decimal(old)) for now, old in
                zip(mixed.coordinates_decimal, baseline.coordinates_decimal)]
        if not any(Decimal(value) != 0 for value in differences[:len(inputs["old_stiffness"])]):
            raise AssertionError("The explicit born forces did not affect the genuine unchanged coordinate prefix")
        write(out / "same_engine_born_force_feedback.json", dict(
            comparison="same current mantle load, old matrix/states and five foundations; only born contact registry differs",
            old_coordinate_prefix_changed=True, coordinate_differences_decimal=differences,
            binary64_coordinate_representatives_may_coincide=True), "feedback_sha256")
    except Exception as error:
        with (out / "failure_traceback.log").open("x") as stream:
            stream.write(traceback.format_exc())
        accepted_after = native_implicit_underthrust_motion_to_dict(previous)
        write(out / "accepted_motion_after_refusal.json", accepted_after)
        after, immutable_after = guard_sources()
        write(out / "sources_after_refusal.json", after)
        if encoded(accepted_before) != encoded(accepted_after) or immutable_after != immutable:
            raise AssertionError("The refused mixed trial altered accepted history or inherited sources") from error
        receipt = dict(format="native-born-reduced-trial-refusal-1", status="REFUSED",
            final_release_gate=False, development=development, seconds=time.perf_counter()-started,
            exception_type=type(error).__name__, message=str(error),
            inherited_732_sources_unchanged=True, accepted_interval_preserved=True,
            parameters_automatically_changed=False, whole_native_interval_admitted=False,
            source_map_before_sha256=digest(out / "sources_before.json"),
            source_map_after_sha256=digest(out / "sources_after_refusal.json"),
            traceback_sha256=digest(out / "failure_traceback.log"),
            genuine_old_trial_seconds=trial_seconds, native_assembly_seconds=assembly_seconds)
        write(out / "failure_receipt.json", receipt, "receipt_sha256")
        print(encoded(receipt).decode(), flush=True)
        return dict(receipt=receipt, case=case, error=error)
    if encoded(case["source_payload"]) != case["fixed_declaration_bytes"]:
        raise AssertionError("The birth declaration changed after its genuine force result")
    contact_before, contact_after, foundation_before, foundation_after = [], [], [], []
    for item, step in zip(inputs["born_attachments"], mixed.born_contact_steps):
        expected = dict(expected_parameters=item.parameters, expected_reference=item.reference)
        contact_before.append(born_contact_state_to_dict(item.previous_state, **expected))
        contact_after.append(born_contact_state_to_dict(step.state, **expected))
    for item, step in zip(inputs["born_foundations"], mixed.born_foundation_steps):
        expected = dict(expected_parameters=item.parameters, expected_reference=item.reference)
        foundation_before.append(born_foundation_state_to_dict(item.previous_state, **expected))
        foundation_after.append(born_foundation_state_to_dict(step.state, **expected))
    write(out / "constitutive_packet.json", dict(format="native-born-reduced-constitutive-packet-1",
        cases=[dict(label="genuine-native-declared-birth", contact_states=contact_before, foundation_states=foundation_before),
            dict(label="genuine-native-reduced-force-result", contact_states=contact_after, foundation_states=foundation_after)],
        genuine_native_source_used=True, native_geometry_admitted=False), "packet_sha256")
    omega = tuple((owner, tuple(float(mixed.coordinates[3*index+axis]/inputs["dt_s"]*SECONDS_PER_MYR)
        for axis in range(3))) for index, owner in enumerate(previous.plate_ids))
    inventory_result = None
    inventory_seconds = 0.
    if inventory:
        from analysis.geometric_native_birth_contact_validation.coverage_inventory import (
            build_coverage_inventory, write_inventory,
        )
        progress_count = [0]
        def progress(row):
            progress_count[0] += 1
            print("mixed pair inventory", progress_count[0], row["status"], flush=True)
        inventory_result = build_coverage_inventory(previous.current_surface, omega,
            inputs["dt_s"]/SECONDS_PER_MYR, time_cells=1, precision=128, order=20,
            provenance="Exhaustive read-only pair inventory for this genuinely solved reduced mixed candidate",
            progress=progress)
        write_inventory(out / "mixed_pair_inventory.json", inventory_result)
        inventory_seconds = inventory_result["elapsed_s"]
    accepted_after = native_implicit_underthrust_motion_to_dict(previous)
    write(out / "accepted_motion_after.json", accepted_after)
    if encoded(accepted_before) != encoded(accepted_after) or len(previous.receipts) != 1:
        raise AssertionError("An unadmitted reduced mixed candidate altered accepted history")
    for path, old_bytes in trial["first"]["original"]["original_files"].items():
        if path.read_bytes() != old_bytes:
            raise AssertionError("An inherited actual source artifact changed")
    after, immutable_after = guard_sources(require_frozen=not development)
    write(out / "sources_after.json", after)
    if immutable_after != immutable or not development and after != before:
        raise AssertionError("Immutable sources changed during the mixed trial")
    metrics = dict(mixed.diagnostics)
    report = dict(format="native-born-reduced-genuine-force-control-1", status="PASS",
        development=development, final_release_gate=not development,
        seconds=time.perf_counter()-started, source_count=len(before),
        genuine_old_trial_seconds=trial_seconds, native_assembly_seconds=assembly_seconds,
        actual_mixed_force_and_export_seconds=force_seconds,
        same_engine_feedback_seconds=baseline_seconds, pair_inventory_seconds=inventory_seconds,
        inherited_732_sources_unchanged=True, source_before_after_unchanged=before == after,
        full_current_source_map_changes=sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name)),
        genuine_current_mantle_source_used=True, genuine_mixed_equilibrium_solved=True,
        original_point_count=16, original_coordinate_count=14, mixed_coordinate_count=len(mixed.coordinates),
        born_contact_count=len(mixed.born_contact_steps), independent_foundation_count=len(mixed.born_foundation_steps),
        original_accepted_interval_count=len(previous.receipts), accepted_interval_preserved=True,
        birth_declaration_sha256=case["fixed_declaration_sha256"], declaration_fixed_before_force_result=True,
        birth_activation_work_j=metrics["born_activation_work_j"], diagnostics=metrics,
        force_candidate_omega_rad_per_myr=[[owner, list(vector)] for owner, vector in omega],
        force_candidate_dt_s=inputs["dt_s"],
        force_packet_sha256=digest(out / "actual_mixed_network.json"),
        same_engine_no_born_packet_sha256=digest(out / "same_engine_without_born_contacts.json"),
        same_engine_born_force_feedback_sha256=digest(out / "same_engine_born_force_feedback.json"),
        unchanged_old_coordinate_prefix_changed_by_born_forces=True,
        constitutive_packet_sha256=digest(out / "constitutive_packet.json"),
        pair_inventory_sha256=None if inventory_result is None else digest(out / "mixed_pair_inventory.json"),
        pair_classification_counts=None if inventory_result is None else inventory_result["inventory"]["classification_counts"],
        byte_pairs=[dict(first="accepted_motion_before.json", second="accepted_motion_after.json",
            bytes=(out / "accepted_motion_before.json").stat().st_size, sha256=digest(out / "accepted_motion_before.json"))],
        continuum_quadrature_error_certified=False, complete_all_pairs_temporal_coverage=False,
        unresolved_positive_vertex_front_count=2, whole_native_interval_admitted=False,
        baseline_witness_source="immutable genuine cold-only second candidate; four witnesses and two vertex fronts belong to that velocity",
        fresh_mixed_geometry_evidence="mixed_pair_inventory.json only; baseline witnesses are not reused as its admission",
        point_1_complete=False, parameters_automatically_changed=False,
        scope=case["scope"])
    write(out / "report.json", report, "report_sha256")
    print(encoded({key: report[key] for key in ("status", "seconds", "mixed_coordinate_count",
        "born_contact_count", "independent_foundation_count", "pair_classification_counts")}).decode(), flush=True)
    return dict(report=report, case=case, mixed=mixed, inventory=inventory_result)
