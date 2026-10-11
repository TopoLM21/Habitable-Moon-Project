"""Replay the genuine candidate and transport supported material ports once.

The explicit receiver-attached configuration law retains every original local
state, interface measure, beam coordinate and work ledger. The eight supported
incoming ports are evaluated at the refused candidate endpoint; neither force
equilibrium nor full native bank/clamp support nor an accepted step is claimed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
PHYSICAL = ROOT / "analysis/geometric_native_physical_binding_validation"
MATERIAL = ROOT / "analysis/geometric_native_first_entry_validation"
SOURCE = MATERIAL / "control_release_02"
LAW_PROVENANCE = (
    "Explicit receiver-attached complete loaded lumped configuration transport; "
    "original interface measure and beam Ritz basis retained"
)
sys.path.insert(0, str(ROOT))


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def value_sha(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def write(path, value, *, seal=None):
    value = deepcopy(value)
    if seal is not None:
        value[seal] = value_sha(value)
    with Path(path).open("xb") as stream:
        stream.write(encoded(value) + b"\n")
    return value


def sealed(path, key):
    value = read(path)
    expected = value.pop(key)
    if expected != value_sha(value):
        raise ValueError("Source selfseal mismatch: " + str(path))
    return value


def rational(value):
    value = value if type(value) is Fraction else Fraction.from_float(value)
    return [value.numerator, value.denominator]


def sources():
    return {str(path.relative_to(ROOT)): digest(path)
        for directory in ("tectonics", "tests", "scripts", "analysis")
        for path in sorted((ROOT / directory).rglob("*.py"))}


def guard_sources():
    current = sources()
    starting = read(BASE / "starting_sources.json")
    if (len(starting) != 664
            or starting != read(PHYSICAL / "release_gates_01/sources_after.json")
            or starting != read(PHYSICAL / "control_release_01/sources_after.json")):
        raise ValueError("Handover must start at the exact authenticated 664-source release")
    changed = [name for name, expected in starting.items() if current.get(name) != expected]
    if changed:
        raise ValueError("A frozen original source changed: " + repr(changed))
    return current, starting


def verify_previous_sources():
    releases = ((PHYSICAL / "release_validation_01", 56, "physical"),
                (MATERIAL / "release_validation_02", 83, "material"))
    for directory, count, label in releases:
        summary = sealed(directory / "validation_summary.json", "summary_sha256")
        manifest = sealed(directory / "artifact_manifest.json", "manifest_sha256")
        if (summary["status"] != "PASS" or manifest["count"] != count
                or len(manifest["artifacts"]) != count
                or summary["artifact_manifest"]["sha256"] != digest(directory / "artifact_manifest.json")):
            raise ValueError("Previous authenticated release differs: " + label)
        for name, record in manifest["artifacts"].items():
            path = ROOT / name
            if (not path.is_file() or digest(path) != record["sha256"]
                    or path.stat().st_size != record["bytes"]):
                raise ValueError("Previous artifact differs: " + name)
        keys = ("fresh_scoped_gate", "actual_control", "independent_audit") if label == "physical" else (
            "release_gate", "actual_control", "independent_audit")
        for key in keys:
            reference = summary[key]
            if digest(ROOT / reference["path"]) != reference["sha256"]:
                raise ValueError("Previous evidence reference differs: " + key)
    physical = sealed(PHYSICAL / "release_validation_01/validation_summary.json", "summary_sha256")
    if (physical["source_count"] != 664
            or physical["fresh_scoped_gate"]["tests_passed"] != 253
            or physical["prior_authoritative_validation"]["tests_passed"] != 770
            or physical["prior_authoritative_validation"]["tests_reexecuted"] is not False):
        raise ValueError("Previous source/test ancestry changed")
    source_paths = (
        PHYSICAL / "release_validation_01/validation_summary.json",
        PHYSICAL / "release_validation_01/artifact_manifest.json",
        PHYSICAL / "release_gates_01/receipt.json",
        PHYSICAL / "control_release_01/report.json",
        PHYSICAL / "control_release_01/independent_binding_receipt.json",
        PHYSICAL / "control_release_01/original_history.json",
        MATERIAL / "release_validation_02/validation_summary.json",
        MATERIAL / "release_validation_02/artifact_manifest.json",
        SOURCE / "original_force/initial.json",
        SOURCE / "original_force/native_source.json",
        SOURCE / "original_force/force_candidate.json",
        SOURCE / "material_state.json", SOURCE / "point_certificates.json",
        BASE / "starting_sources.json",
    )
    return dict(format="native-material-port-handover-source-validation-1", status="PASS",
        original_source_count=664, prior_physical_artifact_count=56,
        prior_material_artifact_count=83, prior_physical_tests_passed=253,
        prior_material_tests_passed=770, prior_tests_reexecuted=False,
        source_artifacts={str(path.relative_to(ROOT)):dict(sha256=digest(path),
            bytes=path.stat().st_size) for path in source_paths})


def get_physical_case():
    """The established actual-source fixture, without a pytest dependency."""
    from tectonics.geometric_native_first_entry_material import native_first_entry_material_from_dict
    from tectonics.geometric_native_mantle_loading import native_spatial_mantle_load_from_dict
    from tectonics.geometric_native_physical_history import (
        _candidate_payload, build_native_physical_history_binding,
    )
    from tectonics.geometric_native_underthrust_motion import (
        UnresolvedNativeUnderthrustMotion, advance_native_underthrust_motion,
    )
    from tectonics.geometric_native_underthrust_motion_io import load_native_underthrust_motion_checkpoint
    from tectonics.mesh import build_icosphere

    paths = tuple(SOURCE / name for name in ("material_state.json", "point_certificates.json",
        "original_force/initial.json", "original_force/after_refusal.json",
        "original_force/native_source.json", "original_force/force_candidate.json"))
    original_files = {path: path.read_bytes() for path in paths}
    mesh = build_icosphere(0)
    print("Loading the typed original checkpoint and its independent physical source", flush=True)
    origin, before, provenance = load_native_underthrust_motion_checkpoint(
        SOURCE / "original_force/initial.json", mesh=mesh)
    current_load = native_spatial_mantle_load_from_dict(json.loads(
        original_files[SOURCE / "original_force/native_source.json"]), before.current_surface, mesh=mesh)
    saved_candidate = json.loads(original_files[SOURCE / "original_force/force_candidate.json"])
    print("Re-solving all fields of the genuine refused physical candidate", flush=True)
    try:
        advance_native_underthrust_motion(mesh, origin, before, current_load,
            saved_candidate["dt_s"], provenance=saved_candidate["provenance"])
    except UnresolvedNativeUnderthrustMotion as refusal:
        candidate = refusal.candidate
        overlaps = refusal.overlaps
        candidate_raw = _candidate_payload(candidate, overlaps)
        if encoded(candidate_raw) != encoded(saved_candidate) or len(candidate_raw) != 14:
            raise AssertionError("Fresh candidate differs from an original physical field")
    else:
        raise AssertionError("The original physical candidate unexpectedly became accepted")
    print("Replaying the complete once-only material state and all original histories", flush=True)
    initial_raw = json.loads(original_files[SOURCE / "original_force/initial.json"])
    material = native_first_entry_material_from_dict(json.loads(
        original_files[SOURCE / "material_state.json"]),
        expected_original_mechanics_payload=initial_raw["native_underthrust_motion"]["initial_mechanics"])
    checkpoint_sha = hashlib.sha256(original_files[SOURCE / "original_force/initial.json"]).hexdigest()
    history = build_native_physical_history_binding(mesh, origin, before, candidate,
        material, current_load=current_load, original_checkpoint_sha256=checkpoint_sha)
    if any(path.read_bytes() != expected for path, expected in original_files.items()):
        raise AssertionError("Actual-source replay changed an authoritative input file")
    return dict(mesh=mesh, origin=origin, before=before, provenance=provenance,
        candidate=candidate, overlaps=overlaps, candidate_raw=candidate_raw, current_load=current_load,
        material_state=material, checkpoint_sha256=checkpoint_sha, history=history,
        original_files=original_files)


def transport_ledger(history, bundle):
    nodes = {node.point_id:node for node in history.quadrature_histories}
    ports = {port.point_id:port for port in bundle.ports}
    retained = tuple(bundle.retained_point_ids)
    if (len(nodes) != 16 or len(ports) != 8 or len(retained) != 8
            or len(set(retained)) != 8 or set(ports) & set(retained)
            or set(ports) | set(retained) != set(nodes)):
        raise AssertionError("Every original quadrature history must be represented exactly once")
    bindings = {row.trace_id:row for row in history.material_state.trace_bindings}
    eligible_pairs = {(bindings[port.trace_id].donor_fragment_id,
                       bindings[port.trace_id].receiver_fragment_id) for port in bundle.ports}
    if eligible_pairs != {("octant:6", "octant:2"), ("octant:7", "octant:3")}:
        raise AssertionError("The supported original handover pairs changed")
    rows, aggregates = [], {}
    positive_cumulative = positive_pressure = nonzero_plastic = 0
    total_area = transported_area = retained_area = Fraction(0)
    for identity, node in nodes.items():
        old = node.unaccepted_candidate_state
        new = ports[identity].transport_state.local_state if identity in ports else old
        if new is not old or encoded(asdict(new)) != encoded(asdict(old)):
            raise AssertionError("Handover altered an original candidate LocalState")
        if old.area_m2 != node.original_area_m2 or old.cumulative_slip_m <= 0:
            raise AssertionError("Original positive cumulative history or interface measure changed")
        if old.plastic_slip_m == 0 or old.traction_normal_pa >= 0:
            raise AssertionError("Original loaded plastic history or compressive pressure disappeared")
        positive_cumulative += 1
        nonzero_plastic += 1
        positive_pressure += 1
        area = Fraction.from_float(node.original_area_m2)
        total_area += area
        if identity in ports:
            transported_area += area
        else:
            retained_area += area
        raw = asdict(old)
        for name, value in raw.items():
            if name.endswith("_j"):
                aggregates[name] = aggregates.get(name, Fraction(0)) + Fraction.from_float(value)
        binding = bindings[node.trace_id]
        rows.append(dict(point_id=identity, trace_id=node.trace_id,
            disposition="transported_once" if identity in ports else "retained_once",
            source_fragment_pair=[binding.donor_fragment_id, binding.receiver_fragment_id],
            original_area_m2=rational(area), source_local_state=raw,
            transported_local_state=asdict(new), local_state_fields_identical=True))
    if total_area != transported_area + retained_area:
        raise AssertionError("Interface measure was duplicated or lost")
    beam = dict(parameters=[asdict(row) for row in history.initial_mechanics.bending_parameters],
        candidate_transverse_displacement_m=history.actual_candidate.transverse_displacement_m,
        original_stiffness=history.original_reference.stiffness)
    return dict(format="native-material-port-handover-once-only-ledger-1",
        original_node_count=16, transported_port_count=8, retained_port_count=8,
        eligible_source_fragment_pairs=[list(row) for row in sorted(eligible_pairs)],
        quadrature_histories=rows, total_original_area_m2=rational(total_area),
        transported_area_m2=rational(transported_area), retained_area_m2=rational(retained_area),
        original_local_energy_work_totals_j={name:rational(value) for name, value in aggregates.items()},
        local_energy_work_counted_once=True, source_work_before=history.before_work_ledgers,
        unaccepted_candidate_work=history.unaccepted_candidate_work_ledgers,
        beam_configuration_before=beam, beam_configuration_after=deepcopy(beam),
        unchanged_global_beam_dofs_and_stiffness=True,
        positive_cumulative_slip_count=positive_cumulative,
        nonzero_signed_plastic_slip_count=nonzero_plastic,
        positive_compressive_pressure_count=positive_pressure,
        signed_plastic_memory_preserved=True, transfer_external_work_j=0,
        transfer_dissipation_j=0, accepted_loaded_native_intervals=0,
        accepted_source_work_j=0)


def rehash_declared_seals(value):
    """Make mutation receipts distinguish source replay from checksum checks."""
    if type(value) is list:
        for item in value:
            rehash_declared_seals(item)
    elif type(value) is dict:
        for item in value.values():
            rehash_declared_seals(item)
        mapping = {
            "native-candidate-material-port-handover-1":"handover_sha256",
            "native-original-physical-history-binding-1":"binding_sha256",
            "native-contact-reference-transport-1":"state_sha256",
            "native-source-beam-material-chart-1":"chart_sha256",
            "native-original-physical-pair-tip-time-maps-1":"maps_sha256",
        }
        key = mapping.get(value.get("format"))
        if value.get("format") in ("native-original-physical-pair-tip-time-maps-1",
                                   "native-source-beam-material-chart-1"):
            value["source_sha256"] = value_sha(value["source"])
        if key in value:
            value[key] = value_sha({name:item for name, item in value.items() if name != key})


def mutation_refusals(raw, *, history, history_raw, law):
    from tectonics.geometric_native_material_port_handover import (
        certify_native_candidate_material_port_handover,
        native_candidate_material_port_handover_from_dict,
    )

    def shifted(bounds, amount):
        for key in ("lo", "hi"):
            bounds[key] = rational(Fraction(*bounds[key]) + amount)

    def tiny_axis(body):
        records = body["source_history"]["actual_candidate"]["omega_rad_per_myr"]
        row = next(row for row in records if row[0] == 1)
        if row[1][0] == 0:
            raise AssertionError("Expected the authentic tiny transverse component")
        row[1][0] = 0.0

    def source_memory(body):
        body["source_history"]["actual_candidate"]["states"][0][1]["cumulative_slip_m"] += 0.001

    def measure(body):
        body["ports"][0]["transport_state"]["original_area_m2"] *= 2

    def chart_width(body):
        chart_source = body["ports"][0]["target_chart"]["source"]
        chart_source["width_m"] = rational(Fraction(*chart_source["width_m"]) + 1)

    def chart_amplitude(body):
        shifted(body["ports"][0]["target_chart"]["source"]["tip_displacement_m"], Fraction(1, 1000))

    def chart_phi(body):
        shifted(body["ports"][0]["target_chart"]["shape"], Fraction(1, 64))

    def receiver_frame(body):
        body["ports"][0]["normal_operator"]["receiver_frame_couple_included"] = False

    def operator_couple(body):
        shifted(body["ports"][0]["normal_operator"]["over_angular"][0], Fraction(1, 64))

    def duplicate_port(body):
        body["ports"].append(deepcopy(body["ports"][0]))

    def duplicate_retained(body):
        body["retained_point_ids"].append(body["retained_point_ids"][0])

    def plastic(body):
        body["ports"][0]["transport_state"]["original_local_state"]["plastic_slip_m"] = 0.0

    def work(body):
        body["ports"][0]["transport_state"]["original_local_state"]["cumulative_external_work_j"] = 0.0

    def provenance(body):
        body["transport_law"]["provenance"] += " changed"

    cases = (("tiny_owner_axis_removed", "source_history.actual_candidate.omega_rad_per_myr", tiny_axis),
        ("source_candidate_memory_changed", "source_history.actual_candidate.states", source_memory),
        ("original_measure_doubled", "ports[0].transport_state.original_area_m2", measure),
        ("source_beam_width_changed", "ports[0].target_chart.source.width_m", chart_width),
        ("source_beam_amplitude_changed", "ports[0].target_chart.source.tip_displacement_m", chart_amplitude),
        ("source_beam_shape_changed", "ports[0].target_chart.shape", chart_phi),
        ("receiver_frame_couple_removed", "ports[0].normal_operator.receiver_frame_couple_included", receiver_frame),
        ("receiver_angular_couple_changed", "ports[0].normal_operator.over_angular", operator_couple),
        ("duplicate_port", "ports", duplicate_port),
        ("duplicate_retained_history", "retained_point_ids", duplicate_retained),
        ("candidate_plastic_reset", "ports[0].transport_state.original_local_state.plastic_slip_m", plastic),
        ("candidate_work_reset", "ports[0].transport_state.original_local_state.cumulative_external_work_j", work),
        ("transport_law_provenance_changed", "transport_law.provenance", provenance))
    results = []
    for label, path, mutate in cases:
        forged = deepcopy(raw)
        mutate(forged)
        rehash_declared_seals(forged)
        try:
            native_candidate_material_port_handover_from_dict(forged,
                expected_history=history, expected_source_history_dto=history_raw, expected_law=law)
        except (ValueError, TypeError) as error:
            results.append(dict(label=label, changed_path=path, rejected=True,
                valid_rehashed_handover_checksum=True, exception_type=type(error).__name__,
                message=str(error), mutated_dto_sha256=value_sha(forged)))
        else:
            raise AssertionError("Rehashed material-port mutation was accepted: " + label)
    forged_source = deepcopy(history_raw)
    forged_source["actual_candidate"]["states"][0][1]["cumulative_slip_m"] += 0.001
    rehash_declared_seals(forged_source)
    try:
        certify_native_candidate_material_port_handover(history, law=law,
            expected_source_history_dto=forged_source)
    except (ValueError, TypeError) as error:
        results.append(dict(label="independent_complete_source_changed",
            changed_path="expected_source_history_dto.actual_candidate.states", rejected=True,
            valid_rehashed_source_checksum=True, exception_type=type(error).__name__,
            message=str(error), mutated_dto_sha256=value_sha(forged_source)))
    else:
        raise AssertionError("A forged independently supplied full source was accepted")
    return results


def run_control(out):
    from tectonics.geometric_native_contact_reference_transport import (
        ContactHistoryTransportLaw, TRANSPORT_LAW_MODEL,
        evaluate_contact_reference_free_anchors,
    )
    from tectonics.geometric_native_material_port_handover import (
        certify_native_candidate_material_port_handover,
        native_candidate_material_port_handover_to_dict,
        native_candidate_material_port_handover_from_dict,
    )
    from tectonics.geometric_native_physical_history import (
        _candidate_payload, native_physical_history_binding_to_dict,
    )
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
        raise AssertionError("The full genuine source history changed from its authoritative release")
    checkpoint_before = save_native_underthrust_motion_checkpoint(out / "source_checkpoint_before.json",
        case["mesh"], case["origin"], case["before"], provenance=case["provenance"])
    if checkpoint_before.read_bytes() != case["original_files"][SOURCE / "original_force/initial.json"]:
        raise AssertionError("The typed prepared checkpoint did not preserve its original bytes")
    write(out / "force_candidate_before.json", case["candidate_raw"])
    law = ContactHistoryTransportLaw(model=TRANSPORT_LAW_MODEL, provenance=LAW_PROVENANCE)
    write(out / "declared_handover_law.json", asdict(law))
    print("Replacing the eight proved incoming candidate attachments exactly once", flush=True)
    bundle = certify_native_candidate_material_port_handover(history, law=law,
        expected_source_history_dto=source_raw, precision=128)
    handover_raw = native_candidate_material_port_handover_to_dict(bundle)
    write(out / "candidate_handover.json", handover_raw)
    print("Replaying the complete handover against the original physical source and law", flush=True)
    replay = native_candidate_material_port_handover_from_dict(deepcopy(handover_raw),
        expected_history=history, expected_source_history_dto=source_raw, expected_law=law)
    write(out / "candidate_handover_replayed.json", native_candidate_material_port_handover_to_dict(replay))
    ledger = transport_ledger(history, bundle)
    write(out / "quadrature_transport_ledger.json", ledger, seal="ledger_sha256")
    observations = []
    for port in bundle.ports:
        state = port.transport_state
        anchors = evaluate_contact_reference_free_anchors(state, state.reference_geometry,
            geometry_context=json.loads(state.reference_geometry_context_json))
        expected = (Fraction.from_float(state.local_state.gap_free_m),
                    Fraction.from_float(state.local_state.jump_free_m))
        if any(value.lo != old or value.hi != old for value, old in zip(anchors, expected)):
            raise AssertionError("Identical handover expressions did not cancel exactly")
        observations.append(dict(point_id=port.point_id,
            source_free_anchor=[rational(value) for value in expected],
            anchors_at_transfer=[dict(lo=rational(value.lo), hi=rational(value.hi)) for value in anchors],
            source_reference_geometry_context=json.loads(state.reference_geometry_context_json),
            correlated_geometry_cancelled_symbolically=True,
            time_fraction=[1, 1], source_epoch="unaccepted_candidate"))
    write(out / "port_reference_observations.json", dict(
        format="native-material-port-transfer-reference-observations-1", count=len(observations),
        observations=observations), seal="receipt_sha256")
    print("Rejecting rehashed source, chart, local-state, measure and frame mutations", flush=True)
    refusals = mutation_refusals(handover_raw, history=history, history_raw=source_raw, law=law)
    write(out / "mutation_refusals.json", dict(format="native-material-port-handover-mutation-refusals-1",
        count=len(refusals), mutations=refusals), seal="receipt_sha256")
    write(out / "source_history_after.json", native_physical_history_binding_to_dict(history, mesh=case["mesh"]))
    save_native_underthrust_motion_checkpoint(out / "source_checkpoint_after.json",
        case["mesh"], case["origin"], case["before"], provenance=case["provenance"])
    write(out / "force_candidate_after.json", _candidate_payload(case["candidate"], case["overlaps"]))
    pairs = (("source_history_before.json", "source_history_after.json"),
        ("source_checkpoint_before.json", "source_checkpoint_after.json"),
        ("force_candidate_before.json", "force_candidate_after.json"),
        ("candidate_handover.json", "candidate_handover_replayed.json"))
    for first, second in pairs:
        if (out / first).read_bytes() != (out / second).read_bytes():
            raise AssertionError("Unchanged physical source or replay bytes differ: " + first)
    if (case["before"].receipts or case["before"].elapsed_time_s != 0
            or case["before"].cumulative_source_work_j != 0
            or case["before"].current_inventory.records):
        raise AssertionError("A handover changed the accepted native trajectory or work")
    if any(path.read_bytes() != expected for path, expected in case["original_files"].items()):
        raise AssertionError("A handover changed an authoritative source artifact")
    after_sources, baseline_after = guard_sources()
    if before_sources != after_sources or baseline != baseline_after:
        raise AssertionError("A Python source changed while computing the handover control")
    write(out / "sources_after.json", after_sources)
    report = dict(format="native-candidate-material-port-handover-control-1", status="PASS",
        seconds=time.perf_counter() - started, original_source_count=664,
        source_count=len(before_sources), added_python_sources=sorted(set(before_sources) - set(baseline)),
        existing_sources_changed=[], source_before_after_unchanged=True,
        original_typed_checkpoint_loaded=True, original_force_re_solved=True,
        history_builder_original_force_re_solved=True, prior_material_state_replayed=True,
        original_candidate_all_fields_identical=True, original_candidate_field_count=14,
        source_epoch="unaccepted_candidate", time_fraction=[1, 1],
        original_node_count=16, transported_port_count=8, retained_port_count=8,
        transported_source_trace_count=2, original_local_states_preserved=True,
        positive_cumulative_slip_count=ledger["positive_cumulative_slip_count"],
        nonzero_signed_plastic_slip_count=ledger["nonzero_signed_plastic_slip_count"],
        positive_compressive_pressure_count=ledger["positive_compressive_pressure_count"],
        source_history_bytes_unchanged=True, source_checkpoint_bytes_unchanged=True,
        source_candidate_bytes_unchanged=True, source_quad_measure_work_counted_once=True,
        port_reference_observation_count=len(observations),
        correlated_geometry_cancelled_symbolically=True,
        unchanged_global_beam_dofs_and_stiffness=True, mutation_refusal_count=len(refusals),
        byte_pairs=[list(pair) for pair in pairs],
        law=asdict(law), transported_candidate_is_force_equilibrium=False,
        native_full_bank_and_clamp_support_certified=False,
        finite_measure_assigned_to_first_entry_point=False,
        candidate_mechanics_adopted=False, physical_pair_admission_authorized=False,
        global_native_inventory_updated=False, transfer_external_work_j=0,
        transfer_dissipation_j=0, accepted_loaded_native_intervals=0, accepted_source_work_j=0,
        artifacts_sha256={str(path.relative_to(out)):digest(path)
            for path in sorted(out.rglob("*")) if path.is_file()})
    saved = write(out / "report.json", report, seal="report_sha256")
    print(json.dumps({name:saved[name] for name in ("status", "seconds", "source_count",
        "transported_port_count", "retained_port_count", "mutation_refusal_count",
        "accepted_loaded_native_intervals")}), flush=True)
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run_control(args.out.resolve())


if __name__ == "__main__":
    main()
