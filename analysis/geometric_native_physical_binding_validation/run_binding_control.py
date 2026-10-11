"""Authenticate the original physical history and incoming point binding.

The original force is solved after a typed checkpoint load.  Its geometric
refusal remains authoritative; all physical-history queries are diagnostics.
The prior once-only material state is replayed independently and never changes
the accepted native trajectory, material inventory, spring memory or work.
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
PRIOR = ROOT / "analysis/geometric_native_first_entry_validation"
PRIOR_CONTROL = PRIOR / "control_release_02"
QUERY_TIMES = (Fraction(0), Fraction(3, 8), Fraction(1))
sys.path.insert(0, str(ROOT))


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value, *, seal=None):
    value = deepcopy(value)
    if seal is not None:
        value[seal] = hashlib.sha256(encoded(value)).hexdigest()
    with Path(path).open("xb") as stream:
        stream.write(encoded(value) + b"\n")
    return value


def sealed(path, key):
    value = read(path)
    expected = value.pop(key)
    if expected != hashlib.sha256(encoded(value)).hexdigest():
        raise ValueError("Inconsistent source seal: " + str(path))
    return value


def sources():
    return {str(path.relative_to(ROOT)): digest(path)
        for directory in ("tectonics", "tests", "scripts", "analysis")
        for path in sorted((ROOT / directory).rglob("*.py"))}


def guard_sources():
    before = sources()
    starting = read(BASE / "starting_sources.json")
    if len(starting) != 654:
        raise ValueError("Physical binding requires the frozen 654-source baseline")
    changed = [name for name, expected in starting.items()
        if before.get(name) != expected]
    if changed:
        raise ValueError("Original Python source changed: " + repr(changed))
    return before, starting


def verify_prior_sources():
    summary_path = PRIOR / "release_validation_02/validation_summary.json"
    manifest_path = PRIOR / "release_validation_02/artifact_manifest.json"
    summary = sealed(summary_path, "summary_sha256")
    manifest = sealed(manifest_path, "manifest_sha256")
    if summary["status"] != "PASS" or manifest["count"] != 83:
        raise ValueError("Expected the authoritative 83-artifact first-entry release")
    if summary["artifact_manifest"]["sha256"] != digest(manifest_path):
        raise ValueError("First-entry release points to a different artifact manifest")
    for name, record in manifest["artifacts"].items():
        path = ROOT / name
        if (not path.is_file() or digest(path) != record["sha256"]
                or path.stat().st_size != record["bytes"]):
            raise ValueError("Prior authoritative artifact differs: " + name)
    for key in ("release_gate", "actual_control", "independent_audit"):
        reference = summary[key]
        if digest(ROOT / reference["path"]) != reference["sha256"]:
            raise ValueError("Prior first-entry evidence reference differs: " + key)
    if summary["source_count"] != 654 or summary["starting_source_count"] != 643:
        raise ValueError("Unexpected first-entry source ancestry")
    gate = read(ROOT / summary["release_gate"]["path"])
    report = sealed(PRIOR_CONTROL / "report.json", "report_sha256")
    audit = sealed(PRIOR_CONTROL / "independent_first_entry_receipt.json", "receipt_sha256")
    if (any(item["status"] != "PASS" for item in (gate, report, audit))
            or gate["counts"]["passed"] != 770):
        raise ValueError("Prior first-entry release is not an authenticated PASS")
    baseline = read(BASE / "starting_sources.json")
    if baseline != read(PRIOR / "release_gates_02/sources_after.json"):
        raise ValueError("Physical binding does not start at the authoritative 654 sources")
    source_paths = (summary_path, manifest_path,
        PRIOR / "release_gates_02/receipt.json",
        PRIOR_CONTROL / "report.json",
        PRIOR_CONTROL / "independent_first_entry_receipt.json",
        PRIOR_CONTROL / "original_force/initial.json",
        PRIOR_CONTROL / "original_force/force_candidate.json",
        PRIOR_CONTROL / "original_force/constitutive_time_support.json",
        PRIOR_CONTROL / "material_state.json",
        PRIOR_CONTROL / "point_certificates.json",
        PRIOR_CONTROL / "material_point_queries.json", BASE / "starting_sources.json")
    return dict(format="native-physical-binding-source-validation-1", status="PASS",
        prior_artifact_count=83, prior_tests_passed=770, original_source_count=654,
        prior_starting_source_count=643,
        source_artifacts={str(path.relative_to(ROOT)):dict(
            sha256=digest(path), bytes=path.stat().st_size) for path in source_paths})


def run_fresh_force(out):
    from analysis.geometric_underthrust_validation.underthrust_case import (
        DT_S, SOURCE_INTERVALS,
    )
    from analysis.geometric_native_underthrust_validation.run_native_underthrust_control import state_snapshot
    from tectonics.geometric_native_contact_time_support import (
        ADMISSIBLE, MODEL, assess_native_contact_time_support,
        build_native_contact_time_path, native_contact_time_support_to_dict,
    )
    from tectonics.geometric_native_mantle_loading import (
        assemble_native_physical_spatial_mantle_load, native_spatial_mantle_load_to_dict,
    )
    from tectonics.geometric_native_polygons import native_polygon_to_dict
    from tectonics.geometric_native_underthrust_motion import (
        UnresolvedNativeUnderthrustMotion, advance_native_underthrust_motion,
    )
    from tectonics.geometric_native_underthrust_motion_io import (
        load_native_underthrust_motion_checkpoint, save_native_underthrust_motion_checkpoint,
    )
    from tectonics.mesh import build_icosphere

    out.mkdir()
    source = PRIOR_CONTROL / "original_force"
    mesh = build_icosphere(0)
    origin, state, provenance = load_native_underthrust_motion_checkpoint(
        source / "initial.json", mesh=mesh)
    initial = save_native_underthrust_motion_checkpoint(out / "initial.json", mesh,
        origin, state, provenance=provenance)
    if initial.read_bytes() != (source / "initial.json").read_bytes():
        raise ValueError("Typed original checkpoint did not reproduce its exact source bytes")
    before = state_snapshot(state)
    write(out / "declared_source.json", asdict(SOURCE_INTERVALS[0]))
    print("Solving the original force from its typed authoritative checkpoint", flush=True)
    load = assemble_native_physical_spatial_mantle_load(mesh, state.current_surface,
        SOURCE_INTERVALS[0])
    write(out / "native_source.json", native_spatial_mantle_load_to_dict(state.current_surface, load))
    original = read(source / "force_candidate.json")
    try:
        advance_native_underthrust_motion(mesh, origin, state, load, DT_S,
            provenance=original["provenance"])
    except UnresolvedNativeUnderthrustMotion as refusal:
        candidate, overlaps = refusal.candidate, refusal.overlaps
    else:
        raise AssertionError("Original force was admitted instead of preserving its geometric refusal")
    raw = dict(before_surface_sha256=candidate.before_surface_sha256,
        current_load_sha256=candidate.current_load_sha256, dt_s=candidate.dt_s,
        provenance=candidate.provenance,
        rotation_increment_radians=candidate.rotation_increment_radians,
        angular_velocity_rad_s=candidate.angular_velocity_rad_s,
        omega_rad_per_myr=candidate.omega_rad_per_myr,
        states=[(identity, asdict(local)) for identity, local in candidate.states],
        transverse_displacement_m=candidate.transverse_displacement_m,
        spatial_step=asdict(candidate.spatial_step),
        diagnostics=dict(candidate.candidate_diagnostics), admission=candidate.admission_audit,
        physical_points=candidate.points,
        unresolved=[dict(fragment_a=item.fragment_a, fragment_b=item.fragment_b,
            plate_a=item.plate_a, plate_b=item.plate_b, area_km2=item.area_km2,
            reason=item.reason, geometry=native_polygon_to_dict(item.geometry)) for item in overlaps])
    if encoded(raw) != encoded(original):
        raise AssertionError("The freshly solved candidate changed an original field")
    probe_path = ROOT / "analysis/geometric_native_underthrust_validation/force_control_release_02/force_candidate_refusal_probe.json"
    probe = read(probe_path)
    if (any(encoded(raw[key]) != encoded(value) for key, value in probe["candidate"].items())
            or encoded(raw["admission"]) != encoded(probe["admission"])):
        raise AssertionError("The fresh candidate changed the original independent refusal probe")
    write(out / "force_candidate.json", raw)
    print("Checking all original constitutive trace and time domains", flush=True)
    path = build_native_contact_time_path(state.initial_mechanics, candidate.spatial_step,
        state.free_anchor_fields, path_model=MODEL)
    certificates = [assess_native_contact_time_support(path, row.trace_id,
        0, row.extent_radians, 0, 1, precision=80)
        for row in candidate.spatial_step.certificates]
    if len(certificates) != 4 or any(row.status != ADMISSIBLE for row in certificates):
        raise AssertionError("Original constitutive path was not certified")
    support = dict(path_sha256=path.path_sha256, path_model=MODEL,
        coordinate_model="original_retained_angle_scaled_from_exact_native_axis_quarter",
        certificates=[native_contact_time_support_to_dict(row) for row in certificates],
        candidate_sha256=hashlib.sha256(encoded(raw)).hexdigest(),
        geometry_permission=False, candidate_memory_adopted=False,
        scope="four_original_inviscid_pairs_whole_material_traces_whole_time_interval")
    if encoded(support) != encoded(read(source / "constitutive_time_support.json")):
        raise AssertionError("Original constitutive support changed")
    write(out / "constitutive_time_support.json", support)
    after = save_native_underthrust_motion_checkpoint(out / "after_refusal.json", mesh,
        origin, state, provenance=provenance)
    if (initial.read_bytes() != after.read_bytes() or state_snapshot(state) != before
            or state.receipts or state.current_inventory.records
            or state.elapsed_time_s != 0 or state.cumulative_source_work_j != 0):
        raise AssertionError("The refused force changed original memory, material or work")
    report = dict(status="PASS", original_typed_checkpoint_loaded=True,
        original_force_re_solved=True, original_candidate_all_fields_identical=True,
        original_candidate_field_count=len(original),
        source_initial_sha256=digest(source / "initial.json"),
        prepared_checkpoint_byte_identical=True,
        accepted_loaded_native_intervals=0, accepted_source_work_j=0,
        refusal_reasons=sorted({item.reason for item in overlaps}),
        source_probe_path=str(probe_path.relative_to(ROOT)), source_probe_sha256=digest(probe_path))
    write(out / "report.json", report, seal="report_sha256")
    return dict(mesh=mesh, origin=origin, state=state, candidate=candidate,
        load=load, contact_time_path=path, candidate_raw=raw,
        support_raw=support, report=report)


def run_control(out):
    from tectonics.geometric_native_first_entry_material import (
        native_first_entry_material_from_dict,
        native_first_entry_material_to_dict,
        native_first_entry_material_point_chronology_from_dict,
        native_first_entry_material_point_chronology_to_dict,
    )
    from tectonics.geometric_native_first_entry_physical_binding import (
        certify_native_first_entry_physical_binding,
        native_first_entry_physical_binding_to_dict,
        native_first_entry_physical_binding_from_dict,
    )
    from tectonics.geometric_native_physical_history import (
        build_native_physical_history_binding,
        native_physical_history_binding_to_dict,
        native_physical_history_binding_from_dict,
        query_native_original_pair_history,
        native_original_pair_history_to_dict,
        native_original_pair_history_from_dict,
    )
    from tectonics.geometric_native_pose import FractionInterval as I

    started = time.perf_counter()
    before, starting = guard_sources()
    authority = verify_prior_sources()
    out.mkdir(parents=True, exist_ok=False)
    write(out / "sources_before.json", before)
    write(out / "source_validation_receipt.json", authority, seal="receipt_sha256")
    force = run_fresh_force(out / "original_force")
    initial = read(out / "original_force/initial.json")
    mechanics = initial["native_underthrust_motion"]["initial_mechanics"]
    print("Replaying the authoritative once-only material state", flush=True)
    material_raw = read(PRIOR_CONTROL / "material_state.json")
    material = native_first_entry_material_from_dict(deepcopy(material_raw),
        expected_original_mechanics_payload=mechanics)
    if encoded(native_first_entry_material_to_dict(material)) != encoded(material_raw):
        raise AssertionError("The prior once-only material state did not replay exactly")
    history_args = (force["mesh"], force["origin"], force["state"], force["candidate"], material)
    history_kwargs = dict(current_load=force["load"],
        original_checkpoint_sha256=digest(out / "original_force/initial.json"),
        contact_time_path=force["contact_time_path"])
    print("Binding every original quadrature history to an independent force replay", flush=True)
    history = build_native_physical_history_binding(*history_args, **history_kwargs)
    history_raw = native_physical_history_binding_to_dict(history, mesh=force["mesh"])
    write(out / "original_history.json", history_raw)
    print("Replaying the complete physical-history DTO from its external sources", flush=True)
    replay = native_physical_history_binding_from_dict(deepcopy(history_raw),
        *history_args, **history_kwargs)
    write(out / "original_history_replayed.json",
        native_physical_history_binding_to_dict(replay, mesh=force["mesh"]))
    queries, replayed_queries = [], []
    indices = {}
    for node in history.quadrature_histories:
        index = node.original_point.node_index
        indices[node.trace_id] = indices.get(node.trace_id, 0) + 1
        for instant in QUERY_TIMES:
            query = native_original_pair_history_to_dict(
                query_native_original_pair_history(history, node.point_id, I(instant)),
                binding=history)
            replayed = native_original_pair_history_to_dict(
                native_original_pair_history_from_dict(deepcopy(query), binding=replay),
                binding=replay)
            metadata = dict(point_id=node.point_id, trace_id=node.trace_id,
                quadrature_index=index,
                time_fraction=[instant.numerator, instant.denominator])
            queries.append(dict(**metadata, query=query))
            replayed_queries.append(dict(**metadata, query=replayed))
    if len(history.quadrature_histories) != 16 or len(indices) != 4 or len(queries) != 48:
        raise AssertionError("Original Gauss quadrature coverage changed")
    write(out / "original_nodal_queries.json", queries)
    write(out / "original_nodal_queries_replayed.json", replayed_queries)
    print("Binding all seven original point contexts at their retained first-entry windows", flush=True)
    incoming, incoming_replayed, chronologies = [], [], []
    for record in read(PRIOR_CONTROL / "point_certificates.json"):
        chronology = native_first_entry_material_point_chronology_from_dict(
            deepcopy(record["certificate"]), state=material)
        chronology_raw = native_first_entry_material_point_chronology_to_dict(chronology)
        if encoded(chronology_raw) != encoded(record["certificate"]):
            raise AssertionError("An incoming context changed its original first-entry chronology")
        chronologies.append(dict(label=record["label"], certificate=chronology_raw))
        binding = certify_native_first_entry_physical_binding(chronology, history, precision=256)
        raw = native_first_entry_physical_binding_to_dict(binding)
        restored = native_first_entry_physical_binding_from_dict(deepcopy(raw),
            chronology=chronology, history=replay)
        incoming.append(dict(label=record["label"], binding=raw))
        incoming_replayed.append(dict(label=record["label"],
            binding=native_first_entry_physical_binding_to_dict(restored)))
    if len(incoming) != 7:
        raise AssertionError("Incoming binding requires all seven original point contexts")
    write(out / "point_certificates.json", chronologies)
    write(out / "incoming_first_entry_bindings.json", incoming)
    write(out / "incoming_first_entry_bindings_replayed.json", incoming_replayed)
    pairs = (
        ("original_force/initial.json", "original_force/after_refusal.json"),
        ("original_history.json", "original_history_replayed.json"),
        ("original_nodal_queries.json", "original_nodal_queries_replayed.json"),
        ("incoming_first_entry_bindings.json", "incoming_first_entry_bindings_replayed.json"),
    )
    for first, second in pairs:
        if (out / first).read_bytes() != (out / second).read_bytes():
            raise AssertionError("Physical-history replay bytes differ: " + first)
    after, starting_after = guard_sources()
    if before != after or starting != starting_after:
        raise AssertionError("A Python source changed while running the physical binding control")
    write(out / "sources_after.json", after)
    report = dict(format="native-original-physical-binding-control-1", status="PASS",
        seconds=time.perf_counter() - started,
        original_source_count=654, source_count=len(before),
        existing_sources_changed=[], source_before_after_unchanged=True,
        added_python_sources=sorted(set(before) - set(starting)),
        original_typed_checkpoint_loaded=True, original_force_re_solved=True,
        original_candidate_all_fields_identical=True,
        original_candidate_field_count=force["report"]["original_candidate_field_count"],
        history_builder_candidate_replayed=True, history_dto_external_sources_replayed=True,
        prior_material_state_replayed=True, original_trace_count=4, original_node_count=16,
        query_times=[[instant.numerator, instant.denominator] for instant in QUERY_TIMES],
        query_count=len(queries), incoming_context_count=len(incoming),
        byte_pairs=[list(pair) for pair in pairs],
        original_quadrature_memory_measure_work_preserved=True,
        intermediate_work_computed=False, work_assigned_to_material_volume=False,
        candidate_mechanics_adopted=False, physical_pair_admission_authorized=False,
        global_native_inventory_updated=False, accepted_loaded_native_intervals=0,
        accepted_source_work_j=0, force_refusal_reasons=force["report"]["refusal_reasons"],
        artifacts_sha256={str(path.relative_to(out)):digest(path)
            for path in sorted(out.rglob("*")) if path.is_file()})
    saved = write(out / "report.json", report, seal="report_sha256")
    print(json.dumps({key:saved[key] for key in ("status", "seconds", "source_count",
        "query_count", "incoming_context_count", "accepted_loaded_native_intervals")}), flush=True)
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run_control(args.out.resolve())


if __name__ == "__main__":
    main()
