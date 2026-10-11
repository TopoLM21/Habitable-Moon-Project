"""Seal the handover control, independent algebra audit and scoped gate."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from analysis.geometric_native_physical_handover_validation.run_handover_control import (
    PHYSICAL, MATERIAL, digest, guard_sources, read, sealed, verify_previous_sources, write,
)
from analysis.geometric_native_physical_handover_validation.run_release_gates import TEST_PATHS


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def check(path, expected):
    if digest(path) != expected:
        raise ValueError("Handover evidence artifact differs: " + str(path))


def consolidate(gate_dir, control_dir, audit_path, out):
    gate_dir, control_dir, audit_path, out = (Path(path).resolve()
        for path in (gate_dir, control_dir, audit_path, out))
    gate = sealed(gate_dir / "receipt.json", "receipt_sha256")
    report = sealed(control_dir / "report.json", "report_sha256")
    audit = sealed(audit_path, "receipt_sha256")
    if any(row["status"] != "PASS" for row in (gate, report, audit)):
        raise ValueError("The fresh gate, actual handover control and independent audit must all pass")
    if (gate["exit_code"] != 0 or gate["changed_paths"] or gate["is_full_profile_execution"]
            or gate["original_253_tests_reexecuted"] or gate["original_770_tests_reexecuted"]
            or tuple(gate["test_paths"]) != TEST_PATHS or gate["counts"].get("passed", 0) <= 0):
        raise ValueError("The gate does not have the declared five-file scope")
    for key, name in (("log_sha256", "pytest.log"),
            ("sources_before_sha256", "sources_before.json"),
            ("sources_after_sha256", "sources_after.json")):
        check(gate_dir / name, gate[key])
    current, baseline = guard_sources()
    for directory in (gate_dir, control_dir):
        for name in ("sources_before.json", "sources_after.json"):
            if read(directory / name) != current:
                raise ValueError("The complete gate/control/current source maps differ")
    if report["original_source_count"] != 664 or gate["original_source_count"] != 664:
        raise ValueError("The original 664-source ancestry changed")
    for name, expected in report["artifacts_sha256"].items():
        check(control_dir / name, expected)
    for name, expected in audit["input_artifact_sha256"].items():
        check(control_dir / name, expected)
    required = ("original_typed_checkpoint_loaded", "original_force_re_solved",
        "history_builder_original_force_re_solved", "prior_material_state_replayed",
        "original_candidate_all_fields_identical", "original_local_states_preserved",
        "source_history_bytes_unchanged", "source_checkpoint_bytes_unchanged",
        "source_candidate_bytes_unchanged", "source_quad_measure_work_counted_once",
        "unchanged_global_beam_dofs_and_stiffness", "correlated_geometry_cancelled_symbolically",
        "source_before_after_unchanged")
    if any(report[name] is not True for name in required):
        raise ValueError("The handover control lacks a required full-source preservation proof")
    if (report["original_candidate_field_count"] != 14 or report["original_node_count"] != 16
            or report["transported_port_count"] != 8 or report["retained_port_count"] != 8
            or report["transported_source_trace_count"] != 2
            or report["source_epoch"] != "unaccepted_candidate" or report["time_fraction"] != [1, 1]
            or report["port_reference_observation_count"] != 8 or report["mutation_refusal_count"] != 14):
        raise ValueError("The original candidate port/time/mutation coverage changed")
    for name in ("positive_cumulative_slip_count", "nonzero_signed_plastic_slip_count",
            "positive_compressive_pressure_count"):
        if report[name] != 16:
            raise ValueError("Loaded original constitutive history was dropped")
    for name in ("accepted_loaded_native_intervals", "accepted_source_work_j",
            "transfer_external_work_j", "transfer_dissipation_j"):
        if report[name] != 0:
            raise ValueError("Configuration transport created accepted work, dissipation or a native step")
    for name in ("transported_candidate_is_force_equilibrium", "native_full_bank_and_clamp_support_certified",
            "finite_measure_assigned_to_first_entry_point", "candidate_mechanics_adopted",
            "physical_pair_admission_authorized", "global_native_inventory_updated"):
        if report[name] is not False:
            raise ValueError("The bounded handover control exceeded its certified physical scope")
    byte_pairs = []
    for first, second in report["byte_pairs"]:
        if (control_dir / first).read_bytes() != (control_dir / second).read_bytes():
            raise ValueError("Handover replay or unchanged source bytes differ: " + first)
        byte_pairs.append(dict(first=first, second=second, sha256=digest(control_dir / first),
            bytes=(control_dir / first).stat().st_size, accepted_native_force_checkpoint=False))
    if len(byte_pairs) != 4:
        raise ValueError("The handover control must retain all four bytepairs")
    authority = verify_previous_sources()
    if authority != sealed(control_dir / "source_validation_receipt.json", "receipt_sha256"):
        raise ValueError("The previous 139-artifact ancestry changed during the new control")
    prior_path = PHYSICAL / "release_validation_01/validation_summary.json"
    prior = sealed(prior_path, "summary_sha256")
    plan = deepcopy(prior["original_six_point_plan"])
    plan[0] = dict(number=1, status="partial",
        completed_here="receiver-attached loaded configuration transport for eight supported original candidate Gauss ports, eight once-retained histories, source beam interior charts and exact unchanged local/beam/work ledgers",
        remaining="whole incoming first-entry field and moving bank/clamp support, joint implicit material-force admission, accepted native interval and joint restart")
    remaining = [
        dict(number=1, title="Сохраняющий перенос физических связей на входящий материал с памятью, энергией и работой",
            status="partial",
            completed="узловой оператор, собственная балочная карта, восемь настоящих присоединений прежних пар и самостоятельное символическое семейство",
            remaining="полный неявный первый вход, смена пары 6→3 и связь памяти/работы с материальной мерой"),
        dict(number=2, title="Подключение неявных материальных областей к силовым опорам и пространственному допуску",
            status="pending", remaining="полные банки и заделки, обновлённые силы и пространственный допуск"),
        dict(number=3, title="Принятое силовое движение, последующие интервалы и совместный restart с проверкой балансов",
            status="pending", remaining="принятый нативный силовой шаг, последующие интервалы и совместный restart"),
    ]
    from scripts.cloud_handoff import verify as verify_cloud_sources
    restored = verify_cloud_sources()
    if restored != dict(files=9, source_bytes=2110696, verified=True):
        raise ValueError("The original nine immutable cloud sources changed")
    out.mkdir(parents=True, exist_ok=False)
    write(out / "restored_source_receipt.json", restored)
    added = sorted(set(current) - set(baseline))
    artifacts = {path.resolve() for directory in (gate_dir, control_dir)
        for path in directory.rglob("*") if path.is_file()}
    artifacts.update((audit_path, BASE / "starting_sources.json", out / "restored_source_receipt.json"))
    artifacts.update(ROOT / name for name in authority["source_artifacts"])
    artifacts.update(ROOT / name for name in added)
    manifest = dict(format="native-material-port-handover-artifact-manifest-1", count=len(artifacts),
        artifacts={relative(path):dict(sha256=digest(path), bytes=path.stat().st_size)
            for path in sorted(artifacts)}, accepted_loaded_native_intervals=0)
    write(out / "artifact_manifest.json", manifest, seal="manifest_sha256")
    summary = dict(format="native-material-port-handover-validation-1", status="PASS",
        prior_physical_validation=dict(path=relative(prior_path), sha256=digest(prior_path),
            verified_artifacts=56, tests_passed=253, tests_reexecuted=False, full_profile_executed=False),
        prior_material_validation=dict(path=relative(MATERIAL / "release_validation_02/validation_summary.json"),
            sha256=digest(MATERIAL / "release_validation_02/validation_summary.json"),
            verified_artifacts=83, tests_passed=770, tests_reexecuted=False, full_profile_executed=False),
        fresh_scoped_gate=dict(path=relative(gate_dir / "receipt.json"), sha256=digest(gate_dir / "receipt.json"),
            tests_passed=gate["counts"]["passed"], counts=gate["counts"], test_paths=list(TEST_PATHS),
            test_files=len(TEST_PATHS), seconds=gate["seconds"], full_profile_executed=False),
        actual_control=dict(path=relative(control_dir / "report.json"), sha256=digest(control_dir / "report.json"),
            seconds=report["seconds"], original_typed_checkpoint_loaded=True,
            original_force_re_solved=True, history_builder_original_force_re_solved=True,
            original_candidate_field_count=14, original_node_count=16,
            transported_port_count=8, retained_port_count=8, mutation_refusal_count=14,
            port_reference_observation_count=8, byte_pairs=byte_pairs),
        independent_audit=dict(path=relative(audit_path), sha256=digest(audit_path),
            scope="exact source/lineage/measure/work and source-cubic/operator algebra; independent high-precision point geometry/virtual-power checks; no whole native bank support or equilibrium/admission proof"),
        source_count=len(current), starting_source_count=664, added_python_sources=added,
        existing_sources_changed=[], source_before_after_unchanged=True,
        restored_sources=restored, original_six_point_plan=plan,
        remaining_work_blocks=remaining, remaining_work_block_count=3, law=report["law"],
        transported_candidate_is_force_equilibrium=False, native_full_bank_and_clamp_support_certified=False,
        finite_measure_assigned_to_first_entry_point=False, candidate_mechanics_adopted=False,
        physical_pair_admission_authorized=False, global_native_inventory_updated=False,
        transfer_external_work_j=0, transfer_dissipation_j=0,
        accepted_loaded_native_intervals=0, accepted_source_work_j=0,
        artifact_manifest=dict(path=relative(out / "artifact_manifest.json"),
            sha256=digest(out / "artifact_manifest.json"), count=len(artifacts)))
    write(out / "validation_summary.json", summary, seal="summary_sha256")
    print(json.dumps(dict(status="PASS", fresh_scoped_tests=gate["counts"]["passed"],
        preserved_prior_tests=[253, 770], sources=len(current), unchanged_existing_sources=664,
        new_python_files=len(added), artifacts=len(artifacts), transported_ports=8,
        retained_ports=8, accepted_native_force_intervals=0)), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    consolidate(args.gate, args.control, args.audit, args.out)
