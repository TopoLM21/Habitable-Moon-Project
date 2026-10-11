"""Seal the frozen scoped release without promoting the refused second interval.

This consumes saved evidence only. Historical tests and force equations are not
executed here; the seven-module gate and both controls supply execution evidence.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.geometric_native_active_line_validation.validation_support import (
    BASE, PRIOR, digest, encoded, guard_sources, read, sealed, verify_baseline, write,
)


TEST_PATHS = {
    "tests/test_geometric_native_active_contact_line.py",
    "tests/test_geometric_native_active_contact_line_work.py",
    "tests/test_geometric_native_implicit_force_support.py",
    "tests/test_geometric_native_implicit_material_continuation.py",
    "tests/test_geometric_native_implicit_mantle_loading.py",
    "tests/test_geometric_native_implicit_underthrust_motion.py",
    "tests/test_geometric_native_implicit_epoch_admission.py",
}


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def reference(path, **extra):
    return dict(path=relative(path), sha256=digest(path), **extra)


def check_hash(path, expected):
    if digest(path) != expected:
        raise ValueError("Release artifact hash differs: " + str(path))


def require(value, true=(), false=(), equal=None):
    if (any(value[name] is not True for name in true)
            or any(value[name] is not False for name in false)
            or any(value[name] != expected for name, expected in (equal or {}).items())):
        raise ValueError("Release evidence scope or status differs: " + value.get("format", "unknown"))


def check_maps(directory, names, frozen):
    for name in names:
        if read(directory / name) != frozen:
            raise ValueError("Execution did not use the complete frozen source map: " + str(directory / name))


def check_pairs(directory, report, expected_count):
    pairs = report["byte_pairs"]
    if len(pairs) != expected_count:
        raise ValueError("Complete replay byte-pair coverage differs")
    for row in pairs:
        first, second = directory / row["first"], directory / row["second"]
        if (first.read_bytes() != second.read_bytes()
                or first.stat().st_size != row["bytes"] or digest(first) != row["sha256"]):
            raise ValueError("Replay or refusal preservation bytes differ: " + str(first))
    return pairs


def inherited_records(baseline):
    """Retain every old record and also count its distinct artifact paths."""
    rows = [baseline["prior_authoritative_validation"], *baseline["ancestry"]]
    records, unique = [], {}
    for row in rows:
        manifest = sealed(ROOT / row["manifest_path"], "manifest_sha256")
        for name, record in manifest["artifacts"].items():
            if name in unique and unique[name] != record:
                raise ValueError("Inherited manifests disagree about an artifact")
            unique[name] = record
            records.append(dict(manifest_path=row["manifest_path"], path=name,
                sha256=record["sha256"], bytes=record["bytes"]))
    if (len(records) != 374 or len(unique) != baseline["verified_unique_artifacts"]):
        raise ValueError("The complete 374-record historical ancestry was not inherited")
    return rows, records, unique


def check_prerelease_evidence(immutable):
    """Authenticate failed probe bodies, including the declared runtime archive."""
    results = []
    for directory in sorted(BASE.glob("prerelease_next_motion_probe_*")):
        if not directory.is_dir():
            continue
        failure_path = directory / "failure_receipt.json"
        failure = sealed(failure_path, "receipt_sha256")
        require(failure, true=("prior_704_sources_unchanged",),
            false=("final_release_gate",), equal={"status": "FAIL"})
        for key, name in (("source_before_sha256", "sources_before.json"),
                ("source_after_sha256", "sources_after_failure.json"),
                ("traceback_sha256", "failure_traceback.log")):
            check_hash(directory / name, failure[key])
        for name in ("sources_before.json", "sources_after_failure.json"):
            observed = read(directory / name)
            if any(observed.get(path) != sha for path, sha in immutable.items()):
                raise ValueError("A failed prerelease probe changed an immutable source")
        archives_path = directory / "failure_imported_source_archive.json"
        archives = read(archives_path)
        runtime = 0
        for source, record in archives.items():
            check_hash(ROOT / record["archive"], record["sha256"])
            runtime += source.startswith(".venv/")
        scope_path = directory / "failure_archive_scope.json"
        scope_ref = None
        if scope_path.is_file():
            scope = sealed(scope_path, "scope_sha256")
            if (scope["archived_files"] != len(archives)
                    or scope["runtime_environment_python_file_count"] != runtime
                    or scope["project_additive_python_source_count"] != len(archives)-runtime):
                raise ValueError("Failure archive runtime/project scope differs")
            scope_ref = reference(scope_path)
        elif runtime:
            raise ValueError("Runtime failure archives need an explicit separate scope")
        recovery_path = directory / "recovered_next_force_receipt.json"
        recovery_ref = None
        if recovery_path.is_file():
            recovery = sealed(recovery_path, "receipt_sha256")
            require(recovery, true=("original_704_sources_unchanged", "fresh_source_freeze_and_final_gate_required"),
                false=("final_release_gate",), equal={"status": "OBSERVED", "accepted_second_intervals": 0})
            check_hash(ROOT / recovery["full_candidate_path"], recovery["full_candidate_sha256"])
            recovery_ref = reference(recovery_path, scope=recovery["scope"])
        results.append(dict(failure=reference(failure_path), stage=failure["stage"],
            status="FAIL", final_release_gate=False, accepted_second_intervals=0,
            source_archive=reference(archives_path), archived_file_count=len(archives),
            project_source_archive_count=len(archives)-runtime, runtime_archive_count=runtime,
            archive_scope=scope_ref, recovered_observation=recovery_ref,
            scope="prerelease construction/export failure and saved bodies; not passing final gate or second acceptance"))
    if len(results) != 2:
        raise ValueError("Both declared failed prerelease probes must remain visible")
    return results


def check_failed_final_gates(immutable):
    """Keep interrupted frozen gates distinct from exploratory source probes.

    Relocation preserves the failed gate's raw archive map. Its recorded body
    paths therefore use the original release directory; actual body reads use
    the corresponding basename inside the preserved failed directory.
    """
    results = []
    for directory in sorted(BASE.glob("failed_release_validation_*")):
        if not directory.is_dir():
            continue
        receipt_path = directory / "receipt.json"
        receipt = sealed(receipt_path, "receipt_sha256")
        preservation_path = directory / "failure_preservation_receipt.json"
        preservation = sealed(preservation_path, "receipt_sha256")
        require(receipt, false=("is_full_profile_execution", "prior_136_tests_reexecuted", "prior_341_tests_reexecuted"),
            equal={"status": "FAIL", "immutable_baseline_source_count": 704, "changed_paths": [],
                "canonical_pytest_import_mode": "importlib"})
        require(preservation, true=("prior_704_sources_unchanged", "source_before_after_unchanged"),
            false=("same_process_controls_executed", "point_1_completed"),
            equal={"status": "FAIL", "immutable_prior_source_count": 704,
                "source_count": receipt["source_count"]})
        if (receipt["exit_code"] == 0 or receipt["counts"].get("failed", 0) == 0
                or set(receipt["test_paths"]) != TEST_PATHS or len(receipt["test_paths"]) != 7):
            raise ValueError("Preserved failed gate must be the interrupted seven-module run")
        for key, name in (("log_sha256", "pytest.log"), ("sources_before_sha256", "sources_before.json"),
                ("sources_after_sha256", "sources_after.json")):
            check_hash(directory / name, receipt[key])
        check_hash(receipt_path, preservation["test_receipt_sha256"])
        frozen_path = directory / "frozen_release_sources.json"
        check_hash(frozen_path, preservation["frozen_map_sha256"])
        frozen = read(frozen_path)
        if (len(frozen) != receipt["source_count"]
                or any(frozen.get(name) != sha for name, sha in immutable.items())):
            raise ValueError("Failed final gate did not preserve the complete immutable source map")
        check_maps(directory, ("sources_before.json", "sources_after.json"), frozen)
        freeze_path = directory / "frozen_release_receipt.json"
        freeze = sealed(freeze_path, "receipt_sha256")
        require(freeze, equal={"status": "PASS", "immutable_baseline_source_count": 704,
            "frozen_source_count": len(frozen), "prior_python_sources_changed": [],
            "added_python_sources": sorted(set(frozen)-set(immutable))})
        check_hash(frozen_path, freeze["frozen_sources_sha256"])
        check_hash(BASE / "immutable_baseline_receipt.json", freeze["immutable_baseline_receipt_sha256"])
        archive_path = directory / "frozen_additive_source_archive.json"
        check_hash(archive_path, preservation["source_archive_sha256"])
        archives = read(archive_path)
        if (set(archives) != set(frozen)-set(immutable)
                or len(archives) != preservation["archived_additive_source_count"]):
            raise ValueError("The failed final freeze needs every additive source body")
        relocation_path = directory / "relocation_receipt.json"
        relocation = sealed(relocation_path, "receipt_sha256")
        original_directory = relative(BASE / "release_validation_01")
        recorded_prefix = original_directory + "/frozen_additive_source_bodies/"
        relocated_rows = {}
        for name, row in archives.items():
            basename = name.replace("/", "__") + ".txt"
            if row["path"] != recorded_prefix + basename:
                raise ValueError("The relocated archive changed its recorded source path")
            actual = directory / "frozen_additive_source_bodies" / basename
            if (row["sha256"] != frozen[name] or digest(actual) != row["sha256"]
                    or actual.stat().st_size != row["bytes"]):
                raise ValueError("A preserved failed-freeze source body changed")
            relocated_rows[name] = dict(source=name, recorded_path=row["path"],
                preserved_actual_path=relative(actual), sha256=row["sha256"])
        attempt = directory.name.removeprefix("failed_release_validation_")
        freeze_archive = BASE / ("failed_source_freeze_" + attempt)
        require(relocation, true=("all_failed_receipt_bytes_unchanged",), equal={"status": "PRESERVED",
            "original_directory": original_directory, "preserved_directory": relative(directory),
            "recorded_source_archive_sha256": digest(archive_path),
            "original_freeze_directory": relative(freeze_archive)})
        observed_relocations = relocation["source_body_relocations"]
        if (len(observed_relocations) != len(relocated_rows)
                or {row["source"]: row for row in observed_relocations} != relocated_rows):
            raise ValueError("The archive relocation receipt does not bind every actual preserved body")
        for name in ("frozen_release_sources.json", "frozen_release_receipt.json"):
            if (freeze_archive / name).read_bytes() != (directory / name).read_bytes():
                raise ValueError("The original failed source-freeze bytes changed during relocation")
        if archives["analysis/geometric_native_active_line_validation/run_release_gates.py"]["sha256"] != receipt["runner_sha256"]:
            raise ValueError("The failed gate runner differs from its saved execution body")
        if any((directory / name).exists() for name in ("suite_receipt.json", "control_console.log")):
            raise ValueError("The failed gate unexpectedly claims same-process controls")
        results.append(dict(attempt=attempt, gate=reference(receipt_path), preservation=reference(preservation_path),
            status="FAIL", source_count=len(frozen), counts=receipt["counts"],
            stage=preservation["stage"], same_process_controls_executed=False,
            frozen_source_map=reference(frozen_path), frozen_source_receipt=reference(freeze_path),
            relocation=reference(relocation_path),
            original_failed_freeze_directory=relative(freeze_archive),
            additive_source_archive=reference(archive_path, source_body_count=len(archives),
                recorded_path_prefix=recorded_prefix,
                actual_body_directory=relative(directory / "frozen_additive_source_bodies")),
            scope="failed frozen final-test attempt; partial passes are not the fresh passing gate and its controls were not run"))
    expected_counts = {"01": {"passed": 89, "failed": 1}, "02": {"passed": 103, "failed": 1},
        "03": {"passed": 117, "failed": 1}}
    if {row["attempt"]: row["counts"] for row in results} != expected_counts:
        raise ValueError("All three exact interrupted final-gate outcomes must remain in the release history")
    return results


def check_source_amendments(failed_gates, frozen, immutable):
    """Bind each unchanged amendment receipt to its actual preserved freeze.

    Earlier amendments record their then-current canonical freeze filenames.
    Those bytes now live in the corresponding failed_source_freeze directory;
    relocation binds each actual directory without rewriting prior receipts.
    """
    paths = sorted(BASE.glob("source_amendment_*.json"))
    if {path.name for path in paths} != {"source_amendment_02.json", "source_amendment_03.json", "source_amendment_04.json"}:
        raise ValueError("All three explicit post-failure source amendments are required")
    failures = {int(row["attempt"]): row for row in failed_gates}
    results = []

    def bind(row, actual, *, recorded_canonical=False):
        actual = Path(actual)
        allowed = {relative(actual)}
        if recorded_canonical:
            allowed.add(relative(BASE / actual.name))
        if set(row) != {"path", "sha256"} or row["path"] not in allowed:
            raise ValueError("Source amendment reference has an unexpected recorded path")
        check_hash(actual, row["sha256"])
        return dict(recorded_path=row["path"], actual_path=relative(actual), sha256=row["sha256"])

    for path in paths:
        amendment = sealed(path, "receipt_sha256")
        number = int(path.stem.removeprefix("source_amendment_"))
        previous = failures[number-1]
        old_directory = ROOT / previous["original_failed_freeze_directory"]
        new_directory = (ROOT / failures[number]["original_failed_freeze_directory"]
            if number in failures else BASE)
        old_map_path = old_directory / "frozen_release_sources.json"
        new_map_path = new_directory / "frozen_release_sources.json"
        old, new = read(old_map_path), read(new_map_path)
        require(amendment, true=("final_checks_required", "old_receipt_and_source_bodies_preserved",
            "prior_704_sources_unchanged"), false=("physical_parameters_or_law_changed",),
            equal={"format": "native-frozen-scoped-gate-source-amendment-1", "status": "PASS",
                "immutable_prior_source_count": 704, "source_count": len(frozen)})
        if (set(old) != set(new) or len(old) != len(frozen)
                or any(old.get(name) != sha or new.get(name) != sha for name, sha in immutable.items())):
            raise ValueError("A post-failure amendment changed the original source set or immutable704")
        changes = {name: dict(before_sha256=old[name], after_sha256=new[name])
            for name in old if old[name] != new[name]}
        if not changes or amendment["changes"] != changes:
            raise ValueError("The exact post-failure source hash changes differ from their frozen maps")
        if number == 3:
            expected = {"analysis/geometric_native_active_line_validation/consolidate_release.py",
                "tests/test_geometric_native_implicit_underthrust_motion.py"}
            if set(changes) != expected or amendment.get("production_sources_changed") != []:
                raise ValueError("Freeze3 must change only the declared test and release accounting source")
        elif number == 4:
            service_sources = {"analysis/geometric_native_active_line_validation/consolidate_release.py",
                "analysis/geometric_native_active_line_validation/run_release_gates.py"}
            required_sources = {"analysis/geometric_native_active_line_validation/consolidate_release.py",
                "tests/test_geometric_native_implicit_material_continuation.py"}
            if (not required_sources <= set(changes) <= service_sources | TEST_PATHS
                    or amendment.get("production_sources_changed") != [] or new != frozen):
                raise ValueError("Freeze4 may change only the declared new tests and release service sources")
        bindings = {}
        for name, actual, canonical in (
                ("old_frozen_sources", old_map_path, False),
                ("old_frozen_receipt", old_directory / "frozen_release_receipt.json", False),
                ("old_additive_source_bodies", ROOT / previous["additive_source_archive"]["path"], False),
                ("new_frozen_sources", new_map_path, True),
                ("new_frozen_receipt", new_directory / "frozen_release_receipt.json", True)):
            bindings[name] = bind(amendment[name], actual, recorded_canonical=canonical)
        results.append(dict(receipt=reference(path), changes=changes, reason=amendment["reason"],
            verified_references=bindings,
            old_freeze_relocation=previous["relocation"],
            new_freeze_relocation=failures[number]["relocation"] if number in failures else None,
            scope="declared additive-source amendment after failed frozen gate; final137 rerun and controls required"))
    return results


def check_reload_cache_scope(failed_gates, frozen, motion_dir):
    """Verify retained-interpreter scope against each execution freeze and DTO."""
    failures = {int(row["attempt"]): row for row in failed_gates}
    results = []
    motion = read(motion_dir / "accepted_motion_before.json")
    material_candidate = read(motion_dir / "unadopted_material_candidate.json")
    load = read(motion_dir / "subsequent_source_load.json")
    refusal = read(motion_dir / "subsequent_temporal_refusal.json")
    contexts = dict(retained_before_state_sha256=motion["state_sha256"],
        retained_current_load_sha256=load["load_sha256"],
        retained_material_epoch_sha256=material_candidate["epoch_sha256"],
        retained_refusal_sha256=refusal["refusal_sha256"])
    for number in (3, 4):
        path = BASE / ("test_reload_cache_scope_" + str(number).zfill(2) + ".json")
        receipt = sealed(path, "receipt_sha256")
        require(receipt, true=("authenticated_fixture_caches_retained", "test_module_reloaded_after_freeze",
            "all_137_test_assertions_full_rerun_required", "unchanged_fixture_ast_confirmed",
            "source_history_matches_authoritative_complete_dto"), false=("fresh_independent_process_execution",),
            equal={"format": "native-frozen-test-reload-authenticated-cache-scope-1", "status": "PASS",
                "source_count": len(frozen), "production_sources_changed_since_physical_construction": []})
        directory = ROOT / failures[number]["original_failed_freeze_directory"] if number in failures else BASE
        map_path = directory / "frozen_release_sources.json"
        execution_map = read(map_path)
        check_hash(map_path, receipt["frozen_sources_sha256"])
        amendment_path = BASE / ("source_amendment_" + str(number).zfill(2) + ".json")
        check_hash(amendment_path, receipt["amendment_" + str(number).zfill(2) + "_sha256"])
        amendment = sealed(amendment_path, "receipt_sha256")
        expected_tests = set(amendment["changes"]) & TEST_PATHS
        loaded = receipt.get("loaded_test_sources_sha256")
        if loaded is None:
            loaded = {receipt["test_path"]: receipt["loaded_test_file_sha256"]}
        if set(loaded) != expected_tests or any(execution_map.get(name) != sha for name, sha in loaded.items()):
            raise ValueError("The reloaded amended test bodies differ from their exact execution freeze")
        for key, expected in contexts.items():
            if key in receipt and receipt[key] != expected:
                raise ValueError("Retained authenticated cache context differs from the final exported force/refusal DTO")
        if "retained_fixture_constructed_in" in receipt:
            origin = receipt["retained_fixture_constructed_in"]
            if origin not in {"failed_release_validation_" + row["attempt"] for row in failed_gates}:
                raise ValueError("The retained fixture construction origin is not preserved")
        if number == 4 and "analysis/geometric_native_active_line_validation/run_release_gates.py" in amendment["changes"]:
            require(receipt, true=("runner_reloaded_after_freeze",))
            if receipt["loaded_runner_sha256"] != execution_map["analysis/geometric_native_active_line_validation/run_release_gates.py"]:
                raise ValueError("The amended runner was not reloaded from its frozen source")
        results.append(reference(path, frozen_source_map=reference(map_path),
            amendment=reference(amendment_path), fresh_independent_process_execution=False,
            authenticated_fixture_caches_retained=True, amended_tests_reloaded_after_freeze=True))
    return results


def consolidate(gate_dir=None, control_dir=None, out=None):
    gate_dir = Path(gate_dir or BASE / "release_validation_01").resolve()
    control_dir = Path(control_dir or BASE / "control_release_01").resolve()
    out = Path(out or gate_dir).resolve()
    if not out.is_dir():
        raise ValueError("The release output directory must already contain its passing gate")
    if any((out / name).exists() for name in ("artifact_manifest.json", "validation_summary.json")):
        raise ValueError("An existing sealed release cannot be overwritten")
    baseline = verify_baseline()
    frozen, immutable = guard_sources(require_frozen=True)
    freeze_path = BASE / "frozen_release_receipt.json"
    freeze = sealed(freeze_path, "receipt_sha256")
    require(freeze, equal={"status": "PASS", "immutable_baseline_source_count": 704,
        "frozen_source_count": len(frozen), "prior_python_sources_changed": [],
        "added_python_sources": sorted(set(frozen)-set(immutable))})
    check_hash(BASE / "frozen_release_sources.json", freeze["frozen_sources_sha256"])
    check_hash(BASE / "immutable_baseline_receipt.json", freeze["immutable_baseline_receipt_sha256"])
    saved_baseline = sealed(BASE / "immutable_baseline_receipt.json", "receipt_sha256")
    require(saved_baseline, true=("immutable_704_sources_unchanged", "original_694_sources_unchanged"),
        false=("prior_tests_reexecuted",), equal={"status": "PASS", "verified_manifest_records": 374})

    gate_path, suite_path = gate_dir / "receipt.json", gate_dir / "suite_receipt.json"
    gate, suite = sealed(gate_path, "receipt_sha256"), sealed(suite_path, "receipt_sha256")
    require(gate, false=("is_full_profile_execution", "prior_136_tests_reexecuted", "prior_341_tests_reexecuted"),
        equal={"status": "PASS", "exit_code": 0, "changed_paths": [], "counts": {"passed": 137},
            "canonical_pytest_import_mode": "importlib", "immutable_baseline_source_count": 704,
            "source_count": len(frozen), "execution_mode": "single_pytest_process_shared_actual_source_fixture"})
    if len(gate["test_paths"]) != 7 or set(gate["test_paths"]) != TEST_PATHS:
        raise ValueError("The agreed seven-module scoped regression gate differs")
    for key, name in (("log_sha256", "pytest.log"), ("sources_before_sha256", "sources_before.json"),
            ("sources_after_sha256", "sources_after.json")):
        check_hash(gate_dir / name, gate[key])
    check_hash(BASE / "run_release_gates.py", gate["runner_sha256"])
    require(suite, true=("source_before_after_unchanged", "actual_source_fixture_shared_with_control"),
        equal={"status": "PASS", "source_count": len(frozen), "control_output": relative(control_dir),
            "control_callable": "analysis.geometric_native_active_line_validation.run_complete_control:run_complete_control"})
    check_hash(gate_path, suite["pytest_gate_receipt_sha256"])
    for key, name in (("sources_before_sha256", "sources_before.json"),
            ("suite_sources_after_sha256", "suite_sources_after.json"),
            ("control_console_sha256", "control_console.log")):
        check_hash(gate_dir / name, suite[key])
    check_maps(gate_dir, ("sources_before.json", "sources_after.json", "suite_sources_after.json"), frozen)

    combined_path = control_dir / "combined_report.json"
    combined = sealed(combined_path, "report_sha256")
    require(combined, true=("prior_704_python_sources_unchanged", "source_before_after_unchanged",
        "subsequent_candidate_solved", "subsequent_candidate_refused", "joint_restart_passed"),
        false=("subsequent_interval_accepted", "continuous_work_added_to_Gauss_ledger", "entire_point_1_completed"),
        equal={"status": "PASS", "accepted_loaded_native_intervals": 1, "source_count": len(frozen)})
    for key, name in (("sources_before_sha256", "sources_before.json"), ("sources_after_sha256", "sources_after.json")):
        check_hash(control_dir / name, combined[key])
    check_maps(control_dir, ("sources_before.json", "sources_after.json"), frozen)
    line_dir, motion_dir = control_dir / "line", control_dir / "motion"
    line_path, motion_path = line_dir / "report.json", motion_dir / "motion_control_report.json"
    for key, path in (("line_control", line_path), ("motion_control", motion_path)):
        if combined[key] != dict(path=str(path.relative_to(control_dir)), sha256=digest(path)):
            raise ValueError("Composite control does not bind its actual child report")
    line, motion = sealed(line_path, "report_sha256"), sealed(motion_path, "report_sha256")
    require(line, true=("prior_704_sources_unchanged", "source_before_after_unchanged",
        "shared_authenticated_actual_source", "whole_line_selection_proved",
        "whole_line_contact_compression_and_monotone_down_dip_proved", "continuous_interface_measure_preserved",
        "positive_width_strips_retained", "signed_literal_compliance_residual_retained",
        "transfer_work_energy_and_dissipation_exactly_zero", "full_bank_live_clamp_support_observed"),
        false=("original_atoms_added_to_continuous_work", "candidate_mechanics_adopted", "entire_point_1_completed"),
        equal={"status": "PASS", "source_count": len(frozen), "original_actual_force_fields_preserved": 14,
            "weighted_receiving_pair_count": 16, "original_quarter_count": 4, "original_gauss_atom_count": 16,
            "accepted_native_force_intervals": 0, "accepted_source_work_j": 0})
    require(motion, true=("prior_704_sources_unchanged", "source_before_after_unchanged",
        "native_force_admission_proved", "full_first_interval_balances_passed", "subsequent_candidate_solved",
        "subsequent_candidate_refused", "subsequent_force_balances_passed", "refusal_preserved_before_bytes",
        "positive_newborn_contact_proved", "immediate_unsupported_newborn_entry_proved", "joint_restart_passed",
        "joint_restart_repeats_original_genuine_force", "complete_balances_passed"),
        false=("subsequent_interval_accepted", "subsequent_intervals_passed", "entire_point_1_completed",
        "safe_truncation_with_same_euler_vectors_exists", "physical_inputs_invented",
        "continuum_work_added_to_Gauss_ledger", "subsequent_whole_line_down_dip_claimed"),
        equal={"status": "PASS", "accepted_loaded_native_intervals": 1, "source_count": len(frozen),
            "native_force_admission_scope": "first original physical interval only",
            "subsequent_continuous_branch_counts": {"up_dip_plastic": 2, "down_dip_plastic": 2}})
    for directory in (line_dir, motion_dir):
        check_maps(directory, ("sources_before.json", "sources_after.json"), frozen)
    line_pairs, motion_pairs = check_pairs(line_dir, line, 7), check_pairs(motion_dir, motion, 6)
    line_source = line_dir / "source_history_before.json"
    if (line_source.read_bytes() != (motion_dir / "source_history_before.json").read_bytes()
            or read(line_source) != read(PRIOR / "control_release_02/source_history_before.json")):
        raise ValueError("Line and motion controls do not share the complete authoritative actual source")

    line_audit_path = line_dir / "independent_active_line_receipt.json"
    motion_audit_path = motion_dir / "independent_motion_receipt.json"
    line_audit, motion_audit = sealed(line_audit_path, "receipt_sha256"), sealed(motion_audit_path, "receipt_sha256")
    require(line_audit, true=("strict_negative_seam_retention_and_positive_fresh_gate_checked",
        "positive_width_strips_retained", "signed_literal_compliance_residual_retained"),
        false=("imports_production_modules", "entire_point_1_completed"),
        equal={"status": "PASS", "verified_prior_manifest_records": 374, "immutable_prior_python_sources": 704,
            "current_frozen_python_sources": len(frozen), "independent_exact_measure_quarters": 4,
            "exact_receiving_pair_count": 16, "independent_common_potential_scalar_identity_checks": 96,
            "independent_240_digit_real_harmonic_core_integrals": 4, "verified_original_parent_supports": 8,
            "verified_live_clamp_records": 8, "verified_byte_pairs": 7, "accepted_native_force_intervals": 0})
    require(motion_audit, true=("endpoint_area_witness_checked_against_production_rational_lower_bounds",
        "prior_704_sources_unchanged"), false=("production_force_equations_independently_resolved_here",
        "native_endpoint_intersection_area_reproved_here", "subsequent_interval_accepted",
        "entire_point_1_completed", "new_physical_inputs_invented"),
        equal={"status": "PASS", "accepted_loaded_native_intervals": 1, "source_count": len(frozen),
            "verified_inherited_manifest_records": 374, "positive_newborn_contact_count": 4,
            "independently_positive_continuous_edge_count": 2, "original_gauss_atom_count": 16,
            "semantic_refusal_count": 5, "byte_pairs_verified": 6})

    prewarm_path = BASE / "source_prewarm_receipt.json"
    prewarm = sealed(prewarm_path, "receipt_sha256")
    require(prewarm, true=("prior_704_sources_unchanged", "source_history_equal_saved_complete_dto",
        "new_modules_imported_after_freeze_required"), false=("final_release_gate", "imported_new_production_modules"),
        equal={"status": "PASS"})
    for name in ("source_prewarm_sources_before.json", "source_prewarm_sources_after.json"):
        observed = read(BASE / name)
        if any(observed.get(path) != sha for path, sha in immutable.items()):
            raise ValueError("Authenticated prewarming changed an immutable source")
    failed = check_prerelease_evidence(immutable)
    failed_final_gates = check_failed_final_gates(immutable)
    source_amendments = check_source_amendments(failed_final_gates, frozen, immutable)
    reload_cache_scope = check_reload_cache_scope(failed_final_gates, frozen, motion_dir)
    ancestry, old_records, old_artifacts = inherited_records(baseline)
    prior_summary = sealed(ROOT / ancestry[0]["path"], "summary_sha256")
    plan, blocks = deepcopy(prior_summary["original_six_point_plan"]), deepcopy(prior_summary["remaining_work_blocks"])
    if len(plan) != 6 or len(blocks) != 3 or [row["number"] for row in blocks] != [1, 2, 3]:
        raise ValueError("The original six-point plan and three work blocks must be preserved")
    plan[0].update(status="partial", completed_here="whole original receiving line, weighted balances, full original supports, one genuine accepted force interval and joint restart",
        remaining=motion["remaining_block"])
    blocks[0].update(status="completed_for_original_physical_source", remaining="",
        completed="Вся исходная принимающая линия, единственная сохраняющая привязка, взвешенные парные балансы и строгий restart")
    blocks[1].update(status="completed_for_original_physical_source",
        completed="Полные исходные банки и заделки, все 16 исходных Gauss-носителей, пространственный допуск первого интервала",
        remaining="Физическое рождение новых контактных пар и банков требуется для последующих интервалов; блок 3")
    blocks[2].update(status="partial",
        completed="Один принятый исходный силовой интервал, совместный restart и настоящий второй решённый кандидат с доказанным отказом",
        remaining=motion["remaining_block"])

    artifacts = {ROOT / name for name in old_artifacts}
    artifacts.update(ROOT / row[key] for row in ancestry for key in ("path", "manifest_path"))
    artifacts.update(ROOT / name for name in set(frozen)-set(immutable))
    for directory in {BASE, gate_dir, control_dir}:
        artifacts.update(path.resolve() for path in directory.rglob("*") if path.is_file()
            and "__pycache__" not in path.parts and path.suffix != ".pyc"
            and path not in (out / "artifact_manifest.json", out / "validation_summary.json"))
    if guard_sources(require_frozen=True)[0] != frozen:
        raise ValueError("Sources changed before final sealing")
    manifest_records = {}
    for path in sorted(artifacts):
        name = relative(path)
        scope = "inherited_immutable_evidence" if name in old_artifacts else "frozen_scoped_release_evidence"
        if "prerelease_next_motion_probe_" in name:
            scope = "failed_prerelease_probe_or_recovered_observation_not_final_gate_not_second_acceptance"
        elif "failed_release_validation_" in name or "failed_source_freeze_" in name:
            scope = "preserved_failed_final_gate_or_prior_freeze_not_passing_gate_controls_not_run"
        elif "source_amendment_" in name:
            scope = "declared_post_failure_source_amendment_requires_fresh_full_scoped_gate"
        elif "test_reload_cache_scope_" in name:
            scope = "retained_authenticated_interpreter_cache_and_reloaded_amended_tests_not_fresh_process"
        elif "source_prewarm" in name:
            scope = "authenticated_old_source_prewarm_not_final_release_gate"
        manifest_records[name] = dict(sha256=digest(path), bytes=path.stat().st_size, scope=scope)
    manifest = dict(format="native-active-line-artifact-manifest-1", count=len(manifest_records),
        unique_artifact_count=len(manifest_records), artifacts=manifest_records,
        inherited_manifest_record_count=374, inherited_unique_artifact_count=len(old_artifacts),
        inherited_records=old_records, accepted_loaded_native_intervals=1,
        failed_final_gate_attempts=failed_final_gates,
        verified_source_amendments=source_amendments,
        verified_reload_cache_scope=reload_cache_scope,
        subsequent_interval_accepted=False, entire_point_1_completed=False,
        excluded_self_referential_documents=[relative(out / "artifact_manifest.json"), relative(out / "validation_summary.json")])
    write(out / "artifact_manifest.json", manifest, "manifest_sha256")
    summary = dict(format="native-active-line-validation-1", status="PASS",
        prior_authoritative_validation=dict(ancestry[0], tests_reexecuted=False),
        inherited_validation_ancestry=ancestry, verified_prior_manifest_records=374,
        inherited_unique_artifact_count=len(old_artifacts), prior_136_tests_reexecuted=False,
        prior_341_tests_reexecuted=False, fresh_scoped_gate=reference(gate_path,
            tests_passed=137, test_files=7, test_paths=gate["test_paths"], seconds=gate["seconds"], full_profile_executed=False),
        same_process_suite=reference(suite_path, authenticated_actual_source_fixture_shared=True),
        actual_control=reference(combined_path, scope=combined["scope"]),
        line_control=reference(line_path, byte_pairs=line_pairs, diagnostic_accepted_native_intervals=0),
        motion_control=reference(motion_path, byte_pairs=motion_pairs, accepted_loaded_native_intervals=1),
        independent_audits=[reference(line_audit_path, scope=line_audit["scope"]),
            reference(motion_audit_path, scope=motion_audit["scope"], force_equations_resolved_here=False,
                endpoint_area_reproved_here=False)],
        source_freeze=reference(freeze_path), immutable_baseline=reference(BASE / "immutable_baseline_receipt.json"),
        source_count=len(frozen), immutable_baseline_source_count=704, original_source_count=694,
        existing_sources_changed=[], added_python_sources=sorted(set(frozen)-set(immutable)),
        source_before_after_unchanged=True, frozen_source_map=reference(BASE / "frozen_release_sources.json"),
        authenticated_source_prewarm=reference(prewarm_path, final_release_gate=False), failed_prerelease_probes=failed,
        failed_final_gate_attempts=failed_final_gates,
        source_amendments=source_amendments,
        reload_cache_scope=reload_cache_scope, fresh_independent_process_execution=False,
        restored_sources=baseline["restored_sources"], original_six_point_plan=plan,
        remaining_work_blocks=blocks, original_work_block_count=3,
        completed_work_block_count=2, remaining_work_block_count=1,
        all_original_receiver_angles_scheduled=True, per_receiving_pair_weighted_contact_integrals_evaluated=True,
        full_original_bank_and_clamp_support_proved=True, accepted_loaded_native_intervals=1,
        first_interval_adopted=True, subsequent_candidate_solved=True, subsequent_candidate_refused=True,
        subsequent_interval_accepted=False, subsequent_intervals_passed=False,
        positive_newborn_contact_proved=True, immediate_unsupported_newborn_entry_proved=True,
        safe_truncation_with_same_euler_vectors_exists=False, joint_restart_passed=True,
        physical_inputs_invented=False, continuum_work_added_to_Gauss_ledger=False,
        entire_point_1_completed=False, remaining_scope=motion["remaining_block"],
        artifact_manifest=reference(out / "artifact_manifest.json", count=len(manifest_records)))
    write(out / "validation_summary.json", summary, "summary_sha256")
    print(encoded(dict(status="PASS", fresh_scoped_tests=137, inherited_manifest_records=374,
        source_count=len(frozen), artifact_manifest_count=len(manifest_records),
        accepted_loaded_native_intervals=1, subsequent_interval_accepted=False,
        entire_point_1_completed=False)).decode(), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, default=BASE / "release_validation_01")
    parser.add_argument("--control", type=Path, default=BASE / "control_release_01")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    consolidate(args.gate, args.control, args.out)
