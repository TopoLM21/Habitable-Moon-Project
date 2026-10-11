"""Summarize finished cloud checks without rerunning tests or physics controls.

Run from any directory after final_cloud_tests.log and cloud_test_receipt.json
have been written. Incomplete checks or changed inputs prevent publication.
"""
from pathlib import Path
import hashlib
import json
import math
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE_COMMIT = "224a07ac1b981b95163cdf6633f424af1599b2f2"
EXPECTED_BRANCH = "fix/genesis-mantle-convection"
EXPECTED_PROFILE_PASSED = 1039
NEW_TEST_COUNTS = {
    "moving_reference": 36,
    "segmented_bending_core": 39,
    "incremental_contact_gauge": 21,
    "physical_rigid_mantle_source": 19,
    "compact_owner_rotations": 40,
    "segmented_core_checkpoint_io": 19,
    "actual_physical_motion_and_checkpoint_io": 12,
}
NEW_TEST_FILES = (
    "test_geometric_bending_reference.py",
    "test_geometric_segmented_bending_dynamics.py",
    "test_geometric_contact_incremental_gauge.py",
    "test_geometric_rigid_mantle_loading.py",
    "test_geometric_compact_rotations.py",
    "test_geometric_segmented_bending_io.py",
    "test_geometric_bending_motion.py",
    "test_geometric_bending_motion_io.py",
)
RESTART_PAIRS = (
    ("changing_source_core", "direct_final.json", "resumed_final.json"),
    ("nonempty_passive_archive", "archive_direct_final.json", "archive_resumed_final.json"),
    ("actual_physical_common_motion", "physical_direct_final.json", "physical_resumed_final.json"),
)
RESERVOIRS = (
    "area_km2", "oceanic_volume_km3", "cold_mantle_volume_km3", "density_excess_mass_kg",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(payload):
    return sha(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8"))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def relative(path):
    return str(path.relative_to(ROOT))


def verify_hashes(mapping, directory, label, expected_count=None):
    require(isinstance(mapping, dict) and mapping, f"Missing {label} hash inventory")
    if expected_count is not None:
        require(len(mapping) == expected_count, f"Unexpected {label} hash count")
    for name, expected in mapping.items():
        path = directory / name
        require(path.is_file() and sha(path.read_bytes()) == expected,
                f"Changed or missing {label} file: {name}")
    return len(mapping)


def git(*arguments):
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def summarize():
    # This gate runs before opening the destination, so interrupted or stale
    # runs cannot acquire a complete validation summary.
    test_receipt_path = HERE / "cloud_test_receipt.json"
    receipt = read_json(test_receipt_path)
    require(receipt.get("exit_code") == 0, "The final cloud profile did not exit successfully")
    require(receipt.get("code_and_tests_unchanged") is True,
            "The final cloud profile did not certify unchanged code and tests")
    log_path = HERE / "final_cloud_tests.log"
    require(receipt.get("log") == relative(log_path), "The receipt names another cloud test log")
    log = log_path.read_text(encoding="utf-8")
    match = re.search(r"^=*[ \t]*(\d+) passed in ([0-9.]+)s[^\n]*$", log, re.MULTILINE)
    require(match is not None and int(match.group(1)) == EXPECTED_PROFILE_PASSED,
            "The final cloud log has no complete 1039-passed result")
    require(not re.search(r"^FAILED |^ERROR |\d+ failed|\d+ errors?", log, re.MULTILINE),
            "The final cloud log contains a failure or error")
    profile_hash_count = verify_hashes(receipt.get("code_and_test_sha256"), ROOT,
                                      "profile code and test")
    for filename in NEW_TEST_FILES:
        require("tests/" + filename in receipt["code_and_test_sha256"],
                f"The final profile does not bind new test file {filename}")

    report_path = HERE / "cloud_control/report.json"
    audit_path = HERE / "cloud_control/recovered_control_audit_receipt.json"
    report_bytes, audit = report_path.read_bytes(), read_json(audit_path)
    report = json.loads(report_bytes)
    require(report.get("status") == "complete" and report.get("code_unchanged") is True,
            "The physical control is incomplete or changed code during execution")
    require(audit.get("status") == "verified_existing_complete_control_without_rerun"
            and audit.get("report_sha256") == sha(report_bytes)
            and audit.get("report_byte_count") == len(report_bytes),
            "The portable control report differs from the independent audit")
    output = (ROOT / audit["source_report"]).parent
    require((output / "report.json").read_bytes() == report_bytes,
            "The portable report is not an exact copy of the original report")
    verify_hashes(report.get("code_sha256"), ROOT, "control code", 154)
    verify_hashes(report.get("checkpoint_sha256"), output, "checkpoint", 10)
    checkpoints = {}
    for name in report["checkpoint_sha256"]:
        checkpoint = read_json(output / name)
        expected = checkpoint.pop("payload_sha256", None)
        require(digest(checkpoint) == expected, f"Invalid checkpoint payload digest: {name}")
        checkpoints[name] = checkpoint
    restarts = []
    for control, direct, resumed in RESTART_PAIRS:
        data = (output / direct).read_bytes()
        require(data == (output / resumed).read_bytes(), f"Restart bytes differ for {control}")
        restarts.append(dict(control=control, direct_file=direct, resumed_file=resumed,
                             bytes_identical=True, sha256=sha(data), byte_count=len(data)))
    require(report.get("restart_exact") is True and report.get("checkpoint_bytes_identical") is True,
            "The control did not certify restart equality")

    verify_hashes(report.get("cloud_source_bundle_sha256"), ROOT / "handoff", "source bundle", 2)
    manifest = read_json(ROOT / "handoff/cloud_sources_manifest.json")
    require(len(manifest["files"]) == 9, "The restored source manifest does not contain nine files")
    require(sha((ROOT / "handoff" / manifest["archive"]).read_bytes()) == manifest["archive_sha256"],
            "The source archive differs from its manifest")
    for item in manifest["files"]:
        data = (ROOT / item["path"]).read_bytes()
        require(len(data) == item["size"] and sha(data) == item["sha256"],
                f"Restored source differs from manifest: {item['path']}")
    source_bytes = sum(item["size"] for item in manifest["files"])
    require(source_bytes == 2110696, "Unexpected restored source byte count")

    baseline_path = ROOT / "analysis/geometric_fault_transport_validation/legacy_slab_constraint_baseline_receipt.json"
    baseline = read_json(baseline_path)
    require(baseline.get("source_identity_matches") is True and baseline.get("base_commit") == BASE_COMMIT
            and baseline.get("status") == "known_original_head_failure_reproduced_identically",
            "The legacy failure has no original-HEAD reproduction receipt")
    require(len(baseline["source_sha256"]) == 3, "Unexpected legacy baseline source inventory")
    for name, hashes in baseline["source_sha256"].items():
        require(hashes["working_tree"] == hashes["original_head"] == sha((ROOT / name).read_bytes()),
                f"Legacy baseline solver, test or fixture changed: {name}")
    dependency_path = HERE / "dependency_test_receipt.json"
    dependency = read_json(dependency_path)
    dependency_log_path = ROOT / dependency["log"]
    dependency_log = dependency_log_path.read_text(encoding="utf-8")
    dependency_match = re.search(r"^1 failed, 223 passed in ([0-9.]+)s$", dependency_log, re.MULTILINE)
    failed_tests = re.findall(r"^FAILED (\S+)", dependency_log, re.MULTILINE)
    require(dependency.get("exit_code") == 1 and dependency_match is not None
            and failed_tests == [baseline["test"]],
            "Expanded dependencies have a failure other than the known original-HEAD result")

    core = report["final_state"]
    require(checkpoints["direct_final.json"]["provenance"]["segmented_bending"] == core,
            "The reported core endpoint differs from its checkpoint")
    exact_work = math.fsum([core["initial_mechanics"]["cumulative_source_work_j"]]
                          + [segment["source_work_j"] for segment in core["segments"]])
    require(exact_work == report["source_work"]["exact_seed_plus_segment_work_j"]
            and math.isclose(exact_work, core["cumulative_source_work_j"], rel_tol=2e-14),
            "Changing-source accumulated work does not equal seed plus segment work")
    physical = report["accepted_physical_common_motion"]
    require(physical["accepted_physical_steps"] == report["accepted_physical_geometry_steps"] == 3
            and physical["cohort_count"] == 2 and physical["actual_archive_transactions_added"] == 3,
            "Unexpected accepted physical control extent")
    require(physical["restart_exact"] is True and physical["checkpoint_bytes_identical"] is True
            and physical["actual_surface_and_archive_are_matching_current_endpoint"] is True
            and physical["original_loss_payloads_and_material_properties_preserved"] is True
            and physical["differential_rejection_preserves_geometry_archive_mechanics_and_work"] is True,
            "Physical control lacks required conservation, restart or atomicity receipts")
    require(report["underthrust_admitted_material_km2"] == 0. and report["new_slab_cohorts"] == 0,
            "The control scope changed to material admission")
    for control in (physical, report["nonempty_passive_archive"]):
        require(set(control["material_total_residual"]) == set(RESERVOIRS)
                and all(control["material_total_residual"][key] == 0. for key in RESERVOIRS),
                "A physical or passive archive material balance is nonzero")
    joint = checkpoints["physical_direct_final.json"]
    motion = joint["provenance"]["bending_motion"]
    require(joint["surface"] == motion["current_surface"] and joint["boundary"] == motion["current_inventory"]
            and joint["surface"]["time_myr"] == joint["boundary"]["time_myr"],
            "The physical joint checkpoint does not contain the actual current pair")
    require(git("branch", "--show-current") == EXPECTED_BRANCH,
            "Validation must use the requested branch")
    validated_head = git("rev-parse", "HEAD")
    subprocess.run(["git", "diff", "--check"], cwd=ROOT, check=True)

    return dict(
        format="geometric-segmented-bending-cloud-validation-1",
        status="complete_with_known_original_head_legacy_failure",
        date=audit["audit_utc"][:10], branch=EXPECTED_BRANCH, base_commit=BASE_COMMIT,
        validated_head=validated_head,
        runtime=report["runtime"],
        cloud_sources=dict(files=9, source_bytes=source_bytes, verified=True,
                           bundle_sha256=report["cloud_source_bundle_sha256"]),
        checks=dict(
            focused_cloud_profile=dict(passed=int(match.group(1)), seconds=float(match.group(2)),
                runner_seconds=receipt["seconds"], log=relative(log_path), receipt=relative(test_receipt_path),
                code_and_tests_unchanged=True, bound_code_and_test_file_count=profile_hash_count),
            expanded_legacy_dependencies=dict(passed=223, failed=1, seconds=float(dependency_match.group(1)),
                runner_seconds=dependency["seconds"], log=relative(dependency_log_path),
                receipt=relative(dependency_path), baseline_reproduction_receipt=relative(baseline_path)),
            new_targeted_tests_included_in_profile=dict(total=sum(NEW_TEST_COUNTS.values()), **NEW_TEST_COUNTS),
            git_diff_check="passed", all_current_control_code_hashes_match=True,
            current_control_code_hash_count=154, checkpoint_hash_count=10,
            all_checkpoint_payload_digests_match=True, all_three_restart_pairs_byte_identical=True),
        control=dict(
            report=relative(report_path), report_sha256=sha(report_bytes), report_byte_count=len(report_bytes),
            independent_recovery_audit_receipt=relative(audit_path), full_output_directory=relative(output),
            elapsed_seconds=report["elapsed_seconds"], point_count=report["point_count"],
            changing_source_segments=report["source_segment_count"], retained_seed=report["retained_seed"],
            source_work=report["source_work"], sliding_point_steps=report["sliding_point_steps"],
            mixed_branch_trace_steps=report["mixed_branch_trace_steps"],
            maximum_resultant_torque_relative_residual=report["maximum_resultant_torque_relative_residual"],
            maximum_generalized_force_relative_residual=report["maximum_generalized_force_relative_residual"],
            maximum_incremental_work_relative_residual=report["maximum_incremental_work_relative_residual"],
            maximum_physical_virtual_work_relative_error=report["maximum_physical_virtual_work_relative_error"],
            independent_locked_continuum_control=report["independent_locked_continuum_control"],
            passive_cartesian_covariance_control=report["passive_cartesian_covariance_control"],
            nonempty_passive_archive=report["nonempty_passive_archive"],
            accepted_physical_common_motion={key: value for key, value in physical.items() if key != "steps"},
            accepted_physical_geometry_steps=3, actual_physical_cohort_count=2,
            underthrust_admitted_material_km2=0., new_slab_cohorts=0,
            all_four_material_total_residuals_zero=True, restart_pairs=restarts),
        known_original_head_failure=dict(
            test=baseline["test"], source_identity_matches=True, unchanged_solver_test_fixture_count=3,
            primal_relative_violation=baseline["primal_relative_violation"],
            test_primal_tolerance=baseline["test_primal_tolerance"],
            production_primal_tolerance=baseline["production_primal_tolerance"],
            independent_primal_oracle_success=baseline["independent_primal_oracle_success"],
            baseline_receipt=relative(baseline_path), interpretation=baseline["interpretation"]),
        original_six_step_plan=[
            dict(step=1, status="partial", scope="Bending and transverse deformation, physical fault motion and spatial underthrust admission"),
            dict(step=2, status="pending", scope="Couple geometric events and cooling in one simulation clock"),
            dict(step=3, status="pending", scope="Full slab material dynamics, stretching, thinning, breakoff, slab pull and ridge forces"),
            dict(step=4, status="pending", scope="Stress-driven fracture, new faults, history transfer and topology changes"),
            dict(step=5, status="pending", scope="Main loop and GUI integration, compatible save formats"),
            dict(step=6, status="pending", scope="Long runs, time and mesh convergence, performance optimization"),
        ],
        scope=report["scope"], limitations=audit["limits"],
        next_step="Spatial material-pair matching and differential underthrust admission, preserving existing force/source work and history before conservative acceptance.",
        evidence_sha256={relative(path): sha(path.read_bytes()) for path in
            (test_receipt_path, log_path, dependency_path, dependency_log_path, baseline_path, audit_path,
             Path(__file__).resolve())},
        notes=["New targeted counts are included in the 1039-test profile; counts overlap.",
               "The final control finished during a transport outage and was independently audited after environment recovery; it was not rerun.",
               "Partial and interrupted profile logs are not accepted as final validation.",
               "All changes remain local in the requested branch; no commit or push was performed."],
    )


def main():
    try:
        summary = summarize()
        target = HERE / "validation_summary.json"
        target.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"Validation summary not published: {error}", file=sys.stderr)
        return 1
    print(json.dumps(dict(status=summary["status"],
                          focused_passed=summary["checks"]["focused_cloud_profile"]["passed"],
                          new_targeted_included=sum(NEW_TEST_COUNTS.values()),
                          summary=relative(target)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
