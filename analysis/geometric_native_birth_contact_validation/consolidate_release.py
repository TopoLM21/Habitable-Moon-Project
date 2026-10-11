"""Seal the frozen scoped newborn experiments without widening their scope."""
from __future__ import annotations

from pathlib import Path

from analysis.geometric_native_birth_contact_validation.validation_support import (
    BASE, ROOT, SUMMARY, digest, guard_sources, read, sealed, value_sha, verify_baseline, write,
)


TEST_PATHS = {
    "tests/test_geometric_native_birth_contact.py",
    "tests/test_geometric_native_birth_contact_network.py",
    "tests/test_geometric_native_birth_contact_support.py",
}


def reference(path, **extra):
    return dict(path=str(Path(path).resolve().relative_to(ROOT)), sha256=digest(path), **extra)


def require(value, true=(), false=(), equal=None):
    if (any(value.get(name) is not True for name in true)
            or any(value.get(name) is not False for name in false)
            or any(value.get(name) != expected for name, expected in (equal or {}).items())):
        raise ValueError("Release evidence status or scope differs: " + value.get("format", "unknown"))


def check_map(path, expected):
    if read(path) != expected:
        raise ValueError("Execution source map differs from the declared freeze: " + str(path))


def check_pairs(directory, report):
    for row in report["byte_pairs"]:
        first, second = directory / row["first"], directory / row["second"]
        if (first.read_bytes() != second.read_bytes() or first.stat().st_size != row["bytes"]
                or digest(first) != row["sha256"]):
            raise ValueError("A complete restart or accepted-state preservation pair differs")


def check_development(immutable):
    rows = []
    for directory in sorted(BASE.glob("development_native_control_*")):
        before = read(directory / "sources_before.json")
        archive = sealed(directory / "development_source_archive.json", "archive_sha256")
        if (set(archive["sources"]) != set(before)-set(immutable)
                or archive["count"] != len(archive["sources"])):
            raise ValueError("A development trial lacks its complete additive body archive")
        for name, row in archive["sources"].items():
            path = ROOT / row["path"]
            if row["sha256"] != before[name] or digest(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
                raise ValueError("A development source body changed")
        path = directory / "failure_receipt.json"
        if path.is_file():
            receipt = sealed(path, "receipt_sha256")
            require(receipt, true=("inherited_732_sources_unchanged", "accepted_interval_preserved"),
                false=("final_release_gate", "parameters_automatically_changed", "whole_native_interval_admitted"),
                equal={"status": "REFUSED"})
            after_path = directory / "sources_after_refusal.json"
            if (digest(after_path) != receipt["source_map_after_sha256"]
                    or digest(directory / "failure_traceback.log") != receipt["traceback_sha256"]):
                raise ValueError("A development refusal's raw evidence changed")
            if (directory / "accepted_motion_before.json").read_bytes() != (directory / "accepted_motion_after_refusal.json").read_bytes():
                raise ValueError("A development refusal changed accepted physical history")
        else:
            path = directory / "report.json"
            receipt = sealed(path, "report_sha256")
            require(receipt, true=("development", "inherited_732_sources_unchanged", "genuine_mixed_equilibrium_solved"),
                false=("final_release_gate", "whole_native_interval_admitted", "parameters_automatically_changed"),
                equal={"status": "PASS"})
            after_path = directory / "sources_after.json"
            check_pairs(directory, receipt)
        after = read(after_path)
        if any(before.get(name) != sha or after.get(name) != sha for name, sha in immutable.items()):
            raise ValueError("A development attempt changed an inherited source")
        rows.append(dict(receipt=reference(path), additive_source_archive=reference(directory / "development_source_archive.json"),
            source_body_count=archive["count"], status=receipt["status"], final_release_gate=False,
            scope="Preserved development source version and actual numerical outcome; not the final passing gate"))
    return rows


def consolidate():
    verify_baseline()
    current, immutable = guard_sources(require_frozen=True)
    freeze = sealed(BASE / "frozen_release_receipt.json", "receipt_sha256")
    require(freeze, equal={"status": "PASS", "inherited_python_source_count": 732,
        "frozen_source_count": len(current), "inherited_sources_changed": [],
        "added_python_sources": sorted(set(current)-set(immutable))})
    if digest(BASE / "frozen_release_sources.json") != freeze["frozen_sources_sha256"]:
        raise ValueError("The frozen source map raw bytes differ")
    gate_dir, control_dir = BASE / "release_validation_01", BASE / "control_release_01"
    gate = sealed(gate_dir / "receipt.json", "receipt_sha256")
    require(gate, false=("is_full_profile_execution", "historical_tests_reexecuted", "whole_native_interval_admitted_by_test_gate"),
        equal={"status": "PASS", "source_count": len(current), "inherited_source_count": 732,
            "changed_paths": [], "exit_code": 0, "canonical_pytest_import_mode": "importlib"})
    if set(gate["test_paths"]) != TEST_PATHS or len(gate["test_paths"]) != 3 or gate["counts"].get("passed", 0) < 1:
        raise ValueError("The complete agreed three-module gate did not run")
    for name in ("sources_before.json", "sources_after.json", "suite_sources_after.json"):
        check_map(gate_dir / name, current)
    suite = sealed(gate_dir / "suite_receipt.json", "receipt_sha256")
    require(suite, true=("source_before_after_unchanged",), equal={"status": "PASS", "source_count": len(current)})
    if suite["pytest_gate_receipt_sha256"] != digest(gate_dir / "receipt.json"):
        raise ValueError("The same-process suite references another gate")
    combined = sealed(control_dir / "combined_report.json", "report_sha256")
    require(combined, true=("inherited_732_sources_unchanged", "source_before_after_unchanged", "genuine_reduced_mixed_force_solved"),
        false=("whole_point_1_completed",), equal={"status": "PASS", "new_whole_native_intervals_accepted": 0,
            "original_native_intervals_accepted": 1, "remaining_original_work_block_count": 1})
    for name in ("sources_before.json", "sources_after.json"):
        check_map(control_dir / name, current)
    local_dir, native_dir = control_dir / "constitutive", control_dir / "native"
    local = sealed(local_dir / "report.json", "report_sha256")
    native = sealed(native_dir / "report.json", "report_sha256")
    require(local, true=("negative_birth_energy_paid", "virgin_shear_work_zero", "opening_reclosure_proved",
        "tiny_positive_areas_retained", "foundation_counted_once", "inherited_732_sources_unchanged", "final_release_gate"),
        false=("development", "genuine_native_source_used", "whole_native_interval_admitted"), equal={"status": "PASS"})
    require(native, true=("final_release_gate", "source_before_after_unchanged", "genuine_current_mantle_source_used",
        "genuine_mixed_equilibrium_solved", "accepted_interval_preserved", "declaration_fixed_before_force_result",
        "unchanged_old_coordinate_prefix_changed_by_born_forces"),
        false=("development", "continuum_quadrature_error_certified", "complete_all_pairs_temporal_coverage",
            "whole_native_interval_admitted", "point_1_complete", "parameters_automatically_changed"),
        equal={"status": "PASS", "original_point_count": 16, "original_coordinate_count": 14,
            "mixed_coordinate_count": 19, "born_contact_count": 2, "independent_foundation_count": 5,
            "original_accepted_interval_count": 1})
    for directory, report in ((local_dir, local), (native_dir, native)):
        check_pairs(directory, report)
        for name in ("sources_before.json", "sources_after.json"):
            check_map(directory / name, current)
        if digest(directory / "constitutive_packet.json") != report["constitutive_packet_sha256"]:
            raise ValueError("A constitutive packet raw reference differs")
    inventory = read(native_dir / "mixed_pair_inventory.json")
    inventory_body = dict(inventory["inventory"])
    expected = inventory_body.pop("inventory_sha256")
    if expected != value_sha(inventory_body) or inventory_body["exhaustive_canonical_pair_count"] != 42:
        raise ValueError("The exhaustive current pair inventory seal or count differs")
    require(inventory_body, false=("all_pairs_constitutively_admitted", "candidate_mechanics_adopted", "force_admission_authorized"),
        equal={"omega_rad_per_myr": native["force_candidate_omega_rad_per_myr"]})
    audits = []
    for path in (local_dir / "independent_energy_receipt.json", native_dir / "independent_energy_receipt.json",
            native_dir / "independent_network_receipt.json", native_dir / "independent_baseline_network_receipt.json"):
        audit = sealed(path, "receipt_sha256")
        require(audit, false=("production_mechanics_imported", "whole_native_interval_admitted"), equal={"status": "PASS"})
        if digest(ROOT / audit["input_packet"]["path"]) != audit["input_packet"]["sha256"]:
            raise ValueError("An independent audit packet reference differs")
        for key in ("auditor_source", "independent_energy_oracle"):
            if key in audit and current.get(audit[key]["path"]) != audit[key]["sha256"]:
                raise ValueError("An independent auditor source does not match the complete final freeze")
        audits.append(reference(path, scope=audit["scope"]))
    development = check_development(immutable)
    prior_summary = sealed(SUMMARY, "summary_sha256")
    prior_manifest = sealed(ROOT / prior_summary["artifact_manifest"]["path"], "manifest_sha256")
    artifacts = dict(prior_manifest["artifacts"])
    excluded = {gate_dir / "artifact_manifest.json", gate_dir / "validation_summary.json"}
    paths = [path for path in BASE.rglob("*") if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc" and path not in excluded]
    paths.extend(ROOT / name for name in set(current)-set(immutable))
    model_doc = ROOT / "GEOMETRIC_NATIVE_BIRTH_CONTACT_IMPLEMENTATION.md"
    if model_doc.is_file():
        paths.append(model_doc)
    for path in paths:
        name = str(path.relative_to(ROOT))
        record = dict(bytes=path.stat().st_size, sha256=digest(path))
        if name in artifacts and artifacts[name] != record:
            raise ValueError("An inherited artifact was replaced by newborn evidence")
        artifacts[name] = record
    manifest = dict(format="native-birth-contact-artifact-manifest-1", count=len(artifacts),
        unique_artifact_count=len(artifacts), inherited_unique_artifact_count=1100,
        artifacts=dict(sorted(artifacts.items())), whole_native_interval_admitted=False,
        whole_point_1_completed=False,
        excluded_self_referential_documents=[str(path.relative_to(ROOT)) for path in sorted(excluded)])
    manifest_path = gate_dir / "artifact_manifest.json"
    write(manifest_path, manifest, "manifest_sha256")
    summary = dict(format="native-birth-contact-scoped-validation-summary-1", status="PASS",
        source_count=len(current), immutable_inherited_source_count=732, inherited_sources_changed=[],
        added_python_sources=sorted(set(current)-set(immutable)),
        inherited_validation=reference(SUMMARY), inherited_artifact_manifest=prior_summary["artifact_manifest"],
        verified_inherited_unique_artifacts=1100, historical_137_tests_reexecuted=False,
        source_freeze=reference(BASE / "frozen_release_receipt.json"), frozen_source_map=reference(BASE / "frozen_release_sources.json"),
        fresh_scoped_gate=reference(gate_dir / "receipt.json", tests_passed=gate["counts"]["passed"], seconds=gate["seconds"], test_files=3),
        same_process_suite=reference(gate_dir / "suite_receipt.json"), actual_control=reference(control_dir / "combined_report.json"),
        local_constitutive_control=reference(local_dir / "report.json"), native_reduced_control=reference(native_dir / "report.json"),
        independent_audits=audits, development_trials=development,
        artifact_manifest=reference(manifest_path, count=len(artifacts)),
        original_native_intervals_accepted=1, new_whole_native_intervals_accepted=0,
        explicit_birth_law_implemented=True, negative_birth_activation_energy_paid=True,
        born_foundations_counted_once=True, original_point_histories_preserved=True,
        genuine_reduced_current_source_force_solved=True,
        reduced_born_contact_atoms=2, independent_foundation_count=5, complete_coordinate_count=19,
        same_engine_old_coordinate_feedback_proved=True,
        whole_native_interval_admitted=False, continuum_quadrature_error_certified=False,
        complete_all_pairs_temporal_constitutive_coverage=False, whole_point_1_completed=False,
        current_pair_inventory=reference(native_dir / "mixed_pair_inventory.json", pair_count=42,
            classification_counts=inventory_body["classification_counts"]),
        baseline_positive_vertex_fronts_are_not_fresh_mixed_geometry=True,
        original_work_block_count=3, completed_work_block_count=2, remaining_work_block_count=1,
        remaining_scope="Within original block 3: certify continuum contact fronts/weights and birth-time mechanics for every actual possible pair, then admit a genuine material/force continuation and joint restart",
        original_six_point_plan="Point 1 partial; points 2 through 6 pending")
    return write(gate_dir / "validation_summary.json", summary, "summary_sha256")


if __name__ == "__main__":
    summary = consolidate()
    print({key: summary[key] for key in ("status", "source_count", "fresh_scoped_gate", "artifact_manifest",
        "current_pair_inventory", "remaining_work_block_count")})
