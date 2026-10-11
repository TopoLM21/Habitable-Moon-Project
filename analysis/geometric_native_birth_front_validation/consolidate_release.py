"""Seal one current scope gate, actual source experiments and immutable ancestry."""
from __future__ import annotations

from pathlib import Path

from .validation_support import (BASE,ROOT,SUMMARY,digest,guard_sources,read,sealed,verify_baseline,write)


TEST_PATHS={
    "tests/test_geometric_native_birth_contact_continuum.py",
    "tests/test_geometric_native_birth_contact_front.py",
    "tests/test_geometric_native_birth_pair_exclusion.py",
    "tests/test_geometric_native_birth_curved_pair_support.py",
    "tests/test_geometric_native_birth_continuum_motion.py",
    "tests/test_geometric_native_birth_finite_pose.py",
    "tests/test_geometric_native_birth_independent_audits.py",
}


def reference(path,**extra):
    path=Path(path)
    return dict(path=str(path.resolve().relative_to(ROOT)),sha256=digest(path),**extra)


def check(value,expected):
    if any(value.get(name)!=actual for name,actual in expected.items()):
        raise ValueError("The scoped release evidence differs: "+value.get("format","unknown"))


def consolidate():
    verify_baseline()
    current,immutable=guard_sources(require_frozen=True)
    freeze=sealed(BASE/"frozen_release_receipt.json","receipt_sha256")
    check(freeze,dict(status="PASS",inherited_python_source_count=751,frozen_source_count=len(current),
        inherited_sources_changed=[],added_python_sources=sorted(set(current)-set(immutable))))
    if digest(BASE/"frozen_release_sources.json")!=freeze["frozen_sources_sha256"]:
        raise ValueError("The complete frozen source map raw bytes differ")
    gate_dir,control_dir,audit_dir=BASE/"release_validation_01",BASE/"control_release_01",BASE/"independent_release_01"
    gate=sealed(gate_dir/"receipt.json","receipt_sha256")
    check(gate,dict(status="PASS",source_count=len(current),inherited_source_count=751,changed_paths=[],
        exit_code=0,canonical_pytest_import_mode="importlib",historical_69_tests_reexecuted=False,
        historical_137_tests_reexecuted=False,whole_native_interval_admitted_by_test_gate=False))
    if set(gate["test_paths"])!=TEST_PATHS or gate["counts"]!={"passed":135}:
        raise ValueError("The agreed complete current 135-test gate did not run")
    for path in (gate_dir/"sources_before.json",gate_dir/"sources_after.json",gate_dir/"suite_sources_after.json",
            control_dir/"sources_before.json",control_dir/"sources_after.json"):
        if read(path)!=current:
            raise ValueError("A current execution source map differs from the freeze")
    suite=sealed(gate_dir/"suite_receipt.json","receipt_sha256")
    check(suite,dict(status="PASS",source_count=len(current),source_before_after_unchanged=True))
    if suite["pytest_gate_receipt_sha256"]!=digest(gate_dir/"receipt.json"):
        raise ValueError("The genuine same-process control references another gate")
    control=sealed(control_dir/"report.json","report_sha256")
    check(control,dict(status="PASS",source_count=len(current),inherited_source_count=751,
        accepted_original_native_intervals=1,accepted_new_native_intervals=0,whole_point_1_completed=False,
        original_point_count=16,full_continuum_arc_count=2,born_midpoint_atoms=0,foundation_patch_count=5,
        coordinate_count=19,complete_first_accepted_history_unchanged=True,
        complete_geometric_pair_count=42,causal_hot_front_whole_clock_proved=True,
        native_admission_status="REFUSED",whole_native_interval_admitted=False,
        all_positive_pair_contact_time_domains_proved=False,born_finite_pose_force_equilibrium_proved=False))
    classification=sealed(control_dir/"complete_geometric_pair_classification.json","classification_sha256")
    check(classification,dict(status="PASS",pair_count=42,hot_pair_count=26,
        classification_counts={"excluded_for_whole_interval":33,"positive_material_subset":7,
            "positive_interior_material_subset":2},
        hot_classification_counts={"excluded_for_whole_interval":24,"positive_material_subset":2,
            "positive_interior_material_subset":0},native_mechanical_admission_authorized=False))
    for directory,report in ((control_dir/"causal_front",sealed(control_dir/"causal_front/report.json","report_sha256")),
            (control_dir/"pair_exclusions",sealed(control_dir/"pair_exclusions/report.json","report_sha256"))):
        if report["status"]!="PASS":
            raise ValueError("A complete source replay subcontrol failed")
        for first,second in report["byte_pairs"]:
            if (directory/first).read_bytes()!=(directory/second).read_bytes():
                raise ValueError("A complete source replay byte pair differs")
    audits=sealed(audit_dir/"report.json","report_sha256")
    check(audits,dict(status="PASS",source_count=len(current),audit_count=4,
        fresh_process_without_production_modules=True,whole_native_interval_admitted=False))
    if audits["input_candidate"]["sha256"]!=digest(control_dir/"candidate_packet.json"):
        raise ValueError("The fresh independent audit used another force packet")
    for name,sha in audits["auditor_sources"].items():
        if current.get(name)!=sha:
            raise ValueError("An independent auditor differs from the complete current freeze")
    for row in audits["independent_audits"]:
        if digest(ROOT/row["path"])!=row["sha256"]:
            raise ValueError("An independent audit receipt raw reference differs")
        sealed(ROOT/row["path"],"receipt_sha256")
    review=sealed(BASE/"mechanical_admission_review.json","review_sha256")
    if review["status"]!="ACTUAL_COLD_INCOMING_UP_DIP_COUNTEREXAMPLE":
        raise ValueError("The preserved actual mechanical counterexample was lost")
    equivalence=sealed(control_dir/"mechanical_review_authority_equivalence.json","equivalence_sha256")
    check(equivalence,dict(status="PASS",entire_candidate_physical_payload_identical=True,
        entire_first_entry_slice_physical_payload_identical=True,
        final_force_candidate_sha256=control["candidate_sha256"],native_mechanical_admission_authorized=False))
    for path in BASE.rglob("additive_source_archive.json"):
        archive=sealed(path,"archive_sha256")
        if archive["count"]!=len(archive["sources"]):
            raise ValueError("An additive development source archive count differs")
        for row in archive["sources"].values():
            archived=ROOT/row["path"]
            if digest(archived)!=row["sha256"] or archived.stat().st_size!=row["bytes"]:
                raise ValueError("A preserved exact development source body changed")
    prior_summary=sealed(SUMMARY,"summary_sha256")
    prior_manifest=sealed(ROOT/prior_summary["artifact_manifest"]["path"],"manifest_sha256")
    artifacts=dict(prior_manifest["artifacts"])
    excluded={gate_dir/"artifact_manifest.json",gate_dir/"validation_summary.json"}
    paths=[path for path in BASE.rglob("*") if path.is_file() and "__pycache__"not in path.parts
        and path.suffix!=".pyc" and path not in excluded]
    paths.extend(ROOT/name for name in set(current)-set(immutable))
    paths.extend(path for path in (ROOT/"GEOMETRIC_NATIVE_BIRTH_FRONT_IMPLEMENTATION.md",
        ROOT/"analysis/geometric_native_birth_contact_validation/front_prerelease_tests_01.log") if path.is_file())
    for path in paths:
        name=str(path.relative_to(ROOT))
        record=dict(bytes=path.stat().st_size,sha256=digest(path))
        if name in artifacts and artifacts[name]!=record:
            raise ValueError("A current artifact replaced an immutable inherited artifact")
        artifacts[name]=record
    manifest_path=gate_dir/"artifact_manifest.json"
    write(manifest_path,dict(format="native-birth-front-artifact-manifest-1",count=len(artifacts),
        unique_artifact_count=len(artifacts),inherited_unique_artifact_count=1271,
        artifacts=dict(sorted(artifacts.items())),whole_native_interval_admitted=False,whole_point_1_completed=False,
        excluded_self_referential_documents=[str(path.relative_to(ROOT)) for path in sorted(excluded)]),"manifest_sha256")
    return write(gate_dir/"validation_summary.json",dict(format="native-birth-front-scoped-validation-summary-1",status="PASS",
        source_count=len(current),immutable_inherited_source_count=751,inherited_sources_changed=[],
        added_python_sources=sorted(set(current)-set(immutable)),inherited_validation=reference(SUMMARY),
        inherited_artifact_manifest=prior_summary["artifact_manifest"],verified_inherited_unique_artifacts=1271,
        historical_69_tests_reexecuted=False,historical_137_tests_reexecuted=False,
        source_freeze=reference(BASE/"frozen_release_receipt.json"),frozen_source_map=reference(BASE/"frozen_release_sources.json"),
        fresh_scoped_gate=reference(gate_dir/"receipt.json",tests_passed=135,seconds=gate["seconds"],test_files=7),
        same_process_suite=reference(gate_dir/"suite_receipt.json"),actual_control=reference(control_dir/"report.json"),
        independent_audit_suite=reference(audit_dir/"report.json"),independent_audits=audits["independent_audits"],
        actual_mechanical_counterexample_review=reference(BASE/"mechanical_admission_review.json"),
        mechanical_review_current_physical_authority_equivalence=reference(control_dir/"mechanical_review_authority_equivalence.json"),
        mechanical_counterexample_uses_preserved_development_full_arc_candidate=True,
        artifact_manifest=reference(manifest_path,count=len(artifacts)),original_native_intervals_accepted=1,
        new_whole_native_intervals_accepted=0,original_point_histories_preserved=True,
        genuine_full_continuum_current_source_force_solved=True,full_continuum_arc_count=2,
        born_midpoint_atoms=0,independent_foundation_count=5,complete_coordinate_count=19,
        paid_birth_activation_energy_decimal=control["diagnostics"]["born_activation_energy_decimal"],
        complete_geometric_pair_classification=reference(control_dir/"complete_geometric_pair_classification.json",
            pair_count=42,classification_counts=classification["classification_counts"],
            hot_classification_counts=classification["hot_classification_counts"]),
        causal_hot_shared_arc_whole_clock_proved=True,complete_all_pairs_temporal_constitutive_coverage=False,
        born_finite_pose_force_equilibrium_proved=False,whole_native_interval_admitted=False,whole_point_1_completed=False,
        original_work_block_count=3,completed_work_block_count=2,remaining_work_block_count=1,
        remaining_scope=control["remaining_physical_obligations"],
        original_six_point_plan="Point 1 partial; points 2 through 6 pending"),"summary_sha256")


if __name__=="__main__":
    result=consolidate()
    print({name:result[name] for name in ("status","source_count","fresh_scoped_gate","artifact_manifest")})
