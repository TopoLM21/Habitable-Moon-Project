"""Summarize measured cases and prove old production/source files unchanged."""
import hashlib
import json
from pathlib import Path
import re

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    baseline = json.loads((OUT/"baseline.json").read_text(encoding="utf-8"))
    changed = {category: [name for name, expected in baseline[category].items()
                          if not (ROOT/name).is_file() or sha(ROOT/name) != expected]
               for category in ("production_sha256", "source_sha256")}
    cases = {path.parent.name: json.loads(path.read_text(encoding="utf-8"))
             for path in OUT.glob("*_common_sub*/report.json")}
    differential = {}
    for path in OUT.glob("*_actual_motion_final/report.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        differential[path.parent.name] = dict(
            status=record["status"], completed_steps=record["completed_steps"],
            unresolved_overlap_count=len(record.get("unresolved", {}).get("overlaps", [])),
            source_unchanged=record["source_unchanged"], code_changed_during_run=record["code_changed_during_run"],
            cumulative_losses=record["cumulative_losses"], cumulative_births=record["cumulative_births"],
            report_sha256=sha(path), wall_seconds=record["wall_seconds"],
            production_sha256=record["production_sha256"])
    artifacts = list(OUT.glob("*.png"))+list(OUT.glob("*_common_sub*/report.json"))
    static = None
    static_path = OUT/"static_contact_probe_final.json"
    if static_path.is_file():
        static_record = json.loads(static_path.read_text(encoding="utf-8"))
        static = {name: {key: value for key, value in record.items() if key != "source"}
                  for name, record in static_record.items()}
        artifacts.append(static_path)
    tests = None
    test_log = OUT/"final_tests.log"
    if test_log.is_file():
        matches = re.findall(r"(\d+) passed in ([\d.]+)s", test_log.read_text(encoding="utf-8"))
        if matches:
            count, seconds = matches[-1]
            tests = dict(passed=int(count), elapsed_seconds=float(seconds),
                         log=test_log.name, sha256=sha(test_log))
    result = dict(
        scope="Persisted polygon/contact geometry with fixed velocities; no slab, ridge or thermal force integration",
        existing_production_files_changed=changed["production_sha256"],
        original_source_files_changed=changed["source_sha256"],
        existing_production_unchanged=not changed["production_sha256"],
        original_sources_unchanged=not changed["source_sha256"],
        baseline_production_files=len(baseline["production_sha256"]),
        baseline_source_files=len(baseline["source_sha256"]),
        tests=tests,
        final_actual_source_contact_geometry=static,
        production_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in
                           sorted((ROOT/"tectonics").glob("geometric*.py"))
                           +[ROOT/"tectonics/spherical_polygons.py", ROOT/"run_geometric_contact_probe.py"] if p.is_file()},
        completed_common_rotation_cases=cases,
        actual_differential_motion_preflight=differential,
        preserved_incomplete_case_folders=[path.name for path in OUT.glob("*_common_sub*")
                                          if path.is_dir() and not (path/"report.json").exists()],
        artifact_sha256={p.relative_to(OUT).as_posix(): sha(p) for p in artifacts},
    )
    assert result["existing_production_unchanged"] and result["original_sources_unchanged"], changed
    (OUT/"validation_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"cases": list(cases), "existing_production_unchanged": result["existing_production_unchanged"],
                      "original_sources_unchanged": result["original_sources_unchanged"]}))


if __name__ == "__main__":
    main()
