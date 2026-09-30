"""Reproducible final evidence for local polarity and passive geometric slabs."""
import hashlib
import json
from pathlib import Path
import re

from verify_baseline import verify

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    baseline = verify(ROOT)
    for category in ("production_sha256", "source_sha256", "previous_geometry_tests_sha256"):
        assert not baseline[category]["changed"] and not baseline[category]["missing"]
    assert set(baseline["previous_geometry_sha256"]["changed"]) == {"tectonics/geometric_transport.py"}
    toy = read(OUT/"passive_connection_case/report.json")
    actual = read(OUT/"saved50_inherited_final/report.json")
    assert toy["status"] == "complete" and toy["exact_restart"] and toy["code_unchanged"]
    assert actual["source_unchanged"] and not actual["code_changed_during_run"]
    for report in (toy, actual):
        hashes = report.get("code_sha256", report.get("production_sha256"))
        assert all(sha(ROOT/path) == expected for path, expected in hashes.items())
    content = (OUT/"final_tests.log").read_bytes()
    log = content.decode("utf-16" if content[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig")
    result = re.search(r"(\d+) passed in ([\d.]+)s", log)
    assert result is not None and "failed" not in log.lower()
    reasons = {}
    for row in (actual.get("unresolved") or {}).get("overlaps", []):
        reasons[row["reason"]] = reasons.get(row["reason"], 0)+1
    output = dict(scope="Local polarity memory and passive slab cohorts; no initiation or force feedback",
        tests={"passed": int(result[1]), "seconds": float(result[2]), "log_sha256": sha(OUT/"final_tests.log")},
        baseline=baseline, synthetic_case=toy,
        actual_source_case={"path": "saved50_inherited_final/report.json", "sha256": sha(OUT/"saved50_inherited_final/report.json"),
            "status": actual["status"], "completed_steps": actual["completed_steps"],
            "unresolved_reasons": reasons, "source_unchanged": actual["source_unchanged"],
            "code_changed_during_run": actual["code_changed_during_run"], "wall_seconds": actual["wall_seconds"]},
        source_audit_sha256=sha(OUT/"source_polarity_audit.json"),
        source_mapping_sha256=sha(OUT/"source_polarity_mapping.json"),
        production_sha256={path.relative_to(ROOT).as_posix(): sha(path) for path in
            [ROOT/"run_geometric_slab_probe.py", *sorted((ROOT/"tectonics").glob("geometric_*.py"))]})
    (OUT/"validation_summary.json").write_text(json.dumps(output, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"tests": output["tests"], "source_case": output["actual_source_case"]}, indent=2))


if __name__ == "__main__":
    main()
