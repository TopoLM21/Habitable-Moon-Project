"""Verify final artifacts against current code and summarize reproducible checks."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((HERE/name).read_text(encoding="utf-8"))


def main():
    baseline = read("preexisting_code_sha256.json")
    changed = [name for name, expected in baseline.items()
               if not (ROOT/name).is_file() or digest(ROOT/name) != expected]
    assert not changed, changed
    positive = read("admission_case/report.json")
    actual = read("saved50_final/report.json")
    audit = read("source_mechanics_audit.json")
    for manifest in (positive["code_sha256"], actual["production_sha256"], audit["production_sha256"]):
        for name, expected in manifest.items():
            assert digest(ROOT/name) == expected, name
    assert positive["code_unchanged"] and not actual["code_changed_during_run"]
    assert actual["source_unchanged"] and positive["source_preserved"]
    for name, expected in actual["input_sha256"].items():
        assert digest(Path(name)) == expected, name
    for source in audit["sources"].values():
        assert source["source_hashes_unchanged"]
        for name, expected in source["provenance"]["source_sha256"].items():
            assert digest(Path(name)) == expected, name
    assert positive["exact_restart"] and positive["checkpoint_bytes_identical"]
    for name, expected in positive["checkpoint_sha256"].items():
        assert digest(HERE/"admission_case"/name) == expected
    assert actual["completed_steps"] == 0 and actual["checkpoint_file"] is None
    assert actual["passive_boundary"]["cohort_count"] == 0
    assert not actual["passive_boundary"]["forces_active"]
    assert all(case["status"] == "blocked" and case["material_unchanged"] and case["birth_calls"] == 0
               for case in positive["controls"].values())
    log = HERE/"final_tests.log"
    raw = log.read_bytes()
    text = raw.decode("utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig")
    count, seconds = re.findall(r"(\d+) passed in ([\d.]+)s", text)[-1]
    assert not re.search(r"\d+ failed|\d+ errors", text)
    result = dict(format="geometric-initiation-validation-summary-1",
        tests=dict(passed=int(count), seconds=float(seconds), log_sha256=digest(log)),
        previous_code=dict(files_checked=len(baseline), changed=changed),
        positive_case=dict(path="admission_case/report.json", sha256=digest(HERE/"admission_case/report.json"),
            cohorts=positive["accepted_boundary"]["cohort_count"], removed=positive["removed"],
            attachment_counts=positive["accepted_boundary"]["attachment_counts"],
            maximum_relative_material_ledger_residuals=positive["maximum_relative_material_ledger_residuals"],
            exact_restart=True, checkpoint_bytes_identical=True, code_unchanged=True,
            blocked_controls=list(positive["controls"])),
        real_source=dict(path="saved50_final/report.json", sha256=digest(HERE/"saved50_final/report.json"),
            status=actual["status"], completed_steps=actual["completed_steps"],
            contact_status_counts=actual["initiation"]["contact_status_counts"],
            overlap_reasons=dict(Counter(item["reason"] for item in actual["unresolved"]["overlaps"])),
            source_unchanged=True, code_unchanged=True, wall_seconds=actual["wall_seconds"],
            checkpoint_file=None, forces_active=False),
        source_audit=dict(path="source_mechanics_audit.json", sha256=digest(HERE/"source_mechanics_audit.json")),
        production_sha256=actual["production_sha256"])
    (HERE/"validation_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("tests", "previous_code", "real_source")}, indent=2))


if __name__ == "__main__":
    main()
