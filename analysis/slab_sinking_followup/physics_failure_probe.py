"""Read-only frozen force/failure probes with source and code provenance."""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tectonics.plate_velocity_diagnostics import diagnose_checkpoint


def hashes(root):
    saved = json.loads((root/"continuation.json").read_text(encoding="utf-8"))
    return {name: sha256((root/name).read_bytes()).hexdigest() for name in saved["checkpoint_sha256"]}


def main():
    cases = [
        (50, "validated_sinking_sub4_dt1"), (100, "validated_sinking_sub4_dt1"),
        (200, "validated_sinking_fixed2_sub4_dt1"), (400, "validated_sinking_fixed2_sub4_dt1")]
    rows = []
    for age, arm in cases:
        root = ROOT/"analysis/slab_sinking_validation/runs"/arm/f"elapsed_{age:04d}"
        before = hashes(root)
        report = diagnose_checkpoint(root)
        assert hashes(root) == before
        failures = report["trace"]["slab_neck_failures"]
        for failure in failures:
            tension = failure["tension_n"]
            reaction = failure["no_eduction_reaction_n"]
            failure["no_eduction_reaction_fraction_of_tension"] = reaction/tension
            failure["would_fail_without_no_eduction_reaction_at_this_solution"] = (
                tension-reaction > failure["capacity_n"])
        rows.append(dict(elapsed_myr=age, source=str(root), source_sha256=before,
            actual_time_myr=report["time_myr"], failures=failures,
            failure_count=len(failures),
            failures_without_reaction=sum(f["would_fail_without_no_eduction_reaction_at_this_solution"] for f in failures),
            final_sections=report["trace"]["slab_sections"],
            final_force_speed_mm_yr=report["stages_speed_km_per_myr"]["target"],
            gravity_m_s2=report["controls"]["dynamics"]["gravity_m_s2"],
            source_shell=report["controls"]["shell"], source_starter=report["controls"]["starter"]))
    output = dict(interpretation="Frozen read-only probes of saved 0.4 states, not evolved0.5 outcomes or decomposition of the original1023 failures.",
        production_sha256={name:sha256((ROOT/name).read_bytes()).hexdigest() for name in (
            "tectonics/young_plate_dynamics.py", "tectonics/young_slab_sinking.py", "tectonics/young_slab_constraints.py")},
        probes=rows)
    path = Path(__file__).with_suffix(".json")
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps([dict(age=r["elapsed_myr"], failures=r["failure_count"], without_reaction=r["failures_without_reaction"]) for r in rows]))


if __name__ == "__main__":
    main()
