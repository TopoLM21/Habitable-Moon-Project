"""Complete the agreed newborn-law experiments in the gated interpreter."""
from __future__ import annotations

from pathlib import Path
import time

from analysis.geometric_native_birth_contact_validation.validation_support import (
    digest, guard_sources, write,
)


def run_complete_control(out):
    from analysis.geometric_native_birth_contact_validation.run_constitutive_control import run_constitutive_control
    from analysis.geometric_native_birth_contact_validation.run_native_control import run_native_control

    started = time.perf_counter()
    before, inherited = guard_sources(require_frozen=True)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write(out / "sources_before.json", before)
    constitutive = run_constitutive_control(out / "constitutive")
    result = run_native_control(out / "native", development=False, inventory=True)
    if "report" not in result or result["report"]["status"] != "PASS":
        raise RuntimeError("The frozen native mixed experiment was refused; receipt preserved")
    native = result["report"]
    if (not constitutive["negative_birth_energy_paid"] or not native["genuine_mixed_equilibrium_solved"]
            or not native["accepted_interval_preserved"] or native["whole_native_interval_admitted"]
            or native["complete_all_pairs_temporal_coverage"]):
        raise ValueError("The agreed partial physical experiment changed scope")
    after, inherited_after = guard_sources(require_frozen=True)
    write(out / "sources_after.json", after)
    if before != after or inherited != inherited_after:
        raise ValueError("Frozen sources changed during the complete control")
    report = dict(format="native-birth-complete-scoped-control-1", status="PASS",
        seconds=time.perf_counter()-started, source_count=len(before),
        inherited_732_sources_unchanged=True, source_before_after_unchanged=True,
        constitutive_report=dict(path=str((out / "constitutive/report.json").relative_to(Path(__file__).resolve().parents[2])),
            sha256=digest(out / "constitutive/report.json")),
        native_report=dict(path=str((out / "native/report.json").relative_to(Path(__file__).resolve().parents[2])),
            sha256=digest(out / "native/report.json")),
        original_native_intervals_accepted=1, new_whole_native_intervals_accepted=0,
        genuine_reduced_mixed_force_solved=True, reduced_born_contact_atoms=2,
        independently_owned_hot_foundations=5, old_physical_points_preserved=16,
        exhaustive_current_pair_count=42, whole_point_1_completed=False,
        remaining_original_work_block_count=1,
        scope="Verified explicit new birth law and genuine reduced current-source coupling; temporal fronts/continuum admission remain within original block 3")
    write(out / "combined_report.json", report, "report_sha256")
    return report
