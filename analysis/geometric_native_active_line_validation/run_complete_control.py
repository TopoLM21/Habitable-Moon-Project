"""Shared-process line and genuine force controls with separate acceptance scope."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time

from .validation_support import digest, guard_sources, sealed, write


def run_complete_control(out):
    """Keep the continuum ledger separate from the accepted Gauss mechanics."""
    from .run_active_line_control import run_control
    from .run_implicit_motion_control import run_implicit_motion_control

    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    before, immutable = guard_sources(require_frozen=True)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write(out / "sources_before.json", before)
    run_control(out / "line")
    run_implicit_motion_control(out / "motion")
    line_path = out / "line/report.json"
    motion_path = out / "motion/motion_control_report.json"
    line = sealed(line_path, "report_sha256")
    motion = sealed(motion_path, "report_sha256")
    if line["status"] != "PASS" or line["accepted_native_force_intervals"] != 0:
        raise ValueError("The line control changed its diagnostic acceptance scope")
    required = ("native_force_admission_proved", "full_first_interval_balances_passed",
        "subsequent_candidate_solved", "subsequent_candidate_refused",
        "subsequent_force_balances_passed", "refusal_preserved_before_bytes",
        "positive_newborn_contact_proved", "joint_restart_passed",
        "joint_restart_repeats_original_genuine_force", "complete_balances_passed")
    if (motion["status"] != "PASS" or motion["accepted_loaded_native_intervals"] != 1
            or any(motion[name] is not True for name in required)
            or motion["subsequent_interval_accepted"] is not False
            or motion["subsequent_intervals_passed"] is not False
            or motion["entire_point_1_completed"] is not False
            or motion["physical_inputs_invented"] is not False
            or motion["continuum_work_added_to_Gauss_ledger"] is not False):
        raise ValueError("The force control changed its first-accepted, second-refused scope")
    after, immutable_after = guard_sources(require_frozen=True)
    if after != before or immutable_after != immutable:
        raise ValueError("Sources changed during the combined physical control")
    write(out / "sources_after.json", after)
    report = dict(format="native-active-line-complete-scoped-control-1", status="PASS",
        started_utc=started_utc, finished_utc=datetime.now(timezone.utc).isoformat(),
        seconds=time.perf_counter()-started, source_count=len(before),
        prior_704_python_sources_unchanged=True, source_before_after_unchanged=True,
        sources_before_sha256=digest(out / "sources_before.json"),
        sources_after_sha256=digest(out / "sources_after.json"),
        line_control=dict(path="line/report.json", sha256=digest(line_path)),
        motion_control=dict(path="motion/motion_control_report.json", sha256=digest(motion_path)),
        accepted_loaded_native_intervals=1, subsequent_candidate_solved=True,
        subsequent_candidate_refused=True, subsequent_interval_accepted=False,
        joint_restart_passed=True, continuous_work_added_to_Gauss_ledger=False,
        entire_point_1_completed=False,
        remaining_block=motion["remaining_block"],
        scope="whole original receiving line and supports; one accepted physical interval; genuine second candidate refused on newborn contact")
    return write(out / "combined_report.json", report, "report_sha256")


run_control = run_complete_control
