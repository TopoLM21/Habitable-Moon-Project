"""Recheck the saved 2026-09-03 numerical study and write a compact report.

Run after all study cases and the headless GUI smoke have finished. No input
checkpoints, images, or existing study outputs are modified.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.benchmark_cpu_modes import compare_checkpoints
from analysis.benchmark_render_modes import compare_pngs


def main():
    import numpy
    import scipy
    base = ROOT / "results/cpu_performance"
    studies = {}
    cases = (("numeric_cells_scaling_20260903", 12), ("numeric_cells_merger_20260903", 2),
             ("numeric_cells_split_20260903", 2), ("numeric_cells_stages_20260903", 2),
             ("numeric_cells_sub6_frame4_20260903", 3))
    for name, expected_count in cases:
        rows = json.loads((base / name / "measurements.json").read_text(encoding="utf-8"))
        if len(rows) != expected_count:
            raise RuntimeError(f"Incomplete study: {name}")
        compact = []
        for row in rows:
            if not (row["checkpoint_comparison"]["exact"] and row["png_comparison"]["png_exact"]):
                raise RuntimeError(f"Failed study case: {name}/{row['case']}")
            compact.append({key: row[key] for key in (
                "case", "numeric_mode", "wall_seconds", "instrumented", "cprofile",
                "exclusive_wall_seconds", "numerical_execution", "transport_coverage",
                "checkpoint_comparison", "png_comparison")})
        studies[name] = compact
    rows = studies["numeric_cells_scaling_20260903"]
    medians = {mode: statistics.median(row["wall_seconds"] for row in rows if row["numeric_mode"] == mode)
               for mode in dict.fromkeys(row["numeric_mode"] for row in rows)}
    reference = base / "render_gui_serial_reference"
    actual = base / "numeric_gui_20260903"
    gui = {f"checkpoint_{time}": compare_checkpoints(reference / f"gui_checkpoint_{time:06d}_Myr",
                                                     actual / f"gui_checkpoint_{time:06d}_Myr")
           for time in (4, 8, 12)}
    if not all(result["exact"] for result in gui.values()):
        raise RuntimeError("GUI checkpoint mismatch")
    gui["png"] = compare_pngs(reference, actual)
    hashes = lambda folder: {str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in folder.rglob("*.gif")}
    expected_gifs, actual_gifs = hashes(reference), hashes(actual)
    gui["gif"] = {"exact": expected_gifs == actual_gifs, "count": len(actual_gifs)}
    if not gui["png"]["png_exact"] or not gui["gif"]["exact"] or not actual_gifs:
        raise RuntimeError("GUI image mismatch or missing GIFs")
    resume = json.loads((base / "numeric_checkpoint_resume_20260903/validation.json").read_text())
    if len(resume) != 2 or not all(row["exact"] for row in resume):
        raise RuntimeError("Cross-runner checkpoint continuation failed")
    report = {"date": "2026-09-03", "base_commit": "0bfb810",
              "scope": "New exact-order numerical paths on top of shared initialization; no physical changes",
              "environment": {"python": platform.python_version(), "numpy": numpy.__version__,
                              "scipy": scipy.__version__, "platform": platform.platform()},
              "measurement_limitations": ["Uncontrolled desktop background load; ComfyUI processes present",
                  "Only two uninstrumented repeats on subdivision 5; one on subdivision 6",
                  "No final long-GIF assembly in timing samples; no 4.5 Gyr timing claim"],
              "plain_median_seconds": medians,
              "default_time_reduction_percent": 100 * (1 - medians["cells1"] / medians["legacy"]),
              "excluded_incompatible_test": {
                  "path": "results/cpu_performance/numeric_cells_sub6_20260903",
                  "reason": "Test command used frame interval 20, reference used 4; rerun with matching 4",
                  "state_exact": True, "missing_pngs": 8, "different_shared_pngs": 0},
              "studies": studies, "gui_exact_comparisons": gui, "checkpoint_continuation": resume}
    destination = ROOT / "performance_reports/numeric_kernels_study.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(destination), "gui": gui, "resume": resume,
                      "plain_median_seconds": medians}, indent=2))


if __name__ == "__main__":
    main()
