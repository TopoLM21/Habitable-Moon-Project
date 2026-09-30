"""Compare uniform and ordered buoyancy on unchanged saved0.4 states.

The ordered arm is an explicitly labelled in-memory counterfactual. It neither
relabels a saved checkpoint nor advances time or commits detachments.
"""
from __future__ import annotations

from copy import deepcopy
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "analysis/slab_sinking_validation")]
from run_case import code_hashes, force_metrics, source_hashes, write_json

SOURCES = {
    50: "validated_sinking_sub4_dt1",
    100: "validated_sinking_sub4_dt1",
    400: "validated_sinking_fixed2_sub4_dt1",
}


def main():
    import tectonics.simulation as simulation
    import tectonics.genesis_starter_continuation as continuation
    from tectonics.plate_velocity_diagnostics import diagnose_checkpoint
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "frozen04_counterfactual")
    args = parser.parse_args()
    target = args.output.resolve()
    if target.exists():
        raise ValueError("Frozen output must be new")
    target.mkdir()
    records = []
    for age, folder in SOURCES.items():
        source = ROOT / "analysis/slab_sinking_validation/runs" / folder / f"elapsed_{age:04d}"
        before = source_hashes(source)
        hashes = code_hashes()
        baseline = diagnose_checkpoint(source)
        load_config, load_cp = simulation.load_config, continuation._load_cp

        def counterfactual_config(*args, **kwargs):
            cfg = deepcopy(load_config(*args, **kwargs))
            if cfg["young_shell"]["mechanics_model_version"] != "young-mechanics-0.4":
                raise ValueError("Counterfactual requires an explicit saved0.4 source")
            cfg["young_shell"]["mechanics_model_version"] = "young-mechanics-0.5"
            cfg["plate_dynamics"]["young_slab_buoyancy_model"] = "ordered_thermal_cohorts_v1"
            cfg["subduction_memory"]["young_slab_buoyancy_model"] = "ordered_thermal_cohorts_v1"
            return cfg

        def counterfactual_cp(*args, **kwargs):
            cp = load_cp(*args, **kwargs)
            cp.subduction_memory.young_boundary_state.buoyancy_geometry_model = "ordered_thermal_cohorts_v1"
            return cp

        simulation.load_config = counterfactual_config
        continuation._load_cp = counterfactual_cp
        try:
            ordered = diagnose_checkpoint(source)
        finally:
            simulation.load_config, continuation._load_cp = load_config, load_cp
        unchanged = before == source_hashes(source)
        pair = {}
        for label, report in (("saved_uniform04", baseline), ("ordered05_counterfactual", ordered)):
            report.update(evaluation=("saved_uniform04_frozen_probe" if label == "saved_uniform04"
                else "explicit_in_memory_ordered_buoyancy_counterfactual_on_saved04_no_evolution"),
                source_unchanged=unchanged, source_sha256=before, production_sha256=hashes,
                metrics=force_metrics(report["trace"]))
            write_json(target / f"elapsed_{age:04d}_{label}.json", report)
            pair[label] = dict(speeds=report["stages_speed_km_per_myr"], metrics=report["metrics"],
                surviving_section_count=len(report["trace"].get("slab_sections", [])),
                hypothetical_neck_failures=len(report["trace"].get("slab_neck_failures", [])))
        record = dict(elapsed_myr=age, source=str(source), source_unchanged=unchanged,
            production_unchanged=hashes == code_hashes(), cases=pair)
        records.append(record)
        print(json.dumps(record), flush=True)
        if not unchanged or hashes != code_hashes():
            raise ValueError("Frozen source or production changed during comparison")
    write_json(target / "summary.json", dict(evaluation="force_only_counterfactual_not_evolved_trajectory",
        note="Only config version and buoyancy selectors were changed on loaded copies; saved materials, time, plate motion, coefficients and files were unchanged.",
        cases=records))


if __name__ == "__main__":
    main()
