"""Explain the precise diagnostic-only delta in the saved0.4 resume check."""
from pathlib import Path
import json

HERE=Path(__file__).resolve().parent


def main():
    legacy=json.loads((HERE/"validation_legacy04_parity.json").read_text(encoding="utf-8"))
    allowed={"bending_feed_resistance_n","cold_hinge_thickness_km","feed_m_s","gravitational_feed_force_n",
        "mantle_feed_resistance_n","neck_strength_pa","no_eduction_reaction_n","source_face","trench_length_km","unconstrained_feed_m_s"}
    differences=legacy["scientific_metadata_differences"]
    enrichment_only=all(row["field"].startswith("subduction_memory.young_boundary_state.mechanical_detachments[")
        and row["field"].rsplit(".",1)[-1] in allowed and row["first"] is None and row["second"] is not None
        for row in differences)
    physical_equal=all(row["bitwise_identical"] for row in legacy["arrays"].values()) and legacy["physical_report_equal"]
    parity=json.loads((HERE/"validation_cpu_parity.json").read_text(encoding="utf-8"))
    result=dict(legacy04_state_arrays_and_report_bitwise_equal=physical_equal,
        legacy04_metadata_difference_is_only_new_failure_diagnostics=enrichment_only,
        legacy04_new_diagnostic_field_count=len(differences),
        legacy04_raw_comparison="validation_legacy04_parity.json",
        ordered05_cpu1_cpu4_scientific_state_bitwise_identical=parity["scientific_state_bitwise_identical"],
        ordered05_raw_comparison="validation_cpu_parity.json",
        all_checks_passed=physical_equal and enrichment_only and parity["scientific_state_bitwise_identical"]
            and legacy["all_checks_passed"] and parity["all_checks_passed"])
    (HERE/"validation_compatibility.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit("Compatibility difference exceeded diagnostic enrichment")


if __name__=="__main__":
    main()
