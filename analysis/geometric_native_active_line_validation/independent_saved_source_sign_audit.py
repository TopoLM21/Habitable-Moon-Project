"""Prove saved source depth signs on entire angular sectors, without a solver."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.geometric_native_active_line_validation.independent_line_oracle import (
    axes_from_source, fresh_depth_whole_two_clock_bounds, initial_depth_divided_clock_bounds,
    quarter_columns,
)
from analysis.geometric_native_active_line_validation.verify_baseline import (
    BASE, digest, encoded, read, verify_baseline, write,
)


SOURCE = ROOT / "analysis/geometric_native_active_binding_validation/control_release_02/source_history_before.json"


def fraction(value):
    return [value.numerator, value.denominator]


def bounds(pair):
    return dict(lo=fraction(pair[0]), hi=fraction(pair[1]))


def audit():
    verify_baseline()
    source = read(SOURCE)
    donor, receiver = axes_from_source(source)
    if any(value == 0 for axis in (donor, receiver) for value in axis):
        raise ValueError("The independent audit must preserve every nonzero saved Euler component")
    depth = initial_depth_divided_clock_bounds(source, degree=18)
    fresh = fresh_depth_whole_two_clock_bounds(source, degree=10)
    if any(not 0 < lo <= hi for lo, hi in (*depth, *fresh)):
        raise ValueError("Independent whole-clock Y/Z depth coefficients did not prove positive")
    ratio_low = min(depth[1][0]/depth[0][1], fresh[1][0]/fresh[0][1])
    ratio_high = max(depth[1][1]/depth[0][0], fresh[1][1]/fresh[0][0])
    if not 0 < ratio_low < ratio_high:
        raise ValueError("The strict mixed-quarter transition band is not positive and ordered")
    quarters = quarter_columns(source)
    if {row[0] for row in quarters} != {"octant:0", "octant:1", "octant:2", "octant:3"}:
        raise ValueError("The saved line does not retain every receiving quarter")
    return dict(format="native-active-line-independent-source-sign-audit-1", status="PASS",
        source_history_path=str(SOURCE.relative_to(ROOT)), source_history_sha256=digest(SOURCE),
        imports_production_modules=False, force_solver_executed=False,
        all_saved_euler_components_retained=True,
        original_quarter_columns=[dict(receiver_fragment_id=row[0], trace_id=row[1],
            start=[fraction(value) for value in row[2]], end=[fraction(value) for value in row[3]])
            for row in quarters],
        initial_depth_over_clock_yz_bounds=[bounds(pair) for pair in depth],
        fresh_depth_all_event_and_owner_clock_yz_bounds=[bounds(pair) for pair in fresh],
        common_mixed_quarter_ratio_band=bounds((ratio_low, ratio_high)),
        full_quarter_sign_proofs=[
            dict(receiver_fragment_id="octant:0", coordinates="y<0,z<0",
                initial_depth_sign="negative", fresh_owner_depth_derivative_sign="negative"),
            dict(receiver_fragment_id="octant:3", coordinates="y>0,z>0",
                initial_depth_sign="positive", fresh_owner_depth_derivative_sign="positive")],
        mixed_quarter_sector_sign_proofs=[
            dict(receiver_fragment_id="octant:1", exact_homogeneous_carrier="(0,-lambda,1)",
                below_ratio_low_sign="positive", above_ratio_high_sign="negative"),
            dict(receiver_fragment_id="octant:2", exact_homogeneous_carrier="(0,lambda,-1)",
                below_ratio_low_sign="negative", above_ratio_high_sign="positive")],
        proof_scope="All positive original clocks for qX/u; every event and owner clock in [0,1]^2 for F(u,v). Exact rational joint-matrix Taylor tail bounds, no angular or clock samples.",
        mixed_ratio_band_selection_proved=False,
        geometric_quarter_exit_same_as_positive_fresh_entry=False,
        unchanged_initial_prepared_contact_can_be_carried=False,
        unchanged_initial_prepared_contact_scope="No constitutive retention law is inferred by this sign audit.",
        all_angle_active_path_closed=False, accepted_native_force_intervals=0)


if __name__ == "__main__":
    receipt = write(BASE / "independent_saved_source_sign_receipt.json", audit(), "receipt_sha256")
    print(encoded({key: receipt[key] for key in ("status", "imports_production_modules",
        "force_solver_executed", "all_angle_active_path_closed")}).decode())
