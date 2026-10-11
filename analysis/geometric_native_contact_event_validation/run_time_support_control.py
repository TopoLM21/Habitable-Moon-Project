"""Live original-force candidate with certified constitutive time diagnostics.

The finite-kinematic force candidate still refuses.  This control proves the
declared inviscid affine-anchor dense extension over all four original traces
and the whole time interval.  It grants neither swept geometry nor transient
material ancestry and adopts no candidate state, memory or work.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.geometric_native_underthrust_validation.run_native_underthrust_control import (
    encoded, file_sha256, state_snapshot, write_json,
)
from analysis.geometric_underthrust_validation.underthrust_case import (
    ANCHOR_ERROR_BOUND_M, DT_S, SOURCE_INTERVALS, physical_inputs,
)
from tectonics.geometric_native_contact_time_support import (
    ADMISSIBLE, MODEL, assess_native_contact_time_support,
    build_native_contact_time_path, native_contact_time_support_to_dict,
)
from tectonics.geometric_native_mantle_loading import (
    assemble_native_physical_spatial_mantle_load, native_spatial_mantle_load_to_dict,
)
from tectonics.geometric_native_polygons import native_polygon_to_dict
from tectonics.geometric_native_underthrust_motion import (
    UnresolvedNativeUnderthrustMotion, advance_native_underthrust_motion,
    initialize_native_underthrust_motion,
)
from tectonics.geometric_native_underthrust_motion_io import (
    save_native_underthrust_motion_checkpoint,
)
from tectonics.geometric_underthrust_motion import HotBirthParameters


def code_snapshot():
    paths = {path for directory in ("tectonics", "tests", "scripts", "analysis")
             for path in (ROOT/directory).rglob("*.py")}
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in sorted(paths)}


def run_control(out):
    started, before_code = time.perf_counter(), code_snapshot()
    write_json(out/"sources_before.json", before_code)
    mesh, origin, inventory, seed, original_load = physical_inputs()
    if any((t.law.viscosity_pa_s_m, t.law.cohesion_pa, t.law.friction_dry,
            t.law.friction_wet) != (0., 0., 0., 0.) for t in seed.fault_traces.traces):
        raise AssertionError("The bounded original source experiment changed its physical law")
    state = initialize_native_underthrust_motion(mesh, origin, inventory, seed, original_load,
        birth_parameters=HotBirthParameters(2., 0., 0.,
            "Declared hot departure-footprint closure for the native physical control"),
        max_anchor_transport_error_m=ANCHOR_ERROR_BOUND_M)
    metadata = dict(control="original native force with declared constitutive time support",
        path_model=MODEL, accepted_loaded_native_intervals=0, cooling="frozen")
    initial = save_native_underthrust_motion_checkpoint(out/"initial.json", mesh, origin,
        state, provenance=metadata)
    before = state_snapshot(state)
    write_json(out/"declared_source.json", asdict(SOURCE_INTERVALS[0]))
    print("Assembling the unchanged prescribed source on native material footprints", flush=True)
    load = assemble_native_physical_spatial_mantle_load(mesh, state.current_surface,
        SOURCE_INTERVALS[0])
    write_json(out/"native_source.json", native_spatial_mantle_load_to_dict(state.current_surface, load))
    print("Solving the genuine full force candidate; retaining its geometric refusal", flush=True)
    try:
        advance_native_underthrust_motion(mesh, origin, state, load, DT_S,
            provenance="Independent affine source drives the original native material pairs")
    except UnresolvedNativeUnderthrustMotion as refusal:
        candidate, overlaps = refusal.candidate, refusal.overlaps
    else:
        raise AssertionError("This bounded diagnostic must not adopt a loaded native interval")
    if (state_snapshot(state) != before or state.receipts or state.current_inventory.records
            or state.elapsed_time_s != 0. or state.cumulative_source_work_j != 0.):
        raise AssertionError("Unaccepted contact-time diagnostics changed persistent material or work")
    candidate_raw = dict(before_surface_sha256=candidate.before_surface_sha256,
        current_load_sha256=candidate.current_load_sha256, dt_s=candidate.dt_s,
        provenance=candidate.provenance,
        rotation_increment_radians=candidate.rotation_increment_radians,
        angular_velocity_rad_s=candidate.angular_velocity_rad_s,
        omega_rad_per_myr=candidate.omega_rad_per_myr,
        states=[(identity, asdict(local)) for identity, local in candidate.states],
        transverse_displacement_m=candidate.transverse_displacement_m,
        spatial_step=asdict(candidate.spatial_step),
        diagnostics=dict(candidate.candidate_diagnostics),
        admission=candidate.admission_audit, physical_points=candidate.points,
        unresolved=[dict(fragment_a=o.fragment_a, fragment_b=o.fragment_b,
            plate_a=o.plate_a, plate_b=o.plate_b, area_km2=o.area_km2,
            reason=o.reason, geometry=native_polygon_to_dict(o.geometry)) for o in overlaps])
    write_json(out/"force_candidate.json", candidate_raw)
    print("Certifying all four original material traces for the entire declared time path", flush=True)
    path = build_native_contact_time_path(seed, candidate.spatial_step, state.free_anchor_fields,
        path_model=MODEL)
    certificates = [assess_native_contact_time_support(path, certificate.trace_id,
        0, certificate.extent_radians, 0, 1, precision=80)
        for certificate in candidate.spatial_step.certificates]
    if len(certificates) != 4 or any(c.status != ADMISSIBLE for c in certificates):
        raise AssertionError("The declared whole-time inviscid support could not be proved")
    write_json(out/"constitutive_time_support.json", dict(
        path_sha256=path.path_sha256, path_model=MODEL,
        coordinate_model="original_retained_angle_scaled_from_exact_native_axis_quarter",
        certificates=[native_contact_time_support_to_dict(c) for c in certificates],
        candidate_sha256=hashlib.sha256(encoded(candidate_raw)).hexdigest(),
        geometry_permission=False, candidate_memory_adopted=False,
        scope="four_original_inviscid_pairs_whole_material_traces_whole_time_interval"))
    after = save_native_underthrust_motion_checkpoint(out/"after_refusal.json", mesh, origin,
        state, provenance=metadata)
    if initial.read_bytes() != after.read_bytes() or state_snapshot(state) != before:
        raise AssertionError("Constitutive time diagnostics changed the prepared checkpoint")
    after_code = code_snapshot()
    write_json(out/"sources_after.json", after_code)
    if before_code != after_code:
        raise AssertionError("Source code changed while running the bounded time-path control")
    artifacts = {path.name: dict(sha256=file_sha256(path), bytes=path.stat().st_size)
                 for path in sorted(out.iterdir()) if path.is_file()}
    report = dict(status="declared_constitutive_time_support_verified_without_adoption",
        finished_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter()-started, original_trace_count=len(certificates),
        accepted_loaded_native_intervals=0, accepted_source_work_j=0.,
        geometry_permission=False, candidate_memory_adopted=False,
        input_unchanged=True, prepared_checkpoint_byte_identical=True,
        prepared_checkpoint_sha256=file_sha256(initial),
        prepared_state_sha256=hashlib.sha256(before).hexdigest(),
        candidate_sha256=hashlib.sha256(encoded(candidate_raw)).hexdigest(),
        path_sha256=path.path_sha256, path_model=MODEL,
        physical_law=dict(viscosity_pa_s_m=0., cohesion_pa=0., friction=0.),
        original_source_selected_before_force=True,
        all_original_full_rotation_components_retained=True,
        material_time_domain="each complete original trace x fraction [0,1]",
        initial_time_model="inviscid continuous right derivative at fraction zero",
        pressure_minimum_lower_bound_pa=min(float(c.pressure_bounds_pa.lo) for c in certificates),
        maximum_plastic_rate_upper_bound_m_s=max(float(c.plastic_rate_bounds_m_s.hi)
            for c in certificates), source_file_count=len(before_code),
        source_code_unchanged=True, refusal_reasons=sorted({o.reason for o in overlaps}),
        unaccepted_candidate_work_j=dict(candidate.candidate_diagnostics)["source_work_j"],
        artifacts=artifacts,
        remaining_within_point_1="certified swept event arcs and transient material ancestry; no endpoint-only adoption")
    write_json(out/"report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True,
        help="New output directory; existing directories are refused")
    out = parser.parse_args().out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = run_control(out)
    print(f"PASS: {report['original_trace_count']} whole-trace/time proofs; "
          "0 loaded intervals adopted; prepared checkpoint unchanged", flush=True)


if __name__ == "__main__":
    main()
