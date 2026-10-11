"""Archive exact development source bodies before a genuine continuum solve.

This exclusive writer preserves refusals and loaded source versions. A pass
is an affine mechanical experiment, never a native material admission.
"""
from __future__ import annotations

import importlib
from pathlib import Path
import sys
import time
import traceback

from .validation_support import ROOT, archive_additions, guard_sources, write, digest


def refresh_new_modules(current, original):
    """Refresh additive code only; retain authenticated heavy original caches."""
    names = [
        "tectonics.geometric_native_birth_contact_front",
        "tectonics.geometric_native_birth_pair_exclusion",
        "tectonics.geometric_native_birth_curved_pair_support",
        "tectonics.geometric_native_birth_contact_continuum",
        "tectonics.geometric_native_birth_continuum_motion",
        "tectonics.geometric_native_birth_finite_pose",
        "analysis.geometric_native_birth_front_validation.native_continuum_case",
        "analysis.geometric_native_birth_front_validation.pair_exclusion_control",
        "analysis.geometric_native_birth_front_validation.curved_pair_control",
        "analysis.geometric_native_birth_front_validation.run_front_control",
        "analysis.geometric_native_birth_front_validation.native_control",
    ]
    rows = []
    importlib.invalidate_caches()
    for name in names:
        path = ROOT / (name.replace(".", "/")+".py")
        relative = str(path.relative_to(ROOT))
        if not path.is_file():
            continue
        if relative in original or relative not in current:
            raise ValueError("Module refresh must touch additive source bodies only")
        previous = sys.modules.get(name)
        cached = getattr(previous, "__cached__", None)
        if cached is None:
            cached = importlib.util.cache_from_source(str(path))
        cached_path = Path(cached)
        if cached_path.is_file():
            cached_path.unlink()
        module = importlib.import_module(name) if previous is None else importlib.reload(previous)
        actual = digest(path)
        if actual != current[relative]:
            raise ValueError("A module changed after its archived development snapshot")
        rows.append(dict(module=name, path=relative, sha256=actual,
            previously_loaded=previous is not None, loaded_filename=module.__file__))
    return rows


def run_development_preflight(directory):
    directory = Path(directory)
    directory.mkdir(exist_ok=False)
    started = time.monotonic()
    current, original = guard_sources()
    write(directory / "source_map_before.json", current)
    archive = archive_additions(directory, current, original)
    stages, case, candidate = {}, None, None
    body = dict(format="native-birth-front-development-preflight-1",
        status="REFUSED", final_release_gate=False,
        native_material_admission_authorized=False, accepted_new_native_intervals=0,
        original_native_intervals_accepted=1, decimal_precision=160,
        archived_additive_source_count=archive["count"])
    try:
        refreshed = refresh_new_modules(current, original)
        body["new_module_refresh"] = refreshed
        from .native_continuum_case import (build_native_continuum_case,
            solve_native_continuum_case, export_native_continuum_case)
        tick = time.monotonic()
        case = build_native_continuum_case()
        stages["canonical_case_construction_s"] = time.monotonic()-tick
        print("FRONT canonical genuine source ready", stages, flush=True)
        tick = time.monotonic()
        candidate = solve_native_continuum_case(case,
            provenance="Archived development genuine full-arc native source force experiment")
        stages["genuine_continuum_force_solve_s"] = time.monotonic()-tick
        print("FRONT genuine continuum force solved", stages, flush=True)
        tick = time.monotonic()
        packet = export_native_continuum_case(case, candidate)
        stages["strict_full_source_and_force_export_replay_s"] = time.monotonic()-tick
        write(directory / "candidate_packet.json", packet)
        body.update(status="PASS", before_state_sha256=candidate.before_state_sha256,
            current_load_sha256=candidate.current_load_sha256,
            birth_support_sha256=candidate.birth_support_sha256,
            candidate_sha256=packet["candidate_sha256"],
            candidate_packet_raw_sha256=digest(directory / "candidate_packet.json"),
            born_finite_pose_branch_admission_proved=False,
            continuum_carriers=len(case["inputs"]["born_continuum_attachments"]),
            born_midpoint_atoms=0, foundation_count=len(case["inputs"]["born_foundations"]),
            old_point_count=len(case["inputs"]["old_point_ids"]),
            coordinate_count=len(candidate.network_step.coordinates),
            diagnostics=dict(candidate.network_step.diagnostics),
            finite_anchor_error_m=candidate.finite_anchor_error_m,
            relative_owner_angle_radians=candidate.relative_owner_angle_radians,
            omega_rad_per_myr=candidate.omega_rad_per_myr, dt_s=case["dt_s"])
    except Exception as error:
        body.update(refusal_type=type(error).__name__, refusal=str(error), traceback=traceback.format_exc())
        print("FRONT genuine continuum force refused", type(error).__name__, str(error), flush=True)
    after, _ = guard_sources()
    write(directory / "source_map_after.json", after)
    body.update(stage_elapsed_s=stages, elapsed_s=time.monotonic()-started,
        added_sources_changed_during_run=sorted(name for name in set(current)|set(after)
            if current.get(name) != after.get(name)), immutable_751_preserved=True,
        archived_code_is_this_attempt_authority=True)
    write(directory / "preflight_receipt.json", body, "receipt_sha256")
    return dict(case=case, candidate=candidate, report=body, directory=directory)
