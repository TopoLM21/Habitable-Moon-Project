"""Clearly separated development probe for the genuinely solved next Omega.

The probe may run while additive sources are changing.  It records those
changes and cannot produce a release PASS or authenticate final test sources.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from fractions import Fraction
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def portable(value):
    if is_dataclass(value):
        return {field.name: portable(getattr(value, field.name)) for field in fields(value)}
    if type(value) is dict:
        return {key: portable(item) for key, item in value.items()}
    if type(value) in (tuple, list):
        return [portable(item) for item in value]
    if type(value) is Fraction:
        return dict(exact_fraction=[value.numerator, value.denominator])
    if type(value) is bytes:
        return dict(exact_bytes_hex=value.hex())
    if hasattr(value, "tolist"):
        return portable(value.tolist())
    return value


def run_probe(out, *, actual_case=None):
    from analysis.geometric_native_active_line_validation.validation_support import (
        digest, encoded, guard_sources, read, sources, write,
    )
    from tests.test_geometric_native_physical_history import get_physical_case
    before, immutable = guard_sources()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write(out / "sources_before.json", before)
    started = time.perf_counter()
    case = get_physical_case() if actual_case is None else actual_case
    print("Development probe: original force authenticated; importing additive construction modules", flush=True)
    from tectonics.geometric_native_contact_reference_transport import ContactHistoryTransportLaw, TRANSPORT_LAW_MODEL
    from tectonics.geometric_native_physical_history import native_physical_history_binding_to_dict
    from tectonics.geometric_native_implicit_underthrust_motion import (
        accept_native_implicit_underthrust_motion, native_implicit_underthrust_motion_to_dict,
        solve_native_implicit_underthrust_candidate,
    )
    from tectonics.geometric_native_implicit_mantle_loading import assemble_native_implicit_physical_spatial_mantle_load
    from analysis.geometric_underthrust_validation.underthrust_case import DT_S, SOURCE_INTERVALS
    history = case["binding"]
    source = native_physical_history_binding_to_dict(history)
    law = ContactHistoryTransportLaw(TRANSPORT_LAW_MODEL,
        "Development probe: whole line positive fresh seam switch; proved negative seam retains frozen material")
    first = accept_native_implicit_underthrust_motion(history, law=law,
        expected_source_history_dto=source)
    write(out / "first_interval_state.json", native_implicit_underthrust_motion_to_dict(first.state))
    print("Development probe: first interval constructed; assembling the changed second source", flush=True)
    load = assemble_native_implicit_physical_spatial_mantle_load(case["mesh"], first.state.current_surface,
        SOURCE_INTERVALS[1])
    print("Development probe: solving genuine next physical force candidate", flush=True)
    trial = solve_native_implicit_underthrust_candidate(case["mesh"], first.state, load, DT_S,
        provenance=SOURCE_INTERVALS[1].provenance)
    write(out / "next_force_candidate.json", portable(trial))
    after, immutable_after = guard_sources()
    write(out / "sources_after.json", after)
    if immutable != immutable_after:
        raise AssertionError("The immutable baseline map changed during the development probe")
    changed = sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))
    imported = {}
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if path is None:
            continue
        path = Path(path)
        if path.suffix == ".py" and path.is_relative_to(ROOT):
            relative = str(path.relative_to(ROOT))
            if relative not in immutable:
                imported[relative] = dict(before_sha256=before.get(relative), after_sha256=after.get(relative))
    write(out / "imported_additive_module_sources.json", imported)
    receipt = dict(format="native-implicit-motion-prerelease-construction-probe-1", status="OBSERVED",
        scope="Development construction only; not final release validation or admission of the second interval",
        seconds=time.perf_counter()-started, original_704_sources_unchanged=True,
        source_history_sha256=history.binding_sha256,
        source_development_changes=changed, imported_additive_modules=sorted(imported),
        sources_before_sha256=digest(out / "sources_before.json"),
        sources_after_sha256=digest(out / "sources_after.json"),
        full_next_candidate_sha256=digest(out / "next_force_candidate.json"),
        next_dt_s=trial.dt_s, next_omega_rad_per_myr=portable(trial.omega_rad_per_myr),
        next_rotation_increment_radians=portable(trial.rotation_increment_radians),
        next_diagnostics=portable(trial.diagnostics),
        first_interval_count_in_exploratory_state=len(first.state.receipts),
        accepted_second_intervals=0, final_release_gate=False,
        fresh_source_freeze_and_complete_final_gate_required=True)
    saved = write(out / "receipt.json", receipt, "receipt_sha256")
    print(encoded({key: saved[key] for key in ("status", "next_dt_s", "next_omega_rad_per_myr",
        "source_development_changes", "accepted_second_intervals")}).decode(), flush=True)
    return saved


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    run_probe(parser.parse_args().out)
