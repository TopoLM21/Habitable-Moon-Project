"""Prepare authenticated inherited source objects in a fresh Python process.

This is a preparation observation, not a frozen release gate.  Only inherited
mechanics are imported; new newborn-law modules must be imported after freeze.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.geometric_native_birth_contact_validation.validation_support import (
    BASE, digest, encoded, guard_sources, read, value_sha, write,
)

started = time.perf_counter()
output = BASE / "source_prewarm_01"
output.mkdir(exist_ok=False)
source_map_before, original = guard_sources()
write(output / "sources_before.json", source_map_before)

from tests.test_geometric_native_physical_history import get_physical_case
from tests.test_geometric_native_implicit_underthrust_motion import get_implicit_motion_case
from tectonics.geometric_native_physical_history import native_physical_history_binding_to_dict

physical = get_physical_case()
first = get_implicit_motion_case()
accepted_state = first["step"].state
actual_source = native_physical_history_binding_to_dict(physical["binding"])
saved_source_path = ROOT / "analysis/geometric_native_active_binding_validation/control_release_02/source_history_before.json"
saved_motion_path = ROOT / "analysis/geometric_native_active_line_validation/control_release_01/motion/accepted_motion_before.json"
if encoded(actual_source) != encoded(read(saved_source_path)):
    raise AssertionError("The fresh genuine physical source differs from its complete inherited DTO")
if encoded(first["raw"]) != encoded(read(saved_motion_path)):
    raise AssertionError("The fresh first accepted physical interval differs from its complete inherited DTO")
if len(accepted_state.receipts) != 1:
    raise AssertionError("The preparation must recover exactly the accepted original interval")
imported_new = sorted(name for name in sys.modules if name.startswith("tectonics.geometric_native_birth"))
if imported_new:
    raise AssertionError("New production mechanics were imported before the source freeze")
write(output / "source_history.json", actual_source)
write(output / "accepted_motion.json", first["raw"])
source_map_after, original_after = guard_sources()
write(output / "sources_after.json", source_map_after)
if original_after != original:
    raise AssertionError("The inherited source map changed during preparation")
receipt = dict(format="native-birth-authenticated-source-prewarm-1", status="PASS",
    verified_utc=datetime.now(timezone.utc).isoformat(), seconds=time.perf_counter()-started,
    final_release_gate=False, fresh_python_interpreter=True,
    inherited_732_sources_unchanged=True, inherited_physical_source_resolved=True,
    source_history_equal_complete_saved_dto=True, accepted_motion_equal_complete_saved_dto=True,
    inherited_first_interval_count=len(accepted_state.receipts),
    source_history_sha256=digest(output / "source_history.json"),
    accepted_motion_sha256=digest(output / "accepted_motion.json"),
    saved_source_sha256=digest(saved_source_path), saved_motion_sha256=digest(saved_motion_path),
    source_map_before_sha256=digest(output / "sources_before.json"),
    source_map_after_sha256=digest(output / "sources_after.json"),
    new_production_modules_imported=imported_new,
    full_current_map_frozen=False,
    scope="Fresh inherited source/first-state reconstruction before newborn-source freeze; no new interval admission")
write(output / "receipt.json", receipt, "receipt_sha256")
print(encoded({key: receipt[key] for key in ("status", "seconds", "inherited_732_sources_unchanged",
    "source_history_equal_complete_saved_dto", "accepted_motion_equal_complete_saved_dto")}).decode(), flush=True)
