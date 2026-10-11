"""Refresh only additive Python modules while preserving inherited fixtures."""
from __future__ import annotations

from datetime import datetime, timezone
import importlib
from pathlib import Path
import sys

from analysis.geometric_native_birth_contact_validation.validation_support import (
    ROOT, digest, guard_sources, write,
)


def refresh_additive_modules(out, *, require_frozen=False):
    before, original = guard_sources(require_frozen=require_frozen)
    loaded = {}
    for name, module in tuple(sys.modules.items()):
        if name == __name__ or not getattr(module, "__file__", None):
            continue
        path = Path(module.__file__).resolve()
        try:
            relative = str(path.relative_to(ROOT))
        except ValueError:
            continue
        if path.suffix == ".py" and relative in before and relative not in original:
            loaded[name] = dict(source=relative, source_sha256=before[relative],
                old_cache=getattr(module, "__cached__", None))
    for name in loaded:
        cached = getattr(sys.modules[name], "__cached__", None)
        if cached:
            cache = Path(cached).resolve()
            if ROOT in cache.parents and cache.suffix == ".pyc" and cache.exists():
                cache.unlink()
    importlib.invalidate_caches()
    def priority(name):
        if name.startswith("tectonics."):
            order = {"tectonics.geometric_native_birth_contact": 0,
                "tectonics.geometric_native_birth_contact_network": 1,
                "tectonics.geometric_native_birth_contact_support": 2}
            return (0, order.get(name, 3), name)
        if name.startswith("tests."):
            return (1, 0, name)
        if name.endswith("validation_support") or name.endswith("verify_baseline"):
            return (2, 0, name)
        if name.endswith("native_case"):
            return (3, 0, name)
        return (4, 0, name)
    order = sorted(loaded, key=priority)
    for name in order:
        importlib.reload(sys.modules[name])
    after, original_after = guard_sources(require_frozen=require_frozen)
    if after != before or original_after != original:
        raise ValueError("Sources changed while additive modules were refreshed")
    for name, row in loaded.items():
        module = sys.modules[name]
        if str(Path(module.__file__).resolve().relative_to(ROOT)) != row["source"]:
            raise ValueError("A refreshed module changed its source location")
        if digest(ROOT / row["source"]) != row["source_sha256"]:
            raise ValueError("A refreshed module changed its source body")
    receipt = dict(format="native-birth-additive-module-cache-refresh-1", status="PASS",
        refreshed_utc=datetime.now(timezone.utc).isoformat(), frozen_source_required=require_frozen,
        whole_current_source_count=len(before), inherited_732_sources_unchanged=True,
        inherited_modules_reloaded=False, inherited_heavy_fixture_caches_retained=True,
        only_additive_pyc_removed=True, imported_module_reload_order=order,
        refreshed_modules=loaded, fresh_independent_interpreter=False,
        scope="New classes and factory definitions reloaded from unchanged declared source bytes; inherited source objects remain authenticated separately")
    return write(out, receipt, "receipt_sha256")
