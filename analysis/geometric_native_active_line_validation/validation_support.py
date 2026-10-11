"""Source and evidence guards for the additive whole-line continuation."""
from __future__ import annotations

from pathlib import Path

from .verify_baseline import (BASE, PRIOR, ROOT, digest, encoded, read, sealed,
    sources, value_sha, verify_baseline, write)


def guard_sources(*, require_frozen=False):
    current = sources()
    original = read(PRIOR / "frozen_release_sources.json")
    if len(original) != 704:
        raise ValueError("Line work must preserve the authenticated 704-source baseline")
    changed = sorted(name for name, expected in original.items() if current.get(name) != expected)
    if changed:
        raise ValueError("An immutable prior Python source changed: " + repr(changed))
    frozen_path = BASE / "frozen_release_sources.json"
    if require_frozen and (not frozen_path.is_file() or read(frozen_path) != current):
        raise ValueError("The full current source map must be frozen before execution")
    return current, original


def freeze_sources():
    verify_baseline()
    current, original = guard_sources()
    map_path = BASE / "frozen_release_sources.json"
    write(map_path, current)
    receipt = dict(format="native-active-line-source-freeze-1", status="PASS",
        immutable_baseline_source_count=len(original), frozen_source_count=len(current),
        prior_python_sources_changed=[], added_python_sources=sorted(set(current)-set(original)),
        frozen_sources_sha256=digest(map_path),
        immutable_baseline_receipt_sha256=digest(BASE / "immutable_baseline_receipt.json"))
    return write(BASE / "frozen_release_receipt.json", receipt, "receipt_sha256")


if __name__ == "__main__":
    print(encoded(freeze_sources()).decode())
