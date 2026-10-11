"""Source guards and exclusive evidence writers for newborn-contact work."""
from __future__ import annotations

from pathlib import Path

from .verify_baseline import (BASE, ROOT, SUMMARY, digest, encoded, read,
    sealed, sources, value_sha, verify_baseline, write)


def inherited_sources():
    summary = sealed(SUMMARY, "summary_sha256")
    reference = summary["frozen_source_map"]
    if digest(ROOT / reference["path"]) != reference["sha256"]:
        raise ValueError("The inherited frozen source map changed")
    return read(ROOT / reference["path"])


def guard_sources(*, require_frozen=False):
    current, original = sources(), inherited_sources()
    if len(original) != 732:
        raise ValueError("Birth-contact work must preserve all 732 inherited sources")
    changed = sorted(name for name, expected in original.items() if current.get(name) != expected)
    if changed:
        raise ValueError("An immutable inherited Python source changed: " + repr(changed))
    frozen = BASE / "frozen_release_sources.json"
    if require_frozen and (not frozen.is_file() or read(frozen) != current):
        raise ValueError("Freeze the complete current source map before executing release checks")
    return current, original


def freeze_sources():
    verify_baseline()
    current, original = guard_sources()
    additions = sorted(set(current)-set(original))
    archive = BASE / "frozen_additive_source_bodies"
    archive.mkdir(exist_ok=False)
    rows = {}
    for name in additions:
        target = archive / (name.replace("/", "__") + ".txt")
        with target.open("xb") as stream:
            stream.write((ROOT / name).read_bytes())
        rows[name] = dict(path=str(target.relative_to(ROOT)), bytes=target.stat().st_size,
            sha256=digest(target))
    archive_manifest = BASE / "frozen_additive_source_archive.json"
    write(archive_manifest, dict(format="native-birth-source-body-archive-1",
        status="PASS", count=len(rows), sources=rows), "archive_sha256")
    map_path = BASE / "frozen_release_sources.json"
    write(map_path, current)
    receipt = dict(format="native-birth-contact-source-freeze-1", status="PASS",
        inherited_python_source_count=len(original), frozen_source_count=len(current),
        inherited_sources_changed=[], added_python_sources=additions,
        frozen_sources_sha256=digest(map_path),
        additive_source_archive=dict(path=str(archive_manifest.relative_to(ROOT)),
            sha256=digest(archive_manifest), count=len(rows)),
        immutable_baseline_receipt_sha256=digest(BASE / "immutable_baseline_receipt.json"))
    return write(BASE / "frozen_release_receipt.json", receipt, "receipt_sha256")


if __name__ == "__main__":
    print(encoded(freeze_sources()).decode())
