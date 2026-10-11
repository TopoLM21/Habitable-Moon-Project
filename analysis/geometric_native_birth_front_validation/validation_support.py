"""Exclusive evidence writing and immutable source guards for front work."""
from __future__ import annotations

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
    if len(original) != 751:
        raise ValueError("Front work must preserve all 751 inherited sources")
    changed = sorted(name for name, expected in original.items() if current.get(name) != expected)
    if changed:
        raise ValueError("An immutable inherited Python source changed: " + repr(changed))
    frozen = BASE / "frozen_release_sources.json"
    if require_frozen and (not frozen.is_file() or read(frozen) != current):
        raise ValueError("Freeze the complete current source map before release execution")
    return current, original


def archive_additions(directory, current, original):
    archive = directory / "additive_source_bodies"
    archive.mkdir(exist_ok=False)
    rows = {}
    for name in sorted(set(current)-set(original)):
        target = archive / (name.replace("/", "__") + ".txt")
        with target.open("xb") as stream:
            stream.write((ROOT / name).read_bytes())
        if digest(target) != current[name]:
            raise ValueError("A source changed during its archival snapshot: " + name)
        rows[name] = dict(path=str(target.relative_to(ROOT)), bytes=target.stat().st_size, sha256=digest(target))
    return write(directory / "additive_source_archive.json", dict(
        format="native-birth-front-source-body-archive-1", status="PRESERVED",
        count=len(rows), sources=rows), "archive_sha256")


def freeze_sources():
    verify_baseline()
    current, original = guard_sources()
    archive = archive_additions(BASE, current, original)
    map_path = BASE / "frozen_release_sources.json"
    write(map_path, current)
    receipt = dict(format="native-birth-front-source-freeze-1", status="PASS",
        inherited_python_source_count=751, frozen_source_count=len(current),
        inherited_sources_changed=[], added_python_sources=sorted(set(current)-set(original)),
        frozen_sources_sha256=digest(map_path),
        additive_source_archive=dict(path=str((BASE / "additive_source_archive.json").relative_to(ROOT)),
            sha256=digest(BASE / "additive_source_archive.json"), count=archive["count"]),
        immutable_baseline_receipt_sha256=digest(BASE / "immutable_baseline_receipt.json"))
    return write(BASE / "frozen_release_receipt.json", receipt, "receipt_sha256")


if __name__ == "__main__":
    print(encoded(freeze_sources()).decode())
