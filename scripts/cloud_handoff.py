"""Package and restore the nine immutable source files needed by cloud probes.

This uses only the standard library. It never overwrites an existing file.
The ZIP preserves exact bytes independently of Git checkout line endings.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile


ROOT = Path(__file__).resolve().parent.parent
STARTER = "results/genesis_runs/starter_20260928_192306_547551"
SAVED50 = "analysis/slab_sinking_followup/runs/ordered_sub4_dt1/elapsed_0050"
CHECKPOINT_FILES = (
    "mature_checkpoint/meta.json", "mature_checkpoint/state.npz", "mature_config.yaml",
    "young_context/starter_checkpoint.npz", "young_context/parameters.json",
    "young_context/fracture_memory.npz",
)
SOURCE_FILES = tuple(sorted((
    f"{STARTER}/starter_checkpoint.npz", f"{STARTER}/parameters.json",
    f"{SAVED50}/continuation.json", *(f"{SAVED50}/{name}" for name in CHECKPOINT_FILES),
)))
FORMAT = "moon-cloud-sources-1"
ARCHIVE = "cloud_sources.zip"
MANIFEST = "cloud_sources_manifest.json"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _target(root: Path, name: str) -> Path:
    """Accept portable relative paths and reject any existing symlink component."""
    pure = PurePosixPath(name)
    if (not name or "\\" in name or ":" in name or pure.is_absolute()
            or any(part in ("", ".", "..") for part in name.split("/"))):
        raise ValueError(f"Unsafe bundle path: {name!r}")
    root = Path(root).absolute()
    if root.is_symlink():
        raise ValueError(f"Destination root is a symlink: {root}")
    root = root.resolve()
    target = root.joinpath(*pure.parts)
    for candidate in (target, *target.parents):
        if candidate == root:
            break
        if candidate.is_symlink():
            raise ValueError(f"Symlink in source/destination path: {candidate}")
        if candidate != target and candidate.exists() and not candidate.is_dir():
            raise ValueError(f"Parent is not a directory: {candidate}")
    if not target.resolve().is_relative_to(root):
        raise ValueError(f"Destination escapes repository: {name}")
    return target


def _check_internal(payloads: dict[str, bytes]) -> None:
    try:
        report = json.loads(payloads[f"{SAVED50}/continuation.json"])
        expected = report["checkpoint_sha256"]
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError("Invalid saved50 continuation manifest") from exc
    if not isinstance(expected, dict) or set(expected) != set(CHECKPOINT_FILES):
        raise ValueError("Saved50 must name exactly the six known checkpoint files")
    for name in CHECKPOINT_FILES:
        if _sha(payloads[f"{SAVED50}/{name}"]) != expected[name]:
            raise ValueError(f"Saved50 internal checksum mismatch: {name}")


def _bundle_dir(bundle_dir: Path | None) -> Path:
    return Path(bundle_dir) if bundle_dir is not None else ROOT / "handoff"


def pack(root: Path = ROOT, bundle_dir: Path | None = None) -> dict:
    """Create a new ZIP and its external checksum manifest from local sources."""
    destination = _bundle_dir(bundle_dir)
    if any((destination / name).exists() or (destination / name).is_symlink()
           for name in (ARCHIVE, MANIFEST)):
        raise ValueError("Bundle already exists; packing never overwrites it")
    payloads = {name: _target(root, name).read_bytes() for name in SOURCE_FILES}
    _check_internal(payloads)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in SOURCE_FILES:
            entry = zipfile.ZipInfo(name)
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(entry, payloads[name], compresslevel=9)
    data = stream.getvalue()
    manifest = dict(format=FORMAT, archive=ARCHIVE, archive_sha256=_sha(data),
                    files=[dict(path=name, size=len(payloads[name]), sha256=_sha(payloads[name]))
                           for name in SOURCE_FILES])
    encoded = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / ARCHIVE).open("xb") as handle:
        handle.write(data)
    with (destination / MANIFEST).open("xb") as handle:
        handle.write(encoded)
    return dict(files=len(payloads), source_bytes=sum(map(len, payloads.values())),
                archive_bytes=len(data), archive_sha256=manifest["archive_sha256"])


def _load_bundle(bundle_dir: Path | None) -> dict[str, bytes]:
    source = _bundle_dir(bundle_dir)
    manifest = json.loads((source / MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT or manifest.get("archive") != ARCHIVE:
        raise ValueError("Unsupported cloud source bundle")
    entries = manifest.get("files")
    if (not isinstance(entries, list) or len(entries) != len(SOURCE_FILES)
            or any(not isinstance(item, dict) for item in entries)
            or {item.get("path") for item in entries} != set(SOURCE_FILES)):
        raise ValueError("Bundle manifest must contain exactly the nine source files")
    expected = {item["path"]: item for item in entries}
    data = (source / ARCHIVE).read_bytes()
    if _sha(data) != manifest.get("archive_sha256"):
        raise ValueError("Bundle archive checksum mismatch")
    payloads = {}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        infos = archive.infolist()
        if len(infos) != len(SOURCE_FILES) or {item.filename for item in infos} != set(SOURCE_FILES):
            raise ValueError("Unexpected, duplicated, or unsafe archive members")
        for entry in infos:
            if entry.is_dir() or stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError("Archive contains a directory or symlink")
            record = expected[entry.filename]
            if (type(record.get("size")) is not int or record["size"] < 0
                    or entry.file_size != record["size"]):
                raise ValueError(f"Bundle size mismatch: {entry.filename}")
            payload = archive.read(entry)
            if _sha(payload) != record.get("sha256"):
                raise ValueError(f"Bundle payload checksum mismatch: {entry.filename}")
            payloads[entry.filename] = payload
    _check_internal(payloads)
    return payloads


def restore(root: Path = ROOT, bundle_dir: Path | None = None) -> dict:
    """Validate everything before writes; skip identical files and reject conflicts."""
    payloads = _load_bundle(bundle_dir)
    pending = []
    for name, payload in payloads.items():
        target = _target(root, name)
        if target.exists():
            if not target.is_file() or target.read_bytes() != payload:
                raise ValueError(f"Existing destination differs; nothing restored: {name}")
        else:
            pending.append((target, payload))
    for target, payload in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(payload)
    return dict(files=len(payloads), restored=len(pending), skipped=len(payloads)-len(pending),
                source_bytes=sum(map(len, payloads.values())))


def verify(root: Path = ROOT, bundle_dir: Path | None = None) -> dict:
    """Check the archive plus all restored sources, including saved50's own hashes."""
    payloads = _load_bundle(bundle_dir)
    for name, payload in payloads.items():
        target = _target(root, name)
        if not target.is_file() or target.read_bytes() != payload:
            raise ValueError(f"Restored source missing or changed: {name}")
    return dict(files=len(payloads), source_bytes=sum(map(len, payloads.values())), verified=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pack", "restore", "verify"))
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="Repository containing sources or receiving restored files")
    parser.add_argument("--bundle-dir", type=Path, default=ROOT/"handoff",
                        help="Directory holding cloud_sources.zip and its manifest")
    args = parser.parse_args()
    result = {"pack": pack, "restore": restore, "verify": verify}[args.command](args.root, args.bundle_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
