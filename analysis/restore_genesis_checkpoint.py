#!/usr/bin/env python3
"""Verify or restore the split, portable Genesis checkpoint using the stdlib.

Run without flags to verify the complete compressed archive.  --restore writes
only missing repository files.  --restore-anchors PATH additionally redirects
workspace anchors to PATH; used alone, it restores only those anchors.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tarfile
import tempfile


CHUNK = 1024 * 1024
FORMAT = "genesis-checkpoint-v1"


class CheckpointError(ValueError):
    pass


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _relative_path(value):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise CheckpointError(f"Unsafe relative path: {value!r}")
    pieces = value.split("/")
    if any(piece in ("", ".", "..") or "\x00" in piece for piece in pieces):
        raise CheckpointError(f"Unsafe relative path: {value!r}")
    return pieces


def _integer(value, label):
    if type(value) is not int or value < 0:
        raise CheckpointError(f"Invalid nonnegative integer: {label}")
    return value


def _digest(value, label):
    if not isinstance(value, str) or len(value) != 64:
        raise CheckpointError(f"Invalid SHA-256: {label}")
    if any(char not in "0123456789abcdef" for char in value):
        raise CheckpointError(f"Invalid SHA-256: {label}")
    return value


def _safe_target(root, relative):
    """Do not resolve a destination through symlinks, even inside the root."""
    target = root
    pieces = _relative_path(relative)
    for index, piece in enumerate(pieces):
        target = target / piece
        try:
            info = target.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode):
            raise CheckpointError(f"Symlink destination refused: {target}")
        if index < len(pieces) - 1 and not stat.S_ISDIR(info.st_mode):
            raise CheckpointError(f"Destination parent is not a directory: {target}")
    return target


def _root(path, *, may_be_missing=False):
    path = Path(path).absolute()
    # Reject a symlink in any supplied root component before resolving it.
    current = Path(path.anchor)
    for piece in path.parts[1:]:
        current = current / piece
        try:
            info = current.lstat()
        except FileNotFoundError:
            if may_be_missing:
                continue
            raise CheckpointError(f"Missing directory: {current}") from None
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise CheckpointError(f"Root contains a symlink or nondirectory: {current}")
    return path


def load_manifest(bundle):
    manifest_path = _safe_target(bundle, "manifest.json")
    if not manifest_path.is_file():
        raise CheckpointError(f"Missing manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle, object_pairs_hook=_json_object)
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise CheckpointError("Unsupported checkpoint manifest format")
    for field in ("raw_file_count", "raw_bytes", "archive_bytes"):
        _integer(manifest.get(field), field)
    _digest(manifest.get("archive_sha256"), "archive_sha256")
    for field in ("parts", "files"):
        if not isinstance(manifest.get(field), list):
            raise CheckpointError(f"Manifest {field} must be a list")
        seen = set()
        for entry in manifest[field]:
            if not isinstance(entry, dict):
                raise CheckpointError(f"Invalid manifest {field} entry")
            _relative_path(entry.get("path"))
            if entry["path"] in seen:
                raise CheckpointError(f"Duplicate manifest {field} path: {entry['path']}")
            seen.add(entry["path"])
            _integer(entry.get("size"), f"{entry['path']}: size")
            _digest(entry.get("sha256"), entry["path"])
            if field == "files":
                mode = _integer(entry.get("mode"), f"{entry['path']}: mode")
                if mode > 0o7777:
                    raise CheckpointError(f"Invalid file mode: {entry['path']}")
    if not manifest["parts"]:
        raise CheckpointError("Checkpoint has no archive parts")
    if sum(entry["size"] for entry in manifest["parts"]) != manifest["archive_bytes"]:
        raise CheckpointError("Archive part sizes disagree with archive_bytes")
    if len(manifest["files"]) != manifest["raw_file_count"]:
        raise CheckpointError("File list disagrees with raw_file_count")
    if sum(entry["size"] for entry in manifest["files"]) != manifest["raw_bytes"]:
        raise CheckpointError("File sizes disagree with raw_bytes")
    file_names = {entry["path"] for entry in manifest["files"]}
    for name in file_names:
        pieces = name.split("/")
        if any("/".join(pieces[:index]) in file_names for index in range(1, len(pieces))):
            raise CheckpointError(f"A manifest file is another file's parent: {name}")
    prefix = manifest.get("workspace_archive_prefix")
    if prefix is not None:
        _relative_path(prefix)
        for entry in manifest["files"]:
            if entry["path"] == prefix:
                raise CheckpointError("Workspace archive prefix names a file")
    return manifest


class _PartsReader(io.RawIOBase):
    """Concatenate parts without a temporary archive; verify while reading."""

    def __init__(self, bundle, manifest):
        super().__init__()
        self.bundle = bundle
        self.manifest = manifest
        self.index = 0
        self.handle = None
        self.part_hash = None
        self.part_bytes = 0
        self.archive_hash = hashlib.sha256()
        self.archive_bytes = 0
        self.finished = False

    def readable(self):
        return True

    def readinto(self, buffer):
        if self.finished:
            return 0
        view = memoryview(buffer)
        used = 0
        while used < len(view):
            if self.handle is None:
                if self.index == len(self.manifest["parts"]):
                    if self.archive_bytes != self.manifest["archive_bytes"]:
                        raise CheckpointError("Concatenated archive size mismatch")
                    if self.archive_hash.hexdigest() != self.manifest["archive_sha256"]:
                        raise CheckpointError("Concatenated archive SHA-256 mismatch")
                    self.finished = True
                    break
                entry = self.manifest["parts"][self.index]
                path = _safe_target(self.bundle, entry["path"])
                info = path.lstat()
                if not stat.S_ISREG(info.st_mode) or info.st_size != entry["size"]:
                    raise CheckpointError(f"Archive part type or size mismatch: {path}")
                self.handle = path.open("rb")
                self.part_hash = hashlib.sha256()
                self.part_bytes = 0
            count = self.handle.readinto(view[used:])
            if not count:
                entry = self.manifest["parts"][self.index]
                self.handle.close()
                self.handle = None
                if self.part_bytes != entry["size"] or self.part_hash.hexdigest() != entry["sha256"]:
                    raise CheckpointError(f"Archive part SHA-256 mismatch: {entry['path']}")
                self.index += 1
                continue
            block = view[used:used + count]
            self.part_hash.update(block)
            self.archive_hash.update(block)
            self.part_bytes += count
            self.archive_bytes += count
            used += count
        return used

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        super().close()


def _hash_file(path):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def _anchor_relative(manifest, path):
    prefix = manifest.get("workspace_archive_prefix")
    if prefix and path.startswith(prefix + "/"):
        relative = path[len(prefix) + 1:]
        _relative_path(relative)
        return relative
    return None


def _destinations(manifest, repo, *, restore, anchors, working_tree):
    destinations = {}
    for entry in manifest["files"]:
        anchor_relative = _anchor_relative(manifest, entry["path"])
        if anchor_relative is not None and anchors is not None:
            target = _safe_target(anchors, anchor_relative)
        elif working_tree and anchor_relative is not None and manifest.get("workspace_anchor_root"):
            anchor_root = _root(manifest["workspace_anchor_root"])
            target = _safe_target(anchor_root, anchor_relative)
        elif working_tree or restore:
            target = _safe_target(repo, entry["path"])
        else:
            continue
        destinations[entry["path"]] = target
    targets = list(destinations.values())
    if len(set(targets)) != len(targets):
        raise CheckpointError("Manifest paths map to duplicate destinations")
    target_set = set(targets)
    for target in targets:
        if any(parent in target_set for parent in target.parents):
            raise CheckpointError(f"A destination file is another file's parent: {target}")
    return destinations


def _preflight(manifest, destinations, *, allow_missing):
    missing = {}
    for entry in manifest["files"]:
        target = destinations.get(entry["path"])
        if target is None:
            continue
        try:
            info = target.lstat()
        except FileNotFoundError:
            if not allow_missing:
                raise CheckpointError(f"Working-tree file is missing: {target}") from None
            missing[entry["path"]] = target
            continue
        if not stat.S_ISREG(info.st_mode):
            raise CheckpointError(f"Existing destination is not a regular file: {target}")
        if info.st_size != entry["size"] or _hash_file(target)[1] != entry["sha256"]:
            raise CheckpointError(f"Conflicting existing file; nothing overwritten: {target}")
    # Group by filesystem so several roots on the same disk cannot oversubscribe it.
    required = {}
    for entry in manifest["files"]:
        if entry["path"] not in missing:
            continue
        ancestor = missing[entry["path"]].parent
        while not ancestor.exists():
            ancestor = ancestor.parent
        device = ancestor.stat().st_dev
        prior_path, prior_bytes = required.get(device, (ancestor, 0))
        required[device] = (prior_path, prior_bytes + entry["size"])
    for ancestor, size in required.values():
        free = shutil.disk_usage(ancestor).free
        if size > free:
            raise CheckpointError(f"Insufficient disk space at {ancestor}: need {size}, available {free}")
    return missing


def _publish(temp_path, target, entry):
    """An exclusive create prevents overwriting a concurrent destination."""
    try:
        # Linking avoids a second full-sized copy and never replaces a file.
        os.link(temp_path, target)
    except FileExistsError:
        raise CheckpointError(f"Destination appeared during restore: {target}") from None
    except OSError:
        # Filesystems without hard links can still restore with exclusive create.
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, entry["mode"])
        try:
            with os.fdopen(descriptor, "wb") as output, temp_path.open("rb") as source:
                shutil.copyfileobj(source, output, CHUNK)
        except BaseException:
            target.unlink(missing_ok=True)
            raise
    os.chmod(target, entry["mode"])


def scan_archive(bundle, manifest, missing=None):
    records = {entry["path"]: entry for entry in manifest["files"]}
    seen = set()
    restored = 0
    with _PartsReader(bundle, manifest) as raw, io.BufferedReader(raw, CHUNK) as buffered:
        with gzip.GzipFile(fileobj=buffered, mode="rb") as decompressed:
            with tarfile.open(fileobj=decompressed, mode="r|") as archive:
                for member in archive:
                    _relative_path(member.name)
                    if not member.isfile() or member.name not in records:
                        raise CheckpointError(f"Unexpected archive member: {member.name}")
                    if member.name in seen:
                        raise CheckpointError(f"Duplicate archive member: {member.name}")
                    seen.add(member.name)
                    entry = records[member.name]
                    if member.size != entry["size"]:
                        raise CheckpointError(f"Archive member size mismatch: {member.name}")
                    digest = hashlib.sha256()
                    size = 0
                    temp_path = None
                    output = None
                    try:
                        target = missing.get(member.name) if missing else None
                        if target is not None:
                            # Repeat the symlink check immediately before writing.
                            safe_root = Path(target.anchor)
                            _safe_target(safe_root, target.as_posix()[len(target.anchor):])
                            target.parent.mkdir(parents=True, exist_ok=True)
                            output = tempfile.NamedTemporaryFile(mode="wb", prefix=".genesis-restore-", dir=target.parent, delete=False)
                            temp_path = Path(output.name)
                        source = archive.extractfile(member)
                        if source is None:
                            raise CheckpointError(f"Unreadable archive member: {member.name}")
                        with source:
                            while block := source.read(CHUNK):
                                size += len(block)
                                digest.update(block)
                                if output is not None:
                                    output.write(block)
                        if size != entry["size"] or digest.hexdigest() != entry["sha256"]:
                            raise CheckpointError(f"Archive member SHA-256 mismatch: {member.name}")
                        if output is not None:
                            output.close()
                            output = None
                            _publish(temp_path, target, entry)
                            restored += 1
                    finally:
                        if output is not None:
                            output.close()
                        if temp_path is not None:
                            temp_path.unlink(missing_ok=True)
                # Consume tar padding and the gzip trailer, including its CRC.
                while block := archive.fileobj.read(CHUNK):
                    if any(block):
                        raise CheckpointError("Unexpected nonzero bytes after the tar end marker")
            # Ensure every compressed byte, including any final part, was consumed.
            while decompressed.read(CHUNK):
                raise CheckpointError("Unexpected data after the tar stream")
        if not raw.finished:
            while buffered.read(CHUNK):
                raise CheckpointError("Unexpected trailing compressed bytes")
    if seen != set(records):
        absent = sorted(set(records) - seen)
        raise CheckpointError(f"Archive is missing {len(absent)} files: {absent[:3]}")
    return restored


def run(bundle, repo, *, restore=False, restore_anchors=None, verify_working_tree=False):
    bundle = _root(bundle)
    repo = _root(repo)
    anchors = _root(restore_anchors, may_be_missing=True) if restore_anchors is not None else None
    manifest = load_manifest(bundle)
    if anchors is not None and not manifest.get("workspace_archive_prefix"):
        raise CheckpointError("This manifest has no workspace anchor prefix")
    scan_archive(bundle, manifest)
    destinations = _destinations(manifest, repo, restore=restore, anchors=anchors, working_tree=verify_working_tree)
    missing = _preflight(manifest, destinations, allow_missing=restore or anchors is not None)
    if verify_working_tree and missing:
        raise CheckpointError("--verify-working-tree requires every mapped file to exist")
    restored = scan_archive(bundle, manifest, missing) if missing else 0
    return {
        "status": "restored" if restore or anchors is not None else "verified",
        "format": FORMAT,
        "verified_files": manifest["raw_file_count"],
        "verified_bytes": manifest["raw_bytes"],
        "archive_bytes": manifest["archive_bytes"],
        "restored_files": restored,
        "checked_existing_files": len(destinations) - len(missing),
    }


def main(argv=None):
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", nargs="?", type=Path, default=repo / "artifact_checkpoints/20261011-genesis")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--verify", action="store_true", help="verify the full archive without extracting (default)")
    action.add_argument("--restore", action="store_true", help="restore missing repository files; refuse conflicting files")
    parser.add_argument("--restore-anchors", type=Path, metavar="PATH", help="restore workspace anchors under PATH; alone, restore anchors only")
    parser.add_argument("--verify-working-tree", action="store_true", help="also verify the original repository and workspace-anchor files")
    arguments = parser.parse_args(argv)
    if arguments.verify and arguments.restore_anchors is not None:
        parser.error("--verify cannot be combined with --restore-anchors")
    try:
        result = run(arguments.bundle, repo, restore=arguments.restore, restore_anchors=arguments.restore_anchors, verify_working_tree=arguments.verify_working_tree)
    except (CheckpointError, OSError, ValueError, tarfile.TarError, EOFError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
