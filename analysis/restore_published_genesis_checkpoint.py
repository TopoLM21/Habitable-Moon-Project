#!/usr/bin/env python3
"""Verify or restore the exact Genesis checkpoint published in split parts.

Default: fetch the separate data branch (or download legacy release assets)
and verify the split Git bundle. --verify-local checks the
already available original commit and omitted working-tree files without any
downloads or writes. --restore imports a separate checkpoint ref and creates
only missing omitted files; it never switches or advances the active branch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile


FORMAT = "genesis-publication-v1"
CHUNK = 1024 * 1024
GIT_OBJECT_RESERVE = 4 * 1024 ** 3
MAX_PART_BYTES = 64 * 1024 ** 2


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


def _safe_target(root, relative):
    target = root
    pieces = _relative_path(relative)
    for index, piece in enumerate(pieces):
        target = target / piece
        try:
            info = target.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode):
            raise CheckpointError(f"Symlink path refused: {target}")
        if index < len(pieces) - 1 and not stat.S_ISDIR(info.st_mode):
            raise CheckpointError(f"Path parent is not a directory: {target}")
    return target


def _root(path, *, may_be_missing=False):
    path = Path(path).absolute()
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


def _integer(value, label):
    if type(value) is not int or value < 0:
        raise CheckpointError(f"Invalid nonnegative integer: {label}")
    return value


def _digest(value, length, label):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{" + str(length) + r"}", value):
        raise CheckpointError(f"Invalid digest: {label}")
    return value


def load_manifest(path):
    if path.is_symlink() or not path.is_file():
        raise CheckpointError(f"Missing or symlinked publication manifest: {path}")
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle, object_pairs_hook=_json_object)
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise CheckpointError("Unsupported publication manifest format")
    storage = manifest.get("storage", "release")
    if storage not in ("release", "git-branch"):
        raise CheckpointError("Unsupported publication storage")
    if not isinstance(manifest.get("repository"), str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", manifest["repository"]):
        raise CheckpointError("Invalid GitHub repository")
    tag = manifest.get("release_tag")
    if not isinstance(tag, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", tag) or ".." in tag:
        raise CheckpointError("Unsafe release tag")
    _digest(manifest.get("original_commit"), 40, "original_commit")
    ref = manifest.get("bundle_ref")
    if not isinstance(ref, str) or not ref.startswith("refs/heads/") or any(char in ref for char in "\x00\r\n :~^?*[\\") or ".." in ref:
        raise CheckpointError("Unsafe bundle ref")
    if storage == "git-branch":
        branch = manifest.get("data_branch")
        if not isinstance(branch, str) or not branch or branch.startswith("-") or any(char in branch for char in "\x00\r\n :~^?*[\\") or ".." in branch:
            raise CheckpointError("Unsafe data branch")
        if "data_commit" in manifest:
            _digest(manifest["data_commit"], 40, "data_commit")
    _integer(manifest.get("bundle_bytes"), "bundle_bytes")
    _digest(manifest.get("bundle_sha256"), 64, "bundle_sha256")
    parts = manifest.get("parts")
    files = manifest.get("omitted_files")
    if not isinstance(parts, list) or not parts or not isinstance(files, list):
        raise CheckpointError("Invalid parts or omitted_files list")
    names = set()
    part_paths = set()
    for part in parts:
        if not isinstance(part, dict):
            raise CheckpointError("Invalid publication part")
        name = part.get("name")
        if len(_relative_path(name)) != 1 or name.startswith("-") or any(char in name for char in "*?[]"):
            raise CheckpointError(f"Unsafe release asset name: {name!r}")
        if name in names:
            raise CheckpointError(f"Duplicate release asset: {name}")
        names.add(name)
        _integer(part.get("size"), name)
        if not 0 < part["size"] <= MAX_PART_BYTES:
            raise CheckpointError(f"Release asset must be between 1 byte and 64 MiB: {name}")
        _digest(part.get("sha256"), 64, name)
        if storage == "git-branch":
            _relative_path(part.get("path"))
            if part["path"] in part_paths:
                raise CheckpointError(f"Duplicate data-branch path: {part['path']}")
            part_paths.add(part["path"])
            _digest(part.get("blob_sha1"), 40, name)
    if sum(part["size"] for part in parts) != manifest["bundle_bytes"]:
        raise CheckpointError("Part sizes disagree with bundle_bytes")
    paths = set()
    for entry in files:
        if not isinstance(entry, dict):
            raise CheckpointError("Invalid omitted file record")
        _relative_path(entry.get("path"))
        if entry["path"] == ".git" or entry["path"].startswith(".git/"):
            raise CheckpointError("Git metadata cannot be restored as an omitted file")
        if entry["path"] in paths:
            raise CheckpointError(f"Duplicate omitted file: {entry['path']}")
        paths.add(entry["path"])
        _integer(entry.get("size"), entry["path"])
        _digest(entry.get("blob_sha1"), 40, entry["path"])
        if entry.get("mode") not in ("100644", "100755"):
            raise CheckpointError(f"Unsupported Git file mode: {entry['path']}")
    for path in paths:
        pieces = path.split("/")
        if any("/".join(pieces[:index]) in paths for index in range(1, len(pieces))):
            raise CheckpointError(f"An omitted file is another file's parent: {path}")
    return manifest


def _command(arguments, *, input=None, allow_missing=False, env=None):
    result = subprocess.run(arguments, input=input, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    if result.returncode:
        if allow_missing and result.returncode == 1:
            return None
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise CheckpointError(f"{arguments[0]} failed ({result.returncode}): {detail[:1500]}")
    return result.stdout


def _git_environment():
    environment = os.environ.copy()
    # A local verification must not download promised partial-clone objects.
    environment["GIT_NO_LAZY_FETCH"] = "1"
    return environment


def _git(repo, *arguments, **kwargs):
    return _command(["git", "-C", str(repo), *arguments], env=_git_environment(), **kwargs)


def _ref_commit(repo, reference):
    output = _git(repo, "rev-parse", "--verify", "--quiet", reference, allow_missing=True)
    return output.decode("ascii").strip() if output is not None else None


def _hash_file(path, *, blob_size=None):
    digest = hashlib.sha256() if blob_size is None else hashlib.sha1()
    if blob_size is not None:
        digest.update(f"blob {blob_size}\0".encode("ascii"))
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(CHUNK):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def _preflight_files(repo, files, *, allow_missing):
    destinations = {entry["path"]: _safe_target(repo, entry["path"]) for entry in files}
    targets = set(destinations.values())
    if len(targets) != len(destinations):
        raise CheckpointError("Omitted paths map to duplicate destinations")
    for target in targets:
        if any(parent in targets for parent in target.parents):
            raise CheckpointError(f"A destination file is another file's parent: {target}")
    missing = {}
    for entry in files:
        target = destinations[entry["path"]]
        try:
            info = target.lstat()
        except FileNotFoundError:
            if not allow_missing:
                raise CheckpointError(f"Missing preserved file: {target}") from None
            missing[entry["path"]] = target
            continue
        if not stat.S_ISREG(info.st_mode):
            raise CheckpointError(f"Existing destination is not a regular file: {target}")
        if info.st_size != entry["size"] or _hash_file(target, blob_size=entry["size"])[1] != entry["blob_sha1"]:
            raise CheckpointError(f"Conflicting existing file; no files overwritten: {target}")
    return missing


def _verify_original_tree(repo, manifest):
    original = manifest["original_commit"]
    _git(repo, "cat-file", "-e", original + "^{commit}")
    tree = {}
    for record in _git(repo, "ls-tree", "-rz", "--full-tree", original).split(b"\0"):
        if record:
            metadata, path = record.split(b"\t", 1)
            mode, kind, oid = metadata.decode("ascii").split(" ")
            tree[path.decode("utf-8", "surrogateescape")] = (mode, kind, oid)
    for entry in manifest["omitted_files"]:
        expected = (entry["mode"], "blob", entry["blob_sha1"])
        if tree.get(entry["path"]) != expected:
            raise CheckpointError(f"Original commit tree disagrees with manifest: {entry['path']}")
    # Check stored object type and size without decompressing their contents.
    records = manifest["omitted_files"]
    request = b"".join((entry["blob_sha1"] + "\n").encode("ascii") for entry in records)
    response = _git(repo, "cat-file", "--batch-check", input=request).splitlines()
    if len(response) != len(records):
        raise CheckpointError("Git object-size readback returned the wrong record count")
    for entry, line in zip(records, response):
        if line.decode("ascii").split() != [entry["blob_sha1"], "blob", str(entry["size"])]:
            raise CheckpointError(f"Git object type or size mismatch: {entry['path']}")


def _check_space(requirements):
    grouped = {}
    for location, size in requirements:
        ancestor = location
        while not ancestor.exists():
            ancestor = ancestor.parent
        device = ancestor.stat().st_dev
        path, total = grouped.get(device, (ancestor, 0))
        grouped[device] = (path, total + size)
    for location, required in grouped.values():
        available = shutil.disk_usage(location).free
        if required > available:
            raise CheckpointError(f"Insufficient disk space at {location}: need {required} bytes, available {available}")


def _part_valid(path, part):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(info.st_mode) or info.st_size != part["size"]:
        raise CheckpointError(f"Cached release asset has the wrong type or size: {path}")
    if _hash_file(path)[1] != part["sha256"]:
        raise CheckpointError(f"Cached release asset SHA-256 mismatch: {path}")
    if "blob_sha1" in part and _hash_file(path, blob_size=part["size"])[1] != part["blob_sha1"]:
        raise CheckpointError(f"Part Git blob SHA-1 mismatch: {path}")
    return True


def _download_part(cache, manifest, part):
    target = _safe_target(cache, part["name"])
    with tempfile.TemporaryDirectory(prefix=".genesis-download-", dir=cache) as directory:
        _command(["gh", "release", "download", manifest["release_tag"], "--repo", manifest["repository"], "--pattern", part["name"], "--dir", directory])
        downloaded = _safe_target(Path(directory), part["name"])
        if not _part_valid(downloaded, part):
            raise CheckpointError(f"Release asset was not downloaded: {part['name']}")
        # Cache paths are exclusively created too; failed downloads stay temporary.
        created = False
        try:
            with downloaded.open("rb") as source, target.open("xb") as output:
                created = True
                shutil.copyfileobj(source, output, CHUNK)
        except BaseException:
            if created:
                target.unlink(missing_ok=True)
            raise


def _assemble_bundle(parts_root, manifest, cache):
    digest = hashlib.sha256()
    size = 0
    temp = tempfile.NamedTemporaryFile(mode="wb", prefix="genesis-", suffix=".bundle", dir=cache, delete=False)
    path = Path(temp.name)
    try:
        with temp:
            for part in manifest["parts"]:
                part_path = _safe_target(parts_root, part["name"])
                if not _part_valid(part_path, part):
                    raise CheckpointError(f"Missing bundle part: {part_path}")
                with part_path.open("rb") as source:
                    while block := source.read(CHUNK):
                        temp.write(block)
                        digest.update(block)
                        size += len(block)
        if size != manifest["bundle_bytes"] or digest.hexdigest() != manifest["bundle_sha256"]:
            raise CheckpointError("Concatenated Git bundle size or SHA-256 mismatch")
        return path
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _verify_data_tree(repo, manifest, commit):
    _git(repo, "cat-file", "-e", commit + "^{commit}")
    expected_commit = manifest.get("data_commit")
    if expected_commit is not None and commit != expected_commit:
        raise CheckpointError("Data branch does not identify the declared data_commit")
    tree = {}
    for record in _git(repo, "ls-tree", "-rz", "--full-tree", commit).split(b"\0"):
        if record:
            metadata, path = record.split(b"\t", 1)
            mode, kind, oid = metadata.decode("ascii").split(" ")
            tree[path.decode("utf-8", "surrogateescape")] = (mode, kind, oid)
    for part in manifest["parts"]:
        entry = tree.get(part["path"])
        if entry is None or entry[0] not in ("100644", "100755") or entry[1:] != ("blob", part["blob_sha1"]):
            raise CheckpointError(f"Data branch tree disagrees with bundle part: {part['path']}")
    request = b"".join((part["blob_sha1"] + "\n").encode("ascii") for part in manifest["parts"])
    response = _git(repo, "cat-file", "--batch-check", input=request).splitlines()
    if len(response) != len(manifest["parts"]):
        raise CheckpointError("Data branch object-size readback returned the wrong record count")
    for part, line in zip(manifest["parts"], response):
        if line.decode("ascii").split() != [part["blob_sha1"], "blob", str(part["size"])]:
            raise CheckpointError(f"Data branch blob type or size mismatch: {part['path']}")


def _find_local_data(repo, manifest, data_ref):
    candidates = []
    if manifest.get("data_commit"):
        candidates.append(manifest["data_commit"])
    for reference in (data_ref, "refs/heads/" + manifest["data_branch"], "refs/remotes/origin/" + manifest["data_branch"]):
        commit = _ref_commit(repo, reference)
        if commit is not None:
            candidates.append(commit)
    for commit in dict.fromkeys(candidates):
        try:
            _verify_data_tree(repo, manifest, commit)
        except CheckpointError:
            continue
        return commit
    return None


def _assemble_git_bundle(repo, manifest, cache):
    """Stream Git part blobs into one bundle; never copy/cache individual parts."""
    output = tempfile.NamedTemporaryFile(mode="wb", prefix="genesis-", suffix=".bundle", dir=cache, delete=False)
    path = Path(output.name)
    process = subprocess.Popen(["git", "-C", str(repo), "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=_git_environment())
    whole_digest = hashlib.sha256()
    whole_size = 0
    try:
        with output:
            for part in manifest["parts"]:
                process.stdin.write((part["blob_sha1"] + "\n").encode("ascii"))
                process.stdin.flush()
                header = process.stdout.readline(256).decode("ascii").split()
                if header != [part["blob_sha1"], "blob", str(part["size"])]:
                    raise CheckpointError(f"Unexpected Git part header: {part['path']}")
                part_digest = hashlib.sha256()
                blob_digest = hashlib.sha1(f"blob {part['size']}\0".encode("ascii"))
                remaining = part["size"]
                while remaining:
                    block = process.stdout.read(min(CHUNK, remaining))
                    if not block:
                        raise CheckpointError(f"Truncated Git part: {part['path']}")
                    output.write(block)
                    whole_digest.update(block)
                    part_digest.update(block)
                    blob_digest.update(block)
                    whole_size += len(block)
                    remaining -= len(block)
                if process.stdout.read(1) != b"\n" or part_digest.hexdigest() != part["sha256"] or blob_digest.hexdigest() != part["blob_sha1"]:
                    raise CheckpointError(f"Git part SHA-256 or blob SHA-1 mismatch: {part['path']}")
        process.stdin.close()
        if process.wait() != 0:
            raise CheckpointError("git cat-file failed while reading checkpoint parts")
        if whole_size != manifest["bundle_bytes"] or whole_digest.hexdigest() != manifest["bundle_sha256"]:
            raise CheckpointError("Concatenated Git bundle size or SHA-256 mismatch")
        return path
    except BaseException:
        output.close()
        path.unlink(missing_ok=True)
        raise
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        for handle in (process.stdin, process.stdout):
            if handle is not None:
                handle.close()


def _restore_blobs(repo, records, missing):
    restored = 0
    process = subprocess.Popen(["git", "-C", str(repo), "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=_git_environment())
    try:
        for entry in records:
            target = missing.get(entry["path"])
            if target is None:
                continue
            _safe_target(repo, entry["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            process.stdin.write((entry["blob_sha1"] + "\n").encode("ascii"))
            process.stdin.flush()
            header = process.stdout.readline(256).decode("ascii").split()
            if header != [entry["blob_sha1"], "blob", str(entry["size"])]:
                raise CheckpointError(f"Unexpected Git blob header: {entry['path']}")
            created = False
            try:
                with target.open("xb") as output:
                    created = True
                    digest = hashlib.sha1(f"blob {entry['size']}\0".encode("ascii"))
                    remaining = entry["size"]
                    while remaining:
                        block = process.stdout.read(min(CHUNK, remaining))
                        if not block:
                            raise CheckpointError(f"Truncated Git blob: {entry['path']}")
                        output.write(block)
                        digest.update(block)
                        remaining -= len(block)
                if process.stdout.read(1) != b"\n" or digest.hexdigest() != entry["blob_sha1"]:
                    raise CheckpointError(f"Restored Git blob SHA-1 mismatch: {entry['path']}")
                os.chmod(target, 0o755 if entry["mode"] == "100755" else 0o644)
                restored += 1
            except BaseException:
                if created:
                    target.unlink(missing_ok=True)
                raise
        process.stdin.close()
        if process.wait() != 0:
            raise CheckpointError("git cat-file failed during restoration")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        for handle in (process.stdin, process.stdout):
            if handle is not None:
                handle.close()
    return restored


def run(manifest_path, repo, *, restore=False, verify_local=False, parts_dir=None, cache_dir=None):
    repo = _root(repo)
    manifest_path = Path(manifest_path).absolute()
    _safe_target(Path(manifest_path.anchor), manifest_path.as_posix()[len(manifest_path.anchor):])
    manifest = load_manifest(manifest_path)
    storage = manifest.get("storage", "release")
    _git(repo, "check-ref-format", manifest["bundle_ref"])
    import_ref = "refs/checkpoints/" + manifest["release_tag"]
    _git(repo, "check-ref-format", import_ref)
    data_ref = "refs/checkpoint-data/" + manifest["release_tag"]
    if storage == "git-branch":
        _git(repo, "check-ref-format", "refs/heads/" + manifest["data_branch"])
        _git(repo, "check-ref-format", data_ref)
    original = manifest["original_commit"]
    total_bytes = sum(entry["size"] for entry in manifest["omitted_files"])
    if verify_local:
        local_ref = manifest["bundle_ref"]
        value = _ref_commit(repo, local_ref)
        if value is None:
            local_ref = import_ref
            value = _ref_commit(repo, local_ref)
        if value != original:
            raise CheckpointError("Local checkpoint ref does not identify original_commit")
        _verify_original_tree(repo, manifest)
        _preflight_files(repo, manifest["omitted_files"], allow_missing=False)
        return {"status": "verified-local", "storage": storage, "original_commit": original, "checkpoint_ref": local_ref, "verified_files": len(manifest["omitted_files"]), "verified_bytes": total_bytes, "published_parts_verified": False, "release_assets_verified": False, "restored_files": 0}

    # All existing-file and ref conflicts are checked before downloads or writes.
    missing = _preflight_files(repo, manifest["omitted_files"], allow_missing=True) if restore else {}
    existing_ref = _ref_commit(repo, import_ref)
    if restore and existing_ref not in (None, original):
        raise CheckpointError(f"Conflicting checkpoint ref: {import_ref}")
    cache = _root(cache_dir or Path(tempfile.gettempdir()) / "genesis-publication" / manifest["release_tag"], may_be_missing=True)
    branch_mode = storage == "git-branch" and parts_dir is None
    data_commit = _find_local_data(repo, manifest, data_ref) if branch_mode else None
    parts_root = _root(parts_dir) if parts_dir is not None else cache
    absent = []
    if not branch_mode:
        for part in manifest["parts"]:
            if not _part_valid(_safe_target(parts_root, part["name"]), part):
                if parts_dir is not None:
                    raise CheckpointError(f"Missing local bundle part: {part['name']}")
                absent.append(part)
    missing_bytes = sum(entry["size"] for entry in manifest["omitted_files"] if entry["path"] in missing)
    requirements = [(cache, manifest["bundle_bytes"] + sum(part["size"] for part in absent) + max((part["size"] for part in absent), default=0))]
    if branch_mode and data_commit is None:
        git_dir = Path(_git(repo, "rev-parse", "--absolute-git-dir").decode("utf-8").strip())
        requirements.append((git_dir, max(GIT_OBJECT_RESERVE, manifest["bundle_bytes"] + manifest["bundle_bytes"] // 10)))
    if restore:
        git_dir = Path(_git(repo, "rev-parse", "--absolute-git-dir").decode("utf-8").strip())
        requirements.extend([(repo, missing_bytes), (git_dir, GIT_OBJECT_RESERVE if existing_ref is None else 0)])
    _check_space(requirements)
    cache.mkdir(parents=True, exist_ok=True)
    if branch_mode:
        if data_commit is None:
            _git(repo, "fetch", "--no-auto-gc", "--no-tags", "--no-write-fetch-head", "origin", "refs/heads/" + manifest["data_branch"] + ":" + data_ref)
            fetched_tip = _ref_commit(repo, data_ref)
            if fetched_tip is None:
                raise CheckpointError("Fetched data branch has no checkpoint commit")
            data_commit = manifest.get("data_commit", fetched_tip)
            if _git(repo, "merge-base", "--is-ancestor", data_commit, fetched_tip, allow_missing=True) is None:
                raise CheckpointError("Declared data_commit is not an ancestor of the fetched data branch")
            _verify_data_tree(repo, manifest, data_commit)
        bundle = _assemble_git_bundle(repo, manifest, cache)
    else:
        for part in absent:
            _download_part(cache, manifest, part)
        bundle = _assemble_bundle(parts_root, manifest, cache)
    restored = 0
    try:
        heads = _git(repo, "bundle", "list-heads", str(bundle)).decode("utf-8").splitlines()
        if heads != [original + " " + manifest["bundle_ref"]]:
            raise CheckpointError("Bundle does not contain exactly the declared checkpoint ref")
        _git(repo, "bundle", "verify", str(bundle))
        if restore:
            if existing_ref is None:
                _git(repo, "fetch", "--no-auto-gc", "--no-tags", "--no-write-fetch-head", str(bundle), manifest["bundle_ref"] + ":" + import_ref)
            if _ref_commit(repo, import_ref) != original:
                raise CheckpointError("Imported checkpoint ref does not identify original_commit")
            _verify_original_tree(repo, manifest)
            restored = _restore_blobs(repo, manifest["omitted_files"], missing) if missing else 0
    finally:
        bundle.unlink(missing_ok=True)
    return {"status": "restored" if restore else "verified-publication", "storage": storage, "original_commit": original, "checkpoint_ref": import_ref if restore else manifest["bundle_ref"], "data_commit": data_commit, "bundle_bytes": manifest["bundle_bytes"], "verified_parts": len(manifest["parts"]), "verified_files": len(manifest["omitted_files"]) if restore else 0, "verified_bytes": total_bytes if restore else 0, "published_parts_verified": True, "release_assets_verified": storage == "release", "restored_files": restored}


def main(argv=None):
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path, default=repo / "artifact_checkpoints/20261011-genesis/publication_manifest.json")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--verify", action="store_true", help="verify the published checkpoint bundle (default)")
    action.add_argument("--restore", action="store_true", help="import a separate checkpoint ref and restore only missing omitted files")
    action.add_argument("--verify-local", action="store_true", help="verify original local Git ref, tree, and preserved files without downloads or writes")
    parser.add_argument("--parts-dir", type=Path, help="use existing verified bundle parts instead of fetching/downloading")
    parser.add_argument("--cache-dir", type=Path, help="directory for the temporary bundle and any legacy release downloads")
    arguments = parser.parse_args(argv)
    try:
        result = run(arguments.manifest, repo, restore=arguments.restore, verify_local=arguments.verify_local, parts_dir=arguments.parts_dir, cache_dir=arguments.cache_dir)
    except (CheckpointError, OSError, ValueError, EOFError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
