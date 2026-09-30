"""Portable source bundle integrity and non-overwriting restoration contract."""
import hashlib
import io
import json
import stat
import zipfile

import pytest

from scripts import cloud_handoff as bundle


@pytest.fixture
def packed(tmp_path):
    original, archive = tmp_path/"original", tmp_path/"bundle"
    for index, name in enumerate(bundle.SOURCE_FILES):
        target = original/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(f"exact source {index}\r\n".encode())
    report = dict(checkpoint_sha256={
        name: hashlib.sha256((original/bundle.SAVED50/name).read_bytes()).hexdigest()
        for name in bundle.CHECKPOINT_FILES})
    (original/bundle.SAVED50/"continuation.json").write_bytes(json.dumps(report).encode())
    bundle.pack(original, archive)
    return original, archive, tmp_path/"destination"


def test_roundtrip_is_exact_and_second_restore_preserves_files(packed):
    original, archive, destination = packed
    assert bundle.restore(destination, archive)["restored"] == 9
    mtimes = {name: (destination/name).stat().st_mtime_ns for name in bundle.SOURCE_FILES}
    assert bundle.restore(destination, archive) == dict(files=9, restored=0, skipped=9,
        source_bytes=sum((original/name).stat().st_size for name in bundle.SOURCE_FILES))
    assert mtimes == {name: (destination/name).stat().st_mtime_ns for name in bundle.SOURCE_FILES}
    assert bundle.verify(destination, archive)["verified"]
    assert all((destination/name).read_bytes() == (original/name).read_bytes()
               for name in bundle.SOURCE_FILES)


def test_existing_conflict_preflight_writes_nothing(packed):
    _, archive, destination = packed
    target = destination/bundle.SOURCE_FILES[-1]
    target.parent.mkdir(parents=True)
    target.write_bytes(b"user data")
    with pytest.raises(ValueError, match="nothing restored"):
        bundle.restore(destination, archive)
    assert [p for p in destination.rglob("*") if p.is_file()] == [target]
    assert target.read_bytes() == b"user data"


def test_pack_never_overwrites_bundle(packed):
    original, archive, _ = packed
    before = {p.name: p.read_bytes() for p in archive.iterdir()}
    with pytest.raises(ValueError, match="never overwrites"):
        bundle.pack(original, archive)
    assert before == {p.name: p.read_bytes() for p in archive.iterdir()}


def test_corrupt_archive_is_rejected_before_writes(packed):
    _, archive, destination = packed
    path = archive/bundle.ARCHIVE
    path.write_bytes(path.read_bytes()+b"changed")
    with pytest.raises(ValueError, match="archive checksum"):
        bundle.restore(destination, archive)
    assert not destination.exists()


def _rewrite_archive(archive, transform):
    path = archive/bundle.ARCHIVE
    with zipfile.ZipFile(path) as old:
        entries = [(entry, old.read(entry)) for entry in old.infolist()]
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as new:
        for index, (entry, payload) in enumerate(entries):
            entry, payload = transform(index, entry, payload)
            new.writestr(entry, payload)
    path.write_bytes(stream.getvalue())
    manifest_path = archive/bundle.MANIFEST
    manifest = json.loads(manifest_path.read_text())
    manifest["archive_sha256"] = hashlib.sha256(stream.getvalue()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))


@pytest.mark.parametrize("name", ["../outside", "/absolute", "C:/outside", "..\\outside"])
def test_malicious_member_is_rejected_even_with_valid_archive_hash(packed, name):
    _, archive, destination = packed
    def change(index, entry, payload):
        if index == 0:
            entry = zipfile.ZipInfo(name)
        return entry, payload
    _rewrite_archive(archive, change)
    with pytest.raises(ValueError, match="archive members"):
        bundle.restore(destination, archive)
    assert not destination.exists()


def test_payload_checksum_is_verified_independently(packed):
    _, archive, destination = packed
    _rewrite_archive(archive, lambda index, entry, payload:
                     (entry, b"!"+payload[1:]) if index == 0 else (entry, payload))
    with pytest.raises(ValueError, match="payload checksum"):
        bundle.restore(destination, archive)
    assert not destination.exists()


def test_zip_symlink_is_rejected(packed):
    _, archive, destination = packed
    def change(index, entry, payload):
        if index == 0:
            entry.external_attr = (stat.S_IFLNK | 0o777) << 16
        return entry, payload
    _rewrite_archive(archive, change)
    with pytest.raises(ValueError, match="symlink"):
        bundle.restore(destination, archive)
    assert not destination.exists()


def test_saved50_internal_manifest_must_match_before_pack(packed, tmp_path):
    original, _, _ = packed
    (original/bundle.SAVED50/bundle.CHECKPOINT_FILES[0]).write_bytes(b"changed")
    output = tmp_path/"bad_bundle"
    with pytest.raises(ValueError, match="internal checksum"):
        bundle.pack(original, output)
    assert not output.exists()


def test_verify_rejects_changed_restored_source(packed):
    _, archive, destination = packed
    bundle.restore(destination, archive)
    (destination/bundle.SOURCE_FILES[0]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="missing or changed"):
        bundle.verify(destination, archive)


def test_destination_symlink_is_rejected(packed, tmp_path):
    _, archive, destination = packed
    destination.mkdir()
    outside = tmp_path/"outside"
    outside.mkdir()
    try:
        (destination/"analysis").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Creating symlinks requires platform privileges")
    with pytest.raises(ValueError, match="Symlink"):
        bundle.restore(destination, archive)
    assert not list(outside.iterdir())
