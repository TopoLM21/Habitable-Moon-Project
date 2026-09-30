"""Joint geometry/passive-slab checkpoints with clock and inventory binding."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .fractional_surface import SurfaceParcel
from .geometric_surface import GeometricFragment, GeometricSurfaceState, audit_partition
from .geometric_boundary import boundary_to_dict, boundary_from_dict, surface_digest, _validate_contacts
from .geometric_contacts import extract_contacts

FORMAT = "geometric-boundary-checkpoint-1"


def _encoded(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _validate_pair(surface, inventory):
    if surface.phase != "partition":
        raise ValueError("Joint boundary checkpoint requires resolved surface geometry")
    if (inventory.time_myr != surface.time_myr or inventory.radius_km != surface.radius_km
            or inventory.surface_digest != surface_digest(surface)):
        raise ValueError("Boundary inventory and geometric surface do not describe the same state")
    expected = {contact.contact_id for contact in extract_contacts(surface.fragments, surface.radius_km)}
    if {contact.contact_id for contact in inventory.contacts} != expected:
        raise ValueError("Boundary inventory is missing current geometric contacts")
    _validate_contacts(surface, inventory.contacts)


def save_boundary_checkpoint(path, surface, inventory, *, provenance=None):
    _validate_pair(surface, inventory)
    audit_partition(surface)
    boundary = boundary_to_dict(inventory)
    # Validate the serialized schema before creating a destination file.
    boundary_from_dict(boundary)
    payload = dict(format=FORMAT, surface=asdict(surface), boundary=boundary,
                   provenance={} if provenance is None else provenance)
    payload["payload_sha256"] = hashlib.sha256(_encoded(payload)).hexdigest()
    encoded = _encoded(payload)+b"\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(encoded)
    return path


def load_boundary_checkpoint(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != FORMAT:
        raise ValueError("Expected a joint geometric surface/slab checkpoint; no implicit legacy migration")
    expected = payload.pop("payload_sha256", None)
    if expected != hashlib.sha256(_encoded(payload)).hexdigest():
        raise ValueError("Joint geometric boundary checkpoint integrity mismatch")
    data = dict(payload["surface"])
    fragments = tuple(GeometricFragment(f["fragment_id"], tuple(map(tuple, f["polygon"])),
        SurfaceParcel(**f["parcel"]), f.get("parent_fragment_id")) for f in data.pop("fragments"))
    surface = GeometricSurfaceState(fragments=fragments, **data)
    inventory = boundary_from_dict(payload["boundary"])
    _validate_pair(surface, inventory)
    audit_partition(surface)
    return surface, inventory, payload.get("provenance", {})
