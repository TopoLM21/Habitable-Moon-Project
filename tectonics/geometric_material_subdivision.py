"""Verify direct conservative material subdivision without moving a footprint.

This is a read-only gate for rebinding mechanics to new material bookkeeping.
It does not remap a contact trace, average material history, or reconstruct a
footprint from its area. Every new child must remain inside its stated donor;
the complete old and new surfaces must separately pass the existing strict
spherical partition audit.
"""
from __future__ import annotations

from dataclasses import fields
import math

import numpy as np

from .fractional_surface import EXTENSIVE_FIELDS, SurfaceParcel
from .geometric_basal_loading import validate_geometric_basal_load
from .geometric_surface import AREA_REL_TOL, GeometricSurfaceState, audit_partition
from .spherical_polygons import polygon_area, subtract_convex

_MATERIAL_ROUNDOFF = 64.*np.finfo(float).eps
_OPERATOR_ROUNDOFF = 512.*np.finfo(float).eps
_INTENSIVE_FIELDS = tuple(item.name for item in fields(SurfaceParcel)
                          if item.name not in (*EXTENSIVE_FIELDS, "cell"))


def _roundoff_equal(actual, expected, label):
    # A zero reservoir remains exactly zero. This only covers multiplication
    # and summation roundoff, independently of the geometric area tolerance.
    if not math.isclose(actual, expected, rel_tol=_MATERIAL_ROUNDOFF, abs_tol=0.):
        raise ValueError(f"Material subdivision changed {label}")


def validate_material_subdivision(old_surface, new_surface):
    """Return immutable sorted ``(new_fragment_id, old_fragment_id)`` lineage.

    Surviving IDs preserve their exact saved vertices and material values;
    integration-cell labels may change. All other IDs must name direct children
    of one old fragment. Child material is inherited exactly, and every
    extensive reservoir is proportional to the same normalized spherical
    measure used by :func:`geometric_surface.split_geometry`.

    Containment uses the existing robust spherical clipping predicates, with
    no positive-area cutoff. ``AREA_REL_TOL`` is used only for each donor's
    total geometric measure; arithmetic material checks use 64 binary64 eps.
    """
    if (not isinstance(old_surface, GeometricSurfaceState)
            or not isinstance(new_surface, GeometricSurfaceState)):
        raise ValueError("Material subdivision requires explicit geometric surfaces")
    if old_surface.phase != "partition" or new_surface.phase != "partition":
        raise ValueError("Material subdivision requires two resolved partitions")
    for name in ("time_myr", "radius_km", "phase", "version", "known_material_ids"):
        if getattr(old_surface, name) != getattr(new_surface, name):
            raise ValueError(f"Material subdivision changed surface {name}")

    audit_partition(old_surface)
    audit_partition(new_surface)
    old = {fragment.fragment_id: fragment for fragment in old_surface.fragments}
    groups = {key: [] for key in old}
    lineage = []
    for child in new_surface.fragments:
        if child.fragment_id in old:
            parent = old[child.fragment_id]
            if (child.polygon != parent.polygon
                    or child.parent_fragment_id != parent.parent_fragment_id):
                raise ValueError("A surviving material ID changed its exact footprint or lineage")
            if any(getattr(child.parcel, name) != getattr(parent.parcel, name)
                   for name in (*EXTENSIVE_FIELDS, *_INTENSIVE_FIELDS)):
                raise ValueError("A surviving material ID changed its material history")
        else:
            if child.parent_fragment_id not in old:
                raise ValueError("Material subdivision requires direct old-fragment lineage")
            parent = old[child.parent_fragment_id]
            if any(getattr(child.parcel, name) != getattr(parent.parcel, name)
                   for name in _INTENSIVE_FIELDS):
                raise ValueError("A material child changed its owner's material history")
            # Equal total area cannot prove containment: a moved child can be
            # compensated by another donor's missing area. Reject any resolved
            # outside piece instead of permitting a global balanced swap.
            if subtract_convex(child.polygon, parent.polygon):
                raise ValueError("A material child extends outside its stated donor")
        groups[parent.fragment_id].append(child)
        lineage.append((child.fragment_id, parent.fragment_id))

    for parent_id, children in groups.items():
        if not children:
            raise ValueError("Material subdivision removed an old fragment")
        if any(child.fragment_id == parent_id for child in children):
            if len(children) != 1:
                raise ValueError("A surviving fragment cannot also donate children")
            continue
        parent = old[parent_id]
        measures = tuple(polygon_area(child.polygon) for child in children)
        total_measure = math.fsum(measures)
        parent_measure = polygon_area(parent.polygon)
        if (any(value <= 0. for value in measures)
                or not math.isclose(total_measure, parent_measure,
                                    rel_tol=AREA_REL_TOL, abs_tol=0.)):
            raise ValueError("Material children do not cover their complete donor footprint")
        weights = np.asarray(measures)/total_measure
        weights[int(np.argmax(weights))] += 1.-math.fsum(weights)
        for child, weight in zip(children, weights):
            for name in EXTENSIVE_FIELDS:
                _roundoff_equal(getattr(child.parcel, name),
                                getattr(parent.parcel, name)*float(weight), name)
        for name in EXTENSIVE_FIELDS:
            _roundoff_equal(math.fsum(getattr(child.parcel, name) for child in children),
                            getattr(parent.parcel, name), f"donor total {name}")
    return tuple(sorted(lineage))


def validate_subdivision_basal_rebind(old_surface, new_surface, old_load, new_load, mapping):
    """Check the same prescribed SI source after a verified material split.

    Bound-load validation checks full record integrity and exact surface
    signatures. It is not source authentication: trusted source reassembly is
    still the existing basal loader's responsibility. Source parameters, mesh,
    provenance, actual plate IDs and each inherited thermal lid must agree.

    Per-owner and global drag differences may contain 512 binary64 eps of their
    Frobenius scale. Torque checks use each owner's physical envelope
    ``R * area * prescribed_traction`` as well as its resultant norm; this
    covers cancellation of individually nonzero cell torques without fitting
    forces to any target velocity. A zero-amplitude source has no absolute
    tolerance. No original operator or material value is changed.
    """
    actual_mapping = validate_material_subdivision(old_surface, new_surface)
    if (not isinstance(mapping, tuple)
            or any(not isinstance(item, tuple) or len(item) != 2 for item in mapping)
            or mapping != actual_mapping):
        raise ValueError("Basal rebinding requires the verified complete immutable lineage")
    validate_geometric_basal_load(old_surface, old_load)
    validate_geometric_basal_load(new_surface, new_load)
    for name in ("mesh_sha256", "plate_ids", "source_parameters", "provenance"):
        if getattr(old_load, name) != getattr(new_load, name):
            raise ValueError(f"Material subdivision changed basal source {name}")
    old_lid = dict(old_load.thermal_lid_by_fragment)
    new_lid = dict(new_load.thermal_lid_by_fragment)
    if any(new_lid[child] != old_lid[parent] for child, parent in mapping):
        raise ValueError("Material subdivision changed inherited full thermal lid")

    old_drag, new_drag = np.asarray(old_load.drag_nm_s), np.asarray(new_load.drag_nm_s)
    old_torque, new_torque = np.asarray(old_load.torque_nm), np.asarray(new_load.torque_nm)
    amplitude = dict(old_load.source_parameters)["convective_traction_pa"]
    areas = {plate: math.fsum(fragment.parcel.area_km2*1e6
                             for fragment in old_surface.fragments if fragment.parcel.plate == plate)
             for plate in old_load.plate_ids}
    torque_scales = []
    for index, plate in enumerate(old_load.plate_ids):
        block = slice(3*index, 3*index+3)
        drag_scale = max(float(np.linalg.norm(old_drag[block, block])),
                         float(np.linalg.norm(new_drag[block, block])))
        if np.linalg.norm(new_drag[block, block]-old_drag[block, block]) > _OPERATOR_ROUNDOFF*drag_scale:
            raise ValueError("Material subdivision changed an owner's basal drag")
        torque_scale = max(float(np.linalg.norm(old_torque[block])),
                           float(np.linalg.norm(new_torque[block])),
                           old_surface.radius_km*1000.*areas[plate]*amplitude)
        torque_scales.append(torque_scale)
        if np.linalg.norm(new_torque[block]-old_torque[block]) > _OPERATOR_ROUNDOFF*torque_scale:
            raise ValueError("Material subdivision changed an owner's basal driving torque")
    if np.linalg.norm(new_drag-old_drag) > _OPERATOR_ROUNDOFF*max(
            float(np.linalg.norm(old_drag)), float(np.linalg.norm(new_drag))):
        raise ValueError("Material subdivision changed global basal drag")
    if np.linalg.norm(new_torque-old_torque) > _OPERATOR_ROUNDOFF*math.hypot(*torque_scales):
        raise ValueError("Material subdivision changed global basal driving torque")
    return new_load
