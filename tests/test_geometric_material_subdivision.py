"""Direct material lineage cannot hide changed geometry, history or basal load."""
from dataclasses import asdict, replace
import hashlib
import json

import numpy as np
import pytest

from tectonics.basal_coupling import PrescribedBasalParameters
from tectonics.fractional_surface import EXTENSIVE_FIELDS, FractionalSurfaceState, SurfaceParcel
from tectonics.geometric_basal_loading import (assemble_geometric_basal_load,
    basal_load_from_dict, basal_load_to_dict)
from tectonics.geometric_material_subdivision import (validate_material_subdivision,
    validate_subdivision_basal_rebind)
from tectonics.geometric_surface import from_fractional_surface, split_geometry
from tectonics.mesh import build_icosphere

PARAMETERS = PrescribedBasalParameters(17, 80000., 1e14, 2.)
PROVENANCE = "Explicit synthetic source shared by conservative material views"


def world():
    mesh = build_icosphere(0)
    radius = 1000.
    areas = mesh.physical_cell_areas_km2(radius)
    parcels = tuple(SurfaceParcel(i, 7 if mesh.centroids[i, 0] > 0 else 10**6, f"origin:{i}",
        float(area), float(area*2.), float(area*3.), float(area*3e9), 10.+i,
        (("damage", i/20.), ("water", .5))) for i, area in enumerate(areas))
    surface = from_fractional_surface(mesh, FractionalSurfaceState(10., tuple(areas), parcels), radius)
    lid = {fragment.fragment_id: .1+8*(mesh.centroids[fragment.parcel.cell, 2]+1)/2
           for fragment in surface.fragments}
    return mesh, surface, lid


def divide(fragment):
    a, b, c = np.asarray(fragment.polygon)
    middle = b+c
    middle /= np.linalg.norm(middle)
    return split_geometry(fragment, (np.array([a, b, middle]), np.array([a, middle, c])),
                          cells=(123, 456), label="material-cut")


def split(surface, lid):
    parent = surface.fragments[0]
    children = divide(parent)
    new_surface = replace(surface, fragments=children+surface.fragments[1:])
    new_lid = {key: value for key, value in lid.items() if key != parent.fragment_id}
    new_lid.update({child.fragment_id: lid[parent.fragment_id] for child in children})
    return new_surface, new_lid


def frozen(value):
    return json.dumps(asdict(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def source(mesh, surface, lid, parameters=PARAMETERS, provenance=PROVENANCE):
    return assemble_geometric_basal_load(mesh, surface, parameters, lid, provenance)


def rewritten_load(surface, load, mutate):
    raw = basal_load_to_dict(surface, load)
    mutate(raw)
    body = {key: value for key, value in raw.items() if key != "load_sha256"}
    raw["load_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    return basal_load_from_dict(raw, surface)


def test_direct_split_has_immutable_complete_mapping_and_does_not_mutate_history():
    _, original, lid = world()
    subdivided, _ = split(original, lid)
    before = (frozen(original), frozen(subdivided))
    mapping = validate_material_subdivision(original, subdivided)
    assert isinstance(mapping, tuple)
    assert mapping == tuple(sorted(mapping))
    assert set(dict(mapping)) == {fragment.fragment_id for fragment in subdivided.fragments}
    for child, parent in mapping:
        assert parent == (original.fragments[0].fragment_id if "/material-cut:" in child else child)
    assert (frozen(original), frozen(subdivided)) == before
    reordered = replace(subdivided, fragments=tuple(reversed(subdivided.fragments)))
    assert validate_material_subdivision(original, reordered) == mapping


def test_surviving_fragment_keeps_saved_parent_and_allows_only_integration_cell_change():
    _, original, lid = world()
    subdivided, _ = split(original, lid)
    # These already split children retain their earlier parent metadata while
    # acting as unchanged current IDs for this next verification transaction.
    relabelled = replace(subdivided, fragments=tuple(replace(fragment,
        parcel=replace(fragment.parcel, cell=fragment.parcel.cell+1000))
        for fragment in subdivided.fragments))
    mapping = validate_material_subdivision(subdivided, relabelled)
    assert all(child == parent for child, parent in mapping)


def test_small_positive_child_is_preserved_without_an_area_cutoff():
    _, original, _ = world()
    parent = original.fragments[0]
    a, b, c = np.asarray(parent.polygon)
    fraction = 1e-6
    ab = (1.-fraction)*a+fraction*b
    ac = (1.-fraction)*a+fraction*c
    ab /= np.linalg.norm(ab)
    ac /= np.linalg.norm(ac)
    children = split_geometry(parent, (np.array([a, ab, ac]), np.array([ab, b, c, ac])))
    assert 0. < children[0].parcel.area_km2 < parent.parcel.area_km2*1e-10
    subdivided = replace(original, fragments=children+original.fragments[1:])
    mapping = dict(validate_material_subdivision(original, subdivided))
    assert mapping[children[0].fragment_id] == parent.fragment_id
    assert len(mapping) == len(subdivided.fragments)


@pytest.mark.parametrize("field", ["time_myr", "radius_km", "known_material_ids", "phase"])
def test_surface_context_cannot_be_rebound(field):
    _, original, lid = world()
    subdivided, _ = split(original, lid)
    if field == "time_myr":
        changed = replace(subdivided, time_myr=subdivided.time_myr+1.)
    elif field == "known_material_ids":
        changed = replace(subdivided, known_material_ids=subdivided.known_material_ids+("retired",))
    elif field == "phase":
        changed = replace(subdivided, phase="advected_candidates")
    else:
        scale = 2.
        changed = replace(subdivided, radius_km=subdivided.radius_km*scale,
            fragments=tuple(replace(fragment, parcel=replace(fragment.parcel,
                **{name: getattr(fragment.parcel, name)*scale**2 for name in EXTENSIVE_FIELDS}))
                for fragment in subdivided.fragments))
    with pytest.raises(ValueError, match="surface|resolved"):
        validate_material_subdivision(original, changed)


@pytest.mark.parametrize("field", ["plate", "material_id", "age_myr", "material_fields", "specific_properties"])
def test_children_cannot_change_actual_owner_or_material_history(field):
    _, original, lid = world()
    subdivided, _ = split(original, lid)
    child = subdivided.fragments[0]
    changes = {
        "plate": child.parcel.plate+1,
        "material_id": original.fragments[1].parcel.material_id,
        "age_myr": child.parcel.age_myr+1,
        "material_fields": (("damage", .9), ("water", .5)),
        "specific_properties": (4., 3., 3e9),
    }
    values = {field: changes[field]}
    if field == "specific_properties":
        values["oceanic_volume_km3"] = child.parcel.area_km2*4.
    changed_child = replace(child, parcel=replace(child.parcel, **values))
    changed = replace(subdivided, fragments=(changed_child,)+subdivided.fragments[1:])
    with pytest.raises(ValueError, match="history"):
        validate_material_subdivision(original, changed)


def test_equal_area_balanced_geometry_swap_is_rejected_even_when_global_partition_is_valid():
    _, original, _ = world()
    first, second = original.fragments[:2]
    left, right = divide(first), divide(second)
    # Geometry remains the same complete sphere; each donor gets the other
    # donor's footprint and retains its own material. Global area cannot catch
    # this swap, whereas explicit per-donor containment must do so.
    swapped = tuple(replace(child, polygon=other.polygon)
                    for children, others in ((left, right), (right, left))
                    for child, other in zip(children, others))
    changed = replace(original, fragments=swapped+original.fragments[2:])
    with pytest.raises(ValueError, match="outside"):
        validate_material_subdivision(original, changed)


def test_positive_overlap_is_rejected_without_a_minimum_area_filter():
    _, original, lid = world()
    subdivided, _ = split(original, lid)
    child = replace(subdivided.fragments[1], polygon=subdivided.fragments[0].polygon)
    overlapping = replace(subdivided, fragments=(subdivided.fragments[0], child)+subdivided.fragments[2:])
    with pytest.raises(ValueError, match="overlapping"):
        validate_material_subdivision(original, overlapping)


def test_area_proportions_use_arithmetic_tolerance_not_the_geometric_area_tolerance():
    _, original, lid = world()
    subdivided, _ = split(original, lid)
    children = []
    for child, factor in zip(subdivided.fragments[:2], (1.+1e-11, 1.-1e-11)):
        children.append(replace(child, parcel=replace(child.parcel,
            **{name: getattr(child.parcel, name)*factor for name in EXTENSIVE_FIELDS})))
    changed = replace(subdivided, fragments=tuple(children)+subdivided.fragments[2:])
    with pytest.raises(ValueError, match="subdivision changed area"):
        validate_material_subdivision(original, changed)


@pytest.mark.parametrize("what", ["missing_parent", "grandparent", "retained_id_geometry", "retained_id_lineage"])
def test_rewriting_lineage_or_reusing_old_id_cannot_hide_a_subdivision(what):
    _, original, lid = world()
    subdivided, _ = split(original, lid)
    if what in ("missing_parent", "grandparent"):
        child = replace(subdivided.fragments[0], parent_fragment_id=None if what == "missing_parent" else "grandparent")
        changed = replace(subdivided, fragments=(child,)+subdivided.fragments[1:])
    else:
        fragment = original.fragments[0]
        changed_fragment = (replace(fragment, polygon=fragment.polygon[1:]+fragment.polygon[:1])
                            if what == "retained_id_geometry" else replace(fragment, parent_fragment_id="new-lineage"))
        changed = replace(original, fragments=(changed_fragment,)+original.fragments[1:])
    with pytest.raises(ValueError, match="lineage|footprint"):
        validate_material_subdivision(original, changed)


def test_basal_source_rebind_inherits_full_lid_and_operators_with_only_roundoff():
    mesh, original, lid = world()
    subdivided, new_lid = split(original, lid)
    old_load, new_load = source(mesh, original, lid), source(mesh, subdivided, new_lid)
    before = (frozen(original), frozen(subdivided), frozen(old_load), frozen(new_load))
    mapping = validate_material_subdivision(original, subdivided)
    assert validate_subdivision_basal_rebind(original, subdivided, old_load, new_load, mapping) is new_load
    assert (frozen(original), frozen(subdivided), frozen(old_load), frozen(new_load)) == before


@pytest.mark.parametrize("what", ["mapping", "lid", "parameters", "provenance", "mesh", "torque", "drag"])
def test_basal_rebind_rejects_changed_source_despite_consistent_record_checksums(what):
    mesh, original, lid = world()
    subdivided, new_lid = split(original, lid)
    old_load = source(mesh, original, lid)
    mapping = validate_material_subdivision(original, subdivided)
    if what == "lid":
        new_lid[subdivided.fragments[0].fragment_id] += .1
    new_mesh = build_icosphere(1) if what == "mesh" else mesh
    new_load = source(new_mesh, subdivided, new_lid,
        replace(PARAMETERS, seed=18) if what == "parameters" else PARAMETERS,
        "changed source" if what == "provenance" else PROVENANCE)
    if what == "mapping":
        mapping = mapping[:-1]
    elif what == "torque":
        def mutate(raw):
            raw["torque_nm"] = list(raw["torque_nm"])
            raw["torque_nm"][0] += np.linalg.norm(raw["torque_nm"])*1e-4
        new_load = rewritten_load(subdivided, new_load, mutate)
    elif what == "drag":
        def mutate(raw):
            drag = np.asarray(raw["drag_nm_s"])
            drag[0, 0] *= 1.+1e-4
            raw["drag_nm_s"] = drag.tolist()
        new_load = rewritten_load(subdivided, new_load, mutate)
    with pytest.raises(ValueError, match="lineage|source|thermal lid|basal"):
        validate_subdivision_basal_rebind(original, subdivided, old_load, new_load, mapping)


def test_zero_source_has_no_absolute_torque_tolerance():
    mesh, original, lid = world()
    subdivided, new_lid = split(original, lid)
    parameters = replace(PARAMETERS, convective_traction_pa=0.)
    old_load, new_load = source(mesh, original, lid, parameters), source(mesh, subdivided, new_lid, parameters)
    def mutate(raw):
        raw["torque_nm"] = list(raw["torque_nm"])
        raw["torque_nm"][0] = 1.
    fabricated = rewritten_load(subdivided, new_load, mutate)
    mapping = validate_material_subdivision(original, subdivided)
    with pytest.raises(ValueError, match="basal driving torque"):
        validate_subdivision_basal_rebind(original, subdivided, old_load, fabricated, mapping)
