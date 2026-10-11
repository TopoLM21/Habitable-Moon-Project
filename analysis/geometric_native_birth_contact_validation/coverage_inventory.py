"""Exhaustive pair inventory for a fresh mixed cold/HOT motion candidate.

This is a geometry inventory, never a force-admission receipt.  It examines
every saved donor/receiver material component, including both sides of the
signed curved correction.  Exact implicit material masks remain authoritative;
native polygons are only certified inner/outer measure enclosures.

The lower temporal enclosure is the endpoint subset.  The upper enclosure
uses the existing full SO(3) Taylor proof over every rational time cell.  A
zero upper measure excludes a pair for the entire interval; a positive lower
measure proves that a pair needs its own physical support; an upper-only pair
remains unresolved.  None of these areas is a material debit or interface area.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time

from tectonics.geometric_native_implicit_epoch_admission import (
    _immediate_edges, _portable_fingerprint_sha256,
)
from tectonics.geometric_native_implicit_material_continuation import (
    NativeImplicitContinuationEpoch, _base_components, _bounded_area,
    _rationalize, _swept_enclosures,
)
from tectonics.geometric_native_implicit_material_epoch import (
    NativeImplicitMaterialEpoch, validate_native_implicit_material_epoch,
)
from tectonics.geometric_native_pose import FractionInterval as I


FORMAT = "native-exhaustive-cold-hot-temporal-pair-inventory-1"


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()


def _hash(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _interval(value):
    return {name: [number.numerator, number.denominator]
        for name, number in (("lo", value.lo), ("hi", value.hi))}


def _source_component(component):
    return dict(component_id=component.component_id,
        material_id=component.material_id,
        original_parent_id=component.original_parent_id,
        owner=component.owner, material_kind=component.material_kind,
        domain_expression=component.domain_expression,
        area_bounds_sr=_interval(component.area_bounds_sr),
        lower_geometry_count=len(component.lower_geometries),
        upper_geometry_count=len(component.upper_geometries))


def _validate_arguments(epoch, records, dt_myr, time_cells, precision, order, provenance):
    if type(epoch) not in (NativeImplicitMaterialEpoch, NativeImplicitContinuationEpoch):
        raise ValueError("Inventory requires a typed complete native material partition")
    epoch = validate_native_implicit_material_epoch(epoch)
    if epoch.phase != "partition":
        raise ValueError("Inventory cannot use unresolved free material transport")
    owners = tuple(owner for owner, _ in epoch.additional_owner_histories)
    if owners != (0, 1):
        raise ValueError("This declared native geometry has donor 1 and receiver 0")
    if (type(records) is not tuple or len(records) != 2
            or any(type(row) is not tuple or len(row) != 2
                or type(row[0]) is not int or type(row[1]) is not tuple
                or len(row[1]) != 3 or any(type(x) is not float or not math.isfinite(x)
                    for x in row[1]) for row in records)
            or tuple(owner for owner, _ in records) != owners):
        raise ValueError("Inventory requires the literal complete current Euler vectors")
    if type(dt_myr) is not float or not math.isfinite(dt_myr) or dt_myr <= 0:
        raise ValueError("Inventory requires its actual positive physical duration")
    if (type(time_cells) is not int or not 1 <= time_cells <= 256
            or type(precision) is not int or not 128 <= precision <= 512
            or type(order) is not int or not 12 <= order <= 40):
        raise ValueError("Inventory needs explicit bounded full-time proof resources")
    if type(provenance) is not str or not provenance.strip():
        raise ValueError("Inventory needs explicit source and candidate provenance")
    return epoch


def build_coverage_inventory(epoch, omega_rad_per_myr, dt_myr, *, time_cells=1,
        precision=128, order=20, provenance, progress=None):
    """Return every canonical material pair, without adopting any mechanics.

    ``progress`` is an optional observation callback.  It receives a completed
    pair record; it cannot alter the proof inputs or the resulting inventory.
    The caller must authenticate ``epoch`` and the genuine mechanical candidate
    independently.  In particular, a literal Euler vector is not solve authority.
    """
    epoch = _validate_arguments(epoch, omega_rad_per_myr, dt_myr,
        time_cells, precision, order, provenance)
    began = time.monotonic()
    exact_components = tuple(sorted(_base_components(epoch), key=lambda row: row.component_id))
    before = {row.component_id: dict(
        lower=_rationalize(row.lower_geometries, False, precision),
        upper=_rationalize(row.upper_geometries, True, precision))
        for row in exact_components}
    donors = tuple(row for row in exact_components if row.owner == 1)
    receivers = tuple(row for row in exact_components if row.owner == 0)
    motions = dict(omega_rad_per_myr)
    fractions = {owner: tuple(Fraction.from_float(x) for x in vector)
        for owner, vector in omega_rad_per_myr}
    pairs = []
    for donor in donors:
        for receiver in receivers:
            if motions[0] == motions[1]:
                # The trusted exact complete partition and one common SO(3)
                # bijection prove pairwise disjointness at every time.  Chart
                # enclosure overlap cannot override that exact identity.
                lower, upper, area = (), (), I(0)
            else:
                lower, upper = _swept_enclosures(before[donor.component_id]["upper"],
                    before[donor.component_id]["lower"],
                    before[receiver.component_id]["lower"],
                    before[receiver.component_id]["upper"],
                    motions[1], motions[0], dt_myr, time_cells, precision, order)
                area = _bounded_area(lower, upper, I(0, donor.area_bounds_sr.hi))
            immediate = tuple(edge
                for dg in donor.lower_geometries for rg in receiver.lower_geometries
                for edge in _immediate_edges(dg, rg, fractions[1], fractions[0], precision))
            status = ("excluded_for_whole_interval" if area.hi == 0 else
                "positive_material_subset" if area.lo > 0 else "unresolved_upper_only")
            row = dict(donor_component_id=donor.component_id,
                receiver_component_id=receiver.component_id,
                donor_original_parent_id=donor.original_parent_id,
                receiver_original_parent_id=receiver.original_parent_id,
                donor_material_kind=donor.material_kind,
                receiver_material_kind=receiver.material_kind,
                involves_hot_material=(donor.material_kind == "hot_birth"
                    or receiver.material_kind == "hot_birth"),
                exact_temporal_domain_expression=(
                    f"prior_live({donor.component_id}@{epoch.epoch_sha256}) intersect "
                    f"union_0<=u<=1 inverse(R_1(u))(R_0(u)("
                    f"prior_live({receiver.component_id}@{epoch.epoch_sha256})))"),
                temporal_material_area_bounds_sr=_interval(area),
                inner_endpoint_polygon_count=len(lower),
                outer_whole_time_polygon_count=len(upper),
                status=status,
                immediate_incoming_edge_count=len(immediate),
                constitutive_time_admission_proved=False,
                force_admission_authorized=False)
            pairs.append(row)
            if progress is not None:
                progress(json.loads(_encoded(row)))
    expected = {(donor.component_id, receiver.component_id)
        for donor in donors for receiver in receivers}
    actual = {(row["donor_component_id"], row["receiver_component_id"]) for row in pairs}
    if len(pairs) != len(expected) or actual != expected:
        raise AssertionError("Inventory lost or duplicated a complete material pair")
    classifications = ("excluded_for_whole_interval", "positive_material_subset", "unresolved_upper_only")
    body = dict(format=FORMAT,
        source_epoch_sha256=epoch.epoch_sha256,
        complete_source_epoch_fingerprint_sha256=_portable_fingerprint_sha256(epoch),
        omega_rad_per_myr=[[owner, list(vector)] for owner, vector in omega_rad_per_myr],
        dt_myr=dt_myr, time_cells=time_cells, precision=precision, order=order,
        provenance=provenance,
        donor_owner=1, receiver_owner=0,
        donor_component_count=len(donors), receiver_component_count=len(receivers),
        material_component_count=len(exact_components),
        exhaustive_canonical_pair_count=len(pairs),
        original_original_pair_count=sum(not row["involves_hot_material"] for row in pairs),
        hot_involving_pair_count=sum(row["involves_hot_material"] for row in pairs),
        classification_counts={name: sum(row["status"] == name for row in pairs)
            for name in classifications},
        components=[_source_component(row) for row in exact_components],
        signed_curved_correction_component_ids=[dict(
            negative_component_id=row.negative_component_id,
            positive_component_id=row.positive_component_id)
            for row in epoch.paired_corrections],
        pairs=pairs,
        exact_implicit_material_masks_authoritative=True,
        native_polygons_are_disposable_measure_enclosures=True,
        upper_only_pairs_are_unresolved=True,
        endpoint_lower_subset_is_not_full_time_lower_union=True,
        all_pairs_constitutively_admitted=False,
        material_area_used_as_inclined_interface_area=False,
        geometric_measure_midpoint_is_material_debit=False,
        force_solve_authority_supplied_by_inventory=False,
        candidate_mechanics_adopted=False,
        force_admission_authorized=False)
    body["inventory_sha256"] = _hash(body)
    # Elapsed time is an execution observation outside the sealed physics body.
    return dict(inventory=body, elapsed_s=time.monotonic()-began)


def write_inventory(path, result):
    """Exclusively save an already produced result; never overwrite evidence."""
    with Path(path).open("xb") as stream:
        stream.write(_encoded(result)+b"\n")
    return path
