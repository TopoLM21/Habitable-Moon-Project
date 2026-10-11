"""World-space integration charts for an exact implicit material partition.

The native polygons in this module are disposable integration charts.  They
are not archive assignments: the curved first-entry domain stays implicit.
One chart contains that domain and its two signed material corrections share
the same exact measure.  Consequently a curved donor loss and the hot fill of
the same vacancy cancel in every owner/cell loading operator, without rounding
away either material reservoir.

An epoch is a material geometry object, never a force acceptance receipt.
Additional owner poses can be recorded, but differential free transport is
marked unresolved until its new archive/vacancy closure has been proved.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
import hashlib
import json

from .geometric_intrinsic_geometry import (
    RotationIncrement, rotation_increment_from_dict, rotation_increment_to_dict,
)
from .geometric_native_first_entry_material import (
    NativeFirstEntryMaterialState, native_first_entry_material_to_dict,
    validate_native_first_entry_material,_typed_fingerprint,
)
from .geometric_native_polygons import (
    NativePlane, NativePolygon, NativeRay, _cross, native_polygon_to_dict,
    native_subtract, validate_native_polygon,
)
from .geometric_native_pose import (
    FractionInterval as I, _rounded_interval, certify_rotation_history,
)


FORMAT = "native-implicit-material-integration-epoch-1"


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _hash(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _interval(value):
    return {name: [number.numerator, number.denominator]
            for name, number in (("lo", value.lo), ("hi", value.hi))}


def _move(geometry, history):
    for step in history:
        geometry = geometry.rotated(step.omega_rad_per_myr, step.dt_myr)
    return geometry


@dataclass(frozen=True, slots=True)
class NativeImplicitMaterialComponent:
    component_id: str
    material_id: str
    original_parent_id: str
    owner: int
    material_kind: str
    domain_expression: str
    proxy_geometries: tuple[NativePolygon, ...]
    area_bounds_sr: I


@dataclass(frozen=True, slots=True)
class NativeImplicitPairedCorrection:
    correction_id: str
    negative_component_id: str
    positive_component_id: str
    owner: int
    measure_reference: str
    area_bounds_sr: I
    containing_chart: NativePolygon


@dataclass(frozen=True, slots=True)
class NativeImplicitMaterialEpoch:
    source_material_state: NativeFirstEntryMaterialState
    additional_owner_histories: tuple
    components: tuple[NativeImplicitMaterialComponent, ...]
    paired_corrections: tuple[NativeImplicitPairedCorrection, ...]
    time_myr: Fraction
    phase: str
    epoch_sha256: str

    @property
    def radius_km(self):
        return self.source_material_state.journal.source_surface.radius_km

    @property
    def original_parent_allocations(self):
        return self.source_material_state.parent_allocations

    @property
    def archive_cohorts(self):
        return self.source_material_state.archive_cohorts

    @property
    def force_admission_authorized(self):
        return False


def _body(epoch):
    return dict(format=FORMAT,
        source_material_state_sha256=epoch.source_material_state.state_sha256,
        additional_owner_histories=[[owner, [rotation_increment_to_dict(step)
            for step in history]] for owner, history in epoch.additional_owner_histories],
        components=[dict(component_id=row.component_id, material_id=row.material_id,
            original_parent_id=row.original_parent_id, owner=row.owner,
            material_kind=row.material_kind, domain_expression=row.domain_expression,
            proxy_geometries=[native_polygon_to_dict(poly) for poly in row.proxy_geometries],
            area_bounds_sr=_interval(row.area_bounds_sr)) for row in epoch.components],
        paired_corrections=[dict(correction_id=row.correction_id,
            negative_component_id=row.negative_component_id,
            positive_component_id=row.positive_component_id, owner=row.owner,
            measure_reference=row.measure_reference, area_bounds_sr=_interval(row.area_bounds_sr),
            containing_chart=native_polygon_to_dict(row.containing_chart),
            chart_is_archive_domain=False, signed_measure_cancels_per_owner_and_cell=True)
            for row in epoch.paired_corrections],
        time_myr=[epoch.time_myr.numerator, epoch.time_myr.denominator], phase=epoch.phase,
        original_parent_count=len(epoch.original_parent_allocations),
        exact_material_partition="every original live/archive complement and every hot vacancy preserved",
        native_polygon_archive_assignment=False, force_admission_authorized=False)


def _checked_histories(source, supplied):
    owners = tuple(sorted({row.owner for row in source.parent_allocations}))
    if type(supplied) is not tuple or tuple(owner for owner, _ in supplied) != owners:
        raise ValueError("Implicit epoch needs all original owners in canonical order")
    for owner, history in supplied:
        if type(owner) is not int or type(history) is not tuple:
            raise ValueError("Implicit epoch histories require literal immutable owner records")
        certify_rotation_history(history, 64)
    if len({len(history) for _, history in supplied}) != 1:
        raise ValueError("Implicit epoch owner clocks must contain the same number of intervals")
    if supplied:
        for index in range(len(supplied[0][1])):
            if len({history[index].dt_myr for _, history in supplied}) != 1:
                raise ValueError("Implicit epoch owner clocks disagree")
    return supplied


def _curved_chart(certificate, donor_motion, additional_history):
    # The whole analytic envelope proof gives x in [0,H], y in
    # [-q_max,-k_terminal], z=1.  Its gnomonic rectangle is a containing
    # chart only; neither its area nor its polygon is assigned to an archive.
    height = _rounded_interval(certificate.terminal_height_bounds, 96).hi
    low_y = -_rounded_interval(certificate.maximum_ratio_bounds, 96).hi
    high_y = -_rounded_interval(certificate.terminal_slope_bounds, 96).lo
    if height <= 0 or high_y <= low_y:
        raise ValueError("Implicit curved correction needs its certified positive containing chart")
    points = ((Fraction(0), low_y, Fraction(1)),
              (height, low_y, Fraction(1)),
              (height, high_y, Fraction(1)),
              (Fraction(0), high_y, Fraction(1)))
    rays = tuple(NativeRay("seed", point) for point in points)
    planes = tuple(NativePlane(_cross(a.reference, b.reference))
                   for a, b in zip(rays, rays[1:]+rays[:1]))
    chart = validate_native_polygon(NativePolygon(rays, planes))
    return _move(chart, (donor_motion,)+additional_history)


def _build(source, histories):
    source = validate_native_first_entry_material(source)
    histories = _checked_histories(source, histories)
    extra_histories = dict(histories)
    journal, certificate = source.journal, source.sweep_certificate
    motions = {owner: RotationIncrement(omega, journal.dt_myr)
               for owner, omega in journal.omega_rad_per_myr}
    domains = {row.fragment_id: row for row in certificate.donor_domains}
    allocations = {row.fragment_id: row for row in source.parent_allocations}
    components = []
    for parent in journal.source_surface.fragments:
        allocation = allocations[parent.fragment_id]
        domain = domains.get(parent.fragment_id)
        proxy = ((parent.geometry,) if domain is None or domain.endpoint_domain is None
                 else native_subtract(parent.geometry, domain.endpoint_domain))
        history = (motions[allocation.owner],)+extra_histories[allocation.owner]
        components.append(NativeImplicitMaterialComponent(allocation.live_domain_id,
            allocation.material_id, parent.fragment_id, allocation.owner, "original_live",
            "original_parent minus once_only_complete_first_entry_archive",
            tuple(_move(poly, history) for poly in proxy),
            I(max(Fraction(0), allocation.parent_area_bounds_sr.lo-allocation.archived_area_bounds_sr.hi),
              allocation.parent_area_bounds_sr.hi-allocation.archived_area_bounds_sr.lo)))
    newborns = {}
    for vacancy in source.endpoint_vacancies:
        newborn = vacancy.newborn
        if newborn is None:
            continue
        newborns[vacancy.fixed_original_parent_id] = newborn
        raw = tuple(row.geometry for row in journal.raw_vacancies
                    if row.fixed_original_parent_id == vacancy.fixed_original_parent_id)
        components.append(NativeImplicitMaterialComponent(newborn.vacancy_domain_id,
            newborn.material_id, newborn.fixed_original_parent_id, newborn.owner, "hot_birth",
            "fixed_original_parent minus union_of_all_moved_live_original_domains",
            tuple(_move(poly, extra_histories[newborn.owner]) for poly in raw),
            vacancy.area_bounds_sr))
    corrections = []
    for domain in certificate.donor_domains:
        if domain.extra_swept_area_bounds_sr.hi == 0:
            continue
        parent = allocations[domain.fragment_id]
        newborn = newborns[certificate.extra_endpoint_fixed_parent_id]
        if parent.owner != newborn.owner:
            raise ValueError("Implicit paired correction cannot claim cancellation between different owners")
        chart = _curved_chart(certificate, motions[parent.owner], extra_histories[parent.owner])
        corrections.append(NativeImplicitPairedCorrection(
            "curved-correction:"+_hash(dict(state=source.state_sha256, parent=domain.fragment_id)),
            parent.live_domain_id, newborn.vacancy_domain_id, parent.owner,
            "native_area("+parent.archive_domain_id+" minus endpoint_archive)",
            domain.extra_swept_area_bounds_sr, chart))
    elapsed = sum((Fraction.from_float(step.dt_myr) for step in histories[0][1]), Fraction(0))
    time = (Fraction.from_float(journal.source_surface.time_myr)
            +Fraction.from_float(journal.dt_myr)+elapsed)
    phase = "partition" if all(history == histories[0][1] for _, history in histories) else "unresolved_free_transport"
    epoch = NativeImplicitMaterialEpoch(source, histories, tuple(components),
        tuple(corrections), time, phase, "")
    return replace(epoch, epoch_sha256=_hash(_body(epoch)))


def initialize_native_implicit_material_epoch(source_material_state):
    """Construct the exact material endpoint's independently usable load chart."""
    source = validate_native_first_entry_material(source_material_state)
    histories = tuple((owner, ()) for owner in sorted({row.owner for row in source.parent_allocations}))
    return _build(source, histories)


def transport_native_implicit_material_epoch(epoch, omega_rad_per_myr, dt_myr):
    """Record a complete free SO(3) trial; differential trials stay unresolved."""
    epoch = validate_native_implicit_material_epoch(epoch)
    if type(epoch) is not NativeImplicitMaterialEpoch:
        raise ValueError("A continuation trial must preserve every new archive and birth; use fresh material continuation closure")
    if type(omega_rad_per_myr) is not dict or set(omega_rad_per_myr) != {owner for owner, _ in epoch.additional_owner_histories}:
        raise ValueError("Implicit transport needs every original material owner")
    histories = tuple((owner, history+(RotationIncrement(omega_rad_per_myr[owner], dt_myr),))
                      for owner, history in epoch.additional_owner_histories)
    return _build(epoch.source_material_state, histories)


def validate_native_implicit_material_epoch(epoch):
    from .geometric_native_implicit_material_continuation import (
        NativeImplicitContinuationEpoch, validate_native_implicit_material_continuation,
    )
    if type(epoch) is NativeImplicitContinuationEpoch:
        return validate_native_implicit_material_continuation(epoch)
    if type(epoch) is not NativeImplicitMaterialEpoch:
        raise ValueError("Expected a complete typed implicit material epoch")
    replay = _build(epoch.source_material_state, epoch.additional_owner_histories)
    if (_typed_fingerprint(epoch)!=_typed_fingerprint(replay) or
        _encoded(_body(epoch)) != _encoded(_body(replay)) or epoch.epoch_sha256 != replay.epoch_sha256):
        raise ValueError("Implicit material epoch differs from its whole typed source replay")
    return replay


def native_implicit_material_epoch_to_dict(epoch, *, include_source=True):
    epoch = validate_native_implicit_material_epoch(epoch)
    from .geometric_native_implicit_material_continuation import (
        NativeImplicitContinuationEpoch, native_implicit_material_continuation_to_dict,
    )
    if type(epoch) is NativeImplicitContinuationEpoch:
        return native_implicit_material_continuation_to_dict(epoch,include_source=include_source)
    raw = dict(_body(epoch), epoch_sha256=epoch.epoch_sha256)
    if include_source:
        raw["source_material_state"] = native_first_entry_material_to_dict(epoch.source_material_state)
    return raw


def native_implicit_material_epoch_from_dict(raw, *, expected_source_material_state):
    """Rebuild every chart against an independently supplied typed source."""
    source = validate_native_first_entry_material(expected_source_material_state)
    from .geometric_native_implicit_material_continuation import (
        FORMAT as CONTINUATION_FORMAT, native_implicit_material_continuation_from_dict,
    )
    if type(raw) is dict and raw.get("format")==CONTINUATION_FORMAT:
        return native_implicit_material_continuation_from_dict(raw,expected_source_material_state=source)
    if type(raw) is not dict or raw.get("format") != FORMAT:
        raise ValueError("Invalid implicit material epoch schema")
    try:
        records = raw["additional_owner_histories"]
        if type(records) is not list or any(type(row) is not list or len(row) != 2
                or type(row[0]) is not int or type(row[1]) is not list for row in records):
            raise ValueError("Implicit epoch histories need canonical JSON owner records")
        histories = tuple((owner, tuple(rotation_increment_from_dict(step) for step in steps))
                          for owner, steps in records)
        replay = _build(source, histories)
        expected = native_implicit_material_epoch_to_dict(replay,
            include_source="source_material_state" in raw)
        if _encoded(raw) != _encoded(expected):
            raise ValueError("Saved implicit epoch differs from complete original source and SO(3) replay")
        return replay
    except (KeyError, TypeError) as error:
        raise ValueError("Invalid complete implicit epoch payload") from error
