"""Certified cell loading of a live implicit material mask.

Only material integration weights change.  The independent mantle field,
fixed global physical mesh centroids, lever arm, and positive drag law retain
their established meaning.  Per-material/cell areas are rational enclosures;
they never become archive debits.  Correlated curved removal/hot-fill weights
cancel exactly before assembling owner torque and drag.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from functools import lru_cache
import math

from .geometric_basal_loading import mesh_fingerprint
from .geometric_native_first_entry_material import _typed_fingerprint
from .geometric_native_implicit_material_epoch import (
    NativeImplicitMaterialEpoch, _encoded, _hash, _interval,
    validate_native_implicit_material_epoch,
)
from .geometric_native_mantle_loading import _canonical_source, _mesh_cells, _traction
from .geometric_native_polygons import _side, native_intersection
from .geometric_native_pose import FractionInterval as I, _rounded_interval, interval_sqrt
from .geometric_spatial_mantle_loading import (
    MODEL, SOURCE_FRAME, PrescribedSpatialMantleSource,
)


FORMAT = "native-implicit-physical-spatial-mantle-loading-1"
QUADRATURE = "exact_implicit_mask_native_mesh_intersections_physical_cell_centroids_v1"
_ASSEMBLY_CACHE={}


@dataclass(frozen=True, slots=True)
class NativeImplicitMaterialCellMeasure:
    cell: int
    component_id: str
    material_id: str
    original_parent_id: str
    owner: int
    material_kind: str
    proxy_area_bounds_sr: I
    signed_correction_bounds_sr: I
    area_bounds_sr: I
    area_bounds_m2: I
    correlation_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NativeImplicitCorrectionCellMeasure:
    cell: int
    correction_id: str
    area_bounds_sr: I
    negative_component_id: str
    positive_component_id: str
    owner: int


@dataclass(frozen=True, slots=True)
class NativeImplicitPhysicalSpatialMantleLoad:
    epoch_sha256: str
    mesh_sha256: str
    plate_ids: tuple[int, ...]
    source: PrescribedSpatialMantleSource
    material_cell_measures: tuple[NativeImplicitMaterialCellMeasure, ...]
    correction_cell_measures: tuple[NativeImplicitCorrectionCellMeasure, ...]
    owner_cell_area_bounds_m2: tuple
    drag_nm_s: tuple[tuple[float, ...], ...]
    torque_nm: tuple[float, ...]
    drag_bounds_nm_s: tuple
    torque_bounds_nm: tuple
    load_sha256: str

    @property
    def surface_sha256(self):
        return self.epoch_sha256

    @property
    def provenance(self):
        return self.source.provenance


def _body(load):
    return dict(format=FORMAT, model=MODEL, quadrature=QUADRATURE,
        source_frame=SOURCE_FRAME, epoch_sha256=load.epoch_sha256,
        mesh_sha256=load.mesh_sha256, plate_ids=list(load.plate_ids), source=asdict(load.source),
        material_cell_measures=[dict(cell=row.cell, component_id=row.component_id,
            material_id=row.material_id, original_parent_id=row.original_parent_id,
            owner=row.owner, material_kind=row.material_kind,
            proxy_area_bounds_sr=_interval(row.proxy_area_bounds_sr),
            signed_correction_bounds_sr=_interval(row.signed_correction_bounds_sr),
            area_bounds_sr=_interval(row.area_bounds_sr), area_bounds_m2=_interval(row.area_bounds_m2),
            correlation_ids=list(row.correlation_ids)) for row in load.material_cell_measures],
        correction_cell_measures=[dict(cell=row.cell, correction_id=row.correction_id,
            area_bounds_sr=_interval(row.area_bounds_sr),
            negative_component_id=row.negative_component_id,
            positive_component_id=row.positive_component_id, owner=row.owner)
            for row in load.correction_cell_measures],
        owner_cell_area_bounds_m2=[dict(cell=cell, owner=owner, area_bounds_m2=_interval(area))
            for cell, owner, area in load.owner_cell_area_bounds_m2],
        drag_nm_s=[list(row) for row in load.drag_nm_s], torque_nm=list(load.torque_nm),
        drag_bounds_nm_s=[[_interval(value) for value in row] for row in load.drag_bounds_nm_s],
        torque_bounds_nm=[_interval(value) for value in load.torque_bounds_nm],
        curved_weights_cancel_per_owner_cell_before_operator_assembly=True,
        geometric_measure_midpoint_is_material_debit=False,
        archived_material_receives_basal_loading=False)


@lru_cache(maxsize=128)
def _measure_intersections(geometries, cells):
    amounts = [I(0) for _ in cells]
    for geometry in geometries:
        for cell, footprint in enumerate(cells):
            overlap = native_intersection(geometry, footprint)
            if overlap is not None:
                amounts[cell] += overlap.area_bounds()
    return tuple(_rounded_interval(value, 96) for value in amounts)


@lru_cache(maxsize=64)
def _correction_cells(correction, cells):
    for cell, geometry in enumerate(cells):
        if all(_side(ray.rotated(correction.containing_chart.pose), plane) >= 0
               for plane in geometry.support_planes for ray in correction.containing_chart.vertices):
            # The already audited native mesh is a disjoint sphere partition.
            # A complete containing chart in one cell localizes its true
            # curved subset there without measuring or clipping that subset.
            return tuple(correction.area_bounds_sr if index == cell else I(0)
                         for index in range(len(cells)))
    supports = _measure_intersections((correction.containing_chart,), cells)
    upper = tuple(min(correction.area_bounds_sr.hi, support.hi) for support in supports)
    if sum(upper, Fraction(0)) < correction.area_bounds_sr.lo:
        raise ValueError("Native mesh does not cover the complete implicit curved correction")
    result = []
    for cell, high in enumerate(upper):
        if not high:
            result.append(I(0))
            continue
        low = max(Fraction(0), correction.area_bounds_sr.lo
                  -sum((other for index, other in enumerate(upper) if index != cell), Fraction(0)))
        result.append(_rounded_interval(I(low, high), 96))
    return tuple(result)


def _operators(owners, owner_areas, source, radius, positions):
    index = {owner: slot for slot, owner in enumerate(owners)}
    size = 3*len(owners)
    drag = [[I(0) for _ in range(size)] for _ in range(size)]
    torque = [I(0) for _ in range(size)]
    radius_exact = Fraction.from_float(radius)
    factor = Fraction.from_float(source.basal_drag_pa_s_m)*radius_exact*radius_exact
    for cell, owner, area in owner_areas:
        x = tuple(Fraction.from_float(value) for value in positions[cell])
        norm_squared = sum((value*value for value in x), Fraction(0))
        traction = _traction(source, radius, positions[cell])
        exact_traction = tuple(Fraction.from_float(value) for value in traction)
        cross = (x[1]*exact_traction[2]-x[2]*exact_traction[1],
                 x[2]*exact_traction[0]-x[0]*exact_traction[2],
                 x[0]*exact_traction[1]-x[1]*exact_traction[0])
        start = 3*index[owner]
        for i in range(3):
            torque[start+i] += radius_exact*cross[i]*area
            for j in range(3):
                coefficient = factor*((norm_squared if i == j else 0)-x[i]*x[j])
                drag[start+i][start+j] += coefficient*area
    # Evaluate the complete fixed-cell rule using exact rational coefficients.
    # A longdouble accumulation can introduce O(100 N m) transverse residues
    # by cancellation of O(1e22 N m) source terms.  Such residues can dominate
    # the actual tiny full-axis rotation and are not physical source inputs.
    # A displayed operator midpoint is only a quadrature representative; every
    # actual geometric operator remains enclosed and no reservoir is debited.
    numerical_drag = tuple(tuple(float(value.midpoint) for value in row) for row in drag)
    numerical_torque = tuple(float(value.midpoint) for value in torque)
    if any(not math.isfinite(value) for row in numerical_drag for value in row) or any(
            not math.isfinite(value) for value in numerical_torque):
        raise ValueError("Implicit physical source operators are not representable")
    # This enclosure includes only the recorded physical-centroid rule and its
    # arithmetic conversion.  It does not claim a continuum spatial quadrature
    # bound that the inherited physical source model does not supply.
    def include_display(enclosure, value):
        exact = Fraction.from_float(value)
        return _rounded_interval(I(min(enclosure.lo, exact), max(enclosure.hi, exact)), 96)
    drag_bounds = tuple(tuple(include_display(value, numerical_drag[i][j])
                              for j, value in enumerate(row)) for i, row in enumerate(drag))
    torque_bounds = tuple(include_display(value, numerical_torque[i])
                          for i, value in enumerate(torque))
    return numerical_drag, numerical_torque, drag_bounds, torque_bounds


def assemble_native_implicit_physical_spatial_mantle_load(mesh, epoch, source):
    epoch = validate_native_implicit_material_epoch(epoch)
    if epoch.phase != "partition":
        raise ValueError("Implicit mantle loading needs the new archive/vacancy closure of differential free transport")
    source = _canonical_source(source)
    mesh_sha=mesh_fingerprint(mesh)
    key=(epoch.epoch_sha256,mesh_sha,_encoded(asdict(source)))
    if key in _ASSEMBLY_CACHE:
        cached,body_bytes,seal,fingerprint=_ASSEMBLY_CACHE[key]
        if (_typed_fingerprint(cached)==fingerprint and _encoded(_body(cached))==body_bytes
            and cached.load_sha256==seal):
            return cached
        del _ASSEMBLY_CACHE[key]
    cells = _mesh_cells(mesh)
    radius = epoch.radius_km*1000.
    radius_squared = Fraction.from_float(radius)**2
    positions = tuple(tuple(map(float, row)) for row in mesh.centroids)
    from .geometric_native_implicit_material_continuation import NativeImplicitContinuationEpoch
    if type(epoch) is NativeImplicitContinuationEpoch:
        result=_assemble_bounded_epoch_load(mesh,epoch,source,cells,radius,radius_squared,positions)
        if len(_ASSEMBLY_CACHE)>=16:
            _ASSEMBLY_CACHE.clear()
        _ASSEMBLY_CACHE[key]=(result,_encoded(_body(result)),result.load_sha256,_typed_fingerprint(result))
        return result
    proxy = {row.component_id: _measure_intersections(row.proxy_geometries, cells)
             for row in epoch.components}
    signed = {row.component_id: [I(0) for _ in cells] for row in epoch.components}
    correlations = {row.component_id: [[] for _ in cells] for row in epoch.components}
    correction_measures = []
    for correction in epoch.paired_corrections:
        for cell, area in enumerate(_correction_cells(correction, cells)):
            if not area.hi:
                continue
            signed[correction.negative_component_id][cell] -= area
            signed[correction.positive_component_id][cell] += area
            for identity in (correction.negative_component_id, correction.positive_component_id):
                correlations[identity][cell].append(correction.correction_id)
            correction_measures.append(NativeImplicitCorrectionCellMeasure(cell,
                correction.correction_id, area, correction.negative_component_id,
                correction.positive_component_id, correction.owner))
    measures = []
    owner_values = {}
    for component in epoch.components:
        for cell, base in enumerate(proxy[component.component_id]):
            adjustment = signed[component.component_id][cell]
            area = base+adjustment
            area = I(max(Fraction(0), area.lo), max(Fraction(0), area.hi))
            if area.hi:
                measures.append(NativeImplicitMaterialCellMeasure(cell,
                    component.component_id, component.material_id, component.original_parent_id,
                    component.owner, component.material_kind, base, adjustment,
                    _rounded_interval(area, 96), _rounded_interval(area*radius_squared, 96),
                    tuple(correlations[component.component_id][cell])))
            # The signed correction belongs to two actual material components
            # with identical owner and exact geometric domain.  Cancel it as
            # an identity here, not by adding independent interval enclosures.
            key = (cell, component.owner)
            owner_values[key] = owner_values.get(key, I(0))+base
    owners = tuple(sorted({row.owner for row in epoch.components}))
    owner_areas = tuple((cell, owner, _rounded_interval(area*radius_squared, 96))
        for (cell, owner), area in sorted(owner_values.items()) if area.hi > 0)
    for cell, geometry in enumerate(cells):
        actual = sum((area for index, _, area in owner_areas if index == cell), I(0))
        expected = geometry.area_bounds()*radius_squared
        if actual.hi < expected.lo or actual.lo > expected.hi:
            raise ValueError("Implicit owner loading does not conservatively cover its complete native physical cell")
    drag, torque, drag_bounds, torque_bounds = _operators(owners, owner_areas, source, radius, positions)
    load = NativeImplicitPhysicalSpatialMantleLoad(epoch.epoch_sha256, mesh_fingerprint(mesh),
        owners, source, tuple(sorted(measures, key=lambda row:(row.cell, row.component_id))),
        tuple(sorted(correction_measures, key=lambda row:(row.cell, row.correction_id))),
        owner_areas, drag, torque, drag_bounds, torque_bounds, "")
    result=replace(load,load_sha256=_hash(_body(load)))
    if len(_ASSEMBLY_CACHE)>=16:
        _ASSEMBLY_CACHE.clear()
    _ASSEMBLY_CACHE[key]=(result,_encoded(_body(result)),result.load_sha256,_typed_fingerprint(result))
    return result


def _assemble_bounded_epoch_load(mesh,epoch,source,cells,radius,radius_squared,positions):
    measures=[]
    owner_values={}
    for component in epoch.components:
        lower=_measure_intersections(component.lower_geometries,cells)
        upper=_measure_intersections(component.upper_geometries,cells)
        for cell,(low,high) in enumerate(zip(lower,upper)):
            # The exact total component measure additionally constrains every
            # local cell.  This preserves positive tiny inherited reservoirs
            # when a containing chart has been split into several cells.
            lo=max(low.lo,component.area_bounds_sr.lo-sum((value.hi for index,value in enumerate(upper)
                if index!=cell),Fraction(0)),Fraction(0))
            hi=min(high.hi,component.area_bounds_sr.hi-sum((value.lo for index,value in enumerate(lower)
                if index!=cell),Fraction(0)))
            if lo>hi:
                raise ValueError("Fresh implicit material/cell bounds contradict their exact total mask measure")
            area=_rounded_interval(I(lo,max(Fraction(0),hi)),96)
            if area.hi==0:
                continue
            physical=_rounded_interval(area*radius_squared,96)
            measures.append(NativeImplicitMaterialCellMeasure(cell,component.component_id,component.material_id,
                component.original_parent_id,component.owner,component.material_kind,area,I(0),area,physical,()))
            key=(cell,component.owner)
            owner_values[key]=owner_values.get(key,I(0))+physical
    owners=tuple(sorted({row.owner for row in epoch.components}))
    owner_areas=[]
    for cell,geometry in enumerate(cells):
        cell_area=geometry.area_bounds()*radius_squared
        total=sum((owner_values.get((cell,owner),I(0)) for owner in owners),I(0))
        if total.hi<cell_area.lo or total.lo>cell_area.hi:
            raise ValueError("Fresh exact implicit masks do not conservatively fill the physical native source cell")
        for owner in owners:
            value=owner_values.get((cell,owner),I(0))
            other=sum((owner_values.get((cell,identity),I(0)) for identity in owners if identity!=owner),I(0))
            lo=max(Fraction(0),value.lo,cell_area.lo-other.hi)
            hi=min(value.hi,cell_area.hi-other.lo)
            if lo>hi:
                raise ValueError("Fresh owner masks contradict the complete-cell complement identity")
            if hi>0:
                owner_areas.append((cell,owner,_rounded_interval(I(lo,hi),96)))
    owner_areas=tuple(owner_areas)
    drag,torque,drag_bounds,torque_bounds=_operators(owners,owner_areas,source,radius,positions)
    load=NativeImplicitPhysicalSpatialMantleLoad(epoch.epoch_sha256,mesh_fingerprint(mesh),owners,source,
        tuple(sorted(measures,key=lambda row:(row.cell,row.component_id))),(),owner_areas,
        drag,torque,drag_bounds,torque_bounds,"")
    return replace(load,load_sha256=_hash(_body(load)))


def validate_native_implicit_physical_spatial_mantle_load(mesh, epoch, load, *, source=None):
    if type(load) is not NativeImplicitPhysicalSpatialMantleLoad:
        raise ValueError("Expected the complete typed implicit spatial mantle load")
    if source is not None and source != load.source:
        raise ValueError("Independently prescribed source changed")
    replay = assemble_native_implicit_physical_spatial_mantle_load(mesh, epoch, load.source)
    if (_typed_fingerprint(replay)!=_typed_fingerprint(load) or
        _encoded(_body(replay)) != _encoded(_body(load)) or replay.load_sha256 != load.load_sha256):
        raise ValueError("Implicit mantle load differs from complete native mask/mesh/source replay")
    return replay


def native_implicit_spatial_mantle_load_to_dict(mesh, epoch, load):
    load = validate_native_implicit_physical_spatial_mantle_load(mesh, epoch, load)
    return dict(_body(load), load_sha256=load.load_sha256)


def native_implicit_spatial_mantle_load_from_dict(raw, mesh, epoch, *, expected_source):
    """Reassemble every saved material/cell weight and operator independently."""
    if type(raw) is not dict or raw.get("format") != FORMAT:
        raise ValueError("Invalid implicit spatial mantle load schema")
    replay = assemble_native_implicit_physical_spatial_mantle_load(mesh, epoch, expected_source)
    expected = dict(_body(replay), load_sha256=replay.load_sha256)
    if _encoded(raw) != _encoded(expected):
        raise ValueError("Saved implicit mantle load differs from full trusted epoch/mesh/source replay")
    return replay


def native_implicit_load_force_error_bounds(load, omega_rad_per_myr):
    """Bound the complete solved force residual caused by mask quadrature.

    The numerical operators are representatives inside the proved mask/cell
    enclosures.  This function checks their effect at the newly solved Euler
    vector; its component bounds include every uncertain torque and drag
    entry.  It does not debit any material or authorize force acceptance.
    """
    from .thermal import SECONDS_PER_MYR
    if type(load) is not NativeImplicitPhysicalSpatialMantleLoad or type(omega_rad_per_myr) is not dict:
        raise ValueError("Implicit source-error check needs the complete load and newly solved owner map")
    if set(omega_rad_per_myr)!=set(load.plate_ids):
        raise ValueError("Implicit source-error check omitted a complete actual owner")
    omega=[]
    for owner in load.plate_ids:
        values=omega_rad_per_myr[owner]
        if type(values) is not tuple or len(values)!=3 or any(type(value) is not float or not math.isfinite(value) for value in values):
            raise ValueError("Implicit source-error check needs all three exact saved float Euler components")
        omega.extend(Fraction.from_float(value)/Fraction.from_float(float(SECONDS_PER_MYR)) for value in values)
    residual=[]
    displayed_drag=[]
    for i,(actual_torque,displayed_torque) in enumerate(zip(load.torque_bounds_nm,load.torque_nm)):
        error=actual_torque-Fraction.from_float(displayed_torque)
        drag_value=Fraction(0)
        for j,angular in enumerate(omega):
            error-=(load.drag_bounds_nm_s[i][j]-Fraction.from_float(load.drag_nm_s[i][j]))*angular
            drag_value+=Fraction.from_float(load.drag_nm_s[i][j])*angular
        residual.append(_rounded_interval(error,96))
        displayed_drag.append(drag_value)
    squared=sum((max(abs(value.lo),abs(value.hi))**2 for value in residual),Fraction(0))
    absolute=interval_sqrt(I(squared),96)
    source_squared=sum((Fraction.from_float(value)**2 for value in load.torque_nm),Fraction(0))
    drag_squared=sum((value**2 for value in displayed_drag),Fraction(0))
    scale=interval_sqrt(I(max(source_squared,drag_squared)),96)
    if scale.lo<=0:
        raise ValueError("Implicit source-error check requires a resolved positive physical force scale")
    return dict(component_residual_bounds_nm=tuple(residual),absolute_residual_norm_upper_nm=absolute.hi,
        relative_residual_norm_upper=absolute.hi/scale.lo,
        source_power_error_bounds_w=_rounded_interval(sum((value*angular for value,angular in zip(residual,omega)),I(0)),96),
        material_midpoint_debit=False,load_sha256=load.load_sha256)
