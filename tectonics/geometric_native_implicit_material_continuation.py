"""Fresh whole-interval material closure on an implicit accepted epoch.

Exact masks are the recursively defined domain expressions.  Native inner and
outer polygon unions are disposable measure enclosures of those masks.  The
outer temporal union uses full SO(3) relative-normal Taylor bounds, including
the preceding ordered pose; it never interpolates endpoint coordinates or
reuses an earlier first-entry chronology.  All original and newborn material
parents retain a once-only archive/live complement and explicit hot births.

This is a material law.  It supplies no physical admission for a newborn
contact and cannot authorize a force interval by itself.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from functools import lru_cache
from math import factorial

from .fractional_surface import EXTENSIVE_FIELDS
from .geometric_native_first_entry_material import _typed_fingerprint
from .geometric_intrinsic_geometry import RotationIncrement, rotation_increment_to_dict
from .geometric_native_donor_sweep import _normal_coefficients
from .geometric_native_implicit_material_epoch import (
    NativeImplicitMaterialEpoch, _encoded, _hash, _interval, _move,
    validate_native_implicit_material_epoch,
)
from .geometric_native_polygons import (
    NativePlane, UnresolvedNativeGeometry, _plane_values, native_clip, native_intersection,
    native_polygon_to_dict, native_subtract,
)
from .geometric_native_pose import FractionInterval as I, _rounded_interval
from .geometric_native_surface import _sphere_octants


FORMAT = "native-fresh-implicit-material-continuation-1"
_BUILD_CACHE={}


def _sum_area(parts):
    return _rounded_interval(sum((poly.area_bounds() for poly in parts), I(0)), 96)


def _intersection(left,right):
    if left==right:
        return left
    try:
        return native_intersection(left,right)
    except UnresolvedNativeGeometry:
        # Intersection is commutative; choosing the other inherited ray chart
        # can authenticate an incidence the first symbolic chart did not name.
        return native_intersection(right,left)


def _subtract(parts, cutters):
    for cutter in cutters:
        remaining = []
        for part in parts:
            if _intersection(part, cutter) is None:
                remaining.append(part)
            else:
                remaining.extend(native_subtract(part, cutter))
        parts = tuple(remaining)
        if not parts:
            break
    return parts


def _union(parts):
    result = ()
    for part in parts:
        result += _subtract((part,), result)
    return result


def _intersect(left, right):
    return tuple(overlap for a in left for b in right
                 if (overlap := _intersection(a, b)) is not None)


def _bounded_area(lower, upper, known=None):
    low, high = _sum_area(lower).lo, _sum_area(upper).hi
    if known is not None:
        low, high = max(low, known.lo), min(high, known.hi)
    if low > high:
        raise ValueError("Fresh implicit mask enclosures contradict their exact material measure")
    return _rounded_interval(I(max(Fraction(0), low), max(Fraction(0), high)), 96)


@lru_cache(maxsize=256)
def _rational_charts(poly, outward, precision):
    """Disposable rational world charts with a complete true-pose proof.

    On each exact axis octant a lower normal corner gives an inner halfspace
    and an upper normal corner gives an outer halfspace.  All downstream
    chart predicates then have exact rational incidence, while the saved
    analytic material mask and its full SO(3) provenance stay authoritative.
    """
    if (not poly.pose and all(not plane.rotations for plane in poly.supports)
        and all(not ray.rotations and all(not plane.rotations for plane in ray.planes) for ray in poly.vertices)):
        return (poly,)
    planes=tuple(_plane_values(plane,precision)[1] for plane in poly.support_planes)
    pieces=[]
    for octant in _sphere_octants():
        signs=tuple(next(plane.normal[axis] for plane in octant.supports if plane.normal[axis]) for axis in range(3))
        clipped=octant
        for bounds in planes:
            corner=tuple((value.hi if sign>0 else value.lo) if outward
                         else (value.lo if sign>0 else value.hi) for sign,value in zip(signs,bounds))
            clipped=native_clip(clipped,NativePlane(corner))
            if clipped is None:
                break
        if clipped is not None:
            pieces.append(clipped)
    return tuple(pieces)


def _rationalize(parts,outward,precision):
    return tuple(chart for poly in parts for chart in _rational_charts(poly,outward,precision))


@dataclass(frozen=True, slots=True)
class NativeImplicitBoundedMaterialComponent:
    component_id: str
    material_id: str
    original_parent_id: str
    owner: int
    material_kind: str
    domain_expression: str
    lower_geometries: tuple
    upper_geometries: tuple
    area_bounds_sr: I
    parent_area_bounds_sr: I
    archived_area_bounds_sr: I
    birth_time_myr: Fraction
    age_myr: Fraction
    material_fields: tuple
    specific_properties: tuple
    parent_amounts: tuple

    @property
    def proxy_geometries(self):
        # Kept for geometry inspection only.  Loads must use both enclosures.
        return self.lower_geometries


@dataclass(frozen=True, slots=True)
class NativeImplicitContinuationArchive:
    cohort_id: str
    component_id: str
    material_id: str
    owner: int
    receiver_owner: int
    loss_domain_expression: str
    newly_archived_area_bounds_sr: I
    total_archived_area_bounds_sr: I
    parent_area_bounds_sr: I
    reservoirs: tuple
    positive_measure_proved: bool
    newborn_physical_pair_admission_authorized: bool = False


@dataclass(frozen=True, slots=True)
class NativeImplicitMaterialReservoir:
    reservoir: str
    parent_amount_bounds: I
    archive_amount_bounds: I
    live_amount_bounds: I
    archive_expression: str
    live_expression: str
    conservation_identity: str = "live_amount + once_only_archive_amount = parent_amount"


@dataclass(frozen=True, slots=True)
class NativeImplicitContinuationEpoch:
    previous_epoch: object
    omega_rad_per_myr: tuple
    dt_myr: float
    time_cells: int
    precision: int
    order: int
    provenance: str
    components: tuple
    new_archives: tuple
    time_myr: Fraction
    epoch_sha256: str
    phase: str = "partition"

    @property
    def radius_km(self):
        return self.previous_epoch.radius_km

    @property
    def source_material_state(self):
        return self.previous_epoch.source_material_state

    @property
    def original_parent_allocations(self):
        return self.source_material_state.parent_allocations

    @property
    def current_original_parent_reservoirs(self):
        rows=[]
        for component in self.components:
            if component.material_kind!="original_live":
                continue
            ratio=component.archived_area_bounds_sr/component.parent_area_bounds_sr
            ratio=I(max(Fraction(0),ratio.lo),min(Fraction(1),ratio.hi))
            reference=f"native_area(total_once_only_archive({component.component_id}@{self.epoch_sha256})) / native_area(original_parent({component.original_parent_id}))"
            reservoirs=tuple(NativeImplicitMaterialReservoir(field,parent,parent*ratio,parent*(I(1)-ratio),
                f"original_parent_amount * {reference}","original_parent_amount minus total_once_only_archive_amount")
                for field,parent in component.parent_amounts)
            rows.append((component.original_parent_id,reservoirs))
        return tuple(rows)

    @property
    def current_material_parent_reservoirs(self):
        output=[]
        radius_squared=Fraction.from_float(self.radius_km)**2
        for component in self.components:
            if component.material_kind=="original_live":
                reservoirs=dict(self.current_original_parent_reservoirs)[component.original_parent_id]
            else:
                factors={"area_km2":radius_squared,"oceanic_volume_km3":radius_squared*component.specific_properties[0],
                    "cold_mantle_volume_km3":Fraction(0),"density_excess_mass_kg":Fraction(0)}
                reservoirs=tuple(NativeImplicitMaterialReservoir(field,parent,
                    component.archived_area_bounds_sr*factors[field],component.area_bounds_sr*factors[field],
                    f"native_area(total_once_only_archive({component.component_id})) * physical_density({field})",
                    f"native_area(current_live({component.component_id})) * physical_density({field})")
                    for field,parent in component.parent_amounts)
            output.append((component.component_id,component.material_id,component.owner,component.material_kind,reservoirs))
        return tuple(output)

    @property
    def paired_corrections(self):
        return ()

    @property
    def archive_cohorts(self):
        return self.previous_epoch.archive_cohorts+self.new_archives

    @property
    def additional_owner_histories(self):
        supplied = dict(self.omega_rad_per_myr)
        return tuple((owner, history+(RotationIncrement(supplied[owner], self.dt_myr),))
                     for owner, history in self.previous_epoch.additional_owner_histories)

    @property
    def force_admission_authorized(self):
        return False


def _body(epoch):
    return dict(format=FORMAT, previous_epoch_sha256=epoch.previous_epoch.epoch_sha256,
        source_material_state_sha256=epoch.source_material_state.state_sha256,
        omega_rad_per_myr=[[owner, list(omega)] for owner, omega in epoch.omega_rad_per_myr],
        dt_myr=epoch.dt_myr, time_cells=epoch.time_cells, precision=epoch.precision,
        order=epoch.order, provenance=epoch.provenance,
        time_myr=[epoch.time_myr.numerator, epoch.time_myr.denominator], phase=epoch.phase,
        components=[dict(component_id=row.component_id, material_id=row.material_id,
            original_parent_id=row.original_parent_id, owner=row.owner, material_kind=row.material_kind,
            domain_expression=row.domain_expression,
            lower_geometries=[native_polygon_to_dict(poly) for poly in row.lower_geometries],
            upper_geometries=[native_polygon_to_dict(poly) for poly in row.upper_geometries],
            area_bounds_sr=_interval(row.area_bounds_sr), parent_area_bounds_sr=_interval(row.parent_area_bounds_sr),
            archived_area_bounds_sr=_interval(row.archived_area_bounds_sr),
            birth_time_myr=[row.birth_time_myr.numerator,row.birth_time_myr.denominator],
            age_myr=[row.age_myr.numerator,row.age_myr.denominator],
            material_fields=[list(value) for value in row.material_fields],
            specific_properties=[[value.numerator,value.denominator] for value in row.specific_properties],
            parent_amounts=[dict(reservoir=name,amount_bounds=_interval(amount)) for name,amount in row.parent_amounts])
            for row in epoch.components],
        new_archives=[dict(cohort_id=row.cohort_id, component_id=row.component_id,
            material_id=row.material_id, owner=row.owner, receiver_owner=row.receiver_owner,
            loss_domain_expression=row.loss_domain_expression,
            newly_archived_area_bounds_sr=_interval(row.newly_archived_area_bounds_sr),
            total_archived_area_bounds_sr=_interval(row.total_archived_area_bounds_sr),
            parent_area_bounds_sr=_interval(row.parent_area_bounds_sr),
            reservoirs=[dict(reservoir=value.reservoir,parent_amount_bounds=_interval(value.parent_amount_bounds),
                archive_amount_bounds=_interval(value.archive_amount_bounds),live_amount_bounds=_interval(value.live_amount_bounds),
                archive_expression=value.archive_expression,live_expression=value.live_expression,
                conservation_identity=value.conservation_identity) for value in row.reservoirs],
            positive_measure_proved=row.positive_measure_proved,
            newborn_physical_pair_admission_authorized=False) for row in epoch.new_archives],
        exact_domain_law="new_archive = prior_live intersect union_{0<=u<=1} inverse(R_donor(u))(R_receiver(u)(prior_receiver_live)); new_live = R_owner(1)(prior_live minus new_archive)",
        exact_birth_law="each fixed prior live material footprint intersect complement of ALL moved retained material",
        native_polygons_are_disposable_inner_outer_measure_enclosures=True,
        chronological_archive_reuse=False, material_midpoint_debit=False,
        force_admission_authorized=False)


def _base_components(epoch):
    if type(epoch) is NativeImplicitContinuationEpoch:
        return epoch.components
    source = epoch.source_material_state
    allocations = {row.fragment_id:row for row in source.parent_allocations}
    parents = {row.fragment_id:row for row in source.journal.source_surface.fragments}
    newborns = {row.newborn.material_id:row.newborn for row in source.endpoint_vacancies if row.newborn is not None}
    radius_squared = Fraction.from_float(epoch.radius_km)**2
    output = []
    for component in epoch.components:
        lower, upper = component.proxy_geometries, component.proxy_geometries
        for correction in epoch.paired_corrections:
            if correction.negative_component_id == component.component_id:
                lower = _subtract(lower, (correction.containing_chart,))
            if correction.positive_component_id == component.component_id:
                upper = _union(upper+(correction.containing_chart,))
        if component.material_kind == "original_live":
            allocation = allocations[component.original_parent_id]
            parent = parents[component.original_parent_id]
            total, archived = allocation.parent_area_bounds_sr, allocation.archived_area_bounds_sr
            birth = Fraction.from_float(source.journal.source_surface.time_myr)-allocation.original_age_myr
            age = epoch.time_myr-birth
            fields = allocation.material_fields
            properties = tuple(Fraction.from_float(value) for value in allocation.specific_properties)
            amounts = tuple((field,I(Fraction.from_float(getattr(parent.parcel,field)))) for field in EXTENSIVE_FIELDS)
        else:
            newborn = newborns[component.material_id]
            total, archived = component.area_bounds_sr, I(0)
            birth, age = newborn.birth_time_myr, epoch.time_myr-newborn.birth_time_myr
            fields, properties = newborn.material_fields, newborn.specific_properties
            amounts = (("area_km2",newborn.area_km2_bounds),
                       ("oceanic_volume_km3",newborn.oceanic_volume_km3_bounds),
                       ("cold_mantle_volume_km3",I(0)),("density_excess_mass_kg",I(0)))
        output.append(NativeImplicitBoundedMaterialComponent(component.component_id,component.material_id,
            component.original_parent_id,component.owner,component.material_kind,component.domain_expression,
            lower,upper,component.area_bounds_sr,total,archived,birth,age,fields,properties,amounts))
    return tuple(output)


@lru_cache(maxsize=2048)
def _normal_series(plane, donor_omega, receiver_omega, duration, order, precision):
    _, original = _plane_values(plane, precision)
    donor = tuple(Fraction.from_float(value)*duration for value in donor_omega)
    receiver = tuple(Fraction.from_float(value)*duration for value in receiver_omega)
    basis = tuple(_normal_coefficients(tuple(int(i==j) for i in range(3)),donor,receiver,order)
                  for j in range(3))
    coefficients = tuple(tuple(_rounded_interval(sum((basis[j][k][i]*original[j] for j in range(3)),I(0)),precision)
                               for i in range(3)) for k in range(order+1))
    rate = sum(map(abs,donor),Fraction(0))+sum(map(abs,receiver),Fraction(0))
    scale = sum((max(abs(value.lo),abs(value.hi)) for value in original),Fraction(0))
    remainder = scale*rate**(order+1)/factorial(order+1)/(1-rate/Fraction(order+2))
    return coefficients, _rounded_interval(I(-remainder,remainder),precision)


def _normal_bounds(plane, donor_omega,receiver_omega,duration,domain,order,precision):
    coefficients, tail = _normal_series(plane,donor_omega,receiver_omega,duration,order,precision)
    result = []
    for axis in range(3):
        value = I(0)
        for row in reversed(coefficients):
            value = _rounded_interval(value*domain+row[axis],precision)
        result.append(_rounded_interval(value+tail,precision))
    return tuple(result)


def _relative_geometry(geometry, donor_omega,receiver_omega,time):
    if not time:
        return geometry
    return _move(geometry,(RotationIncrement(receiver_omega,time),
        RotationIncrement(tuple(-value for value in donor_omega),time)))


def _swept_enclosures(donor_parts,donor_inner_parts,receiver_lower,receiver_upper,donor_omega,receiver_omega,
                       dt_myr,time_cells,precision,order):
    duration = Fraction.from_float(dt_myr)
    # Every displayed sample is itself an exact saved binary64 physical time
    # within the interval.  Only the outer proof uses exact rational subcells.
    # The endpoint supplies a certified subset of the whole temporal union.
    # Receiver and donor lower components are already disjoint material
    # subsets, so no duplicate sample-time union is counted as lower area.
    sample_times = (dt_myr,)
    endpoint_receivers=tuple(chart for receiver in receiver_lower
        for chart in _rational_charts(_relative_geometry(receiver,donor_omega,receiver_omega,dt_myr),False,precision))
    inner = tuple(overlap for receiver in endpoint_receivers
        for donor in donor_inner_parts if (overlap:=_intersection(donor,receiver)) is not None)
    outer = ()
    for octant in _sphere_octants():
        signs = tuple(next(plane.normal[axis] for plane in octant.supports if plane.normal[axis]) for axis in range(3))
        pieces = _intersect(donor_parts,(octant,))
        if not pieces:
            continue
        for index in range(time_cells):
            domain=I(Fraction(index,time_cells),Fraction(index+1,time_cells))
            for receiver in receiver_upper:
                for piece in pieces:
                    clipped = piece
                    for plane in receiver.support_planes:
                        bounds = _normal_bounds(plane,donor_omega,receiver_omega,duration,domain,order,precision)
                        corner = tuple(value.hi if sign>0 else value.lo for sign,value in zip(signs,bounds))
                        # On this fixed-sign octant this one corner dominates
                        # every true full-time support normal componentwise.
                        clipped = native_clip(clipped,NativePlane(corner))
                        if clipped is None:
                            break
                    if clipped is not None:
                        # An overlapping outer cover remains a valid upper
                        # measure enclosure.  It is never a finite material
                        # debit or a lower union area.
                        outer += (clipped,)
    return inner,outer


def _validate_previous(epoch):
    return (validate_native_implicit_material_continuation(epoch)
            if type(epoch) is NativeImplicitContinuationEpoch
            else validate_native_implicit_material_epoch(epoch))


def advance_native_implicit_material_epoch(epoch,omega_rad_per_myr,dt_myr,*,time_cells=1,
        precision=128,order=20,provenance):
    epoch = _validate_previous(epoch)
    if epoch.phase != "partition":
        raise ValueError("Fresh closure requires the complete prior implicit partition")
    owners=tuple(owner for owner,_ in epoch.additional_owner_histories)
    if type(omega_rad_per_myr) is not dict or set(omega_rad_per_myr)!=set(owners) or owners!=(0,1):
        raise ValueError("Fresh material law requires every actual owner and declared donor1/receiver0")
    steps={owner:RotationIncrement(omega_rad_per_myr[owner],dt_myr) for owner in owners}
    if type(time_cells) is not int or not 1<=time_cells<=256 or type(precision) is not int or not 96<=precision<=512 or type(order) is not int or not 12<=order<=40:
        raise ValueError("Fresh material closure needs explicit bounded temporal/analytic resources")
    if type(provenance) is not str or not provenance.strip():
        raise ValueError("Fresh material closure needs explicit provenance")
    motions=tuple((owner,steps[owner].omega_rad_per_myr) for owner in owners)
    context=_hash(dict(previous=epoch.epoch_sha256,motions=[[owner,list(omega)] for owner,omega in motions],
        dt_myr=steps[0].dt_myr,time_cells=time_cells,precision=precision,order=order,provenance=provenance))
    if context in _BUILD_CACHE:
        cached,body_bytes,seal,fingerprint=_BUILD_CACHE[context]
        if (_typed_fingerprint(cached)==fingerprint and _encoded(_body(cached))==body_bytes
            and cached.epoch_sha256==seal):
            return cached
        del _BUILD_CACHE[context]
    before=tuple(replace(row,lower_geometries=_rationalize(row.lower_geometries,False,precision),
        upper_geometries=_rationalize(row.upper_geometries,True,precision)) for row in _base_components(epoch))
    if steps[0].omega_rad_per_myr==steps[1].omega_rad_per_myr:
        # A common SO(3) bijection preserves the exact complete prior material
        # partition.  Its owner domains stay disjoint at every time, and the
        # endpoint has no vacancy.  Outer-chart overlap is not real material.
        endpoint=epoch.time_myr+Fraction.from_float(steps[0].dt_myr)
        moved=tuple(replace(row,
            domain_expression=f"common_SO3({row.component_id}@{epoch.epoch_sha256})",
            lower_geometries=_rationalize(tuple(_move(poly,(steps[row.owner],)) for poly in row.lower_geometries),False,precision),
            upper_geometries=_rationalize(tuple(_move(poly,(steps[row.owner],)) for poly in row.upper_geometries),True,precision),
            age_myr=endpoint-row.birth_time_myr) for row in before)
        result=NativeImplicitContinuationEpoch(epoch,motions,steps[0].dt_myr,time_cells,precision,order,
            provenance,moved,(),endpoint,"")
        result=replace(result,epoch_sha256=_hash(_body(result)))
        _REPLAY_CACHE[(epoch.epoch_sha256,_encoded(_body(result)),result.epoch_sha256,_typed_fingerprint(result))]=True
        if len(_BUILD_CACHE)>=8:
            _BUILD_CACHE.clear()
        _BUILD_CACHE[context]=(result,_encoded(_body(result)),result.epoch_sha256,_typed_fingerprint(result))
        return result
    receiver_lower=tuple(poly for row in before if row.owner==0 for poly in row.lower_geometries)
    receiver_upper=tuple(poly for row in before if row.owner==0 for poly in row.upper_geometries)
    retained,archives=[],[]
    endpoint=epoch.time_myr+Fraction.from_float(steps[0].dt_myr)
    for row in before:
        loss_lower,loss_upper=(),()
        if row.owner==1:
            sweep_lower,sweep_upper=_swept_enclosures(row.upper_geometries,row.lower_geometries,receiver_lower,receiver_upper,
                steps[1].omega_rad_per_myr,steps[0].omega_rad_per_myr,steps[0].dt_myr,time_cells,precision,order)
            loss_lower=sweep_lower
            loss_upper=sweep_upper
        loss=_bounded_area(loss_lower,loss_upper,I(0,row.area_bounds_sr.hi))
        archived_interval=row.archived_area_bounds_sr+loss
        total_archive=_rounded_interval(I(archived_interval.lo,
            min(row.parent_area_bounds_sr.hi,archived_interval.hi)),96)
        kept_lower=_subtract(row.lower_geometries,loss_upper)
        kept_upper=_subtract(row.upper_geometries,loss_lower)
        known=I(max(Fraction(0),row.area_bounds_sr.lo-loss.hi),max(Fraction(0),row.area_bounds_sr.hi-loss.lo))
        area=_bounded_area(kept_lower,kept_upper,known)
        expression=f"R_owner(1)({row.component_id}@{epoch.epoch_sha256} minus first_entry_union@{context})"
        retained.append(replace(row,domain_expression=expression,
            lower_geometries=_rationalize(tuple(_move(poly,(steps[row.owner],)) for poly in kept_lower),False,precision),
            upper_geometries=_rationalize(tuple(_move(poly,(steps[row.owner],)) for poly in kept_upper),True,precision),
            area_bounds_sr=area,archived_area_bounds_sr=total_archive,age_myr=endpoint-row.birth_time_myr))
        if loss.hi>0:
            measure=f"native_area(this_new_first_entry_cohort({row.component_id}@{context})) / native_area(material_parent({row.component_id}))"
            ratio=(loss/row.parent_area_bounds_sr if row.parent_area_bounds_sr.lo>0 else I(0,1))
            ratio=I(max(Fraction(0),ratio.lo),min(Fraction(1),ratio.hi))
            reservoirs=[]
            for field,parent in row.parent_amounts:
                if row.material_kind=="hot_birth":
                    factor=(Fraction.from_float(epoch.radius_km)**2 if field=="area_km2" else
                        Fraction.from_float(epoch.radius_km)**2*row.specific_properties[0] if field=="oceanic_volume_km3" else Fraction(0))
                    amount=loss*factor
                    expression=f"native_area(this_new_first_entry_cohort({row.component_id}@{context})) * physical_density({field})"
                    complement=I(max(Fraction(0),parent.lo-amount.hi),max(Fraction(0),parent.hi-amount.lo))
                else:
                    amount=parent*ratio
                    expression=f"parent_amount * {measure}"
                    complement=parent*(I(1)-ratio)
                reservoirs.append(NativeImplicitMaterialReservoir(field,parent,amount,complement,expression,
                    "parent_amount minus this incremental cohort archive_amount",
                    "parent_outside_this_cohort + incremental_cohort_amount = parent_amount"))
            reservoirs=tuple(reservoirs)
            archives.append(NativeImplicitContinuationArchive("implicit-archive-cohort:"+_hash(dict(context=context,parent=row.component_id)),
                row.component_id,row.material_id,row.owner,0,
                f"prior_live({row.component_id}) intersect union_0<=u<=1 inverse(R_1(u))(R_0(u)(ALL prior_live_owner0))",
                loss,total_archive,row.parent_area_bounds_sr,reservoirs,loss.lo>0))
    all_lower=tuple(poly for row in retained for poly in row.lower_geometries)
    all_upper=tuple(poly for row in retained for poly in row.upper_geometries)
    components=list(retained)
    radius_squared=Fraction.from_float(epoch.radius_km)**2
    parameters=epoch.source_material_state.birth_parameters
    thickness=Fraction.from_float(parameters.crust_thickness_km)
    for row in before:
        birth_lower=_subtract(row.lower_geometries,all_upper)
        birth_upper=_subtract(row.upper_geometries,all_lower)
        area=_bounded_area(birth_lower,birth_upper)
        if area.hi==0:
            continue
        identity="implicit-hot-birth:"+_hash(dict(context=context,parent=row.component_id))
        actual_area=area*radius_squared
        amounts=(("area_km2",actual_area),("oceanic_volume_km3",actual_area*thickness),
                 ("cold_mantle_volume_km3",I(0)),("density_excess_mass_kg",I(0)))
        components.append(NativeImplicitBoundedMaterialComponent(identity,identity,row.original_parent_id,row.owner,"hot_birth",
            f"prior_live({row.component_id}@{epoch.epoch_sha256}) intersect complement(ALL moved retained material@{context})",
            birth_lower,birth_upper,area,area,I(0),endpoint,Fraction(0),
            (("damage",parameters.damage),("water",parameters.water)),(thickness,Fraction(0),Fraction(0)),amounts))
    result=NativeImplicitContinuationEpoch(epoch,motions,steps[0].dt_myr,time_cells,precision,order,
        provenance,tuple(components),tuple(archives),endpoint,"")
    result=replace(result,epoch_sha256=_hash(_body(result)))
    body=_encoded(_body(result))
    if len(_REPLAY_CACHE)>=8:
        _REPLAY_CACHE.clear()
    _REPLAY_CACHE[(epoch.epoch_sha256,body,result.epoch_sha256,_typed_fingerprint(result))]=True
    if len(_BUILD_CACHE)>=8:
        _BUILD_CACHE.clear()
    _BUILD_CACHE[context]=(result,body,result.epoch_sha256,_typed_fingerprint(result))
    return result


_REPLAY_CACHE={}


def validate_native_implicit_material_continuation(epoch):
    if type(epoch) is not NativeImplicitContinuationEpoch:
        raise ValueError("Expected a complete fresh implicit material continuation")
    # Keys bind every displayed and authoritative field; a seal alone does not
    # confer validity and bool/int aliasing cannot hit the cache.
    body=_encoded(_body(epoch))
    key=(epoch.previous_epoch.epoch_sha256,body,epoch.epoch_sha256,_typed_fingerprint(epoch))
    previous=_validate_previous(epoch.previous_epoch)
    if key in _REPLAY_CACHE:
        return epoch
    replay=advance_native_implicit_material_epoch(previous,dict(epoch.omega_rad_per_myr),epoch.dt_myr,
        time_cells=epoch.time_cells,precision=epoch.precision,order=epoch.order,provenance=epoch.provenance)
    if (_typed_fingerprint(epoch)!=_typed_fingerprint(replay) or
        body!=_encoded(_body(replay)) or epoch.epoch_sha256!=replay.epoch_sha256):
        raise ValueError("Fresh implicit material closure differs from complete prior mask and new SO(3) replay")
    if len(_REPLAY_CACHE)>=8:
        _REPLAY_CACHE.clear()
    _REPLAY_CACHE[key]=True
    return epoch


def native_implicit_material_continuation_to_dict(epoch,*,include_previous=True,include_source=True):
    epoch=validate_native_implicit_material_continuation(epoch)
    raw=dict(_body(epoch),epoch_sha256=epoch.epoch_sha256)
    if include_previous:
        from .geometric_native_implicit_material_epoch import native_implicit_material_epoch_to_dict
        raw["previous_epoch"]=native_implicit_material_epoch_to_dict(epoch.previous_epoch,include_source=include_source)
    return raw


def native_implicit_material_continuation_from_dict(raw,*,expected_source_material_state,expected_previous_epoch=None):
    if type(raw) is not dict or raw.get("format")!=FORMAT:
        raise ValueError("Invalid fresh implicit material continuation schema")
    from .geometric_native_implicit_material_epoch import native_implicit_material_epoch_from_dict
    try:
        from .geometric_native_first_entry_material import (
            native_first_entry_material_to_dict,validate_native_first_entry_material,
        )
        trusted_source=validate_native_first_entry_material(expected_source_material_state)
        previous=(native_implicit_material_epoch_from_dict(raw["previous_epoch"],expected_source_material_state=expected_source_material_state)
                  if "previous_epoch" in raw else _validate_previous(expected_previous_epoch))
        if (previous.source_material_state.state_sha256!=trusted_source.state_sha256 or
            _encoded(native_first_entry_material_to_dict(previous.source_material_state))!=
            _encoded(native_first_entry_material_to_dict(trusted_source))):
            raise ValueError("Fresh material continuation belongs to another independently supplied original material source")
        if expected_previous_epoch is not None:
            from .geometric_native_implicit_material_epoch import native_implicit_material_epoch_to_dict
            trusted=_validate_previous(expected_previous_epoch)
            if (previous.epoch_sha256!=trusted.epoch_sha256 or
                _encoded(native_implicit_material_epoch_to_dict(previous,include_source=False))!=
                _encoded(native_implicit_material_epoch_to_dict(trusted,include_source=False))):
                raise ValueError("Saved fresh material continuation belongs to another independently supplied prior force epoch")
        motions=raw["omega_rad_per_myr"]
        if type(motions) is not list or any(type(row) is not list or len(row)!=2 or type(row[0]) is not int
            or type(row[1]) is not list for row in motions):
            raise ValueError("Fresh material motions require canonical JSON owner records")
        replay=advance_native_implicit_material_epoch(previous,{owner:tuple(omega) for owner,omega in motions},raw["dt_myr"],
            time_cells=raw["time_cells"],precision=raw["precision"],order=raw["order"],provenance=raw["provenance"])
        source_node=raw.get("previous_epoch",{})
        while type(source_node) is dict and "previous_epoch" in source_node:
            source_node=source_node["previous_epoch"]
        expected=native_implicit_material_continuation_to_dict(replay,include_previous="previous_epoch" in raw,
            include_source=("source_material_state" in source_node))
        if _encoded(raw)!=_encoded(expected):
            raise ValueError("Fresh material continuation differs from full source/previous masks/new SO(3) replay")
        return replay
    except (KeyError,TypeError) as error:
        raise ValueError("Invalid complete fresh implicit material continuation payload") from error
