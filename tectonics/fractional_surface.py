"""Pure conservative parcel bookkeeping for experimental fractional transport.

The surface is a complete tiling in *area*, with multiple material parcels per
cell. Plate ownership and material histories remain parcel properties. This
module supplies no overlap, subduction-polarity, ridge, or mechanical law and
does not activate fractional transport in the mature raster runner.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from numbers import Integral
from typing import Iterable


FORMAT = "fractional-surface-1"
MATERIAL_MODEL = "oceanic_only_v1"
EXTENSIVE_FIELDS = ("area_km2", "oceanic_volume_km3", "cold_mantle_volume_km3",
                    "density_excess_mass_kg")
_REL_TOL = 2e-12
_ROUND_TOL = 64.*math.ulp(1.)


def _finite(value, label, *, positive=False):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite physical number")
    value = float(value)
    if not math.isfinite(value) or (value <= 0. if positive else value < 0.):
        raise ValueError(f"{label} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def _index(value, label):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return int(value)


def _close(actual, expected):
    return math.isclose(actual, expected, rel_tol=_REL_TOL, abs_tol=0.)


def _roundoff_close(actual, expected):
    # No absolute floor: a zero reservoir must stay exactly zero even when
    # a parcel is tiny. Local splitting adds only multiplication roundoff.
    return math.isclose(actual, expected, rel_tol=_ROUND_TOL, abs_tol=0.)


@dataclass(frozen=True, slots=True)
class SurfaceParcel:
    """One homogeneous oceanic material history, possibly only part of a cell.

    ``material_id`` identifies its origin and survives splitting. Intensives
    are immutable named scalars; categorical masks can use exact zero/one.
    Separate histories are never replaced by their mean age or mean damage.
    """
    cell: int
    plate: int
    material_id: str
    area_km2: float
    oceanic_volume_km3: float
    cold_mantle_volume_km3: float
    density_excess_mass_kg: float
    age_myr: float
    material_fields: tuple[tuple[str, float], ...] = ()
    # Carried exactly through splits: recomputing V/A after every multiply can
    # change its final bit and prevent identical material from recombining.
    specific_properties: tuple[float, float, float] = ()

    def __post_init__(self):
        object.__setattr__(self, "cell", _index(self.cell, "Parcel cell"))
        object.__setattr__(self, "plate", _index(self.plate, "Parcel plate"))
        if not isinstance(self.material_id, str) or not self.material_id:
            raise ValueError("Parcel material_id must be a nonempty stable string")
        for name in EXTENSIVE_FIELDS:
            object.__setattr__(self, name, _finite(getattr(self, name), name,
                positive=name in ("area_km2", "oceanic_volume_km3")))
        object.__setattr__(self, "age_myr", _finite(self.age_myr, "Parcel age"))
        if self.cold_mantle_volume_km3 == 0. and self.density_excess_mass_kg != 0.:
            raise ValueError("Density excess mass requires a cold mantle volume")
        specific = tuple(self.specific_properties)
        if not specific:
            specific = tuple(getattr(self, name)/self.area_km2 for name in EXTENSIVE_FIELDS[1:])
        if len(specific) != 3:
            raise ValueError("Parcel specific properties require three areal densities")
        specific = tuple(_finite(value, "Parcel specific property") for value in specific)
        if any(not _roundoff_close(getattr(self, name), self.area_km2*value)
               for name, value in zip(EXTENSIVE_FIELDS[1:], specific)):
            raise ValueError("Parcel extensive values disagree with its inherited specific properties")
        object.__setattr__(self, "specific_properties", specific)
        fields = []
        seen = set()
        for name, value in self.material_fields:
            if not isinstance(name, str) or not name or name in seen:
                raise ValueError("Material field names must be nonempty and unique")
            if not math.isfinite(float(value)):
                raise ValueError("Material fields must be finite")
            seen.add(name)
            fields.append((name, float(value)))
        object.__setattr__(self, "material_fields", tuple(sorted(fields)))

    @property
    def mantle_thickness_km(self):
        return self.cold_mantle_volume_km3/self.area_km2

    @property
    def density_anomaly_kg_m3(self):
        return (self.density_excess_mass_kg/(self.cold_mantle_volume_km3*1e9)
                if self.cold_mantle_volume_km3 else 0.)


@dataclass(frozen=True, slots=True)
class FractionalSurfaceState:
    time_myr: float
    cell_areas_km2: tuple[float, ...]
    parcels: tuple[SurfaceParcel, ...]
    material_model: str = MATERIAL_MODEL
    version: str = FORMAT
    known_material_ids: tuple[str, ...] = ()

    def __post_init__(self):
        if self.version != FORMAT or self.material_model != MATERIAL_MODEL:
            raise ValueError("Unsupported fractional surface format or material model; oceanic only")
        object.__setattr__(self, "time_myr", _finite(self.time_myr, "Surface time"))
        areas = tuple(_finite(value, "Cell area", positive=True) for value in self.cell_areas_km2)
        if not areas:
            raise ValueError("Fractional surface requires positive cell areas")
        object.__setattr__(self, "cell_areas_km2", areas)
        parcels = tuple(self.parcels)
        _validate_parcels(parcels, len(areas))
        _validate_capacity(areas, parcels)
        object.__setattr__(self, "parcels", parcels)
        origins = {parcel.material_id for parcel in parcels}
        known = tuple(self.known_material_ids)
        if any(not isinstance(identity, str) or not identity for identity in known):
            raise ValueError("Historical material origin IDs must be nonempty strings")
        if known and not origins.issubset(known):
            raise ValueError("Surface material origin is missing from its historical ID registry")
        object.__setattr__(self, "known_material_ids", tuple(sorted(set(known) | origins)))

    @property
    def cell_count(self):
        return len(self.cell_areas_km2)


@dataclass(frozen=True, slots=True)
class TransferPiece:
    """A geometric remap fraction of one source parcel, before overlap policy."""
    source_index: int
    target_cell: int
    fraction: float

    def __post_init__(self):
        object.__setattr__(self, "source_index", _index(self.source_index, "Transfer source"))
        object.__setattr__(self, "target_cell", _index(self.target_cell, "Transfer target"))
        fraction = _finite(self.fraction, "Transfer fraction", positive=True)
        if fraction > 1.:
            raise ValueError("Transfer fraction cannot exceed one")
        object.__setattr__(self, "fraction", fraction)


@dataclass(frozen=True, slots=True)
class IncomingPiece:
    """Transferred material with original donor lineage retained for accounting."""
    source_index: int
    fraction: float
    parcel: SurfaceParcel

    def __post_init__(self):
        object.__setattr__(self, "source_index", _index(self.source_index, "Incoming source"))
        fraction = _finite(self.fraction, "Incoming fraction", positive=True)
        if fraction > 1.:
            raise ValueError("Incoming fraction cannot exceed one")
        object.__setattr__(self, "fraction", fraction)
        if not isinstance(self.parcel, SurfaceParcel):
            raise ValueError("Incoming material must be a SurfaceParcel")


def _validate_parcels(parcels, cell_count):
    for parcel in parcels:
        if not isinstance(parcel, SurfaceParcel) or parcel.cell >= cell_count:
            raise ValueError("Surface parcel references an invalid target cell")


def _validate_capacity(areas, parcels):
    by_cell = [[] for _ in areas]
    for parcel in parcels:
        by_cell[parcel.cell].append(parcel.area_km2)
    for cell, (area, pieces) in enumerate(zip(areas, by_cell)):
        if not _close(math.fsum(pieces), area):
            raise ValueError(f"Surface target capacity does not close in cell {cell}")


def _validate_partition(state, pieces):
    fractions = [[] for _ in state.parcels]
    for piece in pieces:
        if piece.source_index >= len(state.parcels):
            raise ValueError("Transfer references an invalid source parcel")
        fractions[piece.source_index].append(piece.fraction)
    for source, values in enumerate(fractions):
        if not _close(math.fsum(values), 1.):
            raise ValueError(f"Surface donor budget does not close for source {source}")


def split_parcel(parcel, fraction, *, target_cell=None, age_increment_myr=0.):
    """Split every extensive quantity by one fraction; preserve all intensives."""
    fraction = _finite(fraction, "Parcel split fraction", positive=True)
    if fraction > 1.:
        raise ValueError("Parcel split fraction cannot exceed one")
    age = parcel.age_myr+_finite(age_increment_myr, "Age increment")
    values = {name: getattr(parcel, name)*fraction for name in EXTENSIVE_FIELDS}
    return replace(parcel, cell=parcel.cell if target_cell is None else target_cell,
                   age_myr=age, **values)


def split_incoming(piece, fraction):
    """Take a fraction of a candidate, preserving its original-source fraction."""
    parcel = split_parcel(piece.parcel, fraction)
    return IncomingPiece(piece.source_index, piece.fraction*float(fraction), parcel)


def remap_surface(state, transfers: Iterable[TransferPiece], dt_myr=0.):
    """Apply a complete donor map, allowing unresolved target overlaps and gaps.

    A caller supplies the geometric transfer. No nearest-donor guessing, area
    repair, or ridge generation occurs here. Target capacity is enforced only
    at commit, after the external overlap policy has partitioned candidates.
    """
    dt = _finite(dt_myr, "Transport timestep")
    transfers = tuple(transfers)
    if any(not isinstance(piece, TransferPiece) or piece.target_cell >= state.cell_count
           for piece in transfers):
        raise ValueError("Geometric transfer must reference valid target cells")
    _validate_partition(state, transfers)
    return tuple(IncomingPiece(piece.source_index, piece.fraction,
        split_parcel(state.parcels[piece.source_index], piece.fraction,
                     target_cell=piece.target_cell, age_increment_myr=dt))
        for piece in transfers)


def _validate_lineage(state, pieces, dt):
    for piece in pieces:
        if not isinstance(piece, IncomingPiece) or piece.source_index >= len(state.parcels):
            raise ValueError("Incoming piece must reference a valid original donor")
        source = state.parcels[piece.source_index]
        parcel = piece.parcel
        if parcel.cell >= state.cell_count:
            raise ValueError("Incoming piece references an invalid target cell")
        if (parcel.plate != source.plate or parcel.material_id != source.material_id
                or parcel.material_fields != source.material_fields
                or parcel.specific_properties != source.specific_properties
                or parcel.age_myr != source.age_myr+dt):
            raise ValueError("Incoming material history does not match its source lineage")
        if any(not _roundoff_close(getattr(parcel, name), getattr(source, name)*piece.fraction)
               for name in EXTENSIVE_FIELDS):
            raise ValueError("Incoming extensive material does not match its source fraction")


def _merge_identical_histories(parcels):
    """Only coalesce identical origin/history and specific material properties.

    A material origin can develop different local histories; such parcels stay
    distinct even in the same cell. No lossy tolerance-bin merging is used.
    """
    groups = {}
    for parcel in parcels:
        key = (parcel.cell, parcel.plate, parcel.material_id, parcel.age_myr,
               parcel.material_fields, parcel.specific_properties)
        groups.setdefault(key, []).append(parcel)
    result = []
    for _, group in sorted(groups.items()):
        result.append(replace(group[0], **{name: math.fsum(getattr(p, name) for p in group)
                                         for name in EXTENSIVE_FIELDS}))
    return tuple(result)


def commit_surface(state, retained: Iterable[IncomingPiece], losses: Iterable[IncomingPiece],
                   newborn: Iterable[SurfaceParcel], dt_myr=0.):
    """Validate and atomically construct a complete post-transport surface.

    Retained and lost pieces partition each original donor once. Losses are
    returned/consumed by the caller's physical ledger, never subtracted again.
    Newborn parcels fill only actual capacity left after retained material.
    They must carry fresh origin IDs; this routine invents no ridge material.
    All checks precede construction and inputs are immutable on every failure.
    """
    dt = _finite(dt_myr, "Transport timestep", positive=True)
    end_time = state.time_myr+dt
    if not math.isfinite(end_time) or end_time <= state.time_myr:
        raise ValueError("Surface commit requires a finite, representably advancing clock")
    retained, losses, newborn = tuple(retained), tuple(losses), tuple(newborn)
    pieces = retained+losses
    _validate_lineage(state, pieces, dt)
    _validate_partition(state, pieces)
    _validate_parcels(newborn, state.cell_count)
    original_ids = set(state.known_material_ids)
    for parcel in newborn:
        if parcel.material_id in original_ids:
            raise ValueError("Newborn material must have a fresh origin ID")
        if parcel.age_myr != 0.:
            raise ValueError("Newborn oceanic material must start at age zero")
    proposed = tuple(piece.parcel for piece in retained)+newborn
    _validate_capacity(state.cell_areas_km2, proposed)
    merged = _merge_identical_histories(proposed)
    return FractionalSurfaceState(end_time, state.cell_areas_km2, merged,
                                  state.material_model, state.version,
                                  tuple(sorted(original_ids | {p.material_id for p in newborn})))


def surface_totals(state):
    return {name: math.fsum(getattr(parcel, name) for parcel in state.parcels)
            for name in EXTENSIVE_FIELDS}


def state_to_json(state):
    """JSON-safe snapshot of the complete sparse state; no history projection."""
    return {"version": state.version, "material_model": state.material_model,
            "time_myr": state.time_myr, "cell_areas_km2": list(state.cell_areas_km2),
            "known_material_ids": list(state.known_material_ids),
            "parcels": [asdict(parcel) for parcel in state.parcels]}


def state_from_json(data):
    """Load only an explicit complete fractional snapshot, validating budgets."""
    if "known_material_ids" not in data:
        raise ValueError("Fractional snapshot is missing its historical material ID registry")
    return FractionalSurfaceState(time_myr=data["time_myr"],
        cell_areas_km2=tuple(data["cell_areas_km2"]),
        parcels=tuple(SurfaceParcel(**row) for row in data["parcels"]),
        material_model=data["material_model"], version=data["version"],
        known_material_ids=tuple(data["known_material_ids"]))


__all__ = ["FORMAT", "MATERIAL_MODEL", "EXTENSIVE_FIELDS", "SurfaceParcel",
           "FractionalSurfaceState", "TransferPiece", "IncomingPiece", "split_parcel",
           "split_incoming", "remap_surface", "commit_surface", "surface_totals",
           "state_to_json", "state_from_json"]
