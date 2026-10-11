"""Explicit source-kind dispatch without changing existing source meanings."""
from __future__ import annotations

import hashlib
import json
import numpy as np

from .geometric_basal_loading import (
    FORMAT as DEFAULT_FORMAT, GeometricBasalLoad,
    basal_load_from_dict, basal_load_to_dict, validate_geometric_basal_load,
)
from .geometric_rigid_mantle_loading import (
    FORMAT as RIGID_FORMAT, PhysicalRigidMantleLoad, PrescribedRigidMantleSource,
    rigid_mantle_load_from_dict, rigid_mantle_load_to_dict,
    validate_physical_rigid_mantle_load,
)
from .geometric_spatial_mantle_loading import (
    FORMAT as SPATIAL_FORMAT, PhysicalSpatialMantleLoad, PrescribedSpatialMantleSource,
    spatial_mantle_load_from_dict, spatial_mantle_load_to_dict,
    validate_physical_spatial_mantle_load,
)


def validate_basal_source(surface, load, *, mesh=None, parameters=None,
                          thermal_lid_by_fragment=None, provenance=None, source=None):
    if isinstance(load, GeometricBasalLoad):
        if source is not None:
            raise ValueError("Quadratic/thermal basal loading does not use a rigid mantle source")
        return validate_geometric_basal_load(surface, load, mesh=mesh, parameters=parameters,
            thermal_lid_by_fragment=thermal_lid_by_fragment, provenance=provenance)
    if isinstance(load, PhysicalRigidMantleLoad):
        if thermal_lid_by_fragment is not None:
            raise ValueError("Explicit fully transmitted rigid mantle source has no thermal lid coupling")
        if parameters is not None:
            if not isinstance(parameters, PrescribedRigidMantleSource) or (source is not None and source != parameters):
                raise ValueError("Rigid mantle source parameters cannot be quadratic source metadata")
            source = parameters
        return validate_physical_rigid_mantle_load(surface, load, mesh=mesh,
                                                  source=source, provenance=provenance)
    if isinstance(load, PhysicalSpatialMantleLoad):
        if thermal_lid_by_fragment is not None:
            raise ValueError("Explicit fully transmitted spatial mantle source has no thermal lid coupling")
        if parameters is not None:
            if not isinstance(parameters, PrescribedSpatialMantleSource) or (source is not None and source != parameters):
                raise ValueError("Spatial mantle source parameters must be explicit spatial source metadata")
            source = parameters
        return validate_physical_spatial_mantle_load(surface, load, mesh=mesh,
                                                    source=source, provenance=provenance)
    raise ValueError("Unknown explicit physical basal source kind")


def basal_source_to_dict(surface, load):
    if isinstance(load, GeometricBasalLoad):
        return basal_load_to_dict(surface, load)
    if isinstance(load, PhysicalRigidMantleLoad):
        return rigid_mantle_load_to_dict(surface, load)
    if isinstance(load, PhysicalSpatialMantleLoad):
        return spatial_mantle_load_to_dict(surface, load)
    raise ValueError("Unknown explicit physical basal source kind")


def basal_source_from_dict(raw, surface, *, mesh=None, parameters=None,
                           thermal_lid_by_fragment=None, provenance=None, source=None):
    if not isinstance(raw, dict):
        raise ValueError("Explicit basal source checkpoint must be an object")
    if raw.get("format") == DEFAULT_FORMAT:
        if source is not None:
            raise ValueError("Quadratic/thermal basal loading does not use a rigid mantle source")
        return basal_load_from_dict(raw, surface, mesh=mesh, parameters=parameters,
            thermal_lid_by_fragment=thermal_lid_by_fragment, provenance=provenance)
    if raw.get("format") == RIGID_FORMAT:
        if thermal_lid_by_fragment is not None:
            raise ValueError("Explicit fully transmitted rigid mantle source has no thermal lid coupling")
        if parameters is not None:
            if not isinstance(parameters, PrescribedRigidMantleSource) or (source is not None and source != parameters):
                raise ValueError("Rigid mantle source parameters cannot be quadratic source metadata")
            source = parameters
        return rigid_mantle_load_from_dict(raw, surface, mesh=mesh, source=source, provenance=provenance)
    if raw.get("format") == SPATIAL_FORMAT:
        if thermal_lid_by_fragment is not None:
            raise ValueError("Explicit fully transmitted spatial mantle source has no thermal lid coupling")
        if parameters is not None:
            if not isinstance(parameters, PrescribedSpatialMantleSource) or (source is not None and source != parameters):
                raise ValueError("Spatial mantle source parameters must be explicit spatial source metadata")
            source = parameters
        return spatial_mantle_load_from_dict(raw, surface, mesh=mesh, source=source, provenance=provenance)
    raise ValueError("Unknown basal source checkpoint format; no implicit model migration")


def basal_source_operators(surface, load):
    validate_basal_source(surface, load)
    signature = hashlib.sha256(json.dumps(basal_source_to_dict(surface, load), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    return tuple(load.plate_ids), np.asarray(load.drag_nm_s, dtype=float), np.asarray(load.torque_nm, dtype=float), signature
