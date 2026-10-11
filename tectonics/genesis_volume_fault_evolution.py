"""Source-bound prospective weakening of the connected Genesis volume.

The inherited source arrays remain literal history.  This adapter advances
water availability, diffuse damage and the persistent weak-plane timer using
calculated current stresses.  A ready weak plane is an explicit boundary for
the current elastic volume solver, not an invented sharp crack or plate.

The Helmholtz graph keeps the source's material angular conductances and uses
the current flat physical top areas.  Source spherical and flat areas differ:
this is an explicit prospective geometry closure, not the old spherical law
silently evaluated on a fabricated SphereMesh.  Mechanical damage energy is
accounted for by the caller's full-cohort volume loading, not by this adapter.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from functools import lru_cache
import json
import math
from numbers import Real
from types import SimpleNamespace

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from .genesis_bulk_export import validate_genesis_bulk_material
from .genesis_fault_law import select_plane
from .genesis_onset import update_water_access
from .genesis_onset_support import _arc
from .genesis_shell import principal_tensile
from .genesis_tides import TidalOrbitState, tidal_strain_cycle, validate_tidal_orbit
from .genesis_volume_thermal import _context, _json

FORMAT = "genesis-source-bound-volume-fault-evolution-0.1"
NONLOCAL_GEOMETRY = "source_material_angular_conductance_with_current_physical_chord_area"
STRESS_SCOPE = "calculated_volume_face_mean_projected_in_current_top_face_axes"
_ARRAYS = ("damage", "water_access", "peak_tensile_pa", "weak_duration_myr",
    "tidal_stress_mpa", "fault_active", "plane_normal", "activation_time_myr",
    "fault_candidate_age_myr", "cumulative_shear", "signed_shear",
    "last_shear_increment", "shear_stress_pa", "normal_stress_pa", "shear_strength_pa",
    "friction_work_cell_j", "viscous_work_cell_j")


def _number(value, name, *, nonnegative=True):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"Fault evolution {name} must be a finite real")
    value = float(value)
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f"Fault evolution {name} is outside its finite range")
    return value


def _array(value, shape, name, *, boolean=False):
    a = np.asarray(value)
    if a.shape != shape or (boolean and a.dtype.kind != "b"):
        raise ValueError(f"Fault evolution {name} has invalid shape or type")
    if not boolean:
        if a.dtype.kind not in "fiu" or not np.isfinite(a).all():
            raise ValueError(f"Fault evolution {name} must be finite numeric data")
        a = np.asarray(a, dtype=float)
    # Owned immutable bytes, not a read-only view of a mutable caller buffer.
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _sha(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(f"Fault evolution {name} must be a SHA256 identity")


@dataclass(frozen=True, slots=True)
class VolumeFaultHistory:
    source_checkpoint_sha256: str
    source_identity: str
    initial_time_myr: float
    time_myr: float
    accepted_steps: int
    membrane_established: bool
    first_fracture_time_myr: float | None
    damage: np.ndarray
    water_access: np.ndarray
    peak_tensile_pa: np.ndarray
    weak_duration_myr: np.ndarray
    tidal_stress_mpa: np.ndarray
    fault_active: np.ndarray
    plane_normal: np.ndarray
    activation_time_myr: np.ndarray
    fault_candidate_age_myr: np.ndarray
    cumulative_shear: np.ndarray
    signed_shear: np.ndarray
    last_shear_increment: np.ndarray
    shear_stress_pa: np.ndarray
    normal_stress_pa: np.ndarray
    shear_strength_pa: np.ndarray
    friction_work_cell_j: np.ndarray
    viscous_work_cell_j: np.ndarray
    friction_work_j: float
    viscous_fault_work_j: float

    def __post_init__(self):
        _sha(self.source_checkpoint_sha256, "source checkpoint")
        _sha(self.source_identity, "birth source")
        for name in ("initial_time_myr", "time_myr", "friction_work_j", "viscous_fault_work_j"):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        if (self.time_myr < self.initial_time_myr or type(self.accepted_steps) is not int
                or self.accepted_steps < 0 or type(self.membrane_established) is not bool):
            raise ValueError("Fault evolution clock, counter or membrane gate is invalid")
        if self.first_fracture_time_myr is not None:
            first = _number(self.first_fracture_time_myr, "first diffuse fracture time")
            if first > self.time_myr:
                raise ValueError("Fault evolution first diffuse fracture lies in the future")
            object.__setattr__(self, "first_fracture_time_myr", first)
        n = np.asarray(self.damage).size
        if np.asarray(self.damage).shape != (n,) or n == 0:
            raise ValueError("Fault evolution requires one damage value per source face")
        for name in _ARRAYS:
            shape = (n, 2) if name == "plane_normal" else (n,)
            object.__setattr__(self, name, _array(getattr(self, name), shape, name,
                boolean=name == "fault_active"))
        for name in ("damage", "water_access"):
            a = getattr(self, name)
            if np.any((a < 0) | (a > 1)):
                raise ValueError(f"Fault evolution {name} must lie in [0, 1]")
        for name in ("peak_tensile_pa", "weak_duration_myr", "tidal_stress_mpa",
                "fault_candidate_age_myr", "cumulative_shear", "shear_strength_pa",
                "friction_work_cell_j", "viscous_work_cell_j"):
            if np.any(getattr(self, name) < 0):
                raise ValueError(f"Fault evolution {name} must be nonnegative")
        if (np.any(self.weak_duration_myr > self.time_myr+1e-12)
                or np.any(self.fault_candidate_age_myr > self.time_myr+1e-12)):
            raise ValueError("Fault evolution persistence exceeds the material clock")
        active = self.fault_active
        if (np.any(active) and not self.membrane_established
                or not np.allclose(np.linalg.norm(self.plane_normal[active], axis=1), 1., atol=1e-10, rtol=0)
                or np.any(self.plane_normal[~active] != 0)
                or np.any(self.activation_time_myr[~active] != -1)
                or np.any((self.activation_time_myr[active] < 0)
                          | (self.activation_time_myr[active] > self.time_myr))
                or np.any(np.abs(self.signed_shear) > self.cumulative_shear+1e-12)
                or np.any(np.abs(self.last_shear_increment) > self.cumulative_shear+1e-12)
                or np.any(self.cumulative_shear[~active] != 0)):
            raise ValueError("Fault evolution material plane or shear history is invalid")
        for name, total in (("friction_work_cell_j", self.friction_work_j),
                            ("viscous_work_cell_j", self.viscous_fault_work_j)):
            a = getattr(self, name)
            if np.any(a[~active] != 0) or not math.isclose(float(a.sum()), total, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Fault evolution fault work disagrees with inherited cells")

    def __deepcopy__(self, memo):
        return self

    @property
    def birth_source_identity(self):
        return self.source_identity


@lru_cache(maxsize=4)
def _validated_source(source_json):
    packet = json.loads(source_json)
    validate_genesis_bulk_material(packet)
    return _context(source_json)


def _source(packet):
    if not isinstance(packet, dict):
        raise ValueError("Fault evolution requires an authenticated actual Genesis packet")
    return _validated_source(_json(packet))


def validate_volume_fault_history(history, packet):
    if type(history) is not VolumeFaultHistory:
        raise ValueError("Fault evolution requires immutable inherited history")
    c = _source(packet)
    if (history.source_checkpoint_sha256 != c.packet["source"]["checkpoint_sha256"]
            or history.source_identity != c.packet["birth_source_identity"]
            or history.initial_time_myr != c.state.time_myr
            or history.damage.shape != (c.mesh.cell_count,)
            or history.membrane_established != c.state.membrane_established):
        raise ValueError("Fault evolution history lost its source, face coverage or membrane gate")
    fp = c.model.fault_p
    if (np.any(np.abs(history.last_shear_increment) > fp.max_shear_increment)
            or np.any(history.activation_time_myr[history.fault_active]+1e-12 < fp.activation_persistence_myr)
            or (not fp.enabled and (np.any(history.fault_active)
                or history.friction_work_j != 0 or history.viscous_fault_work_j != 0))):
        raise ValueError("Fault evolution history contradicts source material parameters")
    return c


def initial_volume_fault_history(packet):
    """Inherit every Fault array and its accumulated work without reset."""
    c = _source(packet)
    values = {name: getattr(c.state, name) for name in _ARRAYS}
    return VolumeFaultHistory(c.packet["source"]["checkpoint_sha256"],
        c.packet["birth_source_identity"], c.state.time_myr, c.state.time_myr, 0,
        c.state.membrane_established, c.state.first_fracture_time_myr,
        **values, friction_work_j=c.state.friction_work_j,
        viscous_fault_work_j=c.state.viscous_fault_work_j)


class VolumeFaultEvolutionStepLimit(ValueError):
    """A prospective damage update needs a smaller accepted interval."""


def _physical_top(c, vertices):
    x = _array(vertices, (c.mesh.vertex_count, 3), "physical top vertices")
    triangles = x[c.mesh.faces]
    cross = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    area = .5*np.linalg.norm(cross, axis=1)/1e6
    if np.any(area <= 0) or not np.isfinite(area).all():
        raise ValueError("Fault evolution physical top has a degenerate face")
    # Only the explicit face geometry protocol used by tidal_strain_cycle.
    # These physical vertices are not normalized or wrapped in SphereMesh.
    return SimpleNamespace(vertices=x, faces=c.mesh.faces, cell_count=c.mesh.cell_count), area


def _nonlocal_loading(c, values, active, area_km2):
    out = np.zeros_like(values)
    indices = np.flatnonzero(active)
    if not len(indices):
        return out
    length = c.model.onset_p.regularization_km
    edges = np.asarray(c.mesh.shared_edges, dtype=int)
    left, right = edges[:, 0], edges[:, 1]
    use = active[left] & active[right]
    conductance = (_arc(c.mesh.vertices[edges[:, 2]], c.mesh.vertices[edges[:, 3]])
        / _arc(c.mesh.centroids[left], c.mesh.centroids[right]))[use]*length**2
    lookup = np.full(c.mesh.cell_count, -1, dtype=int)
    lookup[indices] = np.arange(len(indices))
    a, b = lookup[left[use]], lookup[right[use]]
    L = sparse.coo_matrix((np.r_[conductance, conductance, -conductance, -conductance],
        (np.r_[a, b, a, b], np.r_[a, b, b, a])), shape=(len(indices), len(indices))).tocsc()
    weights = area_km2[indices]
    out[indices] = splu(sparse.diags(weights, format="csc")+L).solve(weights*values[indices])
    if not np.isfinite(out).all() or np.any(out < 0):
        raise ValueError("Fault evolution nonlocal positive loading is unresolved")
    return out


def prepare_volume_fault_activation(history, packet, face_stress_pa):
    """Prepare an actual ready weak-plane event, leaving accepted state intact.

    Returns ``(candidate_history, report)``; with no event this is the original
    history and None.  The current elastic solver must stop at this boundary
    rather than publishing an unsupported plastic/contact interval.
    """
    c = validate_volume_fault_history(history, packet)
    stress = _array(face_stress_pa, (c.mesh.cell_count, 3), "current projected face stress")
    fp = c.model.fault_p
    ready = (~history.fault_active & (history.damage >= fp.activation_damage)
        & (history.fault_candidate_age_myr >= fp.activation_persistence_myr))
    if not fp.enabled or not history.membrane_established or not np.any(ready):
        return history, None
    normals, times = history.plane_normal.copy(), history.activation_time_myr.copy()
    mu = fp.friction_dry+(fp.friction_wet-fp.friction_dry)*history.water_access
    normals[ready] = select_plane(stress[ready], mu[ready])
    times[ready] = history.time_myr
    candidate = replace(history, fault_active=history.fault_active | ready,
        plane_normal=normals, activation_time_myr=times)
    report = {"event": "calculated_persistent_material_weak_plane_activation",
        "time_myr": history.time_myr, "face_indices": np.flatnonzero(ready).tolist(),
        "activation_damage": fp.activation_damage,
        "activation_persistence_myr": fp.activation_persistence_myr,
        "stress_scope": STRESS_SCOPE, "sharp_banks_created": False,
        "future_volume_plastic_slip_implemented": False,
        "physical_fault_contact_admission_claimed": False}
    return candidate, report


def evaluate_volume_fault_evolution(before, packet, *, after_time_myr,
        physical_vertices_m, face_stress_pa, face_mean_temperature_k,
        retained_fraction, active_material, surface_temperature_k, ocean_fraction,
        orbit_before, orbit_after, weak_plane_slip_available=False):
    """Pure accepted-interval candidate for stress-driven weakening.

    ``face_stress_pa`` is calculated with retained inherited damage, before
    the new damage solve.  The caller must then re-solve volume mechanics with
    the candidate coefficients and debit exact full-cohort damage energy.
    """
    c = validate_volume_fault_history(before, packet)
    if type(weak_plane_slip_available) is not bool:
        raise ValueError("Volume weak-plane implementation availability must be boolean")
    if np.any(before.fault_active) and not weak_plane_slip_available:
        raise ValueError("Active weak planes require the future volume plastic material law")
    n = c.mesh.cell_count
    after_time = _number(after_time_myr, "endpoint clock")
    dt = after_time-before.time_myr
    if dt <= 0:
        raise ValueError("Fault evolution must advance the accepted common clock")
    for orbit, time in ((orbit_before, before.time_myr), (orbit_after, after_time)):
        if type(orbit) is not TidalOrbitState or orbit.time_myr != time:
            raise ValueError("Fault evolution and tidal orbit clocks must agree")
        validate_tidal_orbit(orbit, c.model.tides_p)
    temperature = _array(face_mean_temperature_k, (n,), "mean solid temperature")
    retained = _array(retained_fraction, (n,), "retained fraction")
    active = _array(active_material, (n,), "active material", boolean=True)
    if (np.any(temperature < 0) or np.any(temperature[active] <= 0)
            or np.any((retained < 0) | (retained > 1))):
        raise ValueError("Fault evolution requires positive temperatures on active material, "
                         "nonnegative unavailable temperatures and retained fractions in [0, 1]")
    surface = _number(surface_temperature_k, "surface temperature")
    ocean = _number(ocean_fraction, "ocean fraction")
    if surface <= 0 or ocean > 1:
        raise ValueError("Fault evolution thermal boundary is outside its physical range")
    stress = _array(face_stress_pa, (n, 3), "current projected face stress")
    top, areas = _physical_top(c, physical_vertices_m)
    damage0 = before.damage*retained
    water = update_water_access(before.water_access, damage0, retained, active,
        surface, ocean, dt, c.model.onset_p)
    p = c.model.p
    strength = p.tensile_strength_pa*(1-(1-c.model.onset_p.wet_strength_fraction)*water)
    tides = replace(c.model.tides_p,
        eccentricity=.5*(orbit_before.eccentricity+orbit_after.eccentricity),
        semimajor_axis_km=.5*(orbit_before.semimajor_axis_km+orbit_after.semimajor_axis_km))
    D = np.array([[1., p.poisson_ratio, 0.], [p.poisson_ratio, 1., 0.],
        [0., 0., (1-p.poisson_ratio)/2]])/(1-p.poisson_ratio**2)
    tide = np.einsum("ab,pfb->pfa", D, tidal_strain_cycle(top, tides))
    degradation = p.residual_stiffness+(1-p.residual_stiffness)*(1-damage0)**2
    tide *= (p.young_modulus_pa*degradation)[None, :, None]
    tide[:, ~active] = 0.
    constitutive_stress = np.where(active[:, None], stress, 0.)
    tensile = np.array([principal_tensile(constitutive_stress+t) for t in tide])
    raw_loading = np.mean(np.maximum(tensile/strength-1, 0)**2, axis=0)/p.damage_timescale_myr
    # A face with no live solid has no constitutive stress/temperature driver.
    # Its zero temperature marks unavailable material, not a cold solid.
    raw_loading[~active] = 0.
    loading = _nonlocal_loading(c, raw_loading, active, areas)
    hot = np.clip((temperature-700)/650, 0, 1)
    healing = 1/p.cold_healing_timescale_myr+hot**4/p.hot_healing_timescale_myr
    healing[~active] = 0.
    rate = loading+healing
    equilibrium = np.divide(loading, rate, out=np.zeros_like(loading), where=rate > 0)
    damage = np.where(active,
        np.clip(equilibrium+(damage0-equilibrium)*np.exp(-rate*dt), 0., 1.), 0.)
    damage_increment = float(np.max(np.abs(damage-damage0)))
    if damage_increment > c.model.mobile_p.max_damage_increment:
        raise VolumeFaultEvolutionStepLimit("volume_fault_damage_step_limit")
    age = before.fault_candidate_age_myr
    if c.model.fault_p.enabled:
        age = np.where(before.membrane_established & (damage >= c.model.fault_p.activation_damage),
            age+dt, 0.)
    age = np.where(active, age, 0.)
    peak = np.maximum(before.peak_tensile_pa, principal_tensile(constitutive_stress))
    tidal_peak = np.max(np.array([principal_tensile(t) for t in tide]), axis=0)/1e6
    first = before.first_fracture_time_myr
    if first is None and np.any(active & (damage >= p.damage_threshold)):
        first = after_time
    after = replace(before, time_myr=after_time, accepted_steps=before.accepted_steps+1,
        damage=damage, water_access=water, peak_tensile_pa=peak,
        first_fracture_time_myr=first,
        weak_duration_myr=np.where(damage >= p.damage_threshold, before.weak_duration_myr+dt, 0.),
        tidal_stress_mpa=tidal_peak, fault_candidate_age_myr=age)
    report = {"format": FORMAT, "before_time_myr": before.time_myr,
        "after_time_myr": after_time, "dt_myr": dt,
        "source_checkpoint_sha256": before.source_checkpoint_sha256,
        "birth_source_identity": before.source_identity, "stress_scope": STRESS_SCOPE,
        "nonlocal_geometry_closure": NONLOCAL_GEOMETRY,
        "nonlocal_area_conservation_relative_residual": float(abs(np.dot(areas, loading-raw_loading*active))
            /max(abs(np.dot(areas, raw_loading*active)), 1.)),
        "source_spherical_filter_equals_current_chord_filter_claimed": False,
        "physical_top_area_km2": areas.tolist(), "raw_loading_per_myr": raw_loading.tolist(),
        "nonlocal_loading_per_myr": loading.tolist(), "healing_per_myr": healing.tolist(),
        "mean_solid_temperature_available": active.tolist(),
        "retained_fraction": retained.tolist(), "strength_pa": strength.tolist(),
        "max_damage_increment": damage_increment,
        "max_damage": float(damage.max()), "max_candidate_age_myr": float(age.max()),
        "activation_gate_checked_at_next_start": True,
        "source_fault_work_preserved": True, "sharp_banks_created": False,
        "future_volume_plastic_slip_implemented": weak_plane_slip_available,
        "physical_fault_contact_admission_claimed": False,
        "mechanical_damage_energy_accounted_by_full_cohort_caller": True,
        "combined_heat_mechanical_energy_closed": False}
    return after, report


def volume_fault_history_to_dict(history):
    if type(history) is not VolumeFaultHistory:
        raise ValueError("Fault history serialization requires typed immutable state")
    body = {f.name: getattr(history, f.name) for f in fields(history)}
    for name in _ARRAYS:
        body[name] = body[name].tolist()
    return {"format": FORMAT, **body}


def volume_fault_history_from_dict(raw, packet):
    required = {f.name for f in fields(VolumeFaultHistory)} | {"format"}
    if not isinstance(raw, dict) or set(raw) != required or raw["format"] != FORMAT:
        raise ValueError("Fault history schema is invalid")
    history = VolumeFaultHistory(**{k: v for k, v in raw.items() if k != "format"})
    validate_volume_fault_history(history, packet)
    return history
