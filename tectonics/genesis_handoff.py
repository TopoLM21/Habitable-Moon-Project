"""Measure whether genesis fragments admit the mature rigid-plate description.

This is an observation contract, not an onset mechanism. Fits never split a
region, invent velocities, or declare a frozen contact run a physical handoff.
Thresholds are explicit numerical screening choices, not calibrated rock laws.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from numbers import Integral, Real

import numpy as np

from .genesis import diagnose
from .genesis_onset_support import face_velocities
from .genesis_seam_diagnostics import seam_connectivity
from .mesh import connected_components


@dataclass(frozen=True)
class HandoffScreenParameters:
    min_region_cells: int = 4
    min_region_area_fraction: float = .01
    min_fit_condition_ratio: float = .001
    max_rigid_residual_fraction: float = .1
    min_relative_speed_km_myr: float = .1
    max_rotation_change_fraction: float = .1
    min_observation_years: float = 50000.

    def validate(self):
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid handoff screen parameter {item.name}")
        if not isinstance(self.min_region_cells, Integral) or self.min_region_cells < 2:
            raise ValueError("At least two cells are needed to determine an Euler vector")
        if any(getattr(self, key) > 1 for key in ("min_region_area_fraction", "min_fit_condition_ratio", "max_rigid_residual_fraction",
                                                  "max_rotation_change_fraction")):
            raise ValueError("Handoff screening fractions must not exceed one")


def component_labels(mesh):
    """Deterministic labels of existing face components; no new boundaries."""
    regions = sorted(connected_components(range(mesh.cell_count), mesh.neighbors), key=min)
    labels = np.empty(mesh.cell_count, dtype=np.int64)
    for label, region in enumerate(regions):
        labels[region] = label
    return labels


def _euler_fit(position, velocity, weights):
    normal = weights.sum()*np.eye(3)-np.einsum("f,fi,fj->ij", weights, position, position)
    rhs = np.sum(weights[:, None]*np.cross(position, velocity), axis=0)
    omega, _, rank, singular = np.linalg.lstsq(normal, rhs, rcond=1e-12)
    return omega, int(rank), float(singular[-1]/singular[0]) if singular[0] else 0.


def fit_rigid_regions(mesh, labels, velocity_km_myr, radius_km):
    """Area-weighted v = omega x (R r) fits, with omega in rad/Myr.

    Subtract the best common rotation before measuring relative speed or fit
    quality: a large rigid rotation of the entire shell must not conceal
    deformation. Single-cell fits are explicitly rank deficient.
    """
    labels, velocity = np.asarray(labels), np.asarray(velocity_km_myr, dtype=float)
    position, area = np.asarray(mesh.centroids), np.asarray(mesh.areas_unit_sphere)
    n = mesh.cell_count
    if (labels.shape != (n,) or labels.dtype.kind not in "iu" or not n
            or not np.array_equal(np.unique(labels), np.arange(int(labels.max())+1))):
        raise ValueError("Region labels must be compact nonnegative integers")
    if (velocity.shape != (n, 3) or not np.isfinite(velocity).all()
            or area.shape != (n,) or not np.isfinite(area).all() or np.any(area <= 0)
            or not np.isfinite(position).all() or position.shape != (n, 3)
            or not np.allclose(np.linalg.norm(position, axis=1), 1., atol=1e-12, rtol=1e-10)):
        raise ValueError("Invalid positions, velocities or face areas")
    if isinstance(radius_km, bool) or not np.isfinite(radius_km) or radius_km <= 0:
        raise ValueError("Positive finite radius is required")
    radial = np.einsum("fi,fi->f", velocity, position)
    if np.max(np.abs(radial)) > 1e-8*max(1., float(np.linalg.norm(velocity, axis=1).max())):
        raise ValueError("Face velocities must be tangential")
    common, _, _ = _euler_fit(position, velocity, area)
    relative = velocity-np.cross(common, position)
    predicted = np.empty_like(velocity)
    regions = []
    omega_vectors = []
    for label in range(int(labels.max())+1):
        mask = labels == label
        r, v, w = position[mask], relative[mask], area[mask]
        relative_omega, rank, condition = _euler_fit(r, v, w)
        omega = common+relative_omega
        fitted = np.cross(relative_omega, r)
        residual = v-fitted
        rms = lambda values: float(np.sqrt(np.sum(w*np.sum(values**2, axis=1))/w.sum()))
        speed, error = rms(v), rms(residual)
        predicted[mask] = np.cross(omega, r)
        omega_vectors.append(omega/radius_km)
        regions.append({"region_id": label, "cell_count": int(mask.sum()),
                        "area_fraction": float(w.sum()/area.sum()), "fit_rank": rank,
                        "fit_condition_ratio": condition,
                        "relative_rms_speed_km_myr": speed,
                        "rigid_residual_km_myr": error,
                        "rigid_residual_fraction": error/max(speed, 1e-8),
                        "omega_rad_per_myr": (omega/radius_km).tolist()})
    return {"regions": regions, "omega_rad_per_myr": np.asarray(omega_vectors),
            "common_omega_rad_per_myr": common/radius_km,
            "predicted_velocity_km_myr": predicted,
            "residual_velocity_km_myr": velocity-predicted,
            "relative_velocity_km_myr": relative}


def _shared_region_vertices(mesh, labels):
    owners = [set() for _ in range(mesh.vertex_count)]
    for face, label in zip(mesh.faces, labels):
        for vertex in face:
            owners[int(vertex)].add(int(label))
    return sum(len(owner) > 1 for owner in owners)


def assess_contact_handoff(model, observations, parameters=None):
    """Assess consecutive snapshots of the same split-bank contact model.

    Return JSON-compatible report and per-cell arrays. The source thermal age
    and elapsed mechanical time stay separate. At least two measured increments
    are required for the short-window stability screen; geological persistence
    is a separate test. Missing composition/remap/energy physics blocks export
    regardless of an excellent rigid-motion fit.
    """
    p = parameters or HandoffScreenParameters()
    p.validate()
    if len(observations) < 2:
        raise ValueError("At least two contact observations are required")
    from .genesis_contact import _validate_contact_state
    for state in observations:
        _validate_contact_state(model, state)
    times = np.asarray([state.elapsed_years for state in observations])
    if np.any(np.diff(times) <= 0):
        raise ValueError("Contact observation times must increase")
    labels = component_labels(model.topology.mesh)
    previous = model.mesh_for(observations[0])
    samples, fits = [], []
    for before, after in zip(observations, observations[1:]):
        current = model.mesh_for(after)
        radius = (model.radius_m+after.displacement_m[-1])/1000.
        velocity = face_velocities(previous.centroids, current.centroids, radius,
                                   (after.elapsed_years-before.elapsed_years)/1e6)
        fit = fit_rigid_regions(current, labels, velocity, radius)
        fits.append(fit)
        samples.append({"start_elapsed_years": float(before.elapsed_years),
                        "end_elapsed_years": float(after.elapsed_years), "regions": fit["regions"]})
        previous = current
    regions = fits[-1]["regions"]
    sufficiently_resolved = all(region["cell_count"] >= p.min_region_cells
        and region["area_fraction"] >= p.min_region_area_fraction and region["fit_rank"] == 3
        and region["fit_condition_ratio"] >= p.min_fit_condition_ratio
        for region in regions)
    rigid = all(region["rigid_residual_fraction"] <= p.max_rigid_residual_fraction
                for sample in samples for region in sample["regions"])
    # Area-weighted relative speed after removing a common rigid rotation.
    speeds = [float(np.sqrt(np.average(np.sum(fit["relative_velocity_km_myr"]**2, axis=1),
                                      weights=model.topology.mesh.areas_unit_sphere))) for fit in fits]
    changes = []
    for a, b in zip(fits, fits[1:]):
        wa = a["omega_rad_per_myr"]-a["common_omega_rad_per_myr"]
        wb = b["omega_rad_per_myr"]-b["common_omega_rad_per_myr"]
        weights = np.asarray([region["area_fraction"] for region in regions])
        denominator = max(float(np.sqrt(np.average(np.sum(wa**2, axis=1), weights=weights))),
                          p.min_relative_speed_km_myr/(model.radius_m/1000))
        changes.append(float(np.sqrt(np.average(np.sum((wa-wb)**2, axis=1), weights=weights)))/denominator)
    stable = bool(changes) and max(changes) <= p.max_rotation_change_fraction
    shared = _shared_region_vertices(model.topology.mesh, labels)
    inter_region = labels[model.topology.seam_faces[:, 0]] != labels[model.topology.seam_faces[:, 1]]
    endpoint_mask = np.repeat(inter_region, 2)
    last = observations[-1]
    connectivity = seam_connectivity(model.topology, last.interface_damage < 1.)
    # Cohesive bridges remain real forces even though nodes have been split.
    weights = model.interface_area_m2[endpoint_mask]
    cohesive_fraction = (float(np.average(last.interface_damage[endpoint_mask] < 1., weights=weights))
                         if weights.size else 0.)
    checks = {
        "multiple_regions": len(regions) >= 2,
        "resolved_regions": sufficiently_resolved,
        "rigid_motion": rigid,
        "relative_motion": min(speeds) >= p.min_relative_speed_km_myr,
        "short_window_stability": stable,
        "independent_vertices": shared == 0,
        "geological_persistence": times[-1]-times[0] >= p.min_observation_years,
        "accepted_solver_states": all(state.stopped_reason is None for state in observations),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    messages = {
        "multiple_regions": "Нет нескольких отделённых областей оболочки.",
        "resolved_regions": "Часть фрагментов слишком мала для надёжного определения движения плиты.",
        "rigid_motion": "Движение внутри фрагментов плохо описывается вращением жёсткой плиты.",
        "relative_motion": "Не обнаружено достаточного относительного движения фрагментов.",
        "short_window_stability": "Устойчивость вращений на последовательных интервалах не подтверждена.",
        "independent_vertices": "Области ещё соединены общими узлами на концах разломов.",
        "geological_persistence": "Короткий контактный опыт не подтверждает длительное существование плит.",
        "accepted_solver_states": "Расчёт достиг ограничения применимости контактной модели.",
    }
    blockers = [{"code": key, "message": messages[key]} for key, value in checks.items() if not value]
    # These are concrete representation gaps of this input format, not
    # requirements to reimplement mature subduction/continents/mantle.
    blockers.extend([
        {"code": "crack_path_localization_missing", "message": "Выделение рёбер по направлению слабых плоскостей создаёт сеточные осколки; геометрия физических разломов ещё не подтверждена."},
        {"code": "frozen_thermal_orbit", "message": "Остывание, орбита и память напряжений в контактном опыте зафиксированы; общей эволюции пока нет."},
        {"code": "crust_inventory_missing", "message": "Нет отдельного учёта состава, массы и возраста коры; твёрдая покрышка не равна коре."},
        {"code": "conservative_remap_missing", "message": "Разорванная материальная сетка ещё не перенесена с сохранением вещества на закрытую сетку зрелого движка."},
        {"code": "thermal_water_transfer_missing", "message": "Нужен согласованный перенос тепла, расплава и пара в более простые тепловую модель и океан зрелого движка."},
        {"code": "mantle_forcing_transfer_missing", "message": "Нагрузка оболочки ещё не преобразована в согласованное начальное поле движения мантии."},
    ])
    thermal = diagnose(model.thermal_state, model.source_model.thermal)
    report = {"format": "genesis-handoff-screen-0.1", "handoff_ready": False,
              "kinematic_screen_passed": bool(all(checks.values())), "checks": checks,
              "parameters": asdict(p), "blockers": blockers,
              "source_time_myr": float(model.source_time_myr),
              "start_elapsed_years": float(times[0]), "end_elapsed_years": float(times[-1]),
              "observation_years": float(times[-1]-times[0]),
              "component_count": len(regions), **connectivity, "shared_vertex_count": shared,
              "cohesive_inter_region_area_fraction": cohesive_fraction,
              "relative_rms_speed_km_myr": speeds, "rotation_change_fraction": changes,
              "samples": samples, "regions": regions,
              "surface_temperature_k": thermal["surface_temperature_k"],
              "mantle_temperature_k": thermal["mantle_temperature_k"],
              "mantle_melt_fraction": thermal["mantle_melt_fraction"],
              "column_mass_kg": float(model.layer_mass_kg.sum()),
              "column_energy_j": float(np.sum(model.layer_mass_kg*model.column_enthalpy)),
              "water_inventory_km3": model.source_model.thermal.water_volume_km3,
              "signed_area_coverage_residual": float((previous.areas_unit_sphere.sum()-4*np.pi)/(4*np.pi)),
              "coverage_note": "Signed area is not an overlap/gap measurement and cannot validate remapping.",
              "scope": "Numerical screening of observed contact motion; no calibrated physical onset threshold or mature checkpoint."}
    arrays = {"region_labels": labels, "vertices": previous.vertices, "faces": previous.faces,
              "centroids": previous.centroids, "areas_unit_sphere": previous.areas_unit_sphere,
              "omega_rad_per_myr": fits[-1]["omega_rad_per_myr"],
              "velocity_km_myr": velocity,
              "residual_velocity_km_myr": fits[-1]["residual_velocity_km_myr"],
              "layer_mass_kg": model.layer_mass_kg.copy(), "column_enthalpy_j_kg": model.column_enthalpy.copy()}
    return report, arrays
