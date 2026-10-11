"""Producer-free audit of prospective weakening of the actual Genesis volume.

The frozen volume auditor supplies independent Cartesian/Duffy/mass-q/time
quadrature.  This extension separately derives the predictor's momentum
equation and current-axis stress, the source-bound water/tidal/nonlocal damage
law, and literal inactive fault history.  Its private mechanics views replace
only the damage input for quadrature; they are never archived or interpreted
as thermal states.  No tectonics producer modules are imported.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np

from analysis.genesis_volume_continuation_validation import independent_audit as volume


FORMAT = "genesis-volume-fault-checkpoint-0.1"
SCOPE = "actual_genesis_source_replayed_connected_volume_damage_until_weak_plane_gate"
HISTORY_FORMAT = "genesis-source-bound-volume-fault-evolution-0.1"
DAMAGE_COUPLING = "thermal_first_predictor_damage_then_original_memory_resolve"
SOURCE_SHA256 = "cd79ee2a5139c4b91abdb1b4714a761c9153cc9f4490f5aa63fa2facba458c89"
NONLOCAL_GEOMETRY = "source_material_angular_conductance_with_current_physical_chord_area"
STRESS_SCOPE = "calculated_volume_face_mean_projected_in_current_top_face_axes"
FAULT_ARRAYS = ("damage", "water_access", "peak_tensile_pa", "weak_duration_myr",
    "tidal_stress_mpa", "fault_active", "plane_normal", "activation_time_myr",
    "fault_candidate_age_myr", "cumulative_shear", "signed_shear",
    "last_shear_increment", "shear_stress_pa", "normal_stress_pa", "shear_strength_pa",
    "friction_work_cell_j", "viscous_work_cell_j")
EVOLVED_ARRAYS = {"damage", "water_access", "peak_tensile_pa", "weak_duration_myr",
                  "tidal_stress_mpa", "fault_candidate_age_myr"}
FROZEN_ARRAYS = set(FAULT_ARRAYS)-EVOLVED_ARRAYS


def formation_fault_data(packet):
    """Bind literal source bytes before any mechanics-only damage adaptation."""
    binding = volume.source_dto_binding(packet)
    if binding["checkpoint_sha256"] != SOURCE_SHA256:
        raise ValueError("Fault audit requires the pinned actual molten-formation source")
    encoded = base64.b64decode(packet["source_packet"]["source"]["checkpoint_base64"], validate=True)
    with np.load(io.BytesIO(encoded), allow_pickle=False) as source:
        metadata = json.loads(str(source["metadata"]))
        arrays = {name: np.array(source[name], copy=True) for name in FAULT_ARRAYS}
    return metadata, arrays, binding


def inherited_fault_binding(packet):
    """All inactive plane/slip/work fields stay exactly inherited, never zeroed."""
    metadata, arrays, source_binding = formation_fault_data(packet)
    state, source = packet["runtime_state"], packet["source_packet"]
    history = state["fault_history"]
    count = state["mechanics"]["state"]["accepted_steps"]
    expected_fields = set(FAULT_ARRAYS) | {"format", "source_checkpoint_sha256", "source_identity",
        "initial_time_myr", "time_myr", "accepted_steps", "membrane_established",
        "first_fracture_time_myr", "friction_work_j", "viscous_fault_work_j"}
    if (set(history) != expected_fields or history["format"] != HISTORY_FORMAT
            or state["damage_coupling"] != DAMAGE_COUPLING
            or history["source_checkpoint_sha256"] != SOURCE_SHA256
            or history["source_identity"] != source["birth_source_identity"]
            or history["initial_time_myr"] != source["time_myr"]
            or history["time_myr"] != state["geometry"]["time_myr"]
            or history["accepted_steps"] != count
            or type(count) is not int or count < 0
            or type(history["membrane_established"]) is not bool
            or history["membrane_established"] != metadata["state"]["membrane_established"]):
        raise ValueError("Fault history lost literal source/schema/common-clock binding")
    volume.endpoint_clock_binding(state, history["time_myr"])
    thermal, mechanics = state["thermal"], state["mechanics"]["state"]
    if (thermal["source_checkpoint_sha256"] != SOURCE_SHA256
            or thermal["source_identity"] != source["birth_source_identity"]
            or thermal["initial_time_myr"] != source["time_myr"]
            or thermal["source_accepted_intervals"] != len(source["bulk_birth_history"]["intervals"])
            or thermal["accepted_steps"] != count
            or mechanics["source_identity"] != source["birth_source_identity"]):
        raise ValueError("Fault continuation replaced its runtime thermal/mechanical source context")
    volume._assert_close(thermal["layer_mass_kg"], source["current_columns"]["layer_mass_kg"],
        "unchanged literal column mass", rtol=0, atol=0)
    volume._assert_close(state["thermal"]["damage"], arrays["damage"],
        "unchanged literal thermal source damage", rtol=0, atol=0)
    for name in FAULT_ARRAYS:
        actual = np.asarray(history[name])
        if actual.shape != arrays[name].shape or not np.isfinite(actual).all():
            raise ValueError("Fault history changed source face coverage: "+name)
        if name == "fault_active" and actual.dtype.kind != "b":
            raise ValueError("Fault activity must remain literal boolean data")
        if name in FROZEN_ARRAYS or count == 0:
            if not np.array_equal(actual, arrays[name]):
                raise ValueError("Inactive fault history was reset or forged: "+name)
    if np.any(history["fault_active"]):
        raise ValueError("The current volume branch cannot accept active slip or physical banks")
    for name in ("friction_work_j", "viscous_fault_work_j"):
        if history[name] != metadata["state"][name]:
            raise ValueError("Inherited fault work changed without a resolved slip law")
    first = history["first_fracture_time_myr"]
    if ((count == 0 and first != metadata["state"]["first_fracture_time_myr"])
            or (first is not None and (type(first) not in (int, float)
                or not math.isfinite(first) or not 0 <= first <= history["time_myr"]))):
        raise ValueError("Fault diffuse fracture time lost its inherited/common-clock binding")
    for name in ("damage", "water_access"):
        values = volume._array(history[name])
        if np.any((values < 0) | (values > 1)):
            raise ValueError("Fault fraction is outside [0,1]: "+name)
    for name in ("peak_tensile_pa", "weak_duration_myr", "tidal_stress_mpa", "fault_candidate_age_myr"):
        if np.any(volume._array(history[name]) < 0):
            raise ValueError("Fault persistence/loading is negative: "+name)
    birth = state["history"]
    original = source["bulk_birth_history"]
    for key in set(original)-{"cohorts", "intervals", "front_fraction", "time_myr", "sha256"}:
        if volume._canonical(birth[key]) != volume._canonical(original[key]):
            raise ValueError("Fault continuation replaced its Genesis birth context: "+key)
    if volume._canonical(birth["intervals"][:len(original["intervals"])]) != volume._canonical(original["intervals"]):
        raise ValueError("Fault continuation replaced its accepted Genesis birth intervals")
    current = {int(c["id"]): c for c in birth["cohorts"]}
    for cohort in original["cohorts"]:
        live = current.get(int(cohort["id"]))
        if live is None or any(volume._canonical(live[key]) != volume._canonical(value)
                for key, value in cohort.items() if key != "live_q_hi"):
            raise ValueError("Fault continuation replaced a literal Genesis material cohort")
        if not float(cohort["q_lo"]) <= volume._high(live) <= volume._high(cohort):
            raise ValueError("Inherited source cohort gained replacement historical material")
    steps = state["accepted_step_sizes_myr"]
    if (len(steps) != count or any(type(dt) not in (int, float) or not math.isfinite(dt) or dt <= 0 for dt in steps)):
        raise ValueError("Fault continuation has an invalid accepted-step journal")
    return {**source_binding, "literal_inactive_plane_slip_and_work_preserved": True,
        "source_active_plane_count": int(np.count_nonzero(arrays["fault_active"])),
        "thermal_source_damage_unchanged": True, "original_birth_prefix_preserved": True}


def physical_face_geometry(vertices, faces):
    xyz = volume._array(vertices)[np.asarray(faces, int)]
    edge = xyz[:, 1]-xyz[:, 0]
    cross = np.cross(edge, xyz[:, 2]-xyz[:, 0])
    lengths, twice_area = np.linalg.norm(edge, axis=1), np.linalg.norm(cross, axis=1)
    if np.any(lengths <= 0) or np.any(twice_area <= 0):
        raise ValueError("Independent fault driver has a degenerate physical face")
    x, z = edge/lengths[:, None], cross/twice_area[:, None]
    if np.any(np.einsum("fi,fi->f", cross, xyz.mean(axis=1)) <= 0):
        raise ValueError("Independent fault driver physical face inverted")
    return np.stack((x, np.cross(z, x), z), axis=2), twice_area*.5/1e6


def retained_fraction(before_cohorts, after_cohorts, column_mass):
    old = {int(c["id"]): c for c in before_cohorts}
    retained, total = np.zeros(len(column_mass)), np.zeros(len(column_mass))
    for cohort in after_cohorts:
        face, cid = int(cohort["face_index"]), int(cohort["id"])
        lo, hi = float(cohort["q_lo"]), volume._high(cohort)
        total[face] += column_mass[face]*(hi-lo)
        if cid in old:
            retained[face] += column_mass[face]*max(0., min(hi, volume._high(old[cid]))-lo)
    if np.any(retained > total*(1+1e-14)):
        raise ValueError("Independent retained source material exceeds endpoint material")
    return np.divide(retained, total, out=np.zeros_like(total), where=total > 0)


def face_stress_from_coefficients(vertices, faces, column_mass, density, cohorts,
                                 strain0, strain1, temperatures, damage, current_vertices, shell):
    """Separate volume integration in source axes, projection in current axes."""
    vertices, faces = volume._array(vertices), np.asarray(faces, int)
    e0 = volume._array(strain0, (len(cohorts), 3, 6))
    e1 = volume._array(strain1, e0.shape)
    temperature = volume._array(temperatures, (len(cohorts),))
    current_frames, area = physical_face_geometry(current_vertices, faces)
    world_sum, volume_sum, temperature_sum = np.zeros((len(faces), 3, 3)), np.zeros(len(faces)), np.zeros(len(faces))
    N, wa = volume.angular_rule()
    interpolation = N@np.linalg.inv(volume.SOURCE_ANGULAR_POINTS)
    for i, cohort in enumerate(cohorts):
        face = int(cohort["face_index"])
        triangle = vertices[faces[face]]
        frame, _ = physical_face_geometry(vertices, faces[face][None, :])
        frame = frame[0]
        q, wr = volume.radial_rule(float(cohort["q_lo"]), volume._high(cohort))
        lam, _ = volume.conical_scales(triangle, column_mass[face], density, q)
        average = np.einsum("p,r,pri->i", wa, wr,
            (interpolation@e0[i])[:, None, :]+(interpolation@e1[i])[:, None, :]/lam[None, :, None])
        C = volume.isotropic_matrix(shell["young_modulus_pa"], shell["poisson_ratio"], damage[face], shell["residual_stiffness"])
        stress = C@average
        tensor = np.array(((stress[0], stress[3], stress[4]), (stress[3], stress[1], stress[5]),
                           (stress[4], stress[5], stress[2])))
        v = column_mass[face]*(volume._high(cohort)-float(cohort["q_lo"]))/density
        world_sum[face] += v*(frame@tensor@frame.T)
        volume_sum[face] += v
        temperature_sum[face] += v*temperature[i]
    active = volume_sum > 0
    world_sum[active] /= volume_sum[active, None, None]
    temperature_sum[active] /= volume_sum[active]
    local = current_frames.transpose(0, 2, 1)@world_sum@current_frames
    stress = np.stack((local[:, 0, 0], local[:, 1, 1], local[:, 0, 1]), axis=1)
    return {"stress_pa": stress, "mean_temperature_k": temperature_sum,
            "active_material": active, "physical_top_area_km2": area}


def ocean_fraction(surface_temperature, thermal):
    area = 4*math.pi*(1000*thermal["radius_km"])**2
    water = thermal["water_volume_km3"]*1e9*thermal["water_density_kg_m3"]/area
    t = float(surface_temperature)
    if t >= 647.096:
        return 0.
    if t < 273.16:
        pressure = 611.657*math.exp(5420*(1/273.16-1/max(t, 1.)))
    else:
        theta = 1-t/647.096
        pressure = 22.064e6*math.exp(647.096/t*sum(a*theta**b for a, b in
            ((-7.85951783, 1), (1.84408259, 1.5), (-11.7866497, 3),
             (22.6807411, 3.5), (-15.9618719, 4), (1.80122502, 7.5))))
    vapor = min(water, pressure/thermal["surface_gravity_m_s2"])
    if water*thermal["surface_gravity_m_s2"] > 22.064e6:
        x = min(1., max(0., (647.096-t)/thermal["critical_transition_width_k"]))
        vapor = water+x*x*(3-2*x)*(vapor-water)
    return (water-vapor)/water if water else 0.


def current_tidal_strain(vertices, faces, tidal, orbit_before, orbit_after):
    """Derive quadratic-potential spherical Hessians in current physical axes."""
    frames, _ = physical_face_geometry(vertices, faces)
    n = frames[:, :, 2]
    tangent = frames[:, :, :2]
    eccentricity = .5*(orbit_before["eccentricity"]+orbit_after["eccentricity"])
    axis = .5*(orbit_before["semimajor_axis_km"]+orbit_after["semimajor_axis_km"])*1000
    count = int(tidal["phase_samples"])
    if not tidal["enabled"] or eccentricity == 0:
        return np.zeros((count, len(faces), 3))
    strength = 6.67430e-11*tidal["primary_mass_kg"]*(tidal["satellite_radius_km"]*1000)/(tidal["surface_gravity_m_s2"]*axis**3)
    harmonics = []
    for potential in (np.diag((3., -1.5, -1.5)), np.array(((0., 3., 0.), (3., 0., 0.), (0., 0., 0.)))):
        scalar = np.einsum("fi,ij,fj->f", n, potential, n)
        projection = tangent.transpose(0, 2, 1)@potential@tangent
        tensor = strength*eccentricity*(2*tidal["love_l2"]*projection
            +(tidal["love_h2"]-2*tidal["love_l2"])*scalar[:, None, None]*np.eye(2))
        harmonics.append(np.stack((tensor[:, 0, 0], tensor[:, 1, 1], 2*tensor[:, 0, 1]), axis=1))
    phase = 2*math.pi*np.arange(count)/count
    return np.cos(phase)[:, None, None]*harmonics[0]+np.sin(phase)[:, None, None]*harmonics[1]


def nonlocal_loading(source_vertices, faces, current_area_km2, active, raw_loading, length_km):
    """Dense independent graph solve using source material angular conductances."""
    source_vertices, faces = volume._array(source_vertices), np.asarray(faces, int)
    raw, area = volume._array(raw_loading), volume._array(current_area_km2)
    active = np.asarray(active, bool)
    if raw.shape != area.shape or active.shape != raw.shape or np.any(raw < 0) or np.any(area <= 0):
        raise ValueError("Independent positive nonlocal loading inputs are invalid")
    centers = source_vertices[faces].sum(axis=1)
    centers /= np.linalg.norm(centers, axis=1)[:, None]
    edge_map = {}
    for face, nodes in enumerate(faces):
        for i in range(3):
            edge_map.setdefault(tuple(sorted((int(nodes[i]), int(nodes[(i+1)%3])))), []).append(face)
    arc = lambda a, b: math.atan2(float(np.linalg.norm(np.cross(a, b))), float(np.dot(a, b)))
    indices = np.flatnonzero(active)
    result = np.zeros(len(faces))
    if not len(indices):
        return result
    position = {int(face): i for i, face in enumerate(indices)}
    matrix = np.diag(area[indices])
    for (a, b), neighboring in edge_map.items():
        if len(neighboring) != 2:
            raise ValueError("Independent source material topology is not a closed two-face edge graph")
        left, right = neighboring
        if active[left] and active[right]:
            conductance = length_km**2*arc(source_vertices[a], source_vertices[b])/arc(centers[left], centers[right])
            i, j = position[left], position[right]
            matrix[i, i] += conductance
            matrix[j, j] += conductance
            matrix[i, j] -= conductance
            matrix[j, i] -= conductance
    result[indices] = np.linalg.solve(matrix, area[indices]*raw[indices])
    if not np.isfinite(result).all() or np.any(result < -2e-14*max(float(raw.max()), 1.)):
        raise ValueError("Independent nonlocal loading lost positivity")
    residual = abs(np.dot(area, result-raw*active))/max(abs(np.dot(area, raw*active)), 1.)
    if residual > 2e-12:
        raise ValueError("Independent current-area Helmholtz conservation failed")
    return result


def tensile_principal(stress):
    stress = volume._array(stress)
    return np.maximum(0., .5*(stress[..., 0]+stress[..., 1])
        +np.hypot(.5*(stress[..., 0]-stress[..., 1]), stress[..., 2]))


def independent_damage_law(before, driver, source, orbit_before, orbit_after, after_time):
    parameters = source["source"]["parameters"]
    shell, onset, tides, faults = (parameters[name] for name in ("shell", "onset", "tides", "faults"))
    dt = after_time-before["time_myr"]
    if dt <= 0:
        raise ValueError("Independent fault law requires a positive common-clock interval")
    retained = volume._array(driver["retained_fraction"])
    active = np.asarray(driver["active_material"], bool)
    temperature = volume._array(driver["mean_temperature_k"])
    if (temperature.shape != active.shape or np.any(temperature < 0)
            or np.any(temperature[active] <= 0) or retained.shape != active.shape
            or np.any((retained < 0) | (retained > 1))):
        raise ValueError("Independent fault law requires actual temperatures on active material")
    damage0 = volume._array(before["damage"])*retained
    inherited_water = volume._array(before["water_access"])*retained
    if onset["water_weakening"] and driver["surface_temperature_k"] < onset["water_access_temperature_k"] and driver["ocean_fraction"] > 1e-6:
        permeability = onset["intact_permeability_fraction"]+(1-onset["intact_permeability_fraction"])*damage0**2
        wetting = permeability/onset["access_timescale_myr"]*min(1., driver["ocean_fraction"]/.01)
        water = 1-(1-inherited_water)*np.exp(-wetting*dt)
    else:
        water = inherited_water*np.exp(-dt/onset["loss_timescale_myr"])
    water = np.where(active, np.clip(water, 0., 1.), 0.)
    strength = shell["tensile_strength_pa"]*(1-(1-onset["wet_strength_fraction"])*water)
    faces = np.asarray(source["current_geometry"]["faces"], int)
    _, areas = physical_face_geometry(driver["physical_vertices_m"], faces)
    cycle = current_tidal_strain(driver["physical_vertices_m"], faces, tides, orbit_before, orbit_after)
    nu = shell["poisson_ratio"]
    C = shell["young_modulus_pa"]*np.array(((1., nu, 0.), (nu, 1., 0.), (0., 0., (1-nu)/2)))/(1-nu**2)
    degradation = shell["residual_stiffness"]+(1-shell["residual_stiffness"])*(1-damage0)**2
    tide = np.einsum("ab,pfb->pfa", C, cycle)*degradation[None, :, None]
    tide[:, ~active] = 0.
    constitutive_stress = np.where(active[:, None], volume._array(driver["stress_pa"]), 0.)
    raw = np.mean(np.maximum(tensile_principal(constitutive_stress[None, :, :]+tide)/strength-1., 0.)**2, axis=0)/shell["damage_timescale_myr"]
    raw[~active] = 0.
    loading = nonlocal_loading(source["current_geometry"]["vertices_unit_xyz"], faces,
        areas, active, raw, onset["regularization_km"])
    hot = np.clip((temperature-700)/650, 0., 1.)
    healing = 1/shell["cold_healing_timescale_myr"]+hot**4/shell["hot_healing_timescale_myr"]
    healing[~active] = 0.
    rate = loading+healing
    equilibrium = np.divide(loading, rate, out=np.zeros_like(loading), where=rate > 0)
    damage = np.where(active, np.clip(equilibrium+(damage0-equilibrium)*np.exp(-rate*dt), 0., 1.), 0.)
    age = volume._array(before["fault_candidate_age_myr"])
    if faults["enabled"]:
        age = np.where(before["membrane_established"] & (damage >= faults["activation_damage"]), age+dt, 0.)
    age = np.where(active, age, 0.)
    expected = dict(damage=damage, water_access=water,
        peak_tensile_pa=np.maximum(before["peak_tensile_pa"], tensile_principal(constitutive_stress)),
        weak_duration_myr=np.where(damage >= shell["damage_threshold"], volume._array(before["weak_duration_myr"])+dt, 0.),
        tidal_stress_mpa=tensile_principal(tide).max(axis=0)/1e6, fault_candidate_age_myr=age)
    if np.max(np.abs(damage-damage0)) > parameters["mobile"]["max_damage_increment"]:
        raise ValueError("Independent damage update exceeds the actual source accepted-step limit")
    return expected, dict(physical_top_area_km2=areas, raw_loading_per_myr=raw,
        nonlocal_loading_per_myr=loading, healing_per_myr=healing, retained_fraction=retained,
        strength_pa=strength, max_damage_increment=float(np.max(np.abs(damage-damage0))),
        max_damage=float(damage.max()), max_candidate_age_myr=float(age.max()),
        nonlocal_area_conservation_relative_residual=float(abs(np.dot(areas, loading-raw*active))
            /max(abs(np.dot(areas, raw*active)), 1.)))


def predictor_binding(before_packet, after_packet):
    """Reconstruct endpoint predictor memory AND independent force equation."""
    before, after = before_packet["runtime_state"], after_packet["runtime_state"]
    source = after_packet["source_packet"]
    driver = after["last_report"]["fault_evolution"]["predictor_driver"]
    geometry = after["geometry"]
    X, faces = volume._array(geometry["vertices_m"]), np.asarray(geometry["faces"], int)
    mass, rho = volume._array(geometry["column_mass_kg"]), float(geometry["density_kg_m3"])
    bcohorts, acohorts = before["geometry"]["cohorts"], geometry["cohorts"]
    bm = {int(c["id"]): c for c in before["mechanics"]["state"]["cohorts"]}
    bmap = {int(c["id"]): c for c in bcohorts}
    amap = {int(c["id"]): c for c in acohorts}
    born = [c for c in acohorts if int(c["id"]) not in bmap]
    removed = [{**c, "q_lo": c["q_lo"] if int(c["id"]) not in amap else volume._high(amap[int(c["id"])])}
        for c in bcohorts if int(c["id"]) not in amap or volume._high(amap[int(c["id"])]) < volume._high(c)]
    shell, thermal = (source["source"]["parameters"][name] for name in ("shell", "thermal"))
    q0, q1, v0 = (volume._array(value) for value in (before["displacement_m"], driver["displacement_m"], before["velocity_m_s"]))
    if q1.shape != q0.shape:
        raise ValueError("Independent predictor displacement shape changed")
    dq = q1-q0
    volume._assert_close(driver["physical_vertices_m"], X+q1.reshape(-1, 6)[:, :3],
        "predictor actual physical top", rtol=0, atol=0)
    retained = retained_fraction(bcohorts, acohorts, mass)
    volume._assert_close(driver["retained_fraction"], retained, "predictor retained material", rtol=2e-13, atol=2e-15)
    damage0 = volume._array(before["fault_history"]["damage"])*retained
    volume._assert_close(driver["damage"], damage0, "predictor retained inherited damage", rtol=0, atol=0)
    dt = (after["geometry"]["time_myr"]-before["geometry"]["time_myr"])*365.25*86400*1e6
    tm0, ts0 = volume.global_temperatures(before["thermal"]["global_state"], thermal)
    tm1, ts1 = volume.global_temperatures(after["thermal"]["global_state"], thermal)
    volume._assert_close(driver["surface_temperature_k"], ts1, "predictor actual surface temperature", rtol=2e-12, atol=1e-8)
    volume._assert_close(driver["ocean_fraction"], ocean_fraction(ts1, thermal), "predictor actual condensed water", rtol=2e-10, atol=1e-12)
    column0, column1 = (volume.conductive_temperatures(state["thermal"], thermal) for state in (before, after))
    layer_mass = volume._array(after["thermal"]["layer_mass_kg"])
    expected0, expected1, temperatures = [], [], []
    average_force = np.zeros(len(q0))
    mechanical_work = 0.
    for c in acohorts:
        cid, face = int(c["id"]), int(c["face_index"])
        old = bm.get(cid)
        old0, old1 = ((np.zeros((3, 6)), np.zeros((3, 6))) if old is None
            else (volume._array(old["strain0"]), volume._array(old["strain1"])))
        lo, hi = float(c["q_lo"]), volume._high(c)
        old_temperature = thermal["solidus_k"] if old is None else volume.profile_mean(layer_mass[face], column0[face], ts0, tm0, lo, hi)
        temperature = volume.profile_mean(layer_mass[face], column1[face], ts1, tm1, lo, hi)
        temperatures.append(temperature)
        thermal0 = np.zeros((3, 6)); thermal0[:, :3] = shell["linear_expansion_per_k"]*(temperature-old_temperature)
        viscosity = np.clip(shell["viscosity_reference_pa_s"]*np.exp(np.clip(
            shell["activation_energy_j_mol"]/8.314462618*(1/temperature-1/shell["viscosity_reference_temperature_k"]), -60, 60)), shell["viscosity_min_pa_s"], shell["viscosity_max_pa_s"])
        ratio = 0. if old is None else dt*shell["young_modulus_pa"]/viscosity
        beta = 1. if ratio == 0 else -math.expm1(-ratio)/ratio
        frame, _ = physical_face_geometry(X, faces[face][None, :])
        frame = frame[0]
        dofs = (6*faces[face, :, None]+np.arange(6)).ravel()
        B0, B1 = volume.independent_strain_operators(X[faces[face]], frame, volume.SOURCE_ANGULAR_POINTS)
        expected0.append(math.exp(-ratio)*old0+beta*(B0@dq[dofs]-thermal0))
        expected1.append(math.exp(-ratio)*old1+beta*(B1@dq[dofs]))
        C = volume.isotropic_matrix(shell["young_modulus_pa"], shell["poisson_ratio"], damage0[face], shell["residual_stiffness"])
        result = volume.quadrature_cohort(X[faces[face]], frame, mass[face], rho, c, dq[dofs], old0, old1, thermal0, ratio, C)
        np.add.at(average_force, dofs, result["average_force_n"])
        mechanical_work += result["mechanical_work_j"]
    volume._assert_close(driver["elastic_strain0"], expected0, "predictor constant coefficient memory", rtol=2e-9, atol=2e-12)
    volume._assert_close(driver["elastic_strain1"], expected1, "predictor radial coefficient memory", rtol=2e-9, atol=2e-12)
    response = face_stress_from_coefficients(X, faces, mass, rho, acohorts, expected0, expected1,
        temperatures, damage0, driver["physical_vertices_m"], shell)
    for name in ("stress_pa", "mean_temperature_k"):
        volume._assert_close(driver[name], response[name], "calculated current-axis predictor "+name, rtol=3e-10, atol=1e-5 if name == "stress_pa" else 1e-8)
    if (np.asarray(driver["active_material"]).dtype.kind != "b"
            or not np.array_equal(driver["active_material"], response["active_material"])):
        raise ValueError("Independent predictor active material mask differs")
    inputs = after["last_report"]["audit_input"]
    traction = volume.prescribed_traction(X, faces, mass, rho, acohorts, source["current_geometry"]["radius_km"], shell)
    D, F = volume.basal_drag_and_force(X, faces, mass, rho, acohorts, inputs["basal_drag_pa_s_m"], traction[faces])
    M0, M1, Min, Mout = (volume.mass_matrix(X, faces, mass, rho, rows) for rows in (bcohorts, acohorts, born, removed))
    v1 = 2*dq/dt-v0
    balance = volume.dynamic_balance(M0, M1, Min, Mout, D, F, average_force, q0, q1, v0, v1, dt, mechanical_work)
    return {"full_predictor_momentum_and_work": balance,
        "predictor_live_cohort_count": len(acohorts), "stress_source": "independent_full_cohort_quadrature_then_current_axis_projection",
        "predictor_is_private_trial": True, "predictor_substituted_for_final_accepted_state": False}


def mechanical_view(packet):
    """Private quadrature argument only; raw thermal source remains immutable."""
    result = copy.deepcopy(packet)
    result["runtime_state"]["thermal"]["damage"] = copy.deepcopy(packet["runtime_state"]["fault_history"]["damage"])
    return result


def declared_drag_binding(before, after):
    """Reported loading cannot silently replace the continuation parameter."""
    if before["contact_parameters_json"] != after["contact_parameters_json"]:
        raise ValueError("Independent fault interval changed its actual contact parameters")
    contact = json.loads(after["contact_parameters_json"])
    drag = contact["basal_drag_pa_s_m"]
    if type(drag) not in (int, float) or not math.isfinite(drag) or drag < 0:
        raise ValueError("Independent declared physical basal drag is invalid")
    volume._assert_close(after["last_report"]["audit_input"]["basal_drag_pa_s_m"], drag,
        "actual declared basal drag", rtol=0, atol=0)


def final_stress_binding(packet):
    """Peak memory must also include the final accepted post-damage response."""
    after = packet["runtime_state"]
    geometry = after["geometry"]
    X = volume._array(geometry["vertices_m"])
    q = volume._array(after["displacement_m"]).reshape(-1, 6)
    memories = {int(c["id"]): c for c in after["mechanics"]["state"]["cohorts"]}
    ordered = [memories[int(c["id"])] for c in geometry["cohorts"]]
    response = face_stress_from_coefficients(X, geometry["faces"], volume._array(geometry["column_mass_kg"]),
        geometry["density_kg_m3"], geometry["cohorts"], [c["strain0"] for c in ordered],
        [c["strain1"] for c in ordered], [c["temperature_k"] for c in ordered],
        volume._array(after["fault_history"]["damage"]), X+q[:, :3],
        packet["source_packet"]["source"]["parameters"]["shell"])
    tensile = tensile_principal(response["stress_pa"])
    volume._assert_close(after["last_report"]["fault_evolution"]["final_face_tensile_pa"], tensile,
        "final accepted current-axis tensile stress", rtol=3e-10, atol=1e-5)
    return tensile


def audit_fault_interval(before_packet, after_packet):
    before, after = before_packet["runtime_state"], after_packet["runtime_state"]
    if volume._canonical(before_packet["source_packet"]) != volume._canonical(after_packet["source_packet"]):
        raise ValueError("Independent fault interval changed the complete actual formation source")
    inherited_fault_binding(before_packet)
    inherited_fault_binding(after_packet)
    declared_drag_binding(before, after)
    expected_count = before["fault_history"]["accepted_steps"]+1
    if after["fault_history"]["accepted_steps"] != expected_count:
        raise ValueError("Independent fault interval skips an accepted mechanical step")
    if after["accepted_step_sizes_myr"][:-1] != before["accepted_step_sizes_myr"]:
        raise ValueError("Independent fault continuation changed the accepted step journal")
    if after["last_report"]["requested_dt_myr"] != after["accepted_step_sizes_myr"][-1]:
        raise ValueError("Independent accepted-step request is not bound to its journal")
    for key in ("static_equilibrium_claimed", "old_C5_to_C1_admission_claimed", "full_historical_3d_recovery_claimed"):
        if after["last_report"][key] is not False:
            raise ValueError("Independent volume interval makes an unsupported physical claim: "+key)
    report = after["last_report"]["fault_evolution"]
    availability = np.asarray(report["mean_solid_temperature_available"])
    if (availability.dtype.kind != "b"
            or not np.array_equal(availability, report["predictor_driver"]["active_material"])):
        raise ValueError("Independent solid temperature availability differs from live material")
    if (report["format"] != HISTORY_FORMAT or report["damage_coupling"] != DAMAGE_COUPLING
            or report["stress_scope"] != STRESS_SCOPE or report["nonlocal_geometry_closure"] != NONLOCAL_GEOMETRY
            or report["source_checkpoint_sha256"] != SOURCE_SHA256
            or report["birth_source_identity"] != after_packet["source_packet"]["birth_source_identity"]):
        raise ValueError("Independent fault evolution changed its constitutive/geometry/source semantics")
    for key in ("source_spherical_filter_equals_current_chord_filter_claimed", "sharp_banks_created",
            "future_volume_plastic_slip_implemented", "physical_fault_contact_admission_claimed",
            "combined_heat_mechanical_energy_closed", "combined_heat_mechanical_reservoir_closed", "physical_banks_admitted"):
        if report[key] is not False:
            raise ValueError("Independent fault report makes an unsupported physical claim: "+key)
    for key in ("activation_gate_checked_at_next_start", "source_fault_work_preserved", "mechanical_damage_energy_accounted_by_full_cohort_caller"):
        if report[key] is not True:
            raise ValueError("Independent fault report removed a required closure declaration: "+key)
    source = after_packet["source_packet"]
    fault_parameters = source["source"]["parameters"]["faults"]
    ready = (np.asarray(before["fault_history"]["damage"]) >= fault_parameters["activation_damage"]) & (np.asarray(before["fault_history"]["fault_candidate_age_myr"]) >= fault_parameters["activation_persistence_myr"])
    if fault_parameters["enabled"] and before["fault_history"]["membrane_established"] and np.any(ready):
        raise ValueError("Independent fault branch advanced across a ready weak-plane activation gate")
    predictor = predictor_binding(before_packet, after_packet)
    expected, law = independent_damage_law(before["fault_history"], report["predictor_driver"], source,
        before["thermal"]["orbit"], after["thermal"]["orbit"], after["fault_history"]["time_myr"])
    final_tensile = final_stress_binding(after_packet)
    expected["peak_tensile_pa"] = np.maximum(expected["peak_tensile_pa"], final_tensile)
    for name, value in expected.items():
        volume._assert_close(after["fault_history"][name], value, "source-bound fault law "+name, rtol=3e-10, atol=1e-12 if name != "peak_tensile_pa" else 1e-5)
    first = before["fault_history"]["first_fracture_time_myr"]
    if first is None and np.any(np.asarray(report["predictor_driver"]["active_material"], bool)
            & (expected["damage"] >= source["source"]["parameters"]["shell"]["damage_threshold"])):
        first = after["fault_history"]["time_myr"]
    if after["fault_history"]["first_fracture_time_myr"] != first:
        raise ValueError("Independent diffuse fracture event clock differs")
    for name, value in law.items():
        volume._assert_close(report[name], value, "fault report independent "+name, rtol=3e-10, atol=1e-12)
    if (report["before_time_myr"] != before["fault_history"]["time_myr"]
            or report["after_time_myr"] != after["fault_history"]["time_myr"]
            or report["dt_myr"] != report["after_time_myr"]-report["before_time_myr"]):
        raise ValueError("Independent fault law report clock differs from accepted endpoints")
    volume_result = volume.audit_interval(mechanical_view(before_packet), mechanical_view(after_packet))
    damage_energy = volume_result["quadrature"]["damage_energy_change_j"]
    for name in ("mechanical_work_j", "thermal_work_j", "viscous_dissipation_j", "damage_energy_change_j", "remelt_energy_removed_j"):
        delta = after["mechanics"]["state"][name]-before["mechanics"]["state"][name]
        volume._assert_close(delta, volume_result["quadrature"][name], "cumulative full-cohort "+name, rtol=3e-7, atol=8192.)
    volume._assert_close(report["damage_energy_change_j"], damage_energy, "reported signed full-cohort damage energy", rtol=3e-7, atol=8192.)
    return {**volume_result, "fault_evolution": {"independently_recomputed_water_damage_persistence_and_tides": True,
        "nonlocal_geometry": NONLOCAL_GEOMETRY, "maximum_damage": law["max_damage"],
        "maximum_candidate_age_myr": law["max_candidate_age_myr"], "signed_damage_energy_change_j": damage_energy,
        "peak_includes_both_predictor_and_final_accepted_stress": True,
        "literal_inactive_fault_history_preserved": True, "actual_new_active_plane_count": 0,
        "physical_banks_or_underthrust_admitted": False}, "predictor": predictor}


def audit_directory(directory):
    directory = Path(directory)
    snapshots, bindings = [], []
    for path in sorted(directory.glob("*.json")):
        raw = json.loads(path.read_text())
        if raw.get("format") != FORMAT:
            continue
        if (set(raw) != {"format", "scope", "source_packet", "source_checkpoint_sha256", "runtime_state", "checkpoint_sha256"}
                or raw["scope"] != SCOPE):
            raise ValueError("Independent fault checkpoint schema or physical scope changed")
        body = {key: value for key, value in raw.items() if key != "checkpoint_sha256"}
        if hashlib.sha256(volume._canonical(body)).hexdigest() != raw["checkpoint_sha256"]:
            raise ValueError("Independent fault checkpoint seal mismatch")
        bindings.append({"name": path.name, **inherited_fault_binding(raw)})
        snapshots.append((raw["runtime_state"]["mechanics"]["state"]["accepted_steps"], path, raw))
    if not snapshots:
        raise ValueError("No actual Genesis volume fault checkpoints in the directory")
    grouped = {}
    for count, path, raw in snapshots:
        grouped.setdefault(count, []).append((path, raw))
    if sorted(grouped) != list(range(max(grouped)+1)):
        raise ValueError("Independent fault audit requires every accepted endpoint including initial")
    if max(grouped) == 0:
        raise ValueError("Independent fault audit requires an actually advanced interval")
    restart_pairs = []
    for count, rows in grouped.items():
        for path, raw in rows[1:]:
            if volume._canonical(rows[0][1]) != volume._canonical(raw):
                raise ValueError("Complete direct/restarted fault JSON differs")
            restart_pairs.append(dict(accepted_steps=count, first=rows[0][0].name,
                second=path.name, complete_json_equal=True))
    initial = volume.audit_initial(grouped[0][0][1])
    intervals = [audit_fault_interval(grouped[count-1][0][1], grouped[count][0][1]) for count in range(1, max(grouped)+1)]
    report = dict(format="genesis-volume-fault-independent-audit-0.1", status="PASS", producer_imports=False,
        quadrature="Duffy4x4_angular_Gauss32_massq_Gauss32_adaptive_time",
        source_checkpoint_sha256=SOURCE_SHA256,
        input_files=[dict(name=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for _, path, _ in snapshots],
        literal_formation_and_fault_source_bindings=bindings, initial_source_audit=initial,
        intervals=intervals, restart_pairs=restart_pairs, original_point_1_completed=False,
        physical_fault_contact_admission_claimed=False, old_C5_to_C1_admission_claimed=False,
        global_heat_and_mechanical_reservoir_closed=False)
    return {**report, "report_sha256": hashlib.sha256(volume._canonical(report)).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = audit_directory(arguments.directory)
    encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+"\n"
    if arguments.output:
        arguments.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
