"""Material-aware tectonic topography (v0.15) with physical ocean-depth control.

The v0.6-v0.9.2 model used a sqrt(age) oceanic subsidence law clipped at
7.5 km and then added trench forcing edge-by-edge.  On a triangular mesh,
several boundary edges may touch one cell, so the same trench could be counted
multiple times.  Long runs therefore approached the global -12 km numerical
floor.

v0.9.3 replaces that behavior with two explicit pieces:

1. Normal oceanic bathymetry follows a configurable plate-cooling depth-age
   curve.  The young branch is sqrt(age); the old branch approaches an
   asymptotic depth smoothly instead of requiring a hard ocean-depth clip.
2. Trench deflection is a bounded *local anomaly* determined by the age of the
   subducting oceanic lithosphere and the convergence rate.  Boundary segments
   combine by maximum geological effect, not by additive edge count.

Zero elevation remains a reference datum, not solved sea level.  The remaining
very-wide elevation bounds are numerical safety rails only and should never be
active in a calibrated run.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .kinematics import BoundaryRecord, BoundaryType
from .lithosphere import CrustType, LithosphereState, continental_material_fields, effective_continental_thickness_km
from .mesh import SphereMesh
from .flexure import FlexureParameters, FlexureDiagnostics, solve_flexural_response

Array = np.ndarray


@dataclass(slots=True)
class TopographyParameters:
    continental_reference_thickness_km: float = 35.0
    continental_reference_elevation_m: float = 350.0
    continental_density_kg_m3: float = 2800.0
    mantle_density_kg_m3: float = 3300.0
    sediment_density_kg_m3: float = 2400.0

    # Normal ocean-floor depth-age relation.  Defaults are close to the classic
    # Parsons-Sclater plate-model fit, used here as an effective calibration.
    ridge_axis_depth_m: float = 2500.0
    ocean_young_subsidence_m_per_sqrt_myr: float = 350.0
    ocean_plate_transition_age_myr: float = 20.0
    ocean_plate_asymptotic_depth_m: float = 6400.0
    ocean_plate_exponential_amplitude_m: float = 3200.0
    ocean_plate_tau_myr: float = 62.8

    ridge_uplift_m: float = 1500.0

    # Trench depth is now a bounded anomaly relative to the normal ocean floor.
    trench_min_extra_depth_m: float = 1800.0
    trench_max_extra_depth_m: float = 4300.0
    trench_age_scale_myr: float = 55.0
    trench_rate_scale_km_per_myr: float = 55.0
    trench_age_weight: float = 0.60
    trench_rate_weight: float = 0.40

    arc_uplift_m: float = 1200.0
    continental_collision_uplift_m: float = 1800.0
    boundary_one_ring_fraction: float = 0.40
    isostatic_relaxation_myr: float = 4.0
    erosion_diffusion_per_myr: float = 0.018
    max_erosion_fraction_per_step: float = 0.22

    # Safety rails only.  A healthy run should have zero cells touching these.
    numerical_min_elevation_m: float = -18000.0
    numerical_max_elevation_m: float = 12000.0

    # v0.15: topography follows the independent continental-material layer
    # rather than the legacy >=50% binary crust raster.  A mixed cell is an
    # unresolved area-weighted combination of an oceanic thermal endmember and
    # an Airy-supported continental endmember.  This is passive: it changes
    # relief/sea level, not plate forces or material transport.
    material_aware_isostasy: bool = True
    material_aware_boundary_forcing: bool = True
    material_fraction_epsilon: float = 1.0e-9

    # Backwards-compatible aliases used by older validation code.  In v0.9.3
    # they mean numerical safety rails, not physical bathymetric limits.
    @property
    def min_elevation_m(self) -> float:
        return self.numerical_min_elevation_m

    @property
    def max_elevation_m(self) -> float:
        return self.numerical_max_elevation_m


@dataclass(slots=True)
class TopographyState:
    time_myr: float
    elevation_m: Array


@dataclass(slots=True)
class TopographyDiagnostics:
    time_myr: float
    dt_myr: float
    min_elevation_m: float
    max_elevation_m: float
    mean_elevation_m: float
    mean_continental_elevation_m: float
    mean_oceanic_elevation_m: float
    reference_exposed_fraction: float
    ridge_cells: int
    trench_cells: int
    arc_cells: int
    collision_cells: int
    eroded_volume_km3: float
    numerical_min_clip_cells: int = 0
    numerical_max_clip_cells: int = 0
    deepest_normal_ocean_m: float = 0.0
    deepest_trench_anomaly_m: float = 0.0
    mean_continental_material_elevation_m: float = 0.0
    mixed_material_cells: int = 0
    max_effective_continental_thickness_km: float = 0.0
    mean_elastic_thickness_km: float = 0.0
    min_elastic_thickness_km: float = 0.0
    max_elastic_thickness_km: float = 0.0
    mean_flexural_parameter_km: float = 0.0
    max_abs_flexural_correction_m: float = 0.0
    rms_flexural_correction_m: float = 0.0
    flexure_cg_iterations: int = 0
    flexure_cg_converged: bool = True
    flexure_area_mean_source_m: float = 0.0
    flexure_area_mean_response_m: float = 0.0


def oceanic_plate_depth_m(age_myr: Array | float, params: TopographyParameters) -> Array:
    """Normal oceanic basement depth below the reference datum.

    Young lithosphere follows d = d0 + k*sqrt(t).  Older lithosphere follows an
    asymptotic plate-cooling branch d_inf - A*exp(-t/tau).  The transition is
    blended over a narrow interval to avoid a derivative jump if the configured
    branches are not exactly coincident.
    """
    age = np.maximum(np.asarray(age_myr, dtype=np.float64), 0.0)
    young = float(params.ridge_axis_depth_m) + float(params.ocean_young_subsidence_m_per_sqrt_myr) * np.sqrt(age)
    old = float(params.ocean_plate_asymptotic_depth_m) - float(params.ocean_plate_exponential_amplitude_m) * np.exp(
        -age / max(float(params.ocean_plate_tau_myr), 1e-9)
    )
    t0 = float(params.ocean_plate_transition_age_myr)
    # 4 Myr smooth blend centered on t0; narrow compared with geological ages
    # but wide enough to remove a numerical kink on the 4 Myr long-run step.
    half_width = 2.0
    w = np.clip((age - (t0 - half_width)) / (2.0 * half_width), 0.0, 1.0)
    w = w * w * (3.0 - 2.0 * w)  # smoothstep
    return (1.0 - w) * young + w * old


def _legacy_base_elevation(state: LithosphereState, params: TopographyParameters) -> Array:
    """Pre-v0.15 binary base-elevation rule, retained for compatibility."""
    n = len(state.crust_type)
    out = np.empty(n, dtype=np.float64)
    ocean = state.crust_type == int(CrustType.OCEANIC)
    cont = ~ocean

    age = np.maximum(np.asarray(state.crust_age_myr, dtype=np.float64), 0.0)
    ocean_depth = oceanic_plate_depth_m(age, params)
    out[ocean] = -ocean_depth[ocean]

    rho_c = float(params.continental_density_kg_m3)
    rho_m = float(params.mantle_density_kg_m3)
    airy_factor = max((rho_m - rho_c) / max(rho_m, 1e-9), 0.0)
    excess_km = np.asarray(state.crust_thickness_km, dtype=np.float64) - float(params.continental_reference_thickness_km)
    out[cont] = float(params.continental_reference_elevation_m) + 1000.0 * airy_factor * excess_km[cont]
    return out


def material_topography_endmembers(
    mesh: SphereMesh,
    state: LithosphereState,
    radius_km: float,
    params: TopographyParameters,
) -> tuple[Array, Array, Array, Array]:
    """Return (fraction, effective continental thickness, ocean, continent).

    The material layer describes unresolved footprint coverage, not a binary
    crust label.  The continental endmember uses the *material's own* thickness
    V/(A f); the ocean endmember retains the existing thermal depth-age law.
    Mixing them linearly by footprint fraction is the conservative area-mean
    topographic closure for a single scalar elevation per mesh cell.
    """
    areas = mesh.physical_cell_areas_km2(float(radius_km))
    frac, volume = continental_material_fields(state, areas)
    eps = max(float(params.material_fraction_epsilon), 0.0)
    frac = np.where(frac > eps, np.clip(frac, 0.0, 1.0), 0.0)
    h_cont = effective_continental_thickness_km(frac, volume, areas, eps=max(eps, 1e-15))

    age = np.maximum(np.asarray(state.crust_age_myr, dtype=np.float64), 0.0)
    ocean_endmember = -oceanic_plate_depth_m(age, params)

    rho_c = float(params.continental_density_kg_m3)
    rho_m = float(params.mantle_density_kg_m3)
    airy_factor = max((rho_m - rho_c) / max(rho_m, 1e-9), 0.0)
    continental_endmember = (
        float(params.continental_reference_elevation_m)
        + 1000.0 * airy_factor
        * (h_cont - float(params.continental_reference_thickness_km))
    )
    # No material means the continental endmember is diagnostically irrelevant;
    # keeping it finite avoids NaNs in component maps.
    continental_endmember = np.where(frac > eps, continental_endmember, float(params.continental_reference_elevation_m))
    return frac, h_cont, ocean_endmember, continental_endmember


def material_subgrid_surface_elevations(
    mesh: SphereMesh,
    state: LithosphereState,
    topography: TopographyState,
    radius_km: float,
    params: TopographyParameters,
) -> tuple[Array, Array, Array]:
    """Return (continental_fraction, ocean_surface_m, continental_surface_m).

    v0.15 keeps one scalar topographic state for relaxation/erosion, but the
    continental-material field is sub-cell.  For hydrosphere capacity we recover
    two unresolved surface patches per cell.  Their area-weighted mean is
    exactly the stored scalar topography: the residual between current relief
    and the material-aware base is applied equally to both endmembers.
    """
    frac, _, ocean_end, cont_end = material_topography_endmembers(mesh, state, radius_km, params)
    base = (1.0 - frac) * ocean_end + frac * cont_end
    residual = np.asarray(topography.elevation_m, dtype=np.float64) - base
    return frac, ocean_end + residual, cont_end + residual


def _base_elevation(
    mesh: SphereMesh,
    state: LithosphereState,
    params: TopographyParameters,
    radius_km: float | None = None,
) -> Array:
    if not bool(params.material_aware_isostasy) or radius_km is None:
        return _legacy_base_elevation(state, params)
    frac, _, ocean_endmember, continental_endmember = material_topography_endmembers(
        mesh, state, float(radius_km), params
    )
    return (1.0 - frac) * ocean_endmember + frac * continental_endmember


def _subducting_face(state: LithosphereState, b: BoundaryRecord) -> tuple[int | None, int | None]:
    """Return (subducting_face, overriding_face) for a convergent segment."""
    ta = int(state.crust_type[b.face_a]); tb = int(state.crust_type[b.face_b])
    if ta == int(CrustType.OCEANIC) and tb == int(CrustType.CONTINENTAL):
        return int(b.face_a), int(b.face_b)
    if tb == int(CrustType.OCEANIC) and ta == int(CrustType.CONTINENTAL):
        return int(b.face_b), int(b.face_a)
    if ta == int(CrustType.OCEANIC) and tb == int(CrustType.OCEANIC):
        aa = float(state.crust_age_myr[b.face_a]); ab = float(state.crust_age_myr[b.face_b])
        if aa > ab + 1e-9:
            return int(b.face_a), int(b.face_b)
        if ab > aa + 1e-9:
            return int(b.face_b), int(b.face_a)
        if int(b.plate_a) <= int(b.plate_b):
            return int(b.face_a), int(b.face_b)
        return int(b.face_b), int(b.face_a)
    return None, None


def trench_extra_depth_m(state: LithosphereState, b: BoundaryRecord, subducting_face: int, params: TopographyParameters) -> float:
    """Effective trench deflection below normal oceanic bathymetry.

    Older oceanic lithosphere and faster convergence deepen the flexural trench,
    but the anomaly saturates.  This avoids both an arbitrary absolute seafloor
    floor and unbounded edge-count accumulation.
    """
    age = max(float(state.crust_age_myr[int(subducting_face)]), 0.0)
    convergence = max(-float(b.normal_rate_km_per_myr), 0.0)
    age_factor = 1.0 - np.exp(-age / max(float(params.trench_age_scale_myr), 1e-9))
    rate_factor = 1.0 - np.exp(-convergence / max(float(params.trench_rate_scale_km_per_myr), 1e-9))
    wa = max(float(params.trench_age_weight), 0.0)
    wr = max(float(params.trench_rate_weight), 0.0)
    norm = max(wa + wr, 1e-12)
    maturity = np.clip((wa * age_factor + wr * rate_factor) / norm, 0.0, 1.0)
    lo = float(params.trench_min_extra_depth_m)
    hi = max(float(params.trench_max_extra_depth_m), lo)
    return lo + (hi - lo) * float(maturity)


def _max_positive_with_halo(mesh: SphereMesh, field: Array, seed: int, amount: float, one_ring_fraction: float) -> None:
    i = int(seed)
    field[i] = max(float(field[i]), float(amount))
    if one_ring_fraction <= 0.0:
        return
    halo = float(amount) * float(one_ring_fraction)
    for nb in mesh.neighbors[i]:
        field[int(nb)] = max(float(field[int(nb)]), halo)


def _max_negative_depth_with_halo(mesh: SphereMesh, depth_field: Array, seed: int, depth: float, one_ring_fraction: float) -> None:
    i = int(seed)
    depth_field[i] = max(float(depth_field[i]), float(depth))
    if one_ring_fraction <= 0.0:
        return
    halo = float(depth) * float(one_ring_fraction)
    for nb in mesh.neighbors[i]:
        depth_field[int(nb)] = max(float(depth_field[int(nb)]), halo)


def tectonic_forcing(
    mesh: SphereMesh,
    state: LithosphereState,
    boundaries: list[BoundaryRecord],
    params: TopographyParameters,
    radius_km: float | None = None,
    arc_uplift_forcing: Array | None = None,
) -> tuple[Array, dict[str, set[int]], dict[str, Array]]:
    """Compute non-additive local tectonic topographic anomalies in metres."""
    ridge = np.zeros(mesh.cell_count, dtype=np.float64)
    trench_depth = np.zeros(mesh.cell_count, dtype=np.float64)
    arc = np.zeros(mesh.cell_count, dtype=np.float64)
    collision = np.zeros(mesh.cell_count, dtype=np.float64)
    tags = {"ridge": set(), "trench": set(), "arc": set(), "collision": set()}
    halo = float(params.boundary_one_ring_fraction)
    material_fraction = None
    if bool(params.material_aware_boundary_forcing) and radius_km is not None:
        areas = mesh.physical_cell_areas_km2(float(radius_km))
        material_fraction, _ = continental_material_fields(state, areas)
        material_fraction = np.clip(material_fraction, 0.0, 1.0)
    external_arc = None
    if arc_uplift_forcing is not None:
        external_arc = np.clip(np.asarray(arc_uplift_forcing, dtype=np.float64), 0.0, 2.0)
        if external_arc.shape != (mesh.cell_count,):
            raise ValueError("arc_uplift_forcing must have shape (cell_count,)")
        arc[:] = float(params.arc_uplift_m) * external_arc
        tags["arc"].update(int(i) for i in np.flatnonzero(external_arc > 1e-8))

    for b in boundaries:
        if b.boundary_type == BoundaryType.DIVERGENT:
            for face in (int(b.face_a), int(b.face_b)):
                if material_fraction is None:
                    scale = 1.0 if int(state.crust_type[face]) == int(CrustType.OCEANIC) else 0.35
                else:
                    # Oceanic spreading has the full ridge anomaly; continental
                    # rifting retains the legacy 35% endmember continuously.
                    scale = 1.0 - 0.65 * float(material_fraction[face])
                _max_positive_with_halo(mesh, ridge, face, float(params.ridge_uplift_m) * scale, halo)
                tags["ridge"].add(face)

        elif b.boundary_type == BoundaryType.CONVERGENT:
            if material_fraction is None:
                ta = int(state.crust_type[b.face_a]); tb = int(state.crust_type[b.face_b])
                if ta == int(CrustType.CONTINENTAL) and tb == int(CrustType.CONTINENTAL):
                    for face in (int(b.face_a), int(b.face_b)):
                        _max_positive_with_halo(mesh, collision, face, float(params.continental_collision_uplift_m), halo)
                        tags["collision"].add(face)
                else:
                    sub, over = _subducting_face(state, b)
                    if sub is not None:
                        depth = trench_extra_depth_m(state, b, sub, params)
                        _max_negative_depth_with_halo(mesh, trench_depth, sub, depth, halo)
                        tags["trench"].add(sub)
                    if over is not None and external_arc is None:
                        _max_positive_with_halo(mesh, arc, over, float(params.arc_uplift_m), halo)
                        tags["arc"].add(over)
            else:
                a = int(b.face_a); c = int(b.face_b)
                fa = float(material_fraction[a]); fb = float(material_fraction[c])
                # The shared continental fraction participates in collision;
                # any remaining oceanic part may still support trench/arc relief.
                collision_strength = min(fa, fb)
                if collision_strength > 1e-12:
                    amount = float(params.continental_collision_uplift_m) * collision_strength
                    for face in (a, c):
                        _max_positive_with_halo(mesh, collision, face, amount, halo)
                        tags["collision"].add(face)

                oceanic_strength = max(1.0 - collision_strength, 0.0)
                if oceanic_strength > 1e-12:
                    # Prefer the less continental side as the subducting mixed
                    # endmember.  If both have equal coverage, fall back to the
                    # established age/plate-id rule.
                    if fa + 1e-9 < fb:
                        sub, over = a, c
                    elif fb + 1e-9 < fa:
                        sub, over = c, a
                    else:
                        sub, over = _subducting_face(state, b)
                    if sub is not None:
                        sub_ocean = 1.0 - float(material_fraction[int(sub)])
                        strength = oceanic_strength * max(sub_ocean, 0.0)
                        if strength > 1e-12:
                            depth = trench_extra_depth_m(state, b, int(sub), params) * strength
                            _max_negative_depth_with_halo(mesh, trench_depth, int(sub), depth, halo)
                            tags["trench"].add(int(sub))
                            if over is not None and external_arc is None:
                                _max_positive_with_halo(mesh, arc, int(over), float(params.arc_uplift_m) * strength, halo)
                                tags["arc"].add(int(over))

    # Different geological mechanisms may coexist, so combine the independent
    # fields once.  Duplicate mesh edges of the *same* mechanism cannot stack.
    forcing = ridge + arc + collision - trench_depth
    components = {"ridge": ridge, "trench_depth": trench_depth, "arc": arc, "collision": collision}
    return forcing, tags, components


def _local_airy_thickness_anomaly_m(
    mesh: SphereMesh,
    state: LithosphereState,
    params: TopographyParameters,
    radius_km: float | None,
) -> Array:
    """Local Airy relief caused by *thickness departures* from reference crust.

    v0.22 deliberately does not flex the full continent-ocean reference-height
    contrast.  A normal 35-km continent keeps its reference elevation locally;
    only excess/thinned crustal support is treated as a mechanical load.  This
    prevents the elastic solver from smearing the several-kilometre ocean-vs-
    continent datum step across coastlines while still allowing mountain belts
    and thinned margins to generate foreland/forearc flexure.
    """
    rho_c=float(params.continental_density_kg_m3)
    rho_m=float(params.mantle_density_kg_m3)
    airy=max((rho_m-rho_c)/max(rho_m,1e-9),0.0)
    ref=float(params.continental_reference_thickness_km)
    if bool(params.material_aware_isostasy) and radius_km is not None:
        f,h,_,_=material_topography_endmembers(mesh,state,float(radius_km),params)
        return np.asarray(f,dtype=np.float64)*1000.0*airy*(np.asarray(h,dtype=np.float64)-ref)
    cont=np.asarray(state.crust_type)==int(CrustType.CONTINENTAL)
    out=np.zeros(mesh.cell_count,dtype=np.float64)
    th=np.asarray(state.crust_thickness_km,dtype=np.float64)
    out[cont]=1000.0*airy*(th[cont]-ref)
    return out


def _equilibrium_build(
    mesh: SphereMesh,
    state: LithosphereState,
    boundaries: list[BoundaryRecord],
    params: TopographyParameters,
    radius_km: float | None = None,
    arc_uplift_forcing: Array | None = None,
    flexure_params: FlexureParameters | None = None,
    gravity_m_s2: float | None = None,
) -> tuple[Array, dict[str,set[int]], dict[str,Array], FlexureDiagnostics]:
    """Build local-isostatic and elastic-plate equilibrium topography.

    Thermal ocean-floor depth, the reference continental datum and hot ridge
    uplift remain local/background terms.  Flexure is applied only to loads that
    mechanically bend the plate: continental thickness anomaly, volcanic-arc
    load, collision load, and trench deflection.
    """
    base=_base_elevation(mesh,state,params,radius_km)
    forcing,tags,comp=tectonic_forcing(mesh,state,boundaries,params,radius_km,arc_uplift_forcing)
    local_airy=_local_airy_thickness_anomaly_m(mesh,state,params,radius_km)
    sediment_h=np.zeros(mesh.cell_count,dtype=np.float64)
    sediment_load=np.zeros(mesh.cell_count,dtype=np.float64)
    if radius_km is not None and state.sediment_volume_km3 is not None:
        _A=mesh.physical_cell_areas_km2(float(radius_km))
        sediment_h=1000.0*np.maximum(np.asarray(state.sediment_volume_km3,dtype=np.float64),0.0)/np.maximum(_A,1e-30)
        # Deposit adds its geometric thickness but loads the floating plate.
        # In the local Airy limit this gives a net surface rise
        # h*(rho_m-rho_s)/rho_m; finite rigidity spreads the subsidence.
        sediment_load=-float(params.sediment_density_kg_m3)/max(float(params.mantle_density_kg_m3),1e-9)*sediment_h
    flex_source=(local_airy + np.asarray(comp['arc'],dtype=np.float64)
                 + np.asarray(comp['collision'],dtype=np.float64)
                 - np.asarray(comp['trench_depth'],dtype=np.float64)
                 + sediment_load)
    # Remove only the thickness-dependent Airy anomaly from the background;
    # ridge uplift is intentionally not flexed because it is a thermal buoyancy
    # anomaly rather than an imposed surface load in this effective model.
    background=base-local_airy+np.asarray(comp['ridge'],dtype=np.float64)+sediment_h
    if flexure_params is not None and radius_km is not None and gravity_m_s2 is not None:
        flex_response,fdiag,Te,alpha=solve_flexural_response(
            mesh,state,flex_source,float(radius_km),float(gravity_m_s2),flexure_params)
    else:
        flex_response=flex_source.copy()
        Te=np.zeros(mesh.cell_count,dtype=np.float64)
        alpha=np.zeros(mesh.cell_count,dtype=np.float64)
        fdiag=FlexureDiagnostics(
            area_mean_source_m=float(np.mean(flex_source)),
            area_mean_response_m=float(np.mean(flex_source)),
        )
    target=background+flex_response
    target=np.clip(target,float(params.numerical_min_elevation_m),float(params.numerical_max_elevation_m))
    out={'base':base,'forcing':forcing,**comp,
         'local_airy_thickness_anomaly':local_airy,
         'flexural_local_source':flex_source,
         'flexural_response':np.asarray(flex_response,dtype=np.float64),
         'flexural_correction':np.asarray(flex_response,dtype=np.float64)-flex_source,
         'effective_elastic_thickness_km':np.asarray(Te,dtype=np.float64),
         'flexural_parameter_km':np.asarray(alpha,dtype=np.float64)}
    out['sediment_thickness_m']=sediment_h
    out['sediment_load_isostatic_m']=sediment_load
    if bool(params.material_aware_isostasy) and radius_km is not None:
        f,h,ocean_end,cont_end=material_topography_endmembers(mesh,state,float(radius_km),params)
        out.update({'continental_fraction':f,'effective_continental_thickness_km':h,
                    'ocean_endmember':ocean_end,'continental_endmember':cont_end})
    out['equilibrium']=target
    return target,tags,out,fdiag


def equilibrium_elevation(
    mesh: SphereMesh,
    state: LithosphereState,
    boundaries: list[BoundaryRecord],
    params: TopographyParameters,
    radius_km: float | None = None,
    arc_uplift_forcing: Array | None = None,
    flexure_params: FlexureParameters | None = None,
    gravity_m_s2: float | None = None,
) -> tuple[Array, dict[str, set[int]]]:
    target,tags,_,_=_equilibrium_build(mesh,state,boundaries,params,radius_km,arc_uplift_forcing,flexure_params,gravity_m_s2)
    return target,tags


def topography_components(
    mesh: SphereMesh,
    state: LithosphereState,
    boundaries: list[BoundaryRecord],
    params: TopographyParameters,
    radius_km: float | None = None,
    arc_uplift_forcing: Array | None = None,
    flexure_params: FlexureParameters | None = None,
    gravity_m_s2: float | None = None,
) -> dict[str, Array]:
    """Return diagnostic component fields used to build equilibrium relief."""
    _,_,out,_=_equilibrium_build(mesh,state,boundaries,params,radius_km,arc_uplift_forcing,flexure_params,gravity_m_s2)
    return out


def initialize_topography(
    mesh: SphereMesh,
    state: LithosphereState,
    boundaries: list[BoundaryRecord],
    params: TopographyParameters,
    radius_km: float | None = None,
    flexure_params: FlexureParameters | None = None,
    gravity_m_s2: float | None = None,
) -> TopographyState:
    target,_=equilibrium_elevation(mesh,state,boundaries,params,radius_km,None,flexure_params,gravity_m_s2)
    return TopographyState(time_myr=float(state.time_myr),elevation_m=target.copy())

def _erode_positive_relief(mesh: SphereMesh, elevation: Array, areas_km2: Array, dt_myr: float, params: TopographyParameters) -> tuple[Array, float]:
    old = np.asarray(elevation, dtype=np.float64)
    neighbor_index = np.asarray(mesh.neighbors, dtype=np.int32)
    neigh_mean = np.mean(old[neighbor_index], axis=1)
    local_excess = np.maximum(old - neigh_mean, 0.0)
    active = old > 0.0
    frac = min(float(params.erosion_diffusion_per_myr) * float(dt_myr), float(params.max_erosion_fraction_per_step))
    removed_m = np.zeros_like(old)
    removed_m[active] = frac * local_excess[active]
    new = old - removed_m
    volume_km3 = float(np.sum(areas_km2 * removed_m) / 1000.0)
    return new, volume_km3


def advance_topography(
    mesh: SphereMesh,
    lithosphere: LithosphereState,
    boundaries: list[BoundaryRecord],
    previous: TopographyState,
    dt_myr: float,
    radius_km: float,
    params: TopographyParameters,
    arc_uplift_forcing: Array | None = None,
    flexure_params: FlexureParameters | None = None,
    gravity_m_s2: float | None = None,
) -> tuple[TopographyState, TopographyDiagnostics, Array]:
    if dt_myr <= 0.0:
        raise ValueError("dt_myr must be positive")
    target,tags,components,fdiag=_equilibrium_build(
        mesh,lithosphere,boundaries,params,radius_km,arc_uplift_forcing,flexure_params,gravity_m_s2)
    alpha=1.0-np.exp(-float(dt_myr)/max(float(params.isostatic_relaxation_myr),1e-9))
    elev=np.asarray(previous.elevation_m,dtype=np.float64)+alpha*(target-np.asarray(previous.elevation_m,dtype=np.float64))
    areas=mesh.physical_cell_areas_km2(radius_km)
    elev,eroded=_erode_positive_relief(mesh,elev,areas,dt_myr,params)

    low_before=int(np.sum(elev<float(params.numerical_min_elevation_m)))
    high_before=int(np.sum(elev>float(params.numerical_max_elevation_m)))
    elev=np.clip(elev,float(params.numerical_min_elevation_m),float(params.numerical_max_elevation_m))

    out=TopographyState(time_myr=float(lithosphere.time_myr),elevation_m=elev)
    cont=lithosphere.crust_type==int(CrustType.CONTINENTAL)
    ocean=~cont
    material_frac=components.get('continental_fraction')
    if material_frac is None: material_frac=cont.astype(np.float64)
    material_frac=np.clip(np.asarray(material_frac,dtype=np.float64),0.0,1.0)
    material_weight=areas*material_frac
    ocean_component=components.get('ocean_endmember',components['base'])
    ocean_base=np.asarray(ocean_component)[(1.0-material_frac)>1e-12]
    trench_field=components['trench_depth']
    diag=TopographyDiagnostics(
        time_myr=float(lithosphere.time_myr),dt_myr=float(dt_myr),
        min_elevation_m=float(np.min(elev)),max_elevation_m=float(np.max(elev)),mean_elevation_m=float(np.mean(elev)),
        mean_continental_elevation_m=float(np.mean(elev[cont])) if np.any(cont) else 0.0,
        mean_oceanic_elevation_m=float(np.mean(elev[ocean])) if np.any(ocean) else 0.0,
        reference_exposed_fraction=float(np.sum(areas[elev>0.0])/np.sum(areas)),
        ridge_cells=len(tags['ridge']),trench_cells=len(tags['trench']),arc_cells=len(tags['arc']),collision_cells=len(tags['collision']),
        eroded_volume_km3=float(eroded),numerical_min_clip_cells=low_before,numerical_max_clip_cells=high_before,
        deepest_normal_ocean_m=float(np.min(ocean_base)) if ocean_base.size else 0.0,
        deepest_trench_anomaly_m=-float(np.max(trench_field)) if np.any(trench_field>0.0) else 0.0,
        mean_continental_material_elevation_m=(float(np.sum(elev*material_weight)/np.sum(material_weight)) if np.sum(material_weight)>0.0 else 0.0),
        mixed_material_cells=int(np.sum((material_frac>1e-9)&(material_frac<1.0-1e-9))),
        max_effective_continental_thickness_km=float(np.max(components.get('effective_continental_thickness_km',np.zeros_like(elev)))),
        mean_elastic_thickness_km=float(fdiag.mean_elastic_thickness_km),min_elastic_thickness_km=float(fdiag.min_elastic_thickness_km),max_elastic_thickness_km=float(fdiag.max_elastic_thickness_km),
        mean_flexural_parameter_km=float(fdiag.mean_flexural_parameter_km),max_abs_flexural_correction_m=float(fdiag.max_abs_flexural_correction_m),rms_flexural_correction_m=float(fdiag.rms_flexural_correction_m),
        flexure_cg_iterations=int(fdiag.cg_iterations),flexure_cg_converged=bool(fdiag.cg_converged),
        flexure_area_mean_source_m=float(fdiag.area_mean_source_m),flexure_area_mean_response_m=float(fdiag.area_mean_response_m),
    )
    return out,diag,target

