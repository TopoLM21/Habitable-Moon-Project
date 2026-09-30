"""Work-conjugate sinking and bending of accepted, connected young slabs.

Each trench is tied to its overriding plate. Along-slab feed is relative
convergence; slab motion through a fixed ambient mantle additionally includes
trench motion. All resistance is assembled from positive Rayleigh dissipation,
not an empirical transmission fraction. This is a prescribed local planar
shape and two-face Couette-envelope approximation, not a mantle Stokes solver.

The force uses the current *thermal mantle* excess mass. Compositional crust
buoyancy, free rollback, and a dynamically evolving dip are not supplied here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


_QUADRATURE_X, _QUADRATURE_W = np.polynomial.legendre.leggauss(24)


@dataclass(frozen=True, slots=True)
class SlabSinkingSystem:
    drag_bending_nm_s: np.ndarray
    drag_mantle_nm_s: np.ndarray
    driving_torque_nm: np.ndarray
    sections: tuple[dict, ...]

    @property
    def drag_total_nm_s(self):
        return self.drag_bending_nm_s + self.drag_mantle_nm_s

    @property
    def feed_matrix_m(self):
        """Maps rad/s plate velocities to signed along-slab feed in m/s."""
        return np.asarray([s["feed_row_m"] for s in self.sections], dtype=float).reshape(
            len(self.sections), self.drag_bending_nm_s.shape[0])

    def power_diagnostics(self, omega_rad_s):
        """Evaluate powers at an explicitly supplied target or actual velocity."""
        omega = np.asarray(omega_rad_s, dtype=float).reshape(-1)
        if omega.shape != (self.drag_bending_nm_s.shape[0],) or not np.isfinite(omega).all():
            raise ValueError("Slab power needs finite angular velocities in rad/s")
        feeds = np.array([np.dot(s["feed_row_m"], omega) for s in self.sections])
        feed_roundoff_scale = np.linalg.norm(self.feed_matrix_m, axis=1)*np.linalg.norm(omega)
        return dict(
            slab_gravitational_power_w=float(self.driving_torque_nm.ravel() @ omega),
            slab_bending_dissipation_w=float(omega @ self.drag_bending_nm_s @ omega),
            slab_mantle_dissipation_w=float(omega @ self.drag_mantle_nm_s @ omega),
            negative_feed_section_count=int(np.sum(feeds < -1e-12*feed_roundoff_scale)),
            minimum_feed_m_s=float(feeds.min()) if feeds.size else 0.,
            maximum_feed_m_s=float(feeds.max()) if feeds.size else 0.,
        )


def _positive(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be positive and finite") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _cross_matrix(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def _bend_angle(s, bend_length, dip_rad):
    """Integral of smooth nonnegative curvature; stable near the trench."""
    x = np.clip(np.asarray(s, dtype=float)/bend_length, 0., 1.)
    small = x < 1e-3
    value = x-np.sin(2.*math.pi*x)/(2.*math.pi)
    if np.any(small):
        y = x[small]
        value[small] = ((2.*math.pi**2/3.)*y**3
                        -(2.*math.pi**4/15.)*y**5
                        +(4.*math.pi**6/315.)*y**7)
    return dip_rad*value


def _curvature_gradient_square_integral(length, bend_length, dip_rad):
    """Exact integral of (d curvature/ds)^2 over the existing slab [1/m^3]."""
    x = min(length/bend_length, 1.)
    if x < 1e-3:
        integral_sine_square = ((4.*math.pi**2/3.)*x**3
                                -(16.*math.pi**4/15.)*x**5
                                +(128.*math.pi**6/315.)*x**7)
    else:
        integral_sine_square = x/2.-math.sin(4.*math.pi*x)/(8.*math.pi)
    return 4.*math.pi**2*dip_rad**2/bend_length**3*integral_sine_square


def _shape_quadrature(length, bend_length, dip_rad):
    """Integrate only the curved portion numerically; its long straight tail is exact."""
    curved = min(length, bend_length)
    positions = .5*curved*(_QUADRATURE_X+1.)
    weights = .5*curved*_QUADRATURE_W
    angles = _bend_angle(positions, bend_length, dip_rad)
    if length > bend_length:
        weights = np.append(weights, length-bend_length)
        angles = np.append(angles, dip_rad)
    return weights, angles


def _shape_sine_integral(start, end, bend_length, dip_rad):
    """Vertical drop over a material interval, without subtracting close totals."""
    curved_end = min(end, bend_length)
    result = 0.
    if start < curved_end:
        half = .5*(curved_end-start)
        points = start+half*(_QUADRATURE_X+1.)
        result += half*float(_QUADRATURE_W @ np.sin(_bend_angle(points, bend_length, dip_rad)))
    if end > bend_length:
        result += (end-max(start, bend_length))*math.sin(dip_rad)
    return result


def _thermal_gravity(section, length, bend_length, dip, gravity, uniform_sine):
    """Project each retained thermal cohort at its actual along-slab interval."""
    layers = getattr(section, "buoyancy_layers", ())
    total_mass = float(section.density_excess_mass_kg)
    if not layers:
        return gravity*total_mass*uniform_sine, uniform_sine, "uniform_thermal_mass_v1", ()
    rows = []
    previous = 0.
    length_tolerance = 1e-10*max(length, 1.)
    for layer in sorted(layers, key=lambda item: item.arc_start_km):
        start, end = 1000.*float(layer.arc_start_km), 1000.*float(layer.arc_end_km)
        mass = float(layer.density_excess_mass_kg)
        if (not all(math.isfinite(x) for x in (start, end, mass))
                or start < 0. or end <= start or mass < 0.
                or abs(start-previous) > length_tolerance or end > length+length_tolerance):
            raise ValueError("Ordered slab thermal layers must form finite contiguous material intervals")
        sine = _shape_sine_integral(start, end, bend_length, dip)/(end-start)
        rows.append(dict(arc_start_km=start/1000., arc_end_km=end/1000.,
            age_myr=float(layer.age_myr), thermal_excess_mass_kg=mass,
            effective_sine=sine, gravitational_feed_force_n=gravity*mass*sine))
        previous = end
    if abs(previous-length) > length_tolerance:
        raise ValueError("Ordered slab thermal intervals must span the retained slab length")
    layer_mass = math.fsum(row["thermal_excess_mass_kg"] for row in rows)
    if not math.isclose(layer_mass, total_mass, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("Ordered slab thermal mass must match its section total")
    force = math.fsum(row["gravitational_feed_force_n"] for row in rows)
    sine = force/(gravity*total_mass) if total_mass > 0. else uniform_sine
    return force, sine, "ordered_thermal_cohorts_v1", tuple(rows)


def slab_sinking_system(sections, plate_count, radius_km, mantle_viscosity_pa_s,
                        mantle_depth_km, params, *, incoming_thickness_km=None):
    """Return global SI drag tensors and the thermal-gravity torque.

    Required parameters are ``gravity_m_s2``, ``young_slab_viscosity_contrast``,
    ``young_slab_bend_radius_thickness_ratio``, and
    ``young_slab_mantle_shear_length_fraction``. The latter fraction multiplies
    physical mantle depth, never a numerical slab cap. ``incoming_thickness_km``
    is the cold mechanical mantle thickness indexed by source face; a zero
    newborn cell falls back to the retained slab's area-weighted cubic
    thickness because a connected old hinge has not vanished with that cell.

    Curvature is kappa(s)=theta/l_b*(1-cos(2*pi*s/l_b)) on [0,l_b], zero
    afterward, with l_b=2*theta*R_b. Thus max(kappa)=1/R_b. Its incomplete
    young portion contributes its actual smaller dip and bending dissipation.
    Sections with explicit buoyancy layers project each cohort's current excess
    mass on its own retained along-slab interval. Old sections without layers
    retain the uniform excess-mass approximation for checkpoint compatibility.
    """
    if isinstance(plate_count, (bool, np.bool_)) or int(plate_count) != plate_count or plate_count < 1:
        raise ValueError("plate_count must be a positive integer")
    count = int(plate_count)
    radius = 1000.*_positive(radius_km, "radius_km")
    viscosity = _positive(mantle_viscosity_pa_s, "mantle_viscosity_pa_s")
    shear_distance = (1000.*_positive(mantle_depth_km, "mantle_depth_km")
        *_positive(params.young_slab_mantle_shear_length_fraction, "mantle shear length fraction"))
    contrast = _positive(params.young_slab_viscosity_contrast, "slab viscosity contrast")
    radius_ratio = _positive(params.young_slab_bend_radius_thickness_ratio, "bend radius/thickness ratio")
    gravity = _positive(params.gravity_m_s2, "gravity_m_s2")
    incoming = None if incoming_thickness_km is None else np.asarray(incoming_thickness_km, dtype=float)
    if incoming is not None and (incoming.ndim != 1 or not np.isfinite(incoming).all() or np.any(incoming < 0)):
        raise ValueError("Incoming slab thickness must be a finite nonnegative cell field")
    bending = np.zeros((3*count, 3*count))
    mantle = np.zeros_like(bending)
    drive = np.zeros(3*count)
    diagnostics = []
    for section in sections:
        area = float(section.accepted_area_km2)
        if not math.isfinite(area) or area < 0:
            raise ValueError("Accepted slab area must be finite and nonnegative")
        if area == 0:
            continue
        sub, over = int(section.subducting_plate), int(section.overriding_plate)
        if not (0 <= sub < count and 0 <= over < count and sub != over):
            raise ValueError("Slab section needs distinct valid plate owners")
        width = 1000.*_positive(section.trench_length_km, "trench_length_km")
        length = area*1e6/width
        dip = math.radians(float(section.dip_deg))
        if not math.isfinite(dip) or not 0. < dip <= math.pi/2.:
            raise ValueError("Slab dip must be in (0, 90] degrees")
        mass = float(section.density_excess_mass_kg)
        if not math.isfinite(mass):
            raise ValueError("Slab excess mass must be finite")
        thickness_cubed = float(section.area_thickness_cubed_km5)/area
        if thickness_cubed == 0. and mass == 0. and section.cold_mantle_volume_km3 == 0.:
            # A chemical oceanic parcel can be accepted before it contains
            # any load-bearing cold mantle. This thermal-mantle-only closure
            # supplies no slab body or hinge for that parcel.
            continue
        if not math.isfinite(thickness_cubed) or thickness_cubed <= 0:
            raise ValueError("A finite accepted slab needs positive thickness")
        thickness = 1000.*thickness_cubed**(1./3.)
        thickness_origin = "retained_cubic_mean"
        if incoming is not None:
            source = int(section.source_face)
            if not 0 <= source < incoming.size:
                raise ValueError("Slab source face is outside the incoming thickness field")
            if incoming[source] > 0:
                thickness = 1000.*float(incoming[source])
                thickness_origin = "incoming_cold_mantle"
        bend_radius = radius_ratio*thickness
        bend_length = 2.*dip*bend_radius
        radial = np.asarray(section.midpoint, dtype=float)
        axis = np.asarray(section.torque_direction, dtype=float)
        if (radial.shape != (3,) or axis.shape != (3,)
                or not np.isfinite(radial).all() or not np.isfinite(axis).all()
                or np.linalg.norm(radial) < 1e-14 or np.linalg.norm(axis) < 1e-14):
            raise ValueError("Slab section requires finite nondegenerate geometry")
        radial = radial/np.linalg.norm(radial)
        normal = np.cross(axis, radial)
        norm = np.linalg.norm(normal)
        if norm < 1e-14:
            raise ValueError("Slab torque direction must have a tangential component")
        normal /= norm
        velocity_map = -radius*_cross_matrix(radial)
        feed = np.zeros(3*count)
        feed[3*sub:3*sub+3] = normal @ velocity_map
        feed[3*over:3*over+3] = -normal @ velocity_map
        trench_map = np.zeros((3, 3*count))
        trench_map[:, 3*over:3*over+3] = velocity_map
        weights, angles = _shape_quadrature(length, bend_length, dip)
        descent = float(weights @ np.sin(angles))
        mean_sine = descent/length
        gravitational_force, buoyancy_sine, buoyancy_model, buoyancy_layers = _thermal_gravity(
            section, length, bend_length, dip, gravity, mean_sine)
        drive += gravitational_force*feed
        gradient_integral = _curvature_gradient_square_integral(length, bend_length, dip)
        bend_coefficient = viscosity*contrast*width*thickness**3/3.*gradient_integral
        bending += bend_coefficient*np.outer(feed, feed)
        mantle_coefficient_per_length = 2.*viscosity*width/shear_distance
        # Build a six-DOF Gram matrix before global scatter. This preserves
        # positive dissipation without allocating a global matrix per point.
        indices = np.r_[np.arange(3*sub, 3*sub+3), np.arange(3*over, 3*over+3)]
        tangent = np.cos(angles)[:, None]*normal-np.sin(angles)[:, None]*radial
        slab_map = (trench_map[:, indices][None, :, :]
                    +tangent[:, :, None]*feed[indices][None, None, :])
        local_mantle = mantle_coefficient_per_length*np.einsum(
            "qki,qkj,q->ij", slab_map, slab_map, weights)
        mantle[np.ix_(indices, indices)] += local_mantle
        mantle_feed_drag = mantle_coefficient_per_length*(
            (weights @ tangent) @ trench_map+length*feed)
        diagnostics.append(dict(
            contact_key=section.contact_key, subducting_plate=sub, overriding_plate=over,
            accepted_area_km2=area, slab_length_km=length/1000., trench_length_km=width/1000.,
            nominal_dip_deg=float(section.dip_deg),
            tip_dip_deg=float(np.rad2deg(_bend_angle(np.array([length]), bend_length, dip)[0])),
            geometric_depth_km=descent/1000., effective_sine=buoyancy_sine,
            geometric_mean_sine=mean_sine,
            cold_hinge_thickness_km=thickness/1000., thickness_origin=thickness_origin,
            bend_radius_km=bend_radius/1000., bend_length_km=bend_length/1000.,
            mantle_shear_distance_km=shear_distance/1000.,
            mantle_viscosity_pa_s=viscosity, effective_bending_viscosity_pa_s=contrast*viscosity,
            thermal_excess_mass_kg=mass, gravitational_feed_force_n=gravitational_force,
            uniform_buoyancy_feed_force_n=gravity*mass*mean_sine,
            buoyancy_distribution_model=buoyancy_model, buoyancy_layers=buoyancy_layers,
            bending_coefficient_n_s_m=bend_coefficient,
            mantle_coefficient_n_s_m=mantle_coefficient_per_length*length,
            mantle_feed_drag_row_n_s=mantle_feed_drag.tolist(),
            feed_row_m=feed.tolist(), buoyancy_model="thermal_mantle_only",
            geometry_model=("smooth_prescribed_bend_uniform_excess_mass" if
                buoyancy_model == "uniform_thermal_mass_v1" else "smooth_prescribed_bend_ordered_thermal_cohorts"),
            mantle_frame="fixed_prescribed_source_zero_deep_flow",
        ))
    return SlabSinkingSystem(bending, mantle, drive.reshape(count, 3), tuple(diagnostics))
