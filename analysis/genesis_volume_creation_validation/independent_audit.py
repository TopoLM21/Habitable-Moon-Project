"""Producer-free edge-surface and prospective creation-work checks.

The ruled surfaces are explicit computational supports.  An oracle deriving
their measure and work does not select a physical Genesis fracture, release
ties, authorize underthrust, or supply missing historical material properties.
No tectonics module is imported here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from analysis.genesis_volume_bank_validation.independent_audit import (
    ACTUAL_PARENT_RAW_SHA256, ACTUAL_PARENT_CHECKPOINT_SHA256,
    _actual_parent, _source_fields, canonical, reference_topology,
)
from analysis.genesis_volume_continuation_validation.independent_audit import (
    coefficient_energy, endpoint_clock_binding, isotropic_matrix, mass_matrix,
)

FORMAT = "genesis-volume-creation-control-0.1"
GROWTH_FORMAT = "genesis-volume-creation-birth-growth-control-0.1"
LAW_KEYS = ("fracture_energy_j_m2", "peak_traction_pa",
            "cohesive_stiffness_pa_m", "compression_stiffness_pa_m")


def array(value, shape=None, integer=False):
    raw = np.asarray(value)
    if raw.size and raw.dtype.kind not in ("iu" if integer else "iuf"):
        raise ValueError("Independent creation oracle requires literal numeric arrays")
    result = np.asarray(value, np.int64 if integer else float)
    if shape is not None and not result.size and not np.prod(shape):
        result = result.reshape(shape)
    if (shape is not None and result.shape != shape) or not np.isfinite(result).all():
        raise ValueError("Independent creation oracle received invalid dimensions or values")
    return result


def close(actual, expected, name, rtol=3e-11, atol=2e-12):
    a, b = array(actual), array(expected)
    if not a.size and not b.size:
        a = a.reshape(b.shape)
    if a.shape != b.shape:
        raise ValueError(f"Independent creation audit: {name} shape differs")
    scale = max(float(np.max(np.abs(b), initial=0.)), 1e-300)
    error = float(np.max(np.abs(a-b), initial=0.))
    if error > atol+rtol*scale:
        raise ValueError(f"Independent creation audit: {name} differs")
    return error/max(scale, atol, 1e-300)


def validate_law(law):
    if type(law) is not dict or set(law) != set(LAW_KEYS):
        raise ValueError("Independent creation law needs four explicit parameters")
    values = []
    for key in LAW_KEYS:
        value = law[key]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)):
            raise ValueError("Independent creation parameters must be real numbers")
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Independent creation parameters must be positive")
        values.append(float(value))
    Gc, peak, stiffness, compression = values
    onset, failure = peak/stiffness, 2*Gc/peak
    if not np.isfinite(onset+failure) or not failure > onset:
        raise ValueError("Independent creation law needs a positive softening interval")
    return Gc, peak, stiffness, compression, onset, failure


def law_point(law, gap, maximum):
    """Independent instantaneous traction, energy and irreversible state.

    Cohesive modes use a Euclidean vector norm, whereas compressive normal
    work is stored separately.  The maximum history cannot heal on unload.
    """
    Gc, peak, stiffness, compression, onset, failure = validate_law(law)
    g = array(gap)
    if g.shape[-1:] != (3,):
        raise ValueError("Independent creation gap must have three components")
    maximum = array(maximum)
    if maximum.shape != g.shape[:-1] or np.any(maximum < 0):
        raise ValueError("Independent creation maximum history shape differs")
    positive = g.copy(); positive[..., 0] = np.maximum(g[..., 0], 0.)
    radius = np.hypot.reduce(positive, axis=-1)
    kappa = np.maximum(maximum, radius)
    # Equivalent secant traction is derived from the triangular envelope.
    secant = np.ones_like(kappa)
    soften = (kappa > onset) & (kappa < failure)
    secant[soften] = onset*(failure-kappa[soften])/(kappa[soften]*(failure-onset))
    secant[kappa >= failure] = 0.
    t = stiffness*secant[..., None]*positive
    t[..., 0] += compression*np.minimum(g[..., 0], 0.)
    psi = .5*stiffness*secant*radius**2+.5*compression*np.minimum(g[..., 0], 0.)**2
    dissipation = Gc*np.clip((kappa-onset)/(failure-onset), 0., 1.)
    tangent = np.zeros(g.shape[:-1]+(3, 3))
    for i in range(3):
        tangent[..., i, i] = stiffness*secant
    negative = g[..., 0] < 0
    tangent[..., 0, 0] = np.where(negative, compression, tangent[..., 0, 0])
    loading = (radius > maximum) & (radius > onset) & (radius < failure)
    derivative = np.zeros_like(radius)
    derivative[loading] = -onset*failure/((failure-onset)*radius[loading]**3)
    tangent += stiffness*derivative[..., None, None]*positive[..., :, None]*positive[..., None, :]
    return dict(traction_pa=t, tangent_pa_m=tangent, maximum_separation_m=kappa,
                damage=1-secant, stored_energy_j_m2=psi,
                fracture_dissipation_j_m2=dissipation)


def _path_breaks(law, old, new, maximum):
    """Split a straight vector segment at its actual constitutive events."""
    *_, onset, failure = validate_law(law)
    delta = new-old
    cuts = [0., 1.]
    if delta[0] and 0 < -old[0]/delta[0] < 1:
        cuts.append(float(-old[0]/delta[0]))
    normal_cuts = sorted(cuts)
    for lo, hi in zip(normal_cuts[:-1], normal_cuts[1:]):
        use_normal = (old[0]+.5*(lo+hi)*delta[0]) > 0
        a, b = old.copy(), delta.copy()
        if not use_normal:
            a[0] = b[0] = 0.
        A, B, C = float(b@b), float(2*a@b), float(a@a)
        if not A:
            continue
        for radius in (onset, failure, maximum):
            discriminant = B*B-4*A*(C-radius*radius)
            if discriminant < 0:
                continue
            for root in ((-B-np.sqrt(discriminant))/(2*A),
                         (-B+np.sqrt(discriminant))/(2*A)):
                if lo < root < hi:
                    cuts.append(float(root))
    return sorted(set(cuts))


def integrate_traction_work(law, old_gap, new_gap, old_maximum):
    """Direct quadrature of t(g(t), kappa(t)) dot dg, not endpoint energy.

    Convexity of the unilateral vector norm makes the largest radius on a
    straight interval one of its endpoints.  A consistent old maximum thus
    suffices to integrate every load, unload and compression crossing.
    """
    old, new = array(old_gap), array(new_gap)
    maximum = array(old_maximum)
    if old.shape != new.shape or old.shape[-1:] != (3,) or maximum.shape != old.shape[:-1]:
        raise ValueError("Independent creation segment dimensions differ")
    old_radius = np.linalg.norm(np.column_stack((np.maximum(old.reshape(-1, 3)[:, 0], 0.),
                                                old.reshape(-1, 3)[:, 1:])), axis=-1)
    if np.any(maximum.reshape(-1)+2e-14 < old_radius) or np.any(maximum < 0):
        raise ValueError("Independent creation segment has inconsistent old history")
    nodes, weights = np.polynomial.legendre.leggauss(48)
    result = np.empty(len(old.reshape(-1, 3)))
    for j, (a, b, history) in enumerate(zip(old.reshape(-1, 3), new.reshape(-1, 3), maximum.ravel())):
        delta = b-a; total = 0.
        cuts = _path_breaks(law, a, b, float(history))
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            times = .5*(lo+hi)+.5*(hi-lo)*nodes
            points = a[None]+times[:, None]*delta
            traction = law_point(law, points, np.full(len(nodes), history))["traction_pa"]
            total += .5*(hi-lo)*float(weights@(traction@delta))
        result[j] = total
    return result.reshape(maximum.shape)


def creation_segment(law, old_gap, new_gap, old_maximum):
    # Use the declared scalar history after validating its independently
    # derived old effective length.  Stable hypot avoids false branch changes
    # at a previously reached maximum; the producer declares unloading there.
    old_gap = array(old_gap)
    effective = old_gap.copy(); effective[..., 0] = np.maximum(effective[..., 0], 0.)
    radius = np.hypot.reduce(effective, axis=-1)
    maximum = array(old_maximum)
    tolerance = 32*np.finfo(float).eps*np.maximum(np.maximum(maximum, radius), validate_law(law)[4])
    if np.any(maximum < radius-tolerance):
        raise ValueError("Independent creation segment has inconsistent old history")
    old_maximum = np.maximum(maximum, radius)
    old = law_point(law, old_gap, old_maximum)
    new = law_point(law, new_gap, old_maximum)
    work = integrate_traction_work(law, old_gap, new_gap, old_maximum)
    increment = new["fracture_dissipation_j_m2"]-old["fracture_dissipation_j_m2"]
    balance = new["stored_energy_j_m2"]-old["stored_energy_j_m2"]+increment
    close(work, balance, "direct cohesive work versus energy/dissipation", rtol=2e-10)
    return {**new, "fracture_dissipation_increment_j_m2": increment,
            "mechanical_work_j_m2": work}


def physical_side_intervals(geometry):
    """Check literal depth endpoints against an independently inverted cone."""
    vertices = array(geometry["vertices_m"])
    faces = array(geometry["faces"], integer=True)
    masses = array(geometry["column_mass_kg"])
    density = float(geometry["density_kg_m3"])
    if density <= 0 or len(masses) != len(faces):
        raise ValueError("Independent surface has invalid conical masses")
    coefficients = np.asarray([6*m/(density*np.linalg.det(vertices[row].T))
                               for m, row in zip(masses, faces)])
    if np.any(coefficients <= 0):
        raise ValueError("Independent surface has inward conical faces")
    by_face = [[] for _ in faces]
    for index, cohort in enumerate(geometry["cohorts"]):
        face = int(cohort["face_index"]); coefficient = coefficients[face]
        qlo, qhi = float(cohort["q_lo"]), float(cohort["live_q_hi"])
        if not 0 <= qlo < qhi <= 1 or coefficient*qhi >= 1:
            raise ValueError("Independent surface cohort has invalid live depth")
        expected = np.exp(np.log1p(-coefficient*np.asarray((qhi, qlo)))/3)
        lo, hi = float(cohort["inner_scale"]), float(cohort["outer_scale"])
        close((lo, hi), expected, "source cohort physical depth", rtol=5e-15, atol=0)
        if not 0 < lo < hi <= 1:
            raise ValueError("Independent surface cohort has invalid physical interval")
        by_face[face].append((lo, hi, index))
    for current in by_face:
        ordered = sorted(current)
        if any(a[1] > b[0] for a, b in zip(ordered[:-1], ordered[1:])):
            raise ValueError("Independent surface side cohorts physically overlap")
    return vertices, faces, coefficients, by_face


def _side_q(cohort, radial, coefficient):
    if radial == cohort["outer_scale"]:
        return float(cohort["q_lo"])
    if radial == cohort["inner_scale"]:
        return float(cohort["live_q_hi"])
    return float(-np.expm1(3*np.log(radial))/coefficient)


def surface_oracle(geometry, cut_edges):
    """Pair cones by interval intersection; never pair mass-q or birth index."""
    vertices, faces, coefficients, by_face = physical_side_intervals(geometry)
    topology = reference_topology(vertices, faces, cut_edges)
    cohorts = geometry["cohorts"]
    paired, unmatched = [], []
    normals, tangent1, tangent2 = [], [], []
    for edge_index, (edge, owners) in enumerate(zip(topology["cut_edges"], topology["seam_faces"])):
        u, v = edge; x0, x1 = vertices[u], vertices[v]
        area_factor = np.linalg.norm(np.cross(x0, x1))
        normal = np.cross(x0, x1)/area_factor
        third = next(vertex for vertex in faces[owners[0]] if vertex not in edge)
        if normal@vertices[third] > 0:
            normal = -normal
        opposite = next(vertex for vertex in faces[owners[1]] if vertex not in edge)
        if normal@vertices[opposite] <= 0:
            raise ValueError("Independent surface cannot orient its source sides")
        tangent = (x1-x0)/np.linalg.norm(x1-x0)
        normals.append(normal); tangent1.append(tangent); tangent2.append(np.cross(normal, tangent))
        sides = [by_face[int(face)] for face in owners]
        for left in sides[0]:
            for right in sides[1]:
                lo, hi = max(left[0], right[0]), min(left[1], right[1])
                if lo < hi:
                    paired.append((edge_index, lo, hi, left[2], right[2], area_factor))
        # The solid-only complement is derived per material cohort by
        # subtracting occupied depths of the opposite cone, independently
        # of the producer's combined-endpoint ownership algorithm.
        for side, current in enumerate(sides):
            for lo, hi, index in current:
                remaining = [(lo, hi)]
                for other_lo, other_hi, _ in sides[1-side]:
                    replacement = []
                    for a, b in remaining:
                        if other_hi <= a or other_lo >= b:
                            replacement.append((a, b))
                        else:
                            if a < other_lo:
                                replacement.append((a, other_lo))
                            if other_hi < b:
                                replacement.append((other_hi, b))
                    remaining = replacement
                unmatched.extend((edge_index, a, b, side, index, area_factor)
                                 for a, b in remaining if a < b)
    paired.sort(key=lambda p: (p[0], p[1], p[2]))
    unmatched.sort(key=lambda p: (p[0], p[1], p[2]))
    nedge, npatch, nunmatched = len(topology["cut_edges"]), len(paired), len(unmatched)
    result = dict(cut_edges=topology["cut_edges"], side_face_indices=topology["seam_faces"],
        edge_normals_xyz=np.asarray(normals).reshape(nedge, 3),
        edge_tangent1_xyz=np.asarray(tangent1).reshape(nedge, 3),
        edge_tangent2_xyz=np.asarray(tangent2).reshape(nedge, 3),
        patch_edge_indices=np.asarray([p[0] for p in paired], int),
        side_cohort_indices=np.asarray([[p[3], p[4]] for p in paired], int).reshape(npatch, 2),
        side_cohort_ids=np.asarray([[cohorts[p[3]]["id"], cohorts[p[4]]["id"]]
                                   for p in paired], int).reshape(npatch, 2),
        side_owner_ids=[[cohorts[p[3]]["owner_id"], cohorts[p[4]]["owner_id"]] for p in paired],
        side_material_ids=[[cohorts[p[3]]["material_id"], cohorts[p[4]]["material_id"]] for p in paired],
        side_q_bounds=np.asarray([[[_side_q(cohorts[i], p[2], coefficients[cohorts[i]["face_index"]]),
                                  _side_q(cohorts[i], p[1], coefficients[cohorts[i]["face_index"]])]
                                 for i in (p[3], p[4])] for p in paired]).reshape(npatch, 2, 2),
        lambda_bounds=np.asarray([[p[1], p[2]] for p in paired]).reshape(npatch, 2),
        unmatched_edge_indices=np.asarray([p[0] for p in unmatched], int),
        unmatched_side_indices=np.asarray([p[3] for p in unmatched], int),
        unmatched_cohort_indices=np.asarray([p[4] for p in unmatched], int),
        unmatched_cohort_ids=np.asarray([cohorts[p[4]]["id"] for p in unmatched], int),
        unmatched_lambda_bounds=np.asarray([[p[1], p[2]] for p in unmatched]).reshape(nunmatched, 2),
        unmatched_area_m2=np.asarray([p[5]*(p[2]-p[1])*(p[2]+p[1])/2 for p in unmatched]),
        bank_ndof=6*len(topology["parent_vertex"]), topology=topology)
    gauss, _ = np.polynomial.legendre.leggauss(2)
    points, weights, traces, dofs, samples_s, samples_lam = [], [], [], [], [], []
    patch_areas, edge_areas = [], np.zeros(nedge)
    for edge, lo, hi, left, right, factor in paired:
        s = np.repeat((gauss+1)/2, 2)
        lam = np.tile((lo+hi)/2+(hi-lo)*gauss/2, 2)
        x0, x1 = vertices[topology["cut_edges"][edge]]
        samples_s.append(s); samples_lam.append(lam)
        points.append(lam[:, None]*((1-s[:, None])*x0+s[:, None]*x1))
        weights.append(lam*factor*(hi-lo)/4)
        local_trace = np.zeros((4, 3, 24))
        for point, (where, depth) in enumerate(zip(s, lam)):
            coefficients_trace = np.outer(np.asarray((-1., 1.)), np.asarray((1-where, where))).ravel()
            for endpoint, coefficient in enumerate(coefficients_trace):
                local_trace[point, :, 6*endpoint:6*endpoint+3] = coefficient*np.eye(3)
                local_trace[point, :, 6*endpoint+3:6*endpoint+6] = coefficient*(depth-1)*np.eye(3)
        traces.append(local_trace)
        dofs.append((6*topology["bank_vertices"][edge, :, :, None]+np.arange(6)).ravel())
        area = factor*(hi-lo)*(hi+lo)/2
        patch_areas.append(area); edge_areas[edge] += area
    result.update(quadrature_s=np.asarray(samples_s).reshape(npatch, 4),
        quadrature_lambda=np.asarray(samples_lam).reshape(npatch, 4),
        quadrature_xyz_m=np.asarray(points).reshape(npatch, 4, 3),
        area_weights_m2=np.asarray(weights).reshape(npatch, 4),
        jump_trace=np.asarray(traces).reshape(npatch, 4, 3, 24),
        trace_dofs=np.asarray(dofs, int).reshape(npatch, 24),
        patch_area_m2=np.asarray(patch_areas), edge_area_m2=edge_areas)
    close(result["area_weights_m2"].sum(axis=1), result["patch_area_m2"], "surface area quadrature", atol=0)
    return result


def surface_jump(surface, bank_motion):
    """Direct evaluate endpoint vector fields, rather than multiply trace."""
    motion = array(bank_motion, (surface["bank_ndof"],)).reshape(-1, 6)
    result = np.zeros((len(surface["patch_edge_indices"]), 4, 3))
    for patch, edge in enumerate(surface["patch_edge_indices"]):
        vertices = surface["topology"]["bank_vertices"][edge]
        values = []
        s, lam = surface["quadrature_s"][patch], surface["quadrature_lambda"][patch]
        for bank in vertices:
            amplitudes = motion[bank]
            A = (1-s[:, None])*amplitudes[0, :3]+s[:, None]*amplitudes[1, :3]
            P = (1-s[:, None])*amplitudes[0, 3:]+s[:, None]*amplitudes[1, 3:]
            values.append(A+(lam[:, None]-1)*P)
        result[patch] = values[1]-values[0]
    return result


def surface_force(surface, traction):
    traction = array(traction, (len(surface["patch_edge_indices"]), 4, 3))
    force = np.zeros(surface["bank_ndof"])
    for patch, edge in enumerate(surface["patch_edge_indices"]):
        local = surface["topology"]["bank_vertices"][edge]
        for side, sign in enumerate((-1., 1.)):
            for endpoint, vertex in enumerate(local[side]):
                s, lam = surface["quadrature_s"][patch], surface["quadrature_lambda"][patch]
                shape = s if endpoint else 1-s
                point_forces = sign*shape[:, None]*surface["area_weights_m2"][patch, :, None]*traction[patch]
                force[6*vertex:6*vertex+3] += point_forces.sum(axis=0)
                force[6*vertex+3:6*vertex+6] += ((lam[:, None]-1)*point_forces).sum(axis=0)
    return force


def audit_surface_claims(case, source_geometry):
    expected = surface_oracle(source_geometry, case["cut_edges"])
    if case["bank_ndof"] != expected["bank_ndof"]:
        raise ValueError("Independent creation audit bank dimension differs")
    for key in ("parent_vertex", "faces"):
        if not np.array_equal(array(case["topology"][key], integer=True), expected["topology"][key]):
            raise ValueError(f"Independent creation audit topology {key} differs")
    geometry = case["geometry"]
    errors = {}
    for key, reference in expected.items():
        if key in ("topology", "bank_ndof"):
            continue
        if key in ("side_owner_ids", "side_material_ids"):
            if canonical(geometry[key]) != canonical(reference):
                raise ValueError(f"Independent creation audit {key} differs")
            continue
        raw = array(geometry[key], integer=np.asarray(reference).dtype.kind in "iu")
        if not raw.size:
            raw = raw.reshape(np.shape(reference))
        errors[key] = close(raw, reference, key, rtol=3e-11, atol=2e-12)
    for field in ("source_checkpoint_sha256", "source_identity", "time_myr"):
        if geometry[field] != source_geometry[field]:
            raise ValueError(f"Independent creation audit source binding {field} differs")
    if geometry["scope"] != "mesh_conforming_reference_radial_solid_side_support_not_physical_fracture":
        raise ValueError("Independent creation audit surface scope differs")
    return expected, errors


def audit_case(case, parent):
    if case.get("computational_fixture") is not True:
        raise ValueError("Independent creation audit refuses an unmarked physical fixture")
    if type(case.get("paths")) is not list or not case["paths"]:
        raise ValueError("Independent creation audit requires nonempty computational paths")
    expected, errors = audit_surface_claims(case, parent["runtime_state"]["geometry"])
    geometry = case["geometry"]
    law = case["law"]
    validate_law(law)
    frames = np.stack((expected["edge_normals_xyz"], expected["edge_tangent1_xyz"],
                       expected["edge_tangent2_xyz"]), axis=-1)[expected["patch_edge_indices"]]
    weights = expected["area_weights_m2"]
    result_paths = []
    previous_motion = np.zeros(expected["bank_ndof"])
    previous_maximum = np.zeros(weights.shape)
    previous_local = np.zeros(weights.shape+(3,))
    cumulative_work, cumulative_dissipation = 0., 0.
    for path in case["paths"]:
        old_motion = array(path["old_bank_displacement_from_fixture_origin_m"], (expected["bank_ndof"],))
        new_motion = array(path["bank_displacement_from_fixture_origin_m"], (expected["bank_ndof"],))
        old_world = surface_jump(expected, old_motion)
        world = surface_jump(expected, new_motion)
        old_local = np.einsum("pqa,pab->pqb", old_world, frames)
        local = np.einsum("pqa,pab->pqb", world, frames)
        old_maximum = array(path["old_kappa_m"], weights.shape)
        close(old_motion, previous_motion, "computational path motion continuity", rtol=0, atol=0)
        close(old_maximum, previous_maximum, "computational path irreversible history continuity", rtol=0, atol=0)
        # Geometry is checked independently first.  The law's input vectors
        # are then literal point values, resolving its declared kink side
        # without mistaking quadrature roundoff for a new loading branch.
        published_local = array(path["local_jump_m"], local.shape)
        close(published_local, local, "independently reconstructed local jump")
        close(previous_local, old_local, "independently reconstructed old local jump")
        response = creation_segment(law, previous_local, published_local, old_maximum)
        world_traction = np.einsum("pqb,pab->pqa", response["traction_pa"], frames)
        force = surface_force(expected, world_traction)
        paths_errors = dict(
            world_jump=close(path["world_jump_m"], world, "world jump"),
            local_jump=close(path["local_jump_m"], local, "local jump"),
            traction_world=close(path["traction_world_pa"], world_traction, "world traction"),
        )
        for reported, computed in (
                ("kappa_m", "maximum_separation_m"), ("damage", "damage"),
                ("traction_local_pa", "traction_pa"), ("tangent_local_pa_m", "tangent_pa_m"),
                ("stored_energy_density_j_m2", "stored_energy_j_m2"),
                ("cumulative_dissipation_density_j_m2", "fracture_dissipation_j_m2"),
                ("dissipation_increment_density_j_m2", "fracture_dissipation_increment_j_m2"),
                ("mechanical_work_increment_density_j_m2", "mechanical_work_j_m2")):
            paths_errors[reported] = close(path[reported], response[computed], reported,
                                           rtol=2e-10, atol=2e-10)
        # Cartesian top amplitude and radial-slope forces must each be
        # sensitive; a global A scale could hide a wrong thin P block.
        reported_force = array(path["bank_force_n"], (expected["bank_ndof"],))
        for start, component in ((0, "A"), (3, "P")):
            paths_errors[f"bank_force_{component}"] = close(
                reported_force.reshape(-1, 6)[:, start:start+3], force.reshape(-1, 6)[:, start:start+3],
                f"bank force {component}", rtol=5e-11, atol=1e-5)
        # The dual force check uses a separate vector probe.  Endpoint
        # traction dot whole accepted interval is not cohesive path work.
        probe = np.sin(.117*np.arange(expected["bank_ndof"])+.39)
        trace_work = float(np.einsum("pqa,pqa,pq->", world_traction, surface_jump(expected, probe), weights))
        paths_errors["virtual_work_duality"] = close(force@probe, trace_work,
            "surface virtual work duality", rtol=5e-11, atol=1e-4)
        integrals = {}
        for reported, computed in (
                ("stored_energy_j", "stored_energy_j_m2"),
                ("cumulative_dissipation_j", "fracture_dissipation_j_m2"),
                ("dissipation_increment_j", "fracture_dissipation_increment_j_m2"),
                ("mechanical_work_increment_j", "mechanical_work_j_m2")):
            value = float(np.sum(weights*response[computed]))
            paths_errors[reported] = close(path[reported], value, reported, rtol=2e-10, atol=1e-4)
            integrals[reported] = value
        result_paths.append(dict(name=path["name"], direct_traction_path_work_checked=True,
                                 relative_errors=paths_errors, **integrals))
        cumulative_work += integrals["mechanical_work_increment_j"]
        cumulative_dissipation += integrals["dissipation_increment_j"]
        previous_motion = new_motion
        previous_maximum = array(path["kappa_m"], weights.shape)
        previous_local = published_local
    terminal = case["terminal_state"]
    close(terminal["bank_displacement_from_fixture_origin_m"], previous_motion, "terminal computational motion")
    close(terminal["maximum_separation_m"], previous_maximum, "terminal computational maximum")
    close(terminal["local_gap_m"], local, "terminal computational gap")
    close(terminal["cumulative_fracture_work_j"], cumulative_dissipation, "terminal creation ledger", rtol=2e-10)
    close(terminal["cumulative_cohesive_work_j"], cumulative_work, "terminal cohesive work ledger", rtol=2e-10)
    close(terminal["stored_interface_energy_j"], integrals["stored_energy_j"], "terminal recoverable energy", rtol=2e-10)
    if terminal["physical_creation_admitted"] is not False or terminal["physical_release_admitted"] is not False:
        raise ValueError("Independent creation audit refuses physical admission from a computational state")
    terminal_fields = {"format", "scope", "context_binding_sha256", "surface_binding_sha256",
        "bank_displacement_from_fixture_origin_m", "local_gap_m", "maximum_separation_m",
        "stored_interface_energy_j", "cumulative_fracture_work_j", "cumulative_cohesive_work_j",
        "accepted_path_segments", "physical_creation_admitted", "physical_release_admitted", "state_sha256"}
    if set(terminal) != terminal_fields:
        raise ValueError("Independent creation audit terminal whole schema differs")
    if (terminal["format"] != "genesis-computational-cohesive-surface-history-0.1"
            or terminal["scope"] != "prospective_material_surface_work_kernel_not_physical_Genesis_release"
            or terminal["surface_binding_sha256"] != geometry["binding_sha256"]):
        raise ValueError("Independent creation audit terminal scope or surface identifier differs")
    paths_literal = [path["bank_displacement_from_fixture_origin_m"] for path in case["paths"]]
    if canonical(terminal["accepted_path_segments"]) != canonical(paths_literal):
        raise ValueError("Independent creation audit terminal accepted path catalogue differs")
    terminal_digest = hashlib.sha256(canonical({k: v for k, v in terminal.items() if k != "state_sha256"})).hexdigest()
    if terminal["state_sha256"] != terminal_digest:
        raise ValueError("Independent creation audit terminal state seal differs")
    return dict(name=case["name"], computational_fixture=True,
                paired_patch_count=len(expected["patch_edge_indices"]),
                unmatched_interval_count=len(expected["unmatched_edge_indices"]),
                paired_area_m2=float(expected["patch_area_m2"].sum()),
                nonlinear_spatial_integration="declared_four_point_surface_quadrature",
                surface_binding_sha256_is_opaque_identifier=True,
                computational_context_binding_sha256_is_opaque_identifier=True,
                terminal_whole_schema_seal_and_path_catalogue_checked=True,
                geometry_relative_errors=errors, paths=result_paths)


def audit_context(context, parent, law):
    if law is not None:
        validate_law(law)
    runtime = parent["runtime_state"]
    geometry = runtime["geometry"]
    digest = lambda x: hashlib.sha256(canonical(x)).hexdigest()
    body = dict(format="genesis-volume-creation-context-0.1",
        scope="prospective_material_surface_work_kernel_not_physical_Genesis_release",
        formation_checkpoint_sha256=parent["source_checkpoint_sha256"],
        parent_checkpoint_sha256=parent["checkpoint_sha256"],
        source_identity=geometry["source_identity"], time_myr=geometry["time_myr"],
        accepted_step_sizes_myr=runtime["accepted_step_sizes_myr"],
        cohort_ids=[int(c["id"]) for c in geometry["cohorts"]],
        geometry_sha256=digest(geometry), history_sha256=digest(runtime["history"]),
        mechanics_sha256=digest(runtime["mechanics"]), fault_sha256=digest(runtime["fault_history"]),
        thermal_sha256=digest(runtime["thermal"]),
        inherited_genesis_ledgers=dict(
            runtime={k: v for k, v in runtime.items() if k.endswith("_j")},
            mechanics={k: v for k, v in runtime["mechanics"]["state"].items() if k.endswith("_j")},
            fault={k: v for k, v in runtime["fault_history"].items() if k.endswith("_j")}),
        declared_prospective_creation_law=law,
        parameter_provenance=None if law is None else "explicit_new_constitutive_input_not_historical_toughness",
        source_calculated_creation_law=None, active_weak_plane_faces=[],
        physical_cut_status="no_eligible_cut", physical_creation_admitted=False,
        physical_release_admitted=False, creation_work_debited_from_Genesis_j=0.,
        historical_dissipation_available_for_creation_j=0.,
        existing_damage_or_slip_work_reclassified=False, material_history_reinitialized=False,
        interface_area_convention="paired_Gamma_once_Gc_includes_both_banks")
    expected = {**body, "context_sha256": digest(body)}
    if canonical(context) != canonical(expected):
        raise ValueError("Independent creation audit accepted context or inherited work differs")
    return expected


def audit_fixture(fixture, parent):
    if fixture.get("format") != FORMAT:
        raise ValueError("Independent creation audit format differs")
    if fixture.get("parent_raw_sha256") != ACTUAL_PARENT_RAW_SHA256:
        raise ValueError("Independent creation audit parent raw source differs")
    if canonical(fixture["parent_checkpoint"]) != canonical(parent):
        raise ValueError("Independent creation audit literal accepted parent differs")
    if parent["checkpoint_sha256"] != ACTUAL_PARENT_CHECKPOINT_SHA256:
        raise ValueError("Independent creation audit accepted checkpoint pin differs")
    _source_fields(parent)  # Independent source/source-frame/clock/catalogue gate.
    if "fixture_sha256" in fixture:
        pin = hashlib.sha256(canonical({k: v for k, v in fixture.items() if k != "fixture_sha256"})).hexdigest()
        if fixture["fixture_sha256"] != pin:
            raise ValueError("Independent creation audit fixture seal differs")
    scope = "source_bound_prospective_surface_and_creation_kernel_not_physical_genesis_fracture"
    if fixture.get("scope") != scope or fixture.get("no_physical_fracture_or_contact_admission") is not True:
        raise ValueError("Independent creation audit physical scope differs")
    if (fixture.get("bank_displacement_semantics") !=
            "from_zero_computational_fixture_origin_not_absolute_accepted_Genesis_pose"
            or fixture.get("cohesive_energy_measure") !=
            "one_original_paired_interface_area_counts_both_created_banks"
            or fixture.get("no_prestress_replacement_or_release_equilibrium_claim") is not True):
        raise ValueError("Independent creation audit displacement, area or release semantics differ")
    if type(fixture.get("fixture_cases")) is not list or not fixture["fixture_cases"]:
        raise ValueError("Independent creation audit requires nonempty fixture cases")
    audit_context(fixture["actual_creation_context"], parent, law=None)
    declared_law = fixture["computational_creation_context"]["declared_prospective_creation_law"]
    audit_context(fixture["computational_creation_context"], parent, law=declared_law)
    if any(canonical(case["law"]) != canonical(declared_law) for case in fixture["fixture_cases"]):
        raise ValueError("Independent creation audit case law differs from its computational context")
    cases = [audit_case(case, parent) for case in fixture["fixture_cases"]]
    report = dict(format="independent-genesis-volume-creation-audit-0.1", status="PASS",
        producer_imports=False, parent_raw_sha256=ACTUAL_PARENT_RAW_SHA256,
        parent_checkpoint_sha256=ACTUAL_PARENT_CHECKPOINT_SHA256,
        actual_time_myr=parent["runtime_state"]["geometry"]["time_myr"],
        actual_cohorts=len(parent["runtime_state"]["geometry"]["cohorts"]),
        actual_active_faults=0, actual_no_physical_cut=True,
        physical_fracture_selection_inferred=False, physical_release_admitted=False,
        surface_area_counts_paired_interface_once=True,
        historical_slip_or_damage_work_credited_as_creation=False, cases=cases)
    report["sha256"] = hashlib.sha256(canonical(report)).hexdigest()
    return report


def birth_identity_delta(previous_geometry, current_geometry):
    """Preserve material identity while allowing a calculated front to grow."""
    before = {int(c["id"]): c for c in previous_geometry["cohorts"]}
    after = {int(c["id"]): c for c in current_geometry["cohorts"]}
    if len(before) != len(previous_geometry["cohorts"]) or len(after) != len(current_geometry["cohorts"]):
        raise ValueError("Independent growth audit refuses duplicate material IDs")
    if not set(before) <= set(after):
        raise ValueError("Independent growth audit lost an accepted live cohort")
    for identity, old in before.items():
        for field in ("id", "face_index", "layer_index", "q_lo", "material_id", "owner_id"):
            if canonical(old.get(field)) != canonical(after[identity].get(field)):
                raise ValueError("Independent growth audit replaced accepted material identity")
    return sorted(set(after)-set(before))


def audit_endpoint_energy(packet):
    """Independently integrate full inherited memory and consistent inertia.

    This checks endpoint quantities only, without claiming to rerun the
    accepted time solver or reconstruct unavailable early 3D history.
    """
    runtime, source = packet["runtime_state"], packet["source_packet"]
    geometry = runtime["geometry"]
    vertices, faces = array(geometry["vertices_m"]), array(geometry["faces"], integer=True)
    masses, rho = array(geometry["column_mass_kg"]), float(geometry["density_kg_m3"])
    cohorts = geometry["cohorts"]
    memory = {int(c["id"]): c for c in runtime["mechanics"]["state"]["cohorts"]}
    if (len(memory) != len(runtime["mechanics"]["state"]["cohorts"])
            or set(memory) != {int(c["id"]) for c in cohorts}):
        raise ValueError("Independent growth audit memory lost or invented material IDs")
    shell = source["source"]["parameters"]["shell"]
    energy = 0.
    for cohort in cohorts:
        face, identity = int(cohort["face_index"]), int(cohort["id"])
        C = isotropic_matrix(shell["young_modulus_pa"], shell["poisson_ratio"],
            runtime["fault_history"]["damage"][face], shell["residual_stiffness"])
        energy += coefficient_energy(vertices[faces[face]], masses[face], rho, cohort,
            memory[identity]["strain0"], memory[identity]["strain1"], C)
        close(cohort["mass_kg"], masses[face]*(cohort["live_q_hi"]-cohort["q_lo"]),
              "accepted cohort live mass", rtol=3e-13, atol=0)
    M = mass_matrix(vertices, faces, masses, rho, cohorts)
    velocity = array(runtime["velocity_m_s"], (len(M),))
    kinetic = float(.5*velocity@M@velocity)
    elastic_error = close(runtime["mechanics"]["state"]["elastic_energy_j"], energy,
                           "independent accepted full-cohort energy", rtol=5e-12, atol=1e4)
    kinetic_error = close(runtime["kinetic_energy_j"], kinetic,
                           "independent accepted consistent kinetic energy", rtol=5e-10, atol=1e-6)
    return dict(elastic_energy_j=energy, kinetic_energy_j=kinetic,
                elastic_relative_error=elastic_error, kinetic_relative_error=kinetic_error,
                total_live_mass_kg=float(sum(c["mass_kg"] for c in cohorts)))


def audit_growth_fixture(fixture, parent, directory):
    """Audit source-bound real birth endpoints and computational side support."""
    if (fixture.get("format") != GROWTH_FORMAT or fixture.get("scope") !=
            "real_Genesis_material_birth_growth_with_prospective_uncut_side_support"):
        raise ValueError("Independent growth audit format or physical scope differs")
    if (fixture.get("parent_raw_sha256") != ACTUAL_PARENT_RAW_SHA256
            or canonical(fixture.get("parent_checkpoint")) != canonical(parent)
            or fixture.get("formation_checkpoint_sha256") != parent["source_checkpoint_sha256"]):
        raise ValueError("Independent growth audit original accepted source differs")
    if fixture.get("requested_dt_years") != 100. or fixture.get("real_accepted_intervals") != 2:
        raise ValueError("Independent growth audit interval control differs")
    for flag in ("unchanged_source_forcing_birth_and_activation_parameters", "continuous_material_history",
                 "no_replacement_initialization", "surface_supports_are_computational_fixtures_only",
                 "no_new_creation_law_applied_to_Genesis", "no_physical_fracture_or_contact_admission"):
        if fixture.get(flag) is not True:
            raise ValueError("Independent growth audit changed its admitted source scope")
    if fixture.get("fixture_sha256") != hashlib.sha256(canonical({k: v for k, v in fixture.items()
                                                                if k != "fixture_sha256"})).hexdigest():
        raise ValueError("Independent growth audit fixture seal differs")
    snapshots = fixture.get("snapshots")
    if type(snapshots) is not list or [s.get("name") for s in snapshots] != ["initial", "after_first", "direct_after_second"]:
        raise ValueError("Independent growth audit requires all three calculated endpoints")
    source, first_runtime = parent["source_packet"], parent["runtime_state"]
    previous = parent
    reports, previous_raw = [], None
    for index, snapshot in enumerate(snapshots):
        if snapshot.get("computational_fixture") is not True:
            raise ValueError("Independent growth audit side surface is not marked computational")
        if snapshot.get("actual_physical_surface_created") is not False or snapshot.get("actual_creation_work_assigned") is not False:
            raise ValueError("Independent growth audit refuses uncalculated physical creation")
        name = snapshot["checkpoint_file"]
        if type(name) is not str or Path(name).name != name:
            raise ValueError("Independent growth audit checkpoint must be a local evidence member")
        raw = (Path(directory)/name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != snapshot["checkpoint_raw_sha256"]:
            raise ValueError("Independent growth audit saved checkpoint raw pin differs")
        checkpoint = snapshot["checkpoint"]
        if canonical(json.loads(raw)) != canonical(checkpoint):
            raise ValueError("Independent growth audit snapshot differs from saved checkpoint")
        body = {k: v for k, v in checkpoint.items() if k != "checkpoint_sha256"}
        if checkpoint["checkpoint_sha256"] != hashlib.sha256(canonical(body)).hexdigest():
            raise ValueError("Independent growth audit checkpoint self seal differs")
        if (checkpoint.get("format") != "genesis-volume-slip-checkpoint-0.1"
                or checkpoint.get("scope") != "actual_genesis_source_replayed_connected_volume_finite_width_weak_plane_slip"
                or checkpoint["source_checkpoint_sha256"] != parent["source_checkpoint_sha256"]
                or canonical(checkpoint["source_packet"]) != canonical(source)):
            raise ValueError("Independent growth audit rewrote the formation source or continuation model")
        runtime, geometry = checkpoint["runtime_state"], checkpoint["runtime_state"]["geometry"]
        time = first_runtime["geometry"]["time_myr"]+index*100/1e6
        close(geometry["time_myr"], time, "real growth shared clock", rtol=0, atol=2e-14)
        endpoint_clock_binding(runtime, geometry["time_myr"])
        if (runtime["fault_history"]["time_myr"] != geometry["time_myr"]
                or runtime["fault_history"]["accepted_steps"] != runtime["mechanics"]["state"]["accepted_steps"]):
            raise ValueError("Independent growth audit fault history has a different accepted clock")
        if snapshot["time_myr"] != geometry["time_myr"] or snapshot["cohort_count"] != 640+80*index or len(geometry["cohorts"]) != snapshot["cohort_count"]:
            raise ValueError("Independent growth audit actual clock or cohort count differs")
        expected_steps = first_runtime["accepted_step_sizes_myr"]+[100/1e6]*index
        if (runtime["accepted_step_sizes_myr"] != expected_steps
                or runtime["mechanics"]["state"]["accepted_steps"] != first_runtime["mechanics"]["state"]["accepted_steps"]+index):
            raise ValueError("Independent growth audit accepted interval ancestry differs")
        for field in ("vertices_m", "faces", "column_mass_kg", "density_kg_m3", "source_identity", "source_checkpoint_sha256"):
            if canonical(geometry[field]) != canonical(first_runtime["geometry"][field]):
                raise ValueError("Independent growth audit changed immutable reference or source material")
        for field in ("contact_parameters_json", "activation_closure", "damage_coupling", "kinematics",
                      "plane_transport", "slip_closure", "velocity_closure", "mechanical_work_closure"):
            if canonical(runtime[field]) != canonical(first_runtime[field]):
                raise ValueError("Independent growth audit changed source forcing, activation or constitutive closure")
        if any(runtime["fault_history"]["fault_active"]):
            raise ValueError("Independent growth audit manufactured an active fault")
        audit_context(snapshot["creation_context"], checkpoint, law=None)
        if index == 0 and (hashlib.sha256(raw).hexdigest() != ACTUAL_PARENT_RAW_SHA256 or canonical(checkpoint) != canonical(parent)):
            raise ValueError("Independent growth audit initial endpoint differs from accepted parent")
        new_ids = birth_identity_delta(previous["runtime_state"]["geometry"], geometry)
        if snapshot["new_cohort_ids"] != new_ids:
            raise ValueError("Independent growth audit newborn material IDs differ")
        old_history = previous["runtime_state"]["history"]
        history = runtime["history"]
        old_birth = {int(c["id"]): c for c in old_history["cohorts"]}
        birth = {int(c["id"]): c for c in history["cohorts"]}
        if len(birth) != len(history["cohorts"]) or set(birth) != {int(c["id"]) for c in geometry["cohorts"]}:
            raise ValueError("Independent growth audit material birth history differs from live catalogue")
        for cohort in geometry["cohorts"]:
            for field in ("face_index", "layer_index", "q_lo", "live_q_hi"):
                if birth[int(cohort["id"])][field] != cohort[field]:
                    raise ValueError("Independent growth audit live side depth differs from literal birth record")
        for identity, record in old_birth.items():
            if canonical(record) != canonical(birth[identity]):
                raise ValueError("Independent growth audit rewrote an accepted birth record")
        if index:
            if history["intervals"][:-1] != old_history["intervals"]:
                raise ValueError("Independent growth audit rewrote accepted material time history")
            interval = history["intervals"][-1]
            if interval["before_time_myr"] != old_history["time_myr"] or interval["after_time_myr"] != history["time_myr"]:
                raise ValueError("Independent growth audit newborn interval clock differs")
            for identity in new_ids:
                record = birth[identity]
                if (record["birth_before_time_myr"] != old_history["time_myr"]
                        or record["birth_after_time_myr"] != history["time_myr"]):
                    raise ValueError("Independent growth audit newborn material lacks its calculated time interval")
                cohort = next(c for c in geometry["cohorts"] if int(c["id"]) == identity)
                for field in ("face_index", "layer_index", "q_lo", "live_q_hi"):
                    if record[field] != cohort[field]:
                        raise ValueError("Independent growth audit newborn side depth differs from birth record")
        surface, errors = audit_surface_claims(snapshot, geometry)
        new_set = set(new_ids)
        cut_faces = set(surface["side_face_indices"].ravel().tolist())
        cut_ids = sorted(int(c["id"]) for c in geometry["cohorts"] if int(c["id"]) in new_set and c["face_index"] in cut_faces)
        paired_ids = sorted(new_set & set(surface["side_cohort_ids"].ravel().tolist()))
        unmatched_ids = sorted(new_set & set(surface["unmatched_cohort_ids"].tolist()))
        for key, expected in (("new_cut_face_cohort_ids", cut_ids),
                ("new_cohort_ids_in_paired_surface", paired_ids), ("new_cohort_ids_in_unmatched_surface", unmatched_ids)):
            if snapshot[key] != expected:
                raise ValueError(f"Independent growth audit {key} differs")
        patches_new = int(np.isin(surface["side_cohort_ids"], new_ids).any(axis=1).sum())
        if snapshot["paired_patches_containing_newborn_material"] != patches_new or not set(cut_ids) <= set(paired_ids+unmatched_ids):
            raise ValueError("Independent growth audit newborn material is missing from its side support")
        reports.append(dict(name=snapshot["name"], time_myr=geometry["time_myr"], cohort_count=len(geometry["cohorts"]),
            checkpoint_raw_sha256=snapshot["checkpoint_raw_sha256"], checkpoint_sha256=checkpoint["checkpoint_sha256"],
            newborn_cohort_count=len(new_ids), newborn_cut_face_ids=cut_ids, newborn_paired_ids=paired_ids,
            newborn_unmatched_ids=unmatched_ids, paired_patches_containing_newborn_material=patches_new,
            paired_patch_count=len(surface["patch_edge_indices"]), unmatched_interval_count=len(surface["unmatched_edge_indices"]),
            paired_area_m2=float(surface["patch_area_m2"].sum()), geometry_relative_errors=errors,
            endpoint_energy=audit_endpoint_energy(checkpoint), physical_creation_admitted=False))
        previous, previous_raw = checkpoint, raw
    resumed = (Path(directory)/"resumed_after_second.json").read_bytes()
    if previous_raw != resumed:
        raise ValueError("Independent growth audit entire direct and resumed real archives differ")
    report = dict(format="independent-genesis-volume-creation-birth-growth-audit-0.1", status="PASS",
        producer_imports=False, parent_raw_sha256=ACTUAL_PARENT_RAW_SHA256,
        original_formation_source_unchanged=True, real_accepted_intervals=2,
        snapshots=reports, all_calculated_birth_ids_and_side_supports_checked=True,
        full_cohort_endpoint_energy_and_inertia_checked=True,
        complete_accepted_time_solver_independently_reexecuted=False,
        actual_source_creation_law_unknown=True, actual_active_faults=0,
        historical_slip_or_damage_work_credited_as_creation=False,
        entire_direct_and_resumed_real_archive_bytes_equal=True,
        surface_supports_are_computational_fixtures_only=True,
        physical_fracture_selection_inferred=False, physical_release_admitted=False)
    report["sha256"] = hashlib.sha256(canonical(report)).hexdigest()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fixture", type=Path)
    mode.add_argument("--growth-fixture", type=Path)
    parser.add_argument("--parent-checkpoint", type=Path, default=Path(__file__).resolve().parents[2]/
        "analysis/genesis_volume_slip_validation/evidence/actual_control/direct_after_second.json")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    parent = _actual_parent(arguments.parent_checkpoint)
    if arguments.growth_fixture is not None:
        fixture = json.loads(arguments.growth_fixture.read_bytes())
        report = audit_growth_fixture(fixture, parent, arguments.growth_fixture.parent)
    else:
        fixture = json.loads(arguments.fixture.read_bytes())
        report = audit_fixture(fixture, parent)
    arguments.output.write_bytes(canonical(report)+b"\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
