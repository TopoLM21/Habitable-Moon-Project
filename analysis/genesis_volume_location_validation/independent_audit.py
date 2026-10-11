"""Producer-free current material stress and explicit-plane section oracle.

All pointwise fields are read from the accepted cohort memory and declared
material chart.  A supplied plane is a computational section, never a
calculated physical fracture location, release, or source creation law.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.genesis_volume_continuation_validation.independent_audit import (
    SOURCE_ANGULAR_POINTS, conical_scales, isotropic_matrix, radial_rule,
)
from analysis.genesis_volume_insertion_validation.independent_audit import (
    PARENT_RAW_SHA256, PARENT_CHECKPOINT_SHA256, FORMATION_SHA256, audit_onset,
)

FORMAT = "genesis-volume-material-location-control-0.1"
SCOPE = "accepted_full_material_stress_and_declared_current_sections_not_physical_localization"
STRESS_SCOPE = "accepted_endpoint_elastic_memory_at_explicit_material_coordinates_no_fracture_location"
FALSE_STRESS = ("history_reinitialized", "clock_advanced", "activation_accepted",
    "physical_creation_admitted", "physical_release_admitted",
    "physical_fault_contact_admission_claimed", "historical_three_dimensional_memory_recovered",
    "instantaneous_future_birth_traction_claimed")
EXPLICIT_NORMALS = ((1., 2., 3.), (-2., 1., .5), (.7, -1.3, 1.1))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError("Independent material location audit: "+message)


def array(value, shape=None):
    def contains_boolean(item):
        return isinstance(item, (bool, np.bool_)) or (isinstance(item, (list, tuple))
            and any(contains_boolean(child) for child in item))
    require(not contains_boolean(value), "boolean is not a material coordinate")
    raw = np.asarray(value)
    require(raw.dtype.kind in "fiu" and (shape is None or raw.shape == shape)
            and np.isfinite(raw).all(), "invalid finite numeric array")
    return np.asarray(raw, float)


def close(actual, expected, name, *, rtol=8e-11, atol=1e-8):
    a, b = array(actual), array(expected)
    require(a.shape == b.shape, name+" dimensions differ")
    scale = max(float(np.max(abs(b), initial=0.)), np.finfo(float).tiny)
    error = float(np.max(abs(a-b), initial=0.))
    require(error <= atol+rtol*scale, name+" differs")
    return error/max(scale, atol, np.finfo(float).tiny)


def seal(packet, name):
    require(type(packet) is dict and packet.get(name) == digest(
        {k: v for k, v in packet.items() if k != name}), name+" seal differs")


def tensor(voigt, *, engineering=False):
    a = array(voigt)
    require(a.shape[-1:] == (6,), "six tensor components required")
    result = np.zeros(a.shape[:-1]+(3, 3))
    result[..., np.arange(3), np.arange(3)] = a[..., :3]
    for i, j, k in ((0, 1, 3), (0, 2, 4), (1, 2, 5)):
        result[..., i, j] = result[..., j, i] = a[..., k]/(2 if engineering else 1)
    return result


def source_frames(parent):
    tangents = array(parent["source_packet"]["current_geometry"]["face_frames_xyz"])
    return np.concatenate((tangents, np.cross(tangents[:, :, 0],
        tangents[:, :, 1])[:, :, None]), axis=2)


def material_samples(parent, cohort_ids, angular_barycentric, mass_coordinate_q,
                     *, normal_world=None):
    """Rebuild full local memory with an independent affine vertex lift."""
    runtime = parent["runtime_state"]; geometry = runtime["geometry"]
    require(not any(isinstance(i, (bool, np.bool_)) for i in cohort_ids), "integer cohort IDs required")
    raw_ids = np.asarray(cohort_ids)
    require(raw_ids.ndim == 1 and raw_ids.dtype.kind in "iu", "integer cohort IDs required")
    identities = [int(i) for i in raw_ids]
    N, q = array(angular_barycentric, (len(identities), 3)), array(mass_coordinate_q, (len(identities),))
    require(np.all(N >= 0) and np.allclose(N.sum(axis=1), 1., rtol=0, atol=8*np.finfo(float).eps),
            "barycentric samples leave their material face")
    catalogue = {int(c["id"]): c for c in geometry["cohorts"]}
    memories = {int(c["id"]): c for c in runtime["mechanics"]["state"]["cohorts"]}
    require(all(i in catalogue and i in memories for i in identities), "sample lost live cohort identity")
    faces = np.asarray(geometry["faces"], int); vertices = array(geometry["vertices_m"])
    pose = array(runtime["displacement_m"]).reshape(-1, 6)
    frames = source_frames(parent)
    shell = parent["source_packet"]["source"]["parameters"]["shell"]
    # Source quadrature values determine an affine angular field. Lifting to
    # its three vertices is separate from producer interpolation at the probe.
    inverse = np.linalg.solve(SOURCE_ANGULAR_POINTS, np.eye(3))
    fields = {k: [] for k in ("radial_scale", "reference_position_m", "current_position_m",
        "elastic_strain_local6", "elastic_strain_local_tensor", "elastic_strain_world_tensor",
        "stress_local6", "stress_local_tensor", "stress_world_tensor", "elastic_energy_density_j_m3")}
    face_indices, owners, materials = [], [], []
    for j, cid in enumerate(identities):
        c, memory = catalogue[cid], memories[cid]
        require(float(c["q_lo"]) <= q[j] <= float(c["live_q_hi"]), "sample outside live cohort mass interval")
        f = int(c["face_index"]); face_indices.append(f)
        X = vertices[faces[f]]; A, P = pose[faces[f], :3], pose[faces[f], 3:]
        lam = float(conical_scales(X, geometry["column_mass_kg"][f], geometry["density_kg_m3"], q[j])[0])
        e0 = inverse@array(memory["strain0"], (3, 6))
        e1 = inverse@array(memory["strain1"], (3, 6))
        strain = N[j]@e0+(N[j]@e1)/lam
        C = isotropic_matrix(shell["young_modulus_pa"], shell["poisson_ratio"],
            runtime["fault_history"]["damage"][f], shell["residual_stiffness"])
        stress = C@strain; et, st = tensor(strain, engineering=True), tensor(stress)
        Q = frames[f]
        values = dict(radial_scale=lam, reference_position_m=lam*(N[j]@X),
            current_position_m=N[j]@(lam*(X+P)+(A-P)), elastic_strain_local6=strain,
            elastic_strain_local_tensor=et, elastic_strain_world_tensor=Q@et@Q.T,
            stress_local6=stress, stress_local_tensor=st, stress_world_tensor=Q@st@Q.T,
            elastic_energy_density_j_m3=.5*float(strain@stress))
        for k, value in values.items(): fields[k].append(value)
        owners.append(c["owner_id"]); materials.append(c["material_id"])
    shapes = {"radial_scale": (), "elastic_energy_density_j_m3": (), "reference_position_m": (3,), "current_position_m": (3,),
        "elastic_strain_local6": (6,), "stress_local6": (6,)}
    fields = {k: np.asarray(v).reshape((len(identities),)+shapes.get(k, (3, 3))) for k, v in fields.items()}
    result = dict(cohort_ids=np.asarray(identities), face_indices=np.asarray(face_indices),
        angular_barycentric=N, mass_coordinate_q=q, owner_ids=owners, material_ids=materials, **fields)
    if normal_world is not None:
        normals = array(normal_world, (len(identities), 3))
        require(np.allclose(np.linalg.norm(normals, axis=1), 1., rtol=0, atol=2e-12), "traction normals must be unit")
        result.update(normal_world=normals, traction_world_pa=np.einsum("pij,pj->pi", fields["stress_world_tensor"], normals))
    return result


def plane_parameters(normal_xyz, offset_m):
    normal = array(normal_xyz, (3,))
    offset = array(offset_m, ())
    length = float(np.linalg.norm(normal))
    require(length > 0, "explicit world plane has zero normal")
    return normal/length, float(offset)/length


def section_coefficients(X, A, P, normal_xyz, offset_m):
    X, A, P = (array(v, (3, 3)) for v in (X, A, P))
    n, d = plane_parameters(normal_xyz, offset_m)
    Y, Z = X+P, A-P
    return dict(Y=Y, Z=Z, normal_xyz=n, offset_m=d, y=Y@n, z=Z@n)


def section_at_scale(coefficients, radial_scale):
    """Intersect a complete constant-lambda material triangle with a plane.

    The oracle uses barycentric vertices and every edge pair. It does not
    interpolate the cohort's geometry tetrahedra or snap near-plane vertices.
    """
    lam = float(array(radial_scale, ()))
    require(lam > 0, "section radial scale must be positive")
    y, z, d = coefficients["y"], coefficients["z"], coefficients["offset_m"]
    phi = lam*y+z-d
    if np.all(phi == 0):
        return dict(kind="full_triangle", barycentric=np.eye(3),
                    current_points_m=lam*coefficients["Y"]+coefficients["Z"])
    points, edges = [], []
    for i in range(3):
        if phi[i] == 0:
            points.append(np.eye(3)[i]); edges.append((i, i))
    for i, j in ((0, 1), (0, 2), (1, 2)):
        if phi[i]*phi[j] < 0:
            s = -phi[i]/(phi[j]-phi[i]); bary = np.zeros(3)
            bary[i], bary[j] = 1-s, s
            points.append(bary); edges.append((i, j))
    N = np.asarray(points).reshape(-1, 3)
    kind = "empty" if len(N) == 0 else "touch" if len(N) == 1 else "segment"
    require(len(N) <= 2, "non-coplanar triangle intersection has excess endpoints")
    xyz = N@(lam*coefficients["Y"]+coefficients["Z"])
    return dict(kind=kind, barycentric=N, current_points_m=xyz, edges=edges)


def section_breaks(coefficients, inner_scale, outer_scale):
    lo, hi = float(inner_scale), float(outer_scale)
    require(0 < lo < hi, "section requires positive resolved cohort depth")
    values = [lo, hi]
    for y, z in zip(coefficients["y"], coefficients["z"]):
        if y != 0:
            crossing = (coefficients["offset_m"]-z)/y
            if lo < crossing < hi: values.append(float(crossing))
    return np.asarray(sorted(set(values)))


def section_point(coefficients, radial_scale, *, fraction=.5, barycentric=None):
    cut = section_at_scale(coefficients, radial_scale)
    if cut["kind"] == "full_triangle":
        require(barycentric is not None, "full triangle section needs an explicit barycentric point")
        N = array(barycentric, (3,))
        require(np.all(N >= 0) and abs(N.sum()-1) <= 8*np.finfo(float).eps, "invalid full triangle barycentric point")
    else:
        require(cut["kind"] == "segment", "section probe needs a nonempty segment")
        s = float(array(fraction, ()))
        require(0 <= s <= 1, "section fraction outside segment")
        # Coordinate order is independent from producer's directed-edge
        # traversal. Comparisons use explicit N or set-valued endpoints.
        order = np.lexsort(cut["barycentric"].T[::-1])
        endpoints = cut["barycentric"][order]
        N = (1-s)*endpoints[0]+s*endpoints[1]
    xyz = N@(float(radial_scale)*coefficients["Y"]+coefficients["Z"])
    return dict(angular_barycentric=N, current_position_m=xyz)


def section_measure(coefficients, inner_scale, outer_scale, *, order=12,
                    include_inner_boundary=True, include_outer_boundary=True):
    """Independent geometric area integral of the ruled current section.

    This helper is tested on mathematical fixtures. It is not an accepted
    interface-area rule or a material creation-energy/force admission. Full
    triangles on the two closed depth boundaries belong to both adjacent
    closed cohort supports; an additive material partition must supply a
    single-sided boundary ownership explicitly.
    """
    breaks = section_breaks(coefficients, inner_scale, outer_scale)
    x, w = np.polynomial.legendre.leggauss(order)
    fractions, fraction_weights = (x+1)/2, w/2
    area = 0.; moment = np.zeros(3)
    for lo, hi in zip(breaks, breaks[1:]):
        for lam, weight in zip(lo+(hi-lo)*(x+1)/2, (hi-lo)*w/2):
            cut = section_at_scale(coefficients, lam)
            if cut["kind"] != "segment": continue
            N, derivatives = cut["barycentric"], []
            for i, j in cut["edges"]:
                if i == j:
                    derivatives.append(np.zeros(3)); continue
                denominator = lam*(coefficients["y"][j]-coefficients["y"][i])+coefficients["z"][j]-coefficients["z"][i]
                numerator = coefficients["offset_m"]-lam*coefficients["y"][i]-coefficients["z"][i]
                ds = (-coefficients["y"][i]*denominator-numerator*(coefficients["y"][j]-coefficients["y"][i]))/denominator**2
                derivative = np.zeros(3); derivative[i], derivative[j] = -ds, ds
                derivatives.append(derivative)
            dN = np.asarray(derivatives)
            for fraction, wf in zip(fractions, fraction_weights):
                bary = (1-fraction)*N[0]+fraction*N[1]
                dbary = (1-fraction)*dN[0]+fraction*dN[1]
                X = lam*coefficients["Y"]+coefficients["Z"]
                point = bary@X
                dlam = dbary@X+bary@coefficients["Y"]
                dfraction = (N[1]-N[0])@X
                jacobian = float(np.linalg.norm(np.cross(dlam, dfraction)))
                area += weight*wf*jacobian; moment += weight*wf*jacobian*point
    # A full triangle at one constant lambda is a legitimate plane section
    # with positive area, although it has zero width in this parametrization.
    for lam in breaks:
        if ((lam == inner_scale and not include_inner_boundary)
                or (lam == outer_scale and not include_outer_boundary)):
            continue
        cut = section_at_scale(coefficients, lam)
        if cut["kind"] == "full_triangle":
            pts = cut["current_points_m"]
            a = .5*np.linalg.norm(np.cross(pts[1]-pts[0], pts[2]-pts[0]))
            area += a; moment += a*pts.mean(axis=0)
    return dict(area_m2=float(area), first_moment_m3=moment)


def field_variance(parent):
    """Observe all current saved angular/radial variations, without an event."""
    cohorts = parent["runtime_state"]["geometry"]["cohorts"]
    ids, bary, q = [], [], []
    for c in cohorts:
        for depth in (c["q_lo"], c["live_q_hi"]):
            for N in np.eye(3):
                ids.append(c["id"]); bary.append(N); q.append(depth)
    values = material_samples(parent, ids, bary, q)
    stress = values["stress_local6"].reshape(len(cohorts), 2, 3, 6)
    strain = values["elastic_strain_local6"].reshape(len(cohorts), 2, 3, 6)
    radial = np.linalg.norm(stress[:, 1]-stress[:, 0], axis=2)
    angular = np.linalg.norm(stress-stress.mean(axis=2, keepdims=True), axis=3)
    return dict(cohorts_examined=len(cohorts), samples_examined=len(ids),
        maximum_radial_stress_difference_pa=float(radial.max(initial=0.)),
        maximum_angular_stress_difference_pa=float(angular.max(initial=0.)),
        maximum_elastic_strain_component=float(abs(strain).max(initial=0.)),
        radial_coefficient_memory_nonzero=any(np.any(np.asarray(c["strain1"]) != 0)
            for c in parent["runtime_state"]["mechanics"]["state"]["cohorts"]))


def fixed(raw, expected, label):
    for key, value in expected.items():
        require(key in raw and canonical(raw[key]) == canonical(value), label+" "+key+" differs")


def stress_claims():
    return dict(scope=STRESS_SCOPE,
        strain_interpolation_closure="affine_material_barycentric_and_inverse_reference_radial_scale",
        placement_closure="accepted_conical_A_P_displacement_on_original_material_mass_chart",
        stress_scope="accepted_endpoint_not_interval_average_or_future_birth",
        selected_sharp_material_surface=None, source_creation_energy_law=None,
        creation_work_debited_from_Genesis_j=0., **{name: False for name in FALSE_STRESS})


def audit_observer(raw, parent):
    seal(raw, "binding_sha256")
    runtime = parent["runtime_state"]; geometry = runtime["geometry"]
    cohorts = geometry["cohorts"]
    memories = {int(c["id"]): c for c in runtime["mechanics"]["state"]["cohorts"]}
    face = np.asarray([c["face_index"] for c in cohorts], int)
    shell = parent["source_packet"]["source"]["parameters"]["shell"]
    fixed_fields = dict(format="genesis-volume-material-stress-observer-0.1",
        parent_checkpoint_sha256=parent["checkpoint_sha256"],
        formation_checkpoint_sha256=parent["source_checkpoint_sha256"],
        source_identity=geometry["source_identity"], time_myr=geometry["time_myr"],
        accepted_steps=runtime["mechanics"]["state"]["accepted_steps"],
        accepted_step_sizes_myr=runtime["accepted_step_sizes_myr"],
        geometry=geometry,
        displacement_m=runtime["displacement_m"], velocity_m_s=runtime["velocity_m_s"],
        cohort_ids=[c["id"] for c in cohorts], face_indices=face.tolist(),
        face_damage=runtime["fault_history"]["damage"],
        elastic_constant=[memories[c["id"]]["strain0"] for c in cohorts],
        elastic_inverse_scale=[memories[c["id"]]["strain1"] for c in cohorts], **stress_claims())
    fixed(raw, fixed_fields, "full accepted observer")
    require(set(raw) == set(fixed_fields)|{"basis_xyz", "elasticity_pa", "binding_sha256"},
            "whole observer schema differs")
    basis_error = close(raw["basis_xyz"], source_frames(parent)[face], "inherited source bases", atol=2e-14)
    elasticity = [isotropic_matrix(shell["young_modulus_pa"], shell["poisson_ratio"],
        runtime["fault_history"]["damage"][f], shell["residual_stiffness"]) for f in face]
    C_error = close(raw["elasticity_pa"], elasticity, "current material constitutive law", atol=1e-3)
    return dict(full_source_geometry_pose_velocity_and_memory_checked=True,
        all_current_material_constitutive_parameters_checked=True,
        basis_relative_error=basis_error, elasticity_relative_error=C_error,
        cohort_count=len(cohorts), observer_binding_sha256=raw["binding_sha256"])


def audit_samples(raw, parent, observer):
    seal(raw, "sample_sha256")
    expected = material_samples(parent, raw["cohort_ids"], raw["angular_barycentric"],
        raw["mass_coordinate_q"], normal_world=raw.get("normal_world"))
    fixed_fields = dict(format="genesis-volume-material-stress-samples-0.1",
        parent_checkpoint_sha256=parent["checkpoint_sha256"],
        formation_checkpoint_sha256=parent["source_checkpoint_sha256"],
        source_identity=parent["runtime_state"]["geometry"]["source_identity"],
        time_myr=parent["runtime_state"]["geometry"]["time_myr"],
        observer_binding_sha256=observer["binding_sha256"],
        supplied_normal_is_fracture_orientation=False, **stress_claims())
    for name in ("owner_ids", "material_ids"):
        fixed_fields[name] = expected[name]
    if raw.get("normal_world") is None:
        fixed_fields.update(normal_world=None, traction_world_pa=None)
    fixed(raw, fixed_fields, "full source-bound pointwise sample")
    fields = set(expected)-{"owner_ids", "material_ids"}
    require(set(raw) == set(fixed_fields)|fields|{"sample_sha256"}, "whole pointwise sample schema differs")
    errors = {name: close(raw[name], expected[name], "pointwise material "+name,
        rtol=2e-10, atol=1e-3 if "stress" in name or "traction" in name else
        2e-8 if "position" in name else 1e-12 if "energy" in name else 2e-14)
        for name in sorted(fields)}
    return dict(samples_examined=len(expected["cohort_ids"]), source_local_six_strain_memory_checked=True,
        full_current_A_and_P_placement_checked=True, inherited_source_frame_stress_checked=True,
        current_damage_elasticity_and_energy_density_checked=True,
        independent_field_relative_errors=errors, sample_sha256=raw["sample_sha256"])


def audit_location(raw, parent, onset):
    seal(raw, "location_sha256")
    runtime, geometry = parent["runtime_state"], parent["runtime_state"]["geometry"]
    expected = dict(format="genesis-source-bound-material-plane-location-family-0.1",
        scope="calculated_orientation_full_current_offset_family_without_surface_selection",
        current_material_chart="x(N,lambda)=N@(lambda*(X+P)+(A-P))",
        parent_checkpoint_sha256=parent["checkpoint_sha256"],
        formation_checkpoint_sha256=parent["source_checkpoint_sha256"],
        source_identity=geometry["source_identity"], time_myr=geometry["time_myr"],
        accepted_steps=runtime["mechanics"]["state"]["accepted_steps"],
        accepted_step_sizes_myr=runtime["accepted_step_sizes_myr"], onset_sha256=onset["onset_sha256"],
        source_new_activation_event=None, newly_activating_faces=[], existing_active_faces=[],
        orientation_offset_families=[], status="no_calculated_activation_event",
        source_nucleation_location_law=None, source_physical_dip_law=None,
        source_surface_creation_energy_law=None, selected_sharp_material_surface=None,
        unresolved_physical_inputs=["source_nucleation_location_and_extent",
            "source_through_depth_dip_and_propagation", "source_surface_creation_energy_and_funding",
            "physical_insertion_release_work_and_contact_admission"],
        inherited_runtime_ledgers={k: v for k, v in runtime.items() if k.endswith("_j")},
        physical_creation_admitted=False, physical_release_admitted=False,
        physical_fault_contact_admission_claimed=False, history_reinitialized=False,
        activation_accepted=False, clock_advanced=False, creation_work_debited_from_Genesis_j=0.)
    require(set(raw) == set(expected)|{"location_sha256"}, "whole material location schema differs")
    fixed(raw, expected, "actual source material location")
    return dict(no_current_source_activation_event=True, empty_actual_orientation_families_checked=True,
        no_offset_or_dip_or_creation_law_invented=True, inherited_runtime_ledgers_unchanged=True,
        location_sha256=raw["location_sha256"])


def resolved_section_at_scale(coefficients, lam):
    """Apply only a declared machine-resolution endpoint classification.

    The independent plane roots and point placement remain unrounded. This
    handles a represented vertex exactly on the plane, without a physical
    contact or localization tolerance.
    """
    y, z, d = coefficients["y"], coefficients["z"], coefficients["offset_m"]
    scale = max(float(np.max(abs(lam*y))), float(np.max(abs(z))), abs(d), 1.)
    phi = lam*y/scale+z/scale-d/scale
    threshold = 16*np.finfo(float).eps*np.maximum(abs(lam*y/scale)+abs(z/scale)+abs(d/scale), 1/scale)
    phi[abs(phi) <= threshold] = 0.
    # Rebuilding phi would undo the classified zero through cancellation;
    # use the three scalar roots directly for this classification alone.
    if np.all(phi == 0):
        return dict(kind="triangle", barycentric=np.eye(3), edges=((0, 0), (1, 1), (2, 2)))
    N, edges = [], []
    for i in np.flatnonzero(phi == 0):
        N.append(np.eye(3)[i]); edges.append((int(i), int(i)))
    for i, j in ((0, 1), (0, 2), (1, 2)):
        if phi[i]*phi[j] < 0:
            s = -phi[i]/(phi[j]-phi[i]); point = np.zeros(3)
            point[i], point[j] = 1-s, s; N.append(point); edges.append((i, j))
    require(len(N) <= 2, "section endpoint classification has excess multiplicity")
    return dict(kind="empty" if not N else "point" if len(N) == 1 else "segment",
        barycentric=np.asarray(N).reshape(-1, 3), edges=tuple(edges))


def audit_section_case(case, parent, observer):
    geometry, runtime = parent["runtime_state"]["geometry"], parent["runtime_state"]
    require(case.get("computational_fixture") is True, "explicit section is not marked computational")
    fixed(case, dict(plane_provenance="explicit_normal_and_centroid_anchor_not_source_nucleation_or_dip_law",
        physical_surface_selected=False, source_creation_energy_law=None,
        creation_work_debited_from_Genesis_j=0., physical_creation_admitted=False,
        physical_release_admitted=False, physical_contact_admitted=False,
        dofs_created_or_released=False, material_memory_changed=False,
        integration_scope="analytic_slice_geometry_no_area_or_volume_quadrature_claim"), "explicit section scope")
    require(set(case) == {"name", "computational_fixture", "plane_provenance", "normal_xyz",
        "explicitly_supplied_unnormalized_normal", "offset_m", "face_indices", "origin_xyz_m", "anchor",
        "sections", "probes", "probe_material_stress_samples", "physical_surface_selected",
        "source_creation_energy_law", "creation_work_debited_from_Genesis_j", "physical_creation_admitted",
        "physical_release_admitted", "physical_contact_admitted", "dofs_created_or_released",
        "material_memory_changed", "integration_scope"}, "whole explicit section case schema differs")
    normal, offset = plane_parameters(case["normal_xyz"], case["offset_m"])
    declared = array(case["explicitly_supplied_unnormalized_normal"], (3,))
    close(normal, declared/np.linalg.norm(declared), "explicit fixture normal", atol=2e-14)
    anchor = case["anchor"]
    require(set(anchor) == {"cohort_id", "angular_barycentric", "mass_coordinate_q", "radial_scale"},
        "whole explicit anchor schema differs")
    anchor_catalogue = {c["id"]: c for c in geometry["cohorts"]}
    require(anchor["cohort_id"] in anchor_catalogue
        and anchor_catalogue[anchor["cohort_id"]]["face_index"] in case["face_indices"],
        "computational anchor is outside the declared original face")
    anchor_point = material_samples(parent, [anchor["cohort_id"]], [anchor["angular_barycentric"]], [anchor["mass_coordinate_q"]])
    close(case["origin_xyz_m"], anchor_point["current_position_m"][0], "explicit computational anchor", atol=2e-8)
    close(anchor["radial_scale"], anchor_point["radial_scale"][0], "explicit anchor depth", atol=2e-14)
    close(offset, normal@anchor_point["current_position_m"][0], "explicit anchor plane offset", atol=2e-8)
    raw = case["sections"]; seal(raw, "sections_sha256")
    selected = case["face_indices"]
    require(type(selected) is list and all(type(f) is int and 0 <= f < len(geometry["faces"]) for f in selected)
        and selected == sorted(set(selected)), "selected original face indices differ")
    fixed(raw, dict(format="genesis-declared-current-volume-material-sections-0.1",
        scope="explicit_world_plane_support_no_localization_or_physical_admission",
        source_checkpoint_sha256=parent["source_checkpoint_sha256"],
        source_identity=geometry["source_identity"], time_myr=geometry["time_myr"],
        displacement_sha256=hashlib.sha256(array(runtime["displacement_m"]).tobytes()).hexdigest(),
        face_indices=selected, calculated_location_selected=False, source_creation_energy_law=None,
        physical_creation_admitted=False, physical_release_admitted=False,
        physical_contact_admitted=False, material_memory_changed=False,
        dofs_created_or_released=False, clock_advanced=False,
        creation_work_debited_from_Genesis_j=0.,
        integration_scope="analytic_slice_geometry_no_area_or_volume_quadrature_claim"), "section geometry scope")
    require(set(raw) == {"geometry_binding_sha256", "source_checkpoint_sha256", "source_identity",
        "time_myr", "displacement_sha256", "normal_xyz", "offset_m", "face_indices", "cohorts", "format", "scope",
        "calculated_location_selected", "source_creation_energy_law", "physical_creation_admitted",
        "physical_release_admitted", "physical_contact_admitted", "material_memory_changed",
        "dofs_created_or_released", "clock_advanced", "creation_work_debited_from_Genesis_j",
        "integration_scope", "sections_sha256"}, "whole sections schema differs")
    require(type(raw["geometry_binding_sha256"]) is str and len(raw["geometry_binding_sha256"]) == 64
        and all(c in "0123456789abcdef" for c in raw["geometry_binding_sha256"]),
        "opaque geometry binding must be a SHA256 identity")
    close(raw["normal_xyz"], normal, "declared section normal", atol=2e-14)
    close(raw["offset_m"], offset, "declared section offset", atol=2e-8)
    expected_cohorts = [(i, c) for i, c in enumerate(geometry["cohorts"]) if c["face_index"] in selected]
    require(len(raw["cohorts"]) == len(expected_cohorts), "section omitted selected original cohort")
    X = array(geometry["vertices_m"]); faces = np.asarray(geometry["faces"], int)
    pose = array(runtime["displacement_m"]).reshape(-1, 6)
    records = {}; piece_count = 0; worst_point_error = 0.
    for record, (index, c) in zip(raw["cohorts"], expected_cohorts):
        face = c["face_index"]; triangle = X[faces[face]]
        A, P = pose[faces[face], :3], pose[faces[face], 3:]
        coefficients = section_coefficients(triangle, A, P, normal, offset)
        coefficient = 6*geometry["column_mass_kg"][face]/(geometry["density_kg_m3"]*np.linalg.det(triangle.T))
        fixed(record, dict(cohort_index=index, cohort_id=c["id"], face_index=face,
            owner_id=c["owner_id"], material_id=c["material_id"], mass_kg=c["mass_kg"],
            volume_m3=c["volume_m3"], q_bounds=[c["q_lo"], c["live_q_hi"]],
            lambda_bounds=[c["inner_scale"], c["outer_scale"]], triangle_m=triangle.tolist(),
            displacement_A_m=A.tolist(), slope_P_m=P.tolist()), "original cohort section")
        close(record["normal_xyz"], normal, "cohort section normal", atol=2e-14)
        close(record["offset_m"], offset, "cohort section offset", atol=2e-8)
        close(record["conical_mass_coefficient"], coefficient, "original mass chart coefficient", atol=2e-14)
        close(record["projection_slopes"], coefficients["y"], "full current plane projection slopes", atol=2e-8)
        close(record["projection_offsets"], coefficients["z"], "full current plane projection offsets", atol=2e-8)
        breaks = section_breaks(coefficients, c["inner_scale"], c["outer_scale"])
        close(record["lambda_breakpoints"], breaks, "all material vertex crossing depths", atol=2e-14)
        intervals = []
        for lo, hi in zip(breaks, breaks[1:]):
            cut = resolved_section_at_scale(coefficients, lo+.5*(hi-lo))
            require(cut["kind"] != "triangle", "current material volume collapsed onto declared plane")
            if cut["kind"] in ("point", "segment"):
                intervals.append(dict(lambda_bounds=[float(lo), float(hi)], edge_pairs=[list(e) for e in cut["edges"]], slice_kind=cut["kind"]))
        require(len(record["intervals"]) == len(intervals), "section radial pieces differ")
        for actual, expected in zip(record["intervals"], intervals):
            close(actual["lambda_bounds"], expected["lambda_bounds"], "section piece depth", atol=2e-14)
            fixed(actual, {k: expected[k] for k in ("edge_pairs", "slice_kind")}, "section piece topology")
            require(set(actual) == set(expected), "section piece schema differs")
        constant = []
        boundary_kinds = []
        for lam in breaks:
            kind = resolved_section_at_scale(coefficients, lam)["kind"]; boundary_kinds.append(kind)
            if kind == "triangle": constant.append(dict(lambda_value=float(lam),
                location="inner_boundary" if lam == breaks[0] else "outer_boundary" if lam == breaks[-1] else "interior"))
        require(len(record["constant_lambda_sections"]) == len(constant), "full triangle sections differ")
        for actual, expected in zip(record["constant_lambda_sections"], constant):
            require(set(actual) == set(expected), "whole constant section schema differs")
            close(actual["lambda_value"], expected["lambda_value"], "constant section depth", atol=2e-14)
            fixed(actual, dict(location=expected["location"]), "constant section placement")
        corners = np.concatenate((c["inner_scale"]*coefficients["y"]+coefficients["z"],
            c["outer_scale"]*coefficients["y"]+coefficients["z"]))
        interior = bool(corners.min() < offset < corners.max())
        status = ("interior_cut" if interior else "boundary_support") if intervals or constant else \
            "boundary_touch" if any(k in ("point", "segment") for k in boundary_kinds) else "no_intersection"
        fixed(record, dict(interior_cut=interior, status=status), "section support status")
        require(set(record) == {"cohort_index", "cohort_id", "face_index", "owner_id", "material_id", "mass_kg", "volume_m3",
            "q_bounds", "lambda_bounds", "conical_mass_coefficient", "triangle_m", "displacement_A_m", "slope_P_m",
            "normal_xyz", "offset_m", "projection_slopes", "projection_offsets", "lambda_breakpoints", "intervals",
            "constant_lambda_sections", "interior_cut", "status"}, "whole cohort section schema differs")
        records[c["id"]] = (record, coefficients, coefficient); piece_count += len(intervals)
    samples = case["probe_material_stress_samples"]
    expected_probe_catalogue = [(record["cohort_id"], index, fraction)
        for record in raw["cohorts"] for index in range(len(record["intervals"]))
        for fraction in (.23, .77)]
    require(len(case["probes"]) > 0 and len(samples["cohort_ids"]) == len(case["probes"])
        and [(p["cohort_id"], p["interval_index"], p["fraction"]) for p in case["probes"]] == expected_probe_catalogue,
        "section lost the complete two-probe-per-material-piece catalogue")
    close(samples["normal_world"], np.repeat(normal[None], len(case["probes"]), axis=0),
        "all section traction probes use the declared plane normal", atol=2e-14)
    for j, probe in enumerate(case["probes"]):
        record, coefficients, coefficient = records[probe["cohort_id"]]
        interval = record["intervals"][probe["interval_index"]]
        lam = float(probe["radial_scale"])
        close(lam, .5*sum(interval["lambda_bounds"]), "declared midpoint probe depth", atol=2e-14)
        fraction = float(probe["fraction"]); require(0 <= fraction <= 1, "probe fraction leaves segment")
        cut = resolved_section_at_scale(coefficients, lam)
        require(cut["kind"] == "segment", "probe is not in section segment")
        N = (1-fraction)*cut["barycentric"][0]+fraction*cut["barycentric"][1]
        point = probe["point"]
        require(set(probe) == {"cohort_id", "face_index", "interval_index", "radial_scale", "fraction", "point"}
            and set(point) == {"barycentric", "lambda_value", "q", "xyz_m", "plane_residual_m"}, "whole section probe schema differs")
        close(point["barycentric"], N, "independent explicit plane barycentric root", atol=2e-12)
        close(point["lambda_value"], lam, "explicit probe lambda", atol=2e-14)
        q = -np.expm1(3*np.log(lam))/coefficient
        close(point["q"], q, "explicit probe original material mass coordinate", atol=2e-12)
        xyz = N@(lam*coefficients["Y"]+coefficients["Z"])
        worst_point_error = max(worst_point_error, close(point["xyz_m"], xyz,
            "world plane full A/P section point", rtol=2e-10, atol=2e-8))
        close(point["plane_residual_m"], normal@array(point["xyz_m"])-offset,
            "reported world plane residual", rtol=0, atol=4e-8)
        fixed(probe, dict(face_index=record["face_index"]), "probe original face")
        require(samples["cohort_ids"][j] == probe["cohort_id"], "pointwise stress switched section cohort")
        close(samples["angular_barycentric"][j], point["barycentric"], "pointwise section angular coordinates", atol=2e-14)
        close(samples["mass_coordinate_q"][j], point["q"], "pointwise section material depth", atol=2e-14)
        close(samples["current_position_m"][j], point["xyz_m"], "sample and section current placement", atol=2e-8)
    sample_report = audit_samples(samples, parent, observer)
    return dict(name=case["name"], selected_original_faces=selected,
        all_selected_original_cohorts_checked=len(records), nondegenerate_depth_pieces=piece_count,
        explicit_section_probes_checked=len(case["probes"]), worst_section_point_relative_error=worst_point_error,
        all_six_current_kinematic_amplitudes_checked=True,
        original_material_q_and_one_sided_cohort_ids_preserved=True,
        supplied_plane_not_accepted_source_surface=True,
        declared_plane_traction_normal_checked=True,
        integration_scope="analytic_slice_geometry_no_area_or_volume_quadrature_claim", material_stress=sample_report)


def audit_barrier_records(fixture):
    """Check the actual producer-refusal log's complete resealed DTO bindings.

    These are reported producer source-context refusals. The separate actual
    independent negative-control run exercises this oracle's own refusals.
    """
    records = fixture["observed_resealed_source_context_mutation_barriers"]
    specs = (
        ("accepted_cohort_memory_replaced", fixture["material_stress_observer"], "binding_sha256",
         ["elastic_constant", 0, 0, 0], 1e-4, "VolumeMaterialStressRefusal"),
        ("pointwise_stress_fabricated", fixture["material_stress_samples"], "sample_sha256",
         ["stress_world_tensor", 0, 0, 0], 1e6, "VolumeMaterialStressRefusal"),
        ("explicit_probe_promoted_to_actual_surface", fixture["bound_location"], "location_sha256",
         ["selected_sharp_material_surface"], {"computational_plane_claimed_as_physical_surface": True}, "ValueError"),
        ("through_depth_P_warp_replaced", fixture["section_cases"][0]["sections"], "sections_sha256",
         ["cohorts", 0, "slope_P_m", 0, 0], 1., "ValueError"))
    require(type(records) is list and len(records) == len(specs), "four observed source-context refusal logs required")
    for record, (name, packet, key, path, delta, exception) in zip(records, specs):
        forged = json.loads(canonical(packet)); target = forged
        for part in path[:-1]: target = target[part]
        old = target[path[-1]]; new = delta if isinstance(delta, dict) else old+delta
        target[path[-1]] = new
        forged[key] = digest({k: v for k, v in forged.items() if k != key})
        fixed(record, dict(name=name, status="REJECTED", resealed_descriptor_sha256=forged[key],
            descriptor_seal_recomputed=True, source_context_unchanged=True,
            mutation_path=path, old_value=old, new_value=new, exception_type=exception),
            "observed resealed source-context refusal")
        require(set(record) == {"name", "status", "resealed_descriptor_sha256", "descriptor_seal_recomputed",
            "source_context_unchanged", "mutation_path", "old_value", "new_value", "exception_type", "reason"}
            and type(record["reason"]) is str and len(record["reason"]) > 0,
            "complete observed refusal log schema differs")
    return dict(count=len(records), names=[r["name"] for r in records],
        real_producer_refusal_records_source_and_resealed_deltas_checked=True,
        independent_negative_controls_are_a_separate_run=True)


def audit_fixture(fixture, parent, control_directory=None):
    seal(fixture, "fixture_sha256")
    require(parent.get("checkpoint_sha256") == PARENT_CHECKPOINT_SHA256
        and digest({k: v for k, v in parent.items() if k != "checkpoint_sha256"}) == PARENT_CHECKPOINT_SHA256,
        "accepted immutable parent semantic pin differs")
    require(canonical(fixture["parent_checkpoint"]) == canonical(parent), "complete authentic parent changed")
    fixed(fixture, dict(format=FORMAT, scope=SCOPE, parent_raw_sha256=PARENT_RAW_SHA256,
        stress_scope="current_accepted_endpoint_full_cohort_elastic_memory_not_last_interval_average",
        stress_representation="inherited_fixed_orthonormal_cohort_basis_small_strain_world_tensor_no_finite_deformation_pushforward",
        material_chart="x(N,lambda)=N@(lambda*(X+P)+(A-P))",
        source_nucleation_location_law=None, source_physical_dip_law=None,
        source_surface_creation_energy_law=None, selected_sharp_material_surface=None,
        new_Genesis_time_steps=0, actual_activation_accepted=False,
        creation_work_debited_from_Genesis_j=0., physical_creation_admitted=False,
        physical_release_admitted=False, physical_contact_admitted=False,
        material_memory_changed=False, dofs_created_or_released=False,
        history_reinitialized=False, point_one_complete=False), "authentic source scope")
    require(set(fixture) == {"format", "scope", "parent_raw_sha256", "parent_path", "parent_checkpoint",
        "actual_onset_prerequisite", "bound_location", "material_stress_observer", "material_stress_samples",
        "material_stress_sample_index", "section_cases", "observed_resealed_source_context_mutation_barriers",
        "stress_scope", "stress_representation", "material_chart", "source_nucleation_location_law",
        "source_physical_dip_law", "source_surface_creation_energy_law", "selected_sharp_material_surface",
        "new_Genesis_time_steps", "actual_activation_accepted", "creation_work_debited_from_Genesis_j",
        "physical_creation_admitted", "physical_release_admitted", "physical_contact_admitted",
        "material_memory_changed", "dofs_created_or_released", "history_reinitialized", "point_one_complete",
        "fixture_sha256"}, "whole actual fixture schema differs")
    onset = audit_onset(fixture["actual_onset_prerequisite"], parent)
    observer = audit_observer(fixture["material_stress_observer"], parent)
    samples = audit_samples(fixture["material_stress_samples"], parent, fixture["material_stress_observer"])
    location = audit_location(fixture["bound_location"], parent, fixture["actual_onset_prerequisite"])
    require(type(fixture["section_cases"]) is list and len(fixture["section_cases"]) == 3,
        "three explicit current material plane cases required")
    selected_faces = [0, len(parent["runtime_state"]["geometry"]["faces"])//2,
        len(parent["runtime_state"]["geometry"]["faces"])-1]
    require([c.get("name") for c in fixture["section_cases"]] == [f"supplied_current_plane_face_{f}" for f in selected_faces]
        and [c.get("face_indices") for c in fixture["section_cases"]] == [[f] for f in selected_faces],
        "three distinct original face cases differ")
    geometry_bindings = [c["sections"]["geometry_binding_sha256"] for c in fixture["section_cases"]]
    require(len(set(geometry_bindings)) == 1, "opaque sections do not share the same accepted geometry binding")
    for case, declared in zip(fixture["section_cases"], EXPLICIT_NORMALS):
        fixed(case, dict(explicitly_supplied_unnormalized_normal=list(declared)), "declared computational plane catalogue")
    indices = fixture["material_stress_sample_index"]
    require(len(indices) == len(fixture["material_stress_samples"]["cohort_ids"]) == 81,
        "all eighty-one corner/depth material probes required")
    catalogue = {c["id"]: c for c in parent["runtime_state"]["geometry"]["cohorts"]}
    expected_index_catalogue = []
    for f in selected_faces:
        rows = [c for c in catalogue.values() if c["face_index"] == f]
        for c in (rows[0], rows[len(rows)//2], rows[-1]):
            for N in np.eye(3).tolist():
                for fraction in (0., .5, 1.):
                    expected_index_catalogue.append((c["id"], N, fraction))
    require([(r["cohort_id"], r["angular_barycentric"], r["depth_fraction_of_actual_live_q_interval"])
        for r in indices] == expected_index_catalogue,
        "sample catalogue lost distinct original faces/cohorts/angular corners/material depths")
    for j, index in enumerate(indices):
        samples_packet = fixture["material_stress_samples"]; c = catalogue[index["cohort_id"]]
        fixed(index, dict(cohort_id=samples_packet["cohort_ids"][j], face_index=samples_packet["face_indices"][j],
            owner_id=c["owner_id"], material_id=c["material_id"],
            angular_barycentric=samples_packet["angular_barycentric"][j],
            mass_coordinate_q=samples_packet["mass_coordinate_q"][j], normal_world=samples_packet["normal_world"][j],
            normal_provenance="explicit_computational_input"), "material sample index")
        require(set(index) == {"cohort_id", "face_index", "owner_id", "material_id", "angular_barycentric",
            "mass_coordinate_q", "depth_fraction_of_actual_live_q_interval", "normal_world", "normal_provenance"},
            "whole material sample index schema differs")
        fraction = index["depth_fraction_of_actual_live_q_interval"]
        require(fraction in (0., .5, 1.), "material sample index changed explicit depth fraction")
        expected_q = c["q_lo"] if fraction == 0 else c["live_q_hi"] if fraction == 1 else .5*(c["q_lo"]+c["live_q_hi"])
        close(index["mass_coordinate_q"], expected_q, "sample index original live depth fraction", atol=2e-14)
    barriers = audit_barrier_records(fixture)
    cases = [audit_section_case(case, parent, fixture["material_stress_observer"]) for case in fixture["section_cases"]]
    variance = field_variance(parent)
    require(variance["cohorts_examined"] == 800 and variance["radial_coefficient_memory_nonzero"]
        and variance["maximum_angular_stress_difference_pa"] > 0
        and variance["maximum_radial_stress_difference_pa"] > 0,
        "accepted pointwise stored memory variation was discarded")
    external = False
    if control_directory is not None:
        before, after = (control_directory/name for name in ("parent_before.json", "parent_after.json"))
        require(before.read_bytes() == after.read_bytes()
            and hashlib.sha256(before.read_bytes()).hexdigest() == PARENT_RAW_SHA256,
            "entire accepted parent before/after bytes differ")
        for filename, packet in (("actual_onset_prerequisite.json", fixture["actual_onset_prerequisite"]),
            ("bound_location.json", fixture["bound_location"]),
            ("material_stress_observer.json", fixture["material_stress_observer"]),
            ("material_stress_samples.json", fixture["material_stress_samples"])):
            require(canonical(json.loads((control_directory/filename).read_bytes())) == canonical(packet),
                "external full material control artifact differs")
        for case in fixture["section_cases"]:
            require(canonical(json.loads((control_directory/(case["name"]+".json")).read_bytes())) == canonical(case),
                "external full section artifact differs")
        external = True
    report = dict(format="independent-genesis-volume-material-location-audit-0.1", status="PASS", scope=SCOPE,
        producer_imports=False, parent_raw_sha256=PARENT_RAW_SHA256,
        parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256, formation_checkpoint_sha256=FORMATION_SHA256,
        actual_time_myr=parent["runtime_state"]["geometry"]["time_myr"], actual_cohorts=800,
        actual_active_planes=0, actual_activation_event=None, actual_Genesis_time_advanced=False,
        actual_creation_work_debited_j=0., physical_creation_admitted=False, physical_release_admitted=False,
        normal_contact_admitted=False, actual_surface_selected=False,
        source_nucleation_location_law_unknown=True, source_physical_dip_law_unknown=True,
        actual_creation_law_unknown=True, point_one_complete=False,
        source_calculated_onset=onset, material_stress_observer=observer, material_stress_samples=samples,
        bound_location=location, section_cases=cases, stored_field_variance=variance,
        observed_resealed_source_context_mutation_barriers=barriers,
        complete_material_sample_index_checked=True,
        external_parent_bytes_checked=external,
        opaque_producer_geometry_binding_linked_not_reconstructed=True)
    report["report_sha256"] = digest(report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, default=ROOT/
        "analysis/genesis_volume_creation_validation/evidence/actual_continuation/direct_after_second.json")
    parser.add_argument("--control-directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parent_bytes = args.parent_checkpoint.read_bytes()
    require(hashlib.sha256(parent_bytes).hexdigest() == PARENT_RAW_SHA256, "actual immutable parent raw pin differs")
    parent = json.loads(parent_bytes); fixture = json.loads(args.fixture.read_bytes())
    report = audit_fixture(fixture, parent, args.control_directory)
    args.output.write_bytes(canonical(report)+b"\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

