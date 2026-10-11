"""Producer-free prospective nucleation and embedded material surface oracle.

Saved six-component elastic memory and the conical A/P chart are reconstructed
from the immutable parent.  The independent integrations preserve reference
small-strain work measure; explicit examples do not admit a physical event.
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
# This legacy oracle is itself producer-free and checks the inherited
# weak-plane event context, including its independent volume integration.
from analysis.genesis_volume_insertion_validation.independent_audit import audit_onset

PARENT_RAW_SHA256 = "62579a0bad128830a66e2f44579cf2b1f5d3835b691225b8d431102f3e4486d7"
PARENT_CHECKPOINT_SHA256 = "5d282ceeccc8d76acc38427d08a7e65575fddb509fc4b93115cbe6d318e91349"
FORMATION_SHA256 = "cd79ee2a5139c4b91abdb1b4714a761c9153cc9f4490f5aa63fa2facba458c89"
SOURCE_ANGULAR_POINTS = np.array(((2/3, 1/6, 1/6), (1/6, 2/3, 1/6), (1/6, 1/6, 2/3)))
STRESS_SCOPE = "accepted_endpoint_elastic_memory_at_explicit_material_coordinates_no_fracture_location"
FALSE_STRESS = ("history_reinitialized", "clock_advanced", "activation_accepted",
    "physical_creation_admitted", "physical_release_admitted",
    "physical_fault_contact_admission_claimed", "historical_three_dimensional_memory_recovered",
    "instantaneous_future_birth_traction_claimed")


def conical_scales(triangle, column_mass_kg, density_kg_m3, mass_fraction):
    determinant = np.linalg.det(array(triangle, (3, 3)).T)
    require(determinant > 0 and column_mass_kg > 0 and density_kg_m3 > 0,
            "conical material mass or orientation invalid")
    coefficient = 6*float(column_mass_kg)/(float(density_kg_m3)*determinant)
    z = -coefficient*array(mass_fraction)
    require(np.all(z > -1), "conical material reaches centre")
    minus_one = np.expm1(np.log1p(z)/3)
    return 1+minus_one, minus_one


def isotropic_matrix(young_modulus_pa, poisson_ratio, damage, residual_stiffness):
    E, nu = float(young_modulus_pa), float(poisson_ratio)
    shear = E/(2*(1+nu)); lame = E*nu/((1+nu)*(1-2*nu))
    C = np.zeros((6, 6)); C[:3, :3] = lame
    C[np.arange(3), np.arange(3)] += 2*shear
    C[3:, 3:] = np.eye(3)*shear
    return (float(residual_stiffness)+(1-float(residual_stiffness))*(1-float(damage))**2)*C


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError("Independent embedded nucleation audit: "+message)


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




def normalized_plane(normal_xyz, offset_m):
    n = array(normal_xyz, (3,)); d = float(array(offset_m, ()))
    length = np.linalg.norm(n)
    require(length > 0, "declared plane normal is zero")
    return n/length, d/length


def plane_coefficients(X, A, P, normal_xyz, offset_m):
    X, A, P = (array(v, (3, 3)) for v in (X, A, P))
    n, d = normalized_plane(normal_xyz, offset_m)
    return dict(X=X, A=A, P=P, Y=X+P, Z=A-P, n=n, d=d,
                y=(X+P)@n, z=(A-P)@n)


def signed_at(c, lam):
    terms = (lam*c["y"], c["z"], c["d"])
    scale = max(float(np.max(abs(terms[0]))), float(np.max(abs(terms[1]))), abs(terms[2]), 1.)
    a, b, d = terms[0]/scale, terms[1]/scale, terms[2]/scale
    phi = a+b-d
    error = 16*np.finfo(float).eps*np.maximum(abs(a)+abs(b)+abs(d), 1/scale)
    return np.where(abs(phi) <= error, 0., phi)


def section_partition(c, inner, outer):
    require(0 < inner < outer, "cohort section depth is not positive")
    breaks = [float(inner), float(outer)]
    for vertex, (slope, intercept) in enumerate(zip(c["y"], c["z"])):
        if slope == 0: continue
        root = float((c["d"]-intercept)/slope)
        # Machine coincidence keeps literal material depth; there is no
        # physical spatial acceptance tolerance in this oracle.
        for endpoint in (inner, outer):
            if (abs(root-endpoint) <= 4*max(abs(np.spacing(root)), abs(np.spacing(endpoint)))
                    and signed_at(c, endpoint)[vertex] == 0):
                root = float(endpoint)
                break
        if inner < root < outer: breaks.append(root)
    breaks = sorted(set(breaks))
    segments, triangles = [], []
    for lo, hi in zip(breaks, breaks[1:]):
        value = signed_at(c, lo+(hi-lo)/2)
        edges = [(int(i), int(i)) for i in np.flatnonzero(value == 0)]
        for i, j in ((0, 1), (0, 2), (1, 2)):
            if value[i]*value[j] < 0: edges.append((i, j))
        if len(edges) == 2: segments.append((lo, hi, tuple(edges)))
        require(len(edges) <= 2 or np.all(value == 0), "section edge multiplicity unresolved")
    for lam in breaks:
        if inner < lam < outer and np.all(signed_at(c, lam) == 0): triangles.append(lam)
    return segments, triangles


def edge_barycentric(c, edge, lam):
    i, j = edge
    N, derivative = np.zeros(3), np.zeros(3)
    if i == j:
        N[i] = 1.
        return N, derivative
    # Common scaling avoids losing cancellation for a planetary position.
    scale = max(float(np.max(abs(lam*c["y"]))), float(np.max(abs(c["z"]))), abs(c["d"]), 1.)
    phi = lam*(c["y"]/scale)+(c["z"]/scale-c["d"]/scale)
    den = phi[j]-phi[i]
    require(den != 0, "intersection derivative is singular")
    t = -phi[i]/den
    dt = (-c["y"][i]/scale*den+phi[i]*(c["y"][j]-c["y"][i])/scale)/den**2
    N[i], N[j] = 1-t, t
    derivative[i], derivative[j] = -dt, dt
    return N, derivative


def parent_trace(N, lam):
    """Three Cartesian trace components on each of 3 original A/P vertices."""
    N = array(N, (3,)); result = np.zeros((3, 18))
    for i in range(3):
        result[:, 6*i:6*i+3] = N[i]*np.eye(3)
        result[:, 6*i+3:6*i+6] = (lam-1)*N[i]*np.eye(3)
    return result


def current_chart_jacobians(c, N, lam):
    current = np.column_stack((lam*(c["Y"][1]-c["Y"][0])+(c["Z"][1]-c["Z"][0]),
        lam*(c["Y"][2]-c["Y"][0])+(c["Z"][2]-c["Z"][0]), N@c["Y"]))
    reference = np.column_stack((lam*(c["X"][1]-c["X"][0]),
        lam*(c["X"][2]-c["X"][0]), N@c["X"]))
    return current, reference


def whole_cohort_chart_positive(c, inner, outer):
    """A determinant is affine in N, quadratic in lambda: check exact extrema."""
    Y, Z = c["Y"], c["Z"]
    dY = Y[1:]-Y[0]; dZ = Z[1:]-Z[0]
    coefficients = np.array((np.cross(dZ[0], dZ[1])@Y.T,
        (np.cross(dY[0], dZ[1])+np.cross(dZ[0], dY[1]))@Y.T,
        np.cross(dY[0], dY[1])@Y.T))
    minima = []
    for vertex in range(3):
        a0, a1, a2 = coefficients[:, vertex]
        candidates = [inner, outer]
        if a2 != 0:
            stationary = -a1/(2*a2)
            if inner < stationary < outer: candidates.append(stationary)
        values = [(a2*lam+a1)*lam+a0 for lam in candidates]
        require(min(values) > 0, "current chart is inverted or collapsed in whole cohort")
        minima.append(min(values))
    return min(minima)


def embedded_quadrature(parent, normal_xyz, offset_m, *, face_indices, cohort_ids=None, order=8):
    """Derive every material node, differential, oriented metric and trace."""
    require(type(order) is int and 2 <= order <= 64, "quadrature order outside supported domain")
    runtime = parent["runtime_state"]; geometry = runtime["geometry"]
    Xall, faces = array(geometry["vertices_m"]), np.asarray(geometry["faces"], int)
    u = array(runtime["displacement_m"]).reshape(-1, 6)
    nodes, w = np.polynomial.legendre.leggauss(order)
    nodes, w = (nodes+1)/2, w/2
    fields = {k: [] for k in ("point_patch_indices", "cohort_indices", "cohort_ids", "face_indices",
        "barycentric", "radial_scale", "mass_coordinate_q", "current_position_m", "reference_position_m",
        "current_differentials_m", "reference_differentials_m", "current_normal_xyz", "reference_normal_xyz",
        "current_area_weights_m2", "reference_area_weights_m2", "parent_trace", "paired_jump_trace", "trace_dofs",
        "current_volume_jacobian_m", "reference_volume_jacobian_m", "current_volume_determinant_m3", "reference_volume_determinant_m3", "current_to_reference_volume_ratio")}
    patch_ids, patch_kinds, patch_bounds, owners, materials, selected, selected_indices = [], [], [], [], [], [], []
    for index, cohort in enumerate(geometry["cohorts"]):
        f, cid = int(cohort["face_index"]), int(cohort["id"])
        if f not in face_indices or (cohort_ids is not None and cid not in cohort_ids): continue
        selected.append(cid); selected_indices.append(index)
        vertices = faces[f]; X = Xall[vertices]
        c = plane_coefficients(X, u[vertices, :3], u[vertices, 3:], normal_xyz, offset_m)
        inner, outer = float(cohort["inner_scale"]), float(cohort["outer_scale"])
        whole_cohort_chart_positive(c, inner, outer)
        corners = np.concatenate((inner*c["y"]+c["z"], outer*c["y"]+c["z"]))
        segments, triangles = (section_partition(c, inner, outer)
            if float(corners.min()) < c["d"] < float(corners.max()) else ([], []))
        coefficient = 6*geometry["column_mass_kg"][f]/(geometry["density_kg_m3"]*np.linalg.det(X.T))

        def add(N, lam, dNs, dNl, parameter_weight, patch, *, triangle=False):
            pose = lam*c["Y"]+c["Z"]
            if triangle:
                dc = pose[1:]-pose[0]; dr = lam*(X[1:]-X[0])
            else:
                dc = np.array((dNs@pose, dNl@pose+N@c["Y"]))
                dr = np.array((lam*(dNs@X), N@X+lam*(dNl@X)))
            cc, cr = np.cross(dc[0], dc[1]), np.cross(dr[0], dr[1])
            jc, jr = np.linalg.norm(cc), np.linalg.norm(cr)
            require(jc > 0 and jr > 0, "embedded differential metric collapsed")
            orientation = 1. if cc@c["n"] > 0 else -1.
            T = parent_trace(N, lam)
            Jc, Jr = current_chart_jacobians(c, N, lam)
            q = -np.expm1(3*np.log(lam))/coefficient
            values = dict(point_patch_indices=patch, cohort_indices=index, cohort_ids=cid, face_indices=f,
                barycentric=N, radial_scale=lam, mass_coordinate_q=q,
                current_position_m=N@pose, reference_position_m=lam*(N@X),
                current_differentials_m=dc, reference_differentials_m=dr,
                current_normal_xyz=orientation*cc/jc, reference_normal_xyz=orientation*cr/jr,
                current_area_weights_m2=parameter_weight*jc, reference_area_weights_m2=parameter_weight*jr,
                parent_trace=T, paired_jump_trace=np.concatenate((-T, T), axis=1),
                trace_dofs=(6*vertices[:, None]+np.arange(6)).ravel(),
                current_volume_jacobian_m=Jc, reference_volume_jacobian_m=Jr,
                current_volume_determinant_m3=np.linalg.det(Jc), reference_volume_determinant_m3=np.linalg.det(Jr),
                current_to_reference_volume_ratio=np.linalg.det(Jc)/np.linalg.det(Jr))
            for name in fields: fields[name].append(values[name])
            owners.append(cohort["owner_id"]); materials.append(cohort["material_id"])

        for lo, hi, edges in segments:
            patch = len(patch_ids); patch_ids.append(cid); patch_kinds.append("lambda_segment"); patch_bounds.append((lo, hi))
            for radial, wr in zip(nodes, w):
                lam = lo+(hi-lo)*radial
                N0, D0 = edge_barycentric(c, edges[0], lam); N1, D1 = edge_barycentric(c, edges[1], lam)
                for s, ws in zip(nodes, w):
                    add((1-s)*N0+s*N1, lam, N1-N0, (1-s)*D0+s*D1, (hi-lo)*wr*ws, patch)
        for lam in triangles:
            patch = len(patch_ids); patch_ids.append(cid); patch_kinds.append("constant_lambda_triangle"); patch_bounds.append((lam, lam))
            for r, wr in zip(nodes, w):
                for t, wt in zip(nodes, w):
                    add(np.array(((1-r)*(1-t), r, (1-r)*t)), lam, None, None, wr*wt*(1-r), patch, triangle=True)
    shapes = dict(barycentric=(3,), current_position_m=(3,), reference_position_m=(3,),
        current_differentials_m=(2, 3), reference_differentials_m=(2, 3), current_normal_xyz=(3,), reference_normal_xyz=(3,),
        parent_trace=(3, 18), paired_jump_trace=(3, 36), trace_dofs=(18,),
        current_volume_jacobian_m=(3, 3), reference_volume_jacobian_m=(3, 3))
    integers = {"point_patch_indices", "cohort_indices", "cohort_ids", "face_indices", "trace_dofs"}
    count = len(owners)
    result = {k: np.asarray(v, int if k in integers else float).reshape((count,)+shapes.get(k, ())) for k, v in fields.items()}
    result.update(owner_ids=owners, material_ids=materials, selected_cohort_ids=selected,
        selected_cohort_indices=selected_indices, patch_cohort_ids=patch_ids, patch_kinds=patch_kinds,
        patch_lambda_bounds=np.asarray(patch_bounds).reshape(-1, 2), parent_ndof=6*len(Xall), quadrature_order=order)
    return result


def point_trace(surface, parent_increment):
    u = array(parent_increment, (surface["parent_ndof"],))
    return np.einsum("pai,pi->pa", surface["parent_trace"], u[surface["trace_dofs"]])


def paired_force(surface, traction, *, reference=True):
    t = array(traction, (len(surface["cohort_ids"]), 3))
    weights = surface["reference_area_weights_m2" if reference else "current_area_weights_m2"]
    local = np.einsum("pai,pa,p->pi", surface["parent_trace"], t, weights)
    force = np.zeros(surface["parent_ndof"])
    np.add.at(force, surface["trace_dofs"].ravel(), local.ravel())
    return np.stack((-force, force))


def paired_virtual_work(surface, traction, increments, *, reference=True):
    u = array(increments, (2, surface["parent_ndof"]))
    jump = point_trace(surface, u[1]-u[0])
    weights = surface["reference_area_weights_m2" if reference else "current_area_weights_m2"]
    return float(np.einsum("pa,pa,p->", array(traction, jump.shape), jump, weights))


def prestress_fields(parent, surface):
    values = material_samples(parent, surface["cohort_ids"], surface["barycentric"], surface["mass_coordinate_q"])
    traction = np.einsum("pij,pj->pi", values["stress_world_tensor"], surface["reference_normal_xyz"])
    ratio = surface["reference_area_weights_m2"]/surface["current_area_weights_m2"]
    current_density = traction*ratio[:, None]
    wrong_current = np.einsum("pij,pj->pi", values["stress_world_tensor"], surface["current_normal_xyz"])
    return dict(samples=values, reference_traction_world_pa=traction,
        born_current_equivalent_traction_world_pa=current_density,
        paired_force_n=paired_force(surface, traction),
        wrong_current_normal_measure_paired_force_n=paired_force(surface, wrong_current, reference=False))


NUCLEATION_FALSE = ("physical_creation_admitted", "physical_release_admitted",
    "physical_fault_contact_admission_claimed", "funding_admitted", "history_reinitialized",
    "activation_accepted", "clock_advanced", "positive_finite_seed_area_certified",
    "insertion_reaction_replaced", "historical_dissipation_reclassified_as_funding")
PROVENANCE = "explicit_prospective_constitutive_inputs_not_recovered_history"
CRITERION = "Rankine_and_classical_unclamped_Mohr_Coulomb_convex_envelope"
CORNER_ORDER = "outer_then_inner_radial_endpoint_each_original_angular_vertex"


def nucleation_criterion(stress, law):
    """Independent spectral convex envelope, tension-positive convention."""
    s = array(stress)
    require(s.ndim >= 2 and s.shape[-2:] == (3, 3) and s.size, "criterion needs full stress tensors")
    require(np.allclose(s, s.swapaxes(-1, -2), rtol=0,
        atol=64*np.finfo(float).eps*max(float(abs(s).max()), np.finfo(float).tiny)), "criterion stress is asymmetric")
    T, c, mu = (float(law[k]) for k in ("tensile_strength_pa", "shear_cohesion_pa", "friction_coefficient"))
    require(np.isfinite((T, c, mu)).all() and T > 0 and c > 0 and mu >= 0, "invalid prospective criterion law")
    eigen = np.linalg.eigvalsh(s)
    factor = np.hypot(1., mu)+mu
    tensile = eigen[..., 2]/T-1
    shear = .5*(factor*eigen[..., 2]/c-eigen[..., 0]/(factor*c))-1
    return dict(principal_stress_pa=eigen, tensile_score=tensile, shear_score=shear,
        yield_score=np.maximum(tensile, shear))


def saved_corner_catalogue(parent, law):
    cohorts = parent["runtime_state"]["geometry"]["cohorts"]
    ids, N, q = [], [], []
    for cohort in cohorts:
        for depth in (cohort["q_lo"], cohort["live_q_hi"]):
            ids.extend([cohort["id"]]*3); N.extend(np.eye(3)); q.extend([depth]*3)
    samples = material_samples(parent, ids, N, q)
    stresses = samples["stress_world_tensor"].reshape(len(cohorts), 2, 3, 3, 3)
    return dict(samples=samples, stresses=stresses, **nucleation_criterion(stresses, law))


def validate_prospective_law(law):
    require(type(law) is dict and set(law) == {"tensile_strength_pa", "shear_cohesion_pa",
        "friction_coefficient", "seed_radius_m", "creation_law"}, "whole prospective law schema differs")
    for name in ("tensile_strength_pa", "shear_cohesion_pa", "seed_radius_m"):
        value = array(law[name], ())
        require(value > 0, "prospective parameter is not positive: "+name)
    require(array(law["friction_coefficient"], ()) >= 0, "prospective friction is negative")
    creation = law["creation_law"]
    keys = {"fracture_energy_j_m2", "peak_traction_pa", "cohesive_stiffness_pa_m", "compression_stiffness_pa_m"}
    require(type(creation) is dict and set(creation) == keys, "whole creation law schema differs")
    for key in keys: require(array(creation[key], ()) > 0, "creation parameter is not positive: "+key)
    Gc, peak, k = (float(creation[name]) for name in ("fracture_energy_j_m2", "peak_traction_pa", "cohesive_stiffness_pa_m"))
    require(2*Gc/peak > peak/k, "prospective cohesive softening cannot resolve a positive interval")


def audit_nucleation(raw, parent, observer, onset, law):
    seal(raw, "nucleation_sha256")
    if law is not None: validate_prospective_law(law)
    runtime = parent["runtime_state"]; geometry = runtime["geometry"]
    # Existing diffuse planes cannot be retrospectively renamed as a newly
    # sharp surface. Its authentic no-event context is checked independently.
    require(not onset["newly_activating_faces"] and not onset["existing_active_faces"],
            "actual inherited source no-event gate differs")
    expected = dict(format="genesis-source-bound-prospective-volume-nucleation-0.1",
        scope="explicit_new_convex_source_gated_seed_proposal_not_funded_physical_surface", criterion=CRITERION,
        parent_checkpoint_sha256=parent["checkpoint_sha256"], formation_checkpoint_sha256=FORMATION_SHA256,
        source_identity=geometry["source_identity"], time_myr=geometry["time_myr"], accepted_steps=4,
        accepted_step_sizes_myr=runtime["accepted_step_sizes_myr"], observer_binding_sha256=observer["binding_sha256"],
        source_onset=onset, status="no_calculated_activation_event", newly_activating_faces=[], existing_active_faces=[],
        declared_prospective_law=law, parameter_provenance=None if law is None else PROVENANCE,
        source_recovered_creation_energy_law=None, corner_order=CORNER_ORDER, corner_catalogue=[],
        maximum_certificate="convex_tensor_envelope_on_affine_angular_inverse_lambda_corner_hull",
        numerical_uniqueness_tolerance="128_machine_epsilon_scaled_no_physical_threshold_shift",
        candidate=None, computational_candidate_proposals=[], refusal=None, selected_sharp_material_surface=None,
        missing_physical_admission=["finite_seed_clipping_and_positive_reference_area",
            "embedded_or_constrained_subdomain_force_reaction_replacement",
            "calculated_incremental_work_and_creation_funding", "full_time_contact_and_release_trajectory"],
        inherited_runtime_ledgers={k: v for k, v in runtime.items() if k.endswith("_j")},
        historical_dissipation_available_for_creation_j=0., creation_work_debited_from_Genesis_j=0.,
        **{k: False for k in NUCLEATION_FALSE})
    require(set(raw) == set(expected)|{"nucleation_sha256"}, "whole authentic nucleation schema differs")
    fixed(raw, expected, "authentic source nucleation")
    return dict(nucleation_sha256=raw["nucleation_sha256"], inherited_diffuse_source_gate_no_event_checked=True,
        new_constitutive_law_is_explicit_not_recovered_history=law is not None,
        prospective_law_cannot_override_source_gate=True, no_creation_funding_or_debit=True)


SURFACE_FALSE = ("bulk_energy_force_identity_claimed", "finite_stress_pushforward_claimed",
    "physical_creation_admitted", "physical_release_admitted", "physical_contact_admitted",
    "dofs_created_or_released", "material_memory_changed", "material_history_reinitialized", "clock_advanced")


def audit_embedded_surface(raw, parent, *, normal_xyz, offset_m, face_indices, cohort_ids=None, order=8):
    seal(raw, "embedded_surface_sha256")
    expected = embedded_quadrature(parent, normal_xyz, offset_m, face_indices=face_indices,
        cohort_ids=cohort_ids, order=order)
    geometry = parent["runtime_state"]["geometry"]
    fixed_fields = dict(format="genesis-original-material-embedded-surface-trace-0.1",
        scope="explicit_current_plane_material_quadrature_paired_virtual_trace_no_physical_insertion",
        source_checkpoint_sha256=FORMATION_SHA256, source_identity=geometry["source_identity"],
        time_myr=geometry["time_myr"], source_geometry_bound=True,
        declared_normal_xyz=array(normal_xyz, (3,)).tolist(), declared_offset_m=float(array(offset_m, ())),
        declared_plane_input_semantics="literal_numeric_constructor_coefficients_before_normalization",
        quadrature_rule="tensor_Gauss_on_analytic_lambda_pieces_and_Duffy_triangle",
        original_parent_modes="A_xyz_and_P_xyz_metres_per_original_vertex",
        paired_trace_scope="two_virtual_parent_fields_not_created_or_released_mesh_DOFs",
        material_support_semantics="birth_pullback_N_q_frozen_not_world_plane_reintersection",
        area_conventions="current_and_reference_separate_interface_counted_once",
        support_scope="relative_interior_cohort_sheets_boundary_only_support_has_zero_area",
        numerical_integration_scope="approximate_geometry_integrals_exact_discrete_trace_work_at_declared_order",
        geometry_integrals_exact=False,
        recorded_metrics_semantics="fixed_declared_plane_pullback_snapshot_not_transported_after_follow",
        creation_work_debited_from_Genesis_j=0., **{k: False for k in SURFACE_FALSE})
    for name in ("owner_ids", "material_ids", "selected_cohort_ids", "selected_cohort_indices", "patch_cohort_ids",
            "patch_kinds", "parent_ndof", "quadrature_order"):
        fixed_fields[name] = expected[name]
    fixed(raw, fixed_fields, "full embedded material surface context")
    # Opaque dataclass byte bindings are linked with the source-produced case.
    # Numerical geometry, constitutive history and every quadrature node are
    # reconstructed independently, rather than treating those hashes as proof.
    require(type(raw.get("material_sections_binding_sha256")) is str
        and len(raw["material_sections_binding_sha256"]) == 64
        and all(character in "0123456789abcdef" for character in raw["material_sections_binding_sha256"]),
        "embedded section binding lacks an integrity link")
    numeric = set(expected)-set(fixed_fields)
    errors = {}
    for name in numeric:
        require(name in raw, "embedded surface omits "+name)
        errors[name] = close(raw[name], expected[name], "embedded surface "+name,
            rtol=8e-10, atol=3e-8 if "position" in name else 2e-9 if "area" in name else 2e-10)
    close(raw["total_current_area_m2"], expected["current_area_weights_m2"].sum(), "total born current area", atol=1e-4)
    close(raw["total_reference_area_m2"], expected["reference_area_weights_m2"].sum(), "total reference area", atol=1e-4)
    allowed = set(fixed_fields)|numeric|{"material_sections_binding_sha256", "total_current_area_m2",
        "total_reference_area_m2", "embedded_surface_sha256"}
    require(set(raw) == allowed, "whole embedded surface schema differs")
    return expected, dict(points_examined=len(expected["cohort_ids"]), patches_examined=len(expected["patch_cohort_ids"]),
        original_material_angular_and_mass_identity_checked=True, full_A_P_dual_trace_checked=True,
        separate_reference_and_current_metric_checked=True, whole_cohort_positive_material_chart_checked=True,
        opaque_section_binding_scope="integrity_link_only_not_independently_reconstructed",
        numerical_relative_errors=errors, embedded_surface_sha256=raw["embedded_surface_sha256"])


def birth_basis(normals):
    n = array(normals)
    axes = np.eye(3)[np.argmin(abs(n), axis=1)]
    tangent = axes-n*np.sum(axes*n, axis=1)[:, None]
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    return np.stack((n, tangent, np.cross(n, tangent)), axis=2)


def cohesive_prestress_seed(law, traction):
    t = array(traction)
    Gc, peak, stiffness, compression = (float(law[k]) for k in
        ("fracture_energy_j_m2", "peak_traction_pa", "cohesive_stiffness_pa_m", "compression_stiffness_pa_m"))
    offset = t/stiffness
    offset[..., 0] = np.where(t[..., 0] < 0, t[..., 0]/compression, offset[..., 0])
    opening = offset.copy(); opening[..., 0] = np.maximum(opening[..., 0], 0)
    radius = np.linalg.norm(opening, axis=-1); onset = peak/stiffness
    require(np.all(radius <= onset+32*np.finfo(float).eps*np.maximum(radius, onset)),
        "prospective prestress virgin seed is overstrength")
    factor = np.ones(radius.shape); snap = radius > onset
    factor[snap] = onset*(1-4*np.finfo(float).eps)/radius[snap]
    offset[..., 1:] *= factor[..., None]
    offset[..., 0] = np.where(offset[..., 0] > 0, offset[..., 0]*factor, offset[..., 0])
    opening = offset.copy(); opening[..., 0] = np.maximum(opening[..., 0], 0)
    radius = np.linalg.norm(opening, axis=-1)
    reference = stiffness*offset
    reference[..., 0] = np.where(offset[..., 0] < 0, compression*offset[..., 0], reference[..., 0])
    B = .5*stiffness*radius**2; C = .5*compression*np.minimum(offset[..., 0], 0)**2
    return dict(law=law, initial_traction_pa=t, reference_traction_pa=reference,
        reference_peak_snap_factor=factor, reference_offset_m=offset, initial_maximum_separation_m=radius,
        reference_cohesive_energy_j_m2=B, reference_compression_energy_j_m2=C,
        auxiliary_fracture_energy_j_m2=Gc+B, final_separation_m=2*(Gc+B)/peak,
        kernel_only=True, physical_release_admitted=False, normal_contact_admitted=False)



def check_force_modes(actual, expected, name):
    """Check A and P separately so a large translation force cannot hide P."""
    a, b = array(actual).reshape(2, -1, 6), array(expected).reshape(2, -1, 6)
    require(a.shape == b.shape, name+" paired force dimensions differ")
    return [close(a[:, :, start:start+3], b[:, :, start:start+3], name+" "+family,
        rtol=2e-9, atol=1e-3) for start, family in ((0, "A"), (3, "P"))]


def audit_prestress(raw, parent, observer, surface_raw, surface, *, creation_law=None):
    seal(raw, "embedded_prestress_sha256")
    sample_report = audit_samples(raw["source_stress_samples"], parent, observer)
    calculated = prestress_fields(parent, surface)
    samples = calculated["samples"]
    close(raw["source_stress_samples"]["angular_barycentric"], surface["barycentric"], "prestress uses born material barycentric")
    close(raw["source_stress_samples"]["mass_coordinate_q"], surface["mass_coordinate_q"], "prestress uses born material q", atol=2e-13)
    fixed(raw["source_stress_samples"], dict(cohort_ids=surface["cohort_ids"].tolist()), "prestress full cohort identity")
    pullback = raw["pullback"]; seal(pullback, "pullback_sha256")
    basis = birth_basis(surface["current_normal_xyz"])
    local = np.einsum("pji,pj->pi", basis, calculated["born_current_equivalent_traction_world_pa"])
    reference = calculated["reference_traction_world_pa"]
    reference_local = np.einsum("pji,pj->pi", basis, reference)
    seed = None if creation_law is None else cohesive_prestress_seed(creation_law, reference_local)
    fields = dict(stress_world_pa=samples["stress_world_tensor"], reference_traction_world_pa=reference,
        birth_area_ratio=surface["reference_area_weights_m2"]/surface["current_area_weights_m2"],
        birth_traction_world_pa=calculated["born_current_equivalent_traction_world_pa"], birth_basis_xyz=basis,
        birth_traction_local_pa=local, reference_traction_local_pa=reference_local, paired_parent_force_n=calculated["paired_force_n"])
    claims = dict(surface_sha256=surface_raw["embedded_surface_sha256"],
        traction_measure="fixed_inherited_small_strain_sigma_times_reference_normal_per_reference_area",
        equivalent_birth_measure="reference_traction_times_reference_area_over_fixed_birth_area",
        creation_area_measure="original_reference_material_area_Gc_counts_both_banks_once",
        seed_traction_measure="reference_area_traction_in_fixed_birth_normal_tangent_basis",
        work_scope="algebraic_duality_under_the_same_declared_quadrature",
        bulk_reaction_replaced=False, finite_stress_pushforward_claimed=False,
        physical_surface_selected=False, physical_creation_admitted=False,
        physical_release_admitted=False, funding_admitted=False, creation_work_debited_from_Genesis_j=0.)
    fixed(pullback, claims, "full reference-work prestress context")
    errors = {name: close(pullback[name], value, "reference-work prestress "+name, rtol=2e-9,
        atol=1e-2 if "force" in name else 1e-3 if "stress" in name or "traction" in name else 2e-12)
        for name, value in fields.items()}
    errors["paired_force_A_P"] = check_force_modes(pullback["paired_parent_force_n"],
        calculated["paired_force_n"], "reference-work paired force")
    if seed is None:
        fixed(pullback, dict(seed=None, full_separation_cost_on_explicit_surface_j=None), "unconfigured prestress")
    else:
        raw_seed = pullback["seed"]
        require(type(raw_seed) is dict and set(raw_seed) == set(seed), "whole virgin prestress seed schema differs")
        fixed(raw_seed, {k: seed[k] for k in ("law", "kernel_only", "physical_release_admitted", "normal_contact_admitted")}, "prestress seed scope")
        for name in set(seed)-{"law", "kernel_only", "physical_release_admitted", "normal_contact_admitted"}:
            errors["seed_"+name] = close(raw_seed[name], seed[name], "virgin cohesive seed "+name, rtol=2e-9, atol=1e-10)
        close(pullback["full_separation_cost_on_explicit_surface_j"],
            creation_law["fracture_energy_j_m2"]*surface["reference_area_weights_m2"].sum(),
            "explicit surface separation price is not available work", atol=1e-4)
    require(set(pullback) == set(claims)|set(fields)|{"seed", "full_separation_cost_on_explicit_surface_j", "pullback_sha256"},
        "whole reference-work pullback schema differs")
    geometry = parent["runtime_state"]["geometry"]; runtime = parent["runtime_state"]
    expected_context = dict(format="genesis-source-bound-embedded-reference-work-prestress-0.1",
        scope="explicit_material_boundary_reference_work_not_partitioned_bulk_reaction_or_physical_release",
        parent_checkpoint_sha256=parent["checkpoint_sha256"], formation_checkpoint_sha256=FORMATION_SHA256,
        source_identity=geometry["source_identity"], time_myr=geometry["time_myr"], accepted_steps=4,
        accepted_step_sizes_myr=runtime["accepted_step_sizes_myr"], observer_binding_sha256=observer["binding_sha256"],
        surface_sha256=surface_raw["embedded_surface_sha256"],
        parameter_provenance=None if creation_law is None else "explicit_new_constitutive_input_not_historical_toughness",
        inherited_runtime_ledgers={k: v for k, v in runtime.items() if k.endswith("_j")},
        source_historical_surface_creation_energy_law=None, selected_sharp_material_surface=None,
        partitioned_bulk_domains_prepared=False, dynamic_bank_reaction_replaced=False,
        physical_creation_admitted=False, physical_release_admitted=False,
        physical_fault_contact_admission_claimed=False, funding_admitted=False,
        history_reinitialized=False, material_memory_changed=False, activation_accepted=False,
        clock_advanced=False, creation_work_debited_from_Genesis_j=0.)
    fixed(raw, expected_context, "authentic embedded prestress")
    require(set(raw) == set(expected_context)|{"source_stress_samples", "pullback", "embedded_prestress_sha256"},
        "whole source-bound prestress schema differs")
    force_equivalent = paired_force(surface, calculated["born_current_equivalent_traction_world_pa"], reference=False)
    check_force_modes(force_equivalent, calculated["paired_force_n"], "reference/born traction force duality")
    wrong = np.linalg.norm(calculated["wrong_current_normal_measure_paired_force_n"]-calculated["paired_force_n"])
    scale = max(np.linalg.norm(calculated["paired_force_n"]), np.finfo(float).tiny)
    return calculated, dict(source_stress_samples=sample_report, reference_small_strain_work_measure_checked=True,
        reference_and_born_equivalent_density_force_checked=True,
        wrong_current_normal_measure_relative_force_difference=float(wrong/scale),
        explicit_price_is_not_creation_funding=True, numerical_relative_errors=errors,
        embedded_prestress_sha256=raw["embedded_prestress_sha256"])


FORMAT = "genesis-volume-nucleation-embedded-control-0.1"
SCOPE = "prospective_source_bound_birth_gate_and_computational_embedded_trace_not_funded_physical_release"
DECLARED_LAW = dict(tensile_strength_pa=12e6, shear_cohesion_pa=4e6, friction_coefficient=.6,
    seed_radius_m=100., creation_law=dict(fracture_energy_j_m2=1e6, peak_traction_pa=1e8,
        cohesive_stiffness_pa_m=1e10, compression_stiffness_pa_m=1e10))


def force_family_errors(actual, expected):
    actual, expected = array(actual).reshape(2, -1, 6), array(expected).reshape(2, -1, 6)
    return [float(np.linalg.norm(actual[:, :, start:start+3]-expected[:, :, start:start+3])/
        max(np.linalg.norm(expected[:, :, start:start+3]), 1.)) for start in (0, 3)]


def audit_virtual_work(raw, parent, surface, prestress):
    ndof = surface["parent_ndof"]
    index = np.arange(ndof, dtype=float)
    increments = np.stack((1e-5*np.sin(.173*(index+1)), 1e-5*np.cos(.119*(index+2))))
    fixed(raw, dict(parent_ndof=ndof,
        virtual_increment_provenance="two_explicit_deterministic_parent_A_P_fields_not_released_DOFs",
        exact_zero_jump_for_equal_fields=True,
        virtual_work_claim="discrete_trace_duality_not_parent_constraint_replacement_or_release_balance"),
        "full paired virtual field context")
    close(raw["paired_parent_virtual_increments_m"], increments, "declared deterministic virtual A/P fields", atol=2e-20)
    jump = point_trace(surface, increments[1]-increments[0])
    ref_force = prestress["paired_force_n"]
    born_force = paired_force(surface, prestress["born_current_equivalent_traction_world_pa"], reference=False)
    reference_work = paired_virtual_work(surface, prestress["reference_traction_world_pa"], increments)
    born_work = paired_virtual_work(surface, prestress["born_current_equivalent_traction_world_pa"], increments, reference=False)
    nodal_work = float(np.sum(ref_force*increments))
    fields = dict(virtual_jump_m=jump, reference_paired_parent_force_n=ref_force,
        equivalent_birth_paired_parent_force_n=born_force, reference_virtual_work_j=reference_work,
        birth_virtual_work_j=born_work, paired_nodal_virtual_work_j=nodal_work)
    errors = {name: close(raw[name], value, "paired virtual work "+name, rtol=2e-9,
        atol=1e-3 if "force" in name else 1e-6 if "work" in name else 2e-15) for name, value in fields.items()}
    errors["reference_force_A_P"] = check_force_modes(raw["reference_paired_parent_force_n"], ref_force, "virtual reference force")
    errors["birth_force_A_P"] = check_force_modes(raw["equivalent_birth_paired_parent_force_n"], born_force, "virtual born force")
    close(reference_work, nodal_work, "force dot paired increment equals surface integral", atol=1e-6)
    close(reference_work, born_work, "metric converted virtual work equals reference work", atol=1e-6)
    equal_jump = point_trace(surface, increments[0]-increments[0])
    require(not np.any(equal_jump), "equal virtual banks have a nonzero jump")
    actual_positions = surface["reference_position_m"]+point_trace(surface, parent["runtime_state"]["displacement_m"])
    pose_error = float(np.max(np.linalg.norm(actual_positions-surface["current_position_m"], axis=1), initial=0))
    bound = 128*np.finfo(float).eps*max(float(np.linalg.norm(actual_positions, axis=1).max(initial=0)), 1.)
    require(pose_error <= bound, "frozen material absolute A/P placement differs")
    recorded_error = float(array(raw["frozen_material_absolute_pose_reproduction_max_error_m"], ()))
    require(0 <= recorded_error <= bound, "frozen absolute pose reproduction error exceeds resolution")
    for name in ("reference_saved_force_relative_errors_A_P", "birth_reference_force_relative_errors_A_P"):
        require(array(raw[name], (2,)).min(initial=0) >= 0 and max(raw[name]) < 3e-12,
            "recorded paired force duality errors invalid")
    require(array(raw["virtual_work_relative_errors"], (2,)).min(initial=0) >= 0
        and max(raw["virtual_work_relative_errors"]) < 3e-12, "recorded paired work duality errors invalid")
    require(set(raw) == {"parent_ndof", "virtual_increment_provenance", "paired_parent_virtual_increments_m",
        "virtual_jump_m", "exact_zero_jump_for_equal_fields", "frozen_material_absolute_pose_reproduction_max_error_m",
        "reference_paired_parent_force_n", "equivalent_birth_paired_parent_force_n",
        "reference_saved_force_relative_errors_A_P", "birth_reference_force_relative_errors_A_P",
        "reference_virtual_work_j", "birth_virtual_work_j", "paired_nodal_virtual_work_j",
        "virtual_work_relative_errors", "virtual_work_claim"}, "whole virtual work schema differs")
    return dict(full_parent_A_P_paired_force_and_work_checked=True, tied_equal_fields_have_exact_zero_jump=True,
        frozen_material_full_absolute_pose_reproduction_checked=True, numerical_relative_errors=errors)


def audit_convergence(records, comparisons, parent, *, normal_xyz, offset_m, face_indices):
    require(type(records) is list and [r.get("quadrature_order") for r in records] == [4, 8, 12],
        "complete three-order quadrature convergence catalogue differs")
    recomputed = []
    ndof = 6*len(parent["runtime_state"]["geometry"]["vertices_m"])
    j = np.arange(ndof, dtype=float)
    increments = np.stack((1e-5*np.sin(.173*(j+1)), 1e-5*np.cos(.119*(j+2))))
    for record in records:
        order = record["quadrature_order"]
        surface = embedded_quadrature(parent, normal_xyz, offset_m, face_indices=face_indices, order=order)
        stress = prestress_fields(parent, surface)
        expected = dict(quadrature_order=order, point_count=len(surface["cohort_ids"]),
            patch_count=len(surface["patch_cohort_ids"]),
            total_reference_area_m2=float(surface["reference_area_weights_m2"].sum()),
            total_current_area_m2=float(surface["current_area_weights_m2"].sum()),
            reference_paired_parent_force_n=stress["paired_force_n"],
            reference_virtual_work_j=paired_virtual_work(surface, stress["reference_traction_world_pa"], increments),
            equivalent_birth_virtual_work_j=paired_virtual_work(surface,
                stress["born_current_equivalent_traction_world_pa"], increments, reference=False),
            full_separation_cost_on_explicit_surface_j=DECLARED_LAW["creation_law"]["fracture_energy_j_m2"]*surface["reference_area_weights_m2"].sum(),
            physical_creation_admitted=False, funding_admitted=False)
        require(set(record) == set(expected), "whole quadrature convergence row schema differs")
        for name, value in expected.items():
            if name in ("quadrature_order", "point_count", "patch_count", "physical_creation_admitted", "funding_admitted"):
                fixed(record, {name: value}, "quadrature convergence")
            else:
                close(record[name], value, "independent order "+str(order)+" "+name, rtol=2e-9, atol=1e-3)
        check_force_modes(record["reference_paired_parent_force_n"], expected["reference_paired_parent_force_n"],
            "order "+str(order)+" paired force")
        recomputed.append(expected)
    require(type(comparisons) is list and len(comparisons) == 2, "quadrature comparison catalogue differs")
    reference = recomputed[-1]
    for comparison, record in zip(comparisons, recomputed[:-1]):
        area_errors = [abs(record[name]-reference[name])/max(abs(reference[name]), 1.)
            for name in ("total_reference_area_m2", "total_current_area_m2")]
        force_errors = force_family_errors(record["reference_paired_parent_force_n"], reference["reference_paired_parent_force_n"])
        require(max(area_errors+force_errors) < 2e-7, "independent surface integration failed convergence")
        fixed(comparison, dict(quadrature_order=record["quadrature_order"], reference_order=12), "quadrature comparison orders")
        close(comparison["area_relative_errors_reference_current"], area_errors, "quadrature convergence areas", rtol=1e-4, atol=2e-9)
        close(comparison["reference_force_relative_errors_A_P"], force_errors, "quadrature convergence full A/P forces", rtol=1e-4, atol=2e-9)
        require(set(comparison) == {"quadrature_order", "reference_order", "area_relative_errors_reference_current",
            "reference_force_relative_errors_A_P"}, "whole quadrature comparison schema differs")
    return dict(quadrature_orders_independently_rebuilt=[4, 8, 12], separate_metrics_and_full_A_P_force_convergence_checked=True)


def audit_case(case, parent, observer):
    fixed_fields = dict(computational_fixture=True,
        plane_provenance="same_prior_explicit_normal_and_centroid_anchor_not_physical_nucleation",
        creation_law_provenance="explicit_new_computational_constitutive_example_not_historical_Gc",
        physical_seed_sphere_clipping_performed=False, actual_nucleation_candidate=None,
        physical_creation_admitted=False, physical_release_admitted=False, physical_contact_admitted=False,
        funding_admitted=False, dofs_created_or_released=False, material_memory_changed=False,
        creation_work_debited_from_Genesis_j=0.,
        paired_force_scope="reference_material_stress_trace_duality_not_parent_constraint_reaction")
    fixed(case, fixed_fields, "explicit embedded case scope")
    require(type(case["face_indices"]) is list and len(case["face_indices"]) == 1
        and type(case["face_indices"][0]) is int and case["face_indices"][0] in (0, 40, 79),
        "embedded case outside declared original faces")
    face = case["face_indices"][0]
    require(case["name"] == f"explicit_embedded_plane_face_{face}", "embedded case name differs")
    pin = case["previous_plane_source_pin"]
    path = ROOT/pin["path"]
    require(path.resolve().is_relative_to(ROOT) and path.is_file(), "previous plane source is missing")
    data = path.read_bytes(); prior = json.loads(data)
    require(hashlib.sha256(data).hexdigest() == pin["raw_sha256"], "previous plane source raw pin differs")
    fixed(pin, dict(plane_provenance=prior["plane_provenance"], inherited_anchor=prior["anchor"],
        inherited_origin_xyz_m=prior["origin_xyz_m"]), "unchanged inherited explicit plane")
    require(set(pin) == {"path", "raw_sha256", "plane_provenance", "inherited_anchor", "inherited_origin_xyz_m"},
        "whole inherited plane pin schema differs")
    require(prior["computational_fixture"] is True and prior["physical_surface_selected"] is False
        and prior["physical_creation_admitted"] is False and prior["face_indices"] == [face]
        and prior["probe_material_stress_samples"]["parent_checkpoint_sha256"] == PARENT_CHECKPOINT_SHA256,
        "previous declared plane scope differs")
    close(case["normal_xyz"], prior["normal_xyz"], "unchanged supplied plane normal", atol=2e-14)
    close(case["offset_m"], prior["offset_m"], "unchanged supplied plane offset", atol=2e-8)
    surface, surface_report = audit_embedded_surface(case["embedded_surface"], parent,
        normal_xyz=case["normal_xyz"], offset_m=case["offset_m"], face_indices=[face])
    require(len(surface["selected_cohort_ids"]) == 10 and len(set(surface["cohort_ids"])) == 10,
        "embedded plane lost a live original cohort")
    prestress, stress_report = audit_prestress(case["embedded_prestress"], parent, observer,
        case["embedded_surface"], surface, creation_law=DECLARED_LAW["creation_law"])
    work_report = audit_virtual_work(case["virtual_work"], parent, surface, prestress)
    convergence_report = audit_convergence(case["quadrature_convergence_records"], case["quadrature_convergence_comparisons"],
        parent, normal_xyz=case["normal_xyz"], offset_m=case["offset_m"], face_indices=[face])
    require(set(case) == set(fixed_fields)|{"name", "previous_plane_source_pin", "normal_xyz", "offset_m",
        "face_indices", "embedded_surface", "embedded_prestress", "virtual_work", "quadrature_convergence_records",
        "quadrature_convergence_comparisons"}, "whole computational embedded case schema differs")
    return dict(name=case["name"], original_cohorts=10, surface=surface_report,
        prestress=stress_report, virtual_work=work_report, convergence=convergence_report)


def audit_barriers(barriers, fixture):
    import copy
    require(type(barriers) is list and len(barriers) == 4, "full resealed producer barrier catalogue differs")
    source_nucleation = fixture["nucleation_explicit_computational_law"]
    source_surface = fixture["embedded_cases"][0]["embedded_surface"]
    source_prestress = fixture["embedded_cases"][0]["embedded_prestress"]
    specifications = (
        ("declared_seed_extent_replaced", source_nucleation, "nucleation_sha256", ["declared_prospective_law", "seed_radius_m"], 101.),
        ("physical_seed_invented_on_actual_noevent_source", source_nucleation, "nucleation_sha256", ["candidate"],
            {"forged_eligible_Gamma_on_inactive_actual_source": True}),
        ("frozen_original_material_q_replaced", source_surface, "embedded_surface_sha256", ["mass_coordinate_q", 0],
            source_surface["mass_coordinate_q"][0]+1e-5),
        ("saved_pointwise_reference_traction_replaced", source_prestress, "embedded_prestress_sha256",
            ["pullback", "reference_traction_world_pa", 0, 0], source_prestress["pullback"]["reference_traction_world_pa"][0][0]+1e6))
    for record, (name, source, seal_key, path, new_value) in zip(barriers, specifications):
        forged = copy.deepcopy(source); cursor = forged
        for key in path[:-1]: cursor = cursor[key]
        old = cursor[path[-1]]; cursor[path[-1]] = new_value
        if name == "saved_pointwise_reference_traction_replaced":
            forged["pullback"]["pullback_sha256"] = digest({k: v for k, v in forged["pullback"].items() if k != "pullback_sha256"})
            fixed(record, dict(nested_descriptor_seals_recomputed=["pullback_sha256"]), "nested resealed barrier")
        forged[seal_key] = digest({k: v for k, v in forged.items() if k != seal_key})
        fixed(record, dict(name=name, status="REJECTED", resealed_descriptor_sha256=forged[seal_key],
            descriptor_seal_recomputed=True, source_context_unchanged=True, mutation_path=path,
            old_value=old, new_value=new_value), "genuine resealed descriptor barrier")
        require(type(record.get("exception_type")) is str and type(record.get("reason")) is str
            and record["exception_type"] and record["reason"], "barrier lacks actual refusal metadata")
        expected_keys = {"name", "status", "resealed_descriptor_sha256", "descriptor_seal_recomputed", "source_context_unchanged",
            "mutation_path", "old_value", "new_value", "exception_type", "reason"}
        if name == "saved_pointwise_reference_traction_replaced": expected_keys.add("nested_descriptor_seals_recomputed")
        require(set(record) == expected_keys, "whole mutation barrier schema differs")
    return dict(full_resealed_descriptor_barriers_checked=4, exact_source_bound_deltas_and_nested_seals_checked=True,
        actual_producer_refusal_execution_attested_by_control_capture=True)


def audit_manufactured(raw, law):
    fixed(raw, dict(name="explicit_manufactured_overstress_not_actual_eligibility", computational_fixture=True,
        inherited_planetary_fields_replaced=False, actual_activation_accepted=False,
        physical_creation_admitted=False, creation_work_debited_from_Genesis_j=0.), "manufactured constitutive scope")
    stress = np.array((((20e6, 0., 0.), (0., -2e6, 0.), (0., 0., -8e6)),
        ((-10e6, 20e6, 0.), (20e6, -10e6, 0.), (0., 0., -10e6))))
    close(raw["stress_tensor_pa"], stress, "explicit manufactured full 3D stress", atol=1e-9)
    result = nucleation_criterion(stress, law)
    require(set(raw["criterion"]) == set(result), "whole manufactured criterion schema differs")
    for name, value in result.items(): close(raw["criterion"][name], value, "manufactured convex criterion "+name, atol=1e-12)
    require(np.all(result["yield_score"] > 0), "manufactured positive stress criterion is missing")
    require(set(raw) == {"name", "computational_fixture", "stress_tensor_pa", "criterion",
        "inherited_planetary_fields_replaced", "actual_activation_accepted", "physical_creation_admitted",
        "creation_work_debited_from_Genesis_j"}, "whole manufactured fixture schema differs")
    return dict(cases_examined=2, full_3D_convex_stress_criterion_checked=True,
        manufactured_overstress_does_not_change_authentic_source_eligibility=True)


def audit_fixture(fixture, parent, control_directory=None):
    seal(fixture, "fixture_sha256")
    require(parent.get("checkpoint_sha256") == PARENT_CHECKPOINT_SHA256
        and digest({k: v for k, v in parent.items() if k != "checkpoint_sha256"}) == PARENT_CHECKPOINT_SHA256,
        "complete authentic parent pin differs")
    require(parent.get("source_checkpoint_sha256") == FORMATION_SHA256, "authentic formation ancestor differs")
    require(canonical(fixture["parent_checkpoint"]) == canonical(parent), "complete accepted parent history or fields changed")
    context = dict(format=FORMAT, scope=SCOPE, parent_raw_sha256=PARENT_RAW_SHA256, original_parent_ndof=252,
        prospective_nucleation_law_provenance="explicit_new_convex_constitutive_choice_with_inherited_damage_persistence_gate",
        stress_scope="current_accepted_endpoint_full_cohort_elastic_memory_not_last_interval_average",
        stress_representation="inherited_fixed_frame_small_strain_reference_energy_stress_not_finite_Cauchy_pushforward",
        material_chart="x(N,lambda)=N@(lambda*(X+P)+(A-P))", nucleation_seed_clipping_implemented=False,
        physical_Gamma_selected_on_actual_source=False, actual_activation_accepted=False,
        new_Genesis_time_steps=0, creation_work_debited_from_Genesis_j=0., physical_creation_admitted=False,
        physical_release_admitted=False, physical_contact_admitted=False, funding_admitted=False,
        parent_constraint_reaction_replaced=False, actual_birth_energy_balance_accepted=False,
        material_memory_changed=False, dofs_created_or_released=False, history_reinitialized=False, point_one_complete=False)
    fixed(fixture, context, "whole prospective authentic source control context")
    require(type(fixture["parent_path"]) is str and fixture["parent_path"], "control lacks literal parent provenance path")
    runtime, geometry = parent["runtime_state"], parent["runtime_state"]["geometry"]
    require(len(geometry["cohorts"]) == 800 and abs(geometry["time_myr"]-.8684) < 1e-14
        and runtime["accepted_step_sizes_myr"] == [.0001]*4,
        "authentic material catalogue, clock or accepted intervals differ")
    require(not any(runtime["fault_history"]["fault_active"]), "authentic source has an unexpected active plane")
    observer, onset = fixture["material_stress_observer"], fixture["actual_onset_prerequisite"]
    observer_report = audit_observer(observer, parent)
    onset_report = audit_onset(onset, parent)
    none_report = audit_nucleation(fixture["nucleation_unconfigured"], parent, observer, onset, None)
    law_report = audit_nucleation(fixture["nucleation_explicit_computational_law"], parent, observer, onset, DECLARED_LAW)
    manufactured_report = audit_manufactured(fixture["manufactured_criterion"], DECLARED_LAW)
    require(type(fixture["embedded_cases"]) is list
        and [case.get("face_indices") for case in fixture["embedded_cases"]] == [[0], [40], [79]],
        "three distinct inherited explicit embedded planes differ")
    case_reports = [audit_case(case, parent, observer) for case in fixture["embedded_cases"]]
    barrier_report = audit_barriers(fixture["observed_resealed_source_context_mutation_barriers"], fixture)
    require(set(fixture) == set(context)|{"parent_path", "parent_checkpoint", "actual_onset_prerequisite",
        "material_stress_observer", "nucleation_unconfigured", "nucleation_explicit_computational_law",
        "manufactured_criterion", "embedded_cases", "observed_resealed_source_context_mutation_barriers", "fixture_sha256"},
        "whole actual control fixture schema differs")
    corner = saved_corner_catalogue(parent, DECLARED_LAW)
    require(len(corner["samples"]["cohort_ids"]) == 4800, "all actual cohort corner field catalogue differs")
    if control_directory is not None:
        directory = Path(control_directory)
        before, after = (directory/name for name in ("parent_before.json", "parent_after.json"))
        require(before.read_bytes() == after.read_bytes()
            and hashlib.sha256(before.read_bytes()).hexdigest() == PARENT_RAW_SHA256,
            "whole literal accepted planet changed in control")
        copies = {"control_fixture.json": fixture, "material_stress_observer.json": observer,
            "actual_onset_prerequisite.json": onset, "nucleation_unconfigured.json": fixture["nucleation_unconfigured"],
            "nucleation_explicit_computational_law.json": fixture["nucleation_explicit_computational_law"],
            "manufactured_criterion.json": fixture["manufactured_criterion"]}
        copies.update({case["name"]+".json": case for case in fixture["embedded_cases"]})
        for name, packet in copies.items():
            require(canonical(json.loads((directory/name).read_bytes())) == canonical(packet),
                "external source-bound archived artifact differs: "+name)
    report = dict(format="independent-genesis-volume-nucleation-embedded-audit-0.1", status="PASS", scope=SCOPE,
        producer_imports=False, producer_functions_called=False, public_planetary_replay_performed=False,
        parent_raw_sha256=PARENT_RAW_SHA256, parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
        formation_checkpoint_sha256=FORMATION_SHA256, authentic_parent_full_fields_and_history_checked=True,
        entire_before_after_parent_byte_equality_checked=control_directory is not None,
        time_myr=geometry["time_myr"], cohort_count=800, new_Genesis_time_steps=0,
        material_stress_observer=observer_report, inherited_onset=onset_report,
        actual_nucleation_unconfigured=none_report, actual_nucleation_explicit_law=law_report,
        manufactured_criterion=manufactured_report, embedded_cases=case_reports, mutation_barriers=barrier_report,
        computational_all_live_field_criterion=dict(cohorts_examined=800, corners_examined=4800,
            maximum_tensile_score=float(corner["tensile_score"].max()),
            maximum_shear_score=float(corner["shear_score"].max()), maximum_yield_score=float(corner["yield_score"].max()),
            all_fields_reconstructed_from_saved_six_component_memory=True,
            criterion_does_not_override_authentic_noevent_gate=True),
        physical_Gamma_selected=False, physical_creation_admitted=False, physical_release_admitted=False,
        funding_admitted=False, parent_constraint_reaction_replaced=False,
        actual_birth_energy_balance_accepted=False, point_one_complete=False,
        fixture_sha256=fixture["fixture_sha256"])
    return {**report, "report_sha256": digest(report)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, default=ROOT/
        "analysis/genesis_volume_creation_validation/evidence/actual_continuation/direct_after_second.json")
    parser.add_argument("--control-directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parent_bytes = args.parent_checkpoint.read_bytes()
    require(hashlib.sha256(parent_bytes).hexdigest() == PARENT_RAW_SHA256, "immutable literal parent raw pin differs")
    parent = json.loads(parent_bytes)
    fixture = json.loads(args.fixture.read_bytes())
    report = audit_fixture(fixture, parent, args.control_directory)
    require(not args.output.exists(), "independent output destination already exists")
    args.output.write_bytes(canonical(report)+b"\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
