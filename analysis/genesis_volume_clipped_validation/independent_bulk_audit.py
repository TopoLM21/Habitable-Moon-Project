"""Producer-free checks of source-bound clipped bulk field contractions.

Nodal reference gradients are obtained directly from inverse X.T.  No
producer field, clipping, operator, memory factory or planetary restore is
called.  Literal float64 archives have their own conversion arithmetic;
finite-order observations never certify a physical fracture or weak geometry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.genesis_volume_clipped_validation.independent_geometry_audit import read_source

LD = np.longdouble
EPS = np.finfo(float).eps
MATRIX_NAMES = ("stiffness_cross", "stiffness_enriched", "mass_cross", "mass_enriched")
FIELD_NAMES = ("parent_strain_operator", "enrichment_strain_operator", "parent_trace",
               "enrichment_trace", "point_elastic_strain")


class IndependentBulkRefusal(ValueError):
    pass


def _require(test, message):
    if not test:
        raise IndependentBulkRefusal(message)


def _array(value, shape, name):
    a = np.asarray(value, LD)
    _require(a.shape == shape and np.isfinite(a).all(), name + " shape/nonfinite values differ")
    return a


def _inverse(a):
    """Adjugate of the literal three-row reference triangle."""
    a = np.asarray(a, LD)
    determinant = a[0] @ np.cross(a[1], a[2])
    _require(determinant > 0 and np.isfinite(determinant), "Reference triangle orientation differs")
    return np.column_stack((np.cross(a[1], a[2]), np.cross(a[2], a[0]),
                            np.cross(a[0], a[1]))) / determinant


def _sym(v, g):
    """Engineering strains; v=(3,m), g=(p,3,m)."""
    return np.stack((v[0]*g[:, 0], v[1]*g[:, 1], v[2]*g[:, 2],
                     v[0]*g[:, 1]+v[1]*g[:, 0],
                     v[0]*g[:, 2]+v[2]*g[:, 0],
                     v[1]*g[:, 2]+v[2]*g[:, 1]), axis=1)


def independent_point_fields(domain, elasticity_pa, basis_xyz, memory0, memory1):
    """Differentiate N_i[A_i+(lambda-1)P_i] before assembling any Gram."""
    source = read_source(domain)
    N = np.asarray(domain["barycentric"], LD)
    lam = np.asarray(domain["radial_scale"], LD)
    count = len(lam)
    _require(N.shape == (count, 3) and count > 0 and np.isfinite(N).all(), "Original N points differ")
    _require(np.all(N >= 0) and np.max(np.abs(N.sum(axis=1)-1)) <= 32*EPS,
             "Original N leaves the cohort")
    b, a = map(LD, source.bounds)
    _require(np.all(lam > b) and np.all(lam < a), "Original lambda leaves the cohort")
    relative = _array(domain["centered_birth_offsets_m"], (count, 3), "Centered offsets")
    side = _array(domain["side"], (count,), "Literal Gamma sides")
    _require(set(map(float, side)) <= {-.5, .5}, "Gamma side convention differs")
    C = _array(elasticity_pa, (6, 6), "Elasticity")
    Q = _array(basis_xyz, (3, 3), "Source frame")
    e0, e1 = (_array(v, (3, 6), "Retained material memory") for v in (memory0, memory1))
    _require(np.max(np.abs(Q.T @ Q-np.eye(3))) <= 2e-12 and np.linalg.det(np.asarray(Q, float)) > 0,
             "Source frame differs")
    _require(np.max(np.abs(C-C.T)) <= 64*EPS*np.max(np.abs(C))
             and np.linalg.eigvalsh(np.asarray(C, float)).min() > 0, "Source elasticity differs")
    # Rows of inverse X.T are gradients of lambda*N_i.  Their sum is grad lambda.
    inverse = _inverse(source.X).T
    grad_lambda = inverse.sum(axis=0)
    gradients = (inverse[None]-N[:, :, None]*grad_lambda[None, None]) / lam[:, None, None]
    gradient_p = (lam-1)[:, None, None]*gradients+N[:, :, None]*grad_lambda[None, None]
    g = np.empty((count, 3, 18), LD)
    T = np.zeros((count, 3, 18), LD)
    v = np.zeros((3, 18), LD)
    for vertex in range(3):
        for axis in range(3):
            i = 6*vertex+axis
            g[:, :, i] = gradients[:, vertex]
            g[:, :, i+3] = gradient_p[:, vertex]
            T[:, axis, i], T[:, axis, i+3] = N[:, vertex], (lam-1)*N[:, vertex]
            v[axis, i] = v[axis, i+3] = 1
    local_g = np.einsum("ij,pjm->pim", Q.T, g)
    B = _sym(Q.T @ v, local_g)
    r2 = np.einsum("pi,pi->p", relative, relative)/LD(source.radius)**2
    _require(np.all(r2 < 1), "Source point lies outside the sphere")
    t = (lam-b)/(a-b)
    sphere, angular, radial = (1-r2)**2, 27*np.prod(N, axis=1), 4*t*(1-t)
    w = sphere*angular*radial
    Y = lam[:, None, None]*source.C[None]+source.D[None]
    d1, d2, dl = Y[:, 1]-Y[:, 0], Y[:, 2]-Y[:, 0], N @ source.C
    sphere_grad = -4*(1-r2)[:, None]/LD(source.radius)**2*np.column_stack(
        (np.einsum("pi,pi->p", d1, relative), np.einsum("pi,pi->p", d2, relative),
         np.einsum("pi,pi->p", dl, relative)))
    ag = np.column_stack((27*N[:, 2]*(N[:, 0]-N[:, 1]),
                           27*N[:, 1]*(N[:, 0]-N[:, 2]), np.zeros(count)))
    rg = np.zeros((count, 3), LD)
    rg[:, 2] = 4*(1-2*t)/(a-b)
    parameter = sphere[:, None]*radial[:, None]*ag+angular[:, None]*radial[:, None]*sphere_grad
    parameter += angular[:, None]*sphere[:, None]*rg
    # Use independently derived grad N1, grad N2, grad lambda to convert.
    gradient = parameter[:, 0, None]*gradients[:, 1]+parameter[:, 1, None]*gradients[:, 2]
    gradient += parameter[:, 2, None]*grad_lambda
    local_t = np.einsum("ij,pjm->pim", Q.T, T)
    local_gradient = gradient @ Q
    extra = np.stack((local_t[:, 0]*local_gradient[:, 0, None],
                       local_t[:, 1]*local_gradient[:, 1, None],
                       local_t[:, 2]*local_gradient[:, 2, None],
                       local_t[:, 0]*local_gradient[:, 1, None]+local_t[:, 1]*local_gradient[:, 0, None],
                       local_t[:, 0]*local_gradient[:, 2, None]+local_t[:, 2]*local_gradient[:, 0, None],
                       local_t[:, 1]*local_gradient[:, 2, None]+local_t[:, 2]*local_gradient[:, 1, None]), axis=1)
    E = side[:, None, None]*w[:, None, None]*T
    G = side[:, None, None]*(w[:, None, None]*B+extra)
    angular_samples = np.array(((2/3, 1/6, 1/6), (1/6, 2/3, 1/6), (1/6, 1/6, 2/3)))
    interpolation = N @ np.asarray(np.linalg.inv(angular_samples), LD)
    elastic = interpolation @ e0+(interpolation @ e1)/lam[:, None]
    # Independent construction operands are local to each field component.
    # A nearly zero rotated shear still has the operands of its own dot
    # product; a large neighbouring strain cannot supply its allowance.
    inverse_own = np.abs(inverse)
    lambda_own = inverse_own.sum(axis=0)
    grad_own = (inverse_own[None]+np.abs(N)[:,:,None]*lambda_own[None,None])/lam[:,None,None]
    p_own = np.abs(lam-1)[:,None,None]*grad_own+np.abs(N)[:,:,None]*lambda_own[None,None]
    global_own = np.empty_like(g)
    for vertex in range(3):
        for axis in range(3):
            i = 6*vertex+axis
            global_own[:,:,i], global_own[:,:,i+3] = grad_own[:,vertex], p_own[:,vertex]
    local_own = np.einsum("ij,pjm->pim", np.abs(Q.T), global_own)
    B_own = _sym(np.abs(Q.T) @ np.abs(v), local_own)
    parameter_own = np.abs(sphere[:,None]*radial[:,None]*ag)
    parameter_own += np.abs(angular[:,None]*radial[:,None]*sphere_grad)
    parameter_own += np.abs(angular[:,None]*sphere[:,None]*rg)
    gradient_own = parameter_own[:,0,None]*grad_own[:,1]+parameter_own[:,1,None]*grad_own[:,2]
    gradient_own += parameter_own[:,2,None]*lambda_own
    lt_own = np.einsum("ij,pjm->pim", np.abs(Q.T), np.abs(T))
    lg_own = gradient_own @ np.abs(Q)
    extra_own = np.stack((lt_own[:,0]*lg_own[:,0,None], lt_own[:,1]*lg_own[:,1,None],
        lt_own[:,2]*lg_own[:,2,None],
        lt_own[:,0]*lg_own[:,1,None]+lt_own[:,1]*lg_own[:,0,None],
        lt_own[:,0]*lg_own[:,2,None]+lt_own[:,2]*lg_own[:,0,None],
        lt_own[:,1]*lg_own[:,2,None]+lt_own[:,2]*lg_own[:,1,None]), axis=1)
    construction = dict(parent_strain_operator=B_own,
        enrichment_strain_operator=np.abs(side)[:,None,None]*(np.abs(w)[:,None,None]*B_own+extra_own),
        parent_trace=np.abs(T), enrichment_trace=np.abs(E),
        point_elastic_strain=np.abs(interpolation) @ np.abs(e0)
            +(np.abs(interpolation) @ np.abs(e1))/lam[:,None])
    return dict(parent_strain_operator=B, enrichment_strain_operator=G, parent_trace=T,
                enrichment_trace=E, point_elastic_strain=elastic,
                bubble=w, bubble_gradient_reference_m_inverse=gradient,
                construction_operands=construction)


def _gram(left, right, weights):
    value = np.einsum("pai,paj,p->ij", left, right, weights)
    operands = np.einsum("pai,paj,p->ij", np.abs(left), np.abs(right), weights)
    return value, operands


def recompute_bulk(domain, C, Q, e0, e1, density):
    fields = independent_point_fields(domain, C, Q, e0, e1)
    weights = np.asarray(domain["reference_volume_weights_m3"], LD)
    _require(weights.shape == (len(fields["bubble"]),) and np.all(weights > 0), "Reference weights differ")
    C = np.asarray(C, LD)
    _require(np.isfinite(density) and density > 0, "Retained density differs")
    B, G, T, E = (fields[k] for k in FIELD_NAMES[:4])
    CG = np.einsum("ab,pbi->pai", C, G)
    absCG = np.einsum("ab,pbi->pai", np.abs(C), np.abs(G))
    values, operands = {}, {}
    for name, left, right in (("stiffness_cross", B, CG), ("stiffness_enriched", G, CG),
                               ("mass_cross", T, E), ("mass_enriched", E, E)):
        value, own = _gram(left, right, weights)
        if name.startswith("mass"):
            value, own = LD(density)*value, LD(density)*own
        else:
            own = np.einsum("pai,paj,p->ij", np.abs(left), absCG, weights)
        values[name], operands[name] = value, own
    stress = fields["point_elastic_strain"] @ C.T
    values["enrichment_force_n"] = np.einsum("pai,pa,p->i", G, stress, weights)
    operands["enrichment_force_n"] = np.einsum("pai,pa,p->i", np.abs(G),
        np.abs(fields["point_elastic_strain"]) @ np.abs(C).T, weights)
    return fields, values, operands


def own_component_check(actual, expected, own_operands, *, relative_tolerance=2e-9, operations=1,
                        name="quantity", arithmetic_allowance=None):
    """Compare each quantity to its actual contraction operands, with no floor."""
    actual, expected, own = map(lambda v: np.asarray(v, LD), (actual, expected, own_operands))
    _require(actual.shape == expected.shape == own.shape and np.isfinite(actual).all()
             and np.isfinite(expected).all() and np.isfinite(own).all() and np.all(own >= 0), name+" shape/operand budget differs")
    _require(np.isfinite(relative_tolerance) and relative_tolerance >= 0, "Relative tolerance differs")
    # This is archive-conversion/reconstruction arithmetic, separate from the
    # producer's live-LD reduction and from physical convergence tolerances.
    eps_operations = LD(EPS)*max(int(operations), 64)
    _require(eps_operations < LD(.01), "Archive operation count is unresolved")
    bound = LD(relative_tolerance)*np.maximum(np.abs(actual), np.abs(expected))
    bound += 256*eps_operations/(1-eps_operations)*own
    if arithmetic_allowance is not None:
        allowance = _array(arithmetic_allowance,expected.shape,"Local archive coordinate-conversion allowance")
        _require(np.all(allowance >= 0), "Coordinate-conversion allowance is negative")
        bound += allowance
    error = np.abs(actual-expected)
    _require(np.all(error <= bound), "Independent clipped bulk "+name+" own component differs")
    ratios = np.divide(error, bound, out=np.zeros_like(error), where=bound != 0)
    return dict(maximum_error_budget_ratio=float(np.max(ratios)), maximum_absolute_difference=float(np.max(error)),
                absolute_floor_used=False, coordinate_conversion_allowance_used=arithmetic_allowance is not None,
                scope="literal_archive_conversion_and_independent_current_operands")


def audit_bulk_operator(raw, *, relative_tolerance=2e-9):
    """Independently check every K/M component and retained-memory force."""
    domain = raw["domain"]
    C, Q, e0, e1 = (raw[k] for k in ("elasticity_pa", "basis_xyz", "memory0", "memory1"))
    fields, values, operands = recompute_bulk(domain, C, Q, e0, e1, raw["density_kg_m3"])
    reports = {}
    for name, value in values.items():
        reports[name] = own_component_check(raw[name], value, operands[name],
            relative_tolerance=relative_tolerance, operations=6*len(fields["bubble"])+18, name=name)
        if "roundoff_component_bounds" in raw:
            budget = _array(raw["roundoff_component_bounds"][name],value.shape,"Live reduction budget")
            count = len(fields["bubble"])+32+math.ceil(len(fields["bubble"])/512)
            error = LD(np.finfo(LD).eps)*count
            # Four times the independently reconstructed current-operand
            # reduction envelope tolerates distinct nodal construction. This
            # validates saved live budgets; it grants no physical/work credit.
            cap = 4*error/(1-error)*operands[name]
            _require(np.all(budget >= 0) and np.all(budget <= cap),
                     "Inflated or invalid live component reduction budget: "+name)
    checked_fields = []
    conversion = {}
    if raw.get("points_included"):
        from analysis.genesis_volume_clipped_validation.independent_point_archive_bounds import point_archive_conversion_bounds
        conversion = point_archive_conversion_bounds(domain,C,Q,e0,e1)
    for name in FIELD_NAMES:
        if name in raw:
            value = fields[name]
            # Direct differentiation and source-vertex interpolation have
            # distinct float64 construction operands.  The bound is local to
            # each field component and never enters physical work admission.
            own = fields["construction_operands"][name]
            reports[name] = own_component_check(raw[name], value, own,
                relative_tolerance=relative_tolerance, operations=256, name=name,
                arithmetic_allowance=conversion.get(name))
            checked_fields.append(name)
    return dict(status="PASS", point_count=len(fields["bubble"]), producer_imports=False,
                matrix_components_independently_recomputed=True, retained_source_memory_force_verified=True,
                complete_source_point_fields_checked=checked_fields, checks=reports,
                physical_contraction_accuracy_certified=False, physical_release_admitted=False,
                funding_admitted=False, roundoff_credit_for_funding=False)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode()


def check_seal(value, key):
    _require(isinstance(value, dict) and key in value, "Missing independent input seal: "+key)
    _require(value[key] == hashlib.sha256(canonical({k:v for k,v in value.items() if k != key})).hexdigest(),
             "Independent input seal differs: "+key)


def raw_pin(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1 << 20):
            sha.update(block)
    return dict(path=str(path), bytes=Path(path).stat().st_size, raw_sha256=sha.hexdigest())


def _equal(actual, expected, name):
    _require(np.array_equal(np.asarray(actual), np.asarray(expected)), "Original source "+name+" differs")


def audit_source_binding(fixture, parent, previous, *, case_records=None, case_callback=None):
    """Read literal accepted Genesis and all 800 memories; never restore it."""
    check_seal(fixture, "fixture_sha256")
    check_seal(parent, "checkpoint_sha256")
    check_seal(previous, "fixture_sha256")
    _require(parent["checkpoint_sha256"] == "5d282ceeccc8d76acc38427d08a7e65575fddb509fc4b93115cbe6d318e91349"
        and parent["source_checkpoint_sha256"] == fixture["formation_checkpoint_sha256"]
        and fixture["parent_checkpoint_sha256"] == parent["checkpoint_sha256"]
        and fixture["previous_memory_fixture_sha256"] == previous["fixture_sha256"], "Original Genesis lineage differs")
    runtime = parent["runtime_state"]
    geo = runtime["geometry"]
    _require(geo["time_myr"] == fixture["source_time_myr"] == .8684000000000005
        and len(geo["cohorts"]) == fixture["cohort_count"] == 800
        and len(runtime["accepted_step_sizes_myr"]) == fixture["accepted_interval_count"] == 4
        and not any(runtime["fault_history"]["fault_active"]), "Accepted Genesis history/activation differs")
    old_cases = {case["name"]:case for case in previous["cases"]}
    memories = {c["id"]:c for c in runtime["mechanics"]["state"]["cohorts"]}
    _require(len(fixture["cases"]) == len(old_cases) == 3, "Original three source cases differ")
    results, seen = [], set()
    for case in fixture["cases"] if case_records is None else case_records:
        _require(case["name"] in old_cases and case["name"] not in seen, "Duplicate or missing original source case")
        seen.add(case["name"])
        src = case["source_inputs"]
        prior = old_cases[case["name"]]
        local = prior["embedded_partition"]["local"]
        section = src["section"]
        _require(section == local["section"], "Original material section was replaced")
        cohort = geo["cohorts"][section["cohort_index"]]
        face = geo["faces"][cohort["face_index"]]
        X = np.asarray(geo["vertices_m"])[face]
        u = np.asarray(runtime["displacement_m"]).reshape(-1,6)[face]
        for name, expected in (("triangle_m", X), ("displacement_A_m", u[:,:3]), ("slope_P_m", u[:,3:])):
            _equal(section[name], expected, name)
        _require(section["cohort_id"] == cohort["id"] and section["material_id"] == cohort["material_id"]
            and section["owner_id"] == cohort["owner_id"] and section["mass_kg"] == cohort["mass_kg"]
            and section["volume_m3"] == cohort["volume_m3"], "Original material identity/mass differs")
        memory = memories[cohort["id"]]
        for name, saved in (("memory0", "strain0"), ("memory1", "strain1")):
            _equal(src[name], local[name], name)
            _equal(src[name], memory[saved], name)
            _equal(prior["embedded_partition"][name], [memories[c["id"]][saved] for c in geo["cohorts"]],
                   "all 800 "+name)
        for name in ("elasticity_pa", "basis_xyz"):
            _equal(src[name], local[name], name)
        _require(set(src["previous_local_bulk_blocks"]) == {"parent_stiffness","parent_mass",
            *MATRIX_NAMES,"enrichment_force_n"}, "Complete inherited local block coverage differs")
        for name, value in src["previous_local_bulk_blocks"].items():
            _equal(value,local[name],"inherited local "+name)
        _require(src["original_source_memory_birth"] == prior["source_memory_birth"]
            and src["original_accepted_interval_reaction"] == prior["accepted_interval_reaction"],
            "Original source-born memory or reaction differs")
        for field, archived in (("accepted_interval_mean_e0","interval_average_elastic_constant"),
                                ("accepted_interval_mean_e1","interval_average_elastic_inverse_scale")):
            _equal(src[field],prior["accepted_interval_reaction"][archived][section["cohort_index"]],field)
        _require(src["source_selected_prospective_law"] is None and src["actual_nucleation_candidate"] is None
            and src["physical_Gamma_selected"] is False and src["material_birth_history_replaced"] is False,
            "Computational support was promoted to a physical event")
        _require(src["original_parent_mechanics_sha256"] == hashlib.sha256(canonical(runtime["mechanics"])).hexdigest(),
                 "Original complete parent mechanics differs")
        n = src["parent_ndof"]
        _require(n == prior["embedded_partition"]["parent_ndof"] == 252
            and set(src["literal_original_parent_blocks"]) == {"stiffness","mass_matrix","basal_drag_matrix"},
            "Complete original parent K/M/D or dimension differs")
        for name, archived in src["literal_original_parent_blocks"].items():
            old = prior["embedded_partition"][name]
            data, indices, indptr = [], [], [0]
            for row in range(n):
                for k in range(old["indptr"][row],old["indptr"][row+1]):
                    if old["indices"][k] < n:
                        data.append(old["data"][k]); indices.append(old["indices"][k])
                indptr.append(len(data))
            _require(archived == dict(shape=[n,n],data=data,indices=indices,indptr=indptr),
                     "Original parent "+name+" was reassembled or replaced")
        if case_callback is not None:
            results.append(case_callback(case))
    _require(seen == set(old_cases), "Complete original three-case coverage differs")
    return dict(status="PASS", cases=3, complete_old_memory_cohorts=800, planetary_restore_count=0,
                source_history_reinitialized=False, original_parent_K_M_D_exact=True,
                operator_case_checks=results)


def independently_check_pair(left, right, declared, tolerance):
    """Verify every saved full18 comparison flag on its own signed scale."""
    reports, agreed = {}, True
    A = np.array((0,1,2,6,7,8,12,13,14)); P = A+3
    for name in (*MATRIX_NAMES, "enrichment_force_n"):
        lo, hi = (np.asarray(item["bulk_operator"][name],LD) for item in (left,right))
        budgets = [np.asarray(item["bulk_operator"]["roundoff_component_bounds"][name],LD)
                   for item in (left,right)]
        _require(all(b.shape == lo.shape and np.isfinite(b).all() and np.all(b >= 0) for b in budgets),
                 "Saved component comparison budget differs")
        mask = np.abs(hi-lo) <= LD(tolerance)*np.maximum(np.abs(lo),np.abs(hi))+budgets[0]+budgets[1]
        saved = declared["components"][name]
        _require(saved["agreed"] == bool(np.all(mask))
            and saved["failed_component_count"] == np.count_nonzero(~mask)
            and saved["own_component_relative_scale"] is True
            and saved["positive_scale_floor_used"] is False, "Saved own component finite comparison differs: "+name)
        if mask.ndim == 2:
            for label, rows, cols in (("AA",A,A),("AP",A,P),("PA",P,A),("PP",P,P)):
                local = mask[np.ix_(rows,cols)]
                _require(saved["A_P_blocks"][label] == dict(agreed=bool(np.all(local)),
                    failed_component_count=int(np.count_nonzero(~local))), "Saved A/P block comparison differs")
        reports[name] = dict(agreed=bool(np.all(mask)), failed_component_count=int(np.count_nonzero(~mask)))
        agreed = agreed and bool(np.all(mask))
    _require(set(declared["components"]) == {*MATRIX_NAMES,"enrichment_force_n"},
             "Complete full18 component comparison is missing")
    projected = declared.get("declared_projected_components", {})
    if "declared_projected_blocks" in left:
        _require(set(projected) == {*MATRIX_NAMES,"enrichment_force_n"}, "Complete projected comparison is missing")
    if projected:
        for name, saved in projected.items():
            low, high = (item["declared_projected_blocks"]["blocks"][name] for item in (left,right))
            lo, hi = np.asarray(low,LD), np.asarray(high,LD)
            lb, hb = (np.asarray(item["declared_projected_blocks"]["roundoff_component_bounds"][name],LD)
                      for item in (left,right))
            mask = np.abs(hi-lo) <= LD(tolerance)*np.maximum(np.abs(lo),np.abs(hi))+lb+hb
            _require(saved["agreed"] == bool(np.all(mask))
                and saved["failed_component_count"] == np.count_nonzero(~mask),
                "Saved point-projected comparison differs: "+name)
            agreed = agreed and bool(np.all(mask))
    if "declared_probe_contractions" in left:
        _require([p["name"] for p in declared["declared_probe_results"]]
            == [p["name"] for p in left["declared_probe_contractions"]]
            == [p["name"] for p in right["declared_probe_contractions"]], "Complete declared probe comparison is missing")
    for saved_probe in declared["declared_probe_results"]:
        l = next(p["contractions"] for p in left["declared_probe_contractions"] if p["name"] == saved_probe["name"])
        h = next(p["contractions"] for p in right["declared_probe_contractions"] if p["name"] == saved_probe["name"])
        for name, saved in saved_probe["contractions"].items():
            if name in l["terms"]:
                lo, hi = (LD(v["terms"][name]["value"]) for v in (l,h))
                lb, hb = (LD(v["terms"][name]["absolute_roundoff_budget"]) for v in (l,h))
            else:
                lo, hi = (np.asarray(v[name],LD) for v in (l,h))
                key = name.replace("_force_n", "_force_budget_n")
                lb, hb = (np.asarray(v[key],LD) for v in (l,h))
            mask = np.abs(hi-lo) <= LD(tolerance)*np.maximum(np.abs(lo),np.abs(hi))+lb+hb
            _require(saved["agreed"] == bool(np.all(mask))
                and saved["failed_component_count"] == np.count_nonzero(~mask),
                "Saved declared probe comparison differs: "+name)
            agreed = agreed and bool(np.all(mask))
    _require(declared["finite_order_agreement"] == agreed, "Saved pair convergence summary differs")
    return dict(finite_order_agreement=agreed, full18_components=reports,
                arbitrary_contraction_certified=False)


def audit_case(case):
    """Audit a single source case so literal highest-order tables stay bounded."""
    from analysis.genesis_volume_clipped_validation.independent_contractions_audit import audit_order_contractions
    src = case["source_inputs"]
    _require([p["name"] for p in case["declared_probes"]] ==
        ["mixed_A_P","A_only","P_only","small_mixed","saved_accepted_interval_mean"],
        "Required fixed declared probe coverage differs")
    for field in ("mean_e0","mean_e1"):
        _equal(case["declared_probes"][-1][field],src["accepted_interval_"+field],field)
    _require(case.get("fixed_100m_clipped_support") is not None, "Required fixed 100 m support is missing")
    _require((case.get("original_radius_baseline") is not None) == (src["section"]["cohort_id"] == 1),
             "Required original-radius baseline coverage differs")
    supports = []
    for key in ("fixed_100m_clipped_support", "original_radius_baseline"):
        support = case.get(key)
        if support is None:
            continue
        orders = support["orders"]
        expected_orders = [4,8,12,16] if key == "fixed_100m_clipped_support" else [12,16]
        expected_radius = 100. if key == "fixed_100m_clipped_support" else src["previous_seed_definition"]["seed_radius_m"]
        _require(support["fixed_orders"] == expected_orders and support["required_final_orders"] == [12,16]
            and support["seed_radius_m"] == expected_radius
            and support["relative_tolerance_fixed_before_execution"] == 2e-6,
            "Fixed source radius/orders/relative tolerance were changed")
        virtual = support["virtual_surface_basis"]
        check_seal(virtual,"virtual_basis_sha256")
        check_seal(virtual["finite_seed"],"finite_seed_sha256")
        check_seal(virtual["jump_modes"],"release_modes_sha256")
        seed = virtual["finite_seed"]
        check_seal(seed["surface"],"embedded_surface_sha256")
        _equal(seed["seed_center_world_m"],src["previous_seed_definition"]["seed_center_world_m"],"virtual seed center")
        _require(seed["seed_radius_m"] == expected_radius and seed["selected_cohort_id"] == src["section"]["cohort_id"],
                 "Virtual seed support/source differs")
        _require(virtual["source_section_binding_sha256"] == src["section"]["binding_sha256"]
            and virtual["seed_radius_m"] == expected_radius
            and virtual["jump_modes"]["rank"] == 15 and virtual["jump_modes"]["nullity"] == 3
            and virtual["analytic_complete_kernel_tied"] is True,
            "Declared computational surface basis source/kernel differs")
        _require([r["order"] for r in orders] == support["fixed_orders"], "Declared integration orders differ")
        rows = []
        for item in orders:
            raw = item["bulk_operator"]
            check_seal(raw, "binding_sha256")
            check_seal(raw["domain"], "binding_sha256")
            check_seal(item["declared_projected_blocks"],"archive_sha256")
            _equal(raw["parent_blocks"]["stiffness"],src["previous_local_bulk_blocks"]["parent_stiffness"],
                   "selected old full-cohort parent stiffness")
            _equal(raw["parent_blocks"]["mass"],src["previous_local_bulk_blocks"]["parent_mass"],
                   "selected old full-cohort parent mass")
            for name in (*MATRIX_NAMES,"enrichment_force_n"):
                _equal(raw[name],raw["blocks"][name],"bulk block alias "+name)
                _equal(raw["roundoff_component_bounds"][name],raw["reduction_budgets"][name],"bulk budget alias "+name)
            _require(item["domain"] == raw["domain"], "Archived domain aliases differ")
            for name in ("elasticity_pa", "basis_xyz", "memory0", "memory1"):
                _equal(raw[name], src[name], name)
            domain = raw["domain"]
            _require(domain["source_section_binding_sha256"] == src["section"]["binding_sha256"],
                     "Original domain material section binding differs")
            for name, value in domain["geometry_source"].items():
                if name in src["section"]:
                    _equal(value, src["section"][name], name)
            _equal(domain["seed_center_world_m"], src["previous_seed_definition"]["seed_center_world_m"], "seed center")
            _require(domain["seed_radius_m"] == support["seed_radius_m"], "Declared seed radius changed")
            result = audit_bulk_operator(raw)
            result["declared_contractions"] = audit_order_contractions(item, src, case["declared_probes"],
                support["virtual_surface_basis"]["observable_force_basis"])
            rows.append(dict(order=item["order"], checks=result))
        comparison = support["all_order_diagnostic_comparison"]
        check_seal(comparison,"comparison_sha256")
        _require(len(comparison["comparison_packets"]) == len(orders)-1,"Complete neighboring comparison packets differ")
        for index,packet in enumerate(comparison["comparison_packets"]):
            check_seal(packet,"binding_sha256")
            _require(packet["pair_results"] == [comparison["pair_results"][index]]
                and packet["relative_tolerance"] == 2e-6 and packet["probes"] == case["declared_probes"]
                and packet["operator_bindings"] == [item["bulk_operator"]["operator_binding_sha256"] for item in orders[index:index+2]],
                "Saved comparison packet/operator/probe binding differs")
            _equal(packet["declared_added_basis"],virtual["observable_force_basis"],"comparison source basis")
        pairs = [independently_check_pair(lo,hi,pair,support["relative_tolerance_fixed_before_execution"])
                 for lo,hi,pair in zip(orders,orders[1:],comparison["pair_results"])]
        _require(len(pairs) == len(orders)-1 and support["required_final_order_pair"] == comparison["pair_results"][-1],
                 "Required fixed final order pair differs")
        _require(comparison["finite_order_agreement"] == all(p["finite_order_agreement"] for p in pairs),
                 "All-order diagnostic summary differs")
        supports.append(dict(name=key, orders=rows, pairs=pairs, required_final_pair_agreed=pairs[-1]["finite_order_agreement"]))
    return dict(name=case["name"], supports=supports)


def load_case_records(fixture, directory):
    """Read one sealed case at a time; references must stay within its control."""
    for descriptor in fixture["cases"]:
        if "source_inputs" in descriptor:
            yield descriptor
            continue
        path = (directory/descriptor["case_file"]).resolve()
        _require(path.is_relative_to(directory.resolve()) and path.is_file() and not path.is_symlink(),
                 "Sharded source case path differs")
        pinned = raw_pin(path)
        _require(pinned["raw_sha256"] == descriptor["case_pin"]["raw_sha256"]
            and pinned["bytes"] == descriptor["case_pin"]["bytes"], "Sharded source case raw pin differs")
        with path.open() as stream:
            case = json.load(stream)
        check_seal(case,"case_sha256")
        _require(case["case_sha256"] == descriptor["case_sha256"] and case["name"] == descriptor["name"],
                 "Sharded source case identity differs")
        _require(case["source_inputs"]["section"]["cohort_id"] == descriptor["original_cohort_id"],
                 "Sharded source cohort identity differs")
        yield case


def audit_fixture(fixture, *, case_records=None):
    """Pure structural/operator audit; source lineage is checked by the CLI."""
    check_seal(fixture,"fixture_sha256")
    cases = list(audit_case(case) for case in (fixture["cases"] if case_records is None else case_records))
    final_agreement = all(s["required_final_pair_agreed"] for c in cases for s in c["supports"])
    _require(fixture["numerical_convergence_status"] == ("PASS" if final_agreement else "OPEN"),
             "Overall finite-order convergence summary differs")
    return dict(format="independent-source-bound-clipped-bulk-audit-0.1", status="PASS",
                producer_imports=False, public_planetary_restore_count=0, cases=cases,
                finite_order_numerical_convergence_status="PASS" if final_agreement else "OPEN",
                physical_contraction_accuracy_certified=False, physical_release_admitted=False,
                funding_admitted=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--previous-memory-fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), "Independent audit output must be fresh")
    started = time.monotonic()
    pins = {name:raw_pin(path) for name,path in (("fixture",args.fixture),
        ("parent",args.parent_checkpoint),("previous_memory",args.previous_memory_fixture))}
    _require(pins["parent"]["raw_sha256"] == "62579a0bad128830a66e2f44579cf2b1f5d3835b691225b8d431102f3e4486d7"
        and pins["previous_memory"]["raw_sha256"] == "a98ff7cc89aeb59eb004c05b8ac87f75f2e39a4f10d4454fdf93c288153c9c0e",
        "Independent immutable Genesis input raw pin differs")
    with args.fixture.open() as stream:
        fixture = json.load(stream)
    _require(fixture["declared_fixed_radius_m"] == 100. and fixture["declared_orders"] == [4,8,12,16]
        and fixture["declared_relative_tolerance"] == 2e-6 and fixture["declared_max_points_per_order"] == 250000,
        "Fixed source-control declaration was changed")
    with args.parent_checkpoint.open() as stream:
        parent = json.load(stream)
    with args.previous_memory_fixture.open() as stream:
        previous = json.load(stream)
    for flag in ("history_reinitialized", "accepted_planet_memory_changed", "actual_dynamic_reaction_replaced",
        "physical_Gamma_selected", "automated_goal_enabled", "point_one_complete", "physical_creation_admitted",
        "physical_release_admitted", "physical_contraction_certified", "physical_force_work_admission", "funding_admitted",
        "available_for_creation_funding"):
        _require(fixture[flag] is False,"Physical/refusal flag was promoted: "+flag)
    _require(fixture["new_Genesis_time_steps"] == fixture["new_material_cohorts"] == 0
        and fixture["creation_work_debited_from_Genesis_j"] == 0 and fixture["public_planetary_restore_count"] == 0,
        "Genesis work/material/time was advanced")
    root = Path(__file__).resolve().parents[2]
    source_paths = dict(parent=args.parent_checkpoint,previous_memory_fixture=args.previous_memory_fixture,
        previous_release_report=root/"analysis/genesis_volume_release_validation/evidence/actual_control/report.json",
        previous_memory_final_readback=Path("/workspace/genesis_volume_memory_final_readback_20261010.json"),
        previous_release_final_readback=Path("/workspace/genesis_volume_release_final_readback_20261010.json"))
    _require(set(source_paths) == set(fixture["source_pins"]),"Complete original source pin coverage differs")
    for name,named in fixture["source_pins"].items():
        pinned = raw_pin(source_paths[name])
        _require(pinned["raw_sha256"] == named["raw_sha256"] and pinned["bytes"] == named["bytes"],
                 "Declared immutable source pin differs")
    source = audit_source_binding(fixture,parent,previous,
        case_records=load_case_records(fixture,args.fixture.parent),case_callback=audit_case)
    final_agreement = all(s["required_final_pair_agreed"] for c in source["operator_case_checks"] for s in c["supports"])
    _require(fixture["numerical_convergence_status"] == ("PASS" if final_agreement else "OPEN"),
             "Overall finite-order convergence summary differs")
    for path in (args.fixture.parent/"parent_before.json",args.fixture.parent/"parent_after.json"):
        _require(raw_pin(path)["raw_sha256"] == pins["parent"]["raw_sha256"], "Literal parent before/after changed")
    _require(all(raw_pin(path) == pins[name] for name,path in (("fixture",args.fixture),
        ("parent",args.parent_checkpoint),("previous_memory",args.previous_memory_fixture))),
        "Independent audit original input bytes changed")
    report = dict(format="independent-source-bound-clipped-bulk-audit-0.1", status="PASS",
        scope="independent_source_fields_full_components_declared_virtual_work_and_saved_comparison_flags",
        input_pins=pins, fixture_sha256=fixture["fixture_sha256"], source_binding=source,
        finite_order_numerical_convergence_status="PASS" if final_agreement else "OPEN",
        producer_imports=False, public_planetary_restore_count=0,
        physical_contraction_accuracy_certified=False, physical_release_admitted=False,
        funding_admitted=False, roundoff_credit_for_funding=False,
        actual_seconds=time.monotonic()-started)
    report["sha256"] = hashlib.sha256(canonical(report)).hexdigest()
    with args.output.open("xb") as stream:
        stream.write(canonical(report)+b"\n")
    print(json.dumps(dict(status=report["status"], numerical_convergence_status=report["finite_order_numerical_convergence_status"],
                         actual_seconds=report["actual_seconds"], sha256=report["sha256"])))


if __name__ == "__main__":
    main()
