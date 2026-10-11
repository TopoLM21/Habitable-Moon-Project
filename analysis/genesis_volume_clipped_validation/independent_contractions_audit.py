"""Producer-free source-frame work and operative-point contraction checks.

Every small work term and force component is recomputed from the original
material field.  Parent work uses an independently differentiated full-cohort
analytic integral.  No global energy, rounded full-18 Gram contraction,
producer admission tolerance, or large neighbouring quantity supplies credit.
"""
from __future__ import annotations

import numpy as np

from .independent_bulk_audit import (
    IndependentBulkRefusal, independent_point_fields, own_component_check,
    _inverse, _sym,
)

LD = np.longdouble
EPS = np.finfo(float).eps
TERMS = ("parent_work_j", "cross_parent_work_j", "cross_added_work_j", "self_work_j",
         "stiffness_cross_bilinear_j", "stiffness_self_bilinear_j",
         "mass_cross_bilinear_kg_m2", "mass_self_bilinear_kg_m2")
BLOCKS = ("stiffness_cross", "stiffness_enriched", "mass_cross", "mass_enriched", "enrichment_force_n")


def _require(value, message):
    if not value:
        raise IndependentBulkRefusal(message)


def _array(value, shape, name):
    raw = np.asarray(value)
    _require(raw.dtype.kind in "ifu" and raw.shape == shape, name+" shape/type differs")
    result = np.asarray(raw, LD)
    _require(np.isfinite(result).all(), name+" nonfinite values differ")
    return result


def _project(field, own, coefficients):
    return (np.einsum("pai,ij->paj", field, coefficients),
            np.einsum("pai,ij->paj", own, np.abs(coefficients)))


def _inner(left, right, left_own, right_own, C, weights):
    return (np.einsum("pa,ab,pb,p->", left, C, right, weights),
            np.einsum("pa,ab,pb,p->", left_own, np.abs(C), right_own, weights))


def _gram(left, right, left_own, right_own, C, weights):
    return (np.einsum("pai,ab,pbj,p->ij", left, C, right, weights),
            np.einsum("pai,ab,pbj,p->ij", left_own, np.abs(C), right_own, weights))


def _parent_work(source, Q, C, e0, e1, q):
    """Direct inverse-X derivatives and independently written radial moments."""
    X = _array(source["triangle_m"], (3, 3), "Original parent triangle")
    inverse = _inverse(X).T
    grad_lambda = inverse.sum(axis=0)
    N = np.array(((2/3, 1/6, 1/6), (1/6, 2/3, 1/6), (1/6, 1/6, 2/3)), LD)
    B0, B1 = np.zeros((3, 6, 18), LD), np.zeros((3, 6, 18), LD)
    own0, own1 = np.zeros_like(B0), np.zeros_like(B1)
    for vertex in range(3):
        base = inverse[vertex][None]-N[:, vertex, None]*grad_lambda[None]
        base_own = np.abs(inverse[vertex])[None]+np.abs(N[:, vertex, None])*np.abs(inverse).sum(axis=0)[None]
        for axis in range(3):
            v = Q.T[:, axis, None]
            g0 = np.tile(Q.T@inverse[vertex], (3, 1))[:, :, None]
            g1 = (base@Q)[:, :, None]
            o0 = np.tile(np.abs(Q.T)@np.abs(inverse[vertex]), (3, 1))[:, :, None]
            o1 = (base_own@np.abs(Q))[:, :, None]
            i = 6*vertex+axis
            B0[:, :, i+3] = _sym(v, g0)[:, :, 0]
            B1[:, :, i], B1[:, :, i+3] = _sym(v, g1)[:, :, 0], -_sym(v, g1)[:, :, 0]
            own0[:, :, i+3] = _sym(np.abs(v), o0)[:, :, 0]
            own1[:, :, i], own1[:, :, i+3] = _sym(np.abs(v), o1)[:, :, 0], _sym(np.abs(v), o1)[:, :, 0]
    b, a = map(LD, source["lambda_bounds"])
    denominator = a*a+a*b+b*b
    mean, variance = LD(1.5)*(a+b)/denominator, LD(.75)*((a-b)/denominator)**2
    volume = LD(source["volume_m3"])
    Bmean, Bmeanown = B0+mean*B1, own0+abs(mean)*own1
    emean, emeanown = e0+mean*e1, np.abs(e0)+abs(mean)*np.abs(e1)
    dqmean = np.einsum("pai,i->pa", Bmean, q)
    dqown = np.einsum("pai,i->pa", Bmeanown, np.abs(q))
    dq1, dq1own = np.einsum("pai,i->pa", B1, q), np.einsum("pai,i->pa", own1, np.abs(q))
    first, firstown = _inner(dqmean, emean, dqown, emeanown, C, np.ones(3, LD))
    last, lastown = _inner(dq1, e1, dq1own, np.abs(e1), C, np.ones(3, LD))
    return volume/3*(first+variance*last), volume/3*(firstown+abs(variance)*lastown)


def _claimed_budget_check(saved, expected, operands, operations, name):
    """Reject invented reported roundoff large enough to launder agreement."""
    saved, expected, own = (np.asarray(v, LD) for v in (saved, expected, operands))
    _require(saved.shape == expected.shape == own.shape and np.isfinite(saved).all() and np.all(saved >= 0),
             name+" reported roundoff shape/nonfinite/negative differs")
    cost = LD(EPS)*max(int(operations), 64)
    _require(cost < LD(.01), name+" independent operation count is unresolved")
    allowance = 256*cost/(1-cost)*own
    _require(np.all(saved <= allowance), name+" reported roundoff exceeds its independent own operands")


def _probe_expected(raw, source, probe, fields):
    domain = raw["domain"]
    C, Q = np.asarray(raw["elasticity_pa"], LD), np.asarray(raw["basis_xyz"], LD)
    q = _array(probe["parent_increment_m"], (18,), "Declared parent probe")
    a = _array(probe["enrichment_increment_m"], (18,), "Declared added probe")
    z = _array(probe.get("mean_z_m", np.zeros(18)), (18,), "Declared mean added memory")
    e0 = _array(probe.get("mean_e0", raw["memory0"]), (3, 6), "Declared complete mean memory0")
    e1 = _array(probe.get("mean_e1", raw["memory1"]), (3, 6), "Declared complete mean memory1")
    N, lam = np.asarray(domain["barycentric"], LD), np.asarray(domain["radial_scale"], LD)
    samples = np.array(((2/3, 1/6, 1/6), (1/6, 2/3, 1/6), (1/6, 1/6, 2/3)))
    angular = N@np.asarray(np.linalg.inv(samples), LD)
    elastic = angular@e0+(angular@e1)/lam[:, None]
    elasticown = np.abs(angular)@np.abs(e0)+(np.abs(angular)@np.abs(e1))/lam[:, None]
    weights = np.asarray(domain["reference_volume_weights_m3"], LD)
    own = fields["construction_operands"]
    projections = {}
    for name, coefficients, key in (("Bq", q, "parent_strain_operator"),
        ("Ga", a, "enrichment_strain_operator"), ("Gz", z, "enrichment_strain_operator"),
        ("Tq", q, "parent_trace"), ("Ea", a, "enrichment_trace")):
        value, operands = _project(fields[key], own[key], coefficients[:, None])
        projections[name] = value[:, :, 0], operands[:, :, 0]
    Bq, Bqo = projections["Bq"]; Ga, Gao = projections["Ga"]; Gz, Gzo = projections["Gz"]
    Tq, Tqo = projections["Tq"]; Ea, Eao = projections["Ea"]
    density = LD(raw["density_kg_m3"])
    terms = dict(parent_work_j=_parent_work(source, Q, C, e0, e1, q),
        cross_parent_work_j=_inner(Bq, Gz, Bqo, Gzo, C, weights),
        cross_added_work_j=_inner(Ga, elastic, Gao, elasticown, C, weights),
        self_work_j=_inner(Ga, Gz, Gao, Gzo, C, weights),
        stiffness_cross_bilinear_j=_inner(Bq, Ga, Bqo, Gao, C, weights),
        stiffness_self_bilinear_j=_inner(Ga, Ga, Gao, Gao, C, weights),
        mass_cross_bilinear_kg_m2=tuple(density*v for v in _inner(Tq, Ea, Tqo, Eao, np.eye(3, dtype=LD), weights)),
        mass_self_bilinear_kg_m2=tuple(density*v for v in _inner(Ea, Ea, Eao, Eao, np.eye(3, dtype=LD), weights)))
    force, forceown = _gram(fields["enrichment_strain_operator"], (elastic+Gz)[:, :, None],
        own["enrichment_strain_operator"], (elasticown+Gzo)[:, :, None], C, weights)
    parentforce, parentforceown = _gram(fields["parent_strain_operator"], Gz[:, :, None],
        own["parent_strain_operator"], Gzo[:, :, None], C, weights)
    return terms, dict(enrichment_force_n=(force[:, 0], forceown[:, 0]),
                       added_parent_force_n=(parentforce[:, 0], parentforceown[:, 0]))


def _projected_expected(raw, fields, W):
    W = _array(W, np.asarray(W).shape, "Declared operative basis")
    _require(W.ndim == 2 and W.shape[0] == 18 and 1 <= W.shape[1] <= 18,
             "Declared operative basis dimension differs")
    own = fields["construction_operands"]
    G, Go = _project(fields["enrichment_strain_operator"], own["enrichment_strain_operator"], W)
    E, Eo = _project(fields["enrichment_trace"], own["enrichment_trace"], W)
    C, weights = np.asarray(raw["elasticity_pa"], LD), np.asarray(raw["domain"]["reference_volume_weights_m3"], LD)
    density = LD(raw["density_kg_m3"])
    values = {}
    for name, left, right, lo, ro, middle, factor in (
        ("stiffness_cross", fields["parent_strain_operator"], G, own["parent_strain_operator"], Go, C, LD(1)),
        ("stiffness_enriched", G, G, Go, Go, C, LD(1)),
        ("mass_cross", fields["parent_trace"], E, own["parent_trace"], Eo, np.eye(3, dtype=LD), density),
        ("mass_enriched", E, E, Eo, Eo, np.eye(3, dtype=LD), density),
        ("enrichment_force_n", G, fields["point_elastic_strain"][:, :, None], Go,
            own["point_elastic_strain"][:, :, None], C, LD(1))):
        value, operands = _gram(left, right, lo, ro, middle, weights)
        if name == "enrichment_force_n":
            value, operands = value[:, 0], operands[:, 0]
        values[name] = factor*value, factor*operands
    return values


def audit_order_contractions(item, source_inputs, declared_probes, W=None):
    """Check one archived order; source identity is verified by the caller."""
    raw = item.get("bulk_operator", item.get("operator", item))
    fields = independent_point_fields(raw["domain"], raw["elasticity_pa"], raw["basis_xyz"], raw["memory0"], raw["memory1"])
    operations = 6*len(fields["bubble"])+64
    saved = item.get("declared_probe_contractions")
    _require(type(saved) is list and len(saved) == len(declared_probes), "Declared probe rows differ")
    reports = []
    for row, probe in zip(saved, declared_probes):
        _require(row.get("name") == probe.get("name"), "Declared probe identity/order differs")
        result = row.get("result", row.get("contractions"))
        _require(type(result) is dict and result.get("projection_before_Gram") is True,
                 "Declared work was not projected at live points before Gram")
        _require(result.get("physical_contraction_certified") is False
                 and result.get("available_for_creation_funding") is False,
                 "Declared virtual probe claims physical admission or funding")
        expected, forces = _probe_expected(raw, source_inputs.get("section", raw["domain"]["geometry_source"]), probe, fields)
        _require(set(result.get("terms", {})) == set(TERMS), "Complete signed small-work terms differ")
        checks = {}
        for name, (value, own) in expected.items():
            term = result["terms"][name]
            checks[name] = own_component_check(term["value"], value, own, operations=operations, name=name)
            _claimed_budget_check(term["absolute_roundoff_budget"], value, own, operations, name)
        for name, (value, own) in forces.items():
            checks[name] = own_component_check(result[name], value, own, operations=operations, name=name)
            budget_name = "enrichment_force_budget_n" if name == "enrichment_force_n" else "added_parent_force_budget_n"
            _claimed_budget_check(result[budget_name], value, own, operations, name)
        reports.append(dict(name=probe["name"], status="PASS", checks=checks))
    projected_report = None
    if W is not None:
        saved_projection = item.get("declared_projected_blocks")
        _require(type(saved_projection) is dict and saved_projection.get("projection_before_Gram") is True,
                 "Source operative projected blocks are missing")
        _require(np.array_equal(np.asarray(saved_projection.get("added_basis")), np.asarray(W)),
                 "Source operative projected basis differs")
        _require(np.array_equal(np.asarray(saved_projection.get("parent_basis")), np.eye(18)),
                 "Source operative original parent coordinates differ")
        _require(saved_projection.get("physical_contraction_certified") is False
                 and saved_projection.get("available_for_creation_funding") is False,
                 "Source operative blocks claim physical admission or funding")
        blocks = saved_projection.get("blocks", saved_projection)
        budgets = saved_projection.get("roundoff_component_bounds", saved_projection.get("reduction_budgets"))
        _require(type(budgets) is dict, "Source operative own roundoff bounds are missing")
        checks = {}
        for name, (value, own) in _projected_expected(raw, fields, W).items():
            checks[name] = own_component_check(blocks[name], value, own, operations=operations, name="projected "+name)
            _claimed_budget_check(budgets[name], value, own, operations, "projected "+name)
        projected_report = dict(status="PASS", coordinate_count=int(np.asarray(W).shape[1]), checks=checks)
    return dict(status="PASS", producer_imports=False, projection_before_Gram=True,
        complete_signed_work_parts_verified=True, declared_probe_results=reports,
        declared_projected_blocks=projected_report, physical_contraction_certified=False,
        arbitrary_contraction_certified=False, funding_admitted=False,
        roundoff_credit_for_funding=False)
