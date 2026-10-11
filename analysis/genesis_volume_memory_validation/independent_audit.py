"""Producer-free oracle for frozen-chart enriched Maxwell memory and reactions.

Only preserved Genesis data and older independent numerical oracles are used.
The new constitutive producer and planetary checkpoint loader are never imported.
Nonzero enrichment paths are private mathematical probes: this audit does not
accept a Genesis interval, select a physical fracture or authorize its funding.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.genesis_volume_seed_validation import independent_audit as seed
from analysis.genesis_volume_continuation_validation import independent_audit as volume

PARENT_RAW_SHA256 = seed.PARENT_RAW_SHA256
PARENT_CHECKPOINT_SHA256 = seed.PARENT_CHECKPOINT_SHA256
FORMATION_SHA256 = seed.FORMATION_SHA256
SECONDS_PER_MYR = 365.25*86400*1e6
PREVIOUS_SEED_FIXTURE_RAW_SHA256 = "9c9a669639bb8d7c08a15266481f883894c86227a26a6cc9d2bd0759e51509bb"
PREVIOUS_SEED_FIXTURE_PATH = ROOT/"analysis/genesis_volume_seed_validation/evidence/actual_control/control_fixture.json"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError("Independent enriched memory audit: "+message)


def array(value, shape=None):
    raw = np.asarray(value)
    require(raw.dtype.kind in "fiu" and (shape is None or raw.shape == shape)
            and np.isfinite(raw).all(), "invalid finite numeric array")
    return np.asarray(raw, float)


def close(actual, expected, name, *, rtol=2e-8, atol=1e-10):
    a, b = array(actual), array(expected)
    require(a.shape == b.shape, name+" dimensions differ")
    scale = max(float(np.max(abs(b), initial=0.)), np.finfo(float).tiny)
    error = float(np.max(abs(a-b), initial=0.))
    require(error <= atol+rtol*scale, name+" differs")
    return error/max(scale, atol, np.finfo(float).tiny)


def exact(actual, expected, name):
    require(canonical(actual) == canonical(expected), name+" differs")


def seal(raw, key):
    require(type(raw) is dict and raw.get(key) == digest(
        {k: v for k, v in raw.items() if k != key}), key+" seal differs")


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [plain(v) for v in value]
    return value


def authenticate_parent(parent):
    require(parent.get("checkpoint_sha256") == PARENT_CHECKPOINT_SHA256,
            "accepted parent checkpoint differs")
    seal(parent, "checkpoint_sha256")
    require(parent.get("source_checkpoint_sha256") == FORMATION_SHA256,
            "molten formation ancestor differs")
    runtime = parent["runtime_state"]
    geometry = runtime["geometry"]
    require(len(geometry["cohorts"]) == 800 and len(geometry["vertices_m"]) == 42
            and abs(geometry["time_myr"]-.8684) < 1e-14
            and runtime["accepted_step_sizes_myr"] == [.0001]*4
            and not any(runtime["fault_history"]["fault_active"]),
            "authentic catalogue, clock, accepted steps or fault history differs")
    return geometry


def source_context(parent, definition, *, order=12):
    """Rebuild source coefficients, full material integral and compact G.

    Radial integration uses independently generated mass-fraction Gauss nodes.
    Compact geometry uses the preserved older scalar chart-inverse oracle.
    Neither radial moments nor matrices from the new producer are consulted.
    """
    geometry = parent["runtime_state"]["geometry"]
    cohorts = geometry["cohorts"]
    ids = [c["id"] for c in cohorts]
    selected = definition["original_cohort_id"]
    require(ids.count(selected) == 1, "selected original live cohort differs")
    ci = ids.index(selected)
    frames = seed.prior.source_frames(parent)
    vertices = array(geometry["vertices_m"])
    faces = np.asarray(geometry["faces"], int)
    shell = parent["source_packet"]["source"]["parameters"]["shell"]
    memories = {c["id"]: c for c in parent["runtime_state"]["mechanics"]["state"]["cohorts"]}
    require(set(memories) == set(ids), "full inherited material memory coverage differs")
    radial, weights, volumes, matrices, b0s, b1s, dofs = [], [], [], [], [], [], []
    for cohort in cohorts:
        face = int(cohort["face_index"])
        X, Q = vertices[faces[face]], frames[face]
        q, wr = volume.radial_rule(cohort["q_lo"], cohort["live_q_hi"], order=32)
        lam = volume.conical_scales(X, geometry["column_mass_kg"][face],
            geometry["density_kg_m3"], q)[0]
        radial.append(lam); weights.append(wr)
        volumes.append(geometry["column_mass_kg"][face]*(cohort["live_q_hi"]-cohort["q_lo"])
                       /geometry["density_kg_m3"])
        matrices.append(volume.isotropic_matrix(shell["young_modulus_pa"], shell["poisson_ratio"],
            parent["runtime_state"]["fault_history"]["damage"][face], shell["residual_stiffness"]))
        b0, b1 = volume.independent_strain_operators(X, Q, seed.prior.SOURCE_ANGULAR_POINTS)
        b0s.append(b0); b1s.append(b1)
        dofs.append((6*faces[face, :, None]+np.arange(6)).ravel())
    local = seed.compact_bulk_terms(parent, definition["normal_xyz"], definition["offset_m"],
        definition["seed_center_world_m"], definition["seed_radius_m"],
        cohort_id=selected, order=order)
    context = dict(cohort_ids=ids, selected_index=ci, parent_ndof=6*len(vertices),
        radial_scale=array(radial), radial_weights=array(weights), volume_m3=array(volumes),
        elasticity_pa=array(matrices), B0=array(b0s), B1=array(b1s), dofs=np.asarray(dofs, int),
        inherited0=array([memories[i]["strain0"] for i in ids]),
        inherited1=array([memories[i]["strain1"] for i in ids]),
        local=local, interpolation=local["barycentric"]@np.linalg.inv(seed.prior.SOURCE_ANGULAR_POINTS),
        selected_parent_dofs=np.asarray(dofs[ci], int), parent=parent, definition=definition)
    n = context["parent_ndof"]
    K = np.zeros((n, n))
    for j in range(len(ids)):
        lam = context["radial_scale"][j]
        B = context["B0"][j, :, None]+context["B1"][j, :, None]/lam[None, :, None, None]
        weighted = B*np.sqrt(context["volume_m3"][j]*context["radial_weights"][j][None, :, None, None]/3)
        flat = weighted.reshape(-1, 18)
        stiffness = flat.T@(context["elasticity_pa"][j]@weighted).reshape(-1, 18)
        K[np.ix_(context["dofs"][j], context["dofs"][j])] += stiffness
    context["original_stiffness"] = K
    context["original_mass"] = volume.mass_matrix(geometry["vertices_m"], geometry["faces"], geometry["column_mass_kg"],
        geometry["density_kg_m3"], geometry["cohorts"])
    inputs = parent["runtime_state"]["last_report"]["audit_input"]
    context["original_drag"], context["original_external"] = volume.basal_drag_and_force(
        geometry["vertices_m"], geometry["faces"], geometry["column_mass_kg"],
        geometry["density_kg_m3"], geometry["cohorts"], inputs["basal_drag_pa_s_m"],
        array(inputs["basal_traction_at_vertices_pa"])[faces])
    return context


def field_inner_decomposition(context, a0, a1, za, b0, b1, zb, *, cohort_factors=None):
    """Independent full inner product as original, signed cross, enriched self."""
    n = len(context["cohort_ids"])
    a0, a1, b0, b1 = (array(v, (n, 3, 6)) for v in (a0, a1, b0, b1))
    za, zb = array(za, (18,)), array(zb, (18,))
    factors = np.ones(n) if cohort_factors is None else array(cohort_factors, (n,))
    lam = context["radial_scale"][:, None, :, None]
    ea, eb = a0[:, :, None, :]+a1[:, :, None, :]/lam, b0[:, :, None, :]+b1[:, :, None, :]/lam
    original = np.einsum("cari,cij,carj,cr,c,c->", ea, context["elasticity_pa"], eb,
        context["radial_weights"], context["volume_m3"], factors)/3
    ci, local = context["selected_index"], context["local"]
    P, l = context["interpolation"], local["radial_scale"][:, None]
    sa, sb = P@a0[ci]+(P@a1[ci])/l, P@b0[ci]+(P@b1[ci])/l
    ga, gb = np.einsum("pai,i->pa", local["enrichment_strain_operator"], za), np.einsum(
        "pai,i->pa", local["enrichment_strain_operator"], zb)
    inner = lambda a, b: np.einsum("pi,ij,pj,p->", a, local["elasticity_pa"], b,
                                 local["reference_volume_weights_m3"])
    cross = factors[ci]*(inner(sa, gb)+inner(ga, sb))
    enriched = factors[ci]*inner(ga, gb)
    result = np.asarray((original, cross, enriched), float)
    require(np.isfinite(result).all(), "full material inner product overflowed")
    return result


def full_generalized_force(context, mean0, mean1, mean_z):
    """Rebuild force from the full averaged regular strain field."""
    n = len(context["cohort_ids"])
    mean0, mean1 = array(mean0, (n, 3, 6)), array(mean1, (n, 3, 6))
    mean_z = array(mean_z, (18,))
    result = np.zeros(context["parent_ndof"]+18)
    for ci in range(n):
        lam = context["radial_scale"][ci]
        e = mean0[ci, :, None, :]+mean1[ci, :, None, :]/lam[None, :, None]
        B = context["B0"][ci, :, None]+context["B1"][ci, :, None]/lam[None, :, None, None]
        stress = e@context["elasticity_pa"][ci].T
        localforce = context["volume_m3"][ci]/3*np.einsum("arbi,arb,r->i", B, stress,
                                                                       context["radial_weights"][ci])
        np.add.at(result, context["dofs"][ci], localforce)
    ci, local = context["selected_index"], context["local"]
    P, lam = context["interpolation"], local["radial_scale"][:, None]
    old = P@mean0[ci]+(P@mean1[ci])/lam
    added = np.einsum("pai,i->pa", local["enrichment_strain_operator"], mean_z)
    stress = (old+added)@local["elasticity_pa"].T
    extra_parent = np.einsum("pai,pa,p->i", local["parent_strain_operator"],
        added@local["elasticity_pa"].T, local["reference_volume_weights_m3"])
    np.add.at(result, context["selected_parent_dofs"], extra_parent)
    result[-18:] = np.einsum("pai,pa,p->i", local["enrichment_strain_operator"], stress,
                            local["reference_volume_weights_m3"])
    return result


def enriched_parent_force(context, mean_z):
    local = context["local"]
    strain = np.einsum("pai,i->pa", local["enrichment_strain_operator"], array(mean_z, (18,)))
    force = np.einsum("pai,pa,p->i", local["parent_strain_operator"],
        strain@local["elasticity_pa"].T, local["reference_volume_weights_m3"])
    result = np.zeros(context["parent_ndof"])
    np.add.at(result, context["selected_parent_dofs"], force)
    return result


def audit_force_decomposition(raw, context, expected):
    n = context["parent_ndof"]
    original = full_generalized_force(context, expected["mean0"], expected["mean1"], np.zeros(18))[:n]
    cross = enriched_parent_force(context, expected["mean_coefficients_m"])
    old_error = seed.mode_errors(raw["parent_mean_force_n"], original, "separate old average force", rtol=5e-7, atol=1e-5)
    cross_error = seed.mode_errors(raw["enriched_parent_force_n"], cross, "separate enriched parent coupling force", rtol=3e-6, atol=1e-12)
    require(np.array_equal(array(raw["mean_generalized_force_n"])[:n],
        array(raw["parent_mean_force_n"], (n,))+array(raw["enriched_parent_force_n"], (n,))),
        "literal full parent force decomposition differs")
    return dict(enriched_parent_coupling_checked_separately=True, literal_parent_force_sum_exact=True,
        old_force_relative_errors=old_error, enriched_coupling_relative_errors=cross_error)


def maxwell_step(context, old0, old1, old_z, increment0, increment1, increment_a, ratio):
    """Solve constant-driving Maxwell ODE and independently integrate its work.

    Time averages and temporal Gram terms are numerical Gauss integrals from
    the older independent ODE oracle, not the new producer's beta/theta law.
    Frozen temperatures, material bounds and elasticity imply zero thermal,
    damage and remelt terms in this explicitly private mathematical step.
    """
    n = len(context["cohort_ids"])
    old0, old1, increment0, increment1 = (array(v, (n, 3, 6))
        for v in (old0, old1, increment0, increment1))
    old_z, increment_a = array(old_z, (18,)), array(increment_a, (18,))
    ratio = array(ratio, (n,))
    require(np.all(ratio >= 0), "negative Maxwell ratio")
    factors = np.asarray([volume.temporal_factors(x) for x in ratio])
    relax = np.exp(-ratio)
    driving = np.asarray([1. if x == 0 else -math.expm1(-float(x))/x for x in ratio])
    end0, end1 = relax[:, None, None]*old0+driving[:, None, None]*increment0, relax[:, None, None]*old1+driving[:, None, None]*increment1
    mean0 = factors[:, 0, None, None]*old0+factors[:, 1, None, None]*increment0
    mean1 = factors[:, 0, None, None]*old1+factors[:, 1, None, None]*increment1
    ci = context["selected_index"]
    end_z = relax[ci]*old_z+driving[ci]*increment_a
    mean_z = factors[ci, 0]*old_z+factors[ci, 1]*increment_a
    inner = lambda a0, a1, az, b0, b1, bz, f=None: field_inner_decomposition(
        context, a0, a1, az, b0, b1, bz, cohort_factors=f)
    before = .5*inner(old0, old1, old_z, old0, old1, old_z)
    energy = .5*inner(end0, end1, end_z, end0, end1, end_z)
    work = inner(increment0, increment1, increment_a, mean0, mean1, mean_z)
    loss = (inner(old0, old1, old_z, old0, old1, old_z, factors[:, 2])
        +inner(old0, old1, old_z, increment0, increment1, increment_a, factors[:, 3])
        +inner(increment0, increment1, increment_a, increment0, increment1, increment_a, factors[:, 4]))
    balance = energy-before-work+loss
    scales = np.maximum.reduce((abs(energy), abs(before), abs(work), abs(loss), np.ones(3)))
    require(np.all(abs(balance) <= 3e-10*scales), "independent separated full-field Maxwell work identity failed")
    require(loss.sum() >= -2e-12*max(abs(loss.sum()), abs(before.sum()), 1.),
            "full Maxwell dissipation is negative")
    return dict(strain0=end0, strain1=end1, coefficients_m=end_z, mean0=mean0, mean1=mean1,
        mean_coefficients_m=mean_z, energy_decomposition_j=energy,
        work_decomposition_j=work, dissipation_decomposition_j=loss,
        before_energy_decomposition_j=before, elastic_energy_j=float(energy.sum()),
        mechanical_work_j=float(work.sum()), viscous_dissipation_j=float(loss.sum()),
        balance_residual_j=float(balance.sum()), mean_generalized_force_n=full_generalized_force(
            context, mean0, mean1, mean_z))


def audit_response(raw, expected):
    """Check small enrichment terms independently of the large old energy."""
    errors = {}
    for key in ("strain0", "strain1", "coefficients_m", "mean0", "mean1", "mean_coefficients_m"):
        errors[key] = close(raw[key], expected[key], "Maxwell "+key, rtol=2e-9, atol=2e-13)
    for key in ("energy_decomposition_j", "work_decomposition_j", "dissipation_decomposition_j"):
        actual, wanted = array(raw[key], (3,)), array(expected[key], (3,))
        errors[key] = [close(actual[i], wanted[i], key+" "+name,
            rtol=2e-9 if i == 0 else 3e-6, atol=1e-10)
            for i, name in enumerate(("original", "signed cross", "enriched self"))]
    for key in ("elastic_energy_j", "mechanical_work_j", "viscous_dissipation_j"):
        close(raw[key], expected[key], "full-field "+key, rtol=2e-9, atol=1e-5)
        close(raw[key], sum(raw[{"elastic_energy_j": "energy_decomposition_j",
            "mechanical_work_j": "work_decomposition_j",
            "viscous_dissipation_j": "dissipation_decomposition_j"}[key]]),
            "literal decomposition "+key, rtol=3e-15, atol=1e-10)
    force = array(raw["mean_generalized_force_n"])
    n = len(force)-18
    require(n == len(expected["mean_generalized_force_n"])-18, "full averaged force dimensions differ")
    seed.mode_errors(force[:n], expected["mean_generalized_force_n"][:n], "averaged old parent force",
                     rtol=5e-7, atol=1e-5)
    seed.mode_errors(force[n:], expected["mean_generalized_force_n"][n:], "averaged added bulk force",
                     rtol=5e-7, atol=1e-5)
    for key in ("thermal_work_j", "damage_energy_change_j", "remelt_energy_removed_j"):
        if key in raw: require(raw[key] == 0., "frozen private response "+key+" is nonzero")
    scale = max(abs(raw["elastic_energy_j"]), abs(raw["mechanical_work_j"]),
                raw["viscous_dissipation_j"], 1.)
    require(abs(raw["balance_residual_j"]) <= 5e-12*scale, "full response balance differs")
    return dict(independent_temporal_quadrature=True, original_cross_self_terms_checked_separately=True,
        full_dissipation_nonnegative=True, signed_cross_dissipation_not_creation_funding=True,
        numerical_relative_errors=errors)


def source_maxwell_ratios(parent, before_time_myr, after_time_myr):
    """Regenerate frozen-temperature viscosities and private clock ratios."""
    require(type(before_time_myr) in (int, float) and type(after_time_myr) in (int, float)
            and math.isfinite(before_time_myr) and math.isfinite(after_time_myr)
            and after_time_myr > before_time_myr, "private clock is not later finite time")
    shell = parent["source_packet"]["source"]["parameters"]["shell"]
    temperatures = array([c["temperature_k"] for c in parent["runtime_state"]["mechanics"]["state"]["cohorts"]])
    require(np.all(temperatures > 0), "inherited material temperatures are not positive")
    viscosity = np.clip(shell["viscosity_reference_pa_s"]*np.exp(np.clip(
        shell["activation_energy_j_mol"]/8.314462618*(1/temperatures
            -1/shell["viscosity_reference_temperature_k"]), -60, 60)),
        shell["viscosity_min_pa_s"], shell["viscosity_max_pa_s"])
    return (after_time_myr-before_time_myr)*SECONDS_PER_MYR*shell["young_modulus_pa"]/viscosity


def state_coefficients(raw):
    records = raw["state"]["parent_mechanics"]["state"]["cohorts"]
    return array([r["strain0"] for r in records]), array([r["strain1"] for r in records])


MEMORY_SCOPE = "private_fixed_material_frozen_birth_Maxwell_no_release_or_funding"
STATE_FIELDS = {"parent_mechanics", "partition_binding_sha256", "birth_parent_memory_sha256",
    "birth_time_myr", "displacement_m", "coefficients_m", "initial_energy_j", "elastic_energy_j",
    "mechanical_work_j", "viscous_dissipation_j", "thermal_work_j", "damage_energy_change_j",
    "remelt_energy_removed_j", "balance_residual_j", "journal", "birth_operator_energy_roundoff_j"}


def audit_state_archive(raw, context, partition_sha256, *, expected0, expected1,
        expected_z, expected_a, expected_time, expected_steps, expected_journal,
        expected_ledgers=None, birth=False):
    """Bind serialization to the authentic birth and independently replayed path."""
    seal(raw, "packet_sha256")
    require(set(raw) == {"format", "scope", "state", "physical_release_admitted", "funding_admitted",
        "new_Genesis_time_myr", "new_material_birth_count", "packet_sha256"}, "private memory archive schema differs")
    exact(raw["format"], "genesis-volume-enriched-memory-0.1", "private memory format")
    exact(raw["scope"], MEMORY_SCOPE, "private memory scope")
    require(raw["physical_release_admitted"] is False and raw["funding_admitted"] is False
            and raw["new_Genesis_time_myr"] == 0 and raw["new_material_birth_count"] == 0,
            "private archive admits physical release, funding or planetary time")
    state = raw["state"]
    require(type(state) is dict and set(state) == STATE_FIELDS, "private state schema differs")
    parent = context["parent"]
    birth_memory = parent["runtime_state"]["mechanics"]
    exact(state["partition_binding_sha256"], partition_sha256, "frozen compact partition binding")
    exact(state["birth_parent_memory_sha256"], birth_memory["packet_sha256"], "authentic inherited birth memory binding")
    exact(state["birth_time_myr"], context["parent"]["runtime_state"]["geometry"]["time_myr"], "authentic birth clock")
    exact(state["birth_operator_energy_roundoff_j"], context["birth_operator_energy_roundoff_j"],
          "recorded inherited scalar reduction roundoff")
    exact(state["journal"], expected_journal, "private path journal")
    mechanics = state["parent_mechanics"]
    seal(mechanics, "packet_sha256")
    require(set(mechanics) == set(birth_memory), "original memory packet schema differs")
    for key in ("format", "energy_scope", "voigt_order"):
        exact(mechanics[key], birth_memory[key], "original mechanics "+key)
    old = birth_memory["state"]
    current = mechanics["state"]
    require(set(current) == set(old), "original memory state schema differs")
    exact(current["source_identity"], old["source_identity"], "original material source identity")
    exact(current["time_myr"], expected_time, "private mechanics clock")
    exact(current["accepted_steps"], old["accepted_steps"]+expected_steps, "private mechanics step count")
    require(len(current["cohorts"]) == len(old["cohorts"]), "original material coverage differs")
    for got, original in zip(current["cohorts"], old["cohorts"]):
        require(set(got) == set(original), "original cohort memory schema differs")
        exact(got["id"], original["id"], "original material cohort ID")
        exact(got["temperature_k"], original["temperature_k"], "fixed inherited cohort temperature")
    e0, e1 = state_coefficients(raw)
    close(e0, expected0, "replayed original constant memory", rtol=2e-9, atol=2e-13)
    close(e1, expected1, "replayed original radial memory", rtol=2e-9, atol=2e-13)
    close(state["coefficients_m"], expected_z, "replayed enriched elastic history", rtol=2e-9, atol=2e-13)
    close(state["displacement_m"], expected_a, "replayed enriched displacement history", rtol=0, atol=0)
    for key in ("initial_energy_j", "thermal_work_j", "damage_energy_change_j", "remelt_energy_removed_j"):
        exact(state[key], old[key], "unchanged inherited full ledger "+key)
        exact(current[key], old[key], "unchanged inherited original ledger "+key)
    if birth:
        exact(mechanics, birth_memory, "birth full original memory packet")
        require(not np.any(array(state["coefficients_m"], (18,)))
                and not np.any(array(state["displacement_m"], (18,)))
                and state["journal"] == [], "birth must have exact zero new amplitudes and history")
        for key in STATE_FIELDS-{"parent_mechanics", "partition_binding_sha256", "birth_parent_memory_sha256",
            "birth_time_myr", "displacement_m", "coefficients_m", "journal", "birth_operator_energy_roundoff_j"}:
            exact(state[key], old[key], "literal birth ledger "+key)
    if expected_ledgers is not None:
        for key, wanted in expected_ledgers.items():
            close(state[key], wanted, "replayed complete ledger "+key, rtol=3e-12, atol=1e-5)
    energy_parts = .5*field_inner_decomposition(context, e0, e1, array(state["coefficients_m"]),
        e0, e1, array(state["coefficients_m"]))
    close(current["elastic_energy_j"], energy_parts[0], "replayed original-only elastic energy", rtol=3e-12, atol=1e-5)
    close(state["elastic_energy_j"], energy_parts.sum(), "replayed full elastic energy", rtol=3e-12, atol=1e-5)
    cumulative = (state["initial_energy_j"]+state["mechanical_work_j"]+state["thermal_work_j"]
        -state["viscous_dissipation_j"]+state["damage_energy_change_j"]-state["remelt_energy_removed_j"])
    scale = max(*(abs(state[k]) for k in ("initial_energy_j", "elastic_energy_j", "mechanical_work_j",
        "thermal_work_j", "viscous_dissipation_j", "damage_energy_change_j", "remelt_energy_removed_j")), 1.)
    require(abs(state["elastic_energy_j"]-cumulative) <= 5e-12*scale
            and abs(state["elastic_energy_j"]-cumulative-state["balance_residual_j"]) <= 5e-12*scale,
            "replayed cumulative full-field balance differs")
    return dict(authentic_birth_memory_preserved=True, full_old_cohort_memory_and_ids_checked=True,
        zero_amplitude_birth_exact=birth, private_path_replayed=True,
        displacement_and_elastic_memory_separate=True,
        source_history_reinitialized=False, planetary_time_advanced=False,
        cross_and_self_energy_j=energy_parts[1:].tolist())


REACTION_SCOPE = "hypothetical_frozen_endpoint_compact_basis_tied_over_last_accepted_interval"


def dynamic_reaction_expected(context):
    """Independently reconstruct all last-accepted momentum equation terms."""
    parent, local = context["parent"], context["local"]
    runtime, geometry = parent["runtime_state"], parent["runtime_state"]["geometry"]
    report = runtime["last_report"]
    audit, slip = report["audit_input"], report["weak_plane_slip"]
    n = context["parent_ndof"]
    dt = report["dt_myr"]*SECONDS_PER_MYR
    require(report["after_time_myr"] == geometry["time_myr"]
        and report["requested_dt_myr"] == runtime["accepted_step_sizes_myr"][-1]
        and report["dt_myr"] == report["after_time_myr"]-report["before_time_myr"]
        and dt > 0 and math.isfinite(dt), "accepted interval clock binding differs")
    inc, v0, q0 = (array(audit[key], (n,)) for key in
        ("increment_m", "before_velocity_m_s", "before_displacement_m"))
    require(np.array_equal(q0+inc, array(runtime["displacement_m"], (n,)))
        and np.array_equal(2*inc/dt-v0, array(runtime["velocity_m_s"], (n,))),
        "accepted source kinematics binding differs")
    mean0, mean1 = (array(slip[key], context["inherited0"].shape) for key in
        ("average_strain0", "average_strain1"))
    M, D, external = context["original_mass"], context["original_drag"], context["original_external"]
    meanforce = full_generalized_force(context, mean0, mean1, np.zeros(18))
    inertia, drag = 2*(M@(inc-dt*v0))/(dt*dt), D@inc/dt
    added_inertia = 2*local["mass_cross"].T@(inc[context["selected_parent_dofs"]]
        -dt*v0[context["selected_parent_dofs"]])/(dt*dt)
    parent_residual = inertia+drag+meanforce[:n]-external
    added_residual = added_inertia+meanforce[n:]
    return dict(dt_s=dt, parent_increment_m=inc, parent_before_velocity_m_s=v0,
        interval_average_elastic_constant=mean0, interval_average_elastic_inverse_scale=mean1,
        parent_inertia_force_n=inertia, parent_drag_force_n=drag,
        parent_average_internal_force_n=meanforce[:n], parent_external_force_n=external,
        parent_residual_n=parent_residual, added_inertia_force_n=added_inertia,
        added_drag_force_n=np.zeros(18), added_average_internal_force_n=meanforce[n:],
        added_external_force_n=np.zeros(18), added_residual_n=added_residual,
        endpoint_static_constraint_reaction_n=-local["enrichment_force_n"],
        constraint_multipliers_n=-added_residual,
        constraint_force_n=np.r_[np.zeros(n), -added_residual],
        remaining_residual_n=np.r_[parent_residual, np.zeros(18)],
        enrichment_increment_m=np.zeros(18), enrichment_before_velocity_m_s=np.zeros(18),
        enrichment_elastic_memory_m=np.zeros(18))


def audit_dynamic_reaction(raw, context, partition_raw, finite_seed_sha256):
    """Retain inertia even if too small to alter the rounded total reaction."""
    seal(raw, "seed_reaction_sha256")
    expected = dynamic_reaction_expected(context)
    numerical_fields = set(expected)
    body = {k: v for k, v in raw.items() if k not in numerical_fields|{"seed_reaction_sha256", "reaction_context_sha256"}}
    require(raw.get("reaction_context_sha256") == digest(body), "accepted reaction context seal differs")
    parent, runtime = context["parent"], context["parent"]["runtime_state"]
    pinned = dict(format="genesis-accepted-interval-tied-compact-seed-reaction-0.1", scope=REACTION_SCOPE,
        accepted_parent_checkpoint_sha256=parent["checkpoint_sha256"], formation_checkpoint_sha256=parent["source_checkpoint_sha256"],
        source_identity=runtime["geometry"]["source_identity"], time_myr=runtime["geometry"]["time_myr"],
        before_time_myr=runtime["last_report"]["before_time_myr"], after_time_myr=runtime["geometry"]["time_myr"],
        accepted_step_sizes_myr=runtime["accepted_step_sizes_myr"],
        accepted_final_report_sha256=digest(runtime["last_report"]), full_memory_sha256=digest(runtime["mechanics"]),
        birth_history_sha256=digest(runtime["history"]), fault_history_sha256=digest(runtime["fault_history"]),
        endpoint_geometry_sha256=digest(runtime["geometry"]), finite_seed_binding_sha256=finite_seed_sha256,
        embedded_partition_binding_sha256=partition_raw["embedded_partition_sha256"],
        partition_quadrature_order=partition_raw["local"]["quadrature_order"],
        selected_cohort_id=context["cohort_ids"][context["selected_index"]], parent_ndof=context["parent_ndof"],
        parent_operator_blocks_preserved_exactly=True, parent_residual_preserved_exactly=True,
        constraint_force_parent_pullback_exact_zero=True, added_residual_after_constraints_exact_zero=True,
        tied_constraint_work_j=0., exact_canonical_jump_work_j=0., added_mode_basal_drag_and_external_force_exact_zero=True,
        mass_convention="after_mass_with_accepted_start_comoving_velocity",
        average_response_scope="saved_last_accepted_interval_not_endpoint_static_stress",
        basis_scope="accepted_endpoint_frozen_computational_basis_hypothetically_extended_over_saved_interval",
        added_memory_scope="new_zero_compact_coefficients_old_calculated_memory_unchanged",
        reaction_semantics="negative_added_inertia_plus_accepted_average_internal_residual",
        surface_force_is_dynamic_reaction=False, surface_traction_replacement=None,
        new_accepted_intervals=0, new_material_births=0, released_physical_dofs=0,
        transition_time_increment_myr=0., creation_work_debited_from_Genesis_j=0.)
    pinned.update({key: False for key in ("physical_creation_admitted", "physical_release_admitted", "funding_admitted",
        "historical_seed_claimed", "source_memory_reinitialized", "surface_traction_replacement_admitted", "activation_accepted")})
    for key, wanted in pinned.items(): exact(raw.get(key), wanted, "accepted reaction "+key)
    n = context["parent_ndof"]
    # Force families keep very small P and weak signed mass modes observable.
    errors = {}
    for key in numerical_fields:
        if key in ("parent_increment_m", "parent_before_velocity_m_s", "interval_average_elastic_constant",
            "interval_average_elastic_inverse_scale", "dt_s", "enrichment_increment_m",
            "enrichment_before_velocity_m_s", "enrichment_elastic_memory_m"):
            require(np.array_equal(array(raw[key]), array(expected[key])), "literal accepted reaction "+key+" differs")
        elif key == "parent_residual_n":
            # This small residual is a cancellation of separately audited terms.
            archived = array(raw[key], (n,))
            require(np.array_equal(archived, array(raw["parent_inertia_force_n"])+array(raw["parent_drag_force_n"])
                +array(raw["parent_average_internal_force_n"])-array(raw["parent_external_force_n"])),
                "literal parent momentum residual differs")
        elif key in ("constraint_force_n", "remaining_residual_n"):
            continue
        else:
            errors[key] = seed.mode_errors(raw[key], expected[key], "dynamic reaction "+key,
                rtol=3e-6 if "inertia" in key else 5e-7,
                atol=1e-35 if "inertia" in key else 1e-5)
    ai, ad, am, ae = (array(raw[k], (18,)) for k in ("added_inertia_force_n", "added_drag_force_n",
        "added_average_internal_force_n", "added_external_force_n"))
    ar, multiplier = array(raw["added_residual_n"], (18,)), array(raw["constraint_multipliers_n"], (18,))
    require(np.array_equal(ar, ai+ad+am-ae) and np.array_equal(multiplier, -ar),
            "literal added momentum residual or negative constraint reaction differs")
    require(not np.any(ad) and not np.any(ae), "bubble boundary drag or external force is nonzero")
    force, remaining = array(raw["constraint_force_n"], (n+18,)), array(raw["remaining_residual_n"], (n+18,))
    require(np.array_equal(force, np.r_[np.zeros(n), multiplier])
        and np.array_equal(remaining, np.r_[array(raw["parent_residual_n"]), ar]+force)
        and not np.any(remaining[n:]) and not np.any(force[:n]),
        "tied reaction changes parent residual or fails to cancel added modes")
    # The original parent momentum tolerance is assessed against its operands.
    pr = array(raw["parent_residual_n"])
    scale = max(np.linalg.norm(array(raw["parent_average_internal_force_n"])),
        np.linalg.norm(array(raw["parent_external_force_n"])), 1.)
    relative = float(np.linalg.norm(pr)/scale)
    close(raw["recomputed_parent_momentum_relative_residual"], relative, "literal parent momentum relative residual",
          rtol=2e-12, atol=1e-14)
    exact(raw["parent_momentum_relative_residual"], runtime["last_report"]["momentum_equation_relative_residual"],
          "reported accepted momentum residual")
    return dict(saved_accepted_average_response_checked=True, endpoint_stress_not_substituted=True,
        added_inertia_checked_separately_even_below_total_force_roundoff=True,
        exact_negative_added_residual_checked=True, original_parent_residual_preserved=True,
        all_18_modes_tied=True, zero_constraint_work=True,
        historical_seed_claimed=False, dynamic_reaction_replaced_in_planet=False,
        numerical_relative_errors=errors)


def audit_partition_basis(raw, context):
    """Bind the private memory's entire frozen operator to calculated material."""
    seal(raw, "embedded_partition_sha256")
    parent, n = context["parent"], context["parent_ndof"]
    geometry = parent["runtime_state"]["geometry"]
    require(raw["source_checkpoint_sha256"] == parent["source_checkpoint_sha256"]
        and raw["source_identity"] == geometry["source_identity"] and raw["time_myr"] == geometry["time_myr"]
        and raw["parent_ndof"] == n and raw["ndof"] == n+18
        and raw["selected_cohort_index"] == context["selected_index"]
        and raw["selected_cohort_id"] == context["definition"]["original_cohort_id"],
        "frozen partition source, clock or cohort identity differs")
    for key in ("physical_creation_admitted", "physical_release_admitted", "funding_admitted"):
        require(raw[key] is False, "frozen partition admits "+key)
    require(raw["released_dof_count"] == 0 and raw["new_material_birth_count"] == 0
            and raw["creation_work_debited_from_Genesis_j"] == 0, "frozen partition physical accounting differs")
    require(np.array_equal(array(raw["memory0"]), context["inherited0"])
        and np.array_equal(array(raw["memory1"]), context["inherited1"])
        and np.array_equal(array(raw["selected_parent_dofs"]), context["selected_parent_dofs"]),
        "frozen partition does not preserve literal full inherited memory or original DOFs")
    local, wanted = raw["local"], context["local"]
    errors = {}
    for key in ("barycentric", "radial_scale", "reference_volume_weights_m3", "birth_volume_weights_m3",
        "parent_trace", "enrichment_trace", "parent_strain_operator", "enrichment_strain_operator",
        "elasticity_pa", "basis_xyz", "memory0", "memory1"):
        errors[key] = close(local[key], wanted[key], "independent frozen compact "+key,
            rtol=0 if key in {"barycentric", "radial_scale"} else 8e-7 if "strain_operator" in key
                else 5e-8 if "weights" in key or "enrichment" in key else 2e-9,
            atol=3e-13 if key in {"barycentric", "radial_scale"} else 1e-12)
    for key in ("stiffness_cross", "stiffness_enriched", "mass_cross", "mass_enriched"):
        errors[key] = seed.matrix_mode_errors(local[key], wanted[key], "independent frozen "+key, rtol=3e-6, atol=1e-20)
    matrices = {"stiffness": context["original_stiffness"], "mass_matrix": context["original_mass"],
                "basal_drag_matrix": context["original_drag"]}
    for key, original in matrices.items():
        matrix = seed.dense_csr(raw[key], "private partition "+key)
        seed.matrix_mode_errors(matrix[:n, :n], original, "original tied "+key, rtol=3e-9, atol=1e-20)
        if key != "basal_drag_matrix":
            stem = "stiffness" if key == "stiffness" else "mass"
            cross = np.zeros((n, 18)); cross[context["selected_parent_dofs"]] = wanted[stem+"_cross"]
            seed.matrix_mode_errors(matrix[:n, n:], cross, "whole private "+stem+" cross", rtol=3e-6, atol=1e-20)
            seed.matrix_mode_errors(matrix[n:, :n], cross.T, "whole private "+stem+" transpose cross", rtol=3e-6, atol=1e-20)
            seed.matrix_mode_errors(matrix[n:, n:], wanted[stem+"_enriched"], "whole private "+stem+" enriched", rtol=3e-6, atol=1e-20)
        else:
            require(not np.any(matrix[n:]) and not np.any(matrix[:, n:]), "frozen bubble adds basal drag")
    seed.mode_errors(raw["external_force_n"][:n], context["original_external"], "original private external force", rtol=3e-9, atol=1e-5)
    require(not np.any(array(raw["external_force_n"])[n:]), "frozen bubble adds external force")
    R = seed.dense_csr(raw["prolongation"], "private tied prolongation")
    J = seed.dense_csr(raw["tie_operator"], "private tied operator")
    require(np.array_equal(R, np.r_[np.eye(n), np.zeros((18, n))])
        and np.array_equal(J, np.c_[np.zeros((18, n)), np.eye(18)]),
        "private partition releases a mode or changes canonical tie")
    initial = full_generalized_force(context, context["inherited0"], context["inherited1"], np.zeros(18))
    seed.mode_errors(raw["initial_force_n"][:n], initial[:n], "birth original saved force", rtol=5e-7, atol=1e-5)
    seed.mode_errors(raw["initial_force_n"][n:], initial[n:], "birth added saved force", rtol=5e-7, atol=1e-5)
    old_energy = parent["runtime_state"]["mechanics"]["state"]["elastic_energy_j"]
    close(raw["initial_elastic_energy_j"], old_energy, "birth arithmetic operator reduction", rtol=5e-12, atol=0.)
    context["birth_operator_energy_roundoff_j"] = raw["initial_elastic_energy_j"]-old_energy
    return dict(original_full_K_M_D_forcing_rebuilt=True, all_18_new_modes_tied=True,
        frozen_material_G_rebuilt=True, original_saved_memory_preserved_exactly=True,
        birth_operator_energy_roundoff_j=context["birth_operator_energy_roundoff_j"],
        roundoff_difference_not_physical_work_or_funding=True, numerical_relative_errors=errors)


def audit_case(case, parent):
    definition = case["seed_definition"]
    face = definition["original_face_index"]
    seed.audit_seed_definition(definition, parent, face)
    partition = case["embedded_partition"]
    order = partition["local"]["quadrature_order"]
    require(order == 12, "final memory compact quadrature must use preserved order twelve")
    context = source_context(parent, definition, order=order)
    partition_report = audit_partition_basis(partition, context)
    pin = partition["embedded_partition_sha256"]
    birth = case["birth_memory"]
    old = parent["runtime_state"]["mechanics"]["state"]
    birth_report = audit_state_archive(birth, context, pin, expected0=context["inherited0"],
        expected1=context["inherited1"], expected_z=np.zeros(18), expected_a=np.zeros(18),
        expected_time=old["time_myr"], expected_steps=0, expected_journal=[], birth=True)
    context["reaction_binding_sha256"] = case["accepted_interval_reaction"]["seed_reaction_sha256"]
    source_birth_report = audit_source_memory_birth(case["source_memory_birth"], context, birth, partition)
    steps = case["private_steps"]
    require(type(steps) is list and len(steps) == 2, "two private nonzero continuations are required")
    before0, before1, beforez, beforea, time = (context["inherited0"], context["inherited1"],
        np.zeros(18), np.zeros(18), old["time_myr"])
    full_ledgers = {key: old[key] for key in ("elastic_energy_j", "mechanical_work_j", "viscous_dissipation_j")}
    original_work, original_loss = old["mechanical_work_j"], old["viscous_dissipation_j"]
    journal, reports, before_archive = [], [], birth
    for index, step in enumerate(steps):
        ratio = source_maxwell_ratios(parent, time, step["after_time_myr"])
        close(step["ratio"], ratio, "independent frozen-source Maxwell ratios", rtol=2e-12, atol=0.)
        dq, da = array(step["parent_increment_m"], (context["parent_ndof"],)), array(step["enrichment_increment_m"], (18,))
        require(np.any(da), "private test must exercise nonzero enriched displacement")
        inc0 = np.einsum("cqai,ci->cqa", context["B0"], dq[context["dofs"]])
        inc1 = np.einsum("cqai,ci->cqa", context["B1"], dq[context["dofs"]])
        expected = maxwell_step(context, before0, before1, beforez, inc0, inc1, da, ratio)
        response = step["response"]
        saved_ratio = array(step["ratio"])
        beta = np.divide(-np.expm1(-saved_ratio), saved_ratio,
            out=np.ones_like(saved_ratio), where=saved_ratio != 0)
        binding = dict(partition_binding_sha256=pin, before_state_sha256=digest(before_archive["state"]),
            after_time_myr=step["after_time_myr"], ratio=step["ratio"], beta=beta.tolist())
        exact(response["trial_binding_sha256"], digest(binding), "private trial source/state binding")
        response_report = audit_response(response, expected)
        response_report["separate_parent_force_decomposition"] = audit_force_decomposition(response, context, expected)
        close(response["parent_increment_m"], dq, "literal response parent increment", rtol=0, atol=0)
        close(response["enrichment_increment_m"], da, "literal response enriched increment", rtol=0, atol=0)
        close(response["displacement_m"], beforea+da, "separate cumulative enriched displacement", rtol=0, atol=0)
        parent_response = response["parent_response"]
        close(parent_response["strain0"], expected["strain0"], "retained original response memory0", rtol=2e-9, atol=2e-13)
        close(parent_response["strain1"], expected["strain1"], "retained original response memory1", rtol=2e-9, atol=2e-13)
        for key, parts in (("elastic_energy_j", "energy_decomposition_j"), ("mechanical_work_j", "work_decomposition_j"),
            ("viscous_dissipation_j", "dissipation_decomposition_j")):
            close(parent_response[key], expected[parts][0], "retained original response "+key, rtol=3e-12, atol=1e-5)
        for key in ("thermal_work_j", "damage_energy_change_j", "remelt_energy_removed_j"):
            require(parent_response[key] == 0., "private original response changed thermal or material law")
        full_ledgers["elastic_energy_j"] = expected["elastic_energy_j"]
        full_ledgers["mechanical_work_j"] += expected["mechanical_work_j"]
        full_ledgers["viscous_dissipation_j"] += expected["viscous_dissipation_j"]
        original_work += expected["work_decomposition_j"][0]
        original_loss += expected["dissipation_decomposition_j"][0]
        journal.append(dict(after_time_myr=step["after_time_myr"], parent_increment_m=step["parent_increment_m"],
            enrichment_increment_m=step["enrichment_increment_m"], ratio=step["ratio"]))
        after = step["after_memory"]
        state_report = audit_state_archive(after, context, pin, expected0=expected["strain0"],
            expected1=expected["strain1"], expected_z=expected["coefficients_m"], expected_a=beforea+da,
            expected_time=step["after_time_myr"], expected_steps=index+1, expected_journal=journal,
            expected_ledgers=full_ledgers)
        original = after["state"]["parent_mechanics"]["state"]
        close(original["mechanical_work_j"], original_work, "replayed original cumulative work", rtol=3e-12, atol=1e-5)
        close(original["viscous_dissipation_j"], original_loss, "replayed original cumulative loss", rtol=3e-12, atol=1e-5)
        reports.append(dict(index=index+1, private_after_time_myr=step["after_time_myr"],
            constitutive=response_report, serialized_state=state_report))
        before0, before1, beforez, beforea, time = expected["strain0"], expected["strain1"], expected["coefficients_m"], beforea+da, step["after_time_myr"]
        before_archive = after
    # Their equality binds each independently checked state to the restored
    # continuation; seals alone never authorize a replacement history.
    exact(case["restarted_after_first_memory"], steps[0]["after_memory"], "first source-journal restart")
    exact(case["restarted_after_second_memory"], steps[1]["after_memory"], "continued source-journal restart")
    require(np.any(beforez) and not np.array_equal(beforez, beforea),
        "two-step private path does not distinguish relaxed memory from displacement")
    reaction = audit_dynamic_reaction(case["accepted_interval_reaction"], context, partition,
                                     case["finite_seed_binding_sha256"])
    convergence = audit_reaction_convergence(case["convergence"], case["accepted_interval_reaction"], context)
    return dict(original_cohort_id=definition["original_cohort_id"], original_face_index=face,
        frozen_partition=partition_report, source_memory_birth=source_birth_report, birth_memory=birth_report,
        private_two_step_constitutive_path=reports, nonzero_memory_restart_checked=True,
        accepted_interval_reaction=reaction, convergence=convergence, accepted_planet_changed=False)


def audit_source_memory_birth(raw, context, birth, partition):
    """Check the source adapter without importing its typed producer."""
    seal(raw, "source_memory_birth_sha256")
    parent, runtime = context["parent"], context["parent"]["runtime_state"]
    geometry = runtime["geometry"]
    inherited = runtime["mechanics"]["state"]["elastic_energy_j"]
    wanted = dict(format="genesis-source-bound-compact-memory-birth-0.1",
        scope="unchanged_accepted_Genesis_to_private_fixed_material_Maxwell_memory",
        accepted_parent_checkpoint_sha256=parent["checkpoint_sha256"],
        formation_checkpoint_sha256=parent["source_checkpoint_sha256"],
        source_identity=geometry["source_identity"], accepted_time_myr=geometry["time_myr"],
        accepted_step_sizes_myr=runtime["accepted_step_sizes_myr"],
        reaction_binding_sha256=context["reaction_binding_sha256"],
        partition_binding_sha256=partition["embedded_partition_sha256"],
        birth_parent_memory_sha256=runtime["mechanics"]["packet_sha256"],
        shell_parameters=parent["source_packet"]["source"]["parameters"]["shell"],
        fixed_cohort_damage=[runtime["fault_history"]["damage"][c["face_index"]] for c in geometry["cohorts"]],
        initial_memory=birth, saved_parent_elastic_energy_j=inherited,
        independently_reduced_parent_elastic_energy_j=partition["initial_elastic_energy_j"],
        inherited_energy_reduction_difference_j=partition["initial_elastic_energy_j"]-inherited,
        energy_difference_scope="reported_arithmetic_reduction_difference_not_physical_work",
        physical_creation_admitted=False, physical_release_admitted=False, funding_admitted=False,
        accepted_planet_memory_changed=False, new_Genesis_time_myr=0., new_material_birth_count=0,
        creation_work_debited_from_Genesis_j=0.)
    require(set(raw) == set(wanted)|{"source_memory_birth_sha256"}, "source memory birth schema differs")
    for key, expected in wanted.items(): exact(raw[key], expected, "source memory birth "+key)
    return dict(accepted_parent_and_constitutive_parameters_exact=True,
        original_memory_birth_exact=True, inherited_scalar_reduction_roundoff_reported=True,
        roundoff_not_physical_work_or_funding=True)


def audit_reaction_convergence(raw, reaction, context=None):
    exact(raw["orders"], [8, 12], "reaction convergence orders")
    exact(raw["observed_relative_tolerance"], 2e-6, "reaction convergence tolerance")
    records = {}
    for stem in ("added_inertia", "added_average_force"):
        source = "added_inertia_force_n" if stem == "added_inertia" else "added_average_internal_force_n"
        before, after = array(raw[source+"_order8"], (18,)).reshape(-1, 6), array(reaction[source], (18,)).reshape(-1, 6)
        errors = [float(np.linalg.norm(before[:, i:i+3]-after[:, i:i+3])/
            max(float(np.linalg.norm(after[:, i:i+3])), np.finfo(float).tiny)) for i in (0, 3)]
        close(raw[stem+"_errors_A_P"], errors, "literal observed convergence "+stem, rtol=2e-12, atol=1e-20)
        require(max(errors) <= 2e-6, "accepted average or inertia has unresolved quadrature convergence")
        records[stem+"_relative_errors_A_P"] = errors
    if context is not None:
        definition = context["definition"]
        lower = seed.compact_bulk_terms(context["parent"], definition["normal_xyz"], definition["offset_m"],
            definition["seed_center_world_m"], definition["seed_radius_m"],
            cohort_id=definition["original_cohort_id"], order=8)
        lower_context = {**context, "local": lower,
            "interpolation": lower["barycentric"]@np.linalg.inv(seed.prior.SOURCE_ANGULAR_POINTS)}
        lower_expected = dynamic_reaction_expected(lower_context)
        for key in ("added_inertia_force_n", "added_average_internal_force_n"):
            seed.mode_errors(raw[key+"_order8"], lower_expected[key], "independent order eight "+key,
                rtol=3e-6 if "inertia" in key else 5e-7,
                atol=1e-35 if "inertia" in key else 1e-5)
    return dict(observed_order_eight_twelve_agreement=True, relative_tolerance=2e-6,
        separate_inertia_and_average_force_A_P_checked=True,
        order_eight_components_independently_rebuilt=context is not None, numerical_errors=records)


def audit_fixture(fixture, parent, control_directory=None, previous_seed_fixture=None):
    seal(fixture, "fixture_sha256")
    authenticate_parent(parent)
    expected = dict(parent_raw_sha256=PARENT_RAW_SHA256, parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
        formation_checkpoint_sha256=FORMATION_SHA256, parent_before_sha256=PARENT_RAW_SHA256,
        parent_after_sha256=PARENT_RAW_SHA256, new_Genesis_time_steps=0, new_material_cohorts=0,
        creation_work_debited_from_Genesis_j=0., physical_release_admitted=False,
        funding_admitted=False, point_one_complete=False)
    for key, wanted in expected.items(): exact(fixture.get(key), wanted, "full memory fixture "+key)
    for key in ("history_reinitialized", "physical_creation_admitted", "activation_accepted",
        "physical_contact_admitted", "actual_dynamic_reaction_replaced", "automated_goal_restarted",
        "accepted_planet_memory_changed", "physical_Gamma_selected", "automated_goal_enabled"):
        if key in fixture: require(fixture[key] is False, "fixture admits "+key)
    exact(fixture["format"], "genesis-source-bound-enriched-memory-and-full-tied-reaction-control-0.1", "whole control format")
    exact(fixture["scope"], "private_fixed_material_Maxwell_and_hypothetical_accepted_interval_tied_reaction", "whole control scope")
    previous_bytes = PREVIOUS_SEED_FIXTURE_PATH.read_bytes() if previous_seed_fixture is None else previous_seed_fixture
    require(hashlib.sha256(previous_bytes).hexdigest() == PREVIOUS_SEED_FIXTURE_RAW_SHA256,
        "preserved previous finite-seed evidence pin differs")
    exact(fixture["previous_seed_fixture_pin"], dict(bytes=len(previous_bytes),
        raw_sha256=PREVIOUS_SEED_FIXTURE_RAW_SHA256), "previous finite-seed source pin")
    previous = json.loads(previous_bytes)
    seal(previous, "fixture_sha256")
    cases = fixture.get("cases")
    require(type(cases) is list and len(cases) == 3 and [c["seed_definition"]["original_face_index"]
        for c in cases] == [0, 40, 79], "three original genuine finite-seed cases differ")
    for case, inherited_case in zip(cases, previous["finite_seed_cases"]):
        exact(case["seed_definition"], inherited_case["seed_definition"], "preserved genuine finite-seed definition")
        exact(case["name"], inherited_case["name"], "preserved genuine finite-seed case name")
        exact(case["finite_seed_binding_sha256"], inherited_case["finite_seed"]["finite_seed_sha256"],
              "preserved source-bound finite-seed integrity link")
    reports = [audit_case(case, parent) for case in cases]
    if control_directory is not None:
        directory = Path(control_directory)
        before, after = (directory/name for name in ("parent_before.json", "parent_after.json"))
        require(before.read_bytes() == after.read_bytes()
            and hashlib.sha256(before.read_bytes()).hexdigest() == PARENT_RAW_SHA256,
            "full accepted planetary checkpoint changed")
        require(canonical(json.loads((directory/"control_fixture.json").read_bytes())) == canonical(fixture),
                "saved control fixture differs")
    report = dict(format="independent-genesis-volume-enriched-memory-audit-0.1", status="PASS",
        scope=MEMORY_SCOPE, producer_imports=False, producer_functions_called=False,
        public_planetary_replay_performed=False, parent_raw_sha256=PARENT_RAW_SHA256,
        parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256, formation_checkpoint_sha256=FORMATION_SHA256,
        time_myr=parent["runtime_state"]["geometry"]["time_myr"], cohort_count=800,
        authentic_parent_full_fields_and_history_checked=True,
        entire_before_after_parent_byte_equality_checked=control_directory is not None,
        new_Genesis_time_steps=0, new_material_cohorts=0, creation_work_debited_from_Genesis_j=0.,
        physical_creation_admitted=False, physical_release_admitted=False, funding_admitted=False,
        actual_dynamic_reaction_replaced=False, point_one_complete=False,
        finite_seed_cases=reports, fixture_sha256=fixture["fixture_sha256"])
    report = plain(report)
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
    require(hashlib.sha256(parent_bytes).hexdigest() == PARENT_RAW_SHA256, "immutable genuine parent raw pin differs")
    fixture = json.loads(args.fixture.read_bytes())
    report = audit_fixture(fixture, json.loads(parent_bytes), args.control_directory)
    require(not args.output.exists(), "independent output destination already exists")
    output = canonical(report)+b"\n"
    args.output.write_bytes(output)
    print(json.dumps(dict(status="PASS", report={"bytes": len(output),
        "raw_sha256": hashlib.sha256(output).hexdigest()}, self_sha256=report["report_sha256"]), sort_keys=True))


if __name__ == "__main__":
    main()
