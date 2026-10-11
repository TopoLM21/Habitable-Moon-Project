"""Bounded producer-free geometric audit of actual-source clipped bulk records.

This audits source lineage and every saved node before independently attempting
the fixed 100 m sphere/original-cohort moments by radial-line quadrature.  A
bounded numerical refusal is retained as OPEN, while successful source and node
checks remain separately visible.  No planetary checkpoint loader, mechanics
producer, or new initialization is imported or called.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.genesis_volume_clipped_validation.independent_geometry_audit import (
    IndependentGeometryRefusal, audit_archived_geometry, integrate_radial_lines,
)

PARENT = ROOT / "analysis/genesis_volume_creation_validation/evidence/actual_continuation/direct_after_second.json"
PREVIOUS = ROOT / "analysis/genesis_volume_memory_validation/evidence/actual_control/control_fixture.json"
PARENT_RAW = "62579a0bad128830a66e2f44579cf2b1f5d3835b691225b8d431102f3e4486d7"
PARENT_SELF = "5d282ceeccc8d76acc38427d08a7e65575fddb509fc4b93115cbe6d318e91349"
PREVIOUS_RAW = "a98ff7cc89aeb59eb004c05b8ac87f75f2e39a4f10d4454fdf93c288153c9c0e"
PREVIOUS_SELF = "2ba89795666f61987e513fa78971b9605db0d08bd6d83406eb9ebbe6295587c3"
FORMATION = "cd79ee2a5139c4b91abdb1b4714a761c9153cc9f4490f5aa63fa2facba458c89"
FIXED_RADIUS_M = 100.
FIXED_RELATIVE_TOLERANCE = 2e-6
FIXED_MAX_ANGULAR_CELLS = 2000
FIXED_MAX_DEPTH = 24
CORE_SOURCE_KEYS = ("triangle_m", "displacement_A_m", "slope_P_m", "lambda_bounds",
                    "normal_xyz", "offset_m", "cohort_id", "face_index", "owner_id",
                    "material_id", "mass_kg", "volume_m3", "q_bounds", "conical_mass_coefficient")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def pin(path):
    sha, size = hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        while data := stream.read(1 << 20):
            sha.update(data); size += len(data)
    return {"bytes": size, "raw_sha256": sha.hexdigest()}


def require(condition, message):
    if not condition:
        raise IndependentGeometryRefusal("Actual source geometric audit: "+message)


def seal_check(raw, key, expected=None):
    require(type(raw) is dict and raw.get(key) == digest({k: v for k, v in raw.items() if k != key}),
            "object seal differs: "+key)
    require(expected is None or raw[key] == expected, "pinned object identity differs: "+key)


def exact(a, b, message):
    require(canonical(a) == canonical(b), message)


def _authenticate_source(fixture):
    require(pin(PARENT)["raw_sha256"] == PARENT_RAW and pin(PREVIOUS)["raw_sha256"] == PREVIOUS_RAW,
            "accepted parent or previously reconstructed birth archive raw pin differs")
    parent = json.loads(PARENT.read_bytes())
    seal_check(parent, "checkpoint_sha256", PARENT_SELF)
    previous = json.loads(PREVIOUS.read_bytes())
    seal_check(previous, "fixture_sha256", PREVIOUS_SELF)
    original = {case["name"]: {"section": case["embedded_partition"]["local"]["section"],
                  "seed_definition": case["seed_definition"]} for case in previous["cases"]}
    del previous
    require(fixture["parent_raw_sha256"] == PARENT_RAW
            and fixture["parent_checkpoint_sha256"] == PARENT_SELF
            and fixture["formation_checkpoint_sha256"] == FORMATION
            and fixture["previous_memory_fixture_sha256"] == PREVIOUS_SELF,
            "source lineage differs")
    runtime = parent["runtime_state"]
    geometry = runtime["geometry"]
    require(parent["source_checkpoint_sha256"] == FORMATION
            and geometry["time_myr"] == .8684000000000005
            and len(geometry["cohorts"]) == 800
            and len(runtime["accepted_step_sizes_myr"]) == 4
            and not any(runtime["fault_history"]["fault_active"]),
            "accepted material birth, time, or activation differs")
    require(len(fixture["cases"]) == 3
            and [c["source_inputs"]["section"]["cohort_id"] for c in fixture["cases"]] == [1, 41, 80],
            "three original material cases differ")
    for flag in ("physical_creation_admitted", "physical_release_admitted", "physical_contraction_certified",
                 "physical_force_work_admission", "funding_admitted", "available_for_creation_funding",
                 "history_reinitialized", "accepted_planet_memory_changed", "actual_dynamic_reaction_replaced",
                 "physical_Gamma_selected", "automated_goal_enabled", "point_one_complete"):
        require(fixture[flag] is False, "a source refusal was upgraded: "+flag)
    require(fixture["public_planetary_restore_count"] == fixture["new_Genesis_time_steps"]
            == fixture["new_material_cohorts"] == 0
            and fixture["creation_work_debited_from_Genesis_j"] == 0,
            "source restore/time/material/work accounting differs")
    return parent, original


def _case_source(case, parent, original):
    source = case["source_inputs"]
    section = source["section"]
    require(case["name"] in original, "unknown prior source case")
    prior = original[case["name"]]
    exact(section, prior["section"], "section differs from pinned original birth archive")
    runtime = parent["runtime_state"]
    geometry = runtime["geometry"]
    cohort = geometry["cohorts"][section["cohort_index"]]
    face = np.asarray(geometry["faces"][cohort["face_index"]], int)
    vertices = np.asarray(geometry["vertices_m"])[face].tolist()
    displacement = np.asarray(runtime["displacement_m"]).reshape(-1, 6)[face]
    exact(section["triangle_m"], vertices, "source section replaced actual original triangle")
    exact(section["displacement_A_m"], displacement[:, :3].tolist(), "source section replaced accepted A")
    exact(section["slope_P_m"], displacement[:, 3:].tolist(), "source section replaced accepted P")
    for key in ("cohort_id", "face_index", "owner_id", "material_id", "mass_kg", "volume_m3"):
        target = "id" if key == "cohort_id" else key
        exact(section[key], cohort[target], "source original identity differs: "+key)
    exact(section["lambda_bounds"], [cohort["inner_scale"], cohort["outer_scale"]], "original radial chart differs")
    exact(section["q_bounds"], [cohort["q_lo"], cohort["live_q_hi"]], "original mass-coordinate chart differs")
    require(source["source_activation_accepted"] is False
            and source["source_selected_prospective_law"] is None
            and source["physical_Gamma_selected"] is False
            and source["material_birth_history_replaced"] is False,
            "source absence of selected nucleation was replaced")
    return section, prior["seed_definition"]


def _domain_source(domain, section, seed_definition, expected_radius):
    seal_check(domain, "binding_sha256")
    exact(domain["geometry_source"], {k: section[k] for k in CORE_SOURCE_KEYS},
          "clipped domain changed the exact original section operands")
    exact(domain["source_section_binding_sha256"], section["binding_sha256"],
          "clipped domain changed its original material section binding")
    exact(domain["seed_center_world_m"], seed_definition["seed_center_world_m"],
          "clipped domain moved the inherited center")
    require(domain["seed_radius_m"] == expected_radius, "clipped domain changed its declared fixed radius")
    for flag in ("numerical_integral_convergence_certified", "physical_creation_admitted",
                 "physical_release_admitted", "physical_force_work_admission", "available_for_creation_funding"):
        require(domain[flag] is False, "domain geometry became an unearned physical certificate: "+flag)


def _audit_case(case, parent, original):
    section, definition = _case_source(case, parent, original)
    node_records = []
    supports = [("fixed_100m_clipped_support", case["fixed_100m_clipped_support"], FIXED_RADIUS_M)]
    baseline = case["original_radius_baseline"]
    if baseline is not None:
        supports.append(("original_radius_baseline", baseline, definition["seed_radius_m"]))
    for label, support, radius in supports:
        require(support["seed_radius_m"] == radius and support["radius_unchanged_for_all_orders"] is True,
                "support changed its declared radius")
        expected_orders = [4, 8, 12, 16] if label == "fixed_100m_clipped_support" else [12, 16]
        exact([r["order"] for r in support["orders"]], expected_orders, "saved source order sequence differs")
        for record in support["orders"]:
            domain = record["domain"]
            _domain_source(domain, section, definition, radius)
            node = audit_archived_geometry(domain)
            node_records.append(dict(support=label, order=record["order"],
                point_count=len(domain["barycentric"]), domain_sha256=domain["binding_sha256"], **node))
    primary = case["fixed_100m_clipped_support"]["orders"][-1]["domain"]
    started = time.perf_counter()
    try:
        independent = integrate_radial_lines(primary, relative_tolerance=FIXED_RELATIVE_TOLERANCE,
            max_cells=FIXED_MAX_ANGULAR_CELLS, max_depth=FIXED_MAX_DEPTH)
        agreement = audit_archived_geometry(primary, independent)
        numerical = dict(status="PASS", independent_radial_line_report=independent,
                         highest_order_moment_agreement=agreement,
                         general_contraction_accuracy_certified=False)
    except IndependentGeometryRefusal as error:
        numerical = dict(status="OPEN", refusal_class=type(error).__name__, refusal_reason=str(error),
                         independent_radial_line_report=None, highest_order_moment_agreement=None,
                         general_contraction_accuracy_certified=False)
    numerical["wall_seconds"] = time.perf_counter()-started
    numerical["fixed_radius_m"] = FIXED_RADIUS_M
    numerical["fixed_relative_tolerance"] = FIXED_RELATIVE_TOLERANCE
    numerical["fixed_max_angular_cells"] = FIXED_MAX_ANGULAR_CELLS
    numerical["fixed_max_depth"] = FIXED_MAX_DEPTH
    # No repeated source integration, tolerance chase, or known-exhaustive
    # contraction attempt is performed after a bounded geometric refusal.
    contraction = dict(status="OPEN", independent_work_contraction_integration_attempted=False,
        reason="Fixed-budget independent geometric integral is prerequisite; general cross/self integral accuracy requires separate convergence evidence",
        general_contraction_accuracy_certified=False)
    return dict(name=case["name"], cohort_id=section["cohort_id"], source_section_exact=True,
        source_and_all_saved_nodes_status="PASS", saved_domains=node_records,
        independent_geometric_numerical_convergence=numerical,
        independent_work_contraction_integral=contraction)


def run(fixture_path, output):
    require(not output.exists() and output.resolve().is_relative_to(Path("/workspace"))
            and not output.resolve().is_relative_to(ROOT), "output must be fresh and external inside /workspace")
    started = time.perf_counter()
    fixture_pin = pin(fixture_path)
    fixture = json.loads(fixture_path.read_bytes())
    seal_check(fixture, "fixture_sha256")
    parent, original = _authenticate_source(fixture)
    cases = []
    for case in fixture["cases"]:
        result = _audit_case(case, parent, original)
        cases.append(result)
        print(json.dumps(dict(name=result["name"], source_and_all_saved_nodes_status="PASS",
            independent_geometric_numerical_convergence_status=result["independent_geometric_numerical_convergence"]["status"])), flush=True)
    numerical_pass = all(c["independent_geometric_numerical_convergence"]["status"] == "PASS" for c in cases)
    require(pin(fixture_path) == fixture_pin and pin(PARENT)["raw_sha256"] == PARENT_RAW
            and pin(PREVIOUS)["raw_sha256"] == PREVIOUS_RAW, "audited source bytes changed")
    report = dict(format="independent-actual-original-cohort-clipped-geometry-audit-0.1",
        status="COMPLETED" if numerical_pass else "COMPLETED_WITH_NUMERICAL_REFUSALS",
        source_and_all_saved_nodes_status="PASS",
        independent_geometric_numerical_convergence_status="PASS" if numerical_pass else "OPEN",
        independent_work_contraction_integral_status="OPEN", fixture_pin=fixture_pin,
        fixture_sha256=fixture["fixture_sha256"], parent_raw_sha256=PARENT_RAW,
        parent_checkpoint_sha256=PARENT_SELF, formation_checkpoint_sha256=FORMATION,
        previous_memory_fixture_raw_sha256=PREVIOUS_RAW,
        public_planetary_restore_count=0, source_time_myr=.8684000000000005,
        cohort_count=800, accepted_interval_count=4, new_Genesis_time_steps=0, new_material_cohorts=0,
        history_reinitialized=False, accepted_planet_memory_changed=False,
        physical_creation_admitted=False, physical_release_admitted=False,
        physical_contraction_certified=False, physical_force_work_admission=False,
        funding_admitted=False, conditioning_or_roundoff_available_for_funding=False,
        automated_goal_enabled=False, point_one_complete=False,
        creation_work_debited_from_Genesis_j=0., cases=cases,
        wall_seconds=time.perf_counter()-started)
    report["report_sha256"] = digest(report)
    output.write_bytes(canonical(report)+b"\n")
    print(json.dumps(dict(status=report["status"], source_and_all_saved_nodes_status="PASS",
        independent_geometric_numerical_convergence_status=report["independent_geometric_numerical_convergence_status"],
        report_sha256=report["report_sha256"], output_pin=pin(output))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.fixture, args.output)
