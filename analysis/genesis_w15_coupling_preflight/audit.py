"""Read-only provenance/shape audit; no mechanical or stability certificate."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    pins = {}

    def read(path):
        p = ROOT / path
        raw = p.read_bytes()
        pins[path] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        return json.loads(raw)

    case = read("analysis/genesis_w15_case_validation/case.json")
    fixture_path = case["input_pins"]["fixture"]["path"]
    source = read(fixture_path)["cases"][0]["source_inputs"]
    assert pins[fixture_path]["sha256"] == case["input_pins"]["fixture"]["sha256"]
    original_path = "analysis/genesis_volume_clipped_validation/evidence/independent/resealed_negatives/original_source_case_000.json"
    original = read(original_path)["source_inputs"]
    assert pins[original_path]["sha256"] == "30d35b0dd82bdbc4c9f4efb5cb84c0a6aab6629505650e167456b1b8c3f419a7"
    n = source["parent_ndof"]
    assert n == 252
    dofs = source["selected_parent_dofs"]
    assert len(dofs) == len(set(dofs)) == 18 and all(0 <= i < n for i in dofs)
    shapes = {}
    for name, block in original["literal_original_parent_blocks"].items():
        assert block["shape"] == [n, n]
        ptr = block["indptr"]
        assert len(ptr) == n + 1 and ptr[0] == 0
        assert all(a <= b for a, b in zip(ptr, ptr[1:]))
        assert ptr[-1] == len(block["indices"]) == len(block["data"])
        assert all(0 <= i < n for i in block["indices"])
        shapes[name] = block["shape"]
    old_radius = source["previous_seed_definition"]["seed_radius_m"]
    assert old_radius == 4.024215510221975 and case["seed_radius_m"] == 100.0
    assert old_radius != case["seed_radius_m"]
    assert source["source_selected_prospective_law"] is None
    assert source["actual_nucleation_candidate"] is None
    assert source["physical_Gamma_selected"] is False
    reaction = original["original_accepted_interval_reaction"]
    assert reaction["parent_ndof"] == n
    assert reaction["added_mode_basal_drag_and_external_force_exact_zero"] is True
    for key in ["parent_external_force_n", "parent_before_velocity_m_s", "parent_increment_m"]:
        assert len(reaction[key]) == n
    angular = read("/workspace/genesis-precision-full-01/case_000/angular.json")
    assert angular["relative_tolerance"] == 2e-6
    assert angular["angular_accuracy_certified"] is False
    print(json.dumps({
        "status": "PASS_PROVENANCE_AND_SHAPES_ONLY",
        "base_code_sha": "6de23727ae3a4a78ac77fe5dad1df23b5ea8e257",
        "pins": pins, "parent_shapes": shapes, "selected_parent_dofs": dofs,
        "old_local_block_radius_m": old_radius, "current_radius_m": case["seed_radius_m"],
        "accepted_interval_myr": [reaction["before_time_myr"], reaction["after_time_myr"]],
        "angular_component_count": angular["component_count"],
        "angular_failed_component_count": angular["failed_component_count"],
        "source_selected_law": None, "coupled_test_status": "BLOCKED",
        "mechanical_solves": 0, "new_integrals": 0,
        "coupled_balance_audited": False, "coupled_stability_certified": False,
        "physical_admission": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
