"""Fresh normal-support regressions against the sealed 814-source generation.

A successful gate means that the invalid placement is correctly rejected.
The saved curved-force source packet is authenticated and copied; no genuine
case is rebuilt, no inherited force certificate admits the new placement,
and no physical native interval is accepted by these checks.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from fractions import Fraction
import importlib
import importlib.util
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.geometric_native_curved_equilibrium_validation import validate as previous
from analysis.geometric_native_finite_equilibrium_validation.validate import _purge_added_source_modules
from analysis.geometric_native_spherical_solid_validation.validate import _verify_outputs

RELEASE = ROOT / "analysis/geometric_native_curved_equilibrium_validation/release_validation_01"
EXPECTED_REPORT = "42da56ee7a9be0d254aba11dee50d078db0b99014197af09d732cebd97b57fe2"
EXPECTED_REPORT_SEAL = "c10fb1d83a5b605bdcd41af4eecf1062cd8688eaa804b2c2c2762c861d5ac88c"
EXPECTED_NATIVE_REPORT = "192bdc9394ece287510cd56a29b7b59984f12ebb5ad67288f43736344cd59024"
EXPECTED_NATIVE_REPORT_SEAL = "63ca13c35b3d6c424ae23bace2e50cad26efeb38d3f413968a5e59ade2976d58"
EXPECTED_PACKET = "0e3554a55a8c93eb8ecbdae8351f3280a2ae4bd8596254950afef7f77b98d2cb"
EXPECTED_PACKET_SEAL = "f192cb4a9fbdcfd83ab4bad69e1c5c12307bfe4df78626aa514deb3d4f84cccd"
EXPECTED_SOURCE_MAP = "aebd927438930fe8069818e88a02388054aa12691ccd2828f722c6a4d59c7418"
PHYSICAL_REJECTION = "REJECTED_INVALID_NORMAL_SUPPORT"
SOURCE_FIELDS = (
    "before", "current_load", "original_unaccepted_trial_birth_support",
    "full_reference_face_birth_support", "saved_joint_reference_field",
)
ADDED_SOURCE_PATHS = frozenset((
    "tectonics/geometric_native_normal_support.py",
    "tectonics/geometric_native_bulk_elasticity.py",
    "tests/test_geometric_native_normal_support.py",
    "tests/test_geometric_native_bulk_elasticity.py",
    "analysis/geometric_native_normal_support_validation/validate.py",
    "analysis/geometric_native_normal_support_validation/native_control.py",
))

encoded = previous.encoded
digest = previous.digest
value_digest = previous.value_digest
read = previous.read
sealed = previous.sealed
source_map = previous.source_map
parse_counts = previous.parse_counts


def _literal_fraction(value):
    if (type(value) is not list or len(value) != 2
            or any(type(endpoint) is not int for endpoint in value) or value[1] <= 0):
        raise ValueError("A witness literal requires exact numerator/denominator")
    return Fraction(*value)


def _interval(value):
    if type(value) is not dict or set(value) != {"lo", "hi"}:
        raise ValueError("A witness bound requires two outward exact endpoints")
    lo, hi = _literal_fraction(value["lo"]), _literal_fraction(value["hi"])
    if lo > hi:
        raise ValueError("A witness interval is reversed")
    return lo, hi


def _check_witnesses(witnesses, packet):
    """Check strict signs, positive measure, source rods and every owner roof."""
    support = packet["full_reference_face_birth_support"]
    epoch = packet["before"]["snapshot"]["current_material_epoch"]
    reference = packet["saved_joint_reference_field"]
    carriers = {row["declaration"]["family_id"]: row for row in support["interface_carriers"]}
    components = {row["component_id"]: row for row in epoch["components"]}
    tops = {identity: _interval(bounds) for identity, bounds in reference["top_radial_displacements_m"]}
    radius = Fraction.from_float(epoch["source_material_state"]["journal"]["source_surface"]["radius_km"])*1000
    if (type(witnesses) is not list or len(witnesses) != 2
            or {row.get("family_id") for row in witnesses} != set(carriers)):
        raise ValueError("Both unique genuine curved-face rejection witnesses are required")
    for row in witnesses:
        carrier = carriers[row["family_id"]]
        declaration = carrier["declaration"]
        receiver = components[declaration["receiver_component_id"]]
        donor = components[declaration["donor_component_id"]]
        angle = _interval(row["angle_bounds_radians"])
        native_angle = _interval(row["native_angle_end_bounds_radians"])
        depth = _interval(row["interior_depth_m"])
        end_radius = _interval(row["ray_endpoint_radius_bounds_m"])
        max_roof = _interval(row["owner_max_roof_radius_bounds_m"])
        margin = _interval(row["outside_roof_margin_bounds_m"])
        ray_radius = _interval(row["reference_radius_bounds_m"])
        radial_normal = _interval(row["receiving_radial_normal_bounds"])
        columns = tuple(_literal_fraction(value) for value in row["column_lengths_m"])
        source_columns = declaration["parameters"]["columns"]
        expected_columns = tuple(Fraction.from_float(source_columns[name])
            for name in ("sub_length_m", "over_length_m"))
        expected_roofs = {identity: (radius+lo, radius+hi)
            for identity, (lo, hi) in tops.items()
            if components[identity]["owner"] == receiver["owner"]}
        roofs = {identity: _interval(bounds) for identity, bounds in row["all_same_owner_roofs"]}
        expected_max = (max(lo for lo, hi in expected_roofs.values()),
                        max(hi for lo, hi in expected_roofs.values()))
        donor_top = radius+tops[donor["component_id"]][0]
        receiver_top = radius+tops[receiver["component_id"]][0]
        common_top = min(donor_top, receiver_top)
        # All three arguments are positive on this source witness. Thus the
        # squared ray radius is monotone in r, n_r and the full rod distance.
        radial_square_lo = (ray_radius[0]**2
            +2*ray_radius[0]*radial_normal[0]*columns[1]+columns[1]**2)
        radial_square_hi = (ray_radius[1]**2
            +2*ray_radius[1]*radial_normal[1]*columns[1]+columns[1]**2)
        if (row["source_packet_raw_sha256"] != EXPECTED_PACKET
                or row["source_support_sha256"] != support["support_sha256"]
                or row["source_epoch_sha256"] != epoch["epoch_sha256"]
                or row["saved_field_declaration_precision"] != 128
                or _literal_fraction(row["activation_time_fraction"]) != 0
                or row["receiver_component_id"] != receiver["component_id"]
                or row["donor_component_id"] != donor["component_id"]
                or row["receiver_owner"] != receiver["owner"]
                or row["donor_owner"] != donor["owner"]
                or native_angle != _interval(carrier["native_angle_bounds_radians"])
                or not 0 < native_angle[0] <= native_angle[1]
                or angle != (Fraction(0), native_angle[1])
                or depth != (Fraction(100), Fraction(500))
                or columns != expected_columns or columns != (Fraction(1000),)*2
                or not 0 < ray_radius[0] <= ray_radius[1]
                or not 0 < radial_normal[0] <= radial_normal[1] <= 1
                or _literal_fraction(row["common_reference_top_radius_m"]) != common_top
                or ray_radius[0]**3 > common_top**3-3*radius**2*depth[1]
                or ray_radius[1]**3 < common_top**3-3*radius**2*depth[0]
                or end_radius[0] <= 0 or end_radius[0]**2 > radial_square_lo
                or end_radius[1]**2 < radial_square_hi
                or _interval(row["ray_distance_bounds_m"]) != (columns[1],)*2
                or _literal_fraction(row["source_preload_gap_m"]) != Fraction.from_float(declaration["birth_gap_free_m"])
                or row["positive_measure_activation_strip_proved"] is not True
                or row["full_source_column_support_rejected"] is not True
                or len(row["all_same_owner_roofs"]) != len(expected_roofs)
                or roofs != expected_roofs or max_roof != expected_max
                or end_radius[0]-max_roof[1] <= 356 or margin[0] <= 356):
            raise ValueError("A positive-area full-column source roof exclusion differs")


def _check_bulk_declarations(native, packet):
    """Check the explicitly new tensor and source-driven local examples."""
    support = packet["full_reference_face_birth_support"]
    closures = native.get("bulk_closures")
    checks = native.get("local_bulk_energy_checks")
    if (type(closures) is not list or type(checks) is not list
            or encoded(closures) != encoded(packet.get("bulk_closures"))
            or native.get("bulk_tensor_newly_declared") is not True
            or native.get("actual_body_stiffness_assembled") is not False
            or native.get("actual_whole_mask_field_solved") is not False):
        raise ValueError("New local bulk data were promoted to a whole-body solve")
    expected_closures, expected_checks = {}, {}
    for carrier in support["interface_carriers"]:
        declaration = carrier["declaration"]
        parameters = declaration["parameters"]
        columns = parameters["columns"]
        scalar = lambda key: Fraction.from_float(columns[key])
        compliance = (scalar("sub_length_m")/scalar("sub_normal_modulus_pa")
            +scalar("over_length_m")/scalar("over_normal_modulus_pa")
            +1/Fraction.from_float(parameters["law"]["normal_stiffness_pa_m"]))
        pressure = -min(Fraction.from_float(declaration["birth_gap_free_m"]), Fraction(0))/compliance
        for side, bank in (("donor", "sub"), ("receiver", "over")):
            identity = carrier[side]["component_id"]
            young, shear = scalar(bank+"_normal_modulus_pa"), scalar(bank+"_shear_modulus_pa")
            nu = young/(2*shear)-1
            lam = 2*shear*nu/(1-2*nu)
            values = dict(young_modulus_pa=young, shear_modulus_pa=shear,
                          poisson_ratio=nu, lame_lambda_pa=lam)
            if identity in expected_closures and expected_closures[identity] != (columns, values):
                raise ValueError("One true shared bank received conflicting tensors")
            expected_closures[identity] = (columns, values)
            strain = pressure/young
            density = young*strain**2/2
            expected_checks[declaration["family_id"], side] = (identity, dict(
                preload_pressure_pa=pressure, source_derived_strain=strain,
                old_one_dimensional_density_j_m3=density,
                laterally_free_bulk_density_j_m3=density,
                isochoric_robin_linear_density_j_m3=2*shear*strain**2,
                isochoric_to_one_dimensional_energy_ratio=4*shear/young))
    if (len(closures) != len(expected_closures) or len(checks) != len(expected_checks)
            or {row["component_id"] for row in closures} != set(expected_closures)
            or {(row["family_id"], row["bank"]) for row in checks} != set(expected_checks)):
        raise ValueError("All three true-bank closures and four local source checks are required")
    for row in closures:
        columns, values = expected_closures[row["component_id"]]
        if (encoded(row["source_columns"]) != encoded(columns)
                or row["source_support_sha256"] != support["support_sha256"]
                or row["lateral_closure"] != "laterally_free_standard_isotropic"
                or type(row["lateral_closure_provenance"]) is not str
                or not row["lateral_closure_provenance"].strip()
                or row["new_lateral_closure_declared"] is not True
                or any(row[key] is not False for key in (
                    "previous_three_dimensional_tensor_inherited",
                    "source_column_lengths_are_actual_bank_geometry",
                    "actual_whole_mask_field_solved",
                    "complete_three_dimensional_strain_energy_certified"))):
            raise ValueError("A new closure lost its source or explicit local scope")
        for key, value in values.items():
            lo, hi = _interval(row[key])
            if not lo <= value <= hi:
                raise ValueError("A new bulk coefficient differs from its source moduli")
    for row in checks:
        identity, values = expected_checks[row["family_id"], row["bank"]]
        if (row["component_id"] != identity or row["local_constitutive_check_only"] is not True
                or row["actual_whole_mask_field_solved"] is not False):
            raise ValueError("A source-driven local energy check was promoted to a body field")
        for key, value in values.items():
            lo, hi = _interval(row[key])
            if not lo <= value <= hi:
                raise ValueError("A local bulk energy differs from its source-driven strain")


def verify_prior():
    """Authenticate old 814 bodies and every historical sealed output."""
    inherited = previous.verify_prior()
    path = RELEASE / "report.json"
    if digest(path) != EXPECTED_REPORT:
        raise ValueError("Authoritative curved-force report changed")
    report = sealed(path, "report_sha256")
    if (read(path).get("report_sha256") != EXPECTED_REPORT_SEAL
            or report.get("status") != "PASS"
            or report.get("source_count") != 814
            or report.get("pytest_counts") != {"passed": 69}
            or report.get("changed_python_paths") != []
            or report.get("prior_artifacts_and_python_sources_preserved_after") is not True
            or report.get("preloaded_case_source_generation_preserved_after") is not True
            or report.get("source_map_before_sha256") != EXPECTED_SOURCE_MAP
            or report.get("source_map_after_sha256") != EXPECTED_SOURCE_MAP
            or len(report.get("added_python_sources", {})) != 8
            or set(inherited) & set(report["added_python_sources"])):
        raise ValueError("Curved-force generation or regression scope differs")
    inherited.update(report["added_python_sources"])
    if len(inherited) != 814 or value_digest(inherited) != EXPECTED_SOURCE_MAP:
        raise ValueError("The original 814-source generation differs")
    for name, expected in inherited.items():
        if digest(ROOT / name) != expected:
            raise ValueError("An original Python source changed: " + name)
    _verify_outputs(RELEASE, report, 4)
    native_path = RELEASE / "native_control/report.json"
    packet_path = RELEASE / "native_control/candidate_packet.json"
    if digest(native_path) != EXPECTED_NATIVE_REPORT or digest(packet_path) != EXPECTED_PACKET:
        raise ValueError("Authoritative curved-force native evidence changed")
    native = sealed(native_path, "report_sha256")
    packet = sealed(packet_path, "packet_sha256")
    if (read(native_path).get("report_sha256") != EXPECTED_NATIVE_REPORT_SEAL
            or read(packet_path).get("packet_sha256") != EXPECTED_PACKET_SEAL
            or native.get("status") != "PASS"
            or native.get("accepted_original_native_intervals") != 1
            or native.get("accepted_new_native_intervals") != 0
            or native.get("native_curved_force_equilibrium_certified") is not True
            or native.get("original_point_1_completed") is not False
            or native.get("accepted_input_complete_typed_state_unchanged") is not True
            or native.get("whole_clock_material_admission_authorized") is not False
            or native.get("complete_three_dimensional_strain_energy_certified") is not False
            or native.get("global_physical_solid_noninterpenetration_certified") is not False
            or native.get("all_42_surface_pairs_reexecuted") is not False
            or native.get("packet_sha256") != EXPECTED_PACKET_SEAL
            or packet.get("accepted_input_unchanged") is not True
            or packet.get("material_admission_authorized") is not False):
        raise ValueError("Preserved curved-force scope or complete source binding differs")
    return inherited


def _check_native_control(out):
    path = out / "native_control/report.json"
    packet_path = out / "native_control/source_packet.json"
    native = sealed(path, "report_sha256")
    packet = sealed(packet_path, "packet_sha256")
    original = sealed(RELEASE / "native_control/candidate_packet.json", "packet_sha256")
    false_scope = (
        "original_point_1_completed", "whole_clock_material_admission_authorized",
        "complete_three_dimensional_strain_energy_certified",
        "global_physical_solid_noninterpenetration_certified",
        "all_42_surface_pairs_reexecuted", "new_native_curved_force_equilibrium_certified",
        "full_genuine_source_reconstructed", "inherited_force_certificate_for_placement",
        "material_admission_authorized",
        "normal_columns_clipped",
        "alternative_finite_body_deformation_models_ruled_out",
    )
    if (native.get("status") != PHYSICAL_REJECTION
            or native.get("physical_candidate_status") != PHYSICAL_REJECTION
            or native.get("accepted_original_native_intervals") != 1
            or native.get("accepted_new_native_intervals") != 0
            or native.get("original_point_1_remaining_work_blocks") != 1
            or type(native.get("remaining_physical_conditions")) is not list
            or len(native["remaining_physical_conditions"]) != 3
            or native.get("accepted_input_complete_typed_state_unchanged") is not True
            or any(native.get(name) is not False for name in false_scope)
            or native.get("source_packet_sha256") != read(packet_path)["packet_sha256"]
            or packet.get("authenticated_curved_packet_raw_sha256") != EXPECTED_PACKET
            or packet.get("authenticated_curved_packet_seal") != EXPECTED_PACKET_SEAL
            or packet.get("material_admission_authorized") is not False
            or packet.get("accepted_input_unchanged") is not True
            or packet.get("inherited_force_certificate_for_placement") is not False
            or packet.get("full_genuine_source_reconstructed") is not False):
        raise ValueError("Native rejection changed this gate's physical scope")
    for name in SOURCE_FIELDS:
        if encoded(packet.get(name)) != encoded(original[name]):
            raise ValueError("A complete authenticated source payload differs: " + name)
    for name in ("coordinates_decimal", "literal_owner_axes"):
        if encoded(packet.get(name)) != encoded(original["actual_curved_force_trial"][name]):
            raise ValueError("The canonical saved curved-force pose differs: " + name)
    reference = packet["saved_joint_reference_field"]
    if reference["declaration_precision"] != 128 or reference["component_count"] != 13:
        raise ValueError("The literal saved source reference field differs")
    witnesses = native.get("normal_support_witnesses")
    _check_witnesses(witnesses, packet)
    _check_bulk_declarations(native, packet)
    return dict(
        native_control_report_raw_sha256=digest(path),
        native_control_source_packet_raw_sha256=digest(packet_path),
        physical_candidate_status=PHYSICAL_REJECTION,
        accepted_original_native_intervals=1, accepted_new_native_intervals=0,
        accepted_input_complete_typed_state_unchanged=True,
        full_genuine_source_reconstructed=False,
        inherited_force_certificate_for_placement=False,
        new_native_curved_force_equilibrium_certified=False,
        material_admission_authorized=False,
        complete_three_dimensional_strain_energy_certified=False,
        global_physical_solid_noninterpenetration_certified=False,
        whole_clock_material_admission_authorized=False,
        all_42_surface_pairs_reexecuted=False,
        original_point_1_completed=False,
        original_point_1_remaining_work_blocks=1,
        complete_authenticated_source_packet_preserved=True,
        positive_area_invalid_normal_support_witness_count=len(witnesses),
    )


def run(out, test_paths, control):
    paths = tuple(test_paths)
    if (len(paths) != 2 or len(set(paths)) != 2
            or any(not isinstance(name, str) or not name.startswith("tests/test_")
                or not name.endswith(".py") or not (ROOT / name).is_file() for name in paths)
            or not isinstance(control, str) or ":" not in control):
        raise ValueError("Two existing distinct new tests and module:function are required")
    inherited = verify_prior()
    if any(name in inherited for name in paths):
        raise ValueError("This gate accepts only newly added regression tests")
    before = source_map()
    added = set(before) - set(inherited)
    if len(before) != 820 or added != ADDED_SOURCE_PATHS:
        raise ValueError("Exactly six additive Python sources are required")
    removed_caches = []
    for name in sorted(added):
        cache = Path(importlib.util.cache_from_source(str(ROOT / name)))
        if cache.is_file():
            cache.unlink()
            removed_caches.append(str(cache.relative_to(ROOT)))
    own_path = str(Path(__file__).resolve().relative_to(ROOT))
    purged_modules = _purge_added_source_modules(added - {own_path})
    importlib.invalidate_caches()
    if source_map() != before:
        raise ValueError("A Python source changed while preparing final imports")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = dict(
        format="native-normal-support-refusal-same-process-validation-1",
        status="FAIL", regression_validation_status="FAIL",
        physical_candidate_status="NOT_CHECKED",
        started_utc=datetime.now(timezone.utc).isoformat(),
        verified_prior_artifacts=1633, verified_solid_placement_outputs=5,
        verified_radial_normalization_diagnosis=True, verified_prior_python_sources=814,
        verified_finite_equilibrium_outputs=4, verified_failed_release_outputs=3,
        verified_spherical_solid_outputs=4, verified_joint_spherical_outputs=4,
        verified_curved_equilibrium_outputs=4,
        prior_curved_report_raw_sha256=EXPECTED_REPORT,
        prior_curved_native_report_raw_sha256=EXPECTED_NATIVE_REPORT,
        prior_curved_packet_raw_sha256=EXPECTED_PACKET,
        prior_source_map_sha256=EXPECTED_SOURCE_MAP,
        source_count=len(before), source_map_before_sha256=value_digest(before),
        added_python_sources={name: before[name] for name in sorted(added)},
        removed_added_source_bytecode_caches=removed_caches,
        purged_previously_imported_added_source_modules=purged_modules,
        historical_tests_reexecuted=False,
        test_paths=list(paths), control=control,
        execution_mode="same_python_process", pytest_import_mode="importlib",
        preloaded_genuine_case_used=False,
        full_genuine_source_reconstructed=False,
        inherited_force_certificate_for_placement=False,
        whole_native_interval_admitted_by_test_gate=False,
    )
    try:
        import pytest
        command = ["-q", "-p", "no:cacheprovider", "--import-mode=importlib", *paths]
        pytest_started = time.perf_counter()
        original_directory = Path.cwd()
        try:
            os.chdir(ROOT)
            with (out / "pytest.log").open("x", buffering=1) as stream:
                with redirect_stdout(stream), redirect_stderr(stream):
                    exit_code = int(pytest.main(command))
        finally:
            os.chdir(original_directory)
        summary, counts = parse_counts(out / "pytest.log")
        report.update(pytest_exit_code=exit_code, pytest_summary=summary,
            pytest_counts=counts, pytest_seconds=time.perf_counter()-pytest_started)
        if (exit_code != 0 or counts.get("passed", 0) < 1
                or any(counts.get(name, 0) for name in
                    ("failed", "skipped", "xfailed", "xpassed", "error", "errors"))):
            raise RuntimeError("Fresh normal-support regressions failed")
        if source_map() != before:
            raise ValueError("A Python source changed during pytest")
        module, function = control.split(":", 1)
        callback = getattr(importlib.import_module(module), function)
        control_started = time.perf_counter()
        with (out / "control_console.log").open("x", buffering=1) as stream:
            with redirect_stdout(stream), redirect_stderr(stream):
                callback(out / "native_control")
        report["native_control_seconds"] = time.perf_counter()-control_started
        report.update(_check_native_control(out))
        expected_outputs = {
            "pytest.log", "control_console.log",
            "native_control/report.json", "native_control/source_packet.json",
        }
        if {str(path.relative_to(out)) for path in out.rglob("*") if path.is_file()} != expected_outputs:
            raise ValueError("The refusal gate requires exactly its four declared output files")
        report["status"] = report["regression_validation_status"] = "PASS"
    except BaseException as error:
        report["failure_type"] = type(error).__name__
        report["failure_message"] = str(error)
        report["failure_traceback"] = traceback.format_exc()
    finally:
        after = source_map()
        report["source_map_after_sha256"] = value_digest(after)
        report["changed_python_paths"] = sorted(name for name in set(before) | set(after)
            if before.get(name) != after.get(name))
        try:
            verify_prior()
            report["prior_artifacts_and_python_sources_preserved_after"] = True
        except BaseException as error:
            report["prior_artifacts_and_python_sources_preserved_after"] = False
            report["prior_after_error"] = str(error)
        if (report["changed_python_paths"]
                or not report["prior_artifacts_and_python_sources_preserved_after"]):
            report["status"] = report["regression_validation_status"] = "FAIL"
        report["seconds"] = time.perf_counter()-started
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report["outputs"] = {str(path.relative_to(out)): dict(
            sha256=digest(path), bytes=path.stat().st_size)
            for path in sorted(out.rglob("*")) if path.is_file()}
        report["report_sha256"] = value_digest(report)
        with (out / "report.json").open("xb") as stream:
            stream.write(encoded(report)+b"\n")
        print(encoded({key: report.get(key) for key in
            ("status", "regression_validation_status", "physical_candidate_status",
             "pytest_counts", "seconds", "source_count", "changed_python_paths")}).decode(), flush=True)
    if report["status"] != "PASS":
        raise RuntimeError("Validation refused; see " + str(out / "report.json"))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("test_paths", nargs="+")
    options = parser.parse_args()
    run(options.out, options.test_paths, options.control)
