"""One frozen complete HOT and finite base-mean endpoint gate; prior sealed bytes preserved.

Only the new HOT/base-mean tests and native control run. Prior force tests, tensor
quadrature, solvers and saved reader/auditor mains are not replayed.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[2]
PARENT = Path(__file__).resolve().parent
STAGE = Path('/workspace/hot_mean_baseline_20261007')
PRESERVATION_SOURCE = STAGE / 'baseline_899_preservation_source.txt'
PRESERVATION_RAW = '7410ed4264a972cd4a08e3af4fae4c2f0b7113225a81c8fd9d525e25f0ccbc62'
OLD_FROZEN = ROOT / 'analysis/geometric_native_common_cold_continuum_energy_validation/release_validation_01_frozen_source_map_899.json'
ADDED = frozenset((
    'tectonics/geometric_native_common_hot_mean.py',
    'tests/test_geometric_native_common_hot_mean.py',
    'analysis/geometric_native_common_hot_mean_validation/native_control.py',
    'analysis/geometric_native_common_hot_mean_validation/validate.py',
))
TEST_PATH = 'tests/test_geometric_native_common_hot_mean.py'
PHYSICAL_SCOPE = 'DECLARED_COMPLETE_HOT_FINITE_BASE_MEAN_ENDPOINT_ONLY'
EXPECTED_PAYLOADS = frozenset(('pytest.log', 'control_console.log',
                             'native_control/report.json', 'native_control/source_packet.json'))
REMAINING_CONDITIONS = [
    'Whole-source physical material and old-force/prestress/work handover, HOT law/mean chart, loads/drag and NEW mixed equilibrium',
    'Whole-clock all-pair admission including reverse C5 to C1',
    'Accepted continuation with independent path work, balances and joint restart',
]
EXPECTED_NATIVE = dict(
    physical_candidate_status=PHYSICAL_SCOPE,
    complete_cold_continuum_operator_certified=True,
    complete_hot_material_constitutive_operator_certified=True,
    finite_hot_base_mean_chart_certified=True,
    whole_cold_material_support_geometry_certified=True,
    physical_source_force_prestress_work_handover_certified=False,
    historical_archived_three_dimensional_prestress_recovered=False,
    prepared_reference_full_system_work_certified=False,
    hot_constitutive_law_defined=True, finite_mean_chart_defined=True,
    finite_mean_equilibrium_certified=False,
    new_mixed_force_equilibrium_certified=False,
    physical_material_admission_authorized=False,
    whole_clock_all_pair_admission_authorized=False,
    full_system_temporal_path_work_certified=False,
    clocks_advanced=False, proposed_source_damage_adopted=False,
    reference_change_full_system_work_certified=False,
    historical_force_solve_reexecuted=False, previous_tensor_quadrature_reexecuted=False,
    source_geometry_elapsed_time_s=0, contact_geometry_elapsed_time_s=2000000,
    memory_elapsed_time_s=1000000, material_geometry_elapsed_time_s=1000000,
    accepted_original_native_intervals=1, accepted_new_native_intervals=0,
    original_point_1_remaining_work_blocks=1, original_point_1_completed=False,
    remaining_physical_conditions=REMAINING_CONDITIONS,
)
sys.path.insert(0, str(ROOT))


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()


def value_digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact(actual, expected, label):
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(label)
    if type(actual) in (dict, list, tuple) and encoded(actual) != encoded(expected):
        raise ValueError(label + ' complete literal encoding')


def sealed(path, key):
    value = json.loads(Path(path).read_bytes())
    if type(value) is not dict or key not in value:
        raise ValueError('Complete selfsealed object required: ' + str(path))
    exact(value[key], value_digest({k: v for k, v in value.items() if k != key}),
          'Complete selfseal required: ' + str(path))
    return value


def source_map():
    return {str(path.relative_to(ROOT)): digest(path)
            for directory in ('tectonics', 'tests', 'scripts', 'analysis')
            for path in sorted((ROOT / directory).rglob('*.py'))}


def preservation(additions, output):
    exact(digest(PRESERVATION_SOURCE), PRESERVATION_RAW, 'Reviewed immutable899 guard unchanged')
    module = types.ModuleType('_hot_mean_stdlib_preservation')
    module.__file__ = str(PRESERVATION_SOURCE)
    exec(compile(PRESERVATION_SOURCE.read_bytes(), str(PRESERVATION_SOURCE), 'exec'), module.__dict__)
    receipt = module.run(additions, output_path=output)
    expected = dict(
        status='PASS_IMMUTABLE899_WITH_EXACT_FOUR_DECLARED_SOURCE_ADDITIONS',
        current_python_source_count=903, prior_python_source_count=899,
        additional_python_source_sha256=additions, all899_python_sources_unchanged=True,
        complete_actual_source_map_checked_before_after=True,
        preserved_current899_manifest_file_count=36,
        preserved_current899_final_readback_archive_file_count=4,
        preserved_prior895_manifest_file_count=50,
        preserved_prior895_final_readback_archive_file_count=4,
        preserved_prior891_manifest_file_count=44,
        preserved_prior891_final_readback_archive_file_count=4,
        preserved_prior887_manifest_file_count=41,
        preserved_prior887_final_readback_archive_file_count=4,
        preserved_prior883_raw_dependency_pin_count=341,
        temporary_dependency_copy_association_count=15,
        authoritative_ancestry_release_closure_count=11,
        preserved_current899_release_closed_file_count=5,
        project_modules_imported_reloaded_or_replaced=False,
        prior_reader_or_auditor_or_preservation_main_reexecuted=False,
        math_or_solver_or_test_or_quadrature_reexecuted=False,
    )
    for key, value in expected.items():
        exact(receipt[key], value, 'Complete previous-source preservation result: ' + key)
    return receipt


def snapshot_rows(directory):
    rows = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.is_symlink() or path.resolve() != path:
            raise ValueError('Only canonical regular executing snapshot files are allowed')
        rows[str(path.relative_to(ROOT))] = digest(path)
    return rows


class TestResults:
    def __init__(self):
        self.collected = 0
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def pytest_collection_finish(self, session):
        self.collected = len(session.items)

    def pytest_runtest_logreport(self, report):
        if report.when == 'call' and report.passed:
            self.passed += 1
        if report.failed:
            self.failed += 1
        if report.skipped:
            self.skipped += 1


def run(release_name):
    if (type(release_name) is not str or not release_name.startswith('release_validation_')
            or '/' in release_name or '\\' in release_name or '..' in release_name):
        raise ValueError('A fresh simple release directory name is required')
    release = PARENT / release_name
    release.mkdir(exist_ok=False)
    started = time.monotonic()
    report = dict(
        format='native-common-hot-mean-validation-1', status='FAIL',
        started_at_utc=datetime.now(timezone.utc).isoformat(), source_count=903,
        previous_source_count=899, python_executable=sys.executable, python_version=sys.version,
        old_force_tests_or_tensor_quadrature_reexecuted=False,
        saved_geometry_and_material_restored_and_validated=False,
        current_physical_scope=PHYSICAL_SCOPE,
        tests_exit_code=None, native_control_succeeded=False,
    )
    frozen = None
    frozen_path = None
    snapshots = None
    expected_snapshots = None
    additions = {}
    pending_error = None
    try:
        frozen = source_map()
        additions = {name: frozen[name] for name in sorted(ADDED) if name in frozen}
        prior = json.loads(OLD_FROZEN.read_bytes())
        if (len(prior) != 899 or len(frozen) != 903 or set(frozen) - set(prior) != ADDED
                or any(frozen.get(name) != pin for name, pin in prior.items())):
            raise ValueError('Exactly four additive903 sources are required, with all899 unchanged')
        frozen_path = PARENT / (release_name + '_frozen_source_map_903.json')
        with frozen_path.open('xb') as stream:
            stream.write(encoded(frozen) + b'\n')
        snapshots = PARENT / 'source_snapshots' / release_name
        snapshots.mkdir(parents=True, exist_ok=False)
        expected_snapshots = {
            str((snapshots / (name.replace('/', '__') + '.txt')).relative_to(ROOT)): pin
            for name, pin in additions.items()
        }
        for name in sorted(ADDED):
            shutil.copyfile(ROOT / name, snapshots / (name.replace('/', '__') + '.txt'))
        report.update(
            frozen_source_map_sha256=value_digest(frozen),
            frozen_source_map_raw_sha256=digest(frozen_path),
            frozen_source_map_path=str(frozen_path.relative_to(ROOT)),
            additional_source_sha256=additions,
            executing_source_snapshots=snapshot_rows(snapshots),
        )
        exact(report['executing_source_snapshots'], expected_snapshots,
              'Exactly four executing snapshots equal frozen903 additions BEFORE')
        before_path = STAGE / (release_name + '_before.json')
        before = preservation(additions, before_path)
        shutil.copyfile(before_path, PARENT / (release_name + '_prior899_before.json'))
        report['before_preservation_sha256'] = before['preservation_sha256']
        import pytest
        plugin = TestResults()
        with (release / 'pytest.log').open('x') as console, redirect_stdout(console), redirect_stderr(console):
            tests_started = time.monotonic()
            report['tests_exit_code'] = int(pytest.main(
                [str(ROOT / TEST_PATH), '-q', '-p', 'no:cacheprovider'], plugins=[plugin]))
            report['tests_elapsed_seconds'] = time.monotonic() - tests_started
        report.update(tests_collected=plugin.collected, tests_passed=plugin.passed,
                      tests_failed=plugin.failed, tests_skipped=plugin.skipped)
        if (report['tests_exit_code'] != 0 or plugin.collected == 0
                or plugin.passed != plugin.collected or plugin.failed or plugin.skipped):
            raise ValueError('Every actually collected new HOT/base-mean test must pass')
        from analysis.geometric_native_common_hot_mean_validation import native_control
        with (release / 'control_console.log').open('x') as console, redirect_stdout(console), redirect_stderr(console):
            native_started = time.monotonic()
            native = native_control.run(release / 'native_control')
            report['native_control_elapsed_seconds'] = time.monotonic() - native_started
        saved_native = sealed(release / 'native_control/report.json', 'report_sha256')
        packet = sealed(release / 'native_control/source_packet.json', 'packet_sha256')
        exact(saved_native, native, 'Returned native result equals actual complete saved report')
        exact(native['status'], 'PASS', 'New complete HOT/base-mean native control PASS')
        for key, value in EXPECTED_NATIVE.items():
            exact(native[key], value, 'Native report preserved endpoint scope: ' + key)
            exact(packet[key], value, 'Native packet preserved endpoint scope: ' + key)
        exact(native['source_packet_sha256'], packet['packet_sha256'], 'Actual complete native packet selfseal link')
        exact(native['source_packet_raw_sha256'], digest(release / 'native_control/source_packet.json'),
              'Actual complete native packet raw link')
        report.update(native_control_succeeded=True, saved_geometry_and_material_restored_and_validated=True,
                      native_control_report_sha256=native['report_sha256'],
                      complete_cold_continuum_operator_certified=True,
                      complete_hot_material_constitutive_operator_certified=True,
                      finite_hot_base_mean_chart_certified=True)
        exact(source_map(), frozen, 'Complete actual903 sources unchanged after native gate')
        exact({str(path.relative_to(release)) for path in release.rglob('*') if path.is_file()},
              set(EXPECTED_PAYLOADS), 'Exactly four payloads before sealed gate report')
        report['status'] = 'PASS'
    except Exception as error:
        report['failure'] = type(error).__name__ + ': ' + str(error)
        pending_error = error
    finally:
        try:
            after_path = STAGE / (release_name + '_after.json')
            after = preservation(additions, after_path)
            shutil.copyfile(after_path, PARENT / (release_name + '_prior899_after.json'))
            report['after_preservation_sha256'] = after['preservation_sha256']
            exact(source_map(), frozen, 'Complete actual903 sources unchanged AFTER preservation')
            exact(digest(frozen_path), report['frozen_source_map_raw_sha256'],
                  'Executing frozen903 source map unchanged AFTER')
            exact(snapshot_rows(snapshots), expected_snapshots,
                  'Exactly four frozen903 executing snapshot rows unchanged AFTER')
            if report['status'] == 'PASS':
                exact({str(path.relative_to(release)) for path in release.rglob('*') if path.is_file()},
                      set(EXPECTED_PAYLOADS), 'Four-payload release unchanged AFTER preservation')
        except Exception as error:
            report['status'] = 'FAIL'
            report['after_preservation_failure'] = type(error).__name__ + ': ' + str(error)
            if pending_error is None:
                pending_error = error
        report['elapsed_seconds'] = time.monotonic() - started
        report['output_raw_sha256'] = {
            str(path.relative_to(release)): digest(path)
            for path in sorted(release.rglob('*')) if path.is_file() and path != release / 'report.json'
        }
        if report['status'] == 'PASS':
            exact(set(report['output_raw_sha256']), set(EXPECTED_PAYLOADS),
                  'PASS output raw map includes all four payloads and native report')
        report['report_sha256'] = value_digest(report)
        with (release / 'report.json').open('xb') as stream:
            stream.write(encoded(report) + b'\n')
        print(json.dumps(report, sort_keys=True))
    if pending_error is not None:
        raise pending_error
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', default='release_validation_01')
    run(parser.parse_args().release)
