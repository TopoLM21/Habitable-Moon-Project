"""Production benchmark guards without launching real children or CUDA."""
import json
from pathlib import Path

import pytest

import analysis.validate_assignment_candidate as validation
from test_assignment_validation import harness, report


@pytest.fixture
def production_harness(harness, monkeypatch):
    child = validation.subprocess.run

    def production_child(command, **kwargs):
        assert Path(command[1]).name == 'run_long_evolution_v131_gpu.py'
        assert '--reference' not in command and '--with-boundary' not in command
        result = child(command, **kwargs)
        path = Path(command[command.index('--output') + 1]) / 'render_timings.json'
        execution = json.loads(path.read_text(encoding='utf-8'))
        assignment = '--assignment-columns' in command
        boundary = '--boundary-forces' in command
        numerical = {
            'assignment_columns': {'enabled': assignment, 'backend': 'cpu_compact_columns',
                                   'calls': 439 if assignment else 0},
            'boundary_forces': {'enabled': boundary, 'backend': 'prepared_cpu_boundary_forces',
                                'calls': 50 if boundary else 0},
        }
        if assignment:
            if harness.failure == 'assignment_missing':
                numerical.pop('assignment_columns')
            elif harness.failure == 'assignment_disabled':
                numerical['assignment_columns']['enabled'] = False
            elif harness.failure == 'assignment_zero':
                numerical['assignment_columns']['calls'] = 0
            elif harness.failure == 'boundary_partial':
                numerical['boundary_forces']['calls'] = 49
            elif harness.failure == 'boundary_backend':
                numerical['boundary_forces']['backend'] = 'incorrect'
        elif harness.failure == 'baseline_leak':
            numerical['boundary_forces']['calls'] = 1
        execution['numerical_execution'] = numerical
        path.write_text(json.dumps(execution), encoding='utf-8')
        return result

    monkeypatch.setattr(validation.subprocess, 'run', production_child)
    monkeypatch.setattr(validation.sys, 'argv', [*validation.sys.argv, '--production', '--with-boundary'])
    return harness


def test_production_uses_ordinary_runner_and_explicit_on_off(production_harness):
    assert validation.main() == 0
    result = report(production_harness)
    assert result['production_runner'] is True
    assert result['inputs_and_sources_unchanged'] is True
    for row in result['runs']:
        enabled = row['mode'] == 'candidate'
        assert ('--assignment-columns' in row['command']) == enabled
        assert ('--boundary-forces' in row['command']) == enabled
        assert ('--no-assignment-columns' in row['command']) != enabled
        assert ('--no-boundary-forces' in row['command']) != enabled
        assert row['checkpoint']['exact'] and row['png']['png_exact']


@pytest.mark.parametrize('failure', [
    'assignment_missing', 'assignment_disabled', 'assignment_zero',
    'boundary_partial', 'boundary_backend', 'baseline_leak',
])
def test_production_execution_mismatch_cannot_publish_speedup(production_harness, failure):
    production_harness.failure = failure
    assert validation.main() == 1
    result = report(production_harness)
    assert result['status'] == 'failed' and result['error']
    assert 'observed_time_reduction_percent' not in result


def test_production_assignment_only_and_workers_are_forwarded(production_harness, monkeypatch):
    argv = [arg for arg in validation.sys.argv if arg != '--with-boundary']
    monkeypatch.setattr(validation.sys, 'argv', [*argv, '--cpu-workers', '2'])
    assert validation.main() == 0
    for command in production_harness.commands:
        assert '--no-boundary-forces' in command
        assert command[command.index('--cpu-workers') + 1] == '2'
