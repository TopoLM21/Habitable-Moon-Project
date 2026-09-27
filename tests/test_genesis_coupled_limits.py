"""Reference-limit diagnostics retain the solver's cumulative safety bounds."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tectonics.genesis_contact import ContactParameters
from tectonics.genesis_coupled import CoupledModel, _RetryStep
from tectonics.genesis_shell import maximum_total_strain
from test_genesis_contact import _source_bytes


class GeometryProbe:
    """Independent strain and jump probes isolate each rejection boundary."""

    def __init__(self, *, cuts=True):
        self.edge_length_m = np.array([100.]) if cuts else np.empty(0)
        self.jump_operator = np.zeros((4 if cuts else 0, 3))
        self.strain = np.zeros((1, 3))
        self.actual = np.zeros((2 if cuts else 0, 2))
        self.invalid_mesh = False
        self.mesh_calls = 0

    def _strain(self, q):
        return self.strain

    def mesh_for(self, contact):
        self.mesh_calls += 1
        if self.invalid_mesh:
            raise ValueError("test mesh distortion")
        return self

    def _geometric_jump(self, contact, mesh=None):
        assert mesh is self
        return self.actual


def _probe(**parameters):
    model = object.__new__(CoupledModel)
    model.radius_m = 100000.
    model.contact_parameters = ContactParameters(**parameters)
    return model, GeometryProbe(), SimpleNamespace(displacement_m=np.zeros(3))


@pytest.mark.parametrize("guard", ["strain", "motion", "radial"])
def test_total_reference_limit_survives_arbitrarily_small_latest_increment(guard):
    model, geometry, contact = _probe()
    if guard == "strain":
        geometry.strain[0, 0] = np.nextafter(.005, np.inf)
    elif guard == "motion":
        contact.displacement_m[0] = np.nextafter(2., np.inf)
    else:
        contact.displacement_m[-1] = np.nextafter(500., np.inf)
    previous = SimpleNamespace(displacement_m=contact.displacement_m*(1.-1e-12))
    with pytest.raises(_RetryStep, match="coupled_reference_geometry_limit"):
        model._check_geometry(geometry, contact, previous)
    assert geometry.mesh_calls == 0  # Reference failure retains precedence.


def test_exact_motion_boundary_is_allowed_and_next_float_is_rejected():
    model, geometry, contact = _probe()
    contact.displacement_m[0] = 2.
    model._check_geometry(geometry, contact)
    contact.displacement_m[0] = np.nextafter(2., np.inf)
    with pytest.raises(_RetryStep, match="coupled_reference_geometry_limit"):
        model._check_geometry(geometry, contact)


@pytest.mark.parametrize("linear,geometric", [(6., 0.), (0., 6.), (3., 6.)])
def test_penetration_guard_uses_worse_of_linear_and_moved_bank_gap(linear, geometric):
    model, geometry, contact = _probe()
    contact.displacement_m[0] = .1
    geometry.jump_operator[0, 0] = -linear/.1
    geometry.actual[0, 0] = -geometric
    metrics = model._geometry_utilization(geometry, contact, geometric_gap_m=geometry.actual[:, 0])
    assert metrics["max_contact_penetration_m"] == max(linear, geometric)
    assert metrics["penetration_utilization"] == max(linear, geometric)/5.
    with pytest.raises(_RetryStep, match="coupled_penetration_limit"):
        model._check_geometry(geometry, contact)


def test_mesh_failure_precedes_penetration_and_sliding():
    model, geometry, contact = _probe()
    geometry.invalid_mesh = True
    geometry.actual[0, 0] = -6.
    with pytest.raises(_RetryStep, match="coupled_mesh_quality_limit"):
        model._check_geometry(geometry, contact)


def test_sliding_and_step_jump_are_distinct_cumulative_and_incremental_limits():
    model, geometry, contact = _probe(max_step_jump_m=1.)
    contact.displacement_m[0] = .1
    geometry.jump_operator[1, 0] = 15.
    previous = SimpleNamespace(displacement_m=np.zeros(3))
    metrics = model._geometry_utilization(geometry, contact, previous)
    assert metrics["max_linear_jump_m"] == pytest.approx(1.5)
    assert metrics["small_sliding_utilization"] == pytest.approx(.75)
    assert metrics["step_jump_utilization"] == pytest.approx(1.5)
    model._check_geometry(geometry, contact)  # Total sliding is admissible.
    with pytest.raises(_RetryStep, match="coupled_jump_step_limit"):
        model._check_geometry(geometry, contact, previous)
    geometry.jump_operator[1, 0] = 30.
    with pytest.raises(_RetryStep, match="coupled_small_sliding_limit"):
        model._check_geometry(geometry, contact, previous)


def test_continuous_shell_has_finite_metrics_without_invented_previous_step():
    model, _, contact = _probe()
    geometry = GeometryProbe(cuts=False)
    contact.displacement_m[0] = 25.
    metrics = model._geometry_utilization(geometry, contact, geometric_gap_m=np.empty(0))
    assert metrics["motion_reference_length_m"] == 10000.
    assert metrics["motion_utilization"] == .125
    assert metrics["small_sliding_utilization"] == metrics["penetration_utilization"] == 0.
    assert "step_jump_utilization" not in metrics
    assert np.isfinite(list(metrics.values())).all()
    model._check_geometry(geometry, contact)


def test_evolving_checkpoint_diagnostics_expose_actual_guard_quantities(tmp_path):
    source = _source_bytes(tmp_path)[0]
    model = CoupledModel(source)
    state = model.initial()
    state, thermal, orbit, _ = model.step(state, model.source.thermal_state, model.source.orbit, 10e-6)
    assert state.stopped_reason is None
    row = model.diagnostics(state, thermal, orbit)
    geometry = model._geometry(state.cut_edges)
    q = state.contact.displacement_m
    motion = np.linalg.norm(q[:-1].reshape(-1, 2), axis=1).max()
    jump = (geometry.jump_operator@q).reshape(-1, 2)
    actual = geometry._geometric_jump(state.contact)
    assert row["max_motion_m"] == motion
    assert row["max_added_strain"] == maximum_total_strain(geometry._strain(q))
    assert row["max_jump_edge_fraction"] == np.max(np.abs(jump)/np.repeat(geometry.edge_length_m, 2)[:, None])
    assert row["max_contact_penetration_m"] == max(0., -jump[:, 0].min(), -actual[:, 0].min())
    # Preserve the old plotted penetration field's geometric meaning.
    assert row["max_penetration_m"] == row["max_geometric_penetration_m"]
    assert row["elastic_strain_utilization"] == maximum_total_strain(state.elastic_strain)/model.source_model.mobile_p.max_elastic_strain
    assert row["interface_area_utilization"] == state.interface_area_change_fraction/model.parameters.max_interface_area_change_fraction
    assert row["cohort_count_utilization"] == len(state.cohorts.trace_index)/model.parameters.max_cohort_count
    assert row["reference_geometry_utilization"] <= 1.
    assert "step_jump_utilization" not in row  # A saved state alone has no previous q.

    # Ratios use the checkpoint's actual thresholds, not hard-coded defaults.
    original = row["motion_utilization"]
    model.contact_parameters = replace(model.contact_parameters, max_motion_edge_fraction=.01)
    assert model.diagnostics(state, thermal, orbit)["motion_utilization"] == 2*original
