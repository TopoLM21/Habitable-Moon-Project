"""Independent work, force and history checks for paired-bank mechanics."""
from dataclasses import replace

import numpy as np
import pytest

from tectonics.genesis_contact import ContactModel, ContactParameters, SECONDS_PER_YEAR
from tectonics.genesis_contact_law import ContactLawParameters
from test_genesis_contact import _source_bytes


def make_model(tmp_path, *, weak=False):
    law = ContactLawParameters()
    if weak:
        law = replace(law, cohesion_pa=1000., friction_dry=0., friction_wet=0.)
    return ContactModel(_source_bytes(tmp_path)[0],
                        ContactParameters(alignment_degrees=90.), law)


def direct_interface_energy(model, state):
    jumps = (model.jump_operator@state.displacement_m).reshape(-1, 2)
    gap, tangential = jumps.T
    law = model.law_parameters
    normal = .5*law.normal_stiffness_pa_m*np.where(
        gap < 0., gap**2, (1-state.interface_damage)*gap**2)
    shear = .5*law.tangential_stiffness_pa_m*(tangential-state.plastic_slip_m)**2
    return float(model.interface_area_m2@(normal+shear))


def test_diagnostics_reads_stored_energy_without_an_extra_viscous_return(tmp_path, monkeypatch):
    model = make_model(tmp_path, weak=True)
    state = model.step(model.initial(), 1.)
    assert state.stopped_reason is None
    assert state.viscous_work_cell_j.sum() > 0
    expected = direct_interface_energy(model, state)
    again = model._response(state, state.displacement_m, state.last_step_years*SECONDS_PER_YEAR)
    relaxed = float(model.interface_area_m2@again["recoverable_energy_j_m2"])
    assert relaxed < expected  # Repeating the return would silently advance slip.
    def forbidden(*args, **kwargs):
        raise AssertionError("Diagnostics must not propose another constitutive update")
    monkeypatch.setattr(model, "_response", forbidden)
    row = model.diagnostics(state)
    assert row["interface_elastic_energy_j"] == pytest.approx(expected, rel=2e-14)
    assert model.diagnostics(state) == row


def test_bulk_internal_force_is_derivative_of_physical_stored_energy(tmp_path):
    model = make_model(tmp_path)
    rng = np.random.default_rng(871)
    displacement = rng.normal(size=model.membrane.ndof)*.3
    direction = rng.normal(size=model.membrane.ndof)
    direction /= np.linalg.norm(direction)
    force = model.initial_bulk_force+model.bulk_matrix@displacement
    h = 1.
    derivative = (model._bulk_energy(displacement+h*direction)
                  -model._bulk_energy(displacement-h*direction))/(2*h)
    assert derivative == pytest.approx(float(force@direction), rel=2e-8, abs=1e7)


def test_uniform_radial_increment_has_physical_force_and_energy_scale(tmp_path):
    model = make_model(tmp_path)
    q = np.zeros(model.membrane.ndof)
    q[-1] = 1.  # One metre, not one kilometre or one unit-sphere radius.
    expected_strain = np.tile([1/model.radius_m, 1/model.radius_m, 0.], (len(model.depth_m), 1))
    np.testing.assert_allclose(model._strain(q), expected_strain, rtol=0, atol=0)
    radial_stiffness = np.einsum("i,fij,j->f", [1., 1., 0.], model.elasticity, [1., 1., 0.])
    expected_force = float(np.dot(model.topology.mesh.areas_unit_sphere*model.depth_m, radial_stiffness))
    assert (model.bulk_matrix@q)[-1] == pytest.approx(expected_force, rel=2e-14)


def test_interface_force_is_equal_opposite_and_uses_physical_edge_area(tmp_path):
    model = make_model(tmp_path)
    index = next(i for i, banks in enumerate(model.topology.bank_vertices)
                 if banks[0, 0] != banks[1, 0])
    point = 2*index
    tractions = np.zeros((len(model.interface_area_m2), 2))
    tractions[point] = [1e6, -3e6]
    force = model.jump_operator.T@(tractions*model.interface_area_m2[:, None]).ravel()
    spatial = np.einsum("vij,vj->vi", model.membrane.vertex_basis, force[:-1].reshape(-1, 2))
    first, second = model.topology.bank_vertices[index, :, 0]
    expected = model.interface_area_m2[point]*(1e6*model.interface_normal[point]
               -3e6*model.interface_tangent[point])
    np.testing.assert_allclose(spatial[first], -expected, rtol=2e-14, atol=1.)
    np.testing.assert_allclose(spatial[second], expected, rtol=2e-14, atol=1.)
    np.testing.assert_allclose(spatial.sum(axis=0), 0., atol=1.)
    assert force[-1] == 0.  # Common radius motion does not open the paired trace.


def test_accepted_newton_state_balances_stored_forces_and_drag(tmp_path):
    model = make_model(tmp_path, weak=True)
    initial = model.initial()
    state = model.step(initial, 1.)
    assert state.stopped_reason is None and state.accepted_steps == 1
    q = state.displacement_m
    internal = model.bulk_matrix@q+model.initial_bulk_force
    contact = model.jump_operator.T@(state.traction_pa*model.interface_area_m2[:, None]).ravel()
    drag_coefficient = model.drag_area_m2*model.parameters.basal_drag_pa_s_m/SECONDS_PER_YEAR
    drag_force = drag_coefficient*(q-initial.displacement_m)
    residual = internal+contact+drag_force-model.external_force
    scale = max(np.linalg.norm(internal), np.linalg.norm(contact), np.linalg.norm(model.external_force))
    assert np.linalg.norm(residual)/scale <= model.parameters.equilibrium_tolerance
    assert state.drag_work_j == pytest.approx(float(drag_force@(q-initial.displacement_m)), rel=2e-14)
    assert state.external_work_j == pytest.approx(float(model.external_force@(q-initial.displacement_m)), rel=2e-14)


def test_elastic_step_work_difference_is_backward_euler_remainder(tmp_path):
    model = make_model(tmp_path)
    state = model.step(model.initial(), .01)
    assert state.stopped_reason is None
    assert state.accepted_steps == 1
    np.testing.assert_array_equal(state.interface_damage, 0.)
    np.testing.assert_array_equal(state.plastic_slip_m, 0.)
    q = state.displacement_m
    interface_energy = direct_interface_energy(model, state)
    measured = (model.initial_elastic_energy_j+state.external_work_j-model._bulk_energy(q)
                -interface_energy-state.drag_work_j)
    expected = .5*float(q@(model.bulk_matrix@q))+interface_energy
    assert expected > 0
    assert measured == pytest.approx(expected, rel=2e-4, abs=2e8)
