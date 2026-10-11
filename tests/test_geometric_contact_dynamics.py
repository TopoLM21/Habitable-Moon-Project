"""Independent force/work and rejection checks of the frozen Euler solve."""
from dataclasses import replace
import json
import math

import numpy as np
import pytest

from analysis.geometric_initiation_validation.admission_case import fixture, snapshot
from tectonics.basal_coupling import PrescribedBasalParameters
from tectonics.genesis_contact_law import ContactLawParameters
from tectonics.geometric_basal_loading import assemble_geometric_basal_load
from tectonics.geometric_boundary import initialize_boundary
from tectonics.geometric_contact_dynamics import (
    UnresolvedContactDynamics, advance_contact_dynamics, contact_dynamics_from_dict,
    contact_dynamics_to_dict, initialize_contact_dynamics, validate_contact_dynamics_state,
)
from tectonics.geometric_contact_mechanics import ElasticColumnParameters
from tectonics.geometric_contact_mechanics_adapter import ContactExperiment
from tectonics.geometric_contacts import extract_contacts
from tectonics.geometric_initiation import OrientedFault
from tectonics.geometric_polarity import PolarityEvidence


DT = 1e10


def setup(*, traction=1e3, friction=.2, viscosity=1e14, cohesion=2e6,
          dip=math.pi/6., gap=-20., bound=.01):
    mesh, surface = fixture()
    parameters = PrescribedBasalParameters(31, traction, 1e10, 2.)
    lid = {item.fragment_id: .1+item.parcel.cell for item in surface.fragments}
    load = assemble_geometric_basal_load(mesh, surface, parameters, lid,
        "synthetic nonuniform explicit thermal lid, prescribed mantle source and independent drag")
    law = ContactLawParameters(normal_stiffness_pa_m=1e7, tangential_stiffness_pa_m=1e7,
        tensile_strength_pa=2e6, fracture_energy_j_m2=2e7, cohesion_pa=cohesion,
        friction_dry=friction, friction_wet=friction/2., wet_cohesion_fraction=.5,
        viscosity_pa_s_m=viscosity)
    columns = ElasticColumnParameters(5e4, 5e10, 2e10, 5e4, 5e10, 2e10)
    contacts = extract_contacts(surface.fragments, surface.radius_km)
    experiments = tuple(ContactExperiment(OrientedFault(f"declared:{index}", contact.contact_id,
        1, 0, dip, None, cohesion, friction, "explicit synthetic existing dipping plane"),
        columns, law, 1e3, 0., 0., "explicit synthetic columns and thin radial strip")
        for index, contact in enumerate(contacts))
    offsets = {experiment.fault.fault_id: (gap, 0.) for experiment in experiments}
    initial = initialize_contact_dynamics(surface, experiments, offsets, load,
        initial_loading_dt_s=DT, max_rotation_radians=bound)
    return surface, contacts, experiments, offsets, load, initial


def test_global_locked_solution_matches_independent_linear_spring_system():
    surface, contacts, experiments, _, load, initial = setup()
    step = advance_contact_dynamics(surface, initial, load, DT)
    matrix = np.asarray(load.drag_nm_s)/DT
    initial_force = np.zeros(6)
    local = dict(initial.states)
    rows, weights = [], []
    for contact, experiment in zip(contacts, experiments):
        sign = 1. if experiment.fault.overriding_plate == contact.plate_b else -1.
        mean = sign*np.asarray(contact.moment_arm_cross_normal_km2)/contact.length_km*1000.
        row = np.zeros((2, 3))
        row[experiment.fault.subducting_plate] = -mean
        row[experiment.fault.overriding_plate] = mean
        row = row.ravel()
        sine, cosine = math.sin(experiment.fault.dip_radians), math.cos(experiment.fault.dip_radians)
        area = contact.length_km*1000.*experiment.contact_height_m/sine
        normal = 1./(1./experiment.columns.normal_stiffness_pa_m+1./experiment.law.normal_stiffness_pa_m)
        shear = 1./(1./experiment.columns.shear_stiffness_pa_m+1./experiment.law.tangential_stiffness_pa_m)
        matrix = matrix+area*(normal*sine*sine+shear*cosine*cosine)*np.outer(row, row)
        old = local[experiment.fault.fault_id]
        initial_force += area*(sine*old.traction_normal_pa+cosine*old.traction_shear_pa)*row
        rows.append(row)
        weights.append(contact.length_km*1000.)
    expected = np.linalg.solve(matrix, np.asarray(load.torque_nm)-initial_force)
    np.testing.assert_allclose(np.asarray(step.state.rotation_radians).ravel(), expected,
                               rtol=2e-10, atol=2e-18)
    assert all(result.step.slip_increment_m == 0. for result in step.contacts)
    assert step.torque_signal_relative_residual < 2e-10
    assert step.incremental_energy_balance_relative_error < 2e-9
    assert step.minimum_certified_eigenvalue > 0.
    # Full-sphere rigid trace cannot converge on every boundary without the
    # independently supplied precompression offsets.
    np.testing.assert_allclose(np.asarray(weights)@np.asarray(rows), 0., atol=1e-5)
    actual_source = np.asarray(load.torque_nm)@expected
    assert step.source_work_j == pytest.approx(actual_source, rel=2e-10)


def test_sliding_global_work_and_internal_torques_are_paired_without_material_transfer():
    surface, _, _, _, load, initial = setup(friction=0., cohesion=0.)
    archive = initialize_boundary(surface)
    before = snapshot(surface), snapshot(archive), snapshot(initial)
    step = advance_contact_dynamics(surface, initial, load, DT)
    assert any(abs(result.step.slip_increment_m) > 0. for result in step.contacts)
    assert step.viscous_work_j > 0.
    assert step.friction_work_j == 0.
    assert step.numerical_dissipation_j >= 0.
    combined = math.fsum((step.basal_dissipation_j, step.stored_energy_increment_j,
        step.friction_work_j, step.viscous_work_j, step.numerical_dissipation_j))
    assert abs(step.source_work_j-combined) <= 5e-10*max(abs(step.source_work_j), abs(combined))+step.energy_rounding_tolerance_j
    np.testing.assert_allclose(np.sum(step.contact_torque_nm, axis=0), 0., atol=1e6)
    for contact in step.contacts:
        np.testing.assert_array_equal(contact.torque_on_sub_nm, -np.asarray(contact.torque_on_over_nm))
    assert (snapshot(surface), snapshot(archive), snapshot(initial)) == before


def test_exact_global_restart_keeps_preloads_and_old_elastic_reference():
    surface, _, _, _, load, initial = setup()
    first = advance_contact_dynamics(surface, initial, load, DT).state
    raw = json.loads(json.dumps(contact_dynamics_to_dict(surface, first, load)))
    restored = contact_dynamics_from_dict(raw, surface, load)
    assert restored == first
    direct = advance_contact_dynamics(surface, first, load, DT).state
    resumed = advance_contact_dynamics(surface, restored, load, DT).state
    assert direct == resumed
    assert json.dumps(contact_dynamics_to_dict(surface, direct, load), sort_keys=True) == json.dumps(
        contact_dynamics_to_dict(surface, resumed, load), sort_keys=True)
    assert direct.step_count == 2
    assert direct.elapsed_time_s == 2*DT
    assert all(local.elapsed_time_s == 3*DT for _, local in direct.states)


def test_restart_rejects_balanced_preparation_and_source_ledger_tamper():
    surface, _, _, _, load, initial = setup()
    first = advance_contact_dynamics(surface, initial, load, DT).state
    tampered = replace(first, preparation_work_j=first.preparation_work_j+1e10,
                       cumulative_source_work_j=first.cumulative_source_work_j-1e10)
    with pytest.raises(ValueError, match="source work"):
        validate_contact_dynamics_state(surface, tampered, load)
    with pytest.raises(ValueError, match="preparation work"):
        validate_contact_dynamics_state(surface, replace(first,
            preparation_work_j=first.preparation_work_j+1e10), load)


def test_missing_planes_alternatives_and_conflicting_history_reject_atomically():
    surface, contacts, experiments, offsets, load, initial = setup()
    before = snapshot(surface), snapshot(initial)
    with pytest.raises(UnresolvedContactDynamics, match="exactly one"):
        initialize_contact_dynamics(surface, experiments[:-1],
            {key: value for key, value in offsets.items() if key != experiments[-1].fault.fault_id},
            load, initial_loading_dt_s=DT, max_rotation_radians=.01)
    alternate = replace(experiments[0], fault=replace(experiments[0].fault,
        fault_id="alternate", subducting_plate=0, overriding_plate=1))
    with pytest.raises(UnresolvedContactDynamics, match="exactly one"):
        initialize_contact_dynamics(surface, experiments+(alternate,), dict(offsets, alternate=(-20., 0.)),
            load, initial_loading_dt_s=DT, max_rotation_radians=.01)
    history = tuple(PolarityEvidence(f"old:{sub}", contacts[0].contact_id, sub, 1-sub,
        "explicit opposed old material") for sub in (0, 1))
    with pytest.raises(UnresolvedContactDynamics, match="Conflicting"):
        initialize_contact_dynamics(surface, experiments, offsets, load,
            initial_loading_dt_s=DT, max_rotation_radians=.01, history=history)
    assert (snapshot(surface), snapshot(initial)) == before


def test_history_is_retained_and_revalidated_after_restart():
    surface, contacts, experiments, offsets, load, _ = setup()
    history = (PolarityEvidence("old:1", contacts[0].contact_id, 1, 0, "explicit old material"),)
    prepared = initialize_contact_dynamics(surface, experiments, offsets, load,
        initial_loading_dt_s=DT, max_rotation_radians=.01, history=history)
    raw = contact_dynamics_to_dict(surface, prepared, load)
    assert contact_dynamics_from_dict(raw, surface, load).polarity_history == history
    raw["polarity_history"][0]["subducting_plate"], raw["polarity_history"][0]["overriding_plate"] = 0, 1
    with pytest.raises(UnresolvedContactDynamics, match="opposed"):
        contact_dynamics_from_dict(raw, surface, load)


def test_uncertified_coulomb_network_is_rejected_without_picking_a_root():
    surface, _, _, _, load, initial = setup(friction=1.8, viscosity=0., dip=.1)
    before = snapshot(initial)
    with pytest.raises(UnresolvedContactDynamics, match="uniqueness certificate"):
        advance_contact_dynamics(surface, initial, load, 1e16)
    assert snapshot(initial) == before


def test_unsupported_opening_and_excessive_fixed_frame_rotation_reject():
    surface, _, _, _, load, initial = setup(traction=1e12, gap=-.001)
    before = snapshot(initial)
    with pytest.raises(UnresolvedContactDynamics, match="[Cc]losed|opening"):
        advance_contact_dynamics(surface, initial, load, DT)
    assert snapshot(initial) == before
    surface, _, _, _, load, initial = setup(bound=1e-12)
    with pytest.raises(UnresolvedContactDynamics, match="validity bound"):
        advance_contact_dynamics(surface, initial, load, DT)


@pytest.mark.parametrize("dt", [0., -1., float("nan"), float("inf"), True])
def test_invalid_step_time_is_rejected(dt):
    surface, _, _, _, load, initial = setup()
    with pytest.raises(ValueError, match="dt_s"):
        advance_contact_dynamics(surface, initial, load, dt)
