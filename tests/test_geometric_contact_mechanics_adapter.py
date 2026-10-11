"""Physical signs, covariance and restart of the frozen local contact solve."""
from dataclasses import asdict, replace
import json
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from analysis.geometric_initiation_validation.admission_case import fixture, snapshot
from tectonics.genesis_contact_law import ContactLawParameters
from tectonics.geometric_boundary import initialize_boundary
from tectonics.geometric_boundary_io import load_boundary_checkpoint, save_boundary_checkpoint
from tectonics.geometric_contact_mechanics import ElasticColumnParameters
from tectonics.geometric_contact_mechanics_adapter import (
    ContactExperiment, mechanics_from_dict, mechanics_to_dict,
    solve_contact_mechanics, solved_fault_snapshot,
)
from tectonics.geometric_contacts import extract_contacts
from tectonics.geometric_initiation import OrientedFault, evaluate_fault
from tectonics.geometric_initiation_adapter import (
    fault_snapshot_from_dict, make_forced_underthrust_resolver,
)
from tectonics.geometric_polarity import PolarityEvidence


DT = 1e10


def setup(*, viscosity=1e15, water=0., sub=1):
    _, surface = fixture()
    contact = extract_contacts(surface.fragments, surface.radius_km)[0]
    law = ContactLawParameters(normal_stiffness_pa_m=1e7, tangential_stiffness_pa_m=1e7,
        tensile_strength_pa=2e6, fracture_energy_j_m2=2e7, cohesion_pa=2e6,
        friction_dry=.2, friction_wet=.1, wet_cohesion_fraction=.5, viscosity_pa_s_m=viscosity)
    columns = ElasticColumnParameters(5e4, 5e10, 2e10, 5e4, 5e10, 2e10)
    cohesion = law.cohesion_pa*(1-(1-law.wet_cohesion_fraction)*water)
    friction = law.friction_dry+(law.friction_wet-law.friction_dry)*water
    fault = OrientedFault(f"declared:{sub}", contact.contact_id, sub, 1-sub,
        math.pi/6., None, cohesion, friction, "synthetic pre-existing inclined plane")
    experiment = ContactExperiment(fault, columns, law, 1e3, water, 0.,
        "synthetic displacement clamps and elastic normal/shear columns; zero pore pressure")
    horizontal = np.asarray(contact.normal_a_to_b)*(1 if fault.overriding_plate == contact.plate_b else -1)
    return surface, contact, experiment, horizontal


def anchors(horizontal, amount):
    result = np.zeros((2, 3))
    result[1] = amount*horizontal
    return result


def convergent_omega(contact):
    # A diagnostic relative rotation converges everywhere on the minor arc.
    result = np.zeros((2, 3))
    result[contact.plate_b] = -.01*np.cross(contact.midpoint, contact.normal_a_to_b)
    return result


def test_locked_contact_keeps_shortening_energy_and_all_material():
    surface, contact, experiment, horizontal = setup()
    archive = initialize_boundary(surface)
    before = snapshot(surface), snapshot(archive)
    history, (result,) = solve_contact_mechanics(surface, (experiment,), anchors(horizontal, 10.), DT)
    assert result.down_dip_plastic_increment_m == 0.
    assert result.step.state.gap_m < 0. and result.step.state.jump_m < 0.
    assert result.step.state.bulk_energy_j > 0. and result.step.state.contact_energy_j > 0.
    assert evaluate_fault(result.solved_fault, contact, convergent_omega(contact), surface.radius_km).status == "locked"
    assert (snapshot(surface), snapshot(archive)) == before
    assert mechanics_to_dict(surface, history)["mechanics_evolution"].endswith("no_material_transfer")


def test_solved_traction_and_signed_slip_feed_existing_admission_without_new_law():
    surface, contact, experiment, horizontal = setup()
    history, (result,) = solve_contact_mechanics(surface, (experiment,), anchors(horizontal, 40.), DT)
    assert result.down_dip_plastic_increment_m > 0.
    assessment = evaluate_fault(result.solved_fault, contact, convergent_omega(contact), surface.radius_km)
    assert assessment.status == "forced_underthrust_admissible"
    assert assessment.normal_stress_pa == pytest.approx(result.step.normal_pressure_pa)
    assert assessment.shear_stress_pa == pytest.approx(-result.step.traction_pa[1])
    assert assessment.strength_pa == pytest.approx(result.step.shear_strength_pa)
    assert assessment.margin_pa == pytest.approx(experiment.law.viscosity_pa_s_m*result.down_dip_plastic_increment_m/DT)
    np.testing.assert_array_equal(np.asarray(result.force_on_sub_n), -np.asarray(result.force_on_over_n))
    np.testing.assert_array_equal(np.asarray(result.torque_on_sub_nm), -np.asarray(result.torque_on_over_nm))
    payload = solved_fault_snapshot(surface, (result,))
    assert fault_snapshot_from_dict(payload, surface) == (result.solved_fault,)
    assert result.step.force_relative_residual < 2e-12
    assert result.step.energy_balance_relative_error < 2e-12
    assert history.states[0][1].plastic_slip_m < 0.


def test_reverse_shear_is_not_down_dip_and_rate_independent_yield_is_not_overstress():
    surface, contact, experiment, horizontal = setup()
    radial = np.asarray(contact.midpoint)
    normal = .5*horizontal+math.cos(math.pi/6)*radial
    down = math.cos(math.pi/6)*horizontal-.5*radial
    driven = np.zeros((2, 3))
    driven[1] = 20.*normal-40.*down
    _, (reverse,) = solve_contact_mechanics(surface, (experiment,), driven, DT)
    assert reverse.down_dip_plastic_increment_m < 0.
    assert evaluate_fault(reverse.solved_fault, contact, convergent_omega(contact), surface.radius_km).status == "locked"
    surface, contact, experiment, horizontal = setup(viscosity=0.)
    _, (plastic,) = solve_contact_mechanics(surface, (experiment,), anchors(horizontal, 40.), DT)
    assert plastic.down_dip_plastic_increment_m > 0.
    assert evaluate_fault(plastic.solved_fault, contact, convergent_omega(contact), surface.radius_km).status == "locked"


def test_rotation_and_owner_renumbering_preserve_physical_contact_outputs():
    surface, contact, experiment, horizontal = setup()
    driven = anchors(horizontal, 40.)
    _, (original,) = solve_contact_mechanics(surface, (experiment,), driven, DT)
    matrix = Rotation.from_rotvec([.31, -.17, .24]).as_matrix()
    rotated = replace(surface, fragments=tuple(replace(item,
        polygon=tuple(map(tuple, np.asarray(item.polygon)@matrix.T))) for item in surface.fragments))
    _, (result,) = solve_contact_mechanics(rotated, (experiment,), driven@matrix.T, DT)
    assert result.step.traction_pa == pytest.approx(original.step.traction_pa, rel=1e-12)
    np.testing.assert_allclose(result.force_on_sub_n, matrix@original.force_on_sub_n, rtol=1e-12, atol=.01)
    np.testing.assert_allclose(result.torque_on_sub_nm, matrix@original.torque_on_sub_nm, rtol=1e-12, atol=1000.)
    np.testing.assert_allclose(result.solved_fault.effective_stress_pa,
        matrix@np.asarray(original.solved_fault.effective_stress_pa)@matrix.T, rtol=1e-12, atol=1e-7)
    labels = {0: 7, 1: 3}
    relabeled = replace(surface, fragments=tuple(replace(item,
        parcel=replace(item.parcel, plate=labels[item.parcel.plate])) for item in reversed(surface.fragments)))
    fault = replace(experiment.fault, subducting_plate=3, overriding_plate=7)
    renumbered = np.zeros((8, 3))
    renumbered[3], renumbered[7] = driven[1], driven[0]
    _, (result,) = solve_contact_mechanics(relabeled, (replace(experiment, fault=fault),), renumbered, DT)
    assert asdict(result.step) == asdict(original.step)
    assert result.force_on_sub_n == original.force_on_sub_n


def test_reaction_and_torque_match_independent_spherical_arc_quadrature():
    surface, contact, experiment, horizontal = setup()
    _, (result,) = solve_contact_mechanics(surface, (experiment,), anchors(horizontal, 40.), DT)
    start, end = np.asarray(contact.start), np.asarray(contact.end)
    angle = math.atan2(np.linalg.norm(np.cross(start, end)), start@end)
    tangent = (end-start*math.cos(angle))/math.sin(angle)
    nodes, weights = np.polynomial.legendre.leggauss(24)
    positions = (nodes+1)*angle/2.
    radial = np.cos(positions)[:, None]*start+np.sin(positions)[:, None]*tangent
    sine, cosine = math.sin(experiment.fault.dip_radians), math.cos(experiment.fault.dip_radians)
    n = sine*horizontal+cosine*radial
    d = cosine*horizontal-sine*radial
    traction = result.step.traction_pa[0]*n+result.step.traction_pa[1]*d
    line_weights = weights*angle/2.*surface.radius_km*1000.
    width = experiment.contact_height_m/sine
    force = width*np.sum(line_weights[:, None]*traction, axis=0)
    torque = width*np.sum(line_weights[:, None]*np.cross(surface.radius_km*1000.*radial, traction), axis=0)
    np.testing.assert_allclose(result.force_on_sub_n, force, rtol=2e-13, atol=1.)
    np.testing.assert_allclose(result.torque_on_sub_nm, torque, rtol=2e-13, atol=1e5)


def test_mirrored_alternatives_and_conflicting_history_are_not_selected_by_id():
    surface, contact, experiment, horizontal = setup()
    mirror = replace(experiment, fault=replace(experiment.fault, fault_id="opposite",
        subducting_plate=0, overriding_plate=1))
    _, results = solve_contact_mechanics(surface, (mirror, experiment), anchors(horizontal, 40.), DT)
    assert results[0].step.traction_pa == pytest.approx(results[1].step.traction_pa)
    faults = tuple(item.solved_fault for item in results)
    _, audit = make_forced_underthrust_resolver(surface, convergent_omega(contact), .1, faults=faults)
    row = next(row for row in audit["contacts"] if row["contact_id"] == contact.contact_id)
    assert row["status"] == "multiple_admissible_directions"
    history = tuple(PolarityEvidence(f"old:{sub}", contact.contact_id, sub, 1-sub,
        "synthetic opposed accepted material") for sub in (0, 1))
    _, audit = make_forced_underthrust_resolver(surface, convergent_omega(contact), .1,
        faults=faults, history=history)
    row = next(row for row in audit["contacts"] if row["contact_id"] == contact.contact_id)
    assert row["status"] == "conflicting_local_history"


def test_joint_checkpoint_restart_preserves_bulk_reference_and_exact_bytes(tmp_path):
    surface, _, experiment, horizontal = setup()
    first, _ = solve_contact_mechanics(surface, (experiment,), anchors(horizontal, 10.), DT)
    archive = initialize_boundary(surface)
    save_boundary_checkpoint(tmp_path/"first.json", surface, archive,
        provenance={"local_contact": mechanics_to_dict(surface, first)})
    loaded, inventory, provenance = load_boundary_checkpoint(tmp_path/"first.json")
    resumed = mechanics_from_dict(provenance["local_contact"], loaded)
    assert resumed == first
    outputs = []
    for state, boundary, previous in ((surface, archive, first), (loaded, inventory, resumed)):
        for amount in (40., 30., 45.):
            previous, _ = solve_contact_mechanics(state, (experiment,), anchors(horizontal, amount), DT,
                                                 previous=previous)
        outputs.append(previous)
        save_boundary_checkpoint(tmp_path/f"final-{len(outputs)}.json", state, boundary,
            provenance={"local_contact": mechanics_to_dict(state, previous)})
    assert outputs[0] == outputs[1]
    assert (tmp_path/"final-1.json").read_bytes() == (tmp_path/"final-2.json").read_bytes()
    assert snapshot(inventory) == snapshot(archive)
    assert outputs[0].states[0][1].step_count == 4
    assert outputs[0].states[0][1].elapsed_time_s == 4*DT


def test_restart_rejects_stale_surface_changed_controls_and_tampered_energy():
    surface, _, experiment, horizontal = setup()
    history, _ = solve_contact_mechanics(surface, (experiment,), anchors(horizontal, 40.), DT)
    with pytest.raises(ValueError, match="surface"):
        solve_contact_mechanics(replace(surface, time_myr=surface.time_myr+.1), (experiment,),
            anchors(horizontal, 40.), DT, previous=history)
    with pytest.raises(ValueError, match="boundary model"):
        solve_contact_mechanics(surface, (replace(experiment, contact_height_m=2e4),),
            anchors(horizontal, 40.), DT, previous=history)
    payload = json.loads(json.dumps(mechanics_to_dict(surface, history)))
    payload["states"][experiment.fault.fault_id]["bulk_energy_j"] *= 2.
    with pytest.raises(ValueError, match="bulk energy"):
        mechanics_from_dict(payload, surface)


def test_solved_traction_cannot_be_rebound_to_another_surface_with_same_contact_ids():
    surface, _, experiment, horizontal = setup()
    _, results = solve_contact_mechanics(surface, (experiment,), anchors(horizontal, 40.), DT)
    moved_clock = replace(surface, time_myr=surface.time_myr+.1)
    assert {item.contact_id for item in extract_contacts(surface.fragments, surface.radius_km)} == {
        item.contact_id for item in extract_contacts(moved_clock.fragments, moved_clock.radius_km)}
    with pytest.raises(ValueError, match="no implicit stress rebinding"):
        solved_fault_snapshot(moved_clock, results)
    matrix = Rotation.from_rotvec([.31, -.17, .24]).as_matrix()
    rotated = replace(surface, fragments=tuple(replace(item,
        polygon=tuple(map(tuple, np.asarray(item.polygon)@matrix.T))) for item in surface.fragments))
    with pytest.raises(ValueError, match="no implicit stress rebinding"):
        solved_fault_snapshot(rotated, results)


def test_wet_strength_is_the_existing_law_and_unsupported_opening_is_rejected():
    surface, contact, experiment, horizontal = setup(water=.7)
    _, (result,) = solve_contact_mechanics(surface, (experiment,), anchors(horizontal, 40.), DT)
    assessment = evaluate_fault(result.solved_fault, contact, convergent_omega(contact), surface.radius_km)
    assert assessment.strength_pa == pytest.approx(result.step.shear_strength_pa)
    with pytest.raises(ValueError, match="actual wet contact law"):
        replace(experiment, fault=replace(experiment.fault, friction=.9))
    with pytest.raises(ValueError, match="opening"):
        solve_contact_mechanics(surface, (experiment,), anchors(horizontal, -40.), DT)
    with pytest.raises(ValueError, match="without a prescribed stress"):
        replace(experiment, fault=result.solved_fault)
    with pytest.raises(ValueError, match="zero pore pressure"):
        replace(experiment, pore_pressure_pa=1e6)


def test_no_declared_plane_produces_no_fictitious_stress():
    _, surface = fixture()
    history, results = solve_contact_mechanics(surface, (), np.zeros((2, 3)), DT)
    assert history.states == results == ()
    assert solved_fault_snapshot(surface, results)["faults"] == []


def test_checkpoint_does_not_restore_missing_parameters_from_defaults():
    surface, _, experiment, horizontal = setup()
    history, _ = solve_contact_mechanics(surface, (experiment,), anchors(horizontal, 40.), DT)
    payload = json.loads(json.dumps(mechanics_to_dict(surface, history)))
    del payload["experiments"][0]["law"]["viscosity_pa_s_m"]
    with pytest.raises(ValueError, match="every physical parameter"):
        mechanics_from_dict(payload, surface)


@pytest.mark.parametrize("dt", [-1., 0., float("nan"), float("inf"), True])
def test_empty_experiment_still_validates_mechanical_time(dt):
    _, surface = fixture()
    with pytest.raises(ValueError, match="dt_s"):
        solve_contact_mechanics(surface, (), np.zeros((2, 3)), dt)


def test_alternative_branches_cannot_restart_with_different_clocks():
    surface, _, experiment, horizontal = setup()
    mirror = replace(experiment, fault=replace(experiment.fault, fault_id="opposite",
        subducting_plate=0, overriding_plate=1))
    first, _ = solve_contact_mechanics(surface, (experiment, mirror), anchors(horizontal, 10.), DT)
    second, _ = solve_contact_mechanics(surface, (experiment, mirror), anchors(horizontal, 40.), DT, previous=first)
    inconsistent = replace(second, states=(first.states[0], second.states[1]))
    with pytest.raises(ValueError, match="clock"):
        solve_contact_mechanics(surface, (experiment, mirror), anchors(horizontal, 40.), DT, previous=inconsistent)
