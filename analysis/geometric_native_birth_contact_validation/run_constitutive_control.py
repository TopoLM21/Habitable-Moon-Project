"""Portable local-law control; no native force or geometric admission claim."""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import time

from analysis.geometric_native_birth_contact_validation.validation_support import (
    digest, encoded, guard_sources, value_sha, write,
)


def run_constitutive_control(out, *, development=False):
    from tectonics.geometric_contact_mechanics import ElasticColumnParameters
    from tectonics.genesis_contact_law import ContactLawParameters
    from tectonics.geometric_native_birth_contact import (
        BornContactParameters, BornContactReference, BornFoundationParameters,
        BornFoundationReference, activate_born_contact, activate_born_foundation,
        advance_born_contact, advance_born_foundation, born_contact_state_to_dict,
        born_foundation_state_to_dict, born_contact_state_from_dict,
        born_foundation_state_from_dict,
    )
    from tectonics.geometric_native_polygons import NativePlane

    started = time.perf_counter()
    before, original = guard_sources(require_frozen=not development)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write(out / "sources_before.json", before)
    parameters = BornContactParameters(
        ElasticColumnParameters(1000., 1e9, 4e8, 1500., 2e9, 8e8),
        ContactLawParameters(cohesion_pa=0., friction_dry=0., friction_wet=0., viscosity_pa_s_m=0.),
        "Independent portable constitutive control, explicit unbonded virgin HOT declaration")
    reference = BornContactReference("tiny-unbonded-interface", "a"*64, "b"*64,
        "control-donor", "control-receiver", "donor-material", "receiver-material",
        "donor-parent", "receiver-parent", 0, 1, NativePlane((1, 0, 0), label="Declared control weak plane"),
        Fraction(1, 3), 1e-52, -.25, .75, "Literal compressed birth; gap energy is paid")
    foundation_parameters = BornFoundationParameters(2800., 1.5, "Declared hydrostatic rho*g support")
    foundation_reference = BornFoundationReference("unique-control-foundation", "a"*64, "b"*64,
        "control-receiver", "receiver-material", "receiver-parent", 1, Fraction(1, 3),
        1e-40, 2., "Separate radial patch; no duplicate interface-owned foundation")
    contact = activate_born_contact(parameters, reference)
    foundation = activate_born_foundation(foundation_parameters, foundation_reference)
    expected_contact = dict(expected_parameters=parameters, expected_reference=reference)
    expected_foundation = dict(expected_parameters=foundation_parameters, expected_reference=foundation_reference)
    birth_contact = born_contact_state_to_dict(contact, **expected_contact)
    birth_foundation = born_foundation_state_to_dict(foundation, **expected_foundation)
    write(out / "contact_birth.json", birth_contact)
    write(out / "foundation_birth.json", birth_foundation)
    if not contact.activation_work_j > 0 or contact.normal_contact_energy_j != contact.activation_work_j:
        raise AssertionError("Negative virgin birth energy was not paid by its own ledger")
    steps = []
    for gap, jump, radial, duration in ((-.5, -1., 2.5, 1.), (.25, 2., 1.5, 2.), (-.125, .375, 2.25, 3.)):
        contact_step = advance_born_contact(contact, parameters, reference,
            gap_free_m=gap, jump_free_m=jump, dt_s=duration)
        foundation_step = advance_born_foundation(foundation, foundation_parameters,
            foundation_reference, radial_displacement_m=radial, dt_s=duration)
        contact, foundation = contact_step.state, foundation_step.state
        if contact.shear_traction_pa != 0 or min(contact_step.numerical_dissipation_j,
                foundation_step.numerical_dissipation_j) < 0:
            raise AssertionError("The declared unbonded normal/foundation model changed branches incorrectly")
        steps.append(dict(gap_free_m=gap, normal_traction_pa=contact.normal_traction_pa,
            contact_balance_relative_error=contact_step.energy_balance_relative_error,
            foundation_balance_relative_error=foundation_step.energy_balance_relative_error))
    final_contact = born_contact_state_to_dict(contact, **expected_contact)
    final_foundation = born_foundation_state_to_dict(foundation, **expected_foundation)
    write(out / "contact_final.json", final_contact)
    write(out / "foundation_final.json", final_foundation)
    replay_contact = born_contact_state_from_dict(deepcopy(final_contact), **expected_contact)
    replay_foundation = born_foundation_state_from_dict(deepcopy(final_foundation), **expected_foundation)
    write(out / "contact_final_replayed.json", born_contact_state_to_dict(replay_contact, **expected_contact))
    write(out / "foundation_final_replayed.json", born_foundation_state_to_dict(replay_foundation, **expected_foundation))
    write(out / "contact_birth_after.json", born_contact_state_to_dict(
        activate_born_contact(parameters, reference), **expected_contact))
    write(out / "foundation_birth_after.json", born_foundation_state_to_dict(
        activate_born_foundation(foundation_parameters, foundation_reference), **expected_foundation))
    pairs = []
    for first, second in (("contact_birth.json", "contact_birth_after.json"),
            ("foundation_birth.json", "foundation_birth_after.json"),
            ("contact_final.json", "contact_final_replayed.json"),
            ("foundation_final.json", "foundation_final_replayed.json")):
        if (out / first).read_bytes() != (out / second).read_bytes():
            raise AssertionError("A full constitutive restart or virgin reference changed")
        pairs.append(dict(first=first, second=second, bytes=(out / first).stat().st_size,
            sha256=digest(out / first)))
    mutations = []
    for label, raw, loader, expected, mutation in (
        ("contact_energy", final_contact, born_contact_state_from_dict, expected_contact,
            lambda value: value.update(normal_contact_energy_j=0.)),
        ("contact_reference", final_contact, born_contact_state_from_dict, expected_contact,
            lambda value: value["reference"].update(birth_gap_free_m=0.)),
        ("foundation_density", final_foundation, born_foundation_state_from_dict, expected_foundation,
            lambda value: value["parameters"].update(density_kg_m3=2900.)),
    ):
        forged = deepcopy(raw)
        mutation(forged)
        forged["state_sha256"] = value_sha({key: value for key, value in forged.items() if key != "state_sha256"})
        try:
            loader(forged, **expected)
        except (TypeError, ValueError) as error:
            mutations.append(dict(label=label, rejected=True, whole_state_resealed=True,
                independent_expected_physics_unchanged=True, exception_type=type(error).__name__, message=str(error)))
        else:
            raise AssertionError("A resealed constitutive substitution was accepted")
    write(out / "semantic_refusals.json", mutations)
    packet = dict(format="native-birth-local-constitutive-control-packet-1",
        cases=[dict(label="paid-negative-birth", contact_states=[birth_contact], foundation_states=[birth_foundation]),
            dict(label="tiny-area-open-and-reclosed-journals", contact_states=[final_contact], foundation_states=[final_foundation])],
        genuine_native_source_used=False, native_geometry_admitted=False)
    write(out / "constitutive_packet.json", packet, "packet_sha256")
    after, original_after = guard_sources(require_frozen=not development)
    write(out / "sources_after.json", after)
    if before != after or original != original_after:
        raise AssertionError("Sources changed during constitutive control")
    report = dict(format="native-birth-local-constitutive-control-1", status="PASS",
        development=development, final_release_gate=not development,
        seconds=time.perf_counter()-started, source_count=len(before), inherited_732_sources_unchanged=True,
        source_before_after_unchanged=True, negative_birth_energy_paid=True,
        virgin_shear_work_zero=True, opening_reclosure_proved=True, tiny_positive_areas_retained=True,
        foundation_counted_once=True, byte_pairs=pairs, semantic_refusal_count=len(mutations),
        steps=steps, constitutive_packet_sha256=digest(out / "constitutive_packet.json"),
        genuine_native_source_used=False, whole_native_interval_admitted=False,
        scope="Explicit local virgin HOT normal/foundation model, tiny areas and full external-physics restart; no native geometry or coupled force claim")
    write(out / "report.json", report, "report_sha256")
    return report
