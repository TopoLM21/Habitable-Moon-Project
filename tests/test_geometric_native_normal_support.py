"""Actual source collars must fit material banks before any 3-D admission.

The authenticated diagnostic source is read from fixed packets; no genuine
force case is rebuilt. Independent radial inequalities exclude the union of
all receiver-owner banks on a positive material strip, rather than one probe.
"""
from dataclasses import replace
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json

import pytest

from tectonics.geometric_native_pose import FractionInterval as I


def _api():
    from tectonics import geometric_native_normal_support as module
    return module


@lru_cache(maxsize=1)
def _source():
    return _api().load_native_normal_support_source()


@lru_cache(maxsize=1)
def _proofs():
    return _api().preflight_native_normal_support(_source())


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()


def _resealed(value):
    body = {name: item for name, item in value.items() if name != "packet_sha256"}
    value["packet_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    return _canonical(value)


def test_authentication_preserves_all_complete_source_payloads_and_the_saved_field():
    source = _source()
    curved = json.loads(_api().CURVED_PACKET.read_bytes())
    joint = json.loads(_api().JOINT_PACKET.read_bytes())
    assert source.curved_packet_raw_sha256 == hashlib.sha256(_api().CURVED_PACKET.read_bytes()).hexdigest()
    assert source.joint_packet_raw_sha256 == hashlib.sha256(_api().JOINT_PACKET.read_bytes()).hexdigest()
    assert source.saved_field_declaration_precision == 128
    assert len(source.banks) == 13
    assert len(source.families) == 2
    assert tuple(name for name, _ in source.complete_source_dto_json) == (
        "before", "current_load", "original_unaccepted_trial_birth_support",
        "full_reference_face_birth_support")
    for name, text in source.complete_source_dto_json:
        assert json.loads(text) == curved[name] == joint[name]
    assert json.loads(source.saved_field_json) == curved["saved_joint_reference_field"]


def test_positive_source_strip_retains_both_complete_native_angle_ends_and_preload():
    source = _source()
    families = {row.family_id: row for row in source.families}
    assert {proof.family_id for proof in _proofs()} == {"born-shared-0", "born-shared-3"}
    for proof in _proofs():
        family = families[proof.family_id]
        assert proof.activation_time_fraction == 0
        assert proof.interior_depth_m == I(F(100), F(500))
        assert proof.angle_bounds_radians == I(F(0), family.native_angle_end_bounds_radians.hi)
        assert proof.native_angle_end_bounds_radians == family.native_angle_end_bounds_radians
        assert proof.native_angle_end_bounds_radians.lo > 0
        assert proof.native_angle_end_bounds_radians.width > 0
        assert proof.column_lengths_m == (F(1000), F(1000))
        assert proof.ray_distance_bounds_m == I(F(1000))
        assert proof.source_preload_gap_m == F.from_float(-.01)
        assert proof.positive_measure_activation_strip_proved
        assert proof.full_source_column_support_rejected


def test_actual_receiving_rods_exit_every_same_owner_roof_by_more_than_356_metres():
    source = _source()
    for proof in _proofs():
        actual_roofs = {bank.component_id: source.radius_m+bank.roof_radial_displacement_m
            for bank in source.banks if bank.owner == proof.receiver_owner}
        assert set(identity for identity, _ in proof.all_same_owner_roofs) == set(actual_roofs)
        assert {identity: roof.lo for identity, roof in proof.all_same_owner_roofs} == actual_roofs
        assert all(roof.width == 0 for _, roof in proof.all_same_owner_roofs)
        assert proof.owner_max_roof_radius_bounds_m == I(max(actual_roofs.values()))
        assert proof.outside_roof_margin_bounds_m.lo > F(356)
        assert proof.ray_endpoint_radius_bounds_m.lo-max(actual_roofs.values()) > F(356)
        # This sufficient global exclusion does not rely on a particular mask
        # polygon, a pressure cutoff, or one angle/depth sample.
        assert proof.receiver_owner == 0 and proof.donor_owner == 1
        assert proof.receiver_material_kind == "hot_birth"


def test_interior_strip_lies_inside_both_actual_material_sources_before_rod_extension():
    source = _source()
    banks = {bank.component_id: bank for bank in source.banks}
    for proof in _proofs():
        donor, receiver = banks[proof.donor_component_id], banks[proof.receiver_component_id]
        assert proof.donor_material_depth_bounds_m.lo > 0
        assert proof.receiver_material_depth_bounds_m.lo > 0
        assert proof.donor_material_depth_bounds_m.hi < donor.areal_material_thickness_m
        assert proof.receiver_material_depth_bounds_m.hi < receiver.areal_material_thickness_m
        shared_top = min(source.radius_m+donor.roof_radial_displacement_m,
            source.radius_m+receiver.roof_radial_displacement_m)
        assert proof.common_reference_top_radius_m == shared_top
        physical_depth = proof.interior_depth_m
        reference = proof.reference_radius_bounds_m
        assert reference.lo**3 <= shared_top**3-3*source.radius_m**2*physical_depth.hi
        assert reference.hi**3 >= shared_top**3-3*source.radius_m**2*physical_depth.lo


def test_radial_endpoint_exclusion_has_an_independent_algebraic_interval_oracle():
    for proof in _proofs():
        r = proof.reference_radius_bounds_m
        n = proof.receiving_radial_normal_bounds
        assert F(4, 5) < n.lo <= n.hi < 1
        length = proof.ray_distance_bounds_m.lo
        lower_squared = r.lo**2+2*r.lo*n.lo*length+length**2
        upper_squared = r.hi**2+2*r.hi*n.hi*length+length**2
        endpoint = proof.ray_endpoint_radius_bounds_m
        assert endpoint.lo**2 <= lower_squared
        assert endpoint.hi**2 >= upper_squared
        assert endpoint.lo > proof.owner_max_roof_radius_bounds_m.hi
        assert proof.outside_roof_margin_bounds_m.lo == endpoint.lo-proof.owner_max_roof_radius_bounds_m.hi


def test_roof_exclusion_is_failure_of_declared_full_columns_and_confers_no_admission():
    source = _source()
    assert not source.complete_three_dimensional_strain_energy_certified
    assert not source.physical_material_admission_authorized
    for proof in _proofs():
        assert not proof.full_normal_collars_placeable
        assert not proof.normal_columns_clipped
        assert not proof.source_normal_stiffness_reused_as_spatial_energy
        assert not proof.previous_endpoint_force_certificate_inherited
        assert not proof.complete_three_dimensional_strain_energy_certified
        assert not proof.physical_material_admission_authorized


@pytest.mark.parametrize("target", ["columns", "saved_depth_field", "preload"])
def test_self_consistent_edited_packet_cannot_supply_new_source_or_field_authority(target):
    packet = json.loads(_api().CURVED_PACKET.read_bytes())
    if target == "columns":
        packet["full_reference_face_birth_support"]["interface_carriers"][0]["declaration"]["parameters"]["columns"]["over_length_m"] = 900.
    elif target == "saved_depth_field":
        packet["saved_joint_reference_field"]["declaration_precision"] = 160
    else:
        packet["full_reference_face_birth_support"]["interface_carriers"][0]["declaration"]["birth_gap_free_m"] = 0.
    forged = _resealed(packet)
    assert json.loads(forged)["packet_sha256"] != json.loads(_api().CURVED_PACKET.read_bytes())["packet_sha256"]
    with pytest.raises((ValueError, TypeError), match="authenticated|raw bytes|packet"):
        _api().load_native_normal_support_source(curved_packet_bytes=forged)
    assert _source().families[0].column_lengths_m == (F(1000), F(1000))


def test_stale_source_field_binding_is_refused_even_if_the_receipt_is_resealed():
    source = _source()
    forged = replace(source, saved_field_declaration_precision=160)
    # Match the receipt's complete structural digest; this is still not the
    # independently authenticated original source registered by packet read.
    forged = replace(forged, source_snapshot_sha256=_api()._hash_snapshot(forged))
    with pytest.raises((ValueError, TypeError), match="re-sealed|altered|authentication"):
        _api().validate_native_normal_support_source(forged)
    assert _api().validate_native_normal_support_source(source) is source


def test_forged_zero_margin_cannot_replace_semantic_geometric_replay():
    genuine = _proofs()[0]
    forged = replace(genuine, outside_roof_margin_bounds_m=I(F(0)))
    with pytest.raises((ValueError, TypeError), match="replay|certificate"):
        _api().validate_native_normal_support_certificate(forged, source=_source())


def test_serialized_preflight_retains_full_material_sources_and_all_failure_scope_flags():
    source = _source()
    payload = _api().native_normal_support_to_dict(source, _proofs())
    assert payload["status"] == "SOURCE_COLUMN_SUPPORT_REJECTED"
    assert len(payload["source"]["banks"]) == 13
    assert len(payload["certificates"]) == 2
    assert not payload["normal_columns_clipped"]
    assert not payload["physical_material_admission_authorized"]
    assert not payload["complete_three_dimensional_strain_energy_certified"]
    assert not payload["previous_endpoint_force_certificate_inherited"]


def test_pure_normal_ray_radius_preserves_unit_frame_correlation():
    result = _api().normal_ray_radius_bounds(I(F(3)), I(F(3, 5)), I(F(4)))
    squared = F(197, 5)
    assert result.lo**2 <= squared <= result.hi**2
    assert result.width < F(1, 10**150)
    assert _api().normal_ray_radius_bounds(I(F(3)), I(F(0)), I(F(4))) == I(F(5))
    assert _api().normal_ray_radius_bounds(I(F(3)), I(F(1)), I(F(4))) == I(F(7))


def test_inverse_roof_helper_recovers_exact_first_outward_intersection():
    result = _api().normal_ray_outer_roof_exit_bounds(I(F(5)), I(F(3, 5)), I(F(17, 2)))
    assert result.lo <= F(9, 2) <= result.hi
    assert result.width < F(1, 10**150)
    endpoint = _api().normal_ray_radius_bounds(I(F(5)), I(F(3, 5)), result)
    assert endpoint.lo <= F(17, 2) <= endpoint.hi
    radial = _api().normal_ray_outer_roof_exit_bounds(I(F(3)), I(F(1)), I(F(5)))
    assert radial == I(F(2))


@pytest.mark.parametrize("arguments", [(I(F(0)), I(F(1)), I(F(2))),
    (I(F(3)), I(F(6, 5)), I(F(4))), (3., I(F(1)), I(F(4)))])
def test_radius_helper_rejects_nonphysical_radius_normal_or_nonliteral_interval(arguments):
    with pytest.raises((ValueError, TypeError)):
        _api().normal_ray_radius_bounds(*arguments)


@pytest.mark.parametrize("arguments", [(I(F(3)), I(F(0)), I(F(5))),
    (I(F(3)), I(F(-1)), I(F(5))), (I(F(5)), I(F(1)), I(F(4)))])
def test_inverse_roof_helper_requires_an_interior_point_and_outward_normal(arguments):
    with pytest.raises((ValueError, TypeError)):
        _api().normal_ray_outer_roof_exit_bounds(*arguments)


@pytest.mark.parametrize("depth", [I(F(0), F(500)), I(F(100), F(100)),
    I(F(100), F(10000))])
def test_negative_witness_does_not_replace_material_by_an_outside_or_zero_measure_strip(depth):
    with pytest.raises((ValueError, TypeError)):
        _api().bound_native_normal_collar_roof_exclusion(_source(), "born-shared-0",
            depth_below_shared_top_bounds_m=depth)


def test_inverse_saved_joint_radial_coordinate_reports_roof_exterior_without_new_mask_authority():
    inverse = _api().inverse_joint_spherical_material_depth_bounds(
        radius_m=F(10), top_radial_displacement_m=F(1), point_radius_bounds_m=I(F(9), F(12)))
    assert inverse == I(F(-397, 300), F(301, 150))
    for proof in _proofs():
        for _, roof in proof.all_same_owner_roofs:
            outside = _api().inverse_joint_spherical_material_depth_bounds(
                radius_m=_source().radius_m,
                top_radial_displacement_m=roof.lo-_source().radius_m,
                point_radius_bounds_m=proof.ray_endpoint_radius_bounds_m)
            assert outside.hi < 0


def test_roof_certificate_retains_world_placement_and_complete_receiving_normal_frame():
    for proof in _proofs():
        assert len(proof.reference_frame_unit_normal_bounds) == 3
        assert len(proof.activation_world_ray_endpoint_bounds_m) == 3
        assert all(type(bound) is I for bound in proof.activation_world_ray_endpoint_bounds_m)
        assert all(type(bound) is I for bound in proof.reference_frame_unit_normal_bounds)
        assert _api().validate_native_normal_support_certificate(proof, source=_source()) == proof
