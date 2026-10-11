"""Independent finite-face coverage and physical column-placement checks.

The geometry oracle below starts from independently declared rest lengths,
not the implementation's returned surface positions.  Paid elastic energy is
an endpoint energy; placing its columns must never pay it for a second time.
"""
from dataclasses import replace
from fractions import Fraction as F
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from tectonics.geometric_contact_mechanics import ElasticColumnParameters
from tectonics.genesis_contact_law import ContactLawParameters
from tectonics.geometric_native_pose import FractionInterval as I
from tectonics.geometric_solid_contact_closure import (
    compute_series_normal_closure, require_contact_zone_compaction,
)
from tectonics.geometric_native_solid_face_closure import (
    declare_native_full_face_first_birth_support, declare_physical_face_coverage,
    native_contact_face_coverage_from_packet, place_solid_contact_normal_segments,
    place_solid_contact_robin_segments, robin_column_material_point,
    require_complete_face_force_coverage,
    require_unloaded_face_separation,
)


def _closure(gap=F(-3, 7), *, zone=F(2)):
    columns = ElasticColumnParameters(1000., 1e9, 4e8,
        2000., 2e9, 8e8)
    law = ContactLawParameters(normal_stiffness_pa_m=1e6,
        tangential_stiffness_pa_m=1e6, cohesion_pa=0., friction_dry=0.,
        friction_wet=0., viscosity_pa_s_m=0.)
    normal = compute_series_normal_closure(columns=columns, law=law,
        free_gap_m=gap)
    return normal if zone is None else require_contact_zone_compaction(normal,
        contact_zone_rest_normal_thickness_m=zone)


def _placement(normal, *, owner="donor", origin=F(13, 5), **changes):
    # These are independently declared normal rest lengths, not radial layer
    # thicknesses.  Neither the pressure nor preload determines the zone.
    rest = (F.from_float(normal.columns.sub_length_m)
        +F.from_float(normal.columns.over_length_m)
        +normal.contact_zone_rest_normal_thickness_m)
    arguments = dict(normal=normal, donor_anchor_m=origin,
        receiver_anchor_m=origin+rest+normal.free_gap_m,
        contact_zone_owner=owner,
        geometry_source_id="Independent declared normal-column and contact-zone geometry")
    arguments.update(changes)
    return place_solid_contact_normal_segments(**arguments)


def _coverage(**changes):
    arguments = dict(donor_component_id="actual-donor-material",
        receiver_component_id="actual-receiver-material",
        source_epoch_sha256="1"*64, birth_support_sha256="2"*64,
        donor_thickness_m=F(2000), receiver_thickness_m=F(2000),
        force_contact_height_m=F(1000),
        inclined_measure_density_m2_per_radian=F(200_000_000),
        native_angle_bounds=I(F(1, 10**30), F(1, 10**30)+F(1, 10**100)),
        provenance="Exact source carrier; no force-height extension")
    arguments.update(changes)
    return declare_physical_face_coverage(**arguments)


@pytest.mark.parametrize("owner", ["donor", "receiver"])
def test_compressed_rest_columns_place_nonnegative_physical_void_without_repaying(owner):
    normal = _closure()
    result = _placement(normal, owner=owner)
    # Independent compliance is 1e-6 + 1e-6 + 1e-6 m/Pa.
    pressure = F(1_000_000, 7)
    shortening = F(1, 7)
    assert normal.normal_pressure_pa == pressure
    assert result.donor_column_end_m == F(13, 5)+1000-shortening
    assert result.receiver_column_end_m == result.receiver_anchor_m-2000+shortening
    assert result.physical_face_void_m == 0
    assert result.receiver_physical_face_m-result.donor_physical_face_m == 0
    assert result.contact_zone_end_m-result.contact_zone_start_m == 2-shortening
    independent_energy = 3*pressure*shortening/2
    assert result.total_energy_j_m2 == independent_energy == normal.total_energy_j_m2
    assert result.added_external_work_j_m2 == 0
    assert result.activation_paid_again is False
    assert result.anchor_balance_residual_m == 0
    assert result.material_admission_authorized is False


def test_contact_zone_owner_changes_declared_zone_side_and_preserves_columns_force_energy():
    normal = _closure()
    donor = _placement(normal, owner="donor")
    receiver = _placement(normal, owner="receiver")
    assert donor.contact_zone_start_m == donor.donor_column_end_m
    assert receiver.contact_zone_end_m == receiver.receiver_column_end_m
    assert donor.donor_column_end_m == receiver.donor_column_end_m
    assert donor.receiver_column_end_m == receiver.receiver_column_end_m
    assert donor.physical_face_void_m == receiver.physical_face_void_m == 0
    assert donor.total_energy_j_m2 == receiver.total_energy_j_m2
    assert donor.added_external_work_j_m2 == receiver.added_external_work_j_m2 == 0


@pytest.mark.parametrize("gap", [F(0), F(3, 11)])
def test_unloaded_and_open_geometry_leave_every_independent_rest_element_intact(gap):
    normal = _closure(gap)
    result = _placement(normal)
    assert result.donor_column_end_m-result.donor_anchor_m == 1000
    assert result.receiver_anchor_m-result.receiver_column_end_m == 2000
    assert result.contact_zone_end_m-result.contact_zone_start_m == 2
    assert result.physical_face_void_m == gap
    assert result.receiver_physical_face_m-result.donor_physical_face_m == gap
    assert result.total_energy_j_m2 == result.added_external_work_j_m2 == 0
    assert result.activation_paid_again is False


def test_uniform_world_translation_changes_all_surfaces_and_preserves_normal_physics():
    normal, shift = _closure(), F(10**40, 7)
    first, translated = _placement(normal), _placement(normal, origin=F(13, 5)+shift)
    for name in ("donor_anchor_m", "receiver_anchor_m", "donor_column_end_m",
            "receiver_column_end_m", "donor_physical_face_m", "receiver_physical_face_m",
            "contact_zone_start_m", "contact_zone_end_m"):
        assert getattr(translated, name)-getattr(first, name) == shift
    assert translated.total_energy_j_m2 == first.total_energy_j_m2
    assert translated.physical_face_void_m == first.physical_face_void_m


def test_independent_contact_zone_length_changes_geometry_but_never_pressure_or_energy():
    first, second = _placement(_closure(zone=F(2))), _placement(_closure(zone=F(5)))
    assert second.contact_zone_end_m-second.contact_zone_start_m == (
        first.contact_zone_end_m-first.contact_zone_start_m)+3
    assert second.total_energy_j_m2 == first.total_energy_j_m2
    assert second.added_external_work_j_m2 == first.added_external_work_j_m2 == 0


def test_negative_constitutive_contact_gap_is_not_a_negative_physical_void():
    normal, result = _closure(), _placement(_closure())
    assert normal.constitutive_contact_gap_m == -F(1, 7)
    assert result.physical_face_void_m == 0
    assert result.contact_zone_end_m > result.contact_zone_start_m


def test_anchor_geometry_must_match_each_existing_preloaded_rest_element_exactly():
    normal = _closure()
    rest = F(3002)+normal.free_gap_m
    with pytest.raises(ValueError):
        _placement(normal, receiver_anchor_m=F(13, 5)+rest+F(1, 10**80))
    with pytest.raises(ValueError):
        _placement(normal, receiver_anchor_m=F(13, 5)+3002)


def test_missing_independent_contact_zone_geometry_cannot_be_reconstructed_from_preload():
    normal = _closure(zone=None)
    with pytest.raises(ValueError):
        place_solid_contact_normal_segments(normal=normal, donor_anchor_m=F(0),
            receiver_anchor_m=F(3000)+normal.free_gap_m,
            contact_zone_owner="donor", geometry_source_id="No independently declared zone")


def test_forged_pressure_partition_is_not_a_physical_placement_authority():
    normal = _closure()
    with pytest.raises(ValueError):
        _placement(replace(normal, sub_column_shortening_m=normal.sub_column_shortening_m+1))


@pytest.mark.parametrize("changed", [dict(contact_zone_owner="split"),
    dict(geometry_source_id=""), dict(donor_anchor_m=0.),
    dict(receiver_anchor_m=True)])
def test_missing_or_aliased_normal_geometry_declarations_refuse(changed):
    with pytest.raises(ValueError):
        _placement(_closure(), **changed)


def test_actual_one_kilometre_carrier_retains_uncovered_lower_half_of_two_kilometre_face():
    coverage = _coverage()
    assert coverage.active_depth_bounds_m == I(0, 1000)
    assert coverage.uncovered_depth_parts_m == (I(1000, 2000),)
    assert coverage.whole_physical_face_force_coverage is False
    assert coverage.inclined_measure_density_m2_per_radian == 200_000_000
    assert coverage.activation_paid_again is False
    assert coverage.material_admission_authorized is False
    with pytest.raises(ValueError):
        require_complete_face_force_coverage(coverage)


def test_complete_carrier_has_no_unloaded_face_and_uses_minimum_actual_thickness():
    coverage = _coverage(donor_thickness_m=F(10_000), force_contact_height_m=F(2000))
    assert coverage.active_depth_bounds_m == I(0, 2000)
    assert coverage.uncovered_depth_parts_m == ()
    assert coverage.whole_physical_face_force_coverage is True
    assert require_complete_face_force_coverage(coverage) == 2000


def test_microscopic_positive_arc_is_not_discarded_or_converted_to_overlap_area():
    tiny = I(F(1, 10**800), F(1, 10**800)+F(1, 10**950))
    coverage = _coverage(native_angle_bounds=tiny)
    assert coverage.native_angle_bounds == tiny
    assert coverage.inclined_measure_density_m2_per_radian == 200_000_000
    assert coverage.active_depth_bounds_m == I(0, 1000)
    assert not coverage.whole_physical_face_force_coverage


@pytest.mark.parametrize("changed", [dict(source_epoch_sha256="bad"),
    dict(birth_support_sha256=""), dict(donor_component_id=""),
    dict(receiver_component_id="actual-donor-material"),
    dict(donor_thickness_m=2000.), dict(force_contact_height_m=F(0)),
    dict(native_angle_bounds=I(0, 1)), dict(provenance="")])
def test_missing_typed_carrier_geometry_or_source_identity_refuses(changed):
    with pytest.raises(ValueError):
        _coverage(**changed)


def _uncovered_witness(coverage, **changes):
    arguments = dict(angle_bounds_radians=I(0, coverage.native_angle_bounds.hi),
        depth_bounds_m=I(1000, 2000), time_fraction_bounds=I(0, 1),
        physical_face_void_bounds_m=I(0, F(1, 1000)),
        geometry_source_id="Independently proved complete retained face geometry")
    arguments.update(changes)
    return require_unloaded_face_separation(coverage, **arguments)


def test_uncovered_face_cannot_be_called_unloaded_until_entire_face_and_clock_are_separated():
    witness = _uncovered_witness(_coverage())
    assert witness.additional_contact_force == witness.additional_activation_energy_j == 0
    assert witness.material_admission_authorized is False
    for changes in (dict(time_fraction_bounds=I(1)), dict(depth_bounds_m=I(1000)),
            dict(angle_bounds_radians=I(0, F(1, 10**31))),
            dict(physical_face_void_bounds_m=I(-F(1, 10**100), 0))):
        with pytest.raises(ValueError):
            _uncovered_witness(_coverage(), **changes)


@pytest.mark.parametrize("adopted", [(object(),), [object()], False])
def test_full_face_first_birth_branch_refuses_any_adopted_history_before_source_replay(adopted):
    # The branch can only initialize fresh birth from the accepted old state.
    # No packet or typed history may be supplied to extend its loaded height.
    with pytest.raises(ValueError, match="adopted born history"):
        declare_native_full_face_first_birth_support(object(), object(),
            provenance="Explicit full-face first-birth trial", adopted_born_states=adopted)


def test_full_face_first_birth_constructor_refuses_untyped_checkpoint_or_support():
    with pytest.raises(ValueError, match="accepted original-only"):
        declare_native_full_face_first_birth_support({}, {},
            provenance="A hash-shaped dictionary is not an accepted source state")


TRUSTED_PACKET_BODY_SHA256 = "0dbdc4f041121197533aaf555417de3c420036c5be4933436cf4387af546d351"


@pytest.fixture(scope="module")
def genuine_unchanged_packet():
    # Reading the preserved authenticated DTO does not rebuild the expensive
    # source solve.  Its body seal is independently pinned from the previous
    # completed finite-force release, rather than trusted from the file alone.
    root = Path(__file__).resolve().parents[1]
    path = root/"analysis/geometric_native_finite_equilibrium_validation/release_validation_02/native_control/candidate_packet.json"
    packet = json.loads(path.read_text())
    assert packet["packet_sha256"] == TRUSTED_PACKET_BODY_SHA256
    return packet


def test_genuine_packet_face_coverage_uses_source_material_amounts_and_original_carrier(genuine_unchanged_packet):
    packet = genuine_unchanged_packet
    carriers = packet["birth_support"]["interface_carriers"]
    assert len(carriers) == 2
    for carrier in carriers:
        coverage = native_contact_face_coverage_from_packet(packet,
            carrier["declaration"]["family_id"],
            expected_packet_sha256=TRUSTED_PACKET_BODY_SHA256)
        assert coverage.donor_component_id == carrier["donor"]["component_id"]
        assert coverage.receiver_component_id == carrier["receiver"]["component_id"]
        assert coverage.receiver_thickness_m == 2000
        assert coverage.donor_thickness_m == (2000 if carrier["donor"]["material_kind"] == "hot_birth" else 10000)
        assert coverage.force_contact_height_m == 1000
        assert coverage.inclined_measure_density_m2_per_radian == F(*carrier["inclined_measure_density_m2_per_radian"])
        assert coverage.active_interface_area_bounds_m2.lo > 0
        assert coverage.uncovered_depth_parts_m == (I(1000, 2000),)
        assert coverage.added_activation_energy_j == 0
        assert coverage.material_admission_authorized is False


def test_self_consistent_modified_face_packet_cannot_replace_independent_genuine_source(genuine_unchanged_packet):
    packet = deepcopy(genuine_unchanged_packet)
    family = packet["birth_support"]["interface_carriers"][0]["declaration"]["family_id"]
    packet["birth_support"]["interface_carriers"][0]["declaration"]["vertical_contact_height_m"] = 2000.
    body = {name: value for name, value in packet.items() if name != "packet_sha256"}
    packet["packet_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    with pytest.raises(ValueError, match="independently trusted complete source"):
        native_contact_face_coverage_from_packet(packet, family,
            expected_packet_sha256=TRUSTED_PACKET_BODY_SHA256)


def test_genuine_packet_unknown_material_family_has_no_inferred_face(genuine_unchanged_packet):
    with pytest.raises(ValueError, match="unique requested"):
        native_contact_face_coverage_from_packet(genuine_unchanged_packet,
            "unknown-unrelated-material-pair", expected_packet_sha256=TRUSTED_PACKET_BODY_SHA256)


def _robin(normal=None, *, fraction=F(2, 5), origin=F(13, 5), **changes):
    normal = _closure(zone=None) if normal is None else normal
    arguments = dict(normal=normal, donor_anchor_m=origin,
        receiver_anchor_m=origin+F(3000)+normal.free_gap_m,
        fault_indentation_donor_fraction=fraction,
        geometry_source_id="Independent volume-compensated existing-bank Robin model")
    arguments.update(changes)
    return place_solid_contact_robin_segments(**arguments)


@pytest.mark.parametrize("fraction", [F(0), F(2, 5), F(1)])
def test_robin_boundary_mode_retains_independent_bulk_pressure_and_exact_compliant_partition(fraction):
    result = _robin(fraction=fraction)
    assert result.normal.normal_pressure_pa == F(1_000_000, 7)
    assert result.donor_bulk_column_end_m == F(13, 5)+1000-F(1, 7)
    assert result.receiver_bulk_column_end_m == result.receiver_anchor_m-2000+F(1, 7)
    assert result.donor_robin_boundary_shortening_m == fraction/7
    assert result.receiver_robin_boundary_shortening_m == (1-fraction)/7
    assert result.donor_physical_face_m == result.donor_bulk_column_end_m-fraction/7
    assert result.receiver_physical_face_m == result.receiver_bulk_column_end_m+(1-fraction)/7
    assert result.physical_face_void_m == 0
    assert result.donor_physical_face_m == result.receiver_physical_face_m
    assert result.total_energy_j_m2 == 3*F(1_000_000, 7)*F(1, 7)/2
    assert result.added_external_work_j_m2 == result.additional_material_volume_m3 == 0
    assert result.activation_paid_again is False
    assert result.material_admission_authorized is False
    assert result.whole_spherical_solid_embedding_certified is False


def _determinant(matrix):
    a, b, c = matrix
    return a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])


def test_robin_declared_normal_strike_compensation_preserves_full_local_volume_exactly():
    result = _robin()
    assert result.donor_normal_stretch == F(4999, 5000)
    assert result.receiver_normal_stretch == F(8749, 8750)
    for bank in ("donor", "receiver"):
        deformation = getattr(result, bank+"_local_deformation_gradient")
        assert _determinant(deformation) == getattr(result, bank+"_local_volume_jacobian") == 1
        assert deformation[1][1] > 1
    # Independently compute one entire material-box volume from its mapped
    # endpoint differences.  This does not certify adjoining spherical masks.
    donor_anchor = robin_column_material_point(result, bank="donor",
        normal_coordinate_m=F(0), strike_coordinate_m=F(0), depth_coordinate_m=F(0))
    donor_corner = robin_column_material_point(result, bank="donor",
        normal_coordinate_m=F(1000), strike_coordinate_m=F(7), depth_coordinate_m=F(2000))
    assert (donor_corner[0]-donor_anchor[0])*(donor_corner[1]-donor_anchor[1])*(donor_corner[2]-donor_anchor[2]) == 14_000_000


@pytest.mark.parametrize("bank,length", [("donor", F(1000)), ("receiver", F(2000))])
def test_robin_material_point_interior_and_faces_keep_bank_label_and_complete_column(bank, length):
    result = _robin()
    alpha = getattr(result, bank+"_normal_stretch")
    anchor = getattr(result, bank+"_anchor_m")
    sign = 1 if bank == "donor" else -1
    for coordinate in (F(0), F(1, 10**80), length/3, length):
        point = robin_column_material_point(result, bank=bank,
            normal_coordinate_m=coordinate, strike_coordinate_m=F(3, 11),
            depth_coordinate_m=F(17, 7))
        assert point == (anchor+sign*alpha*coordinate, F(3, 11)/alpha, F(17, 7))
    with pytest.raises(ValueError):
        robin_column_material_point(result, bank=bank,
            normal_coordinate_m=length+F(1, 10**80),
            strike_coordinate_m=F(0), depth_coordinate_m=F(0))


def test_open_robin_columns_retain_full_rest_geometry_and_only_physical_void():
    result = _robin(_closure(F(3, 11), zone=None))
    assert result.donor_normal_stretch == result.receiver_normal_stretch == 1
    assert result.physical_face_void_m == F(3, 11)
    assert result.donor_robin_boundary_shortening_m == result.receiver_robin_boundary_shortening_m == 0
    assert result.total_energy_j_m2 == result.added_external_work_j_m2 == 0


def test_robin_boundary_model_cannot_also_allocate_a_separate_fault_zone_material_layer():
    with pytest.raises(ValueError, match="separate contact-zone"):
        _robin(_closure())


def test_robin_full_column_collapse_refuses_even_when_bulk_series_columns_remain_positive():
    normal = _closure(F(-2500), zone=None)
    assert normal.deformed_sub_column_length_m > 0
    assert normal.deformed_over_column_length_m > 0
    with pytest.raises(ValueError, match="consumes an existing"):
        _robin(normal, fraction=F(1))


@pytest.mark.parametrize("fraction", [F(-1, 10), F(11, 10), 0., True])
def test_robin_indentation_partition_requires_literal_fraction_in_physical_range(fraction):
    with pytest.raises(ValueError):
        _robin(fraction=fraction)


def test_robin_forged_placement_and_boolean_pressure_partition_cannot_authorize_material_point():
    result = _robin()
    with pytest.raises(ValueError):
        robin_column_material_point(replace(result, donor_normal_stretch=F(1)),
            bank="donor", normal_coordinate_m=F(1),
            strike_coordinate_m=F(0), depth_coordinate_m=F(0))
    with pytest.raises(ValueError):
        _robin(replace(result.normal, physical_face_void_m=False))
