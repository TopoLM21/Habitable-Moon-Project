"""Independent controls for the declared same-material reduced source subsystem.

The COMMON body modes are present in the extended chart but do not drive this
source component.  Its original material labels, internal radial coordinates,
owner moments and separate global beam mechanism remain explicit.
"""
from copy import deepcopy
from dataclasses import fields, replace
from decimal import Decimal as D, localcontext
from fractions import Fraction as F
from functools import lru_cache
import json
import hashlib

import pytest

from tectonics.geometric_native_pose import FractionInterval as I


EPOCHS = ("prepared_state", "previous_state", "unaccepted_candidate_state")
STORE_FIELDS = (
    "contact_energy_j", "bulk_energy_j", "cumulative_external_work_j",
    "cumulative_friction_work_j", "cumulative_viscous_work_j",
    "cumulative_numerical_dissipation_j",
)


@lru_cache(maxsize=1)
def _api():
    from tectonics import geometric_native_common_material_transfer
    return geometric_native_common_material_transfer


@lru_cache(maxsize=1)
def _context():
    from analysis.geometric_native_common_material_transfer_validation.native_control import source_context
    return source_context()


@lru_cache(maxsize=1)
def _system():
    from analysis.geometric_native_common_material_transfer_validation.native_control import material_system
    return material_system()


@lru_cache(maxsize=1)
def _before():
    return json.loads(dict(_context()[3].complete_source_dto_json)["before"])


def _atoms():
    return _before()["original_physical_history"]["quadrature_histories"]


def _source_order():
    return _context()[4].coordinate_order


def _source_increments(moved=False):
    return tuple(F(1, 1000000) if moved and coordinate[0] == "radial" and coordinate[-1] == "sub"
                 else F(-1, 2000000) if moved and coordinate[0] == "radial"
                 else F(0) for coordinate in _source_order())


def _global_increments(moved=False):
    return tuple(F(1, 1000000) if moved and coordinate[0] == "source_internal" and coordinate[-1] == "sub"
                 else F(-1, 2000000) if moved and coordinate[0] == "source_internal"
                 else F(0) for coordinate in _system().coordinate_order)


@lru_cache(maxsize=6)
def _query_response(increments, branch):
    return _api().evaluate_native_common_material_transfer(_system(),
        coordinate_increments=increments, branch=branch, precision=96)


def _response(branch="relaxed_zero_strength_shear", moved=False):
    return _query_response(_global_increments(moved), branch)


def _beam_rows():
    before = _before()
    history = before["original_physical_history"]
    mechanics = history["accepted_before"]["initial_mechanics"]
    stiffness = history["original_reference"]["stiffness"]
    radius = _context()[1].radius_m
    tips = {row[0]: row[1:] for row in before["snapshot"]["transverse_displacement_m"]}
    rows = []
    for trace_index, trace in enumerate(mechanics["fault_traces"]["traces"]):
        for side_index, side in enumerate(("sub", "over")):
            index = 3*len(mechanics["plate_ids"])+2*trace_index+side_index
            assert all(not value for j, value in enumerate(stiffness[index]) if j != index)
            rows.append((("radial", trace["trace_id"], side),
                _literal(stiffness[index][index])/radius**2,
                _literal(tips[trace["trace_id"]][side_index])))
    return tuple(rows)


def _literal(value):
    assert type(value) is float
    return F.from_float(value)


def _vector(value):
    return tuple(_literal(x) for x in value)


def _dot(a, b):
    return sum((x*y for x, y in zip(a, b)), a[0]*b[0]*0)


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2],
            a[0]*b[1]-a[1]*b[0])


def _scale(value, vector):
    return tuple(value*x for x in vector)


def _add(a, b):
    return tuple(x+y for x, y in zip(a, b))


def _sym_rotation(a, b, vector):
    return _scale(F(1, 2), _add(_cross(a, _cross(b, vector)),
                              _cross(b, _cross(a, vector))))


def _axis(name):
    return tuple(F(int(name == value)) for value in "xyz")


def _scalar_oracle(atom, radius, order, increments):
    """Reduced h = rho_over*(e0.q) - (R_over*e0).(R_sub*p_sub).

    This cancellation precedes differentiation.  It is independent of the
    production product-rule jet and keeps literal nonunit source vectors.
    """
    point, material = atom["original_point"], atom["original_material"]
    q = _vector(point["position"])
    owners = material["subducting_plate"], material["overriding_plate"]
    radial = {side: increments[order.index(("radial", atom["trace_id"], side))]
              for side in ("sub", "over")}
    p_sub = _scale(radius+radial["sub"], q)
    size = len(order)
    values, first, second = [], [], []
    for director_name in ("normal", "down_dip"):
        e = _vector(point[director_name])
        values.append((radial["over"]-radial["sub"])*_dot(q, e))
        B = [F(0)]*size
        H = [[F(0)]*size for _ in range(size)]
        for i, coordinate in enumerate(order):
            if coordinate[0] == "owner":
                lever = _cross(p_sub, e)["xyz".index(coordinate[2])]
                B[i] = -lever if coordinate[1] == owners[0] else lever
            elif coordinate[1] == atom["trace_id"]:
                B[i] = (-1 if coordinate[2] == "sub" else 1)*_dot(q, e)
            for j in range(i, size):
                other = order[j]
                result = F(0)
                if coordinate[0] == other[0] == "owner":
                    a, b = _axis(coordinate[2]), _axis(other[2])
                    if coordinate[1] == other[1] == owners[0]:
                        result = -_dot(e, _sym_rotation(a, b, p_sub))
                    elif coordinate[1] == other[1] == owners[1]:
                        result = -_dot(_sym_rotation(a, b, e), p_sub)
                    else:
                        sub_axis, over_axis = (a, b) if coordinate[1] == owners[0] else (b, a)
                        result = -_dot(_cross(over_axis, e), _cross(sub_axis, p_sub))
                elif {coordinate[0], other[0]} == {"owner", "radial"}:
                    angular, radial_coordinate = ((coordinate, other)
                        if coordinate[0] == "owner" else (other, coordinate))
                    if radial_coordinate[1:] == (atom["trace_id"], "sub"):
                        a = _axis(angular[2])
                        result = (-_dot(e, _cross(a, q)) if angular[1] == owners[0]
                                  else -_dot(_cross(a, e), q))
                H[i][j] = H[j][i] = result
        first.append(tuple(B))
        second.append(tuple(tuple(row) for row in H))
    return tuple(values), tuple(first), tuple(second)


def _series_oracle(atom, branch):
    material, state = atom["original_material"], atom["unaccepted_candidate_state"]
    c, law = material["columns"], material["law"]
    stiffness = tuple(1/(1/_literal(law[key])+sum(
        (_literal(c[side+"_length_m"])/_literal(c[side+"_"+modulus])
         for side in ("sub", "over")), F(0)))
        for key, modulus in (("normal_stiffness_pa_m", "normal_modulus_pa"),
                             ("tangential_stiffness_pa_m", "shear_modulus_pa")))
    if branch == "relaxed_zero_strength_shear":
        stiffness = stiffness[0], F(0)
    traction = tuple(_literal(state[name]) for name in
                     ("traction_normal_pa", "traction_shear_pa"))
    energy = sum((_literal(state[name]) for name in ("contact_energy_j", "bulk_energy_j")), F(0))
    return _literal(atom["original_area_m2"]), traction, stiffness, energy


def _potential_oracle(atoms, radius, order, increments, branch, beam_rows):
    size = len(order)
    energy, gradient = F(0), [F(0)]*size
    hessian = [[F(0)]*size for _ in range(size)]
    for atom in atoms:
        eta, B, H = _scalar_oracle(atom, radius, order, increments)
        area, t0, stiffness, energy0 = _series_oracle(atom, branch)
        traction = tuple(t+k*x for t, k, x in zip(t0, stiffness, eta))
        energy += energy0+area*sum((t*x+k*x*x/2
            for t, k, x in zip(t0, stiffness, eta)), F(0))
        for i in range(size):
            gradient[i] += area*sum((t*row[i] for t, row in zip(traction, B)), F(0))
            for j in range(size):
                hessian[i][j] += area*sum((k*row[i]*row[j]+t*matrix[i][j]
                    for k, row, t, matrix in zip(stiffness, B, traction, H)), F(0))
    energy += _literal(dict(_before()["original_physical_history"]
                            ["unaccepted_candidate_work_ledgers"])["bending_energy_increment_j"])
    for coordinate, stiffness, reference in beam_rows:
        i = order.index(coordinate)
        total = reference+increments[i]
        energy += stiffness*(reference*increments[i]+increments[i]*increments[i]/2)
        gradient[i] += stiffness*total
        hessian[i][i] += stiffness
    return energy, tuple(gradient), tuple(tuple(row) for row in hessian)


def _decimal(value):
    value = F(value)
    return D(value.numerator)/D(value.denominator)


def _decimal_rotation(vector, rotation_vector):
    """Independent Cartesian exponential series, no project pose helper."""
    result, term = vector, vector
    for n in range(1, 181):
        term = _scale(D(1)/D(n), _cross(rotation_vector, term))
        result = _add(result, term)
    return result


def _decimal_material_point(q, history, radius, roof, common_rate, depth=F(0)):
    with localcontext() as context:
        context.prec = 130
        ray = tuple(_decimal(value) for value in q)
        ray = _scale(D(1)/_dot(ray, ray).sqrt(), ray)
        for increment in history:
            dt = _literal(increment["dt_myr"])
            rotation = tuple(_decimal(_literal(value)*dt)
                             for value in increment["omega_rad_per_myr"])
            ray = _decimal_rotation(ray, rotation)
        cube = _decimal((radius+roof)**3-3*radius**2*depth)
        r = _decimal(radius+roof)
        for _ in range(20):
            r = (2*r+cube/(r*r))/3
        tau = (_decimal(radius)**3-r**3)/(3*_decimal(radius)**2)
        rotation = tuple(_decimal(value)*tau for value in common_rate)
        return _scale(r, _decimal_rotation(ray, rotation))


def _encloses(interval, value):
    assert type(interval) is I
    assert interval.lo <= value <= interval.hi


def _encloses_decimal(interval, value):
    with localcontext() as context:
        context.prec = 130
        # Decimal roundoff at this budget is far below the certified physical
        # enclosures.  The exp-series tail for ||rotation|| < 1/4 is < 1e-400.
        numerical_guard = D("1e-100")
        assert _decimal(interval.lo)-numerical_guard <= value <= _decimal(interval.hi)+numerical_guard


def _reseal(raw, kind):
    body = {field.name: raw[field.name] for field in fields(kind)
            if field.name != "binding_sha256"}
    raw["binding_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    return raw


def _shift_interval_one(raw):
    for endpoint in ("lo", "hi"):
        raw[endpoint][0] += raw[endpoint][1]


@lru_cache(maxsize=4)
def _expected_potential(branch="relaxed_zero_strength_shear", moved=False):
    return _potential_oracle(_atoms(), _context()[1].radius_m, _source_order(),
        _source_increments(moved), branch, _beam_rows())


def test_one_global_injection_extends_common_with_eight_real_internal_channels():
    system = _system()
    source_order = _source_order()
    assert len(system.coordinate_order) == 100
    assert system.coordinate_order[:6] == source_order[:6]
    assert all(coordinate[0] == "body" for coordinate in system.coordinate_order[6:92])
    assert system.coordinate_order[92:] == tuple(("source_internal", trace, side)
        for _, trace, side in source_order[6:])
    assert system.source_support_indices == tuple(range(6))+tuple(range(92, 100))
    assert len(system.source_to_global_injection) == 100
    for i, row in enumerate(system.source_to_global_injection):
        assert row == tuple(F(int(i == target)) for target in system.source_support_indices)
    response = _response()
    gradient = _api().expand_native_common_material_transfer_gradient(response)
    hessian = _api().expand_native_common_material_transfer_hessian(response)
    assert len(gradient) == len(hessian) == 100
    assert response.global_zero_coordinate_indices == tuple(range(6, 92))
    for i in range(6, 92):
        assert gradient[i] == I(0)
        assert hessian[i] == (I(0),)*100
        assert all(row[i] == I(0) for row in hessian)
    for i, target_i in enumerate(system.source_support_indices):
        assert gradient[target_i] == response.potential_gradient_bounds[i]
        for j, target_j in enumerate(system.source_support_indices):
            assert hessian[target_i][target_j] == response.potential_hessian_bounds[i][j]


def test_one_reference_potential_keeps_serial_stores_and_distinct_literal_beam_ledger():
    response, system = _response(), _system()
    energy, gradient, hessian = _expected_potential()
    _encloses(response.potential_j, energy)
    for actual, expected in zip(response.potential_gradient_bounds, gradient):
        _encloses(actual, expected)
    for actual_row, expected_row in zip(response.potential_hessian_bounds, hessian):
        for actual, expected in zip(actual_row, expected_row):
            _encloses(actual, expected)
    mathematical_beam = sum((k*w*w/2 for _, k, w in _beam_rows()), F(0))
    literal_beam = _literal(dict(_before()["original_physical_history"]
        ["unaccepted_candidate_work_ledgers"])["bending_energy_increment_j"])
    assert system.beam_reference_energy_j == literal_beam
    assert system.beam_mathematical_reference_energy_j == mathematical_beam
    assert system.beam_energy_rounding_residual_j == literal_beam-mathematical_beam
    assert system.beam_energy_rounding_residual_j != 0
    serial = sum((_literal(atom["unaccepted_candidate_state"][name])
        for atom in _atoms() for name in ("contact_energy_j", "bulk_energy_j")), F(0))
    assert energy == serial+literal_beam
    assert response.source_pullback_gradient_bounds == response.potential_gradient_bounds
    assert response.source_pullback_hessian_bounds == response.potential_hessian_bounds
    expected_beams = {coordinate[1:]: (k, w) for coordinate, k, w in _beam_rows()}
    assert len(system.beams) == len(expected_beams) == 8
    stiffness = _before()["original_physical_history"]["original_reference"]["stiffness"]
    for beam in system.beams:
        k, w = expected_beams[beam.trace_id, beam.side]
        assert (beam.stiffness_exact_n_m, beam.reference_displacement_m,
            beam.reference_energy_exact_j, beam.reference_force_exact_n) == (k, w, k*w*w/2, k*w)
        old_k = stiffness[beam.original_global_stiffness_index][beam.original_global_stiffness_index]
        k_binary = old_k/(float(_context()[1].radius_m)*float(_context()[1].radius_m))
        force_binary, energy_binary = k_binary*float(w), k_binary*float(w)*float(w)/2
        assert beam.stiffness_binary64_n_m == _literal(k_binary)
        assert beam.literal_reduced_beam_binary_force_evaluation_n == _literal(force_binary)
        assert beam.literal_reduced_beam_binary_energy_evaluation_j == _literal(energy_binary)
        assert beam.binary_force_evaluation_minus_exact_n == _literal(force_binary)-k*w
        assert beam.binary_energy_evaluation_minus_exact_j == _literal(energy_binary)-k*w*w/2
        assert beam.archived_old_binary_force_replayed is False


def test_all_eight_finite_internal_increments_have_one_exact_path_work_and_hessian():
    for branch in ("relaxed_zero_strength_shear", "frozen_elastic_predictor"):
        response = _response(branch, True)
        energy, gradient, hessian = _expected_potential(branch, True)
        _encloses(response.potential_j, energy)
        assert response.source_coordinate_increments == _source_increments(True)
        for actual, expected in zip(response.potential_gradient_bounds, gradient):
            _encloses(actual, expected)
        for actual_row, expected_row in zip(response.potential_hessian_bounds, hessian):
            for actual, expected in zip(actual_row, expected_row):
                _encloses(actual, expected)
        energy0, gradient0, _ = _expected_potential(branch)
        delta = _source_increments(True)
        trapezoid_work = sum(((g0+g1)*x/2 for g0, g1, x in zip(gradient0, gradient, delta)), F(0))
        assert energy-energy0 == trapezoid_work
        _encloses(response.potential_j-_response(branch).potential_j, trapezoid_work)


def test_fresh_rigid_owner_rotation_preserves_energy_and_angular_virtual_work():
    for moved in (False, True):
        response = _response(moved=moved)
        _, exact_gradient, exact_hessian = _expected_potential(moved=moved)
        for axis in "xyz":
            indices = [i for i, c in enumerate(_source_order()) if c[0] == "owner" and c[2] == axis]
            assert len(indices) == 2
            assert sum((exact_gradient[i] for i in indices), F(0)) == 0
            _encloses(sum((response.potential_gradient_bounds[i] for i in indices), I(0)), F(0))
            for other_axis in "xyz":
                other = [i for i, c in enumerate(_source_order()) if c[0] == "owner" and c[2] == other_axis]
                assert sum((exact_hessian[i][j] for i in indices for j in other), F(0)) == 0
                _encloses(sum((response.potential_hessian_bounds[i][j]
                    for i in indices for j in other), I(0)), F(0))


def test_all_thirty_two_attachments_keep_true_original_material_masks_and_cohorts():
    system = _system()
    assert len(system.attachments) == 32
    assert len({(a.point_id, a.side) for a in system.attachments}) == 32
    assert (system.archived_donor_attachment_count, system.live_donor_attachment_count,
        system.live_receiver_attachment_count) == (8, 8, 16)
    material = _before()["snapshot"]["current_material_epoch"]["source_material_state"]
    assert json.loads(system.complete_original_material_state_json) == material
    allocations = {a["fragment_id"]: a for a in material["parent_allocations"]}
    parents = {a["fragment_id"]: a for a in material["journal"]["source_surface"]["fragments"]}
    cohorts = {a["original_parent_id"]: a for a in material["archive_cohorts"]}
    assert set(cohorts) == {"octant:5", "octant:6", "octant:7"}
    atoms = {a["point_id"]: a for a in _atoms()}
    for attachment in system.attachments:
        atom = atoms[attachment.point_id]
        q = _vector(atom["original_point"]["position"])
        allocation, parent = allocations[attachment.original_parent_id], parents[attachment.original_parent_id]
        archived = attachment.side == "sub" and attachment.original_parent_id in ("octant:6", "octant:7")
        assert attachment.original_direction_literal == q
        assert attachment.original_material_id == allocation["material_id"] == parent["parcel"]["material_id"]
        assert attachment.owner == allocation["owner"] == parent["parcel"]["plate"]
        assert json.loads(attachment.complete_parent_allocation_json) == allocation
        assert json.loads(attachment.complete_original_parent_json) == parent
        assert attachment.original_label_membership == (
            "ARCHIVED_ORIGINAL_MATERIAL" if archived else "LIVE_ORIGINAL_MATERIAL")
        assert attachment.original_material_membership_authenticated is True
        assert attachment.current_live_cold_mask_membership_claimed is False
        assert attachment.archived_three_dimensional_body_motion_certified is False
        for observed, plane in zip(attachment.original_parent_support_predicate_bounds,
                                   parent["geometry"]["supports"]):
            assert observed == I(_dot(tuple(F(value) for value in plane["normal"]), q))
            assert observed.lo >= 0
        if archived:
            assert attachment.archive_cohort_id == cohorts[attachment.original_parent_id]["cohort_id"]
            assert json.loads(attachment.complete_archive_cohort_json) == cohorts[attachment.original_parent_id]
            assert attachment.whole_clock_receiver_predicate_divided_by_u_bounds.lo > 0
        else:
            assert attachment.archive_cohort_id is None
            assert attachment.complete_archive_cohort_json is None
            if attachment.side == "sub":
                assert attachment.whole_clock_receiver_predicate_divided_by_u_bounds.hi < 0
        assert attachment.live_domain_id == allocation["live_domain_id"]
        assert attachment.archive_domain_id == allocation["archive_domain_id"]
    # This parent has a real archive, but its four old source atoms are live.
    donor_five = [a for a in system.attachments if a.side == "sub" and a.original_parent_id == "octant:5"]
    assert len(donor_five) == 4 and all(not a.archived for a in donor_five)


def test_authentic_current_phi_uses_homogeneous_ray_and_full_depth_without_changing_source_q():
    system, reference = _system(), _context()[1]
    histories = dict(_before()["snapshot"]["owner_histories"])
    banks = {b.original_parent_id: b for b in reference.banks if b.material_kind == "original_live"}
    nonunit_literal_seen = False
    for attachment in system.attachments:
        q = attachment.original_direction_literal
        nonunit_literal_seen |= _dot(q, q) != 1
        bank = banks[attachment.original_parent_id]
        assert attachment.material_depth_bounds_m == I(0)
        assert attachment.areal_material_thickness_m == bank.areal_material_thickness_m
        assert attachment.roof_radial_displacement_m == bank.roof_radial_displacement_m
        expected = _decimal_material_point(q, histories[attachment.owner], reference.radius_m,
            bank.roof_radial_displacement_m, reference.angular_rate_per_physical_volume_depth_m)
        for actual, value in zip(attachment.genuine_current_attachment_world_m, expected):
            _encloses_decimal(actual, value)
        with localcontext() as context:
            context.prec = 130
            norm = _dot(tuple(_decimal(x) for x in q), tuple(_decimal(x) for x in q)).sqrt()
            for actual, value in zip(attachment.normalized_material_ray_bounds, q):
                _encloses_decimal(actual, _decimal(value)/norm)
    assert nonunit_literal_seen
    representatives = [next(a for a in system.attachments if a.archived),
                       next(a for a in system.attachments if a.side == "over")]
    for attachment in representatives:
        depth = attachment.areal_material_thickness_m/2
        point = _api().evaluate_native_common_material_point(system, attachment.point_id, attachment.side,
            material_depth_bounds_m=I(depth), precision=96)
        assert point.original_direction_literal == attachment.original_direction_literal
        assert point.original_material_membership_authenticated is True
        assert point.existing_common_live_mask_membership_authenticated is False
        assert point.physical_material_admission_authorized is False
        assert point.determinant_exact == 1
        assert point.signed_material_chart_jacobian_exact == -reference.radius_m**2
        expected = _decimal_material_point(attachment.original_direction_literal, histories[attachment.owner],
            reference.radius_m, attachment.roof_radial_displacement_m,
            reference.angular_rate_per_physical_volume_depth_m, depth)
        for actual, value in zip(point.world_position_bounds_m, expected):
            _encloses_decimal(actual, value)
        raw = _api().native_common_material_point_to_dict(point)
        assert _api().native_common_material_point_to_dict(
            _api().native_common_material_point_from_dict(raw, system=system)) == raw
        forged = deepcopy(raw)
        forged["original_material_id"] = "invented-hot-successor"
        _reseal(forged, type(point))
        with pytest.raises(ValueError):
            _api().native_common_material_point_from_dict(forged, system=system)


def test_archival_driver_rebase_preserves_three_epochs_six_stores_and_distinct_clocks():
    system, carrier_system = _system(), _context()[4]
    assert (system.source_geometry_elapsed_time_s, system.material_geometry_elapsed_time_s,
        system.source_memory_elapsed_time_s, system.source_contact_elapsed_time_s,
        system.source_contact_step_count) == (F(0), F(1000000), F(1000000), F(2000000), 2)
    assert system.complete_current_source_states_json == carrier_system.complete_current_source_states_json
    assert system.complete_current_source_memory_json == carrier_system.complete_current_source_memory_json
    assert json.loads(system.complete_before_work_ledgers_json) == _before()["original_physical_history"]["before_work_ledgers"]
    assert json.loads(system.complete_unaccepted_candidate_work_ledgers_json) == _before()["original_physical_history"]["unaccepted_candidate_work_ledgers"]
    assert len(set(system.original_source_work_lineages)) == 16
    attachments = {(a.point_id, a.side): a for a in system.attachments}
    carriers = {a.point_id: a for a in carrier_system.carriers}
    current = dict(_before()["snapshot"]["states"])
    count = 0
    for response in _response().atom_responses:
        carrier = carriers[response.point_id]
        atom = json.loads(carrier.complete_original_atom_history_json)
        assert response.original_source_state_json == carrier.source_local_state_json
        assert json.loads(response.original_source_state_json) == atom["unaccepted_candidate_state"] == current[response.point_id]
        assert response.scalar_increment_bounds_m == (I(0), I(0))
        assert response.scalar_free_anchor_bounds_m == tuple(I(_literal(atom["unaccepted_candidate_state"][name]))
            for name in ("gap_free_m", "jump_free_m"))
        for epoch in EPOCHS:
            for field in STORE_FIELDS:
                assert type(atom[epoch][field]) is float
                count += 1
        q = _vector(atom["original_point"]["position"])
        for side, position in (("sub", response.sub_position_bounds_m), ("over", response.over_position_bounds_m)):
            attachment = attachments[response.point_id, side]
            for actual, value in zip(position, _scale(_context()[1].radius_m, q)):
                _encloses(actual, value)
            assert attachment.source_tip_reference_bounds_m == position
        assert response.archival_literal_reaction_replay_certified is True
        assert json.loads(response.archival_literal_world_reactions_json) == json.loads(carrier.archival_literal_world_reactions_json)
        assert response.archival_mathematical_minus_literal_reaction_bounds == carrier.archival_mathematical_minus_literal_reaction_bounds
    assert count == 288


def test_all_scalar_first_second_variations_and_decimal_rotating_receiver_oracle():
    atoms = {a["point_id"]: a for a in _atoms()}
    for moved in (False, True):
        for response in _response(moved=moved).atom_responses:
            expected_eta, B, H = _scalar_oracle(atoms[response.point_id], _context()[1].radius_m,
                _source_order(), _source_increments(moved))
            for actual, expected in zip(response.scalar_increment_bounds_m, expected_eta):
                _encloses(actual, expected)
            for actual_row, expected_row in zip(response.scalar_first_derivative_bounds, B):
                for actual, expected in zip(actual_row, expected_row):
                    _encloses(actual, expected)
            for actual_matrix, expected_matrix in zip(response.scalar_second_derivative_bounds, H):
                for actual_row, expected_row in zip(actual_matrix, expected_matrix):
                    for actual, expected in zip(actual_row, expected_row):
                        _encloses(actual, expected)
    atom = _atoms()[0]
    _, _, H = _scalar_oracle(atom, _context()[1].radius_m, _source_order(), _source_increments(True))
    sub, over = atom["original_material"]["subducting_plate"], atom["original_material"]["overriding_plate"]
    pairs = ((("owner", sub, "y"), ("owner", sub, "z")),
             (("owner", over, "z"), ("owner", over, "z")),
             (("owner", sub, "y"), ("owner", over, "z")))
    with localcontext() as context:
        context.prec = 130
        step = D("1e-24")
        q = tuple(_decimal(x) for x in _vector(atom["original_point"]["position"]))
        ds = _source_increments(True)[_source_order().index(("radial", atom["trace_id"], "sub"))]
        ps = _scale(_decimal(_context()[1].radius_m+ds), q)
        for scalar, name in enumerate(("normal", "down_dip")):
            director = tuple(_decimal(x) for x in _vector(atom["original_point"][name]))
            for left, right in pairs:
                samples = []
                for sign_left, sign_right in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    rotations = {owner: [D(0)]*3 for owner in (sub, over)}
                    rotations[left[1]]["xyz".index(left[2])] += sign_left*step
                    rotations[right[1]]["xyz".index(right[2])] += sign_right*step
                    # The rho_over e0.q term is owner invariant and cancels.
                    samples.append(-_dot(_decimal_rotation(director, tuple(rotations[over])),
                                         _decimal_rotation(ps, tuple(rotations[sub]))))
                finite = (samples[0]-samples[1]-samples[2]+samples[3])/(4*step*step)
                expected = _decimal(H[scalar][_source_order().index(left)][_source_order().index(right)])
                truncation_bound = 50*_decimal(_context()[1].radius_m+abs(ds))*step*step+D("1e-70")
                assert abs(finite-expected) <= truncation_bound


def test_authentic_attachment_forces_levers_receiver_couples_and_virtual_work_cancel():
    atoms = {a["point_id"]: a for a in _atoms()}
    beta = (F(1, 7), F(-2, 9), F(3, 11), F(-1, 13), F(2, 17), F(-3, 19))
    for moved in (False, True):
        for response in _response(moved=moved).atom_responses:
            atom = atoms[response.point_id]
            eta, B, _ = _scalar_oracle(atom, _context()[1].radius_m, _source_order(), _source_increments(moved))
            area, t0, K, _ = _series_oracle(atom, "relaxed_zero_strength_shear")
            traction = tuple(t+k*x for t, k, x in zip(t0, K, eta))
            directors = [_vector(atom["original_point"][name]) for name in ("normal", "down_dip")]
            force = tuple(area*sum((t*e[i] for t, e in zip(traction, directors)), F(0)) for i in range(3))
            q = _vector(atom["original_point"]["position"])
            ds = _source_increments(moved)[_source_order().index(("radial", atom["trace_id"], "sub"))]
            do = _source_increments(moved)[_source_order().index(("radial", atom["trace_id"], "over"))]
            ps, po = _scale(_context()[1].radius_m+ds, q), _scale(_context()[1].radius_m+do, q)
            torque = _cross(ps, force)
            receiver = _cross(tuple(x-y for x, y in zip(po, ps)), force)
            world = dict(response.world_reactions)
            for actual, expected in zip(world["force_on_sub_n"], force):
                _encloses(actual, expected)
            for actual, expected in zip(response.complete_attachment_torque_on_sub_nm, torque):
                _encloses(actual, expected)
            for actual, expected in zip(response.complete_attachment_torque_on_over_nm, _scale(F(-1), torque)):
                _encloses(actual, expected)
            for actual, expected in zip(response.receiver_carrier_couple_on_over_nm, receiver):
                _encloses(actual, expected)
            for side in response.attachment_torque_closure_residual_bounds_nm:
                for value in side:
                    _encloses(value, F(0))
            indices = {owner: tuple(_source_order().index(("owner", owner, name)) for name in "xyz")
                       for owner in (atom["original_material"]["subducting_plate"], atom["original_material"]["overriding_plate"])}
            sub, over = atom["original_material"]["subducting_plate"], atom["original_material"]["overriding_plate"]
            beta_sub, beta_over = tuple(beta[i] for i in indices[sub]), tuple(beta[i] for i in indices[over])
            velocity_s = _add(_cross(beta_sub, response.sub_attachment_position_bounds_m), _scale(F(2, 23), response.sub_direction_bounds))
            velocity_o = _add(_cross(beta_over, response.over_attachment_position_bounds_m), _scale(F(-3, 29), response.over_direction_bounds))
            work = (_dot(world["force_on_sub_n"], velocity_s)+_dot(world["force_on_over_n"], velocity_o)
                +_dot(response.lever_couple_on_sub_nm, beta_sub)
                +_dot(_add(response.lever_couple_on_over_nm, response.receiver_carrier_couple_on_over_nm), beta_over))
            virtual = list(beta)+[F(0)]*8
            virtual[_source_order().index(("radial", atom["trace_id"], "sub"))] = F(2, 23)
            virtual[_source_order().index(("radial", atom["trace_id"], "over"))] = F(-3, 29)
            expected_work = -area*sum((t*_dot(row, virtual) for t, row in zip(traction, B)), F(0))
            _encloses(work, expected_work)


def test_inactive_preloaded_atoms_keep_real_restoring_force_and_variable_energy():
    initial = {r.point_id: r for r in _response().atom_responses}
    inactive = [r for r in _response(moved=True).atom_responses if r.source_accounting_bucket.startswith("inactive:")]
    assert len(inactive) == 8
    for response in inactive:
        old = initial[response.point_id]
        assert old.traction_bounds_pa[0].hi < 0
        assert response.traction_bounds_pa[0] != old.traction_bounds_pa[0]
        assert (response.potential_bounds_j-old.potential_bounds_j).lo > 0
        i = _source_order().index(("radial", response.trace_id, "sub"))
        assert response.potential_hessian_bounds[i][i].lo > 0
    assert all(row.hessian_n_m > 0 for row in _response().beam_responses)


def test_relaxed_shear_and_frozen_predictor_preserve_memory_and_work_without_adoption():
    relaxed, frozen = _response(moved=True), _response("frozen_elastic_predictor", True)
    expected_extra = F(0)
    atoms = {a["point_id"]: a for a in _atoms()}
    for left, right in zip(relaxed.atom_responses, frozen.atom_responses):
        atom = atoms[left.point_id]
        eta, _, _ = _scalar_oracle(atom, _context()[1].radius_m, _source_order(), _source_increments(True))
        area, _, K, _ = _series_oracle(atom, "frozen_elastic_predictor")
        assert left.point_id == right.point_id
        assert left.original_source_state_json == right.original_source_state_json
        assert left.source_work_lineage_id == right.source_work_lineage_id
        assert left.effective_tangent_stiffness_pa_m[1] == 0
        assert right.effective_tangent_stiffness_pa_m[1] == K[1] > 0
        assert left.traction_bounds_pa[1] == I(0)
        _encloses(right.traction_bounds_pa[1], K[1]*eta[1])
        _encloses(left.proposed_plastic_increment_bounds_m, eta[1])
        assert right.proposed_plastic_increment_bounds_m == I(0)
        expected_extra += area*K[1]*eta[1]*eta[1]/2
    assert expected_extra > 0
    _encloses(frozen.potential_j-relaxed.potential_j, expected_extra)
    for response in (relaxed, frozen):
        assert response.source_state_or_work_advanced is False
        assert response.complete_unaccepted_candidate_work_ledgers_json == _system().complete_unaccepted_candidate_work_ledgers_json
        assert response.original_source_work_lineages == _system().original_source_work_lineages


def test_scope_and_admission_reject_duplicate_energy_or_unproved_coordinate_drivers():
    system, api = _system(), _api()
    raw = api.native_common_material_transfer_to_dict(system)
    assert raw["scope"] == "COMMON_ORIGINAL_MATERIAL_EXTENDED_REDUCED_SOURCE_COMPONENT_ONLY"
    for name in ("physical_source_material_handover_certified", "authentic_archived_three_dimensional_body_motion_certified",
                 "source_internal_links_reduced_into_common_body_modes", "common_body_modal_coupling_certified",
                 "new_mixed_force_equilibrium_certified", "whole_clock_all_pair_admission_authorized",
                 "physical_material_admission_authorized", "source_history_advanced", "original_point_1_completed"):
        assert raw[name] is False
    assert system.reduced_coordinate_map_second_derivative_exact_zero is True
    assert system.common_body_source_coupling_exact_zero is True
    for kwargs in (dict(common_calibrated_beam_reference_energized=True),
                   dict(continuous_source_measure_energized=True),
                   dict(common_calibrated_beam_reference_energized=1)):
        with pytest.raises(ValueError):
            api.validate_native_common_material_transfer_energy_assembly(system, **kwargs)
    with pytest.raises(ValueError):
        api.evaluate_native_common_material_transfer(system, common_calibrated_beam_energy_active=True)
    with pytest.raises(ValueError):
        api.bind_native_common_material_transfer(_context()[4], reference=_context()[1],
            force_basis="current_objective", provenance="Implicit current-basis replacement", precision=96)
    for values in ((F(0),)*92, (0.,)*100, [F(0)]*100):
        with pytest.raises(ValueError):
            api.evaluate_native_common_material_transfer(system, coordinate_increments=values)
    for index in (0, 6, 91):
        values = list(_global_increments())
        values[index] = F(1, 1000000)
        with pytest.raises(ValueError):
            api.evaluate_native_common_material_transfer(system, coordinate_increments=tuple(values))
    values = list(_global_increments())
    values[92:] = (-_context()[1].radius_m,)*8
    with pytest.raises(ValueError):
        api.evaluate_native_common_material_transfer(system, coordinate_increments=tuple(values))
    with pytest.raises(ValueError):
        api.evaluate_native_common_material_transfer(system, branch="undeclared-hot-law")


def test_system_typed_replay_rejects_resealed_material_budget_beam_and_embedding_changes():
    system, api = _system(), _api()
    raw = api.native_common_material_transfer_to_dict(system)
    assert _reseal(deepcopy(raw), type(system))["binding_sha256"] == system.binding_sha256
    replay = api.native_common_material_transfer_from_dict(raw, carrier_system=_context()[4], reference=_context()[1])
    assert api.native_common_material_transfer_to_dict(replay) == raw
    changed = []
    archive = next(i for i, a in enumerate(system.attachments) if a.archived)
    value = deepcopy(raw)
    value["attachments"][archive]["original_material_id"] = "invented-hot-successor"
    value["attachments"][archive]["archive_cohort_id"] = None
    changed.append(value)
    value = deepcopy(raw)
    value["attachments"][1] = deepcopy(value["attachments"][0])
    changed.append(value)
    value = deepcopy(raw)
    value["source_to_global_injection"][92][6] = [0, 1]
    changed.append(value)
    value = deepcopy(raw)
    value["beams"][0]["reference_force_exact_n"][0] += value["beams"][0]["reference_force_exact_n"][1]
    changed.append(value)
    value = deepcopy(raw)
    material = json.loads(value["complete_original_material_state_json"])
    material["parent_allocations"][0]["material_id"] = "substituted-reservoir"
    value["complete_original_material_state_json"] = json.dumps(material, sort_keys=True, separators=(",", ":"))
    changed.append(value)
    for forged in changed:
        _reseal(forged, type(system))
        with pytest.raises(ValueError):
            api.native_common_material_transfer_from_dict(forged, carrier_system=_context()[4], reference=_context()[1])
    forged = replace(system, beam_energy_rounding_residual_j=system.beam_energy_rounding_residual_j+1)
    resealed = _reseal(api._payload(forged), type(system))["binding_sha256"]
    with pytest.raises(ValueError):
        api.validate_native_common_material_transfer_system(replace(forged, binding_sha256=resealed))
    with pytest.raises(ValueError):
        api.validate_native_common_material_transfer_system(system,
            carrier_system=replace(_context()[4], provenance="Forged source with the same saved hash"))
    with pytest.raises(ValueError):
        api.validate_native_common_material_transfer_system(system,
            reference=replace(_context()[1], provenance="Forged reference with the same saved hash"))
    carrier_system, reference = _context()[4], _context()[1]
    original_carriers = carrier_system.carriers
    altered = replace(original_carriers[0], original_area_m2=original_carriers[0].original_area_m2+1)
    try:
        object.__setattr__(carrier_system, "carriers", (altered,)+original_carriers[1:])
        with pytest.raises(ValueError):
            api.evaluate_native_common_material_transfer(system)
    finally:
        object.__setattr__(carrier_system, "carriers", original_carriers)
    original_banks = reference.banks
    attachment = system.attachments[0]
    index = next(i for i, b in enumerate(original_banks) if b.component_id == attachment.geometry_field_component_id)
    altered_bank = replace(original_banks[index], roof_radial_displacement_m=original_banks[index].roof_radial_displacement_m+1)
    try:
        object.__setattr__(reference, "banks", original_banks[:index]+(altered_bank,)+original_banks[index+1:])
        with pytest.raises(ValueError):
            api.evaluate_native_common_material_point(system, attachment.point_id, attachment.side,
                material_depth_bounds_m=I(0), precision=96)
    finally:
        object.__setattr__(reference, "banks", original_banks)
    # The authenticated immutable dependency values are restored for all later
    # queries, including cached ones; no source DTO or work history is adopted.
    assert api.validate_native_common_material_transfer_system(system) is system


def test_response_full_replay_and_controller_rows_reject_valid_query_transplants():
    api, system = _api(), _system()
    response = _response(moved=True)
    raw = api.native_common_material_transfer_response_to_dict(response)
    assert _reseal(deepcopy(raw), type(response))["binding_sha256"] == response.binding_sha256
    replay = api.native_common_material_transfer_response_from_dict(raw, system=system)
    assert api.native_common_material_transfer_response_to_dict(replay) == raw
    with pytest.raises(ValueError):
        api.validate_native_common_material_transfer_response(response,
            system=replace(system, provenance="Forged response context with the same saved hash"))
    for path in (("potential_gradient_bounds", 6), ("source_pullback_gradient_bounds", 6)):
        forged = deepcopy(raw)
        _shift_interval_one(forged[path[0]][path[1]])
        _reseal(forged, type(response))
        with pytest.raises(ValueError):
            api.native_common_material_transfer_response_from_dict(forged, system=system)
    forged = deepcopy(raw)
    forged["atom_responses"][0]["original_source_state_json"] = "{}"
    _reseal(forged, type(response))
    with pytest.raises(ValueError):
        api.native_common_material_transfer_response_from_dict(forged, system=system)
    for field, index in (("scalar_second_derivative_bounds", (0, 0, 3)),
                         ("receiver_carrier_couple_on_over_nm", (0,))):
        forged = deepcopy(raw)
        interval = forged["atom_responses"][0][field]
        for i in index:
            interval = interval[i]
        _shift_interval_one(interval)
        _reseal(forged, type(response))
        with pytest.raises(ValueError):
            api.native_common_material_transfer_response_from_dict(forged, system=system)
    from analysis.geometric_native_common_material_transfer_validation import native_control as controller
    queries = []
    for arguments in controller.query_cases(system):
        candidate = _query_response(arguments["coordinate_increments"], arguments["branch"])
        queries.append(dict(arguments=controller.payload(arguments),
            response=api.native_common_material_transfer_response_to_dict(candidate)))
    points = []
    for arguments in controller.material_query_cases(system):
        point = api.evaluate_native_common_material_point(system, **arguments)
        points.append(dict(arguments=controller.payload(arguments),
            response=api.native_common_material_point_to_dict(point)))
    packet = dict(format="native-common-original-material-reduced-source-packet-1",
        prior_867_packet_raw_sha256=controller.PRIOR_PACKET_RAW,
        prior_867_wrapper_raw_sha256=controller.PRIOR_WRAPPER_RAW,
        material_transfer=api.native_common_material_transfer_to_dict(system),
        global_queries=queries, material_point_queries=points, **controller.summary(6))
    packet["packet_sha256"] = controller.digest(packet)
    assert controller.replay_packet(packet) == controller.summary(6)
    for left, right in ((0, 1), (0, 2)):
        forged = deepcopy(packet)
        forged["global_queries"][left]["response"], forged["global_queries"][right]["response"] = (
            forged["global_queries"][right]["response"], forged["global_queries"][left]["response"])
        forged["packet_sha256"] = controller.digest({k: v for k, v in forged.items() if k != "packet_sha256"})
        with pytest.raises(ValueError, match="query"):
            controller.replay_packet(forged)
    for left, right in ((0, 1), (0, 2)):
        forged = deepcopy(packet)
        forged["material_point_queries"][left]["response"], forged["material_point_queries"][right]["response"] = (
            forged["material_point_queries"][right]["response"], forged["material_point_queries"][left]["response"])
        forged["packet_sha256"] = controller.digest({k: v for k, v in forged.items() if k != "packet_sha256"})
        with pytest.raises(ValueError, match="query"):
            controller.replay_packet(forged)
