"""Source-complete mechanical carrier controls, with independent SO(3) algebra.

The archival frozen solver basis and CURRENT objective source basis are
different physical references.  Neither is an authenticated COMMON attachment.
The Decimal oracle composes Rodrigues rotations from source literals and
derives the reduced objective driver separately from production jets.
"""
from copy import deepcopy
from dataclasses import fields, replace
from decimal import Decimal as D, localcontext
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json

import pytest

from tectonics.geometric_native_pose import FractionInterval as I
from tectonics import geometric_native_common_source_carrier as api
from tests.test_geometric_native_common_source_allocation import _allocation, _cuts, _reference, _before


PROVENANCE = "Explicit literal-atom objective source carrier; no COMMON attachment"
STORE_FIELDS = (
    "contact_energy_j", "bulk_energy_j", "cumulative_external_work_j",
    "cumulative_friction_work_j", "cumulative_viscous_work_j",
    "cumulative_numerical_dissipation_j",
)
EPOCHS = ("prepared_state", "previous_state", "unaccepted_candidate_state")
ZERO = (F(0),) * 14


@lru_cache(maxsize=1)
def _system():
    return api.bind_native_common_source_carrier_system(_allocation(),
        reference=_reference(), cuts=_cuts(), provenance=PROVENANCE, precision=96)


def _atoms():
    return _before()["original_physical_history"]["quadrature_histories"]


def _atom(point_id):
    return next(row for row in _atoms() if row["point_id"] == point_id)


@lru_cache(maxsize=128)
def _response(point_id, basis="current_objective", branch="relaxed_zero_strength_shear", coordinates=ZERO):
    return api.evaluate_native_common_source_carrier(_system(), point_id,
        basis=basis, branch=branch, coordinate_increments=coordinates, precision=96)


def _fraction(value):
    return F.from_float(value)


def _exact_dot(a, b):
    return sum((x*y for x, y in zip(a, b)), F(0))


def _archival_exact_reactions(atom, radius):
    """Exact rational mathematical reactions from the unchanged source floats."""
    point, state = atom["original_point"], atom["unaccepted_candidate_state"]
    q = tuple(_fraction(value) for value in point["position"])
    directors = tuple(tuple(_fraction(value) for value in point[name])
                      for name in ("normal", "down_dip"))
    traction = tuple(_fraction(state[name]) for name in ("traction_normal_pa", "traction_shear_pa"))
    area = _fraction(atom["original_area_m2"])
    force = tuple(area*sum((t*e[i] for t, e in zip(traction, directors)), F(0)) for i in range(3))
    position = tuple(radius*value for value in q)
    torque = _cross(position, force)
    radial = _exact_dot(q, force)
    return dict(force_on_sub_n=force, force_on_over_n=tuple(-value for value in force),
        torque_on_sub_nm=torque, torque_on_over_nm=tuple(-value for value in torque),
        force_torque_on_over_nm=tuple(-value for value in torque),
        carrier_couple_on_over_nm=(F(0),)*3, radial_force_n=(radial, -radial))


def _old_augmented_operator(atom, radius, owners, trace_ids):
    """Replay the old binary horizontal-row construction without old solvers.

    The saved directions and five-term histories are used literally. No
    quadrature points, normals, fields or contact states are regenerated.
    This old construction differs from the exact literal-director two-jet.
    """
    point, material = atom["original_point"], atom["original_material"]
    lever = tuple(float(radius)*value for value in _cross(point["position"], point["horizontal"]))
    row = [0.]*14
    for owner, sign in ((material["subducting_plate"], -1), (material["overriding_plate"], 1)):
        start = 3*owners.index(owner)
        row[start:start+3] = tuple(sign*value for value in lever)
    sine, cosine = atom["original_frame"]
    operator = [[component*value for value in row] for component in (sine, cosine)]
    start = 6+2*trace_ids.index(atom["trace_id"])
    operator[0][start:start+2] = float(radius)*(-cosine), float(radius)*cosine
    operator[1][start:start+2] = float(radius)*sine, float(radius)*(-sine)
    return tuple(row), tuple(tuple(values) for values in operator)


def _archival_operator_budget(atom, radius, owners, trace_ids):
    """Separate source director rounding from old binary arithmetic errors."""
    point, material = atom["original_point"], atom["original_material"]
    q = tuple(_fraction(value) for value in point["position"])
    horizontal = tuple(_fraction(value) for value in point["horizontal"])
    sine, cosine = tuple(_fraction(value) for value in atom["original_frame"])
    cross_horizontal = _cross(q, horizontal)
    # These authenticated horizontal vectors are axis vectors. Thus their
    # cross products with the saved q require only exact signs and zeros;
    # subsequent rounding enters at R*lever and frame*row, two multiplies.
    assert sum(value != 0 for value in horizontal) == 1
    assert all(value in (F(-1), F(0), F(1)) for value in horizontal)
    assert tuple(_fraction(value) for value in _cross(point["position"], point["horizontal"])) == cross_horizontal
    unit_roundoff = F(1, 2**53)
    gamma1 = unit_roundoff/(1-unit_roundoff)
    gamma2 = 2*unit_roundoff/(1-2*unit_roundoff)
    start = 6+2*trace_ids.index(atom["trace_id"])
    mathematical, ideal, source_geometry, arithmetic_bound = [], [], [], []
    for name, horizontal_coefficient, radial_coefficient in (
            ("normal", sine, cosine), ("down_dip", cosine, -sine)):
        director = tuple(_fraction(value) for value in point[name])
        ideal_director = tuple(horizontal_coefficient*h+radial_coefficient*v for h, v in zip(horizontal, q))
        director_residual = tuple(actual-intended for actual, intended in zip(director, ideal_director))
        cross_director, cross_residual = _cross(q, director), _cross(q, director_residual)
        exact_row, ideal_row, geometry_row, bound_row = ([F(0)]*14 for _ in range(4))
        for owner, sign in ((material["subducting_plate"], -1), (material["overriding_plate"], 1)):
            owner_start = 3*owners.index(owner)
            for j in range(3):
                i = owner_start+j
                exact_row[i] = sign*radius*cross_director[j]
                ideal_row[i] = sign*radius*horizontal_coefficient*cross_horizontal[j]
                geometry_row[i] = sign*radius*cross_residual[j]
                bound_row[i] = gamma2*abs(ideal_row[i])
        # The old radial operator assumed q squared=1 and q.horizontal=0.
        # Retain both source literal departures in an exact signed receipt.
        radial_geometry = -horizontal_coefficient*_exact_dot(q, horizontal) \
            -radial_coefficient*(_exact_dot(q, q)-1)-_exact_dot(q, director_residual)
        exact_row[start:start+2] = -_exact_dot(q, director), _exact_dot(q, director)
        ideal_row[start:start+2] = -radial_coefficient, radial_coefficient
        geometry_row[start:start+2] = radial_geometry, -radial_geometry
        bound_row[start:start+2] = gamma1*abs(radial_coefficient), gamma1*abs(radial_coefficient)
        assert tuple(x-y for x, y in zip(exact_row, ideal_row)) == tuple(geometry_row)
        mathematical.append(tuple(exact_row)); ideal.append(tuple(ideal_row))
        source_geometry.append(tuple(geometry_row)); arithmetic_bound.append(tuple(bound_row))
    return tuple(mathematical), tuple(ideal), tuple(source_geometry), tuple(arithmetic_bound)


def _decimal(value):
    value = value if type(value) is F else _fraction(value)
    return D(value.numerator) / D(value.denominator)


def _dot(a, b):
    return sum((x*y for x, y in zip(a, b)), D(0))


def _cross(a, b):
    return a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]


def _scale(a, scale):
    return tuple(scale*x for x in a)


def _add(a, b):
    return tuple(x+y for x, y in zip(a, b))


def _subtract(a, b):
    return tuple(x-y for x, y in zip(a, b))


def _sym(a, b, vector):
    return _scale(_add(_cross(a, _cross(b, vector)), _cross(b, _cross(a, vector))), D("0.5"))


def _rotate(vector, angle):
    """Independent stable Rodrigues, without normalizing material vectors."""
    square = _dot(angle, angle)
    if not square:
        return vector
    sinc, cosc = D(1), D("0.5")
    sine_term, cosine_term = sinc, cosc
    for k in range(1, 180):
        sine_term *= -square / D((2*k)*(2*k+1))
        cosine_term *= -square / D((2*k+1)*(2*k+2))
        sinc += sine_term
        cosc += cosine_term
        if max(abs(sine_term), abs(cosine_term)) < D("1e-115"):
            break
    else:
        raise AssertionError("Independent source Rodrigues oracle did not converge")
    return _add(vector, _add(_scale(_cross(angle, vector), sinc),
                             _scale(_cross(angle, _cross(angle, vector)), cosc)))


def _poses(vector, owner, basis):
    if basis == "archival_old_reference":
        return vector
    histories = dict(_before()["snapshot"]["owner_histories"])[owner]
    for record in histories:
        angle = tuple(_decimal(value)*_decimal(record["dt_myr"])
                      for value in record["omega_rad_per_myr"])
        vector = _rotate(vector, angle)
    return vector


def _coordinate_order():
    owners = _before()["original_physical_history"]["original_reference"]["plate_ids"]
    traces = sorted({atom["trace_id"] for atom in _atoms()})
    return tuple(("owner", owner, label) for owner in owners for label in "xyz") + tuple(
        ("radial", trace_id, side) for trace_id in traces for side in ("sub", "over"))


def _radial_indices(atom):
    traces = sorted({row["trace_id"] for row in _atoms()})
    sub = 6+2*traces.index(atom["trace_id"])
    return sub, sub+1


def _radial_coordinates(atom, sub=F(0), over=F(0)):
    result = list(ZERO)
    i, j = _radial_indices(atom)
    result[i], result[j] = sub, over
    return tuple(result)


def _geometry(atom, basis, coordinates=ZERO):
    """Source literals only; no production geometry, jets or interval helpers."""
    point, material = atom["original_point"], atom["original_material"]
    sub, over = material["subducting_plate"], material["overriding_plate"]
    q = tuple(_decimal(value) for value in point["position"])
    qs, qo = _poses(q, sub, basis), _poses(q, over, basis)
    directions = tuple(_poses(tuple(_decimal(value) for value in point[name]), over, basis)
                       for name in ("normal", "down_dip"))
    radius = _decimal(_before()["original_physical_history"]["origin_surface"]["radius_km"])*1000
    radial_source = (_before()["original_physical_history"]["accepted_before"]["snapshot"]
                     if basis == "archival_old_reference" else _before()["snapshot"])
    w = next(row[1:] for row in radial_source["transverse_displacement_m"] if row[0] == atom["trace_id"])
    si, oi = _radial_indices(atom)
    ws, wo = _decimal(w[0])+_decimal(coordinates[si]), _decimal(w[1])+_decimal(coordinates[oi])
    xs, xo = _scale(qs, radius+ws), _scale(qo, radius+wo)
    geometry = tuple(_dot(e, _subtract(xo, xs)) for e in directions)
    return xs, xo, qs, qo, directions, geometry


def _analytic_driver(atom, basis, coordinates=ZERO):
    xs, xo, qs, qo, directions, geometry = _geometry(atom, basis, coordinates)
    material = atom["original_material"]
    sub, over = material["subducting_plate"], material["overriding_plate"]
    owner_ids = _before()["original_physical_history"]["original_reference"]["plate_ids"]
    si, oi = _radial_indices(atom)
    axes = tuple(tuple(D(int(i == j)) for i in range(3)) for j in range(3))
    gradients, hessians = [], []
    for e in directions:
        row, hessian = [D(0)]*14, [[D(0)]*14 for _ in range(14)]
        for owner in owner_ids:
            start = 3*owner_ids.index(owner)
            sign = -1 if owner == sub else 1 if owner == over else 0
            row[start:start+3] = _scale(_cross(xs, e), D(sign))
        row[si], row[oi] = -_dot(qs, e), _dot(qo, e)
        ss, oo = 3*owner_ids.index(sub), 3*owner_ids.index(over)
        for i, a in enumerate(axes):
            hessian[ss+i][si] = hessian[si][ss+i] = -_dot(e, _cross(a, qs))
            hessian[oo+i][si] = hessian[si][oo+i] = -_dot(_cross(a, e), qs)
            for j, b in enumerate(axes):
                hessian[ss+i][ss+j] = -_dot(e, _sym(a, b, xs))
                hessian[oo+i][oo+j] = -_dot(_sym(a, b, e), xs)
                mixed = -_dot(_cross(b, e), _cross(a, xs))
                hessian[ss+i][oo+j] = hessian[oo+j][ss+i] = mixed
        gradients.append(tuple(row))
        hessians.append(tuple(tuple(values) for values in hessian))
    return geometry, tuple(gradients), tuple(hessians), (xs, xo, qs, qo, directions)


def _enclosed(bounds, value, tolerance=F(0)):
    expected = F(value)
    # A 120-digit independent oracle may leave a tiny residual on an exact
    # singleton cancellation; it must not supply a physical error tolerance.
    numeric = F(1, 10**110)*max(F(1), abs(expected))
    assert bounds.lo-tolerance-numeric <= expected <= bounds.hi+tolerance+numeric


def _reseal(raw, value):
    body = {field.name: raw[field.name] for field in fields(value)
            if field.name != "binding_sha256"}
    raw["binding_sha256"] = hashlib.sha256(api._encoded(body)).hexdigest()
    return raw


def test_complete_source_system_has_every_literal_atom_and_one_mechanical_lineage():
    system = _system()
    assert api.validate_native_common_source_carrier_system(system,
        allocation=_allocation(), reference=_reference(), cuts=_cuts()) is system
    assert len(system.carriers) == 16
    assert tuple(carrier.point_id for carrier in system.carriers) == tuple(atom["point_id"] for atom in _atoms())
    assert len({carrier.point_id for carrier in system.carriers}) == 16
    assert sum(carrier.source_accounting_bucket.startswith("inactive:") for carrier in system.carriers) == 8
    assert sum(carrier.source_accounting_bucket.startswith("retained-contact:") for carrier in system.carriers) == 8
    assert system.coordinate_order == _coordinate_order()
    assert len({carrier.original_source_work_lineage_id for carrier in system.carriers}) == 16
    assert (system.archival_geometry_elapsed_time_s, system.current_geometry_elapsed_time_s,
        system.source_memory_elapsed_time_s, system.source_contact_elapsed_time_s) == (F(0), F(1000000), F(1000000), F(2000000))
    assert (system.source_memory_stage_count, system.source_contact_step_count) == (1, 2)
    assert system.source_current_node_epoch == "unaccepted_candidate"
    raw = api.native_common_source_carrier_system_to_dict(system)
    for flag in ("source_carriers_attached_to_authentic_common_material",
                 "old_rod_three_dimensional_material_map_certified", "continuous_C_dtheta_measure_energized",
                 "current_objective_basis_equals_archival_force_basis", "new_mixed_force_equilibrium_certified",
                 "whole_clock_all_pair_admission_certified", "physical_material_admission_authorized",
                 "source_state_or_thermal_state_advanced", "new_native_interval_accepted", "original_point_1_completed"):
        assert raw[flag] is False


def test_every_original_state_energy_and_work_store_survives_all_three_epochs():
    for carrier, atom in zip(_system().carriers, _atoms()):
        assert carrier.original_area_m2 == _fraction(atom["original_area_m2"])
        # This examines retained immutable payloads rather than rebuilding an
        # equivalent approximate state from a continuous memory polynomial.
        retained = json.loads(carrier.complete_original_atom_history_json)
        assert retained == atom
        assert json.loads(carrier.source_local_state_json) == atom["unaccepted_candidate_state"]
        for epoch in EPOCHS:
            for key in STORE_FIELDS:
                assert _fraction(retained[epoch][key]) == _fraction(atom[epoch][key])
        response = _response(atom["point_id"])
        state = atom["unaccepted_candidate_state"]
        assert json.loads(response.source_local_state_json) == state
        assert response.stored_source_energy_components_j == (_fraction(state["contact_energy_j"]), _fraction(state["bulk_energy_j"]))
        assert response.source_energy_exact_j == sum(response.stored_source_energy_components_j, F(0))
    history = _before()["original_physical_history"]
    system = _system()
    assert json.loads(system.complete_before_work_ledgers_json) == history["before_work_ledgers"]
    assert json.loads(system.complete_unaccepted_candidate_work_ledgers_json) == history["unaccepted_candidate_work_ledgers"]
    assert dict(system.all16_energy_and_work_stores_j) == {
        key: sum((_fraction(atom["unaccepted_candidate_state"][key]) for atom in _atoms()), F(0))
        for key in STORE_FIELDS}
    assert system.complete_historical_saved_field_json == _allocation().complete_historical_saved_field_json
    assert json.loads(system.complete_current_source_memory_json) == _before()["snapshot"]["spatial_memory"]


def test_archival_frozen_basis_reproduces_all_sixteen_saved_world_reactions():
    history = _before()["original_physical_history"]
    assert history["accepted_before"]["snapshot"]["elapsed_time_s"] == 0.
    assert _before()["snapshot"]["elapsed_time_s"] == 1_000_000.
    radius = _fraction(history["origin_surface"]["radius_km"])*1000
    for carrier, atom, saved in zip(_system().carriers, _atoms(), history["actual_candidate"]["physical_points"]):
        response = _response(atom["point_id"], "archival_old_reference")
        reactions = dict(response.world_reactions)
        exact_reactions = _archival_exact_reactions(atom, radius)
        source_archive = json.dumps(saved, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False)
        assert carrier.archival_literal_world_reactions_json == source_archive
        assert response.archival_literal_world_reactions_json == source_archive
        assert carrier.archival_literal_reaction_replay_certified
        assert response.archival_literal_reaction_replay_certified
        signed_rounding_receipts = dict(carrier.archival_mathematical_minus_literal_reaction_bounds)
        assert set(reactions) == set(saved)
        assert set(signed_rounding_receipts) == set(saved)
        for key, original in saved.items():
            for actual, literal, exact, receipt in zip(reactions[key], original,
                    exact_reactions[key], signed_rounding_receipts[key]):
                # A rounded binary operation sequence can cancel two large
                # cross products to 0 or a few N m. Preserve its signed
                # discrepancy from exact arithmetic on the same literals.
                assert actual.lo <= exact <= actual.hi
                expected_residual = exact-_fraction(literal)
                assert receipt.lo <= expected_residual <= receipt.hi
                assert receipt == actual-I(_fraction(literal))
        assert all(value == I(0) for value in reactions["carrier_couple_on_over_nm"])


def test_old_operator_radial_scaling_is_converted_to_metres_not_owner_rates():
    history = _before()["original_physical_history"]
    radius = _fraction(history["origin_surface"]["radius_km"])*1000
    owners = tuple(history["original_reference"]["plate_ids"])
    trace_ids = tuple(sorted({atom["trace_id"] for atom in _atoms()}))
    original = history["original_reference"]
    for atom, old, old_horizontal_row in zip(_atoms(), original["operators"], original["rows"]):
        response = _response(atom["point_id"], "archival_old_reference")
        assert response.coordinate_order == _coordinate_order()
        binary_row, binary_operator = _old_augmented_operator(atom, radius, owners, trace_ids)
        assert tuple(value.hex() for value in binary_row) == tuple(value.hex() for value in old_horizontal_row)
        assert tuple(tuple(value.hex() for value in row) for row in binary_operator) == tuple(
            tuple(value.hex() for value in row) for row in old)
        mathematical, ideal, geometry_residual, arithmetic_bound = _archival_operator_budget(atom, radius, owners, trace_ids)
        for actual_row, binary_values, exact_values, ideal_values, source_residuals, bounds in zip(
                response.scalar_first_derivative_bounds, old, mathematical, ideal, geometry_residual, arithmetic_bound):
            for column, (actual, binary_value, exact, intended, source_residual, rounding_bound) in enumerate(zip(
                    actual_row, binary_values, exact_values, ideal_values, source_residuals, bounds)):
                # Convert the OLD w/R column to metres exactly. Owner columns
                # stay radians. The new operator must enclose the mathematical
                # driver of the literal directors, with no comparison tolerance.
                literal = _fraction(binary_value)/(radius if column >= 6 else 1)
                assert actual.lo <= exact <= actual.hi
                arithmetic_residual = literal-intended
                assert abs(arithmetic_residual) <= rounding_bound
                assert literal-exact == arithmetic_residual-source_residual
                # Keep the actual signed difference from the old bit pattern;
                # do not round a nearly cancelling owner component back to 0.
                difference = actual-I(literal)
                assert difference.lo <= exact-literal <= difference.hi
        # A stored rate times its duration may differ in the last binary bit
        # from the separately stored increment. The owner chart uses radians.
        assert response.coordinate_increments == ZERO
        assert atom["unaccepted_candidate_state"]["elapsed_time_s"] == 2_000_000.
    conversions = json.loads(_system().complete_owner_rate_duration_conversion_json)
    candidate = history["actual_candidate"]
    assert conversions["angular_variation_units"] == "radians"
    assert conversions["radial_variation_units"] == "metres"
    assert conversions["omega_rad_per_myr"] == candidate["omega_rad_per_myr"]
    assert conversions["angular_velocity_rad_s"] == candidate["angular_velocity_rad_s"]
    assert conversions["rotation_increment_radians"] == candidate["rotation_increment_radians"]
    for owner, rate_duration, velocity_duration, stored, residual in conversions["per_owner"]:
        rates = dict(candidate["omega_rad_per_myr"])[owner]
        velocity = candidate["angular_velocity_rad_s"][owner]
        assert tuple(F(*pair) for pair in rate_duration) == tuple(_fraction(value)*_fraction(history["literal_dt_myr"]) for value in rates)
        assert tuple(F(*pair) for pair in velocity_duration) == tuple(_fraction(value)*_fraction(history["literal_dt_s"]) for value in velocity)
        assert tuple(F(*pair) for pair in stored) == tuple(_fraction(value) for value in candidate["rotation_increment_radians"][owner])
        assert tuple(F(*pair) for pair in residual) == tuple(F(*a)-F(*b) for a, b in zip(rate_duration, velocity_duration))


def test_current_owner_and_radial_jacobians_match_independent_objective_driver():
    with localcontext() as context:
        context.prec = 120
        for atom in _atoms():
            response = _response(atom["point_id"])
            geometry, gradients, _, (xs, xo, qs, qo, directions) = _analytic_driver(atom, "current_objective")
            for field, expected in ((response.sub_position_bounds_m, xs), (response.over_position_bounds_m, xo),
                                    (response.sub_direction_bounds, qs), (response.over_direction_bounds, qo)):
                for bounds, value in zip(field, expected):
                    _enclosed(bounds, value)
            for actual, expected in zip(response.receiver_director_bounds, directions):
                for bounds, value in zip(actual, expected):
                    _enclosed(bounds, value)
            for bounds, value in zip(response.projected_geometry_bounds_m, geometry):
                _enclosed(bounds, value)
            for actual, expected in zip(response.scalar_first_derivative_bounds, gradients):
                for bounds, value in zip(actual, expected):
                    _enclosed(bounds, value)


def test_complete_geometric_hessians_include_rotating_receiver_and_radial_cross_terms():
    with localcontext() as context:
        context.prec = 120
        for atom in _atoms():
            coordinates = _radial_coordinates(atom, F(1, 1024), F(-1, 2048))
            response = _response(atom["point_id"], coordinates=coordinates)
            _, _, hessians, _ = _analytic_driver(atom, "current_objective", coordinates)
            for actual, expected in zip(response.scalar_second_derivative_bounds, hessians):
                assert len(actual) == 14 and all(len(row) == 14 for row in actual)
                for i, row in enumerate(actual):
                    for j, bounds in enumerate(row):
                        assert bounds == actual[j][i]
                        _enclosed(bounds, expected[i][j])
            _, over = _radial_indices(atom)
            for hessian in response.scalar_second_derivative_bounds:
                for value in hessian[over]:
                    _enclosed(value, D(0))


def test_decimal_finite_left_rotation_second_variations_match_the_mechanical_jet():
    atom = _atoms()[3]
    response = _response(atom["point_id"])
    owner_ids = _before()["original_physical_history"]["original_reference"]["plate_ids"]
    sub, over = atom["original_material"]["subducting_plate"], atom["original_material"]["overriding_plate"]
    sub_start, over_start = 3*owner_ids.index(sub), 3*owner_ids.index(over)
    with localcontext() as context:
        context.prec = 120
        xs, xo, _, _, directions, _ = _geometry(atom, "current_objective")
        step = D("1e-30")
        a, b = (D(1), D(0), D(0)), (D(0), D(0), D(1))
        for component, e in enumerate(directions):
            # Receiver-carried projection: the entire over-side term is
            # invariant under its rotation. This is independent of the
            # production full product-rule implementation.
            def h(s, o):
                return _dot(e, xo)-_dot(_rotate(e, _scale(b, o)), _rotate(xs, _scale(a, s)))
            first = (h(step, D(0))-h(-step, D(0)))/(2*step)
            second = (h(step, step)-h(step, -step)-h(-step, step)+h(-step, -step))/(4*step*step)
            tolerance = F(1, 10**48)
            _enclosed(response.scalar_first_derivative_bounds[component][sub_start], first, tolerance)
            _enclosed(response.scalar_second_derivative_bounds[component][sub_start][over_start+2], second, tolerance)


def test_potential_gradient_hessian_and_world_force_are_one_variational_response():
    with localcontext() as context:
        context.prec = 120
        for atom in _atoms():
            coordinates = _radial_coordinates(atom, F(1, 4096), F(-1, 8192))
            response = _response(atom["point_id"], branch="frozen_elastic_predictor", coordinates=coordinates)
            geometry, first, second, _ = _analytic_driver(atom, "current_objective", coordinates)
            base = _geometry(atom, "current_objective")[-1]
            area = _decimal(atom["original_area_m2"])
            initial = atom["unaccepted_candidate_state"]
            stiffness = tuple(_decimal(value) for value in response.effective_tangent_stiffness_pa_m)
            initial_traction = tuple(_decimal(initial[name]) for name in ("traction_normal_pa", "traction_shear_pa"))
            eta = tuple(now-old for now, old in zip(geometry, base))
            traction = tuple(t+k*increment for t, k, increment in zip(initial_traction, stiffness, eta))
            energy = sum((_decimal(initial[name]) for name in ("bulk_energy_j", "contact_energy_j")), D(0))
            energy += area*sum((t*increment+D("0.5")*k*increment*increment
                                for t, k, increment in zip(initial_traction, stiffness, eta)), D(0))
            _enclosed(response.potential_bounds_j, energy)
            for bounds, value in zip(response.traction_bounds_pa, traction):
                _enclosed(bounds, value)
            for i, bounds in enumerate(response.potential_gradient_bounds):
                _enclosed(bounds, area*sum((traction[a]*first[a][i] for a in range(2)), D(0)))
                for j, actual in enumerate(response.potential_hessian_bounds[i]):
                    expected = area*sum((stiffness[a]*first[a][i]*first[a][j]+traction[a]*second[a][i][j]
                                         for a in range(2)), D(0))
                    _enclosed(actual, expected)
                    assert actual == response.potential_hessian_bounds[j][i]


def test_receiver_couple_closes_world_torque_and_virtual_work_at_displaced_tips():
    atom = _atoms()[0]
    coordinates = _radial_coordinates(atom, F(1, 512), F(-1, 256))
    response = _response(atom["point_id"], coordinates=coordinates)
    with localcontext() as context:
        context.prec = 120
        xs, xo, qs, qo, directions, _ = _geometry(atom, "current_objective", coordinates)
        area = _decimal(atom["original_area_m2"])
        traction = tuple(_decimal(bounds.midpoint) for bounds in response.traction_bounds_pa)
        force = _scale(_add(_scale(directions[0], traction[0]), _scale(directions[1], traction[1])), area)
        ms, couple = _cross(xs, force), _cross(_subtract(xo, xs), force)
        reactions = dict(response.world_reactions)
        expected = dict(force_on_sub_n=force, force_on_over_n=_scale(force, D(-1)),
            torque_on_sub_nm=ms, torque_on_over_nm=_scale(ms, D(-1)),
            force_torque_on_over_nm=_cross(xo, _scale(force, D(-1))),
            carrier_couple_on_over_nm=couple,
            radial_force_n=(_dot(qs, force), -_dot(qo, force)))
        for key, values in expected.items():
            for bounds, value in zip(reactions[key], values):
                _enclosed(bounds, value, F(1, 10**75))
        assert max(abs(value) for value in couple) > D(1)
        # Forces applied to the actual separate tips alone leave this torque.
        assert max(abs(value) for value in _add(ms, expected["force_torque_on_over_nm"])) > D(1)
        for axis in range(3):
            _enclosed(response.potential_gradient_bounds[axis], -expected["torque_on_over_nm"][axis], F(1, 10**75))
            _enclosed(response.potential_gradient_bounds[3+axis], -expected["torque_on_sub_nm"][axis], F(1, 10**75))
        si, oi = _radial_indices(atom)
        _enclosed(response.potential_gradient_bounds[si], -expected["radial_force_n"][0], F(1, 10**75))
        _enclosed(response.potential_gradient_bounds[oi], -expected["radial_force_n"][1], F(1, 10**75))


def test_common_rigid_rotation_has_no_scalar_or_potential_virtual_work():
    with localcontext() as context:
        context.prec = 120
        common = (D("0.012"), D("-0.009"), D("0.003"))
        for atom in _atoms():
            response = _response(atom["point_id"])
            xs, xo, _, _, directions, geometry = _geometry(atom, "current_objective")
            for component, e in enumerate(directions):
                rotated = _dot(_rotate(e, common), _subtract(_rotate(xo, common), _rotate(xs, common)))
                assert abs(rotated-geometry[component]) < D("1e-108")
                first = response.scalar_first_derivative_bounds[component]
                hessian = response.scalar_second_derivative_bounds[component]
                for axis in range(3):
                    _enclosed(first[axis]+first[3+axis], D(0))
                    for other in range(3):
                        _enclosed(hessian[axis][other]+hessian[axis][3+other]
                            +hessian[3+axis][other]+hessian[3+axis][3+other], D(0))
            for axis in range(3):
                _enclosed(response.potential_gradient_bounds[axis]+response.potential_gradient_bounds[3+axis], D(0))


def test_stored_free_anchors_rebase_complete_configuration_without_paid_work():
    with localcontext() as context:
        context.prec = 120
        for atom in _atoms():
            state = atom["unaccepted_candidate_state"]
            for basis in ("archival_old_reference", "current_objective"):
                base = _response(atom["point_id"], basis)
                trial_coordinates = _radial_coordinates(atom, F(1, 16384), F(1, 32768))
                trial = _response(atom["point_id"], basis, coordinates=trial_coordinates)
                for a, key in enumerate(("gap_free_m", "jump_free_m")):
                    assert base.scalar_increment_bounds_m[a] == I(0)
                    assert base.scalar_free_anchor_bounds_m[a] == I(_fraction(state[key]))
                    _enclosed(trial.scalar_free_anchor_bounds_m[a],
                        _decimal(state[key])+_decimal(trial.scalar_increment_bounds_m[a].midpoint), F(1, 10**85))
                assert base.potential_bounds_j == I(base.source_energy_exact_j)
                assert json.loads(trial.source_local_state_json) == state
                # CURRENT physical projection is a geometry diagnostic, and
                # is different from the saved integrated scalar anchors.
                assert not (base.projected_geometry_bounds_m[0].lo <= _fraction(state["gap_free_m"])
                            <= base.projected_geometry_bounds_m[0].hi)


def test_inactive_source_carriers_retain_live_preload_and_restoring_response():
    inactive = tuple(carrier for carrier in _system().carriers if carrier.source_accounting_bucket.startswith("inactive:"))
    assert len(inactive) == 8
    for carrier in inactive:
        atom = _atom(carrier.point_id)
        base = _response(carrier.point_id)
        trial = _response(carrier.point_id, coordinates=_radial_coordinates(atom, F(1, 4096)))
        si, _ = _radial_indices(atom)
        assert base.potential_gradient_bounds[si].lo*base.potential_gradient_bounds[si].hi > 0
        assert trial.potential_bounds_j.hi < base.potential_bounds_j.lo or trial.potential_bounds_j.lo > base.potential_bounds_j.hi
        assert json.loads(trial.source_local_state_json) == atom["unaccepted_candidate_state"]


def test_relaxed_zero_strength_shear_and_frozen_predictor_have_distinct_tangents():
    atom = _atoms()[0]
    coordinates = _radial_coordinates(atom, F(1, 2048), F(-1, 4096))
    relaxed = _response(atom["point_id"], coordinates=coordinates)
    predictor = _response(atom["point_id"], branch="frozen_elastic_predictor", coordinates=coordinates)
    assert relaxed.traction_bounds_pa[1] == I(0)
    assert relaxed.effective_tangent_stiffness_pa_m[1] == F(0)
    assert predictor.effective_tangent_stiffness_pa_m[1] > 0
    assert predictor.traction_bounds_pa[1].lo*predictor.traction_bounds_pa[1].hi > 0
    assert relaxed.proposed_plastic_increment_bounds_m == relaxed.scalar_increment_bounds_m[1]
    assert predictor.proposed_plastic_increment_bounds_m == I(0)
    assert predictor.potential_bounds_j.lo > relaxed.potential_bounds_j.hi
    assert relaxed.source_local_state_json == predictor.source_local_state_json


def test_exact_serial_stiffness_keeps_binary64_source_rounding_visible():
    for atom in _atoms():
        response = _response(atom["point_id"])
        columns, law = atom["original_material"]["columns"], atom["original_material"]["law"]
        expected, binary = [], []
        for normal, contact_key in ((True, "normal_stiffness_pa_m"), (False, "tangential_stiffness_pa_m")):
            modulus = "normal_modulus_pa" if normal else "shear_modulus_pa"
            compliance = sum((_fraction(columns[side+"_length_m"])/_fraction(columns[side+"_"+modulus])
                              for side in ("sub", "over")), F(0))
            expected.append(1/(1/_fraction(law[contact_key])+compliance))
            column_k = 1./(columns["sub_length_m"]/columns["sub_"+modulus]
                           +columns["over_length_m"]/columns["over_"+modulus])
            # Match the old compressed-contact algorithm's stable TWO-stage
            # binary operation, rather than another algebraically equivalent
            # operation with potentially different last bits.
            binary.append(_fraction(law[contact_key]/(1.+law[contact_key]/column_k)))
        assert response.series_stiffness_exact_pa_m == tuple(expected)
        assert response.series_stiffness_binary64_pa_m == tuple(binary)
        assert any(exact != literal for exact, literal in zip(expected, binary))
        state = atom["unaccepted_candidate_state"]
        tractions = _fraction(state["traction_normal_pa"]), _fraction(state["traction_shear_pa"])
        assert response.traction_rounding_residual_pa == (
            tractions[0]-expected[0]*_fraction(state["gap_free_m"]),
            tractions[1]-expected[1]*(_fraction(state["jump_free_m"])-_fraction(state["plastic_slip_m"])))
        assert response.energy_rounding_residual_j == response.source_energy_exact_j-_fraction(atom["original_area_m2"])*sum(
            (t*t/(2*k) for t, k in zip(tractions, expected)), F(0))


def test_current_geometry_and_archival_world_force_bases_are_explicitly_distinct():
    atom = _atoms()[0]
    archival = _response(atom["point_id"], "archival_old_reference")
    current = _response(atom["point_id"])
    assert archival.basis == "archival_old_reference" and current.basis == "current_objective"
    assert archival.source_local_state_json == current.source_local_state_json
    assert archival.source_energy_exact_j == current.source_energy_exact_j
    assert archival.potential_bounds_j == current.potential_bounds_j
    old, now = dict(archival.world_reactions), dict(current.world_reactions)
    assert any(a.hi < b.lo or b.hi < a.lo for a, b in zip(old["force_on_sub_n"], now["force_on_sub_n"]))
    assert any(a.hi < b.lo or b.hi < a.lo for a, b in zip(archival.sub_position_bounds_m, current.sub_position_bounds_m))
    with localcontext() as context:
        context.prec = 120
        position = tuple(_decimal(value) for value in atom["original_point"]["position"])
        literal_square = sum((_fraction(value)**2 for value in atom["original_point"]["position"]), F(0))
        assert literal_square != 1
        for geometry in (archival, current):
            preserved_square = sum((bounds.midpoint*bounds.midpoint for bounds in geometry.sub_direction_bounds), F(0))
            assert abs(preserved_square-literal_square) < F(1, 10**85)
        assert abs(_dot(position, position)-D(1)) > D("1e-20")


def test_carrier_queries_reject_finite_owner_motion_and_unauthenticated_inputs():
    atom = _atoms()[0]
    for basis in ("unknown", "COMMON_cut_frame", ""):
        with pytest.raises(ValueError):
            api.evaluate_native_common_source_carrier(_system(), atom["point_id"], basis=basis)
    for branch in ("default_virgin_contact", "frozen_memory_predictor", ""):
        with pytest.raises(ValueError):
            api.evaluate_native_common_source_carrier(_system(), atom["point_id"], branch=branch)
    for coordinates in ([F(0)]*14, ZERO[:-1], (F(1, 10**400),)+ZERO[1:],
                        (0,)+ZERO[1:], (False,)+ZERO[1:]):
        with pytest.raises(ValueError):
            api.evaluate_native_common_source_carrier(_system(), atom["point_id"], coordinate_increments=coordinates)
    with pytest.raises(ValueError):
        api.evaluate_native_common_source_carrier(_system(), "new-HOT-material-point")
    unrelated = list(ZERO)
    unrelated[next(i for i, coordinate in enumerate(_coordinate_order())
        if coordinate[0] == "radial" and coordinate[1] != atom["trace_id"])] = F(1, 4096)
    with pytest.raises(ValueError):
        api.evaluate_native_common_source_carrier(_system(), atom["point_id"], coordinate_increments=tuple(unrelated))
    with pytest.raises(ValueError):
        api.evaluate_native_common_source_carrier(_system(), atom["point_id"], coordinate_increments=_radial_coordinates(atom, F(-100001)))
    with pytest.raises(ValueError):
        api.bind_native_common_source_carrier_system(_allocation(), reference=_reference(), cuts=_cuts(), provenance="")


def test_complete_source_carrier_system_roundtrip_replays_source_and_rejects_resealed_forgery():
    system = _system()
    raw = api.native_common_source_carrier_system_to_dict(system)
    rebuilt = api.native_common_source_carrier_system_from_dict(deepcopy(raw),
        allocation=_allocation(), reference=_reference(), cuts=_cuts())
    assert rebuilt == system
    changed = deepcopy(raw)
    changed["carriers"] = changed["carriers"][:-1]
    _reseal(changed, system)
    with pytest.raises(ValueError):
        api.native_common_source_carrier_system_from_dict(changed,
            allocation=_allocation(), reference=_reference(), cuts=_cuts())
    changed = deepcopy(raw)
    changed["carriers"] = changed["carriers"][:-1]+changed["carriers"][:1]
    _reseal(changed, system)
    with pytest.raises(ValueError):
        api.native_common_source_carrier_system_from_dict(changed,
            allocation=_allocation(), reference=_reference(), cuts=_cuts())
    with pytest.raises(ValueError):
        api.validate_native_common_source_carrier_system(replace(system, carriers=system.carriers[:-1]))


def test_full_mechanical_response_roundtrip_rejects_altered_force_jet_anchor_and_scope():
    atom = _atoms()[0]
    response = _response(atom["point_id"], branch="frozen_elastic_predictor",
        coordinates=_radial_coordinates(atom, F(1, 8192)))
    raw = api.native_common_source_carrier_response_to_dict(response)
    assert api.native_common_source_carrier_response_from_dict(deepcopy(raw), system=_system()) == response
    altered = [
        ("source_local_state_json", "{}"), ("basis", "archival_old_reference"),
        ("potential_gradient_bounds", []), ("scalar_second_derivative_bounds", []),
        ("scalar_free_anchor_bounds_m", []), ("world_reactions", []),
    ]
    for field, value in altered:
        changed = deepcopy(raw); changed[field] = value; _reseal(changed, response)
        with pytest.raises(ValueError):
            api.native_common_source_carrier_response_from_dict(changed, system=_system())
    changed = deepcopy(raw); changed["physical_material_admission_authorized"] = True
    with pytest.raises(ValueError):
        api.native_common_source_carrier_response_from_dict(changed, system=_system())
    with pytest.raises(ValueError):
        api.validate_native_common_source_carrier_response(replace(response, source_energy_exact_j=response.source_energy_exact_j+1), system=_system())
