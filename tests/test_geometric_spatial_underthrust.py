"""Continuous branch/support controls independent of sampled Gauss stresses."""
from dataclasses import asdict, replace
import json
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from analysis.geometric_initiation_validation.admission_case import (
    DT as GEOMETRIC_DT, MOTION, birth, fixture, ledger,
)
from tests.test_geometric_distributed_contact_invariants import prepare, world
from tectonics.genesis_contact_law import ContactLawParameters
from tectonics.geometric_boundary import consume_geometric_transaction, initialize_boundary
from tectonics.geometric_contact_mechanics import ElasticColumnParameters, solve_compressed_contact
from tectonics.geometric_contacts import extract_contacts
from tectonics.geometric_distributed_contact_dynamics import initialize_distributed_contact_dynamics
from tectonics.geometric_fault_traces import (
    PhysicalFaultTrace, TraceQuadrature, declare_fault_trace_set, fault_trace_points,
)
from tectonics.geometric_polarity import PolarityEvidence, rotate_points
from tectonics.geometric_rigid_mantle_loading import PrescribedRigidMantleSource, assemble_physical_rigid_mantle_load
from tectonics.geometric_spatial_underthrust import (
    SpatialUnderthrustMemory, TrigPolynomial, _assess_interval, _close,
    _material_support_interval, _partial_velocity_bounds, _spatial_intersect_arcs,
    advance_spatial_underthrust_memory, initialize_spatial_underthrust_memory,
    make_spatial_underthrust_resolver, spatial_underthrust_memory_from_dict,
    spatial_underthrust_memory_to_dict, trig_polynomial_extrema, trig_polynomial_roots,
)
from tectonics.geometric_transport import UnresolvedPolarityError, advance_geometric_surface
from tectonics.thermal import SECONDS_PER_MYR
from tectonics.spherical_polygons import intersect_convex, polygon_area


def constant(value):
    return TrigPolynomial((value, 0., 0., 0., 0.))


def advanced_nodes(surface, seed, previous_states, anchor_fields, dt):
    """Use the production scalar return map, with angles recovered geometrically."""
    old = dict(previous_states)
    traces = {trace.trace_id: trace for trace in seed.fault_traces.traces}
    result = []
    for point in fault_trace_points(surface, seed.fault_traces):
        trace = traces[point.trace_id]
        start = np.asarray(trace.start)
        horizontal = np.asarray(trace.normal_sub_to_over)
        tangent = np.cross(horizontal, start)
        theta = math.atan2(float(tangent@np.asarray(point.position)), float(start@np.asarray(point.position)))
        gap, jump = anchor_fields[trace.trace_id]
        step = solve_compressed_contact(columns=trace.columns, law=trace.law, area_m2=point.area_m2,
            gap_free_m=gap(theta), jump_free_m=jump(theta), dt_s=dt, water=trace.water,
            previous=old[point.point_id])
        result.append((point.point_id, step.state))
    return tuple(result)


def prepared():
    mesh, surface, lid = world()
    _, _, seed = prepare(mesh, surface, lid, panels=1, order=4)
    return mesh, surface, seed


@pytest.mark.parametrize("field,expected", [
    ((0., 1., 0., 0., 0.), (math.pi/2.,)),
    ((0., 0., 0., 1., 0.), (math.pi/4., 3.*math.pi/4.)),
    ((0., 0., 0., 0., 1.), (math.pi/2.,)),
    ((-.25, 0., 0., 1., 0.), (.5*math.acos(.25), math.pi-.5*math.acos(.25))),
])
def test_quartic_roots_find_all_signed_degree_two_crossings(field, expected):
    roots = trig_polynomial_roots(TrigPolynomial(field), .001, math.pi-.001)
    np.testing.assert_allclose(roots, expected, rtol=0., atol=3e-14)


def test_degree_two_extrema_include_stationary_points_between_nodes():
    field = TrigPolynomial((.3, 0., 0., -.7, .2))
    low, high = trig_polynomial_extrema(field, .01, math.pi-.01)
    amplitude = math.hypot(.7, .2)
    assert low == pytest.approx(.3-amplitude, rel=2e-15)
    assert high == pytest.approx(.3+amplitude, rel=2e-15)


def test_tangent_double_root_is_retained_without_fake_sign_change():
    # (cos(theta)-.2)^2, represented exactly to degree two.
    field = TrigPolynomial((.54, -.4, 0., .5, 0.))
    roots = trig_polynomial_roots(field, 0., math.pi-.001)
    assert roots == pytest.approx((math.acos(.2),), abs=3e-8)
    low, _ = trig_polynomial_extrema(field, 0., math.pi-.001)
    assert abs(low) <= field.rounding_bound


def test_tiny_tilted_saved_endpoint_has_positive_shared_arc_without_plane_distance_floor():
    receiver = np.asarray([
        [-8.950365874871867e-7, .9999999999995994, 1.7584044141472642e-30],
        [-3.366688077358961e-27, 1.7553911051400742e-30, -1.],
    ])
    actual = np.asarray([
        [-8.950365874871867e-7, .9999999999995994, 1.7584044141472646e-30],
        [-3.366688077358961e-27, 1.7553911051400746e-30, -1.],
    ])
    assert _spatial_intersect_arcs(*receiver, *actual) is not None
    # A physically different meridian remains unproved even at a small offset.
    foreign = actual@Rotation.from_rotvec((0., 0., 1e-8)).as_matrix().T
    assert _spatial_intersect_arcs(*receiver, *foreign) is None


def test_spatial_arc_scope_requires_positive_length_and_is_coordinate_covariant():
    a, b, c = (np.asarray(point) for point in ((1., 0., 0.), (0., 1., 0.), (-1., 0., 0.)))
    assert _spatial_intersect_arcs(a, b, b, c) is None
    matrix = Rotation.from_rotvec((.21, -.17, .33)).as_matrix()
    assert _spatial_intersect_arcs(matrix@a, matrix@b, matrix@a, matrix@b) is not None


def test_tiny_closing_pocket_on_opening_parent_uses_actual_saved_world_points():
    _, _, seed = prepared()
    # The whole negative-y quarter opens under the negative z rotation.
    # An independently retained tiny y rotation closes its actual north-pole
    # endpoint pocket.  Absolute theta=pi/2 would lose this positive arc.
    trace = replace(seed.fault_traces.traces[0], start=(0., -1., 0.), end=(0., 0., 1.))
    first = np.asarray((0., -1e-20, 1.))
    last = np.asarray(trace.end)
    shared = _spatial_intersect_arcs(trace.start, trace.end, first, last)
    assert shared is not None
    omega = {trace.subducting_plate: (0., 0., 0.),
             trace.overriding_plate: (0., 1e-19, -1.)}
    minimum, maximum, bound = _partial_velocity_bounds(trace, omega, 100., *shared)
    assert minimum == pytest.approx(-1e-17, rel=2e-15)
    assert maximum == pytest.approx(-9e-18, rel=2e-15)
    assert maximum < -bound
    low, high = _material_support_interval(trace, *shared, math.pi/2.)
    assert low == np.nextafter(math.pi/2., -math.inf)
    assert high == math.pi/2.
    # Its true saved coordinates and positive length were never enlarged.
    np.testing.assert_array_equal(shared[0], first)
    np.testing.assert_array_equal(shared[1], last)


def test_partial_velocity_keeps_zero_endpoint_but_rejects_true_opening():
    _, _, seed = prepared()
    trace = replace(seed.fault_traces.traces[0], start=(0., -1., 0.), end=(0., 0., 1.))
    omega = {trace.subducting_plate: (0., 0., 0.),
             trace.overriding_plate: (0., 1e-19, -1.)}
    minimum, maximum, bound = _partial_velocity_bounds(trace, omega, 100.,
        np.asarray((0., -1e-19, 1.)), np.asarray(trace.end))
    assert minimum < -bound
    assert abs(maximum) <= bound
    _, opening, bound = _partial_velocity_bounds(trace, omega, 100.,
        np.asarray((0., -2e-19, 1.)), np.asarray(trace.end))
    assert opening > bound


@pytest.mark.parametrize("bad", [(True, 0., 0., 0., 0.), (0.,)*4, (float("nan"), 0., 0., 0., 0.)])
def test_polynomial_rejects_boolean_nonfinite_or_missing_coefficients(bad):
    with pytest.raises(ValueError):
        TrigPolynomial(bad)


def test_zero_shear_comparison_uses_the_actual_elastic_plastic_subtraction_operations():
    tiny = TrigPolynomial((0., 0., -1.5673258010963833e-32, 0., 0.))
    _close(0., tiny, .10906328583662661, "shear traction", operation_scale=3e4)
    # The same bound still rejects a materially different reaction.
    with pytest.raises(ValueError):
        _close(.1, tiny, .10906328583662661, "shear traction", operation_scale=3e4)


def test_constant_preparation_preserves_preexisting_local_plastic_memory():
    _, _, seed = prepared()
    shifted_traces = tuple(replace(trace, initial_jump_m=-100.) for trace in seed.fault_traces.traces)
    # Explicitly prepare a new declared experiment at those loaded anchors.
    locals_ = []
    by_trace = {trace.trace_id: trace for trace in shifted_traces}
    for identity, old in seed.states:
        trace = by_trace[identity.split(":panel:")[0]]
        step = solve_compressed_contact(columns=trace.columns, law=trace.law,
            area_m2=old.area_m2, gap_free_m=trace.initial_gap_m,
            jump_free_m=trace.initial_jump_m, dt_s=seed.initial_loading_dt_s, water=trace.water)
        locals_.append((identity, step.state))
    seed = replace(seed, states=tuple(locals_), fault_traces=replace(seed.fault_traces, traces=shifted_traces))
    memory = initialize_spatial_underthrust_memory(seed)
    assert all(piece.plastic_slip.coefficients[0] < 0.
               for trace in memory.traces for piece in trace.pieces)
    for trace in memory.traces:
        local = next(local for identity, local in seed.states if identity.startswith(trace.trace_id+":panel:"))
        assert trace.pieces[0].plastic_slip(.7) == local.plastic_slip_m
        assert trace.pieces[0].cumulative_slip(.7) == local.cumulative_slip_m


def test_two_changing_degree_two_fields_preserve_piecewise_signed_history_and_node_response():
    _, surface, seed = prepared()
    memory = initialize_spatial_underthrust_memory(seed)
    previous = seed.states
    fields = [
        (TrigPolynomial((-2., .1, -.15, .04, -.03)), TrigPolynomial((-10., 70., -55., 23., 12.))),
        (TrigPolynomial((-3., -.2, .1, -.03, .07)), TrigPolynomial((15., -80., 40., -21., 19.))),
    ]
    for gap, jump in fields:
        anchors = {trace.trace_id: (gap, jump) for trace in seed.fault_traces.traces}
        current = advanced_nodes(surface, seed, previous, anchors, 2e8)
        step = advance_spatial_underthrust_memory(seed, memory, anchors, 2e8,
            previous_point_states=previous, current_point_states=current)
        memory, previous = step.memory, current
    assert memory.stage_count == 2 and memory.elapsed_time_s == 4e8
    assert any(len(trace.pieces) > 2 for trace in memory.traces)
    assert {piece.branch for cert in step.certificates for piece in cert.pieces} == {
        "locked", "up_dip_plastic", "down_dip_plastic"}
    decoded = spatial_underthrust_memory_from_dict(json.loads(json.dumps(
        spatial_underthrust_memory_to_dict(memory))), seed)
    assert decoded == memory


def test_narrow_locked_pocket_between_all_gauss_nodes_blocks_whole_contact():
    _, surface, seed = prepared()
    memory = initialize_spatial_underthrust_memory(seed)
    trace = seed.fault_traces.traces[0]
    kn = 1/(1/trace.columns.normal_stiffness_pa_m+1/trace.law.normal_stiffness_pa_m)
    kt = 1/(1/trace.columns.shear_stiffness_pa_m+1/trace.law.tangential_stiffness_pa_m)
    pressure = kn
    mu = trace.law.friction_dry+(trace.law.friction_wet-trace.law.friction_dry)*trace.water
    y = trace.law.cohesion_pa*(1-(1-trace.law.wet_cohesion_fraction)*trace.water)+mu*pressure
    center = .731123
    excess = TrigPolynomial((1.-1e-9, -math.cos(center), -math.sin(center), 0., 0.)).scaled(1e8)
    jump = (excess+constant(y)).scaled(-1/kt)
    anchors = {trace.trace_id: (constant(-1.), jump) for trace in seed.fault_traces.traces}
    current = advanced_nodes(surface, seed, seed.states, anchors, 1e8)
    assert all(local.plastic_slip_m < 0. for _, local in current)
    step = advance_spatial_underthrust_memory(seed, memory, anchors, 1e8,
        previous_point_states=seed.states, current_point_states=current)
    cert = step.certificates[0]
    locked = [piece for piece in cert.pieces if piece.branch == "locked"]
    assert locked and sum(piece.high_radians-piece.low_radians for piece in locked) < 1e-3
    assert _assess_interval(cert, 0., cert.extent_radians) == "no_down_dip_plastic_sliding"


@pytest.mark.parametrize("alteration", ["anchors", "point_id", "clock", "state_order"])
def test_continuous_authentication_rejects_independent_anchor_identity_or_clock_changes(alteration):
    _, surface, seed = prepared()
    memory = initialize_spatial_underthrust_memory(seed)
    anchors = {trace.trace_id: (constant(-2.), constant(-100.)) for trace in seed.fault_traces.traces}
    current = advanced_nodes(surface, seed, seed.states, anchors, 1e8)
    if alteration == "anchors":
        anchors = {identity: (gap, jump+constant(.01)) for identity, (gap, jump) in anchors.items()}
    elif alteration == "point_id":
        current = (("counterfeit-point", current[0][1]), *current[1:])
    elif alteration == "clock":
        current = ((current[0][0], replace(current[0][1], elapsed_time_s=current[0][1].elapsed_time_s+1.)), *current[1:])
    else:
        current = tuple(reversed(current))
    with pytest.raises(ValueError):
        advance_spatial_underthrust_memory(seed, memory, anchors, 1e8,
            previous_point_states=seed.states, current_point_states=current)


def admission_fixture(*, viscosity=0., locked=False, owner_labels=(0, 1), matrix=None):
    mesh, surface = fixture()
    matrix = np.eye(3) if matrix is None else np.asarray(matrix)
    surface = replace(surface, fragments=tuple(replace(fragment,
        polygon=tuple(map(tuple, np.asarray(fragment.polygon)@matrix.T)),
        parcel=replace(fragment.parcel, plate=owner_labels[fragment.parcel.plate]))
        for fragment in surface.fragments))
    law = ContactLawParameters(normal_stiffness_pa_m=1e6, tangential_stiffness_pa_m=1e6,
        tensile_strength_pa=2e6, fracture_energy_j_m2=2e7, cohesion_pa=2e6,
        friction_dry=.4, friction_wet=.2, wet_cohesion_fraction=.5, viscosity_pa_s_m=viscosity)
    columns = ElasticColumnParameters(1000., 1e9, 4e8, 1000., 1e9, 4e8)
    traces = []
    for contact in extract_contacts(surface.fragments, surface.radius_km):
        start, end = contact.start, contact.end
        wanted = np.asarray(contact.normal_a_to_b)*(1. if contact.plate_b == owner_labels[0] else -1.)
        if float(np.cross(start, end)@wanted) < 0.:
            start, end = end, start
        traces.append(PhysicalFaultTrace("material-plane:"+contact.contact_id, start, end,
            owner_labels[1], owner_labels[0], math.pi/6., columns, law, 1000., .7, 0., -1., 0.,
            "Explicit synthetic weak plane with solved local continuum permission"))
    trace_set = declare_fault_trace_set(surface, traces, TraceQuadrature(1, 4))
    load = assemble_physical_rigid_mantle_load(mesh, surface,
        PrescribedRigidMantleSource(tuple(matrix@np.asarray((1e-12, 2e-12, 3e-12))), 1e14,
            "Independent initial physical source"))
    seed = initialize_distributed_contact_dynamics(surface, trace_set, load,
        initial_loading_dt_s=1e6, max_rotation_radians=.01)
    anchors = {trace.trace_id: (constant(-2.), constant(-1. if locked else -100.)) for trace in traces}
    dt = GEOMETRIC_DT*SECONDS_PER_MYR
    current = advanced_nodes(surface, seed, seed.states, anchors, dt)
    step = advance_spatial_underthrust_memory(seed, initialize_spatial_underthrust_memory(seed),
        anchors, dt, previous_point_states=seed.states, current_point_states=current)
    return mesh, surface, seed, step


@pytest.mark.parametrize("viscosity", [0., 1e14])
def test_solved_down_dip_permission_accepts_true_local_polygon_mass_and_archive_once(viscosity):
    mesh, surface, seed, step = admission_fixture(viscosity=viscosity)
    resolver, audit = make_spatial_underthrust_resolver(surface, MOTION, GEOMETRIC_DT,
        initial_mechanics=seed, carrier_traces=seed.fault_traces.traces, step=step)
    assert audit["contact_status_counts"] == {"nonconvergent": 2, "spatial_underthrust_admissible": 2}
    transaction = advance_geometric_surface(mesh, surface, MOTION, GEOMETRIC_DT,
        birth_factory=birth, polarity_resolver=resolver)
    inventory = consume_geometric_transaction(surface, transaction, initialize_boundary(surface), MOTION, GEOMETRIC_DT)
    assert len(transaction.losses) == len(inventory.cohorts) == 2
    assert all(loss.fragment.parcel.plate == 1 and loss.receiver_plate == 0 for loss in transaction.losses)
    assert all(loss.polarity_basis == "solved_spatial_underthrust_admission" for loss in transaction.losses)
    assert max(ledger(surface, transaction.state, inventory, transaction.births).values()) < 5e-12
    assert all(cohort.supports and not cohort.unresolved_support for cohort in inventory.cohorts)
    with pytest.raises(ValueError):
        consume_geometric_transaction(surface, transaction, inventory, MOTION, GEOMETRIC_DT)


def test_zero_viscosity_endpoint_at_yield_is_admitted_from_actual_plastic_branch():
    _, _, _, step = admission_fixture(viscosity=0.)
    for cert in step.certificates:
        for piece in cert.pieces:
            assert piece.branch == "down_dip_plastic"
            assert (piece.shear_traction.scaled(-1.)-piece.strength)(.7) == pytest.approx(0., abs=1e-7)
            assert piece.plastic_increment(.7) < 0.


@pytest.mark.parametrize("owner_labels", [(0, 1), (10**12, 7)])
def test_spatial_permission_is_coordinate_covariant_and_uses_actual_sparse_owners(owner_labels):
    matrix = Rotation.from_rotvec((.31, -.17, .24)).as_matrix()
    mesh, surface, seed, step = admission_fixture(owner_labels=owner_labels, matrix=matrix)
    omega = {owner_labels[index]: matrix@MOTION[index] for index in range(2)}
    resolver, audit = make_spatial_underthrust_resolver(surface, omega, GEOMETRIC_DT,
        initial_mechanics=seed, carrier_traces=seed.fault_traces.traces, step=step)
    actual = advance_geometric_surface(mesh, surface, omega, GEOMETRIC_DT,
        birth_factory=birth, polarity_resolver=resolver)
    assert audit["contact_status_counts"] == {"nonconvergent": 2, "spatial_underthrust_admissible": 2}
    assert all(loss.fragment.parcel.plate == owner_labels[1] and loss.receiver_plate == owner_labels[0]
               for loss in actual.losses)
    old_mesh, old_surface, old_seed, old_step = admission_fixture()
    original_resolver, _ = make_spatial_underthrust_resolver(old_surface, MOTION, GEOMETRIC_DT,
        initial_mechanics=old_seed, carrier_traces=old_seed.fault_traces.traces, step=old_step)
    original = advance_geometric_surface(old_mesh, old_surface, MOTION, GEOMETRIC_DT,
        birth_factory=birth, polarity_resolver=original_resolver)
    initial_losses = {loss.fragment.parcel.material_id: loss.fragment.parcel.area_km2 for loss in original.losses}
    transformed_losses = {loss.fragment.parcel.material_id: loss.fragment.parcel.area_km2 for loss in actual.losses}
    assert transformed_losses == pytest.approx(initial_losses, rel=3e-11)


def test_locked_mechanics_rejects_before_any_birth_and_keeps_surface_bytes_unchanged():
    mesh, surface, seed, step = admission_fixture(locked=True)
    before = json.dumps(asdict(surface), sort_keys=True)
    resolver, audit = make_spatial_underthrust_resolver(surface, MOTION, GEOMETRIC_DT,
        initial_mechanics=seed, carrier_traces=seed.fault_traces.traces, step=step)
    calls = []
    def forbidden_birth(*args):
        calls.append(args)
        raise AssertionError("Blocked permission reached material birth")
    with pytest.raises(UnresolvedPolarityError):
        advance_geometric_surface(mesh, surface, MOTION, GEOMETRIC_DT,
            birth_factory=forbidden_birth, polarity_resolver=resolver)
    assert not calls and json.dumps(asdict(surface), sort_keys=True) == before
    assert audit["contact_status_counts"] == {"no_down_dip_plastic_sliding": 2, "nonconvergent": 2}


def test_one_admitted_receiver_arc_cannot_cover_another_unknown_receiver_arc():
    _, surface, seed, step = admission_fixture()
    # Deliberately adversarial current neighbor geometry against an already
    # authenticated original x-plane seed.  A current receiver island now has
    # a genuine y-plane neighbor absent from its retained original ports.
    # This targets the gate alone; it is not an accepted engine trajectory.
    surface = replace(surface, fragments=tuple(replace(fragment,
        parcel=replace(fragment.parcel, plate=0 if fragment.fragment_id == "octant:0" else 1))
        for fragment in surface.fragments))
    omega = {0: (0., 0., 0.), 1: (-.04, .05, 0.)}
    resolver, audit = make_spatial_underthrust_resolver(surface, omega, GEOMETRIC_DT,
        initial_mechanics=seed, carrier_traces=seed.fault_traces.traces, step=step)
    receiver = next(item for item in surface.fragments if item.fragment_id == "octant:0")
    donor = next(item for item in surface.fragments if item.fragment_id == "octant:6")
    moved = rotate_points(donor.polygon, omega[1], GEOMETRIC_DT)
    overlap = intersect_convex(receiver.polygon, moved)
    assert polygon_area(overlap) > 0.
    decision = resolver(receiver, donor, overlap)
    witnesses = audit["actual_overlap_assessments"][-1]
    assert any(row["status"] == "spatial_underthrust_admissible" for row in witnesses["receiver_arcs"])
    assert not decision.comparison
    assert decision.basis == "no_continuous_physical_fault_support"
    assert len(decision.contact_ids) > len(witnesses["contact_ids"])


def test_conflicting_archived_directions_remain_blocked_after_solved_sliding():
    mesh, surface, seed, step = admission_fixture()
    history = tuple(PolarityEvidence("old:"+contact.contact_id+":"+str(sub), contact.contact_id, sub, 1-sub,
        "Explicit opposed inherited history") for contact in extract_contacts(surface.fragments, surface.radius_km)
        for sub in (0, 1))
    resolver, audit = make_spatial_underthrust_resolver(surface, MOTION, GEOMETRIC_DT,
        initial_mechanics=seed, carrier_traces=seed.fault_traces.traces, step=step, history=history)
    with pytest.raises(UnresolvedPolarityError):
        advance_geometric_surface(mesh, surface, MOTION, GEOMETRIC_DT, birth_factory=birth, polarity_resolver=resolver)
    assert audit["contact_status_counts"] == {"conflicting_local_history": 4}


def test_geometric_permission_cannot_reuse_a_different_constitutive_interval():
    _, surface, seed, step = admission_fixture()
    with pytest.raises(ValueError, match="physical intervals"):
        make_spatial_underthrust_resolver(surface, MOTION, GEOMETRIC_DT/2.,
            initial_mechanics=seed, carrier_traces=seed.fault_traces.traces, step=step)


def test_rehashed_or_rebuilt_certificate_fields_cannot_bypass_actual_node_authentication():
    _, surface, seed, step = admission_fixture()
    changed = tuple((identity, gap, jump+constant(.01)) for identity, gap, jump in step.anchor_fields)
    with pytest.raises(ValueError, match="accepted point"):
        make_spatial_underthrust_resolver(surface, MOTION, GEOMETRIC_DT,
            initial_mechanics=seed, carrier_traces=seed.fault_traces.traces,
            step=replace(step, anchor_fields=changed, node_manifest_sha256="0"*64))


@pytest.mark.parametrize("field", ["stage_count", "elapsed_time_s", "unknown"])
def test_memory_codec_rejects_boolean_or_unknown_aliases(field):
    _, _, seed = prepared()
    raw = json.loads(json.dumps(spatial_underthrust_memory_to_dict(initialize_spatial_underthrust_memory(seed))))
    raw[field] = True
    with pytest.raises(ValueError):
        spatial_underthrust_memory_from_dict(raw, seed)


def test_a_continuous_opening_pocket_between_nodes_is_rejected():
    _, surface, seed = prepared()
    # Positive near center and negative at all four Gauss nodes.
    center = .731123
    gap = TrigPolynomial((-1.+1e-4, math.cos(center), math.sin(center), 0., 0.))
    anchors = {trace.trace_id: (gap, constant(-100.)) for trace in seed.fault_traces.traces}
    current = advanced_nodes(surface, seed, seed.states, anchors, 1e8)
    with pytest.raises(ValueError, match="continuously closed"):
        advance_spatial_underthrust_memory(seed, initialize_spatial_underthrust_memory(seed), anchors, 1e8,
            previous_point_states=seed.states, current_point_states=current)
