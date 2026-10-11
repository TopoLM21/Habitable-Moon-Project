"""Independent precision reference for coordinates entering spherical clipping."""
from decimal import Decimal, localcontext
from dataclasses import replace
from itertools import product
import math

import numpy as np
import pytest

from tectonics.spherical_polygons import _vertices

from tectonics.fractional_surface import EXTENSIVE_FIELDS, SurfaceParcel
from tectonics.geometric_surface import (
    AREA_REL_TOL, GeometricFragment, GeometricSurfaceState,
    _RigidlyAdvectedCandidates, _certify_rigid_candidates, audit_partition,
    rotate_surface, save_geometric_checkpoint,
)
from tectonics.spherical_polygons import (
    _edge_cross, _edge_distance, _plane_crossing_error, _supported, _unit,
    _saved_binary64_rays, _saved_inputs_separated, _saved_ray_side,
    _polygon_projection_area_error, _rigid_rotation_projection,
    intersect_convex, normalize_polygon, polygon_area, rotate_polygon, subtract_convex,
)
import tectonics.spherical_polygons as polygons


# Full SOURCE/SECOND vectors measured by the independently prescribed force
# controls.  Their off-axis components are retained, never projected away.
NEXT_OMEGA = (
    {0: (3.943918887612674e-18, -1.7754383428153527e-19, 28.245230535292283),
     1: (4.171092555597449e-18, 2.4581194098898567e-19, 34.86996946470772)},
    {0: (3.5029521325109036e-18, -7.03581045461479e-20, 31.616288107981852),
     1: (3.754415918516928e-18, -2.1544284047396363e-19, 37.810431892018144)},
)
NEXT_DT_MYR = 3.168808781402895e-8


def thin_partition():
    """Complete disjoint sphere with a resolved tiny triangle next to +y."""
    fragments = []
    for index, signs in enumerate(product((-1., 1.), repeat=3)):
        shapes = [(str(index), normalize_polygon(np.diag(signs)), 0)]
        if signs == (-1., 1., 1.):
            shapes = [
                ("thin", normalize_polygon(((-1e-6, 1., 0.),
                    (0., 1., 1e-33), (0., 1., 0.))), 1),
                ("remainder", normalize_polygon(((-1., 0., 0.),
                    (-1e-6, 1., 0.), (0., 1., 1e-33), (0., 0., 1.))), 0),
            ]
        for identity, polygon, owner in shapes:
            area = polygon_area(polygon)*10000.
            parcel = SurfaceParcel(index, owner, identity, area, 2.*area,
                8.*area, area*1e12, 5., ())
            fragments.append(GeometricFragment(identity, tuple(map(tuple, polygon)), parcel))
    state = GeometricSurfaceState(0., 100., tuple(fragments))
    audit_partition(state)
    return state


def changed_measure(parcel):
    return replace(parcel, **{name: getattr(parcel, name)*(1.+1e-5)
                              for name in EXTENSIVE_FIELDS})


@pytest.mark.skipif(np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant,
                   reason="Platform has no precision beyond binary64")
def test_extended_coordinates_are_normalized_before_binary64_rounding():
    # These exactly representable dyadic offsets model the sub-ULP components
    # retained by a Rodrigues rotation.  Rounding the input first changes the
    # first unit-vector component by one binary64 ULP.
    base, offsets = (1, 2, 3), (-62, 37, 59)
    coordinates = np.asarray(base, dtype=np.longdouble) + np.ldexp(
        np.asarray(offsets, dtype=np.longdouble), -60)
    with localcontext() as context:
        context.prec = 100
        exact = [Decimal(integer) + Decimal(offset) / Decimal(2)**60
                 for integer, offset in zip(base, offsets)]
        length = sum(value * value for value in exact).sqrt()
        reference = np.asarray([float(value / length) for value in exact])
    actual = _vertices([coordinates])
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual[0], reference)


@pytest.mark.parametrize("omega", NEXT_OMEGA)
def test_actual_full_next_rotation_preserves_mass_and_certifies_temporary_thin_geometry(omega):
    before = thin_partition()
    candidate = rotate_surface(before, omega, NEXT_DT_MYR)
    assert isinstance(candidate, _RigidlyAdvectedCandidates)
    assert candidate.phase == "advected_candidates"
    assert candidate._rotation_vectors == tuple(sorted(omega.items()))
    assert candidate._source_surface == before
    bounds = dict(candidate._projection_area_errors_km2)
    for old, moved in zip(before.fragments, candidate.fragments):
        assert moved.parcel == replace(old.parcel, age_myr=old.parcel.age_myr+NEXT_DT_MYR)
        np.testing.assert_array_equal(moved.polygon,
            rotate_polygon(old.polygon, omega[old.parcel.plate], NEXT_DT_MYR))
        measured = polygon_area(moved.polygon)*before.radius_km**2
        assert 0. <= bounds[moved.fragment_id] < measured
        assert abs(measured-old.parcel.area_km2) <= bounds[moved.fragment_id]
    thin = next(fragment for fragment in candidate.fragments if fragment.fragment_id == "thin")
    assert not math.isclose(polygon_area(thin.polygon)*10000., thin.parcel.area_km2,
                            rel_tol=AREA_REL_TOL, abs_tol=0.)
    # The persisted contract has the same tolerance and rejects this projection.
    with pytest.raises(ValueError, match="footprint disagree"):
        GeometricSurfaceState(candidate.time_myr, candidate.radius_km,
            candidate.fragments, candidate.known_material_ids, candidate.phase)


def test_temporary_rigid_projection_cannot_create_a_checkpoint(tmp_path):
    candidate = rotate_surface(thin_partition(), NEXT_OMEGA[0], NEXT_DT_MYR)
    path = tmp_path/"projection.json"
    with pytest.raises(ValueError, match="temporary.*cannot be checkpointed"):
        save_geometric_checkpoint(path, candidate)
    assert not path.exists()
    with pytest.raises(AttributeError, match="immutable"):
        candidate._dt_myr = 1.


def test_common_thin_rotation_keeps_strict_persisted_area_contract():
    before = thin_partition()
    common = {owner: NEXT_OMEGA[0][1] for owner in (0, 1)}
    with pytest.raises(ValueError, match="footprint disagree"):
        rotate_surface(before, common, NEXT_DT_MYR)


def test_rigid_projection_witness_rejects_changed_mass_and_changed_world_coordinates():
    before = thin_partition()
    candidate = rotate_surface(before, NEXT_OMEGA[0], NEXT_DT_MYR)
    index = next(i for i, fragment in enumerate(candidate.fragments) if fragment.fragment_id == "thin")
    moved = list(candidate.fragments)
    fragment = moved[index]
    moved[index] = replace(fragment, parcel=changed_measure(fragment.parcel))
    with pytest.raises(ValueError, match="unchanged material and actual rotation"):
        _certify_rigid_candidates(before, moved, NEXT_OMEGA[0], NEXT_DT_MYR)
    moved[index] = fragment
    coords = np.asarray(fragment.polygon).copy()
    coords[0, 2] = np.nextafter(coords[0, 2], math.inf)
    moved[index] = replace(fragment, polygon=tuple(map(tuple, coords)))
    with pytest.raises(ValueError, match="unchanged material and actual rotation"):
        _certify_rigid_candidates(before, moved, NEXT_OMEGA[0], NEXT_DT_MYR)


def test_imported_microscopic_area_corruption_has_no_uncertainty_allowance():
    before = thin_partition()
    fragments = list(before.fragments)
    index = next(i for i, fragment in enumerate(fragments) if fragment.fragment_id == "thin")
    fragment = fragments[index]
    fragments[index] = replace(fragment, parcel=changed_measure(fragment.parcel))
    with pytest.raises(ValueError, match="footprint disagree"):
        GeometricSurfaceState(0., 100., tuple(fragments))


@pytest.mark.parametrize("dt", [0., -1., False, math.nan, math.inf])
def test_private_rigid_witness_requires_an_actual_advancing_timestep(dt):
    before = thin_partition()
    with pytest.raises(ValueError, match="timestep"):
        _certify_rigid_candidates(before, before.fragments, NEXT_OMEGA[0], dt)


def test_private_rigid_witness_does_not_allow_common_motion():
    before = thin_partition()
    with pytest.raises(ValueError, match="differential"):
        _certify_rigid_candidates(before, before.fragments,
            {0: (0., 0., 0.), 1: (0., 0., 0.)}, NEXT_DT_MYR)


@pytest.mark.parametrize("binary64_working_precision", [False, True])
def test_rigid_area_envelope_is_positive_and_portable_without_longdouble(monkeypatch,
        binary64_working_precision):
    before = thin_partition()
    thin = next(fragment for fragment in before.fragments if fragment.fragment_id == "thin")
    if binary64_working_precision:
        monkeypatch.setattr(polygons, "_LD", np.float64)
    target, errors = _rigid_rotation_projection(thin.polygon, NEXT_OMEGA[0][1], NEXT_DT_MYR)
    envelope = _polygon_projection_area_error(target, errors)
    measured = polygon_area(target)
    assert 0. < envelope < measured
    assert abs(measured-thin.parcel.area_km2/10000.) <= envelope
    assert math.isinf(_polygon_projection_area_error(target, np.ones_like(target)))
    with pytest.raises(ValueError, match="zero area"):
        _rigid_rotation_projection(((1., 0., 0.), (1., 1., 0.), (1., 2., 0.)),
                                   NEXT_OMEGA[0][1], NEXT_DT_MYR)


def test_squared_norm_underflow_is_not_certified_as_zero_physical_rotation(monkeypatch):
    monkeypatch.setattr(polygons, "_LD", np.float64)
    _, errors = _rigid_rotation_projection(((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)),
                                         (1e-300, 0., 0.), 1e300)
    assert np.isinf(errors).all()


@pytest.mark.parametrize("binary64_working_precision", [False, True])
def test_literal_short_saved_edge_sign_survives_construction_plane_uncertainty(
        monkeypatch, binary64_working_precision):
    if binary64_working_precision:
        monkeypatch.setattr(polygons, "_LD", np.float64)
    a = np.asarray((1.238785685359418e-19, -.525010129004093, .8510959783967409))
    b = np.asarray((-1.238785685359418e-19, -.525010129004093, .8510959783967409))
    pole = np.asarray((0., 0., 1.))
    # The endpoints' shared saved yz components define this plane exactly.
    # Construction incidence keeps its existing coordinate uncertainty, while
    # the literal saved-domain separator resolves this exact outside sign.
    assert _saved_ray_side(a, b, pole, {}) == -1
    assert _saved_ray_side(b, a, pole, {}) == 1
    assert _edge_distance(a, a, b, None) == 0.


@pytest.mark.parametrize("binary64_working_precision", [False, True])
@pytest.mark.parametrize("width", [1e-30, 1e-80, 1e-150])
def test_nested_plane_crossing_preserves_representable_ray_during_norm_underflow(
        monkeypatch, binary64_working_precision, width):
    if binary64_working_precision:
        monkeypatch.setattr(polygons, "_LD", np.float64)
    origin = np.asarray((0., 0., 1.))
    first = (origin, np.asarray((width, 0., 1.)))
    second = (origin, np.asarray((0., width, 1.)))
    # Exact Cartesian algebra gives a strictly positive z ray of width².
    # Squaring its norm underflows on binary64 for width1e-80 and1e-150.
    ray = _edge_cross(_edge_cross(*first), _edge_cross(*second))
    assert ray[2] > 0.
    crossing = _unit(ray)
    np.testing.assert_array_equal(crossing, origin)
    envelope = _plane_crossing_error(first, second, crossing)
    assert envelope is not None and np.isfinite(envelope).all()
    assert np.all(envelope >= 0.)
    assert _plane_crossing_error(first, first, crossing) is None


def test_vacancy_cut_reuses_shared_endpoint_with_its_actual_nested_cross_error():
    # Saved construction from the old two-step slab probe: the inherited
    # planes' crossing differed from the known clipping endpoint by364xULPs
    # and one zULP.  An interpolation-only envelope failed to reuse it and
    # tilted the newborn's short top edge into its retained neighbour.
    vertices = [
        (-5.86570315973603e-20, -.5257311121191336, .85065080835204),
        (.8077660387295019, -.5007700159285651, .3110360394265486),
        (.8083902146637106, -.49913371717766525, .3120429348861568),
        (0., -.5253692317276304, .8508743563851947),
        (-9.139771023942028e-20, -.5253692317276304, .8508743563851947),
    ]
    supports = [
        ((-.0013134217111704627, -.5253675899496364, .8508743563851947), vertices[1]),
        ((.5214713041740324, -.8532688198469658, 0.), vertices[1]),
        ((.8090154381976788, -.49874903802981335, .3110360394265486), vertices[3]),
        (vertices[3], (-.001314326411206824, -.5257294692102639, .85065080835204)),
        ((-1.2387856853594142e-19, -.525010129004093, .8510959783967409), vertices[0]),
    ]
    endpoint = (5.86570315973603e-20, -.5257311121191336, .85065080835204)
    covered = [endpoint, vertices[1],
        (.8090107949745042, -.497496166883635, .3130480754625573),
        (1.2387856853594142e-19, -.525010129004093, .8510959783967409)]
    subject = _supported(vertices, tuple((np.asarray(a), np.asarray(b)) for a, b in supports))
    pieces = subtract_convex(subject, covered)
    assert len(pieces) == 1 and polygon_area(pieces[0]) > 0.
    assert any(np.array_equal(vertex, endpoint) for vertex in pieces[0])
    assert any(np.array_equal(vertex, vertices[0]) for vertex in pieces[0])


def test_underflowed_nonaxis_nested_product_cannot_certify_a_zero_component(monkeypatch):
    # Each saved ray is representable, as are the exact normalized crossing's
    # x/y components.  The selected outer-cross products themselves underflow
    # in binary64.  A relative error formula then falsely reports E_xy=0.
    monkeypatch.setattr(polygons, "_LD", np.float64)
    width = 1e-150
    first = (np.asarray((0., 0., 1.)), np.asarray((width, 2.*width, 1.)))
    second = (np.asarray((width, width, 1.)),
              np.asarray((3.*width, 4.*width, 1.)))
    line = _edge_cross(_edge_cross(*first), _edge_cross(*second))
    np.testing.assert_array_equal(line[:2], (0., 0.))
    assert line[2] < 0.
    crossing = _unit(line)
    np.testing.assert_array_equal(crossing, (0., 0., -1.))
    assert _plane_crossing_error(first, second, crossing) is None


@pytest.mark.parametrize("reverse_order", [False, True])
@pytest.mark.parametrize("binary64_working_precision", [False, True])
def test_saved_domain_separation_rejects_false_slab_restart_overlap(
        monkeypatch, binary64_working_precision, reverse_order):
    if binary64_working_precision:
        monkeypatch.setattr(polygons, "_LD", np.float64)
    retained = ((-.0026286446078778564, .5257245404939229, .85065080835204),
        (-.8506401752390879, -.004253236319890511, .5257311121191336),
        (.0026286446078778564, -.5257245404939229, .85065080835204))
    birth = ((1.2387856853594142e-19, -.525010129004093, .8510959783967409),
        (-1.238785685359418e-19, -.525010129004093, .8510959783967409),
        (-9.139771023942317e-20, -.5253692317276304, .8508743563851947),
        (9.139771023942317e-20, -.5253692317276304, .8508743563851947))
    # Exact rational halfspaces prove these saved polygons disjoint.  The old
    # filtered clip falsely included the pole, 2.5e5 times the entire birth.
    assert polygon_area(birth) > 0.
    a, b = (birth, retained) if reverse_order else (retained, birth)
    assert not len(intersect_convex(a, b))


def test_saved_domain_empty_proof_cannot_remove_the_true_raw_restart_sliver():
    retained = ((.5214713041740324, -.8532688198469658, 0.),
        (.8077660387295019, -.5007700159285651, .3110360394265486),
        (-.0013134217111704627, -.5253675899496364, .8508743563851947),
        (-.0026286446078778564, -.5257245404939229, .85065080835204))
    birth = ((-5.86570315973603e-20, -.5257311121191336, .85065080835204),
        (5.865703159735603e-20, -.5257311121191336, .8506508083520399),
        (9.139771023942028e-20, -.5253692317276304, .8508743563851947),
        (0., -.5253692317276304, .8508743563851947),
        (-9.139771023942028e-20, -.5253692317276304, .8508743563851947))
    # The independent exact oracle finds a positive 7.975e-39sr intersection
    # in these old saved coordinates.  Endpoint construction prevents that
    # accidental sliver in rebuilt runs; the empty proof cannot erase it here.
    a, b = normalize_polygon(retained), normalize_polygon(birth)
    assert not _saved_inputs_separated(a, b)
    assert not _saved_inputs_separated(b, a)


@pytest.mark.parametrize("pole_y", [0., 5e-324])
def test_saved_side_uses_exact_arithmetic_when_subnormal_products_flip_sign(monkeypatch, pole_y):
    monkeypatch.setattr(polygons, "_LD", np.float64)
    a = np.asarray((.23138424375648017, 1e-322, .9728624423531015))
    b = np.asarray((.9408900513139096, -9.4e-323, .33871213639093084))
    point = np.asarray((.658442107088969, pole_y, .7526313783069629))
    # The floating determinant is +5e-324 or zero with an alleged zero error;
    # both exact dyadic determinants are negative, so neither certifies a sign.
    assert _saved_ray_side(a, b, point, {}) == -1
    assert _saved_ray_side(b, a, point, {}) == 1


def test_saved_separator_eligibility_preserves_original_input_precision():
    assert _saved_binary64_rays(((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)))
    assert not _saved_binary64_rays(((1, 0., 0.), (0., 1., 0.), (0., 0., 1.)))
    assert not _saved_binary64_rays(np.eye(3, dtype=np.longdouble))
    assert not _saved_binary64_rays(_supported(np.eye(3)))
    with pytest.raises(ValueError):
        intersect_convex((), ((1., 0., 0.), (1., 0., 0.), (0., 1., 0.)))
