"""Independent cantilever shape, energy, boundary and explicit-physics checks."""
from dataclasses import FrozenInstanceError, replace
import copy
import json
import math

import numpy as np
import pytest

from tectonics.geometric_bending import (
    ElasticBendingBankParameters, FaultBendingParameters,
    bank_energy_j, bank_max_surface_strain, bank_max_tip_slope,
    bank_restoring_force_n, bending_bank_rigidity_nm,
    bending_parameters_from_dict, bending_parameters_sha256,
    bending_parameters_to_dict, bending_strip_stiffness_n_m,
    canonical_bending_parameters, cantilever_shape,
    cantilever_shape_curvature, cantilever_shape_slope,
)


def bank():
    return ElasticBendingBankParameters(1e5, 1e4, 7e10, .25,
        "synthetic explicitly clamped elastic cantilever")


def test_exact_cantilever_boundary_conditions_and_tip_load_oracle():
    material = bank()
    length, force = 2.4e5, -2e12
    rigidity = material.young_modulus_pa*material.elastic_thickness_m**3/(12.*(1.-material.poisson_ratio**2))
    displacement = force*material.width_m**3/(3.*rigidity*length)
    assert cantilever_shape(0.) == 1.
    assert cantilever_shape(1.) == 0.
    assert cantilever_shape_slope(1.) == 0.
    assert cantilever_shape_curvature(0.) == 0.
    # Exact beam solution: clamp moment=-P*L and uniform transmitted shear P.
    clamp_moment = -rigidity*length*displacement/material.width_m**2*cantilever_shape_curvature(1.)
    transmitted_shear = rigidity*length*displacement/material.width_m**3*3.
    assert clamp_moment == pytest.approx(-force*material.width_m, rel=3e-15)
    assert transmitted_shear == pytest.approx(force, rel=3e-15)
    assert bank_restoring_force_n(material, length, displacement) == pytest.approx(-force, rel=3e-15)
    assert bank_energy_j(material, length, displacement) == pytest.approx(.5*force*displacement, rel=3e-15)


def test_bending_energy_matches_independent_physical_curvature_quadrature():
    material = bank()
    length, displacement = 1.5e5, -320.
    nodes, weights = np.polynomial.legendre.leggauss(7)
    physical_x = .5*material.width_m*(nodes+1.)
    physical_curvature = 3.*displacement*physical_x/material.width_m**3
    rigidity = material.young_modulus_pa*material.elastic_thickness_m**3/(12.*(1.-material.poisson_ratio**2))
    independent = .5*rigidity*length*.5*material.width_m*float(weights@physical_curvature**2)
    assert bank_energy_j(material, length, displacement) == pytest.approx(independent, rel=3e-15)


def test_tip_force_is_energy_gradient_and_implicit_work_includes_numerical_remainder():
    material, length = bank(), 1e5
    old, new = -100., 240.
    stiffness = bending_strip_stiffness_n_m(material, length)
    delta = new-old
    internal_work = -bank_restoring_force_n(material, length, new)*delta
    stored_increment = bank_energy_j(material, length, new)-bank_energy_j(material, length, old)
    implicit_remainder = .5*stiffness*delta**2
    assert internal_work == pytest.approx(stored_increment+implicit_remainder, rel=3e-15)
    h = 1e-2
    gradient = (bank_energy_j(material, length, new+h)-bank_energy_j(material, length, new-h))/(2.*h)
    assert gradient == pytest.approx(-bank_restoring_force_n(material, length, new), rel=3e-12)


def test_stiffness_scales_with_physical_material_and_dimensions():
    material, length = bank(), 2e5
    stiffness = bending_strip_stiffness_n_m(material, length)
    assert bending_strip_stiffness_n_m(replace(material, young_modulus_pa=2*material.young_modulus_pa), length) == pytest.approx(2*stiffness)
    assert bending_strip_stiffness_n_m(replace(material, elastic_thickness_m=2*material.elastic_thickness_m), length) == pytest.approx(8*stiffness)
    assert bending_strip_stiffness_n_m(replace(material, width_m=2*material.width_m), length) == pytest.approx(stiffness/8.)
    assert bending_strip_stiffness_n_m(material, 2*length) == pytest.approx(2*stiffness)
    assert bank_energy_j(material, length, 0.) == 0.
    assert bank_restoring_force_n(material, length, 0.) == 0.


def test_slope_and_surface_strain_are_maxima_of_the_physical_shape():
    material, displacement = bank(), -280.
    coordinates = np.linspace(0., 1., 1001)
    slopes = displacement/material.width_m*cantilever_shape_slope(coordinates)
    fibre_strains = .5*material.elastic_thickness_m*displacement/material.width_m**2*cantilever_shape_curvature(coordinates)
    assert bank_max_tip_slope(material, displacement) == pytest.approx(max(abs(slopes)), rel=3e-15)
    assert bank_max_surface_strain(material, displacement) == pytest.approx(max(abs(fibre_strains)), rel=3e-15)
    assert np.all(cantilever_shape(coordinates) >= 0.)
    assert np.all(np.diff(cantilever_shape(coordinates)) <= 0.)
    assert cantilever_shape(np.nextafter(1., 0.)) > 0.


@pytest.mark.parametrize("nu", [-.7, 0., .25, .499999, np.nextafter(-1., 0.)])
def test_rigidity_preserves_full_physical_poisson_domain_without_denominator_floor(nu):
    material = replace(bank(), poisson_ratio=nu)
    expected = material.young_modulus_pa*material.elastic_thickness_m**3/(12.*(1.-nu**2))
    assert bending_bank_rigidity_nm(material) == pytest.approx(expected, rel=3e-15)


@pytest.mark.parametrize(("name", "value"), [
    ("width_m", 0.), ("width_m", -1.), ("width_m", True), ("width_m", float("nan")),
    ("elastic_thickness_m", 0.), ("elastic_thickness_m", -1.),
    ("elastic_thickness_m", np.bool_(True)), ("elastic_thickness_m", 1e5),
    ("elastic_thickness_m", 2e5), ("young_modulus_pa", 0.),
    ("young_modulus_pa", float("inf")), ("young_modulus_pa", "7e10"),
    ("poisson_ratio", -1.), ("poisson_ratio", .5), ("poisson_ratio", False),
    ("provenance", ""), ("provenance", "  "), ("provenance", None),
])
def test_invalid_or_inferred_bank_parameters_are_rejected(name, value):
    with pytest.raises(ValueError):
        replace(bank(), **{name: value})


@pytest.mark.parametrize("bad", [-1., 1.01, True, np.bool_(False), float("nan"), [0., float("inf")], [0., True], "0.5"])
def test_shape_rejects_coordinates_outside_supported_strip(bad):
    with pytest.raises(ValueError):
        cantilever_shape(bad)


@pytest.mark.parametrize("bad", [0., -1., True, float("nan"), float("inf")])
def test_stiffness_rejects_nonphysical_trace_lengths(bad):
    with pytest.raises(ValueError):
        bending_strip_stiffness_n_m(bank(), bad)


def test_parameters_are_immutable_and_canonical_with_exact_trace_coverage():
    first = FaultBendingParameters("alpha", bank(), replace(bank(), width_m=1.1e5))
    second = FaultBendingParameters("beta", bank(), bank())
    assert canonical_bending_parameters((second, first), trace_ids=("beta", "alpha")) == (first, second)
    with pytest.raises(FrozenInstanceError):
        first.sub.width_m = 1.
    with pytest.raises(ValueError, match="repeat"):
        canonical_bending_parameters((first, first))
    with pytest.raises(ValueError, match="every physical trace"):
        canonical_bending_parameters((first,), trace_ids=("alpha", "beta"))
    with pytest.raises(ValueError):
        FaultBendingParameters("alpha", {}, bank())


def test_canonical_json_roundtrip_binds_every_parameter_and_provenance():
    parameters = (FaultBendingParameters("alpha", bank(), bank()), FaultBendingParameters("beta", bank(), bank()))
    raw = json.loads(json.dumps(bending_parameters_to_dict(reversed(parameters))))
    restored = bending_parameters_from_dict(raw)
    assert restored == parameters
    digest = bending_parameters_sha256(parameters)
    assert digest == bending_parameters_sha256(restored) == bending_parameters_sha256(reversed(parameters))
    assert len(digest) == 64
    mutations = [replace(parameters[0].sub, width_m=1.01e5),
                 replace(parameters[0].sub, elastic_thickness_m=1.01e4),
                 replace(parameters[0].sub, young_modulus_pa=7.1e10),
                 replace(parameters[0].sub, poisson_ratio=.24),
                 replace(parameters[0].sub, provenance="another physical material provenance")]
    for changed in mutations:
        assert bending_parameters_sha256((replace(parameters[0], sub=changed), parameters[1])) != digest


@pytest.mark.parametrize("mutation", ["extra", "missing", "wrong_bank", "boolean", "duplicate", "noncanonical"])
def test_restart_schema_cannot_hide_parameters_or_reorder_bindings(mutation):
    raw = bending_parameters_to_dict((FaultBendingParameters("alpha", bank(), bank()), FaultBendingParameters("beta", bank(), bank())))
    raw = copy.deepcopy(raw)
    if mutation == "extra":
        raw[0]["hidden_foundation"] = 1.
    elif mutation == "missing":
        raw[0]["sub"].pop("elastic_thickness_m")
    elif mutation == "wrong_bank":
        raw[0]["over"] = []
    elif mutation == "boolean":
        raw[0]["sub"]["poisson_ratio"] = True
    elif mutation == "duplicate":
        raw[1]["trace_id"] = "alpha"
    elif mutation == "noncanonical":
        raw.reverse()
    with pytest.raises(ValueError):
        bending_parameters_from_dict(raw)
