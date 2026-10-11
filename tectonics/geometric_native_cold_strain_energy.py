"""Source-preserving cold seven-mode, calibrated initial-strain energy.

This is an explicitly NEW nonlocal constitutive extension.  The authentic old
tip stiffness and nonzero accepted tip displacement remain the global beam
ledger's reference energy and work-conjugate force.  A positive rank-one
calibration pays the old-tip strain projection once; the complementary full
three-dimensional strain is paid by the actual layered tensor integral.

The reference initial strain is w_ref*sym(grad psi_tip), rather than a claim
to have recovered an old three-dimensional prestress.  Incremental Green
strain makes the new law objective.  Cold fields, all interface laws, loads,
drag, whole-clock admission and an accepted mixed step are still distinct.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from fractions import Fraction as F
import hashlib
import json
import math

from .geometric_native_pose import FractionInterval as I, interval_sqrt
from .geometric_native_spherical_solid import interval_cbrt
from .geometric_native_normal_support import load_native_normal_support_source


MATERIAL_MODEL = "new_layered_tensor_with_positive_nonlocal_old_tip_calibration"
INITIAL_STRAIN_MODEL = "new_initial_strain_w_ref_times_complete_symmetric_tip_gradient"
COORDINATE_ORDER = ("old_tip_increment", "xx", "yy", "zz", "xy", "xz", "yz")
_MATERIALS = {}


def _geometry():
    from . import geometric_native_cold_body as geometry
    return geometry


def _body(value):
    return _geometry().validate_native_cold_body(value)


def _coordinates(value):
    if type(value) is not tuple or len(value) != 7 or any(type(x) is not F for x in value):
        raise ValueError("Cold increments require seven immutable exact Fraction amplitudes in metres")
    return value


def _payload(value):
    if type(value) is F:
        return [value.numerator, value.denominator]
    if type(value) is I:
        return {"lo": _payload(value.lo), "hi": _payload(value.hi)}
    if is_dataclass(value):
        return {item.name: _payload(getattr(value, item.name)) for item in fields(value)}
    if type(value) in (tuple, list):
        return [_payload(x) for x in value]
    if type(value) is dict and all(type(k) is str for k in value):
        return {k: _payload(v) for k, v in value.items()}
    if type(value) in (str, int, bool) or value is None:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise TypeError("Cold material and energy receipts require literal finite source data")


def _encoded(value):
    return json.dumps(_payload(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _material_snapshot(value):
    return tuple(getattr(value, item.name) for item in fields(value)
                 if item.name != "material_snapshot_sha256")


def _positive_fraction(value, name):
    if type(value) is not F or value <= 0:
        raise ValueError(name + " requires a literal positive Fraction")
    return value


@dataclass(frozen=True, slots=True)
class NativeColdStrainMaterial:
    component_id: str
    body_snapshot_sha256: str
    source_snapshot_sha256: str
    source_support_sha256: str
    upper_young_modulus_pa: F
    upper_poisson_ratio: F
    lower_young_modulus_pa: F
    lower_shear_modulus_pa: F
    source_lower_tensor_template_json: str
    material_model: str
    initial_strain_model: str
    lower_layer_extension_provenance: str
    calibration_provenance: str
    material_snapshot_sha256: str

    def lame_coefficients(self, layer):
        """Return exact (lambda, shear modulus) for the declared real layer."""
        if layer == "upper":
            E, nu = self.upper_young_modulus_pa, self.upper_poisson_ratio
            return E*nu/((1+nu)*(1-2*nu)), E/(2*(1+nu))
        if layer == "lower":
            E, G = self.lower_young_modulus_pa, self.lower_shear_modulus_pa
            nu = E/(2*G)-1
            return 2*G*nu/(1-2*nu), G
        raise ValueError("Cold tensor integration requires the explicit upper or lower layer")

    @property
    def nonlocal_constitutive_calibration_declared(self):
        return True

    @property
    def historical_full_three_dimensional_prestress_recovered(self):
        return False

    @property
    def historical_local_three_dimensional_tensor_inherited(self):
        return False

    @property
    def source_elastic_layer_thickness_used_as_areal_depth(self):
        return False


def build_native_cold_strain_material(body):
    """Bind source upper material and a NEW source-E/G lower-layer extension."""
    body = _body(body)
    source = load_native_normal_support_source()
    if (body.source_snapshot_sha256 != source.source_snapshot_sha256
            or body.source_support_sha256 != source.source_support_sha256):
        raise ValueError("Cold material declaration differs from the authentic normal-source snapshot")
    pairs = []
    templates = []
    for family in source.families:
        columns = json.loads(family.complete_carrier_source_json)["declaration"]["parameters"]["columns"]
        templates.append((family.family_id, columns))
        for side in ("sub", "over"):
            pairs.append((F.from_float(columns[side+"_normal_modulus_pa"]),
                          F.from_float(columns[side+"_shear_modulus_pa"])))
    if not pairs or len(set(pairs)) != 1:
        raise ValueError("A uniform lower-layer extension needs one authentic contact-source E/G declaration")
    E, G = pairs[0]
    upper_E = _positive_fraction(body.young_modulus_pa, "Source upper Young modulus")
    nu = body.poisson_ratio
    if type(nu) is not F or not -1 < nu < F(1, 2):
        raise ValueError("Source upper Poisson ratio must define a stable full tensor")
    lower_nu = E/(2*G)-1
    if E <= 0 or G <= 0 or not -1 < lower_nu < F(1, 2):
        raise ValueError("Authentic contact E/G does not define the new lower-layer tensor")
    value = NativeColdStrainMaterial(body.component_id, body.body_snapshot_sha256,
        source.source_snapshot_sha256, source.source_support_sha256, upper_E, nu, E, G,
        json.dumps(templates, sort_keys=True, separators=(",", ":"), allow_nan=False),
        MATERIAL_MODEL, INITIAL_STRAIN_MODEL,
        "Explicit NEW lower cold material extension of the authentic contact-source laterally free E/G tensor; no lower tensor was independently recovered",
        "Positive bank-wide rank-one strain projection preserves the old global beam ledger and reference tip force on the authentic remaining mask; no historical per-material archived strain-energy allocation is recovered",
        "")
    digest = hashlib.sha256(_encoded(_material_snapshot(value))).hexdigest()
    value = replace(value, material_snapshot_sha256=digest)
    _MATERIALS[digest] = _material_snapshot(value)
    return value


def validate_native_cold_strain_material(material, *, body=None):
    if type(material) is not NativeColdStrainMaterial:
        raise ValueError("A source-bound NativeColdStrainMaterial is required")
    snapshot = _material_snapshot(material)
    if (material.material_snapshot_sha256 != hashlib.sha256(_encoded(snapshot)).hexdigest()
            or _MATERIALS.get(material.material_snapshot_sha256) != snapshot):
        raise ValueError("An altered or re-sealed cold material cannot replace its authenticated declaration")
    if body is not None:
        body = _body(body)
        if (body.component_id != material.component_id
                or body.body_snapshot_sha256 != material.body_snapshot_sha256):
            raise ValueError("Cold material belongs to a different authentic supported bank")
    return material


@dataclass(frozen=True, slots=True)
class NativeColdStrainSupport:
    body_snapshot_sha256: str
    coordinates_m: tuple
    reference_radius_bounds_m: I
    reference_tip_gradient_norm_upper_per_m: F
    enrichment_gradient_norm_upper_per_m: F
    initial_metric_eigenvalue_bounds: I
    incremental_gradient_norm_upper: F
    incremental_jacobian_bounds: I
    total_elastic_metric_eigenvalue_bounds: I

    @property
    def initial_metric_positive_definite_certified(self):
        return self.initial_metric_eigenvalue_bounds.lo > 0

    @property
    def whole_mask_local_positive_jacobian_certified(self):
        return self.incremental_jacobian_bounds.lo > 0

    @property
    def total_elastic_metric_positive_definite_certified(self):
        return self.total_elastic_metric_eigenvalue_bounds.lo > 0

    @property
    def whole_mask_global_embedding_certified(self):
        return False

    @property
    def global_physical_solid_noninterpenetration_certified(self):
        return False


def certify_native_cold_strain_support(body, coordinates_m, *, precision=160):
    """Certify reference metric and incremental J by global gradient bounds.

    Unit rotations preserve vector norms.  The entire clamped support obeys
    |chi|<=1, |dchi/dbeta|<=3/(2b), and the full radial/depth flow is retained.
    The conservative cos(b) lower bound only widens the source estimate.
    These are local metric/J proofs; global pair admission is not inherited.
    """
    body = _body(body)
    coordinates = _coordinates(coordinates_m)
    source = load_native_normal_support_source()
    banks = {x.component_id: x for x in source.banks}
    bank = banks.get(body.component_id)
    if bank is None or bank.material_kind != "original_live":
        raise ValueError("Cold strain support requires an authentic original live bank")
    R, top = body.radius_m, body.radius_m+bank.roof_radial_displacement_m
    cube = top**3-3*R**2*body.areal_material_thickness_m
    if cube <= 0:
        raise ValueError("The authentic cold material depth reaches a nonpositive radius")
    inner = interval_cbrt(I(cube), precision)
    radius = I(inner.lo, top)
    reference = json.loads(source.saved_field_json)
    groups = tuple(g for g in reference["groups"] if body.component_id in g["component_ids"])
    if len(groups) != 1:
        raise ValueError("The authentic cold bank lacks one saved128 depth-flow group")
    family_ids = groups[0]["family_ids"]
    rates = tuple(x.shared_depth_axis_per_m for x in source.families if x.family_id in family_ids)
    if rates and len(set(rates)) != 1:
        raise ValueError("The cold reference depth flow is not one literal shared source")
    A = rates[0] if rates else (F(0), F(0), F(0))
    norm_A = interval_sqrt(I(sum((x*x for x in A), F(0))), precision).hi
    b = body.beta_cut_radians
    if type(b) is not F or not 0 < b <= F(1, 4):
        raise ValueError("Global cold gradient proof requires the declared source clamp angle <=1/4")
    cos_lower = 1-b*b/2
    beta_gradient = 1/radius.lo+(top**2/R**2)*norm_A/cos_lower
    tip_norm = 1/radius.lo+F(3, 2)/b*beta_gradient
    enrichment_norm = 1/R+F(3, 2)/b*(top/R)*beta_gradient
    initial_norm = 2*abs(body.accepted_tip_displacement_m)*tip_norm
    if initial_norm >= 1:
        raise ValueError("The declared initial cold strain does not certify a positive reference metric")
    increment_norm = abs(coordinates[0])*tip_norm+sum((abs(x)*enrichment_norm
        for x in coordinates[1:]), F(0))
    if increment_norm >= 1:
        raise ValueError("The proposed cold increments exceed the supported positive-J gradient domain")
    total_metric = I((1-increment_norm)**2-initial_norm,
                     (1+increment_norm)**2+initial_norm)
    if total_metric.lo <= 0:
        raise ValueError("The new initial-strain plus finite increment does not certify a positive elastic metric")
    return NativeColdStrainSupport(body.body_snapshot_sha256, coordinates, radius,
        tip_norm, enrichment_norm, I(1-initial_norm, 1+initial_norm), increment_norm,
        I((1-increment_norm)**3, (1+increment_norm)**3), total_metric)


@dataclass(frozen=True, slots=True)
class NativeColdStrainEnergyResponse:
    body: object
    material: NativeColdStrainMaterial
    raw_integrals: object
    support: NativeColdStrainSupport
    coordinates_m: tuple
    reference_energy_j: F
    reference_tip_force_n: F
    energy_bounds_j: I
    generalized_gradient_bounds_n: tuple
    generalized_hessian_bounds_n_m: tuple
    raw_green_energy_bounds_j: I
    projection_increment_m: I
    total_old_tip_strain_projection_m: I

    @property
    def nonlocal_constitutive_calibration_declared(self):
        return True

    @property
    def accepted_global_beam_reference_energy_preserved(self):
        return True

    @property
    def accepted_global_beam_reference_tip_force_preserved(self):
        return True

    @property
    def old_beam_stiffness_added_as_second_energy(self):
        return False

    @property
    def historical_full_three_dimensional_prestress_recovered(self):
        return False

    @property
    def historical_archived_strain_energy_density_recovered(self):
        return False

    @property
    def complete_three_dimensional_strain_energy_certified(self):
        return False

    @property
    def mixed_force_equilibrium_certified(self):
        return False

    @property
    def material_admission_authorized(self):
        return False

    @property
    def original_point_1_completed(self):
        return False


def _square(value):
    return I(F(0) if value.lo <= 0 <= value.hi else min(value.lo**2, value.hi**2),
             max(value.lo**2, value.hi**2))


def evaluate_native_cold_strain_energy(body, coordinates_m, *, material=None,
        integrals=None, precision=128, taylor_order=2, beta_cell_count=1,
        depth_cell_count=1):
    """Evaluate objective calibrated cold energy, all seven forces/Hessian.

    The integrator retains the correlated remainder Green(F)-q_tip*H0.
    Consequently the reference energy and source tip force are exact literal
    source identities, without dividing independently enclosed copies of k.
    The geometric prestress term remains in every nonlinear Hessian.
    """
    body = _body(body)
    coordinates = _coordinates(coordinates_m)
    if material is None:
        material = build_native_cold_strain_material(body)
    material = validate_native_cold_strain_material(material, body=body)
    support = certify_native_cold_strain_support(body, coordinates, precision=max(precision, 160))
    from .geometric_native_cold_tensor_integration import (
        integrate_native_cold_tensor_response, validate_native_cold_tensor_integral,
    )
    if integrals is None:
        integrals = integrate_native_cold_tensor_response(body, material, coordinates,
            precision=precision, taylor_order=taylor_order,
            beta_cell_count=beta_cell_count, depth_cell_count=depth_cell_count)
    raw = validate_native_cold_tensor_integral(integrals, body=body, material=material)
    if raw.coordinates_m != coordinates:
        raise ValueError("Cold tensor integrals describe different physical modal increments")
    k = raw.reference_tip_strain_stiffness_n_m
    if type(k) is not I or k.lo <= 0:
        raise ValueError("The actual cold old-tip full tensor norm must be positive without a floor")
    prem = raw.projection_remainder_n
    prem_gradient = raw.projection_remainder_gradient_n_m
    rem_norm = raw.remainder_strain_inner_product_j
    rem_gradient = raw.remainder_local_gradient_n
    rem_hessian = raw.remainder_local_hessian_n_m
    projection_hessian = raw.projection_hessian_n_m2
    if (len(prem_gradient) != 7 or len(rem_gradient) != 7
            or len(rem_hessian) != 7 or any(len(x) != 7 for x in rem_hessian)
            or len(projection_hessian) != 7 or any(len(x) != 7 for x in projection_hessian)):
        raise ValueError("All seven true cold strain modes require complete derivative tensors")
    old_k, wref = body.source_old_tip_stiffness_n_m, body.accepted_tip_displacement_m
    reference_energy, reference_force = old_k*wref*wref/2, old_k*wref
    projection_increment = I(coordinates[0])+prem/k
    total_projection = I(wref)+projection_increment
    modal_derivatives = tuple(I(int(i == 0))+prem_gradient[i]/k for i in range(7))
    residual_energy = rem_norm/2-_square(prem)/(2*k)
    # The exact residual is the C-norm of the orthogonal strain complement.
    # Intersecting its known nonnegative mathematical range is not a floor.
    residual_energy = I(max(F(0), residual_energy.lo), residual_energy.hi)
    energy = old_k*_square(total_projection)/2+residual_energy
    gradient = tuple(old_k*total_projection*modal_derivatives[i]+rem_gradient[i]
                     -(prem/k)*prem_gradient[i] for i in range(7))
    hessian = tuple(tuple(old_k*modal_derivatives[i]*modal_derivatives[j]
        +rem_hessian[i][j]-prem_gradient[i]*prem_gradient[j]/k
        +((old_k*total_projection-prem)/k)*projection_hessian[i][j]
        for j in range(7)) for i in range(7))
    if not any(coordinates):
        # These correlated source identities hold for the actual k itself.
        # Its unknown value is never replaced with the interval midpoint.
        energy = I(reference_energy)
        gradient = (I(reference_force),)+gradient[1:]
        first_row = tuple((I(old_k) if j == 0 else old_k*prem_gradient[j]/k)
            +(reference_force/k)*projection_hessian[0][j] for j in range(7))
        hessian = (first_row,)+tuple((first_row[i],)+hessian[i][1:] for i in range(1, 7))
    return NativeColdStrainEnergyResponse(body, material, raw, support, coordinates,
        reference_energy, reference_force, energy, gradient, hessian,
        raw.incremental_strain_inner_product_j/2, projection_increment, total_projection)


def native_cold_strain_material_to_dict(material):
    return _payload(validate_native_cold_strain_material(material))


def native_cold_strain_energy_to_dict(response):
    if type(response) is not NativeColdStrainEnergyResponse:
        raise ValueError("A genuine cold calibrated strain-energy response is required")
    _body(response.body)
    validate_native_cold_strain_material(response.material, body=response.body)
    result = _payload(response)
    result.update(nonlocal_constitutive_calibration_declared=True,
        accepted_global_beam_reference_energy_preserved=True,
        accepted_global_beam_reference_tip_force_preserved=True,
        old_beam_stiffness_added_as_second_energy=False,
        historical_full_three_dimensional_prestress_recovered=False,
        historical_archived_strain_energy_density_recovered=False,
        complete_three_dimensional_strain_energy_certified=False,
        mixed_force_equilibrium_certified=False, material_admission_authorized=False,
        original_point_1_completed=False)
    return result
