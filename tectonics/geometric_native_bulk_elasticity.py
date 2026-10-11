"""Explicit bulk elasticity for a new finite-body interpretation of HOT ports.

The saved column law declares effective normal and shear moduli, not a full
three-dimensional material tensor.  This module makes a NEW closure explicit:
interpret the normal modulus as laterally free Young's modulus, and use the
supplied shear modulus in a standard isotropic tensor.  Its Poisson ratio and
Lame coefficient are consequences of that declaration, not inherited source
authority or additional fitted material numbers.

The exact local energy includes every symmetric strain component.  The
finite-kinematic adapter uses objective Green strain and the same reference
tensor (Saint Venant--Kirchhoff).  Positive Jacobian and conserved reference
mass do not require constant physical volume.  These local formulas certify
neither a globally injective whole-mask placement nor physical admission.

The separate shared-bank condensation helper retains cross-contact coupling
of an explicitly supplied positive definite discrete body stiffness.  It
does not infer a material rod, normal length, boundary condition, integration
domain, or physical stiffness from a source hash.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from fractions import Fraction as F

from .geometric_contact_mechanics import ElasticColumnParameters


LATERAL_CLOSURE = "laterally_free_standard_isotropic"


def _text(value, name):
    if type(value) is not str or not value.strip():
        raise ValueError(name + " requires explicit nonempty source text")
    return value


def _sha(value, name):
    if (type(value) is not str or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(name + " requires a literal lowercase SHA-256")
    return value


def _fraction(value, name, *, positive=False):
    if type(value) is not F or (positive and value <= 0):
        raise ValueError(name + " requires an exact" + (" positive" if positive else "") + " Fraction")
    return value


def _matrix(value, name, *, size=None, width=None):
    if type(value) is not tuple or not value:
        raise ValueError(name + " requires a nonempty immutable exact matrix")
    if size is not None and len(value) != size:
        raise ValueError(name + " has an invalid row count")
    if width is None:
        if type(value[0]) is not tuple or not value[0]:
            raise ValueError(name + " requires nonempty immutable rows")
        width = len(value[0])
    if any(type(row) is not tuple or len(row) != width
            or any(type(entry) is not F for entry in row) for row in value):
        raise ValueError(name + " requires immutable equally sized exact Fraction rows")
    return value


def _symmetric(value, name, *, size=3):
    matrix = _matrix(value, name, size=size, width=size)
    if any(matrix[i][j] != matrix[j][i] for i in range(size) for j in range(i)):
        raise ValueError(name + " must be exactly symmetric")
    return matrix


def _dot(first, second):
    return sum((a*b for a, b in zip(first, second)), F(0))


def _contraction(first, second):
    return sum((_dot(a, b) for a, b in zip(first, second)), F(0))


def _transpose(matrix):
    return tuple(tuple(row[column] for row in matrix) for column in range(len(matrix[0])))


def _multiply(first, second):
    transposed = _transpose(second)
    return tuple(tuple(_dot(row, column) for column in transposed) for row in first)


def _determinant3(matrix):
    a, b, c = matrix
    return (a[0]*(b[1]*c[2]-b[2]*c[1])
            -a[1]*(b[0]*c[2]-b[2]*c[0])
            +a[2]*(b[0]*c[1]-b[1]*c[0]))


@dataclass(frozen=True, slots=True)
class NativeBulkElasticClosure:
    """A source-bound, explicitly NEW laterally free isotropic closure.

    ``source_bank`` selects the supplied ``sub`` or ``over`` moduli.  The
    original source columns, including their historical lengths, are retained
    verbatim.  Those lengths receive no finite-body geometry authority here.
    """
    source_columns: ElasticColumnParameters
    source_bank: str
    bank_source_id: str
    source_support_sha256: str
    lateral_closure_provenance: str
    lateral_closure: str = LATERAL_CLOSURE

    def __post_init__(self):
        if type(self.source_columns) is not ElasticColumnParameters:
            raise ValueError("Bulk closure requires the actual declared source column parameters")
        original = self.source_columns
        replay = ElasticColumnParameters(**{item.name: getattr(original, item.name)
                                            for item in fields(ElasticColumnParameters)})
        if original != replay:
            raise ValueError("Bulk closure source column parameters are not canonical")
        if type(self.source_bank) is not str or self.source_bank not in ("sub", "over"):
            raise ValueError("Bulk closure must retain an explicit sub or over source bank")
        _text(self.bank_source_id, "Actual bank source ID")
        _sha(self.source_support_sha256, "Actual support binding")
        _text(self.lateral_closure_provenance, "New lateral closure provenance")
        if type(self.lateral_closure) is not str or self.lateral_closure != LATERAL_CLOSURE:
            raise ValueError("Bulk closure requires the explicit new laterally free standard isotropic interpretation")
        if not -1 < self.poisson_ratio < F(1, 2):
            raise ValueError("The supplied moduli do not define a stable laterally free isotropic tensor")
        if self.shear_modulus_pa <= 0 or self.bulk_modulus_pa <= 0:
            raise ValueError("The new bulk tensor requires strictly positive shear and bulk moduli")

    @property
    def young_modulus_pa(self):
        return F.from_float(getattr(self.source_columns, self.source_bank + "_normal_modulus_pa"))

    @property
    def shear_modulus_pa(self):
        return F.from_float(getattr(self.source_columns, self.source_bank + "_shear_modulus_pa"))

    @property
    def poisson_ratio(self):
        return self.young_modulus_pa/(2*self.shear_modulus_pa)-1

    @property
    def lame_lambda_pa(self):
        nu = self.poisson_ratio
        return 2*self.shear_modulus_pa*nu/(1-2*nu)

    @property
    def bulk_modulus_pa(self):
        return self.lame_lambda_pa+2*self.shear_modulus_pa/3

    @property
    def new_lateral_closure_declared(self):
        return True

    @property
    def previous_three_dimensional_tensor_inherited(self):
        return False

    @property
    def source_column_lengths_are_actual_bank_geometry(self):
        return False

    @property
    def complete_three_dimensional_strain_energy_certified(self):
        return False

    @property
    def force_equilibrium_certified(self):
        return False

    @property
    def material_admission_authorized(self):
        return False


def validate_native_bulk_elastic_closure(closure):
    """Replay the declaration; a mutable or forged cached tensor is refused."""
    if type(closure) is not NativeBulkElasticClosure:
        raise ValueError("An explicit NativeBulkElasticClosure is required")
    replay = NativeBulkElasticClosure(**{item.name: getattr(closure, item.name)
                                        for item in fields(NativeBulkElasticClosure)})
    if closure != replay:
        raise ValueError("The declared bulk closure differs from its complete replay")
    return closure


@dataclass(frozen=True, slots=True)
class NativeBulkElasticResponse:
    closure: NativeBulkElasticClosure
    symmetric_strain: tuple
    stress_pa: tuple
    energy_density_j_m3: F
    first_variation_j_m3: F | None

    @property
    def complete_three_dimensional_strain_energy_certified(self):
        return False

    @property
    def actual_material_mask_integration_required(self):
        return True

    @property
    def force_equilibrium_certified(self):
        return False

    @property
    def material_admission_authorized(self):
        return False


def evaluate_native_bulk_elasticity(closure, symmetric_strain, strain_direction=None):
    """Evaluate all six strain modes and their exact energetic conjugates.

    Energy is per reference material volume.  Off-diagonal contributions occur
    twice in the tensor contraction; they are not engineering shear strains.
    """
    closure = validate_native_bulk_elastic_closure(closure)
    strain = _symmetric(symmetric_strain, "Full symmetric three-dimensional strain")
    trace = sum((strain[index][index] for index in range(3)), F(0))
    mu, lam = closure.shear_modulus_pa, closure.lame_lambda_pa
    stress = tuple(tuple(2*mu*strain[i][j]+(lam*trace if i == j else F(0))
                         for j in range(3)) for i in range(3))
    energy = mu*_contraction(strain, strain)+lam*trace*trace/2
    if energy < 0:
        raise ValueError("A stable bulk tensor produced negative strain energy")
    variation = None
    if strain_direction is not None:
        direction = _symmetric(strain_direction, "Symmetric strain first-variation direction")
        variation = _contraction(stress, direction)
    return NativeBulkElasticResponse(closure, strain, stress, energy, variation)


@dataclass(frozen=True, slots=True)
class NativeBulkDeformationResponse:
    closure: NativeBulkElasticClosure
    deformation_gradient: tuple
    green_strain: tuple
    second_piola_stress_pa: tuple
    first_piola_stress_pa: tuple
    cauchy_stress_pa: tuple
    energy_density_reference_j_m3: F
    energy_density_physical_j_m3: F
    first_variation_reference_j_m3: F | None
    volume_jacobian: F
    reference_density_kg_m3: F
    reference_volume_m3: F
    conserved_mass_kg: F
    physical_volume_m3: F
    current_density_kg_m3: F

    @property
    def local_positive_jacobian_verified(self):
        return True

    @property
    def reference_mass_conserved(self):
        return True

    @property
    def physical_volume_preservation_imposed(self):
        return False

    @property
    def reference_material_inventory_replaced(self):
        return False

    @property
    def finite_strain_model(self):
        return "explicit objective Saint Venant--Kirchhoff reference-tensor lift"

    @property
    def global_energy_convexity_certified(self):
        return False

    @property
    def complete_three_dimensional_strain_energy_certified(self):
        return False

    @property
    def actual_material_mask_integration_required(self):
        return True

    @property
    def whole_mask_embedding_certified(self):
        return False

    @property
    def force_equilibrium_certified(self):
        return False

    @property
    def material_admission_authorized(self):
        return False


def evaluate_native_bulk_deformation(closure, deformation_gradient, *,
        reference_density_kg_m3, reference_volume_m3,
        deformation_gradient_direction=None):
    """Lift the declared reference tensor objectively and transport mass.

    A positive local determinant excludes local rank loss only.  It does not
    prove injectivity, admissibility of large strains, absence of collisions,
    or integration over a source-bound physical body.
    """
    closure = validate_native_bulk_elastic_closure(closure)
    gradient = _matrix(deformation_gradient, "Actual local deformation gradient", size=3, width=3)
    density = _fraction(reference_density_kg_m3, "Reference material density", positive=True)
    volume = _fraction(reference_volume_m3, "Reference material volume", positive=True)
    jacobian = _determinant3(gradient)
    if jacobian <= 0:
        raise ValueError("Actual local deformation requires a strictly positive volume Jacobian")
    right_cauchy_green = _multiply(_transpose(gradient), gradient)
    strain = tuple(tuple((right_cauchy_green[i][j]-(F(1) if i == j else F(0)))/2
                         for j in range(3)) for i in range(3))
    elastic = evaluate_native_bulk_elasticity(closure, strain)
    piola = _multiply(gradient, elastic.stress_pa)
    spatial_numerator = _multiply(piola, _transpose(gradient))
    cauchy = tuple(tuple(value/jacobian for value in row) for row in spatial_numerator)
    variation = None
    if deformation_gradient_direction is not None:
        direction = _matrix(deformation_gradient_direction,
                            "Deformation-gradient first-variation direction", size=3, width=3)
        variation = _contraction(piola, direction)
    return NativeBulkDeformationResponse(closure, gradient, strain, elastic.stress_pa,
        piola, cauchy, elastic.energy_density_j_m3,
        elastic.energy_density_j_m3/jacobian, variation,
        jacobian, density, volume, density*volume, jacobian*volume, density/jacobian)


def _positive_definite_factor(stiffness):
    """Exact LDL^T factorization: no numerical floor or lost weak mode."""
    count = len(stiffness)
    stiffness = _symmetric(stiffness, "Shared-bank discrete stiffness", size=count)
    lower = [[F(int(i == j)) for j in range(count)] for i in range(count)]
    diagonal = []
    for column in range(count):
        pivot = stiffness[column][column]-sum((lower[column][j]**2*diagonal[j]
                                               for j in range(column)), F(0))
        if pivot <= 0:
            raise ValueError("Shared-bank stiffness must be strictly positive definite after explicit rigid-mode constraints")
        diagonal.append(pivot)
        for row in range(column+1, count):
            lower[row][column] = (stiffness[row][column]-sum((
                lower[row][j]*lower[column][j]*diagonal[j] for j in range(column)), F(0)))/pivot
    return tuple(tuple(row) for row in lower), tuple(diagonal)


def _factor_solve(lower, diagonal, right_hand_side):
    count = len(diagonal)
    forward = []
    for index in range(count):
        forward.append(right_hand_side[index]-_dot(lower[index][:index], forward))
    scaled = tuple(value/pivot for value, pivot in zip(forward, diagonal))
    answer = [F(0)]*count
    for index in range(count-1, -1, -1):
        answer[index] = scaled[index]-sum((lower[j][index]*answer[j]
                                          for j in range(index+1, count)), F(0))
    return tuple(answer)


@dataclass(frozen=True, slots=True)
class NativeSharedBankCondensation:
    """One declared finite-body operator serving all its face functionals."""
    stiffness: tuple
    face_operators: tuple
    face_ids: tuple
    bank_source_id: str
    source_support_sha256: str
    geometry_source_sha256: str
    provenance: str
    unit_face_load_displacements_m_n: tuple
    compliance_m_n: tuple

    @property
    def shared_bank_cross_contact_terms_retained(self):
        return True

    @property
    def material_column_length_inferred(self):
        return False

    @property
    def supplied_stiffness_authenticates_actual_body(self):
        return False

    @property
    def actual_material_mask_integration_required(self):
        return True

    @property
    def complete_three_dimensional_strain_energy_certified(self):
        return False

    @property
    def force_equilibrium_certified(self):
        return False

    @property
    def material_admission_authorized(self):
        return False


def condense_shared_bank_elasticity(*, stiffness, face_operators, face_ids,
        bank_source_id, source_support_sha256, geometry_source_sha256, provenance):
    """Compute B K^-1 B^T exactly, with every declared face kept together.

    K [N/m] is an independently supplied body stiffness after its explicit
    rigid-mode constraints.  Rows of dimensionless B give signed face
    displacement functionals.  The result [m/N] is a positive semidefinite
    operator; opposite face orientations may give negative cross entries.
    Caller-supplied hashes identify declarations and do not validate their
    physical domains or boundary conditions.
    """
    stiffness = _matrix(stiffness, "Shared-bank discrete stiffness")
    count = len(stiffness)
    if any(len(row) != count for row in stiffness):
        raise ValueError("Shared-bank discrete stiffness must be square")
    operators = _matrix(face_operators, "Complete shared-bank face operators", width=count)
    if (type(face_ids) is not tuple or len(face_ids) != len(operators)
            or any(type(identity) is not str or not identity.strip() for identity in face_ids)
            or len(set(face_ids)) != len(face_ids)):
        raise ValueError("All shared-bank faces require unique immutable physical IDs")
    _text(bank_source_id, "Actual shared-bank source ID")
    _sha(source_support_sha256, "Actual shared-bank support binding")
    _sha(geometry_source_sha256, "Declared finite-body geometry source binding")
    _text(provenance, "Explicit body stiffness and boundary-condition provenance")
    lower, diagonal = _positive_definite_factor(stiffness)
    displacements = tuple(_factor_solve(lower, diagonal, row) for row in operators)
    compliance = tuple(tuple(_dot(row, displacement) for displacement in displacements)
                       for row in operators)
    if any(compliance[i][j] != compliance[j][i]
            for i in range(len(compliance)) for j in range(i)):
        raise ValueError("Exact shared-bank condensation lost energetic symmetry")
    return NativeSharedBankCondensation(stiffness, operators, face_ids,
        bank_source_id, source_support_sha256, geometry_source_sha256, provenance,
        displacements, compliance)


def validate_native_shared_bank_condensation(condensation):
    if type(condensation) is not NativeSharedBankCondensation:
        raise ValueError("An explicit NativeSharedBankCondensation is required")
    replay = condense_shared_bank_elasticity(**{name: getattr(condensation, name) for name in
        ("stiffness", "face_operators", "face_ids", "bank_source_id",
         "source_support_sha256", "geometry_source_sha256", "provenance")})
    if condensation != replay:
        raise ValueError("Shared-bank condensation differs from its complete exact replay")
    return condensation


@dataclass(frozen=True, slots=True)
class NativeSharedBankLoadResponse:
    condensation: NativeSharedBankCondensation
    face_loads_n: tuple
    body_displacements_m: tuple
    face_displacements_m: tuple
    body_elastic_gradient_n: tuple
    strain_energy_j: F
    force_residual_n: tuple

    @property
    def discrete_declared_operator_equilibrium_verified(self):
        return not any(self.force_residual_n)

    @property
    def complete_three_dimensional_strain_energy_certified(self):
        return False

    @property
    def force_equilibrium_certified(self):
        return False

    @property
    def material_admission_authorized(self):
        return False


def evaluate_shared_bank_face_loads(condensation, face_loads_n):
    """Recover the shared displacement and exact operator work identity."""
    condensation = validate_native_shared_bank_condensation(condensation)
    if (type(face_loads_n) is not tuple or len(face_loads_n) != len(condensation.face_ids)
            or any(type(value) is not F for value in face_loads_n)):
        raise ValueError("Shared-bank loads require one exact signed force per declared face")
    count = len(condensation.stiffness)
    displacement = tuple(sum((force*row[index] for force, row in
        zip(face_loads_n, condensation.unit_face_load_displacements_m_n)), F(0))
        for index in range(count))
    faces = tuple(_dot(row, displacement) for row in condensation.face_operators)
    gradient = tuple(_dot(row, displacement) for row in condensation.stiffness)
    loads = tuple(sum((row[index]*force for row, force in
        zip(condensation.face_operators, face_loads_n)), F(0)) for index in range(count))
    residual = tuple(first-second for first, second in zip(gradient, loads))
    energy = _dot(displacement, gradient)/2
    if any(residual) or energy < 0 or energy != _dot(face_loads_n, faces)/2:
        raise ValueError("The complete shared-bank operator lost force or work consistency")
    return NativeSharedBankLoadResponse(condensation, face_loads_n, displacement,
        faces, gradient, energy, residual)
