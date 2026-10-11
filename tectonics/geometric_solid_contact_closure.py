"""Exact local elastic closure, distinct from finite solid placement.

The unbonded normal branch consists of the two independently declared elastic
columns and the existing normal contact compliance in series. A negative free
anchor gap is elastic shortening, not a negative physical void. Describing the
contact compliance as compacted material requires an independently declared
rest thickness; neither preload nor spring stiffness supplies that length.

This module does not embed columns in three dimensions, move material, replay a
return map, or authorize archiving. In particular, legitimate reverse shear
motion supplies no permission for a new solid underpass. Opening here is the
unbonded, zero-tension branch, not the cohesive opening branch of ContactLaw.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from fractions import Fraction as F
import math

from .geometric_contact_mechanics import ElasticColumnParameters
from .genesis_contact_law import ContactLawParameters


def _fraction(value, name, *, positive=False, nonnegative=False):
    if type(value) is F:
        result = value
    elif type(value) is int:
        result = F(value)
    elif type(value) is float and math.isfinite(value):
        result = F.from_float(value)
    else:
        raise ValueError(f"{name} must be an exact Fraction or a finite literal number")
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _parameters(columns, law):
    if type(columns) is not ElasticColumnParameters or type(law) is not ContactLawParameters:
        raise ValueError("Closure requires the independently declared column and contact parameter types")
    for item in fields(columns):
        _fraction(getattr(columns, item.name), item.name, positive=True)
    # Reconstructing the declared columns also checks their finite condensed
    # stiffness. Parameter instances forged by bypassing __init__ are refused.
    if ElasticColumnParameters(**{item.name: getattr(columns, item.name)
            for item in fields(columns)}) != columns:
        raise ValueError("The declared elastic columns are not canonical")
    law.validate()
    for item in fields(law):
        _fraction(getattr(law, item.name), item.name, nonnegative=True)
    return columns, law


@dataclass(frozen=True, slots=True)
class SolidNormalClosure:
    columns: ElasticColumnParameters
    law: ContactLawParameters
    free_gap_m: F
    contact_zone_rest_normal_thickness_m: F | None
    sub_normal_compliance_m_pa: F
    over_normal_compliance_m_pa: F
    fault_normal_compliance_m_pa: F
    effective_normal_stiffness_pa_m: F
    normal_pressure_pa: F
    physical_face_void_m: F
    sub_column_shortening_m: F
    over_column_shortening_m: F
    fault_indentation_m: F
    constitutive_contact_gap_m: F
    deformed_sub_column_length_m: F
    deformed_over_column_length_m: F
    deformed_contact_zone_thickness_m: F | None
    sub_column_energy_j_m2: F
    over_column_energy_j_m2: F
    fault_energy_j_m2: F
    total_energy_j_m2: F
    anchor_balance_residual_m: F
    energy_balance_residual_j_m2: F
    branch: str

    @property
    def material_admission_authorized(self):
        return False

    @property
    def finite_solid_placement_authorized(self):
        return False

    @property
    def physical_geometry_authorized(self):
        return False

    @property
    def contact_zone_geometry_declared(self):
        return self.contact_zone_rest_normal_thickness_m is not None


def _validate_normal_result(normal):
    if type(normal) is not SolidNormalClosure:
        raise ValueError("Expected an exact local normal closure")
    optional = {"contact_zone_rest_normal_thickness_m", "deformed_contact_zone_thickness_m"}
    for item in fields(normal):
        if item.name in ("columns", "law", "branch"):
            continue
        value = getattr(normal, item.name)
        if type(value) is not F and not (item.name in optional and value is None):
            raise ValueError("Stored normal closure must retain exact Fraction fields")
    if type(normal.branch) is not str:
        raise ValueError("Stored normal closure branch must be explicit text")


def solve_solid_normal_closure(*, columns, law, free_gap_m,
        contact_zone_rest_normal_thickness_m):
    """Solve exact unbonded series compression or free opening per unit area.

    Every physical compression must remain strictly smaller than the declared
    rest length of its own element. The supplied contact-zone rest thickness
    is an independent geometry declaration, not a new spring parameter. The
    returned energies are endpoint recoverable energies, not newly paid work.
    """
    normal = compute_series_normal_closure(columns=columns, law=law,
        free_gap_m=free_gap_m)
    return require_contact_zone_compaction(normal,
        contact_zone_rest_normal_thickness_m=contact_zone_rest_normal_thickness_m)


def compute_series_normal_closure(*, columns, law, free_gap_m):
    """Recover the genuine series law without inventing a contact-zone depth.

    The physical void of the unbonded reduced contact is zero in compression;
    its negative constitutive gap remains a compliant-element indentation.
    This algebra alone supplies no spatial embedding or contact-zone volume.
    ``contact_zone_geometry_declared`` is False until an independent rest
    thickness has been supplied to ``require_contact_zone_compaction``.
    """
    columns, law = _parameters(columns, law)
    gap = _fraction(free_gap_m, "Free normal anchor gap")
    sub_length = _fraction(columns.sub_length_m, "Sub column length", positive=True)
    over_length = _fraction(columns.over_length_m, "Over column length", positive=True)
    sub = sub_length/_fraction(columns.sub_normal_modulus_pa, "Sub normal modulus", positive=True)
    over = over_length/_fraction(columns.over_normal_modulus_pa, "Over normal modulus", positive=True)
    fault = 1/_fraction(law.normal_stiffness_pa_m, "Normal contact stiffness", positive=True)
    compliance = sub+over+fault
    pressure = -min(gap, F(0))/compliance
    void = max(gap, F(0))
    shortening = tuple(pressure*value for value in (sub, over, fault))
    deformed = tuple(length-displacement for length, displacement in
        zip((sub_length, over_length), shortening[:2]))
    if any(value <= 0 for value in deformed):
        raise ValueError("Normal compaction must be smaller than every independently declared rest thickness")
    energies = tuple(pressure*pressure*value/2 for value in (sub, over, fault))
    energy = sum(energies, F(0))
    return SolidNormalClosure(columns, law, gap, None, sub, over, fault,
        1/compliance, pressure, void, *shortening, void-shortening[2], *deformed, None,
        *energies, energy, void-sum(shortening, F(0))-gap,
        energy-pressure*(-min(gap, F(0)))/2,
        "compressed_compliant_material" if gap < 0 else "open_unbonded" if gap > 0 else "touching_unloaded")


def require_contact_zone_compaction(normal, *, contact_zone_rest_normal_thickness_m):
    """Require an independently declared positive rest length for compaction.

    A valid local series solution need not have such a geometry declaration.
    Supplying a preload, crust thickness or stiffness instead does not create
    one. Callers must retain the actual source of this independent length.
    The returned local length check still does not authorize a 3-D underpass.
    """
    _validate_normal_result(normal)
    replay = compute_series_normal_closure(columns=normal.columns, law=normal.law,
        free_gap_m=normal.free_gap_m)
    without_geometry = replace(normal, contact_zone_rest_normal_thickness_m=None,
        deformed_contact_zone_thickness_m=None)
    if replay != without_geometry:
        raise ValueError("Stored normal closure differs from independently declared elastic physics")
    rest = _fraction(contact_zone_rest_normal_thickness_m,
        "Independent contact-zone rest normal thickness", positive=True)
    if normal.contact_zone_rest_normal_thickness_m is not None:
        if (normal.contact_zone_rest_normal_thickness_m != rest
                or normal.deformed_contact_zone_thickness_m != rest-normal.fault_indentation_m):
            raise ValueError("The independently declared contact-zone rest thickness cannot be rebased")
    remaining = rest-normal.fault_indentation_m
    if remaining <= 0:
        raise ValueError("Normal compaction must be smaller than every independently declared rest thickness")
    return replace(normal, contact_zone_rest_normal_thickness_m=rest,
        deformed_contact_zone_thickness_m=remaining)


@dataclass(frozen=True, slots=True)
class SolidGeometryAnchorCheck:
    free_gap_m: F
    reconstructed_free_gap_m: F
    anchor_residual_m: F
    sub_shortening_residual_m: F
    over_shortening_residual_m: F
    fault_indentation_residual_m: F
    physical_void_residual_m: F
    tolerance_m: F

    @property
    def material_admission_authorized(self):
        return False

    @property
    def finite_solid_placement_authorized(self):
        return False


def geometry_to_anchor(*, normal, physical_face_void_m,
        sub_column_shortening_m, over_column_shortening_m,
        fault_indentation_m, tolerance_m=F(0)):
    """Check independently supplied physical geometry against the actual law.

    Compatibility is ``g_free = void - c_sub - c_over - c_fault``. Matching
    just that sum is insufficient: each element must reproduce the pressure
    partition and positive remaining rest length of the canonical solution.
    The explicit tolerance is a geometric representation allowance, never a
    replacement preload or a material-admission certificate.
    """
    _validate_normal_result(normal)
    replay = compute_series_normal_closure(columns=normal.columns, law=normal.law,
        free_gap_m=normal.free_gap_m)
    if normal.contact_zone_rest_normal_thickness_m is not None:
        replay = require_contact_zone_compaction(replay,
            contact_zone_rest_normal_thickness_m=normal.contact_zone_rest_normal_thickness_m)
    if replay != normal:
        raise ValueError("Stored normal closure differs from independently declared elastic physics")
    void = _fraction(physical_face_void_m, "Physical face void", nonnegative=True)
    sub = _fraction(sub_column_shortening_m, "Sub column shortening", nonnegative=True)
    over = _fraction(over_column_shortening_m, "Over column shortening", nonnegative=True)
    fault = _fraction(fault_indentation_m, "Fault indentation", nonnegative=True)
    tolerance = _fraction(tolerance_m, "Geometry-to-anchor tolerance", nonnegative=True)
    if (sub >= _fraction(normal.columns.sub_length_m, "Sub column length")
            or over >= _fraction(normal.columns.over_length_m, "Over column length")
            or (normal.contact_zone_rest_normal_thickness_m is not None
                and fault >= normal.contact_zone_rest_normal_thickness_m)):
        raise ValueError("The supplied geometry consumes an independently declared rest thickness")
    reconstructed = void-sub-over-fault
    residuals = (reconstructed-normal.free_gap_m,
        sub-normal.sub_column_shortening_m, over-normal.over_column_shortening_m,
        fault-normal.fault_indentation_m, void-normal.physical_face_void_m)
    if any(abs(value) > tolerance for value in residuals):
        raise ValueError("Physical geometry does not reproduce the free anchor and elastic pressure partition")
    return SolidGeometryAnchorCheck(normal.free_gap_m, reconstructed, *residuals, tolerance)


@dataclass(frozen=True, slots=True)
class SolidTangentialClosure:
    columns: ElasticColumnParameters
    law: ContactLawParameters
    free_jump_m: F
    plastic_slip_m: F
    sub_shear_compliance_m_pa: F
    over_shear_compliance_m_pa: F
    fault_shear_compliance_m_pa: F
    shear_traction_pa: F
    sub_column_shear_m: F
    over_column_shear_m: F
    fault_elastic_shear_m: F
    constitutive_contact_jump_m: F
    sub_column_energy_j_m2: F
    over_column_energy_j_m2: F
    fault_energy_j_m2: F
    total_energy_j_m2: F
    anchor_balance_residual_m: F

    @property
    def material_admission_authorized(self):
        return False


def solve_solid_tangential_closure(*, columns, law, free_jump_m, plastic_slip_m):
    """Recover exact elastic compatibility at independently supplied slip.

    ``j_free = slip + tau*(L_sub/G_sub + L_over/G_over + 1/Kt)``.
    This is not a Coulomb/viscous return map and does not assert slip-history
    admissibility. Its traction must be checked against the actual evolved
    constitutive state. Signed slip and deformation can move in either sense.
    """
    columns, law = _parameters(columns, law)
    jump = _fraction(free_jump_m, "Free tangential anchor jump")
    slip = _fraction(plastic_slip_m, "Actual signed plastic slip")
    sub = _fraction(columns.sub_length_m, "Sub column length")/_fraction(
        columns.sub_shear_modulus_pa, "Sub shear modulus", positive=True)
    over = _fraction(columns.over_length_m, "Over column length")/_fraction(
        columns.over_shear_modulus_pa, "Over shear modulus", positive=True)
    fault = 1/_fraction(law.tangential_stiffness_pa_m, "Tangential contact stiffness", positive=True)
    traction = (jump-slip)/(sub+over+fault)
    displacements = tuple(traction*value for value in (sub, over, fault))
    energies = tuple(traction*traction*value/2 for value in (sub, over, fault))
    return SolidTangentialClosure(columns, law, jump, slip, sub, over, fault,
        traction, *displacements, slip+displacements[2], *energies,
        sum(energies, F(0)), slip+sum(displacements, F(0))-jump)
