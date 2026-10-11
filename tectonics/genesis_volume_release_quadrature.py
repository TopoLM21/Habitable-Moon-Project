"""Restricted compact fields contracted before their source quadrature Grams.

The literal archived point fields remain the source.  Long-double arithmetic
preserves weak represented directions that can disappear when an already
rounded eighteen-column Gram is rotated.  This certifies computational
operator arithmetic, not weak geometry or clipped-bulk physical accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json

import numpy as np


SCOPE = "literal_archived_float64_point_fields_restricted_before_Gram_longdouble_computational_arithmetic_only"


class RestrictedSourceQuadratureRefusal(ValueError):
    """A restricted source field is nonfinite or has changed its binding."""


def _immutable(value):
    a = np.asarray(value)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def restricted_point_gram(left, right, weights):
    """Integrate literal point-vector inner products without a norm/weight floor."""
    a, b, w = (np.asarray(v, np.longdouble) for v in (left, right, weights))
    if (a.ndim != 3 or b.ndim != 3 or a.shape[:2] != b.shape[:2]
            or w.shape != (len(a),) or not np.isfinite(a).all() or not np.isfinite(b).all()
            or not np.isfinite(w).all() or np.any(w <= 0.)):
        raise RestrictedSourceQuadratureRefusal("Source point Gram fields or weights differ")
    out = np.einsum("pai,paj,p->ij", a, b, w)
    if not np.isfinite(out).all():
        raise RestrictedSourceQuadratureRefusal("Restricted point Gram overflowed")
    return out


def _numeric_pin(value):
    # Public descriptions use numerical float64 values, never long-double
    # padding bytes.  Live arrays are owned read-only buffers.
    a = np.ascontiguousarray(value, dtype=float)
    return hashlib.sha256(repr(a.shape).encode()+a.tobytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class RestrictedSourceQuadrature:
    source_binding_sha256: str
    basis_w: np.ndarray
    G_w: np.ndarray
    E_w: np.ndarray
    parent_strain_operator: np.ndarray
    parent_trace: np.ndarray
    reference_volume_weights_m3: np.ndarray
    elasticity_pa: np.ndarray
    density_kg_m3: float
    mass_cross: np.ndarray
    mass_enriched: np.ndarray
    stiffness_cross: np.ndarray
    stiffness_enriched: np.ndarray
    surface_trace_local_w: np.ndarray
    surface_area_weights_m2: np.ndarray
    surface_gram: np.ndarray
    binding_sha256: str

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, np.ndarray):
                if not np.isfinite(value).all():
                    raise RestrictedSourceQuadratureRefusal("Restricted source array is not finite")
                object.__setattr__(self, f.name, _immutable(value))


def _body(q):
    return dict(format="genesis-release-restricted-source-quadrature-0.1", scope=SCOPE,
        source_binding_sha256=q.source_binding_sha256, density_kg_m3=q.density_kg_m3,
        array_sha256={f.name: _numeric_pin(getattr(q, f.name)) for f in fields(q)
            if isinstance(getattr(q, f.name), np.ndarray)},
        represented_source_field_dtype="archived_float64",
        contraction_arithmetic="longdouble_before_Gram",
        weak_geometry_accuracy_certified=False, source_selected_clipped_bulk_accuracy_certified=False,
        physical_release_admitted=False, funding_admitted=False)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def prepare_restricted_source_quadrature(context):
    """Use the caller's separately validated immutable source release context."""
    p, l = context.partition, context.partition.local
    w = np.asarray(context.observable_force_basis, np.longdouble)
    g = np.einsum("pai,ij->paj", np.asarray(l.enrichment_strain_operator, np.longdouble), w)
    e = np.einsum("pai,ij->paj", np.asarray(l.enrichment_trace, np.longdouble), w)
    b, t, weights, elasticity = (np.asarray(v, np.longdouble) for v in (
        l.parent_strain_operator, l.parent_trace, l.reference_volume_weights_m3, l.elasticity_pa))
    cg = np.einsum("ab,pbi->pai", elasticity, g)
    density = float(l.section.mass_kg/l.section.volume_m3)
    if hasattr(context, "operative_cohesive"):
        surface_trace = np.asarray(context.operative_cohesive.compact_jump_trace_local, np.longdouble)
    else:
        surface_trace = np.einsum("pai,ij->paj", np.asarray(
            context.cohesive.compact_jump_trace_local, np.longdouble), w)
    area = np.asarray(context.cohesive.reference_area_weights_m2, np.longdouble)
    values = dict(source_binding_sha256=p.binding_sha256, basis_w=np.asarray(w, float),
        G_w=g, E_w=e, parent_strain_operator=b, parent_trace=t,
        reference_volume_weights_m3=weights, elasticity_pa=elasticity, density_kg_m3=density,
        mass_cross=np.longdouble(density)*restricted_point_gram(t, e, weights),
        mass_enriched=np.longdouble(density)*restricted_point_gram(e, e, weights),
        stiffness_cross=restricted_point_gram(b, cg, weights),
        stiffness_enriched=restricted_point_gram(g, cg, weights),
        surface_trace_local_w=surface_trace, surface_area_weights_m2=area,
        surface_gram=restricted_point_gram(surface_trace, surface_trace, area))
    temporary = RestrictedSourceQuadrature(**values, binding_sha256="")
    return RestrictedSourceQuadrature(**values, binding_sha256=_digest(_body(temporary)))


def restricted_source_quadrature_to_dict(value):
    if type(value) is not RestrictedSourceQuadrature or value.binding_sha256 != _digest(_body(value)):
        raise RestrictedSourceQuadratureRefusal("Restricted source quadrature binding differs")
    return {**_body(value), "source_quadrature_sha256": value.binding_sha256}
