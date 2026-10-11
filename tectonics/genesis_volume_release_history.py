"""Source-born native release coordinates beside the unchanged full18 archive.

The fifteen (or other resolved-rank) coordinates are stored independently.
Expanding large weak coordinates to float64 full18 is an arithmetic witness,
never a way to recover or reinitialize an authoritative mature history.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from .genesis_volume_checkpoint import _canonical
from .genesis_volume_creation_law import _numeric, _immutable
from .genesis_volume_enriched_memory import enriched_memory_to_dict
from .genesis_volume_release_quadrature import prepare_restricted_source_quadrature
from .genesis_volume_seed_memory import restart_source_seed_memory


FORMAT = "genesis-source-born-native-release-history-0.1"
SCOPE = "source_born_authoritative_W_coordinates_fixed_material_private_journal"
SECONDS_PER_MYR = 1e6*365.25*24*3600


class NativeReleaseHistoryRefusal(ValueError):
    """Native history does not continue its authentic zero birth and source."""


def _digest(body):
    return hashlib.sha256(_canonical(body)).hexdigest()


def _array(value, shape, name):
    a = _numeric(value, name)
    if a.shape != shape:
        raise NativeReleaseHistoryRefusal(name+" shape differs from native source")
    return a


def expand_native_release_increment(context, increment_native_m):
    p, r = context.partition, context.modes.rank
    d = _array(increment_native_m, (p.parent_ndof+r,), "Native increment")
    expanded = np.asarray(np.asarray(context.observable_force_basis, np.longdouble)
        @np.asarray(d[p.parent_ndof:], np.longdouble), float)
    return np.r_[d[:p.parent_ndof], expanded]


@dataclass(frozen=True, slots=True)
class NativeReleaseState:
    displacement_w_m: np.ndarray
    coefficients_w_m: np.ndarray
    velocity_w_m_s: np.ndarray
    time_myr: float
    journal_json: str
    full18_memory_sha256: str
    context_binding_sha256: str
    payload_json: str
    binding_sha256: str

    def __post_init__(self):
        for name in ("displacement_w_m", "coefficients_w_m", "velocity_w_m_s"):
            object.__setattr__(self, name, _immutable(np.asarray(getattr(self, name), float)))

    @property
    def journal(self):
        # Decoded copies cannot alter the live immutable chain.
        return tuple(json.loads(self.journal_json))

    @property
    def physical_release_admitted(self): return False

    @property
    def funding_admitted(self): return False


def _make(context, full_memory_sha256, c, z, v, time, journal):
    q = prepare_restricted_source_quadrature(context)
    body = dict(format=FORMAT, scope=SCOPE,
        context_binding_sha256=context.binding_sha256,
        source_memory_birth_binding_sha256=context.birth.binding_sha256,
        source_initial_full18_memory_sha256=context.birth.initial_memory.binding_sha256,
        source_quadrature_binding_sha256=q.binding_sha256,
        full18_memory_sha256=full_memory_sha256,
        native_rank=context.modes.rank, accepted_source_birth_time_myr=context.partition.time_myr,
        time_myr=float(time), displacement_w_m=np.asarray(c, float).tolist(),
        coefficients_w_m=np.asarray(z, float).tolist(), velocity_w_m_s=np.asarray(v, float).tolist(),
        journal=journal, native_birth_from_authentic_zero_enrichment=True,
        mature_full18_projection_used_as_native_initialization=False,
        full18_fields_scope="expanded_float64_arithmetic_witness_only_not_authoritative_coordinates",
        cohort_parent_memory_scope="unchanged_original_source_Maxwell_parent_archive",
        accepted_planet_memory_changed=False, physical_release_admitted=False, funding_admitted=False,
        new_accepted_Genesis_time_myr=0., creation_work_debited_from_Genesis_j=0.,
        historical_dissipation_available_for_creation_j=0.)
    payload = _canonical({**body, "native_release_state_sha256": _digest(body)}).decode()
    return NativeReleaseState(c, z, v, float(time), _canonical(journal).decode(),
        full_memory_sha256, context.binding_sha256, payload, hashlib.sha256(payload.encode()).hexdigest())


def native_release_state_to_dict(state):
    if type(state) is not NativeReleaseState:
        raise NativeReleaseHistoryRefusal("Typed source-born native history is required")
    raw = json.loads(state.payload_json)
    if (state.binding_sha256 != hashlib.sha256(state.payload_json.encode()).hexdigest()
            or raw.get("native_release_state_sha256") != _digest({k: v for k, v in raw.items()
                if k != "native_release_state_sha256"})
            or raw.get("context_binding_sha256") != state.context_binding_sha256
            or raw.get("full18_memory_sha256") != state.full18_memory_sha256
            or raw.get("time_myr") != state.time_myr
            or raw.get("journal") != list(state.journal)
            or any(not np.array_equal(raw.get(name), getattr(state, name)) for name in (
                "displacement_w_m", "coefficients_w_m", "velocity_w_m_s"))):
        raise NativeReleaseHistoryRefusal("Immutable native release history changed")
    return raw


def make_native_release_birth(context):
    from .genesis_volume_release_work import validate_volume_release_work
    validate_volume_release_work(context)
    m = context.birth.initial_memory
    if m.journal or np.any(m.displacement_m != 0.) or np.any(m.coefficients_m != 0.):
        raise NativeReleaseHistoryRefusal("Native birth requires the authentic zero enriched source birth")
    zero = np.zeros(context.modes.rank)
    return _make(context, m.binding_sha256, zero, zero, zero, m.time_myr, [])


def _rebuild_native_release_state(context, raw, full_memory):
    """The caller has separately replayed the unchanged full18 source archive."""
    if type(raw) is not dict or raw.get("format") != FORMAT:
        raise NativeReleaseHistoryRefusal("Native release history packet is required")
    journal = raw.get("journal")
    if not isinstance(journal, list) or len(journal) != len(full_memory.journal):
        raise NativeReleaseHistoryRefusal("Native and unchanged full18 journal lengths differ")
    r, p = context.modes.rank, context.partition
    c, z, v, time = np.zeros(r), np.zeros(r), np.zeros(r), p.time_myr
    checked = []
    for step, archive in zip(journal, full_memory.journal):
        dw = _array(step.get("increment_w_m"), (r,), "Journal native increment")
        dq = _array(step.get("parent_increment_m"), (p.parent_ndof,), "Journal original parent increment")
        expanded = expand_native_release_increment(context, np.r_[dq, dw])[p.parent_ndof:]
        after = float(_array(step.get("after_time_myr"), (), "Journal native after time"))
        dt = (after-time)*SECONDS_PER_MYR
        x = float(archive.ratio[p.selected_cohort_index])
        alpha = float(np.exp(-x))
        beta = float(-np.expm1(-x)/x) if x else 1.
        expected = dict(after_time_myr=after, dt_s=dt, parent_increment_m=dq.tolist(),
            increment_w_m=dw.tolist(), source_expanded_increment_full18_m=expanded.tolist(),
            selected_maxwell_ratio=x, selected_maxwell_alpha=alpha, selected_maxwell_beta=beta)
        if (not np.isfinite(dt) or dt <= 0. or after != archive.after_time_myr
                or not np.array_equal(dq, archive.parent_increment_m)
                or not np.array_equal(expanded, archive.enrichment_increment_m)
                or _canonical(step) != _canonical(expected)):
            raise NativeReleaseHistoryRefusal("Native step differs from original source Maxwell or expanded full18 journal")
        c = np.asarray(c+dw, float)
        z = np.asarray(alpha*z+beta*dw, float)
        v = np.asarray(2*dw/dt-v, float)
        time = after
        checked.append(expected)
    if not journal and full_memory.binding_sha256 != context.birth.initial_memory.binding_sha256:
        raise NativeReleaseHistoryRefusal("Native zero birth cannot bind a mature full18 state")
    expected = _make(context, full_memory.binding_sha256, c, z, v, time, checked)
    if _canonical(raw) != _canonical(native_release_state_to_dict(expected)):
        raise NativeReleaseHistoryRefusal("Native coordinates differ from genuine zero-birth journal replay")
    return expected


def restart_native_release_state(context, raw, full_memory):
    from .genesis_volume_release_work import validate_volume_release_work
    validate_volume_release_work(context)
    source = restart_source_seed_memory(context.birth, enriched_memory_to_dict(full_memory))
    if enriched_memory_to_dict(source) != enriched_memory_to_dict(full_memory):
        raise NativeReleaseHistoryRefusal("Full18 witness does not replay from authentic source birth")
    return _rebuild_native_release_state(context, raw, full_memory)


def validate_native_release_state(context, state, full_memory):
    return restart_native_release_state(context, native_release_state_to_dict(state), full_memory)


def _advance_native_release_state(context, before, full_after_memory, increment_native_m, loading):
    """Append one verified source-expanded step; never accepts mature initials."""
    raw = native_release_state_to_dict(before)
    p, r = context.partition, context.modes.rank
    d = _array(increment_native_m, (p.parent_ndof+r,), "Native increment")
    dq, dw = d[:p.parent_ndof], d[p.parent_ndof:]
    time = full_after_memory.time_myr
    dt = (time-before.time_myr)*SECONDS_PER_MYR
    ci = p.selected_cohort_index
    x, alpha, beta = float(loading.ratio[ci]), float(np.exp(-loading.ratio[ci])), float(loading.beta[ci])
    expanded = expand_native_release_increment(context, d)[p.parent_ndof:]
    step = dict(after_time_myr=time, dt_s=dt, parent_increment_m=dq.tolist(),
        increment_w_m=dw.tolist(), source_expanded_increment_full18_m=expanded.tolist(),
        selected_maxwell_ratio=x, selected_maxwell_alpha=alpha, selected_maxwell_beta=beta)
    result = _make(context, full_after_memory.binding_sha256,
        before.displacement_w_m+dw, alpha*before.coefficients_w_m+beta*dw,
        2*dw/dt-before.velocity_w_m_s, time, raw["journal"]+[step])
    return _rebuild_native_release_state(context, native_release_state_to_dict(result), full_after_memory)


def advance_native_release_state(context, before_memory, before, full_after_memory,
        increment_native_m, loading):
    """Guard a direct caller's complete source archive before appending history."""
    from .genesis_volume_release_work import validate_volume_release_work
    validate_volume_release_work(context)
    if before.context_binding_sha256 != context.binding_sha256:
        raise NativeReleaseHistoryRefusal("Native predecessor belongs to another source context")
    validate_native_release_state(context, before, before_memory)
    if (len(full_after_memory.journal) != len(before_memory.journal)+1
            or full_after_memory.journal[:-1] != before_memory.journal):
        raise NativeReleaseHistoryRefusal("Full18 after journal does not append exactly one genuine predecessor step")
    source = restart_source_seed_memory(context.birth, enriched_memory_to_dict(full_after_memory))
    if enriched_memory_to_dict(source) != enriched_memory_to_dict(full_after_memory):
        raise NativeReleaseHistoryRefusal("Full18 after archive differs from authentic source material history")
    return _advance_native_release_state(context, before, full_after_memory, increment_native_m, loading)
