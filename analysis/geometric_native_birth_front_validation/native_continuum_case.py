"""Canonical genuine source wrapper, without a second force construction."""
from __future__ import annotations


def build_native_continuum_case():
    from tests.test_geometric_native_implicit_underthrust_motion import get_implicit_second_trial_case
    from tests.test_geometric_native_birth_contact_support import get_birth_support_case
    from tectonics.geometric_native_birth_continuum_motion import build_native_birth_continuum_inputs
    trial = get_implicit_second_trial_case()
    support = get_birth_support_case()["support"]
    mesh, previous, load = trial["first"]["original"]["mesh"], trial["before"], trial["load"]
    dt_s = float(trial["candidate"].dt_s)
    inputs, frames = build_native_birth_continuum_inputs(mesh, previous, load, support, dt_s,
        decimal_precision=160)
    return dict(trial=trial, mesh=mesh, previous=previous, load=load, support=support,
        dt_s=dt_s, inputs=inputs, frames=frames)


def solve_native_continuum_case(case, *, provenance):
    from tectonics.geometric_native_birth_continuum_motion import solve_native_birth_continuum_candidate
    return solve_native_birth_continuum_candidate(case["mesh"], case["previous"], case["load"],
        case["support"], case["dt_s"], provenance=provenance, decimal_precision=160)


def export_native_continuum_case(case, candidate):
    from tectonics.geometric_native_birth_continuum_motion import native_birth_continuum_candidate_to_dict
    return native_birth_continuum_candidate_to_dict(candidate, mesh=case["mesh"],
        previous=case["previous"], load=case["load"], support=case["support"])
