"""Zero-time tied-bank transfer of the sealed, accepted Genesis state.

Reference edge selections and displacement probes are computational fixtures.
They do not activate a fault, specify a physical fracture surface, release a
tie, or advance material history. The actual source has no eligible cut.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tectonics.genesis_volume_banks import prepare_volume_banks, evaluate_volume_banks
from tectonics.genesis_volume_bank_transfer import (
    bind_volume_bank_transfer, volume_bank_transfer_to_dict,
    validate_volume_bank_transfer_payload,
)
from tectonics.genesis_volume_dynamics import coefficient_arrays
from tectonics.genesis_volume_operator import evaluate_genesis_volume_operator
from tectonics.genesis_volume_slip_checkpoint import (
    load_volume_slip_checkpoint, save_volume_slip_checkpoint,
    volume_slip_checkpoint_to_dict,
)


PARENT_RAW_SHA256 = "34cc53c215b2e9b8300a8f47b33801d49182b611a3ad985217a9c4cef25f2aed"
FORMATION_SHA256 = "cd79ee2a5139c4b91abdb1b4714a761c9153cc9f4490f5aa63fa2facba458c89"
FORMAT = "genesis-volume-bank-control-0.1"
SCOPE = "tied_computational_bank_transfer_from_verified_actual_genesis_volume_state"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, name, *, rtol=3e-12, atol=1e-8):
    a, b = np.asarray(actual, float), np.asarray(expected, float)
    require(a.shape == b.shape, f"{name}: dimensions changed")
    scale = max(float(np.max(np.abs(b), initial=0.)), 1e-300)
    error = float(np.max(np.abs(a-b), initial=0.))
    require(error <= atol+rtol*scale, f"{name}: tied transfer changed the physical quantity")
    return error/max(scale, atol, 1e-300)


def dense(matrix):
    return matrix.toarray().tolist()


def momentum_by_quadrature(assembly, faces, velocity):
    """Integrate physical v and r cross v directly in the reference chart.

    The angular rule integrates quadratic fields, and radial weights are the
    literal lambda**2 volume measure. This differs from forming generalized
    rigid-motion vectors and multiplying the assembled mass matrix.
    """
    N = np.array(((2/3, 1/6, 1/6), (1/6, 2/3, 1/6), (1/6, 1/6, 2/3)))
    x, w = np.polynomial.legendre.leggauss(8)
    P, L = np.zeros(3), np.zeros(3)
    energy = 0.
    velocity = np.asarray(velocity).reshape(-1, 6)
    for cohort, outer, inner in zip(assembly.geometry.cohorts,
                                   assembly.outer_scale, assembly.inner_scale):
        local = velocity[faces[cohort.face_index]]
        lam = .5*(outer+inner)+.5*(outer-inner)*x
        weights = w*lam*lam
        weights /= weights.sum()
        v = (N@local[:, :3])[:, None, :]+(lam[None, :, None]-1)*(N@local[:, 3:])[:, None, :]
        r = (N@np.asarray(cohort.top_triangle_m))[:, None, :]*lam[None, :, None]
        mass = cohort.mass_kg
        P += mass*np.einsum("r,ari->i", weights, v)/3
        L += mass*np.einsum("r,ari->i", weights, np.cross(r, v))/3
        energy += .5*mass*np.einsum("r,ari,ari->", weights, v, v)/3
    return dict(kinetic_energy_j=float(energy), linear_momentum_kg_m_s=P.tolist(),
                angular_momentum_kg_m2_s=L.tolist())


def run(parent_path, output):
    output.mkdir(parents=True, exist_ok=False)
    parent_bytes = parent_path.read_bytes()
    parent_raw_sha = hashlib.sha256(parent_bytes).hexdigest()
    require(parent_raw_sha == PARENT_RAW_SHA256, "Control requires the sealed accepted 0.8682 Myr parent")
    parent_packet = json.loads(parent_bytes)
    # The untrusted archive is replayed once from the literal formation source.
    # No initial_state() replacement and no new model.step() are performed.
    model, state = load_volume_slip_checkpoint(parent_path,
        expected_source_checkpoint_sha256=FORMATION_SHA256)
    require(len(state.geometry.cohorts) == 640 and abs(state.time_myr-.8682) < 1e-14,
            "Accepted material cohort catalogue or clock differs")
    require(not state.fault_history.fault_active.any(), "The actual source unexpectedly has an active fault")
    require(canonical(volume_slip_checkpoint_to_dict(model, state)) == canonical(parent_packet),
            "Semantic parent replay changed the accepted checkpoint")
    save_volume_slip_checkpoint(output/"parent_before.json", model, state)
    require((output/"parent_before.json").read_bytes() == parent_bytes,
            "Restored accepted checkpoint is not byte-identical")
    parent, basal = model._assembly(state.geometry, state.fault_history.damage)
    e0, e1 = coefficient_arrays(state.mechanics)
    parent_response = evaluate_genesis_volume_operator(parent, np.zeros(parent.ndof),
        elastic_constant=e0, elastic_inverse_scale=e1)
    close(parent_response.energy_j, state.mechanics.elastic_energy_j,
          "Accepted elastic memory energy", atol=1e4)
    faces = np.asarray(state.geometry.faces, int)
    parent_momentum = momentum_by_quadrature(parent, faces, state.velocity_m_s)
    close(parent_momentum["kinetic_energy_j"], state.kinetic_energy_j,
          "Accepted reference-chart kinetic energy", atol=1e-7)
    actual = bind_volume_bank_transfer(model, state)
    actual_payload = volume_bank_transfer_to_dict(actual)
    validate_volume_bank_transfer_payload(actual_payload, expected_transfer=actual)
    require(actual_payload["physical_cut_status"] == "no_eligible_cut",
            "Actual zero-active-fault source acquired a physical cut")
    (output/"actual_no_cut_transfer.json").write_bytes(canonical(actual_payload)+b"\n")

    first = faces[0]
    cycle = [(int(first[a]), int(first[b])) for a, b in ((0, 1), (1, 2), (2, 0))]
    fixture_cases, transfer_reports = [], []
    parent_probe = .25*np.sin(.21*np.arange(parent.ndof)+.4)
    for name, cut_edges in (("unchanged_reference_support", []),
                            ("supplied_closed_first_face_reference_cycle", cycle)):
        banks = prepare_volume_banks(parent, cut_edges, basal=basal)
        transfer = bind_volume_bank_transfer(model, state, banks)
        descriptor = volume_bank_transfer_to_dict(transfer)
        validate_volume_bank_transfer_payload(descriptor, expected_transfer=transfer)
        bank_basal = banks.basal
        R, J = banks.R, banks.J
        zero = evaluate_volume_banks(banks, np.zeros(banks.ndof),
            elastic_constant=transfer.elastic_constant,
            elastic_inverse_scale=transfer.elastic_inverse_scale)
        errors = {}
        for label, current, previous in (("stiffness", banks.stiffness, parent.stiffness),
                ("mass", bank_basal.mass_matrix, basal.mass_matrix),
                ("basal_drag", bank_basal.basal_drag_matrix, basal.basal_drag_matrix)):
            errors[label] = close((R.T@current@R).toarray(), previous.toarray(), label)
        errors["prestress_force"] = close(R.T@zero.force, parent_response.force, "Prestress force")
        errors["external_force"] = close(R.T@bank_basal.external_force_n,
                                         basal.external_force_n, "Basal force")
        errors["elastic_energy"] = close(zero.energy_j, parent_response.energy_j, "Elastic energy", atol=1e4)
        require(np.array_equal(zero.memory0, e0) and np.array_equal(zero.memory1, e1),
                "Zero-time transfer changed accepted full-cohort elastic memory")
        require(not np.any(J@transfer.displacement_m) and not np.any(J@transfer.velocity_m_s),
                "Tied support acquired an absolute pose or velocity jump")
        tied = dict(displacement_m=transfer.displacement_m.tolist(),
                    velocity_m_s=transfer.velocity_m_s.tolist(),
                    **momentum_by_quadrature(parent, banks.topology.mesh.faces, transfer.velocity_m_s))
        for key in parent_momentum:
            errors[key] = close(tied[key], parent_momentum[key], key, atol=1e-7)
        close(tied["kinetic_energy_j"], .5*transfer.velocity_m_s@(bank_basal.mass_matrix@transfer.velocity_m_s),
              "Consistent bank kinetic energy", atol=1e-7)
        probes = []
        inputs = [("zero_new_increment_accepted_prestress", np.zeros(parent.ndof), False),
                  ("tied_nonzero_A_and_P_increment", parent_probe, False)]
        if J.shape[0]:
            free = np.asarray(R@parent_probe)
            free[parent.ndof:] += .07*np.cos(.37*np.arange(banks.ndof-parent.ndof)+.2)
            inputs.append(("synthetic_six_dof_duplicate_jump", free, True))
        for probe_name, input_vector, synthetic in inputs:
            increment = input_vector if synthetic else R@input_vector
            response = evaluate_volume_banks(banks, increment,
                elastic_constant=transfer.elastic_constant,
                elastic_inverse_scale=transfer.elastic_inverse_scale)
            work = float(.5*(zero.force+response.force)@increment)
            close(response.energy_j-zero.energy_j, work, "Incremental work identity", rtol=8e-11, atol=2e4)
            if not synthetic:
                old = evaluate_genesis_volume_operator(parent, input_vector,
                    elastic_constant=e0, elastic_inverse_scale=e1)
                close(response.energy_j, old.energy_j, "Tied probe energy", atol=1e4)
                close(R.T@response.force, old.force, "Tied probe force")
            probes.append(dict(name=probe_name,
                parent_increment_m=None if synthetic else input_vector.tolist(),
                bank_increment_m=np.asarray(increment).tolist(), synthetic_nonzero_jump=synthetic,
                elastic_energy_j=response.energy_j, force_n=response.force.tolist(),
                mechanical_work_j=work, jump_coefficients_m=np.asarray(J@increment).tolist(),
                constant_strain=response.memory0.tolist(), inverse_scale_strain=response.memory1.tolist()))
        fixture_cases.append(dict(name=name, synthetic_computational_cut=True,
            cut_edges=banks.topology.cut_edges.tolist(),
            topology=dict(parent_vertex=banks.topology.parent_vertex.tolist(),
                faces=banks.topology.mesh.faces.tolist(), seam_faces=banks.topology.seam_faces.tolist(),
                bank_vertices=banks.topology.bank_vertices.tolist()),
            maps=dict(prolongation=dense(R), jump_operator=dense(J),
                      prolongation_shape=list(R.shape), jump_operator_shape=list(J.shape)),
            operator=dict(stiffness=dense(banks.stiffness), mass_matrix=dense(bank_basal.mass_matrix),
                basal_drag_matrix=dense(bank_basal.basal_drag_matrix),
                external_force_n=bank_basal.external_force_n.tolist(),
                prestress_force_n=zero.force.tolist(), elastic_energy_j=zero.energy_j),
            tied=tied, probes=probes, transfer_descriptor=descriptor))
        transfer_reports.append(dict(name=name, cut_edges=banks.topology.cut_edges.tolist(),
            ndof=banks.ndof, bank_vertex_count=len(banks.topology.parent_vertex),
            independent_algebraic_jump_rows=J.shape[0], tied_relative_errors=errors,
            memory_coefficients_exactly_unchanged=True, physical_cut_status=descriptor["physical_cut_status"],
            transfer_sha256=descriptor["transfer_sha256"], probe_names=[p["name"] for p in probes]))
    fixture = dict(format=FORMAT, scope=SCOPE, parent_raw_sha256=parent_raw_sha,
        parent_path=str(parent_path.resolve()), parent_checkpoint=parent_packet,
        prescribed_basal=dict(traction_face_xyz_pa=basal.basal_traction_at_vertices_pa[faces].tolist(),
            drag_pa_s_per_m=model.contact_parameters.basal_drag_pa_s_m),
        fixture_cases=fixture_cases,
        angular_momentum_position="unchanged_reference_chart_r_equals_lambda_X",
        increment_semantics="new_increment_only_absolute_q_is_not_reapplied_to_accepted_memory",
        actual_no_cut_transfer=actual_payload)
    fixture["fixture_sha256"] = hashlib.sha256(canonical(fixture)).hexdigest()
    (output/"control_fixture.json").write_bytes(canonical(fixture)+b"\n")
    save_volume_slip_checkpoint(output/"parent_after.json", model, state)
    require((output/"parent_after.json").read_bytes() == parent_bytes,
            "Zero-time transfer changed the entire accepted parent archive")
    require(parent_path.read_bytes() == parent_bytes, "Immutable accepted source was changed")
    report = dict(format="actual-genesis-volume-bank-transfer-control-0.1", status="PASS",
        scope=SCOPE, parent_raw_sha256=parent_raw_sha, parent_checkpoint_sha256=parent_packet["checkpoint_sha256"],
        formation_checkpoint_sha256=FORMATION_SHA256, time_myr=state.time_myr,
        cohort_count=len(state.geometry.cohorts), actual_active_weak_plane_count=0,
        source_replayed_once_from_formation_checkpoint=True, no_replacement_initialization=True,
        parent_before_after_entire_packets_byte_equal=True,
        new_time_steps=0, transition_time_increment_myr=0.,
        actual_cut_status=actual_payload["physical_cut_status"], actual_transfer_sha256=actual_payload["transfer_sha256"],
        sharp_physical_banks_created=False, ties_released=False, fracture_work_assigned=False,
        physical_banks_contact_underthrust_admitted=False, point_one_complete=False,
        fixture_sha256=fixture["fixture_sha256"], fixture_cases=transfer_reports,
        angular_momentum_position=fixture["angular_momentum_position"],
        unresolved_physical_inputs=actual_payload["unresolved_physical_inputs"])
    report["report_sha256"] = hashlib.sha256(canonical(report)).hexdigest()
    (output/"control_report.json").write_bytes(canonical(report)+b"\n")
    print(json.dumps(report, sort_keys=True))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", type=Path, default=ROOT/
        "analysis/genesis_volume_slip_validation/evidence/actual_control/direct_after_second.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.parent_checkpoint, args.output)


if __name__ == "__main__":
    main()
