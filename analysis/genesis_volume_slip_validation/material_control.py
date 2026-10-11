"""Explicit synthetic 3-D material control, separate from molten Genesis.

This generator supplies declared cone/material inputs to the producer law.
It neither loads nor alters a planetary formation source.  The independent
oracle and actual-formation continuation have their own separate captures.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from analysis.genesis_volume_slip_validation import independent_audit as audit
from tectonics.genesis_fault_law import WeakPlaneParameters
from tectonics.genesis_volume_fault_slip import prepare_volume_fault_slip, evaluate_volume_fault_slip
from tectonics.genesis_volume_loading import RadialMoments, VolumeCohortMemory, VolumeLoading, VolumeMechanicsState
from tectonics.genesis_volume_operator import conical_strain_operators, ANGULAR_BARYCENTRIC


def declared_material_fixture(*, sign=1., active=True, pure_compression=False):
    """Synthetic cone, full 3-D plane and unequal Maxwell/radial memories.

    This fixture does not instantiate a formation source, replace planetary
    memory, or activate a real Genesis face.  Its cohort coefficients and
    prescribed increment are independently constructed material inputs.
    """
    triangle = np.eye(3)*1e5
    frames, _ = audit.faults.physical_face_geometry(triangle, np.array(((0, 1, 2),)))
    frame = frames[0]
    normal = np.array((1., 2., -1.))/np.sqrt(6.)
    tangent = np.array((2., -1., 0.))/np.sqrt(5.)
    flow, _ = audit.engineering_flow(frame, normal, tangent)
    shell = dict(young_modulus_pa=6e10, poisson_ratio=.28, residual_stiffness=.02)
    damage, water = .3, .65
    params = WeakPlaneParameters()
    C = audit.volume.isotropic_matrix(shell["young_modulus_pa"], shell["poisson_ratio"], damage, shell["residual_stiffness"])
    intervals, ratios = ((0., .08), (.08, .22), (.22, .37)), (0., .4, 3.)
    mass, density, dt = 1e15, 3000., 1e-4
    rows, volumes, means, variances, energies, inc0, inc1 = [], [], [], [], [], [], []
    for index, ((lo, hi), ratio) in enumerate(zip(intervals, ratios)):
        q, weights = audit.volume.radial_rule(lo, hi)
        lam, _ = audit.volume.conical_scales(triangle, mass, density, q)
        mean = float(weights@(1/lam))
        variance = float(weights@((1/lam-mean)**2))
        V = mass*(hi-lo)/density
        old0 = np.tile(sign*.003*(1+.12*index)*flow, (3, 1))
        old0[:, :3] -= 2e-5
        old0 += np.array((-1., 0., 1.))[:, None]*np.array((1., -1., .2, .8, -.7, .6))*1e-5
        old1 = np.tile(sign*(index-1)*1e-4*flow, (3, 1))
        if pure_compression:
            old0[:] = 0.; old0[:, :3] = -.003
            old1[:] = 0.
        thermal = np.zeros((3, 6)); thermal[:, :3] = -4e-7
        dq = np.array(((.003, -.001, .002, .02, .01, -.01),
                       (-.002, .001, .004, -.01, .01, .02),
                       (.001, -.003, -.001, .02, -.01, .01))).ravel()
        if pure_compression:
            dq[:] = 0.
        cohort = dict(id=index+1, face_index=0, q_lo=lo, q_hi=hi)
        rows.append(dict(face_index=0, triangle_m=triangle.tolist(), basis_xyz=frame.tolist(),
            geometry_cohort=cohort, column_mass_kg=mass, density_kg_m3=density,
            displacement_increment18_m=dq.tolist(), old0=old0.tolist(), old1=old1.tolist(),
            thermal0=thermal.tolist(), maxwell_ratio=ratio))
        B0, B1 = audit.volume.independent_strain_operators(triangle, frame, audit.volume.SOURCE_ANGULAR_POINTS)
        inc0.append(B0@dq); inc1.append(B1@dq)
        volumes.append(V); means.append(mean); variances.append(variance)
        energies.append(audit.volume.coefficient_energy(triangle, mass, density, cohort, old0, old1, C))
    fixture = dict(format=audit.FIXTURE_FORMAT, scope=audit.FIXTURE_SCOPE,
        plane_normal_world=[normal.tolist()], slip_direction_world=[tangent.tolist()],
        damage=[damage], water=[water], active=[active], dt_myr=dt,
        elastic_parameters=shell, fault_parameters=asdict(params), cohorts=rows)
    old0, old1, thermal = (np.array([row[key] for row in rows]) for key in ("old0", "old1", "thermal0"))
    ratio = np.array(ratios)
    beta = np.divide(-np.expm1(-ratio), ratio, out=np.ones_like(ratio), where=ratio != 0)
    moments = RadialMoments(tuple(volumes), tuple(means), tuple(variances))
    records = tuple(VolumeCohortMemory(i+1, 1000., old0[i], old1[i]) for i in range(3))
    before = VolumeMechanicsState("a"*64, 0., 0, records, sum(energies), sum(energies))
    loading = VolumeLoading(before, (1, 2, 3), dt, np.full(3, 1000.), moments,
        old0, old1, thermal, np.exp(-ratio)[:, None, None]*old0-beta[:, None, None]*thermal,
        np.exp(-ratio)[:, None, None]*old1, ratio, beta, np.broadcast_to(C, (3, 6, 6)), 0., 0.)
    prepared = prepare_volume_fault_slip(loading, face_indices=np.zeros(3, int),
        basis_xyz=np.repeat(frame[None], 3, axis=0), plane_normal_world=np.array((normal,)),
        slip_direction_world=np.array((tangent,)), active=np.array((active,)),
        damage=np.array((damage,)), water=np.array((water,)), parameters=params)
    result = evaluate_volume_fault_slip(prepared, np.array(inc0), np.array(inc1))
    names = ("average_strain0", "average_strain1", "plastic_increment0", "shear_increment",
        "trial_shear_stress_pa", "shear_stress_pa", "normal_stress_pa", "yield_strength_pa", "yielding",
        "friction_work_cell_j", "viscous_work_cell_j", "friction_work_j", "viscous_fault_work_j",
        "plastic_work_j", "physical_mechanical_work_j", "balance_residual_j")
    producer = {name: getattr(result, name) for name in names}
    for name in ("strain0", "strain1", "elastic_energy_j", "thermal_work_j",
                 "viscous_dissipation_j", "mechanical_work_j"):
        producer[name] = getattr(result.loading_response, name)
    force = []
    for index, row in enumerate(rows):
        B0, B1 = conical_strain_operators(triangle, frame, ANGULAR_BARYCENTRIC)
        Bmean = B0+means[index]*B1
        mean_strain = result.average_strain0[index]+means[index]*result.average_strain1[index]
        local_force = (np.einsum('qai,ab,qb->i', Bmean, C, mean_strain)
            +variances[index]*np.einsum('qai,ab,qb->i', B1, C, result.average_strain1[index]))*volumes[index]/3
        force.append(local_force)
    producer['average_force_cohort_n'] = np.array(force)
    fixture["producer_result"] = {name: value.tolist() if isinstance(value, np.ndarray) else value
                                  for name, value in producer.items()}
    return fixture



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    fixtures = []
    for name, options in (('positive_3d_slip', {}), ('negative_3d_slip', {'sign': -1.}),
                          ('inactive_plane', {'active': False}),
                          ('pure_compression', {'pure_compression': True})):
        raw = declared_material_fixture(**options)
        encoded = json.dumps(raw, sort_keys=True, indent=2, allow_nan=False)+'\n'
        path = args.output/(name+'.json')
        path.write_text(encoded)
        fixtures.append(dict(name=path.name, raw_sha256=hashlib.sha256(encoded.encode()).hexdigest(),
            scope=raw['scope'], yielding_face_count=int(sum(raw['producer_result']['yielding'])),
            shear_increment=raw['producer_result']['shear_increment'],
            plastic_work_j=raw['producer_result']['plastic_work_j']))
    report = dict(format='genesis-volume-slip-synthetic-material-control-0.1', status='PASS',
        scope='synthetic_material_test_not_planet_activation', fixtures=fixtures,
        actual_genesis_initialization_replaced=False, actual_planet_activation_claimed=False,
        sharp_banks_created=False, physical_contact_or_underthrust_admitted=False,
        whole_heat_mechanical_reservoir_closed=False, original_point_1_completed=False)
    report['report_sha256'] = hashlib.sha256(audit.volume._canonical(report)).hexdigest()
    encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+'\n'
    (args.output/'control_report.json').write_text(encoded)
    print(encoded, end='')


if __name__ == '__main__':
    main()
