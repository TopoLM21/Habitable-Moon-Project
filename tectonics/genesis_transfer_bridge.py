"""Loss-accounted data bridge, not a new physical closure or runnable checkpoint.

Version 1 supports the saved fixed-cut conical packet schema. Residual channels
are mandatory. The current Starter/mature consumer cannot evolve them, so this
bridge never certifies a physical handoff merely because serialization succeeds.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .genesis_volume_basal import local_consistent_mass_matrix

FORMAT = 'genesis-transfer-bridge-1'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode()


def finite(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f'{name}: shape or nonfinite value')
    return result


def build_transfer(packet, cuts):
    """Return conservative inventories + explicit residuals without assigning plates.

    Face indices are reference-chart indices, NOT a conservative remap onto the
    mature canonical surface. Memory means are mass-weighted coefficient means,
    not constitutive variables of a reduced model. Their residuals stay active
    requirements of any future consumer, never fictitious heat.
    """
    s = packet['state']; g = s['geometry']; cohorts = g['cohorts']
    X = np.asarray(g['vertices_m'], dtype=float)
    if X.ndim != 2 or X.shape[1] != 3 or not np.isfinite(X).all():
        raise ValueError('vertices')
    nv = len(X); nd = 6 * nv
    faces = np.asarray(g['faces'])
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in 'iu' or np.any(faces < 0) or np.any(faces >= nv):
        raise ValueError('faces')
    nf = len(faces)
    owners = g['owner_ids_per_face']
    if len(owners) != nf or any(not isinstance(x, str) or not x for x in owners):
        raise ValueError('owners')
    clock = float(s['thermal']['time_myr'])
    if not np.isfinite(clock) or clock < 0:
        raise ValueError('clock')
    for value in [packet['thermal'], s['mechanics'], s['fault_history'], packet['orbit_after'], g]:
        if value['time_myr'] != clock:
            raise ValueError('clock mismatch')
    m = finite(packet['thermal']['mass_kg'], (len(cuts)+2,), 'thermal mass')
    H = finite(packet['thermal']['energy_j'], m.shape, 'thermal enthalpy')
    if np.any(m <= 0):
        raise ValueError('nonpositive reservoir mass')
    geom = {c['id']: c for c in cohorts}
    memory = {c['id']: c for c in s['mechanics']['cohorts']}
    if len(geom) != len(cohorts) or len(memory) != len(s['mechanics']['cohorts']) or set(geom) != set(memory):
        raise ValueError('duplicate or mismatched cohort IDs')
    solid_indices = []; seen = set(); groups = [[] for _ in range(nf)]
    for i, cut in enumerate(cuts):
        f = cut['face']
        if not isinstance(f, int) or not 0 <= f < nf or cut['mass_kg'] != m[i]:
            raise ValueError('cut mass/face mismatch')
        if cut['material_id'] is not None:
            cid = cut['material_id']
            if cid not in geom or cid in seen:
                raise ValueError('cut cohort identity')
            c = geom[cid]; seen.add(cid)
            if (c['face_index'] != f or c['mass_kg'] != m[i] or
                c['q_lo'] != cut['q_lo'] or c['live_q_hi'] != cut['q_hi'] or
                c['owner_id'] != owners[f]):
                raise ValueError('cut geometry binding')
            groups[f].append(i); solid_indices.append(i)
    if seen != set(geom) or any(not x for x in groups):
        raise ValueError('incomplete solid chart')
    mass_face = np.array([math.fsum(m[i] for i in indices) for indices in groups])
    enthalpy_face = np.array([math.fsum(H[i] for i in indices) for indices in groups])
    other = [i for i in range(len(m)) if i not in set(solid_indices)]
    # Keep each non-solid/deep/water slot separately, with original indices.
    # Slot meanings and phase/birth provenance remain in the pinned source.
    M = np.zeros((nd, nd))
    for c in cohorts:
        if not 0 < c['inner_scale'] < c['outer_scale'] or c['mass_kg'] <= 0:
            raise ValueError('invalid live cohort')
        dofs = (faces[c['face_index'], :, None]*6 + np.arange(6)).ravel()
        M[np.ix_(dofs, dofs)] += local_consistent_mass_matrix(c['mass_kg'], c['outer_scale'], c['inner_scale'])
    v = finite(s['velocity_m_s'], (nd,), 'velocity')
    u = finite(s['displacement_m'], (nd,), 'displacement').reshape(nv, 6)
    pos = u.copy(); pos[:, :3] += X; pos[:, 3:] += X
    scale = float(np.max(np.linalg.norm(X, axis=1)))
    if scale <= 0:
        raise ValueError('reference radius')
    T = np.zeros((nv, 6, 3)); T[:, :3] = np.eye(3); T = T.reshape(nd, 3)
    R = np.empty((nv, 6, 3))
    for k in range(3):
        R[:, :3, k] = np.cross(np.eye(3)[k], pos[:, :3])/scale
        R[:, 3:, k] = np.cross(np.eye(3)[k], pos[:, 3:])/scale
    R = R.reshape(nd, 3)
    # One global rotation is a diagnostic reference, never a new plate partition.
    coeff = np.linalg.solve(R.T@M@R, R.T@M@v)
    fit = R@coeff; residual = v-fit
    norm2 = float(v@M@v); res2 = float(residual@M@residual)
    moment = lambda q: dict(linear_kg_m_s=(T.T@M@q).tolist(), angular_kg_m2_s=(scale*R.T@M@q).tolist())
    means = {}; residuals = {}; errors = {}
    for field in ('strain0', 'strain1'):
        a = np.array([finite(memory[c['id']][field], (3, 6), field) for c in cohorts])
        mean = np.zeros((nf, 3, 6))
        for c, row in zip(cohorts, a):
            mean[c['face_index']] += c['mass_kg']/mass_face[c['face_index']]*row
        r = a - mean[[c['face_index'] for c in cohorts]]
        weights = np.array([c['mass_kg'] for c in cohorts])
        means[field] = mean.tolist(); residuals[field] = r.tolist()
        errors[field] = dict(mass_weighted_coefficient_rms=float(np.sqrt(np.sum(weights[:, None, None]*r*r)/(weights.sum()*18))),
                             max_abs_coefficient=float(np.max(np.abs(r))),
                             reconstruction_max_abs=float(np.max(np.abs(a-(mean[[c['face_index'] for c in cohorts]]+r)))))
    damage = finite(s['fault_history']['damage'], (nf,), 'damage')
    if np.any((damage < 0) | (damage > 1)):
        raise ValueError('damage range')
    checks = [
        dict(code='FIRST_PARTITION', status='UNPROVEN' if len(set(owners)) >= 2 else 'FAIL', requirement='Source-certified first partition; owner count alone is insufficient.'),
        dict(code='PRIMARY_SOLIDIFICATION', status='UNPROVEN', requirement='Starter material inventory must certify its primary surface reservoir; solid cut coverage is not this criterion.'),
        dict(code='CONSERVATIVE_CANONICAL_REMAP', status='UNPROVEN', requirement='Reference face indices are not a conservative remap of displaced geometry.'),
        dict(code='ACTIVE_MEMORY_AND_MOMENTUM_CLOSURE', status='FAIL', requirement='Existing consumer does not evolve conical memory, residual velocity or their moment/energy ledgers.'),
        dict(code='ACTIVE_THERMAL_CLOSURE', status='FAIL', requirement='Existing mature source mass/enthalpy arrays are archives, not active full-column reservoirs.'),
        dict(code='EXPLICIT_MATURE_FIELDS', status='UNPROVEN', requirement='Independent mantle flow, chemical crust, liquid water, relief and thermal projection must be justified; no continents are inferred from thickness.'),
    ]
    return dict(format=FORMAT, time_myr=clock, physical_handoff_admitted=False,
        target='existing Starter/mature; data exchange only', checks=checks,
        source_owner_ids=owners, source_cohort_ids=[c['id'] for c in cohorts],
        inventory=dict(solid_mass_kg=mass_face.tolist(), solid_enthalpy_j=enthalpy_face.tolist(),
                       other_source_indices=other, other_mass_kg=m[other].tolist(), other_enthalpy_j=H[other].tolist(),
                       total_mass_kg=math.fsum(m), total_enthalpy_j=math.fsum(H),
                       mass_grouping_residual_kg=math.fsum(m)-math.fsum([*mass_face, *m[other]]),
                       enthalpy_grouping_residual_j=math.fsum(H)-math.fsum([*enthalpy_face, *H[other]])),
        motion=dict(reference_rotation_rad_s=(coeff/scale).tolist(), reference_velocity_m_s=fit.tolist(),
                    residual_velocity_m_s=residual.tolist(), source_moments=moment(v), reference_moments=moment(fit), residual_moments=moment(residual),
                    source_kinetic_j=.5*norm2, residual_kinetic_j=.5*res2, reference_kinetic_j=float(.5*fit@M@fit),
                    relative_mass_rms=math.sqrt(res2/norm2) if norm2 else 0.,
                    reconstruction_max_abs_m_s=float(np.max(np.abs(v-(fit+residual))))),
        history=dict(damage=damage.tolist(), fault_history=s['fault_history'], orbit=packet['orbit_after'],
                     elastic_energy_j=s['mechanics']['elastic_energy_j'], mean_coefficients=means,
                     residual_coefficients=residuals, projection_error=errors),
        approximations=['Mass-weighted strain-coefficient means are descriptive, not a reduced constitutive law.',
                        'Global rotation is only a reference projection; residual kinetic energy is NOT heat.',
                        'All original states remain required; no unsupported memory is initialized to zero.'])


def require_admission(transfer):
    """Fail closed: version 1 has no active conservative consumer, even if edited."""
    raise ValueError('PHYSICAL_HANDOFF_NOT_IMPLEMENTED: residual/thermal consumer required')


def write_bundle(destination, *, packet_path, cuts_path, source_path, expected_sha256):
    """Exclusive creation, pinned byte copies, then deterministic readback.

    Hashes establish identity/integrity, not acceptance of a trial trajectory.
    Source and cut-provider packets are retained in full; no CRLF normalization.
    """
    destination = Path(destination)
    paths = dict(packet=Path(packet_path), cuts=Path(cuts_path), source=Path(source_path))
    raw = {key: path.read_bytes() for key, path in paths.items()}
    if set(expected_sha256) != set(paths) or any(digest(raw[k]) != expected_sha256[k] for k in paths):
        raise ValueError('input SHA256 mismatch')
    packet = json.loads(raw['packet'])['packet']
    cuts = json.loads(raw['cuts'])['packet']['thermal_packet']['fixed_cut_geometry']
    source = json.loads(raw['source'])
    if packet['canonical_raw_sha256'] != digest(raw['source']):
        raise ValueError('canonical source binding')
    # Parse source to ensure the archived object is a real source packet.
    if 'source_packet' not in source:
        raise ValueError('source packet missing')
    transfer = build_transfer(packet, cuts)
    destination.mkdir(parents=True, exist_ok=False)
    files = {}
    for key, data in raw.items():
        name = key + '.json'
        with (destination/name).open('xb') as f:
            f.write(data)
        files[name] = digest(data)
    data = encoded(transfer)
    with (destination/'transfer.json').open('xb') as f:
        f.write(data)
    files['transfer.json'] = digest(data)
    manifest = dict(format=FORMAT, files=files, input_paths={k: str(p) for k,p in paths.items()}, new_time_steps=0)
    with (destination/'manifest.json').open('xb') as f:
        f.write(encoded(manifest))
    read_bundle(destination)
    return manifest, transfer


def read_bundle(directory):
    """Verify bytes AND recompute all derived fields before returning data."""
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_bytes())
    names = {'packet.json', 'cuts.json', 'source.json', 'transfer.json'}
    if manifest['format'] != FORMAT or set(manifest['files']) != names:
        raise ValueError('manifest schema')
    raw = {name: (directory/name).read_bytes() for name in names}
    if any(digest(raw[name]) != pin for name,pin in manifest['files'].items()):
        raise ValueError('bundle SHA256 mismatch')
    packet = json.loads(raw['packet.json'])['packet']
    if packet['canonical_raw_sha256'] != digest(raw['source.json']):
        raise ValueError('canonical source binding')
    cuts = json.loads(raw['cuts.json'])['packet']['thermal_packet']['fixed_cut_geometry']
    rebuilt = build_transfer(packet, cuts)
    if encoded(rebuilt) != raw['transfer.json']:
        raise ValueError('derived transfer mismatch')
    return rebuilt
