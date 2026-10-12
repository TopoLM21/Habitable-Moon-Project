import copy
import json

import numpy as np
import pytest

from tectonics.genesis_transfer_bridge import (build_transfer, digest, encoded,
    read_bundle, require_admission, write_bundle)
from tectonics.mesh import build_icosphere


@pytest.fixture
def example():
    mesh = build_icosphere(0); nf = mesh.cell_count; nv = len(mesh.vertices)
    cohorts = []; memory = []; cuts = []
    for f in range(nf):
        for j in range(2):
            cid = 2*f+j
            cohorts.append(dict(id=cid, face_index=f, mass_kg=float(j+1),
                q_lo=j/2, live_q_hi=(j+1)/2, inner_scale=.8+j*.1,
                outer_scale=.9+j*.1, owner_id='one-body'))
            memory.append(dict(id=cid, strain0=np.full((3,6), (j+1)*.001).tolist(),
                strain1=np.full((3,6), (j+1)*.002).tolist()))
            cuts.append(dict(face=f, material_id=cid, mass_kg=float(j+1), q_lo=j/2, q_hi=(j+1)/2))
    rng = np.random.default_rng(42)
    packet = dict(state=dict(thermal=dict(time_myr=1.),
        geometry=dict(vertices_m=(mesh.vertices*10).tolist(), faces=mesh.faces.tolist(),
            cohorts=cohorts, owner_ids_per_face=['one-body']*nf, time_myr=1.),
        mechanics=dict(cohorts=memory, elastic_energy_j=123., time_myr=1.),
        fault_history=dict(time_myr=1., damage=[.2]*nf, peak_tensile_pa=[1e7]*nf),
        displacement_m=[0.]*(nv*6), velocity_m_s=rng.normal(size=nv*6).tolist()),
        thermal=dict(time_myr=1., mass_kg=[c['mass_kg'] for c in cuts]+[10.,20.],
            energy_j=list(range(len(cuts)+2))), orbit_after=dict(time_myr=1., eccentricity=.1))
    return packet, cuts


def test_loss_channels_reconstruct_inventory_momentum_and_memory(example):
    packet, cuts = example; before = encoded(packet)
    t = build_transfer(packet, cuts); inv = t['inventory']; motion = t['motion']
    assert encoded(packet) == before
    assert inv['mass_grouping_residual_kg'] == inv['enthalpy_grouping_residual_j'] == 0
    np.testing.assert_allclose(np.array(motion['reference_velocity_m_s'])+motion['residual_velocity_m_s'], packet['state']['velocity_m_s'], atol=1e-15)
    for key in ('linear_kg_m_s', 'angular_kg_m2_s'):
        np.testing.assert_allclose(np.array(motion['reference_moments'][key])+motion['residual_moments'][key], motion['source_moments'][key], rtol=1e-12, atol=1e-12)
    assert motion['source_kinetic_j'] == pytest.approx(motion['reference_kinetic_j']+motion['residual_kinetic_j'])
    for field in ('strain0','strain1'):
        mean = t['history']['mean_coefficients'][field]; residual = t['history']['residual_coefficients'][field]
        for i, c in enumerate(packet['state']['geometry']['cohorts']):
            np.testing.assert_allclose(np.array(mean[c['face_index']])+residual[i], packet['state']['mechanics']['cohorts'][i][field], atol=1e-18)
        assert t['history']['projection_error'][field]['max_abs_coefficient'] > 0
    assert t['history']['fault_history'] == packet['state']['fault_history']
    assert not t['physical_handoff_admitted']


def test_zero_velocity_is_finite_and_not_physical_admission(example):
    packet,cuts = example; packet['state']['velocity_m_s'] = [0.]*len(packet['state']['velocity_m_s'])
    t = build_transfer(packet,cuts)
    assert t['motion']['relative_mass_rms'] == 0
    t['physical_handoff_admitted'] = True
    with pytest.raises(ValueError, match='PHYSICAL_HANDOFF_NOT_IMPLEMENTED'):
        require_admission(t)


@pytest.mark.parametrize('kind', ['mass', 'identity', 'clock', 'nan', 'damage'])
def test_rejects_inconsistent_source(example,kind):
    p,c = copy.deepcopy(example)
    if kind == 'mass': c[0]['mass_kg'] += 1
    if kind == 'identity': c[1]['material_id'] = c[0]['material_id']
    if kind == 'clock': p['orbit_after']['time_myr'] += 1
    if kind == 'nan': p['state']['velocity_m_s'][0] = float('nan')
    if kind == 'damage': p['state']['fault_history']['damage'][0] = 1.1
    with pytest.raises(ValueError): build_transfer(p,c)


def files(tmp_path,example):
    p,c = example; source = b'{\r\n"source_packet": {}\r\n}\r\n'
    p['canonical_raw_sha256'] = digest(source)
    data = dict(packet=encoded(dict(packet=p)), cuts=encoded(dict(packet=dict(thermal_packet=dict(fixed_cut_geometry=c)))), source=source)
    paths = {}
    for key, raw in data.items():
        paths[key] = tmp_path/(key+'.json'); paths[key].write_bytes(raw)
    kwargs = {k+'_path': v for k,v in paths.items()}
    kwargs['expected_sha256'] = {k:digest(v) for k,v in data.items()}
    return kwargs, data


def test_bundle_readback_preserves_raw_bytes_and_refuses_overwrite(tmp_path,example):
    kwargs,data = files(tmp_path,example); out = tmp_path/'bundle'
    _,t = write_bundle(out,**kwargs)
    assert read_bundle(out) == t
    assert (out/'source.json').read_bytes() == data['source']
    with pytest.raises(FileExistsError): write_bundle(out,**kwargs)
    (out/'packet.json').write_bytes(data['packet']+b' ')
    with pytest.raises(ValueError,match='SHA256'): read_bundle(out)


def test_modified_projection_rejected_even_with_updated_manifest_hash(tmp_path,example):
    kwargs,_ = files(tmp_path,example); out = tmp_path/'bundle'; write_bundle(out,**kwargs)
    t = json.loads((out/'transfer.json').read_bytes()); t['physical_handoff_admitted'] = True
    raw = encoded(t); (out/'transfer.json').write_bytes(raw)
    m = json.loads((out/'manifest.json').read_bytes()); m['files']['transfer.json'] = digest(raw)
    (out/'manifest.json').write_bytes(encoded(m))
    with pytest.raises(ValueError,match='derived transfer mismatch'): read_bundle(out)


def test_wrong_pin_writes_nothing(tmp_path,example):
    kwargs,_ = files(tmp_path,example); kwargs['expected_sha256']['source'] = '0'*64
    out = tmp_path/'bundle'
    with pytest.raises(ValueError,match='SHA256'): write_bundle(out,**kwargs)
    assert not out.exists()
