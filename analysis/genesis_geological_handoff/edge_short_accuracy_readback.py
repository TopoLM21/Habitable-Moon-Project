"""Independent vectorized readback of saved full-field traces; no evolution."""
import hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).parent


def main():
    result=json.loads((ROOT/'edge_short_accuracy.json').read_text());runs=result['runs']
    assert [r['dt_years'] for r in runs]==[.5,.25,.125]
    assert result['declaration']==json.loads(Path('/workspace/genesis-edge-short-accuracy-01/declaration.json').read_text())
    assert result['declaration']['criterion']['traction_max_vector_difference_over_fine_max']==.05
    assert result['declaration']['criterion']['damage_max_absolute_difference']==.02
    arrays=[]
    for r in runs:
        assert r['status']=='COMPLETED' and r['end_time_myr']==result['target_time_myr']
        for kind in ('checkpoint','trace'):
            assert hashlib.sha256(Path(r[kind]).read_bytes()).hexdigest()==r[kind+'_sha256']
        with np.load(r['trace'],allow_pickle=False) as z:arrays.append({k:z[k].copy() for k in z.files})
        assert all(row['saturated']==0 and row['reason']=='NOT_SEPARATING' and row['energy_fraction']<=1 for row in r['rows'])
    checks=[]
    for index,(a,b) in enumerate(zip(arrays,arrays[1:])):
        np.testing.assert_array_equal(a['years'],b['years'][1::2])
        d=np.max(abs(a['damage']-b['damage'][1::2]),axis=1)
        fine=b['shear_vectors_pa'][1::2]
        load=np.max(np.linalg.norm(a['shear_vectors_pa']-fine,axis=2),axis=1)/np.max(np.linalg.norm(fine,axis=2),axis=1)
        reported=result['comparisons'][index]
        np.testing.assert_allclose(d,[r['damage_normalized_difference'] for r in reported['rows']],rtol=1e-14,atol=0)
        np.testing.assert_allclose(load,[r['traction_normalized_difference'] for r in reported['rows']],rtol=1e-14,atol=0)
        ok=bool(d.max()<=.02 and load.max()<=.05)
        assert ok==reported['criterion_pass'];checks.append(dict(max_D_difference=float(d.max()),max_traction_difference=float(load.max()),criterion_pass=ok))
    assert checks[0]['criterion_pass'] is False and checks[1]['criterion_pass'] is True
    out=dict(status='PASS',full_trace_hashes_and_arrays=True,checks=checks,evolution_steps=0)
    (ROOT/'edge_short_accuracy_readback.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))


if __name__=='__main__':main()
