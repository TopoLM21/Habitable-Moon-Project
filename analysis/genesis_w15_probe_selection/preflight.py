"""Read-only basis/probe checks; no integrator, geometry generation or motion solve."""
from fractions import Fraction as F
from pathlib import Path
import hashlib,json
import numpy as np
from analysis.genesis_directional_stiffness_validation.audit import witness


def main():
    path=Path('analysis/genesis_volume_exact_lambda_validation/evidence/actual_control/source_fixture.json')
    fixture=json.loads(path.read_bytes());case=fixture['cases'][0];s=case['supports']['fixed_100m_clipped_support'];raw=case['source_inputs']
    B=s['original_declared_added_basis'];assert len(B)==18 and all(len(x)==15 for x in B)
    matrix=np.asarray(B);records=[]
    for p in s['declared_probes']:
        a=list(map(F,p['enrichment_increment_m']));w=witness(B,a)
        assert all(sum(w[i]*F(B[i][j]) for i in range(18))==0 for j in range(15))
        dot=sum(x*y for x,y in zip(w,a));assert dot!=0
        af=np.asarray(p['enrichment_increment_m']);res=af-matrix@np.linalg.lstsq(matrix,af,rcond=None)[0]
        records.append(dict(name=p['name'],exact_member=False,left_null_witness=list(map(str,w)),
                            witness_dot_a=str(dot),relative_numerical_projection_residual=float(np.linalg.norm(res)/np.linalg.norm(af))))
    alpha=F(1,8192);a=[float(alpha*F(row[0])) for row in B]
    coords=[alpha]+[F()]*14
    assert all(F(a[i])==sum(F(B[i][j])*coords[j] for j in range(15)) for i in range(18))
    # The model's analytic null witnesses: source-plane scalar times Cartesian axes.
    section=raw['section'];X=np.asarray(section['triangle_m']);A=np.asarray(section['displacement_A_m']);P=np.asarray(section['slope_P_m']);normal=np.asarray(section['normal_xyz'])
    za=(X+A)@normal-section['offset_m'];zp=(X+P)@normal
    Z=np.zeros((18,3))
    for vertex in range(3):
        for axis in range(3):Z[6*vertex+axis,axis]=za[vertex];Z[6*vertex+3+axis,axis]=zp[vertex]
    Z/=np.linalg.norm(Z,axis=0)
    leakage=float(np.linalg.norm(Z.T@np.asarray(a))/np.linalg.norm(a))
    limit=256*np.finfo(float).eps
    assert leakage<=limit
    candidate=dict(name='candidate_W15_column0_2powminus13',selection_status='PENDING_USER_CHOICE_NOT_AN_ACTIVE_CASE',
        source_column_zero_based=0,coordinate_amplitude_m=str(alpha),coordinates_w15_m=list(map(str,coords)),
        enrichment_increment_m=a,parent_increment_m=[0.]*18,
        mean_e0=raw['accepted_interval_mean_e0'],mean_e1=raw['accepted_interval_mean_e1'],
        exact_literal_basis_membership=True,analytic_kernel_relative_leakage=leakage,
        model_leakage_threshold=float(limit),analytic_kernel_numerical_check_passed=True,
        full_typed_release_validator_called=False,physical_release_admitted=False,
        interpretation='Stored first singular direction of the area-weighted compact jump trace; generally mixed normal/tangential, not a physically selected opening direction')
    result=dict(source_path=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        support_sha256=s['support_sha256'],virtual_basis_sha256=s['original_virtual_surface_basis_sha256'],
        basis_scope=s['basis_scope'],kernel_tied=s['original_kernel_coordinates_tied'],declared_probes=records,
        old_probe_origin='mixed_A_P increment with saved accepted interval mean memory; no saved motion is asserted',
        candidate=candidate,physical_selection_rule_found=False,new_regional_integrals=0,new_stiffness_integrals=0,
        mechanical_solves=0,history_modified=False)
    output=Path('analysis/genesis_w15_probe_selection/preflight.json')
    with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(probes=[{k:r[k] for k in ('name','exact_member','relative_numerical_projection_residual')} for r in records],
        candidate_exact_member=True,analytic_kernel_leakage=leakage,threshold=float(limit),status='selection_pending')),flush=True)


if __name__=='__main__':main()
