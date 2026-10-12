from fractions import Fraction as F
import pytest
from analysis.genesis_directional_stiffness_validation.bounds import ldl,quadratic,responses


def test_signed_cross_terms_preserved_by_ldl():
    C=[[F(2),F(-1)],[F(-1),F(2)]];L,D=ldl(C)
    assert [[sum(L[i][k]*D[k]*L[j][k] for k in range(2)) for j in range(2)] for i in range(2)]==C
    assert quadratic([(F(1),F(1)),(F(1),F(1))],L,D)==(F(2),F(2))


def test_interval_crosses_zero_square_lower_is_zero():
    L,D=ldl([[F(3)]]);assert quadratic([(F(-1),F(2))],L,D)==(0,12)


def test_indefinite_material_refused():
    with pytest.raises(ValueError):ldl([[F(-1)]])


def test_symmetric_increment_curvature_and_signed_linear_work():
    rows=responses((F(10),F(10)),(F(4),F(4)),F(1))
    for neg,pos in zip(rows[::2],rows[1::2]):
        h=F(pos['t']);assert F(neg['internal_virtual_work_j'][0])==-F(pos['internal_virtual_work_j'][0])
        assert (F(pos['total_directional_force_times_a_j'][0])-F(neg['total_directional_force_times_a_j'][0]))/(2*h)==4
        assert (F(pos['frozen_elastic_energy_change_j'][0])+F(neg['frozen_elastic_energy_change_j'][0]))/h**2==4
        assert F(neg['incremental_directional_force_times_a_j'][1])<0


def test_exact_nonmembership_witness():
    from analysis.genesis_directional_stiffness_validation.audit import witness
    B=[[F(1)],[F(0)]];a=[F(1),F(2)];w=witness(B,a)
    assert sum(w[i]*B[i][0] for i in range(2))==0
    assert sum(w[i]*a[i] for i in range(2))!=0


def test_cauchy_lower_with_signed_cross_work():
    # Finite exact weighted inner product: (-1,2) dot (3,4)=5,
    # stiffness=5, memory norm=25, so the nonzero-work bound is 1<=5.
    strain=[F(-1),F(2)];memory=[F(3),F(4)]
    W=sum(x*y for x,y in zip(strain,memory));K=sum(x*x for x in strain);E=sum(x*x for x in memory)
    assert W==5 and 0<W*W/E<=K
    assert sum(x*y for x,y in zip(strain,memory))!=sum(abs(x*y) for x,y in zip(strain,memory))
