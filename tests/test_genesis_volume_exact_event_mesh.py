import json
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import pytest
from tectonics.genesis_volume_exact_event_mesh import event_cut,event_mesh
from tectonics.genesis_volume_event_partition import linear,signed_double_area,source_events
from tectonics.genesis_volume_source_coverage import exact_node
from tectonics.genesis_volume_bank_bounds import PreparedBankBounds
from analysis.genesis_volume_banks_validation.diagnose import positive_bank
from analysis.genesis_volume_banks_validation.partition_audit import check
from analysis.genesis_volume_exact_events_validation.angular import refinement_key,polynomial_chart
from tectonics.genesis_volume_sphere_event_mesh import SphereEventRefiner
from tectonics.genesis_volume_rational_fields import fraction

ROOT=Path(__file__).resolve().parents[1]
def decode(rows):return tuple(tuple(F(*map(int,v)) for v in row) for row in rows)
def simplex():return tuple(tuple(F(int(i==j)) for j in range(3)) for i in range(3))

def test_exact_cut_keeps_an_arbitrarily_thin_positive_triangle():
    event=linear((F(1,2**100),F(-1),F(-1)))
    children,record=event_cut(simplex(),event)
    assert len(children)==3 and record['split']
    assert sum(signed_double_area(c) for c in children)==1
    assert any(min(event.evaluate(n) for n in c)>=0 and max(event.evaluate(n) for n in c)>0 for c in children)
    assert all(F(*map(int,r['parameter_error']))==0 and F(*map(int,r['event_residual']))==0 for r in record['edge_parameters'])
    check([tuple(n[1:] for n in simplex())],[tuple(n[1:] for n in c) for c in children])

def test_scaled_priority_does_not_let_an_arbitrarily_small_missing_bank_starve_a_large_tensor_error():
    allowance=np.ones(3,np.longdouble)
    tiny=refinement_key(np.array([1e-30,0,1.]),allowance,np.array([1e-30,0]),F(1))
    important=refinement_key(np.array([0.,0.,100.]),allowance,np.zeros(2),F(1,2))
    assert important>tiny

def test_positive_unknown_volume_at_zero_allowance_keeps_infinite_priority():
    key=refinement_key(np.array([1.,0.,1.]),np.array([0.,1.,1.]),np.array([1.,0.]),F(1))
    assert np.isinf(key[0]) and np.isinf(key[1])

def test_all_46_previously_unreachable_positive_banks_are_sampled_after_exact_cuts():
    fixture=json.loads((ROOT/'analysis/genesis_volume_exact_events_validation/known_positive_slivers.json').read_bytes())
    assert len(fixture['cells'])==46
    nodes=tuple(map(fraction,(np.polynomial.legendre.leggauss(5)[0]+1)/2))
    for row in fixture['cells']:
        raw=fixture['geometries'][str(row['case'])];old=decode(row['vertices']);g=PreparedBankBounds(raw)
        witness=tuple(F(*map(int,v)) for v in row['positive_source_witness'])
        assert positive_bank(g,witness,row['side'])
        children,certificate=event_mesh(raw,(old,))
        assert certificate['exact_event_alignment']
        check([tuple(n[1:] for n in old)],[tuple(n[1:] for n in c) for c in children])
        events,_=source_events(raw)
        for child in children:
            for name in ('Gamma_cohort_lower','Gamma_cohort_upper'):
                values=[events[name].evaluate(n) for n in child]
                assert not min(values)<0<max(values)
        assert any(positive_bank(g,exact_node(c,s,t),row['side']) for c in children for s in nodes for t in nodes),(row['case'],row['old_cell'])
        refiner=SphereEventRefiner(raw)
        charts=[(c,refiner.chart(c)[:2]) for c in children]
        assert any(positive_bank(g,polynomial_chart(c,s,t,*chart)[0],row['side']) for c,chart in charts for s in nodes for t in nodes),(row['case'],row['old_cell'],'smoothed chart')


def test_priority_uses_each_component_scale_and_is_invariant_to_component_units():
    allowance=np.array([1.,1.,1e-12],np.longdouble)
    errors=[np.array([10.,0.,0.],np.longdouble),np.array([0.,0.,1e-9],np.longdouble)]
    missing=np.zeros(2,np.longdouble)
    keys=[refinement_key(e,allowance,missing,F(1)) for e in errors]
    assert keys[1]>keys[0]  # Smaller absolute error, larger own-relative error.
    units=np.array([1e6,1e-6,1e20],np.longdouble)
    rescaled=[refinement_key(e*units,allowance*units,missing,F(1)) for e in errors]
    assert rescaled[1]>rescaled[0]


def test_cut_through_a_vertex_retains_both_nonzero_sides():
    event=linear((F(),F(-1),F(1)))
    children,record=event_cut(simplex(),event)
    assert record['split'] and len(children)==2
    assert all(signed_double_area(c)==F(1,2) for c in children)
    check([tuple(n[1:] for n in simplex())],[tuple(n[1:] for n in c) for c in children])


@pytest.mark.parametrize('rotation',[0,1,2])
@pytest.mark.parametrize('order',[3,5])
def test_polynomial_chart_preserves_constant_and_linear_measure(rotation,order):
    raw_nodes,raw_weights=np.polynomial.legendre.leggauss(order)
    nodes=tuple(map(fraction,(raw_nodes+1)/2));weights=tuple(map(fraction,raw_weights/2))
    moments=[F() for _ in range(4)]
    for s,sw in zip(nodes,weights):
        for t,tw in zip(nodes,weights):
            n,j=polynomial_chart(simplex(),s,t,rotation,2)
            assert sum(n)==1 and min(n)>0 and j>0
            for k,value in enumerate((F(1),)+n):moments[k]+=value*j*sw*tw
    for measured,truth in zip(moments,(F(1,2),F(1,6),F(1,6),F(1,6))):
        # Literal binary Gauss operands: the ideal polynomial moments are exact,
        # while the represented weights/nodes have this separately checked residue.
        assert abs(measured-truth)<F(1,10**14)
