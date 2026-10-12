from dataclasses import replace
import numpy as np
import pytest
from analysis.genesis_invariant_fastpath.cache import Bindings,InvariantStore,FastPath,digest


def binding():return Bindings(*[str(i)*64 for i in range(4)])

def square(x):return np.asarray(x)**2

def test_immutable_private_payload_and_all_arguments_in_key():
    s=InvariantStore(binding(),'test');f=s.memo(square);x=np.array([2.,3.]);a=f(x);a[:]=99
    np.testing.assert_array_equal(f(x),[4.,9.]);x[0]=4
    np.testing.assert_array_equal(f(x),[16.,9.]);assert sum(v['hits'] for v in s.stats.values())==1
    assert all(isinstance(v,bytes) for v in s._entries.values())


@pytest.mark.parametrize('field',list(Bindings.__dataclass_fields__))
def test_each_SHA_change_invalidates(field):
    s=InvariantStore(binding(),'test');f=s.memo(square);f([2]);f([2]);s.rebind(replace(binding(),**{field:'f'*64}));f([2])
    assert s.invalidations==1 and sum(v['misses'] for v in s.stats.values())==2


def test_stores_are_disjoint_and_bounded():
    a=InvariantStore(binding(),'producer',128);b=InvariantStore(binding(),'auditor',128)
    a.memo(square)(np.arange(100.));b.memo(square)(np.arange(100.))
    assert a.bytes==b.bytes==0 and a._entries is not b._entries
    assert sum(v['misses'] for v in b.stats.values())==1


def test_typed_and_array_keys_are_unambiguous():
    assert digest(np.array([1,2],dtype='i8'))!=digest(np.array([1,2],dtype='f8'))
    assert digest([1,2])!=digest((1,2))
    assert digest({'b':2,'a':1})==digest({'a':1,'b':2})


def test_context_restores_original_functions_and_auditor_is_independent():
    from tectonics import genesis_volume_operator as op
    from analysis.genesis_volume_continuation_validation import independent_audit as q
    original=op.conical_strain_operators;ref=q.independent_strain_operators
    X=np.array([[0.,0.,10.],[2.,0.,10.],[0.,2.,10.]]);N=np.array([[1/3]*3]);frame=np.eye(3)
    baseline=ref(X,frame,N)
    with FastPath(binding()) as cache:
        producer=op.conical_strain_operators(X,frame,N);producer[0][:]=1e9
        for _ in range(2):
            actual=q.independent_strain_operators(X,frame,N)
            for a,b in zip(actual,baseline):np.testing.assert_array_equal(a,b)
        assert cache.auditor.bytes and cache.producer.bytes
    assert op.conical_strain_operators is original and q.independent_strain_operators is ref


def test_independent_audit_rejects_tampered_memory_after_cache_warmup():
    from analysis.genesis_coupled_split_control.audit import block
    X=np.array([[0.,0.,10.],[2.,0.,10.],[0.,2.,10.]]);frame=np.eye(3)
    c={'q_lo':0.,'live_q_hi':.1};old=np.ones((3,6))*.001;zero=np.zeros((3,6));du=np.zeros(18);C=np.eye(6)*1000
    with FastPath(binding()):
        block(X,frame,100.,3.,c,C,old,zero,old/1.4,zero,zero,.4,du)
        with pytest.raises(ValueError,match='memory'):
            block(X,frame,100.,3.,c,C,old,zero,old/1.4*1.01,zero,zero,.4,du)


def test_keys_preserve_longdouble_bits_and_scalar_types():
    assert digest(1)!=digest('int:1')
    x=np.longdouble(1);y=np.nextafter(x,np.longdouble(2))
    assert x!=y and float(x)==float(y)
    assert digest(x)!=digest(y)


def test_context_restores_after_exception():
    from analysis.genesis_volume_continuation_validation import independent_audit as q
    original=q.mass_matrix
    with pytest.raises(RuntimeError,match='test failure'):
        with FastPath(binding()):raise RuntimeError('test failure')
    assert q.mass_matrix is original


def test_unbound_geometry_requires_new_scope_instead_of_reusing_basal():
    from types import SimpleNamespace
    from tectonics import genesis_volume_dynamics as d
    with FastPath(binding()):
        with pytest.raises(ValueError,match='GEOMETRY_BINDING_CHANGED'):
            d.assemble_genesis_volume_basal(SimpleNamespace(geometry=()))
