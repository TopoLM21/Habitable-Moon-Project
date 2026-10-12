"""Opt-in, process-local invariant cache. Never caches states, solves or audits.

Producer and independent-auditor stores are disjoint. Values are immutable
private pickle bytes created here, never loaded from external pickle files.
Every return is a detached object, so caller mutation cannot poison a cache.
"""
from collections import OrderedDict,defaultdict
from contextlib import ExitStack
from dataclasses import dataclass,fields,is_dataclass
import functools,hashlib,inspect,pickle,struct
from pathlib import Path
from unittest.mock import patch
import numpy as np


def digest(value):
    h=hashlib.sha256()
    def put(x):
        if isinstance(x,np.ndarray):
            if x.dtype.hasobject:raise TypeError('No object-array cache keys')
            h.update(b'array');put(x.dtype.str);put(x.shape);h.update(x.tobytes(order='C'))
        elif is_dataclass(x):
            h.update(b'dataclass');put(len(fields(x)));put(type(x).__module__+'.'+type(x).__qualname__)
            for f in fields(x):put(f.name);put(getattr(x,f.name))
        elif isinstance(x,dict):
            h.update(b'dict');put(len(x))
            for k in sorted(x,key=lambda k:(type(k).__name__,repr(k))):put(k);put(x[k])
        elif isinstance(x,(tuple,list)):
            h.update(type(x).__name__.encode());put(len(x))
            for v in x:put(v)
        elif isinstance(x,str):
            raw=x.encode();h.update(b'str'+struct.pack('!Q',len(raw))+raw)
        elif isinstance(x,bytes):h.update(b'bytes'+struct.pack('!Q',len(x))+x)
        elif x is None:h.update(b'none')
        elif isinstance(x,(bool,np.bool_)):h.update(b'bool'+bytes([bool(x)]))
        elif isinstance(x,np.generic):
            h.update(b'numpy-scalar');put(x.dtype.str);put(x.tobytes())
        elif isinstance(x,int):
            raw=str(x).encode();h.update(b'int'+struct.pack('!Q',len(raw))+raw)
        elif isinstance(x,float):h.update(b'float'+struct.pack('!d',x))
        else:raise TypeError('Unsupported invariant key '+str(type(x)))
    put(value);return h.hexdigest()


@dataclass(frozen=True)
class Bindings:
    geometry_sha256:str
    material_sha256:str
    source_sha256:str
    solver_sha256:str

    def __post_init__(self):
        for f in fields(self):
            x=getattr(self,f.name)
            if len(x)!=64 or any(c not in '0123456789abcdef' for c in x):raise ValueError('SHA256 binding required')


class InvariantStore:
    def __init__(self,bindings,namespace,max_bytes=64*1024**2):
        self.bindings=bindings;self.namespace=namespace;self.max_bytes=max_bytes
        self._entries=OrderedDict();self.bytes=0;self.invalidations=0;self.stats=defaultdict(lambda:dict(hits=0,misses=0))

    def rebind(self,bindings):
        if bindings!=self.bindings:
            self._entries.clear();self.bytes=0;self.invalidations+=1;self.bindings=bindings

    def memo(self,fn,key=None):
        identity=(fn.__module__,fn.__qualname__,hashlib.sha256(Path(inspect.getsourcefile(fn)).read_bytes()).hexdigest())
        name='.'.join(identity[:2])
        @functools.wraps(fn)
        def call(*args,**kw):
            arguments=(args,kw) if key is None else key(*args,**kw)
            address=digest((self.namespace,self.bindings,identity,arguments))
            payload=self._entries.get(address)
            if payload is None:
                self.stats[name]['misses']+=1
                value=fn(*args,**kw)
                payload=pickle.dumps(value,protocol=5)
                if len(payload)<=self.max_bytes:
                    while self._entries and self.bytes+len(payload)>self.max_bytes:
                        _,old=self._entries.popitem(last=False);self.bytes-=len(old)
                    self._entries[address]=payload;self.bytes+=len(payload)
            else:self.stats[name]['hits']+=1;self._entries.move_to_end(address)
            return pickle.loads(payload)
        return call

    def report(self):
        return dict(namespace=self.namespace,bindings={f.name:getattr(self.bindings,f.name) for f in fields(self.bindings)},
                    entries=len(self._entries),bytes=self.bytes,invalidations=self.invalidations,functions=dict(self.stats))


class FastPath:
    """Explicit context for the existing fixed-geometry pipeline, no default hook.

    Dynamic C, temperature, Maxwell rates, RHS, solutions, damage and acceptance
    are always recomputed. A material/geometry change requires a fresh binding.
    """
    def __init__(self,bindings,max_bytes_per_store=64*1024**2):
        self.producer=InvariantStore(bindings,'producer',max_bytes_per_store)
        self.auditor=InvariantStore(bindings,'independent-auditor',max_bytes_per_store)
        self._stack=None

    def __enter__(self):
        if self._stack is not None:raise RuntimeError('FastPath already active')
        from tectonics import genesis_volume_operator as op,genesis_volume_dynamics as dynamics
        from analysis.genesis_volume_continuation_validation import independent_audit as q
        from analysis.genesis_volume_fault_validation import independent_audit as fault
        self._stack=ExitStack()
        def basal_key(operator,contact_parameters=None):
            geometry_hash=digest(operator.geometry)
            if geometry_hash!=self.producer.bindings.geometry_sha256:raise ValueError('GEOMETRY_BINDING_CHANGED: rebind required')
            # Exhaustive non-constitutive inputs consumed by the original basal
            # assembler; C/damage is deliberately absent because it is unused.
            return (type(operator).__module__,type(operator).__qualname__,geometry_hash,
                    operator.source_checkpoint_sha256,operator.source_identity,operator.time_myr,
                    operator.ndof,operator.dofs,operator.cohort_ids,operator.face_indices,contact_parameters)
        targets=[(self.producer,op,'conical_strain_operators',None),
                 (self.producer,op,'conical_radial_moments',None),
                 (self.producer,dynamics,'assemble_genesis_volume_basal',basal_key)]
        targets += [(self.auditor,q,name,None) for name in ('angular_rule','radial_rule','conical_scales','independent_strain_operators','mass_matrix','prescribed_traction','basal_drag_and_force')]
        targets.append((self.auditor,fault,'physical_face_geometry',None))
        for store,module,name,key in targets:
            self._stack.enter_context(patch.object(module,name,store.memo(getattr(module,name),key)))
        return self

    def __exit__(self,*exc):
        self._stack.__exit__(*exc);self._stack=None

    def report(self):return dict(producer=self.producer.report(),independent_auditor=self.auditor.report())


def bindings_for_geometry(geometry,solver_sha256):
    return Bindings(digest(geometry),digest((geometry.cohorts,geometry.column_mass_kg,geometry.density_kg_m3)),
                    digest(geometry.source_packet_json),solver_sha256)
