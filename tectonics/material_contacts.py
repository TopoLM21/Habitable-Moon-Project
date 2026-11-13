"""Material banks are geometry; receiver labels are not physical contacts.

Only coincident, oppositely oriented bank arcs define line contacts. Pairing is
recomputed, so tangential sliding can change material neighbours. This operator
does not repair gaps/overlaps, assign their matter, or infer pressure from them.
It supplies an independent geometric contract and a fail-closed validity check.
"""
from dataclasses import dataclass
import hashlib,math
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class MaterialContact:
    bank_a:int
    bank_b:int
    owner_a:int
    owner_b:int
    start:np.ndarray
    end:np.ndarray
    normal_a_to_b:np.ndarray
    length_km:float


def arc_overlaps(a,b):
    """Intervals of two minor arcs on the same great circle; handles sliding."""
    u,v=a;t=v-u*(u@v);t/=np.linalg.norm(t);pole=np.cross(u,t)
    length=math.atan2(np.linalg.norm(np.cross(u,v)),u@v)
    start=math.atan2(b[0]@t,b[0]@u)
    other=math.atan2(np.linalg.norm(np.cross(*b)),b[0]@b[1])
    end=start+(other if np.cross(*b)@pole>=0 else -other)
    lo,hi=sorted((start,end));result=[]
    for shift in (-2*math.pi,0.,2*math.pi):
        x=max(0.,lo+shift);y=min(length,hi+shift)
        if y>x:result.append((x,y,np.cos(x)*u+np.sin(x)*t,np.cos(y)*u+np.sin(y)*t))
    return result


def _interval_coverage(intervals):
    """Union length and extra coverage; overlaps must not cancel open parts."""
    if not intervals:return 0.,0.
    ordered=sorted(intervals);lo,hi=ordered[0];pieces=[]
    for start,end in ordered[1:]:
        if start>hi:pieces.append(hi-lo);lo,hi=start,end
        else:hi=max(hi,end)
    pieces.append(hi-lo)
    union=math.fsum(pieces)
    extra=max(0.,math.fsum(end-start for start,end in ordered)-union)
    return union,extra


class MaterialBoundaryRegistry:
    """One bank per material side; no independent area or mass is introduced."""
    def __init__(self,mesh,owners):
        self.owners=np.asarray(owners,dtype=np.int64)
        edges=np.asarray(mesh.shared_edges,dtype=np.int64)
        e=edges[self.owners[edges[:,0]]!=self.owners[edges[:,1]]]
        self.faces=e[:,:2].reshape(-1);self.bank_owners=self.owners[self.faces]
        self.reference=np.repeat(mesh.vertices[e[:,2:]],2,axis=0)
        normal=np.cross(self.reference[::2,0],self.reference[::2,1]);normal/=np.linalg.norm(normal,axis=1)[:,None]
        normal*=np.where(np.einsum('ij,ij->i',normal,mesh.centroids[e[:,1]]-mesh.centroids[e[:,0]])>=0,1.,-1.)[:,None]
        self.outward=np.stack([normal,-normal],axis=1).reshape(-1,3)
        self.identity=hashlib.sha256(e.tobytes()+self.owners.tobytes()+self.reference.tobytes()).hexdigest()

    def evaluate(self,quaternions,radius,angular_tolerance=1e-10):
        matrices=Rotation.from_quat(quaternions).as_matrix()[self.bank_owners]
        banks=np.einsum('nij,nkj->nki',matrices,self.reference);normals=np.einsum('nij,nj->ni',matrices,self.outward)
        lengths=np.arctan2(np.linalg.norm(np.cross(banks[:,0],banks[:,1]),axis=1),np.einsum('ij,ij->i',banks[:,0],banks[:,1]))*radius
        # The dot-product prefilter is only cheap rejection. The vector sum
        # enforces the actual angular tolerance, without cancellation near -1.
        i,j=np.nonzero(np.triu(normals@normals.T<-1+1e-12,1))
        valid=(self.bank_owners[i]!=self.bank_owners[j])&(np.linalg.norm(normals[i]+normals[j],axis=1)<=angular_tolerance)
        contacts=[];intervals=[[] for _ in lengths]
        for a,b in zip(i[valid],j[valid]):
            for lo,hi,start,end in arc_overlaps(banks[a],banks[b]):
                length=(hi-lo)*radius
                if length<=angular_tolerance*radius:continue # numerical endpoint only, not a physical length threshold
                intervals[a].append((lo,hi))
                contacts.append(MaterialContact(int(a),int(b),int(self.bank_owners[a]),int(self.bank_owners[b]),start,end,normals[a].copy(),length))
            # Each side uses its own arc coordinate, including after sliding.
            intervals[b].extend((lo,hi) for lo,hi,_,_ in arc_overlaps(banks[b],banks[a]) if hi-lo>angular_tolerance)
        coverage=np.asarray([_interval_coverage(parts) for parts in intervals],dtype=float).reshape(-1,2)*radius
        middle=np.sum(banks,axis=1);middle/=np.linalg.norm(middle,axis=1)[:,None]
        # Signed distance of the old partner's midpoint to the first bank's
        # supporting great circle. Diagnostic only: not an overlap-area budget.
        gap=np.arcsin(np.clip(np.einsum('ij,ij->i',middle[1::2],normals[::2]),-1,1))*radius
        diag=dict(identity_sha256=self.identity,banks=len(banks),material_owner_count=len(np.unique(self.owners)),
                  bank_length_sum_km=math.fsum(lengths),contact_fragments=len(contacts),
                  coincident_contact_length_km=math.fsum(c.length_km for c in contacts),
                  unmatched_bank_length_km=math.fsum(np.maximum(lengths-coverage[:,0],0.)),
                  multiply_covered_bank_length_km=math.fsum(coverage[:,1]),
                  old_partner_signed_separation_min_km=float(min(gap,default=0.)),
                  old_partner_signed_separation_max_km=float(max(gap,default=0.)),
                  angular_coincidence_tolerance=angular_tolerance,
                  scope='coincident material bank arcs; unmatched banks are unresolved gap/overlap/free-boundary geometry, not new plates')
        return contacts,diag

    def require_shared_support(self,quaternions,radius):
        contacts,diag=self.evaluate(quaternions,radius)
        tolerance=2e-10*radius*max(diag['banks'],1)
        if max(diag['unmatched_bank_length_km'],diag['multiply_covered_bank_length_km'])>tolerance:
            raise ValueError(f"Material contacts not geometrically closed: unmatched {diag['unmatched_bank_length_km']:.9g} km, multiply covered {diag['multiply_covered_bank_length_km']:.9g} km; a gap/overlap or deformable-surface law is required")
        return contacts,diag
