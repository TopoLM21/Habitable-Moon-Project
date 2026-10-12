"""One fixed mixed-mode elastic coupling action, exact represented operands."""
from fractions import Fraction as F
import numpy as np
from analysis.genesis_directional_stiffness_validation.expression import Graph, build as strain_graph
from analysis.genesis_volume_operand_validation.independent_coefficients import _literal_vertices, rational_array


Q_DIAGNOSTIC = [F((-1)**i, 8192*(i+1)) for i in range(18)]


def build(raw, probe, center):
    if raw['seed_radius_m'] != 100.:
        raise ValueError('Only the current 100 m source is admitted')
    base = strain_graph(raw, probe, center)
    g = Graph()
    g.nodes = [tuple(n) for n in base['nodes']]
    g.cache = {n:i for i,n in enumerate(g.nodes)}
    u,v,lam = [g.node('x',i) for i in range(3)]
    N = [g.sub(g.sub(g.const(1),u),v),u,v]
    X = np.asarray(raw['section']['triangle_m'])
    B0,B1 = _literal_vertices(X,np.asarray(raw['basis_xyz']))
    B = [[g.sum(g.mul(N[k],g.add(g.scale(lam,B0[k,j,i]),g.const(B1[k,j,i])))
                for k in range(3)) for i in range(18)] for j in range(6)]
    C = rational_array(raw['elasticity_pa'])
    assert np.array_equal(C,C.T)
    Ga = [base['outputs']['strain_'+str(j)] for j in range(6)]
    stress = [g.sum(g.scale(Ga[j],C[i,j]) for j in range(6)) for i in range(6)]
    xr = rational_array(X)
    det = sum(xr[0,j]*(xr[1,(j+1)%3]*xr[2,(j+2)%3]-xr[1,(j+2)%3]*xr[2,(j+1)%3]) for j in range(3))
    assert det > 0
    b = [g.scale(g.sum(g.mul(B[j][i],stress[j]) for j in range(6)),det) for i in range(18)]
    Bq = [g.sum(g.scale(B[j][i],Q_DIAGNOSTIC[i]) for i in range(18)) for j in range(6)]
    # Transposed order: Ga^T C^T Bq; not a sum of the forward b outputs.
    adjoint = g.scale(g.sum(g.mul(Ga[j],g.sum(g.scale(Bq[i],C[i,j]) for i in range(6))) for j in range(6)),det)
    outputs = {k:base['outputs'][k] for k in ('sphere','gamma')}
    outputs.update({'b_'+str(i):node for i,node in enumerate(b)})
    outputs['adjoint_work'] = adjoint
    return dict(nodes=g.nodes,outputs=outputs), B, Ga, str(det)


def bounds(ranges, box):
    volume = (box[0][1]-box[0][0])*(box[1][1]-box[1][0])*(box[2][1]-box[2][0])
    outside = ranges['sphere'][0] > 0
    inside = ranges['sphere'][1] <= 0
    phi = ranges['gamma']
    H = (F(1,2),)*2 if phi[0]>0 else (F(-1,2),)*2 if phi[1]<0 else (F(-1,2),F(1,2))
    out = {}
    for key,value in ranges.items():
        if key in ('sphere','gamma'):continue
        products = [x*y for x in value for y in H]
        if not inside:products.append(F())
        out[key] = (F(),F()) if outside else (volume*min(products),volume*max(products))
    return out


def contract(vector, q=Q_DIAGNOSTIC):
    return tuple(sum(min(q[i]*x for x in vector[i]) if e==0 else max(q[i]*x for x in vector[i])
                     for i in range(18)) for e in (0,1))


def observed(bound):
    lo,hi = bound
    gap = lo if lo>0 else -hi if hi<0 else F()
    return dict(sign='positive' if lo>0 else 'negative' if hi<0 else 'zero' if lo==hi==0 else 'unresolved',
        relative_radius_bound=str((hi-lo)/2/gap) if gap else None,
        relative_2e_minus_6_certified=(hi==lo==0 or bool(gap and (hi-lo)/2 <= F(1,500000)*gap)))
