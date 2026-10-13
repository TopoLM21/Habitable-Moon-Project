"""Opt-in full angular stiffness for the manufactured vector-shell diagnostic.

The frozen Driven/GridControl paths retain their former reduced integration.
Three degree-two triangle points integrate B.T C B exactly for face-constant C.
No physical parameter or checkpoint representation is changed here.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from analysis.genesis_volume_continuation_validation import independent_audit as audit

ANGULAR_N = np.array([[2/3, 1/6, 1/6], [1/6, 2/3, 1/6], [1/6, 1/6, 2/3]])
ANGULAR_W = np.full(3, 1/3)


def frame_for(triangle):
    tangent = triangle[1]-triangle[0]; tangent /= np.linalg.norm(tangent)
    normal = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0]); normal /= np.linalg.norm(normal)
    return np.column_stack((tangent, np.cross(normal, tangent), normal))


def blocks(triangle, inner, N=ANGULAR_N, wa=ANGULAR_W):
    frame = frame_for(triangle)
    b0, b1 = audit.independent_strain_operators(triangle, frame, N)
    radial = np.array([.5-.5/np.sqrt(3), .5+.5/np.sqrt(3)])
    lam = np.cbrt(1-radial*(1-inner**3))
    B = b0[None]+b1[None]/lam[:, None, None, None]
    volume = np.linalg.det(triangle.T)*(1-inner**3)/6
    weights = np.broadcast_to(wa[None]*volume/2, (2,len(wa))).copy()
    points = lam[:,None,None]*(N@triangle)[None]
    return B, weights, points, frame


def first_step(grid, years=.5, integrated=True):
    """One passive, zero-history BE step; identical material, mass and load."""
    from .driven_shell import CONFIG
    a = grid.a
    dt = years*31557600.
    relaxation = 1+dt*CONFIG['young_pa']/CONFIG['maxwell_viscosity_pa_s']
    if integrated:
        inner = 1-CONFIG['thickness_km']/CONFIG['radius_km']
        local = [blocks(tri, inner) for tri in a.mesh.vertices[a.mesh.faces]*CONFIG['radius_km']*1000]
        B = np.array([x[0].reshape(-1,6,18) for x in local])
        w = np.array([x[1].ravel() for x in local])
        C = a.C.reshape(a.nf,2,6,6)[:,0]
    else:
        B = a.B.reshape(a.nf,2,6,18)
        w = a.w.reshape(a.nf,2)
        C = a.C.reshape(a.nf,2,6,6)[:,0]
    Klocal = np.einsum('fqki,fkl,fqlj,fq->fij',B,C,B,w)/relaxation
    K = sparse.coo_matrix((Klocal.ravel(),(a.row,a.col)),shape=a.M.shape).tocsr()
    A = a.M/dt**2+K
    active = A[a.free][:,a.free]; scale = 1/np.sqrt(active.diagonal())
    u = np.zeros(a.nd)
    u[a.free] = scale*spsolve(sparse.diags(scale)@active@sparse.diags(scale),scale*a.force[a.free])
    ids = a.ids.reshape(a.nf,2,18)[:,0]
    strain = np.einsum('fqij,fj->fqi',B,u[ids])/relaxation
    stress = np.einsum('fij,fqj->fqi',C,strain)
    mean = np.einsum('fq,fqi->fi',w,stress)/w.sum(axis=1)[:,None]
    tensor = np.zeros((a.nf,3,3)); tensor[:,range(3),range(3)] = mean[:,:3]
    for k,(i,j) in enumerate(((0,1),(0,2),(1,2)),3):tensor[:,i,j]=tensor[:,j,i]=mean[:,k]
    world = np.einsum('fai,fij,fbj->fab',grid.frames,tensor,grid.frames)
    momentum = np.linalg.norm((A@u-a.force)[a.free])/np.linalg.norm(a.force[a.free])
    elastic = .5*np.einsum('fqi,fqi,fq->',strain,stress,w)
    kinetic = .5*u@(a.M@u)/dt**2
    work = a.force@u
    # BE first step: numerical elastic=elastic, numerical kinetic=kinetic.
    balance = 2*elastic*relaxation+2*kinetic-work
    assert momentum < 1e-8 and abs(balance)/max(abs(work),1.) < 5e-8
    return world,dict(momentum=float(momentum),energy_relative=float(abs(balance)/max(abs(work),1.)),
        elastic_j=float(elastic),work_j=float(work),operator_arrays_bytes=B.nbytes+w.nbytes+Klocal.nbytes)
