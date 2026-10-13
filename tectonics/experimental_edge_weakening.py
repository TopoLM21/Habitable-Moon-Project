"""Opt-in competing damage on vector-shell elastic blocks, with no fracture heat.

Face damage is the maximum incident edge damage. Old and edge damage compete
through max, not multiplication. This is a mesh-dependent diffuse weakening law,
not a traction-free discontinuity or calibrated fracture energy.
"""
import numpy as np

MODE='weakening_then_geological_retirement'


def enabled(m):
    return m.config.get('experimental_geological_edge_fatigue',{}).get('mode')==MODE


def face_damage(mesh,edge):
    edge=np.asarray(edge,dtype=float)
    if edge.shape!=(len(mesh.shared_edges),) or not np.isfinite(edge).all() or np.any((edge<0)|(edge>1)):
        raise ValueError('EXPERIMENTAL_EDGE_INVALID_STATE')
    result=np.zeros(mesh.cell_count)
    for d,(f,g,_,_) in zip(edge,mesh.shared_edges):result[f]=max(result[f],d);result[g]=max(result[g],d)
    return result


def scale(m,old_damage,edge=None):
    if not enabled(m):return m.scale(old_damage)
    if edge is None:edge=m.state.get('experimental_edge_fatigue',{}).get('damage',np.zeros(len(m.mesh.shared_edges)))
    return m.scale(np.maximum(old_damage,face_damage(m.mesh,edge)))


def commit_damage(m,new_edge):
    """Damage substep at fixed memory/mass/enthalpy/velocity. Release -> reserve."""
    previous=m.state.get('experimental_edge_fatigue',{}).get('damage',np.zeros(len(m.mesh.shared_edges)))
    new_edge=np.asarray(new_edge);face_damage(m.mesh,new_edge)
    if np.any(new_edge<previous):raise ValueError('EDGE_HEALING_NOT_SUPPORTED')
    before=scale(m,m.state['damage'],previous);after=scale(m,m.state['damage'],new_edge)
    base=np.array([.5*np.asarray(e).ravel()@np.kron(m.G[f],m.C[f])@np.asarray(e).ravel() for f,e in enumerate(m.state['memory'])])
    for x in m.state.get('born_material',[]):
        e=np.array(x['memory']);base[x['face']]+=.5*x['volume_m3']*e@m.C[x['face']]@e
    releases=base*(before-after)
    if np.any(releases<0) or not np.isfinite(releases).all():raise ValueError('EDGE_RELEASE_SIGN')
    release=float(releases.sum())
    m.state['experimental_edge_fracture_reserve_j']=float(m.state.get('experimental_edge_fracture_reserve_j',0.))+release
    m.state.setdefault('experimental_edge_fatigue',{})['damage']=new_edge.tolist()
    return dict(release_j=release,elastic_before_j=float(base@before),elastic_after_j=float(base@after),
                face_release_j=releases.tolist(),scale_before=before,scale_after=after)
