"""Explicit numerical closure, not recovery of a unique historical subcell field."""
import math


def partition_cell(*,face,layer,q_lo,q_hi,mass_kg,energy_j,anchors,lower_h,upper_h):
    """Anchor disjoint material intervals at accepted mean enthalpy.

    Residual unowned mass gets one constant h chosen by the cell energy budget.
    Anchors: id,q_lo,q_hi,specific_h. No new cut locations except their bounds.
    """
    if not (0<=q_lo<q_hi<=1 and mass_kg>0 and energy_j>=0 and 0<=lower_h<=upper_h):raise ValueError('Cell bounds')
    for a in anchors:
        if not (q_lo<=a['q_lo']<a['q_hi']<=q_hi and lower_h<=a['specific_h']<=upper_h):raise ValueError('Anchor bounds')
    ordered=sorted(anchors,key=lambda a:a['q_lo'])
    if any(b['q_lo']<a['q_hi'] for a,b in zip(ordered,ordered[1:])):raise ValueError('Overlapping material anchors')
    cuts=sorted({q_lo,q_hi,*[a[k] for a in anchors for k in ('q_lo','q_hi')]})
    rows=[]
    for lo,hi in zip(cuts,cuts[1:]):
        owners=[a for a in anchors if a['q_lo']<=lo and hi<=a['q_hi']]
        owner=owners[0] if owners else None;m=mass_kg*(hi-lo)/(q_hi-q_lo)
        rows.append(dict(face=face,layer=layer,q_lo=lo,q_hi=hi,mass_kg=m,material_id=None if owner is None else owner['id'],specific_h=None if owner is None else owner['specific_h']))
    used=math.fsum(c['mass_kg']*c['specific_h'] for c in rows if c['material_id'] is not None)
    free_mass=math.fsum(c['mass_kg'] for c in rows if c['material_id'] is None)
    if free_mass:
        residual=(energy_j-used)/free_mass
        if not lower_h<=residual<=upper_h:raise ValueError('Residual enthalpy violates admissible bounds')
        for c in rows:
            if c['material_id'] is None:c['specific_h']=residual
    elif not math.isclose(used,energy_j,rel_tol=2e-13,abs_tol=1e-9):raise ValueError('Fully anchored cell has inconsistent enthalpy')
    for c in rows:c['energy_j']=c['mass_kg']*c['specific_h']
    if not math.isclose(math.fsum(c['energy_j'] for c in rows),energy_j,rel_tol=2e-13,abs_tol=1e-9):raise ValueError('Energy partition')
    return rows


def mean_preserving_affine_envelope(cell,left_h,right_h,*,solidus_h=None):
    """Extreme symmetric affine alternative within neighbouring-mean bounds.

    +/- amplitude have identical mean and integral. This proves nonuniqueness;
    it is a sensitivity alternative, not a prescribed physical gradient.
    """
    h=cell['specific_h'];lo=min(left_h,h,right_h);hi=max(left_h,h,right_h)
    if solidus_h is not None:hi=min(hi,solidus_h)
    amplitude=max(0.,min(h-lo,hi-h))
    return dict(mean_h=h,left_h=h-amplitude,right_h=h+amplitude,amplitude_h=amplitude,
                mean_preserved=True,additional_cuts=0)
