"""Exact rational affine Gamma cuts, retaining both sides and their measure.

No edge parameter rounding is permitted in this lane. The earlier rounded
experimental mesh is preserved in genesis_volume_event_mesh.py.
"""
from fractions import Fraction as F
from .genesis_volume_event_partition import literal_cell,signed_double_area,source_events,ratio,digest


def event_cut(vertices,event):
    vertices=literal_cell(vertices)
    if event.degree>1:raise ValueError('This event cut requires an affine event')
    values=tuple(event.evaluate(v) for v in vertices)
    if min(values)>=0 or max(values)<=0:return (vertices,),dict(split=False,edge_parameters=[])
    crossing={};records=[]
    for i in range(3):
        j=(i+1)%3;a,b=values[i],values[j]
        if a*b<0:
            exact=a/(a-b)
            if not 0<exact<1:
                raise AssertionError('Exact crossing escaped the open edge')
            point=tuple(x+exact*(y-x) for x,y in zip(vertices[i],vertices[j]))
            residual=event.evaluate(point)
            if residual:
                raise AssertionError('Exact crossing does not lie on the affine event')
            crossing[i]=point;records.append(dict(edge=i,exact=ratio(exact),represented=ratio(exact),
                parameter_error=ratio(F()),event_residual=ratio(residual)))
    children=[]
    for sign in (-1,1):
        polygon=[]
        for i in range(3):
            if sign*values[i]>=0:polygon.append(vertices[i])
            if i in crossing:polygon.append(crossing[i])
        polygon=[p for i,p in enumerate(polygon) if p!=polygon[i-1]]
        for i in range(1,len(polygon)-1):
            cell=(polygon[0],polygon[i],polygon[i+1])
            if signed_double_area(cell)>0:children.append(literal_cell(cell))
    if sum(signed_double_area(c) for c in children)!=signed_double_area(vertices):
        raise AssertionError('Event cut lost exact source measure')
    return tuple(children),dict(split=True,edge_parameters=records)


def event_mesh(raw,cells,*,max_depth=24):
    if type(max_depth) is not int or max_depth<0:raise ValueError('Invalid event depth')
    roots=tuple(literal_cell(c) for c in cells);events,_=source_events(raw)
    current=[(i,0,c) for i,c in enumerate(roots)];cuts=[]
    for name in ('Gamma_cohort_lower','Gamma_cohort_upper'):
        output=[]
        for root,depth,cell in current:
            if depth>=max_depth:
                output.append((root,depth,cell));continue
            children,record=event_cut(cell,events[name])
            if record['split']:cuts.append(dict(root=root,event=name,**record))
            output.extend((root,depth+int(record['split']),c) for c in children)
        current=output
    for i,c in enumerate(roots):
        if sum(signed_double_area(v) for root,depth,v in current if root==i)!=signed_double_area(c):
            raise AssertionError('Initial source root measure changed')
    certificate=dict(format='genesis-exact-Gamma-exact-measure-mesh-1',edge_parameters_exact=True,
        root_cells=[[[ratio(v) for v in row] for row in c] for c in roots],
        cells=[dict(root=i,depth=depth,vertices=[[ratio(v) for v in row] for row in c]) for i,depth,c in current],cuts=cuts,
        exact_partition_of_supplied_cells=True,exact_event_alignment=all(not (min(events[name].evaluate(n) for n in c)<0<max(events[name].evaluate(n) for n in c)) for _,_,c in current for name in ('Gamma_cohort_lower','Gamma_cohort_upper')),
        source_cells_removed=0,source_bank_assignment_from_mesh_labels=False,
        angular_accuracy_certified=False)
    certificate['sha256']=digest(certificate)
    return tuple(c for _,depth,c in current),certificate
