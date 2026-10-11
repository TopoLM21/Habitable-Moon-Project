"""Independent exact triangle-intersection audit of a completed angular mesh."""
from fractions import Fraction as F
from pathlib import Path
import argparse,json,hashlib

def decode(rows):return tuple(tuple(F(int(a),int(b)) for a,b in row[1:]) for row in rows)
def cross(a,b,p):return (b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0])
def area(p):return abs(sum(p[i][0]*p[(i+1)%len(p)][1]-p[i][1]*p[(i+1)%len(p)][0] for i in range(len(p))))/2 if len(p)>2 else F()
def intersection(p,q):
    # Convex half-plane clipping, independent of source coverage and event code.
    p=list(p)
    for a,b in zip(q,q[1:]+q[:1]):
        if not p:return F()
        result=[];prior=p[-1];x=cross(a,b,prior)
        for current in p:
            y=cross(a,b,current)
            if (x<0)!=(y<0):
                t=x/(x-y)
                result.append(tuple(v+t*(w-v) for v,w in zip(prior,current)))
            if y>=0:result.append(current)
            prior,x=current,y
        p=result
    return area(p)
def box(p):return tuple((min(v[j] for v in p),max(v[j] for v in p)) for j in range(2))
def overlap(a,b):return all(x[1]>y[0] and y[1]>x[0] for x,y in zip(a,b))
def pin(p):return dict(path=str(p),bytes=p.stat().st_size,raw_sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def check(roots,cells):
    assert all(cross(*c)>0 for c in cells+roots),'Nonpositive triangle'
    assert all(intersection(c,r)==0 for i,c in enumerate(roots) for r in roots[:i]),'Initial roots overlap'
    assert sum(map(area,roots),F())==sum(map(area,cells),F()),'Exact area mismatch'
    boxes=list(map(box,cells));checks=0
    for i,c in enumerate(cells):
        assert sum((intersection(c,r) for r in roots),F())==area(c),'Cell outside the initial support'
        for j in range(i):
            if overlap(boxes[i],boxes[j]):
                assert intersection(c,cells[j])==0,'Positive-area cell overlap'
                checks+=1
    return dict(cells=len(cells),exact_positive_area_intersection_checks=checks,
        all_cells_inside_initial_support=True,no_positive_area_overlap=True,
        exact_total_parameter_area_preserved=True)

def selftest():
    z=F();o=F(1);h=F(1,2)
    root=((z,z),(o,z),(z,o));a=((z,z),(h,z),(z,h))
    assert intersection(root,a)==F(1,8)
    assert intersection(root,root)==F(1,2)
    outside=((o,o),(o+1,o),(o,o+1))
    assert intersection(root,outside)==0
    middle=((h,z),(h,h),(z,h));b=((h,z),(o,z),(h,h));c=((z,h),(h,h),(z,o))
    check([root],[a,b,c,middle])
    try:check([root],[a,a,b,c])
    except AssertionError:pass
    else:raise AssertionError('Overlap and same-area missing region were accepted')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--control',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--cases',type=int,nargs='+',default=[0,1,2,3])
    args=parser.parse_args();selftest()
    if args.control is None:print('Independent intersection audit self-tests PASS');return
    assert args.output is not None and not args.output.exists();args.output.mkdir()
    rows=[]
    assert len(set(args.cases))==len(args.cases) and all(0<=i<4 for i in args.cases)
    for i in args.cases:
        directory=args.control/f'case_{i:03d}'
        mesh=json.loads((directory/'partition.json').read_bytes());angular=json.loads((directory/'angular.json').read_bytes())
        roots=[decode(v) for v in mesh['root_cells']]
        cells=[decode(c['vertices']) for c in angular['final_cells']]+[decode(c['vertices']) for c in angular['discarded_cells']]
        result=dict(case=i,**check(roots,cells),angular_pin=pin(directory/'angular.json'),mesh_pin=pin(directory/'partition.json'))
        rows.append(result);print(json.dumps(result),flush=True)
    report=dict(format='genesis-independent-final-mesh-intersection-audit-1',cases=rows,auditor_pin=pin(Path(__file__)))
    with (args.output/'report.json').open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')

if __name__=='__main__':main()
