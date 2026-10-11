"""Exact event signs and conservative subdivision of supplied angular cells.

Adapted from the preserved 20261011 source-event prototype (SHA256
8a8d939a5973455bebd31a50b2e9ae78a6ddd3eb325e469e162871022cfdb66a). No physical source or material is changed.
Coverage here means exact partition of the supplied cells, NOT certification
that those cells cover the physical source or that floating mapped nodes do.
"""
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from types import MappingProxyType

import numpy as np


def _literal_source(raw):
    section = raw['section']
    def array(value, shape):
        value = np.asarray(value, dtype=float)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError('Invalid literal source geometry')
        return value
    def exact(value):
        return Fraction(*float(value).as_integer_ratio())
    X, A, P = (array(section[key], (3,3)) for key in
               ('triangle_m','displacement_A_m','slope_P_m'))
    # These binary64 sums are the declared frozen birth convention.
    C, D = X+P, A-P
    center = array(raw['seed_center_world_m'], (3,))
    normal = array(section['normal_xyz'], (3,))
    bounds = tuple(map(exact,array(section['lambda_bounds'], (2,))))
    radius, offset = exact(raw['seed_radius_m']),exact(section['offset_m'])
    if not 0 < bounds[0] < bounds[1] or radius <= 0:
        raise ValueError('Invalid source bounds or radius')
    return dict(C=[tuple(map(exact,C[:,j])) for j in range(3)],
        D=[tuple(map(exact,D[:,j])) for j in range(3)],
        center=tuple(map(exact,center)),normal=tuple(map(exact,normal)),
        radius=radius,offset=offset,bounds=bounds)


def ratio(value):
    value = Fraction(value)
    return [str(value.numerator), str(value.denominator)]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


@dataclass(frozen=True)
class Polynomial:
    variables: int
    coefficients: object

    def __post_init__(self):
        clean = {}
        for powers, value in self.coefficients.items():
            powers = tuple(powers)
            if len(powers) != self.variables or any(type(p) is not int or p < 0 for p in powers):
                raise ValueError('Polynomial exponent shape differs')
            value = Fraction(value)
            if value:
                clean[powers] = clean.get(powers, Fraction())+value
        object.__setattr__(self, 'coefficients', MappingProxyType({p:v for p,v in clean.items() if v}))

    @property
    def degree(self):
        return max((sum(p) for p in self.coefficients), default=0)

    def evaluate(self, coordinates):
        coordinates = tuple(map(Fraction, coordinates))
        if len(coordinates) != self.variables:
            raise ValueError('Polynomial evaluation coordinate shape differs')
        result = Fraction()
        for powers, coefficient in self.coefficients.items():
            term = coefficient
            for x, p in zip(coordinates, powers):
                term *= x**p
            result += term
        return result

    def metadata(self):
        return dict(variables=self.variables, degree=self.degree,
            coefficients=[dict(powers=list(p), value=ratio(v))
                          for p,v in sorted(self.coefficients.items())])


def constant(value, variables=3):
    return Polynomial(variables, {(0,)*variables:Fraction(value)})


def variable(index, variables=3):
    powers = [0]*variables
    powers[index] = 1
    return Polynomial(variables, {tuple(powers):Fraction(1)})


def add(*polynomials):
    if not polynomials or len({p.variables for p in polynomials}) != 1:
        raise ValueError('Polynomial addition dimension differs')
    coefficients = {}
    for polynomial in polynomials:
        for powers, value in polynomial.coefficients.items():
            coefficients[powers] = coefficients.get(powers, Fraction())+value
    return Polynomial(polynomials[0].variables, coefficients)


def scale(polynomial, value):
    return Polynomial(polynomial.variables, {p:Fraction(value)*v for p,v in polynomial.coefficients.items()})


def multiply(left, right):
    if left.variables != right.variables:
        raise ValueError('Polynomial product dimension differs')
    coefficients = {}
    for lp, lv in left.coefficients.items():
        for rp, rv in right.coefficients.items():
            powers = tuple(a+b for a,b in zip(lp,rp))
            coefficients[powers] = coefficients.get(powers, Fraction())+lv*rv
    return Polynomial(left.variables, coefficients)


def power(polynomial, exponent):
    if type(exponent) is not int or exponent < 0:
        raise ValueError('Polynomial power must be nonnegative integer')
    result = constant(1, polynomial.variables)
    for _ in range(exponent):
        result = multiply(result, polynomial)
    return result


def dot(left, right):
    if len(left) != len(right) or not left:
        raise ValueError('Polynomial dot-product shape differs')
    return add(*(multiply(a,b) for a,b in zip(left,right)))


def linear(weights):
    return add(*(scale(variable(i), weight) for i,weight in enumerate(weights)))


def source_events(raw):
    """Use the preserved float64 C=X+P,D=A-P, then exact source algebra."""
    source = _literal_source(raw)
    q = tuple(linear(weights) for weights in source['C'])
    d = tuple(linear(weights) for weights in source['D'])
    r = tuple(add(component, constant(-center)) for component,center in zip(d,source['center']))
    A, B = dot(q,q), dot(q,r)
    E = add(dot(r,r), constant(-source['radius']**2))
    g = add(*(scale(component,n) for component,n in zip(q,source['normal'])))
    h = add(*(scale(component,n) for component,n in zip(d,source['normal'])), constant(-source['offset']))
    Delta = add(power(B,2), scale(multiply(A,E),-1))
    events = dict(A=A,B=B,E=E,Delta=Delta,g=g,h=h)
    for label, endpoint in zip(('lower','upper'),source['bounds']):
        events['sphere_cohort_'+label] = add(scale(A,endpoint**2),scale(B,2*endpoint),E)
        events['sphere_derivative_half_'+label] = add(scale(A,endpoint),B)
        events['Gamma_cohort_'+label] = add(scale(g,endpoint),h)
    events['Gamma_sphere_resultant'] = add(multiply(A,power(h,2)),
        scale(multiply(multiply(B,h),g),-2),multiply(E,power(g,2)))
    events.update({f'N_{i}':variable(i) for i in range(3)})
    if max(p.degree for p in events.values()) > 4:
        raise AssertionError('Source-event polynomial degree exceeds four')
    source_literal = dict(C=[[ratio(v) for v in row] for row in source['C']],
        D=[[ratio(v) for v in row] for row in source['D']],
        center=[ratio(v) for v in source['center']],normal=[ratio(v) for v in source['normal']],
        radius=ratio(source['radius']),offset=ratio(source['offset']),
        lambda_bounds=[ratio(v) for v in source['bounds']])
    return events, source_literal


def on_cell(polynomial, vertices):
    """Substitute N=v0+s(v1-v0)+t(v2-v0), preserving all three N values."""
    vertices = tuple(tuple(map(Fraction,row)) for row in vertices)
    if len(vertices) != 3 or any(len(row) != polynomial.variables for row in vertices):
        raise ValueError('Cell vertex shape differs')
    operands = tuple(Polynomial(2,{(0,0):vertices[0][i],
        (1,0):vertices[1][i]-vertices[0][i],
        (0,1):vertices[2][i]-vertices[0][i]}) for i in range(polynomial.variables))
    result = constant(0,2)
    for powers, coefficient in polynomial.coefficients.items():
        term = constant(coefficient,2)
        for operand, exponent in zip(operands,powers):
            term = multiply(term,power(operand,exponent))
        result = add(result,term)
    return result


def bernstein_coefficients(polynomial, vertices, degree=None):
    local = on_cell(polynomial,vertices)
    degree = local.degree if degree is None else degree
    if type(degree) is not int or not local.degree <= degree <= 4:
        raise ValueError('Bernstein degree must contain the polynomial through degree four')
    result = {}
    for b in range(degree+1):
        for c in range(degree-b+1):
            value = Fraction()
            for (i,j),coefficient in local.coefficients.items():
                if i <= b and j <= c:
                    value += coefficient*Fraction(math.comb(b,i)*math.comb(c,j),
                        math.comb(degree,i+j)*math.comb(i+j,i))
            result[(degree-b-c,b,c)] = value
    return result


def sign_certificate(polynomial, vertices):
    coefficients = bernstein_coefficients(polynomial,vertices)
    lower, upper = min(coefficients.values()), max(coefficients.values())
    if lower > 0:
        classification = 'STRICT_POSITIVE'
    elif upper < 0:
        classification = 'STRICT_NEGATIVE'
    elif lower == upper == 0:
        classification = 'IDENTICALLY_ZERO_REQUIRES_SEMANTIC_CLASSIFICATION'
    else:
        classification = 'ZERO_POSSIBLE_REQUIRES_SPLIT_OR_RETAIN_OPEN'
    body = dict(format='genesis-source-event-Bernstein-certificate-1',
        polynomial=polynomial.metadata(), vertices=[[ratio(v) for v in row] for row in vertices],
        bernstein_coefficients=[dict(powers=list(p),value=ratio(v)) for p,v in sorted(coefficients.items())],
        lower=ratio(lower),upper=ratio(upper),classification=classification,
        event_free_sign_certified=classification in ('STRICT_POSITIVE','STRICT_NEGATIVE'),
        convex_hull_sign_bound_certified=True, original_support_discarded=False,
        source_history_reinitialized=False, production_algorithm_adopted=True,
        angular_accuracy_certified=False,physical_contraction_certified=False,
        available_for_creation_funding=False)
    body['certificate_sha256'] = digest(body)
    return body


def verify_sign_certificate(polynomial,vertices,certificate):
    if sign_certificate(polynomial,vertices) != certificate:
        raise ValueError('Canonical Bernstein certificate differs')
    return True


def literal_cell(vertices):
    def exact(value):
        return Fraction(*value.as_integer_ratio()) if isinstance(value,(float,np.floating)) else Fraction(value)
    vertices=tuple(tuple(exact(v) for v in row) for row in vertices)
    if len(vertices)!=3 or any(len(row)!=3 for row in vertices):
        raise ValueError('A cell needs three literal three-component vertices')
    if any(min(row)<0 or abs(sum(row)-1)>Fraction(1,2**45) for row in vertices):
        raise ValueError('Literal cell leaves original simplex envelope')
    if signed_double_area(vertices)<=0:
        raise ValueError('Cell orientation is not positive')
    return vertices


def signed_double_area(vertices):
    a,b,c=vertices
    return (b[1]-a[1])*(c[2]-a[2])-(b[2]-a[2])*(c[1]-a[1])


def split_cell(vertices):
    """Four exact affine child triangles: no gaps, overlap, or native rounding."""
    vertices=literal_cell(vertices)
    a,b,c=vertices
    def mid(left,right):
        return tuple((x+y)/2 for x,y in zip(left,right))
    ab,bc,ca=mid(a,b),mid(b,c),mid(c,a)
    children=((a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca))
    area=signed_double_area(vertices)
    if any(signed_double_area(child)!=area/4 for child in children):
        raise ValueError('Exact midpoint subdivision lost coverage')
    return children


def radial_support_classification(certificates):
    """Signs of the convex sphere quadratic certify all lines in a cell."""
    def bounds(name):
        c=certificates[name]
        return tuple(Fraction(*map(int,c[k])) for k in ('lower','upper'))
    A=bounds('A'); delta=bounds('Delta')
    lower,upper=bounds('sphere_cohort_lower'),bounds('sphere_cohort_upper')
    dl,du=bounds('sphere_derivative_half_lower'),bounds('sphere_derivative_half_upper')
    if A[0]<=0:
        return 'OPEN'
    if delta[1]<0 or (lower[0]>0 and dl[0]>=0) or (upper[0]>0 and du[1]<=0):
        return 'EMPTY_CERTIFIED_ON_SUPPLIED_CELL'
    if lower[1]<0 or upper[1]<0 or (delta[0]>0 and dl[1]<0 and du[0]>0):
        return 'POSITIVE_CERTIFIED_ON_SUPPLIED_CELL'
    return 'OPEN'


def partition_events(raw, cells, *, max_depth=2, max_cells=256):
    """Refine event crossings; retain every unresolved cell at the work cap.

    Event signs alone do not identify the active sphere root or Gamma bank.
    Identically zero events stay semantically unresolved and are never discarded.
    No angular quadrature is performed and no source support is removed.
    """
    if type(max_depth) is not int or not 0<=max_depth<=12:
        raise ValueError('Event depth must be between 0 and 12')
    if type(max_cells) is not int or not 1<=max_cells<=4096:
        raise ValueError('Event cell cap must be between 1 and 4096')
    roots=tuple(literal_cell(v) for v in cells)
    if not roots or len(roots)>max_cells:
        raise ValueError('Cell cap cannot retain the initial support')
    events,source=source_events(raw)
    pending=[(i,0,vertices) for i,vertices in reversed(tuple(enumerate(roots)))]
    leaves=[]
    splits=0
    while pending:
        root,depth,vertices=pending.pop()
        certificates={name:sign_certificate(polynomial,vertices) for name,polynomial in events.items()}
        crossings=[name for name,c in certificates.items()
                   if c['classification']=='ZERO_POSSIBLE_REQUIRES_SPLIT_OR_RETAIN_OPEN']
        semantic=[name for name,c in certificates.items()
                  if c['classification']=='IDENTICALLY_ZERO_REQUIRES_SEMANTIC_CLASSIFICATION']
        if crossings and depth<max_depth and len(leaves)+len(pending)+4<=max_cells:
            children=split_cell(vertices)
            pending.extend((root,depth+1,child) for child in reversed(children))
            splits+=1
            continue
        leaves.append(dict(root=root,depth=depth,vertices=[[ratio(v) for v in row] for row in vertices],
            signed_double_area=ratio(signed_double_area(vertices)),
            radial_support_classification=radial_support_classification(certificates),
            event_free=not crossings and not semantic,crossing_events=crossings,
            identically_zero_events=semantic,
            sign_certificates={name:dict(lower=c['lower'],upper=c['upper'],classification=c['classification'],
                                       certificate_sha256=c['certificate_sha256']) for name,c in certificates.items()}))
    for root,vertices in enumerate(roots):
        retained=sum((Fraction(*map(int,leaf['signed_double_area'])) for leaf in leaves if leaf['root']==root),Fraction())
        if retained!=signed_double_area(vertices):
            raise ValueError('Original root area was not preserved')
    result=dict(format='genesis-source-event-partition-1',source_literal=source,
        source_events={name:p.metadata() for name,p in events.items()},
        root_vertices=[[[ratio(v) for v in row] for row in vertices] for vertices in roots],
        max_depth=max_depth,max_cells=max_cells,split_count=splits,leaf_count=len(leaves),leaves=leaves,
        event_free_leaf_count=sum(leaf['event_free'] for leaf in leaves),
        unresolved_leaf_count=sum(not leaf['event_free'] for leaf in leaves),
        exact_partition_of_supplied_cells=True,coverage_proof='Repeated exact affine midpoint subdivision with four area-quarter children',
        discarded_cell_count=0,source_physical_support_coverage_certified=False,
        native_node_mapping_accuracy_certified=False,angular_accuracy_certified=False,
        physical_creation_admitted=False,physical_release_admitted=False,
        automated_goal_enabled=False,history_advanced=False,history_reinitialized=False)
    result['sha256']=digest(result)
    return result


def verify_partition(raw,cells,certificate):
    if partition_events(raw,cells,max_depth=certificate['max_depth'],max_cells=certificate['max_cells'])!=certificate:
        raise ValueError('Canonical event partition differs')
    return True
