"""Certified spherical cuts retaining exact reference planes and full poses.

Vertices are original rays or intersections of two inherited reference planes.
Rational proper rotations are scratch values only: every nonzero predicate
needs an enclosure proving its sign for the recorded analytic SO(3) history.
Unproved incidence is refused, rather than snapped or discarded by area.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from functools import lru_cache
import math
from numbers import Integral, Real

from .geometric_intrinsic_geometry import (IntrinsicGeometry, RotationIncrement,
    rotation_increment_to_dict, rotation_increment_from_dict)
from .geometric_native_pose import (FractionInterval as I, certify_rotation_history,
    interval_sqrt, interval_atan, interval_pi)

VERSION = "geometric-native-polygons-1"
PRECISIONS = (64, 128, 256, 512, 1024)


class UnresolvedNativeGeometry(ValueError):
    """The analytic geometry was not certified within the resource budget."""


def _primitive(values):
    values = tuple(values)
    if len(values) != 3:
        raise ValueError("Native reference needs three finite exact ray components")
    rays = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError("Native ray components cannot be booleans")
        if isinstance(value, Fraction):
            rays.append(value)
        elif isinstance(value, Integral):
            rays.append(Fraction(int(value)))
        elif isinstance(value, Real) and math.isfinite(float(value)):
            rays.append(Fraction.from_float(float(value)))
        else:
            raise ValueError("Native ray components must be finite explicit numbers")
    denominator = math.lcm(*(value.denominator for value in rays))
    integers = tuple(value.numerator*(denominator//value.denominator) for value in rays)
    divisor = math.gcd(*integers)
    if not divisor:
        raise ValueError("Native rays and planes cannot be zero")
    return tuple(value//divisor for value in integers)


def _history(values):
    history = tuple(values)
    if any(type(step) is not RotationIncrement for step in history):
        raise ValueError("Native geometry requires full ordered rotation increments")
    return tuple(RotationIncrement(step.omega_rad_per_myr, step.dt_myr) for step in history)


def _dot(a, b):
    return sum((x*y for x, y in zip(a, b)), 0)


def _cross(a, b):
    return tuple(a[j]*b[k]-a[k]*b[j] for j, k in ((1,2),(2,0),(0,1)))


def _map(matrix, vector):
    return tuple(_dot(row, vector) for row in matrix)


@dataclass(frozen=True, slots=True)
class NativePlane:
    normal: tuple[int, int, int]
    rotations: tuple[RotationIncrement, ...] = ()
    label: str = ""

    def __post_init__(self):
        object.__setattr__(self, "normal", _primitive(self.normal))
        object.__setattr__(self, "rotations", _history(self.rotations))
        if not isinstance(self.label, str):
            raise ValueError("Native plane lineage label must be a string")

    def reversed(self):
        return NativePlane(tuple(-value for value in self.normal), self.rotations, self.label)

    def rotated(self, history):
        return NativePlane(self.normal, self.rotations+_history(history), self.label)


def _effective(history):
    # Reduce only exact rotation identities. The authoritative DTO retains
    # every recorded increment; this word is a geometric proof/evaluation aid.
    result = []
    for step in history:
        if not any(step.omega_rad_per_myr):
            continue
        if (result and step.dt_myr == result[-1].dt_myr
                and step.omega_rad_per_myr == tuple(-x for x in result[-1].omega_rad_per_myr)):
            result.pop()
        else:
            result.append(step)
    return tuple(result)


def _pose_certificate(history, precision):
    # Check the domain of the full journal before reducing analytic inverses.
    certify_rotation_history(history,PRECISIONS[0])
    return certify_rotation_history(_effective(history),precision)


def _axis_fixed(reference, history):
    """Prove invariance using every component of every recorded axis."""
    return all(not any(_cross(reference, tuple(Fraction.from_float(x)
                    for x in step.omega_rad_per_myr))) for step in history)


def _vector_history(reference, history):
    return () if _axis_fixed(reference, history) else _effective(history)


def _same_plane(a, b):
    return (_vector_history(a.normal,a.rotations) == _vector_history(b.normal,b.rotations)
            and (a.normal == b.normal or a.normal == tuple(-x for x in b.normal)))


@dataclass(frozen=True, slots=True)
class NativeRay:
    kind: str
    reference: tuple[int, int, int] = ()
    rotations: tuple[RotationIncrement, ...] = ()
    planes: tuple[NativePlane, ...] = ()
    sign: int = 1

    def __post_init__(self):
        if type(self.sign) is not int or self.sign not in (-1,1):
            raise ValueError("Native intersection ray needs an explicit orientation")
        object.__setattr__(self, "rotations", _history(self.rotations))
        object.__setattr__(self, "planes", tuple(self.planes))
        if self.kind == "seed":
            if self.planes or self.sign != 1:
                raise ValueError("Invalid native seed ray schema")
            object.__setattr__(self, "reference", _primitive(self.reference))
        elif self.kind == "cross":
            if (self.reference or self.rotations or len(self.planes) != 2
                    or any(type(plane) is not NativePlane for plane in self.planes)):
                raise ValueError("Native cut vertex requires two explicit inherited planes")
        else:
            raise ValueError("Unknown native ray expression")

    def rotated(self, history):
        history = _history(history)
        if self.kind == "seed":
            return NativeRay("seed", self.reference, self.rotations+history)
        return NativeRay("cross", planes=tuple(p.rotated(history) for p in self.planes), sign=self.sign)

    def world_ray(self):
        """Display projection only; predicates retain this ray's expression."""
        vector,_ = _ray_values(self,PRECISIONS[1])
        scale = max(map(abs,vector))
        values = tuple(float(x/scale) for x in vector)
        norm = math.hypot(*values)
        return tuple(x/norm for x in values)


def _incident(ray, plane):
    if ray.kind == "cross":
        return any(_same_plane(plane, own) for own in ray.planes)
    if _dot(ray.reference, plane.normal) != 0:
        return False
    if _effective(ray.rotations) == _effective(plane.rotations):
        return True
    if (_vector_history(ray.reference,ray.rotations)
            == _vector_history(plane.normal,plane.rotations)):
        return True
    # A ray can move within a fixed plane without itself remaining fixed.
    return (_axis_fixed(plane.normal,plane.rotations)
            and _axis_fixed(plane.normal,ray.rotations))


@lru_cache(maxsize=8192)
def _plane_values(plane, precision):
    certificate = _pose_certificate(plane.rotations, precision)
    vector = tuple(Fraction(x) for x in plane.normal)
    # A homogeneous positive scale avoids enormous initial dyadic integers.
    scale = max(map(abs, vector))
    vector = tuple(x/scale for x in vector)
    return _map(certificate.matrix, vector), _map(certificate.intervals, vector)


@lru_cache(maxsize=16384)
def _ray_values(ray, precision):
    if ray.kind == "seed":
        certificate = _pose_certificate(ray.rotations, precision)
        scale = max(map(abs, ray.reference))
        vector = tuple(Fraction(x, scale) for x in ray.reference)
        return _map(certificate.matrix, vector), _map(certificate.intervals, vector)
    a, ia = _plane_values(ray.planes[0], precision)
    b, ib = _plane_values(ray.planes[1], precision)
    center = tuple(ray.sign*x for x in _cross(a,b))
    bounds = tuple(ray.sign*x for x in _cross(ia,ib))
    scale = max(map(abs, center))
    if not scale:
        raise UnresolvedNativeGeometry("Native support planes do not certify a nonzero intersection ray")
    return tuple(x/scale for x in center), tuple(x/scale for x in bounds)


def _side(ray, plane):
    if _incident(ray,plane):
        return 0
    for precision in PRECISIONS:
        _, point = _ray_values(ray,precision)
        _, normal = _plane_values(plane,precision)
        value = _dot(point,normal)
        if value.lo > 0:
            return 1
        if value.hi < 0:
            return -1
        if value.lo == value.hi == 0:
            return 0
    raise UnresolvedNativeGeometry("Unproved native plane incidence; no snapping or area cutoff")


@dataclass(frozen=True, slots=True)
class NativePolygon:
    vertices: tuple[NativeRay, ...]
    supports: tuple[NativePlane, ...]
    pose: tuple[RotationIncrement, ...] = ()
    version: str = VERSION

    def __post_init__(self):
        object.__setattr__(self,"vertices",tuple(self.vertices))
        object.__setattr__(self,"supports",tuple(self.supports))
        object.__setattr__(self,"pose",_history(self.pose))
        if (self.version != VERSION or len(self.vertices) < 3
                or len(self.vertices) != len(self.supports)
                or any(type(p) is not NativeRay for p in self.vertices)
                or any(type(p) is not NativePlane for p in self.supports)):
            raise ValueError("Native polygon needs an explicit positive cyclic footprint and inherited supports")

    @classmethod
    def from_intrinsic(cls, geometry):
        if type(geometry) is not IntrinsicGeometry:
            raise ValueError("Native import requires explicit intrinsic body geometry")
        geometry = IntrinsicGeometry(geometry.body_rays, geometry.rotations, geometry.version)
        rays = tuple(NativeRay("seed", _primitive(row)) for row in geometry.body_rays)
        supports = tuple(NativePlane(_cross(a.reference,b.reference))
                         for a,b in zip(rays,rays[1:]+rays[:1]))
        result = cls(rays,supports,geometry.rotations)
        return validate_native_polygon(result)

    @property
    def support_planes(self):
        return tuple(plane.rotated(self.pose) for plane in self.supports)

    def rotated(self, omega_rad_per_myr, dt_myr):
        checked = validate_native_polygon(self)
        step = RotationIncrement(omega_rad_per_myr,dt_myr)
        certify_rotation_history((step,),PRECISIONS[0])
        return replace(checked,pose=checked.pose+(step,))

    def area_bounds(self):
        return _certified_area(validate_native_polygon(self))

    @property
    def area_sr(self):
        bounds = self.area_bounds()
        value = float(bounds.midpoint)
        if not math.isfinite(value) or value <= 0.:
            raise UnresolvedNativeGeometry("Positive native area is not representable in the material ledger")
        return value

    def world_polygon(self):
        result = []
        for ray in self.vertices:
            vector,_ = _ray_values(ray.rotated(self.pose),PRECISIONS[1])
            scale = max(map(abs,vector))
            floating = tuple(float(x/scale) for x in vector)
            norm = math.hypot(*floating)
            result.append(tuple(x/norm for x in floating))
        return tuple(result)


def _world(poly):
    if not poly.pose:
        return poly
    return NativePolygon(tuple(ray.rotated(poly.pose) for ray in poly.vertices),
                         poly.support_planes)


def _known_collinear(rays, supports):
    return any(all(_incident(ray,plane) for ray in rays) for plane in supports)


def _atan2_positive(y,x,precision):
    if y.lo <= 0:
        raise UnresolvedNativeGeometry("Native triangle determinant sign is not certified")
    if x.lo > 0:
        return interval_atan(y/x,precision)
    if x.hi < 0:
        return interval_pi(precision)-interval_atan(y/(-x),precision)
    # atan2(y,x) = pi/2 - atan(x/y), monotone in the signed ratio.
    ratio = x/y
    low = (-interval_atan(-ratio.lo,precision) if ratio.lo < 0 else interval_atan(ratio.lo,precision))
    high = (-interval_atan(-ratio.hi,precision) if ratio.hi < 0 else interval_atan(ratio.hi,precision))
    return interval_pi(precision)/2-I(low.lo,high.hi)


@lru_cache(maxsize=4096)
def _certified_area(poly):
    # Uniform post-cut pose is an exact isometry, not an area remeasurement.
    rays = poly.vertices
    for precision in PRECISIONS:
        total = I(0)
        try:
            for index in range(1,len(rays)-1):
                triangle = (rays[0],rays[index],rays[index+1])
                if _known_collinear(triangle,poly.supports):
                    continue
                values = tuple(_ray_values(ray,precision)[1] for ray in triangle)
                a,b,c = values
                numerator = _dot(a,_cross(b,c))
                if numerator.lo == numerator.hi == 0:
                    continue
                norms = tuple(interval_sqrt(_dot(v,v),precision+8) for v in values)
                na,nb,nc = norms
                denominator = na*nb*nc+_dot(a,b)*nc+_dot(b,c)*na+_dot(c,a)*nb
                total += 2*_atan2_positive(numerator,denominator,precision+8)
            if total.lo > 0 and total.width <= total.lo/Fraction(10**13):
                return total
        except (UnresolvedNativeGeometry,ValueError):
            continue
    raise UnresolvedNativeGeometry("Native positive area could not be certified within the precision budget")


def validate_native_polygon(poly):
    if type(poly) is not NativePolygon:
        raise ValueError("Expected an explicit native polygon")
    return _validate_native_polygon(poly)


@lru_cache(maxsize=4096)
def _validate_native_polygon(poly):
    supports = tuple(NativePlane(p.normal,p.rotations,p.label) for p in poly.supports)
    vertices = tuple(_validated_ray(ray) for ray in poly.vertices)
    checked = NativePolygon(vertices,supports,poly.pose,poly.version)
    # Validate the full post-cut pose domain without using its projection.
    certify_rotation_history(checked.pose,PRECISIONS[0])
    for index,plane in enumerate(supports):
        a,b = vertices[index],vertices[(index+1)%len(vertices)]
        # A declared support can also be incident as an exact algebraic
        # consequence of the parent planes. Only a proved zero is accepted;
        # a predicate whose enclosure straddles zero still refuses.
        if ((not _incident(a,plane) and _side(a,plane) != 0)
                or (not _incident(b,plane) and _side(b,plane) != 0)):
            raise ValueError("Native edge endpoints lack their inherited support lineage")
        if any(_side(ray,plane) < 0 for ray in vertices):
            raise ValueError("Native polygon is not exactly convex in its inherited halfspaces")
        orientation = None
        for precision in PRECISIONS:
            _,av = _ray_values(a,precision)
            _,bv = _ray_values(b,precision)
            _,normal = _plane_values(plane,precision)
            proof = _dot(_cross(av,bv),normal)
            if proof.lo > 0:
                orientation = True
                break
            if proof.hi < 0:
                raise ValueError("Native edge orientation opposes its inward support")
        if orientation is None:
            raise UnresolvedNativeGeometry("Native edge does not certify a positive minor arc")
    _certified_area(checked)
    return checked


def _validated_ray(ray):
    if type(ray) is not NativeRay:
        raise ValueError("Native polygon needs typed vertex lineage")
    planes = tuple(NativePlane(p.normal,p.rotations,p.label) for p in ray.planes)
    return NativeRay(ray.kind,ray.reference,ray.rotations,planes,ray.sign)


def _crossing(a,b,support,clip):
    for precision in PRECISIONS:
        na,ia = _plane_values(support,precision)
        nb,ib = _plane_values(clip,precision)
        center = _cross(na,nb)
        ac,ai = _ray_values(a,precision)
        bc,bi = _ray_values(b,precision)
        sign = 1 if _dot(center,tuple(x+y for x,y in zip(ac,bc))) > 0 else -1
        proof = _dot(tuple(sign*x for x in _cross(ia,ib)),tuple(x+y for x,y in zip(ai,bi)))
        if proof.lo > 0:
            return NativeRay("cross",planes=(support,clip),sign=sign)
    raise UnresolvedNativeGeometry("Native plane crossing does not certify the minor-edge branch")


def native_clip(poly,plane):
    poly = validate_native_polygon(poly)
    if type(plane) is not NativePlane:
        raise ValueError("Native clipping requires an explicit typed plane")
    return _clip_checked(poly,plane)


def _clip_checked(poly,plane):
    poly = _world(poly)
    plane = NativePlane(plane.normal,plane.rotations,plane.label)
    sides = tuple(_side(ray,plane) for ray in poly.vertices)
    if min(sides) >= 0:
        return poly
    if max(sides) <= 0:
        return None
    vertices,incoming = [],[]

    def append(ray,support):
        if vertices and _ray_same(ray,vertices[-1]):
            return
        vertices.append(ray)
        incoming.append(support)

    for index,(a,b,support) in enumerate(zip(poly.vertices,
            poly.vertices[1:]+poly.vertices[:1],poly.supports)):
        da,db = sides[index],sides[(index+1)%len(sides)]
        if da >= 0 and db >= 0:
            append(b,support)
        elif da >= 0:
            append(a if da == 0 else _crossing(a,b,support,plane),support)
        elif db >= 0:
            append(b if db == 0 else _crossing(a,b,support,plane),plane)
            if db:
                append(b,support)
    if len(vertices) > 1 and _ray_same(vertices[0],vertices[-1]):
        vertices.pop()
        incoming[0] = incoming.pop()
    if len(vertices) < 3:
        return None
    supports = tuple(incoming[(i+1)%len(incoming)] for i in range(len(incoming)))
    result = NativePolygon(tuple(vertices),supports)
    return validate_native_polygon(result)


def native_intersection(a,b):
    result = validate_native_polygon(a)
    b = validate_native_polygon(b)
    for plane in b.support_planes:
        result = _clip_checked(result,plane)
        if result is None:
            return None
    return result


def native_subtract(a,b):
    remainder,parts = validate_native_polygon(a),[]
    b = validate_native_polygon(b)
    for plane in b.support_planes:
        outside = _clip_checked(remainder,plane.reversed())
        if outside is not None:
            parts.append(outside)
        remainder = _clip_checked(remainder,plane)
        if remainder is None:
            break
    return tuple(parts)


@dataclass(frozen=True, slots=True)
class NativeArc:
    start: NativeRay
    end: NativeRay
    plane: NativePlane
    length_radians: float


def _ray_same(a,b):
    return _ray_key(a) == _ray_key(b)


def _history_key(history):
    return tuple((step.omega_rad_per_myr,step.dt_myr) for step in _effective(history))


def _shared_plane_history(first,second):
    h1,h2 = _effective(first.rotations),_effective(second.rotations)
    if h1 == h2:
        return h1
    if (_axis_fixed(first.normal,h1) and _axis_fixed(first.normal,h2)):
        return h2
    if (_axis_fixed(second.normal,h1) and _axis_fixed(second.normal,h2)):
        return h1
    return None


def _ray_key(ray):
    if ray.kind == "seed":
        return ("seed",ray.reference,_history_key(_vector_history(ray.reference,ray.rotations)))
    first,second = ray.planes
    common = _shared_plane_history(first,second)
    if common is not None:
        # Cross(R n1,R n2) = R cross(n1,n2), including the positive
        # homogeneous scales.  This is an exact identity, not proximity.
        reference = tuple(ray.sign*x for x in _cross(first.normal,second.normal))
        return ("seed",_primitive(reference),_history_key(_vector_history(reference,common)))
    keys,direction = [],ray.sign
    for plane in ray.planes:
        polarity = next(1 if x > 0 else -1 for x in plane.normal if x)
        keys.append((tuple(polarity*x for x in plane.normal),
                     _history_key(_vector_history(plane.normal,plane.rotations))))
        direction *= polarity
    if keys[0] > keys[1]:
        keys.reverse()
        direction *= -1
    return ("cross",tuple(keys),direction)


def _arc_order(a,b,plane):
    ka,kb = _ray_key(a),_ray_key(b)
    if ka == kb:
        return 0
    # Opposite rays have an exactly zero determinant too. They remain
    # distinct endpoints: the other arc membership predicate excludes
    # a remote antipode, and endpoint deduplication uses oriented identity.
    if (ka[0] == kb[0] == "seed" and ka[2] == kb[2]
            and not any(_cross(ka[1],kb[1]))):
        return 0
    if ka[0] == kb[0] == "cross" and ka[1] == kb[1]:
        return 0
    for precision in PRECISIONS:
        _,av = _ray_values(a,precision)
        _,bv = _ray_values(b,precision)
        _,nv = _plane_values(plane,precision)
        determinant = _dot(_cross(av,bv),nv)
        if determinant.lo > 0:
            return 1
        if determinant.hi < 0:
            return -1
        if determinant.lo == determinant.hi == 0:
            return 0
    raise UnresolvedNativeGeometry("Unproved ordering of native shared-arc endpoints")


def _arc_length(a,b):
    for precision in PRECISIONS:
        _,av = _ray_values(a,precision)
        _,bv = _ray_values(b,precision)
        cross = _cross(av,bv)
        try:
            norm = interval_sqrt(_dot(cross,cross),precision+8)
            angle = _atan2_positive(norm,_dot(av,bv),precision+8)
            if angle.lo > 0 and angle.width < angle.lo/Fraction(10**13):
                value = float(angle.midpoint)
                if value > 0:
                    return value
        except (UnresolvedNativeGeometry,ValueError):
            continue
    raise UnresolvedNativeGeometry("Positive native contact length is not certified/representable")


def native_shared_boundary_arcs(a,b):
    a,b = _world(validate_native_polygon(a)),_world(validate_native_polygon(b))
    result = []
    for i,plane in enumerate(a.supports):
        for j,other in enumerate(b.supports):
            if (not _same_plane(plane,other) or
                    plane.normal != tuple(-x for x in other.normal)):
                continue
            start,end = a.vertices[i],a.vertices[(i+1)%len(a.vertices)]
            second_start,second_end = b.vertices[(j+1)%len(b.vertices)],b.vertices[j]
            # Test membership in both oriented minor arcs; signed pairwise
            # ordering alone is insufficient for disjoint arcs across pi.
            points = []
            for point in (start,end,second_start,second_end):
                inside = (_arc_order(start,point,plane) >= 0
                    and _arc_order(point,end,plane) >= 0
                    and _arc_order(second_start,point,plane) >= 0
                    and _arc_order(point,second_end,plane) >= 0)
                if inside and not any(_ray_same(point,own) for own in points):
                    points.append(point)
            if len(points) == 2:
                start,end = points
                if _arc_order(start,end,plane) < 0:
                    start,end = end,start
                result.append(NativeArc(start,end,plane,_arc_length(start,end)))
            elif len(points) > 2:
                raise UnresolvedNativeGeometry("Native contact has unproved endpoint identities")
    return tuple(result)


def native_plane_to_dict(plane):
    plane = NativePlane(plane.normal,plane.rotations,plane.label)
    return dict(normal=list(plane.normal),rotations=[rotation_increment_to_dict(s) for s in plane.rotations],label=plane.label)


def native_plane_from_dict(raw):
    if (not isinstance(raw,dict) or set(raw) != {"normal","rotations","label"}
            or not isinstance(raw["normal"],list) or not isinstance(raw["rotations"],list)
            or any(type(value) is not int for value in raw["normal"])):
        raise ValueError("Invalid native plane schema")
    return NativePlane(tuple(raw["normal"]),tuple(rotation_increment_from_dict(s) for s in raw["rotations"]),raw["label"])


def native_ray_to_dict(ray):
    ray = _validated_ray(ray)
    return dict(kind=ray.kind,reference=list(ray.reference),rotations=[rotation_increment_to_dict(s) for s in ray.rotations],
                planes=[native_plane_to_dict(p) for p in ray.planes],sign=ray.sign)


def native_ray_from_dict(raw):
    if (not isinstance(raw,dict) or set(raw) != {"kind","reference","rotations","planes","sign"}
            or any(not isinstance(raw[key],list) for key in ("reference","rotations","planes"))
            or any(type(value) is not int for value in raw["reference"])):
        raise ValueError("Invalid native ray schema")
    return NativeRay(raw["kind"],tuple(raw["reference"]),tuple(rotation_increment_from_dict(s) for s in raw["rotations"]),
                     tuple(native_plane_from_dict(p) for p in raw["planes"]),raw["sign"])


def native_polygon_to_dict(poly):
    poly = validate_native_polygon(poly)
    return dict(version=poly.version,vertices=[native_ray_to_dict(v) for v in poly.vertices],
                supports=[native_plane_to_dict(p) for p in poly.supports],pose=[rotation_increment_to_dict(s) for s in poly.pose])


def native_polygon_from_dict(raw):
    if (not isinstance(raw,dict) or set(raw) != {"version","vertices","supports","pose"}
            or any(not isinstance(raw[key],list) for key in ("vertices","supports","pose"))):
        raise ValueError("Invalid native polygon schema")
    return validate_native_polygon(NativePolygon(tuple(native_ray_from_dict(v) for v in raw["vertices"]),
        tuple(native_plane_from_dict(p) for p in raw["supports"]),tuple(rotation_increment_from_dict(s) for s in raw["pose"]),raw["version"]))


def native_arc_to_dict(arc):
    return dict(start=native_ray_to_dict(arc.start),end=native_ray_to_dict(arc.end),
                plane=native_plane_to_dict(arc.plane),length_radians=arc.length_radians)
