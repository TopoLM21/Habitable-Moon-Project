"""Certified continuous crossing events for a moving native contact arc.

Both owners follow their complete constant Euler vectors for the declared
interval. Exact Taylor coefficients and a norm remainder prove signs over
the whole interval. No sampled sign, projected coordinate or minimum strip
width authorizes a topology change. This bounded event certificate describes
an initial exit and later reentry band; it does not adopt material or memory.
"""
from dataclasses import dataclass
from fractions import Fraction
from math import comb, factorial
import hashlib
import json

from .geometric_intrinsic_geometry import RotationIncrement
from .geometric_native_polygons import (
    NativePlane, NativeRay, _cross, _dot, _incident, _plane_values, _ray_values,
    _side, native_arc_to_dict, native_plane_to_dict, native_plane_from_dict,
)
from .geometric_native_pose import (
    FractionInterval as I, _rounded_interval, certify_rotation_history, interval_sqrt,
)
from .geometric_native_spatial_underthrust import _arc_from_dict, _validate_arc

FORMAT = "native-continuous-contact-arc-events-1"
MODEL = "complete_two_branch_native_arc_entry_exit_schedule"


class UnresolvedNativeContactEvents(ValueError):
    """The complete event schedule was not proved in the declared domain."""


def _encoded(raw):
    return json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _maximum(interval):
    return max(abs(interval.lo), abs(interval.hi))


def _round(value, precision):
    return _rounded_interval(value, precision+16)


def _normalize(vector, precision):
    norm = interval_sqrt(_dot(vector,vector),precision+16)
    if norm.lo <= 0:
        raise UnresolvedNativeContactEvents("Native event direction is not certified nonzero")
    return tuple(_round(value/norm,precision) for value in vector)


@dataclass(frozen=True, slots=True)
class _Series:
    coefficients: tuple
    scale: Fraction
    rate_bound: Fraction
    precision: int

    def derivative(self, degree):
        return tuple(_round(value*Fraction(factorial(k),factorial(k-degree)),self.precision)
                     for k,value in enumerate(self.coefficients) if k >= degree)

    def remainder(self, degree=0):
        # F(u)=g(u)/u has coefficient bound M L^(k+1)/(k+1)!.
        # The ratio of successive derivative-tail terms is <= L/(N+2-r).
        n = len(self.coefficients)-1
        first = self.scale*self.rate_bound**(n+2)/factorial(n+2)
        first *= Fraction(factorial(n+1),factorial(n+1-degree))
        ratio = self.rate_bound/Fraction(n+2-degree)
        if ratio >= 1:
            raise UnresolvedNativeContactEvents("Event Taylor remainder exceeds its norm domain")
        return first/(1-ratio)

    def bounds(self, domain, degree=0):
        return _polynomial_bounds(self.derivative(degree),domain,
                                  self.remainder(degree),self.precision)


def _polynomial_bounds(coefficients, domain, remainder, precision):
    """Bernstein convex-hull enclosure on an exact rational time interval."""
    n = len(coefficients)-1
    low, width = domain.lo,domain.width
    if not width:
        value=I(0)
        for coefficient in reversed(coefficients):
            value=_round(value*low+coefficient,precision)
        return _round(value+I(-remainder,remainder),precision)
    power = []
    for j in range(n+1):
        value = sum((coefficients[k]*comb(k,j)*low**(k-j)*width**j
                     for k in range(j,n+1)),I(0))
        power.append(_round(value,precision))
    bernstein = []
    for i in range(n+1):
        value = sum((power[k]*Fraction(comb(i,k),comb(n,k))
                     for k in range(i+1)),I(0))
        bernstein.append(_round(value,precision))
    return _round(I(min(v.lo for v in bernstein)-remainder,
                    max(v.hi for v in bernstein)+remainder),precision)


def _zero(plane,ray):
    return _incident(ray,plane) or _side(ray,plane)==0


def _series(plane,ray,donor,receiver,duration,order,precision):
    if not _zero(plane,ray):
        raise UnresolvedNativeContactEvents("Contact-arc certificate needs authenticated initial incidence")
    n = _normalize(_plane_values(plane,precision)[1],precision)
    p = _normalize(_ray_values(ray,precision)[1],precision)
    axis_n = tuple(Fraction.from_float(v)*duration for v in donor)
    axis_p = tuple(Fraction.from_float(v)*duration for v in receiver)
    rate = sum(map(abs,axis_n),Fraction(0))+sum(map(abs,axis_p),Fraction(0))
    scale = sum(map(_maximum,n),Fraction(0))*sum(map(_maximum,p),Fraction(0))
    normals,points = [n],[p]
    for k in range(1,order+2):
        normals.append(tuple(_round(v/k,precision) for v in _cross(axis_n,normals[-1])))
        points.append(tuple(_round(v/k,precision) for v in _cross(axis_p,points[-1])))
    # g(0)=0 is inherited native incidence, not an approximate cancellation.
    coefficients = tuple(_round(sum((_dot(normals[i],points[k-i]) for i in range(k+1)),I(0)),precision)
                         for k in range(1,order+2))
    return _Series(coefficients,scale,rate,precision)


def _turning(a,b,u):
    return a.bounds(u)*b.bounds(u,1)-a.bounds(u,1)*b.bounds(u)


def _isolate(function,low,high,*,increasing,relative_digits=24):
    """Bracket a unique crossing already proved by a continuous sign law."""
    first,last = function(I(low)),function(I(high))
    before = lambda v: v.hi < 0 if increasing else v.lo > 0
    after = lambda v: v.lo > 0 if increasing else v.hi < 0
    if not before(first) or not after(last):
        raise UnresolvedNativeContactEvents("Event root lacks strict oriented endpoint signs")
    for _ in range(768):
        if low > 0 and high-low <= high/Fraction(10**relative_digits):
            return I(low,high)
        middle = (low+high)/2
        value = function(I(middle))
        if before(value):
            low = middle
        elif after(value):
            high = middle
        else:
            # The bracket remains valid. A smaller unproved midpoint must
            # never be promoted to an exact zero or dropped event.
            left,right = (low+middle)/2,(middle+high)/2
            if before(function(I(left))) and after(function(I(right))):
                low,high=left,right
            else:
                raise UnresolvedNativeContactEvents("Event time needs a sharper analytic enclosure")
    raise UnresolvedNativeContactEvents("Event isolation exceeded its exact subdivision budget")


@dataclass(frozen=True, slots=True)
class NativeContactArcEventCertificate:
    donor_plane_before: NativePlane
    receiver_arc_before: object
    donor_omega: tuple
    receiver_omega: tuple
    dt_myr: float
    precision: int
    order: int
    a_over_u_bounds: I
    b_over_u_second_derivative_bounds: I
    turning_derivative_bounds: I
    turning_fraction: I
    turning_time_myr: I
    initial_ratio: I
    maximum_ratio: I
    recrossing_ratio_width: I
    topology: str = "initial_exit_later_entry_positive_material_band"


def certify_native_contact_arc_events(donor_plane_before,receiver_arc_before,
        donor_omega,receiver_omega,dt_myr,*,precision=128,order=20):
    """Prove the entire actual arc's two-branch entry/exit family.

    For normalized endpoint trajectories a,b and positive arc coefficients
    alpha,beta, g=u*(alpha*A+beta*B), A=a/u, B=b/u. A>0; B is strictly
    convex, starts negative and ends at zero, so B<0 inside. V=A B'-A' B
    increases strictly from negative to positive and has one root; its
    derivative is V'=A*B''-A''*B. Thus
    q=-B/A increases then decreases to zero. Its positive maximum above q(0)
    proves the complete nonempty exit/reentry band without spatial sampling.
    The start endpoint (beta=0, infinite ratio) stays inside by A>0.
    """
    if type(donor_plane_before) is not NativePlane:
        raise ValueError("Events require an explicit native donor plane")
    arc = _validate_arc(receiver_arc_before)
    if type(precision) is not int or not 32 <= precision <= 1024:
        raise ValueError("Event precision must be an explicit bounded integer")
    if type(order) is not int or not 4 <= order <= 64:
        raise ValueError("Event Taylor order must be an explicit bounded integer")
    if not isinstance(donor_omega,tuple) or not isinstance(receiver_omega,tuple):
        raise ValueError("Event owners require immutable complete Euler triples")
    dstep,rstep=RotationIncrement(donor_omega,dt_myr),RotationIncrement(receiver_omega,dt_myr)
    certify_rotation_history((dstep,),precision)
    certify_rotation_history((rstep,),precision)
    donor,receiver=dstep.omega_rad_per_myr,rstep.omega_rad_per_myr
    duration=Fraction.from_float(dstep.dt_myr)
    if not _zero(donor_plane_before.rotated((dstep,)),arc.end.rotated((rstep,))):
        raise UnresolvedNativeContactEvents("Terminal contact incidence is not inherited from actual native planes")
    a=_series(donor_plane_before,arc.start,donor,receiver,duration,order,precision)
    b=_series(donor_plane_before,arc.end,donor,receiver,duration,order,precision)
    domain=I(0,1)
    abound,bbound=a.bounds(domain),b.bounds(domain)
    bsecond=b.bounds(domain,2)
    vprime=abound*bsecond-a.bounds(domain,2)*bbound
    if (abound.lo <= 0 or b.coefficients[0].hi >= 0 or bsecond.lo <= 0
            or vprime.lo <= 0):
        raise UnresolvedNativeContactEvents("Complete native arc event topology is not certified in this bounded domain")
    turning=_isolate(lambda u:_turning(a,b,u),Fraction(0),Fraction(1),increasing=True)
    initial=_round(-b.coefficients[0]/a.coefficients[0],precision)
    maximum=_round(-b.bounds(turning)/a.bounds(turning),precision)
    width=_round(maximum-initial,precision)
    if initial.lo <= 0 or width.lo <= 0:
        raise UnresolvedNativeContactEvents("Positive recrossing band was not proved; no strip may be discarded")
    return NativeContactArcEventCertificate(donor_plane_before,arc,donor,receiver,
        dstep.dt_myr,precision,order,abound,bsecond,_round(vprime,precision),turning,
        turning*duration,initial,maximum,width)


@dataclass(frozen=True, slots=True)
class NativeContactRatioEvent:
    kind: str
    time_fraction: I
    time_myr: I


def isolate_native_contact_ratio_events(certificate,ratio):
    """Return every crossing for one exact positive homogeneous arc ratio."""
    if type(certificate) is not NativeContactArcEventCertificate or type(ratio) is not Fraction or ratio<0:
        raise ValueError("Arc event query requires a certified family and an exact nonnegative ratio")
    # Do not trust a forged frozen object. Recompute its complete native proof.
    validate_native_contact_arc_events(certificate)
    c=certificate
    duration=Fraction.from_float(c.dt_myr)
    event=lambda kind,u:NativeContactRatioEvent(kind,u,u*duration)
    if not ratio:
        return (event("exit",I(0)),event("entry",I(1)))
    a=_series(c.donor_plane_before,c.receiver_arc_before.start,c.donor_omega,
        c.receiver_omega,duration,c.order,c.precision)
    b=_series(c.donor_plane_before,c.receiver_arc_before.end,c.donor_omega,
        c.receiver_omega,duration,c.order,c.precision)
    score=lambda u:ratio*a.bounds(u)+b.bounds(u)
    if ratio > c.maximum_ratio.hi:
        return (event("entry",I(0)),)
    if ratio < c.initial_ratio.lo:
        root=_isolate(score,c.turning_fraction.hi,Fraction(1),increasing=True)
        return (event("exit",I(0)),event("entry",root))
    if c.initial_ratio.hi < ratio < c.maximum_ratio.lo:
        first=_isolate(score,Fraction(0),c.turning_fraction.lo,increasing=False)
        second=_isolate(score,c.turning_fraction.hi,Fraction(1),increasing=True)
        return (event("entry",I(0)),event("exit",first),event("entry",second))
    raise UnresolvedNativeContactEvents("Material ratio overlaps a certified topology threshold enclosure")


def _rational(value):
    return [value.numerator,value.denominator]


def _interval(value):
    return dict(lo=_rational(value.lo),hi=_rational(value.hi))


_BOUNDS=("a_over_u_bounds","b_over_u_second_derivative_bounds","turning_derivative_bounds",
    "turning_fraction","turning_time_myr","initial_ratio","maximum_ratio","recrossing_ratio_width")


def _body(c):
    return dict(format=FORMAT,model=MODEL,trajectory=dict(
        donor_plane_before=native_plane_to_dict(c.donor_plane_before),
        receiver_arc_before=native_arc_to_dict(c.receiver_arc_before),
        donor_omega=list(c.donor_omega),receiver_omega=list(c.receiver_omega),dt_myr=c.dt_myr),
        precision=c.precision,order=c.order,topology=c.topology,
        certificate={key:_interval(getattr(c,key)) for key in _BOUNDS},
        admission_authorized=False,
        event_classes=["entry_at_start_and_remains_inside", "one_later_entry",
                       "entry_at_start_exit_and_reentry", "endpoint_exit_at_start_entry_at_end"])


def validate_native_contact_arc_events(certificate):
    if type(certificate) is not NativeContactArcEventCertificate:
        raise ValueError("Expected an explicit native contact arc event certificate")
    c=certificate
    expected=certify_native_contact_arc_events(c.donor_plane_before,c.receiver_arc_before,
        c.donor_omega,c.receiver_omega,c.dt_myr,precision=c.precision,order=c.order)
    if _encoded(_body(c)) != _encoded(_body(expected)):
        raise ValueError("Native contact event certificate differs from complete analytic replay")
    return expected


def native_contact_arc_events_to_dict(certificate):
    raw=_body(validate_native_contact_arc_events(certificate))
    raw["certificate_sha256"]=hashlib.sha256(_encoded(raw)).hexdigest()
    return raw


def native_contact_arc_events_from_dict(raw):
    keys={"format","model","trajectory","precision","order","topology","certificate",
          "admission_authorized","event_classes","certificate_sha256"}
    if not isinstance(raw,dict) or set(raw)!=keys or raw["format"]!=FORMAT or raw["model"]!=MODEL:
        raise ValueError("Expected strict native contact event schema")
    if not isinstance(raw["trajectory"],dict) or set(raw["trajectory"])!={
            "donor_plane_before","receiver_arc_before","donor_omega","receiver_omega","dt_myr"}:
        raise ValueError("Native contact event trajectory schema mismatch")
    trajectory=raw["trajectory"]
    for key in ("donor_omega","receiver_omega"):
        if not isinstance(trajectory[key],list):
            raise ValueError("Native event trajectory requires explicit owner vector lists")
    expected=certify_native_contact_arc_events(native_plane_from_dict(trajectory["donor_plane_before"]),
        _arc_from_dict(trajectory["receiver_arc_before"]),tuple(trajectory["donor_omega"]),
        tuple(trajectory["receiver_omega"]),trajectory["dt_myr"],
        precision=raw["precision"],order=raw["order"])
    if _encoded(raw)!=_encoded(native_contact_arc_events_to_dict(expected)):
        raise ValueError("Saved native contact event proof differs from complete typed analytic replay")
    return expected
