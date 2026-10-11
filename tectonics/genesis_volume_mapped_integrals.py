"""Radial source bounds at exact rational angular nodes.

No N is converted to floating point. Only the integration chart is represented;
source endpoint enclosures include that chart's rounding, as in the native lane.
The unchanged coefficient/contraction implementation evaluates the exact N.
"""
from dataclasses import dataclass
from fractions import Fraction as F
import numpy as np
from .genesis_volume_precision_integrals import PreparedPrecisionIntegrals
from .genesis_volume_rational_fields import fraction, rational
from .genesis_volume_operand_bounds import rational_enclosure
from .genesis_volume_integral_geometry_bounds import (
    SourceGeometryMoments, _moment_vector, _directed_vector, _digest, _ratio,
)
from analysis.genesis_volume_operand_validation.independent_geometry_bounds import _source_interval


@dataclass(frozen=True)
class RationalChart:
    source: object
    barycentric: tuple
    lambda_mid: object
    lambda_halfwidth: object


def represented(q):
    b=rational_enclosure(q)
    return b.lower/np.longdouble(2)+b.upper/np.longdouble(2)


class PreparedMappedIntegrals(PreparedPrecisionIntegrals):
    def __init__(self, source, raw, **options):
        super().__init__(source,raw,**options)
        self.normal=rational(np.asarray(raw['section']['normal_xyz'],float))
        self.offset=fraction(raw['section']['offset_m'])

    def geometry_at(self,N):
        n=tuple(map(fraction,N))
        if len(n)!=3 or min(n)<0 or sum(n)!=1:
            raise ValueError('Exact mapped N must belong to the source simplex')
        u=np.asarray(n,object)@self.fields.C
        d=np.asarray(n,object)@self.fields.D
        v=d-self.fields.center
        A,B,V=u@u,u@v,v@v
        if A<=0:
            raise ValueError('Collapsed exact source radial line')
        line=dict(uu=A,uv=B,vv=V,margin=self.fields.radius2-V+B*B/A,
            closest=-B/A,radius=self.source.geometry.seed_radius_m,
            bounds=(self.fields.lower,self.fields.upper))
        line['radius']=fraction(line['radius'])
        truth=_source_interval(line,256)
        slope,intercept=self.normal@u,self.normal@d-self.offset
        source_binding=self.base.geometry.source_binding_sha256
        if truth is None:
            chart=None; mid=None; half=F()
        else:
            mid=represented(sum(truth['mid'])/2)
            half=represented(sum(truth['half'])/2)
            if half<=0:
                raise ValueError('Represented rational chart collapsed')
            chart=RationalChart(self.source.geometry,n,mid,half)
            mid,half=fraction(mid),fraction(half)
        binding=_digest(dict(source=source_binding,N=list(map(_ratio,n)),
            midpoint=None if mid is None else _ratio(mid),halfwidth=_ratio(half)))
        if truth is None:
            exact=tuple((F(),F()) for _ in range(17)); zero=_directed_vector(exact)
            return chart,SourceGeometryMoments(source_binding,binding,n,None,F(),True,None,None,
                None,None,None,(exact,)*3,(exact,)*2,(zero,)*3,(zero,)*2,256)
        low=tuple((x-mid)/half for x in truth['low'])
        high=tuple((x-mid)/half for x in truth['high'])
        phi0,phi1=slope*mid+intercept,slope*half
        if not phi0 and not phi1:
            raise ValueError('Positive source line lies identically in Gamma')
        root=-phi0/phi1 if phi1 else None
        ordinary=_moment_vector(low,high,half,'ordinary',root,phi1,phi0)
        signed=_moment_vector(low,high,half,'signed',root,phi1,phi0)
        moments=(ordinary,signed,tuple((a/4,b/4) for a,b in ordinary))
        banks=tuple(_moment_vector(low,high,half,mode,root,phi1,phi0) for mode in ('negative','positive'))
        widths=tuple((a[0][0]/half,a[0][1]/half) for a in banks)
        return chart,SourceGeometryMoments(source_binding,binding,n,mid,half,False,
            (truth['low'],truth['high']),(low,high),(phi0,phi1),root,widths,
            moments,banks,tuple(map(_directed_vector,moments)),tuple(map(_directed_vector,banks)),256)

    def evaluate(self,N):
        chart,geometry=self.geometry_at(N)
        return self.evaluate_prepared(N,chart,geometry)
