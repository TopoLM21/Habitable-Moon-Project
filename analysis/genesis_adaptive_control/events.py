"""Opt-in conservative event enclosure/localization; no physical transition law."""
import copy
import math
import hashlib
from pathlib import Path
import time
from dataclasses import asdict, dataclass
from .controller import fingerprint, time_limit


@dataclass(frozen=True)
class Event:
    kind: str
    identity: str
    units: str
    transition: str

    def __post_init__(self):
        if self.kind not in ('phase', 'birth', 'damage_gate'):
            raise ValueError('EVENT_KIND')
        if not all(isinstance(v, str) and v for v in asdict(self).values()):
            raise ValueError('EVENT_METADATA')


class EventPolicy:
    """enclose(snapshot, event, a, b) returns bounds on signed safety margin.

    Positive means admissible. Bounds MUST enclose the complete path, including
    all coupled stages and alternative trial paths, not merely endpoint samples.
    Provider assumptions and source/implementation SHA are explicit provenance.
    No transition is currently supported: a possible boundary stops before trial.
    """
    def __init__(self, events, enclose, *, source_sha256, provider_sha256,
                 assumptions, valid_years, resolution_years, max_evaluations,
                 seconds, clock=time.monotonic):
        self.events=tuple(events);self.enclose=enclose;self.clock=clock
        if len({(e.kind,e.identity) for e in self.events})!=len(self.events):
            raise ValueError('DUPLICATE_EVENT')
        for sha in (source_sha256,provider_sha256):
            if not isinstance(sha,str) or len(sha)!=64 or any(x not in '0123456789abcdef' for x in sha):
                raise ValueError('EVENT_SHA')
        if not isinstance(assumptions,str) or not assumptions.strip():raise ValueError('EVENT_ASSUMPTIONS')
        if len(valid_years)!=2 or not all(math.isfinite(x) for x in valid_years) or valid_years[0]>=valid_years[1]:
            raise ValueError('EVENT_VALIDITY')
        for x in (resolution_years,seconds):
            if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<=0:raise ValueError('EVENT_LIMIT')
        if type(max_evaluations) is not int or max_evaluations<1:raise ValueError('EVENT_LIMIT')
        self.contract=dict(implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),events=[asdict(e) for e in self.events],source_sha256=source_sha256,
            provider_sha256=provider_sha256,assumptions=assumptions,valid_years=list(valid_years),
            resolution_years=resolution_years,max_evaluations=max_evaluations,seconds=seconds)
        self.binding=fingerprint(self.contract)
        self.used=0;self.seconds_used=0.;self.blocked=None

    def state(self):
        return dict(binding=self.binding,used=self.used,seconds_used=self.seconds_used,blocked=copy.deepcopy(self.blocked))

    def restore(self,state):
        if state['binding']!=self.binding or fingerprint(self.contract)!=self.binding:raise ValueError('EVENT_RESTART_BINDING')
        if type(state['used']) is not int or not 0<=state['used']<=self.contract['max_evaluations']:raise ValueError('EVENT_RESTART_BUDGET')
        if not math.isfinite(state['seconds_used']) or state['seconds_used']<0:raise ValueError('EVENT_RESTART_TIME')
        self.used=state['used'];self.seconds_used=state['seconds_used'];self.blocked=copy.deepcopy(state['blocked'])

    def check(self,snapshot,dt):
        if self.blocked is not None:return copy.deepcopy(self.blocked)
        started=self.clock();c=self.contract;a=snapshot['elapsed_years'];b=a+dt
        result=None
        try:
            if fingerprint(self.contract)!=self.binding:raise ValueError('EVENT_CONTRACT_MUTATED')
            if snapshot.get('source_sha256')!=c['source_sha256']:raise ValueError('EVENT_SOURCE_MISMATCH')
            if not math.isfinite(a) or not math.isfinite(b) or dt<=0 or a<c['valid_years'][0] or b>c['valid_years'][1]:
                raise ValueError('EVENT_OUTSIDE_VALIDITY')
            def bounds(e,lo,hi):
                if self.used>=c['max_evaluations']:raise RuntimeError('EVENT_EVALUATION_LIMIT')
                remaining=c['seconds']-self.seconds_used-(self.clock()-started)
                if remaining<=0:raise TimeoutError('EVENT_TIME_LIMIT')
                self.used+=1;s=copy.deepcopy(snapshot);before=fingerprint(s)
                with time_limit(remaining):v=self.enclose(s,e,lo,hi)
                if fingerprint(s)!=before:raise ValueError('EVENT_PROVIDER_MUTATED_INPUT')
                if v is None:raise ValueError('EVENT_BOUND_UNAVAILABLE')
                if len(v)!=2 or not all(math.isfinite(x) for x in v) or v[0]>v[1]:raise ValueError('EVENT_INVALID_ENCLOSURE')
                return v
            # Left-first subdivision finds earliest POSSIBLE contact. A loose bound
            # can stop early; it never proves that an event actually occurred.
            stack=[(a,b)];safe_until=None
            while stack:
                lo,hi=stack.pop();possible=[]
                for e in self.events:
                    v=bounds(e,lo,hi)
                    if v[0]<=0:possible.append(dict(**asdict(e),margin_bounds=list(v)))
                if not possible:
                    safe_until=hi;continue
                if hi-lo<=c['resolution_years']:
                    result=dict(status='BLOCKED',reason='POSSIBLE_UNSUPPORTED_EVENT',
                        bracket_years=[lo,hi],certified_safe_until_years=safe_until,events=possible)
                    break
                mid=lo+(hi-lo)/2
                if mid==lo or mid==hi:raise RuntimeError('EVENT_RESOLUTION_LIMIT')
                stack.extend(((mid,hi),(lo,mid)))
            if result is None:result=dict(status='CLEAR',certified_safe_until_years=b)
        except Exception as exc:
            result=dict(status='UNRESOLVED',reason=f'{type(exc).__name__}: {exc}')
        finally:self.seconds_used+=max(0.,self.clock()-started)
        if result['status']!='CLEAR':self.blocked=copy.deepcopy(result)
        return result
