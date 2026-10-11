"""Exact rational audit of supplied whole-clock scalar support certificates.

This checks polynomial contraction, tail addition and a complete clock cover.
The external source-bound typed loader remains responsible for native rays,
ordered SO3 coefficient construction and exact birth-incidence identities.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction as F
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from analysis.geometric_native_birth_front_validation.verify_baseline import digest,read,write,encoded


def rational(value):
    if (type(value)is not list or len(value)!=2 or any(type(x)is not int for x in value)
            or value[1]<=0):
        raise ValueError("A scalar proof needs an exact primitive rational")
    result=F(*value)
    if [result.numerator,result.denominator]!=value:
        raise ValueError("A scalar proof rational must be canonical")
    return result


def interval(value):
    if type(value)is not dict or set(value)!={"lo","hi"}:
        raise ValueError("A scalar proof needs a literal complete interval")
    low,high=rational(value["lo"]),rational(value["hi"])
    if low>high:
        raise ValueError("A scalar interval is reversed")
    return low,high


def plus(first,second):
    return first[0]+second[0],first[1]+second[1]


def product(first,second):
    values=[a*b for a in first for b in second]
    return min(values),max(values)


def tree_vertices(value,resources=None):
    resources={} if resources is None else resources
    if type(value)is dict:
        resources=dict(resources,**{name:value[name] for name in ("precision","order","time_cells") if name in value})
        if "scalar_coefficients" in value and "cell_sign_bounds" in value:
            yield value,resources
        else:
            for nested in value.values():
                yield from tree_vertices(nested,resources)
    elif type(value)is list:
        for nested in value:
            yield from tree_vertices(nested,resources)


def audit_vertex(vertex,resources=None):
    resources=dict({} if resources is None else resources,**vertex)
    order,precision,cells=resources["order"],resources["precision"],resources["time_cells"]
    if (type(order)is not int or not 12<=order<=40 or type(precision)is not int
            or not 128<=precision<=512 or type(cells)is not int or not 1<=cells<=256):
        raise ValueError("A scalar proof has undeclared or unbounded resources")
    coefficients=[interval(value) for value in vertex["scalar_coefficients"]]
    if len(coefficients)!=order+1:
        raise ValueError("A scalar proof lost a Taylor coefficient")
    incidence=vertex["exact_initial_incidence"]
    divided=vertex["divided_by_birth_clock"]
    if type(incidence)is not bool or type(divided)is not bool or incidence!=divided:
        raise ValueError("An incidence clock division needs an explicit exact identity")
    if incidence and coefficients[0]!=(F(0),F(0)):
        raise ValueError("A birth-clock factor was asserted without a zero constant")
    tail=rational(vertex["scalar_remainder_bound"])
    if tail<0 or len(vertex["cell_sign_bounds"])!=cells:
        raise ValueError("The scalar proof has a negative tail or incomplete time cover")
    selected=coefficients[1:] if divided else coefficients
    power=order if divided else order+1
    rows=[]
    for index,row in enumerate(vertex["cell_sign_bounds"]):
        clock=interval(row["time_fraction"])
        if clock!=(F(index,cells),F(index+1,cells)):
            raise ValueError("The scalar proof left a clock gap or overlapping substitute cell")
        value=(F(0),F(0))
        for coefficient in reversed(selected):
            value=plus(product(value,clock),coefficient)
        remainder=tail*clock[1]**power
        value=plus(value,(-remainder,remainder))
        reported=interval(row["sign_bounds"])
        if reported[0]>value[0] or reported[1]<value[1] or value[1]>0 or reported[1]>0:
            raise ValueError("The complete polynomial and remainder are not certified nonpositive")
        rows.append(dict(time_fraction=[[clock[0].numerator,clock[0].denominator],
            [clock[1].numerator,clock[1].denominator]],
            independently_contracted_upper=[value[1].numerator,value[1].denominator]))
    return dict(order=order,precision=precision,time_cells=cells,
        birth_clock_division=divided,constant_exactly_zero=coefficients[0]==(F(0),F(0)),cells=rows)


def audit_pair_polynomials(paths,*,expected_time_cells=1):
    started=time.perf_counter()
    records=[]
    for path in paths:
        packet=read(path)
        vertices=list(tree_vertices(packet,{"time_cells":expected_time_cells}))
        if not vertices:
            raise ValueError("The input has no complete scalar polynomial certificate")
        rows=[audit_vertex(vertex,resources) for vertex,resources in vertices]
        records.append(dict(path=str(Path(path).resolve().relative_to(ROOT)),sha256=digest(path),
            vertex_count=len(rows),birth_clock_division_count=sum(row["birth_clock_division"] for row in rows),
            vertices=rows))
    return dict(format="native-birth-independent-scalar-polynomial-audit-1",status="PASS",
        verified_utc=datetime.now(timezone.utc).isoformat(),seconds=time.perf_counter()-started,
        auditor_source=dict(path=str(Path(__file__).resolve().relative_to(ROOT)),sha256=digest(__file__)),
        inputs=records,vertex_count=sum(row["vertex_count"] for row in records),
        externally_declared_time_cells=expected_time_cells,
        exact_rational_horner=True,complete_clock_partition_checked=True,
        positive_area_tolerance_used=False,production_geometry_imported=False,
        native_coefficient_generation_reproved_here=False,native_incidence_reproved_here=False,
        finite_pose_mechanical_admission_proved=False,whole_native_interval_admitted=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,nargs="+",required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    receipt=write(args.out,audit_pair_polynomials(args.input),"receipt_sha256")
    print(encoded({key:receipt[key] for key in ("status","seconds","vertex_count")}).decode())
