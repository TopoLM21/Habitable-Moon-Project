"""Exact binary64-ray convex intersection, independent of the geometry kernel.

Gnomonic projection and clipping use Fraction throughout.  Decimal spherical
areas preserve the resulting rational rays; the area witness is never rounded
back to binary64 before evaluation.  Only the final human-readable area rounds.
"""
import argparse
from decimal import Decimal, localcontext
from fractions import Fraction
import json
from pathlib import Path

F = Fraction
D = Decimal


def dot(a, b):
    return sum((x*y for x, y in zip(a, b)), 0)


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def cross2(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def area2(polygon):
    return sum((cross2(polygon[0], polygon[i], polygon[i+1])
        for i in range(1, len(polygon)-1)), F(0)) if polygon else F(0)


def clean(polygon):
    result = []
    for point in polygon:
        if not result or point != result[-1]:
            result.append(point)
    if len(result) > 1 and result[0] == result[-1]:
        result.pop()
    return result


def decimal(value):
    return D(value.numerator)/D(value.denominator)


def atan(value):
    assert abs(value) < D(".5")
    total, power, index = value, value, 1
    while True:
        power *= -value*value
        index += 2
        term = power/index
        total += term
        if abs(term) < D("1e-155"):
            return total


def spherical_area(polygon):
    points = [tuple(decimal(value) for value in point) for point in polygon]
    total = D(0)
    for i in range(1, len(points)-1):
        a, b, c = points[0], points[i], points[i+1]
        na, nb, nc = (dot(point, point).sqrt() for point in (a, b, c))
        numerator = dot(a, cross(b, c))
        denominator = na*nb*nc+dot(a, b)*nc+dot(b, c)*na+dot(c, a)*nb
        total += 2*atan(numerator/denominator)
    return abs(total)


def convex_signs(polygon):
    values = []
    for i, (a, b) in enumerate(zip(polygon, polygon[1:]+polygon[:1])):
        values.extend(dot(cross(a, b), point) for j, point in enumerate(polygon)
            if j not in (i, (i+1) % len(polygon)))
    return dict(positive=sum(value > 0 for value in values),
                negative=sum(value < 0 for value in values), zero=sum(value == 0 for value in values))


def solve(record, index):
    a_record = record.get("a", record.get("fragment_a"))
    b_record = record.get("b", record.get("fragment_b"))
    a_raw, b_raw = (list(tuple(F(value) for value in point) for point in item["polygon"])
                    for item in (a_record, b_record))
    normal = tuple(sum((point[k] for point in a_raw+b_raw), F(0)) for k in range(3))
    drop = max(range(3), key=lambda k: abs(normal[k]))
    keep = [k for k in range(3) if k != drop]

    def project(point):
        denominator = dot(normal, point)
        assert denominator > 0, "Both complete footprints must fit the projection hemisphere"
        return tuple(point[k]/denominator for k in keep)

    a, b = [list(map(project, polygon)) for polygon in (a_raw, b_raw)]
    if area2(a) < 0:
        a.reverse()
    if area2(b) < 0:
        b.reverse()
    result = a
    for u, v in zip(b, b[1:]+b[:1]):
        clipped = []
        for p, q in zip(result, result[1:]+result[:1]):
            dp, dq = cross2(u, v, p), cross2(u, v, q)
            if dp*dq < 0:
                parameter = dp/(dp-dq)
                clipped.append(tuple(x+parameter*(y-x) for x, y in zip(p, q)))
            if dq >= 0:
                clipped.append(q)
        result = clean(clipped)
        if not result:
            break
    world = []
    for point in result:
        vector = [F(0), F(0), F(0)]
        for k, value in zip(keep, point):
            vector[k] = value
        vector[drop] = (1-sum((normal[k]*vector[k] for k in keep), F(0)))/normal[drop]
        world.append(tuple(vector))
    area = spherical_area(world)
    normalized = []
    for vector in world:
        vector_decimal = tuple(map(decimal, vector))
        norm = dot(vector_decimal, vector_decimal).sqrt()
        normalized.append([str(value/norm) for value in vector_decimal])
    containment = []
    for point in record["overlap"]:
        ray = tuple(F(value) for value in point)
        containment.append({label: [i for i, (u, v) in enumerate(zip(polygon, polygon[1:]+polygon[:1]))
            if dot(cross(u, v), ray) < 0] for label, polygon in (("outside_a_edges", a_raw), ("outside_b_edges", b_raw))})
    return dict(index=index, fragment_a=a_record["fragment_id"], fragment_b=b_record["fragment_id"],
        raw_convex_signs_a=convex_signs(a_raw), raw_convex_signs_b=convex_signs(b_raw),
        exact_projected_area=str(decimal(area2(result)/2)), exact_positive=area2(result) > 0,
        spherical_area_sr_decimal=str(area), spherical_area_sr_float=float(area),
        birth_polygon_area_sr_decimal=str(spherical_area(b_raw)),
        reported_kernel_area_sr=record["area_sr"], reported_vertex_containment=containment,
        intersection_vertices_decimal=normalized)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    options = parser.parse_args()
    capture = json.loads(options.input.read_text())
    records = capture.get("records", capture.get("overlaps"))
    with localcontext() as context:
        context.prec = 160
        report = dict(method="exact Fraction binary64 rays / gnomonic halfspaces / Decimal160 spherical area",
            binary64_ray_semantics=True, input=str(options.input),
            records=[solve(record, index) for index, record in enumerate(records)])
    with options.out.open("x") as stream:
        stream.write(json.dumps(report, indent=2)+"\n")
    print(json.dumps([dict(index=row["index"], positive=row["exact_positive"],
        area_sr=row["spherical_area_sr_float"]) for row in report["records"]]))


if __name__ == "__main__":
    main()
