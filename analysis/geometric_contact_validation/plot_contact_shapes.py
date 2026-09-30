"""Two known spherical layouts with identical owner fractions and different arcs."""
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tectonics.fractional_surface import SurfaceParcel
from tectonics.geometric_contacts import contacts_between
from tectonics.mesh import build_icosphere
from tectonics.spherical_polygons import polygon_area


@dataclass(frozen=True)
class Fragment:
    fragment_id: str
    polygon: object
    parcel: object


def main():
    mesh = build_icosphere(0)
    vertices = mesh.vertices[mesh.faces[0]]
    center = vertices.sum(axis=0)
    center /= np.linalg.norm(center)
    e1 = vertices[1]-vertices[0]
    e1 -= center*np.dot(e1, center)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(center, e1)
    basis = np.stack((e1, e2), axis=1)
    project = lambda points: np.asarray(points)@basis/(np.asarray(points)@center)[..., None]
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.3), constrained_layout=True)
    for i, ax in enumerate(axes):
        a, b, c = np.roll(vertices, i, axis=0)
        midpoint = (b+c)/np.linalg.norm(b+c)
        pieces = (np.array((a, b, midpoint)), np.array((a, midpoint, c)))
        fragments = []
        for plate, (piece, color) in enumerate(zip(pieces, ("#72A8D8", "#F2AE67"))):
            area = polygon_area(piece)*100.**2
            parcel = SurfaceParcel(0, plate, f"material:{plate}", area, area, 0., 0., 0.)
            fragments.append(Fragment(chr(97+plate), piece, parcel))
            xy = project(piece)
            ax.fill(*xy.T, color=color, edgecolor="white", lw=1.2)
            ax.text(*project(piece.sum(axis=0)/np.linalg.norm(piece.sum(axis=0))),
                    f"Плита {plate+1}\n50 %", ha="center", va="center", fontsize=10)
        contact, = contacts_between(*fragments, 100.)
        line = project((contact.start, contact.end))
        ax.plot(*line.T, color="#222222", lw=3)
        p, n = np.array(contact.midpoint), np.array(contact.normal_a_to_b)
        derivative = ((n@basis)*(p@center)-(p@basis)*(n@center))/(p@center)**2
        derivative /= np.linalg.norm(derivative)
        origin = project(p)
        ax.annotate("", xy=origin+.13*derivative, xytext=origin,
                    arrowprops=dict(arrowstyle="->", color="#222222", lw=1.8))
        ax.set_title(f"Геометрия {i+1}: другое направление контакта", fontsize=11)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.margins(.15)
    figure.suptitle("Одинаковые доли площади не определяют границу плит", fontsize=15)
    figure.savefig(Path(__file__).with_name("equal_area_different_contacts.png"), dpi=180,
                   metadata={"Description": "Known spherical polygons; gnomonic drawing. Both layouts have exact half-cell owner areas."})
    plt.close(figure)


if __name__ == "__main__":
    main()
