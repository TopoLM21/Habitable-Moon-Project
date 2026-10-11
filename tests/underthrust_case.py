"""Declared physical inputs for differential material-pair force controls.

Eight spherical octants form two x hemispheres.  All four actual physical
fault links are retained, including opening surface arcs: archived donor-bank
measure continues to belong to its original port.  No source parameter reads
the solved force or a desired plate speed.
"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import math

import numpy as np

from analysis.geometric_initiation_validation.admission_case import fixture
from tectonics.genesis_contact_law import ContactLawParameters
from tectonics.geometric_bending import ElasticBendingBankParameters, FaultBendingParameters
from tectonics.geometric_bending_contact_dynamics import initialize_bending_contact_dynamics
from tectonics.geometric_boundary import initialize_boundary
from tectonics.geometric_contacts import extract_contacts
from tectonics.geometric_contact_mechanics import ElasticColumnParameters
from tectonics.geometric_fault_traces import PhysicalFaultTrace, TraceQuadrature, declare_fault_trace_set
from tectonics.geometric_spatial_mantle_loading import (
    PrescribedSpatialMantleSource, assemble_physical_spatial_mantle_load,
)


PROVENANCE = "Independent affine x mantle traction on declared x-hemisphere weak-plane ports"
SOURCE = PrescribedSpatialMantleSource((0., 0., 1e-12), 1e14, 1., (.2, 0., 0.), PROVENANCE)
SECOND_SOURCE = PrescribedSpatialMantleSource((0., 0., 1.1e-12), 1.2e14, 1., (.17, 0., 0.),
    "Explicit second source interval selected before either force result")
DT_S = 1e6
BOUNDS = dict(initial_loading_dt_s=1e6, max_rotation_radians=.01,
    max_tip_slope=.01, max_bending_strain=.001, max_deflection_radius_ratio=.001)


@lru_cache(maxsize=None)
def physical_inputs(*, initial_gap_m=-.1, viscosity_pa_s_m=0., cohesion_pa=0.,
                    friction=0., owners=(0, 1), quadrature_order=4):
    mesh, origin = fixture()
    origin = replace(origin, fragments=tuple(replace(fragment,
        parcel=replace(fragment.parcel, plate=owners[fragment.parcel.plate]))
        for fragment in origin.fragments))
    law = ContactLawParameters(normal_stiffness_pa_m=1e6, tangential_stiffness_pa_m=1e6,
        tensile_strength_pa=2e6, fracture_energy_j_m2=2e7, cohesion_pa=cohesion_pa,
        friction_dry=friction, friction_wet=friction, wet_cohesion_fraction=.5,
        viscosity_pa_s_m=viscosity_pa_s_m)
    columns = ElasticColumnParameters(1000., 1e9, 4e8, 1000., 1e9, 4e8)
    traces = []
    for contact in extract_contacts(origin.fragments, origin.radius_km):
        start, end = contact.start, contact.end
        wanted = np.asarray(contact.normal_a_to_b)*(1. if contact.plate_b == owners[0] else -1.)
        if float(np.cross(start, end)@wanted) < 0.:
            start, end = end, start
        traces.append(PhysicalFaultTrace("x-port:"+contact.contact_id, start, end,
            owners[1], owners[0], math.pi/6., columns, law, 1000., .7, 0.,
            initial_gap_m, 0., "Explicit inclined physical port; pressure from declared preload"))
    trace_set = declare_fault_trace_set(origin, traces, TraceQuadrature(1, quadrature_order))
    sub = ElasticBendingBankParameters(10000., 2000., 7e10, .25,
        "Declared original donor cantilever bank, including archived measure")
    over = ElasticBendingBankParameters(15000., 3000., 8e10, .3,
        "Declared original receiver cantilever bank")
    bending = tuple(FaultBendingParameters(trace.trace_id, sub, over) for trace in trace_set.traces)
    load = assemble_physical_spatial_mantle_load(mesh, origin, SOURCE)
    seed = initialize_bending_contact_dynamics(origin, trace_set, load, bending, **BOUNDS)
    assert seed.rotation_radians == ((0., 0., 0.),)*2
    assert all(sub_w == over_w == 0. for _, sub_w, over_w in seed.transverse_displacement_m)
    assert all(local.plastic_slip_m == local.cumulative_slip_m == local.jump_free_m == 0.
               for _, local in seed.states)
    return mesh, origin, initialize_boundary(origin), seed, load


def initialize_case(*, max_anchor_transport_error_m=1e-4, **input_options):
    from tectonics.geometric_underthrust_motion import (
        HotBirthParameters, initialize_underthrust_motion,
    )
    mesh, origin, inventory, seed, load = physical_inputs(**input_options)
    births = HotBirthParameters(2., 0., 0.,
        "Explicit hot departure-footprint closure for actual material vacancies")
    state = initialize_underthrust_motion(mesh, origin, inventory, seed, load,
        birth_parameters=births, max_anchor_transport_error_m=max_anchor_transport_error_m)
    return mesh, origin, state, load


def advance_case(*, dt_s=DT_S, **input_options):
    from tectonics.geometric_underthrust_motion import advance_underthrust_motion
    mesh, origin, initial, load = initialize_case(**input_options)
    first = advance_underthrust_motion(mesh, origin, initial, load, dt_s, provenance=PROVENANCE)
    return mesh, origin, initial, first
