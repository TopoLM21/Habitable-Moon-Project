"""Birth and private restart of compact memory from accepted Genesis.

The accepted planet is only read. The fixed-material Maxwell branch has its
own journal and never replaces the planetary checkpoint or advances its clock.
Untrusted planetary data must first pass the public slip-checkpoint replay.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

from .genesis_shell import ShellParameters
from .genesis_volume_checkpoint import _canonical
from .genesis_volume_dynamics import radial_moments
from .genesis_volume_enriched_memory import (
    initial_enriched_memory, validate_enriched_memory, enriched_memory_to_dict,
    enriched_memory_from_dict,
)
from .genesis_volume_loading import RadialMoments, prepare_volume_loading, _damage_matrix
from .genesis_volume_seed_reaction import (
    prepare_volume_seed_reaction, validate_volume_seed_reaction,
    volume_seed_reaction_to_dict, volume_seed_reaction_from_dict,
)
from .genesis_volume_slip_checkpoint import volume_slip_checkpoint_to_dict


FORMAT = "genesis-source-bound-compact-memory-birth-0.1"
SCOPE = "unchanged_accepted_Genesis_to_private_fixed_material_Maxwell_memory"


class SourceSeedMemoryRefusal(ValueError):
    """The private branch no longer follows its accepted material birth."""


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class SourceSeedMemoryBirth:
    reaction: object
    initial_memory: object
    payload_json: str
    binding_sha256: str

    @property
    def partition(self):
        return self.reaction.partition


def prepare_source_seed_memory(model, state, finite_seed, *, partition=None,
                               reaction=None, quadrature_order=8):
    """Preserve all old material memory and initialize only the new field."""
    if reaction is None:
        reaction = prepare_volume_seed_reaction(model, state, finite_seed,
            partition=partition, quadrature_order=quadrature_order)
    else:
        # A prepared descriptor alone cannot assert an accepted-source binding.
        reaction = volume_seed_reaction_from_dict(volume_seed_reaction_to_dict(reaction),
            model=model, state=state, finite_seed=finite_seed,
            partition=reaction.partition if partition is None else partition,
            quadrature_order=quadrature_order)
    p = reaction.partition
    memory = initial_enriched_memory(p, state.mechanics)
    packet = volume_slip_checkpoint_to_dict(model, state)
    body = dict(format=FORMAT, scope=SCOPE,
        accepted_parent_checkpoint_sha256=packet["checkpoint_sha256"],
        formation_checkpoint_sha256=packet["source_checkpoint_sha256"],
        source_identity=p.source_identity, accepted_time_myr=state.time_myr,
        accepted_step_sizes_myr=list(state.accepted_step_sizes_myr),
        reaction_binding_sha256=reaction.binding_sha256,
        partition_binding_sha256=p.binding_sha256,
        birth_parent_memory_sha256=memory.birth_parent_memory_sha256,
        shell_parameters=asdict(model.shell_parameters),
        fixed_cohort_damage=np.asarray(state.fault_history.damage)[p.parent_assembly.face_indices].tolist(),
        initial_memory=enriched_memory_to_dict(memory),
        saved_parent_elastic_energy_j=state.mechanics.elastic_energy_j,
        independently_reduced_parent_elastic_energy_j=p.initial_elastic_energy_j,
        inherited_energy_reduction_difference_j=p.initial_elastic_energy_j-state.mechanics.elastic_energy_j,
        energy_difference_scope="reported_arithmetic_reduction_difference_not_physical_work",
        physical_creation_admitted=False, physical_release_admitted=False,
        funding_admitted=False, accepted_planet_memory_changed=False,
        new_Genesis_time_myr=0., new_material_birth_count=0,
        creation_work_debited_from_Genesis_j=0.)
    raw = {**body, "source_memory_birth_sha256": _digest(body)}
    payload = _canonical(raw).decode()
    return SourceSeedMemoryBirth(reaction, memory, payload,
        hashlib.sha256(payload.encode()).hexdigest())


def validate_source_seed_memory(birth):
    if (type(birth) is not SourceSeedMemoryBirth
            or birth.binding_sha256 != hashlib.sha256(birth.payload_json.encode()).hexdigest()):
        raise SourceSeedMemoryRefusal("Source memory birth binding differs")
    validate_volume_seed_reaction(birth.reaction)
    validate_enriched_memory(birth.partition, birth.initial_memory)
    raw = json.loads(birth.payload_json)
    reaction_context = json.loads(birth.reaction.payload_json)
    p, memory = birth.partition, birth.initial_memory
    source_parameters = json.loads(p.parent_assembly.geometry.source_packet_json)["source"]["parameters"]["shell"]
    if (raw.get("shell_parameters") != source_parameters
            or not np.array_equal(_damage_matrix(ShellParameters(**source_parameters),
                raw.get("fixed_cohort_damage"), len(p.memory0)), p.parent_assembly.C)):
        raise SourceSeedMemoryRefusal("Private Maxwell parameters or damage differ from the authentic source")
    body = {k: v for k, v in raw.items() if k != "source_memory_birth_sha256"}
    if (raw.get("format") != FORMAT or raw.get("scope") != SCOPE
            or raw.get("source_memory_birth_sha256") != _digest(body)
            or raw.get("reaction_binding_sha256") != birth.reaction.binding_sha256
            or raw.get("partition_binding_sha256") != birth.partition.binding_sha256
            or raw.get("initial_memory") != enriched_memory_to_dict(birth.initial_memory)
            or raw.get("accepted_parent_checkpoint_sha256") != reaction_context["accepted_parent_checkpoint_sha256"]
            or raw.get("formation_checkpoint_sha256") != reaction_context["formation_checkpoint_sha256"]
            or raw.get("source_identity") != p.source_identity
            or raw.get("accepted_time_myr") != p.time_myr
            or raw.get("accepted_step_sizes_myr") != reaction_context["accepted_step_sizes_myr"]
            or raw.get("birth_parent_memory_sha256") != memory.birth_parent_memory_sha256
            or _digest(enriched_memory_to_dict(memory)["state"]["parent_mechanics"]) != reaction_context["full_memory_sha256"]
            or memory.private_steps != 0 or memory.time_myr != p.time_myr
            or np.any(memory.coefficients_m) or np.any(memory.displacement_m)
            or raw.get("saved_parent_elastic_energy_j") != memory.elastic_energy_j
            or raw.get("independently_reduced_parent_elastic_energy_j") != p.initial_elastic_energy_j
            or raw.get("inherited_energy_reduction_difference_j") != p.initial_elastic_energy_j-memory.elastic_energy_j
            or any(raw.get(k) is not False for k in (
                "physical_creation_admitted", "physical_release_admitted", "funding_admitted",
                "accepted_planet_memory_changed"))
            or any(raw.get(k) != 0 for k in (
                "new_Genesis_time_myr", "new_material_birth_count", "creation_work_debited_from_Genesis_j"))):
        raise SourceSeedMemoryRefusal("Source birth cannot replace history or authorize release")
    return birth


def source_seed_memory_to_dict(birth):
    validate_source_seed_memory(birth)
    return json.loads(birth.payload_json)


def source_seed_memory_from_dict(raw, *, model, state, finite_seed, partition=None,
                                  reaction=None, quadrature_order=8):
    expected = prepare_source_seed_memory(model, state, finite_seed, partition=partition,
        reaction=reaction, quadrature_order=quadrature_order)
    if type(raw) is not dict or _canonical(raw) != _canonical(source_seed_memory_to_dict(expected)):
        raise SourceSeedMemoryRefusal("Source birth differs from accepted Genesis reconstruction")
    return expected


def source_seed_memory_loading(birth, private_state, after_time_myr):
    """Obtain Maxwell factors from the unchanged source constitutive parameters."""
    validate_source_seed_memory(birth)
    p = birth.partition
    validate_enriched_memory(p, private_state)
    if private_state.birth_parent_memory_sha256 != birth.initial_memory.birth_parent_memory_sha256:
        raise SourceSeedMemoryRefusal("Private branch has a different inherited birth memory")
    raw = json.loads(birth.payload_json)
    mechanics = private_state.parent_state
    temperature = [c.temperature_k for c in mechanics.cohorts]
    birth_temperature = [c.temperature_k for c in birth.initial_memory.parent_state.cohorts]
    if not np.array_equal(temperature, birth_temperature):
        raise SourceSeedMemoryRefusal("Private fixed-material temperatures differ from the authentic birth")
    moments = radial_moments(p.parent_assembly)
    n = len(mechanics.cohorts)
    removed = RadialMoments((0.,)*n, moments.inverse_radius_mean, moments.inverse_radius_variance)
    damage = raw["fixed_cohort_damage"]
    return prepare_volume_loading(mechanics, after_ids=p.parent_assembly.cohort_ids,
        after_temperature_k=temperature, birth_temperature_k=temperature,
        radial_moments=moments, before_radial_moments=moments,
        removed_radial_moments=removed, before_damage=damage, after_damage=damage,
        parameters=ShellParameters(**raw["shell_parameters"]), after_time_myr=after_time_myr,
        before_temperature_k=temperature)


def restart_source_seed_memory(birth, raw):
    """Re-evaluate a private journal from its genuine, unchanged Genesis birth."""
    validate_source_seed_memory(birth)
    return enriched_memory_from_dict(raw, birth.partition, birth.initial_memory.parent_state,
        loading_factory=lambda s, t: source_seed_memory_loading(birth, s, t))
