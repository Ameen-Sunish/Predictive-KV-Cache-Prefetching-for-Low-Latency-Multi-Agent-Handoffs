"""Workload generators and traces for TeleportCache."""

from teleportcache.workloads.generators import (
    generate_adversarial_workload,
    generate_branching_workload,
    generate_cyclic_workload,
    generate_hub_and_spoke_workload,
    make_block,
)

__all__ = [
    "generate_adversarial_workload",
    "generate_branching_workload",
    "generate_cyclic_workload",
    "generate_hub_and_spoke_workload",
    "make_block",
]
