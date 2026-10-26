"""Unit tests for workload generators."""

from __future__ import annotations

import pytest

from teleportcache.workloads.generators import (
    generate_adversarial_workload,
    generate_branching_workload,
    generate_cyclic_workload,
    generate_hub_and_spoke_workload,
)


def _assert_valid_workload(workload, expected_steps: int):
    assert len(workload.events) == expected_steps
    for i, ev in enumerate(workload.events):
        assert ev.step == i
        assert ev.from_agent in workload.agents
        assert ev.to_agent in workload.agents
        assert len(ev.required_blocks) > 0
        for b in ev.required_blocks:
            assert b in workload.blocks
            assert workload.blocks[b].block_id == b


def test_generate_cyclic_workload():
    wl = generate_cyclic_workload(num_steps=30, agents=("X", "Y", "Z"), blocks_per_set=4)
    _assert_valid_workload(wl, 30)
    assert wl.agents == ["X", "Y", "Z"]
    # Check cyclic handoffs
    assert wl.events[0].from_agent == "X"
    assert wl.events[0].to_agent == "Y"
    assert wl.events[1].from_agent == "Y"
    assert wl.events[1].to_agent == "Z"
    assert wl.events[2].from_agent == "Z"
    assert wl.events[2].to_agent == "X"


def test_generate_hub_and_spoke_workload():
    wl = generate_hub_and_spoke_workload(num_steps=40, num_workers=3, seed=123)
    _assert_valid_workload(wl, 40)
    assert wl.agents[0] == "Orchestrator"
    assert len(wl.agents) == 4
    # Check that orchestrator is always one endpoint of every handoff
    for ev in wl.events:
        assert ev.from_agent == "Orchestrator" or ev.to_agent == "Orchestrator"


def test_generate_branching_workload():
    wl1 = generate_branching_workload(num_steps=25, seed=42)
    wl2 = generate_branching_workload(num_steps=25, seed=42)
    _assert_valid_workload(wl1, 25)
    # Check determinism with same seed
    for ev1, ev2 in zip(wl1.events, wl2.events):
        assert ev1.from_agent == ev2.from_agent
        assert ev1.to_agent == ev2.to_agent
        assert ev1.required_blocks == ev2.required_blocks


def test_generate_adversarial_workload():
    wl = generate_adversarial_workload(num_steps=50, seed=99)
    _assert_valid_workload(wl, 50)
    # No self-handoffs
    for ev in wl.events:
        assert ev.from_agent != ev.to_agent
