"""Deterministic and parameterized workload generators for multi-agent handoff simulation.

Provides standard execution topologies:
- Cyclic pipeline (A -> B -> C -> A) with alternating working sets
- Hub-and-Spoke orchestrator (Orchestrator -> Worker_i -> Orchestrator)
- Branching pipeline (Planner -> Coder -> Reviewer -> Tester with probabilistic loopback)
- Adversarial / Stochastic pipeline (random next-agent transitions)
"""

from __future__ import annotations

import random
from typing import Sequence

from teleportcache.types import AgentId, BlockId, HandoffEvent, KVBlock, Workload

DEFAULT_BLOCK_TOKENS = 512
DEFAULT_BLOCK_SIZE_BYTES = 1_000_000


def make_block(
    block_id: BlockId,
    num_tokens: int = DEFAULT_BLOCK_TOKENS,
    size_bytes: int = DEFAULT_BLOCK_SIZE_BYTES,
) -> KVBlock:
    """Create a fixed-size KV page block."""
    return KVBlock(block_id=block_id, num_tokens=num_tokens, size_bytes=size_bytes)


def generate_cyclic_workload(
    num_steps: int = 50,
    agents: Sequence[AgentId] = ("A", "B", "C"),
    blocks_per_set: int = 4,
    num_sets: int = 2,
    block_tokens: int = DEFAULT_BLOCK_TOKENS,
    block_size_bytes: int = DEFAULT_BLOCK_SIZE_BYTES,
) -> Workload:
    """Generates a round-robin cyclic pipeline workload.

    Agents hand off cyclically: agents[0] -> agents[1] -> ... -> agents[n-1] -> agents[0].
    Each agent has ``num_sets`` disjoint sets of ``blocks_per_set`` blocks and
    alternates between them across visits.
    """
    agent_list = list(agents)
    n_agents = len(agent_list)
    blocks: dict[BlockId, KVBlock] = {}

    for agent in agent_list:
        for s in range(num_sets):
            for b in range(blocks_per_set):
                b_id = f"{agent}_s{s}_b{b}"
                blocks[b_id] = make_block(b_id, block_tokens, block_size_bytes)

    events: list[HandoffEvent] = []
    agent_visit_counts: dict[AgentId, int] = {agent: 0 for agent in agent_list}

    for step in range(num_steps):
        from_agent = agent_list[step % n_agents]
        to_agent = agent_list[(step + 1) % n_agents]
        visit = agent_visit_counts[to_agent]
        agent_visit_counts[to_agent] += 1

        active_set = visit % num_sets
        required = tuple(f"{to_agent}_s{active_set}_b{b}" for b in range(blocks_per_set))
        events.append(
            HandoffEvent(
                step=step,
                from_agent=from_agent,
                to_agent=to_agent,
                required_blocks=required,
            )
        )

    return Workload(
        name=f"cyclic_{n_agents}agents_{num_steps}steps",
        agents=agent_list,
        blocks=blocks,
        events=events,
    )


def generate_hub_and_spoke_workload(
    num_steps: int = 50,
    num_workers: int = 3,
    blocks_per_agent: int = 4,
    orchestrator_id: AgentId = "Orchestrator",
    block_tokens: int = DEFAULT_BLOCK_TOKENS,
    block_size_bytes: int = DEFAULT_BLOCK_SIZE_BYTES,
    seed: int = 42,
) -> Workload:
    """Generates an Orchestrator-Worker handoff pattern.

    Transitions alternate between Orchestrator -> Worker and Worker -> Orchestrator.
    Workers are chosen round-robin or probabilistically by seed.
    """
    rng = random.Random(seed)
    workers = [f"Worker_{i+1}" for i in range(num_workers)]
    agent_list = [orchestrator_id] + workers
    blocks: dict[BlockId, KVBlock] = {}

    for agent in agent_list:
        for b in range(blocks_per_agent * 2):
            b_id = f"{agent}_b{b}"
            blocks[b_id] = make_block(b_id, block_tokens, block_size_bytes)

    events: list[HandoffEvent] = []
    curr_agent = orchestrator_id

    for step in range(num_steps):
        if curr_agent == orchestrator_id:
            next_agent = rng.choice(workers)
        else:
            next_agent = orchestrator_id

        # Half persistent blocks, half variable
        subset_idx = (step // 2) % 2
        offset = subset_idx * (blocks_per_agent // 2)
        required = tuple(f"{next_agent}_b{offset + i}" for i in range(blocks_per_agent))

        events.append(
            HandoffEvent(
                step=step,
                from_agent=curr_agent,
                to_agent=next_agent,
                required_blocks=required,
            )
        )
        curr_agent = next_agent

    return Workload(
        name=f"hub_spoke_{num_workers}workers_{num_steps}steps",
        agents=agent_list,
        blocks=blocks,
        events=events,
    )


def generate_branching_workload(
    num_steps: int = 50,
    loopback_prob: float = 0.25,
    blocks_per_agent: int = 4,
    block_tokens: int = DEFAULT_BLOCK_TOKENS,
    block_size_bytes: int = DEFAULT_BLOCK_SIZE_BYTES,
    seed: int = 42,
) -> Workload:
    """Generates a software engineering agent pipeline with probabilistic rework loops.

    Pipeline: Planner -> Coder -> Reviewer -> Tester -> Planner...
    With probability ``loopback_prob``:
      - Reviewer may loop back to Coder
      - Tester may loop back to Coder
    """
    rng = random.Random(seed)
    stages = ["Planner", "Coder", "Reviewer", "Tester"]
    blocks: dict[BlockId, KVBlock] = {}

    for stage in stages:
        for b in range(blocks_per_agent * 2):
            b_id = f"{stage}_b{b}"
            blocks[b_id] = make_block(b_id, block_tokens, block_size_bytes)

    events: list[HandoffEvent] = []
    curr_stage_idx = 0

    for step in range(num_steps):
        from_agent = stages[curr_stage_idx]

        if from_agent == "Reviewer" and rng.random() < loopback_prob:
            next_stage_idx = stages.index("Coder")
        elif from_agent == "Tester" and rng.random() < loopback_prob:
            next_stage_idx = stages.index("Coder")
        else:
            next_stage_idx = (curr_stage_idx + 1) % len(stages)

        to_agent = stages[next_stage_idx]
        subset = (step // 4) % 2
        offset = subset * (blocks_per_agent // 2)
        required = tuple(f"{to_agent}_b{offset + i}" for i in range(blocks_per_agent))

        events.append(
            HandoffEvent(
                step=step,
                from_agent=from_agent,
                to_agent=to_agent,
                required_blocks=required,
            )
        )
        curr_stage_idx = next_stage_idx

    return Workload(
        name=f"branching_pipeline_{num_steps}steps",
        agents=stages,
        blocks=blocks,
        events=events,
    )


def generate_adversarial_workload(
    num_steps: int = 50,
    agents: Sequence[AgentId] = ("Agent_X", "Agent_Y", "Agent_Z", "Agent_W"),
    blocks_per_agent: int = 4,
    block_tokens: int = DEFAULT_BLOCK_TOKENS,
    block_size_bytes: int = DEFAULT_BLOCK_SIZE_BYTES,
    seed: int = 42,
) -> Workload:
    """Generates a uniformly random handoff sequence (stress test for predictors)."""
    rng = random.Random(seed)
    agent_list = list(agents)
    blocks: dict[BlockId, KVBlock] = {}

    for agent in agent_list:
        for b in range(blocks_per_agent * 3):
            b_id = f"{agent}_b{b}"
            blocks[b_id] = make_block(b_id, block_tokens, block_size_bytes)

    events: list[HandoffEvent] = []
    curr_agent = agent_list[0]

    for step in range(num_steps):
        candidates = [a for a in agent_list if a != curr_agent]
        to_agent = rng.choice(candidates)
        chosen_blocks = tuple(
            f"{to_agent}_b{b}"
            for b in rng.sample(range(blocks_per_agent * 3), blocks_per_agent)
        )
        events.append(
            HandoffEvent(
                step=step,
                from_agent=curr_agent,
                to_agent=to_agent,
                required_blocks=chosen_blocks,
            )
        )
        curr_agent = to_agent

    return Workload(
        name=f"adversarial_random_{num_steps}steps",
        agents=agent_list,
        blocks=blocks,
        events=events,
    )
