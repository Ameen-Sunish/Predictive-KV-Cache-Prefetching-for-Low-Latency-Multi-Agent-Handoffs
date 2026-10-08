"""Shared deterministic builders for tests/prediction."""

from __future__ import annotations

from teleportcache.types import (
    AgentId,
    BlockId,
    HandoffEvent,
    KVBlock,
    PolicyView,
    SimConfig,
    Workload,
)

BLOCK_TOKENS = 512
BLOCK_SIZE_BYTES = 1_000_000

#: Per-block page-in time for the default sizes: 0.001 + 1e6/1.25e9 = 0.0018 s
SIM_CONFIG = SimConfig(
    capacity_blocks=4,
    bandwidth_bytes_per_s=1.25e9,
    transfer_latency_s=0.001,
    recompute_time_per_token_s=0.001,
    prefetch_window_s=0.05,
    eviction_policy="lru",
)


def make_block(
    block_id: BlockId,
    num_tokens: int = BLOCK_TOKENS,
    size_bytes: int = BLOCK_SIZE_BYTES,
) -> KVBlock:
    """A fixed-size KV page."""
    return KVBlock(block_id=block_id, num_tokens=num_tokens, size_bytes=size_bytes)


def cycle_events(
    num_steps: int,
    cycle: tuple[AgentId, ...] = ("A", "B", "C"),
) -> list[HandoffEvent]:
    """Events whose to_agent sequence repeats ``cycle`` (required_blocks empty).

    Event i goes from cycle[(i-1) % len] to cycle[i % len], so the agent
    visit order is exactly the cycle repeated.
    """
    events: list[HandoffEvent] = []
    for i in range(num_steps):
        to_agent = cycle[i % len(cycle)]
        from_agent = cycle[(i - 1) % len(cycle)]
        events.append(
            HandoffEvent(step=i, from_agent=from_agent, to_agent=to_agent, required_blocks=())
        )
    return events


def cyclic_workload(num_steps: int = 30, capacity_blocks: int = 4) -> Workload:
    """A deterministic A->B->C->A pipeline workload.

    Each agent owns TWO disjoint sets of 4 blocks and alternates between them
    on consecutive visits (visit v uses set v % 2). With per-agent capacity
    ``capacity_blocks`` = 4, demand paging must re-transfer a full set on every
    visit, while a correct predictor can prefetch the needed set in advance —
    the canonical setup where predictive prefetching strictly beats on-demand
    transfer and recompute.
    """
    agents: list[AgentId] = ["A", "B", "C"]
    blocks: dict[BlockId, KVBlock] = {}
    for agent in agents:
        for v in range(2):
            for j in range(4):
                block_id = f"{agent}{v}_{j}"
                blocks[block_id] = make_block(block_id)
    events: list[HandoffEvent] = []
    for i in range(num_steps):
        from_agent = agents[i % 3]
        to_agent = agents[(i + 1) % 3]
        visit = i // 3
        required = tuple(f"{to_agent}{visit % 2}_{j}" for j in range(4))
        events.append(
            HandoffEvent(step=i, from_agent=from_agent, to_agent=to_agent, required_blocks=required)
        )
    return Workload(name="cyclic", agents=agents, blocks=blocks, events=events)


def make_view(
    step: int,
    current_agent: AgentId,
    history: tuple[HandoffEvent, ...] = (),
    cached: dict[AgentId, tuple[BlockId, ...]] | None = None,
    config: SimConfig = SIM_CONFIG,
    agents: tuple[AgentId, ...] = ("A", "B", "C"),
    blocks: dict[BlockId, KVBlock] | None = None,
) -> PolicyView:
    """Hand-built read-only snapshot, as the simulator would pass to a strategy."""
    if blocks is None:
        blocks = {}
    if cached is None:
        cached = {agent: () for agent in agents}
    return PolicyView(
        step=step,
        current_agent=current_agent,
        agents=agents,
        blocks=dict(blocks),
        config=config,
        cached=cached,
        history=history,
    )
