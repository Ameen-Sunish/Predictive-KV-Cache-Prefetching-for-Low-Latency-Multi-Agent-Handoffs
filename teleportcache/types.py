"""FROZEN CONTRACT. Owned by Eldo. Do not change without all 3 teammates agreeing."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

BlockId = str
AgentId = str


@dataclass(frozen=True)
class KVBlock:
    block_id: BlockId
    num_tokens: int          # tokens covered by this fixed-size block (page)
    size_bytes: int          # bytes of KV data in this block


@dataclass(frozen=True)
class HandoffEvent:
    step: int
    from_agent: AgentId
    to_agent: AgentId
    required_blocks: tuple[BlockId, ...]   # blocks to_agent needs resident to start


@dataclass
class Workload:
    name: str
    agents: list[AgentId]
    blocks: dict[BlockId, KVBlock]         # every block that can ever be referenced
    events: list[HandoffEvent]             # ordered, events[i].step == i


@dataclass(frozen=True)
class SimConfig:
    capacity_blocks: int = 16                    # per-agent cache capacity (in blocks)
    bandwidth_bytes_per_s: float = 1.25e9        # ~10 Gbps
    transfer_latency_s: float = 0.001            # fixed per-block transfer overhead
    recompute_time_per_token_s: float = 0.001    # prefill cost per token
    prefetch_window_s: float = 0.05              # time between "Agent A done" and "Agent B starts" usable for prefetching
    eviction_policy: str = "lru"                 # "lru" | "fifo"


@dataclass(frozen=True)
class PrefetchRequest:
    target_agent: AgentId      # whose cache receives the block
    block_id: BlockId


@dataclass(frozen=True)
class PolicyView:
    """Read-only snapshot given to a strategy right after Agent A finishes,
    BEFORE the upcoming handoff is known. It must NOT contain the upcoming event."""
    step: int                                   # index of the upcoming handoff
    current_agent: AgentId                      # Agent A (just finished)
    agents: tuple[AgentId, ...]
    blocks: dict[BlockId, KVBlock]
    config: SimConfig
    cached: dict[AgentId, tuple[BlockId, ...]]  # resident blocks per agent, eviction order (next victim first)
    history: tuple[HandoffEvent, ...]           # completed events only (events[:step])


@runtime_checkable
class Strategy(Protocol):
    name: str
    use_cache: bool          # False => no caching at all (pure recompute baseline)
    miss_handling: str       # "transfer" | "recompute": how a miss is served

    def plan_prefetch(self, view: PolicyView) -> list[PrefetchRequest]: ...
    def observe(self, event: HandoffEvent) -> None: ...   # called after each handoff completes
    def reset(self) -> None: ...                          # called at start of every run


@runtime_checkable
class Predictor(Protocol):
    name: str
    def predict(self, current_agent: AgentId, history: Sequence[HandoffEvent], step: int) -> list[tuple[AgentId, float]]: ...
        # ranked best-first list of (candidate_next_agent, probability)
    def update(self, event: HandoffEvent) -> None: ...
    def reset(self) -> None: ...


@dataclass
class HandoffResult:
    step: int
    from_agent: AgentId
    to_agent: AgentId
    latency_s: float                 # handoff latency = total time Agent B waits on misses
    hits: int
    misses: int
    bytes_transferred: float         # bytes moved on-demand because of misses (0 if recomputed)
    recompute_time_s: float          # time spent recomputing missed blocks
    prefetch_issued_blocks: int      # prefetches that actually fit in the window and completed
    prefetch_bytes: float            # bytes moved by completed prefetches
    prefetch_useful_blocks: int      # prefetched blocks hit for the first time this step
    prefetch_wasted_blocks: int      # prefetched blocks evicted before ever being hit (attributed to the eviction step)
    cache_occupancy: float           # to_agent cache fill fraction after the handoff (0..1)


@dataclass
class RunResult:
    workload_name: str
    strategy_name: str
    config: SimConfig
    handoffs: list[HandoffResult] = field(default_factory=list)
    extras: dict = field(default_factory=dict)   # e.g. {"prediction_accuracy": 0.83}
