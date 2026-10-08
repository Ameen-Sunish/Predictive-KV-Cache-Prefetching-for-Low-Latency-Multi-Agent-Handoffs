"""Deterministic Simulation Engine for Multi-Agent KV-Cache Handoffs.

Simulates agent handoffs in a multi-agent pipeline / DAG, modeling:
1. Context switches between agents.
2. Prefetching of KV-cache memory pages during the inter-agent transition window.
3. Demand paging (on-demand network page transfers) or page recomputation upon cache misses.
4. Page eviction under configured memory limits.
5. Accurate accounting of page hit rates, transfer latencies, compute overhead,
   and useful vs. wasted page prefetches.
"""

from __future__ import annotations

from typing import Any, Callable

from teleportcache.core.cache import AgentCache, make_policy
from teleportcache.core.cost_model import CostModel
from teleportcache.types import (
    AgentId,
    BlockId,
    HandoffEvent,
    HandoffResult,
    PolicyView,
    PrefetchRequest,
    RunResult,
    SimConfig,
    Strategy,
    Workload,
)


class Simulator:
    """Discrete-event simulator for KV-cache memory management and prefetching.
    
    Acts as the OS kernel MMU runtime, executing sequential agent handoffs,
    managing page frame caches, scheduling prefetch I/O within a budget window,
    and resolving page misses.
    """

    def __init__(self, config: SimConfig | None = None) -> None:
        self.config: SimConfig = config if config is not None else SimConfig()
        self.cost_model: CostModel = CostModel(self.config)

    def _validate_workload(self, workload: Workload) -> None:
        """Validate workload integrity before running simulation."""
        agent_set = set(workload.agents)
        for i, event in enumerate(workload.events):
            if event.step != i:
                raise ValueError(f"Event step mismatch at index {i}: expected step {i}, got {event.step}")
            if event.from_agent not in agent_set:
                raise ValueError(f"Unknown from_agent '{event.from_agent}' at step {i}")
            if event.to_agent not in agent_set:
                raise ValueError(f"Unknown to_agent '{event.to_agent}' at step {i}")
            for bid in event.required_blocks:
                if bid not in workload.blocks:
                    raise ValueError(f"Unknown required block '{bid}' at step {i}")

    def run(self, workload: Workload, strategy: Strategy) -> RunResult:
        """Run the simulation for a given workload and strategy deterministically.
        
        Args:
            workload: The multi-agent workflow containing agents, blocks, and events.
            strategy: The caching and prefetching strategy to evaluate.
            
        Returns:
            RunResult containing per-step HandoffResults and strategy statistics.
        """
        self._validate_workload(workload)
        strategy.reset()

        # Initialize fresh per-agent caches
        caches: dict[AgentId, AgentCache] = {
            agent: AgentCache(
                capacity_blocks=self.config.capacity_blocks,
                policy=make_policy(self.config.eviction_policy),
            )
            for agent in workload.agents
        }

        handoff_results: list[HandoffResult] = []

        for i, event in enumerate(workload.events):
            # 1. Build PolicyView before the upcoming event is revealed to the strategy
            view = PolicyView(
                step=i,
                current_agent=event.from_agent,
                agents=tuple(workload.agents),
                blocks=dict(workload.blocks),
                config=self.config,
                cached={agent: caches[agent].resident_blocks() for agent in workload.agents},
                history=tuple(workload.events[:i]),
            )

            prefetch_requests: list[PrefetchRequest] = []
            if strategy.use_cache:
                prefetch_requests = strategy.plan_prefetch(view)

            # 2. Prefetch Phase
            prefetch_issued_blocks = 0
            prefetch_bytes = 0.0
            step_wasted_prefetches = 0
            time_spent_prefetching = 0.0

            if strategy.use_cache and prefetch_requests:
                for req in prefetch_requests:
                    if req.target_agent not in caches:
                        raise ValueError(f"Prefetch requested for unknown target agent: '{req.target_agent}'")
                    if req.block_id not in workload.blocks:
                        raise ValueError(f"Prefetch requested for unknown block: '{req.block_id}'")

                    target_cache = caches[req.target_agent]
                    # Skip blocks already resident in target cache
                    if target_cache.contains(req.block_id):
                        continue

                    block = workload.blocks[req.block_id]
                    transfer_time = self.cost_model.transfer_time(block)

                    # Check if prefetch fits into the remaining prefetch window
                    if time_spent_prefetching + transfer_time <= self.config.prefetch_window_s + 1e-12:
                        time_spent_prefetching += transfer_time
                        prefetch_issued_blocks += 1
                        prefetch_bytes += float(block.size_bytes)

                        evicted = target_cache.put(req.block_id, prefetched=True)
                        for _evicted_id, was_unused in evicted:
                            if was_unused:
                                step_wasted_prefetches += 1
                    else:
                        # Exceeds prefetch window budget; dropped
                        pass

            # 3. Handoff Phase
            target_cache = caches[event.to_agent]
            hits = 0
            misses = 0
            bytes_transferred = 0.0
            recompute_time_s = 0.0
            latency_s = 0.0
            prefetch_useful_blocks = 0

            for block_id in event.required_blocks:
                block = workload.blocks[block_id]
                if strategy.use_cache and target_cache.contains(block_id):
                    # Cache Hit
                    hits += 1
                    was_useful_prefetch = target_cache.touch(block_id)
                    if was_useful_prefetch:
                        prefetch_useful_blocks += 1
                else:
                    # Cache Miss (Page Fault)
                    misses += 1
                    if strategy.miss_handling == "transfer":
                        m_cost = self.cost_model.transfer_time(block)
                        latency_s += m_cost
                        bytes_transferred += float(block.size_bytes)
                    elif strategy.miss_handling == "recompute":
                        m_cost = self.cost_model.recompute_time(block)
                        latency_s += m_cost
                        recompute_time_s += m_cost
                    else:
                        raise ValueError(f"Unknown miss_handling strategy: '{strategy.miss_handling}'")

                    if strategy.use_cache:
                        evicted = target_cache.put(block_id, prefetched=False)
                        for _evicted_id, was_unused in evicted:
                            if was_unused:
                                step_wasted_prefetches += 1

            cache_occupancy = target_cache.occupancy()

            handoff_result = HandoffResult(
                step=i,
                from_agent=event.from_agent,
                to_agent=event.to_agent,
                latency_s=latency_s,
                hits=hits,
                misses=misses,
                bytes_transferred=bytes_transferred,
                recompute_time_s=recompute_time_s,
                prefetch_issued_blocks=prefetch_issued_blocks,
                prefetch_bytes=prefetch_bytes,
                prefetch_useful_blocks=prefetch_useful_blocks,
                prefetch_wasted_blocks=step_wasted_prefetches,
                cache_occupancy=cache_occupancy,
            )
            handoff_results.append(handoff_result)

            # 4. Strategy observes the completed handoff event
            strategy.observe(event)

        # Attribute any leftover prefetched-unused resident blocks to the final step as wasted
        leftover_wasted = sum(len(c.get_prefetched_unused_blocks()) for c in caches.values())
        if leftover_wasted > 0 and handoff_results:
            handoff_results[-1].prefetch_wasted_blocks += leftover_wasted

        # Collect extras / prediction stats from strategy if available
        extras: dict[str, Any] = {}
        if hasattr(strategy, "stats"):
            stats_attr = getattr(strategy, "stats")
            if callable(stats_attr):
                extras = dict(stats_attr())
            elif isinstance(stats_attr, dict):
                extras = dict(stats_attr)

        return RunResult(
            workload_name=workload.name,
            strategy_name=strategy.name,
            config=self.config,
            handoffs=handoff_results,
            extras=extras,
        )
