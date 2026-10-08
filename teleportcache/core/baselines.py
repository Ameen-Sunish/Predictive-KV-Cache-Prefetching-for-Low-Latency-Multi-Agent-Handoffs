"""Standard Baseline Strategies for Multi-Agent KV-Cache Management.

- RecomputeStrategy: Pure recompute baseline without page caching (use_cache=False, miss_handling="recompute").
  Every page access triggers full token prefilling from scratch.
- TransferOnDemandStrategy: Standard demand-paging baseline (use_cache=True, miss_handling="transfer").
  Pages are brought into cache on-demand only when a page fault occurs, with no prefetching.
"""

from __future__ import annotations

from dataclasses import dataclass
from teleportcache.types import HandoffEvent, PolicyView, PrefetchRequest, Strategy


@dataclass
class RecomputeStrategy:
    """Baseline strategy that recomputes all required KV tokens upon handoff.
    
    Analogous to a system without virtual memory page caching or persistence,
    where every task must compute its entire working set from scratch.
    """
    name: str = "recompute"
    use_cache: bool = False
    miss_handling: str = "recompute"

    def plan_prefetch(self, view: PolicyView) -> list[PrefetchRequest]:
        """Recompute baseline never prefetches."""
        return []

    def observe(self, event: HandoffEvent) -> None:
        """Observe completed handoff (no-op for baseline)."""
        pass

    def reset(self) -> None:
        """Reset state across simulation runs."""
        pass


@dataclass
class TransferOnDemandStrategy:
    """Demand-paging baseline that transfers missed blocks over the network on demand.
    
    Analogous to standard OS demand paging with LRU/FIFO replacement, where pages
    are fetched into resident memory only upon a page fault.
    """
    name: str = "transfer_on_demand"
    use_cache: bool = True
    miss_handling: str = "transfer"

    def plan_prefetch(self, view: PolicyView) -> list[PrefetchRequest]:
        """Demand paging does not perform speculative prefetching."""
        return []

    def observe(self, event: HandoffEvent) -> None:
        """Observe completed handoff (no-op for baseline)."""
        pass

    def reset(self) -> None:
        """Reset state across simulation runs."""
        pass
