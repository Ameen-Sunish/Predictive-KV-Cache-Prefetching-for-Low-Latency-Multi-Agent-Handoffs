"""OS-inspired Cache Management and Page Eviction Policies for TeleportCache.

This module models agent KV-cache storage as a physical page cache / page table:
- KV blocks correspond to fixed-size memory pages.
- Eviction policies correspond to OS page replacement algorithms (LRU, FIFO).
- AgentCache manages the resident page set, page faults/hits, and tracks
  prefetched-unused pages to evaluate prefetch accuracy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Sequence

from teleportcache.types import BlockId


class EvictionPolicy(ABC):
    """Abstract Base Class for page replacement / eviction policies.
    
    Analogous to OS kernel page replacement algorithms that select victim frames
    when physical memory capacity is exceeded.
    """

    @abstractmethod
    def on_insert(self, block_id: BlockId) -> None:
        """Record insertion of a new page block into the cache."""
        pass

    @abstractmethod
    def on_access(self, block_id: BlockId) -> None:
        """Record a hit/access to an existing resident page block."""
        pass

    @abstractmethod
    def choose_victim(self) -> BlockId | None:
        """Select the next victim page block for eviction.
        
        Returns None if the cache is empty.
        """
        pass

    @abstractmethod
    def on_remove(self, block_id: BlockId) -> None:
        """Remove a block from tracking (e.g. after eviction or invalidation)."""
        pass

    @abstractmethod
    def victim_order(self) -> tuple[BlockId, ...]:
        """Return resident block IDs in eviction order (next victim first)."""
        pass


class LRUPolicy(EvictionPolicy):
    """Least Recently Used (LRU) page replacement policy.
    
    Tracks page access recency using an OrderedDict for O(1) accesses, insertions,
    and victim selections. The least recently used page is chosen as the victim.
    """

    def __init__(self) -> None:
        self._order: OrderedDict[BlockId, None] = OrderedDict()

    def on_insert(self, block_id: BlockId) -> None:
        if block_id in self._order:
            self._order.move_to_end(block_id, last=True)
        else:
            self._order[block_id] = None

    def on_access(self, block_id: BlockId) -> None:
        if block_id in self._order:
            self._order.move_to_end(block_id, last=True)

    def choose_victim(self) -> BlockId | None:
        if not self._order:
            return None
        return next(iter(self._order))

    def on_remove(self, block_id: BlockId) -> None:
        self._order.pop(block_id, None)

    def victim_order(self) -> tuple[BlockId, ...]:
        return tuple(self._order.keys())


class FIFOPolicy(EvictionPolicy):
    """First-In First-Out (FIFO) page replacement policy.
    
    Evicts pages in the order they were inserted, regardless of subsequent access.
    """

    def __init__(self) -> None:
        self._order: OrderedDict[BlockId, None] = OrderedDict()

    def on_insert(self, block_id: BlockId) -> None:
        if block_id not in self._order:
            self._order[block_id] = None

    def on_access(self, block_id: BlockId) -> None:
        # FIFO ignores access recency
        pass

    def choose_victim(self) -> BlockId | None:
        if not self._order:
            return None
        return next(iter(self._order))

    def on_remove(self, block_id: BlockId) -> None:
        self._order.pop(block_id, None)

    def victim_order(self) -> tuple[BlockId, ...]:
        return tuple(self._order.keys())


def make_policy(name: str) -> EvictionPolicy:
    """Factory function to instantiate an EvictionPolicy by name."""
    normalized = name.strip().lower()
    if normalized == "lru":
        return LRUPolicy()
    elif normalized == "fifo":
        return FIFOPolicy()
    else:
        raise ValueError(f"Unknown eviction policy: '{name}'. Supported: 'lru', 'fifo'")


class AgentCache:
    """Per-agent page frame cache storing resident KV-cache blocks.
    
    Represents an agent's allocated page frames in memory with a fixed capacity.
    Tracks whether resident blocks were prefetched and whether they have been used,
    enabling exact accounting of useful vs wasted prefetches.
    """

    def __init__(self, capacity_blocks: int, policy: EvictionPolicy | str | None = None) -> None:
        if capacity_blocks < 0:
            raise ValueError(f"capacity_blocks must be non-negative, got {capacity_blocks}")
        self._capacity: int = capacity_blocks
        if policy is None:
            self._policy: EvictionPolicy = LRUPolicy()
        elif isinstance(policy, str):
            self._policy = make_policy(policy)
        else:
            self._policy = policy

        # Maps block_id -> bool indicating whether it was prefetched and NOT yet hit
        self._prefetched_unused: dict[BlockId, bool] = {}

    @property
    def capacity(self) -> int:
        """Maximum number of blocks this cache can hold."""
        return self._capacity

    def __len__(self) -> int:
        return len(self._prefetched_unused)

    def contains(self, block_id: BlockId) -> bool:
        """Check if a block is currently resident in the cache."""
        return block_id in self._prefetched_unused

    def __contains__(self, block_id: BlockId) -> bool:
        return self.contains(block_id)

    def touch(self, block_id: BlockId) -> bool:
        """Record an access/hit to a resident block.
        
        Updates eviction policy state (e.g. LRU order). If the block was
        marked 'prefetched-unused', transitions it to 'used'.
        
        Returns:
            bool: True if the block was prefetched and this was its first access
                  (i.e. a useful prefetch), False otherwise.
        Raises:
            KeyError: if block_id is not resident in the cache.
        """
        if block_id not in self._prefetched_unused:
            raise KeyError(f"Block '{block_id}' is not resident in cache")
        self._policy.on_access(block_id)
        was_prefetched_unused = self._prefetched_unused[block_id]
        if was_prefetched_unused:
            self._prefetched_unused[block_id] = False
            return True
        return False

    def get(self, block_id: BlockId) -> bool:
        """Query if block is resident; if hit, touches it and returns True.
        
        Convenience method mapping to page cache lookups.
        """
        if not self.contains(block_id):
            return False
        self.touch(block_id)
        return True

    def put(self, block_id: BlockId, prefetched: bool = False) -> list[tuple[BlockId, bool]]:
        """Insert a block into the cache, performing eviction if capacity is reached.
        
        Args:
            block_id: The ID of the block to insert.
            prefetched: Whether this block was brought in via prefetching.
            
        Returns:
            list[tuple[BlockId, bool]]: List of evicted blocks with their
            prefetched_unused status as (evicted_block_id, was_prefetched_unused).
        """
        evicted: list[tuple[BlockId, bool]] = []
        if self._capacity <= 0:
            return evicted

        if block_id in self._prefetched_unused:
            # Block is already resident: update policy and flag if needed
            self._policy.on_access(block_id)
            if not prefetched:
                self._prefetched_unused[block_id] = False
            return evicted

        # Cache is full: evict victim(s)
        while len(self._prefetched_unused) >= self._capacity:
            victim = self._policy.choose_victim()
            if victim is None:
                break
            was_unused = self._prefetched_unused.pop(victim, False)
            self._policy.on_remove(victim)
            evicted.append((victim, was_unused))

        # Insert new block
        self._prefetched_unused[block_id] = prefetched
        self._policy.on_insert(block_id)
        return evicted

    def resident_blocks(self) -> tuple[BlockId, ...]:
        """Return resident block IDs in eviction order (next victim first)."""
        return self._policy.victim_order()

    def get_prefetched_unused_blocks(self) -> list[BlockId]:
        """Return all resident blocks that were prefetched and never accessed."""
        return [b for b, unused in self._prefetched_unused.items() if unused]

    def occupancy(self) -> float:
        """Return cache occupancy fraction between 0.0 and 1.0."""
        if self._capacity <= 0:
            return 0.0
        return len(self._prefetched_unused) / float(self._capacity)
