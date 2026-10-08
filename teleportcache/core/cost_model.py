"""Cost and Latency Model for KV-Cache Operations in TeleportCache.

Models the OS memory hierarchy and transfer costs:
- Page transfer time models interconnect latency (fixed serialization overhead + bandwidth scaling).
- Page recomputation time models CPU/GPU prefill compute cost per token.
- Prefetch budget models the I/O window during which background page transfers can occur.
"""

from __future__ import annotations

import math
from teleportcache.types import KVBlock, SimConfig


class CostModel:
    """Latency and transfer cost evaluator for KV blocks.
    
    Translates hardware parameters (bandwidth, interconnect latency, compute rate)
    into page transfer and recomputation costs.
    """

    def __init__(self, config: SimConfig) -> None:
        self.config: SimConfig = config

    def transfer_time(self, block: KVBlock) -> float:
        """Compute time required to transfer a KV block across the network/bus.
        
        T_transfer = latency_overhead + (size_bytes / bandwidth)
        """
        if self.config.bandwidth_bytes_per_s <= 0:
            raise ValueError("Bandwidth must be positive")
        return self.config.transfer_latency_s + (block.size_bytes / self.config.bandwidth_bytes_per_s)

    def recompute_time(self, block: KVBlock) -> float:
        """Compute time required to recompute (prefill) the KV cache for a block.
        
        T_recompute = num_tokens * recompute_time_per_token
        """
        return block.num_tokens * self.config.recompute_time_per_token_s

    def miss_cost(self, block: KVBlock, miss_handling: str) -> float:
        """Calculate the latency cost of resolving a cache miss for a block."""
        if miss_handling == "transfer":
            return self.transfer_time(block)
        elif miss_handling == "recompute":
            return self.recompute_time(block)
        else:
            raise ValueError(f"Unknown miss_handling strategy: '{miss_handling}'. Must be 'transfer' or 'recompute'.")

    def prefetch_budget_blocks(self, block: KVBlock) -> int:
        """Calculate how many homogeneous blocks of this size fit in the prefetch window."""
        t_time = self.transfer_time(block)
        if t_time <= 0:
            return 0
        return math.floor(self.config.prefetch_window_s / t_time)
