"""Oracle prefetch strategy — an UPPER BOUND, not a real strategy.

:class:`OraclePrefetchStrategy` cheats: it is allowed to read the upcoming
handoff from the workload (``workload.events[view.step]``) and therefore knows
the true next agent and its exact required KV blocks before the handoff starts.
It issues prefetch requests for exactly the not-yet-resident required blocks,
packed smallest-first so the maximum number of page-ins fits in the window.

Use it only as a latency floor / prefetch ceiling when comparing against real
strategies (:class:`PredictivePrefetchStrategy`, on-demand transfer, recompute).
It reads the future, which is impossible in any deployable system.
"""

from __future__ import annotations

from teleportcache.types import HandoffEvent, PolicyView, PrefetchRequest, Workload


class OraclePrefetchStrategy:
    """Upper-bound strategy with clairvoyant access to the next handoff.

    Implements the ``Strategy`` protocol (``use_cache=True``,
    ``miss_handling="transfer"``). ``observe``/``reset`` are no-ops because the
    oracle never learns — it already knows.

    Args:
        workload: The workload whose events are read ahead (this is the cheat).
    """

    def __init__(self, workload: Workload) -> None:
        self._workload: Workload = workload
        self.name: str = "oracle"
        self.use_cache: bool = True
        self.miss_handling: str = "transfer"

    def plan_prefetch(self, view: PolicyView) -> list[PrefetchRequest]:
        """Request exactly the upcoming required blocks that are not resident.

        Blocks whose individual transfer time exceeds the whole window are
        skipped (they can never complete); the rest are packed smallest-first
        and truncated by the cumulative window budget and the target agent's
        free cache capacity.
        """
        if not (0 <= view.step < len(self._workload.events)):
            return []
        event: HandoffEvent = self._workload.events[view.step]
        target = event.to_agent
        resident = set(view.cached.get(target, ()))
        if view.config.capacity_blocks <= 0:
            return []

        config = view.config
        needed: list[str] = []
        seen: set[str] = set()
        for block_id in event.required_blocks:
            if block_id in resident or block_id in seen:
                continue
            block = view.blocks.get(block_id)
            if block is None:
                continue
            transfer_time = config.transfer_latency_s + block.size_bytes / config.bandwidth_bytes_per_s
            if transfer_time > config.prefetch_window_s:
                continue
            needed.append(block_id)
            seen.add(block_id)

        # Smallest-first packing maximizes the number of completed prefetches.
        def _transfer_time(block_id: str) -> float:
            return config.transfer_latency_s + view.blocks[block_id].size_bytes / config.bandwidth_bytes_per_s

        needed.sort(key=_transfer_time)
        requests: list[PrefetchRequest] = []
        cumulative = 0.0
        for block_id in needed:
            if len(requests) >= view.config.capacity_blocks:
                break
            transfer_time = _transfer_time(block_id)
            if cumulative + transfer_time > config.prefetch_window_s + 1e-12:
                break
            cumulative += transfer_time
            requests.append(PrefetchRequest(target_agent=target, block_id=block_id))
        return requests

    def observe(self, event: HandoffEvent) -> None:
        """No learning: the oracle already knows the schedule."""

    def reset(self) -> None:
        """No state to clear: the oracle reads the workload directly."""
