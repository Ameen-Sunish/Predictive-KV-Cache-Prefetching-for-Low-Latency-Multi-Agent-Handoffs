"""Predictive prefetching strategy: predict next agent -> pick blocks -> prefetch.

The project's main contribution. Right after agent A finishes, the strategy
asks its :class:`~teleportcache.prediction.predictors.Predictor` who runs next,
asks its :class:`~teleportcache.prediction.block_selector.BlockSelector` which
KV pages that agent will need, and issues speculative page-in requests within
the handoff I/O window — all while respecting cache exclusion, the window
budget, and per-agent cache capacity.

Learns only from completed handoffs (``observe`` scores the prediction BEFORE
feeding the event to the predictor/selector, so the event is never used to
predict itself).
"""

from __future__ import annotations

import math

from teleportcache.prediction.block_selector import BlockSelector
from teleportcache.types import (
    AgentId,
    HandoffEvent,
    PolicyView,
    PrefetchRequest,
    SimConfig,
)


class PredictivePrefetchStrategy:
    """Prefetch strategy driven by a next-agent predictor and a block selector.

    Implements the ``Strategy`` protocol: ``use_cache=True``,
    ``miss_handling="transfer"`` (misses are served by on-demand page transfer).

    Args:
        predictor: Next-agent predictor (learns via ``update`` only).
        top_k_agents: How many ranked candidate agents to prefetch for.
        max_prefetch_blocks: Optional cap on requests per handoff; otherwise
            only the window budget and cache capacity limit the count.
        name: Optional strategy name; auto-generated when ``None``.
        selector: Optional injected :class:`BlockSelector` (a fresh default is
            created when ``None``).

    ``stats()`` exposes top-1 / top-k next-agent prediction accuracy and the
    cumulative number of prefetch requests issued.
    """

    def __init__(
        self,
        predictor,
        top_k_agents: int = 1,
        max_prefetch_blocks: int | None = None,
        name: str | None = None,
        selector: BlockSelector | None = None,
    ) -> None:
        if top_k_agents < 1:
            raise ValueError(f"top_k_agents must be >= 1, got {top_k_agents}")
        if max_prefetch_blocks is not None and max_prefetch_blocks < 0:
            raise ValueError(f"max_prefetch_blocks must be >= 0, got {max_prefetch_blocks}")
        self.predictor = predictor
        self.selector: BlockSelector = selector if selector is not None else BlockSelector()
        self.top_k_agents: int = top_k_agents
        self.max_prefetch_blocks: int | None = max_prefetch_blocks
        self.name: str = (
            name if name is not None else f"predictive({predictor.name},k={top_k_agents})"
        )
        self.use_cache: bool = True
        self.miss_handling: str = "transfer"
        self._last_ranking: list[tuple[AgentId, float]] | None = None
        self._pred_total: int = 0
        self._pred_top1: int = 0
        self._pred_topk: int = 0
        self._prefetch_requests: int = 0

    # ------------------------------------------------------------------ #
    # Prefetch planning
    # ------------------------------------------------------------------ #

    def plan_prefetch(self, view: PolicyView) -> list[PrefetchRequest]:
        """Issue prefetch requests for the (predicted) upcoming handoff.

        The view snapshot contains only past information; nothing here touches
        future events.
        """
        ranking = self.predictor.predict(view.current_agent, view.history, view.step)
        self._last_ranking = list(ranking)
        requests = self._build_requests(view, ranking)
        self._prefetch_requests += len(requests)
        return requests

    @staticmethod
    def _transfer_time(config: SimConfig, size_bytes: float) -> float:
        """Per-block page-in time; mirrors the frozen cost model without importing it."""
        return config.transfer_latency_s + size_bytes / config.bandwidth_bytes_per_s

    @classmethod
    def _window_budget(cls, view: PolicyView) -> int:
        """How many blocks could at best complete within ``prefetch_window_s``.

        Computed optimistically against the smallest known block so that the
        simulator's cumulative window check remains the binding constraint.
        """
        if not view.blocks:
            return 0
        min_size = min(block.size_bytes for block in view.blocks.values())
        per_block = cls._transfer_time(view.config, min_size)
        if per_block <= 0:
            return len(view.blocks)
        return int(view.config.prefetch_window_s // per_block)

    def _build_requests(
        self,
        view: PolicyView,
        ranking: list[tuple[AgentId, float]],
    ) -> list[PrefetchRequest]:
        """Turn a predicted ranking into a merged, budget-respecting request list."""
        budget = self._window_budget(view)
        if self.max_prefetch_blocks is not None:
            budget = min(budget, self.max_prefetch_blocks)
        selected = [(agent, prob) for agent, prob in ranking[: self.top_k_agents] if prob > 0]
        if budget <= 0 or not selected:
            return []

        # Confidence-aware budget: the window budget is scaled by the summed
        # prediction confidence, then split across predicted agents proportional
        # to their probability (largest-remainder apportionment). High confidence
        # means close to the full window is used; low confidence prefetches
        # fewer blocks so memory and bandwidth are not wasted.
        total_prob = sum(prob for _, prob in selected)
        effective_budget = budget * total_prob
        exact = [(agent, prob, effective_budget * prob / total_prob) for agent, prob in selected]
        quotas: dict[AgentId, int] = {agent: int(math.floor(share)) for agent, _, share in exact}
        leftovers = max(0, int(math.floor(effective_budget + 1e-9)) - sum(quotas.values()))
        by_fraction = sorted(exact, key=lambda t: (-(t[2] - math.floor(t[2])), t[0]))
        for i in range(leftovers):
            quotas[by_fraction[i % len(by_fraction)][0]] += 1

        scored: list[tuple[float, AgentId, str]] = []
        for agent, prob in selected:
            resident = set(view.cached.get(agent, ()))
            # Never request more blocks than the target cache's capacity;
            # the simulator evicts per policy when prefetches overflow it.
            quota = min(quotas.get(agent, 0), view.config.capacity_blocks)
            if quota <= 0:
                continue
            taken = 0
            for block_id, score in self.selector.rank(agent, exclude=resident):
                if taken >= quota:
                    break
                block = view.blocks.get(block_id)
                if block is None:
                    continue
                if self._transfer_time(view.config, block.size_bytes) > view.config.prefetch_window_s:
                    continue  # can never complete within the window
                scored.append((prob * score, agent, block_id))
                taken += 1

        scored.sort(key=lambda t: (-t[0], t[1], t[2]))
        return [PrefetchRequest(target_agent=agent, block_id=block_id) for _, agent, block_id in scored]

    # ------------------------------------------------------------------ #
    # Learning (past events only)
    # ------------------------------------------------------------------ #

    def observe(self, event: HandoffEvent) -> None:
        """Score the previous prediction against reality, then let models learn.

        The event is scored BEFORE ``predictor.update`` / ``selector.update``
        run, so the prediction could not have used this event.
        """
        if self._last_ranking:  # empty rankings are not scoreable
            self._pred_total += 1
            if self._last_ranking and self._last_ranking[0][0] == event.to_agent:
                self._pred_top1 += 1
            top_k_agents = {agent for agent, _ in self._last_ranking[: self.top_k_agents]}
            if event.to_agent in top_k_agents:
                self._pred_topk += 1
            self._last_ranking = None
        self.predictor.update(event)
        self.selector.update(event)

    def stats(self) -> dict:
        """Prediction accuracy and prefetch volume for this run."""
        return {
            "prediction_accuracy_top1": (
                self._pred_top1 / self._pred_total if self._pred_total else 0.0
            ),
            "prediction_accuracy_topk": (
                self._pred_topk / self._pred_total if self._pred_total else 0.0
            ),
            "prefetch_requests": self._prefetch_requests,
        }

    def reset(self) -> None:
        """Clear all state, including the predictor and selector."""
        self._last_ranking = None
        self._pred_total = 0
        self._pred_top1 = 0
        self._pred_topk = 0
        self._prefetch_requests = 0
        self.predictor.reset()
        self.selector.reset()
